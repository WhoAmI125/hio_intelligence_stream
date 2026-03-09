"""
VLM API Router ??Legacy-compatible endpoints for adhoc_rtsp.html

Maps the original /api/vlm/* endpoints used by the existing frontend HTML
to the new 3-tier architecture's model_server internals.

Original endpoints expected by adhoc_rtsp.html:
    POST /api/vlm/start/    ??start RTSP stream + pipeline
    POST /api/vlm/stop/     ??stop stream
    GET  /api/vlm/video/    ??MJPEG frame stream
    GET  /api/vlm/status/   ??real-time status (running, fps, events, etc.)
    GET  /api/vlm/config/   ??server configuration
    GET  /api/vlm/events/   ??event list
    POST /api/vlm/zones/    ??set cashier/drawer zone polygons
    POST /api/vlm/feedback/ ??submit human feedback
"""

import json
import logging
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from model_server import config as server_config

logger = logging.getLogger("model_server.vlm_api")

router = APIRouter(prefix="/api/vlm", tags=["vlm-legacy"])

# ---------------------------------------------------------------------------
# Runtime state for active VLM workers
# ---------------------------------------------------------------------------
# Each camera_id gets its own state dictionary
_camera_states: dict[str, dict[str, Any]] = {}

_inference_threads: dict[str, threading.Thread] = {}
_worker_locks: dict[str, threading.Lock] = {}

# Global lock to synchronize Florence-2 inference across all threads
_inference_lock = threading.Lock()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = (_PROJECT_ROOT / "data").resolve()


def _to_public_media_url(raw: Any) -> Any:
    """Convert local clip/thumbnail file paths to browser-accessible /media URLs."""
    if not isinstance(raw, str):
        return raw
    value = raw.strip()
    if not value:
        return value
    if value.startswith(("http://", "https://", "/media/")):
        return value

    normalized = value.replace("\\", "/")
    if normalized.startswith("data/"):
        return f"/media/{normalized[len('data/'):]}"
    if normalized.startswith("./data/"):
        return f"/media/{normalized[len('./data/'):]}"
    if normalized.startswith("/"):
        marker = "/data/"
        idx = normalized.find(marker)
        if idx >= 0:
            return f"/media/{normalized[idx + len(marker):]}"

    try:
        resolved = Path(value).expanduser().resolve(strict=False)
        rel = resolved.relative_to(_DATA_ROOT)
        return f"/media/{str(rel).replace(os.sep, '/')}"
    except Exception:
        return value


def _normalize_event_media_links(event: Any) -> Any:
    if not isinstance(event, dict):
        return event
    out = dict(event)
    for key in ("clip_url", "thumbnail_url"):
        if key in out:
            out[key] = _to_public_media_url(out.get(key))
    gem = out.get("gemini")
    if isinstance(gem, dict):
        gem_out = dict(gem)
        if "validation_clip_url" in gem_out:
            gem_out["validation_clip_url"] = _to_public_media_url(gem_out.get("validation_clip_url"))
        out["gemini"] = gem_out
    return out


def _normalize_clip_map(values: Any) -> Any:
    if not isinstance(values, dict):
        return values
    return {
        k: _to_public_media_url(v) if isinstance(v, str) else v
        for k, v in values.items()
    }


def _trim_text(value: Any, max_len: int = 280) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _append_shadow_feedback_record(record: dict[str, Any]) -> None:
    """Persist manual shadow feedback as JSONL for audit/evolution data."""
    try:
        persist_dir = Path(getattr(server_config, "SHADOW_PERSIST_DIR", str(_DATA_ROOT / "shadow_feedback")))
        persist_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y%m%d")
        path = persist_dir / f"manual_feedback_{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning(f"[VLM API] shadow feedback persist skipped: {e}")


def _validate_rtsp_url(rtsp_url: str) -> tuple[str | None, str | None]:
    """
    Validate user-provided RTSP URL before stream connection attempt.

    Returns:
        (normalized_url, error_message)
    """
    raw = str(rtsp_url or "").strip()
    if not raw:
        return None, "rtsp_url is required"

    try:
        parsed = urlparse(raw)
    except Exception:
        return None, "Invalid RTSP URL format"

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"rtsp", "rtsps"}:
        return None, "RTSP URL must start with rtsp:// or rtsps://"
    if not parsed.hostname:
        return None, "RTSP URL must include a valid host"

    return raw, None

def _get_or_create_state(camera_id: str) -> dict[str, Any]:
    if camera_id not in _camera_states:
        _camera_states[camera_id] = {
            "running": False,
            "status": "stopped",
            "run_id": 0,
            "rtsp_url": "",
            "camera_id": camera_id,
            "base_fps": 1.5,
            "rtsp_transport": "tcp",
            "open_timeout_ms": 8000,
            "read_timeout_ms": 8000,
            "event_cooldown_sec": 20,
            "clip_duration_sec": 10,
            "validation_clip_sec": 10,
            "current_fps": 0.0,
            "stream_fps": 0.0,
            "last_error": "",
            "last_vlm": None,
            "last_validation": {},
            "last_clip_path": {},
            "recent_events": [],
            "cashier_zone": [],
            "drawer_zone": [],
            "evidence_mode": str(getattr(server_config, "EVIDENCE_MODE", "hybrid")),
            "last_frame_age_sec": 0.0,
            "last_overlay_age_sec": 0.0,
            "server_start_time": None,
            "frame_count": 0,
            "recent_inference_logs": [],
        }
    if camera_id not in _worker_locks:
        _worker_locks[camera_id] = threading.Lock()
    return _camera_states[camera_id]



def _get_server_modules():
    """Lazy import to avoid circular imports ??gets main module globals."""
    import model_server.main as main_mod
    return main_mod


def shutdown_all_workers(timeout_sec: float = 2.0) -> dict[str, int]:
    """
    Stop all camera workers for process shutdown.

    Returns summary:
        {"total": N, "stopped": X, "alive": Y}
    """
    srv = _get_server_modules()
    camera_ids = list(_camera_states.keys())

    # Step 1: signal stop for every camera state
    for camera_id in camera_ids:
        state = _get_or_create_state(camera_id)
        lock = _worker_locks[camera_id]
        with lock:
            state["running"] = False
            state["status"] = "stopping"
            state["run_id"] += 1
            state["last_error"] = "Server shutting down..."

        if getattr(srv, "stream_manager", None):
            try:
                srv.stream_manager.remove_camera(camera_id)
            except Exception:
                pass

    # Step 2: join best-effort
    stopped = 0
    alive = 0
    for camera_id in camera_ids:
        th = _inference_threads.get(camera_id)
        if th is not None and th.is_alive():
            th.join(timeout=max(0.1, float(timeout_sec)))

        still_alive = bool(th is not None and th.is_alive())
        lock = _worker_locks[camera_id]
        with lock:
            state = _get_or_create_state(camera_id)
            if still_alive:
                state["status"] = "stopping"
                state["last_error"] = "Worker still stopping during shutdown."
                alive += 1
            else:
                if _inference_threads.get(camera_id) is th:
                    _inference_threads[camera_id] = None
                state["status"] = "stopped"
                state["last_error"] = ""
                stopped += 1

    return {"total": len(camera_ids), "stopped": stopped, "alive": alive}


# ---------------------------------------------------------------------------
# POST /api/vlm/start/ ??Start RTSP stream + inference loop
# ---------------------------------------------------------------------------
@router.post("/start/")
async def vlm_start(request: Request):
    body = await request.json()
    rtsp_url, rtsp_error = _validate_rtsp_url(body.get("rtsp_url", ""))
    if rtsp_error:
        return JSONResponse({"success": False, "error": rtsp_error}, status_code=400)

    srv = _get_server_modules()

    camera_id = body.get("camera_id", "adhoc_cam")
    state = _get_or_create_state(camera_id)
    worker_lock = _worker_locks[camera_id]

    # Normalize request settings first.
    req_base_fps = float(body.get("base_fps", 1.5))
    req_transport = body.get("rtsp_transport", "tcp")
    req_open_timeout_ms = int(body.get("open_timeout_ms", 8000))
    req_read_timeout_ms = int(body.get("read_timeout_ms", 8000))
    req_event_cooldown_sec = int(body.get("event_cooldown_sec", 20))
    req_clip_duration_sec = int(body.get("clip_duration_sec", 10))
    req_validation_clip_sec = int(body.get("validation_clip_sec", 10))
    req_evidence_mode_raw = body.get("evidence_mode")
    req_use_video_validation = body.get("use_video_validation")

    def _as_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(v)

    valid_evidence_modes = {
        "hybrid",
        "video_first",
        "video_only",
        "images_first",
        "storyboard",
        "image",
    }
    req_evidence_mode = str(state.get("evidence_mode", "hybrid")).strip().lower()
    if isinstance(req_evidence_mode_raw, str) and req_evidence_mode_raw.strip():
        mode = req_evidence_mode_raw.strip().lower()
        if mode in valid_evidence_modes:
            req_evidence_mode = mode
        else:
            logger.warning(
                f"[VLM API] Invalid evidence_mode '{req_evidence_mode_raw}', keeping {req_evidence_mode}"
            )
    elif req_use_video_validation is not None:
        req_evidence_mode = "video_first" if _as_bool(req_use_video_validation) else "images_first"

    # Block duplicate RTSP across camera IDs to prevent decoder collision.
    if srv.stream_manager:
        try:
            dup_cam = srv.stream_manager.find_camera_by_rtsp(
                rtsp_url,
                exclude_camera_id=camera_id,
                active_only=True,
            )
        except Exception:
            dup_cam = None
        if dup_cam:
            return JSONResponse(
                {
                    "success": False,
                    "error": f"Same RTSP is already active on '{dup_cam}'. Stop it first.",
                    "duplicate_camera_id": dup_cam,
                },
                status_code=200,
            )

    # Idempotent fast-path: if same camera is already running with same core stream params,
    # do not force restart. This avoids unnecessary RTSP decoder re-init races.
    current_thread = _inference_threads.get(camera_id)
    if (
        bool(state.get("running"))
        and current_thread is not None
        and current_thread.is_alive()
        and str(state.get("rtsp_url", "")).strip() == rtsp_url
    ):
        state["base_fps"] = req_base_fps
        state["event_cooldown_sec"] = req_event_cooldown_sec
        state["clip_duration_sec"] = req_clip_duration_sec
        state["validation_clip_sec"] = req_validation_clip_sec
        state["evidence_mode"] = req_evidence_mode
        return {
            "success": True,
            "camera_id": camera_id,
            "already_running": True,
        }

    # Ensure previous worker state is not left running.
    with worker_lock:
        stale = _inference_threads.get(camera_id)
        if stale is not None and stale.is_alive() and not state["running"]:
            stale.join(timeout=2.0)

        # If start is called while already running, force a clean restart.
        if state["running"]:
            state["running"] = False
            state["status"] = "stopping"
            state["run_id"] += 1
            state["last_error"] = "Restarting worker..."

    if state["status"] == "stopping" and srv.stream_manager:
        try:
            srv.stream_manager.remove_camera(camera_id)
        except Exception:
            pass

    with worker_lock:
        stale = _inference_threads.get(camera_id)
        if stale is not None and stale.is_alive() and not state["running"]:
            stale.join(timeout=3.0)
            if stale.is_alive():
                return JSONResponse(
                    {"success": False, "error": "Previous inference loop is still stopping. Retry in a moment."},
                    status_code=409,
                )
            _inference_threads[camera_id] = None

        # Update state from request
        state["rtsp_url"] = rtsp_url
        state["base_fps"] = req_base_fps
        state["rtsp_transport"] = req_transport
        state["open_timeout_ms"] = req_open_timeout_ms
        state["read_timeout_ms"] = req_read_timeout_ms
        state["event_cooldown_sec"] = req_event_cooldown_sec
        state["clip_duration_sec"] = req_clip_duration_sec
        state["validation_clip_sec"] = req_validation_clip_sec
        state["evidence_mode"] = req_evidence_mode

    # Start camera stream
    try:
        stream = srv.stream_manager.add_camera(
            camera_id, rtsp_url,
            base_fps=state["base_fps"],
            rtsp_transport=state["rtsp_transport"],
            open_timeout_ms=state["open_timeout_ms"],
            read_timeout_ms=state["read_timeout_ms"],
        )
        stream.start()
    except RuntimeError as e:
        state["last_error"] = str(e)
        return JSONResponse({"success": False, "error": str(e)}, status_code=409)
    except Exception as e:
        state["last_error"] = str(e)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    with worker_lock:
        state["running"] = True
        state["status"] = "running"
        state["last_error"] = ""
        state["server_start_time"] = datetime.now().isoformat()
        state["frame_count"] = 0
        state["run_id"] += 1
        run_id = state["run_id"]

        # Start background inference loop for this run id.
        new_thread = threading.Thread(
            target=_inference_loop,
            args=(camera_id, run_id),
            daemon=True,
            name=f"vlm-inference-{camera_id}-{run_id}",
        )
        _inference_threads[camera_id] = new_thread
        new_thread.start()

    logger.info(f"[VLM API] Started: {rtsp_url} for camera {camera_id}")
    return {"success": True, "camera_id": camera_id}


# ---------------------------------------------------------------------------
# POST /api/vlm/stop/ ??Stop stream
# ---------------------------------------------------------------------------
@router.post("/stop/")
async def vlm_stop(camera_id: str = "adhoc_cam"):
    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)
    worker_lock = _worker_locks[camera_id]

    with worker_lock:
        state["running"] = False
        state["status"] = "stopping"
        state["run_id"] += 1

    stream_stopped = True
    if srv.stream_manager:
        try:
            stream_stopped = bool(srv.stream_manager.remove_camera(camera_id))
        except Exception:
            stream_stopped = False

    thread = _inference_threads.get(camera_id)
    if thread is not None and thread.is_alive():
        join_timeout = max(3.0, (state["read_timeout_ms"] / 1000.0) + 2.0)
        thread.join(timeout=join_timeout)

    thread_alive = bool(thread is not None and thread.is_alive())
    with worker_lock:
        if not thread_alive:
            if _inference_threads.get(camera_id) is thread:
                _inference_threads[camera_id] = None
            state["status"] = "stopped"
            state["last_error"] = ""
        else:
            state["status"] = "stopping"
            state["last_error"] = "Inference thread still stopping."

    if thread_alive or not stream_stopped:
        logger.warning(f"[VLM API] Stop requested for {camera_id} but inference thread is still alive.")
    else:
        logger.info(f"[VLM API] Stopped for {camera_id}")
    return {
        "success": (not thread_alive) and stream_stopped,
        "inference_thread_alive": thread_alive,
        "stream_stopped": stream_stopped,
    }


# ---------------------------------------------------------------------------
# GET /api/vlm/video/ ??MJPEG streaming
# ---------------------------------------------------------------------------
@router.get("/video/")
def vlm_video(camera_id: str = "adhoc_cam"):
    """Continuous MJPEG stream (multipart/x-mixed-replace).

    Streams at ~15fps for smooth UI display regardless of VLM inference rate.
    The inference loop runs separately in _inference_loop at base_fps.
    """
    UI_FPS = 15  # target display fps (original pipeline uses ~30fps)
    UI_INTERVAL = 1.0 / UI_FPS
    ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 70]

    def frame_generator():
        srv = _get_server_modules()
        state = _get_or_create_state(camera_id)
        last_send = 0.0

        while True:
            if bool(getattr(srv, "is_shutting_down", False)):
                break
            # Exit only when explicitly disconnected (browser closes img)
            # Keep streaming placeholder when stopped so reconnection is seamless
            now = time.time()
            if now - last_send < UI_INTERVAL:
                time.sleep(0.01)
                continue

            frame = None
            if state["running"] and srv.stream_manager:
                frame = srv.stream_manager.get_frame(camera_id)

            if frame is not None:
                try:
                    _, jpeg = cv2.imencode('.jpg', frame, ENCODE_PARAMS)
                except Exception:
                    continue
            else:
                # Placeholder frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                msg = "Waiting for stream..." if not state["running"] else "Connecting..."
                cv2.putText(blank, msg, (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                _, jpeg = cv2.imencode('.jpg', blank)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            last_send = now

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# GET /api/vlm/status/ ??Status polling
# ---------------------------------------------------------------------------
@router.get("/status/")
def vlm_status(camera_id: str = "adhoc_cam"):
    try:
        srv = _get_server_modules()
        state = _get_or_create_state(camera_id)
        thread = _inference_threads.get(camera_id)

        if bool(getattr(srv, "is_shutting_down", False)):
            return {
                "running": False,
                "status": "shutting_down",
                "server_time": datetime.now().isoformat(),
                "current_fps": 0.0,
                "stream_fps": 0.0,
                "base_fps": float(state.get("base_fps", 1.5)),
                "event_cooldown_sec": int(state.get("event_cooldown_sec", 20)),
                "validation_clip_sec": int(state.get("validation_clip_sec", 10)),
                "evidence_mode": str(state.get("evidence_mode", "hybrid")),
                "last_error": "Server shutting down",
                "last_frame_age_sec": float(state.get("last_frame_age_sec", 0.0)),
                "last_overlay_age_sec": float(state.get("last_overlay_age_sec", 0.0)),
                "last_vlm": state.get("last_vlm"),
                "last_validation": state.get("last_validation", {}),
                "last_clip_path": state.get("last_clip_path", {}),
                "recent_events": (state.get("recent_events") or [])[-50:],
                "buffers": {"raw_frames": 0, "raw_buffer_sec": 0.0, "gemini_frames": 0},
                "audit_log_dir": "",
                "router": {"policy_loaded": False},
                "florence_device_requested": str(getattr(srv.config, "FLORENCE_DEVICE", "unknown")),
                "florence_device_actual": "shutting_down",
                "inference_thread_alive": bool(thread is not None and thread.is_alive()),
                "cashier_zone_points": len(state.get("cashier_zone", []) or []),
                "drawer_zone_points": len(state.get("drawer_zone", []) or []),
            }

        florence_device_requested = str(getattr(srv.config, "FLORENCE_DEVICE", "unknown"))
        florence_device_actual = "not_loaded"
        if getattr(srv, "florence_adapter", None) is not None:
            florence_device_actual = str(getattr(srv.florence_adapter, "device", "unknown"))

        # Get stream stats
        stream_stats = {}
        if srv.stream_manager:
            try:
                all_stats = srv.stream_manager.get_all_stats()
                stream_stats = all_stats.get(camera_id, {})
            except Exception:
                stream_stats = {}

        ring_size = int(stream_stats.get("ring_buffer_size", stream_stats.get("buffer_size", 0)) or 0)
        stream_fps = float(stream_stats.get("stream_fps", stream_stats.get("fps", 0)) or 0.0)
        buffer_fps = float(stream_stats.get("buffer_fps_effective", state["base_fps"]) or state["base_fps"])
        inference_thread_alive = bool(thread is not None and thread.is_alive())

        # Get buffer info
        buffers = {
            "raw_frames": ring_size,
            "raw_buffer_sec": ring_size / max(buffer_fps, 0.1),
            "gemini_frames": 0,
            "buffer_fps": buffer_fps,
        }
        last_validation = state["last_validation"] if isinstance(state.get("last_validation"), dict) else {}
        last_clip_path = state["last_clip_path"] if isinstance(state.get("last_clip_path"), dict) else {}
        if "validation_clip_url" in last_validation:
            last_validation = dict(last_validation)
            last_validation["validation_clip_url"] = _to_public_media_url(last_validation.get("validation_clip_url"))
        last_clip_path = _normalize_clip_map(last_clip_path)
        recent_events = [_normalize_event_media_links(ev) for ev in state["recent_events"][-50:]]

        return {
            "running": state["running"],
            "status": state["status"],
            "server_time": datetime.now().isoformat(),
            "current_fps": state["current_fps"],
            "stream_fps": stream_fps,
            "base_fps": state["base_fps"],
            "event_cooldown_sec": state["event_cooldown_sec"],
            "validation_clip_sec": state["validation_clip_sec"],
            "evidence_mode": state["evidence_mode"],
            "last_error": state["last_error"],
            "last_frame_age_sec": state["last_frame_age_sec"],
            "last_overlay_age_sec": state["last_overlay_age_sec"],
            "last_vlm": state["last_vlm"],
            "last_validation": last_validation,
            "last_clip_path": last_clip_path,
            "recent_events": recent_events,
            "buffers": buffers,
            "audit_log_dir": str(srv.config.LOG_DIR) if hasattr(srv, "config") else "",
            "router": {
                "policy_loaded": bool(getattr(srv, "evidence_router", None) and srv.evidence_router.policy_model is not None)
            },
            "florence_device_requested": florence_device_requested,
            "florence_device_actual": florence_device_actual,
            "inference_thread_alive": inference_thread_alive,
            "cashier_zone_points": len(state.get("cashier_zone", []) or []),
            "drawer_zone_points": len(state.get("drawer_zone", []) or []),
        }
    except Exception as e:
        logger.warning(f"[VLM API] status error ({camera_id}): {e}")
        return JSONResponse(
            {
                "running": False,
                "status": "error",
                "server_time": datetime.now().isoformat(),
                "last_error": f"status error: {e}",
            },
            status_code=200,
        )


# ---------------------------------------------------------------------------
# GET /api/vlm/config/ ??Config
# ---------------------------------------------------------------------------
@router.get("/config/")
def vlm_config(camera_id: str = "adhoc_cam"):
    from model_server import config
    state = _get_or_create_state(camera_id)
    cfg = {
        "florence_model": config.FLORENCE_MODEL,
        "florence_backend": config.FLORENCE_BACKEND,
        "florence_device": config.FLORENCE_DEVICE,
        "gemini_model": config.GEMINI_MODEL,
        "base_fps": state["base_fps"],
        "clip_duration_sec": state["clip_duration_sec"],
        "validation_clip_sec": state["validation_clip_sec"],
        "evidence_mode": state["evidence_mode"],
        "cash_threshold": config.CASH_THRESHOLD,
        "violence_threshold": config.VIOLENCE_THRESHOLD,
        "fire_threshold": config.FIRE_THRESHOLD,
        "rtsp_url": state["rtsp_url"],
        "rtsp_transport": state["rtsp_transport"],
        "open_timeout_ms": state["open_timeout_ms"],
        "read_timeout_ms": state["read_timeout_ms"],
        "event_cooldown_sec": state["event_cooldown_sec"],
    }
    return {
        "running": state["running"],
        "config": cfg,
        **cfg,
    }


# ---------------------------------------------------------------------------
# GET /api/vlm/events/ ??Event list
# ---------------------------------------------------------------------------
@router.get("/events/")
def vlm_events(limit: int = 50, date: str | None = None):
    srv = _get_server_modules()

    if srv.local_storage:
        events = srv.local_storage.list_events(date_str=date, limit=limit)
    else:
        events = []
    events = [_normalize_event_media_links(ev) for ev in events]

    # Get available dates
    dates = []
    if srv.local_storage:
        dates = srv.local_storage.get_pending_dates()

    return {
        "events": events,
        "dates": dates,
        "count": len(events),
    }


@router.get("/inference-logs/")
def vlm_inference_logs(camera_id: str = "adhoc_cam", limit: int = 120):
    state = _get_or_create_state(camera_id)
    logs = state.get("recent_inference_logs")
    if not isinstance(logs, list):
        logs = []
    n = max(1, min(int(limit or 120), 1000))
    sliced = logs[-n:]
    normalized_logs: list[dict[str, Any]] = []
    for row in sliced:
        if isinstance(row, dict):
            item = dict(row)
        else:
            item = {"raw": str(row)}
        row_camera_id = str(item.get("camera_id", "")).strip()
        if not row_camera_id:
            item["camera_id"] = camera_id
        elif row_camera_id != camera_id:
            # Safety guard: this endpoint is camera-scoped, so mismatched rows are dropped.
            continue
        normalized_logs.append(item)
    return {
        "camera_id": camera_id,
        "count": len(normalized_logs),
        "logs": normalized_logs,
    }


@router.post("/lora-flourence-feedback/")
async def vlm_lora_flourence_feedback(request: Request):
    """
    Florence inference feedback endpoint wired directly to LoRA collector.
    Spelling is intentionally kept as 'flourence' to match UI contract.
    """
    body = await request.json()
    srv = _get_server_modules()

    if getattr(srv, "data_collector", None) is None:
        return JSONResponse(
            {"success": False, "error": "data_collector_not_available"},
            status_code=503,
        )

    camera_id = str(body.get("camera_id", "")).strip()
    scenario = str(body.get("scenario", "")).strip().lower()
    decision = str(body.get("decision", "")).strip().lower()
    note = str(body.get("note", "")).strip()
    event_id = str(body.get("event_id", "")).strip()
    if not event_id:
        event_id = f"florence_feedback_{int(time.time() * 1000)}_{camera_id or 'unknown'}"

    shared_caption = str(body.get("shared_caption", "")).strip()
    cash_caption = str(body.get("cash_caption", "")).strip()
    caption = cash_caption if scenario == "cash" and cash_caption else (shared_caption or cash_caption)
    summary = body.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    frame = None
    if camera_id and getattr(srv, "stream_manager", None):
        try:
            frame = srv.stream_manager.get_frame(camera_id)
        except Exception:
            frame = None

    result = srv.data_collector.collect_florence_feedback(
        event_id=event_id,
        decision=decision,
        note=note,
        frame=frame,
        caption=caption,
        scenario=scenario,
        camera_id=camera_id,
        summary=summary,
        source="LoRa_Flourence_feedback",
    )

    if not result.get("success"):
        return JSONResponse(result, status_code=400)
    return result


# ---------------------------------------------------------------------------
# POST /api/vlm/zones/ ??Set detection zones
# ---------------------------------------------------------------------------
@router.post("/zones/")
async def vlm_zones(request: Request):
    def _normalize_zone(points: Any) -> list[list[int]]:
        if not isinstance(points, list):
            return []
        out: list[list[int]] = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = int(round(float(p[0])))
                y = int(round(float(p[1])))
            except Exception:
                continue
            out.append([x, y])
        return out

    body = await request.json()
    camera_id = body.get("camera_id", "adhoc_cam")
    state = _get_or_create_state(camera_id)
    
    state["cashier_zone"] = _normalize_zone(body.get("cashier_zone", []))
    state["drawer_zone"] = _normalize_zone(body.get("drawer_zone", []))

    logger.info(
        f"[VLM API] Zones updated for {camera_id}: "
        f"cashier={len(state['cashier_zone'])} pts, "
        f"drawer={len(state['drawer_zone'])} pts"
    )
    return {"success": True}


# ---------------------------------------------------------------------------
# GET /api/vlm/crop/ — Zone crop preview (Florence-2가 실제로 보는 이미지)
# ---------------------------------------------------------------------------
@router.get("/crop/")
async def vlm_crop_preview(zone: str = "cashier", camera_id: str = "adhoc_cam"):
    """Return the cropped zone image as JPEG so you can see what Florence-2 sees."""
    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)

    if not camera_id or not srv.stream_manager:
        return Response(
            content=b"No active stream",
            status_code=404,
            media_type="text/plain",
        )

    frame = srv.stream_manager.get_frame(camera_id)
    if frame is None:
        return Response(
            content=b"No frame available",
            status_code=404,
            media_type="text/plain",
        )

    # Pick the requested zone polygon
    zone_key = f"{zone}_zone"
    zone_polygon = state.get(zone_key, [])

    if len(zone_polygon) >= 3 and srv.florence_adapter:
        cropped, bbox = srv.florence_adapter.crop_zone(frame, zone_polygon)
        label = f"{zone} zone (crop {cropped.shape[1]}x{cropped.shape[0]})"
    else:
        cropped = frame.copy()
        label = f"full frame (no {zone} zone set)"

    # Burn label into image
    cv2.putText(
        cropped, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    _, jpeg = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# POST /api/vlm/feedback/ — Human feedback
# ---------------------------------------------------------------------------
@router.post("/feedback/")
async def vlm_feedback(request: Request):
    body = await request.json()
    event_id = body.get("event_id", "")
    decision = body.get("decision", "")
    note = body.get("note", "")
    camera_id = body.get("camera_id", "adhoc_cam")

    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)

    # Update local storage
    if srv.local_storage:
        ev = srv.local_storage.get_event(event_id)
        if ev:
            ev["human_feedback"] = {
                "decision": decision,
                "note": note,
                "error_type": body.get("error_type", ""),
                "missed_focus": body.get("missed_focus", []),
                "suggestion": body.get("suggestion", ""),
                "at": datetime.now().isoformat(),
            }
            srv.local_storage.save_event(event_id, ev)

    # Also update in-memory recent events
    for ev in state["recent_events"]:
        if ev.get("event_id") == event_id:
            ev["human_feedback"] = {
                "decision": decision,
                "note": note,
                "at": datetime.now().isoformat(),
            }
            break

    # Feed to shadow agent for evolution learning
    for sa in srv.shadow_agents.values():
        sa.enqueue({
            "event_id": event_id,
            "human_feedback": decision,
            "note": note,
        })

    logger.info(f"[VLM API] Feedback for {camera_id}: {event_id} ??{decision}")

    # ── LoRA data collection with feedback ──
    if srv.data_collector is not None:
        try:
            srv.data_collector.collect_feedback(
                event_id=event_id,
                decision=decision,
                note=note,
                scenario=body.get("scenario", ""),
            )
        except Exception as dc_err:
            logger.debug(f"[VLM API] DataCollector feedback error: {dc_err}")

    return {"success": True}


# ---------------------------------------------------------------------------
# Shadow agent endpoints (for monitor_shadow.html)
# ---------------------------------------------------------------------------
@router.get("/shadow/recent/")
def vlm_shadow_recent(limit: int = 120):
    srv = _get_server_modules()
    results = []
    for name, sa in srv.shadow_agents.items():
        stats = sa.get_stats()
        results.append({
            "scenario": name,
            **stats,
        })
    return {"agents": results}


@router.post("/shadow/feedback/")
async def vlm_shadow_feedback(request: Request):
    body = await request.json()

    event_id = str(body.get("event_id", "")).strip()
    if not event_id:
        return JSONResponse({"success": False, "error": "event_id is required"}, status_code=400)

    raw_decision = str(body.get("decision", "")).strip().lower()
    if raw_decision in {"not_decide", "not-decide"}:
        raw_decision = "unsure"
    if raw_decision not in {"accept", "decline", "unsure"}:
        return JSONResponse(
            {"success": False, "error": "decision must be accept|decline|unsure"},
            status_code=400,
        )

    note = str(body.get("note", "")).strip()
    error_type = str(body.get("error_type", "")).strip().lower()
    if error_type not in {"false_positive", "false_negative", "wrong_class", "weak_reason", "other", ""}:
        error_type = "other"

    missed_focus_raw = body.get("missed_focus", [])
    missed_focus: list[str] = []
    if isinstance(missed_focus_raw, list):
        for x in missed_focus_raw:
            item = str(x or "").strip()
            if item:
                missed_focus.append(item[:64])
    suggestion = str(body.get("suggestion", "")).strip()[:400]
    source = str(body.get("source", "api")).strip() or "api"
    override = bool(body.get("override", False))

    srv = _get_server_modules()
    now_iso = datetime.now().isoformat()

    # Try to locate event context from memory and local storage.
    memory_event: dict[str, Any] | None = None
    memory_camera_id = ""
    for cam_id, state in _camera_states.items():
        for ev in state.get("recent_events", []):
            if str(ev.get("event_id", "")).strip() == event_id:
                memory_event = ev
                memory_camera_id = str(cam_id)
                break
        if memory_event is not None:
            break

    stored_event: dict[str, Any] | None = None
    if srv.local_storage:
        try:
            stored_event = srv.local_storage.get_event(event_id)
        except Exception:
            stored_event = None

    scenario = str(body.get("scenario", "")).strip().lower()
    if not scenario and memory_event:
        scenario = str(memory_event.get("event_type") or memory_event.get("scenario") or "").strip().lower()
    if not scenario and stored_event:
        scenario = str(stored_event.get("event_type") or stored_event.get("scenario") or "").strip().lower()

    camera_id = str(body.get("camera_id", "")).strip()
    if not camera_id:
        if memory_event:
            camera_id = str(memory_event.get("camera_id") or "").strip()
        elif stored_event:
            camera_id = str(stored_event.get("camera_id") or "").strip()
    if not camera_id:
        camera_id = memory_camera_id

    feedback_obj = {
        "decision": raw_decision,
        "note": note,
        "error_type": error_type,
        "missed_focus": missed_focus[:12],
        "suggestion": suggestion,
        "override": override,
        "source": source,
        "at": now_iso,
    }

    # Persist feedback into local event JSON (if found).
    local_saved = False
    if stored_event is not None and srv.local_storage:
        try:
            stored_event["shadow_feedback"] = feedback_obj
            srv.local_storage.save_event(event_id, stored_event)
            local_saved = True
        except Exception as e:
            logger.warning(f"[VLM API] shadow feedback local save failed ({event_id}): {e}")

    # Update in-memory recent event cache for active cameras.
    memory_updated = 0
    for state in _camera_states.values():
        for ev in state.get("recent_events", []):
            if str(ev.get("event_id", "")).strip() == event_id:
                ev["shadow_feedback"] = feedback_obj
                memory_updated += 1

    # Feed feedback to scenario shadow-agent path (not main feedback API).
    enqueued = False
    if scenario and getattr(srv, "shadow_agents", None) and scenario in srv.shadow_agents:
        tier1_src = memory_event or stored_event or {}
        tier1_result = {
            "is_detected": bool(tier1_src.get("is_detected", False)),
            "confidence": float(tier1_src.get("confidence", 0.0) or 0.0),
            "matched_keywords": tier1_src.get("matched_keywords", []),
            "event_type": scenario,
            "camera_id": camera_id,
        }
        shadow_event = {
            "event_id": event_id,
            "camera_id": camera_id,
            "scenario": scenario,
            "tier1_result": tier1_result,
            "human_feedback": raw_decision,
            "feedback_meta": feedback_obj,
        }
        try:
            enqueued = bool(srv.shadow_agents[scenario].enqueue(shadow_event))
        except Exception as e:
            logger.warning(f"[VLM API] shadow enqueue failed ({event_id}/{scenario}): {e}")

    # Always append manual feedback record for audit/training corpus.
    _append_shadow_feedback_record(
        {
            "event_id": event_id,
            "camera_id": camera_id,
            "scenario": scenario,
            "feedback": feedback_obj,
            "shadow_enqueued": enqueued,
            "local_saved": local_saved,
            "memory_updated": memory_updated,
            "ts": now_iso,
        }
    )

    return {
        "success": True,
        "event_id": event_id,
        "camera_id": camera_id,
        "scenario": scenario,
        "shadow_enqueued": enqueued,
        "local_saved": local_saved,
        "memory_updated": memory_updated,
        "feedback": feedback_obj,
    }


# ---------------------------------------------------------------------------
# Background inference loop
# ---------------------------------------------------------------------------
def _inference_loop(camera_id: str, run_id: int):
    """
    Continuous frame grab -> Florence -> CaptionAnalyzer detection loop.
    Each loop belongs to one run_id and exits if a new run starts.
    """
    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)
    
    from model_server.scenarios.base_scenario import CaptionAnalyzer
    from model_server.scenarios import ScenarioType

    last_inference = 0.0
    cooldown_tracker: dict[str, float] = {}

    logger.info(f"[VLM API] Inference loop started for {camera_id} (run_id={run_id})")

    while (
        state["running"]
        and state.get("run_id") == run_id
        and not bool(getattr(srv, "is_shutting_down", False))
    ):
        try:
            now = time.time()
            interval = 1.0 / max(float(state["base_fps"]), 0.5)
            if now - last_inference < interval:
                time.sleep(0.05)
                continue

            frame = srv.stream_manager.get_frame(camera_id) if srv.stream_manager else None
            if frame is None:
                state["last_frame_age_sec"] = 999.0
                time.sleep(0.5)
                continue

            state["frame_count"] += 1
            state["last_frame_age_sec"] = 0.0
            state["current_fps"] = 1.0 / max(now - last_inference, 0.001)
            last_inference = now

            cash_zone_applied = False
            cash_zone_bbox: list[int] | None = None
            scenario_results: dict[str, dict[str, Any]] = {}
            full_caption = ""
            cash_caption = ""

            if getattr(srv, "pipeline_orchestrator", None) is not None:
                try:
                    zones = {
                        "cashier": state.get("cashier_zone", []),
                        "drawer": state.get("drawer_zone", [])
                    }
                    with _inference_lock:
                        orch_result = srv.pipeline_orchestrator.process_frame_sequential(frame, zones=zones)
                    
                    scenario_results = {name: sr.to_dict() for name, sr in orch_result.scenario_results.items()}
                    full_caption = orch_result.metadata.get("shared_caption", "")
                    cash_zone_applied = len(zones["cashier"]) >= 3
                    
                    state["last_vlm"] = {
                        "scenario_results": scenario_results,
                        "total_inference_time_ms": orch_result.total_inference_time_ms,
                        "cashier_zone_applied": cash_zone_applied,
                        "cashier_zone_points": len(zones["cashier"]),
                        "shared_caption": full_caption,
                        "cash_caption": (scenario_results.get("cash", {}) or {}).get("raw_response", ""),
                        "source": "orchestrator",
                    }
                except Exception as e:
                    state["last_error"] = f"Orchestrator error: {e}"
            elif getattr(srv, "florence_adapter", None) is not None:
                # Fallback purely to avoid breaking completely if orchestrator failed init
                try:
                    with _inference_lock:
                        full_caption = srv.florence_adapter.infer(frame, "")
                        cash_caption = full_caption
                        cashier_zone = state.get("cashier_zone", []) or []
                        if len(cashier_zone) >= 3:
                            cropped, bbox = srv.florence_adapter.crop_zone(frame, cashier_zone)
                            if cropped is not None and getattr(cropped, "size", 0) > 0:
                                cash_caption = srv.florence_adapter.infer(cropped, "")
                                cash_zone_applied = True
                                cash_zone_bbox = [int(v) for v in bbox]
                except Exception as e:
                    state["last_error"] = f"Florence error: {e}"

                for scenario_name in ["cash", "fire", "violence"]:
                    t0 = time.time()
                    try:
                        scenario_type = ScenarioType[scenario_name.upper()]
                        if scenario_name == "cash" and cash_zone_applied:
                            scenario_caption = f"[ROI]\n{cash_caption}\n\n[GLOBAL]\n{full_caption}"
                        else:
                            scenario_caption = full_caption

                        result = CaptionAnalyzer.analyze(scenario_caption, scenario_type)
                        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
                        result["raw_response"] = scenario_caption
                        result["scenario_type"] = scenario_name
                        result["zone"] = "cashier" if (scenario_name == "cash" and cash_zone_applied) else "full"
                        
                        if scenario_name == "cash" and cash_zone_applied and cash_zone_bbox is not None:
                            result["zone_bbox"] = cash_zone_bbox
                        scenario_results[scenario_name] = result
                    except Exception as e:
                        scenario_results[scenario_name] = {
                            "error": str(e),
                            "is_detected": False,
                            "confidence": 0.0,
                            "zone": "cashier" if (scenario_name == "cash" and cash_zone_applied) else "full",
                        }

                state["last_vlm"] = {
                    "scenario_results": scenario_results,
                    "total_inference_time_ms": round(sum(r.get("inference_time_ms", 0) for r in scenario_results.values()), 1),
                    "cashier_zone_applied": cash_zone_applied,
                    "cashier_zone_points": len(state.get("cashier_zone", []) or []),
                    "shared_caption": full_caption,
                    "cash_caption": cash_caption,
                    "source": "fallback",
                }
            else:
                if state["frame_count"] == 1:
                    state["last_error"] = "Florence-2 not loaded. Caption analysis only."

            # Keep a rolling server-side log of Florence inference results.
            if scenario_results:
                summary = {}
                for name in ("cash", "fire", "violence"):
                    r = scenario_results.get(name, {}) if isinstance(scenario_results.get(name, {}), dict) else {}
                    summary[name] = {
                        "is_detected": bool(r.get("is_detected", False)),
                        "confidence": float(r.get("confidence", 0.0) or 0.0),
                        "zone": str(r.get("zone", "full")),
                    }
                logs = state.get("recent_inference_logs")
                if not isinstance(logs, list):
                    logs = []
                    state["recent_inference_logs"] = logs
                logs.append({
                    "at": datetime.now().isoformat(),
                    "camera_id": camera_id,
                    "frame_count": int(state.get("frame_count", 0)),
                    "total_inference_time_ms": float(state.get("last_vlm", {}).get("total_inference_time_ms", 0.0) or 0.0),
                    "cashier_zone_points": len(state.get("cashier_zone", []) or []),
                    "summary": summary,
                    "shared_caption": _trim_text(full_caption),
                    "cash_caption": _trim_text(cash_caption),
                })
                if len(logs) > 400:
                    state["recent_inference_logs"] = logs[-400:]

            for scenario_name, result in scenario_results.items():
                if not result.get("is_detected"):
                    continue

                last_event_time = cooldown_tracker.get(scenario_name, 0.0)
                if now - last_event_time < float(state["event_cooldown_sec"]):
                    continue

                cooldown_tracker[scenario_name] = now
                event_id = f"ev_{int(now * 1000)}_{scenario_name}_{camera_id}"
                event_caption = result.get("raw_response") or result.get("evidence") or ""

                event = {
                    "event_id": event_id,
                    "at": datetime.now().isoformat(),
                    "event_type": scenario_name,
                    "scenario": scenario_name,
                    "is_detected": True,
                    "confidence": result.get("confidence", 0),
                    "gemini": {
                        "state": "pending",
                        "validated": None,
                        "confidence": None,
                        "reason": "",
                    },
                    "human_feedback": None,
                    "caption": event_caption,
                    "matched_keywords": result.get("matched_keywords", []),
                    "clip_url": "",
                    "zone": result.get("zone", "full"),
                    "cashier_zone_used": bool(scenario_name == "cash" and cash_zone_applied),
                    "drawer_zone_used": bool(len(state.get("drawer_zone", []) or []) >= 3),
                    "camera_id": camera_id,
                }

                # ── Evidence Router / Uncertainty Gate ──
                needs_tier2 = False
                if getattr(srv, "evidence_router", None) is not None:
                    from model_server.episode_manager import Episode, EpisodeState
                    ep = Episode(episode_id=event_id, camera_id=camera_id, event_type=scenario_name)
                    ep.state = EpisodeState.VALIDATING
                    ep.detection_count = 1
                    ep.confidence_history = [result.get("confidence", 0)]
                    
                    action, reason, q, st = srv.evidence_router.select_action(ep, record_decision=True)
                    needs_tier2 = (action in getattr(srv.evidence_router, "TIER2_ACTIONS", {"GEMINI_IMG", "GEMINI_VIDEO"}))
                    event["router_action"] = action
                    event["router_reason"] = reason
                else:
                    from model_server.agents.dynamic_agent import UncertaintyGate
                    needs_tier2 = UncertaintyGate.should_escalate(
                        scenario_name, result, stability=0.5
                    )

                if needs_tier2:
                    event["gemini"]["state"] = "needed"
                else:
                    event["gemini"]["state"] = "skipped"

                # ── Gemini Tier2 Validation ──
                if needs_tier2 and srv.gemini_validator is not None:
                    try:
                        val_seconds = float(state.get("validation_clip_sec", 10))
                        val_entries = srv.stream_manager.get_clip_frames(
                            camera_id, window_sec=val_seconds
                        ) if srv.stream_manager else []

                        val_clip_path = None
                        if val_entries and len(val_entries) >= 2:
                            val_frames = [
                                e["frame"] for e in val_entries
                                if e.get("frame") is not None
                            ]
                            if val_frames and srv.local_storage:
                                # Estimate fps for validation clip
                                ts0 = val_entries[0].get("mono_ts", 0)
                                ts1 = val_entries[-1].get("mono_ts", 0)
                                v_fps = len(val_entries) / max(ts1 - ts0, 0.1)
                                v_fps = min(max(v_fps, 1.0), 30.0)
                                val_clip_path = srv.local_storage.save_clip(
                                    f"val_{event_id}", val_frames, fps=v_fps, allow_s3=False
                                )

                        gemini_ok, gemini_conf, gemini_reason, _ = (
                            srv.gemini_validator.validate_event_evidence(
                                packet={
                                    "event_type": scenario_name,
                                    "tier1_confidence": result.get("confidence", 0),
                                },
                                mode=state.get("evidence_mode", "hybrid"),
                                video_path=val_clip_path,
                                frame=frame,
                            )
                        )

                        val_log = (
                            getattr(srv.gemini_validator, "last_validation_log", {}) or {}
                        )
                        event["gemini"] = {
                            "state": "done",
                            "validated": gemini_ok,
                            "confidence": gemini_conf,
                            "reason": gemini_reason,
                            "at": datetime.now().isoformat(),
                            "validation_type": (
                                str(
                                    val_log.get("input_mode", "")
                                )
                                or ("video" if val_clip_path else "image")
                            ),
                            "input_mode": str(val_log.get("input_mode", "")),
                            "prompt_version": str(val_log.get("prompt_version", "")),
                            "processing_time_ms": int(val_log.get("processing_time_ms", 0) or 0),
                            "media_ref": str(val_log.get("media_ref", "")),
                        }

                        # Clean up temp validation clip
                        if val_clip_path and os.path.exists(val_clip_path):
                            try:
                                os.remove(val_clip_path)
                            except OSError:
                                pass

                        if not gemini_ok:
                            # Keep rejected result for audit/review, but mark as not detected.
                            logger.info(
                                f"[VLM API] Gemini REJECTED ({camera_id}): "
                                f"{scenario_name} conf={gemini_conf:.2f} "
                                f"reason={gemini_reason[:80]}"
                            )
                            event["is_detected"] = False
                            event["rejected_by_gemini"] = True
                            if not isinstance(state.get("last_validation"), dict):
                                state["last_validation"] = {}
                            state["last_validation"][scenario_name] = dict(event["gemini"])
                            # Keep rejected events for Gemini Logs / audit visibility.

                    except Exception as gem_err:
                        logger.warning(
                            f"[VLM API] Gemini validation error ({camera_id}): {gem_err}"
                        )
                        event["gemini"]["state"] = "error"
                        event["gemini"]["reason"] = str(gem_err)
                        # Fail-open: allow event through on error

                # ── Append confirmed event ──
                state["recent_events"].append(event)
                if len(state["recent_events"]) > 100:
                    state["recent_events"] = state["recent_events"][-100:]

                if srv.local_storage:
                    srv.local_storage.save_event(event_id, event)

                # ── Save clip from ring buffer ──
                clip_frames_for_lora = []
                if srv.local_storage and srv.stream_manager:
                    try:
                        clip_seconds = float(state.get("clip_duration_sec", 10))
                        clip_entries = srv.stream_manager.get_clip_frames(
                            camera_id, window_sec=clip_seconds
                        )
                        if clip_entries:
                            clip_frames = [
                                e["frame"] for e in clip_entries
                                if e.get("frame") is not None
                            ]
                            if clip_frames:
                                clip_frames_for_lora = clip_frames
                                # Estimate FPS from timestamps
                                if len(clip_entries) >= 2:
                                    ts_first = clip_entries[0].get("mono_ts", 0)
                                    ts_last = clip_entries[-1].get("mono_ts", 0)
                                    duration = ts_last - ts_first
                                    clip_fps = len(clip_entries) / max(duration, 0.1)
                                    clip_fps = min(max(clip_fps, 1.0), 30.0)
                                else:
                                    clip_fps = 15.0

                                clip_path = srv.local_storage.save_clip(
                                    event_id, clip_frames, fps=clip_fps
                                )
                                if clip_path:
                                    event["clip_url"] = clip_path
                                    srv.local_storage.save_event(event_id, event)
                                    logger.info(
                                        f"[VLM API] Clip saved ({camera_id}): "
                                        f"{len(clip_frames)} frames, "
                                        f"{clip_seconds:.0f}s, fps={clip_fps:.1f}"
                                    )

                                # ── Save thumbnail (last frame of clip) ──
                                last_frame = clip_frames[-1]
                                thumb_path = srv.local_storage.save_thumbnail(
                                    event_id, last_frame
                                )
                                if thumb_path:
                                    event["thumbnail_url"] = thumb_path
                                    srv.local_storage.save_event(event_id, event)
                                    logger.debug(
                                        f"[VLM API] Thumbnail saved ({camera_id}): "
                                        f"{thumb_path}"
                                    )

                    except Exception as clip_err:
                        logger.warning(
                            f"[VLM API] Clip/thumbnail save failed ({camera_id}): {clip_err}"
                        )

                # ── LoRA collection: cash only, Gemini-approved clip evidence only ──
                if srv.data_collector is not None:
                    try:
                        gem = event.get("gemini", {}) if isinstance(event.get("gemini"), dict) else {}
                        if (
                            scenario_name == "cash"
                            and gem.get("state") == "done"
                            and bool(gem.get("validated")) is True
                            and clip_frames_for_lora
                        ):
                            srv.data_collector.collect_gemini_validated_clip(
                                event_id=event_id,
                                scenario=scenario_name,
                                clip_frames=clip_frames_for_lora,
                                caption=str(event.get("caption") or ""),
                                camera_id=camera_id,
                                gemini_confidence=float(gem.get("confidence") or 0.0),
                                matched_keywords=list(result.get("matched_keywords") or []),
                                sample_count=3,
                            )
                    except Exception as dc_err:
                        logger.debug(f"[VLM API] DataCollector gemini-clip error: {dc_err}")

                # ── Trigger burst mode for tighter sampling post-detection ──
                if srv.stream_manager:
                    try:
                        srv.stream_manager.trigger_burst(camera_id)
                    except Exception:
                        pass

                # ── Update dead state variables ──
                if not isinstance(state.get("last_validation"), dict):
                    state["last_validation"] = {}
                state["last_validation"][scenario_name] = dict(event.get("gemini", {}))
                if event.get("clip_url"):
                    if not isinstance(state.get("last_clip_path"), dict):
                        state["last_clip_path"] = {}
                    state["last_clip_path"][scenario_name] = event["clip_url"]

                shadow = srv.shadow_agents.get(scenario_name)
                if shadow:
                    shadow.enqueue({
                        "event_id": event_id,
                        "tier1_result": result,
                    })

                logger.info(
                    f"[VLM API] Detection ({camera_id}): {scenario_name} "
                    f"conf={result.get('confidence', 0):.2f} "
                    f"zone={result.get('zone', 'full')} "
                    f"tier2={'Y' if needs_tier2 else 'N'}"
                )

        except Exception as e:
            state["last_error"] = str(e)
            logger.error(f"[VLM API] Inference error for {camera_id}: {e}")
            time.sleep(1)

    logger.info(f"[VLM API] Inference loop stopped for {camera_id} (run_id={run_id})")
