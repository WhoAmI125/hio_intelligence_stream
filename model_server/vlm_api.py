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
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
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

# Optional global lock to serialize Florence-2 inference across all camera threads
_inference_lock = threading.Lock()
_tier2_validation_slots = threading.BoundedSemaphore(
    max(1, int(getattr(server_config, "GEMINI_MAX_CONCURRENT", 1) or 1))
)

# Per-camera CashierTracker cache (created lazily)
_cashier_trackers: dict[str, Any] = {}
_cashier_tracker_lock = threading.Lock()
_clip_save_slots = threading.BoundedSemaphore(
    max(1, int(getattr(server_config, "CLIP_SAVE_MAX_CONCURRENT", 1) or 1))
)
_florence_log_lock = threading.Lock()
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
    for key in ("clip_url", "thumbnail_url", "overlay_clip_url", "overlay_snapshot_url"):
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


def _save_clip_serialized(
    storage: Any,
    event_id: str,
    frames: list[Any],
    *,
    fps: float,
    allow_s3: bool = True,
    overlay_polygons: Optional[dict[str, list[list[int]]]] = None,
) -> Any:
    if storage is None or not frames:
        return None

    _clip_save_slots.acquire()
    try:
        return storage.save_clip(
            event_id,
            frames,
            fps=fps,
            allow_s3=allow_s3,
            overlay_polygons=overlay_polygons,
        )
    finally:
        _clip_save_slots.release()


def _is_api_error_reason(reason: Any) -> bool:
    return str(reason or "").strip().lower().startswith("api error:")


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
            "clip_buffer_fps": 12.0,
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
            "evidence_mode": str(getattr(server_config, "EVIDENCE_MODE", "video_only")),
            "last_frame_age_sec": 0.0,
            "last_overlay_age_sec": 0.0,
            "server_start_time": None,
            "frame_count": 0,
            "recent_inference_logs": [],
            "last_inference_started_at": None,
            "last_inference_finished_at": None,
            "cooldown_tracker": {},
            "scheduler": {
                "registered": False,
                "pending": False,
                "inflight": False,
                "jobs_enqueued": 0,
                "jobs_completed": 0,
                "jobs_dropped": 0,
            },
            "postprocess": {
                "queue_size": 0,
                "worker_count": 0,
                "workers_alive": 0,
            },
        }
    if camera_id not in _worker_locks:
        _worker_locks[camera_id] = threading.Lock()
    return _camera_states[camera_id]



def _get_server_modules():
    """Lazy import to avoid circular imports ??gets main module globals."""
    import model_server.main as main_mod
    return main_mod


def _get_scheduler_metrics(camera_id: str) -> dict[str, Any]:
    srv = _get_server_modules()
    scheduler = getattr(srv, "inference_scheduler", None)
    if scheduler is None:
        return {}
    try:
        return scheduler.get_metrics(camera_id)
    except Exception:
        return {}


def _get_postprocess_metrics() -> dict[str, Any]:
    srv = _get_server_modules()
    postprocessor = getattr(srv, "event_postprocessor", None)
    if postprocessor is None:
        return {}
    try:
        return postprocessor.get_metrics()
    except Exception:
        return {}


def _append_inference_log(
    camera_id: str,
    state: dict[str, Any],
    *,
    full_caption: str,
    cash_caption: str,
    scenario_results: dict[str, dict[str, Any]],
) -> None:
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

    if not bool(getattr(server_config, "FLORENCE_LOG_PERSIST", True)):
        return

    day_dir = Path(getattr(server_config, "FLORENCE_LOG_DIR", _DATA_ROOT / "florence_logs")) / datetime.now().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    def _trim_text_local(value: Any, limit: int = 500) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _scenario_snapshot(name: str) -> dict[str, Any]:
        row = scenario_results.get(name, {}) if isinstance(scenario_results.get(name, {}), dict) else {}
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        florence_signals = meta.get("florence_signals", {}) if isinstance(meta.get("florence_signals"), dict) else {}
        snapshot = {
            "is_detected": bool(row.get("is_detected", False)),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "zone": str(row.get("zone", "full")),
            "evidence": _trim_text_local(row.get("evidence", ""), 300),
            "raw_response_preview": _trim_text_local(row.get("raw_response", ""), 500),
            "florence_signals": {
                "matched_keywords": list(florence_signals.get("matched_keywords", []) or []),
                "object_hints": list(florence_signals.get("object_hints", []) or []),
                "exclusion_match": list(florence_signals.get("exclusion_match", []) or []),
                "global_keywords": list(florence_signals.get("global_keywords", []) or []),
            },
        }
        for key in ("cash_path", "roi_confidence", "global_handover_score", "global_keywords"):
            if key in meta:
                snapshot[key] = meta.get(key)
        return snapshot

    payload = {
        "at": datetime.now().isoformat(),
        "camera_id": camera_id,
        "frame_count": int(state.get("frame_count", 0)),
        "source": str(state.get("last_vlm", {}).get("source", "")),
        "total_inference_time_ms": float(state.get("last_vlm", {}).get("total_inference_time_ms", 0.0) or 0.0),
        "cashier_zone_points": len(state.get("cashier_zone", []) or []),
        "drawer_zone_points": len(state.get("drawer_zone", []) or []),
        "shared_caption": _trim_text_local(full_caption, 1000),
        "cash_caption": _trim_text_local(cash_caption, 1000),
        "summary": summary,
        "scenarios": {
            "cash": _scenario_snapshot("cash"),
            "fire": _scenario_snapshot("fire"),
            "violence": _scenario_snapshot("violence"),
        },
    }

    log_path = day_dir / f"{camera_id}.jsonl"
    try:
        with _florence_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning("[VLM API] florence log persist failed (%s): %s", camera_id, e)


def _update_recent_event(camera_id: str, event: dict[str, Any]) -> None:
    state = _get_or_create_state(camera_id)
    events = state.get("recent_events")
    if not isinstance(events, list):
        return
    event_id = str(event.get("event_id", "")).strip()
    if not event_id:
        return
    for idx, existing in enumerate(events):
        if str(existing.get("event_id", "")).strip() == event_id:
            events[idx] = event
            break


def _persist_event(camera_id: str, event: dict[str, Any]) -> None:
    _update_recent_event(camera_id, event)
    srv = _get_server_modules()
    if getattr(srv, "local_storage", None) is not None:
        try:
            srv.local_storage.save_event(str(event.get("event_id", "")), event)
        except Exception as e:
            logger.warning(
                "[VLM API] event save failed (%s/%s): %s",
                camera_id,
                str(event.get("event_id", "")),
                e,
            )


def _queue_detection_postprocess(payload: dict[str, Any]) -> bool:
    srv = _get_server_modules()
    postprocessor = getattr(srv, "event_postprocessor", None)
    if postprocessor is None:
        return False
    return bool(postprocessor.submit(payload))


def _get_or_create_cashier_tracker(camera_id: str, state: dict[str, Any]):
    """Return (or lazily create) a CashierTracker for this camera.

    Tracker zones are refreshed from state every call so UI zone edits propagate
    without needing service restart.
    """
    from model_server.cashier_tracker import CashierTracker

    with _cashier_tracker_lock:
        tracker = _cashier_trackers.get(camera_id)
        if tracker is None:
            tracker = CashierTracker(
                camera_id=camera_id,
                cashier_zone_norm=state.get("cashier_zone") or [],
                drawer_zone_norm=state.get("drawer_zone") or [],
                cashier_watch_seconds=float(getattr(server_config, "CASHIER_WATCH_SECONDS", 60.0)),
                cashier_watch_min_obs=int(getattr(server_config, "CASHIER_WATCH_MIN_OBS", 6)),
                wrist_conf_threshold=float(getattr(server_config, "CASHIER_WRIST_CONF_THRESHOLD", 0.3)),
                hand_proximity_px=float(getattr(server_config, "CASH_HAND_PROXIMITY_PX", 120.0)),
                cooldown_sec=float(getattr(server_config, "CASH_TRIGGER_COOLDOWN_SEC", 20.0)),
                max_linger_sec=float(getattr(server_config, "CASHIER_MAX_LINGER_SEC", 30.0)),
                linger_gap_tolerance_sec=float(getattr(server_config, "CASHIER_LINGER_GAP_TOLERANCE_SEC", 1.5)),
                linger_cluster_px=float(getattr(server_config, "CASHIER_LINGER_CLUSTER_PX", 120.0)),
            )
            _cashier_trackers[camera_id] = tracker
        else:
            # Refresh zones if they changed
            tracker.update_zones(
                cashier_zone_norm=state.get("cashier_zone") or [],
                drawer_zone_norm=state.get("drawer_zone") or [],
            )
        return tracker


def _run_yolo_cash_gate(camera_id: str, frame: Any, state: dict[str, Any]) -> Optional[Any]:
    """
    Run YOLO-pose + CashierTracker. Returns trigger event if fired, else None.
    Silently returns None if YOLO disabled or no cashier_zone configured.
    """
    srv = _get_server_modules()
    yolo = getattr(srv, "yolo_pose_adapter", None)
    if yolo is None or not yolo.is_ready():
        return None

    cashier_zone = state.get("cashier_zone") or []
    if len(cashier_zone) < 3:
        return None  # no zone → nothing to gate

    try:
        detections = yolo.detect(frame)
    except Exception as exc:
        logger.debug("[YOLO gate] detect error for %s: %s", camera_id, exc)
        return None

    tracker = _get_or_create_cashier_tracker(camera_id, state)
    try:
        trigger = tracker.update(
            detections=detections,
            frame_shape=frame.shape if hasattr(frame, "shape") else (0, 0, 0),
            now_ts=time.time(),
            now_mono=time.monotonic(),
        )
    except Exception as exc:
        logger.debug("[YOLO gate] tracker error for %s: %s", camera_id, exc)
        return None

    state["last_yolo_detections"] = len(detections)
    if trigger is not None:
        logger.info(
            "[YOLO gate] Cash trigger fired: camera=%s proximity=%.1fpx persons=%d",
            camera_id, trigger.proximity_px, trigger.num_persons,
        )
    return trigger


def _sample_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not entries or count <= 0:
        return []
    if len(entries) <= count:
        return list(entries)
    if count == 1:
        return [entries[-1]]
    out: list[dict[str, Any]] = []
    last_idx = len(entries) - 1
    for i in range(count):
        idx = round((last_idx * i) / (count - 1))
        out.append(entries[idx])
    return out


def _crop_entries(entries: list[dict[str, Any]], polygon: list[Any], crop_zone: Any, limit: int) -> list[Any]:
    if not entries or not polygon or crop_zone is None:
        return []
    out: list[Any] = []
    for entry in _sample_entries(entries, limit):
        frame = entry.get("frame")
        if frame is None:
            continue
        try:
            cropped, _ = crop_zone(frame, polygon)
        except Exception:
            continue
        if cropped is not None and getattr(cropped, "size", 0) > 0:
            out.append(cropped)
    return out


def _build_validation_packet(
    camera_id: str,
    scenario_name: str,
    state: dict[str, Any],
    event: dict[str, Any],
    result: dict[str, Any],
    entries: list[dict[str, Any]],
    anchor_mono_ts: Any,
) -> dict[str, Any]:
    srv = _get_server_modules()
    sampled = _sample_entries(entries, 8)
    global_keyframes = [e.get("frame") for e in sampled if e.get("frame") is not None]
    crop_zone = getattr(getattr(srv, "florence_adapter", None), "crop_zone", None)
    cashier_zone = state.get("cashier_zone", []) or []
    drawer_zone = state.get("drawer_zone", []) or []
    cashier_roi_frames = _crop_entries(entries, cashier_zone, crop_zone, 8)
    drawer_roi_frames = _crop_entries(entries, drawer_zone, crop_zone, 4)
    result_meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    packet = {
        "episode_id": str(event.get("event_id", "")),
        "camera_id": camera_id,
        "event_type": scenario_name,
        "tier1_confidence": float(result.get("confidence", 0.0) or 0.0),
        "router_action": str(event.get("router_action", "")),
        "router_reason": str(event.get("router_reason", "")),
        "focus_hints": [f"scenario:{scenario_name}", f"zone:{event.get('zone', 'full')}"],
        "anchor_mono_ts": anchor_mono_ts,
        "clip_sec_used": float(state.get("validation_clip_sec", 10) or 10.0),
        "global_keyframes": global_keyframes,
        "cashier_roi_frames": cashier_roi_frames,
        "drawer_roi_frames": drawer_roi_frames,
        "florence_signals": result_meta.get("florence_signals", {}),
    }
    return packet


def shutdown_all_workers(timeout_sec: float = 2.0) -> dict[str, int]:
    """
    Stop all camera workers for process shutdown.

    Returns summary:
        {"total": N, "stopped": X, "alive": Y}
    """
    srv = _get_server_modules()
    camera_ids = list(_camera_states.keys())
    scheduler = getattr(srv, "inference_scheduler", None)

    # Step 1: signal stop for every camera state
    for camera_id in camera_ids:
        state = _get_or_create_state(camera_id)
        lock = _worker_locks[camera_id]
        with lock:
            state["running"] = False
            state["status"] = "stopping"
            state["run_id"] += 1
            state["last_error"] = "Server shutting down..."
            if isinstance(state.get("scheduler"), dict):
                state["scheduler"]["registered"] = False

        if scheduler is not None:
            try:
                scheduler.unregister_camera(camera_id)
            except Exception:
                pass

        if getattr(srv, "stream_manager", None):
            try:
                srv.stream_manager.remove_camera(camera_id)
            except Exception:
                pass

    # Step 2: join best-effort
    stopped = 0
    alive = 0
    for camera_id in camera_ids:
        scheduler_metrics = _get_scheduler_metrics(camera_id)
        still_alive = bool(
            scheduler_metrics.get("registered")
            or scheduler_metrics.get("pending")
            or scheduler_metrics.get("inflight")
        )
        lock = _worker_locks[camera_id]
        with lock:
            state = _get_or_create_state(camera_id)
            if still_alive:
                state["status"] = "stopping"
                state["last_error"] = "Worker still stopping during shutdown."
                alive += 1
            else:
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
    scheduler = getattr(srv, "inference_scheduler", None)

    # Optional compatibility mode: keep only one active camera inference loop.
    if bool(getattr(server_config, "SINGLE_CAMERA_MODE", False)):
        stopped_cameras: list[str] = []
        for other_camera_id in list(_camera_states.keys()):
            if str(other_camera_id) == str(camera_id):
                continue
            other_state = _camera_states.get(other_camera_id) or {}
            other_running = bool(other_state.get("running"))
            other_registered = bool((_get_scheduler_metrics(other_camera_id) or {}).get("registered"))
            if not other_running and not other_registered:
                continue
            try:
                await vlm_stop(camera_id=str(other_camera_id))
                stopped_cameras.append(str(other_camera_id))
            except Exception as stop_err:
                logger.warning(
                    f"[VLM API] SINGLE_CAMERA_MODE stop failed for {other_camera_id}: {stop_err}"
                )
        if stopped_cameras:
            logger.info(
                "[VLM API] SINGLE_CAMERA_MODE active: stopped cameras=%s before start(%s)",
                ",".join(stopped_cameras),
                camera_id,
            )

    # Normalize request settings first.
    req_base_fps = float(body.get("base_fps", 1.5))
    req_clip_buffer_fps = float(body.get("clip_buffer_fps", state.get("clip_buffer_fps", 12.0)))
    req_clip_buffer_fps = max(2.0, min(30.0, req_clip_buffer_fps))
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
    req_evidence_mode = str(state.get("evidence_mode", "video_only")).strip().lower()
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
    if (
        bool(state.get("running"))
        and bool((_get_scheduler_metrics(camera_id) or {}).get("registered"))
        and str(state.get("rtsp_url", "")).strip() == rtsp_url
    ):
        state["base_fps"] = req_base_fps
        state["clip_buffer_fps"] = req_clip_buffer_fps
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
        # If start is called while already running, force a clean restart.
        if state["running"]:
            state["running"] = False
            state["status"] = "stopping"
            state["run_id"] += 1
            state["last_error"] = "Restarting worker..."
            if isinstance(state.get("scheduler"), dict):
                state["scheduler"]["registered"] = False

    if scheduler is not None:
        try:
            scheduler.unregister_camera(camera_id)
        except Exception:
            pass

    if state["status"] == "stopping" and srv.stream_manager:
        try:
            srv.stream_manager.remove_camera(camera_id)
        except Exception:
            pass

    with worker_lock:
        # Update state from request
        state["rtsp_url"] = rtsp_url
        state["base_fps"] = req_base_fps
        state["clip_buffer_fps"] = req_clip_buffer_fps
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
            clip_buffer_fps=state["clip_buffer_fps"],
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
        state["last_inference_started_at"] = None
        state["last_inference_finished_at"] = None
        state["cooldown_tracker"] = {}
        state["run_id"] += 1
        if not isinstance(state.get("scheduler"), dict):
            state["scheduler"] = {}
        state["scheduler"]["registered"] = True

    if scheduler is not None:
        scheduler.register_camera(camera_id)
    else:
        if srv.stream_manager:
            try:
                srv.stream_manager.remove_camera(camera_id)
            except Exception:
                pass
        with worker_lock:
            state["running"] = False
            state["status"] = "error"
            state["last_error"] = "Inference scheduler is not available."
        return JSONResponse(
            {"success": False, "error": "Inference scheduler is not available."},
            status_code=503,
        )

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
    scheduler = getattr(srv, "inference_scheduler", None)

    with worker_lock:
        state["running"] = False
        state["status"] = "stopping"
        state["run_id"] += 1
        if isinstance(state.get("scheduler"), dict):
            state["scheduler"]["registered"] = False

    if scheduler is not None:
        try:
            scheduler.unregister_camera(camera_id)
        except Exception:
            pass

    stream_stopped = True
    if srv.stream_manager:
        try:
            stream_stopped = bool(srv.stream_manager.remove_camera(camera_id))
        except Exception:
            stream_stopped = False

    scheduler_metrics = _get_scheduler_metrics(camera_id)
    worker_alive = bool(
        scheduler_metrics.get("registered")
        or scheduler_metrics.get("pending")
        or scheduler_metrics.get("inflight")
    )
    with worker_lock:
        if not worker_alive:
            state["status"] = "stopped"
            state["last_error"] = ""
        else:
            state["status"] = "stopping"
            state["last_error"] = "Inference worker still stopping."

    if worker_alive or not stream_stopped:
        logger.warning(f"[VLM API] Stop requested for {camera_id} but inference worker is still alive.")
    else:
        logger.info(f"[VLM API] Stopped for {camera_id}")
    return {
        "success": (not worker_alive) and stream_stopped,
        "inference_thread_alive": worker_alive,
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
        scheduler_metrics = _get_scheduler_metrics(camera_id)
        postprocess_metrics = _get_postprocess_metrics()

        if bool(getattr(srv, "is_shutting_down", False)):
            return {
                "running": False,
                "status": "shutting_down",
                "server_time": datetime.now().isoformat(),
                "current_fps": 0.0,
                "stream_fps": 0.0,
                "base_fps": float(state.get("base_fps", 1.5)),
                "clip_buffer_fps": float(state.get("clip_buffer_fps", 12.0)),
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
                "inference_thread_alive": bool(scheduler_metrics.get("workers_alive")),
                "cashier_zone_points": len(state.get("cashier_zone", []) or []),
                "drawer_zone_points": len(state.get("drawer_zone", []) or []),
                "scheduler": scheduler_metrics,
                "postprocess": postprocess_metrics,
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
        inference_thread_alive = bool(
            scheduler_metrics.get("registered")
            or scheduler_metrics.get("pending")
            or scheduler_metrics.get("inflight")
            or scheduler_metrics.get("workers_alive")
        )
        state["scheduler"] = scheduler_metrics
        state["postprocess"] = postprocess_metrics

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
            "clip_buffer_fps": state.get("clip_buffer_fps", 12.0),
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
            "scheduler": scheduler_metrics,
            "postprocess": postprocess_metrics,
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
        "clip_buffer_fps": state.get("clip_buffer_fps", 12.0),
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
    decision = str(body.get("decision", "")).strip().lower()
    if decision in {"not_decide", "not-decide"}:
        decision = "unsure"
    note = str(body.get("note", "") or "").strip()
    camera_id = body.get("camera_id", "adhoc_cam")
    labeler = str(body.get("labeler", "") or "").strip()[:40]
    overlay_mode_shown = str(body.get("overlay_mode_shown", "") or "").strip().lower()
    if overlay_mode_shown not in {"plain", "overlay", ""}:
        overlay_mode_shown = ""

    if decision not in {"accept", "decline", "unsure"}:
        return JSONResponse(
            {"success": False, "error": "decision must be accept|decline|unsure"},
            status_code=400,
        )

    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)

    now_iso = datetime.now().isoformat()
    feedback_obj = {
        "decision": decision,
        "note": note,
        "error_type": body.get("error_type", ""),
        "missed_focus": body.get("missed_focus", []),
        "suggestion": body.get("suggestion", ""),
        "labeler": labeler,
        "overlay_mode_shown": overlay_mode_shown,
        "source": str(body.get("source", "api") or "api"),
        "at": now_iso,
    }

    # Update local storage
    if srv.local_storage:
        ev = srv.local_storage.get_event(event_id)
        if ev:
            ev["human_feedback"] = feedback_obj
            srv.local_storage.save_event(event_id, ev)

    # Also update in-memory recent events (cross-camera lookup)
    for cam_state in _camera_states.values():
        for ev in cam_state.get("recent_events", []):
            if ev.get("event_id") == event_id:
                ev["human_feedback"] = feedback_obj
                break

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
# One-shot inference + async post-processing
# ---------------------------------------------------------------------------
def _run_inference_once(camera_id: str, frame: Any, state: dict[str, Any], started_at: float) -> None:
    srv = _get_server_modules()

    from model_server.scenarios.base_scenario import CaptionAnalyzer
    from model_server.scenarios import ScenarioType

    state["frame_count"] = int(state.get("frame_count", 0)) + 1
    state["last_frame_age_sec"] = 0.0
    prev_started = state.get("last_inference_started_at")
    if prev_started:
        state["current_fps"] = 1.0 / max(float(started_at) - float(prev_started), 0.001)
    state["last_inference_started_at"] = started_at

    # YOLO-pose cash gate (runs before Florence; fires only when cashier_zone set
    # AND a customer wrist comes close to a cashier wrist). Returns a trigger
    # event or None. Fire/violence scenarios are untouched by this gate.
    yolo_trigger = None
    if bool(getattr(server_config, "YOLO_POSE_ENABLED", False)):
        yolo_trigger = _run_yolo_cash_gate(camera_id, frame, state)

    cash_zone_applied = False
    cash_zone_bbox: list[int] | None = None
    scenario_results: dict[str, dict[str, Any]] = {}
    full_caption = ""
    cash_caption = ""

    if getattr(srv, "pipeline_orchestrator", None) is not None:
        try:
            zones = {
                "cashier": state.get("cashier_zone", []),
                "drawer": state.get("drawer_zone", []),
            }
            inference_ctx = (
                _inference_lock
                if bool(getattr(server_config, "GLOBAL_INFERENCE_LOCK", True))
                else nullcontext()
            )
            with inference_ctx:
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
            logger.exception("[VLM API] Orchestrator error for %s", camera_id)
            return
    elif getattr(srv, "florence_adapter", None) is not None:
        try:
            inference_ctx = (
                _inference_lock
                if bool(getattr(server_config, "GLOBAL_INFERENCE_LOCK", True))
                else nullcontext()
            )
            with inference_ctx:
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
            logger.exception("[VLM API] Florence fallback error for %s", camera_id)
            return

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
        return

    # If YOLO cash gate is enabled AND a cashier_zone is set for this camera,
    # Florence per-frame cash scenario is no longer a source of cash events.
    # YOLO trigger is the sole cash event source.
    yolo_gate_active = (
        bool(getattr(server_config, "YOLO_POSE_ENABLED", False))
        and len(state.get("cashier_zone", []) or []) >= 3
    )
    if yolo_gate_active and "cash" in scenario_results:
        cash_result = scenario_results.get("cash") or {}
        cash_result["is_detected"] = False
        cash_result["suppressed_by_yolo_gate"] = True
        scenario_results["cash"] = cash_result

    if yolo_trigger is not None and yolo_gate_active:
        # Synthesize a cash scenario_results[cash] entry so downstream code
        # (logging + event creation) picks it up via the usual path.
        scenario_results["cash"] = {
            "is_detected": True,
            "confidence": 0.75,  # placeholder — real confidence decided by Gemini
            "zone": "cashier_yolo",
            "evidence": f"YOLO trigger: wrist proximity={yolo_trigger.proximity_px:.1f}px",
            "raw_response": "",
            "matched_keywords": ["_yolo_trigger"],
            "object_hints": [],
            "scenario_type": "cash",
            "inference_time_ms": 0.0,
            "source": "yolo_trigger",
            "metadata": {
                "yolo_trigger": yolo_trigger.to_dict(),
            },
            "yolo_triggered": True,
        }

    if scenario_results:
        _append_inference_log(
            camera_id,
            state,
            full_caption=full_caption,
            cash_caption=cash_caption,
            scenario_results=scenario_results,
        )

    cooldown_tracker = state.get("cooldown_tracker")
    if not isinstance(cooldown_tracker, dict):
        cooldown_tracker = {}
        state["cooldown_tracker"] = cooldown_tracker

    for scenario_name, result in scenario_results.items():
        if not result.get("is_detected"):
            continue

        last_event_time = float(cooldown_tracker.get(scenario_name, 0.0) or 0.0)
        if started_at - last_event_time < float(state["event_cooldown_sec"]):
            continue

        cooldown_tracker[scenario_name] = started_at
        event_id = f"ev_{int(started_at * 1000)}_{scenario_name}_{camera_id}"
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
            "postprocess_state": "pending",
        }

        needs_tier2 = False
        if getattr(srv, "evidence_router", None) is not None:
            from model_server.episode_manager import Episode, EpisodeState

            ep = Episode(episode_id=event_id, camera_id=camera_id, event_type=scenario_name)
            ep.state = EpisodeState.VALIDATING
            ep.detection_count = 1
            ep.confidence_history = [result.get("confidence", 0)]
            action, reason, _, _ = srv.evidence_router.select_action(ep, record_decision=True)
            needs_tier2 = action in getattr(srv.evidence_router, "TIER2_ACTIONS", {"GEMINI_IMG", "GEMINI_VIDEO"})
            event["router_action"] = action
            event["router_reason"] = reason
        else:
            from model_server.agents.dynamic_agent import UncertaintyGate

            needs_tier2 = UncertaintyGate.should_escalate(scenario_name, result, stability=0.5)

        event["gemini"]["state"] = "needed" if needs_tier2 else "skipped"
        if not isinstance(state.get("last_validation"), dict):
            state["last_validation"] = {}
        state["last_validation"][scenario_name] = dict(event.get("gemini", {}))

        state["recent_events"].append(event)
        if len(state["recent_events"]) > 100:
            state["recent_events"] = state["recent_events"][-100:]
        _persist_event(camera_id, event)

        # Carry YOLO trigger metadata into the postprocess payload. When set,
        # the postprocessor will (1) pull a longer CASH_TRIGGER_CLIP_SEC clip,
        # (2) sample N frames + run Florence batch, (3) only escalate to Gemini
        # if cash or H2H keywords are present (or CASH_TRIGGER_H2H_ESCALATE
        # forces through).
        is_yolo_triggered = bool(result.get("yolo_triggered"))
        if is_yolo_triggered:
            event["yolo_triggered"] = True
            event["yolo_trigger_meta"] = (result.get("metadata") or {}).get("yolo_trigger", {})
            needs_tier2 = True  # always evaluate a YOLO-triggered cash event
            event["gemini"]["state"] = "needed"

        payload = {
            "camera_id": camera_id,
            "scenario_name": scenario_name,
            "event": event,
            "result": result,
            "frame": frame,
            "anchor_mono_ts": time.monotonic(),
            "needs_tier2": bool(needs_tier2),
            "yolo_triggered": is_yolo_triggered,
        }
        queued = _queue_detection_postprocess(payload)
        if queued:
            event["postprocess_state"] = "queued"
            _persist_event(camera_id, event)
        else:
            if needs_tier2:
                event["gemini"]["state"] = "error"
                event["gemini"]["reason"] = "postprocess queue unavailable"
                state["last_validation"][scenario_name] = dict(event["gemini"])
            event["postprocess_state"] = "error"
            _persist_event(camera_id, event)

        if getattr(srv, "inference_scheduler", None) is not None:
            try:
                srv.inference_scheduler.mark_camera_active(camera_id)
            except Exception:
                pass

        if srv.stream_manager:
            try:
                srv.stream_manager.trigger_burst(camera_id)
            except Exception:
                pass

        logger.info(
            "[VLM API] Detection (%s): %s conf=%.2f zone=%s tier2=%s queued=%s",
            camera_id,
            scenario_name,
            float(result.get("confidence", 0) or 0.0),
            result.get("zone", "full"),
            "Y" if needs_tier2 else "N",
            "Y" if queued else "N",
        )

    state["last_inference_finished_at"] = time.time()


def _florence_multi_frame_gate(
    val_entries: list[dict[str, Any]],
    florence_adapter: Any,
    n_frames: int,
) -> tuple[bool, list[str], str]:
    """
    Sample n_frames from the clip, run Florence caption on each, and return
    (has_cash_or_h2h_signal, captions_list, combined_keywords_str).

    Used as a lightweight filter in front of Gemini for YOLO-triggered cash
    events so motion-only triggers that carry no cash/H2H signal don't waste
    Gemini API calls.
    """
    from model_server.scenarios.base_scenario import CaptionAnalyzer
    from model_server.scenarios import ScenarioType

    if not val_entries or florence_adapter is None or not n_frames:
        return False, [], ""

    sampled = _sample_entries(val_entries, max(1, int(n_frames)))
    captions: list[str] = []
    any_cash_signal = False
    any_h2h_signal = False
    all_keywords: set[str] = set()

    for entry in sampled:
        frm = entry.get("frame")
        if frm is None:
            continue
        try:
            cap = florence_adapter.infer(frm, "")
        except Exception as exc:
            logger.debug("[YOLO gate] Florence sample error: %s", exc)
            continue
        if not cap:
            continue
        captions.append(cap)
        analysis = CaptionAnalyzer.analyze(cap, ScenarioType.CASH)
        if analysis.get("is_detected"):
            any_h2h_signal = True
        kws = analysis.get("matched_keywords") or []
        for kw in kws:
            all_keywords.add(str(kw))
        # Treat direct cash vocabulary as the strongest signal.
        low = cap.lower()
        for token in ("cash", "money", "banknote", "banknotes", "paying", "paper money",
                      "won ", "currency"):
            if token in low:
                any_cash_signal = True
                break

    return (any_cash_signal or any_h2h_signal), captions, ", ".join(sorted(all_keywords))


def _process_detection_event(payload: dict[str, Any]) -> None:
    srv = _get_server_modules()
    camera_id = str(payload.get("camera_id", "")).strip()
    scenario_name = str(payload.get("scenario_name", "")).strip().lower()
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    frame = payload.get("frame")
    anchor_mono_ts = payload.get("anchor_mono_ts")
    needs_tier2 = bool(payload.get("needs_tier2"))
    yolo_triggered = bool(payload.get("yolo_triggered"))
    state = _get_or_create_state(camera_id)

    if not event or not camera_id or not scenario_name:
        return

    event["postprocess_state"] = "processing"
    _persist_event(camera_id, event)

    clip_frames_for_lora: list[Any] = []
    val_clip_path = None
    val_entries: list[dict[str, Any]] = []
    postprocess_ok = True

    # Zone overlays for visual hint (cashier yellow, drawer cyan).
    # Only cash uses overlays; fire/violence keep full-frame context.
    overlay_polygons: dict[str, list[list[int]]] = {}
    if scenario_name == "cash":
        cz = state.get("cashier_zone", []) or []
        dz = state.get("drawer_zone", []) or []
        if len(cz) >= 3:
            overlay_polygons["cashier"] = list(cz)
        if len(dz) >= 3:
            overlay_polygons["drawer"] = list(dz)
    event["overlay_applied"] = bool(overlay_polygons)

    try:
        if needs_tier2 and srv.gemini_validator is not None:
            try:
                _tier2_validation_slots.acquire()
                try:
                    # YOLO-triggered cash events use CASH_TRIGGER_CLIP_SEC for a
                    # longer clip that captures the customer approach + exchange.
                    if yolo_triggered and scenario_name == "cash":
                        val_seconds = float(getattr(server_config, "CASH_TRIGGER_CLIP_SEC", 14))
                    else:
                        val_seconds = float(state.get("validation_clip_sec", 10))
                    val_entries = (
                        srv.stream_manager.get_clip_frames(
                            camera_id,
                            window_sec=val_seconds,
                            anchor_mono_ts=anchor_mono_ts,
                        )
                        if srv.stream_manager
                        else []
                    )
                    if val_entries and len(val_entries) >= 2:
                        val_frames = [e["frame"] for e in val_entries if e.get("frame") is not None]
                        if val_frames and srv.local_storage:
                            ts0 = float(val_entries[0].get("mono_ts", 0) or 0)
                            ts1 = float(val_entries[-1].get("mono_ts", 0) or 0)
                            v_fps = len(val_entries) / max(ts1 - ts0, 0.1)
                            v_fps = min(max(v_fps, 1.0), 30.0)
                            # Gemini receives the OVERLAY version of the clip.
                            # For non-cash scenarios overlay_polygons is empty → behaves as before.
                            val_clip_path = _save_clip_serialized(
                                srv.local_storage,
                                f"val_{event['event_id']}",
                                val_frames,
                                fps=v_fps,
                                allow_s3=False,
                                overlay_polygons=overlay_polygons or None,
                            )

                    # ── YOLO trigger → Florence multi-frame keyword gate ──
                    # Decide if this clip shows cash-related activity before
                    # spending a Gemini call on it.
                    florence_gate_pass = True
                    if yolo_triggered and scenario_name == "cash":
                        n_sample = int(getattr(server_config, "CASH_TRIGGER_FLORENCE_FRAMES", 6))
                        has_signal, fl_caps, fl_kws = _florence_multi_frame_gate(
                            val_entries, getattr(srv, "florence_adapter", None), n_sample,
                        )
                        allow_motion_only = bool(getattr(server_config, "CASH_TRIGGER_H2H_ESCALATE", True))
                        florence_gate_pass = bool(has_signal or allow_motion_only)
                        event["florence_multi_frame"] = {
                            "n_frames": len(fl_caps),
                            "has_signal": has_signal,
                            "matched_keywords": fl_kws,
                            "captions_preview": [c[:200] for c in fl_caps[:3]],
                            "gate_pass": florence_gate_pass,
                            "reason": "signal_found" if has_signal else (
                                "motion_only_escalation" if allow_motion_only else "no_signal_skip"
                            ),
                        }
                        logger.info(
                            "[YOLO→Florence gate] %s event=%s n=%d has_signal=%s gate_pass=%s reason=%s kws='%s'",
                            camera_id, event.get("event_id"), len(fl_caps),
                            has_signal, florence_gate_pass,
                            event["florence_multi_frame"]["reason"], fl_kws[:80],
                        )
                        if not florence_gate_pass:
                            # Skip Gemini, mark as rejected by florence gate.
                            event["gemini"] = {
                                "state": "skipped",
                                "validated": False,
                                "confidence": 0.0,
                                "reason": "yolo_trigger_no_florence_signal",
                                "at": datetime.now().isoformat(),
                            }
                            event["is_detected"] = False
                            event["rejected_by_florence_gate"] = True
                            _persist_event(camera_id, event)
                            # release semaphore via finally below; return early after cleanup
                            raise StopIteration("florence_gate_skip")

                    validation_packet = _build_validation_packet(
                        camera_id,
                        scenario_name,
                        state,
                        event,
                        result,
                        val_entries,
                        anchor_mono_ts,
                    )
                    gemini_ok, gemini_conf, gemini_reason, _ = (
                        srv.gemini_validator.validate_event_evidence(
                            packet=validation_packet,
                            mode="video_only",
                            video_path=val_clip_path,
                            frame=frame,
                        )
                    )
                    if (not gemini_ok) and _is_api_error_reason(gemini_reason):
                        gemini_ok = True
                        gemini_conf = 1.0
                        gemini_reason = f"FAIL-OPEN on API error: {gemini_reason}"
                finally:
                    _tier2_validation_slots.release()

                val_log = getattr(srv.gemini_validator, "last_validation_log", {}) or {}
                event["gemini"] = {
                    "state": "done",
                    "validated": gemini_ok,
                    "confidence": gemini_conf,
                    "reason": gemini_reason,
                    "at": datetime.now().isoformat(),
                    "validation_type": str(val_log.get("input_mode", "")) or "video",
                    "input_mode": str(val_log.get("input_mode", "")) or "video",
                    "prompt_version": str(val_log.get("prompt_version", "")),
                    "processing_time_ms": int(val_log.get("processing_time_ms", 0) or 0),
                    "media_ref": str(val_log.get("media_ref", "")),
                }
                if not gemini_ok:
                    logger.info(
                        "[VLM API] Gemini REJECTED (%s): %s conf=%.2f reason=%s",
                        camera_id,
                        scenario_name,
                        float(gemini_conf or 0.0),
                        str(gemini_reason or "")[:80],
                    )
                    event["is_detected"] = False
                    event["rejected_by_gemini"] = True
            except StopIteration:
                # Florence multi-frame gate rejected the YOLO trigger (no cash/H2H
                # signal in sampled captions). We still proceed to persist the
                # event with gemini=skipped and save the permanent clip so the
                # operator can inspect rejected triggers in the UI.
                logger.info(
                    "[VLM API] YOLO trigger rejected by Florence gate (%s): event=%s",
                    camera_id, event.get("event_id"),
                )
            except Exception as gem_err:
                postprocess_ok = False
                logger.warning("[VLM API] Gemini validation error (%s): %s", camera_id, gem_err)
                event["gemini"]["state"] = "error"
                event["gemini"]["reason"] = str(gem_err)

            # Retain val_clip on disk: it is the exact media Gemini scored,
            # and will also serve as fallback if the permanent clip save below
            # ever fails (e.g. ring-buffer eviction race).
        if srv.local_storage and srv.stream_manager:
            try:
                # Align permanent-clip duration with validation clip so UI and
                # Gemini see the same time window.
                clip_seconds = float(state.get("validation_clip_sec", 10))
                # Reuse val_entries when Gemini path already fetched them;
                # this eliminates a second get_clip_frames() call that could
                # race against ring-buffer eviction during Gemini wait.
                if val_entries:
                    clip_entries = val_entries
                else:
                    clip_entries = srv.stream_manager.get_clip_frames(
                        camera_id,
                        window_sec=clip_seconds,
                        anchor_mono_ts=anchor_mono_ts,
                    )
                if not clip_entries:
                    logger.warning(
                        "[VLM API] Permanent clip skipped (%s/%s): empty clip_entries "
                        "(anchor_mono_ts=%s, val_entries=%d, val_clip=%s)",
                        camera_id, event.get("event_id"),
                        anchor_mono_ts, len(val_entries),
                        "present" if val_clip_path else "none",
                    )
                if clip_entries:
                    clip_frames = [e["frame"] for e in clip_entries if e.get("frame") is not None]
                    if clip_frames:
                        clip_frames_for_lora = clip_frames
                        if len(clip_entries) >= 2:
                            ts_first = float(clip_entries[0].get("mono_ts", 0) or 0)
                            ts_last = float(clip_entries[-1].get("mono_ts", 0) or 0)
                            duration = ts_last - ts_first
                            clip_fps = len(clip_entries) / max(duration, 0.1)
                            clip_fps = min(max(clip_fps, 1.0), 30.0)
                        else:
                            clip_fps = 15.0

                        clip_path = _save_clip_serialized(
                            srv.local_storage,
                            event["event_id"],
                            clip_frames,
                            fps=clip_fps,
                        )
                        if clip_path:
                            event["clip_url"] = clip_path
                            logger.info(
                                "[VLM API] Clip saved (%s): %s frames, %.0fs, fps=%.1f",
                                camera_id,
                                len(clip_frames),
                                clip_seconds,
                                clip_fps,
                            )

                        # Additionally save an OVERLAY version (yellow/cyan zone
                        # polygons burned in) for UI/forensics. The Gemini input
                        # uses its own short val_clip_path above.
                        if overlay_polygons:
                            overlay_clip_path = _save_clip_serialized(
                                srv.local_storage,
                                f"{event['event_id']}_roi",
                                clip_frames,
                                fps=clip_fps,
                                overlay_polygons=overlay_polygons,
                            )
                            if overlay_clip_path:
                                event["overlay_clip_url"] = overlay_clip_path

                        thumb_path = srv.local_storage.save_thumbnail(
                            event["event_id"], clip_frames[-1]
                        )
                        if thumb_path:
                            event["thumbnail_url"] = thumb_path

                        if overlay_polygons:
                            overlay_thumb_path = srv.local_storage.save_thumbnail(
                                f"{event['event_id']}_roi",
                                clip_frames[-1],
                                overlay_polygons=overlay_polygons,
                            )
                            if overlay_thumb_path:
                                event["overlay_snapshot_url"] = overlay_thumb_path
            except Exception as clip_err:
                postprocess_ok = False
                logger.warning("[VLM API] Clip/thumbnail save failed (%s): %s", camera_id, clip_err)

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
                        event_id=event["event_id"],
                        scenario=scenario_name,
                        clip_frames=clip_frames_for_lora,
                        caption=str(event.get("caption") or ""),
                        camera_id=camera_id,
                        gemini_confidence=float(gem.get("confidence") or 0.0),
                        matched_keywords=list(result.get("matched_keywords") or []),
                        sample_count=3,
                    )
            except Exception as dc_err:
                logger.debug("[VLM API] DataCollector gemini-clip error: %s", dc_err)
    except Exception as e:
        postprocess_ok = False
        event["postprocess_state"] = "error"
        logger.warning("[VLM API] postprocess error (%s/%s): %s", camera_id, event.get("event_id"), e)
    finally:
        event["postprocess_state"] = "done" if postprocess_ok else "error"
        # Fallback: if permanent clip save produced nothing but the retained
        # val_clip exists, promote val_clip to clip_url so the UI/DB always
        # has a playable media reference.
        if not event.get("clip_url") and val_clip_path and os.path.exists(val_clip_path):
            event["clip_url"] = val_clip_path
            if overlay_polygons and not event.get("overlay_clip_url"):
                # val_clip already has the overlay baked in.
                event["overlay_clip_url"] = val_clip_path
            logger.info(
                "[VLM API] Clip fallback to val_clip (%s/%s): %s",
                camera_id, event.get("event_id"), val_clip_path,
            )
        if not isinstance(state.get("last_validation"), dict):
            state["last_validation"] = {}
        state["last_validation"][scenario_name] = dict(event.get("gemini", {}))
        if event.get("clip_url"):
            if not isinstance(state.get("last_clip_path"), dict):
                state["last_clip_path"] = {}
            state["last_clip_path"][scenario_name] = event["clip_url"]
        _persist_event(camera_id, event)


# ---------------------------------------------------------------------------
# Legacy per-camera loop kept for fallback/testing
# ---------------------------------------------------------------------------
def _inference_loop(camera_id: str, run_id: int):
    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)
    last_started = 0.0
    logger.info(f"[VLM API] Inference loop started for {camera_id} (run_id={run_id})")

    while (
        state["running"]
        and state.get("run_id") == run_id
        and not bool(getattr(srv, "is_shutting_down", False))
    ):
        try:
            now = time.time()
            interval = 1.0 / max(float(state["base_fps"]), 0.5)
            if now - last_started < interval:
                time.sleep(0.05)
                continue
            frame = srv.stream_manager.get_frame(camera_id) if srv.stream_manager else None
            if frame is None:
                state["last_frame_age_sec"] = 999.0
                time.sleep(0.5)
                continue
            last_started = now
            _run_inference_once(camera_id, frame, state, now)
        except Exception as e:
            state["last_error"] = str(e)
            logger.error(f"[VLM API] Inference error for {camera_id}: {e}")
            time.sleep(1)

    logger.info(f"[VLM API] Inference loop stopped for {camera_id} (run_id={run_id})")
