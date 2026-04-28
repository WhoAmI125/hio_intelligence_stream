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
from datetime import datetime, timezone, timedelta

# Operators are in Korea — always stamp event timestamps in KST so the value
# is unambiguous regardless of the host's process timezone.
KST = timezone(timedelta(hours=9))


def now_kst_iso() -> str:
    return datetime.now(KST).isoformat()
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from model_server import config as server_config

logger = logging.getLogger("model_server.vlm_api")

router = APIRouter(prefix="/api/vlm", tags=["vlm-v3"])

# ---------------------------------------------------------------------------
# Runtime state for active VLM workers
# ---------------------------------------------------------------------------
# Each camera_id gets its own state dictionary
_camera_states: dict[str, dict[str, Any]] = {}

_inference_threads: dict[str, threading.Thread] = {}
_worker_locks: dict[str, threading.Lock] = {}

# Optional global lock to serialize GPU inference across all camera threads
_inference_lock = threading.Lock()
_tier2_validation_slots = threading.BoundedSemaphore(
    max(1, int(getattr(server_config, "GEMINI_MAX_CONCURRENT", 1) or 1))
)
_clip_save_slots = threading.BoundedSemaphore(
    max(1, int(getattr(server_config, "CLIP_SAVE_MAX_CONCURRENT", 1) or 1))
)
_proposal_log_lock = threading.Lock()
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
    if isinstance(out.get("candidate_clip_paths"), dict):
        out["candidate_clip_paths"] = _normalize_clip_map(out.get("candidate_clip_paths"))
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


_SUPPORTED_CORRECTED_EVENT_TYPES = {"cash", "violence", "fire", "staff_cash_theft", "none"}


def _normalize_corrected_event_type(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": str(fallback or "").strip().lower(),
        "cash_transaction": "cash",
        "fire_alert": "fire",
        "threat_to_cashier": "violence",
        "not_applicable": "none",
        "no_event": "none",
        "false_positive": "none",
    }
    return aliases.get(text, text)


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
            "base_fps": float(getattr(server_config, "BASE_FPS", 1.0) or 1.0),
            "clip_buffer_fps": 12.0,
            "mjpeg_burst_until": 0.0,
            "rtsp_transport": "tcp",
            "open_timeout_ms": 8000,
            "read_timeout_ms": 8000,
            "event_cooldown_sec": 20,
            "clip_duration_sec": 10,
            "validation_clip_sec": int(getattr(server_config, "V3_VALIDATION_CLIP_SEC", 15)),
            "current_fps": 0.0,
            "stream_fps": 0.0,
            "last_error": "",
            "model_health": {},
            "last_vlm": None,
            "last_validation": {},
            "last_clip_path": {},
            "recent_events": [],
            "cashier_zone": [],
            "drawer_zone": [],
            "exchange_band": [],
            "staff_work_zone": [],
            "evidence_mode": str(getattr(server_config, "EVIDENCE_MODE", "video_only")),
            "last_frame_age_sec": 0.0,
            "last_overlay_age_sec": 0.0,
            "server_start_time": None,
            "frame_count": 0,
            "recent_inference_logs": [],
            "last_inference_started_at": None,
            "last_inference_finished_at": None,
            "last_cash_clip_siglip_ts": 0.0,
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
        "at": now_kst_iso(),
        "camera_id": camera_id,
        "frame_count": int(state.get("frame_count", 0)),
        "total_inference_time_ms": float(state.get("last_vlm", {}).get("total_inference_time_ms", 0.0) or 0.0),
        "cashier_zone_points": len(state.get("cashier_zone", []) or []),
        "exchange_band_points": len(state.get("exchange_band", []) or []),
        "staff_work_zone_points": len(state.get("staff_work_zone", []) or []),
        "summary": summary,
        "shared_caption": _trim_text(full_caption),
        "cash_caption": _trim_text(cash_caption),
    })
    if len(logs) > 400:
        state["recent_inference_logs"] = logs[-400:]

    source = str(state.get("last_vlm", {}).get("source", "") or "")
    if not bool(getattr(server_config, "V3_PROPOSAL_LOG_PERSIST", True)):
        return
    log_root = Path(getattr(server_config, "V3_PROPOSAL_LOG_DIR", _DATA_ROOT / "v3_proposal_logs"))

    day_dir = log_root / datetime.now(KST).strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    def _trim_text_local(value: Any, limit: int = 500) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _scenario_snapshot(name: str) -> dict[str, Any]:
        row = scenario_results.get(name, {}) if isinstance(scenario_results.get(name, {}), dict) else {}
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        snapshot = {
            "is_detected": bool(row.get("is_detected", False)),
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "zone": str(row.get("zone", "full")),
            "evidence": _trim_text_local(row.get("evidence", ""), 300),
            "raw_response_preview": _trim_text_local(row.get("raw_response", ""), 500),
        }
        for key in ("cash_path", "roi_confidence", "global_handover_score", "global_keywords"):
            if key in meta:
                snapshot[key] = meta.get(key)
        return snapshot

    payload = {
        "at": now_kst_iso(),
        "camera_id": camera_id,
        "frame_count": int(state.get("frame_count", 0)),
        "source": source,
        "pipeline_version": str(state.get("last_vlm", {}).get("pipeline_version", "")),
        "total_inference_time_ms": float(state.get("last_vlm", {}).get("total_inference_time_ms", 0.0) or 0.0),
        "cashier_zone_points": len(state.get("cashier_zone", []) or []),
        "drawer_zone_points": len(state.get("drawer_zone", []) or []),
        "exchange_band_points": len(state.get("exchange_band", []) or []),
        "staff_work_zone_points": len(state.get("staff_work_zone", []) or []),
        "proposal_summary": _trim_text_local(full_caption, 1000),
        "cash_proposal": _trim_text_local(cash_caption, 1000),
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
        with _proposal_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning("[VLM API] proposal log persist failed (%s): %s", camera_id, e)


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
    ok = bool(postprocessor.submit(payload))
    if not ok and hasattr(postprocessor, "last_reject_reason"):
        payload["_postprocess_reject_reason"] = postprocessor.last_reject_reason()
    return ok


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


def _build_validation_packet(
    camera_id: str,
    scenario_name: str,
    state: dict[str, Any],
    event: dict[str, Any],
    result: dict[str, Any],
    entries: list[dict[str, Any]],
    anchor_mono_ts: Any,
) -> dict[str, Any]:
    sampled = _sample_entries(entries, 8)
    global_keyframes = [e.get("frame") for e in sampled if e.get("frame") is not None]
    cashier_zone = state.get("cashier_zone", []) or []
    drawer_zone = state.get("drawer_zone", []) or []
    exchange_band = state.get("exchange_band", []) or []
    staff_work_zone = state.get("staff_work_zone", []) or []
    result_meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    packet = {
        "episode_id": str(event.get("event_id", "")),
        "camera_id": camera_id,
        "event_type": scenario_name,
        "proposal_type": str(result_meta.get("proposal_type") or event.get("proposal_type") or ""),
        "tier1_confidence": float(result.get("confidence", 0.0) or 0.0),
        "router_action": str(event.get("router_action", "")),
        "router_reason": str(event.get("router_reason", "")),
        "focus_hints": [f"scenario:{scenario_name}", f"zone:{event.get('zone', 'full')}"],
        "anchor_mono_ts": anchor_mono_ts,
        "clip_sec_used": float(state.get("validation_clip_sec", 15) or 15.0),
        "video_window_sec": float(state.get("validation_clip_sec", 15) or 15.0),
        "polygon_coords": result_meta.get("polygon_coords") or {
            "cashier_zone": cashier_zone,
            "drawer_zone": drawer_zone,
            "exchange_band": exchange_band,
            "staff_work_zone": staff_work_zone,
        },
        "yolo_summary": result_meta.get("yolo_summary", {}),
        "skeleton_summary": result_meta.get("skeleton_summary", {}),
        "candidate_clip_paths": event.get("candidate_clip_paths")
        or result_meta.get("candidate_clip_paths", {}),
        "global_keyframes": global_keyframes,
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
    req_base_fps = float(body.get("base_fps", float(getattr(server_config, "BASE_FPS", 1.0) or 1.0)))
    req_clip_buffer_fps = float(body.get("clip_buffer_fps", state.get("clip_buffer_fps", 12.0)))
    req_clip_buffer_fps = max(2.0, min(30.0, req_clip_buffer_fps))
    req_transport = body.get("rtsp_transport", "tcp")
    req_open_timeout_ms = int(body.get("open_timeout_ms", 8000))
    req_read_timeout_ms = int(body.get("read_timeout_ms", 8000))
    req_event_cooldown_sec = int(body.get("event_cooldown_sec", 20))
    req_clip_duration_sec = int(body.get("clip_duration_sec", 10))
    req_validation_clip_sec = int(body.get("validation_clip_sec", getattr(server_config, "V3_VALIDATION_CLIP_SEC", 15)))
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
        state["server_start_time"] = now_kst_iso()
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

    Preview is intentionally decoupled from the analysis pipeline:
    - YOLO ingest uses the original ring-buffer frame (up to INGEST_DOWNSAMPLE_HEIGHT).
    - MJPEG here downsamples to FRONTEND_MJPEG_WIDTH (cv2.resize returns a new array,
      so the YOLO path is unaffected).
    - FPS ramps up to FRONTEND_MJPEG_BURST_FPS during INFERENCE_ACTIVE_BURST_SEC
      after a detection so the operator sees smooth motion when it matters.
    - Identical back-to-back frames are skipped when FRONTEND_MJPEG_DEDUP_FRAMES=true.
    """
    base_fps = max(1.0, min(30.0, float(getattr(server_config, "FRONTEND_MJPEG_FPS", 3.0) or 3.0)))
    burst_fps_cfg = float(getattr(server_config, "FRONTEND_MJPEG_BURST_FPS", 0.0) or 0.0)
    burst_fps = max(base_fps, min(30.0, burst_fps_cfg)) if burst_fps_cfg > 0 else base_fps
    jpeg_quality = max(10, min(95, int(getattr(server_config, "FRONTEND_MJPEG_QUALITY", 50) or 50)))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    preview_width = int(getattr(server_config, "FRONTEND_MJPEG_WIDTH", 0) or 0)
    dedup_enabled = bool(getattr(server_config, "FRONTEND_MJPEG_DEDUP_FRAMES", True))
    burst_window_sec = float(getattr(server_config, "INFERENCE_ACTIVE_BURST_SEC", 3.0) or 3.0)
    idle_pause_sec = float(getattr(server_config, "FRONTEND_MJPEG_IDLE_PAUSE_SEC", 5.0) or 5.0)

    def _downsample(frame):
        if preview_width <= 0 or frame is None or frame.shape[1] <= preview_width:
            return frame
        scale = preview_width / float(frame.shape[1])
        return cv2.resize(
            frame,
            (preview_width, max(2, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    def frame_generator():
        srv = _get_server_modules()
        state = _get_or_create_state(camera_id)
        last_send = 0.0
        last_frame_id = None  # object identity of previously-sent frame

        while True:
            if bool(getattr(srv, "is_shutting_down", False)):
                break
            now = time.time()

            # A3: adaptive fps. Bump to burst_fps inside active-burst window.
            burst_end = float(state.get("mjpeg_burst_until", 0.0) or 0.0)
            target_fps = burst_fps if burst_end > now else base_fps
            interval = 1.0 / max(target_fps, 0.5)
            if now - last_send < interval:
                time.sleep(min(0.02, interval * 0.25))
                continue

            frame = None
            if state["running"] and srv.stream_manager:
                frame = srv.stream_manager.get_frame(camera_id)

            # A2: dedup — stream_manager returns same ndarray ref when no new
            # frame arrived. Skip encode+send entirely to save CPU and bandwidth.
            if (
                dedup_enabled
                and frame is not None
                and id(frame) == last_frame_id
            ):
                # Long idle: still heartbeat so client MJPEG multipart stays alive
                if now - last_send < idle_pause_sec:
                    time.sleep(min(0.05, interval * 0.5))
                    continue

            if frame is not None:
                try:
                    preview = _downsample(frame)  # A1: MJPEG-only resize
                    _, jpeg = cv2.imencode(".jpg", preview, encode_params)
                except Exception:
                    continue
                last_frame_id = id(frame)
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                msg = "Waiting for stream..." if not state["running"] else "Connecting..."
                cv2.putText(
                    blank, msg, (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2,
                )
                _, jpeg = cv2.imencode(".jpg", blank)
                last_frame_id = None

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
                "server_time": now_kst_iso(),
                "current_fps": 0.0,
                "stream_fps": 0.0,
                "base_fps": float(state.get("base_fps", 1.5)),
                "clip_buffer_fps": float(state.get("clip_buffer_fps", 12.0)),
                "event_cooldown_sec": int(state.get("event_cooldown_sec", 20)),
                "validation_clip_sec": int(state.get("validation_clip_sec", 15)),
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
                "pipeline_mode": "hio_v3_yolo26_siglip_temporal" if getattr(srv, "v3_pipeline", None) is not None else "v3_unloaded",
                "tier1_backend": "YOLO26" if getattr(srv, "v3_pipeline", None) is not None else "legacy",
                "v3_pipeline_loaded": getattr(srv, "v3_pipeline", None) is not None,
                "inference_thread_alive": bool(scheduler_metrics.get("workers_alive")),
                "cashier_zone_points": len(state.get("cashier_zone", []) or []),
                "drawer_zone_points": len(state.get("drawer_zone", []) or []),
                "exchange_band_points": len(state.get("exchange_band", []) or []),
                "staff_work_zone_points": len(state.get("staff_work_zone", []) or []),
                "scheduler": scheduler_metrics,
                "postprocess": postprocess_metrics,
            }

        pipeline_mode = "hio_v3_yolo26_siglip_temporal" if getattr(srv, "v3_pipeline", None) is not None else "v3_unloaded"
        tier1_backend = "YOLO26" if getattr(srv, "v3_pipeline", None) is not None else "legacy"

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
            "server_time": now_kst_iso(),
            "current_fps": state["current_fps"],
            "stream_fps": stream_fps,
            "base_fps": state["base_fps"],
            "clip_buffer_fps": state.get("clip_buffer_fps", 12.0),
            "event_cooldown_sec": state["event_cooldown_sec"],
            "validation_clip_sec": state["validation_clip_sec"],
            "evidence_mode": state["evidence_mode"],
            "last_error": state["last_error"],
            "model_health": state.get("model_health", {}),
            "last_frame_age_sec": state["last_frame_age_sec"],
            "last_overlay_age_sec": state["last_overlay_age_sec"],
            "last_vlm": state["last_vlm"],
            "last_validation": last_validation,
            "last_clip_path": last_clip_path,
            "recent_events": recent_events,
            "buffers": buffers,
            "audit_log_dir": str(srv.config.LOG_DIR) if hasattr(srv, "config") else "",
            "router": {"mode": "v3_gemini_temporal"},
            "pipeline_mode": pipeline_mode,
            "tier1_backend": tier1_backend,
            "v3_pipeline_loaded": getattr(srv, "v3_pipeline", None) is not None,
            "inference_thread_alive": inference_thread_alive,
            "cashier_zone_points": len(state.get("cashier_zone", []) or []),
            "drawer_zone_points": len(state.get("drawer_zone", []) or []),
            "exchange_band_points": len(state.get("exchange_band", []) or []),
            "staff_work_zone_points": len(state.get("staff_work_zone", []) or []),
            "scheduler": scheduler_metrics,
            "postprocess": postprocess_metrics,
        }
    except Exception as e:
        logger.warning(f"[VLM API] status error ({camera_id}): {e}")
        return JSONResponse(
            {
                "running": False,
                "status": "error",
                "server_time": now_kst_iso(),
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
    srv = _get_server_modules()
    pipeline = getattr(srv, "v3_pipeline", None)
    pipeline_health: dict[str, Any] = {}
    if pipeline is not None and hasattr(pipeline, "health"):
        try:
            pipeline_health = pipeline.health() or {}
        except Exception as exc:  # noqa: BLE001
            pipeline_health = {"error": str(exc)}
    cfg = {
        "hio_v3_enabled": bool(config.HIO_V3_ENABLED),
        "pipeline_version": str(getattr(config, "HIO_V3_PIPELINE_VERSION", "")),
        "v3_pipeline_loaded": pipeline is not None,
        "tier1_backend": "YOLO26" if bool(config.HIO_V3_ENABLED) else "legacy",
        "yolo26_pose_weights": str(config.YOLO26_POSE_WEIGHTS),
        "yolo26_detect_weights": str(config.YOLO26_DETECT_WEIGHTS),
        "yolo26_cash_weights": str(config.YOLO26_CASH_WEIGHTS),
        "yolo26_fire_weights": str(config.YOLO26_FIRE_WEIGHTS),
        "yolo26_device": str(config.YOLO26_DEVICE),
        "yolo26_imgsz": int(config.YOLO26_IMGSZ),
        "semantic_filter_enabled": bool(getattr(config, "V3_SEMANTIC_FILTER_ENABLED", True)),
        "semantic_model": str(getattr(config, "V3_SEMANTIC_MODEL", "")),
        "semantic_filter": pipeline_health.get("semantic_filter", {}),
        "cash_siglip_clip_enabled": bool(getattr(config, "V3_CASH_SIGLIP_CLIP_ENABLED", True)),
        "cash_siglip_clip_frames": int(getattr(config, "V3_CASH_SIGLIP_CLIP_FRAMES", 12) or 12),
        "cash_siglip_clip_batch_size": int(getattr(config, "V3_CASH_SIGLIP_CLIP_BATCH_SIZE", 4) or 4),
        "cash_siglip_clip_window_sec": float(getattr(config, "V3_CASH_SIGLIP_CLIP_WINDOW_SEC", 15.0) or 15.0),
        "cash_siglip_clip_min_score": float(getattr(config, "V3_CASH_SIGLIP_CLIP_MIN_SCORE", 0.50) or 0.50),
        "fire_classifier": pipeline_health.get("fire_classifier", {}),
        "action_classifier": pipeline_health.get("action_classifier", {}),
        "gemini_model": config.GEMINI_MODEL,
        "base_fps": state["base_fps"],
        "default_base_fps": float(getattr(config, "BASE_FPS", 1.0) or 1.0),
        "default_burst_fps": float(getattr(config, "BURST_FPS", 3.0) or 3.0),
        "clip_buffer_fps": state.get("clip_buffer_fps", 12.0),
        "clip_duration_sec": state["clip_duration_sec"],
        "validation_clip_sec": state["validation_clip_sec"],
        "evidence_mode": state["evidence_mode"],
        "cash_threshold": config.V3_CASH_PREFILTER_THRESHOLD,
        "violence_threshold": config.V3_VIOLENCE_PREFILTER_THRESHOLD,
        "fire_threshold": config.V3_FIRE_PREFILTER_THRESHOLD,
        "ingest_downsample_height": int(getattr(config, "INGEST_DOWNSAMPLE_HEIGHT", 0) or 0),
        "overlay_fps_divisor": int(getattr(config, "V3_OVERLAY_FPS_DIVISOR", 3) or 3),
        "ffmpeg_preset": str(getattr(config, "FFMPEG_PRESET", "ultrafast")),
        "ffmpeg_crf": int(getattr(config, "FFMPEG_CRF", 28) or 28),
        "frontend_mjpeg_fps": float(getattr(config, "FRONTEND_MJPEG_FPS", 3.0) or 3.0),
        "frontend_mjpeg_burst_fps": float(getattr(config, "FRONTEND_MJPEG_BURST_FPS", 0.0) or 0.0),
        "frontend_mjpeg_quality": int(getattr(config, "FRONTEND_MJPEG_QUALITY", 50) or 50),
        "frontend_mjpeg_width": int(getattr(config, "FRONTEND_MJPEG_WIDTH", 0) or 0),
        "frontend_mjpeg_dedup": bool(getattr(config, "FRONTEND_MJPEG_DEDUP_FRAMES", True)),
        "flush_interval_sec": int(getattr(config, "FLUSH_INTERVAL_SEC", 120) or 120),
        "rtsp_url": state["rtsp_url"],
        "rtsp_transport": state["rtsp_transport"],
        "rtsp_hwaccel_decoder": str(getattr(config, "RTSP_HWACCEL_DECODER", "") or ""),
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
@router.post("/flush-now/")
def vlm_flush_now(include_today: bool = True):
    """Manual flush trigger. Spot interruption watcher calls this with
    include_today=true before systemctl stop so the day-in-progress drains to DB.
    """
    srv = _get_server_modules()
    flush_worker = getattr(srv, "flush_worker", None)
    if flush_worker is None:
        return JSONResponse({"ok": False, "error": "flush_worker not initialized"}, status_code=503)
    try:
        summary = flush_worker.flush(include_today=bool(include_today))
        return {"ok": True, "summary": summary, "include_today": bool(include_today)}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.get("/events/")
def vlm_events(
    limit: int = 50,
    date: str | None = None,
    kst_date: str | None = None,
):
    srv = _get_server_modules()

    events: list[dict[str, Any]] = []
    if srv.local_storage:
        if kst_date:
            # Older events were bucketed by UTC date, so a single KST day spans
            # two on-disk dirs. Read both, filter by the event_id epoch (which
            # is unambiguous UTC ms), and return only rows in the KST window.
            try:
                yyyy, mm, dd = int(kst_date[0:4]), int(kst_date[4:6]), int(kst_date[6:8])
                kst_start = datetime(yyyy, mm, dd, 0, 0, 0, tzinfo=KST)
                kst_end = kst_start + timedelta(days=1)
                start_ms = int(kst_start.timestamp() * 1000)
                end_ms = int(kst_end.timestamp() * 1000)
            except Exception:
                start_ms = end_ms = 0
            if end_ms > start_ms:
                day_before = (datetime(yyyy, mm, dd) - timedelta(days=1)).strftime("%Y%m%d")
                # epoch_ms_min/max lets list_events reject files by filename
                # before any disk read, so we only pay for events actually in
                # the requested KST window.
                bucket_limit = max(limit, 5000)
                pool = srv.local_storage.list_events(
                    date_str=day_before,
                    limit=bucket_limit,
                    epoch_ms_min=start_ms,
                    epoch_ms_max=end_ms,
                )
                pool += srv.local_storage.list_events(
                    date_str=kst_date,
                    limit=bucket_limit,
                    epoch_ms_min=start_ms,
                    epoch_ms_max=end_ms,
                )

                def _epoch_ms(ev: dict[str, Any]) -> int:
                    eid = str(ev.get("event_id", ""))
                    parts = eid.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
                    return 0

                seen: set[str] = set()
                filtered: list[dict[str, Any]] = []
                for ev in pool:
                    eid = str(ev.get("event_id", ""))
                    if eid in seen:
                        continue
                    seen.add(eid)
                    filtered.append(ev)
                filtered.sort(key=_epoch_ms, reverse=True)
                events = filtered[: max(1, limit)]
        else:
            events = srv.local_storage.list_events(date_str=date, limit=limit)
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


@router.post("/proposal-feedback/")
async def vlm_proposal_feedback(request: Request):
    """
    v3 proposal feedback endpoint wired directly to the proposal feedback collector.
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
        event_id = f"proposal_feedback_{int(time.time() * 1000)}_{camera_id or 'unknown'}"

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

    result = srv.data_collector.collect_proposal_feedback(
        event_id=event_id,
        decision=decision,
        note=note,
        frame=frame,
        caption=caption,
        scenario=scenario,
        camera_id=camera_id,
        summary=summary,
        source="v3_proposal_feedback",
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
    state["exchange_band"] = _normalize_zone(body.get("exchange_band", []))
    state["staff_work_zone"] = _normalize_zone(body.get("staff_work_zone", []))

    logger.info(
        f"[VLM API] Zones updated for {camera_id}: "
        f"cashier={len(state['cashier_zone'])} pts, "
        f"drawer={len(state['drawer_zone'])} pts, "
        f"exchange_band={len(state['exchange_band'])} pts, "
        f"staff_work_zone={len(state['staff_work_zone'])} pts"
    )
    return {
        "success": True,
        "cashier_zone_points": len(state["cashier_zone"]),
        "drawer_zone_points": len(state["drawer_zone"]),
        "exchange_band_points": len(state["exchange_band"]),
        "staff_work_zone_points": len(state["staff_work_zone"]),
    }


# ---------------------------------------------------------------------------
# GET /api/vlm/roi-preview/ - Zone ROI preview
# ---------------------------------------------------------------------------
@router.get("/roi-preview/")
async def vlm_roi_preview(zone: str = "cashier", camera_id: str = "adhoc_cam"):
    """Return a zone-focused JPEG for v3 ROI verification in the UI."""
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

    # Pick the requested zone polygon. exchange_band/staff_work_zone are not
    # named with the legacy "<name>_zone" convention.
    zone_key = {
        "cashier": "cashier_zone",
        "drawer": "drawer_zone",
        "exchange": "exchange_band",
        "exchange_band": "exchange_band",
        "staff_work": "staff_work_zone",
        "staff_work_zone": "staff_work_zone",
    }.get(str(zone or "").strip().lower(), f"{zone}_zone")
    zone_polygon = state.get(zone_key, [])

    if len(zone_polygon) >= 3:
        try:
            pts = np.array(zone_polygon, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + max(1, w))
            y2 = min(frame.shape[0], y + max(1, h))
            preview = frame[y:y2, x:x2].copy()
            if preview.size == 0:
                preview = frame.copy()
                label = f"full frame ({zone} ROI preview empty)"
            else:
                label = f"{zone} ROI preview {preview.shape[1]}x{preview.shape[0]}"
        except Exception:
            preview = frame.copy()
            label = f"full frame ({zone} ROI preview failed)"
    else:
        preview = frame.copy()
        label = f"full frame (no {zone} zone set)"

    # Burn label into image
    cv2.putText(
        preview, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

    # FP subtype validation (ported from v1 labeling schema).
    # Allowed set mirrors labeling.html ALLOWED_ERROR so UI and server stay in sync.
    _FP_ERROR_TYPES = {
        "phone_or_device",
        "receipt_or_paper",
        "card",
        "empty_scene",
        "staff_only",
        "no_transfer",
        "other",
    }
    error_type = str(body.get("error_type", "") or "").strip().lower()
    if error_type and error_type not in _FP_ERROR_TYPES:
        return JSONResponse(
            {"success": False, "error": f"error_type must be one of {sorted(_FP_ERROR_TYPES)} or empty"},
            status_code=400,
        )
    if decision == "decline" and not error_type:
        return JSONResponse(
            {
                "success": False,
                "error": "FP 라벨링에는 error_type이 필요합니다 (phone_or_device/receipt_or_paper/card/empty_scene/staff_only/no_transfer/other).",
            },
            status_code=400,
        )
    if decision in {"decline", "unsure"} and not note:
        return JSONResponse(
            {"success": False, "error": "FP / Unclear 라벨링에는 한 줄 이상의 note가 필요합니다."},
            status_code=400,
        )

    srv = _get_server_modules()
    state = _get_or_create_state(camera_id)

    now_iso = now_kst_iso()
    feedback_obj = {
        "decision": decision,
        "note": note,
        "error_type": error_type,
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

    # ── v3 proposal feedback collection ──
    if srv.data_collector is not None:
        try:
            srv.data_collector.collect_feedback(
                event_id=event_id,
                decision=decision,
                note=note,
                scenario=body.get("scenario", ""),
            )
        except Exception as dc_err:
            logger.debug(f"[VLM API] v3 feedback collector error: {dc_err}")

    return {"success": True}


# ---------------------------------------------------------------------------
# One-shot inference + async post-processing
# ---------------------------------------------------------------------------
def _is_cash_clip_siglip_candidate(cash_result: dict[str, Any]) -> bool:
    if not isinstance(cash_result, dict) or bool(cash_result.get("is_detected")):
        return False
    meta = cash_result.get("metadata", {}) if isinstance(cash_result.get("metadata"), dict) else {}
    exchange_features = meta.get("exchange_features", {}) if isinstance(meta.get("exchange_features"), dict) else {}
    recent_hit_count = int(exchange_features.get("recent_hit_count", 0) or 0)
    min_hits = int(exchange_features.get("min_hits_required", 2) or 2)
    return bool(
        exchange_features.get("soft_hit")
        and recent_hit_count >= max(1, min_hits)
        and not bool(exchange_features.get("staff_work_only"))
        and not bool(exchange_features.get("staff_only_near_exchange"))
    )


def _apply_cash_clip_siglip_result(cash_result: dict[str, Any], clip_result: dict[str, Any]) -> bool:
    if not isinstance(cash_result, dict) or not isinstance(clip_result, dict):
        return False
    meta = cash_result.setdefault("metadata", {})
    meta["cash_siglip_clip"] = clip_result
    local_prefilter = meta.setdefault("local_prefilter", {})
    local_prefilter["cash_siglip_clip"] = {
        "enabled": bool(clip_result.get("enabled")),
        "passed": bool(clip_result.get("passed")),
        "aggregate_score": clip_result.get("aggregate_score", 0.0),
        "positive_frame_count": clip_result.get("positive_frame_count", 0),
        "frame_count_scored": clip_result.get("frame_count_scored", 0),
    }
    if not bool(clip_result.get("passed")):
        return False

    try:
        clip_score = float(clip_result.get("aggregate_score", 0.0) or 0.0)
    except Exception:
        clip_score = 0.0
    try:
        current_score = float(cash_result.get("confidence", 0.0) or 0.0)
    except Exception:
        current_score = 0.0
    promoted_score = max(current_score, clip_score)
    reasons = list(cash_result.get("matched_keywords") or [])
    if "cash_siglip_full_clip_passed" not in reasons:
        reasons.append("cash_siglip_full_clip_passed")

    cash_result["is_detected"] = True
    cash_result["confidence"] = round(promoted_score, 3)
    cash_result["matched_keywords"] = reasons
    cash_result["evidence"] = ", ".join(reasons)
    cash_result["raw_response"] = "YOLO26 temporal cash proposal + full-clip SigLIP: " + ", ".join(reasons)
    local_prefilter["passed"] = True
    local_prefilter["cash_siglip_clip_promoted"] = True
    local_prefilter["score"] = round(promoted_score, 3)
    return True


def _run_inference_once(camera_id: str, frame: Any, state: dict[str, Any], started_at: float) -> None:
    srv = _get_server_modules()

    state["frame_count"] = int(state.get("frame_count", 0)) + 1
    state["last_frame_age_sec"] = 0.0
    prev_started = state.get("last_inference_started_at")
    if prev_started:
        state["current_fps"] = 1.0 / max(float(started_at) - float(prev_started), 0.001)
    state["last_inference_started_at"] = started_at

    cash_zone_applied = False
    scenario_results: dict[str, dict[str, Any]] = {}
    full_caption = ""
    cash_caption = ""

    if getattr(srv, "v3_pipeline", None) is not None:
        try:
            zones = {
                "cashier": state.get("cashier_zone", []),
                "drawer": state.get("drawer_zone", []),
                "exchange": state.get("exchange_band", []),
                "staff_work": state.get("staff_work_zone", []),
            }
            inference_ctx = (
                _inference_lock
                if bool(getattr(server_config, "GLOBAL_INFERENCE_LOCK", True))
                else nullcontext()
            )
            with inference_ctx:
                v3_result = srv.v3_pipeline.process_frame(camera_id, frame, zones)

            scenario_results = dict(v3_result.scenario_results)
            cash_clip_siglip_result: dict[str, Any] = {}
            if (
                bool(getattr(server_config, "V3_CASH_SIGLIP_CLIP_ENABLED", True))
                and _is_cash_clip_siglip_candidate(scenario_results.get("cash", {}) or {})
                and srv.stream_manager is not None
            ):
                now_wall = time.time()
                cooldown_sec = max(
                    0.5,
                    float(getattr(server_config, "V3_CASH_SIGLIP_CLIP_COOLDOWN_SEC", 2.0) or 2.0),
                )
                if now_wall - float(state.get("last_cash_clip_siglip_ts", 0.0) or 0.0) >= cooldown_sec:
                    state["last_cash_clip_siglip_ts"] = now_wall
                    try:
                        clip_window_sec = float(
                            getattr(server_config, "V3_CASH_SIGLIP_CLIP_WINDOW_SEC", 15.0) or 15.0
                        )
                        if clip_window_sec <= 0:
                            clip_window_sec = float(state.get("validation_clip_sec", 15) or 15.0)
                        clip_entries = srv.stream_manager.get_clip_frames(
                            camera_id,
                            window_sec=clip_window_sec,
                            anchor_mono_ts=time.monotonic(),
                        )
                        if clip_entries:
                            clip_siglip_started = time.time()
                            with inference_ctx:
                                cash_clip_siglip_result = srv.v3_pipeline.score_cash_clip(clip_entries)
                            cash_clip_siglip_result["processing_time_ms"] = round(
                                (time.time() - clip_siglip_started) * 1000.0,
                                1,
                            )
                            promoted = _apply_cash_clip_siglip_result(
                                scenario_results.get("cash", {}) or {},
                                cash_clip_siglip_result,
                            )
                            if promoted:
                                logger.info(
                                    "[VLM API] Cash promoted by full-clip SigLIP (%s): score=%.3f pos_frames=%s/%s",
                                    camera_id,
                                    float(cash_clip_siglip_result.get("aggregate_score", 0.0) or 0.0),
                                    cash_clip_siglip_result.get("positive_frame_count", 0),
                                    cash_clip_siglip_result.get("frame_count_scored", 0),
                                )
                    except Exception as clip_siglip_err:
                        cash_clip_siglip_result = {
                            "enabled": False,
                            "passed": False,
                            "error": str(clip_siglip_err),
                        }
                        logger.warning(
                            "[VLM API] cash full-clip SigLIP failed (%s): %s",
                            camera_id,
                            clip_siglip_err,
                        )
            cash_zone_applied = len(zones["cashier"]) >= 3
            full_caption = "HIO v3 YOLO26 + SigLIP semantic/full-clip cash + temporal proposal"
            cash_caption = str((scenario_results.get("cash", {}) or {}).get("raw_response", ""))
            yolo_summary = dict(v3_result.metadata.get("yolo_summary", {}) or {})
            yolo_errors = list(yolo_summary.get("errors") or [])
            state["model_health"] = yolo_summary.get("model_health", {})
            if yolo_errors:
                state["last_error"] = "; ".join(str(v) for v in yolo_errors[-3:])
            state["last_vlm"] = {
                "scenario_results": scenario_results,
                "total_inference_time_ms": float(v3_result.total_inference_time_ms),
                "yolo_summary": yolo_summary,
                "cashier_zone_applied": cash_zone_applied,
                "cashier_zone_points": len(zones["cashier"]),
                "drawer_zone_points": len(zones["drawer"]),
                "exchange_band_points": len(zones["exchange"]),
                "staff_work_zone_points": len(zones["staff_work"]),
                "cash_siglip_clip": cash_clip_siglip_result,
                "shared_caption": full_caption,
                "cash_caption": cash_caption,
                "source": "hio_v3_yolo26_siglip_temporal",
                "pipeline_version": str(
                    getattr(server_config, "HIO_V3_PIPELINE_VERSION", "v3-yolo26-tier1-siglip-episode-gemini")
                ),
            }
        except Exception as e:
            state["last_error"] = f"HIO v3 pipeline error: {e}"
            logger.exception("[VLM API] HIO v3 pipeline error for %s", camera_id)
            return
    else:
        if state["frame_count"] == 1:
            state["last_error"] = "HIO v3 pipeline not loaded."
        return

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
        result_meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        event = {
            "event_id": event_id,
            "at": now_kst_iso(),
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
            "exchange_band_used": bool(scenario_name == "cash" and len(state.get("exchange_band", []) or []) >= 3),
            "staff_work_zone_used": bool(scenario_name == "cash" and len(state.get("staff_work_zone", []) or []) >= 3),
            "camera_id": camera_id,
            "postprocess_state": "pending",
        }
        if result_meta.get("v3") or result_meta.get("pipeline_version"):
            event.update({
                "pipeline_version": str(
                    result_meta.get("pipeline_version")
                    or getattr(server_config, "HIO_V3_PIPELINE_VERSION", "v3-yolo26-tier1-siglip-episode-gemini")
                ),
                "proposal_type": str(result_meta.get("proposal_type", "")),
                "tier1_confidence": float(result.get("confidence", 0.0) or 0.0),
                "yolo_summary": result_meta.get("yolo_summary", {}),
                "skeleton_summary": result_meta.get("skeleton_summary", {}),
                "polygon_coords": result_meta.get("polygon_coords", {}),
                "candidate_clip_paths": dict(result_meta.get("candidate_clip_paths", {}) or {}),
                "local_prefilter": result_meta.get("local_prefilter", {}),
            })

        needs_tier2 = False
        if result_meta.get("v3") or result_meta.get("pipeline_version"):
            needs_tier2 = bool(getattr(server_config, "V3_GEMINI_ALWAYS_VALIDATE", True))
            event["router_action"] = "V3_GEMINI_TEMPORAL"
            event["router_reason"] = "v3 semantic + temporal prefilter passed; Gemini validation enabled"
        else:
            needs_tier2 = False
            event["router_action"] = "V3_LOCAL_ONLY"
            event["router_reason"] = "non-v3 proposal metadata"

        event["gemini"]["state"] = "needed" if needs_tier2 else "skipped"
        if not isinstance(state.get("last_validation"), dict):
            state["last_validation"] = {}
        state["last_validation"][scenario_name] = dict(event.get("gemini", {}))

        state["recent_events"].append(event)
        if len(state["recent_events"]) > 100:
            state["recent_events"] = state["recent_events"][-100:]
        _persist_event(camera_id, event)

        admission_mono_ts = time.monotonic()
        admission_clip_entries: list[dict[str, Any]] = []
        if srv.stream_manager:
            try:
                admission_clip_entries = srv.stream_manager.get_clip_frames(
                    camera_id,
                    window_sec=float(state.get("validation_clip_sec", 15)),
                    anchor_mono_ts=admission_mono_ts,
                )
            except Exception as clip_capture_err:
                logger.warning(
                    "[VLM API] Admission clip capture failed (%s/%s): %s",
                    camera_id,
                    event_id,
                    clip_capture_err,
                )

        payload = {
            "camera_id": camera_id,
            "scenario_name": scenario_name,
            "event": event,
            "result": result,
            "frame": frame,
            "anchor_mono_ts": admission_mono_ts,
            "admission_queued_at": time.time(),
            "admission_clip_entries": admission_clip_entries,
            "needs_tier2": bool(needs_tier2),
        }
        queued = _queue_detection_postprocess(payload)
        if queued:
            event["postprocess_state"] = "queued"
            _persist_event(camera_id, event)
        else:
            reject_reason = str(payload.get("_postprocess_reject_reason") or "postprocess queue unavailable")
            if needs_tier2:
                event["gemini"]["state"] = "skipped" if reject_reason == "duplicate_pending" else "error"
                event["gemini"]["reason"] = reject_reason
                state["last_validation"][scenario_name] = dict(event["gemini"])
            event["postprocess_state"] = "dropped" if reject_reason == "duplicate_pending" else "error"
            event["postprocess_reject_reason"] = reject_reason
            if reject_reason == "duplicate_pending":
                event["is_detected"] = False
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

        # A3: tell the MJPEG generator to bump to FRONTEND_MJPEG_BURST_FPS for
        # the active-burst window so the operator sees smooth motion during the
        # event. Outside this window the stream ticks at base FRONTEND_MJPEG_FPS.
        try:
            burst_sec = float(getattr(server_config, "INFERENCE_ACTIVE_BURST_SEC", 3.0) or 3.0)
            state["mjpeg_burst_until"] = time.time() + max(1.0, burst_sec)
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


def _process_detection_event(payload: dict[str, Any]) -> None:
    srv = _get_server_modules()
    camera_id = str(payload.get("camera_id", "")).strip()
    scenario_name = str(payload.get("scenario_name", "")).strip().lower()
    event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    result_meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    frame = payload.get("frame")
    anchor_mono_ts = payload.get("anchor_mono_ts")
    admission_queued_at = float(payload.get("admission_queued_at") or time.time())
    needs_tier2 = bool(payload.get("needs_tier2"))
    state = _get_or_create_state(camera_id)

    if not event or not camera_id or not scenario_name:
        return

    event["postprocess_state"] = "processing"
    _persist_event(camera_id, event)

    clip_frames_for_feedback: list[Any] = []
    val_clip_path = None
    payload_entries = payload.get("admission_clip_entries")
    val_entries: list[dict[str, Any]] = list(payload_entries) if isinstance(payload_entries, list) else []
    postprocess_ok = True

    # Zone overlays are kept for UI/forensics only. Gemini evidence is built
    # by CandidateClipBuilder as full-frame raw/context overlay clips.
    overlay_polygons: dict[str, list[list[int]]] = {}
    if scenario_name == "cash":
        cz = state.get("cashier_zone", []) or []
        dz = state.get("drawer_zone", []) or []
        eb = state.get("exchange_band", []) or []
        sw = state.get("staff_work_zone", []) or []
        if len(cz) >= 3:
            overlay_polygons["cashier"] = list(cz)
        if len(dz) >= 3:
            overlay_polygons["drawer"] = list(dz)
        if len(eb) >= 3:
            overlay_polygons["exchange"] = list(eb)
        if len(sw) >= 3:
            overlay_polygons["staff_work"] = list(sw)
    event["overlay_applied"] = bool(overlay_polygons)
    v3_zones = {
        "cashier": (result_meta.get("polygon_coords", {}) or {}).get("cashier_zone")
        or state.get("cashier_zone", [])
        or [],
        "drawer": (result_meta.get("polygon_coords", {}) or {}).get("drawer_zone")
        or state.get("drawer_zone", [])
        or [],
        "exchange": (result_meta.get("polygon_coords", {}) or {}).get("exchange_band")
        or state.get("exchange_band", [])
        or [],
        "staff_work": (result_meta.get("polygon_coords", {}) or {}).get("staff_work_zone")
        or state.get("staff_work_zone", [])
        or [],
    }

    try:
        if needs_tier2 and srv.gemini_validator is not None:
            try:
                _tier2_validation_slots.acquire()
                try:
                    val_seconds = float(state.get("validation_clip_sec", 15))
                    queue_latency = max(0.0, time.time() - admission_queued_at)
                    if queue_latency > val_seconds:
                        raise RuntimeError(
                            f"validation_error: postprocess queue latency {queue_latency:.1f}s exceeded clip window {val_seconds:.1f}s"
                        )
                    if not val_entries:
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
                            # Keep the validation clip as full-frame raw CCTV.
                            # Gemini gets overlays through candidate_clip_paths,
                            # not through cut-out video or burned-in validation clip.
                            val_clip_path = _save_clip_serialized(
                                srv.local_storage,
                                f"val_{event['event_id']}",
                                val_frames,
                                fps=v_fps,
                                allow_s3=False,
                                overlay_polygons=None,
                            )
                            if result_meta.get("v3") or result_meta.get("pipeline_version"):
                                try:
                                    from model_server.candidates.clip_builder import CandidateClipBuilder

                                    candidate_paths = CandidateClipBuilder(srv.local_storage).save_candidate_clips(
                                        event_id=str(event["event_id"]),
                                        entries=val_entries,
                                        fps=v_fps,
                                        zones=v3_zones,
                                        skeleton_summary=result_meta.get("skeleton_summary", {}),
                                        raw_path=val_clip_path,
                                    )
                                    if candidate_paths:
                                        if val_clip_path:
                                            candidate_paths.setdefault("validation_clip", val_clip_path)
                                        event["candidate_clip_paths"] = candidate_paths
                                        result_meta["candidate_clip_paths"] = candidate_paths
                                        _persist_event(camera_id, event)
                                except Exception as clip_build_err:
                                    logger.warning(
                                        "[VLM API] v3 candidate clip build failed (%s/%s): %s",
                                        camera_id,
                                        event.get("event_id"),
                                        clip_build_err,
                                    )

                    validation_packet = _build_validation_packet(
                        camera_id,
                        scenario_name,
                        state,
                        event,
                        result,
                        val_entries,
                        anchor_mono_ts,
                    )
                    gemini_ok, gemini_conf, gemini_reason, corrected_event_type = (
                        srv.gemini_validator.validate_event_evidence(
                            packet=validation_packet,
                            mode="video_only",
                            video_path=val_clip_path,
                            frame=frame,
                        )
                    )
                finally:
                    _tier2_validation_slots.release()

                corrected_event_type = _normalize_corrected_event_type(
                    corrected_event_type,
                    scenario_name,
                )
                scenario_corrected = False
                correction_rejected = False
                if gemini_ok:
                    if corrected_event_type not in _SUPPORTED_CORRECTED_EVENT_TYPES:
                        correction_rejected = True
                        gemini_ok = False
                        gemini_conf = min(float(gemini_conf or 0.0), 0.20)
                        gemini_reason = (
                            f"{gemini_reason}; rejected: unsupported Gemini "
                            f"event_type_detected={corrected_event_type}"
                        )
                    elif corrected_event_type == "none":
                        correction_rejected = True
                        gemini_ok = False
                        gemini_conf = min(float(gemini_conf or 0.0), 0.20)
                        gemini_reason = (
                            f"{gemini_reason}; rejected: Gemini corrected "
                            "event_type_detected to none"
                        )
                    elif corrected_event_type != scenario_name:
                        scenario_corrected = True
                        event["original_scenario"] = event.get("original_scenario") or scenario_name
                        event["original_event_type"] = event.get("original_event_type") or scenario_name
                        event["scenario"] = corrected_event_type
                        event["event_type"] = corrected_event_type
                        event["corrected_type"] = corrected_event_type
                        event["scenario_corrected"] = True
                        event["correction_source"] = "gemini_event_type_detected"
                        if corrected_event_type == "cash":
                            event["cashier_zone_used"] = bool(len(v3_zones.get("cashier") or []) >= 3)
                            event["exchange_band_used"] = bool(len(v3_zones.get("exchange") or []) >= 3)
                            event["staff_work_zone_used"] = bool(len(v3_zones.get("staff_work") or []) >= 3)
                        event["router_reason"] = (
                            f"{event.get('router_reason', '')}; Gemini corrected "
                            f"event_type {scenario_name}->{corrected_event_type}"
                        ).strip("; ")
                        gemini_reason = (
                            f"{gemini_reason}; corrected_event_type: "
                            f"{scenario_name}->{corrected_event_type} by Gemini event_type_detected"
                        )

                val_log = getattr(srv.gemini_validator, "last_validation_log", {}) or {}
                event["gemini"] = {
                    "state": "done",
                    "validated": gemini_ok,
                    "confidence": gemini_conf,
                    "reason": gemini_reason,
                    "at": now_kst_iso(),
                    "validation_type": str(val_log.get("input_mode", "")) or "video",
                    "input_mode": str(val_log.get("input_mode", "")) or "video",
                    "prompt_version": str(val_log.get("prompt_version", "")),
                    "processing_time_ms": int(val_log.get("processing_time_ms", 0) or 0),
                    "media_ref": str(val_log.get("media_ref", "")),
                    "event_type_detected": corrected_event_type,
                    "corrected_type": corrected_event_type,
                    "scenario_corrected": scenario_corrected,
                    "corrected_from": scenario_name if scenario_corrected else "",
                    "correction_rejected": correction_rejected,
                }
                if event.get("candidate_clip_paths"):
                    event["gemini"]["candidate_clip_paths"] = dict(event.get("candidate_clip_paths", {}) or {})
                if not gemini_ok:
                    reason_lower = str(gemini_reason or "").lower()
                    is_validation_error = (
                        _is_api_error_reason(gemini_reason)
                        or "validation disabled" in reason_lower
                        or "no validation clip" in reason_lower
                        or "validation_error" in reason_lower
                    )
                    logger.info(
                        "[VLM API] Gemini %s (%s): %s conf=%.2f reason=%s",
                        "ERROR" if is_validation_error else "REJECTED",
                        camera_id,
                        scenario_name,
                        float(gemini_conf or 0.0),
                        str(gemini_reason or "")[:80],
                    )
                    event["is_detected"] = False
                    if is_validation_error:
                        event["validation_error"] = True
                        event["gemini"]["state"] = "error"
                    else:
                        event["rejected_by_gemini"] = True
            except Exception as gem_err:
                postprocess_ok = False
                logger.warning("[VLM API] Gemini validation error (%s): %s", camera_id, gem_err)
                event["gemini"]["state"] = "error"
                event["gemini"]["reason"] = str(gem_err)
                event["validation_error"] = True
                event["is_detected"] = False

            # Retain val_clip on disk: it is the exact media Gemini scored,
            # and will also serve as fallback if the permanent clip save below
            # ever fails (e.g. ring-buffer eviction race).
        if srv.local_storage and srv.stream_manager:
            try:
                # Align permanent-clip duration with validation clip so UI and
                # Gemini see the same time window.
                clip_seconds = float(state.get("validation_clip_sec", 15))
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
                        clip_frames_for_feedback = clip_frames
                        if len(clip_entries) >= 2:
                            ts_first = float(clip_entries[0].get("mono_ts", 0) or 0)
                            ts_last = float(clip_entries[-1].get("mono_ts", 0) or 0)
                            duration = ts_last - ts_first
                            clip_fps = len(clip_entries) / max(duration, 0.1)
                            clip_fps = min(max(clip_fps, 1.0), 30.0)
                        else:
                            clip_fps = 15.0

                        clip_path = val_clip_path if val_clip_path and os.path.exists(val_clip_path) else None
                        if not clip_path:
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
                    str(event.get("scenario") or scenario_name).lower() == "cash"
                    and gem.get("state") == "done"
                    and bool(gem.get("validated")) is True
                    and clip_frames_for_feedback
                ):
                    srv.data_collector.collect_validated_clip(
                        event_id=event["event_id"],
                        scenario=str(event.get("scenario") or scenario_name).lower(),
                        clip_frames=clip_frames_for_feedback,
                        caption=str(event.get("caption") or ""),
                        camera_id=camera_id,
                        gemini_confidence=float(gem.get("confidence") or 0.0),
                        matched_keywords=list(result.get("matched_keywords") or []),
                        sample_count=3,
                    )
            except Exception as dc_err:
                logger.debug("[VLM API] v3 feedback collector validated-clip error: %s", dc_err)
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
