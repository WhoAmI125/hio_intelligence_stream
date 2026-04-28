"""HIO v3 model server: YOLO26 temporal proposals + Gemini validation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_server import config
from model_server.flush_worker import FlushWorker
from model_server.local_storage import LocalStorage
from model_server.proposal.feedback_collector import V3ProposalFeedbackCollector
from model_server.stream_manager import StreamManager

logger = logging.getLogger("model_server")


class _LocalTZFormatter(logging.Formatter):
    """Logging formatter that renders asctime in the configured TZ.

    Windows does not honor `os.environ['TZ']` for Python's time module, so
    `logging.Formatter`'s default asctime is raw system time (often UTC on
    dev boxes), not the operator's intended timezone. We explicitly use
    `zoneinfo` to render Asia/Seoul (or whatever TZ env var specifies) so
    log timestamps match what the operator sees on their wall clock.
    """

    _tz = None

    def _resolve_tz(self):
        if self._tz is not None:
            return self._tz
        tz_name = os.getenv("TZ", "Asia/Seoul")
        try:
            from zoneinfo import ZoneInfo
            self._tz = ZoneInfo(tz_name)
        except Exception:
            self._tz = None
        return self._tz

    def formatTime(self, record, datefmt=None):
        from datetime import datetime
        tz = self._resolve_tz()
        dt = datetime.fromtimestamp(record.created, tz=tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{int(record.msecs):03d}"


_root = logging.getLogger()
_root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
for _h in list(_root.handlers):
    _root.removeHandler(_h)
_handler = logging.StreamHandler()
_handler.setFormatter(_LocalTZFormatter(config.LOG_FORMAT))
_root.addHandler(_handler)

stream_manager: StreamManager | None = None
local_storage: LocalStorage | None = None
flush_worker: FlushWorker | None = None
gemini_validator = None
v3_pipeline = None
data_collector: V3ProposalFeedbackCollector | None = None
inference_scheduler = None
event_postprocessor = None
ws_clients: list[WebSocket] = []
is_shutting_down: bool = False
startup_restore_task: asyncio.Task[Any] | None = None


def _env_bool_local(key: str, default: bool) -> bool:
    raw = str(os.getenv(key, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float_local(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _env_int_local(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _http_get_json(url: str, timeout_sec: float = 8.0) -> dict[str, Any]:
    req = urlrequest.Request(url, method="GET")
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read()
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Invalid JSON response")
    return data


def _http_post_json(url: str, body: dict[str, Any], timeout_sec: float = 20.0) -> dict[str, Any]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Invalid JSON response")
    return data


def _normalize_zone_points(points: Any) -> list[list[float]]:
    if not isinstance(points, list):
        return []
    out: list[list[float]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            continue
        out.append([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])
    return out


def _zone_norm_to_pixels(points: list[list[float]], width: int, height: int) -> list[list[int]]:
    w = max(1, int(width))
    h = max(1, int(height))
    return [[int(round(p[0] * w)), int(round(p[1] * h))] for p in points]


def _fetch_saved_cameras() -> list[dict[str, Any]]:
    db_base = str(getattr(config, "DB_SERVER_URL", "http://localhost:8001")).rstrip("/")
    data = _http_get_json(f"{db_base}/api/cameras", timeout_sec=8.0)
    cameras = data.get("cameras")
    if isinstance(cameras, list):
        return [c for c in cameras if isinstance(c, dict)]
    return []


def _wait_frame_size(camera_id: str, timeout_sec: float = 20.0, poll_sec: float = 0.5) -> tuple[int, int] | None:
    deadline = time.time() + max(1.0, float(timeout_sec))
    while time.time() < deadline:
        mgr = stream_manager
        if mgr is None:
            time.sleep(poll_sec)
            continue
        try:
            frame = mgr.get_frame(camera_id)
        except Exception:
            frame = None
        if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
            h = int(frame.shape[0])
            w = int(frame.shape[1])
            if w > 1 and h > 1:
                return (w, h)
        time.sleep(poll_sec)
    return None


def _restore_cameras_sync() -> None:
    if not _env_bool_local("AUTO_RESTORE_CAMERAS_ON_BOOT", True):
        logger.info("Auto-restore disabled by AUTO_RESTORE_CAMERAS_ON_BOOT=false")
        return

    db_retries = max(1, _env_int_local("AUTO_RESTORE_DB_RETRIES", 20))
    db_retry_sec = max(1.0, _env_float_local("AUTO_RESTORE_DB_RETRY_SEC", 3.0))
    frame_wait_sec = max(3.0, _env_float_local("AUTO_RESTORE_FRAME_WAIT_SEC", 20.0))
    between_cam_sec = max(0.2, _env_float_local("AUTO_RESTORE_BETWEEN_CAM_SEC", 1.5))
    model_base = str(os.getenv("AUTO_RESTORE_MODEL_SERVER_URL", "http://127.0.0.1:8000")).rstrip("/")

    cameras: list[dict[str, Any]] = []
    for attempt in range(1, db_retries + 1):
        try:
            cameras = _fetch_saved_cameras()
            break
        except Exception as e:
            logger.warning("Auto-restore: camera list fetch failed (%s/%s): %s", attempt, db_retries, e)
            time.sleep(db_retry_sec)

    if not cameras:
        logger.info("Auto-restore: no saved cameras found; skipping.")
        return

    started = 0
    zone_applied = 0
    failed = 0
    logger.info("Auto-restore: restoring %d camera(s) sequentially...", len(cameras))

    for cam in cameras:
        camera_id = str(cam.get("camera_id", "")).strip()
        rtsp_url = str(cam.get("rtsp_url", "")).strip()
        if not camera_id or not rtsp_url:
            failed += 1
            continue

        start_payload = {
            "camera_id": camera_id,
            "rtsp_url": rtsp_url,
            "base_fps": float(cam.get("base_fps", config.BASE_FPS)),
            "rtsp_transport": str(cam.get("rtsp_transport", config.RTSP_TRANSPORT)),
            "open_timeout_ms": int(cam.get("open_timeout_ms", config.RTSP_OPEN_TIMEOUT_MS)),
            "read_timeout_ms": int(cam.get("read_timeout_ms", config.RTSP_READ_TIMEOUT_MS)),
            "event_cooldown_sec": int(cam.get("event_cooldown_sec", 20)),
            "clip_duration_sec": int(cam.get("clip_duration_sec", 10)),
            "validation_clip_sec": int(cam.get("validation_clip_sec", getattr(config, "V3_VALIDATION_CLIP_SEC", 15))),
            "evidence_mode": str(cam.get("evidence_mode", config.EVIDENCE_MODE)),
        }

        try:
            start_resp = _http_post_json(f"{model_base}/api/vlm/start/", start_payload, timeout_sec=25.0)
            if not bool(start_resp.get("success")):
                failed += 1
                continue
            started += 1
        except (urlerror.HTTPError, Exception) as e:
            failed += 1
            logger.warning("Auto-restore: start error for %s: %s", camera_id, e)
            continue

        cashier_zone_norm = _normalize_zone_points(cam.get("cashier_zone"))
        drawer_zone_norm = _normalize_zone_points(cam.get("drawer_zone"))
        exchange_band_norm = _normalize_zone_points(cam.get("exchange_band"))
        staff_work_zone_norm = _normalize_zone_points(cam.get("staff_work_zone"))
        if not cashier_zone_norm and not drawer_zone_norm and not exchange_band_norm and not staff_work_zone_norm:
            time.sleep(between_cam_sec)
            continue

        size = _wait_frame_size(camera_id, timeout_sec=frame_wait_sec, poll_sec=0.5)
        if size is None:
            time.sleep(between_cam_sec)
            continue

        width, height = size
        zone_payload = {
            "camera_id": camera_id,
            "cashier_zone": _zone_norm_to_pixels(cashier_zone_norm, width, height),
            "drawer_zone": _zone_norm_to_pixels(drawer_zone_norm, width, height),
            "exchange_band": _zone_norm_to_pixels(exchange_band_norm, width, height),
            "staff_work_zone": _zone_norm_to_pixels(staff_work_zone_norm, width, height),
        }
        try:
            zone_resp = _http_post_json(f"{model_base}/api/vlm/zones/", zone_payload, timeout_sec=10.0)
            if bool(zone_resp.get("success")):
                zone_applied += 1
        except Exception as e:
            logger.warning("Auto-restore: zone apply error for %s: %s", camera_id, e)

        time.sleep(between_cam_sec)

    logger.info("Auto-restore complete: started=%d zone_applied=%d failed=%d total=%d", started, zone_applied, failed, len(cameras))


async def _restore_cameras_after_startup() -> None:
    delay_sec = max(0.0, _env_float_local("AUTO_RESTORE_DELAY_SEC", 4.0))
    if delay_sec > 0:
        await asyncio.sleep(delay_sec)
    await asyncio.to_thread(_restore_cameras_sync)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_manager, local_storage, flush_worker
    global gemini_validator, v3_pipeline, data_collector, inference_scheduler, event_postprocessor
    global is_shutting_down, startup_restore_task

    logger.info("=" * 60)
    logger.info("HIO v3 model server starting up...")
    logger.info("=" * 60)
    is_shutting_down = False
    config.load_dotenv()

    os.makedirs(str(config.DATA_DIR), exist_ok=True)
    os.makedirs(str(config.LOG_DIR), exist_ok=True)

    stream_manager = StreamManager(default_config={
        "base_fps": config.BASE_FPS,
        "burst_fps": config.BURST_FPS,
        "burst_duration_sec": config.BURST_DURATION_SEC,
        "rtsp_transport": config.RTSP_TRANSPORT,
        "open_timeout_ms": config.RTSP_OPEN_TIMEOUT_MS,
        "read_timeout_ms": config.RTSP_READ_TIMEOUT_MS,
        "rtsp_hwaccel": config.RTSP_HWACCEL,
        "rtsp_hwaccel_device": config.RTSP_HWACCEL_DEVICE,
        "rtsp_hwaccel_decoder": config.RTSP_HWACCEL_DECODER,
        "rtsp_hwaccel_allow_fallback": config.RTSP_HWACCEL_ALLOW_FALLBACK,
    })

    local_storage = LocalStorage(base_dir=str(config.DATA_DIR))
    flush_worker = FlushWorker(
        db_server_url=config.DB_SERVER_URL,
        flush_endpoint=config.FLUSH_ENDPOINT,
        flush_interval_sec=config.FLUSH_INTERVAL_SEC,
        max_retries=config.FLUSH_MAX_RETRIES,
        local_storage=local_storage,
    )
    flush_worker.start()

    try:
        from model_server.v3_pipeline import HioV3Pipeline

        v3_pipeline = HioV3Pipeline()
        logger.info("HIO v3 pipeline loaded.")
    except Exception as e:
        logger.warning("HIO v3 pipeline not loaded: %s", e)
        v3_pipeline = None

    try:
        from model_server.validators.gemini_temporal_validator_v3 import GeminiTemporalValidatorV3

        gemini_validator = GeminiTemporalValidatorV3(
            api_key=config.GEMINI_API_KEY,
            enabled=bool(config.GEMINI_API_KEY),
        )
        logger.info("Gemini temporal validator ready: %s", config.GEMINI_MODEL)
    except Exception as e:
        logger.warning("Gemini temporal validator init failed: %s", e)
        gemini_validator = None

    data_collector = V3ProposalFeedbackCollector(
        base_dir=config.V3_PROPOSAL_FEEDBACK_DIR,
        enabled=True,
    )
    logger.info("v3 feedback collector ready (dir=%s)", config.V3_PROPOSAL_FEEDBACK_DIR)

    try:
        import model_server.vlm_api as v3_vlm_api
        from model_server.event_postprocessor import EventPostProcessor
        from model_server.inference_scheduler import InferenceScheduler

        event_postprocessor = EventPostProcessor(
            process_fn=v3_vlm_api._process_detection_event,
            workers=int(config.POSTPROCESS_WORKERS),
            queue_size=int(config.POSTPROCESS_QUEUE_SIZE),
            dead_letter_dir=config.DATA_DIR / "dead_letter",
        )
        event_postprocessor.start()

        inference_scheduler = InferenceScheduler(
            stream_manager=stream_manager,
            get_state=v3_vlm_api._get_or_create_state,
            process_fn=v3_vlm_api._run_inference_once,
            workers=int(config.INFERENCE_WORKERS),
            queue_size=int(config.INFERENCE_QUEUE_SIZE),
            active_burst_sec=float(config.INFERENCE_ACTIVE_BURST_SEC),
            active_burst_fps=float(config.INFERENCE_ACTIVE_BURST_FPS),
        )
        inference_scheduler.start()
        logger.info(
            "InferenceScheduler initialized workers=%d queue=%d, postprocess workers=%d queue=%d",
            int(config.INFERENCE_WORKERS),
            int(config.INFERENCE_QUEUE_SIZE),
            int(config.POSTPROCESS_WORKERS),
            int(config.POSTPROCESS_QUEUE_SIZE),
        )
    except Exception as e:
        logger.warning("Scheduler init failed: %s", e)
        inference_scheduler = None
        event_postprocessor = None

    startup_restore_task = asyncio.create_task(_restore_cameras_after_startup())
    logger.info("HIO v3 model server ready.")

    yield

    logger.info("HIO v3 model server shutting down...")
    is_shutting_down = True
    try:
        import model_server.vlm_api as v3_vlm_api

        summary = v3_vlm_api.shutdown_all_workers(timeout_sec=2.0)
        logger.info("VLM workers shutdown summary: stopped=%s alive=%s", summary.get("stopped", 0), summary.get("alive", 0))
    except Exception as e:
        logger.warning("VLM worker shutdown helper failed: %s", e)

    if inference_scheduler:
        inference_scheduler.stop()
    inference_scheduler = None

    if event_postprocessor:
        event_postprocessor.stop()
    event_postprocessor = None

    if stream_manager:
        stream_manager.stop_all()
    if startup_restore_task is not None and not startup_restore_task.done():
        startup_restore_task.cancel()
        try:
            await startup_restore_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Startup restore task shutdown warning: %s", e)
    startup_restore_task = None
    if flush_worker:
        flush_worker.stop()
    logger.info("HIO v3 model server stopped.")


app = FastAPI(
    title="HIO v3 CCTV Model Server",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from model_server.vlm_api import router as vlm_router

app.include_router(vlm_router)


class StartStreamRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    scenarios: list[str] = ["cash", "fire", "violence"]


class StopStreamRequest(BaseModel):
    camera_id: str


class AnalyzeRequest(BaseModel):
    scenario: str = "cash"
    caption: str = ""


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "model_server",
        "pipeline_mode": "hio_v3_yolo26_siglip_temporal",
        "hio_v3_enabled": bool(getattr(config, "HIO_V3_ENABLED", True)),
        "v3_pipeline_loaded": v3_pipeline is not None,
        "tier1_backend": "YOLO26" if v3_pipeline is not None else "unloaded",
        "gemini_model": getattr(config, "GEMINI_MODEL", ""),
    }


@app.get("/status")
def system_status():
    return {
        "streams": stream_manager.get_all_stats() if stream_manager else {},
        "storage": local_storage.get_stats() if local_storage else {},
        "flush": flush_worker.get_stats() if flush_worker else {},
        "feedback": data_collector.get_stats() if data_collector else {},
        "pipeline_mode": "hio_v3_yolo26_siglip_temporal",
        "hio_v3_enabled": bool(getattr(config, "HIO_V3_ENABLED", True)),
        "v3_pipeline_loaded": v3_pipeline is not None,
        "tier1_backend": "YOLO26" if v3_pipeline is not None else "unloaded",
        "gemini_model": getattr(config, "GEMINI_MODEL", ""),
    }


@app.post("/api/start_stream")
def start_stream(req: StartStreamRequest):
    if stream_manager is None:
        return {"error": "Not initialized"}
    stream = stream_manager.add_camera(req.camera_id, req.rtsp_url)
    stream.start()
    return {"status": "started", "camera_id": req.camera_id, "rtsp_url": req.rtsp_url}


@app.post("/api/stop_stream")
def stop_stream(req: StopStreamRequest):
    if stream_manager is None:
        return {"error": "Not initialized"}
    stopped = stream_manager.remove_camera(req.camera_id)
    return {"status": "stopped" if stopped else "not_found", "camera_id": req.camera_id}


@app.get("/api/stream/{camera_id}/frame")
def get_frame(camera_id: str):
    if stream_manager is None:
        return Response(status_code=503, content="Not initialized")
    frame = stream_manager.get_frame(camera_id)
    if frame is None:
        return Response(status_code=404, content="No frame available")

    import cv2

    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@app.post("/api/analyze")
def analyze_caption(req: AnalyzeRequest):
    return {
        "error": "caption_analysis_removed_in_v3",
        "message": "Use /api/analyze_frame or /api/vlm/* so YOLO26 temporal metadata is available.",
        "scenario": req.scenario,
    }


@app.post("/api/analyze_frame")
def analyze_frame(
    camera_id: Annotated[str, Body()],
    scenario: Annotated[str, Body()] = "cash",
):
    if stream_manager is None:
        return {"error": "Not ready (stream_manager not loaded)"}
    if v3_pipeline is None:
        return {"error": "HIO v3 pipeline not loaded"}

    frame = stream_manager.get_frame(camera_id)
    if frame is None:
        return {"error": f"No frame for camera {camera_id}"}

    result = v3_pipeline.process_frame(
        camera_id,
        frame,
        {"cashier": [], "drawer": [], "exchange": [], "staff_work": []},
    )
    selected = result.scenario_results.get(scenario, {})
    if selected.get("is_detected") and local_storage is not None:
        event_id = f"ev_{int(time.time() * 1000)}"
        local_storage.save_event(event_id, selected)
    return {
        "pipeline_version": str(getattr(config, "HIO_V3_PIPELINE_VERSION", "v3-yolo26-tier1-siglip-episode-gemini")),
        "scenario": scenario,
        "result": selected,
        "all_results": result.scenario_results,
        "metadata": result.metadata,
    }


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        if ws in ws_clients:
            ws_clients.remove(ws)


async def broadcast_event(event: dict[str, Any]):
    message = json.dumps(event, default=str)
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)


@app.post("/api/flush")
def manual_flush():
    if flush_worker is None:
        return {"error": "Not initialized"}
    return flush_worker.flush()


@app.get("/api/v3/feedback/stats")
def v3_feedback_stats():
    if data_collector is None:
        return {"error": "Not initialized"}
    return data_collector.get_stats()
