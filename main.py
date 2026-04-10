"""HiO Intelligence Stream v2 — Main FastAPI application."""

from __future__ import annotations

import asyncio
import collections
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from storage import db
from stream.clip_extractor import extract_clip_frames
from stream.stream_reader import StreamReader
from tier1.trigger_accumulator import TriggerAccumulator

logger = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────
streams: dict[str, StreamReader] = {}
cam_configs: dict[str, dict] = {}
accumulator = TriggerAccumulator()

cash_trigger = None
clip_trigger = None
video_analyzer = None

ws_clients: list[WebSocket] = []
_inference_task: asyncio.Task | None = None
_retention_task: asyncio.Task | None = None
_event_tasks: set[asyncio.Task] = set()
_tier1_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tier1")

# Per-camera latest Tier 1 results for live monitoring
_tier1_latest: dict[str, dict] = {}
# Per-camera Tier 1 history log (last 200 entries per camera)
_tier1_history: dict[str, collections.deque] = {}
# Qwen inference lock — prevent concurrent GPU calls
_qwen_lock = asyncio.Lock()


# ── Lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cash_trigger, clip_trigger, video_analyzer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    await db.init_db()

    # Load Tier 1
    logger.info("Loading Tier 1 models...")
    from tier1.cash_trigger import CashTrigger
    from tier1.clip_trigger import CLIPTrigger

    cash_trigger = CashTrigger()
    clip_trigger = CLIPTrigger()

    # Tier 2: Qwen2.5-VL-3B — lazy load on first trigger to save VRAM
    # YOLO(~1GB) + CLIP(~1.5GB) + CUDA overhead(~1.5GB) = ~4GB → leaves ~8GB for Qwen
    logger.info("Tier 2 (Qwen2.5-VL-3B) will load on first event trigger")

    # Auto-restore cameras from DB
    saved = await db.get_all_cameras()
    if saved:
        logger.info("Auto-restoring %d cameras in %ds...", len(saved), config.AUTO_RESTORE_DELAY)
        await asyncio.sleep(config.AUTO_RESTORE_DELAY)
        for cam_cfg in saved:
            try:
                _start_camera(cam_cfg)
                logger.info("Restored camera: %s", cam_cfg["id"])
            except Exception as e:
                logger.error("Failed to restore %s: %s", cam_cfg.get("id"), e)

    # Start inference loop + daily retention cleanup
    global _inference_task, _retention_task
    _inference_task = asyncio.create_task(_inference_loop())
    _retention_task = asyncio.create_task(_retention_loop())

    logger.info("HiO v2 ready — %d cameras", len(streams))
    yield

    if _inference_task:
        _inference_task.cancel()
    if _retention_task:
        _retention_task.cancel()
    for t in list(_event_tasks):
        t.cancel()
    _tier1_executor.shutdown(wait=False)
    for reader in streams.values():
        reader.stop()
    await db.close_db()
    logger.info("HiO v2 shutdown complete")


app = FastAPI(title="HiO Intelligence Stream v2", lifespan=lifespan)
app.mount("/media", StaticFiles(directory=str(config.DATA_DIR), check_dir=False), name="media")


# ── Camera management ──────────────────────────────────────────────
def _start_camera(cam: dict) -> None:
    cam_id = cam["id"]
    if cam_id in streams:
        return
    reader = StreamReader(cam_id, cam["rtsp_url"])
    reader.start()
    streams[cam_id] = reader
    cam_configs[cam_id] = cam


# ── Parallel Tier 1 inference ──────────────────────────────────────
def _run_tier1(frame, cam_cfg: dict) -> dict:
    results = {}
    futures = {}

    if cash_trigger:
        futures["cash"] = _tier1_executor.submit(cash_trigger.detect, frame, cam_cfg)

    if clip_trigger:
        futures["clip"] = _tier1_executor.submit(clip_trigger.detect, frame)

    if "cash" in futures:
        try:
            results["cash"] = futures["cash"].result(timeout=30.0)
        except Exception as e:
            logger.error("Cash trigger failed: %s", e, exc_info=True)
            results["cash"] = {"triggered": False}

    if "clip" in futures:
        try:
            clip_results = futures["clip"].result(timeout=30.0)
            results["fire"] = clip_results.get("fire", {"triggered": False})
            results["violence"] = clip_results.get("violence", {"triggered": False})
        except Exception as e:
            logger.error("CLIP trigger failed: %s", e, exc_info=True)
            results["fire"] = {"triggered": False}
            results["violence"] = {"triggered": False}

    return results


# ── Inference loop ─────────────────────────────────────────────────
async def _inference_loop():
    sample_interval = 1.0 / config.SAMPLE_FPS

    while True:
        try:
            for cam_id, reader in list(streams.items()):
                frame, frame_time = reader.get_inference_frame()
                if frame is None:
                    continue

                cam_cfg = cam_configs.get(cam_id, {})
                tier1_results = await asyncio.to_thread(_run_tier1, frame, cam_cfg)

                # Store latest + append to history
                entry = {
                    **tier1_results,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "accumulator": {
                        s: len(accumulator._triggers.get((cam_id, s), []))
                        for s in ("cash", "fire", "violence")
                    },
                }
                _tier1_latest[cam_id] = entry
                if cam_id not in _tier1_history:
                    _tier1_history[cam_id] = collections.deque(maxlen=200)
                _tier1_history[cam_id].append(entry)

                for scenario, result in tier1_results.items():
                    if result.get("triggered"):
                        reader.trigger_burst()  # Switch to 4 FPS for 3s
                        if accumulator.add(cam_id, scenario, frame_time):
                            task = asyncio.create_task(
                                _handle_event(cam_id, scenario, frame_time, reader)
                            )
                            _event_tasks.add(task)
                            task.add_done_callback(_event_tasks.discard)

            await asyncio.sleep(sample_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Inference loop error: %s", e)
            await asyncio.sleep(1)


async def _handle_event(cam_id: str, scenario: str, trigger_time: float, reader: StreamReader):
    """Lazy pipeline: extract clip → save MP4 → Qwen(MP4) → Gemini(MP4) → alert."""
    global video_analyzer
    try:
        from event.event_pipeline import save_clips, finalize_event

        cam_cfg = cam_configs.get(cam_id, {})

        # ── Phase 1: Extract + Save clips OUTSIDE lock ──
        clip_frames = await asyncio.to_thread(
            extract_clip_frames, reader, trigger_time,
        )
        if not clip_frames or len(clip_frames) < 2:
            logger.warning("[%s] Not enough clip frames (%d), dismissing",
                           cam_id, len(clip_frames) if clip_frames else 0)
            return

        # Save clips to disk (full + zone if applicable)
        full_path, zone_path, thumb_full, thumb_zone = await save_clips(
            cam_id, scenario, clip_frames, cam_cfg,
        )
        if not full_path:
            logger.error("[%s] Failed to save clip, aborting", cam_id)
            return

        # Choose which clip to send to VLMs
        vlm_clip_path = zone_path if zone_path else full_path

        # ── Phase 2: Qwen GPU inference INSIDE lock ──
        async with _qwen_lock:
            if video_analyzer is None:
                logger.info("Loading Tier 2 model (Qwen2.5-VL-3B) on first trigger...")
                from tier2.video_analyzer import VideoAnalyzer
                video_analyzer = await asyncio.to_thread(VideoAnalyzer)
                logger.info("Tier 2 model loaded")

            tier2_result = await asyncio.to_thread(
                video_analyzer.analyze_clip, vlm_clip_path, scenario
            )
            routing = video_analyzer.route_result(tier2_result, scenario)

        logger.info("[%s] Tier 2 result: %s", cam_id, tier2_result)

        # ── Phase 3: Tier 3 + DB + alert OUTSIDE lock ──
        event = await finalize_event(
            cam_id=cam_id,
            scenario=scenario,
            tier2_result=tier2_result,
            routing=routing,
            full_path=full_path,
            zone_path=zone_path,
            thumb_full=thumb_full,
            thumb_zone=thumb_zone,
            vlm_clip_path=vlm_clip_path,
        )
        if event:
            await _broadcast_ws(event)

    except Exception as e:
        logger.error("[%s] Event pipeline error: %s", cam_id, e, exc_info=True)


async def _broadcast_ws(event: dict):
    dead = []
    for ws in list(ws_clients):  # snapshot to avoid mutation during iteration
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── API: Status ────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "cameras": {
            cid: {
                "id": cid,
                "running": r.running,
                "has_frame": r.latest_frame is not None,
                "stale": r.is_stale,
            }
            for cid, r in streams.items()
        },
        "models": {
            "tier1_cash": cash_trigger is not None,
            "tier1_clip": clip_trigger is not None,
            "tier2_qwen": video_analyzer is not None,
        },
    }


@app.get("/api/cameras/{cam_id}/tier1")
async def camera_tier1(cam_id: str):
    """Return latest Tier 1 detection results for a camera."""
    data = _tier1_latest.get(cam_id)
    if not data:
        return {"cam_id": cam_id, "status": "no_data"}
    return {"cam_id": cam_id, **data}


# ── API: Events ────────────────────────────────────────────────────
@app.get("/api/events")
async def get_events(cam_id: str | None = None, scenario: str | None = None, limit: int = 50, offset: int = 0):
    events = await db.get_events(cam_id=cam_id, scenario=scenario, limit=limit, offset=offset)
    total = await db.get_event_count()
    return {"events": events, "total": total}


@app.get("/api/stats")
async def get_stats():
    return await db.get_stats()


# ── API: Cameras ───────────────────────────────────────────────────
@app.get("/api/cameras")
async def list_cameras():
    return {
        "cameras": [
            {
                "id": cid,
                "running": r.running,
                "has_frame": r.latest_frame is not None,
                **{k: v for k, v in cam_configs.get(cid, {}).items() if k not in ("rtsp_url",)},
            }
            for cid, r in streams.items()
        ]
    }


@app.post("/api/cameras/{cam_id}/start")
async def start_camera(cam_id: str, rtsp_url: str = ""):
    if cam_id in streams:
        return {"status": "already_running"}
    if not rtsp_url:
        return JSONResponse({"error": "rtsp_url required"}, status_code=400)

    cam_dict = {"id": cam_id, "rtsp_url": rtsp_url, "cashier_zone": [], "customer_zone": [], "wrist_px": 80}
    _start_camera(cam_dict)
    await db.upsert_camera(cam_dict)
    return {"status": "started", "cam_id": cam_id}


@app.post("/api/cameras/{cam_id}/stop")
async def stop_camera(cam_id: str):
    reader = streams.pop(cam_id, None)
    if reader:
        reader.stop()
        cam_configs.pop(cam_id, None)
        accumulator.reset(cam_id)
        return {"status": "stopped"}
    return JSONResponse({"error": "camera not found"}, status_code=404)


@app.delete("/api/cameras/{cam_id}")
async def remove_camera(cam_id: str):
    reader = streams.pop(cam_id, None)
    if reader:
        reader.stop()
    cam_configs.pop(cam_id, None)
    accumulator.reset(cam_id)
    await db.delete_camera(cam_id)
    return {"status": "removed"}


# ── API: Zones (ROI) ──────────────────────────────────────────────
@app.put("/api/cameras/{cam_id}/zones")
async def update_zones(cam_id: str, request: Request):
    body = await request.json()
    cashier = body.get("cashier_zone", [])
    customer = body.get("customer_zone", [])
    wpx = int(body.get("wrist_px", 80))
    trigger_mode = body.get("trigger_mode", "wrist")

    if cam_id in cam_configs:
        cam_configs[cam_id]["cashier_zone"] = cashier
        cam_configs[cam_id]["customer_zone"] = customer
        cam_configs[cam_id]["wrist_px"] = wpx
        cam_configs[cam_id]["trigger_mode"] = trigger_mode

    await db.update_camera_zones(cam_id, cashier, customer, wpx, trigger_mode)
    return {"status": "ok", "cam_id": cam_id, "trigger_mode": trigger_mode}


# ── API: Snapshot ──────────────────────────────────────────────────
@app.get("/api/cameras/{cam_id}/snapshot")
async def camera_snapshot(cam_id: str):
    import cv2

    reader = streams.get(cam_id)
    if not reader:
        return JSONResponse({"error": "camera not found"}, status_code=404)

    frame, _ = reader.get_frame()
    if frame is None:
        return JSONResponse({"error": "no frame available"}, status_code=503)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


@app.get("/api/cameras/{cam_id}/skeleton")
async def camera_skeleton(cam_id: str):
    """Snapshot with YOLO skeleton overlay. No extra GPU — reuses last YOLO result."""
    reader = streams.get(cam_id)
    if not reader:
        return JSONResponse({"error": "camera not found"}, status_code=404)

    frame, _ = reader.get_frame()
    if frame is None:
        return JSONResponse({"error": "no frame available"}, status_code=503)

    cam_cfg = cam_configs.get(cam_id, {})

    # Run YOLO on this frame (reuses existing model, ~5ms)
    def _draw_skeleton(f, cfg):
        if not cash_trigger:
            return f
        results = cash_trigger.model(f, verbose=False, half=cash_trigger._use_half)
        kps = results[0].keypoints
        boxes = results[0].boxes
        if kps is None or kps.xy is None:
            return f

        vis = f.copy()
        h, w = vis.shape[:2]

        # Draw cashier zone if set
        zone_norm = cfg.get("cashier_zone", [])
        if len(zone_norm) >= 3:
            import numpy as np
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone_norm], np.int32)
            cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
            cv2.putText(vis, "CASHIER ZONE", (pts[0][0], max(pts[0][1] - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # COCO skeleton connections
        skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15), (12, 14), (14, 16),  # legs
            (0, 1), (0, 2), (1, 3), (2, 4),  # face
        ]
        colors = {
            "staff": (0, 255, 100),    # green
            "customer": (255, 150, 0), # orange
            "unknown": (200, 200, 200),  # gray
        }

        import numpy as np
        zone_px = None
        if len(zone_norm) >= 3:
            zone_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone_norm], np.int32)

        for i, kp in enumerate(kps.xy):
            kp_np = kp.cpu().numpy() if hasattr(kp, "cpu") else np.array(kp)

            # Determine role
            from tier1.cash_trigger import _best_position, _point_in_polygon
            pos = _best_position(kp_np)
            role = "unknown"
            if zone_px is not None and pos is not None:
                role = "staff" if _point_in_polygon(pos, zone_px) else "customer"
            color = colors[role]

            # Draw keypoints
            for j, pt in enumerate(kp_np):
                x, y = int(pt[0]), int(pt[1])
                if x == 0 and y == 0:
                    continue
                r = 5 if j in (9, 10) else 3  # wrists bigger
                cv2.circle(vis, (x, y), r, color, -1)

            # Draw skeleton lines
            for a, b in skeleton:
                pa, pb = kp_np[a], kp_np[b]
                if pa.sum() > 0 and pb.sum() > 0:
                    cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1)

            # Label
            if pos is not None:
                label = f"{role}[{i}]"
                cv2.putText(vis, label, (int(pos[0]) - 20, int(pos[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw wrist distances
        trigger_mode = cfg.get("trigger_mode", "wrist")
        wrist_threshold = cfg.get("wrist_px", 250)
        cv2.putText(vis, f"mode: {trigger_mode} | threshold: {wrist_threshold}px",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis

    vis_frame = await asyncio.to_thread(_draw_skeleton, frame, cam_cfg)
    _, buf = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


# ── API: MJPEG Stream ─────────────────────────────────────────────
@app.get("/api/cameras/{cam_id}/mjpeg")
async def camera_mjpeg(cam_id: str):
    import cv2

    reader = streams.get(cam_id)
    if not reader:
        return JSONResponse({"error": "camera not found"}, status_code=404)

    async def generate():
        target_interval = 1.0 / 10.0  # 10 FPS
        try:
            while reader.running:
                frame, _ = reader.get_frame()
                if frame is None:
                    await asyncio.sleep(0.5)
                    continue

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg = buf.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
                await asyncio.sleep(target_interval)
        except asyncio.CancelledError:
            return

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── WebSocket ──────────────────────────────────────────────────────
@app.websocket("/ws/events")
async def websocket_events(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── Retention cleanup (daily) ──────────────────────────────────────
async def _retention_loop():
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            deleted = await db.cleanup_old_data()
            if deleted:
                logger.info("Retention cleanup: deleted %d old events", deleted)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Retention cleanup error: %s", e)


# ── API: Tier 1 History ────────────────────────────────────────────
@app.get("/api/cameras/{cam_id}/tier1/history")
async def camera_tier1_history(cam_id: str, limit: int = 50):
    """Return recent Tier 1 inference history for a camera."""
    hist = _tier1_history.get(cam_id)
    if not hist:
        return {"cam_id": cam_id, "logs": []}
    entries = list(hist)[-limit:]
    # Simplify for JSON (remove numpy arrays from cash distances)
    logs = []
    for e in reversed(entries):
        log = {"timestamp": e.get("timestamp")}
        cash = e.get("cash", {})
        log["cash_triggered"] = cash.get("triggered", False)
        log["cash_staff"] = cash.get("staff_count", 0)
        log["cash_cust"] = cash.get("customer_count", 0)
        log["cash_min_dist"] = cash.get("min_distance")
        fire = e.get("fire", {})
        log["fire_score"] = fire.get("score", 0)
        log["fire_triggered"] = fire.get("triggered", False)
        viol = e.get("violence", {})
        log["violence_score"] = viol.get("score", 0)
        log["violence_triggered"] = viol.get("triggered", False)
        log["accumulator"] = e.get("accumulator", {})
        logs.append(log)
    return {"cam_id": cam_id, "logs": logs}


# ── API: Clip Review (upload MP4 → Tier 1/2/3 evaluation) ─────────
@app.post("/api/clip-review")
async def clip_review(clip: UploadFile, scenario: str = ""):
    """Upload MP4 clip → evaluate ALL scenarios through Tier 1/2/3 pipeline."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    contents = await clip.read()
    tmp.write(contents)
    tmp.flush()
    clip_filename = clip.filename or "clip.mp4"
    tmp_path = tmp.name
    tmp.close()

    try:
        # Get frame count + middle frame for Tier 1
        cap = cv2.VideoCapture(tmp_path)
        total = 0
        mid_frame = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                total += 1
                if mid_frame is None:
                    mid_frame = frame  # will be overwritten to actual middle
            # Re-read middle frame
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, mid_frame = cap.read()
        finally:
            cap.release()

        if total == 0 or mid_frame is None:
            return JSONResponse({"error": "no frames in clip"}, status_code=400)

        result = {
            "filename": clip_filename,
            "total_frames": total,
            "scenarios": {},
        }

        # Tier 1: all scenarios on middle frame
        t1_cash = await asyncio.to_thread(cash_trigger.detect, mid_frame, {}) if cash_trigger else {}
        t1_clip = await asyncio.to_thread(clip_trigger.detect, mid_frame) if clip_trigger else {}

        result["tier1"] = {
            "cash": t1_cash,
            "fire": t1_clip.get("fire", {}),
            "violence": t1_clip.get("violence", {}),
        }

        # Tier 2: Qwen analyzes the uploaded MP4 directly (INSIDE lock)
        global video_analyzer
        tier2_results = {}
        async with _qwen_lock:
            if video_analyzer is None:
                from tier2.video_analyzer import VideoAnalyzer
                video_analyzer = await asyncio.to_thread(VideoAnalyzer)

            for sc in ("cash", "fire", "violence"):
                t2 = await asyncio.to_thread(video_analyzer.analyze_clip, tmp_path, sc)
                tier2_results[sc] = (t2, video_analyzer.route_result(t2, sc))

        # Tier 3: Gemini analyzes the uploaded MP4 (OUTSIDE lock)
        from tier3 import tier3_verifier
        for sc, (t2, routing) in tier2_results.items():
            t3 = None
            if routing == "tier3":
                t3 = await asyncio.to_thread(tier3_verifier.verify, tmp_path, sc, t2)

            result["scenarios"][sc] = {
                "tier2": t2,
                "routing": routing,
                "tier3": t3 or {"verdict": "SKIPPED" if routing != "tier3" else "ERROR"},
            }

        await db.insert_clip_review(clip_filename, total, result)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/clip-reviews")
async def get_clip_reviews(limit: int = 50):
    return {"reviews": await db.get_clip_reviews(limit)}


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=config.SERVER_PORT, reload=False, log_level="info")
