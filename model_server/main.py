"""
Model Server — FastAPI entry point.

Tier 1 (Florence-2) + Tier 2 (Gemini) detection pipeline.

Endpoints:
    GET  /                  → health check
    GET  /status            → system status (streams, shadow, critic)
    POST /api/start_stream  → start RTSP camera stream
    POST /api/stop_stream   → stop camera stream
    POST /api/analyze_frame → one-shot frame analysis (for testing)
    GET  /api/stream/{id}/frame → MJPEG snapshot
    WS   /ws/events         → real-time event push
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_server import config
from model_server.stream_manager import StreamManager
from model_server.local_storage import LocalStorage
from model_server.flush_worker import FlushWorker
from model_server.agents.dynamic_agent import DynamicAgent
from model_server.agents.shadow_agent import ShadowAgent
from model_server.evolution.critic_trainer import CriticTrainer
from model_server.evolution.rule_updater import RuleUpdater
from model_server.lora.data_collector import DataCollector

logger = logging.getLogger("model_server")
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
)

# ---------------------------------------------------------------------------
# Global state (initialised in lifespan)
# ---------------------------------------------------------------------------
stream_manager: StreamManager | None = None
local_storage: LocalStorage | None = None
flush_worker: FlushWorker | None = None
florence_adapter = None
gemini_validator = None          # GeminiValidator instance (Tier 2)
pipeline_orchestrator = None     # ScenarioOrchestrator instance (Tier 1 parallel)
evidence_router = None           # EvidenceRouter instance
agents: dict[str, DynamicAgent] = {}
shadow_agents: dict[str, ShadowAgent] = {}
critic_trainer: CriticTrainer | None = None
rule_updater: RuleUpdater | None = None
data_collector: DataCollector | None = None
ws_clients: list[WebSocket] = []
is_shutting_down: bool = False


def _florence_device_status() -> dict[str, Any]:
    requested = config.FLORENCE_DEVICE
    actual = "not_loaded"
    initialized = False

    if florence_adapter is not None:
        actual = str(getattr(florence_adapter, "device", "unknown"))
        initialized = bool(getattr(florence_adapter, "is_initialized", False))

    return {
        "requested": requested,
        "actual": actual,
        "initialized": initialized,
    }


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_manager, local_storage, flush_worker
    global florence_adapter, gemini_validator, pipeline_orchestrator, evidence_router, agents, shadow_agents
    global critic_trainer, rule_updater, data_collector, is_shutting_down

    logger.info("=" * 60)
    logger.info("Model Server starting up...")
    logger.info("=" * 60)
    is_shutting_down = False

    # Load .env if available
    config.load_dotenv()

    # Data dirs
    os.makedirs(str(config.DATA_DIR), exist_ok=True)
    os.makedirs(str(config.LOG_DIR), exist_ok=True)

    # Infrastructure
    stream_manager = StreamManager(default_config={
        "base_fps": config.BASE_FPS,
        "burst_fps": config.BURST_FPS,
        "burst_duration_sec": config.BURST_DURATION_SEC,
        "rtsp_transport": config.RTSP_TRANSPORT,
        "open_timeout_ms": config.RTSP_OPEN_TIMEOUT_MS,
        "read_timeout_ms": config.RTSP_READ_TIMEOUT_MS,
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

    # Florence-2 adapter (lazy load)
    try:
        from model_server.adapters.florence_adapter import create_florence_adapter
        florence_adapter = create_florence_adapter({
            "model": config.FLORENCE_MODEL,
            "backend": config.FLORENCE_BACKEND,
            "device": config.FLORENCE_DEVICE,
            "input_size": (config.FLORENCE_INPUT_SIZE, config.FLORENCE_INPUT_SIZE),
            "cache_dir": str(config.MODELS_DIR),
            "lora_enabled": config.LORA_ENABLED,
            "lora_adapter_path": config.LORA_ADAPTER_PATH,
        })
        logger.info(f"Florence-2 adapter loaded: {config.FLORENCE_MODEL}")
        dev = _florence_device_status()
        if dev["actual"] != "cuda":
            logger.warning(
                "Florence-2 requested device=%s but actual device=%s",
                dev["requested"], dev["actual"]
            )
        else:
            logger.info("Florence-2 running on CUDA GPU.")
            
        # Initialize PipelineOrchestrator
        from model_server.pipeline_orchestrator import ScenarioOrchestrator, OrchestratorConfig
        pipeline_orchestrator = ScenarioOrchestrator(
            vlm_adapter=florence_adapter,
            config=OrchestratorConfig(
                detect_cash=True,
                detect_violence=True,
                detect_fire=True
            )
        )
        logger.info("PipelineOrchestrator initialized.")
    except Exception as e:
        logger.warning(f"Florence-2 / Orchestrator not loaded (will work without GPU): {e}")
        florence_adapter = None
        pipeline_orchestrator = None

    # Determine router configurations
    try:
        from model_server.evidence_router import EvidenceRouter
        evidence_router = EvidenceRouter()
        logger.info("EvidenceRouter initialized.")
    except Exception as e:
        logger.warning(f"EvidenceRouter init failed: {e}")
        evidence_router = None

    # ── Gemini Validator (Tier 2) ──
    try:
        from model_server.gemini_validator import GeminiValidator
        if config.GEMINI_API_KEY:
            gemini_validator = GeminiValidator(
                api_key=config.GEMINI_API_KEY,
                enabled=True,
            )
            logger.info(f"Gemini validator loaded: {config.GEMINI_MODEL}")
        else:
            logger.warning("GEMINI_API_KEY not set — Tier2 validation disabled (bypass mode)")
    except Exception as gem_init_err:
        logger.warning(f"GeminiValidator init failed (bypass mode): {gem_init_err}")

    # Dynamic Agents (per scenario)
    prompts_dir = str(config.RULE_PROMPTS_DIR)
    for scenario in ["cash", "fire", "violence"]:
        agents[scenario] = DynamicAgent(
            scenario_name=scenario,
            prompts_dir=prompts_dir,
        )

    # Evolution
    critic_trainer = CriticTrainer(
        model_dir=config.CRITIC_MODEL_DIR,
        min_samples=config.CRITIC_MIN_SAMPLES,
    )
    rule_updater = RuleUpdater(
        prompts_dir=prompts_dir,
        versions_dir=config.RULE_VERSIONS_DIR,
        gemini_api_key=config.GEMINI_API_KEY,
    )

    # Shadow Agents
    for scenario in ["cash", "fire", "violence"]:
        sa = ShadowAgent(
            scenario_name=scenario,
            critic_trainer=critic_trainer,
            rule_updater=rule_updater,
            batch_size=config.SHADOW_BATCH_SIZE,
            persist_dir=config.SHADOW_PERSIST_DIR,
            max_queue_size=config.SHADOW_MAX_QUEUE,
            prompts_dir=prompts_dir,
        )
        sa.start()
        shadow_agents[scenario] = sa

    # Data Collector for LoRA training
    data_collector = DataCollector(
        base_dir=config.LORA_DATA_DIR,
        normal_ratio=config.LORA_COLLECT_NORMAL_RATIO,
        max_samples=config.LORA_MAX_SAMPLES,
        enabled=config.LORA_DATA_COLLECTION,
    )
    logger.info(
        f"LoRA DataCollector: {'enabled' if data_collector.enabled else 'disabled'} "
        f"(dir={config.LORA_DATA_DIR})"
    )

    logger.info("Model Server ready.")

    yield  # ← app is running

    # Shutdown
    logger.info("Model Server shutting down...")
    is_shutting_down = True
    try:
        import model_server.vlm_api as legacy_vlm_api
        summary = legacy_vlm_api.shutdown_all_workers(timeout_sec=2.0)
        logger.info(
            "VLM workers shutdown summary: stopped=%s alive=%s",
            summary.get("stopped", 0), summary.get("alive", 0)
        )
    except Exception as e:
        logger.warning(f"VLM worker shutdown helper failed: {e}")

    if stream_manager:
        stream_manager.stop_all()
    for sa in shadow_agents.values():
        sa.stop()
    if flush_worker:
        flush_worker.stop()
    logger.info("Model Server stopped.")


app = FastAPI(
    title="Intelligent CCTV Model Server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register VLM legacy API router (for adhoc_rtsp.html compatibility)
from model_server.vlm_api import router as vlm_router
app.include_router(vlm_router)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class StartStreamRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    scenarios: list[str] = ["cash", "fire", "violence"]


class StopStreamRequest(BaseModel):
    camera_id: str


class AnalyzeRequest(BaseModel):
    scenario: str = "cash"
    caption: str


# ---------------------------------------------------------------------------
# Health / Status endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    dev = _florence_device_status()
    return {
        "status": "ok",
        "service": "model_server",
        "florence_loaded": florence_adapter is not None,
        "florence_device_requested": dev["requested"],
        "florence_device_actual": dev["actual"],
        "florence_initialized": dev["initialized"],
    }


@app.get("/status")
def system_status():
    dev = _florence_device_status()
    result = {
        "streams": stream_manager.get_all_stats() if stream_manager else {},
        "storage": local_storage.get_stats() if local_storage else {},
        "flush": flush_worker.get_stats() if flush_worker else {},
        "shadow_agents": {
            name: sa.get_stats() for name, sa in shadow_agents.items()
        },
        "agents_loaded": list(agents.keys()),
        "florence_loaded": florence_adapter is not None,
        "florence_device_requested": dev["requested"],
        "florence_device_actual": dev["actual"],
        "florence_initialized": dev["initialized"],
    }
    return result


# ---------------------------------------------------------------------------
# Stream management
# ---------------------------------------------------------------------------
@app.post("/api/start_stream")
def start_stream(req: StartStreamRequest):
    if stream_manager is None:
        return {"error": "Not initialized"}

    stream = stream_manager.add_camera(req.camera_id, req.rtsp_url)
    stream.start()

    return {
        "status": "started",
        "camera_id": req.camera_id,
        "rtsp_url": req.rtsp_url,
    }


@app.post("/api/stop_stream")
def stop_stream(req: StopStreamRequest):
    if stream_manager is None:
        return {"error": "Not initialized"}

    stopped = stream_manager.remove_camera(req.camera_id)
    return {"status": "stopped" if stopped else "not_found", "camera_id": req.camera_id}


@app.get("/api/stream/{camera_id}/frame")
def get_frame(camera_id: str):
    """Return latest frame as JPEG (for MJPEG streaming)."""
    if stream_manager is None:
        return Response(status_code=503, content="Not initialized")

    frame = stream_manager.get_frame(camera_id)
    if frame is None:
        return Response(status_code=404, content="No frame available")

    import cv2
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Analysis (testing without RTSP — uses caption text directly)
# ---------------------------------------------------------------------------
@app.post("/api/analyze")
def analyze_caption(req: AnalyzeRequest):
    """Test endpoint: analyze a caption without Florence/camera."""
    from model_server.scenarios.base_scenario import CaptionAnalyzer
    from model_server.scenarios import ScenarioType

    try:
        scenario_type = ScenarioType[req.scenario.upper()]
    except KeyError:
        return {"error": f"Unknown scenario: {req.scenario}"}

    result = CaptionAnalyzer.analyze(req.caption, scenario_type)
    result["caption"] = req.caption
    result["scenario"] = req.scenario

    from model_server.agents.dynamic_agent import UncertaintyGate
    result["would_escalate_to_tier2"] = UncertaintyGate.should_escalate(
        req.scenario, result, stability=0.5
    )

    return result


@app.post("/api/analyze_frame")
def analyze_frame(
    camera_id: Annotated[str, Body()],
    scenario: Annotated[str, Body()] = "cash",
):
    """Grab latest frame from stream and run full pipeline."""
    if stream_manager is None or florence_adapter is None:
        return {"error": "Not ready (stream_manager or florence not loaded)"}

    frame = stream_manager.get_frame(camera_id)
    if frame is None:
        return {"error": f"No frame for camera {camera_id}"}

    agent = agents.get(scenario)
    if agent is None:
        return {"error": f"No agent for scenario: {scenario}"}

    result = agent.process(
        frame=frame,
        florence_adapter=florence_adapter,
        gemini_validator=gemini_validator,
        stability=0.5,
    )

    # Save event locally
    if result.get("is_detected"):
        event_id = f"ev_{int(time.time() * 1000)}"
        local_storage.save_event(event_id, result)

        # Push to shadow for async evaluation
        shadow = shadow_agents.get(scenario)
        if shadow:
            shadow.enqueue({"event_id": event_id, **result})

    return result


# ---------------------------------------------------------------------------
# WebSocket — real-time event push
# ---------------------------------------------------------------------------
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            # Keep alive — clients can send pings
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        ws_clients.remove(ws)


async def broadcast_event(event: dict):
    """Broadcast an event to all connected WebSocket clients."""
    message = json.dumps(event, default=str)
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Manual flush trigger
# ---------------------------------------------------------------------------
@app.post("/api/flush")
def manual_flush():
    """Trigger manual flush to DB server."""
    if flush_worker is None:
        return {"error": "Not initialized"}
    result = flush_worker.flush()
    return result


# ---------------------------------------------------------------------------
# Evolution endpoints
# ---------------------------------------------------------------------------
@app.get("/api/evolution/stats")
def evolution_stats():
    return {
        "critic": {
            "model_loaded": critic_trainer.model is not None if critic_trainer else False,
            "training_count": critic_trainer._training_count if critic_trainer else 0,
        },
        "rule_versions": {
            scenario: rule_updater.get_version_history(scenario)[-3:]
            if rule_updater else []
            for scenario in ["cash", "fire", "violence"]
        },
    }


@app.get("/api/evolution/prompt/{scenario}")
def get_prompt(scenario: str):
    if rule_updater is None:
        return {"error": "Not initialized"}
    prompt = rule_updater.get_current_prompt(scenario)
    if prompt is None:
        return {"error": f"No prompt for {scenario}"}
    return {"scenario": scenario, "prompt": prompt}


# ---------------------------------------------------------------------------
# LoRA data collection endpoints
# ---------------------------------------------------------------------------
@app.get("/api/lora/stats")
def lora_stats():
    """Get LoRA training data collection statistics."""
    if data_collector is None:
        return {"error": "DataCollector not initialized"}
    return data_collector.get_stats()


@app.post("/api/lora/collect/toggle")
def lora_toggle():
    """Toggle data collection on/off."""
    if data_collector is None:
        return {"error": "DataCollector not initialized"}
    new_state = data_collector.toggle()
    return {"enabled": new_state}


@app.get("/api/lora/status")
def lora_status():
    """Get LoRA adapter and training readiness status."""
    from pathlib import Path

    adapter_path = config.LORA_ADAPTER_PATH
    adapter_exists = Path(adapter_path, "adapter_config.json").exists() if adapter_path else False

    result = {
        "lora_enabled": config.LORA_ENABLED,
        "adapter_path": adapter_path,
        "adapter_exists": adapter_exists,
        "data_collection_enabled": data_collector.enabled if data_collector else False,
    }

    if data_collector:
        result["training_readiness"] = data_collector.export_for_training()

    return result

