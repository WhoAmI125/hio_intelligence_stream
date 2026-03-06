"""
Centralized Configuration — Model Server settings management.

Loads from environment variables (.env) with sensible defaults.
All thresholds, model paths, API keys, and FPS settings live here.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _load_dotenv_early() -> None:
    """
    Load .env before computing module-level settings.

    This ensures environment-backed constants below (e.g. FLORENCE_DEVICE)
    use values from project .env on first import.
    """
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return

    default_env = Path(__file__).resolve().parent.parent / ".env"
    env_path = Path(os.getenv("MODEL_SERVER_ENV_FILE", str(default_env)))
    if env_path.exists():
        _load(str(env_path), override=True)


_load_dotenv_early()


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(os.getenv(
    "MODEL_SERVER_BASE_DIR",
    str(Path(__file__).resolve().parent.parent)
))

DATA_DIR = Path(os.getenv("MODEL_SERVER_DATA_DIR", str(BASE_DIR / "data")))
LOG_DIR = Path(os.getenv("MODEL_SERVER_LOG_DIR", str(DATA_DIR / "logs")))
MODELS_DIR = Path(os.getenv("MODEL_SERVER_MODELS_DIR", str(BASE_DIR / "models")))

# ---------------------------------------------------------------------------
# Florence-2 (Tier 1)
# ---------------------------------------------------------------------------

FLORENCE_MODEL = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-large")
FLORENCE_BACKEND = os.getenv("FLORENCE_BACKEND", "pytorch")  # pytorch | openvino
FLORENCE_DEVICE = os.getenv("FLORENCE_DEVICE", "cuda")       # auto | cpu | cuda
FLORENCE_INPUT_SIZE = _env_int("FLORENCE_INPUT_SIZE", 448)
FLORENCE_DTYPE = os.getenv("FLORENCE_DTYPE", "float32")      # float32 | float16

# ---------------------------------------------------------------------------
# Gemini (Tier 2)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE = _env_float("GEMINI_TEMPERATURE", 0.1)
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 1500)
GEMINI_TIMEOUT_SEC = _env_float("GEMINI_TIMEOUT_SEC", 30.0)

# ---------------------------------------------------------------------------
# Detection Thresholds
# ---------------------------------------------------------------------------

CASH_THRESHOLD = _env_float("CASH_THRESHOLD", 0.30)
VIOLENCE_THRESHOLD = _env_float("VIOLENCE_THRESHOLD", 0.30)
FIRE_THRESHOLD = _env_float("FIRE_THRESHOLD", 0.30)

# Uncertainty Gate — tier2 escalation thresholds
TIER2_FIRE_THRESHOLD = _env_float("TIER2_FIRE_THRESHOLD", 0.60)
TIER2_VIOLENCE_THRESHOLD = _env_float("TIER2_VIOLENCE_THRESHOLD", 0.70)
TIER2_CASH_THRESHOLD = _env_float("TIER2_CASH_THRESHOLD", 0.55)

# Confidence + stability thresholds to skip Tier2
SKIP_CONFIDENCE = _env_float("SKIP_CONFIDENCE", 0.85)
SKIP_STABILITY = _env_float("SKIP_STABILITY", 0.90)

# ---------------------------------------------------------------------------
# Stream Sampling (RTSP)
# ---------------------------------------------------------------------------

BASE_FPS = _env_float("BASE_FPS", 1.5)
BURST_FPS = _env_float("BURST_FPS", 4.0)
BURST_DURATION_SEC = _env_float("BURST_DURATION_SEC", 3.0)
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
RTSP_OPEN_TIMEOUT_MS = _env_int("RTSP_OPEN_TIMEOUT_MS", 8000)
RTSP_READ_TIMEOUT_MS = _env_int("RTSP_READ_TIMEOUT_MS", 8000)
STALE_THRESHOLD_SEC = _env_float("STALE_THRESHOLD_SEC", 2.5)
CLIP_BUFFER_SECONDS = _env_int("CLIP_BUFFER_SECONDS", 30)

# ---------------------------------------------------------------------------
# Episode Manager
# ---------------------------------------------------------------------------

EPISODE_MIN_DETECTIONS = _env_int("EPISODE_MIN_DETECTIONS", 2)
EPISODE_STABILITY_THRESHOLD = _env_float("EPISODE_STABILITY_THRESHOLD", 0.65)
EPISODE_COOLDOWN_SEC = _env_int("EPISODE_COOLDOWN_SEC", 60)
EPISODE_MAX_PER_TYPE = _env_int("EPISODE_MAX_PER_TYPE", 3)

# ---------------------------------------------------------------------------
# Evidence Router
# ---------------------------------------------------------------------------

GEMINI_TARGET_RATIO = _env_float("GEMINI_TARGET_RATIO", 0.30)
GEMINI_RATIO_PENALTY = _env_float("GEMINI_RATIO_PENALTY", 0.25)
VIDEO_CLIP_SECONDS = _env_int("VIDEO_CLIP_SECONDS", 10)
EVIDENCE_MODE = os.getenv("EVIDENCE_MODE", "hybrid")

# ---------------------------------------------------------------------------
# Critic / Evolution
# ---------------------------------------------------------------------------

CRITIC_ENABLED = _env_bool("CRITIC_ENABLED", False)
CRITIC_SHADOW_MODE = _env_bool("CRITIC_SHADOW_MODE", True)
CRITIC_MODEL_DIR = str(DATA_DIR / "critic_models")
CRITIC_MIN_SAMPLES = _env_int("CRITIC_MIN_SAMPLES", 30)

# Shadow Agent
SHADOW_BATCH_SIZE = _env_int("SHADOW_BATCH_SIZE", 30)
SHADOW_PERSIST_DIR = str(DATA_DIR / "shadow_feedback")
SHADOW_MAX_QUEUE = _env_int("SHADOW_MAX_QUEUE", 200)
SHADOW_DISAGREE_THRESHOLD = _env_float("SHADOW_DISAGREE_THRESHOLD", 0.30)

# Rule Updater
RULE_PROMPTS_DIR = os.getenv("RULE_PROMPTS_DIR", "")
if not RULE_PROMPTS_DIR:
    RULE_PROMPTS_DIR = str(Path(__file__).resolve().parent / "agents" / "prompts")
RULE_VERSIONS_DIR = str(DATA_DIR / "rule_versions")

# ---------------------------------------------------------------------------
# LoRA Fine-tuning
# ---------------------------------------------------------------------------

LORA_ENABLED = _env_bool("LORA_ENABLED", False)
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", str(DATA_DIR / "lora_output"))
LORA_DATA_COLLECTION = _env_bool("LORA_DATA_COLLECTION", True)
LORA_COLLECT_NORMAL_RATIO = _env_float("LORA_COLLECT_NORMAL_RATIO", 0.05)
LORA_DATA_DIR = os.getenv("LORA_DATA_DIR", str(DATA_DIR / "lora_training"))
LORA_MAX_SAMPLES = _env_int("LORA_MAX_SAMPLES", 50000)

# ---------------------------------------------------------------------------
# Flush Worker (Model -> DB Server)
# ---------------------------------------------------------------------------

DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8001")
FLUSH_ENDPOINT = os.getenv("FLUSH_ENDPOINT", "/api/flush")
FLUSH_INTERVAL_SEC = _env_int("FLUSH_INTERVAL_SEC", 3600)
FLUSH_MAX_RETRIES = _env_int("FLUSH_MAX_RETRIES", 3)

# ---------------------------------------------------------------------------
# Media Export / Storage
# ---------------------------------------------------------------------------

# Keep S3 disabled by default for local-first operation.
USE_S3 = _env_bool("USE_S3", False)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME", "")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# ---------------------------------------------------------------------------
# Router Steps (append-only JSONL for critic training)
# ---------------------------------------------------------------------------

ROUTER_STEPS_PATH = os.getenv(
    "ROUTER_STEPS_PATH",
    str(DATA_DIR / "router_steps.jsonl")
)

# ---------------------------------------------------------------------------
# Cash Dual Path
# ---------------------------------------------------------------------------

CASH_DUAL_PATH_ENABLED = _env_bool("CASH_DUAL_PATH_ENABLED", True)
CASH_GLOBAL_ASSIST_THRESHOLD = _env_float("CASH_GLOBAL_ASSIST_THRESHOLD", 0.30)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)


def as_dict() -> Dict[str, Any]:
    """Export all settings as a flat dictionary (for debugging/logging)."""
    return {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }


def load_dotenv(path: Optional[str] = None) -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv as _load
        env_path = path or os.getenv("MODEL_SERVER_ENV_FILE", str(BASE_DIR / ".env"))
        if os.path.exists(env_path):
            _load(env_path, override=True)
    except ImportError:
        pass
