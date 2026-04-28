"""Centralized HIO v3 model-server configuration."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_early() -> None:
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


BASE_DIR = Path(os.getenv("MODEL_SERVER_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = Path(os.getenv("MODEL_SERVER_DATA_DIR", str(BASE_DIR / "data")))
LOG_DIR = Path(os.getenv("MODEL_SERVER_LOG_DIR", str(DATA_DIR / "logs")))
MODELS_DIR = Path(os.getenv("MODEL_SERVER_MODELS_DIR", str(BASE_DIR / "models")))

# HIO v3 model hot path
HIO_V3_ENABLED = _env_bool("HIO_V3_ENABLED", True)
HIO_V3_PIPELINE_VERSION = os.getenv(
    "HIO_V3_PIPELINE_VERSION",
    "v3-yolo26-tier1-siglip-episode-gemini",
)
V3_SCENARIOS = [
    item.strip().lower()
    for item in os.getenv("V3_SCENARIOS", "cash,fire,violence").split(",")
    if item.strip()
]
YOLO26_POSE_WEIGHTS = os.getenv("YOLO26_POSE_WEIGHTS", str(MODELS_DIR / "yolo26s-pose.pt"))
YOLO26_DETECT_WEIGHTS = os.getenv("YOLO26_DETECT_WEIGHTS", "").strip()
YOLO26_CASH_WEIGHTS = os.getenv("YOLO26_CASH_WEIGHTS", "").strip()
YOLO26_FIRE_WEIGHTS = os.getenv("YOLO26_FIRE_WEIGHTS", "").strip()
YOLO26_DEVICE = os.getenv("YOLO26_DEVICE", "cuda")
ALLOW_CPU_FALLBACK = _env_bool("ALLOW_CPU_FALLBACK", False)
YOLO26_IMGSZ = _env_int("YOLO26_IMGSZ", 640)
YOLO26_POSE_CONF = _env_float("YOLO26_POSE_CONF", 0.25)
YOLO26_DETECT_CONF = _env_float("YOLO26_DETECT_CONF", 0.25)
V3_CASH_EPISODE_WINDOW_SEC = _env_float("V3_CASH_EPISODE_WINDOW_SEC", 5.0)
V3_CASH_MIN_HITS = _env_int("V3_CASH_MIN_HITS", 2)
V3_CASH_EXCHANGE_MARGIN_PX = _env_float("V3_CASH_EXCHANGE_MARGIN_PX", 90.0)
V3_CASH_WRIST_PROXIMITY_PX = _env_float("V3_CASH_WRIST_PROXIMITY_PX", 170.0)

V3_POSE_FPS = _env_float("V3_POSE_FPS", 3.0)
V3_DETECT_FPS = _env_float("V3_DETECT_FPS", 1.5)
V3_VALIDATION_CLIP_SEC = _env_float("V3_VALIDATION_CLIP_SEC", 15.0)
V3_CASH_PREFILTER_THRESHOLD = _env_float("V3_CASH_PREFILTER_THRESHOLD", 0.45)
V3_FIRE_PREFILTER_THRESHOLD = _env_float("V3_FIRE_PREFILTER_THRESHOLD", 0.35)
V3_VIOLENCE_PREFILTER_THRESHOLD = _env_float("V3_VIOLENCE_PREFILTER_THRESHOLD", 0.45)
V3_GEMINI_ALWAYS_VALIDATE = _env_bool("V3_GEMINI_ALWAYS_VALIDATE", True)
V3_SEMANTIC_FILTER_ENABLED = _env_bool("V3_SEMANTIC_FILTER_ENABLED", True)
V3_SEMANTIC_MODEL = os.getenv("V3_SEMANTIC_MODEL", "google/siglip-base-patch16-224")
V3_SEMANTIC_DEVICE = os.getenv("V3_SEMANTIC_DEVICE", YOLO26_DEVICE)
V3_CASH_SIGLIP_CLIP_ENABLED = _env_bool("V3_CASH_SIGLIP_CLIP_ENABLED", True)
V3_CASH_SIGLIP_CLIP_FRAMES = _env_int("V3_CASH_SIGLIP_CLIP_FRAMES", 12)
V3_CASH_SIGLIP_CLIP_BATCH_SIZE = _env_int("V3_CASH_SIGLIP_CLIP_BATCH_SIZE", 4)
V3_CASH_SIGLIP_CLIP_WINDOW_SEC = _env_float("V3_CASH_SIGLIP_CLIP_WINDOW_SEC", 15.0)
V3_CASH_SIGLIP_CLIP_PEAK_WINDOW_SEC = _env_float("V3_CASH_SIGLIP_CLIP_PEAK_WINDOW_SEC", 5.0)
V3_CASH_SIGLIP_CLIP_MIN_SCORE = _env_float("V3_CASH_SIGLIP_CLIP_MIN_SCORE", 0.50)
V3_CASH_SIGLIP_CLIP_FRAME_POSITIVE = _env_float("V3_CASH_SIGLIP_CLIP_FRAME_POSITIVE", 0.48)
V3_CASH_SIGLIP_CLIP_MIN_POSITIVE_FRAMES = _env_int("V3_CASH_SIGLIP_CLIP_MIN_POSITIVE_FRAMES", 2)
V3_CASH_SIGLIP_CLIP_COOLDOWN_SEC = _env_float("V3_CASH_SIGLIP_CLIP_COOLDOWN_SEC", 2.0)
V3_FIRE_SIGLIP_MIN_SCORE = _env_float("V3_FIRE_SIGLIP_MIN_SCORE", 0.52)
V3_FIRE_NEUTRALIZER_THRESHOLD = _env_float("V3_FIRE_NEUTRALIZER_THRESHOLD", 0.58)

# Tier-2 SigLIP GATE: if SigLIP semantic score for the scenario is BELOW the
# gate threshold, the Tier-1 pose/object signal is suppressed (not just no
# bonus). This prevents wasting Gemini calls when SigLIP strongly disagrees
# with pose (e.g. no cashier visible, no fight happening).
V3_CASH_SIGLIP_GATE = _env_float("V3_CASH_SIGLIP_GATE", 0.30)
V3_VIOLENCE_SIGLIP_GATE = _env_float("V3_VIOLENCE_SIGLIP_GATE", 0.25)
V3_FIRE_SIGLIP_FLOOR = _env_float("V3_FIRE_SIGLIP_FLOOR", 0.15)

# Tier-2 fine-tuned SigLIP2 classifier heads (Apache 2.0 open weights)
V3_FIRE_CLASSIFIER_ENABLED = _env_bool("V3_FIRE_CLASSIFIER_ENABLED", True)
V3_FIRE_CLASSIFIER_MODEL = os.getenv(
    "V3_FIRE_CLASSIFIER_MODEL", "prithivMLmods/Fire-Detection-Siglip2"
)
V3_ACTION_CLASSIFIER_ENABLED = _env_bool("V3_ACTION_CLASSIFIER_ENABLED", True)
V3_ACTION_CLASSIFIER_MODEL = os.getenv(
    "V3_ACTION_CLASSIFIER_MODEL", "prithivMLmods/Human-Action-Recognition"
)
V3_CLASSIFIER_DEVICE = os.getenv("V3_CLASSIFIER_DEVICE", YOLO26_DEVICE)
V3_ACTION_FIGHT_MIN_SCORE = _env_float("V3_ACTION_FIGHT_MIN_SCORE", 0.40)
V3_ACTION_NEUTRAL_DAMPEN = _env_float("V3_ACTION_NEUTRAL_DAMPEN", 0.50)

V3_CLIP_ARTIFACT_MODE = os.getenv("V3_CLIP_ARTIFACT_MODE", "minimal").strip().lower()
V3_EPISODE_COOLDOWN_SEC = _env_float("V3_EPISODE_COOLDOWN_SEC", 20.0)
V3_EPISODE_MAX_GAP_SEC = _env_float("V3_EPISODE_MAX_GAP_SEC", 6.0)

# CPU-saving knobs
INGEST_DOWNSAMPLE_HEIGHT = _env_int("INGEST_DOWNSAMPLE_HEIGHT", 720)
V3_OVERLAY_FPS_DIVISOR = _env_int("V3_OVERLAY_FPS_DIVISOR", 3)
# Skeleton overlay on clip:
#   0  (default)  = NO skeleton on any frame — clip shows raw video + cashier ROI
#                   polygon only. No ghost because ROI is truly static.
#   N (>0)        = skeleton drawn on first N frames of overlay clip, then ROI
#                   only. Use if operator needs to see pose snapshot at trigger.
# Set to >0 only if needed; cashier ROI + raw video is enough for Gemini to judge.
V3_OVERLAY_SKELETON_FRAMES = _env_int("V3_OVERLAY_SKELETON_FRAMES", 0)
FFMPEG_PRESET = os.getenv("FFMPEG_PRESET", "ultrafast").strip()
FFMPEG_CRF = _env_int("FFMPEG_CRF", 28)
# Encoder: "libx264" (CPU, ubiquitous), "h264_nvenc" (NVIDIA GPU, 10x faster),
#          or "auto" to probe for nvenc and fall back to libx264.
FFMPEG_ENCODER = os.getenv("FFMPEG_ENCODER", "libx264").strip().lower()
FFMPEG_NVENC_PRESET = os.getenv("FFMPEG_NVENC_PRESET", "p3").strip()
FFMPEG_NVENC_CQ = _env_int("FFMPEG_NVENC_CQ", 28)

# Frontend MJPEG live preview throttle (reduces idle CPU)
# - FPS: base streaming fps when no active event on camera
# - BURST_FPS: bumped fps for INFERENCE_ACTIVE_BURST_SEC after a detection,
#              so live viewer sees smooth motion when it actually matters.
#              Defaults to FPS (no bump) unless explicitly set.
# - WIDTH: downscale MJPEG frames to this width (0 = keep source resolution).
#          YOLO ingest path is unaffected; downscale is MJPEG-only.
# - QUALITY: JPEG quality 10..95
FRONTEND_MJPEG_FPS = _env_float("FRONTEND_MJPEG_FPS", 3.0)
FRONTEND_MJPEG_BURST_FPS = _env_float("FRONTEND_MJPEG_BURST_FPS", 0.0)
FRONTEND_MJPEG_QUALITY = _env_int("FRONTEND_MJPEG_QUALITY", 50)
FRONTEND_MJPEG_WIDTH = _env_int("FRONTEND_MJPEG_WIDTH", 0)
FRONTEND_MJPEG_IDLE_PAUSE_SEC = _env_float("FRONTEND_MJPEG_IDLE_PAUSE_SEC", 5.0)
FRONTEND_MJPEG_DEDUP_FRAMES = _env_bool("FRONTEND_MJPEG_DEDUP_FRAMES", True)

# Gemini temporal validator
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_TEMPERATURE = _env_float("GEMINI_TEMPERATURE", 0.1)
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 1500)
GEMINI_TIMEOUT_SEC = _env_float("GEMINI_TIMEOUT_SEC", 90.0)
GEMINI_MAX_CONCURRENT = _env_int("GEMINI_MAX_CONCURRENT", 2)

# RTSP and scheduling
BASE_FPS = _env_float("BASE_FPS", 1.5)
BURST_FPS = _env_float("BURST_FPS", 4.0)
BURST_DURATION_SEC = _env_float("BURST_DURATION_SEC", 3.0)
SINGLE_CAMERA_MODE = _env_bool("SINGLE_CAMERA_MODE", False)
GLOBAL_INFERENCE_LOCK = _env_bool("GLOBAL_INFERENCE_LOCK", True)
INFERENCE_WORKERS = _env_int("INFERENCE_WORKERS", 1)
INFERENCE_QUEUE_SIZE = _env_int("INFERENCE_QUEUE_SIZE", 128)
INFERENCE_ACTIVE_BURST_SEC = _env_float("INFERENCE_ACTIVE_BURST_SEC", 3.0)
INFERENCE_ACTIVE_BURST_FPS = _env_float("INFERENCE_ACTIVE_BURST_FPS", 3.0)
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
RTSP_OPEN_TIMEOUT_MS = _env_int("RTSP_OPEN_TIMEOUT_MS", 8000)
RTSP_READ_TIMEOUT_MS = _env_int("RTSP_READ_TIMEOUT_MS", 8000)
RTSP_HWACCEL = os.getenv("RTSP_HWACCEL", "none").strip().lower()
RTSP_HWACCEL_DEVICE = os.getenv("RTSP_HWACCEL_DEVICE", "0").strip()
RTSP_HWACCEL_DECODER = os.getenv("RTSP_HWACCEL_DECODER", "").strip()
RTSP_HWACCEL_ALLOW_FALLBACK = _env_bool("RTSP_HWACCEL_ALLOW_FALLBACK", True)
STALE_THRESHOLD_SEC = _env_float("STALE_THRESHOLD_SEC", 2.5)
CLIP_BUFFER_SECONDS = _env_int("CLIP_BUFFER_SECONDS", 30)
VIDEO_CLIP_SECONDS = _env_int("VIDEO_CLIP_SECONDS", 10)
EVIDENCE_MODE = os.getenv("EVIDENCE_MODE", "video_only")

# Persistence
DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8001")
FLUSH_ENDPOINT = os.getenv("FLUSH_ENDPOINT", "/api/flush")
FLUSH_INTERVAL_SEC = _env_int("FLUSH_INTERVAL_SEC", 120)
FLUSH_MAX_RETRIES = _env_int("FLUSH_MAX_RETRIES", 3)
LOCAL_RETENTION_DAYS = _env_int("LOCAL_RETENTION_DAYS", 3)
V3_PROPOSAL_LOG_PERSIST = _env_bool("V3_PROPOSAL_LOG_PERSIST", True)
V3_PROPOSAL_LOG_DIR = Path(os.getenv("V3_PROPOSAL_LOG_DIR", str(DATA_DIR / "v3_proposal_logs")))
V3_PROPOSAL_FEEDBACK_DIR = Path(os.getenv("V3_PROPOSAL_FEEDBACK_DIR", str(DATA_DIR / "v3_feedback")))

# Media export
USE_S3 = _env_bool("USE_S3", False)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME", "")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
CLIP_SAVE_MAX_CONCURRENT = _env_int("CLIP_SAVE_MAX_CONCURRENT", 1)
POSTPROCESS_WORKERS = _env_int("POSTPROCESS_WORKERS", 1)
POSTPROCESS_QUEUE_SIZE = _env_int("POSTPROCESS_QUEUE_SIZE", 128)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def load_dotenv(path: str | None = None) -> None:
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    if path:
        _load(path, override=True)
        return
    default_env = BASE_DIR / ".env"
    if default_env.exists():
        _load(str(default_env), override=True)
