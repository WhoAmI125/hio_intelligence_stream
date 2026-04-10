"""Centralized configuration — env vars + cameras.yaml."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
CONFIGS_DIR = BASE_DIR / "configs"

DB_DIR = BASE_DIR / "db"

for d in (DATA_DIR, LOG_DIR, DB_DIR, DATA_DIR / "clips", DATA_DIR / "events", DATA_DIR / "thumbnails"):
    d.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)


load_env()

# ── GPU ────────────────────────────────────────────────────────────
CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# ── Models ─────────────────────────────────────────────────────────
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26s-pose.pt")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-L/14")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

# ── Tier 1 thresholds ─────────────────────────────────────────────
CASH_WRIST_THRESHOLD_PX = int(os.getenv("CASH_WRIST_THRESHOLD_PX", "250"))
CLIP_FIRE_THRESHOLD = float(os.getenv("CLIP_FIRE_THRESHOLD", "0.25"))
CLIP_VIOLENCE_THRESHOLD = float(os.getenv("CLIP_VIOLENCE_THRESHOLD", "0.30"))

# ── Tier 2 confidence gates ───────────────────────────────────────
QWEN_CONFIDENCE_HIGH = float(os.getenv("QWEN_CONFIDENCE_HIGH", "0.7"))
QWEN_CONFIDENCE_LOW = float(os.getenv("QWEN_CONFIDENCE_LOW", "0.3"))

# ── Tier 3 (cloud verifier) ───────────────────────────────────────
TIER3_API_KEY = os.getenv("GEMINI_API_KEY", "")
TIER3_MODEL = os.getenv("TIER3_MODEL", "gemini-2.5-flash")

# ── Stream ─────────────────────────────────────────────────────────
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "1.0"))
RING_BUFFER_SECONDS = int(os.getenv("RING_BUFFER_SECONDS", "60"))
CLIP_PRE_SECONDS = int(os.getenv("CLIP_PRE_SECONDS", "6"))
CLIP_POST_SECONDS = int(os.getenv("CLIP_POST_SECONDS", "6"))
TRIGGER_COOLDOWN_SECONDS = int(os.getenv("TRIGGER_COOLDOWN_SECONDS", "60"))

# ── Server ─────────────────────────────────────────────────────────
AUTO_RESTORE_DELAY = int(os.getenv("AUTO_RESTORE_DELAY", "5"))
LOCAL_RETENTION_DAYS = int(os.getenv("LOCAL_RETENTION_DAYS", "7"))
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8002"))
DB_PATH = str(Path(os.getenv("DB_PATH", str(DB_DIR / "events.db"))).resolve())

# ── Storage ────────────────────────────────────────────────────────
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
AWS_BUCKET = os.getenv("AWS_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# ── Alert ──────────────────────────────────────────────────────────
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_SLACK_WEBHOOK = os.getenv("ALERT_SLACK_WEBHOOK", "")
ALERT_LINE_TOKEN = os.getenv("ALERT_LINE_TOKEN", "")


# ── Camera config ──────────────────────────────────────────────────
def load_cameras() -> list[dict]:
    """Load camera configs from cameras.yaml."""
    yaml_path = CONFIGS_DIR / "cameras.yaml"
    if not yaml_path.exists():
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("cameras", [])
