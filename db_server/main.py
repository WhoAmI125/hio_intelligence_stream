"""
DB Server — FastAPI entry point with SQLite storage.

Tier 2 backend for event persistence and query APIs.
Replaces Django ORM with direct SQLite for local standalone execution.

Endpoints:
    GET  /             → health check
    POST /api/flush    → receive events + clips from model_server
    GET  /api/events   → list events (paginated, filterable)
    GET  /api/events/{event_id} → single event detail
    POST /api/feedback → submit human feedback
    GET  /api/stats    → aggregate statistics
"""

import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("db_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "data/cctv_events.db")
MEDIA_ROOT = os.getenv("DB_MEDIA_ROOT", "data/media_archive")


# ---------------------------------------------------------------------------
# SQLite helper
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    os.makedirs(MEDIA_ROOT, exist_ok=True)

    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            camera_id TEXT DEFAULT '',
            event_type TEXT DEFAULT '',
            scenario TEXT DEFAULT '',
            confidence REAL DEFAULT 0.0,
            tier INTEGER DEFAULT 1,
            is_detected INTEGER DEFAULT 0,
            gemini_validated INTEGER,
            gemini_confidence REAL,
            gemini_reason TEXT DEFAULT '',
            caption TEXT DEFAULT '',
            matched_keywords TEXT DEFAULT '[]',
            evidence TEXT DEFAULT '',
            clip_path TEXT DEFAULT '',
            human_feedback TEXT,
            event_data TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS episode_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            event_id TEXT DEFAULT '',
            camera_id TEXT DEFAULT '',
            event_type TEXT DEFAULT '',
            final_policy TEXT DEFAULT '',
            is_valid_event INTEGER DEFAULT 0,
            note TEXT DEFAULT '',
            review_status TEXT DEFAULT 'queued',
            reviewer TEXT DEFAULT '',
            gemini_validated INTEGER,
            gemini_confidence REAL,
            gemini_reason TEXT DEFAULT '',
            tier1_snapshot TEXT DEFAULT '{}',
            router_snapshot TEXT DEFAULT '{}',
            florence_signals TEXT DEFAULT '{}',
            feedback_suggestion TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS worker_leases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT UNIQUE NOT NULL,
            instance_id TEXT NOT NULL,
            pid INTEGER DEFAULT 0,
            rtsp_url TEXT DEFAULT '',
            acquired_at TEXT DEFAULT (datetime('now', 'localtime')),
            last_heartbeat TEXT DEFAULT (datetime('now', 'localtime')),
            lease_ttl_sec INTEGER DEFAULT 60
        );

        CREATE TABLE IF NOT EXISTS gemini_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            camera_id TEXT DEFAULT '',
            event_type TEXT DEFAULT '',
            gemini_state TEXT DEFAULT '',
            gemini_validated INTEGER,
            gemini_confidence REAL,
            gemini_reason TEXT DEFAULT '',
            validation_type TEXT DEFAULT '',
            input_mode TEXT DEFAULT '',
            prompt_version TEXT DEFAULT '',
            processing_time_ms INTEGER,
            media_ref TEXT DEFAULT '',
            log_data TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT UNIQUE NOT NULL,
            rtsp_url TEXT NOT NULL,
            base_fps REAL DEFAULT 1.5,
            rtsp_transport TEXT DEFAULT 'tcp',
            open_timeout_ms INTEGER DEFAULT 8000,
            read_timeout_ms INTEGER DEFAULT 8000,
            event_cooldown_sec INTEGER DEFAULT 20,
            clip_duration_sec INTEGER DEFAULT 10,
            validation_clip_sec INTEGER DEFAULT 10,
            evidence_mode TEXT DEFAULT 'hybrid',
            use_video_validation INTEGER DEFAULT 1,
            cashier_zone TEXT DEFAULT '[]',
            drawer_zone TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT DEFAULT (datetime('now', 'localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
        CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id);
        CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
        CREATE INDEX IF NOT EXISTS idx_gemini_logs_event_id ON gemini_logs(event_id);
        CREATE INDEX IF NOT EXISTS idx_gemini_logs_created ON gemini_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_cameras_updated_at ON cameras(updated_at);
    """)
    conn.close()
    logger.info(f"Database initialized: {DB_PATH}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("DB Server ready.")
    yield
    logger.info("DB Server stopped.")


app = FastAPI(
    title="Intelligent CCTV DB Server",
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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    event_id: str
    decision: str  # "accept" | "decline" | "unsure"
    note: str = ""
    reviewer: str = ""


class CameraConfigRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    base_fps: float = 1.5
    rtsp_transport: str = "tcp"
    open_timeout_ms: int = 8000
    read_timeout_ms: int = 8000
    event_cooldown_sec: int = 20
    clip_duration_sec: int = 10
    validation_clip_sec: int = 10
    evidence_mode: str = "hybrid"
    use_video_validation: bool = True
    cashier_zone: list[list[float]] = []
    drawer_zone: list[list[float]] = []


def _is_valid_rtsp_url(rtsp_url: str) -> bool:
    raw = str(rtsp_url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    if (parsed.scheme or "").lower() not in {"rtsp", "rtsps"}:
        return False
    return bool(parsed.hostname)


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
        if not (x == x and y == y):
            continue
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        out.append([x, y])
    return out


def _camera_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    cashier_zone: list[list[float]] = []
    drawer_zone: list[list[float]] = []
    try:
        cashier_zone = _normalize_zone_points(json.loads(row["cashier_zone"] or "[]"))
    except Exception:
        cashier_zone = []
    try:
        drawer_zone = _normalize_zone_points(json.loads(row["drawer_zone"] or "[]"))
    except Exception:
        drawer_zone = []

    return {
        "camera_id": row["camera_id"],
        "rtsp_url": row["rtsp_url"],
        "base_fps": float(row["base_fps"] or 1.5),
        "rtsp_transport": row["rtsp_transport"] or "tcp",
        "open_timeout_ms": int(row["open_timeout_ms"] or 8000),
        "read_timeout_ms": int(row["read_timeout_ms"] or 8000),
        "event_cooldown_sec": int(row["event_cooldown_sec"] or 20),
        "clip_duration_sec": int(row["clip_duration_sec"] or 10),
        "validation_clip_sec": int(row["validation_clip_sec"] or 10),
        "evidence_mode": row["evidence_mode"] or "hybrid",
        "use_video_validation": bool(row["use_video_validation"]),
        "cashier_zone": cashier_zone,
        "drawer_zone": drawer_zone,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "service": "db_server", "db_path": DB_PATH}


@app.get("/api/cameras")
def list_cameras():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM cameras ORDER BY updated_at DESC, camera_id ASC"
    ).fetchall()
    conn.close()
    return {"cameras": [_camera_row_to_dict(r) for r in rows]}


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str):
    camera_id = str(camera_id or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM cameras WHERE camera_id = ?",
        (camera_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"camera": _camera_row_to_dict(row)}


def _upsert_camera(camera_id: str, req: CameraConfigRequest) -> dict[str, Any]:
    camera_id = str(camera_id or "").strip()
    rtsp_url = str(req.rtsp_url or "").strip()

    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")
    if not _is_valid_rtsp_url(rtsp_url):
        raise HTTPException(status_code=400, detail="Valid rtsp_url is required")

    payload = {
        "camera_id": camera_id,
        "rtsp_url": rtsp_url,
        "base_fps": float(req.base_fps or 1.5),
        "rtsp_transport": (str(req.rtsp_transport or "tcp").strip() or "tcp"),
        "open_timeout_ms": int(req.open_timeout_ms or 8000),
        "read_timeout_ms": int(req.read_timeout_ms or 8000),
        "event_cooldown_sec": int(req.event_cooldown_sec or 20),
        "clip_duration_sec": int(req.clip_duration_sec or 10),
        "validation_clip_sec": int(req.validation_clip_sec or 10),
        "evidence_mode": (str(req.evidence_mode or "hybrid").strip() or "hybrid"),
        "use_video_validation": 1 if req.use_video_validation else 0,
        "cashier_zone": json.dumps(_normalize_zone_points(req.cashier_zone), ensure_ascii=False),
        "drawer_zone": json.dumps(_normalize_zone_points(req.drawer_zone), ensure_ascii=False),
    }

    conn = get_db()
    conn.execute(
        """
        INSERT INTO cameras (
            camera_id, rtsp_url, base_fps, rtsp_transport, open_timeout_ms, read_timeout_ms,
            event_cooldown_sec, clip_duration_sec, validation_clip_sec, evidence_mode,
            use_video_validation, cashier_zone, drawer_zone, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'))
        ON CONFLICT(camera_id) DO UPDATE SET
            rtsp_url = excluded.rtsp_url,
            base_fps = excluded.base_fps,
            rtsp_transport = excluded.rtsp_transport,
            open_timeout_ms = excluded.open_timeout_ms,
            read_timeout_ms = excluded.read_timeout_ms,
            event_cooldown_sec = excluded.event_cooldown_sec,
            clip_duration_sec = excluded.clip_duration_sec,
            validation_clip_sec = excluded.validation_clip_sec,
            evidence_mode = excluded.evidence_mode,
            use_video_validation = excluded.use_video_validation,
            cashier_zone = excluded.cashier_zone,
            drawer_zone = excluded.drawer_zone,
            updated_at = datetime('now', 'localtime')
        """,
        (
            payload["camera_id"],
            payload["rtsp_url"],
            payload["base_fps"],
            payload["rtsp_transport"],
            payload["open_timeout_ms"],
            payload["read_timeout_ms"],
            payload["event_cooldown_sec"],
            payload["clip_duration_sec"],
            payload["validation_clip_sec"],
            payload["evidence_mode"],
            payload["use_video_validation"],
            payload["cashier_zone"],
            payload["drawer_zone"],
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM cameras WHERE camera_id = ?",
        (camera_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=500, detail="Camera upsert failed")
    return _camera_row_to_dict(row)


@app.post("/api/cameras")
def create_camera(req: CameraConfigRequest):
    camera = _upsert_camera(str(req.camera_id or "").strip(), req)
    return {"status": "ok", "camera": camera}


@app.put("/api/cameras/{camera_id}")
def update_camera(camera_id: str, req: CameraConfigRequest):
    req_camera_id = str(req.camera_id or "").strip()
    camera_id = str(camera_id or "").strip()
    if req_camera_id and req_camera_id != camera_id:
        raise HTTPException(status_code=400, detail="camera_id mismatch between path and body")
    camera = _upsert_camera(camera_id, req)
    return {"status": "ok", "camera": camera}


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str):
    camera_id = str(camera_id or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")

    conn = get_db()
    cur = conn.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
    conn.commit()
    conn.close()
    if cur.rowcount <= 0:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"status": "ok", "camera_id": camera_id}


@app.post("/api/flush")
async def flush_events(
    metadata: Annotated[str, Form()],
    video_clip: UploadFile | None = File(None),
):
    """
    Receive batched events from model_server's FlushWorker.
    Accepts JSON metadata and optional video clip.
    """
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON metadata"}

    events = payload.get("events", [])
    conn = get_db()
    inserted = 0

    for ev in events:
        try:
            event_id = ev.get("event_id", f"ev_{int(time.time()*1000)}")
            gem = ev.get("gemini", {}) if isinstance(ev.get("gemini", {}), dict) else {}
            gem_valid = gem.get("validated", None)
            gem_valid_i = None
            if gem_valid is True:
                gem_valid_i = 1
            elif gem_valid is False:
                gem_valid_i = 0

            conn.execute("""
                INSERT OR REPLACE INTO events
                (event_id, camera_id, event_type, scenario, confidence, tier,
                 is_detected, gemini_validated, gemini_confidence, gemini_reason,
                 caption, matched_keywords, evidence, clip_path, human_feedback, event_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                ev.get("camera_id", ""),
                ev.get("event_type", ev.get("scenario", "")),
                ev.get("scenario", ""),
                ev.get("confidence", 0.0),
                ev.get("tier", 1),
                1 if ev.get("is_detected") else 0,
                gem_valid_i,
                gem.get("confidence", None),
                str(gem.get("reason", "")),
                ev.get("caption", ""),
                json.dumps(ev.get("matched_keywords", []), ensure_ascii=False),
                ev.get("evidence", ""),
                ev.get("clip_url", ev.get("clip_path", "")),
                (
                    json.dumps(ev.get("human_feedback"), ensure_ascii=False, default=str)
                    if ev.get("human_feedback") is not None
                    else None
                ),
                json.dumps(ev, ensure_ascii=False, default=str),
            ))

            if gem:
                conn.execute("""
                    INSERT OR REPLACE INTO gemini_logs
                    (event_id, camera_id, event_type, gemini_state, gemini_validated,
                     gemini_confidence, gemini_reason, validation_type, input_mode,
                     prompt_version, processing_time_ms, media_ref, log_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    ev.get("camera_id", ""),
                    ev.get("event_type", ev.get("scenario", "")),
                    str(gem.get("state", "")),
                    gem_valid_i,
                    gem.get("confidence", None),
                    str(gem.get("reason", "")),
                    str(gem.get("validation_type", "")),
                    str(gem.get("input_mode", "")),
                    str(gem.get("prompt_version", "")),
                    int(gem.get("processing_time_ms", 0) or 0),
                    str(gem.get("media_ref", "")),
                    json.dumps(gem, ensure_ascii=False, default=str),
                ))
            inserted += 1
        except Exception as e:
            logger.error(f"Insert failed for event: {e}")

    conn.commit()
    conn.close()

    # Save video clip if provided
    clip_path = ""
    if video_clip:
        ext = os.path.splitext(video_clip.filename or ".mp4")[1]
        clip_dir = os.path.join(MEDIA_ROOT, datetime.now().strftime("%Y%m%d"))
        os.makedirs(clip_dir, exist_ok=True)
        clip_path = os.path.join(clip_dir, f"clip_{int(time.time())}{ext}")
        with open(clip_path, "wb") as f:
            content = await video_clip.read()
            f.write(content)

    logger.info(f"Flush received: {inserted}/{len(events)} events, clip={clip_path or 'none'}")
    return {"status": "ok", "inserted": inserted, "total": len(events)}


@app.get("/api/events")
def list_events(
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 20,
    scenario: str | None = None,
    camera_id: str | None = None,
):
    """List events with pagination and optional filtering."""
    conn = get_db()
    offset = (page - 1) * per_page

    where_clauses = []
    params: list = []

    if scenario:
        where_clauses.append("scenario = ?")
        params.append(scenario)
    if camera_id:
        where_clauses.append("camera_id = ?")
        params.append(camera_id)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    # Count
    count = conn.execute(
        f"SELECT COUNT(*) FROM events WHERE {where_sql}", params
    ).fetchone()[0]

    # Fetch
    rows = conn.execute(
        f"SELECT * FROM events WHERE {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()

    conn.close()

    return {
        "page": page,
        "per_page": per_page,
        "total": count,
        "total_pages": (count + per_page - 1) // per_page,
        "events": [dict(r) for r in rows],
    }


@app.get("/api/events/{event_id}")
def get_event(event_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    conn.close()

    if row is None:
        return {"error": "Event not found"}
    return dict(row)


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit human feedback (accept/decline/unsure) for an event."""
    conn = get_db()
    feedback = json.dumps({
        "decision": req.decision,
        "note": req.note,
        "reviewer": req.reviewer,
        "at": datetime.now().isoformat(),
    })
    conn.execute(
        "UPDATE events SET human_feedback = ? WHERE event_id = ?",
        (feedback, req.event_id),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "event_id": req.event_id}


@app.get("/api/stats")
def aggregate_stats():
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    detected = conn.execute("SELECT COUNT(*) FROM events WHERE is_detected = 1").fetchone()[0]

    by_scenario = conn.execute("""
        SELECT scenario, COUNT(*) as cnt, AVG(confidence) as avg_conf
        FROM events GROUP BY scenario
    """).fetchall()

    by_tier = conn.execute("""
        SELECT tier, COUNT(*) as cnt FROM events GROUP BY tier
    """).fetchall()

    recent = conn.execute("""
        SELECT event_id, scenario, confidence, tier, created_at
        FROM events ORDER BY created_at DESC LIMIT 10
    """).fetchall()

    conn.close()

    return {
        "total_events": total,
        "total_detected": detected,
        "by_scenario": [dict(r) for r in by_scenario],
        "by_tier": [dict(r) for r in by_tier],
        "recent": [dict(r) for r in recent],
    }
