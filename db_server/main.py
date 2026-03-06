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
from typing import Annotated

from fastapi import FastAPI, File, Form, Query, UploadFile
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

        CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
        CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id);
        CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
        CREATE INDEX IF NOT EXISTS idx_gemini_logs_event_id ON gemini_logs(event_id);
        CREATE INDEX IF NOT EXISTS idx_gemini_logs_created ON gemini_logs(created_at);
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "service": "db_server", "db_path": DB_PATH}


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
