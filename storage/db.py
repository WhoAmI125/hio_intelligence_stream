"""SQLite WAL database for event + camera config storage."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import aiosqlite

import config

logger = logging.getLogger(__name__)

_db_path: str = config.DB_PATH
_conn: aiosqlite.Connection | None = None
_conn_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _conn_lock
    if _conn_lock is None:
        _conn_lock = asyncio.Lock()
    return _conn_lock


async def _get_conn() -> aiosqlite.Connection:
    """Return a long-lived singleton connection (created on first call)."""
    global _conn
    if _conn is not None:
        return _conn
    async with _get_lock():
        if _conn is None:
            _conn = await aiosqlite.connect(_db_path)
            await _conn.execute("PRAGMA journal_mode=WAL")
            await _conn.execute("PRAGMA busy_timeout = 5000")
    return _conn


async def close_db() -> None:
    """Close the singleton connection (call on app shutdown)."""
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None


# Auto-migrate: if old DB exists under data/ and new path doesn't, move it
_old_db = config.DATA_DIR / "events.db"
if _old_db.exists() and not Path(_db_path).exists():
    import shutil
    for _suffix in ("", "-wal", "-shm"):
        _src = Path(str(_old_db) + _suffix)
        if _src.exists():
            shutil.move(str(_src), str(_db_path) + _suffix)
    logger.info("Migrated DB from %s to %s", _old_db, _db_path)

# ── Schema ─────────────────────────────────────────────────────────
CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cam_id TEXT NOT NULL,
    scenario TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    tier2_detected BOOLEAN,
    tier2_confidence REAL,
    tier2_reason TEXT,
    tier2_yes_count INTEGER DEFAULT 0,
    tier3_verdict TEXT,
    tier3_reason TEXT,
    clip_path TEXT,
    clip_roi_path TEXT DEFAULT '',
    thumbnail_path TEXT,
    thumbnail_roi_path TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
)
"""

CREATE_CAMERAS_TABLE = """
CREATE TABLE IF NOT EXISTS cameras (
    cam_id TEXT PRIMARY KEY,
    rtsp_url TEXT NOT NULL,
    cashier_zone TEXT DEFAULT '[]',
    customer_zone TEXT DEFAULT '[]',
    wrist_px INTEGER DEFAULT 80,
    enabled BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
)
"""

CREATE_CLIP_REVIEWS_TABLE = """
CREATE TABLE IF NOT EXISTS clip_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    total_frames INTEGER,
    result_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_events_cam_ts ON events(cam_id, timestamp)
"""


async def init_db() -> None:
    conn = await _get_conn()
    await conn.execute(CREATE_EVENTS_TABLE)
    await conn.execute(CREATE_CAMERAS_TABLE)
    await conn.execute(CREATE_CLIP_REVIEWS_TABLE)
    await conn.execute(CREATE_INDEX)
    # Schema migration: add tier2_yes_count if missing (existing DBs)
    for migration in [
        "ALTER TABLE events ADD COLUMN tier2_yes_count INTEGER DEFAULT 0",
        "ALTER TABLE cameras ADD COLUMN trigger_mode TEXT DEFAULT 'wrist'",
    ]:
        try:
            await conn.execute(migration)
        except Exception:
            pass  # column already exists
    await conn.commit()
    logger.info("Database initialized: %s", _db_path)


# ── Camera CRUD ────────────────────────────────────────────────────
async def upsert_camera(cam: dict) -> None:
    cam_id = cam.get("cam_id") or cam.get("id")
    rtsp_url = cam.get("rtsp_url", "")
    cashier = json.dumps(cam.get("cashier_zone", []))
    customer = json.dumps(cam.get("customer_zone", []))

    conn = await _get_conn()
    await conn.execute(
        """INSERT INTO cameras (cam_id, rtsp_url, cashier_zone, customer_zone, wrist_px, enabled)
           VALUES (?, ?, ?, ?, ?, 1)
           ON CONFLICT(cam_id) DO UPDATE SET
             rtsp_url=excluded.rtsp_url,
             cashier_zone=excluded.cashier_zone,
             customer_zone=excluded.customer_zone,
             wrist_px=excluded.wrist_px,
             enabled=1,
             updated_at=datetime('now')""",
        (cam_id, rtsp_url, cashier, customer, cam.get("wrist_px", 80)),
    )
    await conn.commit()


async def get_all_cameras() -> list[dict]:
    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute("SELECT * FROM cameras WHERE enabled = 1")
    rows = await cursor.fetchall()
    conn.row_factory = None
    result = []
    for row in rows:
        d = dict(row)
        d["id"] = d.pop("cam_id")
        for key in ("cashier_zone", "customer_zone"):
            try:
                d[key] = json.loads(d.get(key, "[]"))
            except (json.JSONDecodeError, TypeError):
                d[key] = []
        result.append(d)
    return result


async def update_camera_zones(cam_id: str, cashier_zone: list, customer_zone: list, wrist_px: int, trigger_mode: str = "wrist") -> None:
    conn = await _get_conn()
    await conn.execute(
        """UPDATE cameras SET cashier_zone=?, customer_zone=?, wrist_px=?, trigger_mode=?, updated_at=datetime('now')
           WHERE cam_id=?""",
        (json.dumps(cashier_zone), json.dumps(customer_zone), wrist_px, trigger_mode, cam_id),
    )
    await conn.commit()


async def delete_camera(cam_id: str) -> None:
    conn = await _get_conn()
    await conn.execute("DELETE FROM cameras WHERE cam_id=?", (cam_id,))
    await conn.commit()


# ── Events ─────────────────────────────────────────────────────────
async def insert_event(event: dict) -> int:
    conn = await _get_conn()
    cursor = await conn.execute(
        """INSERT INTO events
           (cam_id, scenario, timestamp, tier2_detected, tier2_confidence,
            tier2_reason, tier2_yes_count, tier3_verdict, tier3_reason,
            clip_path, clip_roi_path, thumbnail_path, thumbnail_roi_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event.get("cam_id"),
            event.get("scenario"),
            event.get("timestamp"),
            event.get("tier2_detected"),
            event.get("tier2_confidence"),
            event.get("tier2_reason"),
            event.get("tier2_yes_count", 0),
            event.get("tier3_verdict"),
            event.get("tier3_reason"),
            event.get("clip_path"),
            event.get("clip_roi_path", ""),
            event.get("thumbnail_path"),
            event.get("thumbnail_roi_path", ""),
        ),
    )
    await conn.commit()
    return cursor.lastrowid


async def get_events(
    cam_id: str | None = None,
    scenario: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    query = "SELECT * FROM events WHERE 1=1"
    params: list = []
    if cam_id:
        query += " AND cam_id = ?"
        params.append(cam_id)
    if scenario:
        query += " AND scenario = ?"
        params.append(scenario)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(query, params)
    rows = await cursor.fetchall()
    conn.row_factory = None
    return [dict(row) for row in rows]


async def get_event_count() -> int:
    conn = await _get_conn()
    cursor = await conn.execute("SELECT COUNT(*) FROM events")
    row = await cursor.fetchone()
    return row[0] if row else 0


async def get_stats() -> dict:
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT scenario, COUNT(*) as cnt FROM events GROUP BY scenario"
    )
    rows = await cursor.fetchall()
    return {row[0]: row[1] for row in rows}


# ── Clip Reviews ───────────────────────────────────────────────────
async def insert_clip_review(filename: str, total_frames: int, result: dict) -> int:
    conn = await _get_conn()
    cursor = await conn.execute(
        "INSERT INTO clip_reviews (filename, total_frames, result_json) VALUES (?, ?, ?)",
        (filename, total_frames, json.dumps(result, ensure_ascii=False)),
    )
    await conn.commit()
    return cursor.lastrowid


async def get_clip_reviews(limit: int = 50) -> list[dict]:
    conn = await _get_conn()
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(
        "SELECT * FROM clip_reviews ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = await cursor.fetchall()
    conn.row_factory = None
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["result"] = json.loads(d.pop("result_json"))
        except (json.JSONDecodeError, TypeError):
            d["result"] = {}
        results.append(d)
    return results


# ── Data Retention ─────────────────────────────────────────────────
async def cleanup_old_data() -> int:
    """Delete events + clip/thumbnail files older than LOCAL_RETENTION_DAYS."""
    import shutil
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=config.LOCAL_RETENTION_DAYS)
    cutoff_str = cutoff.isoformat()

    conn = await _get_conn()
    cursor = await conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_str,))
    deleted = cursor.rowcount
    await conn.commit()

    # Delete old date folders
    for subdir in ("clips", "thumbnails"):
        base = config.DATA_DIR / subdir
        if not base.exists():
            continue
        for day_dir in sorted(base.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                dir_date = datetime.strptime(day_dir.name, "%Y%m%d").replace(tzinfo=timezone.utc)
                if dir_date < cutoff:
                    shutil.rmtree(day_dir, ignore_errors=True)
                    logger.info("Cleaned up old data: %s", day_dir)
            except ValueError:
                pass

    return deleted
