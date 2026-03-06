"""
Flush Worker — Daily data transfer from model server to DB server.

Runs as a background thread that periodically checks for pending
date directories in LocalStorage and flushes them to the DB server
via HTTP POST with retry logic.

Flow:
    LocalStorage (events/ + clips/)
        ↓  (once per day or on demand)
    FlushWorker.flush()
        ↓  POST /api/flush (multipart: metadata JSON + clip files)
    DB Server → stores in PostgreSQL + media archive
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FlushWorker:
    """
    Background worker that flushes local events/clips to the DB server.

    Retry strategy: exponential backoff (1s → 2s → 4s → max 60s)
    On success: archives the flushed date directory
    On failure: retries next cycle
    """

    def __init__(
        self,
        db_server_url: str = "http://localhost:8001",
        flush_endpoint: str = "/api/flush",
        flush_interval_sec: int = 3600,  # once per hour by default
        max_retries: int = 3,
        local_storage=None,
    ):
        self.db_server_url = db_server_url.rstrip("/")
        self.flush_endpoint = flush_endpoint
        self.flush_interval_sec = flush_interval_sec
        self.max_retries = max_retries

        # Lazy import to avoid circular dependencies
        if local_storage is not None:
            self.storage = local_storage
        else:
            from model_server.local_storage import LocalStorage
            self.storage = LocalStorage()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._total_flushed = 0
        self._total_failed = 0
        self._last_flush_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="flush-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[FlushWorker] Started. Interval={self.flush_interval_sec}s, "
            f"Target={self.db_server_url}{self.flush_endpoint}"
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(
            f"[FlushWorker] Stopped. "
            f"Flushed={self._total_flushed}, Failed={self._total_failed}"
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.flush()
            except Exception as e:
                logger.error(f"[FlushWorker] Flush cycle error: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(self.flush_interval_sec * 2):
                if not self._running:
                    break
                time.sleep(0.5)

    # ------------------------------------------------------------------
    # Flush logic
    # ------------------------------------------------------------------

    def flush(self) -> Dict[str, Any]:
        """
        Flush all pending date directories to the DB server.

        Returns summary dict with counts.
        """
        pending = self.storage.get_pending_dates()
        if not pending:
            logger.debug("[FlushWorker] No pending dates to flush.")
            return {"flushed": 0, "failed": 0}

        flushed = 0
        failed = 0

        for date_str in pending:
            success = self._flush_date(date_str)
            if success:
                self.storage.archive_date(date_str)
                flushed += 1
                self._total_flushed += 1
            else:
                failed += 1
                self._total_failed += 1

        from datetime import datetime
        self._last_flush_at = datetime.now().isoformat()

        logger.info(
            f"[FlushWorker] Flush complete. "
            f"Flushed={flushed}, Failed={failed}, Dates={pending}"
        )
        return {"flushed": flushed, "failed": failed, "dates": pending}

    def _flush_date(self, date_str: str) -> bool:
        """Flush a single date's events and clips to DB server with retries."""
        events = self.storage.get_events_for_date(date_str)
        clips = self.storage.get_clips_for_date(date_str)

        if not events:
            logger.debug(f"[FlushWorker] No events for {date_str}, skipping.")
            return True

        payload = {
            "date": date_str,
            "events": events,
            "clip_count": len(clips),
        }

        for attempt in range(self.max_retries):
            try:
                success = self._post_flush(payload, clips)
                if success:
                    logger.info(
                        f"[FlushWorker] Flushed {date_str}: "
                        f"{len(events)} events, {len(clips)} clips"
                    )
                    return True
            except Exception as e:
                logger.warning(
                    f"[FlushWorker] Flush {date_str} attempt {attempt + 1} failed: {e}"
                )

            # Exponential backoff
            backoff = min(60.0, 2 ** attempt)
            time.sleep(backoff)

        logger.error(
            f"[FlushWorker] Flush {date_str} failed after {self.max_retries} retries."
        )
        return False

    def _post_flush(self, payload: Dict[str, Any], clip_paths: List[str]) -> bool:
        """
        POST events metadata + clip files to DB server.

        Uses requests library for multipart upload.
        """
        import requests

        url = f"{self.db_server_url}{self.flush_endpoint}"

        # Build multipart form data
        files = []
        try:
            # Events metadata as JSON
            metadata_json = json.dumps(payload, ensure_ascii=False, default=str)

            files_dict = {"metadata": ("metadata.json", metadata_json, "application/json")}

            # Attach clip files
            open_files = []
            for clip_path in clip_paths:
                if os.path.exists(clip_path):
                    name = os.path.basename(clip_path)
                    fh = open(clip_path, "rb")
                    open_files.append(fh)
                    files_dict[f"clip_{name}"] = (name, fh, "video/mp4")

            response = requests.post(
                url,
                files=files_dict,
                timeout=120,
            )

            # Close file handles
            for fh in open_files:
                fh.close()

            if response.status_code == 200:
                return True
            else:
                logger.warning(
                    f"[FlushWorker] Server returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False

        except requests.ConnectionError as e:
            logger.error(f"[FlushWorker] Connection error: {e}")
            raise
        except requests.Timeout as e:
            logger.error(f"[FlushWorker] Timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"[FlushWorker] Unexpected error: {e}")
            raise

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "total_flushed": self._total_flushed,
            "total_failed": self._total_failed,
            "last_flush_at": self._last_flush_at,
            "pending_dates": self.storage.get_pending_dates(),
        }
