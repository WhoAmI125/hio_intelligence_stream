import logging
import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger("model_server.event_postprocessor")


@dataclass
class EventPostProcessJob:
    payload: dict[str, Any]


class EventPostProcessor:
    """Runs heavy event post-processing outside the main inference path."""

    def __init__(
        self,
        *,
        process_fn: Callable[[dict[str, Any]], None],
        workers: int = 1,
        queue_size: int = 128,
        dead_letter_dir: str | Path | None = None,
    ) -> None:
        self.process_fn = process_fn
        self.workers = max(1, int(workers))
        self._queue: queue.Queue[EventPostProcessJob] = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._keys_lock = threading.Lock()
        self._active_keys: set[str] = set()
        self._last_reject_reason = ""
        self._dropped_total = 0
        self.dead_letter_dir = Path(dead_letter_dir) if dead_letter_dir else None
        if self.dead_letter_dir is not None:
            self.dead_letter_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if any(th.is_alive() for th in self._threads):
            return
        self._stop_event.clear()
        self._threads = []
        for idx in range(self.workers):
            th = threading.Thread(
                target=self._worker_loop,
                name=f"event-postprocess-{idx}",
                daemon=True,
            )
            th.start()
            self._threads.append(th)
        logger.info(
            "[EventPostProcessor] started workers=%d queue_size=%d",
            self.workers,
            self._queue.maxsize,
        )

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop_event.set()
        for th in self._threads:
            if th.is_alive():
                th.join(timeout=timeout_sec)
        logger.info("[EventPostProcessor] stopped")

    def submit(self, payload: dict[str, Any]) -> bool:
        key = self._payload_key(payload)
        with self._keys_lock:
            if key and key in self._active_keys:
                self._last_reject_reason = "duplicate_pending"
                self._dropped_total += 1
                self._write_dead_letter(payload, self._last_reject_reason)
                return False
            if key:
                self._active_keys.add(key)
        try:
            self._queue.put_nowait(EventPostProcessJob(payload=payload))
            return True
        except queue.Full:
            with self._keys_lock:
                if key:
                    self._active_keys.discard(key)
            self._last_reject_reason = "queue_full"
            self._dropped_total += 1
            self._write_dead_letter(payload, self._last_reject_reason)
            return False

    def last_reject_reason(self) -> str:
        return self._last_reject_reason

    def get_metrics(self) -> dict[str, Any]:
        return {
            "queue_size": self._queue.qsize(),
            "worker_count": self.workers,
            "workers_alive": sum(1 for th in self._threads if th.is_alive()),
            "active_keys": len(self._active_keys),
            "dropped_total": self._dropped_total,
        }

    @staticmethod
    def _payload_key(payload: dict[str, Any]) -> str:
        return f"{payload.get('camera_id', '')}:{payload.get('scenario_name', '')}"

    @staticmethod
    def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in dict(payload or {}).items():
            if key in {"frame", "admission_clip_entries"}:
                if key == "admission_clip_entries" and isinstance(value, list):
                    out[key] = f"<{len(value)} frame entries>"
                else:
                    out[key] = "<omitted>"
            else:
                out[key] = value
        return out

    def _write_dead_letter(self, payload: dict[str, Any], reason: str) -> None:
        if self.dead_letter_dir is None:
            return
        try:
            path = self.dead_letter_dir / "events_dropped.jsonl"
            row = {
                "at": datetime.now(KST).isoformat(),
                "reason": reason,
                "key": self._payload_key(payload),
                "payload": self._sanitize_payload(payload),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("[EventPostProcessor] dead-letter write failed", exc_info=True)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process_fn(job.payload)
            except Exception:
                logger.exception("[EventPostProcessor] job error")
            finally:
                key = self._payload_key(job.payload)
                if key:
                    with self._keys_lock:
                        self._active_keys.discard(key)
                self._queue.task_done()
