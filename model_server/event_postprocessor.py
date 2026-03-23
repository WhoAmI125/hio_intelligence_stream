import logging
import queue
import threading
from dataclasses import dataclass
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
    ) -> None:
        self.process_fn = process_fn
        self.workers = max(1, int(workers))
        self._queue: queue.Queue[EventPostProcessJob] = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

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
        try:
            self._queue.put_nowait(EventPostProcessJob(payload=payload))
            return True
        except queue.Full:
            return False

    def get_metrics(self) -> dict[str, Any]:
        return {
            "queue_size": self._queue.qsize(),
            "worker_count": self.workers,
            "workers_alive": sum(1 for th in self._threads if th.is_alive()),
        }

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
                self._queue.task_done()
