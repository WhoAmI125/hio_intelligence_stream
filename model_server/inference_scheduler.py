import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("model_server.inference_scheduler")


@dataclass
class InferenceJob:
    camera_id: str
    run_id: int
    frame: Any
    enqueued_at: float


class InferenceScheduler:
    """
    Shared inference scheduler for all active cameras.

    Each camera can have at most one pending/in-flight job so we always prefer
    the latest frame over building backlog.
    """

    def __init__(
        self,
        *,
        stream_manager: Any,
        get_state: Callable[[str], Dict[str, Any]],
        process_fn: Callable[[str, Any, Dict[str, Any], float], None],
        workers: int = 1,
        dispatcher_sleep_sec: float = 0.02,
        queue_size: int = 128,
        active_burst_sec: float = 3.0,
        active_burst_fps: float = 3.0,
    ) -> None:
        self.stream_manager = stream_manager
        self.get_state = get_state
        self.process_fn = process_fn
        self.workers = max(1, int(workers))
        self.dispatcher_sleep_sec = max(0.005, float(dispatcher_sleep_sec))
        self.active_burst_sec = max(0.5, float(active_burst_sec))
        self.active_burst_fps = max(0.5, float(active_burst_fps))
        self._queue: queue.Queue[InferenceJob] = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop_event = threading.Event()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._worker_threads: list[threading.Thread] = []
        self._camera_ids: set[str] = set()
        self._lock = threading.RLock()
        self._camera_runtime: Dict[str, Dict[str, Any]] = {}

    def start(self) -> None:
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            return

        self._stop_event.clear()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="inference-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()

        self._worker_threads = []
        for idx in range(self.workers):
            th = threading.Thread(
                target=self._worker_loop,
                name=f"inference-worker-{idx}",
                daemon=True,
            )
            th.start()
            self._worker_threads.append(th)

        logger.info(
            "[InferenceScheduler] started workers=%d queue_size=%d",
            self.workers,
            self._queue.maxsize,
        )

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop_event.set()
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=timeout_sec)
        for th in self._worker_threads:
            if th.is_alive():
                th.join(timeout=timeout_sec)
        logger.info("[InferenceScheduler] stopped")

    def register_camera(self, camera_id: str) -> None:
        cam = str(camera_id)
        with self._lock:
            self._camera_ids.add(cam)
            self._camera_runtime.setdefault(
                cam,
                {
                    "registered": True,
                    "pending": False,
                    "inflight": False,
                    "last_submit_ts": 0.0,
                    "last_finish_ts": 0.0,
                    "last_active_ts": 0.0,
                    "jobs_enqueued": 0,
                    "jobs_completed": 0,
                    "jobs_dropped": 0,
                },
            )
            self._camera_runtime[cam]["registered"] = True

    def unregister_camera(self, camera_id: str) -> None:
        cam = str(camera_id)
        with self._lock:
            self._camera_ids.discard(cam)
            runtime = self._camera_runtime.setdefault(cam, {})
            runtime["registered"] = False
            runtime["pending"] = False
            runtime["inflight"] = False

    def mark_camera_active(self, camera_id: str) -> None:
        cam = str(camera_id)
        with self._lock:
            runtime = self._camera_runtime.setdefault(cam, {})
            runtime["last_active_ts"] = time.time()

    def get_metrics(self, camera_id: str) -> Dict[str, Any]:
        cam = str(camera_id)
        with self._lock:
            runtime = dict(self._camera_runtime.get(cam, {}))
            runtime["registered"] = cam in self._camera_ids
        runtime["queue_size"] = self._queue.qsize()
        runtime["worker_count"] = self.workers
        runtime["dispatcher_alive"] = bool(
            self._dispatcher_thread is not None and self._dispatcher_thread.is_alive()
        )
        runtime["workers_alive"] = sum(1 for th in self._worker_threads if th.is_alive())
        return runtime

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                camera_ids = list(self._camera_ids)

            for camera_id in camera_ids:
                state = self.get_state(camera_id)
                if not bool(state.get("running")):
                    continue

                runtime = self._camera_runtime.setdefault(camera_id, {})
                if runtime.get("pending") or runtime.get("inflight"):
                    continue

                base_fps = max(float(state.get("base_fps", 1.5) or 1.5), 0.5)
                target_fps = base_fps
                last_active_ts = float(runtime.get("last_active_ts", 0.0) or 0.0)
                if now - last_active_ts <= self.active_burst_sec:
                    target_fps = max(base_fps, self.active_burst_fps)

                interval = 1.0 / max(target_fps, 0.5)
                last_submit_ts = float(runtime.get("last_submit_ts", 0.0) or 0.0)
                if now - last_submit_ts < interval:
                    continue

                frame = self.stream_manager.get_frame(camera_id) if self.stream_manager else None
                if frame is None:
                    state["last_frame_age_sec"] = 999.0
                    continue

                job = InferenceJob(
                    camera_id=camera_id,
                    run_id=int(state.get("run_id", 0)),
                    frame=frame,
                    enqueued_at=now,
                )
                try:
                    self._queue.put_nowait(job)
                    runtime["pending"] = True
                    runtime["last_submit_ts"] = now
                    runtime["jobs_enqueued"] = int(runtime.get("jobs_enqueued", 0)) + 1
                except queue.Full:
                    runtime["jobs_dropped"] = int(runtime.get("jobs_dropped", 0)) + 1

            time.sleep(self.dispatcher_sleep_sec)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            state = self.get_state(job.camera_id)
            runtime = self._camera_runtime.setdefault(job.camera_id, {})
            try:
                if not bool(state.get("running")) or int(state.get("run_id", 0)) != job.run_id:
                    continue

                runtime["pending"] = False
                runtime["inflight"] = True
                started_at = time.time()
                self.process_fn(job.camera_id, job.frame, state, started_at)
                runtime["jobs_completed"] = int(runtime.get("jobs_completed", 0)) + 1
                runtime["last_finish_ts"] = time.time()
            except Exception:
                logger.exception("[InferenceScheduler] worker error camera=%s", job.camera_id)
            finally:
                runtime["pending"] = False
                runtime["inflight"] = False
                self._queue.task_done()
