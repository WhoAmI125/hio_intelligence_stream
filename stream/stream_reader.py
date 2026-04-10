"""RTSP stream reader — cv2.VideoCapture based (clean H.264 decode).

Two frame outputs:
  - display_frame: updated every cap.read() (~stream FPS) for live UI
  - inference_frame: updated at SAMPLE_FPS (default 1 FPS) for Tier 1
                     switches to BURST_FPS (4 FPS) for 3s after event trigger

Ring buffer stores decoded frames for clip extraction.
"""

from __future__ import annotations

import collections
import logging
import random
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class BufferFrame:
    frame: np.ndarray
    time: float  # monotonic


class StreamReader:

    def __init__(self, cam_id: str, rtsp_url: str):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url

        # Ring buffer — decoded frames for clip extraction (~720p, 12fps, 60s = 720 frames ~2GB)
        self._ring: collections.deque[BufferFrame] = collections.deque(maxlen=720)

        # Display frame — always latest
        self._display_frame: np.ndarray | None = None
        self._display_time: float = 0.0
        self._display_lock = threading.Lock()

        # Inference frame — sampled at inference FPS
        self._inference_frame: np.ndarray | None = None
        self._inference_time: float = 0.0
        self._inference_lock = threading.Lock()
        self._last_consumed_time: float = 0.0  # track last frame consumed by inference loop
        self._ring_lock = threading.Lock()

        self.running = False
        self._thread: threading.Thread | None = None

        # Backoff
        self._backoff_attempts = 0
        self._backoff_base = 1.0
        self._backoff_max = 30.0

        # Stale detection
        self._stale_timeout = 2.5

        # Burst mode
        self._in_burst = False
        self._burst_start: float = 0.0
        self._burst_duration = 3.0  # seconds

        # Stats
        self._frame_count = 0
        self._stream_fps: float = 0.0

    @property
    def latest_frame(self):
        with self._display_lock:
            return self._display_frame

    @property
    def is_stale(self) -> bool:
        with self._display_lock:
            if self._display_time == 0.0:
                return False
            return (time.monotonic() - self._display_time) > self._stale_timeout

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._backoff_attempts = 0
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"stream-{self.cam_id}"
        )
        self._thread.start()
        logger.info("Stream started: %s", self.cam_id)

    def stop(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Stream stopped: %s", self.cam_id)

    def trigger_burst(self) -> None:
        """Switch to burst mode (higher inference FPS) for 3 seconds."""
        self._in_burst = True
        self._burst_start = time.monotonic()

    def _get_inference_interval(self) -> float:
        if self._in_burst:
            if time.monotonic() - self._burst_start > self._burst_duration:
                self._in_burst = False
                return 1.0 / config.SAMPLE_FPS
            return 1.0 / 4.0  # 4 FPS burst
        return 1.0 / config.SAMPLE_FPS

    def _read_loop(self) -> None:
        last_inference_time = 0.0
        # Ring buffer sampling: ~12 FPS for smooth clips
        ring_interval = 1.0 / 12.0
        last_ring_time = 0.0

        while self.running:
            cap = None
            try:
                # FFmpeg low-latency options (matching reference project)
                env = (
                    f"rtsp_transport;tcp|"
                    f"fflags;nobuffer|"
                    f"flags;low_delay|"
                    f"analyzeduration;500000|"
                    f"probesize;500000"
                )
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)

                if not cap.isOpened():
                    raise ConnectionError(f"Failed to open {self.rtsp_url}")

                self._stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._backoff_attempts = 0

                logger.info("[%s] Connected — %dx%d @ %.0f fps", self.cam_id, w, h, self._stream_fps)

                consecutive_failures = 0

                while self.running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures > 40:
                            logger.warning("[%s] 40+ consecutive read failures, reconnecting", self.cam_id)
                            break
                        continue

                    consecutive_failures = 0
                    now = time.monotonic()
                    self._frame_count += 1

                    # Resize to max 1280x720 to cap RAM + GPU usage
                    h, w = frame.shape[:2]
                    if w > 1280 or h > 720:
                        scale = min(1280 / w, 720 / h)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    # Always update display frame (every read)
                    with self._display_lock:
                        self._display_frame = frame
                        self._display_time = now

                    # Ring buffer at ~12 FPS
                    if now - last_ring_time >= ring_interval:
                        with self._ring_lock:
                            self._ring.append(BufferFrame(frame=frame, time=now))
                        last_ring_time = now

                    # Inference frame at SAMPLE_FPS (or burst FPS)
                    inf_interval = self._get_inference_interval()
                    if now - last_inference_time >= inf_interval:
                        with self._inference_lock:
                            self._inference_frame = frame
                            self._inference_time = now
                        last_inference_time = now

            except Exception as e:
                self._backoff_attempts += 1
                delay = min(self._backoff_base * (2 ** (self._backoff_attempts - 1)), self._backoff_max)
                jitter = random.uniform(0, 1.0)
                total = delay + jitter
                logger.warning("[%s] RTSP error (attempt %d), reconnecting in %.1fs: %s",
                               self.cam_id, self._backoff_attempts, total, e)
                time.sleep(total)
            finally:
                if cap is not None:
                    cap.release()

    def get_frame(self) -> tuple[np.ndarray | None, float]:
        """Latest display frame (updated every cap.read)."""
        with self._display_lock:
            if self._display_frame is not None:
                age = time.monotonic() - self._display_time
                if age > self._stale_timeout:
                    return None, 0.0
                return self._display_frame.copy(), self._display_time
            return None, 0.0

    def get_inference_frame(self) -> tuple[np.ndarray | None, float]:
        """Latest inference frame (1 FPS normal / 4 FPS burst).

        Returns (None, 0.0) if no new frame since last call.
        """
        with self._inference_lock:
            if self._inference_frame is not None and self._inference_time != self._last_consumed_time:
                self._last_consumed_time = self._inference_time
                return self._inference_frame.copy(), self._inference_time
            return None, 0.0

    def get_clip_frames(
        self,
        trigger_time: float,
        pre_sec: int | None = None,
        post_sec: int | None = None,
        target_fps: float = 8.0,
    ) -> list[np.ndarray]:
        """Extract clip frames from ring buffer around trigger time."""
        pre = pre_sec or config.CLIP_PRE_SECONDS
        post = post_sec or config.CLIP_POST_SECONDS

        # Wait for post-trigger footage
        remaining = (trigger_time + post) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        start = trigger_time - pre
        end = trigger_time + post

        with self._ring_lock:
            candidates = [bf for bf in self._ring if start <= bf.time <= end]

        if not candidates:
            return []

        # Downsample to target_fps
        clip_duration = pre + post
        target_count = int(clip_duration * target_fps)
        step = max(1, len(candidates) // target_count)
        return [candidates[i].frame for i in range(0, len(candidates), step)][:target_count]
