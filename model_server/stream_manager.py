"""
Stream Manager — Multi-camera RTSP reader with ring buffer and frame sampling.

Ported from AdhocRTSPVLMWorker RTSP handling:
- Dedicated reader thread per camera
- Ring buffer (collections.deque) with monotonic timestamps
- Adaptive sampling: base 1.5fps / burst 4fps
- Reconnection with exponential backoff + jitter
- Frame extraction by time window for clip creation

Thread model:
    StreamManager
       └─ per camera_id:
              ├─ reader_thread  →  reads RTSP and pushes to ring buffer
              └─ ring buffer    →  deque[(frame, mono_ts), ...]
"""

import cv2
import logging
import random
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _rtsp_key(rtsp_url: str) -> str:
    """
    Canonical key for RTSP equality checks.
    """
    raw = str(rtsp_url or "").strip()
    if not raw:
        return ""
    try:
        p = urlparse(raw)
        scheme = (p.scheme or "").lower()
        host = (p.hostname or "").lower()
        port = f":{p.port}" if p.port else ""
        auth = ""
        if p.username:
            auth = p.username
            if p.password is not None:
                auth += f":{p.password}"
            auth += "@"
        path = p.path or ""
        query = f"?{p.query}" if p.query else ""
        frag = f"#{p.fragment}" if p.fragment else ""
        return f"{scheme}://{auth}{host}{port}{path}{query}{frag}"
    except Exception:
        return raw


class CameraStream:
    """
    Manages a single RTSP camera stream:
    - cv2.VideoCapture lifecycle
    - Ring buffer of (frame, mono_ts) tuples
    - Reconnection with exponential backoff
    - Frame sampling logic (base / burst modes)
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        *,
        base_fps: float = 1.5,
        burst_fps: float = 4.0,
        burst_duration_sec: float = 3.0,
        buffer_seconds: int = 30,
        rtsp_transport: str = "tcp",
        open_timeout_ms: int = 8000,
        read_timeout_ms: int = 8000,
        stale_threshold_sec: float = 2.5,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.base_fps = base_fps
        self.burst_fps = burst_fps
        self.burst_duration_sec = burst_duration_sec
        self.rtsp_transport = rtsp_transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.stale_threshold_sec = stale_threshold_sec

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

        # Status
        self.status = "stopped"
        self.last_error: Optional[str] = None
        self.stream_fps = 30.0
        self.current_fps = 0.0

        # Ring buffer — stores dicts: {"frame": np.ndarray, "mono_ts": float}
        est_buffer_len = max(60, int(buffer_seconds * 15))  # ~15fps effective
        self._ring: deque = deque(maxlen=est_buffer_len)
        self._ring_lock = threading.Lock()

        # Sampling control
        self._frame_count = 0
        self._sample_interval = 20  # recalculated after connect
        self._burst_interval = 5
        self._burst_duration_frames = 90
        self._in_burst = False
        self._burst_start_frame = 0

        # Current frame for UI/overlay
        self._current_frame = None
        self._current_frame_lock = threading.Lock()

        # Reconnect backoff
        self._reconnect_attempt = 0

        # Stats
        self._frames_read = 0
        self._frames_sampled = 0
        self._reconnect_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if self._running:
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"stream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[CameraStream:{self.camera_id}] Started reader thread.")
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        self._running = False
        self.status = "stopping"

        if self._thread and self._thread.is_alive():
            # read_timeout_ms can block reader thread; wait slightly longer than caller timeout.
            join_timeout = max(float(timeout), (self.read_timeout_ms / 1000.0) + 1.0)
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"[CameraStream:{self.camera_id}] Stop timeout. Reader thread still alive."
                )
                self.last_error = "Reader thread still alive after stop timeout."
                return False

        # Safe release after reader loop exits.
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        self.status = "stopped"
        logger.info(
            f"[CameraStream:{self.camera_id}] Stopped. "
            f"Frames read={self._frames_read}, sampled={self._frames_sampled}, "
            f"reconnects={self._reconnect_count}"
        )
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_frame(self) -> Optional[Any]:
        """Returns the latest raw frame (thread-safe copy)."""
        with self._current_frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
            return None

    def get_buffer_frames(
        self,
        window_sec: float = 10.0,
        anchor_mono_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from the ring buffer within a time window.

        Args:
            window_sec: Duration of the window in seconds
            anchor_mono_ts: Center timestamp (uses latest if None)

        Returns:
            List of dicts with 'frame' and 'mono_ts' keys
        """
        with self._ring_lock:
            if not self._ring:
                return []

            if anchor_mono_ts is None:
                anchor_mono_ts = self._ring[-1]["mono_ts"]

            half = window_sec / 2.0
            w_start = anchor_mono_ts - half
            w_end = anchor_mono_ts + half

            windowed = [
                e for e in self._ring
                if w_start <= e["mono_ts"] <= w_end
            ]

            # Fallback: rear window if symmetric window has too few frames
            if len(windowed) < 8:
                w_start = anchor_mono_ts - window_sec
                w_end = anchor_mono_ts + 0.5
                windowed = [
                    e for e in self._ring
                    if w_start <= e["mono_ts"] <= w_end
                ]

            windowed.sort(key=lambda e: e["mono_ts"])
            return windowed

    def trigger_burst(self) -> None:
        """Switch to burst sampling mode (higher FPS for a short duration)."""
        self._in_burst = True
        self._burst_start_frame = self._frame_count

    def is_active(self) -> bool:
        """
        Best-effort active-state check for duplicate RTSP protection.
        """
        return bool(self._running) or self.status in {
            "running", "connecting", "reconnecting", "stopping"
        }

    # ------------------------------------------------------------------
    # Reader loop (runs in dedicated thread)
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        self.status = "connecting"

        cap = self._try_connect()
        if cap is None:
            self.status = "error"
            self.last_error = "Cannot open RTSP stream after retries"
            self._running = False
            return

        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.stream_fps = float(fps)
        self._update_sampling_params(self.stream_fps)
        self.status = "running"

        last_success = time.time()
        consecutive_fail = 0
        fps_counter = 0
        fps_start = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_fail += 1
                if time.time() - last_success > 15 or consecutive_fail > 40:
                    # Reconnect
                    self.status = "reconnecting"
                    self.last_error = "Stream read failures, reconnecting..."
                    self._reconnect_count += 1
                    self._flush_ring()
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(self._next_backoff())
                    cap = self._try_connect()
                    if cap is None:
                        self.status = "error"
                        self.last_error = "Reconnection failed"
                        self._running = False
                        return
                    self._cap = cap
                    self._reset_backoff()
                    consecutive_fail = 0
                    last_success = time.time()
                    fps = cap.get(cv2.CAP_PROP_FPS) or self.stream_fps
                    self.stream_fps = float(fps)
                    self._update_sampling_params(self.stream_fps)
                    self.status = "running"
                continue

            consecutive_fail = 0
            last_success = time.time()
            self._frame_count += 1
            self._frames_read += 1

            # FPS tracking
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            # Update current frame for UI
            with self._current_frame_lock:
                self._current_frame = frame

            # Sampling decision
            if self._should_sample():
                mono_ts = time.monotonic()
                with self._ring_lock:
                    self._ring.append({"frame": frame, "mono_ts": mono_ts})
                self._frames_sampled += 1

            # Burst mode auto-exit
            if self._in_burst:
                if (self._frame_count - self._burst_start_frame) >= self._burst_duration_frames:
                    self._in_burst = False

        # Cleanup
        try:
            cap.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _should_sample(self) -> bool:
        """Decide whether current frame should be sampled into the ring buffer."""
        interval = self._burst_interval if self._in_burst else self._sample_interval
        return (self._frame_count % interval) == 0

    def _update_sampling_params(self, stream_fps: float) -> None:
        """Recalculate sampling intervals based on actual stream FPS."""
        sfps = max(1.0, stream_fps)
        self._sample_interval = max(1, int(round(sfps / self.base_fps)))
        self._burst_interval = max(1, int(round(sfps / self.burst_fps)))
        self._burst_duration_frames = max(1, int(round(sfps * self.burst_duration_sec)))

        # Resize ring buffer to hold at least 24 seconds of effective frames
        effective_fps = sfps / self._sample_interval
        new_maxlen = max(60, int(round(effective_fps * 30)))  # 30 seconds
        if new_maxlen != self._ring.maxlen:
            with self._ring_lock:
                old_data = list(self._ring)
                self._ring = deque(old_data, maxlen=new_maxlen)

    # ------------------------------------------------------------------
    # Connection / Reconnection
    # ------------------------------------------------------------------

    def _try_connect(self, max_retries: int = 5) -> Optional[cv2.VideoCapture]:
        """Try to connect to RTSP with retries and backoff."""
        for attempt in range(max_retries):
            try:
                cap = self._create_capture()
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self._reset_backoff()
                        logger.info(
                            f"[CameraStream:{self.camera_id}] "
                            f"Connected (attempt {attempt + 1})"
                        )
                        return cap
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception as e:
                self.last_error = f"RTSP connect error: {e}"
            time.sleep(self._next_backoff())

        return None

    def _create_capture(self) -> cv2.VideoCapture:
        """Create an OpenCV VideoCapture with RTSP options."""
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)

        env_flags = {
            "OPENCV_FFMPEG_CAPTURE_OPTIONS": (
                f"rtsp_transport;{self.rtsp_transport}"
                "|fflags;nobuffer"
                "|flags;low_delay"
                "|threads;1"
                "|analyzeduration;500000"
                "|probesize;500000"
            ),
        }
        import os
        for k, v in env_flags.items():
            os.environ[k] = v

        cap.open(self.rtsp_url, cv2.CAP_FFMPEG)
        return cap

    def _next_backoff(self) -> float:
        """Exponential backoff: 1s → 2s → 4s → 8s(max) + jitter."""
        base = min(8.0, float(2 ** min(self._reconnect_attempt, 3)))
        self._reconnect_attempt = min(self._reconnect_attempt + 1, 10)
        return base + random.uniform(0.0, 0.3)

    def _reset_backoff(self) -> None:
        self._reconnect_attempt = 0

    def _flush_ring(self) -> None:
        with self._ring_lock:
            self._ring.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._ring_lock:
            ring_len = len(self._ring)
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "stream_fps": self.stream_fps,
            "current_fps": round(self.current_fps, 1),
            "frames_read": self._frames_read,
            "frames_sampled": self._frames_sampled,
            "ring_buffer_size": ring_len,
            "reconnect_count": self._reconnect_count,
            "in_burst": self._in_burst,
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# StreamManager — orchestrates multiple CameraStream instances
# ---------------------------------------------------------------------------

class StreamManager:
    """
    Multi-camera stream manager.

    Usage:
        mgr = StreamManager()
        mgr.add_camera("cam1", "rtsp://...")
        mgr.add_camera("cam2", "rtsp://...")
        mgr.start_all()

        frame = mgr.get_frame("cam1")
        frames = mgr.get_clip_frames("cam1", window_sec=10)

        mgr.stop_all()
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self._streams: Dict[str, CameraStream] = {}
        self._lock = threading.Lock()
        self._default_config = default_config or {}

    def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        **kwargs,
    ) -> CameraStream:
        """Register a camera stream. Kwargs override default_config."""
        config = {**self._default_config, **kwargs}
        stream = CameraStream(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            base_fps=config.get("base_fps", 1.5),
            burst_fps=config.get("burst_fps", 4.0),
            burst_duration_sec=config.get("burst_duration_sec", 3.0),
            buffer_seconds=config.get("buffer_seconds", 30),
            rtsp_transport=config.get("rtsp_transport", "tcp"),
            open_timeout_ms=config.get("open_timeout_ms", 8000),
            read_timeout_ms=config.get("read_timeout_ms", 8000),
            stale_threshold_sec=config.get("stale_threshold_sec", 2.5),
        )
        with self._lock:
            # Block duplicate RTSP active on another camera.
            conflict_cam = None
            target_key = _rtsp_key(rtsp_url)
            for cam_id, s in self._streams.items():
                if cam_id == camera_id:
                    continue
                if not s.is_active():
                    continue
                if _rtsp_key(s.rtsp_url) == target_key:
                    conflict_cam = cam_id
                    break
            if conflict_cam:
                raise RuntimeError(
                    f"RTSP already active on camera '{conflict_cam}'. "
                    "Stop that camera first."
                )

            # Stop existing if replacing
            if camera_id in self._streams:
                old_stream = self._streams[camera_id]
                stopped = old_stream.stop()
                if not stopped:
                    raise RuntimeError(
                        f"Camera '{camera_id}' is still stopping. Retry in a moment."
                    )
            self._streams[camera_id] = stream
        return stream

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            stream = self._streams.pop(camera_id, None)
        if stream:
            return stream.stop()
        return False

    def start_all(self) -> None:
        with self._lock:
            for stream in self._streams.values():
                stream.start()

    def stop_all(self) -> None:
        with self._lock:
            streams = list(self._streams.values())
        for stream in streams:
            stream.stop()

    def start_camera(self, camera_id: str) -> bool:
        stream = self._get(camera_id)
        return stream.start() if stream else False

    def stop_camera(self, camera_id: str) -> bool:
        stream = self._get(camera_id)
        if stream:
            return stream.stop()
        return False

    def get_frame(self, camera_id: str) -> Optional[Any]:
        stream = self._get(camera_id)
        return stream.get_current_frame() if stream else None

    def get_clip_frames(
        self,
        camera_id: str,
        window_sec: float = 10.0,
        anchor_mono_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        stream = self._get(camera_id)
        return stream.get_buffer_frames(window_sec, anchor_mono_ts) if stream else []

    def trigger_burst(self, camera_id: str) -> None:
        stream = self._get(camera_id)
        if stream:
            stream.trigger_burst()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {cid: s.get_stats() for cid, s in self._streams.items()}

    def _get(self, camera_id: str) -> Optional[CameraStream]:
        with self._lock:
            return self._streams.get(camera_id)

    def find_camera_by_rtsp(
        self,
        rtsp_url: str,
        *,
        exclude_camera_id: Optional[str] = None,
        active_only: bool = True,
    ) -> Optional[str]:
        """
        Find a camera_id using the same RTSP URL.
        """
        target = _rtsp_key(rtsp_url)
        if not target:
            return None
        with self._lock:
            for cam_id, stream in self._streams.items():
                if exclude_camera_id and cam_id == exclude_camera_id:
                    continue
                if active_only and not stream.is_active():
                    continue
                if _rtsp_key(stream.rtsp_url) == target:
                    return cam_id
        return None
