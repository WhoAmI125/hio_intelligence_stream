"""Extract clips from stream reader's ring buffer."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import config
from stream.stream_reader import StreamReader

logger = logging.getLogger(__name__)


def extract_clip_frames(
    reader: StreamReader,
    trigger_time: float,
    pre_sec: int | None = None,
    post_sec: int | None = None,
    target_fps: float = 8.0,
) -> list[np.ndarray]:
    return reader.get_clip_frames(trigger_time, pre_sec, post_sec, target_fps)


def save_clip_mp4(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 8.0,
) -> str | None:
    """Save frames as browser-compatible H.264 MP4.

    Strategy: write temp AVI (MJPG) → ffmpeg to H.264.
    Fallback: cv2 mp4v if ffmpeg unavailable.
    """
    if not frames:
        return None

    tmp_path: str | None = None
    writer: cv2.VideoWriter | None = None

    try:
        h, w = frames[0].shape[:2]
        final_path = Path(output_path)

        # Step 1: write temp AVI with MJPG (reliable, fast)
        tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        tmp_path = tmp.name
        tmp.close()

        fourcc = cv2.VideoWriter.fourcc(*"MJPG")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        writer = None

        # Step 2: convert to H.264 via ffmpeg
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-r", str(fps),
                str(final_path),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and final_path.exists() and final_path.stat().st_size > 0:
                logger.info("Clip saved (H.264): %s (%d frames)", output_path, len(frames))
                return output_path
            else:
                logger.warning("ffmpeg failed (rc=%d), falling back to mp4v", result.returncode)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("ffmpeg unavailable or timed out (%s), falling back to mp4v", type(e).__name__)

        # Fallback: cv2 mp4v (not browser-compatible but at least saves)
        fourcc_mp4 = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc_mp4, fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        writer = None
        logger.info("Clip saved (mp4v fallback): %s (%d frames)", output_path, len(frames))
        return output_path

    except Exception as e:
        logger.error("Failed to save clip: %s", e, exc_info=True)
        return None
    finally:
        # Ensure writer is always released
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        # Always clean up temp AVI
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
