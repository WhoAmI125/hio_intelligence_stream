"""
Local Storage — Event and clip management for the model server.

Stores detection events and video clips locally before daily flush to DB server.
Supports save, query, and delete operations.
Also processes videos with FFmpeg (H.264) and optionally uploads to AWS S3.

Directory layout:
    data/
    ├── events/
    │   └── YYYYMMDD/
    │       ├── ev_1234567890_1.json
    │       └── ev_1234567890_2.json
    ├── clips/
    │   └── YYYYMMDD/
    │       ├── ev_1234567890_1.mp4
    │       └── ev_1234567890_2.mp4
    └── thumbnails/
        └── YYYYMMDD/
            ├── ev_1234567890_1.jpg
            └── ev_1234567890_2.jpg
"""

import json
import logging
import os
import shutil
import subprocess
import cv2
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from model_server import config

logger = logging.getLogger(__name__)

def _upload_to_s3(local_path: str, s3_key: str) -> Optional[str]:
    """Uploads a file to S3 and returns the URL if USE_S3 is enabled."""
    if not getattr(config, "USE_S3", False):
        return None
    if not BOTO3_AVAILABLE:
        logger.warning("[LocalStorage] USE_S3=true but boto3 is not installed.")
        return None
    if not os.path.exists(local_path):
        logger.warning(f"[LocalStorage] S3 upload skipped, file not found: {local_path}")
        return None

    try:
        region = getattr(config, "AWS_REGION", "ap-northeast-2")
        access_key = getattr(config, "AWS_ACCESS_KEY_ID", "")
        secret_key = getattr(config, "AWS_SECRET_ACCESS_KEY", "")
        bucket_name = getattr(config, "AWS_STORAGE_BUCKET_NAME", "")
        if not bucket_name:
            logger.error("[LocalStorage] AWS_STORAGE_BUCKET_NAME not set")
            return None

        client_kwargs = {"region_name": region}
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
        s3_client = boto3.client("s3", **client_kwargs)

        content_type = "application/octet-stream"
        if local_path.endswith(".mp4"):
            content_type = "video/mp4"
        elif local_path.endswith(".jpg"):
            content_type = "image/jpeg"
        elif local_path.endswith(".json"):
            content_type = "application/json"

        s3_client.upload_file(
            local_path,
            bucket_name,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
        logger.info(f"[LocalStorage] Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.error(f"[LocalStorage] S3 upload failed for {local_path}: {e}")
    return None


class LocalStorage:
    """
    Local event/clip storage for the model server.

    Events are stored as JSON files; clips as MP4 files.
    Both are organized by date (YYYYMMDD) for easy daily flush.
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.events_dir = self.base_dir / "events"
        self.clips_dir = self.base_dir / "clips"
        self.thumbnails_dir = self.base_dir / "thumbnails"  # NEW: event preview JPEGs
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def save_event(self, event_id: str, event_data: Dict[str, Any]) -> str:
        """
        Save an event as a JSON file.

        Returns:
            Path to the saved file.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        day_dir = self.events_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        filepath = day_dir / f"{event_id}.json"
        event_data['event_id'] = event_id
        event_data['saved_at'] = datetime.now().isoformat()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2, default=str)

        logger.debug(f"[LocalStorage] Saved event: {filepath}")
        
        # Optionally upload to S3
        s3_url = _upload_to_s3(str(filepath), f"events/{date_str}/{event_id}.json")
        if s3_url:
            return s3_url
            
        return str(filepath)

    def get_event(self, event_id: str, date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load an event by ID. Searches today if date_str is not given."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        filepath = self.events_dir / date_str / f"{event_id}.json"
        if not filepath.exists():
            # Search all dates
            for day_dir in sorted(self.events_dir.iterdir(), reverse=True):
                candidate = day_dir / f"{event_id}.json"
                if candidate.exists():
                    filepath = candidate
                    break
            else:
                return None

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_events(
        self,
        date_str: Optional[str] = None,
        scenario: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List events, optionally filtered by date and scenario."""
        if date_str:
            dirs = [self.events_dir / date_str]
        else:
            dirs = sorted(self.events_dir.iterdir(), reverse=True)

        events = []
        for day_dir in dirs:
            if not day_dir.is_dir():
                continue
            for fp in sorted(day_dir.glob("*.json"), reverse=True):
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        ev = json.load(f)
                    if scenario and ev.get('scenario') != scenario:
                        continue
                    events.append(ev)
                    if len(events) >= limit:
                        return events
                except Exception:
                    continue
        return events

    def delete_event(self, event_id: str) -> bool:
        """Delete an event JSON and its associated clip."""
        deleted = False
        for day_dir in self.events_dir.iterdir():
            if not day_dir.is_dir():
                continue
            filepath = day_dir / f"{event_id}.json"
            if filepath.exists():
                filepath.unlink()
                deleted = True
                break

        # Also delete clip if exists
        for day_dir in self.clips_dir.iterdir():
            if not day_dir.is_dir():
                continue
            for ext in ('.mp4', '.avi', '.mkv'):
                clip_path = day_dir / f"{event_id}{ext}"
                if clip_path.exists():
                    clip_path.unlink()
                    break

        return deleted

    # ------------------------------------------------------------------
    # Clips
    # ------------------------------------------------------------------

    def save_clip(
        self,
        event_id: str,
        frames: List[Any],
        fps: float = 15.0,
        codec: str = "mp4v",
        allow_s3: bool = True,
    ) -> Optional[str]:
        """
        Save a list of OpenCV frames as an MP4 clip.
        Converts to H.264 using FFmpeg for web compatibility.
        Uploads to S3 if configured.

        Args:
            event_id: Event identifier (used as filename)
            frames: List of BGR np.ndarray frames
            fps: Output video FPS
            codec: FourCC codec string for OpenCV fallback path

        Returns:
            URL/Path to saved clip, or None on failure.
        """
        if not frames:
            return None

        date_str = datetime.now().strftime("%Y%m%d")
        day_dir = self.clips_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        final_path = day_dir / f"{event_id}.mp4"
        temp_path = day_dir / f"{event_id}_temp.avi"
        h, w = frames[0].shape[:2]

        try:
            ffmpeg_ok = False
            # Step 1: Write temporary AVI using MJPG (fast, reliable)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))
            for frame in frames:
                writer.write(frame)
            writer.release()

            # Step 2: Convert to H.264 using FFmpeg
            ffmpeg_path = getattr(config, "FFMPEG_PATH", "ffmpeg")
            try:
                result = subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        str(temp_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-r",
                        str(int(fps)),
                        str(final_path),
                    ],
                    capture_output=True,
                    timeout=180,
                )
                if result.returncode == 0 and final_path.exists():
                    ffmpeg_ok = True
                    logger.info(
                        f"[LocalStorage] FFmpeg converted clip: {final_path} "
                        f"({len(frames)} frames, {len(frames)/fps:.1f}s)"
                    )
                else:
                    stderr_txt = (result.stderr or b"").decode(errors="ignore")
                    logger.warning(
                        f"[LocalStorage] FFmpeg failed (code={result.returncode}). "
                        "Falling back to OpenCV mp4v. "
                        f"stderr={stderr_txt[:300]}"
                    )
            except Exception as e:
                logger.warning(
                    "[LocalStorage] FFmpeg unavailable/failed; "
                    f"falling back to OpenCV mp4v: {e}"
                )
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError as cleanup_err:
                        logger.debug(f"[LocalStorage] Temp clip cleanup skipped: {cleanup_err}")

            # Step 3: Fallback when FFmpeg path fails
            if not ffmpeg_ok:
                try:
                    fallback_fourcc = cv2.VideoWriter_fourcc(*codec)
                    fallback = cv2.VideoWriter(str(final_path), fallback_fourcc, fps, (w, h))
                    for frame in frames:
                        fallback.write(frame)
                    fallback.release()
                except Exception as fallback_err:
                    logger.error(f"[LocalStorage] OpenCV fallback save failed: {fallback_err}")
                    return None
                if not final_path.exists():
                    logger.error("[LocalStorage] OpenCV fallback did not create output file.")
                    return None
                logger.info(
                    f"[LocalStorage] Fallback clip saved: {final_path} "
                    f"({len(frames)} frames, {len(frames)/fps:.1f}s)"
                )

            # Step 4: Optional S3 upload
            if allow_s3:
                s3_url = _upload_to_s3(str(final_path), f"clips/{date_str}/{event_id}.mp4")
                if s3_url:
                    return s3_url

            return str(final_path)

        except Exception as e:
            logger.error(f"[LocalStorage] Failed to save clip: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            return None

    def get_clip_path(self, event_id: str) -> Optional[str]:
        """Find the clip file path for a given event ID."""
        for day_dir in sorted(self.clips_dir.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            for ext in ('.mp4', '.avi', '.mkv'):
                clip_path = day_dir / f"{event_id}{ext}"
                if clip_path.exists():
                    return str(clip_path)
        return None

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------

    def save_thumbnail(
        self, event_id: str, frame, quality: int = 85
    ) -> Optional[str]:
        """
        Save the last frame as a JPEG thumbnail for event list preview.

        Args:
            event_id: Event identifier (used as filename)
            frame: BGR np.ndarray (last frame of the clip)
            quality: JPEG quality (0-100)

        Returns:
            Path to saved thumbnail, or None on failure.
        """
        if frame is None:
            return None
        date_str = datetime.now().strftime("%Y%m%d")
        day_dir = self.thumbnails_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        filepath = day_dir / f"{event_id}.jpg"
        try:
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            logger.debug(f"[LocalStorage] Saved thumbnail: {filepath}")
            
            s3_url = _upload_to_s3(str(filepath), f"thumbnails/{date_str}/{event_id}.jpg")
            if s3_url:
                return s3_url
                
            return str(filepath)
        except Exception as e:
            logger.error(f"[LocalStorage] Thumbnail save failed: {e}")
            return None

    def get_thumbnail_path(self, event_id: str) -> Optional[str]:
        """Find the thumbnail file path for a given event ID."""
        for day_dir in sorted(self.thumbnails_dir.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            thumb_path = day_dir / f"{event_id}.jpg"
            if thumb_path.exists():
                return str(thumb_path)
        return None

    # ------------------------------------------------------------------
    # Flush helpers (used by FlushWorker)
    # ------------------------------------------------------------------

    def get_pending_dates(self) -> List[str]:
        """Get list of date directories that have events pending flush."""
        today = datetime.now().strftime("%Y%m%d")
        dates = []
        for day_dir in sorted(self.events_dir.iterdir()):
            if day_dir.is_dir() and day_dir.name < today:  # Don't flush today
                dates.append(day_dir.name)
        return dates

    def get_events_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Get all events for a specific date."""
        return self.list_events(date_str=date_str, limit=10000)

    def get_clips_for_date(self, date_str: str) -> List[str]:
        """Get all clip file paths for a specific date."""
        day_dir = self.clips_dir / date_str
        if not day_dir.exists():
            return []
        return [str(p) for p in day_dir.glob("*.*") if p.is_file()]

    def archive_date(self, date_str: str) -> None:
        """Remove event/clip files for a date after successful flush."""
        for base in (self.events_dir, self.clips_dir):
            day_dir = base / date_str
            if day_dir.exists():
                shutil.rmtree(day_dir, ignore_errors=True)
                logger.info(f"[LocalStorage] Archived {day_dir}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        event_count = sum(
            1 for d in self.events_dir.iterdir() if d.is_dir()
            for _ in d.glob("*.json")
        )
        clip_count = sum(
            1 for d in self.clips_dir.iterdir() if d.is_dir()
            for _ in d.glob("*.*")
        )
        thumbnail_count = sum(
            1 for d in self.thumbnails_dir.iterdir() if d.is_dir()
            for _ in d.glob("*.jpg")
        )
        return {
            "event_count": event_count,
            "clip_count": clip_count,
            "thumbnail_count": thumbnail_count,
            "pending_dates": self.get_pending_dates(),
        }
