"""
LoRA Data Collector — Auto-save frames + captions for Florence-2 fine-tuning.

Hooks into the inference loop to collect training data automatically:
- Every detected event frame + Florence-2 caption → saved as training pair
- Normal frames saved at a configurable ratio (negative samples)
- Human feedback (true/false positive) enriches labels

Data format (annotations.jsonl):
    {"image": "images/ev_xxx.jpg", "prefix": "<MORE_DETAILED_CAPTION>", "suffix": "A cashier ...", "scenario": "cash", "label": "detected", "feedback": null}

Usage:
    collector = DataCollector(base_dir="data/lora_training")
    collector.collect(frame, caption, scenario_results)
    collector.collect_feedback(event_id, decision, note)
"""

import json
import logging
import os
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and manages training data for Florence-2 LoRA fine-tuning.

    Stores:
    - images/          JPEG frames
    - annotations.jsonl   one JSON line per sample
    """

    def __init__(
        self,
        base_dir: str = "data/lora_training",
        normal_ratio: float = 0.05,
        max_samples: int = 50000,
        enabled: bool = True,
    ):
        """
        Args:
            base_dir: Root directory for training data.
            normal_ratio: Probability of saving a normal (no-detection) frame.
                          0.05 = save 5% of normal frames.
            max_samples: Maximum number of samples to store (auto-cleanup oldest).
            enabled: Whether data collection is active.
        """
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.annotations_path = self.base_dir / "annotations.jsonl"
        self.normal_ratio = normal_ratio
        self.max_samples = max_samples
        self.enabled = enabled

        self._lock = threading.Lock()
        self._sample_count = 0
        self._event_count = 0
        self._normal_count = 0
        self._feedback_count = 0

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Count existing samples
        if self.annotations_path.exists():
            with open(self.annotations_path, "r", encoding="utf-8") as f:
                self._sample_count = sum(1 for _ in f)
            logger.info(
                f"[DataCollector] Loaded {self._sample_count} existing samples "
                f"from {self.annotations_path}"
            )

    # ------------------------------------------------------------------
    # Core collection API
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_token(value: Any) -> str:
        """Build a filesystem-safe ASCII token."""
        text = str(value or "").strip()
        if not text:
            return "unknown"
        text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
        text = text.strip("_")
        return text[:80] or "unknown"

    def collect(
        self,
        frame: np.ndarray,
        caption: str,
        scenario_results: Dict[str, Dict[str, Any]],
        camera_id: str = "adhoc_cam",
    ) -> Optional[str]:
        """
        Collect a training sample from the inference loop.

        Called after every Florence-2 inference. Decides whether to save
        based on whether a detection occurred or random normal sampling.

        Args:
            frame: BGR image from OpenCV.
            caption: Florence-2 generated caption for this frame.
            scenario_results: Dict of scenario analysis results.
            camera_id: Camera identifier.

        Returns:
            Sample ID if saved, None if skipped.
        """
        if not self.enabled or frame is None or not caption:
            return None

        # Check if any scenario was detected
        has_detection = any(
            r.get("is_detected", False)
            for r in scenario_results.values()
        )

        # Find best detected scenario
        detected_scenario = None
        detected_confidence = 0.0
        detected_keywords: list = []

        if has_detection:
            for name, result in scenario_results.items():
                if result.get("is_detected") and result.get("confidence", 0) > detected_confidence:
                    detected_scenario = name
                    detected_confidence = result.get("confidence", 0)
                    detected_keywords = result.get("matched_keywords", [])

        # Decision: save or skip
        if has_detection:
            label = "detected"
            scenario = detected_scenario
        elif random.random() < self.normal_ratio:
            label = "normal"
            scenario = "none"
        else:
            return None  # Skip this frame

        # Generate sample ID
        ts = int(time.time() * 1000)
        sample_id = f"{label}_{scenario}_{ts}_{camera_id}"
        image_filename = f"{sample_id}.jpg"
        image_path = self.images_dir / image_filename

        try:
            # Save image
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Build annotation
            annotation = {
                "sample_id": sample_id,
                "image": f"images/{image_filename}",
                "prefix": "<MORE_DETAILED_CAPTION>",
                "suffix": caption,
                "scenario": scenario,
                "label": label,
                "confidence": round(detected_confidence, 4),
                "matched_keywords": detected_keywords,
                "feedback": None,
                "camera_id": camera_id,
                "collected_at": datetime.now().isoformat(),
            }

            # Append to JSONL
            with self._lock:
                with open(self.annotations_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                self._sample_count += 1
                if label == "detected":
                    self._event_count += 1
                else:
                    self._normal_count += 1

            logger.debug(
                f"[DataCollector] Saved {label} sample: {sample_id} "
                f"(total: {self._sample_count})"
            )

            # Auto-cleanup if over limit
            if self._sample_count > self.max_samples:
                self._cleanup_oldest()

            return sample_id

        except Exception as e:
            logger.error(f"[DataCollector] Failed to save sample: {e}")
            return None

    def collect_gemini_validated_clip(
        self,
        *,
        event_id: str,
        scenario: str,
        clip_frames: List[np.ndarray],
        caption: str,
        camera_id: str = "adhoc_cam",
        gemini_confidence: float = 1.0,
        matched_keywords: Optional[List[str]] = None,
        sample_count: int = 3,
    ) -> List[str]:
        """
        Save LoRA samples only from Gemini-approved clip evidence.

        This is intended for strict positive collection:
        - No Tier1-only positives
        - No random normal frames
        """
        if not self.enabled:
            return []
        if not caption:
            return []
        if not clip_frames:
            return []

        valid_frames = [
            f for f in clip_frames
            if f is not None and getattr(f, "size", 0) > 0
        ]
        if not valid_frames:
            return []

        count = max(1, min(int(sample_count), len(valid_frames)))
        if len(valid_frames) <= count:
            indices = list(range(len(valid_frames)))
        else:
            indices = np.linspace(0, len(valid_frames) - 1, num=count, dtype=np.int32).tolist()

        safe_camera = self._safe_token(camera_id)
        safe_event = self._safe_token(event_id)
        safe_scenario = self._safe_token(scenario)
        keys = matched_keywords or []
        ts_base = int(time.time() * 1000)
        saved_ids: List[str] = []

        for order, idx in enumerate(indices):
            frame = valid_frames[int(idx)]
            sample_id = f"detected_{safe_scenario}_gemini_{safe_event}_{ts_base}_{order}_{safe_camera}"
            image_filename = f"{sample_id}.jpg"
            image_path = self.images_dir / image_filename

            try:
                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                annotation = {
                    "sample_id": sample_id,
                    "image": f"images/{image_filename}",
                    "prefix": "<MORE_DETAILED_CAPTION>",
                    "suffix": caption,
                    "scenario": scenario,
                    "label": "detected",
                    "confidence": round(float(gemini_confidence or 0.0), 4),
                    "matched_keywords": keys,
                    "feedback": {
                        "source": "gemini_validated_clip",
                        "event_id": event_id,
                        "clip_frame_index": int(idx),
                    },
                    "camera_id": camera_id,
                    "collected_at": datetime.now().isoformat(),
                }

                with self._lock:
                    with open(self.annotations_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                    self._sample_count += 1
                    self._event_count += 1

                saved_ids.append(sample_id)
            except Exception as e:
                logger.error(f"[DataCollector] Gemini-validated sample save failed: {e}")

        if saved_ids:
            logger.info(
                f"[DataCollector] Saved {len(saved_ids)} Gemini-validated clip samples "
                f"(event={event_id}, scenario={scenario})"
            )

        if self._sample_count > self.max_samples:
            self._cleanup_oldest()

        return saved_ids

    def collect_feedback(
        self,
        event_id: str,
        decision: str,
        note: str = "",
        frame: Optional[np.ndarray] = None,
        caption: str = "",
        scenario: str = "",
    ) -> bool:
        """
        Enrich training data with human feedback.

        When a user marks a detection as true_positive or false_positive,
        save that information to improve training quality.

        Args:
            event_id: Event ID from the detection system.
            decision: 'accept' (true positive) or 'decline' (false positive).
            note: Optional note from the human reviewer.
            frame: Optional frame image (if available).
            caption: Optional caption for new frame-based sample.
            scenario: Scenario name.

        Returns:
            True if feedback was recorded.
        """
        if not self.enabled:
            return False

        try:
            feedback_label = "true_positive" if decision == "accept" else "false_positive"

            # If we have a frame, save as a new high-quality sample
            if frame is not None and caption:
                ts = int(time.time() * 1000)
                sample_id = f"feedback_{feedback_label}_{scenario}_{ts}"
                image_filename = f"{sample_id}.jpg"
                image_path = self.images_dir / image_filename

                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                annotation = {
                    "sample_id": sample_id,
                    "image": f"images/{image_filename}",
                    "prefix": "<MORE_DETAILED_CAPTION>",
                    "suffix": caption,
                    "scenario": scenario,
                    "label": feedback_label,
                    "confidence": 1.0 if feedback_label == "true_positive" else 0.0,
                    "matched_keywords": [],
                    "feedback": {
                        "event_id": event_id,
                        "decision": decision,
                        "note": note,
                    },
                    "camera_id": "",
                    "collected_at": datetime.now().isoformat(),
                }

                with self._lock:
                    with open(self.annotations_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                    self._sample_count += 1
                    self._feedback_count += 1

            else:
                # No frame — just log the feedback for potential future use
                feedback_path = self.base_dir / "feedback_log.jsonl"
                entry = {
                    "event_id": event_id,
                    "decision": decision,
                    "note": note,
                    "scenario": scenario,
                    "at": datetime.now().isoformat(),
                }
                with self._lock:
                    with open(feedback_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    self._feedback_count += 1

            return True

        except Exception as e:
            logger.error(f"[DataCollector] Feedback save failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Stats & Management
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        # Count by label from disk
        label_counts: dict[str, int] = {}
        scenario_counts: dict[str, int] = {}

        if self.annotations_path.exists():
            try:
                with open(self.annotations_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            label = record.get("label", "unknown")
                            scenario = record.get("scenario", "unknown")
                            label_counts[label] = label_counts.get(label, 0) + 1
                            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

        # Disk usage
        disk_bytes = 0
        image_count = 0
        if self.images_dir.exists():
            for f in self.images_dir.iterdir():
                if f.is_file():
                    disk_bytes += f.stat().st_size
                    image_count += 1

        return {
            "enabled": self.enabled,
            "total_samples": self._sample_count,
            "event_samples": self._event_count,
            "normal_samples": self._normal_count,
            "feedback_samples": self._feedback_count,
            "by_label": label_counts,
            "by_scenario": scenario_counts,
            "image_count": image_count,
            "disk_usage_mb": round(disk_bytes / (1024 * 1024), 2),
            "normal_ratio": self.normal_ratio,
            "max_samples": self.max_samples,
            "data_dir": str(self.base_dir),
        }

    def toggle(self, enabled: Optional[bool] = None) -> bool:
        """Toggle or set data collection on/off."""
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = not self.enabled
        logger.info(f"[DataCollector] Collection {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

    def export_for_training(self) -> Dict[str, Any]:
        """
        Prepare data for training.

        Returns summary of available data and readiness status.
        """
        stats = self.get_stats()
        total = stats["total_samples"]

        # Minimum requirements
        min_samples = 50
        is_ready = total >= min_samples

        return {
            "ready": is_ready,
            "total_samples": total,
            "min_required": min_samples,
            "recommended": 200,
            "annotations_path": str(self.annotations_path),
            "images_dir": str(self.images_dir),
            "by_label": stats["by_label"],
            "by_scenario": stats["by_scenario"],
            "message": (
                f"Ready for training with {total} samples."
                if is_ready
                else f"Need at least {min_samples} samples "
                     f"(currently {total}). Keep the system running."
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_oldest(self, keep_ratio: float = 0.8):
        """Remove oldest samples when over max_samples limit."""
        target = int(self.max_samples * keep_ratio)

        try:
            # Read all annotations
            records = []
            with open(self.annotations_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if len(records) <= target:
                return

            # Sort by collected_at, keep newest
            records.sort(key=lambda r: r.get("collected_at", ""), reverse=True)
            to_remove = records[target:]
            to_keep = records[:target]

            # Delete old images
            for r in to_remove:
                img_path = self.base_dir / r.get("image", "")
                if img_path.exists():
                    img_path.unlink()

            # Rewrite annotations
            with self._lock:
                with open(self.annotations_path, "w", encoding="utf-8") as f:
                    for r in to_keep:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                self._sample_count = len(to_keep)

            logger.info(
                f"[DataCollector] Cleaned up {len(to_remove)} old samples. "
                f"Remaining: {len(to_keep)}"
            )

        except Exception as e:
            logger.error(f"[DataCollector] Cleanup failed: {e}")
