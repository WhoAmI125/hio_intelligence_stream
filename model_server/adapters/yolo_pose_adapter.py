"""
YOLO Pose Adapter — Ultralytics YOLO26n-pose wrapper.

Used as the cash-scenario gate. Keeps fire/violence untouched.

Output model:
    detect(frame) -> List[PersonDetection]

PersonDetection fields:
    bbox: (x1, y1, x2, y2)
    bbox_conf: float
    keypoints: np.ndarray shape (17, 2)   COCO-17 order
    kp_conf:   np.ndarray shape (17,)
    left_wrist / right_wrist accessors for kp[9] / kp[10]
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices for wrists
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8


@dataclass
class PersonDetection:
    bbox: Tuple[float, float, float, float]
    bbox_conf: float
    keypoints: np.ndarray  # (17, 2)
    kp_conf: np.ndarray    # (17,)

    def wrist_points(self, min_conf: float = 0.3) -> List[Tuple[float, float]]:
        """Return list of (x, y) for any wrist that clears the confidence threshold."""
        out = []
        for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
            if float(self.kp_conf[idx]) >= min_conf:
                out.append((float(self.keypoints[idx, 0]), float(self.keypoints[idx, 1])))
        return out

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "bbox_conf": self.bbox_conf,
            "left_wrist": [float(self.keypoints[KP_LEFT_WRIST, 0]), float(self.keypoints[KP_LEFT_WRIST, 1])],
            "left_wrist_conf": float(self.kp_conf[KP_LEFT_WRIST]),
            "right_wrist": [float(self.keypoints[KP_RIGHT_WRIST, 0]), float(self.keypoints[KP_RIGHT_WRIST, 1])],
            "right_wrist_conf": float(self.kp_conf[KP_RIGHT_WRIST]),
        }


class YoloPoseAdapter:
    """
    Thread-safe Ultralytics YOLO pose wrapper. Single model, serialized inference
    via internal lock so multiple scheduler workers can call detect() safely.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        input_size: int = 640,
    ):
        self.model_path = str(model_path) or self._default_model_path()
        self.device = str(device).lower().strip() or "cuda"
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = int(input_size)
        self.model = None
        self._lock = threading.Lock()
        self._initialized = False

    @staticmethod
    def _default_model_path() -> str:
        """Locate the shipped yolo26n-pose.pt under <repo>/models/yolo/."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / "models" / "yolo" / "yolo26n-pose.pt"
        if candidate.exists():
            return str(candidate)
        # Fallback: let ultralytics auto-download in current working directory.
        return "yolo26n-pose.pt"

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            logger.error("[YoloPoseAdapter] ultralytics not installed: %s", exc)
            return False

        logger.info("[YoloPoseAdapter] Loading %s", self.model_path)
        self.model = YOLO(self.model_path)
        try:
            self.model.to(self.device)
        except Exception as exc:
            logger.warning("[YoloPoseAdapter] device=%s move failed (%s); falling back to cpu", self.device, exc)
            self.device = "cpu"
            self.model.to("cpu")

        # Warmup (2 dummy forward passes to stabilize cudnn kernels).
        try:
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            for _ in range(2):
                _ = self.model(dummy, verbose=False, conf=self.conf_threshold)
        except Exception as exc:
            logger.warning("[YoloPoseAdapter] warmup skipped: %s", exc)

        self._initialized = True
        logger.info("[YoloPoseAdapter] Ready on %s (conf=%.2f)", self.device, self.conf_threshold)
        return True

    def is_ready(self) -> bool:
        return bool(self._initialized and self.model is not None)

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> List[PersonDetection]:
        """
        Run pose estimation on one BGR frame. Returns list of PersonDetection.
        Empty list if not initialized or no persons above threshold.
        """
        if not self.is_ready():
            return []

        threshold = float(conf) if conf is not None else self.conf_threshold
        with self._lock:
            try:
                results = self.model(
                    frame,
                    verbose=False,
                    conf=threshold,
                    iou=self.iou_threshold,
                    imgsz=self.input_size,
                )
            except Exception as exc:
                logger.debug("[YoloPoseAdapter] inference error: %s", exc)
                return []

        if not results:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()          # (N, 4)
        bbox_conf = r.boxes.conf.cpu().numpy()      # (N,)
        if r.keypoints is None:
            return []
        kps = r.keypoints.xy.cpu().numpy()          # (N, 17, 2)
        kp_conf = r.keypoints.conf.cpu().numpy()    # (N, 17)

        out: List[PersonDetection] = []
        for i in range(len(boxes)):
            out.append(PersonDetection(
                bbox=(float(boxes[i, 0]), float(boxes[i, 1]),
                      float(boxes[i, 2]), float(boxes[i, 3])),
                bbox_conf=float(bbox_conf[i]),
                keypoints=kps[i],
                kp_conf=kp_conf[i],
            ))
        return out

    def cleanup(self) -> None:
        self._initialized = False
        self.model = None
