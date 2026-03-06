"""
Base Detector - Abstract base class for all detectors

Provides common functionality:
- YOLO model loading
- Zone polygon handling
- Keypoint extraction
- Detection result structure
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional


@dataclass
class Detection:
    """Detection result structure"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    metadata: Dict = field(default_factory=dict)
    event_type: str = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'metadata': self.metadata,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class BaseDetector(ABC):
    """Abstract base class for detectors"""

    # COCO Pose keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(self, config: Dict = None):
        """
        Initialize base detector

        Args:
            config: Configuration dictionary with keys:
                - models_dir: Path to models directory
                - device: 'cuda', 'cpu', or 'auto'
                - pose_confidence: Minimum confidence for pose detection
        """
        self.config = config or {}
        self.model = None
        self.device = 'cpu'
        self.is_initialized = False

        # Default settings
        self.models_dir = Path(self.config.get('models_dir', './models'))
        self.pose_confidence = self.config.get('pose_confidence', 0.3)

        # Frame counter
        self.frame_count = 0

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector (load models, etc.)"""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame"""
        pass

    def _setup_device(self) -> str:
        """Setup computation device"""
        device_config = self.config.get('device', 'auto')

        if device_config == 'auto':
            try:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device_config

        return self.device

    def _load_yolo_model(self, model_names: List[str], task: str = None):
        """
        Load YOLO model from priority list

        Args:
            model_names: List of model names to try in order
            task: YOLO task type ('pose', 'detect', etc.)

        Returns:
            Loaded model or None
        """
        try:
            from ultralytics import YOLO

            for model_name in model_names:
                # Try local path first
                local_path = self.models_dir / model_name
                model_path = str(local_path) if local_path.exists() else model_name

                try:
                    model = YOLO(model_path, task=task) if task else YOLO(model_path)
                    model.to(self.device)
                    print(f"[{self.__class__.__name__}] Loaded {model_name} on {self.device}")
                    return model
                except Exception as e:
                    print(f"[{self.__class__.__name__}] {model_name} not available: {e}")
                    continue

            return None

        except ImportError:
            print(f"[{self.__class__.__name__}] ultralytics not installed")
            return None

    # ==================== Zone Utilities ====================

    @staticmethod
    def point_in_polygon(point: Tuple[int, int], polygon: List) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm

        Args:
            point: (x, y) coordinates
            polygon: List of [x, y] points forming the polygon
        """
        if not polygon or len(polygon) < 3:
            return False

        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    # ==================== Keypoint Utilities ====================

    def get_hand_positions(self, keypoints: np.ndarray,
                           confidence_threshold: float = 0.3) -> Dict:
        """
        Extract hand (wrist) positions from pose keypoints

        Args:
            keypoints: Keypoints array from YOLO pose
            confidence_threshold: Minimum confidence to include

        Returns:
            Dict with 'left' and/or 'right' hand positions
        """
        hands = {}

        if keypoints is None or len(keypoints) < 11:
            return hands

        # Left wrist
        if len(keypoints) > self.LEFT_WRIST:
            lw = keypoints[self.LEFT_WRIST]
            if len(lw) >= 3 and lw[2] >= confidence_threshold:
                hands['left'] = (int(lw[0]), int(lw[1]), float(lw[2]))

        # Right wrist
        if len(keypoints) > self.RIGHT_WRIST:
            rw = keypoints[self.RIGHT_WRIST]
            if len(rw) >= 3 and rw[2] >= confidence_threshold:
                hands['right'] = (int(rw[0]), int(rw[1]), float(rw[2]))

        return hands

    def get_person_center(self, keypoints: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get person center point using hip keypoints or bbox fallback

        Args:
            keypoints: Keypoints array
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            (x, y) center point
        """
        # Try hip center
        if keypoints is not None and len(keypoints) > self.RIGHT_HIP:
            left_hip = keypoints[self.LEFT_HIP]
            right_hip = keypoints[self.RIGHT_HIP]

            if (len(left_hip) >= 3 and left_hip[2] > 0.3 and
                len(right_hip) >= 3 and right_hip[2] > 0.3):
                return (
                    int((left_hip[0] + right_hip[0]) / 2),
                    int((left_hip[1] + right_hip[1]) / 2)
                )

            # Try shoulder center
            if len(keypoints) > self.RIGHT_SHOULDER:
                left_sh = keypoints[self.LEFT_SHOULDER]
                right_sh = keypoints[self.RIGHT_SHOULDER]

                if (len(left_sh) >= 3 and left_sh[2] > 0.3 and
                    len(right_sh) >= 3 and right_sh[2] > 0.3):
                    return (
                        int((left_sh[0] + right_sh[0]) / 2),
                        int((left_sh[1] + right_sh[1]) / 2)
                    )

        # Fallback to bbox center
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @staticmethod
    def calculate_distance(p1: Tuple, p2: Tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # ==================== People Extraction ====================

    def extract_people(self, result, cashier_zone: List = None) -> List[Dict]:
        """
        Extract people information from YOLO pose result

        Args:
            result: YOLO result object
            cashier_zone: Optional zone polygon for role assignment

        Returns:
            List of person dictionaries
        """
        people = []

        if result.keypoints is None or result.boxes is None:
            return people

        keypoints_data = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
            bbox = tuple(map(int, box))
            hands = self.get_hand_positions(kpts)
            center = self.get_person_center(kpts, bbox)

            # Determine role if cashier zone provided
            in_cashier_zone = None
            role = 'unknown'
            if cashier_zone:
                in_cashier_zone = self.point_in_polygon(center, cashier_zone)
                role = 'cashier' if in_cashier_zone else 'customer'

            person = {
                'idx': idx,
                'bbox': bbox,
                'hands': hands,
                'keypoints': kpts,
                'center': center,
                'in_cashier_zone': in_cashier_zone,
                'role': role
            }
            people.append(person)

        return people
