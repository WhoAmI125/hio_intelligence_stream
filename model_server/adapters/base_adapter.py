"""
Base VLM Adapter - Abstract base class for Vision-Language Model adapters

Provides unified interface for different VLM backends:
- OpenVINO (CPU optimized)
- ONNX Runtime
- PyTorch (direct)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class VLMInferenceResult:
    """Result from VLM inference"""
    response: str
    inference_time_ms: float
    input_size: Tuple[int, int]
    model_name: str
    backend: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'response': self.response,
            'inference_time_ms': self.inference_time_ms,
            'input_size': self.input_size,
            'model_name': self.model_name,
            'backend': self.backend,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata
        }


class BaseVLMAdapter(ABC):
    """
    Abstract base class for VLM adapters.

    Provides unified interface for running VLM inference
    on images with text prompts.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize VLM adapter.

        Args:
            config: Configuration dictionary with keys:
                - model: Model name or path
                - backend: 'openvino', 'onnx', or 'pytorch'
                - device: 'cpu' or 'cuda'
                - input_size: Target input size (width, height)
                - max_tokens: Maximum output tokens
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'florence-2-large')
        self.backend = self.config.get('backend', 'openvino')
        self.device = self.config.get('device', 'cpu')
        self.input_size = self.config.get('input_size', (448, 448))
        self.max_tokens = self.config.get('max_tokens', 512)

        self.model = None
        self.processor = None
        self.is_initialized = False

        # Statistics
        self.total_inferences = 0
        self.total_inference_time_ms = 0.0
        self.last_inference_time_ms = 0.0

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the VLM model.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def infer(
        self,
        image: np.ndarray,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Run VLM inference on image with prompt.

        Args:
            image: BGR image from OpenCV (HWC format)
            prompt: Text prompt for the model
            **kwargs: Additional inference parameters

        Returns:
            Model's text response
        """
        pass

    def infer_with_result(
        self,
        image: np.ndarray,
        prompt: str,
        **kwargs
    ) -> VLMInferenceResult:
        """
        Run inference and return detailed result.

        Args:
            image: BGR image from OpenCV
            prompt: Text prompt
            **kwargs: Additional parameters

        Returns:
            VLMInferenceResult with response and metadata
        """
        import time

        if not self.is_initialized:
            return VLMInferenceResult(
                response="",
                inference_time_ms=0.0,
                input_size=self.input_size,
                model_name=self.model_name,
                backend=self.backend,
                success=False,
                error="Model not initialized"
            )

        start_time = time.time()

        try:
            response = self.infer(image, prompt, **kwargs)
            inference_time = (time.time() - start_time) * 1000

            # Update statistics
            self.total_inferences += 1
            self.total_inference_time_ms += inference_time
            self.last_inference_time_ms = inference_time

            return VLMInferenceResult(
                response=response,
                inference_time_ms=inference_time,
                input_size=self.input_size,
                model_name=self.model_name,
                backend=self.backend,
                success=True
            )

        except Exception as e:
            inference_time = (time.time() - start_time) * 1000
            return VLMInferenceResult(
                response="",
                inference_time_ms=inference_time,
                input_size=self.input_size,
                model_name=self.model_name,
                backend=self.backend,
                success=False,
                error=str(e)
            )

    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess image for VLM input.

        Args:
            image: BGR image from OpenCV
            target_size: Optional target size (width, height)

        Returns:
            Preprocessed image
        """
        import cv2

        if target_size is None:
            target_size = self.input_size

        # Resize if needed
        h, w = image.shape[:2]
        target_w, target_h = target_size

        if (w, h) != (target_w, target_h):
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def crop_zone(
        self,
        image: np.ndarray,
        zone_polygon: List[List[int]],
        padding: int = 20
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop image to zone bounding box with padding.

        Args:
            image: BGR image
            zone_polygon: List of [x, y] points
            padding: Padding around zone

        Returns:
            (cropped_image, (x1, y1, x2, y2))
        """
        if not zone_polygon:
            return image, (0, 0, image.shape[1], image.shape[0])

        # Get bounding box of polygon
        points = np.array(zone_polygon)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)

        # Add padding
        h, w = image.shape[:2]
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        # Crop
        cropped = image[y1:y2, x1:x2].copy()

        return cropped, (x1, y1, x2, y2)

    def get_inference_time(self) -> float:
        """Return last inference time in milliseconds"""
        return self.last_inference_time_ms

    def get_average_inference_time(self) -> float:
        """Return average inference time in milliseconds"""
        if self.total_inferences == 0:
            return 0.0
        return self.total_inference_time_ms / self.total_inferences

    def get_stats(self) -> Dict:
        """Get adapter statistics"""
        return {
            'model_name': self.model_name,
            'backend': self.backend,
            'device': self.device,
            'is_initialized': self.is_initialized,
            'total_inferences': self.total_inferences,
            'avg_inference_time_ms': self.get_average_inference_time(),
            'last_inference_time_ms': self.last_inference_time_ms
        }

    def cleanup(self):
        """Cleanup resources"""
        self.model = None
        self.processor = None
        self.is_initialized = False
