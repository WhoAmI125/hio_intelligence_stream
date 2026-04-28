"""Fine-tuned SigLIP2 classifier heads for HIO v3 Tier-2.

Two drop-in open-weight classifiers:

* FireClassifierHead  - prithivMLmods/Fire-Detection-Siglip2 (Apache 2.0)
                         3 classes: Fire / Normal / Smoke, 99.41% reported acc.
* ActionClassifierHead - prithivMLmods/Human-Action-Recognition (Apache 2.0)
                         15 action classes including "fighting" (F1 0.84).

Only run when Tier-1 (YOLO/pose) produced a candidate for the scenario. The
base SemanticPrefilter (zero-shot SigLIP cash prompts) continues to own cash.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class _BaseClassifierHead:
    """Common loader with exponential backoff retry.

    Subclasses set ``MODEL_NAME`` (HuggingFace id) and ``LABELS`` (ordered).

    Failure handling: after ``MAX_LOAD_FAILURES`` consecutive load failures the
    classifier is considered permanently failed. ``health()`` returns
    ``permanent_failure=True`` and a single ERROR-level log is emitted so the
    operator is informed that an environment fix + process restart is needed.
    Exponential backoff continues but at the capped 300s interval.
    """

    MODEL_NAME: str = ""
    LABELS: list[str] = []
    MAX_LOAD_FAILURES: int = 8  # after ~10 min of backoff, surface to operator

    def __init__(self, *, model_name: str = "", device: str = "cuda", enabled: bool = True) -> None:
        self.model_name = (model_name or self.MODEL_NAME).strip()
        self.device = str(device or "cpu").strip()
        self.enabled = bool(enabled and self.model_name)
        self._loaded = False
        self._failure_count = 0
        self._next_retry_ts = 0.0
        self._last_error = ""
        self._permanent_failure = False
        self._permanent_logged = False
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> bool:
        if self._loaded:
            return True
        if not self.enabled:
            return False
        if time.time() < self._next_retry_ts:
            return False
        try:
            from model_server import config
            import torch
            from transformers import AutoImageProcessor, SiglipForImageClassification

            if self.device.lower().startswith("cuda") and not torch.cuda.is_available():
                if not bool(getattr(config, "ALLOW_CPU_FALLBACK", False)):
                    raise RuntimeError(
                        f"CUDA requested for {self.model_name} but torch.cuda.is_available() is false"
                    )
                self.device = "cpu"
                logger.warning(
                    "[%s] CUDA unavailable; CPU fallback enabled for %s",
                    type(self).__name__,
                    self.model_name,
                )
            self._torch = torch
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = (
                SiglipForImageClassification.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )
            self._loaded = True
            self._failure_count = 0
            self._last_error = ""
            logger.info(
                "[%s] loaded %s on %s",
                type(self).__name__,
                self.model_name,
                self.device,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - surface every load failure via health
            self._failure_count += 1
            delay = min(300.0, 5.0 * (2 ** min(self._failure_count - 1, 6)))
            self._next_retry_ts = time.time() + delay
            self._last_error = str(exc)
            if self._failure_count >= self.MAX_LOAD_FAILURES:
                self._permanent_failure = True
                if not self._permanent_logged:
                    logger.error(
                        "[%s] PERMANENT LOAD FAILURE after %d attempts: %s. "
                        "Classifier output will not be available until the "
                        "environment is fixed and the service restarted. "
                        "Check /api/vlm/config/ for last_error.",
                        type(self).__name__,
                        self._failure_count,
                        self.model_name,
                    )
                    self._permanent_logged = True
            else:
                logger.warning(
                    "[%s] load failed (%d/%d); retry in %.1fs (%s): %s",
                    type(self).__name__,
                    self._failure_count,
                    self.MAX_LOAD_FAILURES,
                    delay,
                    self.model_name,
                    exc,
                )
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _softmax(self, frame_bgr: Any) -> list[float]:
        if frame_bgr is None or not self._load():
            return []
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {
                k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
            return self._torch.softmax(logits[0].float(), dim=0).cpu().tolist()
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            logger.warning("[%s] inference failed: %s", type(self).__name__, exc)
            return []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "model": self.model_name,
            "device": self.device,
            "enabled": self.enabled,
            "loaded": self._loaded,
            "failure_count": self._failure_count,
            "max_failures": self.MAX_LOAD_FAILURES,
            "permanent_failure": self._permanent_failure,
            "last_error": self._last_error,
            "next_retry_ts": round(self._next_retry_ts, 3) if self._next_retry_ts else 0.0,
        }


class FireClassifierHead(_BaseClassifierHead):
    """SigLIP2-based 3-class fire/smoke/normal classifier."""

    MODEL_NAME = "prithivMLmods/Fire-Detection-Siglip2"
    LABELS = ["Fire", "Normal", "Smoke"]

    def score(self, frame_bgr: Any) -> dict[str, Any]:
        probs = self._softmax(frame_bgr)
        if not probs or len(probs) < 3:
            return {"enabled": False, "error": self._last_error}
        fire, normal, smoke = float(probs[0]), float(probs[1]), float(probs[2])
        return {
            "enabled": True,
            "fire": round(fire, 3),
            "normal": round(normal, 3),
            "smoke": round(smoke, 3),
            "fire_or_smoke": round(max(fire, smoke), 3),
        }


class ActionClassifierHead(_BaseClassifierHead):
    """SigLIP2-based 15-class action classifier.

    We consume the ``fighting`` probability and dampen it by the strongest
    neutral-interaction class (hugging, clapping, laughing, dancing) to keep
    friendly interactions from triggering violence events.
    """

    MODEL_NAME = "prithivMLmods/Human-Action-Recognition"
    LABELS = [
        "calling",
        "clapping",
        "cycling",
        "dancing",
        "drinking",
        "eating",
        "fighting",
        "hugging",
        "laughing",
        "listening_to_music",
        "running",
        "sitting",
        "sleeping",
        "texting",
        "using_laptop",
    ]
    NEUTRAL_KEYS = ("hugging", "clapping", "laughing", "dancing")

    def score(self, frame_bgr: Any) -> dict[str, Any]:
        probs = self._softmax(frame_bgr)
        if not probs or len(probs) < len(self.LABELS):
            return {"enabled": False, "error": self._last_error}
        labeled = {self.LABELS[i]: round(float(probs[i]), 3) for i in range(len(self.LABELS))}
        try:
            from model_server import config

            dampen = float(getattr(config, "V3_ACTION_NEUTRAL_DAMPEN", 0.5))
        except Exception:
            dampen = 0.5
        fighting = labeled.get("fighting", 0.0)
        neutral = max(labeled.get(k, 0.0) for k in self.NEUTRAL_KEYS)
        violence_score = max(0.0, min(1.0, fighting - dampen * neutral))
        labeled["violence_score"] = round(violence_score, 3)
        labeled["neutral_max"] = round(neutral, 3)
        labeled["enabled"] = True
        return labeled
