"""Tier 1 Fire/Violence Trigger — CLIP ViT-L/14 zero-shot classification."""

from __future__ import annotations

import logging

import clip
import numpy as np
import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)


class CLIPTrigger:
    """Zero-shot fire/violence detection via CLIP text-image similarity."""

    def __init__(self, model_name: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            model_name or config.CLIP_MODEL, device=self.device
        )
        self.model.eval()

        self.prompts = {
            "fire": {
                "positive": ["fire burning in a room", "smoke rising indoors"],
                "negative": ["normal room", "steam from cooking", "fog or mist"],
            },
            "violence": {
                "positive": ["people fighting violently", "physical assault or attack"],
                "negative": [
                    "normal interaction",
                    "people playing",
                    "friendly hug",
                ],
            },
        }
        self.thresholds = {
            "fire": config.CLIP_FIRE_THRESHOLD,
            "violence": config.CLIP_VIOLENCE_THRESHOLD,
        }
        self._text_features = self._encode_texts()
        logger.info("CLIPTrigger loaded: %s on %s", config.CLIP_MODEL, self.device)

    @torch.no_grad()
    def _encode_texts(self) -> dict:
        features = {}
        for scenario, prompts in self.prompts.items():
            all_texts = prompts["positive"] + prompts["negative"]
            tokens = clip.tokenize(all_texts).to(self.device)
            text_feats = self.model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            features[scenario] = {
                "features": text_feats,
                "n_positive": len(prompts["positive"]),
            }
        return features

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> dict[str, dict]:
        """Classify a frame for fire and violence.

        Args:
            frame_bgr: BGR numpy array from OpenCV

        Returns:
            {"fire": {"triggered": bool, "score": float},
             "violence": {"triggered": bool, "score": float}}
        """
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB
        image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        results = {}
        for scenario, data in self._text_features.items():
            similarity = (image_features @ data["features"].T).softmax(dim=-1)[0]
            positive_score = similarity[: data["n_positive"]].max().item()
            results[scenario] = {
                "triggered": positive_score > self.thresholds[scenario],
                "score": round(positive_score, 4),
            }

        return results
