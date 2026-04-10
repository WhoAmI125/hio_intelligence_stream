"""Tier 2 Video Analyzer — Qwen2.5-VL-3B (4-bit) clip analysis.

Cash: Evidence extractor (structured slots).
Fire/Violence: Detection with yes_count.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

import config
from tier2.agent_prompts import AGENT_PROMPTS

logger = logging.getLogger(__name__)

MAX_VLM_FRAMES = 12

# Required fields per scenario for validation
_CASH_FIELDS = {
    "cash_like_object": False,
    "hand_to_hand_transfer": False,
    "counter_context": False,
    "staff_customer_roles_clear": False,
    "drawer_or_counting": False,
    "non_cash_object": "unknown",
    "confidence": 0.0,
    "reason": "",
}

_DETECTION_FIELDS = {
    "detected": False,
    "confidence": 0.0,
    "yes_count": 0,
    "reason": "",
}


class VideoAnalyzer:
    """Analyze video clips with Qwen2.5-VL-3B (4-bit NF4)."""

    def __init__(self, model_name: str | None = None):
        model_id = model_name or config.QWEN_MODEL
        logger.info("Loading Qwen2.5-VL model (4-bit NF4): %s ...", model_id)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        logger.info("Qwen2.5-VL loaded (4-bit, ~2.8GB VRAM)")

    @torch.no_grad()
    def analyze_clip(self, clip_path: str, scenario: str) -> dict:
        if not clip_path or not Path(clip_path).exists():
            return _fallback_result(scenario, "no_clip_file")

        prompt = AGENT_PROMPTS.get(scenario)
        if not prompt:
            return _fallback_result(scenario, "unknown_scenario")

        import cv2 as _cv2
        from PIL import Image

        cap = _cv2.VideoCapture(clip_path)
        all_frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
        finally:
            cap.release()

        if not all_frames:
            return _fallback_result(scenario, "no_frames_in_clip")

        if len(all_frames) > MAX_VLM_FRAMES:
            step = len(all_frames) / MAX_VLM_FRAMES
            sampled = [all_frames[int(i * step)] for i in range(MAX_VLM_FRAMES)]
        else:
            sampled = all_frames

        pil_frames = [
            Image.fromarray(_cv2.cvtColor(f, _cv2.COLOR_BGR2RGB))
            for f in sampled
        ]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                videos=[pil_frames],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=200)
            response = self.processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0]

            result = _parse_json(response, scenario)

        except Exception as e:
            logger.error("Qwen analysis failed for %s: %s", scenario, e, exc_info=True)
            result = _fallback_result(scenario, f"error: {e}")
            try:
                if "inputs" in locals():
                    del inputs
                if "output_ids" in locals():
                    del output_ids
                torch.cuda.empty_cache()
            except Exception:
                pass

        return result

    def route_result(self, result: dict, scenario: str = "cash") -> str:
        """Route based on scenario-specific evidence."""
        if scenario == "cash":
            # Cash: evidence-based routing
            cash_like = result.get("cash_like_object", False)
            hand_transfer = result.get("hand_to_hand_transfer", False)
            non_cash = str(result.get("non_cash_object", "unknown")).lower().strip()

            # Non-cash object clearly identified → skip Tier 3
            if non_cash in ("smartphone", "card", "hotel_key_card", "receipt", "document", "envelope"):
                return "dismiss"

            # No cash-like object AND no hand transfer → dismiss
            if not cash_like and not hand_transfer:
                conf = result.get("confidence", 0.0)
                if conf < config.QWEN_CONFIDENCE_LOW:
                    return "dismiss"

            return "tier3"
        else:
            # Fire/Violence: confidence-based routing (existing)
            conf = result.get("confidence", 0.0)
            return "tier3" if conf >= config.QWEN_CONFIDENCE_LOW else "dismiss"


def _fallback_result(scenario: str, reason: str) -> dict:
    if scenario == "cash":
        return {**_CASH_FIELDS, "reason": reason}
    return {**_DETECTION_FIELDS, "reason": reason}


def _parse_json(text: str, scenario: str) -> dict:
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?", "", text).strip()
    try:
        start = clean.index("{")
        end = clean.rindex("}") + 1
        data = json.loads(clean[start:end])
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse Qwen response: %s", text[:300])
        return _fallback_result(scenario, "parse_error")

    # Validate and fill missing fields
    template = _CASH_FIELDS if scenario == "cash" else _DETECTION_FIELDS
    for key, default in template.items():
        if key not in data:
            data[key] = default

    # Ensure confidence is float
    try:
        data["confidence"] = float(data["confidence"])
    except (ValueError, TypeError):
        data["confidence"] = 0.0

    # Cash backward compat: compute detected + yes_count from evidence slots
    if scenario == "cash":
        evidence_fields = ["cash_like_object", "hand_to_hand_transfer",
                           "counter_context", "staff_customer_roles_clear",
                           "drawer_or_counting"]
        yes_count = sum(1 for f in evidence_fields if data.get(f, False))
        data["yes_count"] = yes_count
        data["detected"] = yes_count >= 2

    return data
