"""Optional SigLIP/CLIP semantic prefilter for HIO v3 proposals."""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class SemanticPrefilter:
    """Score the current full frame against event text prompts.

    This is not the final detector. It is the middle semantic layer between
    YOLO metadata and Gemini validation.
    """

    PROMPTS: dict[str, dict[str, list[str]]] = {
        "cash": {
            "positive": [
                "a customer handing Korean banknotes to a cashier at a counter",
                "a payment handover at a hotel front desk",
                "hands exchanging visible cash near a register",
            ],
            "neutral": [
                "an empty hotel lobby counter",
                "a person standing normally at a desk",
                "a quiet reception area with no transaction",
                "a person holding a receipt, white paper, phone, or card with no visible cash",
            ],
        },
        "violence": {
            "positive": [
                "two people physically fighting",
                "a person punching or kicking another person",
                "a weapon threat or aggressive physical assault",
            ],
            "neutral": [
                "people standing calmly",
                "a normal conversation in a lobby",
                "a quiet hotel reception area",
            ],
        },
        "fire": {
            "positive": [
                "visible fire flames",
                "visible smoke filling the scene",
                "a fire emergency in an indoor camera view",
            ],
            "neutral": [
                "normal indoor lighting",
                "a clean hotel lobby with no smoke",
                "a quiet room with no fire",
            ],
            "neutralizer": [
                "sunlight glare or bright reflection with no fire",
                "a TV monitor LED screen or sign that looks bright",
                "a red sign orange sign lamp or decorative light",
                "a fire extinguisher with no smoke or flame",
                "steam fog blur or camera artifact with no fire",
            ],
        },
    }

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        enabled: bool = True,
        *,
        fire_head: Any = None,
        action_head: Any = None,
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.device = str(device or "cpu").strip()
        self.enabled = bool(enabled and self.model_name)
        self._loaded = False
        self._failure_count = 0
        self._next_retry_ts = 0.0
        self._last_error = ""
        self._permanent_failure = False
        self._permanent_logged = False
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self.fire_head = fire_head
        self.action_head = action_head

    MAX_LOAD_FAILURES = 8  # ~10 min of backoff before surfacing permanent failure

    def health(self) -> dict[str, Any]:
        return {
            "name": "SemanticPrefilter",
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
            from transformers import AutoModel, AutoProcessor

            if self.device.lower().startswith("cuda") and not torch.cuda.is_available():
                if not bool(getattr(config, "ALLOW_CPU_FALLBACK", False)):
                    raise RuntimeError("CUDA requested for SigLIP but torch.cuda.is_available() is false")
                self.device = "cpu"
                logger.warning("[SemanticPrefilter] CUDA unavailable; CPU fallback enabled.")
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._loaded = True
            self._failure_count = 0
            self._last_error = ""
            logger.info("[SemanticPrefilter] loaded %s on %s", self.model_name, self.device)
            return True
        except Exception as exc:
            self._failure_count += 1
            delay = min(300.0, 5.0 * (2 ** min(self._failure_count - 1, 6)))
            self._next_retry_ts = time.time() + delay
            self._last_error = str(exc)
            if self._failure_count >= self.MAX_LOAD_FAILURES:
                self._permanent_failure = True
                if not self._permanent_logged:
                    logger.error(
                        "[SemanticPrefilter] PERMANENT LOAD FAILURE after %d attempts: %s. "
                        "Zero-shot SigLIP path disabled until env is fixed and service restarted. "
                        "Check /api/vlm/config/ semantic_filter.last_error.",
                        self._failure_count,
                        self.model_name,
                    )
                    self._permanent_logged = True
            else:
                logger.warning(
                    "[SemanticPrefilter] load failed (%d/%d); retry in %.1fs (%s): %s",
                    self._failure_count,
                    self.MAX_LOAD_FAILURES,
                    delay,
                    self.model_name,
                    exc,
                )
            return False

    @staticmethod
    def _to_pil(frame: Any) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _score_prompt_set(
        self,
        image: Image.Image,
        positive: list[str],
        neutral: list[str],
        neutralizer: list[str] | None = None,
    ) -> dict[str, float]:
        rows = self._score_prompt_set_batch([image], positive, neutral, neutralizer)
        return rows[0] if rows else {
            "score": 0.0,
            "positive_score": 0.0,
            "neutral_score": 0.0,
            "neutralizer_score": 0.0,
        }

    def _score_prompt_set_batch(
        self,
        images: list[Image.Image],
        positive: list[str],
        neutral: list[str],
        neutralizer: list[str] | None = None,
    ) -> list[dict[str, float]]:
        if not images:
            return []
        neutralizer = list(neutralizer or [])
        texts = list(positive) + list(neutral) + neutralizer
        inputs = self._processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        logits = getattr(outputs, "logits_per_image", None)
        if logits is None:
            image_embeds = getattr(outputs, "image_embeds", None)
            text_embeds = getattr(outputs, "text_embeds", None)
            if image_embeds is None or text_embeds is None:
                return [
                    {
                        "score": 0.0,
                        "positive_score": 0.0,
                        "neutral_score": 0.0,
                        "neutralizer_score": 0.0,
                        "error": "model returned no logits/embeds",
                    }
                    for _ in images
                ]
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            logits = image_embeds @ text_embeds.T
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = self._torch.softmax(logits.float(), dim=1)
        neutral_start = len(positive)
        neutral_end = neutral_start + len(neutral)
        rows: list[dict[str, float]] = []
        for row in probs:
            pos = float(row[: len(positive)].max().detach().cpu())
            neu = float(row[neutral_start:neutral_end].max().detach().cpu()) if neutral else 0.0
            ntr = float(row[neutral_end:].max().detach().cpu()) if neutralizer else 0.0
            counter = max(neu, ntr)
            score = max(0.0, min(1.0, pos / max(pos + counter, 1e-6)))
            rows.append(
                {
                    "score": round(score, 3),
                    "positive_score": round(pos, 3),
                    "neutral_score": round(neu, 3),
                    "neutralizer_score": round(ntr, 3),
                }
            )
        return rows

    @staticmethod
    def _uniform_sample(items: list[Any], count: int) -> list[Any]:
        if count <= 0 or not items:
            return []
        if len(items) <= count:
            return list(items)
        if count == 1:
            return [items[-1]]
        last = len(items) - 1
        return [items[round((last * i) / (count - 1))] for i in range(count)]

    def score_clip_frames(
        self,
        frames: list[Any],
        *,
        scenario: str = "cash",
        max_frames: int = 12,
        batch_size: int = 4,
        min_score: float = 0.50,
        frame_positive: float = 0.48,
        min_positive_frames: int = 2,
    ) -> dict[str, Any]:
        scenario = str(scenario or "cash").lower().strip()
        prompts = self.PROMPTS.get(scenario)
        if prompts is None:
            return {"enabled": False, "scenario": scenario, "error": "unknown scenario"}

        valid_frames = [f for f in list(frames or []) if f is not None]
        sampled = self._uniform_sample(valid_frames, max(1, int(max_frames or 1)))
        base_loaded = self._load()
        if not base_loaded:
            return {
                "enabled": False,
                "scenario": scenario,
                "model": self.model_name,
                "device": self.device,
                "frame_count_input": len(valid_frames),
                "frame_count_scored": 0,
                "error": self._last_error,
                "passed": False,
                "aggregate_score": 0.0,
            }

        rows: list[dict[str, Any]] = []
        try:
            images = [self._to_pil(frame) for frame in sampled]
            bs = max(1, int(batch_size or 1))
            for start in range(0, len(images), bs):
                batch = images[start:start + bs]
                batch_rows = self._score_prompt_set_batch(
                    batch,
                    prompts["positive"],
                    prompts["neutral"],
                    prompts.get("neutralizer", []),
                )
                rows.extend(batch_rows)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("[SemanticPrefilter] clip scoring failed: %s", exc)
            return {
                "enabled": False,
                "scenario": scenario,
                "model": self.model_name,
                "device": self.device,
                "frame_count_input": len(valid_frames),
                "frame_count_scored": 0,
                "error": str(exc),
                "passed": False,
                "aggregate_score": 0.0,
            }

        scores = [float(row.get("score", 0.0) or 0.0) for row in rows]
        if not scores:
            return {
                "enabled": True,
                "scenario": scenario,
                "model": self.model_name,
                "device": self.device,
                "frame_count_input": len(valid_frames),
                "frame_count_scored": 0,
                "scores": [],
                "passed": False,
                "aggregate_score": 0.0,
            }

        sorted_scores = sorted(scores, reverse=True)
        top = sorted_scores[: min(3, len(sorted_scores))]
        top_mean = sum(top) / max(len(top), 1)
        max_score = sorted_scores[0]
        mean_score = sum(scores) / max(len(scores), 1)
        aggregate = max(top_mean, (0.70 * max_score) + (0.30 * top_mean))
        positive_count = sum(1 for score in scores if score >= float(frame_positive))
        passed = bool(
            aggregate >= float(min_score)
            and positive_count >= max(1, int(min_positive_frames or 1))
        )
        return {
            "enabled": True,
            "scenario": scenario,
            "model": self.model_name,
            "device": self.device,
            "frame_count_input": len(valid_frames),
            "frame_count_scored": len(scores),
            "max_score": round(max_score, 3),
            "mean_score": round(mean_score, 3),
            "top3_mean_score": round(top_mean, 3),
            "aggregate_score": round(aggregate, 3),
            "frame_positive_threshold": round(float(frame_positive), 3),
            "positive_frame_count": int(positive_count),
            "min_positive_frames": int(min_positive_frames or 1),
            "min_score": round(float(min_score), 3),
            "passed": passed,
            "scores": [round(v, 3) for v in scores],
            "details": rows[:16],
        }

    def score_frame(
        self,
        frame: Any,
        scenarios: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any]:
        wanted = {str(v).lower() for v in scenarios} if scenarios else set(self.PROMPTS) | {"violence"}
        out_scores: dict[str, float] = {}
        out_details: dict[str, dict[str, Any]] = {}
        base_loaded = self._load()
        base_error = self._last_error if not base_loaded else ""

        if base_loaded and frame is not None:
            try:
                image = self._to_pil(frame)
                for name, prompts in self.PROMPTS.items():
                    if name not in wanted:
                        continue
                    row = self._score_prompt_set(
                        image,
                        prompts["positive"],
                        prompts["neutral"],
                        prompts.get("neutralizer", []),
                    )
                    out_details[name] = row
                    out_scores[name] = row["score"]
            except Exception as exc:
                logger.warning("[SemanticPrefilter] zero-shot scoring failed: %s", exc)
                self._last_error = str(exc)
                base_error = str(exc)

        # Fine-tuned classifier heads override/augment fire and violence scores.
        if "fire" in wanted and self.fire_head is not None and frame is not None:
            fire_row = self.fire_head.score(frame)
            if fire_row.get("enabled"):
                classifier_fire = float(fire_row.get("fire_or_smoke", 0.0))
                zero_shot_fire = float(out_scores.get("fire", 0.0))
                out_scores["fire"] = max(classifier_fire, zero_shot_fire)
                detail = out_details.setdefault("fire", {})
                detail["classifier_fire"] = fire_row.get("fire")
                detail["classifier_smoke"] = fire_row.get("smoke")
                detail["classifier_normal"] = fire_row.get("normal")
                detail["classifier_source"] = "Fire-Detection-Siglip2"

        if "violence" in wanted and self.action_head is not None and frame is not None:
            act_row = self.action_head.score(frame)
            if act_row.get("enabled"):
                classifier_viol = float(act_row.get("violence_score", 0.0))
                zero_shot_viol = float(out_scores.get("violence", 0.0))
                out_scores["violence"] = max(classifier_viol, zero_shot_viol)
                detail = out_details.setdefault("violence", {})
                detail["classifier_fighting"] = act_row.get("fighting")
                detail["classifier_neutral_max"] = act_row.get("neutral_max")
                detail["classifier_hugging"] = act_row.get("hugging")
                detail["classifier_clapping"] = act_row.get("clapping")
                detail["classifier_laughing"] = act_row.get("laughing")
                detail["classifier_dancing"] = act_row.get("dancing")
                detail["classifier_source"] = "Human-Action-Recognition"

        enabled = bool(out_scores) or base_loaded
        return {
            "enabled": enabled,
            "model": self.model_name,
            "device": self.device,
            "scores": out_scores,
            "details": out_details,
            "error": base_error if not enabled else "",
            "next_retry_ts": round(self._next_retry_ts, 3) if self._next_retry_ts else 0.0,
            "classifier_heads": {
                "fire": bool(self.fire_head is not None),
                "action": bool(self.action_head is not None),
            },
        }
