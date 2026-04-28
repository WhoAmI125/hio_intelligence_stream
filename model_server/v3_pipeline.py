"""HIO v3 YOLO26 + semantic prefilter + temporal proposal pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from model_server import config
from model_server.proposal.classifier_heads import (
    ActionClassifierHead,
    FireClassifierHead,
)
from model_server.proposal.episode_manager import EpisodeManager
from model_server.proposal.semantic_filter import SemanticPrefilter
from model_server.proposal.temporal_engine import TemporalEventEngine
from model_server.proposal.yolo26_runner import YOLO26Runner

logger = logging.getLogger(__name__)


@dataclass
class V3PipelineResult:
    scenario_results: dict[str, dict[str, Any]]
    total_inference_time_ms: float
    metadata: dict[str, Any]


class HioV3Pipeline:
    """Runtime bridge used by vlm_api._run_inference_once."""

    def __init__(self) -> None:
        self.runner = YOLO26Runner(
            pose_weights=str(config.YOLO26_POSE_WEIGHTS),
            detect_weights=str(config.YOLO26_DETECT_WEIGHTS),
            cash_weights=str(config.YOLO26_CASH_WEIGHTS),
            fire_weights=str(config.YOLO26_FIRE_WEIGHTS),
            device=str(config.YOLO26_DEVICE),
            imgsz=int(config.YOLO26_IMGSZ),
            pose_conf=float(config.YOLO26_POSE_CONF),
            detect_conf=float(config.YOLO26_DETECT_CONF),
        )
        classifier_device = str(
            getattr(config, "V3_CLASSIFIER_DEVICE", config.YOLO26_DEVICE)
        )
        self.fire_head = FireClassifierHead(
            model_name=str(getattr(config, "V3_FIRE_CLASSIFIER_MODEL", "")),
            device=classifier_device,
            enabled=bool(getattr(config, "V3_FIRE_CLASSIFIER_ENABLED", True)),
        )
        self.action_head = ActionClassifierHead(
            model_name=str(getattr(config, "V3_ACTION_CLASSIFIER_MODEL", "")),
            device=classifier_device,
            enabled=bool(getattr(config, "V3_ACTION_CLASSIFIER_ENABLED", True)),
        )
        self.semantic_filter = SemanticPrefilter(
            model_name=str(getattr(config, "V3_SEMANTIC_MODEL", "")),
            device=str(getattr(config, "V3_SEMANTIC_DEVICE", config.YOLO26_DEVICE)),
            enabled=bool(getattr(config, "V3_SEMANTIC_FILTER_ENABLED", True)),
            fire_head=self.fire_head,
            action_head=self.action_head,
        )
        self.engine = TemporalEventEngine()
        self.episode_manager = EpisodeManager()
        logger.info(
            "HIO v3 pipeline initialized (YOLO26 pose+fire, SigLIP zero-shot + Fire/Action SigLIP2 classifiers, episode, Gemini)."
        )

    def health(self) -> dict[str, Any]:
        """Runtime health summary for /api/vlm/config/ and dashboards."""
        sf = self.semantic_filter
        semantic_health = (
            sf.health() if hasattr(sf, "health") else {
                "name": "SemanticPrefilter",
                "model": getattr(sf, "model_name", ""),
                "device": getattr(sf, "device", ""),
                "enabled": bool(getattr(sf, "enabled", False)),
                "loaded": bool(getattr(sf, "_loaded", False)),
                "failure_count": int(getattr(sf, "_failure_count", 0) or 0),
                "last_error": str(getattr(sf, "_last_error", "")),
                "next_retry_ts": float(getattr(sf, "_next_retry_ts", 0.0) or 0.0),
            }
        )
        return {
            "pipeline_version": str(
                getattr(
                    config,
                    "HIO_V3_PIPELINE_VERSION",
                    "v3-yolo26-tier1-siglip2classifier-episode-gemini",
                )
            ),
            "semantic_filter": semantic_health,
            "fire_classifier": self.fire_head.health(),
            "action_classifier": self.action_head.health(),
        }

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

    def _sample_clip_entries_for_cash(self, entries: list[Any]) -> list[Any]:
        rows: list[tuple[int, Any, float | None]] = []
        for idx, entry in enumerate(entries or []):
            if isinstance(entry, dict):
                frame = entry.get("frame")
                if frame is None:
                    continue
                try:
                    mono_ts = float(entry.get("mono_ts"))
                except Exception:
                    mono_ts = None
                rows.append((idx, frame, mono_ts))
            elif entry is not None:
                rows.append((idx, entry, None))
        if not rows:
            return []

        max_frames = max(1, int(getattr(config, "V3_CASH_SIGLIP_CLIP_FRAMES", 12) or 12))
        if len(rows) <= max_frames:
            return [row[1] for row in rows]

        peak_window_sec = max(0.5, float(getattr(config, "V3_CASH_SIGLIP_CLIP_PEAK_WINDOW_SEC", 5.0) or 5.0))
        anchor_ts = rows[-1][2]
        if anchor_ts is None:
            return self._uniform_sample([row[1] for row in rows], max_frames)

        recent = [row for row in rows if row[2] is not None and anchor_ts - float(row[2]) <= peak_window_sec]
        recent_indices = {row[0] for row in recent}
        context = [row for row in rows if row[0] not in recent_indices]
        recent_budget = min(max_frames, max(1, int(round(max_frames * 0.70))))
        context_budget = max(0, max_frames - min(len(recent), recent_budget))

        selected: list[tuple[int, Any, float | None]] = []
        if context_budget > 0:
            selected.extend(self._uniform_sample(context, context_budget))
        selected.extend(self._uniform_sample(recent, max_frames - len(selected)))
        selected = sorted(selected, key=lambda row: row[0])
        return [row[1] for row in selected[:max_frames]]

    def score_cash_clip(self, entries: list[Any]) -> dict[str, Any]:
        frames = self._sample_clip_entries_for_cash(entries)
        return self.semantic_filter.score_clip_frames(
            frames,
            scenario="cash",
            max_frames=len(frames) or int(getattr(config, "V3_CASH_SIGLIP_CLIP_FRAMES", 12) or 12),
            batch_size=int(getattr(config, "V3_CASH_SIGLIP_CLIP_BATCH_SIZE", 4) or 4),
            min_score=float(getattr(config, "V3_CASH_SIGLIP_CLIP_MIN_SCORE", 0.50) or 0.50),
            frame_positive=float(getattr(config, "V3_CASH_SIGLIP_CLIP_FRAME_POSITIVE", 0.48) or 0.48),
            min_positive_frames=int(getattr(config, "V3_CASH_SIGLIP_CLIP_MIN_POSITIVE_FRAMES", 2) or 2),
        )

    def process_frame(self, camera_id: str, frame: Any, zones: dict[str, list[Any]]) -> V3PipelineResult:
        started = time.time()
        yolo_result = self.runner.process_frame(camera_id, frame)
        now_mono = time.monotonic()
        tier1_results = self.engine.evaluate(
            camera_id,
            yolo_result,
            zones,
            now_mono=now_mono,
            append_history=True,
        )
        tier1_passed = [name for name, row in tier1_results.items() if bool(row.get("is_detected"))]
        scenario_results = dict(tier1_results)
        if tier1_passed:
            yolo_result.semantic_scores = self.semantic_filter.score_frame(frame, scenarios=tier1_passed)
            tier2_results = self.engine.evaluate(
                camera_id,
                yolo_result,
                zones,
                now_mono=now_mono,
                append_history=False,
                scenario_names=tier1_passed,
            )
            scenario_results.update(tier2_results)
        else:
            yolo_result.semantic_scores = {"enabled": False, "scores": {}, "details": {}, "skipped": "no_tier1_candidate"}

        scenario_results = self.episode_manager.filter_results(
            camera_id=camera_id,
            scenario_results=scenario_results,
            now_mono=now_mono,
        )
        elapsed_ms = round((time.time() - started) * 1000.0, 1)
        metadata = {
            "pipeline_version": str(getattr(config, "HIO_V3_PIPELINE_VERSION", "v3-yolo26-tier1-siglip-episode-gemini")),
            "yolo_summary": yolo_result.summary(),
            "tier1_passed": tier1_passed,
            "total_inference_time_ms": elapsed_ms,
        }
        return V3PipelineResult(
            scenario_results=scenario_results,
            total_inference_time_ms=elapsed_ms,
            metadata=metadata,
        )
