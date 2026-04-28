"""Small v3 feedback collector for proposal review UI."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


class V3ProposalFeedbackCollector:
    """Persist operator feedback without pulling in legacy fine-tuning code."""

    def __init__(self, base_dir: str | Path, enabled: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.enabled = bool(enabled)
        self.images_dir = self.base_dir / "images"
        self.annotations_path = self.base_dir / "annotations.jsonl"
        self.feedback_log_path = self.base_dir / "feedback_log.jsonl"
        self.validated_log_path = self.base_dir / "validated_clips.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _safe(value: Any) -> str:
        text = str(value or "").strip()
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:120] or "unknown"

    def collect_proposal_feedback(
        self,
        *,
        event_id: str,
        decision: str,
        note: str,
        frame: Any,
        caption: str,
        scenario: str,
        camera_id: str,
        summary: dict[str, Any],
        source: str = "v3_proposal_feedback",
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"success": False, "error": "collector_disabled"}

        ts = datetime.now().isoformat()
        safe_event = self._safe(event_id or f"proposal_feedback_{int(time.time() * 1000)}")
        safe_camera = self._safe(camera_id)
        safe_scenario = self._safe(scenario)
        image_path = ""

        if frame is not None and getattr(frame, "size", 0):
            image_path_obj = self.images_dir / f"{safe_event}_{safe_camera}_{safe_scenario}.jpg"
            try:
                cv2.imwrite(str(image_path_obj), frame)
                image_path = str(image_path_obj)
            except Exception as exc:
                logger.debug("v3 feedback frame save failed: %s", exc)

        annotation = {
            "at": ts,
            "event_id": event_id,
            "camera_id": camera_id,
            "scenario": scenario,
            "decision": decision,
            "note": note,
            "caption": caption,
            "summary": summary,
            "image_path": image_path,
            "source": source,
        }
        self._append_jsonl(self.annotations_path, annotation)
        self._append_jsonl(self.feedback_log_path, annotation)
        return {
            "success": True,
            "saved_annotation": True,
            "annotations_path": str(self.annotations_path),
            "proposal_feedback_log_path": str(self.feedback_log_path),
            "image_path": image_path,
        }

    def collect_feedback(self, *, event_id: str, decision: str, note: str, scenario: str) -> dict[str, Any]:
        payload = {
            "at": datetime.now().isoformat(),
            "event_id": event_id,
            "scenario": scenario,
            "decision": decision,
            "note": note,
            "source": "v3_event_feedback",
        }
        self._append_jsonl(self.feedback_log_path, payload)
        return {"success": True, "proposal_feedback_log_path": str(self.feedback_log_path)}

    def collect_validated_clip(
        self,
        *,
        event_id: str,
        scenario: str,
        camera_id: str,
        gemini_confidence: float,
        matched_keywords: list[Any],
        caption: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        payload = {
            "at": datetime.now().isoformat(),
            "event_id": event_id,
            "scenario": scenario,
            "camera_id": camera_id,
            "gemini_confidence": float(gemini_confidence or 0.0),
            "matched_keywords": list(matched_keywords or []),
            "caption": caption,
            "source": "v3_gemini_validated_clip",
        }
        self._append_jsonl(self.validated_log_path, payload)
        return {"success": True, "validated_log_path": str(self.validated_log_path)}

    def get_stats(self) -> dict[str, Any]:
        def _count(path: Path) -> int:
            if not path.exists():
                return 0
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)

        return {
            "enabled": self.enabled,
            "base_dir": str(self.base_dir),
            "annotations": _count(self.annotations_path),
            "feedback": _count(self.feedback_log_path),
            "validated_clips": _count(self.validated_log_path),
        }
