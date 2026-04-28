"""Candidate clip builder for HIO v3.

Creates the exact media packet Gemini sees: full-frame raw clip plus
full-frame overlays. The CCTV frame is kept intact.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from model_server import config
from model_server.skeleton.pose_features import COCO_LIMBS

logger = logging.getLogger(__name__)


class CandidateClipBuilder:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    @staticmethod
    def estimate_fps(entries: list[dict[str, Any]], default: float = 15.0) -> float:
        if len(entries) >= 2:
            ts0 = float(entries[0].get("mono_ts", 0.0) or 0.0)
            ts1 = float(entries[-1].get("mono_ts", 0.0) or 0.0)
            fps = len(entries) / max(ts1 - ts0, 0.1)
            return min(max(fps, 1.0), 30.0)
        return default

    @staticmethod
    def _frames(entries: list[dict[str, Any]]) -> list[Any]:
        return [e.get("frame") for e in entries if e.get("frame") is not None]

    # COCO pose indices for arms only (shoulders → elbows → wrists).
    # 5=L shoulder 6=R shoulder 7=L elbow 8=R elbow 9=L wrist 10=R wrist.
    _ARM_KEYPOINTS = (5, 6, 7, 8, 9, 10)
    _ARM_LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10))

    @staticmethod
    def _draw_skeleton_overlay(frame: Any, skeleton_summary: dict[str, Any]) -> Any:
        """Draw arms-only pose hint on the frame.

        User requested simplification:
          * only arms (shoulders, elbows, wrists) — no torso/legs/head
          * no left/right labels
          * no person bbox or SoM numbering
          * wrist is marked with a small green dot so cashier-ROI interaction
            is still visually obvious to Gemini + human labeler

        Note: skeleton_summary is a single snapshot captured at event admission
        time, not per-frame pose. Drawing it on every overlay frame causes a
        static "ghost" — unavoidable without per-frame pose inference. Keeping
        the arms-only style minimizes visual clutter from the ghost.
        """
        out = frame.copy()
        persons = skeleton_summary.get("persons") or []
        limb_color = (0, 220, 255)   # cyan yellow
        wrist_color = (0, 255, 0)    # green dot for wrist
        for person in persons[:12]:
            keypoints = person.get("keypoints") or []
            pts: dict[int, tuple[int, int]] = {}
            for idx in CandidateClipBuilder._ARM_KEYPOINTS:
                if idx >= len(keypoints):
                    continue
                kp = keypoints[idx]
                try:
                    conf = float(kp.get("confidence", 0.0) or 0.0)
                    if conf < 0.15:
                        continue
                    pts[idx] = (
                        int(round(float(kp["x"]))),
                        int(round(float(kp["y"]))),
                    )
                except Exception:
                    continue

            for a, b in CandidateClipBuilder._ARM_LIMBS:
                if a in pts and b in pts:
                    cv2.line(out, pts[a], pts[b], limb_color, 2, cv2.LINE_AA)

            # Wrist dots (ids 9, 10) — the single most important signal for cash ROI.
            for idx in (9, 10):
                if idx in pts:
                    cv2.circle(out, pts[idx], 5, wrist_color, -1, cv2.LINE_AA)
        return out

    @staticmethod
    def _draw_zone_polygon(
        frame: Any,
        pts: list[Any],
        *,
        label: str,
        color: tuple[int, int, int],
        thickness: int = 3,
    ) -> Any:
        out = frame.copy()
        if len(pts) < 3:
            return out
        try:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [arr], True, color, thickness, cv2.LINE_AA)
            x0 = int(min(p[0] for p in pts))
            y0 = int(min(p[1] for p in pts))
            cv2.putText(
                out,
                label,
                (x0, max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        except Exception:
            logger.debug("%s overlay draw failed", label, exc_info=True)
        return out

    @staticmethod
    def _draw_static_zones(frame: Any, zones: dict[str, list[Any]]) -> Any:
        out = frame.copy()
        out = CandidateClipBuilder._draw_zone_polygon(
            out,
            zones.get("cashier") or [],
            label="CASHIER ROI",
            color=(0, 0, 255),
            thickness=4,
        )
        out = CandidateClipBuilder._draw_zone_polygon(
            out,
            zones.get("drawer") or [],
            label="DRAWER",
            color=(255, 128, 0),
            thickness=2,
        )
        out = CandidateClipBuilder._draw_zone_polygon(
            out,
            zones.get("exchange") or zones.get("exchange_band") or [],
            label="EXCHANGE BAND",
            color=(0, 255, 255),
            thickness=4,
        )
        out = CandidateClipBuilder._draw_zone_polygon(
            out,
            zones.get("staff_work") or zones.get("staff_work_zone") or [],
            label="STAFF WORK",
            color=(255, 0, 255),
            thickness=2,
        )
        return out

    @staticmethod
    def _draw_cashier_zone_red(frame: Any, zones: dict[str, list[Any]]) -> Any:
        return CandidateClipBuilder._draw_static_zones(frame, zones)

    @staticmethod
    def _draw_context_overlay(frame: Any, zones: dict[str, list[Any]], skeleton_summary: dict[str, Any]) -> Any:
        out = CandidateClipBuilder._draw_skeleton_overlay(frame, skeleton_summary)
        out = CandidateClipBuilder._draw_static_zones(out, zones)
        return out

    @classmethod
    def _make_context_overlay_applier(
        cls,
        zones: dict[str, list[Any]],
        skeleton_summary: dict[str, Any],
        reference_frame: Any,
    ):
        """Pre-render overlay template(s) and return a per-frame applier.

        The skeleton_summary captured at admission time is a SNAPSHOT — drawing
        it on every frame while the real person moves produces a ghost. The
        cashier ROI polygon is the only truly static primitive that can be
        safely drawn on every frame without ghosting.

        Behavior driven by ``V3_OVERLAY_SKELETON_FRAMES``:
          * 0 (default) — ROI-only overlay on every frame. No skeleton anywhere.
                          No ghost. Gemini + operator judge from raw video.
          * N (>0)      — skeleton (arms-only) on first N overlay frames for
                          context, then ROI-only for the rest.

        Uses a closure-local counter so save_clip_stream's ``transform(frame)``
        protocol works without needing explicit index passing.
        """
        if reference_frame is None:
            return lambda f: cls._draw_context_overlay(f, zones, skeleton_summary)
        try:
            from model_server import config as _cfg
            skel_frames = max(0, int(getattr(_cfg, "V3_OVERLAY_SKELETON_FRAMES", 0) or 0))
        except Exception:
            skel_frames = 0
        try:
            h, w = reference_frame.shape[:2]
            template_roi = np.zeros((h, w, 3), dtype=np.uint8)
            template_roi = cls._draw_static_zones(template_roi, zones)
            mask_roi = template_roi.sum(axis=-1) > 0
            has_roi = bool(mask_roi.any())

            template_full = None
            mask_full = None
            has_full = False
            if skel_frames > 0:
                template_full = np.zeros((h, w, 3), dtype=np.uint8)
                template_full = cls._draw_context_overlay(template_full, zones, skeleton_summary)
                mask_full = template_full.sum(axis=-1) > 0
                has_full = bool(mask_full.any())

            if not has_full and not has_roi:
                return lambda f: f.copy() if f is not None else f
        except Exception:
            logger.debug("overlay template pre-render failed; falling back", exc_info=True)
            return lambda f: cls._draw_context_overlay(f, zones, skeleton_summary)

        counter = [0]

        def apply(frame):
            if frame is None:
                return frame
            if frame.shape[:2] != (h, w):
                return cls._draw_context_overlay(frame, zones, skeleton_summary)
            out = frame.copy()
            idx = counter[0]
            counter[0] += 1
            if skel_frames > 0 and idx < skel_frames and has_full:
                out[mask_full] = template_full[mask_full]
            elif has_roi:
                out[mask_roi] = template_roi[mask_roi]
            return out

        return apply

    def _save_transformed(
        self,
        event_id: str,
        frames: list[Any],
        fps: float,
        transform: Any,
    ) -> str | None:
        if not self.storage or not frames:
            return None
        if hasattr(self.storage, "save_clip_stream"):
            return self.storage.save_clip_stream(event_id, frames, fps=fps, transform=transform, allow_s3=False)
        rendered = []
        for frame in frames:
            try:
                rendered.append(transform(frame))
            except Exception:
                rendered.append(frame)
        return self.storage.save_clip(event_id, rendered, fps=fps, allow_s3=False)

    def _save_skeleton_json(self, event_id: str, skeleton_summary: dict[str, Any]) -> str | None:
        if not self.storage:
            return None
        try:
            base_dir = Path(getattr(self.storage, "base_dir"))
            day_dir = base_dir / "clips" / datetime.now().strftime("%Y%m%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            out_path = day_dir / f"{event_id}_skeleton.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(skeleton_summary or {}, f, ensure_ascii=False, indent=2, default=str)
            return str(out_path)
        except Exception:
            logger.debug("skeleton json save failed", exc_info=True)
            return None

    def save_candidate_clips(
        self,
        *,
        event_id: str,
        entries: list[dict[str, Any]],
        fps: float | None,
        zones: dict[str, list[Any]],
        skeleton_summary: dict[str, Any],
        raw_path: str | None = None,
    ) -> dict[str, str]:
        frames = self._frames(entries)
        if not frames or not self.storage:
            return {}
        out: dict[str, str] = {}
        clip_fps = float(fps or self.estimate_fps(entries))

        # Downsample overlay fps to cut per-frame cv2 draw CPU on g4dn.2xlarge.
        try:
            divisor = max(1, int(getattr(config, "V3_OVERLAY_FPS_DIVISOR", 3) or 3))
        except Exception:
            divisor = 3
        overlay_fps = max(1.0, clip_fps / float(divisor))
        if divisor > 1 and len(frames) > divisor:
            overlay_frames = frames[::divisor]
        else:
            overlay_frames = frames

        if raw_path:
            out["raw"] = raw_path
        elif str(getattr(config, "V3_CLIP_ARTIFACT_MODE", "minimal")).lower() == "debug":
            raw = self.storage.save_clip(f"{event_id}_raw", frames, fps=clip_fps, allow_s3=False)
            if raw:
                out["raw"] = raw

        skeleton_json = self._save_skeleton_json(event_id, skeleton_summary)
        if skeleton_json:
            out["skeleton_json"] = skeleton_json

        # Build a pre-rendered overlay applier once per event. skeleton_summary
        # is static for the event so re-running cv2 draw per frame is wasted CPU.
        overlay_applier = self._make_context_overlay_applier(
            zones, skeleton_summary, overlay_frames[0] if overlay_frames else None
        )
        context = self._save_transformed(
            f"{event_id}_context_overlay",
            overlay_frames,
            overlay_fps,
            overlay_applier,
        )
        if context:
            out["context_overlay"] = context

        return out
