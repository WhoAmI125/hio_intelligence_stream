"""Temporal event proposal engine for HIO v3."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from model_server import config
from model_server.proposal.yolo26_runner import DetectionObject, Yolo26FrameResult
from model_server.skeleton.pose_features import (
    LEFT_WRIST,
    RIGHT_WRIST,
    bbox_center,
    bbox_intersects_polygon,
    point_in_polygon,
    summarize_skeletons,
)
from model_server.proposal.cashier_tracker_v3 import CashierTrackerV3


class TemporalEventEngine:
    """Convert YOLO frame observations into coarse event candidates."""

    def __init__(self) -> None:
        self._history: dict[str, Any] = defaultdict(lambda: deque(maxlen=48))
        self.cash_threshold = float(getattr(config, "V3_CASH_PREFILTER_THRESHOLD", 0.45))
        self.fire_threshold = float(getattr(config, "V3_FIRE_PREFILTER_THRESHOLD", 0.35))
        self.violence_threshold = float(getattr(config, "V3_VIOLENCE_PREFILTER_THRESHOLD", 0.45))
        # Violence semantic gate: classifier-aware. Lower than zero-shot prompt gate
        # because Human-Action-Recognition outputs calibrated direct probability.
        self.violence_semantic_gate = float(getattr(config, "V3_ACTION_FIGHT_MIN_SCORE", 0.40))
        # Tier-2 SigLIP SUPPRESSION gates: if SigLIP semantic score is below
        # the gate AND was actually computed (>0), override the Tier-1 pose
        # score to 0. This prevents wasting Gemini calls when SigLIP strongly
        # disagrees with the pose signal (e.g. wrist in ROI but empty counter).
        self.cash_siglip_gate = float(getattr(config, "V3_CASH_SIGLIP_GATE", 0.30))
        self.violence_siglip_gate = float(getattr(config, "V3_VIOLENCE_SIGLIP_GATE", 0.25))
        self.fire_siglip_floor = float(getattr(config, "V3_FIRE_SIGLIP_FLOOR", 0.15))
        self.scenarios = tuple(getattr(config, "V3_SCENARIOS", ["cash", "fire", "violence"]) or ["cash", "fire", "violence"])
        self.cashier_tracker = CashierTrackerV3()
        self.cash_episode_window_sec = float(getattr(config, "V3_CASH_EPISODE_WINDOW_SEC", 5.0))
        self.cash_min_hits = max(1, int(getattr(config, "V3_CASH_MIN_HITS", 2)))
        self.cash_exchange_margin_px = float(getattr(config, "V3_CASH_EXCHANGE_MARGIN_PX", 90.0))
        self.cash_wrist_proximity_px = float(getattr(config, "V3_CASH_WRIST_PROXIMITY_PX", 170.0))
        self._cash_hits: dict[str, Any] = defaultdict(lambda: deque(maxlen=48))
        self._cash_track_history: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=24)))

    @staticmethod
    def _object_in_zone(obj: DetectionObject, zone: list[Any]) -> bool:
        return bbox_intersects_polygon(obj.bbox, zone)

    @staticmethod
    def _zone_bbox(zone: list[Any]) -> tuple[float, float, float, float] | None:
        if not zone:
            return None
        pts: list[tuple[float, float]] = []
        for p in zone:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def _point_near_zone(cls, point: tuple[float, float], zone: list[Any], margin_px: float = 0.0) -> bool:
        if not zone or len(zone) < 3:
            return False
        if point_in_polygon(point, zone):
            return True
        bbox = cls._zone_bbox(zone)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        x, y = point
        margin = max(0.0, float(margin_px or 0.0))
        return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)

    @staticmethod
    def _person_center(person: Any) -> tuple[float, float]:
        return bbox_center(list(getattr(person, "bbox", []) or []))

    @staticmethod
    def _person_wrists(person: Any, min_conf: float = 0.15) -> list[tuple[str, tuple[float, float], float]]:
        keypoints = list(getattr(person, "keypoints", []) or [])
        keypoint_conf = list(getattr(person, "keypoint_conf", []) or [])
        out: list[tuple[str, tuple[float, float], float]] = []
        for idx, name in ((LEFT_WRIST, "left_wrist"), (RIGHT_WRIST, "right_wrist")):
            if idx >= len(keypoints):
                continue
            xy = keypoints[idx]
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            conf = float(keypoint_conf[idx]) if idx < len(keypoint_conf) else 0.0
            if conf < min_conf:
                continue
            try:
                out.append((name, (float(xy[0]), float(xy[1])), conf))
            except Exception:
                continue
        return out

    def _cash_exchange_features(
        self,
        camera_id: str,
        yolo: Yolo26FrameResult,
        zones: dict[str, list[Any]],
        now_mono: float,
        *,
        append_history: bool,
    ) -> dict[str, Any]:
        cashier_zone = zones.get("cashier") or []
        exchange_zone = zones.get("exchange") or zones.get("exchange_band") or []
        staff_work_zone = zones.get("staff_work") or zones.get("staff_work_zone") or []
        exchange_configured = len(exchange_zone) >= 3
        if not exchange_configured:
            exchange_zone = cashier_zone

        wrist_rows: list[dict[str, Any]] = []
        person_rows: list[dict[str, Any]] = []
        track_history = self._cash_track_history[camera_id]

        for idx, person in enumerate(yolo.people):
            track_id = str(getattr(person, "track_id", "") or f"p{idx + 1}")
            center = self._person_center(person)
            wrists = self._person_wrists(person)
            center_near_exchange = self._point_near_zone(center, exchange_zone, self.cash_exchange_margin_px)
            center_in_staff_work = self._point_near_zone(center, staff_work_zone, 0.0)
            person_rows.append(
                {
                    "track_id": track_id,
                    "center": [round(center[0], 1), round(center[1], 1)],
                    "center_near_exchange": center_near_exchange,
                    "center_in_staff_work": center_in_staff_work,
                    "wrist_count": len(wrists),
                }
            )
            if append_history:
                track_history[track_id].append(
                    {
                        "ts": now_mono,
                        "center": center,
                        "wrists": [pt for _, pt, _ in wrists],
                    }
                )
            for wrist_name, point, conf in wrists:
                near_exchange = self._point_near_zone(point, exchange_zone, self.cash_exchange_margin_px)
                in_exchange = self._point_near_zone(point, exchange_zone, 0.0)
                in_cashier = self._point_near_zone(point, cashier_zone, 0.0)
                in_staff_work = self._point_near_zone(point, staff_work_zone, 0.0)
                wrist_rows.append(
                    {
                        "track_id": track_id,
                        "wrist": wrist_name,
                        "point": [round(point[0], 1), round(point[1], 1)],
                        "confidence": round(conf, 3),
                        "near_exchange": near_exchange,
                        "in_exchange": in_exchange,
                        "in_cashier": in_cashier,
                        "in_staff_work": in_staff_work,
                        "center_near_exchange": center_near_exchange,
                        "center_in_staff_work": center_in_staff_work,
                    }
                )

        min_cross_track_wrist_distance = None
        proximity_pair = None
        for i, a in enumerate(wrist_rows):
            ax, ay = a["point"]
            for b in wrist_rows[i + 1:]:
                if a["track_id"] == b["track_id"]:
                    continue
                if not (a["near_exchange"] or b["near_exchange"]):
                    continue
                bx, by = b["point"]
                dist = ((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2) ** 0.5
                if min_cross_track_wrist_distance is None or dist < min_cross_track_wrist_distance:
                    min_cross_track_wrist_distance = dist
                    proximity_pair = [a["track_id"], b["track_id"]]

        exchange_motion_count = 0
        crossing_count = 0
        for wrist in wrist_rows:
            if not wrist["near_exchange"]:
                continue
            track_id = str(wrist["track_id"])
            px, py = [float(v) for v in wrist["point"][:2]]
            history = list(track_history.get(track_id) or [])
            recent = [h for h in history if now_mono - float(h.get("ts", now_mono)) <= self.cash_episode_window_sec]
            moved = False
            crossed = False
            for row in recent[:-1]:
                old_points = list(row.get("wrists") or [])
                if row.get("center") is not None:
                    old_points.append(row.get("center"))
                for old in old_points:
                    if old is None:
                        continue
                    ox, oy = float(old[0]), float(old[1])
                    dist = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
                    if dist >= 24.0:
                        moved = True
                    if (
                        dist >= 18.0
                        and not self._point_near_zone((ox, oy), exchange_zone, 0.0)
                        and self._point_near_zone((px, py), exchange_zone, self.cash_exchange_margin_px)
                    ):
                        crossed = True
            if moved:
                exchange_motion_count += 1
            if crossed:
                crossing_count += 1

        exchange_wrist_count = sum(1 for w in wrist_rows if w["near_exchange"])
        cashier_wrist_count = sum(1 for w in wrist_rows if w["in_cashier"])
        staff_work_wrist_count = sum(1 for w in wrist_rows if w["in_staff_work"])
        customer_near_exchange = any(
            bool(p["center_near_exchange"]) and not bool(p["center_in_staff_work"])
            for p in person_rows
        ) or any(bool(w["near_exchange"]) and not bool(w["center_in_staff_work"]) for w in wrist_rows)
        wrist_proximity = (
            min_cross_track_wrist_distance is not None
            and min_cross_track_wrist_distance <= self.cash_wrist_proximity_px
        )
        loose_wrist_proximity = (
            min_cross_track_wrist_distance is not None
            and min_cross_track_wrist_distance <= self.cash_wrist_proximity_px * 1.35
        )
        cashier_zone_only = bool(cashier_wrist_count > 0 and exchange_wrist_count == 0 and not loose_wrist_proximity)
        staff_work_only = bool(staff_work_wrist_count > 0 and exchange_wrist_count == 0 and not loose_wrist_proximity)
        staff_only_near_exchange = bool(exchange_wrist_count > 0 and not customer_near_exchange and not loose_wrist_proximity)
        pose_hit = bool(
            exchange_wrist_count > 0
            and (
                wrist_proximity
                or loose_wrist_proximity
                or exchange_motion_count > 0
                or crossing_count > 0
            )
            and not staff_work_only
            and not staff_only_near_exchange
        )
        soft_hit = bool(exchange_wrist_count > 0 and not staff_work_only and not staff_only_near_exchange)

        hits = self._cash_hits[camera_id]
        cutoff = now_mono - self.cash_episode_window_sec
        while hits and float(hits[0]) < cutoff:
            hits.popleft()
        if append_history and soft_hit:
            hits.append(now_mono)
        recent_hit_count = sum(1 for ts in hits if float(ts) >= cutoff)

        return {
            "exchange_configured": exchange_configured,
            "exchange_wrist_count": exchange_wrist_count,
            "cashier_wrist_count": cashier_wrist_count,
            "staff_work_wrist_count": staff_work_wrist_count,
            "customer_near_exchange": customer_near_exchange,
            "wrist_proximity": wrist_proximity,
            "loose_wrist_proximity": loose_wrist_proximity,
            "min_cross_track_wrist_distance_px": round(float(min_cross_track_wrist_distance), 1)
            if min_cross_track_wrist_distance is not None
            else None,
            "proximity_pair": proximity_pair,
            "exchange_motion_count": exchange_motion_count,
            "crossing_count": crossing_count,
            "cashier_zone_only": cashier_zone_only,
            "staff_work_only": staff_work_only,
            "staff_only_near_exchange": staff_only_near_exchange,
            "pose_hit": pose_hit,
            "soft_hit": soft_hit,
            "recent_hit_count": recent_hit_count,
            "episode_window_sec": self.cash_episode_window_sec,
            "min_hits_required": self.cash_min_hits,
            "wrist_rows": wrist_rows[:12],
            "person_rows": person_rows[:12],
        }

    @staticmethod
    def _wrist_in_zone(skeleton_summary: dict[str, Any]) -> bool:
        return bool(skeleton_summary.get("any_wrist_in_cashier_zone"))

    @staticmethod
    def _handover_like(skeleton_summary: dict[str, Any]) -> bool:
        wrists = skeleton_summary.get("wrist_events") or []
        persons = {str(w.get("person_id")) for w in wrists if w.get("inside_cashier_zone")}
        return len(persons) >= 2 or (len(wrists) >= 2 and bool(persons))

    @staticmethod
    def _max_conf(objects: list[DetectionObject], event_tag: str) -> float:
        scores = [float(o.confidence) for o in objects if o.event_tag == event_tag]
        return max(scores) if scores else 0.0

    @staticmethod
    def _semantic_score(yolo: Yolo26FrameResult, scenario: str) -> float:
        semantic = getattr(yolo, "semantic_scores", {}) or {}
        scores = semantic.get("scores", {}) if isinstance(semantic, dict) else {}
        try:
            return max(0.0, min(1.0, float(scores.get(scenario, 0.0) or 0.0)))
        except Exception:
            return 0.0

    @staticmethod
    def _semantic_detail(yolo: Yolo26FrameResult, scenario: str) -> dict[str, Any]:
        semantic = getattr(yolo, "semantic_scores", {}) or {}
        details = semantic.get("details", {}) if isinstance(semantic, dict) else {}
        row = details.get(scenario, {}) if isinstance(details, dict) else {}
        return row if isinstance(row, dict) else {}

    def _cash_result(
        self,
        camera_id: str,
        yolo: Yolo26FrameResult,
        zones: dict[str, list[Any]],
        skeleton_summary: dict[str, Any],
        now_mono: float,
        append_history: bool,
    ) -> dict[str, Any]:
        cashier_zone = zones.get("cashier") or []
        exchange_zone = zones.get("exchange") or zones.get("exchange_band") or []
        wrist_in_zone = self._wrist_in_zone(skeleton_summary)
        handover_like = self._handover_like(skeleton_summary)
        exchange_features = self._cash_exchange_features(
            camera_id,
            yolo,
            zones,
            now_mono,
            append_history=append_history,
        )
        role_summary = self.cashier_tracker.summarize(
            camera_id=camera_id,
            people=yolo.people,
            cashier_zone=cashier_zone,
            now_ts=now_mono,
        )
        cash_objects = [
            obj
            for obj in yolo.objects
            if obj.event_tag == "cash_like"
            and (not cashier_zone or self._object_in_zone(obj, cashier_zone))
        ]
        max_cash_conf = max([float(o.confidence) for o in cash_objects], default=0.0)
        semantic_score = self._semantic_score(yolo, "cash")
        score = 0.0
        reasons: list[str] = []
        if wrist_in_zone:
            score += 0.05
            reasons.append("wrist_inside_cashier_zone_context_only")
        if exchange_features["customer_near_exchange"]:
            score += 0.15
            reasons.append("customer_track_near_exchange_band")
        if exchange_features["exchange_wrist_count"] > 0:
            score += 0.20
            reasons.append("wrist_near_exchange_band")
        if exchange_features["wrist_proximity"]:
            score += 0.25
            reasons.append("customer_staff_wrist_proximity_near_exchange")
        elif exchange_features["loose_wrist_proximity"]:
            score += 0.15
            reasons.append("loose_wrist_proximity_near_exchange")
        if exchange_features["exchange_motion_count"] > 0:
            score += 0.15
            reasons.append("exchange_band_wrist_motion")
        if exchange_features["crossing_count"] > 0:
            score += 0.10
            reasons.append("wrist_crossing_exchange_band")
        if handover_like and exchange_features["exchange_wrist_count"] > 0:
            score += 0.10
            reasons.append("handover_like_pose")
        if role_summary.get("triggered") and (
            exchange_features["exchange_wrist_count"] > 0
            or exchange_features["loose_wrist_proximity"]
        ):
            score += 0.15
            reasons.append("cashier_tracker_customer_staff_wrist_proximity")
        if cash_objects:
            score += min(0.35, max_cash_conf * 0.35)
            reasons.append("cash_like_object_in_roi")
        if exchange_features["recent_hit_count"] >= self.cash_min_hits:
            score += 0.05
            reasons.append("cash_episode_min_hits_met")
        if semantic_score >= 0.58:
            score += min(0.20, semantic_score * 0.20)
            reasons.append("siglip_cash_context")
        if not exchange_features["exchange_configured"]:
            reasons.append("exchange_band_missing_cashier_fallback")
        if exchange_features["cashier_zone_only"]:
            score -= 0.30
            reasons.append("penalty_cashier_zone_wrist_only")
        if exchange_features["staff_work_only"]:
            score -= 0.45
            reasons.append("penalty_staff_work_only")
        if exchange_features["staff_only_near_exchange"]:
            score -= 0.35
            reasons.append("penalty_staff_only_near_exchange")
        stable_episode = exchange_features["recent_hit_count"] >= self.cash_min_hits
        if exchange_features["pose_hit"] and not stable_episode:
            reasons.append("suppressed_single_frame_cash_hit")
        score = max(0.0, score)
        score = min(score, 1.0)
        # Tier-2 SigLIP SUPPRESSION GATE.
        # If SigLIP cash score was actually computed (>0; i.e. pipeline did
        # invoke semantic_filter for this scenario) AND is below the gate,
        # the scene lacks cashier/handover semantic evidence. Override pose-only
        # proposal to False to skip Gemini call.
        gate_triggered = False
        pre_gate_score = score
        if semantic_score > 0.0 and semantic_score < self.cash_siglip_gate:
            gate_triggered = True
            score = 0.0
            reasons.append(f"siglip_cash_gate<{self.cash_siglip_gate:.2f}")
        detected = bool(
            score >= self.cash_threshold
            and stable_episode
            and (
                exchange_features["pose_hit"]
                or bool(cash_objects)
                or semantic_score >= 0.75
            )
        )
        return {
            "is_detected": detected,
            "confidence": round(score, 3),
            "scenario_type": "cash",
            "zone": "exchange_band" if len(exchange_zone) >= 3 else ("cashier_fallback" if cashier_zone else "full"),
            "evidence": ", ".join(reasons) or "no_cash_candidate_signal",
            "matched_keywords": reasons,
            "raw_response": "YOLO26 temporal cash proposal: " + (", ".join(reasons) or "none"),
            "metadata": {
                "proposal_type": "cash_payment_candidate",
                "cashier_tracker": role_summary,
                "exchange_features": exchange_features,
                "local_prefilter": {
                    "passed": detected,
                    "score": round(score, 3),
                    "pre_gate_score": round(pre_gate_score, 3),
                    "siglip_cash_score": round(semantic_score, 3),
                    "siglip_gate_triggered": gate_triggered,
                    "stable_episode": stable_episode,
                    "reasons": reasons,
                },
            },
        }

    def _fire_result(self, yolo: Yolo26FrameResult) -> dict[str, Any]:
        score = self._max_conf(yolo.objects, "fire_smoke")
        semantic_score = self._semantic_score(yolo, "fire")
        semantic_detail = self._semantic_detail(yolo, "fire")
        neutralizer_score = float(semantic_detail.get("neutralizer_score", 0.0) or 0.0)
        fire_min_score = float(getattr(config, "V3_FIRE_SIGLIP_MIN_SCORE", 0.52))
        neutralizer_threshold = float(getattr(config, "V3_FIRE_NEUTRALIZER_THRESHOLD", 0.58))
        neutralized = bool(neutralizer_score >= neutralizer_threshold and semantic_score < fire_min_score)
        if neutralized:
            score = min(score, score * 0.35)
        if semantic_score >= 0.65:
            score = max(score, min(0.95, semantic_score * 0.8))
        labels = [o.label for o in yolo.objects if o.event_tag == "fire_smoke"][:8]
        reasons = ["fire_smoke_object"] if score > 0 else []
        if neutralized:
            reasons.append("siglip_fire_neutralizer")
        if semantic_score >= 0.65:
            reasons.append("siglip_fire_context")
        return {
            "is_detected": score >= self.fire_threshold,
            "confidence": round(score, 3),
            "scenario_type": "fire",
            "zone": "full",
            "evidence": ", ".join(labels) or "no_fire_smoke_signal",
            "matched_keywords": labels,
            "raw_response": "YOLO26 fire/smoke proposal: " + (", ".join(labels) or "none"),
            "metadata": {
                "proposal_type": "fire_smoke_candidate",
                "local_prefilter": {
                    "passed": score >= self.fire_threshold,
                    "score": round(score, 3),
                    "reasons": reasons,
                    "siglip_fire_score": round(float(semantic_score), 3),
                    "siglip_neutralizer_score": round(float(neutralizer_score), 3),
                },
            },
        }

    def _violence_result(
        self,
        yolo: Yolo26FrameResult,
        skeleton_summary: dict[str, Any],
    ) -> dict[str, Any]:
        weapon_score = self._max_conf(yolo.objects, "violence_like")
        semantic_score = self._semantic_score(yolo, "violence")
        pose_score = 0.0
        reasons: list[str] = []
        if weapon_score > 0:
            reasons.append("weapon_or_violence_object")
        if skeleton_summary.get("person_count", 0) >= 2 and int(skeleton_summary.get("close_person_pair_count", 0) or 0) > 0:
            pose_score += 0.20
            reasons.append("close_person_pair")
        if int(skeleton_summary.get("cross_person_wrist_pair_count", 0) or 0) > 0:
            pose_score += 0.25
            reasons.append("cross_person_wrist_near_body")
        pose_score = min(pose_score, 0.45)
        semantic_bonus = 0.0
        if semantic_score >= self.violence_semantic_gate:
            semantic_bonus = min(0.45, semantic_score * 0.45)
            reasons.append("siglip_violence_context")
        # Tier-2 SigLIP SUPPRESSION GATE for violence. If both zero-shot
        # violence prompt AND classifier "fighting" class agree this is NOT
        # a fight, suppress even when pose suggests close-person pair (friends
        # hugging, staff handover etc.).
        _violence_gate_triggered = False
        if semantic_score > 0.0 and semantic_score < self.violence_siglip_gate:
            _violence_gate_triggered = True
            reasons.append(f"siglip_violence_gate<{self.violence_siglip_gate:.2f}")
        score = min(1.0, max(weapon_score, 0.0) + pose_score + semantic_bonus)
        pre_gate_score = score
        if _violence_gate_triggered:
            score = 0.0
        return {
            "is_detected": score >= self.violence_threshold,
            "confidence": round(score, 3),
            "scenario_type": "violence",
            "zone": "full",
            "evidence": ", ".join(reasons) or "no_violence_candidate_signal",
            "matched_keywords": reasons,
            "raw_response": "YOLO26 violence proposal: " + (", ".join(reasons) or "none"),
            "metadata": {
                "proposal_type": "violence_candidate",
                "local_prefilter": {
                    "passed": score >= self.violence_threshold,
                    "score": round(score, 3),
                    "pre_gate_score": round(pre_gate_score, 3),
                    "siglip_violence_score": round(semantic_score, 3),
                    "siglip_gate_triggered": _violence_gate_triggered,
                    "reasons": reasons,
                },
            },
        }

    def evaluate(
        self,
        camera_id: str,
        yolo: Yolo26FrameResult,
        zones: dict[str, list[Any]],
        *,
        now_mono: float | None = None,
        append_history: bool = True,
        scenario_names: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        now_mono = float(now_mono or time.monotonic())
        skeleton_summary = summarize_skeletons(yolo.people, zones)
        if append_history:
            self._history[camera_id].append(
                {
                    "ts": now_mono,
                    "person_count": skeleton_summary.get("person_count", 0),
                    "wrist_in_cashier": skeleton_summary.get("any_wrist_in_cashier_zone", False),
                    "objects": [obj.event_tag for obj in yolo.objects],
                }
            )
        base_meta = {
            "v3": True,
            "pipeline_version": str(getattr(config, "HIO_V3_PIPELINE_VERSION", "v3-yolo26-tier1-siglip-episode-gemini")),
            "yolo_summary": yolo.summary(),
            "skeleton_summary": skeleton_summary,
            "polygon_coords": {
                "cashier_zone": zones.get("cashier") or [],
                "drawer_zone": zones.get("drawer") or [],
                "exchange_band": zones.get("exchange") or zones.get("exchange_band") or [],
                "staff_work_zone": zones.get("staff_work") or zones.get("staff_work_zone") or [],
            },
            "candidate_clip_paths": {},
        }
        wanted = tuple(scenario_names or self.scenarios)
        result_builders = {
            "cash": lambda: self._cash_result(camera_id, yolo, zones, skeleton_summary, now_mono, append_history),
            "fire": lambda: self._fire_result(yolo),
            "violence": lambda: self._violence_result(yolo, skeleton_summary),
        }
        results = {
            name: result_builders[name]()
            for name in wanted
            if name in result_builders
        }
        for row in results.values():
            meta = row.setdefault("metadata", {})
            meta.update(base_meta)
        return results
