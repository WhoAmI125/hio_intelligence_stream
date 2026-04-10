"""Tier 1 Cash Trigger — YOLO Pose detection with two trigger modes.

Mode 1 (wrist): Wrist proximity between two people (original).
Mode 2 (zone_intrusion): Any person's wrist/hand enters the cashier zone from outside.
    Works when cashier is hidden behind counter (e.g., Ilsan camera angle).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


class CashTrigger:

    def __init__(self, model_path: str | None = None):
        self.model = YOLO(model_path or config.YOLO_MODEL)
        self._use_half = False  # FP16 disabled — causes CUDA assert with concurrent CLIP
        logger.info("CashTrigger loaded (FP16): %s", model_path or config.YOLO_MODEL)

    def detect(self, frame: np.ndarray, cam_config: dict) -> dict:
        trigger_mode = cam_config.get("trigger_mode", "wrist")
        if trigger_mode == "zone_intrusion":
            return self._detect_zone_intrusion(frame, cam_config)
        return self._detect_wrist(frame, cam_config)

    # ── Mode 1: Wrist proximity (original) ──────────────────────────
    def _detect_wrist(self, frame: np.ndarray, cam_config: dict) -> dict:
        wrist_threshold = cam_config.get("wrist_px", config.CASH_WRIST_THRESHOLD_PX)
        zone_norm = cam_config.get("cashier_zone", [])

        h, w = frame.shape[:2]
        zone_px = _norm_to_pixel(zone_norm, w, h) if len(zone_norm) >= 3 else None

        results = self.model(frame, verbose=False, half=self._use_half)
        keypoints = results[0].keypoints
        total = 0 if keypoints is None else len(keypoints.xy) if keypoints.xy is not None else 0

        if keypoints is None or keypoints.xy is None or total < 1:
            return _empty_result(zone_px is not None, wrist_threshold, "wrist")

        persons = []
        for i, kp in enumerate(keypoints.xy):
            kp_np = kp.cpu().numpy() if hasattr(kp, "cpu") else np.array(kp)
            lw, rw = kp_np[9], kp_np[10]
            has_lw = lw.sum() > 0
            has_rw = rw.sum() > 0
            if not has_lw and not has_rw:
                continue

            pos = _best_position(kp_np)
            role = "unknown"
            if zone_px is not None and pos is not None:
                role = "staff" if _point_in_polygon(pos, zone_px) else "customer"

            persons.append({
                "index": i, "role": role,
                "left_wrist": lw, "right_wrist": rw,
                "has_lw": has_lw, "has_rw": has_rw,
            })

        staff = [p for p in persons if p["role"] == "staff"]
        customers = [p for p in persons if p["role"] == "customer"]

        if staff and customers:
            pairs = [(s, c) for s in staff for c in customers]
            mode = "zone"
        else:
            pairs = [(persons[a], persons[b]) for a in range(len(persons)) for b in range(a + 1, len(persons))]
            mode = "all_pairs"

        distances = []
        min_dist = float("inf")
        for pa, pb in pairs:
            dist = _min_wrist_distance(pa, pb)
            if dist < float("inf"):
                distances.append({
                    "a_idx": pa["index"], "b_idx": pb["index"],
                    "a_role": pa["role"], "b_role": pb["role"],
                    "wrist_px": round(dist, 1),
                })
                if dist < min_dist:
                    min_dist = dist

        triggered = min_dist < wrist_threshold if distances else False

        return {
            "triggered": triggered,
            "trigger_mode": "wrist",
            "persons_detected": total,
            "wrist_persons": len(persons),
            "staff_count": len(staff),
            "customer_count": len(customers),
            "zone_set": zone_px is not None,
            "mode": mode,
            "wrist_threshold": wrist_threshold,
            "min_distance": round(min_dist, 1) if min_dist < float("inf") else None,
            "distances": distances,
        }

    # ── Mode 2: Zone intrusion (for hidden cashier) ─────────────────
    def _detect_zone_intrusion(self, frame: np.ndarray, cam_config: dict) -> dict:
        """Trigger when an outside person's wrist/hand enters the cashier zone.

        Works even when cashier is hidden behind counter:
        - Detect all persons via YOLO
        - Person's body center is OUTSIDE cashier zone
        - But their wrist/elbow/hand keypoint is INSIDE cashier zone
        - This means someone is reaching into the cashier area → transaction signal
        """
        zone_norm = cam_config.get("cashier_zone", [])
        h, w = frame.shape[:2]
        zone_px = _norm_to_pixel(zone_norm, w, h) if len(zone_norm) >= 3 else None

        if zone_px is None:
            return _empty_result(False, 0, "zone_intrusion")

        results = self.model(frame, verbose=False, half=self._use_half)
        keypoints = results[0].keypoints
        boxes = results[0].boxes
        total = 0 if keypoints is None else len(keypoints.xy) if keypoints.xy is not None else 0

        if keypoints is None or keypoints.xy is None or total < 1:
            return _empty_result(True, 0, "zone_intrusion")

        intrusions = []
        outside_persons = 0
        inside_persons = 0

        for i, kp in enumerate(keypoints.xy):
            kp_np = kp.cpu().numpy() if hasattr(kp, "cpu") else np.array(kp)
            pos = _best_position(kp_np)

            if pos is None:
                continue

            is_inside = _point_in_polygon(pos, zone_px)
            if is_inside:
                inside_persons += 1
                continue  # cashier side — skip

            outside_persons += 1

            # Check if any hand/wrist/elbow keypoint enters the zone
            # Keypoints: 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist
            reach_points = [
                ("left_wrist", kp_np[9]),
                ("right_wrist", kp_np[10]),
                ("left_elbow", kp_np[7]),
                ("right_elbow", kp_np[8]),
            ]

            for name, pt in reach_points:
                if pt.sum() > 0 and _point_in_polygon(pt, zone_px):
                    intrusions.append({
                        "person_idx": i,
                        "keypoint": name,
                        "x": round(float(pt[0]), 1),
                        "y": round(float(pt[1]), 1),
                    })

        triggered = len(intrusions) > 0

        return {
            "triggered": triggered,
            "trigger_mode": "zone_intrusion",
            "persons_detected": total,
            "outside_persons": outside_persons,
            "inside_persons": inside_persons,
            "intrusion_count": len(intrusions),
            "intrusions": intrusions[:5],  # limit for JSON size
            "zone_set": True,
            "mode": "zone_intrusion",
            "wrist_threshold": 0,
            "min_distance": None,
            "distances": [],
            "staff_count": inside_persons,
            "customer_count": outside_persons,
        }


def _empty_result(zone_set: bool, threshold: int, trigger_mode: str) -> dict:
    return {
        "triggered": False,
        "trigger_mode": trigger_mode,
        "persons_detected": 0,
        "staff_count": 0,
        "customer_count": 0,
        "zone_set": zone_set,
        "mode": "none",
        "wrist_threshold": threshold,
        "min_distance": None,
        "distances": [],
    }


def _best_position(kp: np.ndarray) -> np.ndarray | None:
    """Get best available position: hip > shoulder > wrist."""
    for idx_a, idx_b in [(11, 12), (5, 6), (9, 10)]:
        a, b = kp[idx_a], kp[idx_b]
        if a.sum() > 0 and b.sum() > 0:
            return (a + b) / 2
        if a.sum() > 0:
            return a
        if b.sum() > 0:
            return b
    return None


def _norm_to_pixel(zone_norm: list, w: int, h: int) -> np.ndarray:
    pts = [[int(p[0] * w), int(p[1] * h)] for p in zone_norm if len(p) >= 2]
    return np.array(pts, dtype=np.int32)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0


def _min_wrist_distance(person_a: dict, person_b: dict) -> float:
    dists = []
    for wa in [person_a["left_wrist"], person_a["right_wrist"]]:
        for wb in [person_b["left_wrist"], person_b["right_wrist"]]:
            if wa.sum() > 0 and wb.sum() > 0:
                dists.append(float(np.linalg.norm(wa - wb)))
    return min(dists) if dists else float("inf")
