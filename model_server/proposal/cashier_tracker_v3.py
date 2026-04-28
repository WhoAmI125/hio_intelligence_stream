"""Cashier/customer role tracker for v3 pose results."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from model_server.skeleton.pose_features import LEFT_WRIST, RIGHT_WRIST, point_in_polygon


class CashierTrackerV3:
    """Infer cashier/customer roles from wrist-in-cashier-zone history.

    The tracker does not decide the final event. It only adds role-aware
    features to the v3 cash prefilter so a customer lingering at the counter is
    less likely to be treated as staff.
    """

    def __init__(
        self,
        *,
        watch_seconds: float = 10.0,
        min_observations: int = 6,
        wrist_conf_threshold: float = 0.25,
        hand_proximity_px: float = 120.0,
        max_linger_sec: float = 30.0,
        linger_gap_tolerance_sec: float = 3.0,
        linger_cluster_px: float = 120.0,
    ) -> None:
        self.watch_seconds = float(watch_seconds)
        self.min_observations = int(min_observations)
        self.wrist_conf_threshold = float(wrist_conf_threshold)
        self.hand_proximity_px = float(hand_proximity_px)
        self.max_linger_sec = float(max_linger_sec)
        self.linger_gap_tolerance_sec = float(linger_gap_tolerance_sec)
        self.linger_cluster_px = float(linger_cluster_px)
        self._history: dict[str, deque[tuple[float, tuple[float, float], bool]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _center(person: Any) -> tuple[float, float]:
        bbox = list(getattr(person, "bbox", []) or [])
        if len(bbox) < 4:
            return (0.0, 0.0)
        return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)

    def _wrists(self, person: Any) -> list[tuple[float, float]]:
        keypoints = list(getattr(person, "keypoints", []) or [])
        confs = list(getattr(person, "keypoint_conf", []) or [])
        out: list[tuple[float, float]] = []
        for idx in (LEFT_WRIST, RIGHT_WRIST):
            if idx >= len(keypoints):
                continue
            conf = float(confs[idx]) if idx < len(confs) else 0.0
            if conf < self.wrist_conf_threshold:
                continue
            xy = keypoints[idx]
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                out.append((float(xy[0]), float(xy[1])))
        return out

    def _continuous_in_zone_streak(
        self,
        history: deque[tuple[float, tuple[float, float], bool]],
        center: tuple[float, float],
        now_ts: float,
    ) -> float:
        streak_start = None
        last_ts = None
        for ts, hist_center, in_zone in reversed(history):
            if math.hypot(hist_center[0] - center[0], hist_center[1] - center[1]) > self.linger_cluster_px:
                continue
            if last_ts is not None and (last_ts - ts) > self.linger_gap_tolerance_sec:
                break
            if not in_zone:
                break
            streak_start = ts
            last_ts = ts
        if streak_start is None:
            return 0.0
        return max(0.0, now_ts - streak_start)

    def summarize(
        self,
        *,
        camera_id: str,
        people: list[Any],
        cashier_zone: list[Any],
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        if not people or len(cashier_zone) < 3:
            return {"triggered": False, "reason": "no_people_or_cashier_zone"}

        now_ts = float(now_ts or time.time())
        with self._lock:
            history = self._history.setdefault(str(camera_id), deque(maxlen=600))
            cutoff = now_ts - self.watch_seconds
            while history and history[0][0] < cutoff:
                history.popleft()

            per_person = []
            for person in people:
                center = self._center(person)
                wrists = self._wrists(person)
                in_zone = any(point_in_polygon(wrist, cashier_zone) for wrist in wrists)
                per_person.append({"person": person, "center": center, "wrists": wrists, "in_zone": in_zone})
                history.append((now_ts, center, in_zone))

            lingering_centers = []
            for row in per_person:
                streak = self._continuous_in_zone_streak(history, row["center"], now_ts)
                row["linger_sec"] = round(streak, 2)
                row["force_customer"] = streak > self.max_linger_sec
                if row["force_customer"]:
                    lingering_centers.append(row["center"])

            def near_lingering(center: tuple[float, float]) -> bool:
                return any(math.hypot(center[0] - c[0], center[1] - c[1]) <= self.linger_cluster_px for c in lingering_centers)

            in_zone_history = [c for _, c, in_zone in history if in_zone and not near_lingering(c)]
            cashier_anchor = None
            if len(in_zone_history) >= self.min_observations:
                cashier_anchor = (
                    float(np.median([c[0] for c in in_zone_history])),
                    float(np.median([c[1] for c in in_zone_history])),
                )

            cashiers = []
            customers = []
            for row in per_person:
                if row["force_customer"]:
                    customers.append(row)
                    continue
                if cashier_anchor is None:
                    (cashiers if row["in_zone"] else customers).append(row)
                    continue
                bbox = list(getattr(row["person"], "bbox", []) or [])
                bbox_width = max(1.0, float(bbox[2]) - float(bbox[0])) if len(bbox) >= 4 else 80.0
                match_radius = max(40.0, min(bbox_width * 0.5, 200.0))
                dist = math.hypot(row["center"][0] - cashier_anchor[0], row["center"][1] - cashier_anchor[1])
                (cashiers if dist <= match_radius else customers).append(row)

            best_dist = None
            for cashier in cashiers:
                for customer in customers:
                    for cw in cashier["wrists"]:
                        for kw in customer["wrists"]:
                            dist = math.hypot(cw[0] - kw[0], cw[1] - kw[1])
                            best_dist = dist if best_dist is None else min(best_dist, dist)

            triggered = best_dist is not None and best_dist <= self.hand_proximity_px
            return {
                "triggered": bool(triggered),
                "cashier_count": len(cashiers),
                "customer_count": len(customers),
                "cashier_anchor": [round(cashier_anchor[0], 1), round(cashier_anchor[1], 1)] if cashier_anchor else None,
                "min_cashier_customer_wrist_distance_px": round(float(best_dist), 1) if best_dist is not None else None,
                "lingering_customer_count": sum(1 for row in per_person if row.get("force_customer")),
                "reason": "cashier_customer_wrist_proximity" if triggered else "role_context_only",
            }
