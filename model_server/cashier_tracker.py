"""
CashierTracker — simple per-camera state for identifying cashier vs customer
and emitting a cash trigger when a customer wrist gets close to a cashier wrist.

Identification rules (intentionally minimal, no full tracking):
    - "Cashier" = a person whose ANY wrist has been inside cashier_zone
      for at least CASHIER_WATCH_SECONDS of observations in the recent window.
    - "Customer" = any other person present in the frame.

Trigger rule:
    - For any (cashier wrist, customer wrist) pair with Euclidean distance
      <= HAND_PROXIMITY_PX, emit a trigger.
    - Global cooldown CASH_TRIGGER_COOLDOWN_SEC suppresses re-triggers.

This is simpler than multi-person ID tracking: we treat each frame mostly
independently and rely on a sliding window of wrist-in-zone observations
to decide which bbox is the cashier. That works for hotel reception where
staff stays behind the counter.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters.yolo_pose_adapter import PersonDetection, KP_LEFT_WRIST, KP_RIGHT_WRIST

logger = logging.getLogger(__name__)


def _point_in_polygon(px: float, py: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon (counts boundary as inside)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _denormalize(polygon_norm: Sequence[Sequence[float]], w: int, h: int) -> List[Tuple[float, float]]:
    """Convert polygon coords from [0, 1] normalized space to pixel space."""
    return [(float(p[0]) * w, float(p[1]) * h) for p in polygon_norm]


def _as_pixel_polygon(polygon: Sequence[Sequence[float]], w: int, h: int) -> List[Tuple[float, float]]:
    """
    Accept polygon in either pixel or normalized coords. Heuristic: if any
    coordinate exceeds 1.5, assume pixel; otherwise assume [0, 1] normalized.
    """
    if not polygon:
        return []
    flat = [float(v) for p in polygon for v in (p[0], p[1])]
    mx = max(flat) if flat else 0.0
    if mx <= 1.5:
        return [(float(p[0]) * w, float(p[1]) * h) for p in polygon]
    return [(float(p[0]), float(p[1])) for p in polygon]


@dataclass
class CashierTriggerEvent:
    camera_id: str
    trigger_ts: float        # time.time() at trigger
    mono_ts: float           # monotonic clock at trigger (ring-buffer alignment)
    proximity_px: float      # distance between cashier wrist and customer wrist
    cashier_wrist: Tuple[float, float]
    customer_wrist: Tuple[float, float]
    cashier_bbox: Tuple[float, float, float, float]
    customer_bbox: Tuple[float, float, float, float]
    num_persons: int
    reason: str = "cashier_customer_wrist_proximity"

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "trigger_ts": self.trigger_ts,
            "proximity_px": round(self.proximity_px, 2),
            "cashier_wrist": list(self.cashier_wrist),
            "customer_wrist": list(self.customer_wrist),
            "cashier_bbox": list(self.cashier_bbox),
            "customer_bbox": list(self.customer_bbox),
            "num_persons": self.num_persons,
            "reason": self.reason,
        }


@dataclass
class _PersonTimeline:
    """Rolling history of per-bbox wrist-in-zone flags, keyed by bbox index."""
    wrist_in_zone_count: int = 0
    total_observations: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class CashierTracker:
    """
    Per-camera tracker. Call update() on every pose inference result.

    Thread-safety: internal lock so multiple worker threads calling from different
    cameras cannot corrupt state (in practice we shard by camera_id).
    """

    def __init__(
        self,
        camera_id: str,
        cashier_zone_norm: Optional[Sequence[Sequence[float]]] = None,
        drawer_zone_norm: Optional[Sequence[Sequence[float]]] = None,
        # Accept either normalized or pixel-space polygons; auto-detected.
        cashier_watch_seconds: float = 10.0,
        cashier_watch_min_obs: int = 8,
        wrist_conf_threshold: float = 0.3,
        hand_proximity_px: float = 120.0,
        cooldown_sec: float = 20.0,
        # If a specific bbox-cluster has had a wrist continuously inside the
        # cashier_zone for more than max_linger_sec, reclassify that person as
        # customer (they are lingering at the counter filling out forms, signing,
        # etc.). Staff's wrist typically dips in and out of the zone; a customer
        # who stays planted is the one this threshold catches.
        max_linger_sec: float = 30.0,
        linger_gap_tolerance_sec: float = 3.0,
        linger_cluster_px: float = 120.0,
    ):
        self.camera_id = str(camera_id)
        self.cashier_zone_norm = list(cashier_zone_norm or [])
        self.drawer_zone_norm = list(drawer_zone_norm or [])
        self.cashier_watch_seconds = float(cashier_watch_seconds)
        self.cashier_watch_min_obs = int(cashier_watch_min_obs)
        self.wrist_conf_threshold = float(wrist_conf_threshold)
        self.hand_proximity_px = float(hand_proximity_px)
        self.cooldown_sec = float(cooldown_sec)
        self.max_linger_sec = float(max_linger_sec)
        self.linger_gap_tolerance_sec = float(linger_gap_tolerance_sec)
        self.linger_cluster_px = float(linger_cluster_px)

        self._lock = threading.Lock()
        # History of (ts, person_idx, wrist_in_zone_bool) for building rolling stats.
        # Since we don't keep persistent IDs, we use bbox center clustering per frame.
        self._history: Deque[Tuple[float, Tuple[float, float], bool]] = deque(maxlen=500)
        self._last_trigger_ts: float = 0.0

    # ------------------------------------------------------------------
    # Config update (called when zones change in UI)
    # ------------------------------------------------------------------
    def update_zones(
        self,
        cashier_zone_norm: Optional[Sequence[Sequence[float]]] = None,
        drawer_zone_norm: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        with self._lock:
            if cashier_zone_norm is not None:
                self.cashier_zone_norm = list(cashier_zone_norm)
            if drawer_zone_norm is not None:
                self.drawer_zone_norm = list(drawer_zone_norm)
            self._history.clear()

    def has_zone(self) -> bool:
        return bool(self.cashier_zone_norm)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._last_trigger_ts = 0.0

    # ------------------------------------------------------------------
    # Linger measurement (used to reclassify long-stay persons as customer)
    # ------------------------------------------------------------------
    def _continuous_in_zone_streak(
        self,
        center: Tuple[float, float],
        now_ts: float,
    ) -> float:
        """
        Return the length (seconds) of the ongoing continuous wrist-in-zone
        streak for a bbox cluster near `center`. A "continuous" streak means
        the spatial cluster has produced wrist-in-zone=True observations with
        no inter-observation gap larger than linger_gap_tolerance_sec.

        The streak ends (becomes 0) when an out-of-zone observation from the
        same cluster is recorded, or when the cluster has no recent obs at all.
        """
        if not self._history:
            return 0.0

        cluster_r = float(self.linger_cluster_px)
        gap_tol = float(self.linger_gap_tolerance_sec)

        # Walk backwards through history, collecting obs whose center is within
        # cluster_r of this detection's center. Streak ends on any out-of-zone
        # obs of the cluster, or on a time gap exceeding gap_tol.
        streak_start: Optional[float] = None
        last_ts: Optional[float] = None
        for t, c, hz in reversed(self._history):
            dx = c[0] - center[0]
            dy = c[1] - center[1]
            if math.hypot(dx, dy) > cluster_r:
                continue
            if last_ts is not None and (last_ts - t) > gap_tol:
                break  # time gap too large → streak ended earlier
            if not hz:
                break  # cluster went out of zone → streak broken
            streak_start = t
            last_ts = t

        if streak_start is None:
            return 0.0
        return max(0.0, now_ts - streak_start)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(
        self,
        detections: Sequence[PersonDetection],
        frame_shape: Tuple[int, int, int],
        now_ts: Optional[float] = None,
        now_mono: Optional[float] = None,
    ) -> Optional[CashierTriggerEvent]:
        """
        Process one frame's pose detections. Returns a CashierTriggerEvent if a
        new trigger fires, else None.
        """
        if not self.cashier_zone_norm:
            return None
        if not detections:
            return None

        h, w = frame_shape[:2]
        zone_px = _as_pixel_polygon(self.cashier_zone_norm, w, h)
        drawer_px = _as_pixel_polygon(self.drawer_zone_norm, w, h) if self.drawer_zone_norm else []

        now_ts = float(now_ts or time.time())
        now_mono = float(now_mono or time.monotonic())

        with self._lock:
            # 1) Compute per-person wrist-in-zone status for this frame.
            per_person = []  # (bbox_center, wrist_points_in_zone_bool, wrist_points_list)
            for det in detections:
                wrist_pts = det.wrist_points(self.wrist_conf_threshold)
                any_wrist_in = False
                for (wx, wy) in wrist_pts:
                    if _point_in_polygon(wx, wy, zone_px):
                        any_wrist_in = True
                        break
                cx = (det.bbox[0] + det.bbox[2]) / 2.0
                cy = (det.bbox[1] + det.bbox[3]) / 2.0
                per_person.append((det, (cx, cy), any_wrist_in, wrist_pts))

            # 2) Record observations into rolling history (time-bounded).
            cutoff = now_ts - self.cashier_watch_seconds
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()
            for det, center, in_zone, _ in per_person:
                self._history.append((now_ts, center, in_zone))

            # 3) Compute per-person continuous wrist-in-zone streak. If a
            #    spatial cluster near this person's bbox center has had
            #    wrist-in-zone observations for > max_linger_sec continuously
            #    (no gap larger than linger_gap_tolerance_sec), force that
            #    person to "customer". We also collect the bbox centers of such
            #    lingering clusters so we can EXCLUDE them from the cashier
            #    anchor computation — otherwise a planted customer pulls the
            #    anchor onto themselves and swaps roles with the real cashier.
            force_customer_flags: List[bool] = []
            lingering_centers: List[Tuple[float, float]] = []
            for det, center, _in_zone, _ in per_person:
                streak_sec = self._continuous_in_zone_streak(center, now_ts)
                is_lingering = streak_sec > self.max_linger_sec
                force_customer_flags.append(is_lingering)
                if is_lingering:
                    lingering_centers.append(center)

            def _near_any_lingering(c: Tuple[float, float]) -> bool:
                for lc in lingering_centers:
                    if math.hypot(c[0] - lc[0], c[1] - lc[1]) <= self.linger_cluster_px:
                        return True
                return False

            # 4) Build cashier anchor from in-zone history, EXCLUDING obs that
            #    belong to any lingering cluster.
            in_zone_history = [
                c for (t, c, hz) in self._history
                if hz and not _near_any_lingering(c)
            ]
            cashier_anchor = None
            if len(in_zone_history) >= self.cashier_watch_min_obs:
                xs = np.array([c[0] for c in in_zone_history])
                ys = np.array([c[1] for c in in_zone_history])
                cashier_anchor = (float(np.median(xs)), float(np.median(ys)))

            cashiers: List[PersonDetection] = []
            customers: List[PersonDetection] = []

            if cashier_anchor is None:
                # Not enough history yet: anyone with wrist in zone *right now*
                # provisionally counts as cashier (still allows trigger once a
                # second person overlaps, though history-based anchor is preferred).
                for (det, _center, in_zone, _), force_cust in zip(per_person, force_customer_flags):
                    if force_cust:
                        customers.append(det)
                    else:
                        (cashiers if in_zone else customers).append(det)
            else:
                ax, ay = cashier_anchor
                for (det, center, in_zone, _), force_cust in zip(per_person, force_customer_flags):
                    if force_cust:
                        customers.append(det)
                        continue
                    dx = center[0] - ax
                    dy = center[1] - ay
                    dist = math.hypot(dx, dy)
                    # Match radius tuned for reception-camera perspective:
                    # use horizontal bbox width (stable across perspective) and
                    # clamp to a tight range so that two nearby persons are
                    # resolved as separate roles.
                    bbox_width = det.bbox[2] - det.bbox[0]
                    match_radius = max(40.0, min(bbox_width * 0.5, 200.0))
                    if dist <= match_radius:
                        cashiers.append(det)
                    else:
                        customers.append(det)

            if not cashiers or not customers:
                return None

            # 4) Cooldown check.
            if (now_ts - self._last_trigger_ts) < self.cooldown_sec:
                return None

            # 5) Compute minimum cashier_wrist ↔ customer_wrist distance.
            best: Optional[Tuple[float, Tuple[float, float], Tuple[float, float], PersonDetection, PersonDetection]] = None
            for c_det in cashiers:
                c_wrists = c_det.wrist_points(self.wrist_conf_threshold)
                if not c_wrists:
                    continue
                for k_det in customers:
                    k_wrists = k_det.wrist_points(self.wrist_conf_threshold)
                    if not k_wrists:
                        continue
                    for cw in c_wrists:
                        for kw in k_wrists:
                            d = math.hypot(cw[0] - kw[0], cw[1] - kw[1])
                            if best is None or d < best[0]:
                                best = (d, cw, kw, c_det, k_det)

            if best is None:
                return None

            dist, cw, kw, c_det, k_det = best
            if dist > self.hand_proximity_px:
                return None

            self._last_trigger_ts = now_ts
            return CashierTriggerEvent(
                camera_id=self.camera_id,
                trigger_ts=now_ts,
                mono_ts=now_mono,
                proximity_px=dist,
                cashier_wrist=cw,
                customer_wrist=kw,
                cashier_bbox=c_det.bbox,
                customer_bbox=k_det.bbox,
                num_persons=len(detections),
            )
