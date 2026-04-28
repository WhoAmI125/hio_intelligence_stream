"""Pose-derived feature helpers for the HIO v3 temporal engine."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_LIMBS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

LEFT_WRIST = 9
RIGHT_WRIST = 10


def point_in_polygon(point: tuple[float, float], polygon: list[Any]) -> bool:
    """Return True when a pixel-space point is inside a polygon."""
    if not polygon or len(polygon) < 3:
        return False
    try:
        arr = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(arr, point, False) >= 0
    except Exception:
        return False


def bbox_center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0)
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _bbox_area(bbox: list[float] | tuple[float, ...]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(union, 1e-6)


def _point_in_expanded_bbox(point: tuple[float, float], bbox: list[float], margin_ratio: float = 0.15) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    mx = w * margin_ratio
    my = h * margin_ratio
    x, y = point
    return (x1 - mx) <= x <= (x2 + mx) and (y1 - my) <= y <= (y2 + my)


def bbox_intersects_polygon(bbox: list[float] | tuple[float, ...], polygon: list[Any]) -> bool:
    """Cheap object-in-ROI test using center and box corners."""
    if not bbox or len(bbox) < 4 or not polygon or len(polygon) < 3:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    checks = [
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    return any(point_in_polygon(pt, polygon) for pt in checks)


def summarize_skeletons(people: list[Any], zones: dict[str, list[Any]]) -> dict[str, Any]:
    """Summarize pose persons into Gemini-friendly, JSON-serializable facts."""
    cashier_zone = zones.get("cashier") or []
    persons: list[dict[str, Any]] = []
    wrist_events: list[dict[str, Any]] = []

    for idx, person in enumerate(people):
        person_id = str(getattr(person, "track_id", None) or f"p{idx + 1}")
        bbox = list(getattr(person, "bbox", []) or [])
        keypoints = list(getattr(person, "keypoints", []) or [])
        keypoint_conf = list(getattr(person, "keypoint_conf", []) or [])
        kp_rows: list[dict[str, Any]] = []
        wrist_rows: list[dict[str, Any]] = []

        for kp_idx, xy in enumerate(keypoints[: len(COCO_KEYPOINTS)]):
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            conf = float(keypoint_conf[kp_idx]) if kp_idx < len(keypoint_conf) else 0.0
            x = float(xy[0])
            y = float(xy[1])
            row = {
                "name": COCO_KEYPOINTS[kp_idx],
                "x": round(x, 1),
                "y": round(y, 1),
                "confidence": round(conf, 3),
            }
            kp_rows.append(row)
            if kp_idx in {LEFT_WRIST, RIGHT_WRIST} and conf >= 0.15:
                wrist = {
                    "person_id": person_id,
                    "wrist": COCO_KEYPOINTS[kp_idx],
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "confidence": round(conf, 3),
                    "inside_cashier_zone": point_in_polygon((x, y), cashier_zone),
                }
                wrist_rows.append(wrist)
                wrist_events.append(wrist)

        persons.append(
            {
                "person_id": person_id,
                "bbox": [round(float(v), 1) for v in bbox[:4]],
                "confidence": round(float(getattr(person, "confidence", 0.0) or 0.0), 3),
                "keypoints": kp_rows,
                "wrists": wrist_rows,
            }
        )

    close_person_pairs: list[dict[str, Any]] = []
    overlapping_person_pairs: list[dict[str, Any]] = []
    cross_person_wrist_pairs: list[dict[str, Any]] = []
    min_center_distance = None

    for i, a in enumerate(persons):
        bbox_a = a.get("bbox") or []
        center_a = bbox_center(bbox_a)
        for j in range(i + 1, len(persons)):
            b = persons[j]
            bbox_b = b.get("bbox") or []
            center_b = bbox_center(bbox_b)
            dx = center_a[0] - center_b[0]
            dy = center_a[1] - center_b[1]
            dist = (dx * dx + dy * dy) ** 0.5
            min_center_distance = dist if min_center_distance is None else min(min_center_distance, dist)

            h_a = max(1.0, float(bbox_a[3]) - float(bbox_a[1])) if len(bbox_a) >= 4 else 1.0
            h_b = max(1.0, float(bbox_b[3]) - float(bbox_b[1])) if len(bbox_b) >= 4 else 1.0
            iou = _bbox_iou(bbox_a, bbox_b)
            close = dist <= (0.75 * ((h_a + h_b) / 2.0)) or iou >= 0.03
            if close:
                close_person_pairs.append(
                    {
                        "a": a.get("person_id"),
                        "b": b.get("person_id"),
                        "center_distance_px": round(dist, 1),
                        "bbox_iou": round(iou, 3),
                    }
                )
            if iou >= 0.03:
                overlapping_person_pairs.append(
                    {"a": a.get("person_id"), "b": b.get("person_id"), "bbox_iou": round(iou, 3)}
                )

            for wrist in a.get("wrists") or []:
                if _point_in_expanded_bbox((float(wrist["x"]), float(wrist["y"])), bbox_b):
                    cross_person_wrist_pairs.append(
                        {"from": a.get("person_id"), "to": b.get("person_id"), "wrist": wrist.get("wrist")}
                    )
            for wrist in b.get("wrists") or []:
                if _point_in_expanded_bbox((float(wrist["x"]), float(wrist["y"])), bbox_a):
                    cross_person_wrist_pairs.append(
                        {"from": b.get("person_id"), "to": a.get("person_id"), "wrist": wrist.get("wrist")}
                    )

    return {
        "person_count": len(persons),
        "persons": persons,
        "wrist_events": wrist_events,
        "any_wrist_in_cashier_zone": any(bool(w.get("inside_cashier_zone")) for w in wrist_events),
        "close_person_pair_count": len(close_person_pairs),
        "overlapping_person_pair_count": len(overlapping_person_pairs),
        "cross_person_wrist_pair_count": len(cross_person_wrist_pairs),
        "min_person_center_distance_px": round(float(min_center_distance), 1) if min_center_distance is not None else None,
        "close_person_pairs": close_person_pairs[:12],
        "cross_person_wrist_pairs": cross_person_wrist_pairs[:12],
    }
