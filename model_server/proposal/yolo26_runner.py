"""Ultralytics-compatible YOLO26 pose/detect runner for HIO v3."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class YoloRuntimeError(RuntimeError):
    """Raised when a required YOLO inference step fails."""


@dataclass
class PosePerson:
    track_id: str
    bbox: list[float]
    confidence: float
    keypoints: list[list[float]]
    keypoint_conf: list[float]


@dataclass
class DetectionObject:
    label: str
    confidence: float
    bbox: list[float]
    source: str
    event_tag: str = "object"


@dataclass
class Yolo26FrameResult:
    people: list[PosePerson] = field(default_factory=list)
    objects: list[DetectionObject] = field(default_factory=list)
    semantic_scores: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    model_health: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "person_count": len(self.people),
            "objects": [
                {
                    "label": obj.label,
                    "confidence": round(float(obj.confidence), 3),
                    "bbox": [round(float(v), 1) for v in obj.bbox[:4]],
                    "source": obj.source,
                    "event_tag": obj.event_tag,
                }
                for obj in self.objects[:30]
            ],
            "semantic_scores": dict(self.semantic_scores),
            "errors": list(self.errors),
            "timings_ms": dict(self.timings_ms),
            "model_health": dict(self.model_health),
        }


class YOLO26Runner:
    """Thin wrapper around local YOLO26 .pt/.engine files."""

    def __init__(
        self,
        *,
        pose_weights: str,
        detect_weights: str,
        cash_weights: str,
        fire_weights: str,
        device: str = "cuda",
        imgsz: int = 640,
        pose_conf: float = 0.25,
        detect_conf: float = 0.25,
    ) -> None:
        self.device = device
        self.requested_device = device
        self.imgsz = int(imgsz or 640)
        self.pose_conf = float(pose_conf or 0.25)
        self.detect_conf = float(detect_conf or 0.25)
        self._track_next_id: dict[str, int] = {}
        self._track_state: dict[str, list[dict[str, Any]]] = {}
        self._YOLO = None
        self.model_health: dict[str, Any] = {
            "device": self.device,
            "requested_device": self.requested_device,
            "runtime": "initializing",
            "errors": [],
            "enabled_models": ["pose_tracking", "fire_smoke"],
        }
        self._runtime_error = self._resolve_runtime_device()
        self.pose_model = self._load_model(pose_weights, "pose")
        if str(detect_weights or "").strip():
            logger.info("[YOLO26Runner] generic detect weights configured but ignored; tier1 uses pose + fire/smoke only.")
        if str(cash_weights or "").strip():
            logger.info("[YOLO26Runner] cash detect weights configured but ignored; cash proposals use pose/exchange-band temporal signals.")
        self.detect_model = None
        self.cash_model = None
        self.fire_model = self._load_model(fire_weights, "fire")

    def _resolve_runtime_device(self) -> str:
        if not str(self.device).lower().startswith("cuda"):
            self.model_health["runtime"] = "ok"
            return ""
        try:
            from model_server import config
            import torch

            if torch.cuda.is_available():
                self.model_health["runtime"] = "ok"
                return ""
            msg = "CUDA requested for YOLO26 but torch.cuda.is_available() is false"
            if bool(getattr(config, "ALLOW_CPU_FALLBACK", False)):
                self.device = "cpu"
                self.model_health["device"] = self.device
                self.model_health["runtime"] = "cpu_fallback"
                self.model_health["errors"].append(msg)
                logger.warning("[YOLO26Runner] %s; CPU fallback enabled.", msg)
                return ""
            self.model_health["runtime"] = "error"
            self.model_health["errors"].append(msg)
            logger.error("[YOLO26Runner] %s; set ALLOW_CPU_FALLBACK=true to allow fallback.", msg)
            return msg
        except Exception as exc:
            msg = f"YOLO26 device check failed: {exc}"
            self.model_health["runtime"] = "error"
            self.model_health["errors"].append(msg)
            logger.error("[YOLO26Runner] %s", msg)
            return msg

    def _load_model(self, path_value: str, name: str) -> Any:
        if not str(path_value or "").strip():
            logger.info("[YOLO26Runner] %s model disabled (no weights configured).", name)
            return None
        path = Path(str(path_value or "")).expanduser()
        if not path.exists():
            logger.warning("[YOLO26Runner] %s weights missing: %s", name, path)
            return None
        try:
            if self._YOLO is None:
                from ultralytics import YOLO

                self._YOLO = YOLO
            model = self._YOLO(str(path))
            logger.info("[YOLO26Runner] loaded %s model: %s", name, path)
            return model
        except Exception as exc:
            msg = f"failed to load {name} model {path}: {exc}"
            self.model_health["errors"].append(msg)
            logger.warning("[YOLO26Runner] %s", msg)
            return None

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0,))
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                return value.numpy()
        except Exception:
            pass
        try:
            return np.asarray(value)
        except Exception:
            return np.empty((0,))

    @staticmethod
    def _classify_event_tag(label: str, source: str) -> str:
        text = str(label or "").lower().strip()
        if text in {"other", "background", "default", "none", "-1"}:
            return "object"
        if any(k in text for k in ("fire", "flame", "smoke")):
            if any(k in text for k in ("extinguisher", "sign", "screen", "monitor", "logo")):
                return "object"
            return "fire_smoke"
        if any(k in text for k in ("cash", "money", "bill", "banknote", "note", "currency", "wallet")):
            return "cash_like"
        if any(k in text for k in ("knife", "gun", "weapon", "fight", "violence", "punch", "kick")):
            return "violence_like"
        return "object"

    def _predict(self, model: Any, frame: Any, conf: float, source: str) -> list[Any]:
        if model is None:
            return []
        if self._runtime_error:
            raise YoloRuntimeError(self._runtime_error)
        try:
            return model.predict(
                frame,
                imgsz=self.imgsz,
                conf=float(conf),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            msg = f"{source} inference failed: {exc}"
            lowered = msg.lower()
            if "out of memory" in lowered or "cuda" in lowered:
                self.model_health["runtime"] = "degraded"
            self.model_health["errors"].append(msg)
            logger.warning("[YOLO26Runner] %s", msg)
            raise YoloRuntimeError(msg) from exc

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        if len(bbox) < 4:
            return (0.0, 0.0)
        return (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        if len(a) < 4 or len(b) < 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    def _assign_track_ids(self, camera_id: str, people: list[PosePerson]) -> list[PosePerson]:
        cam = str(camera_id or "default")
        previous = list(self._track_state.get(cam, []))
        used_previous: set[int] = set()
        next_id = int(self._track_next_id.get(cam, 1) or 1)

        for person in people:
            center = self._bbox_center(person.bbox)
            best_idx = -1
            best_score = 0.0
            for idx, row in enumerate(previous):
                if idx in used_previous:
                    continue
                prev_bbox = list(row.get("bbox") or [])
                prev_center = row.get("center") or self._bbox_center(prev_bbox)
                iou = self._bbox_iou(person.bbox, prev_bbox)
                dist = (
                    (center[0] - float(prev_center[0])) ** 2
                    + (center[1] - float(prev_center[1])) ** 2
                ) ** 0.5
                dist_score = max(0.0, 1.0 - dist / 180.0)
                score = max(iou, dist_score * 0.65)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0 and best_score >= 0.20:
                person.track_id = str(previous[best_idx].get("track_id") or f"t{next_id}")
                used_previous.add(best_idx)
            else:
                person.track_id = f"t{next_id}"
                next_id += 1

        self._track_next_id[cam] = next_id
        self._track_state[cam] = [
            {
                "track_id": person.track_id,
                "bbox": list(person.bbox),
                "center": self._bbox_center(person.bbox),
            }
            for person in people
            if person.bbox
        ][:32]
        return people

    def _run_pose(self, camera_id: str, frame: Any) -> tuple[list[PosePerson], list[str]]:
        errors: list[str] = []
        people: list[PosePerson] = []
        try:
            results = self._predict(self.pose_model, frame, self.pose_conf, "pose")
        except YoloRuntimeError as exc:
            errors.append(str(exc))
            return people, errors
        for result in results:
            boxes_xyxy = self._as_numpy(getattr(getattr(result, "boxes", None), "xyxy", None))
            boxes_conf = self._as_numpy(getattr(getattr(result, "boxes", None), "conf", None))
            keypoints = getattr(result, "keypoints", None)
            kxy = self._as_numpy(getattr(keypoints, "xy", None))
            kconf = self._as_numpy(getattr(keypoints, "conf", None))
            if kxy.ndim != 3:
                continue
            for idx in range(kxy.shape[0]):
                bbox = boxes_xyxy[idx].astype(float).tolist() if idx < len(boxes_xyxy) else []
                conf = float(boxes_conf[idx]) if idx < len(boxes_conf) else 0.0
                kp_conf = kconf[idx].astype(float).tolist() if kconf.ndim == 2 and idx < len(kconf) else []
                people.append(
                    PosePerson(
                        track_id=f"p{len(people) + 1}",
                        bbox=bbox,
                        confidence=conf,
                        keypoints=kxy[idx].astype(float).tolist(),
                        keypoint_conf=kp_conf,
                    )
                )
        people = self._assign_track_ids(camera_id, people)
        return people, errors

    def _run_detect_model(self, model: Any, frame: Any, source: str) -> list[DetectionObject]:
        objects: list[DetectionObject] = []
        try:
            results = self._predict(model, frame, self.detect_conf, source)
        except YoloRuntimeError:
            raise
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            xyxy = self._as_numpy(getattr(boxes, "xyxy", None))
            conf = self._as_numpy(getattr(boxes, "conf", None))
            cls = self._as_numpy(getattr(boxes, "cls", None))
            if xyxy.ndim != 2:
                continue
            for idx, bbox in enumerate(xyxy):
                cls_id = int(cls[idx]) if idx < len(cls) else -1
                label = str(names.get(cls_id, cls_id))
                score = float(conf[idx]) if idx < len(conf) else 0.0
                objects.append(
                    DetectionObject(
                        label=label,
                        confidence=score,
                        bbox=bbox.astype(float).tolist(),
                        source=source,
                        event_tag=self._classify_event_tag(label, source),
                    )
                )
        return objects

    def process_frame(self, camera_id: str, frame: Any) -> Yolo26FrameResult:
        result = Yolo26FrameResult()
        people, errors = self._run_pose(camera_id, frame)
        result.people = people
        result.errors.extend(errors)
        try:
            result.objects.extend(self._run_detect_model(self.fire_model, frame, "fire_smoke"))
        except YoloRuntimeError as exc:
            result.errors.append(str(exc))
        result.model_health = dict(self.model_health)
        return result
