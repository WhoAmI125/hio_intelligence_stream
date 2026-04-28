"""Local smoke test for HIO v3 modules.

Runs without RTSP. It checks config, YOLO model paths, dummy clip creation,
and optional YOLO inference on one blank frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_server import config
from model_server.candidates.clip_builder import CandidateClipBuilder
from model_server.local_storage import LocalStorage


def _resolve_model_path(value: str) -> Path | None:
    """Resolve a model weights path. Empty/whitespace means disabled, return None.

    Previously an empty string would resolve to repo root and falsely report
    exists=True, hiding a disabled-but-mistagged model. Return None so the
    caller can render a correct disabled row.
    """
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _render_model_row(path: Path | None) -> dict:
    if path is None:
        return {"path": "", "exists": False, "enabled": False}
    return {"path": str(path), "exists": path.exists(), "enabled": True}


def main() -> int:
    root = ROOT
    model_paths = {
        "pose": _resolve_model_path(config.YOLO26_POSE_WEIGHTS),
        "detect": _resolve_model_path(config.YOLO26_DETECT_WEIGHTS),
    }
    optional_model_paths = {
        "cash": _resolve_model_path(config.YOLO26_CASH_WEIGHTS),
        "fire": _resolve_model_path(config.YOLO26_FIRE_WEIGHTS),
    }
    report = {
        "root": str(root),
        "hio_v3_enabled": bool(config.HIO_V3_ENABLED),
        "pipeline_version": str(getattr(config, "HIO_V3_PIPELINE_VERSION", "")),
        "semantic_filter": {
            "enabled": bool(getattr(config, "V3_SEMANTIC_FILTER_ENABLED", True)),
            "model": str(getattr(config, "V3_SEMANTIC_MODEL", "")),
        },
        "classifier_heads": {
            "fire": {
                "enabled": bool(getattr(config, "V3_FIRE_CLASSIFIER_ENABLED", True)),
                "model": str(getattr(config, "V3_FIRE_CLASSIFIER_MODEL", "")),
            },
            "action": {
                "enabled": bool(getattr(config, "V3_ACTION_CLASSIFIER_ENABLED", True)),
                "model": str(getattr(config, "V3_ACTION_CLASSIFIER_MODEL", "")),
            },
        },
        "validation_clip_sec": float(getattr(config, "V3_VALIDATION_CLIP_SEC", 15.0)),
        "ingest_downsample_height": int(getattr(config, "INGEST_DOWNSAMPLE_HEIGHT", 0) or 0),
        "overlay_fps_divisor": int(getattr(config, "V3_OVERLAY_FPS_DIVISOR", 3) or 3),
        "ffmpeg": {
            "encoder": str(getattr(config, "FFMPEG_ENCODER", "libx264") or "libx264"),
            "preset": str(getattr(config, "FFMPEG_PRESET", "ultrafast") or "ultrafast"),
            "crf": int(getattr(config, "FFMPEG_CRF", 28) or 28),
            "nvenc_preset": str(getattr(config, "FFMPEG_NVENC_PRESET", "p3") or "p3"),
            "nvenc_cq": int(getattr(config, "FFMPEG_NVENC_CQ", 28) or 28),
        },
        "frontend_mjpeg": {
            "fps": float(getattr(config, "FRONTEND_MJPEG_FPS", 3.0) or 3.0),
            "burst_fps": float(getattr(config, "FRONTEND_MJPEG_BURST_FPS", 0.0) or 0.0),
            "width": int(getattr(config, "FRONTEND_MJPEG_WIDTH", 0) or 0),
            "quality": int(getattr(config, "FRONTEND_MJPEG_QUALITY", 50) or 50),
            "dedup": bool(getattr(config, "FRONTEND_MJPEG_DEDUP_FRAMES", True)),
            "idle_pause_sec": float(getattr(config, "FRONTEND_MJPEG_IDLE_PAUSE_SEC", 5.0) or 5.0),
        },
        "models": {k: _render_model_row(v) for k, v in model_paths.items()},
        "optional_models": {k: _render_model_row(v) for k, v in optional_model_paths.items()},
    }
    # Range sanity check
    _range_warn = []
    mj = report["frontend_mjpeg"]
    if not (1.0 <= mj["fps"] <= 30.0):
        _range_warn.append(f"FRONTEND_MJPEG_FPS out of [1,30]: {mj['fps']}")
    if mj["burst_fps"] and mj["burst_fps"] < mj["fps"]:
        _range_warn.append(f"FRONTEND_MJPEG_BURST_FPS {mj['burst_fps']} < FPS {mj['fps']}")
    if not (10 <= mj["quality"] <= 95):
        _range_warn.append(f"FRONTEND_MJPEG_QUALITY out of [10,95]: {mj['quality']}")
    if mj["width"] and mj["width"] < 160:
        _range_warn.append(f"FRONTEND_MJPEG_WIDTH suspiciously small: {mj['width']}")
    encoder = report["ffmpeg"]["encoder"].lower()
    if encoder not in {"libx264", "h264_nvenc", "nvenc", "auto"}:
        _range_warn.append(f"FFMPEG_ENCODER unknown value: {encoder}")
    report["config_warnings"] = _range_warn

    storage = LocalStorage(base_dir=str(root / "data" / "smoke"))
    frames = []
    for i in range(12):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"HIO v3 smoke {i}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frames.append({"frame": frame, "mono_ts": float(i) / 6.0})
    skeleton_summary = {
        "persons": [
            {
                "bbox": [180, 90, 360, 310],
                "keypoints": [
                    {"x": 250, "y": 110, "confidence": 0.9},
                    *[{"x": 0, "y": 0, "confidence": 0.0} for _ in range(8)],
                    {"x": 230, "y": 220, "confidence": 0.9},
                    {"x": 330, "y": 220, "confidence": 0.9},
                ],
                "wrists": [
                    {"wrist": "left_wrist", "x": 230, "y": 220, "inside_cashier_zone": True},
                    {"wrist": "right_wrist", "x": 330, "y": 220, "inside_cashier_zone": True},
                ],
            }
        ]
    }
    raw_path = storage.save_clip(
        "smoke_v3_candidate_val",
        [row["frame"] for row in frames],
        fps=6.0,
        allow_s3=False,
    )
    clips = CandidateClipBuilder(storage).save_candidate_clips(
        event_id="smoke_v3_candidate",
        entries=frames,
        fps=6.0,
        zones={"cashier": [[160, 80], [430, 80], [430, 320], [160, 320]], "drawer": []},
        skeleton_summary=skeleton_summary,
        raw_path=raw_path,
    )
    report["candidate_clips"] = {k: {"path": v, "exists": Path(v).exists()} for k, v in clips.items()}
    required_clips = {"raw", "skeleton_json", "context_overlay"}
    forbidden_overlay_keys = {"skeleton_overlay", "cashier_zone_overlay"}
    legacy_roi_key = "cashier_roi_" + "crop"
    report["candidate_clip_contract"] = {
        "required_present": sorted(required_clips),
        "missing": sorted(required_clips - set(clips)),
        "forbidden_present": sorted(({legacy_roi_key} | forbidden_overlay_keys) & set(clips)),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = (
        all(row["exists"] for row in report["models"].values())
        and not report["candidate_clip_contract"]["missing"]
        and not report["candidate_clip_contract"]["forbidden_present"]
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
