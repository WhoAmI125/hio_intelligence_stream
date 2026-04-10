"""Event Pipeline — save clips first, then Tier 3 verify + store + alert."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config
from event.alert_sender import send_alert
from storage import db
from stream.clip_extractor import save_clip_mp4

logger = logging.getLogger(__name__)


def _draw_cashier_zone(frames: list[np.ndarray], zone_norm: list) -> list[np.ndarray]:
    """Draw cashier zone polygon border on frames as visual hint for VLM.

    Returns new frames (copies) — does not modify originals.
    """
    if not zone_norm or len(zone_norm) < 3:
        return frames

    result = []
    for frame in frames:
        f = frame.copy()
        h, w = f.shape[:2]
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone_norm], dtype=np.int32)
        cv2.polylines(f, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
        cx, cy = pts[0][0], max(pts[0][1] - 8, 16)
        cv2.putText(f, "CASHIER ZONE", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        result.append(f)
    return result


async def save_clips(
    cam_id: str,
    scenario: str,
    clip_frames: list[np.ndarray],
    cam_config: dict,
) -> tuple[str, str | None, str, str | None]:
    """Save full clip + zone clip + thumbnails to disk.

    Returns: (full_path, zone_path, thumb_full, thumb_zone)
    """
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    clip_dir = config.DATA_DIR / "clips" / date_str
    clip_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = config.DATA_DIR / "thumbnails" / date_str
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Full clip
    full_path = str(clip_dir / f"{cam_id}_{scenario}_{now_str}_full.mp4")
    full_saved = await asyncio.to_thread(save_clip_mp4, clip_frames, full_path)
    if not full_saved:
        logger.warning("[%s] Failed to save full clip", cam_id)
        return "", None, "", None

    # Zone clip (cash with cashier zone only)
    zone_path = None
    cashier_zone = cam_config.get("cashier_zone", [])
    has_zone = scenario == "cash" and len(cashier_zone) >= 3
    if has_zone:
        zone_frames = _draw_cashier_zone(clip_frames, cashier_zone)
        zone_path = str(clip_dir / f"{cam_id}_{scenario}_{now_str}_zone.mp4")
        zone_saved = await asyncio.to_thread(save_clip_mp4, zone_frames, zone_path)
        if not zone_saved:
            zone_path = None

    # Thumbnails
    mid = len(clip_frames) // 2
    thumb_full = str(thumb_dir / f"{cam_id}_{scenario}_{now_str}_full.jpg")
    cv2.imwrite(thumb_full, clip_frames[mid], [cv2.IMWRITE_JPEG_QUALITY, 85])

    thumb_zone = None
    if has_zone:
        zone_thumb = _draw_cashier_zone([clip_frames[mid]], cashier_zone)[0]
        thumb_zone = str(thumb_dir / f"{cam_id}_{scenario}_{now_str}_zone.jpg")
        cv2.imwrite(thumb_zone, zone_thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    logger.info("[%s] Clips saved: full=%s zone=%s", cam_id, full_path, zone_path or "N/A")
    return full_path, zone_path, thumb_full, thumb_zone


async def finalize_event(
    cam_id: str,
    scenario: str,
    tier2_result: dict,
    routing: str,
    full_path: str,
    zone_path: str | None,
    thumb_full: str,
    thumb_zone: str | None,
    vlm_clip_path: str,
) -> dict | None:
    """Phase 3: Tier 3 verify (MP4) + DB + alert. No GPU needed."""

    if routing == "dismiss":
        # Determine actual dismiss reason
        non_cash = str(tier2_result.get("non_cash_object", "")).lower().strip()
        if non_cash in ("smartphone", "card", "hotel_key_card", "receipt", "document", "envelope"):
            dismiss_reason = f"non-cash object detected: {non_cash}"
        else:
            dismiss_reason = f"Tier 2 conf {tier2_result.get('confidence', 0):.2f} < {config.QWEN_CONFIDENCE_LOW}"

        logger.info("[%s] %s dismissed (%s)", cam_id, scenario, dismiss_reason)
        dismiss_event = {
            "cam_id": cam_id,
            "scenario": scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier2_detected": tier2_result.get("detected", False),
            "tier2_confidence": tier2_result.get("confidence", 0.0),
            "tier2_reason": tier2_result.get("reason", ""),
            "tier3_verdict": "DISMISSED",
            "tier3_reason": dismiss_reason,
            "clip_path": _rel(full_path),
            "clip_roi_path": _rel(zone_path),
            "thumbnail_path": _rel(thumb_full),
            "thumbnail_roi_path": _rel(thumb_zone),
        }
        await db.insert_event(dismiss_event)
        return None

    # Tier 3: Gemini verifies the saved MP4 clip
    from tier3 import tier3_verifier
    tier3_result = await asyncio.to_thread(
        tier3_verifier.verify, vlm_clip_path, scenario, tier2_result
    )
    verdict = tier3_result.get("verdict", "CONFIRMED")
    logger.info("[%s] Tier 3 verdict: %s", cam_id, verdict)

    # Compute backward-compatible fields (scenario-aware)
    if scenario == "cash":
        evidence_fields = ["cash_like_object", "hand_to_hand_transfer",
                           "counter_context", "staff_customer_roles_clear",
                           "drawer_or_counting"]
        yes_count = sum(1 for f in evidence_fields if tier2_result.get(f, False))
        detected = yes_count >= 2
    else:
        # fire/violence: use original yes_count from Qwen
        yes_count = tier2_result.get("yes_count", 0)
        detected = tier2_result.get("detected", False)

    event = {
        "cam_id": cam_id,
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier2_detected": detected,
        "tier2_confidence": tier2_result.get("confidence", 0.0),
        "tier2_reason": tier2_result.get("reason", ""),
        "tier2_yes_count": yes_count,
        "tier3_verdict": verdict,
        "tier3_reason": tier3_result.get("reason", "") if tier3_result else "",
        "clip_path": _rel(full_path),
        "clip_roi_path": _rel(zone_path),
        "thumbnail_path": _rel(thumb_full),
        "thumbnail_roi_path": _rel(thumb_zone),
    }

    await db.insert_event(event)
    if verdict == "CONFIRMED":
        await send_alert(event)
    elif verdict == "UNCERTAIN":
        logger.info("[%s] UNCERTAIN — stored but no alert sent", cam_id)

    logger.info("[%s] Event finalized: %s (conf=%.2f, verdict=%s)",
                cam_id, scenario, tier2_result.get("confidence", 0), verdict)
    return event


def _rel(p: str | None) -> str:
    """Convert absolute path to relative from DATA_DIR for /media/ URLs."""
    if not p:
        return ""
    try:
        return str(Path(p).relative_to(config.DATA_DIR)).replace("\\", "/")
    except ValueError:
        return str(Path(p).name)
