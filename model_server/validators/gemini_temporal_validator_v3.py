"""Gemini temporal video validator for HIO v3."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types

    _GEMINI_AVAILABLE = True
    _GEMINI_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - dependency availability differs by host
    genai = None
    types = None
    _GEMINI_AVAILABLE = False
    _GEMINI_IMPORT_ERROR = str(exc)


class GeminiTemporalValidatorV3:
    """Validate v3 event candidates using full-frame raw and overlay clips."""

    PROMPT_VERSION = "hio-v3-tier2-exchange-band-2026-04-28"

    def __init__(self, api_key: str | None = None, enabled: bool = True) -> None:
        from model_server import config

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = str(getattr(config, "GEMINI_MODEL", "gemini-3.1-flash-lite-preview"))
        self.enabled = bool(enabled and self.api_key and _GEMINI_AVAILABLE)
        self.client = None
        self.last_validation_log: dict[str, Any] | None = None
        if self.enabled:
            timeout_sec = max(1, int(float(getattr(config, "GEMINI_TIMEOUT_SEC", 90.0) or 90.0)))
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=timeout_sec * 1000),
            )
        elif enabled and not _GEMINI_AVAILABLE:
            print(f"[GeminiTemporalValidatorV3] SDK unavailable: {_GEMINI_IMPORT_ERROR}")

    @staticmethod
    def _extract_json_text(text: str | None) -> dict[str, Any]:
        if not text:
            return {"error": "No response text"}
        payload = text.strip()
        if payload.startswith("```json"):
            payload = payload[7:]
        if payload.startswith("```"):
            payload = payload[3:]
        if payload.endswith("```"):
            payload = payload[:-3]
        try:
            return json.loads(payload.strip())
        except Exception as exc:
            return {"error": f"JSON parse error: {exc}", "raw": payload[:1000]}

    @staticmethod
    def _sanitize_packet(packet: dict[str, Any]) -> dict[str, Any]:
        drop_keys = {"global_keyframes", "cashier_roi_frames", "drawer_roi_frames"}
        clean: dict[str, Any] = {}
        for key, value in dict(packet or {}).items():
            if key in drop_keys:
                continue
            if key == "candidate_clip_paths" and isinstance(value, dict):
                clean[key] = {k: str(v) for k, v in value.items()}
            else:
                clean[key] = value
        return clean

    def _build_prompt(self, packet: dict[str, Any]) -> str:
        clean = self._sanitize_packet(packet)
        event_type = str(clean.get("event_type") or "cash").lower()
        return (
            "You are the temporal validator for a hotel CCTV safety system.\n"
            "Judge the event only from the attached full-frame CCTV clip and the packet metadata.\n\n"
            "Attached clip:\n"
            "- context_overlay: full-frame CCTV clip with cashier ROI, drawer zone, exchange_band, staff_work_zone, and optional pose skeleton overlaid.\n"
            "- Overlays mark system focus areas only. They are not physical objects or event evidence.\n\n"
            "ROI and polygon instructions:\n"
            "- Use polygon_coords.cashier_zone, drawer_zone, exchange_band, and staff_work_zone as pixel-space visual hints.\n"
            "- For cash, use the full-frame context and focus on hand/wrist motion near exchange_band, customer-staff handover, cash-like object transfer, drawer/till interaction, and cashier/counter context.\n"
            "- Staff keyboard/mouse/register work inside staff_work_zone alone is not a cash transaction.\n"
            "- Do not use overlay boxes, skeleton lines, or labels themselves as event evidence. They are visual hints only.\n"
            "- For fire and violence incidents, use full-frame temporal context and use ROI only as camera context.\n\n"
            "Correction rule:\n"
            "- If the upstream event_type is wrong but another listed event type is clearly present, set event_type_detected to the corrected type.\n"
            "- If no listed event type is clearly present, set event_type_detected to none and is_valid_event to false.\n\n"
            "Cash hard rules:\n"
            "- TRUE cash requires H1 visible Korean cash/banknotes, H2 ownership transfer/payment movement, and H3 cashier/counter/register context.\n"
            "- Korean banknote visual hints: 1,000 KRW is blue, 5,000 KRW is green, 10,000 KRW is orange/red-orange, 50,000 KRW is yellow.\n"
            "- Receipts, white paper, cards, phones, menus, forms, envelopes, and room keys are NOT cash.\n"
            "- If the cash evidence is hedged with appears/likely/maybe/probably or is not visually clear, return FALSE_POSITIVE.\n\n"
            "Fire hard rules:\n"
            "- TRUE fire/smoke requires visible flame, visible smoke plume, or temporally persistent smoke.\n"
            "- Reject sunlight/glare/reflection, TV/LED screen/signage, red/orange signs, lamps, fire extinguishers, fog/steam/blur, or camera artifacts.\n\n"
            "Return strict JSON only with these keys:\n"
            "{\n"
            '  "event_policy": "CASH_TRANSACTION | THREAT_TO_CASHIER | FIRE_ALERT | STAFF_CASH_THEFT_SUSPECT | NONE",\n'
            '  "event_type_detected": "cash | violence | fire | staff_cash_theft | none",\n'
            '  "is_valid_event": true,\n'
            '  "decision": "TRUE_POSITIVE | FALSE_POSITIVE | NOT_APPLICABLE",\n'
            '  "severity_label": "none | low | medium | high | critical",\n'
            '  "confidence": 0.0,\n'
            '  "policy_scores": {},\n'
            '  "cash_hard_gates": {"H1_visible_krw_cash": false, "H2_transfer_or_payment": false, "H3_cashier_context": false, "S_STRONG": false, "no_hedging": true},\n'
            '  "used_overlay_as_evidence": false,\n'
            '  "reason_bullets": ["short visible evidence only"]\n'
            "}\n\n"
            f"Start the first reason bullet with [{self.PROMPT_VERSION}].\n"
            f"Target upstream event_type: {event_type}\n"
            f"Input packet JSON:\n{json.dumps(clean, ensure_ascii=False, default=str)}"
        )

    @staticmethod
    def _candidate_paths(packet: dict[str, Any], video_path: str | None) -> list[tuple[str, str]]:
        clip_map = packet.get("candidate_clip_paths") if isinstance(packet, dict) else {}
        if isinstance(clip_map, dict):
            value = clip_map.get("context_overlay")
            if value and Path(str(value)).exists():
                return [("context_overlay", str(value))]
        return []

    def _parse_result(self, result: dict[str, Any], event_type: str) -> tuple[bool, float, str, str]:
        valid = bool(result.get("is_valid_event", result.get("is_valid", False)))
        try:
            confidence = float(result.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        detected = str(result.get("event_type_detected") or event_type).strip().lower()
        bullets = result.get("reason_bullets")
        if isinstance(bullets, list):
            reason = "; ".join(str(v) for v in bullets[:6])
        else:
            reason = str(result.get("reason") or result.get("decision") or "")
        lowered_reason = reason.lower()
        if bool(result.get("used_overlay_as_evidence")):
            valid = False
            confidence = min(confidence, 0.15)
            reason = f"{reason}; rejected: overlay used as evidence"
        if detected == "cash":
            gates = result.get("cash_hard_gates") if isinstance(result.get("cash_hard_gates"), dict) else {}
            required = [
                bool(gates.get("H1_visible_krw_cash")),
                bool(gates.get("H2_transfer_or_payment")),
                bool(gates.get("H3_cashier_context")),
                bool(gates.get("S_STRONG")),
                bool(gates.get("no_hedging", True)),
            ]
            hedged = any(word in lowered_reason for word in ("appears", "likely", "maybe", "probably", "seems"))
            if not all(required) or hedged:
                valid = False
                confidence = min(confidence, 0.20)
                reason = f"{reason}; rejected: cash hard gate failed"
        if not reason.startswith(f"[{self.PROMPT_VERSION}]"):
            reason = f"[{self.PROMPT_VERSION}] {reason}"
        return valid, max(0.0, min(1.0, confidence)), reason, detected or event_type

    def validate_event_evidence(
        self,
        packet: Any,
        mode: str = "video_only",
        *,
        video_path: str | None = None,
        frame: Any = None,
    ) -> tuple[bool, float, str, str]:
        start = time.time()
        packet_dict = dict(packet or {}) if isinstance(packet, dict) else {}
        event_type = str(packet_dict.get("event_type") or "cash").strip().lower()
        prompt = self._build_prompt(packet_dict)

        if not self.enabled or not self.client:
            self.last_validation_log = {
                "event_type": event_type,
                "is_valid": False,
                "confidence": 0.0,
                "reason": f"[{self.PROMPT_VERSION}] Validation disabled or SDK/API key unavailable",
                "prompt": prompt,
                "response": {},
                "processing_time_ms": int((time.time() - start) * 1000),
                "input_mode": "disabled",
                "prompt_version": self.PROMPT_VERSION,
                "packet_summary": self._sanitize_packet(packet_dict),
                "media_ref": "",
            }
            return False, 0.0, f"[{self.PROMPT_VERSION}] Validation disabled or SDK/API key unavailable", event_type

        paths = self._candidate_paths(packet_dict, video_path)
        if not paths:
            return False, 0.0, "No validation clip available", event_type

        try:
            parts = [types.Part.from_text(text=prompt)]
            media_refs: list[str] = []
            for label, path in paths[:2]:
                with open(path, "rb") as f:
                    data = f.read()
                media_refs.append(f"{label}:{Path(path).name}")
                parts.append(types.Part.from_text(text=f"Next video part label: {label}"))
                parts.append(types.Part.from_bytes(data=data, mime_type="video/mp4"))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0,
                    max_output_tokens=1800,
                    response_mime_type="application/json",
                ),
            )
            result = self._extract_json_text(getattr(response, "text", None))
            if "error" in result:
                raise RuntimeError(str(result.get("error")))
            valid, confidence, reason, corrected = self._parse_result(result, event_type)
            self.last_validation_log = {
                "event_type": event_type,
                "is_valid": valid,
                "confidence": confidence,
                "reason": reason,
                "prompt": prompt,
                "response": result,
                "processing_time_ms": int((time.time() - start) * 1000),
                "input_mode": "multi_video",
                "prompt_version": self.PROMPT_VERSION,
                "packet_summary": self._sanitize_packet(packet_dict),
                "media_ref": ";".join(media_refs),
            }
            return valid, confidence, reason, corrected
        except Exception as exc:
            reason = f"API error: {exc}"
            self.last_validation_log = {
                "event_type": event_type,
                "is_valid": False,
                "confidence": 0.0,
                "reason": reason,
                "prompt": prompt,
                "response": {"error": str(exc)},
                "processing_time_ms": int((time.time() - start) * 1000),
                "input_mode": "multi_video",
                "prompt_version": self.PROMPT_VERSION,
                "packet_summary": self._sanitize_packet(packet_dict),
                "media_ref": ";".join(f"{label}:{Path(path).name}" for label, path in paths),
            }
            return False, 0.0, reason, event_type
