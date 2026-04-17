"""
Gemini AI Validation Service for Detection Events

This module provides AI-powered validation of detected events using Google's Gemini API.
It acts as a secondary validation layer to reduce false positives by analyzing images
before events are stored in the database.

Usage:
    validator = GeminiValidator(api_key="your-api-key")
    is_valid, confidence, reason, corrected_event_type = validator.validate_event(frame, "cash")
"""

import cv2
import json
import os   
import time
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
try:
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
    _GEMINI_IMPORT_ERROR = ""
except Exception as e:
    genai = None
    types = None
    _GEMINI_AVAILABLE = False
    _GEMINI_IMPORT_ERROR = str(e)
    

# API key should be set via environment variable: GEMINI_API_KEY
api_key = os.environ.get('GEMINI_API_KEY', '')

# ============================================================================
# GLOBAL UNIFIED PROMPT (SOFT SCORING: CASH != CARD)
# ============================================================================
DEFAULT_UNIFIED_PROMPT = r"""
You are an AI Retail Security & Safety Analyst.

You will receive:
 1. A CCTV image OR a short CCTV video clip (5-10 seconds)
 2. An upstream detection_type string: "{event_type}"

Visual hint overlay (when present):
- A YELLOW closed polygon labeled "CASHIER ZONE" marks the cashier/counter area.
- A CYAN polygon labeled "DRAWER" may mark a cash drawer region.
- These overlays are attention hints, NOT physical objects in the scene.
- Do NOT describe the polygon itself in reason_bullets. Never say "yellow box",
  "rectangle", "frame overlay", or similar. It is a marker drawn by the system.
- Focus cash-transaction judgment INSIDE the yellow polygon, but KEEP using
  surrounding context (staff, customer, counter layout, body orientation) to
  judge roles and service activity.
- If the yellow polygon is absent, evaluate the full frame as before.

Currency context: This environment uses Korean Won (KRW). Korean banknotes have
distinctive colors (blue 1000, green 5000, orange 10000, yellow 50000) and are
larger than receipts. Do not claim "Korean cash" unless you see these traits.

Your job: visually analyze ONLY what is visible and choose ONE policy:
 1. CASH_TRANSACTION
 2. THREAT_TO_CASHIER
 3. FIRE_ALERT
 4. STAFF_CASH_THEFT_SUSPECT
 5. NONE

=====================================================================
OUTPUT FORMAT
=====================================================================

Return ONLY one JSON object:

{
 "event_policy": "CASH_TRANSACTION | THREAT_TO_CASHIER | FIRE_ALERT | STAFF_CASH_THEFT_SUSPECT | NONE",
 "event_type_detected": "cash | violence | fire | staff_cash_theft | none",
 "is_valid_event": true | false,
 "decision": "TRUE_POSITIVE | FALSE_POSITIVE | NOT_APPLICABLE",
 "severity_label": "none | low | medium | high | critical",
 "confidence": 0.0-1.0,
 "policy_scores": {},
 "reason_bullets": [
  "- [PROMPT_VERSION] v.26.04.17",
  "- (concrete visual facts only, no speculation)"
 ]
}

Rules:
 - Output JSON only (no extra text).
 - Fill all top-level fields.
 - reason_bullets: list of strings starting with "- ".
 - First bullet: "- [PROMPT_VERSION] v.26.04.17"
 - If you need words like "appears", "likely", "suggests", "seems" — evidence is
   insufficient. Choose NONE instead.

=====================================================================
POLICY 1) CASH_TRANSACTION
=====================================================================

Goal: Validate real cash payment only when visual evidence is clear.
When uncertain, choose NONE.

Context:
- CASH = physical paper currency (not receipts/coupons/tickets).
- Thin white paper alone is not cash.
- Hand proximity alone is not exchange.
- Treat upstream types cash, cash_object, potential_cash as "cash-like".

A) HARD RULES (all three must pass)

H1. CASH_VISUAL_CONFIRM
PASS if at least one is clearly visible:
- Banknote-like printing, color, or pattern (not a plain white slip)
- One or more distinct paper bills visible (single bill counts if banknote traits are clear)
- Staff or customer counting/peeling bills

FAIL if:
- Object looks like a plain white slip with no banknote features
- Object is rigid/reflective (card/device) or emits light (smartphone screen)
- Object remains ambiguous

H2. OWNERSHIP_TRANSFER
PASS if:
- Object moves from one person’s hand to another’s (customer to staff or staff to customer)
- Transfer direction is visible (before and after the handover)
- In video: brief occlusion during hand overlap is acceptable if the object is visible
  in the giver’s hand before and the receiver’s hand after

FAIL if:
- Only one person handles the object (no transfer)
- Ownership is unclear due to heavy occlusion across all frames

H3. ACTIVE_TRANSACTION_CONTEXT
PASS if:
- Customer and staff are at a counter/register area
- Staff is performing service actions: operating register, opening drawer, handling goods,
  processing payment, or interacting with customer about a transaction

FAIL if:
- No counter/register context visible
- Staff is clearly doing personal activity (eating, using personal phone, chatting with
  coworker away from counter)

B) SOFT RULES (only evaluated after all hard rules pass)

STRONG (at least 1 required to validate):
S_STRONG_1. Cash drawer is visibly open or object is inserted into cash slot/till
S_STRONG_2. Staff counts, peels, or aligns paper bills (one or more)
S_STRONG_3. Staff gives paper bills or coins back to customer as change

WEAK (supporting evidence):
S_WEAK_1. Handover moment is clearly visible (not heavily occluded)
S_WEAK_2. Customer turns away or leaves after the exchange
S_WEAK_3. Both parties look at the object during exchange

C) SCORE REPORTING

policy_scores = {
  "money_likelihood": 0 | 25 | 40,
  "hand_to_hand": 0 | 35 | 40,
  "safe_drawer": 0 | 40,
  "non_cash_penalty": -30 | -15 | 0,
  "total_score": (sum)
}

Score mapping:
- money_likelihood: 0 if H1 fails. 25 if banknote traits visible (any count). 40 if S_STRONG_2 passes.
- hand_to_hand: 0 if H2 fails. 35 if transfer visible with continuity. 40 if S_STRONG_3 passes.
- safe_drawer: 0 if S_STRONG_1 does not pass. 40 if drawer open/insert visible.
- non_cash_penalty:
    -30 if the transferred object itself is clearly a smartphone, card, or tablet
    -15 if the object is a plain white slip with no banknote features
    0 otherwise
  Note: a phone visible nearby (on counter, in pocket) while a separate cash-like object
  is being exchanged does NOT trigger penalty. Penalty applies only when the transferred
  object itself is non-cash.

D) DECISION RULE

Choose CASH_TRANSACTION only when ALL of these are true:
1. H1, H2, H3 all pass
2. At least one STRONG soft rule passes (S_STRONG_1, S_STRONG_2, or S_STRONG_3)
3. At least one score is 40 (safe_drawer=40 or money_likelihood=40 or hand_to_hand=40)

If any condition fails, choose NONE.

E) OUTPUT MAPPING

Validated: event_policy=CASH_TRANSACTION, is_valid_event=true, decision=TRUE_POSITIVE, severity_label=low
Not validated: event_policy=NONE, is_valid_event=false, decision=FALSE_POSITIVE (if upstream is cash-like) or NOT_APPLICABLE

F) reason_bullets: each bullet describes a concrete visible fact.
If validated, at least one bullet describes the STRONG evidence observed.

G) EXAMPLES

Example 1 — TRUE_POSITIVE:
Scene: Customer hands colored paper bills to staff at counter. Staff opens cash drawer
and places bills inside. Customer receives coins as change.
Output: is_valid_event=true, money_likelihood=40, hand_to_hand=40, safe_drawer=40,
non_cash_penalty=0, total_score=120
reason_bullets: ["- [PROMPT_VERSION] v.26.04.17", "- Customer holds two colored paper
objects with banknote-like appearance", "- Staff receives bills and opens cash drawer",
"- Coins handed back to customer as change"]

Example 2 — FALSE_POSITIVE (phone + receipt):
Scene: Customer places phone on counter showing reservation. Staff hands white paper
slip (receipt). No banknote features visible.
Output: is_valid_event=false, money_likelihood=0, hand_to_hand=0, safe_drawer=0,
non_cash_penalty=0, total_score=0
reason_bullets: ["- [PROMPT_VERSION] v.26.04.17", "- Object on counter is a smartphone
(screen illuminated)", "- White paper handed by staff has no banknote features (receipt)"]

Example 3 — TRUE_POSITIVE (single bill):
Scene: Customer hands one colored paper object to staff at front desk. Object has
visible green tint consistent with 5000 KRW. Staff places it in open drawer.
Output: is_valid_event=true, money_likelihood=25, hand_to_hand=35, safe_drawer=40,
non_cash_penalty=0, total_score=100
reason_bullets: ["- [PROMPT_VERSION] v.26.04.17", "- Single paper object with green
banknote-like color", "- Customer-to-staff transfer visible", "- Cash drawer open,
bill placed inside"]

=====================================================================
POLICY 2) THREAT_TO_CASHIER
=====================================================================

Goal: Threats or violence toward cashier or staff.

Valid cues:
 - aggressive reach across counter
 - punching, pushing, grabbing, throwing objects
 - weapon visible

policy_scores:
{
 "mandatory_score": 0,
 "supporting_score": 0,
 "negative_score": 0,
 "total_score": 0,
 "threat_level": 0-4,
 "threat_label": "CLEAR | TENSE | INTIMIDATION | PHYSICAL | WEAPON"
}

Severity: CLEAR=none, TENSE=low, INTIMIDATION=medium, PHYSICAL=high, WEAPON=critical

=====================================================================
POLICY 3) FIRE_ALERT
=====================================================================

Goal: Detect actual fire situations.

FIRE_ALERT requires at least one clearly visible:
 - active flames
 - visible smoke emitted from an object or area

Fire-related objects alone (extinguisher, alarm, sign) do not qualify.

If flames or smoke are not visible:
 event_policy=NONE, is_valid_event=false, fire_confidence=0.0, smoke_confidence=0.0

policy_scores:
{
 "fire_confidence": 0.0-1.0,
 "smoke_confidence": 0.0-1.0
}

Severity: none / low / medium / high / critical based on visible scale.

=====================================================================
POLICY 4) STAFF_CASH_THEFT_SUSPECT
=====================================================================

Goal: Suspicious cash removal by staff without a valid customer transaction.

If metadata has has_cash_box_roi=true and cash_box_bboxes:
 - cash box access is a strong hint

Otherwise use behavior:
 - cash-like object in staff hand
 - moved toward personal area (pocket, bag, inside clothes)
 - nervous look-around or hiding

policy_scores:
{
 "suspicion_level": 0-3,
 "suspicion_label": "none | low | medium | high",
 "cash_box_access": true/false,
 "looks_around": true/false,
 "moves_cash_to_personal_area": true/false,
 "customer_present": true/false,
 "paperwork_or_reconciliation": true/false
}

Severity: none / low / medium / high

=====================================================================
POLICY PRIORITY: 1.FIRE_ALERT 2.THREAT_TO_CASHIER 3.CASH_TRANSACTION 4.STAFF_CASH_THEFT_SUSPECT 5.NONE
=====================================================================

Justify all decisions with factual visual observations in reason_bullets.
"""
# ============================================================================


class GeminiValidator:
    """
    Validates detection events using Google Gemini Vision API.
    
    This acts as a filter layer - only events confirmed by Gemini are stored.
    Supports custom prompts per camera and logging of all validations.
    """
    
    # Gemini API - use gemini-2.5-flash-lite (cheapest, FREE standard tier)
    # Best for: high volume, cost-efficient image validation
    # Pricing: FREE (standard) | $0.10/1M input + $0.40/1M output (paid)
    # https://ai.google.dev/gemini-api/docs/pricing
    MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    EVIDENCE_PROMPT_VERSION = "evidence-v1.1"
    
    # Legacy prompts (for backward compatibility)
    PROMPTS = {
        'cash': """Analyze this CCTV image from a cash register area. 
Determine if there is a CASH TRANSACTION happening.

Look for these signs of a cash transaction:
1. A cashier behind a counter/register
2. A customer in front of the counter
3. Hands exchanging money, cards, or items
4. Cash register or POS terminal visible
5. Hand reaching into cash drawer

Respond in JSON format ONLY:
{
    "is_cash_transaction": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "details": {
        "cashier_visible": true/false,
        "customer_visible": true/false,
        "cash_exchange_visible": true/false,
        "register_visible": true/false
    }
}""",
        
        'violence': """Analyze this CCTV image for VIOLENCE or PHYSICAL ALTERCATION.

Look for these signs of violence:
1. People in fighting poses
2. Physical contact between people (punching, pushing, grabbing)
3. Aggressive body language
4. People on the ground from being pushed/hit
5. Multiple people surrounding one person aggressively

Do NOT flag as violence:
- Normal standing or walking
- Friendly interaction or handshakes
- People simply close together

Respond in JSON format ONLY: 
{
    "is_violence": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "details": {
        "fighting_pose": true/false,
        "physical_contact": true/false,
        "aggressive_behavior": true/false,
        "people_count": number
    }
}""",
        
        'fire': """Analyze this CCTV image for FIRE or SMOKE.

Look for these signs of fire:
1. Visible flames (orange/red/yellow)
2. Smoke (white, gray, or black)
3. Unusual lighting that could indicate fire
4. Fire on objects, walls, or floor

Do NOT flag as fire:
- Normal lighting
- Red/orange colored objects
- Steam from cooking
- Sunlight reflections

Respond in JSON format ONLY:
{
    "is_fire": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "details": {
        "flames_visible": true/false,
        "smoke_visible": true/false,
        "fire_location": "description or null"
    }
}"""
    }
    
    def __init__(self, api_key: str = None, enabled: bool = True, camera_id: int = None):
        """
        Initialize the Gemini validator.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            enabled: If False, all validations return True (bypass mode).
            camera_id: Camera ID for logging and custom prompts.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.enabled = enabled and bool(self.api_key) and _GEMINI_AVAILABLE
        self.client = None
        self.camera_id = camera_id
        self.custom_prompts = {}  # Custom prompts per event type
        self.last_validation_log = None  # Store last validation for debugging
        
        if self.enabled:
            try:
                from model_server import config

                timeout_sec = max(1, int(float(getattr(config, "GEMINI_TIMEOUT_SEC", 30.0) or 30.0)))
                timeout_ms = max(1000, timeout_sec * 1000)
                self.client = genai.Client(
                    api_key=self.api_key,
                    # google.genai HttpOptions.timeout is in milliseconds.
                    http_options=types.HttpOptions(timeout=timeout_ms),
                )
                print(f"[GeminiValidator] Initialized with model: {self.MODEL_NAME}")
            except Exception as e:
                print(f"[GeminiValidator] Failed to initialize: {e}")
                self.enabled = False
        elif enabled and not _GEMINI_AVAILABLE:
            print(f"[GeminiValidator] Warning: Gemini SDK not available: {_GEMINI_IMPORT_ERROR}")
        
        if not self.api_key and enabled:
            print("[GeminiValidator] Warning: No API key provided, validation disabled")
    
    def set_custom_prompts(self, prompts: Dict[str, str]):
        """Set custom prompts for event types (supports unified prompt)"""
        self.custom_prompts = prompts
    
    def get_prompt(self, event_type: str) -> str:
        """Get prompt for event type - uses unified prompt with {event_type} placeholder"""
        # Check for unified prompt first (stored in 'unified' key from web UI)
        unified_prompt = self.custom_prompts.get('unified', '')
        
        # If the prompt contains {event_type}, it's a unified prompt
        if unified_prompt and '{event_type}' in unified_prompt:
            return unified_prompt.replace('{event_type}', event_type)
        
        # Legacy: check for specific event type prompt
        if self.custom_prompts.get(event_type):
            return self.custom_prompts.get(event_type)
        
        # Default: use global unified prompt
        return DEFAULT_UNIFIED_PROMPT.replace('{event_type}', event_type)

    @staticmethod
    def _packet_meta(packet: Any) -> Dict[str, Any]:
        """Normalize EvidencePacket/dataclass/dict into metadata dict."""
        if packet is None:
            return {}
        if isinstance(packet, dict):
            return dict(packet)
        if hasattr(packet, "to_dict"):
            try:
                return dict(packet.to_dict())
            except Exception:
                pass

        keys = (
            "packet_id",
            "schema_version",
            "episode_id",
            "event_type",
            "tier1_confidence",
            "stability_score",
            "detection_count",
            "duration_seconds",
            "selected_mode",
            "router_action",
            "router_reason",
            "router_q",
            "focus_hints",
            "video_window_sec",
            "florence_signals",
        )
        out: Dict[str, Any] = {}
        for k in keys:
            if hasattr(packet, k):
                out[k] = getattr(packet, k)
        return out

    @staticmethod
    def _normalize_florence_signals(signals: Any) -> Dict[str, List[str]]:
        base = {
            'matched_keywords': [],
            'object_hints': [],
            'exclusion_match': [],
            'global_keywords': [],
        }
        if not isinstance(signals, dict):
            return base

        def _norm(values: Any) -> List[str]:
            if values is None:
                return []
            if isinstance(values, list):
                items = values
            elif isinstance(values, (tuple, set)):
                items = list(values)
            else:
                items = [values]

            out: List[str] = []
            seen = set()
            for item in items:
                s = str(item).strip().lower()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= 8:
                    break
            return out

        return {
            'matched_keywords': _norm(signals.get('matched_keywords')),
            'object_hints': _norm(signals.get('object_hints')),
            'exclusion_match': _norm(signals.get('exclusion_match')),
            'global_keywords': _norm(signals.get('global_keywords')),
        }

    @staticmethod
    def _packet_frames(packet: Any, key: str) -> List[Any]:
        """Extract frame lists from packet-like object."""
        if packet is None:
            return []
        if isinstance(packet, dict):
            v = packet.get(key, [])
            return v if isinstance(v, list) else []
        v = getattr(packet, key, [])
        return v if isinstance(v, list) else []

    def _build_evidence_prompt(self, event_type: str, packet: Any = None) -> str:
        """Build Gemini prompt with minimal upstream context prepended before policy.

        Design notes (2026-04-17):
        - florence_signals (matched_keywords, object_hints) removed to prevent
          anchoring bias — Gemini should judge from visible evidence only.
        - Upstream context placed BEFORE the policy prompt so recency bias
          favors the policy rules (which come last).
        - Only tier1_confidence and episode metadata kept for traceability.
        """
        base_prompt = self.get_prompt(event_type)
        if packet is None:
            return base_prompt

        meta = self._packet_meta(packet)
        lines = [
            "--- UPSTREAM CONTEXT (metadata only, do not use for visual judgment) ---",
            f"upstream_event_type: {event_type}",
            f"episode_id: {meta.get('episode_id') or 'unknown'}",
            f"tier1_confidence: {float(meta.get('tier1_confidence') or 0.0):.3f}",
            f"span: {meta.get('video_window_sec') or []}",
            "--- END UPSTREAM CONTEXT ---",
        ]
        # Upstream BEFORE policy prompt: recency bias favors the policy rules
        return "\n".join(lines) + "\n\n" + base_prompt

    @staticmethod
    def _extract_json_text(text: str) -> Dict[str, Any]:
        """Parse Gemini response text into JSON dict."""
        if text is None:
            return {"error": "No response text"}
        payload = text.strip()
        if payload.startswith('```json'):
            payload = payload[7:]
        if payload.startswith('```'):
            payload = payload[3:]
        if payload.endswith('```'):
            payload = payload[:-3]
        payload = payload.strip()
        try:
            return json.loads(payload)
        except Exception as e:
            return {"error": f"JSON parse error: {e}"}
    
    def _encode_image(self, frame):
        """Convert OpenCV frame to bytes for Gemini API."""
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def _save_validation_image(self, frame, event_type: str) -> Optional[str]:
        """Save validation image for debugging/logging."""
        try:
            from datetime import datetime
            from model_server import config
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{event_type}_{self.camera_id or 'unknown'}_{timestamp}.jpg"
            storage_path = f"gemini_logs/{filename}"
            
            # Local storage fallback
            log_dir = Path(config.DATA_DIR) / 'gemini_logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = log_dir / filename
            cv2.imwrite(str(filepath), frame)
            
            # (S3 upload logic will be handled by caller or local_storage if configured)
            return f'/data/gemini_logs/{filename}'
            
        except Exception as e:
            print(f"[GeminiValidator] Failed to save validation image: {e}")
            return None
    
    def _log_validation(
        self,
        camera_id: int,
        event_type: str,
        is_valid: bool, 
        confidence: float,
        reason: str,
        prompt: str,
        response_raw: str,
        image_path: str,
        processing_time_ms: int,
        validation_type: str = 'image',
        video_path: str = None
    ):
        """Log validation result to file
        In vlm_new_deploy, database recording happens via FlushWorker using the returned result.
        """
        import json
        from model_server import config
        try:
            log_dir = Path(config.LOG_DIR) / 'gemini_validations'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"validation_{camera_id}_{event_type}.jsonl"
            
            log_entry = {
                "camera_id": camera_id,
                "event_type": event_type,
                "validation_type": validation_type,
                "is_validated": is_valid,
                "confidence": confidence,
                "reason": reason,
                "prompt_used": prompt,
                "response_raw": response_raw,
                "image_path": image_path,
                "video_path": video_path,
                "processing_time_ms": processing_time_ms,
                "timestamp": __import__('time').time()
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            print(f"[GeminiValidator] Logged validation for camera {camera_id}, event_type={event_type}, is_validated={is_valid} to file")
        except Exception as e:
            print(f"[GeminiValidator] Failed to log validation to file: {e}")
    
    def _call_gemini_api(self, image_bytes: bytes, prompt: str) -> dict:
        """
        Call Gemini API with image and prompt using official SDK.
        
        Returns:
            dict: Parsed JSON response or error dict
        """
        if not self.client:
            return {"error": "Client not initialized"}
        
        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/jpeg"
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0,
                    max_output_tokens=1500,
                    response_mime_type="application/json"
                )
            )
            
            # Extract text from response
            if response.text:
                result = self._extract_json_text(response.text)
                print(f"[GeminiValidator] API response: {json.dumps(result, indent=2)[:500]}")
                return result
            else:
                return {"error": "No response from Gemini"}
                
        except json.JSONDecodeError as e:
            print(f"[GeminiValidator] JSON parse error: {e}")
            print(f"[GeminiValidator] Response text: {response.text[:500] if response else 'None'}")
            return {"error": "Invalid JSON response"}
        except Exception as e:
            print(f"[GeminiValidator] API error: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _parse_score(value) -> float:
        """Safely parse a score value from Gemini response (may be int, float, or str)."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    def _parse_new_response_format(self, result: dict, original_event_type: str) -> Tuple[bool, float, str, str]:
        """
        Parse the new Gemini response format (soft scoring with policies).
        
        New format fields:
        - event_policy: CASH_TRANSACTION | THREAT_TO_CASHIER | FIRE_ALERT | STAFF_CASH_THEFT_SUSPECT | NONE
        - event_type_detected: cash | violence | fire | staff_cash_theft | none
        - is_valid_event: true/false
        - decision: TRUE_POSITIVE | FALSE_POSITIVE | NOT_APPLICABLE
        - severity_label: none | low | medium | high | critical
        - confidence: 0.0-1.0
        - policy_scores: dict with scoring details
        - reason_bullets: list of strings
        
        Returns:
            Tuple of (is_valid, confidence, reason, corrected_event_type)
        """
        # Check for new format fields
        event_policy = result.get('event_policy', '')
        is_valid_event = result.get('is_valid_event')
        event_type_detected = result.get('event_type_detected', 'none')
        confidence = result.get('confidence', 0.0)
        decision = result.get('decision', '')
        severity_label = result.get('severity_label', 'none')
        policy_scores = result.get('policy_scores', {})
        reason_bullets = result.get('reason_bullets', [])
        
        # Also support legacy format for backwards compatibility
        legacy_is_valid = result.get('is_valid')
        legacy_reason = result.get('reason', '')
        
        # Determine is_valid - prefer new format
        if is_valid_event is not None:
            is_valid = is_valid_event
        elif legacy_is_valid is not None:
            is_valid = legacy_is_valid
        else:
            # Infer from event_policy
            is_valid = event_policy not in ['NONE', '', None]

        # Build reason string from bullets or use legacy
        if reason_bullets and isinstance(reason_bullets, list):
            reason = ' '.join([b.strip() for b in reason_bullets])
        elif legacy_reason:
            reason = legacy_reason
        else:
            reason = f"Policy: {event_policy}, Decision: {decision}, Severity: {severity_label}"

        # Add policy scores to reason if available
        if policy_scores and isinstance(policy_scores, dict):
            total_score = policy_scores.get('total_score', 'N/A')
            reason = f"[Score: {total_score}] {reason}"

        # CODE-LEVEL HARD GATE: enforce score requirements that the prompt requests
        # but Gemini may not always comply with (~70-85% prompt compliance).
        # This catches cases where Gemini sets is_valid_event=true but scores don't support it.
        # Runs AFTER reason is built so the override message prepends to the full reason.
        if is_valid and policy_scores and isinstance(policy_scores, dict):
            gate_failed = False
            gate_reason = ""

            if event_type_detected in ('cash', '') and event_policy in ('CASH_TRANSACTION', ''):
                # CASH hard gate: at least one STRONG score must be 40
                safe_drawer = self._parse_score(policy_scores.get('safe_drawer', 0))
                money_likelihood = self._parse_score(policy_scores.get('money_likelihood', 0))
                hand_to_hand = self._parse_score(policy_scores.get('hand_to_hand', 0))

                if safe_drawer < 40 and money_likelihood < 40 and hand_to_hand < 40:
                    gate_failed = True
                    gate_reason = (
                        f"[HARD_GATE_OVERRIDE] CASH requires at least one STRONG score=40, "
                        f"got safe_drawer={safe_drawer}, money_likelihood={money_likelihood}, "
                        f"hand_to_hand={hand_to_hand}"
                    )

            elif event_type_detected == 'fire' and event_policy == 'FIRE_ALERT':
                # FIRE hard gate: must have fire or smoke confidence > 0
                fire_conf = self._parse_score(policy_scores.get('fire_confidence', 0))
                smoke_conf = self._parse_score(policy_scores.get('smoke_confidence', 0))

                if fire_conf <= 0 and smoke_conf <= 0:
                    gate_failed = True
                    gate_reason = (
                        f"[HARD_GATE_OVERRIDE] FIRE requires fire_confidence>0 or smoke_confidence>0, "
                        f"got fire={fire_conf}, smoke={smoke_conf}"
                    )

            if gate_failed:
                print(f"[GeminiValidator] {gate_reason}")
                is_valid = False
                reason = f"{gate_reason} | {reason}"
        
        # Map event_policy to event_type_detected if not set
        if not event_type_detected or event_type_detected == 'none':
            policy_to_type = {
                'CASH_TRANSACTION': 'cash',
                'THREAT_TO_CASHIER': 'violence',
                'FIRE_ALERT': 'fire',
                'STAFF_CASH_THEFT_SUSPECT': 'staff_cash_theft',
                'NONE': 'none'
            }
            event_type_detected = policy_to_type.get(event_policy, 'none')
        
        # Determine corrected event type
        corrected_event_type = original_event_type
        
        if is_valid and event_type_detected != 'none' and event_type_detected != original_event_type:
            # Handle violence -> cash correction (avoid duplicates)
            if original_event_type == "violence" and event_type_detected == "cash":
                is_valid = False
                corrected_event_type = original_event_type
                reason = f"[BLOCKED] Violence->Cash correction blocked. {reason}"
            else:
                corrected_event_type = event_type_detected
                reason = f"[CORRECTED: {original_event_type}->{corrected_event_type}] {reason}"
        
        return is_valid, confidence, reason, corrected_event_type

    def validate_event_evidence(
        self,
        packet: Any,
        mode: str = "hybrid",
        *,
        video_path: Optional[str] = None,
        frame: Any = None,
    ) -> Tuple[bool, float, str, str]:
        """
        Validate an event with evidence context (packet + media).

        Modes:
        - hybrid: cash prefers storyboard then video, others video then storyboard
        - video_first: video -> storyboard -> image
        - video_only: video only (no storyboard/image fallback)
        - images_first/storyboard: storyboard -> image -> video
        - image: image -> storyboard -> video
        """
        meta = self._packet_meta(packet)
        event_type = str(meta.get('event_type') or 'cash').strip().lower()
        prompt = self._build_evidence_prompt(event_type, packet)
        prompt_version = self.EVIDENCE_PROMPT_VERSION
        start_time = time.time()

        if not self.enabled or not self.client:
            return True, 1.0, "Validation disabled", event_type

        def _finalize(result: Dict[str, Any], input_mode: str, media_ref: Optional[str] = None):
            if not isinstance(result, dict):
                result = {"error": "Invalid Gemini response"}
            if 'error' in result:
                reason = f"API error: {result.get('error')}"
                self.last_validation_log = {
                    'event_type': event_type,
                    'is_valid': False,
                    'confidence': 0.0,
                    'reason': reason,
                    'prompt': prompt,
                    'response': result,
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'input_mode': input_mode,
                    'prompt_version': prompt_version,
                    'packet_summary': meta,
                    'media_ref': media_ref,
                }
                return False, 0.0, reason, event_type

            is_valid, confidence, reason, corrected_event_type = self._parse_new_response_format(result, event_type)
            self.last_validation_log = {
                'event_type': event_type,
                'is_valid': is_valid,
                'confidence': confidence,
                'reason': reason,
                'prompt': prompt,
                'response': result,
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'input_mode': input_mode,
                'prompt_version': prompt_version,
                'packet_summary': meta,
                'media_ref': media_ref,
            }
            return is_valid, confidence, reason, corrected_event_type

        def _storyboard_frames() -> List[Any]:
            roi = self._packet_frames(packet, 'cashier_roi_frames')
            drawer = self._packet_frames(packet, 'drawer_roi_frames')
            global_kf = self._packet_frames(packet, 'global_keyframes')
            if event_type == 'cash':
                frames = roi + drawer + global_kf
            else:
                frames = global_kf + roi + drawer
            out = []
            for f in frames:
                try:
                    if f is not None and getattr(f, 'size', 0) > 0:
                        out.append(f)
                except Exception:
                    continue
            return out

        def _run_storyboard() -> Optional[Tuple[bool, float, str, str]]:
            frames = _storyboard_frames()
            if not frames:
                return None
            try:
                parts = [types.Part.from_text(text=f"{prompt}\n\nAnalyze the keyframe storyboard in temporal order.")]
                for f in frames[:12]:
                    parts.append(types.Part.from_bytes(data=self._encode_image(f), mime_type="image/jpeg"))
                response = self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        top_k=1,
                        top_p=1.0,
                        max_output_tokens=1500,
                        response_mime_type="application/json",
                    ),
                )
                result = self._extract_json_text(response.text) if getattr(response, 'text', None) else {"error": "No response text"}
                return _finalize(result, input_mode='storyboard', media_ref='packet_keyframes')
            except Exception as e:
                return _finalize({'error': str(e)}, input_mode='storyboard')

        def _run_video() -> Optional[Tuple[bool, float, str, str]]:
            if not video_path:
                return None
            try:
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                video_prompt = (
                    f"{prompt}\n\nAnalyze this short validation clip (~6 seconds). "
                    "Use motion continuity before/during/after interaction."
                )
                result = self._call_gemini_api_video(video_bytes, video_prompt)
                return _finalize(result, input_mode='video', media_ref=video_path)
            except Exception as e:
                return _finalize({'error': str(e)}, input_mode='video', media_ref=video_path)

        def _run_image() -> Optional[Tuple[bool, float, str, str]]:
            local_frame = frame
            if local_frame is None:
                frames = _storyboard_frames()
                if frames:
                    local_frame = frames[-1]
            if local_frame is None:
                return None
            try:
                result = self._call_gemini_api(self._encode_image(local_frame), prompt)
                return _finalize(result, input_mode='image')
            except Exception as e:
                return _finalize({'error': str(e)}, input_mode='image')

        mode_norm = str(mode or 'hybrid').strip().lower()
        attempts: List[str]
        if mode_norm == 'video_first':
            attempts = ['video', 'storyboard', 'image']
        elif mode_norm == 'video_only':
            attempts = ['video']
        elif mode_norm in {'images_first', 'storyboard'}:
            attempts = ['storyboard', 'image', 'video']
        elif mode_norm == 'image':
            attempts = ['image', 'storyboard', 'video']
        else:
            attempts = ['storyboard', 'video', 'image'] if event_type == 'cash' else ['video', 'storyboard', 'image']

        runners = {
            'storyboard': _run_storyboard,
            'video': _run_video,
            'image': _run_image,
        }
        last_api_error_result: Optional[Tuple[bool, float, str, str]] = None
        for name in attempts:
            result = runners[name]()
            if result is None:
                continue
            reason = str(result[2] or "")
            if reason.startswith("API error:"):
                last_api_error_result = result
                continue
            if result is not None:
                return result

        if last_api_error_result is not None:
            return last_api_error_result
        return False, 0.0, "No validation clip available", event_type

    def validate_event(
        self, frame, event_type: str, save_image: bool = True
    ) -> Tuple[bool, float, str, str]:
        """
        Validate a detection event using Gemini AI.
        
        Args:
            frame: OpenCV frame (numpy array) to analyze
            event_type: Type of event ('cash', 'violence', 'fire')
            save_image: Whether to save the validation image for logging
            
        Returns:
            Tuple of (is_valid, confidence, reason, corrected_event_type)
            - is_valid: True if Gemini confirms the event
            - confidence: Gemini's confidence score (0.0-1.0)
            - reason: Gemini's explanation
            - corrected_event_type: The actual event type Gemini detected (may differ from input)
        """
        start_time = time.time()
        image_path = None
        prompt = ""
        response_raw = ""
        
        # If disabled or no API key, bypass validation
        if not self.enabled:
            return True, 1.0, "Validation bypassed (no API key)", event_type
        
        # Get prompt (custom or default)
        prompt = self.get_prompt(event_type)
        if not prompt:
            print(f"[GeminiValidator] Unknown event type: {event_type}")
            return True, 1.0, f"Unknown event type: {event_type}", event_type
        
        # Check frame validity
        if frame is None or frame.size == 0:
            return False, 0.0, "Invalid frame", event_type
        
        try:
            # Save image for logging if enabled
            if save_image and self.camera_id:
                image_path = self._save_validation_image(frame, event_type)
            
            # Encode image
            image_bytes = self._encode_image(frame)
            
            # Call Gemini API
            result = self._call_gemini_api(image_bytes, prompt)
            response_raw = json.dumps(result)
            
            # Check for errors
            if 'error' in result:
                print(f"[GeminiValidator] API error: {result['error']}")
                return False, 0.0, f"API error: {result['error']}", event_type
            
            # Parse response using new format parser (handles both old and new formats)
            is_valid, confidence, reason, corrected_event_type = self._parse_new_response_format(result, event_type)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Store for debugging
            self.last_validation_log = {
                'event_type': event_type,
                'is_valid': is_valid,
                'confidence': confidence,
                'reason': reason,
                'prompt': prompt,
                'response': result,
                'image_path': image_path,
                'processing_time_ms': processing_time_ms
            }
            
            # Log to database
            print(f"[GeminiValidator] DEBUG: camera_id={self.camera_id}, will_log={bool(self.camera_id)}")
            if self.camera_id:
                self._log_validation(
                    self.camera_id, event_type, is_valid, confidence, 
                    reason, prompt, response_raw, image_path, processing_time_ms,
                    validation_type='image'
                )
            else:
                print(f"[GeminiValidator] ? ï¸ Skipping database log - no camera_id set!")
            
            print(
                f"[GeminiValidator] IMAGE {event_type}: valid={is_valid}, "
                f"conf={confidence:.2f}, corrected={corrected_event_type}, "
                f"reason={reason[:150]}"
            )
            
            return is_valid, confidence, reason, corrected_event_type
            
        except Exception as e:
            print(f"[GeminiValidator] Exception: {e}")
            return False, 0.0, f"Validation error: {e}", event_type
    
    def validate_event_video(self, video_path: str, event_type: str) -> Tuple[bool, float, str, str]:
        """
        Validate a detection event using a short video clip instead of a single frame.
        Provides better context for Gemini to analyze motion and behavior.
        
        Args:
            video_path: Path to the validation video clip
            event_type: Type of event to validate ('cash', 'violence', 'fire', 'potential_cash')
            
        Returns:
            Tuple of (is_valid, confidence, reason, corrected_event_type)
        """
        if not self.enabled or not self.client:
            return True, 1.0, "Validation disabled", event_type
        
        try:
            import time
            start_time = time.time()
            
            # Read video file
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            # Get appropriate prompt
            prompt = self.get_prompt(event_type)
            prompt = f"{prompt}\n\nAnalyze this short video clip (5-10 seconds) showing the detected event. Consider motion, behavior, and context over time."
            
            # Call Gemini API with video
            result = self._call_gemini_api_video(video_bytes, prompt)
            response_raw = json.dumps(result) if isinstance(result, dict) else str(result)
            
            # Parse response
            if 'error' in result:
                print(f"[GeminiValidator] Video API error: {result['error']}")
                return False, 0.0, f"API error: {result['error']}", event_type
            
            # Check if response is empty
            if not result:
                print(f"[GeminiValidator] Empty response from Gemini API")
                return False, 0.0, "Empty API response", event_type
            
            # Parse response using new format parser (handles both old and new formats)
            is_valid, confidence, reason, corrected_event_type = self._parse_new_response_format(result, event_type)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Store for debugging
            self.last_validation_log = {
                'event_type': event_type,
                'is_valid': is_valid,
                'confidence': confidence,
                'reason': reason,
                'prompt': prompt,
                'response': result,
                'response_raw': response_raw,
                'video_path': video_path,  # Local temp path - will be updated by caller
                'processing_time_ms': processing_time_ms
            }
            
            # NOTE: Database logging is now handled by the caller (validate_detection in utils.py)
            # after it uploads the video to S3 and has the correct URL.
            # This allows us to store S3 URLs instead of temp file paths.
            
            print(f"[GeminiValidator] VIDEO {event_type}: valid={is_valid}, conf={confidence:.2f}, corrected={corrected_event_type}, reason={reason[:150]}")
            
            return is_valid, confidence, reason, corrected_event_type
            
        except Exception as e:
            print(f"[GeminiValidator] Video validation exception: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0, f"Video validation error: {e}", event_type
    
    def _call_gemini_api_video(self, video_bytes: bytes, prompt: str) -> dict:
        """
        Call Gemini API with video and prompt using official SDK.
        
        Returns:
            dict: Parsed JSON response or error dict
        """
        if not self.client:
            return {"error": "Client not initialized"}
        
        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                data=video_bytes,
                                mime_type="video/mp4"
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0,
                    max_output_tokens=1500,
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                result = self._extract_json_text(response.text)
                print(f"[GeminiValidator] Video API response: {result}")
                return result

            print(f"[GeminiValidator] Video API returned no text. Full response: {response}")
            return {"error": "No response text"}
                
        except json.JSONDecodeError as e:
            print(f"[GeminiValidator] JSON parse error for video: {e}")
            print(f"Response text: {response.text if response else 'None'}")
            return {"error": f"JSON parse error: {e}"}
        except Exception as e:
            print(f"[GeminiValidator] API video call error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def validate_h2h_event(
        self,
        frames: list,
        h2h_context: dict,
    ) -> Tuple[bool, float, str, str]:
        """
        Backward-compatible H2H wrapper.

        Legacy callers can keep using this function name, but runtime validation
        now goes through the unified evidence path.
        """
        packet = {
            'packet_id': str(h2h_context.get('packet_id') or 'h2h_legacy'),
            'schema_version': 'h2h-v1',
            'event_type': 'cash',
            'episode_id': str(h2h_context.get('episode_id') or ''),
            'tier1_confidence': float(h2h_context.get('confidence') or 0.0),
            'stability_score': float(
                h2h_context.get('stability') or h2h_context.get('confidence') or 0.0
            ),
            'router_action': str(h2h_context.get('router_action') or 'GEMINI_IMG'),
            'router_reason': str(h2h_context.get('router_reason') or 'legacy_h2h'),
            'router_q': h2h_context.get('router_q') or {},
            'focus_hints': [
                'h2h',
                f"keywords:{','.join((h2h_context.get('matched_keywords') or [])[:8])}",
                f"object_hints:{','.join((h2h_context.get('object_hints') or [])[:8])}",
            ],
            'global_keyframes': list(frames or [])[:12],
            'cashier_roi_frames': list(frames or [])[:8],
            'drawer_roi_frames': [],
        }
        seed_frame = frames[0] if frames else None
        return self.validate_event_evidence(packet, mode='storyboard', frame=seed_frame)

    def validate_cash_transaction(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for cash transaction validation."""
        # NOTE: ?¬ê¸°?'ëŠ” 4ê°'ë? ë¦¬í„´?˜ì?ë§? ?¸ì¶' ì¸¡ì—??3ê°'ë§Œ ë°›ì'¼ë©??Œì´???¸íŒ¨?¹ì'¼ë¡?ì¡°ì ˆ ê°€??
        return self.validate_event(frame, 'cash')
    
    def validate_violence(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for violence validation."""
        return self.validate_event(frame, 'violence')
    
    def validate_fire(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for fire validation."""
        return self.validate_event(frame, 'fire')


# Singleton instance for global access
_validator_instance = None


def get_validator(api_key: str = None) -> GeminiValidator:
    """
    Get or create the global GeminiValidator instance.
    
    Args:
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        GeminiValidator instance
    """
    global _validator_instance
    
    if _validator_instance is None:
        key = api_key or os.environ.get('GEMINI_API_KEY', '')
        from model_server import config
        if not key and getattr(config, 'GEMINI_API_KEY', None):
            key = config.GEMINI_API_KEY
        _validator_instance = GeminiValidator(api_key=key)
    
    return _validator_instance


def validate_detection(frame, event_type: str, api_key: str = None) -> Tuple[bool, float, str, str]:
    """
    Convenience function to validate a detection event.
    
    Args:
        frame: OpenCV frame to analyze
        event_type: Type of event ('cash', 'violence', 'fire')
        api_key: Optional API key
        
    Returns:
        Tuple of (is_valid, confidence, reason, corrected_event_type)
    """
    validator = get_validator(api_key)
    return validator.validate_event(frame, event_type)
