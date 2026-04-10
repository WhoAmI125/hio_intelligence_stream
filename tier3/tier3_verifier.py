"""Tier 3 Cloud Verifier — Hard Gate final judge via Gemini 2.5 Flash.

Cash: 3 Hard Gates + 3 Soft Strong + UNCERTAIN verdict.
Fire/Violence: Simple CONFIRMED/FALSE_ALARM (no UNCERTAIN — safety first).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.TIER3_API_KEY)
    return _client


def _build_cash_prompt(tier2_result: dict) -> str:
    """Build Hard Gate cash verification prompt with Qwen evidence as hints."""

    # Format Qwen evidence as unreliable hints
    qwen_hints = (
        "[LOCAL MODEL HINTS — MAY BE NOISY]\n"
        "로컬 3B 양자화 모델의 사전 분석입니다. 오류가 빈번합니다.\n"
        "당신의 주의를 어디에 집중할지 안내하는 용도로만 사용하세요.\n"
        "최종 판단은 반드시 영상에서 직접 확인한 시각적 증거에 기반해야 합니다.\n\n"
        f"  cash_like_object: {tier2_result.get('cash_like_object', 'unknown')}\n"
        f"  hand_to_hand_transfer: {tier2_result.get('hand_to_hand_transfer', 'unknown')}\n"
        f"  counter_context: {tier2_result.get('counter_context', 'unknown')}\n"
        f"  drawer_or_counting: {tier2_result.get('drawer_or_counting', 'unknown')}\n"
        f"  non_cash_object: {tier2_result.get('non_cash_object', 'unknown')}\n"
        f"  confidence: {tier2_result.get('confidence', 0.0)}\n"
        f"  reason: {tier2_result.get('reason', '')}\n"
    )

    return (
        "당신은 호텔 프런트 CCTV에서 현금 거래를 최종 검증하는 감사자입니다.\n"
        "당신의 역할은 모호한 장면을 현금으로 확대 해석하지 않는 것입니다.\n\n"
        "최우선 원칙:\n"
        "- 보수적으로 판단하세요.\n"
        "- 명확한 시각 증거가 없으면 CONFIRMED가 아니라 UNCERTAIN 또는 FALSE_ALARM 입니다.\n"
        "- 카운터 앞 상호작용, 손 움직임, 종이처럼 보이는 물체만으로는 현금 거래를 확정하지 마세요.\n"
        "- 스마트폰, 카드, 영수증, 문서, 봉투가 명확하면 FALSE_ALARM 입니다.\n\n"
        "먼저 영상만 보고 독립 판단한 뒤,\n"
        "그 다음 참고 자료(Qwen 결과)를 보조적으로만 사용하세요.\n"
        "Qwen 결과보다 영상에서 직접 보이는 증거를 우선합니다.\n\n"
        "Hard Gate 1. CASH_VISUAL_CONFIRM (완화 기준)\n"
        "다음 중 하나라도 보이면 PASS:\n"
        "- 지폐의 색상/패턴/인쇄 특징이 보임\n"
        "- 복수 장의 얇은 물체가 분리되어 보임\n"
        "- 지폐를 세거나 정리하는 동작이 보임\n"
        "- 얇고 작은 평면 물체가 손에서 손으로 전달됨 (지폐일 수 있음)\n"
        "- 접혀 있거나 뭉쳐진 종이류가 전달됨\n"
        "FAIL 조건 (이것만 FAIL):\n"
        "- 스마트폰, 신용카드, 키카드처럼 딱딱하고 반사되는 물체가 명확함\n"
        "- 아무런 물체 전달 없이 대화만 함\n\n"
        "Hard Gate 2. OWNERSHIP_TRANSFER\n"
        "다음이 보여야 PASS:\n"
        "- 물체가 한 사람 손에서 다른 사람 손으로 이동하는 흐름이 보임\n"
        "- 또는 카운터 위에서 물체를 건네받는 동작이 보임\n"
        "FAIL 조건:\n"
        "- 손만 가까울 뿐 물체 이동이 전혀 없음\n"
        "- 각자 자기 물건만 만지고 있음\n\n"
        "Hard Gate 3. TRANSACTION_CONTEXT (완화 기준)\n"
        "다음 중 하나면 PASS:\n"
        "- 호텔 프런트/카운터에서 직원-고객이 대면하고 있음\n"
        "- 카운터 위에서 물건을 주고받는 상호작용이 있음\n"
        "FAIL 조건 (이것만 FAIL):\n"
        "- 사람이 혼자 있음 (상호작용 대상 없음)\n"
        "- 복도/엘리베이터 등 카운터가 아닌 장소\n\n"
        "Soft Strong Evidence:\n"
        "- 현금 서랍이 열리거나 현금이 들어감\n"
        "- 복수 지폐를 직원이 세는 장면\n"
        "- 거스름돈으로 지폐/동전이 반환됨\n\n"
        "Non-cash override:\n"
        "아래가 명확하면 무조건 FALSE_ALARM:\n"
        "smartphone, card, hotel_key_card, receipt, document, envelope\n\n"
        f"{qwen_hints}\n"
        "최종 판정 규칙:\n"
        "- Hard Gate 3개 모두 통과 + Soft Strong 1개 이상 → CONFIRMED\n"
        "- Hard Gate 3개 모두 통과 + Soft Strong 없음 → UNCERTAIN\n"
        "- Hard Gate 2개 통과 (1개 불확실) → UNCERTAIN\n"
        "- Hard Gate 0~1개만 통과 → FALSE_ALARM\n"
        "- Non-cash 물체 명확 → FALSE_ALARM\n\n"
        "반드시 아래 JSON만 출력하세요.\n"
        "{\n"
        '  "verdict": "CONFIRMED | UNCERTAIN | FALSE_ALARM",\n'
        '  "hard_gate": {\n'
        '    "cash_visual_confirm": true,\n'
        '    "ownership_transfer": true,\n'
        '    "active_transaction_context": true\n'
        "  },\n"
        '  "soft_strong": {\n'
        '    "drawer_open_or_insert": false,\n'
        '    "multiple_bills_counted": false,\n'
        '    "change_return_visible": false\n'
        "  },\n"
        '  "non_cash_object": "none",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "핵심 근거 2~3문장"\n'
        "}"
    )


def _build_fire_violence_prompt(scenario: str, tier2_result: dict) -> str:
    """Simple CONFIRMED/FALSE_ALARM prompt for fire/violence."""
    scenario_desc = {
        "fire": "화재 또는 연기 발생",
        "violence": "폭력 행위 (때리기, 밀기, 싸움)",
    }
    return (
        f"당신은 호텔 CCTV 영상 분석 전문가입니다.\n"
        f"이 영상(12초)에서 '{scenario_desc.get(scenario, scenario)}'가 "
        f"실제로 발생하고 있는지 판단하세요.\n\n"
        f"참고 (로컬 모델 결과, 보조):\n"
        f"  detected: {tier2_result.get('detected')}\n"
        f"  confidence: {tier2_result.get('confidence')}\n"
        f"  reason: {tier2_result.get('reason')}\n\n"
        f"CONFIRMED (실제 발생) 또는 FALSE_ALARM (오탐) 중 하나로 답하고,\n"
        f"판단 근거를 2-3줄로 설명하세요."
    )


def verify(
    clip_path: str,
    scenario: str,
    tier2_result: dict,
) -> dict:
    """Send saved MP4 clip to Gemini for verification.

    Cash: Hard Gate 3-stage → CONFIRMED / UNCERTAIN / FALSE_ALARM
    Fire/Violence: Simple → CONFIRMED / FALSE_ALARM
    """
    # Fail policy: cash → UNCERTAIN (conservative), fire/violence → CONFIRMED (safety first)
    fail_verdict = "CONFIRMED" if scenario in ("fire", "violence") else "UNCERTAIN"

    if not config.TIER3_API_KEY:
        logger.warning("Tier 3 API key not configured — fail_verdict=%s", fail_verdict)
        return {"verdict": fail_verdict, "reason": "tier3_api_key_missing"}

    if not clip_path or not Path(clip_path).exists():
        logger.error("Tier 3: clip file not found: %s — fail_verdict=%s", clip_path, fail_verdict)
        return {"verdict": fail_verdict, "reason": "clip_file_missing"}

    # Build scenario-specific prompt
    if scenario == "cash":
        prompt = _build_cash_prompt(tier2_result)
    else:
        prompt = _build_fire_violence_prompt(scenario, tier2_result)

    tmp_copy = None
    try:
        client = _get_client()

        # ASCII-safe path for Gemini SDK
        upload_path = clip_path
        try:
            clip_path.encode("ascii")
        except (UnicodeEncodeError, AttributeError):
            import shutil
            import tempfile
            tmp_copy = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_copy.close()
            shutil.copy2(clip_path, tmp_copy.name)
            upload_path = tmp_copy.name

        uploaded = client.files.upload(file=upload_path)

        # Wait for ACTIVE state
        import time as _time
        for _ in range(30):
            status = client.files.get(name=uploaded.name)
            if status.state.name == "ACTIVE":
                break
            _time.sleep(1)

        response = client.models.generate_content(
            model=config.TIER3_MODEL,
            contents=[uploaded, prompt],
        )

        text = response.text.strip() if response.text else ""

        if scenario == "cash":
            return _parse_cash_response(text)
        else:
            return _parse_simple_response(text)

    except Exception as e:
        logger.error("Tier 3 verification failed: %s", e, exc_info=True)
        return {"verdict": fail_verdict, "reason": f"tier3_error: {e}"}
    finally:
        if tmp_copy is not None:
            Path(tmp_copy.name).unlink(missing_ok=True)


def _parse_cash_response(text: str) -> dict:
    """Parse structured Hard Gate JSON response for cash."""
    clean = re.sub(r"```(?:json)?", "", text).strip()
    try:
        start = clean.index("{")
        end = clean.rindex("}") + 1
        data = json.loads(clean[start:end])

        verdict = data.get("verdict", "").upper()
        if verdict not in ("CONFIRMED", "UNCERTAIN", "FALSE_ALARM"):
            verdict = "UNCERTAIN"

        hard_gate = data.get("hard_gate", {})
        soft_strong = data.get("soft_strong", {})

        # Server-side Hard Gate enforcement (don't trust Gemini's verdict blindly)
        h1 = hard_gate.get("cash_visual_confirm", False)
        h2 = hard_gate.get("ownership_transfer", False)
        h3 = hard_gate.get("active_transaction_context", False)
        hard_pass = h1 and h2 and h3

        s1 = soft_strong.get("drawer_open_or_insert", False)
        s2 = soft_strong.get("multiple_bills_counted", False)
        s3 = soft_strong.get("change_return_visible", False)
        has_soft = s1 or s2 or s3

        non_cash = str(data.get("non_cash_object", "none")).lower().strip()
        hard_count = sum([h1, h2, h3])

        if non_cash in ("smartphone", "card", "hotel_key_card", "receipt", "document", "envelope"):
            enforced_verdict = "FALSE_ALARM"
        elif hard_pass and has_soft:
            enforced_verdict = "CONFIRMED"
        elif hard_pass:
            # All 3 hard gates pass but no soft strong → UNCERTAIN
            enforced_verdict = "UNCERTAIN"
        elif hard_count >= 2:
            # 2/3 hard gates pass → still worth reviewing (UNCERTAIN)
            enforced_verdict = "UNCERTAIN"
        else:
            # 0-1 hard gates → FALSE_ALARM
            enforced_verdict = "FALSE_ALARM"

        # Parse confidence safely AFTER verdict is already enforced
        try:
            conf = float(data.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0

        return {
            "verdict": enforced_verdict,
            "hard_gate": hard_gate,
            "soft_strong": soft_strong,
            "non_cash_object": non_cash,
            "confidence": conf,
            "reason": data.get("reason", text[:500]),
        }

    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse Tier 3 cash JSON (%s), falling back to text: %s", e, text[:300])
        return _parse_simple_response(text, allow_uncertain=True)


def _parse_simple_response(text: str, allow_uncertain: bool = False) -> dict:
    """Fallback: text-based verdict parsing for fire/violence or JSON failure."""
    upper = (text or "").upper().strip()

    # Empty or garbage response → never default to CONFIRMED
    if len(upper) < 5:
        verdict = "UNCERTAIN" if allow_uncertain else "FALSE_ALARM"
        return {"verdict": verdict, "reason": "empty_or_invalid_response"}

    if "FALSE_ALARM" in upper:
        verdict = "FALSE_ALARM"
    elif "CONFIRMED" in upper and "FALSE_ALARM" not in upper:
        verdict = "CONFIRMED"
    elif allow_uncertain and "UNCERTAIN" in upper:
        verdict = "UNCERTAIN"
    else:
        # No keyword matched — conservative default
        verdict = "UNCERTAIN" if allow_uncertain else "FALSE_ALARM"

    return {"verdict": verdict, "reason": text[:500]}
