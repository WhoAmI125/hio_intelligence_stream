"""
Scenario-Specific Prompts for VLM Detection

Based on EVA Q2E Multi-Scenario Decomposition approach.
Each prompt focuses on a SINGLE detection task to maximize
VLM attention on specific visual features.

Key principle: "The depth of the question determines the depth of understanding"
- Don't ask "detect cash, violence, and fire"
- Instead, ask each separately with focused criteria

Reference: https://mellerikat.com/blog_tech/Research/multi_scenario
"""

from enum import Enum
from typing import Dict, Optional


class ScenarioType(Enum):
    """Supported scenario types"""
    CASH = "cash"
    VIOLENCE = "violence"
    FIRE = "fire"
    SMOKE = "smoke"


# ============================================================================
# CASH TRANSACTION SCENARIO
# ============================================================================
CASH_SCENARIO = """
[TASK] Analyze if a CASH TRANSACTION is occurring in this CCTV image.
[FOCUS] Only focus on hand movements and money exchange. Ignore everything else.

DETECTION CRITERIA (must see at least 2):
1. Customer's hand holding paper bills or coins
2. Hand-to-hand exchange of money between customer and cashier
3. Cash drawer/register opening with money visible
4. Cashier counting or handling multiple bills
5. Change being returned as bills or coins

EXCLUSION CRITERIA (NOT a cash transaction):
- Plastic card visible (credit/debit card payment)
- Phone/mobile device near terminal (mobile payment)
- Only receipt being exchanged
- No visible money, just hand movements
- Hands not near counter/register area

RESPONSE FORMAT (JSON only):
{
    "is_cash": true or false,
    "confidence": 0.0 to 1.0,
    "evidence": "brief description of what you see",
    "exclusion_match": null or "which exclusion criteria matched"
}
"""


# ============================================================================
# VIOLENCE SCENARIO
# ============================================================================
VIOLENCE_SCENARIO = """
[TASK] Analyze if VIOLENCE or PHYSICAL AGGRESSION is occurring in this CCTV image.
[FOCUS] Only focus on body contact and aggressive movements. Ignore everything else.

DETECTION CRITERIA (must see at least 2):
1. Two or more people in very close physical contact
2. Pushing, hitting, grabbing, or restraining actions
3. Rapid and aggressive arm/leg movements
4. One person falling, ducking, or in defensive posture
5. Facial expressions of anger or distress (if visible)

EXCLUSION CRITERIA (NOT violence):
- Friendly contact: handshake, hug, pat on back
- Normal walking or running
- People standing close but relaxed
- Single person gesturing alone
- Children playing

RESPONSE FORMAT (JSON only):
{
    "is_violence": true or false,
    "confidence": 0.0 to 1.0,
    "evidence": "brief description of what you see",
    "exclusion_match": null or "which exclusion criteria matched"
}
"""


# ============================================================================
# FIRE SCENARIO
# ============================================================================
FIRE_SCENARIO = """
[TASK] Analyze if FIRE or FLAMES are visible in this CCTV image.
[FOCUS] Only focus on fire characteristics: color, movement, brightness. Ignore everything else.

DETECTION CRITERIA (must see at least 1):
1. Bright orange/yellow/red flames visible
2. Irregular, flickering light patterns
3. Flames spreading or growing
4. Objects on fire with visible combustion
5. Intense bright spots with flame-like edges

EXCLUSION CRITERIA (NOT fire):
- Steady light reflections (windows, mirrors)
- Skin tones (people's faces, hands)
- Fixed orange/red colored objects (signs, clothing)
- Sun reflections or lens flare
- LED lights or screens

RESPONSE FORMAT (JSON only):
{
    "is_fire": true or false,
    "confidence": 0.0 to 1.0,
    "evidence": "brief description of what you see",
    "exclusion_match": null or "which exclusion criteria matched"
}
"""


# ============================================================================
# SMOKE SCENARIO
# ============================================================================
SMOKE_SCENARIO = """
[TASK] Analyze if SMOKE is visible in this CCTV image.
[FOCUS] Only focus on smoke characteristics: color, movement, density. Ignore everything else.

DETECTION CRITERIA (must see at least 1):
1. Gray, white, or black wispy patterns rising upward
2. Hazy area with reduced visibility
3. Smoke spreading or dispersing
4. Billowing or swirling cloud-like formations
5. Gradual obscuring of background elements

EXCLUSION CRITERIA (NOT smoke):
- Steam from cooking (white, rising near kitchen)
- Fog or mist (uniform, low-lying)
- Dust clouds (usually brownish, settles down)
- Shadows or dark areas
- Blurry image quality

RESPONSE FORMAT (JSON only):
{
    "is_smoke": true or false,
    "confidence": 0.0 to 1.0,
    "evidence": "brief description of what you see",
    "exclusion_match": null or "which exclusion criteria matched"
}
"""


# ============================================================================
# ZONE-SPECIFIC CONTEXT TEMPLATES
# ============================================================================
ZONE_CONTEXT_TEMPLATE = """
[ZONE CONTEXT]
This image is cropped from the {zone_name} area.
Focus specifically on: {focus_points}

"""

ZONE_CONTEXTS = {
    'cashier': {
        'zone_name': 'cashier/counter',
        'focus_points': 'the counter area, cashier hands, and customer hands'
    },
    'drawer': {
        'zone_name': 'cash drawer/register',
        'focus_points': 'the cash drawer, money handling, and bill insertion'
    },
    'entrance': {
        'zone_name': 'entrance/exit',
        'focus_points': 'people entering or leaving, suspicious behavior'
    },
    'full': {
        'zone_name': 'full frame',
        'focus_points': 'overall scene activity'
    }
}


# ============================================================================
# SCENARIO REGISTRY
# ============================================================================
SCENARIO_PROMPTS: Dict[ScenarioType, str] = {
    ScenarioType.CASH: CASH_SCENARIO,
    ScenarioType.VIOLENCE: VIOLENCE_SCENARIO,
    ScenarioType.FIRE: FIRE_SCENARIO,
    ScenarioType.SMOKE: SMOKE_SCENARIO,
}


def get_scenario_prompt(
    scenario_type: ScenarioType,
    zone: Optional[str] = None,
    custom_context: Optional[str] = None
) -> str:
    """
    Get the prompt for a specific scenario with optional zone context.

    Args:
        scenario_type: The type of scenario to detect
        zone: Optional zone name ('cashier', 'drawer', 'entrance', 'full')
        custom_context: Optional additional context to prepend

    Returns:
        Complete prompt string for the scenario
    """
    base_prompt = SCENARIO_PROMPTS.get(scenario_type, "")

    if not base_prompt:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    # Add zone context if specified
    zone_context = ""
    if zone and zone in ZONE_CONTEXTS:
        zone_info = ZONE_CONTEXTS[zone]
        zone_context = ZONE_CONTEXT_TEMPLATE.format(**zone_info)

    # Build final prompt
    parts = []
    if custom_context:
        parts.append(custom_context)
    if zone_context:
        parts.append(zone_context)
    parts.append(base_prompt)

    return "\n".join(parts)


def get_all_scenarios() -> Dict[str, str]:
    """Get all scenario prompts as a dictionary"""
    return {
        scenario.value: prompt
        for scenario, prompt in SCENARIO_PROMPTS.items()
    }
