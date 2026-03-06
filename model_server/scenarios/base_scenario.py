"""
Base Scenario - Abstract base class for scenario agents

Each scenario focuses on a single detection task following the
EVA Q2E decomposition principle.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .prompts import ScenarioType, get_scenario_prompt


@dataclass
class ScenarioResult:
    """Result from a single scenario detection"""
    scenario_type: ScenarioType
    is_detected: bool
    confidence: float
    evidence: str
    exclusion_match: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    zone: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None
    inference_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'scenario_type': self.scenario_type.value,
            'is_detected': self.is_detected,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'exclusion_match': self.exclusion_match,
            'bbox': self.bbox,
            'zone': self.zone,
            'timestamp': self.timestamp.isoformat(),
            'raw_response': self.raw_response,
            'inference_time_ms': self.inference_time_ms,
            'metadata': self.metadata
        }

    def to_detection(self):
        """Convert to Detection object for compatibility"""
        from ..base_detector import Detection

        # Map scenario type to label
        label_map = {
            ScenarioType.CASH: 'CASH',
            ScenarioType.VIOLENCE: 'VIOLENCE',
            ScenarioType.FIRE: 'FIRE',
            ScenarioType.SMOKE: 'SMOKE'
        }

        return Detection(
            label=label_map.get(self.scenario_type, self.scenario_type.value.upper()),
            confidence=self.confidence,
            bbox=self.bbox or (0, 0, 0, 0),
            event_type=self.scenario_type.value,
            timestamp=self.timestamp,
            metadata={
                'source': 'vlm_scenario',
                'evidence': self.evidence,
                'exclusion_match': self.exclusion_match,
                'zone': self.zone,
                'inference_time_ms': self.inference_time_ms,
                **self.metadata
            }
        )


class BaseScenario(ABC):
    """
    Abstract base class for scenario agents.

    Each scenario agent is responsible for detecting ONE type of event
    with focused attention on specific visual features.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize scenario agent.

        Args:
            config: Configuration dictionary with keys:
                - confidence_threshold: Minimum confidence to report detection
                - zone: Zone name for context ('cashier', 'drawer', etc.)
                - custom_context: Additional context to add to prompt
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.zone = self.config.get('zone', None)
        self.custom_context = self.config.get('custom_context', None)

        # Statistics
        self.total_inferences = 0
        self.positive_detections = 0
        self.total_inference_time_ms = 0.0

    @property
    @abstractmethod
    def scenario_type(self) -> ScenarioType:
        """Return the scenario type this agent handles"""
        pass

    def get_prompt(self) -> str:
        """Get the prompt for this scenario with zone context"""
        return get_scenario_prompt(
            self.scenario_type,
            zone=self.zone,
            custom_context=self.custom_context
        )

    @abstractmethod
    def process(
        self,
        frame: np.ndarray,
        vlm_adapter: Any
    ) -> ScenarioResult:
        """
        Process a frame for this scenario.

        Args:
            frame: BGR frame from OpenCV
            vlm_adapter: VLM adapter instance (e.g., FlorenceAdapter)

        Returns:
            ScenarioResult with detection details
        """
        pass

    def parse_vlm_response(self, response: str) -> Dict:
        """
        Parse VLM JSON response.

        Args:
            response: Raw text response from VLM

        Returns:
            Parsed dictionary or default values on error
        """
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith('```'):
                lines = response.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```') and not in_json:
                        in_json = True
                        continue
                    elif line.startswith('```') and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = '\n'.join(json_lines)

            # Find JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)

        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        # Return default on parse failure
        return {
            'is_detected': False,
            'confidence': 0.0,
            'evidence': 'Failed to parse VLM response',
            'exclusion_match': None
        }

    def _map_response_key(self, parsed: Dict) -> bool:
        """Map scenario-specific response key to is_detected"""
        key_map = {
            ScenarioType.CASH: 'is_cash',
            ScenarioType.VIOLENCE: 'is_violence',
            ScenarioType.FIRE: 'is_fire',
            ScenarioType.SMOKE: 'is_smoke'
        }

        key = key_map.get(self.scenario_type, 'is_detected')
        return parsed.get(key, parsed.get('is_detected', False))

    def update_stats(self, result: ScenarioResult):
        """Update internal statistics"""
        self.total_inferences += 1
        self.total_inference_time_ms += result.inference_time_ms
        if result.is_detected:
            self.positive_detections += 1

    def get_stats(self) -> Dict:
        """Get statistics summary"""
        avg_time = (
            self.total_inference_time_ms / self.total_inferences
            if self.total_inferences > 0 else 0.0
        )
        detection_rate = (
            self.positive_detections / self.total_inferences
            if self.total_inferences > 0 else 0.0
        )

        return {
            'scenario_type': self.scenario_type.value,
            'total_inferences': self.total_inferences,
            'positive_detections': self.positive_detections,
            'detection_rate': detection_rate,
            'avg_inference_time_ms': avg_time
        }


class CaptionAnalyzer:
    """
    Analyze Florence-2 captions with scenario-specific keyword matching.

    Florence-2 generates scene descriptions, not JSON responses.
    This analyzer extracts detection signals from captions using
    word-boundary-aware keyword matching per scenario.

    v2 Changes (2026-02-06):
    - Word boundary matching (regex \\b) to prevent substring FP
      e.g. "white" no longer matches "hit", "billboard" no longer matches "bill"
    - Removed ambiguous keywords: "hit" (too common in captions)
    - Added hotel CCTV context keywords (front desk, lobby, reception)
    - Added multi-word phrase support for stronger signals
    """

    # Scenario-specific keyword groups with weights
    KEYWORDS = {
        # CASH scenario = H2H Transaction Detection (Tier1)
        # 목적: 카운터에서 물건을 주고받는 모든 상호작용을 탐지 → Tier2(Gemini)가 현금/카드 판별
        # Florence-2는 "cash"를 직접 인식 못함 → "phone", "paper"로 오인
        # 따라서 "무엇을"이 아니라 "주고받는 동작 자체"를 탐지
        # object_hints: Florence-2가 묘사한 물체 디테일을 Gemini에 전달
        ScenarioType.CASH: {
            'strong_positive': [
                # 직접적 현금/결제 언급 (Florence-2가 가끔 잡음)
                'cash', 'money', 'banknote', 'banknotes',
                'currency', 'dollar', 'won', 'coins',
                'paying', 'payment', 'transaction',
                'cash register', 'paper money', 'bill', 'bills',
            ],
            'moderate_positive': [
                # 장소 신호 (카운터/프론트)
                'counter', 'cashier', 'checkout',
                'front desk', 'reception',
                'drawer', 'receipt',
                # 손에 물체를 들고 있는 동작 (h2h 핵심)
                'handing', 'holding', 'passing', 'reaching',
                'exchanging', 'giving', 'receiving',
                # 지갑/소지품
                'wallet', 'purse', 'envelope',
            ],
            'context_phrases': [
                # H2H 전달 동작 (Florence-2가 실제 생성하는 표현)
                'holding a piece of paper', 'holding piece of paper',
                'handing a piece of paper', 'passing a piece of paper',
                'holding a small', 'holding a black',
                'holding a brown', 'holding a white',
                'holding something', 'handing something',
                'passing something', 'giving something',
                # 지갑/주머니 동작
                'reaching into wallet', 'pulling out wallet',
                'taking out wallet', 'opening wallet',
                'reaching into pocket', 'pulling from pocket',
                'taking something out', 'pulling something out',
                # 직접적 현금 동작
                'handing over', 'hands over', 'giving money',
                'receiving money', 'counting money', 'cash drawer',
                'counting bills', 'holding money', 'holding cash',
                # 서랍/금전등록기
                'opening drawer', 'open drawer', 'opening the drawer',
                'putting into drawer', 'taking from drawer',
                'folded paper', 'small rectangular',
                'counting something', 'flipping through',
            ],
            # Florence-2가 묘사한 물체 힌트 → Gemini에 전달 (점수에는 미반영)
            # 현금과 혼동 가능한 물체만 포함 (car key, pen 등 불필요한 것 제외 → 토큰 절약)
            'object_hints': [
                # 종이/지폐 계열 (Florence-2가 현금을 이렇게 묘사)
                'paper', 'piece of paper', 'folded paper',
                'envelope', 'receipt', 'document',
                # 직접적 현금
                'cash', 'money', 'bill', 'bills', 'banknote',
                'coin', 'coins', 'change',
                # 카드 계열
                'card', 'credit card', 'debit card',
                # 지갑 (지갑 꺼내는 동작 = 결제 신호)
                'wallet', 'purse',
                # Florence-2가 현금을 자주 오인하는 대상
                'phone', 'mobile',
                'remote', 'remote control',
                'small object', 'black object', 'object',
            ],
            'negative': [
                # phone/mobile 제거 — Florence-2가 현금을 "phone"으로 자주 오인
                # Tier2(Gemini)가 phone vs cash 구별 담당
                'credit card', 'debit card', 'card reader',
                'swipe', 'contactless', 'terminal',
            ],
            'neutralizing_phrases': {
                'cash': [
                    'cash register',
                ],
                'cash register': [
                    'cash register',
                ],
                'bill': [
                    'billboard',
                ],
            },
            'weights': {'strong': 0.3, 'moderate': 0.1, 'context': 0.3, 'negative': -0.3}
        },
        ScenarioType.VIOLENCE: {
            'strong_positive': [
                'fight', 'fighting', 'punch', 'punching',
                'attack', 'attacking', 'struggle', 'struggling',
                'violent', 'violence', 'assault', 'assaulting',
                'shove', 'shoving', 'slap', 'slapping', 'kick', 'kicking',
            ],
            'moderate_positive': [
                'aggressive', 'angry', 'confrontation', 'conflict',
                'restrain', 'threaten', 'threatening', 'yelling',
                'screaming', 'falling down', 'knocked',
            ],
            'context_phrases': [
                'hitting someone', 'hitting a person', 'hitting him',
                'hitting her', 'pushing someone', 'pushing a person',
                'grabbing someone', 'pulling hair',
                'throwing punch', 'physical altercation',
            ],
            'negative': [
                'handshake', 'hug', 'hugging', 'friendly',
                'greeting', 'playing', 'children',
                'laughing', 'smiling', 'waving',
                'keyboard', 'button', 'typing',
            ],
            'weights': {'strong': 0.35, 'moderate': 0.15, 'context': 0.45, 'negative': -0.25}
        },
        ScenarioType.FIRE: {
            'strong_positive': [
                'fire', 'flame', 'flames', 'burning', 'blaze',
                'smoke', 'ignite', 'combustion', 'inferno',
            ],
            'moderate_positive': [
                'orange glow', 'red glow', 'haze', 'hazy',
                'emergency', 'sprinkler',
                'charred', 'scorched', 'smoldering',
            ],
            'context_phrases': [
                'on fire', 'catching fire', 'thick smoke',
                'smoke rising', 'smoke coming', 'smoke billowing',
                'flames spreading',
            ],
            'neutralizing_phrases': {
                'fire': [
                    'fire extinguisher', 'fire exit', 'fire escape',
                    'fire alarm', 'fire department', 'fire station',
                    'fire hydrant', 'fire truck', 'fire safety',
                    'fire door', 'fire hose', 'fire prevention',
                ],
            },
            'negative': [
                'lamp', 'screen', 'monitor', 'reflection',
                'sunset', 'warm lighting', 'neon', 'candle',
                'fireplace', 'cigarette', 'no smoking sign',
                'extinguisher',
            ],
            'weights': {'strong': 0.4, 'moderate': 0.15, 'context': 0.5, 'negative': -0.2}
        }
    }

    # Pre-compiled word boundary patterns (lazy init)
    _compiled_patterns: Dict = {}

    @classmethod
    def _get_pattern(cls, keyword: str) -> re.Pattern:
        """Get or compile a word-boundary regex pattern for a keyword."""
        if keyword not in cls._compiled_patterns:
            escaped = re.escape(keyword)
            cls._compiled_patterns[keyword] = re.compile(
                rf'\b{escaped}\b', re.IGNORECASE
            )
        return cls._compiled_patterns[keyword]

    @classmethod
    def _match_keywords(cls, caption_lower: str, keyword_list: List[str]) -> List[str]:
        """Match keywords using word boundaries to prevent substring FP."""
        matches = []
        for kw in keyword_list:
            pattern = cls._get_pattern(kw)
            if pattern.search(caption_lower):
                matches.append(kw)
        return matches

    @classmethod
    def analyze(cls, caption: str, scenario_type: ScenarioType) -> Dict:
        """
        Analyze caption for scenario-specific signals.

        Uses word-boundary matching to prevent substring false positives.
        Supports multi-word context phrases for stronger detection signals.

        Returns:
            {
                'is_detected': bool,
                'confidence': float (0-1),
                'evidence': str,
                'matched_keywords': list,
                'exclusion_match': str or None
            }
        """
        caption_lower = caption.lower()
        keywords = cls.KEYWORDS.get(scenario_type, {})
        weights = keywords.get('weights', {
            'strong': 0.3, 'moderate': 0.15, 'context': 0.4, 'negative': -0.2
        })

        # Word-boundary keyword matches
        strong_matches = cls._match_keywords(caption_lower, keywords.get('strong_positive', []))
        moderate_matches = cls._match_keywords(caption_lower, keywords.get('moderate_positive', []))
        context_matches = cls._match_keywords(caption_lower, keywords.get('context_phrases', []))
        negative_matches = cls._match_keywords(caption_lower, keywords.get('negative', []))

        # Neutralize strong keywords that only appear in safe compound phrases
        # e.g. "fire" in "fire extinguisher" should NOT trigger fire detection
        neutralizing = keywords.get('neutralizing_phrases', {})
        if neutralizing:
            neutralized = []
            for kw in strong_matches:
                if kw in neutralizing:
                    # Check if ALL occurrences of this keyword are inside a neutralizing phrase
                    safe_phrases = cls._match_keywords(caption_lower, neutralizing[kw])
                    if safe_phrases:
                        # Mask out neutralizing phrases, then check if keyword still appears
                        masked = caption_lower
                        for phrase in safe_phrases:
                            masked = masked.replace(phrase, ' ' * len(phrase))
                        pattern = cls._get_pattern(kw)
                        if not pattern.search(masked):
                            neutralized.append(kw)
            for kw in neutralized:
                strong_matches.remove(kw)

        # Calculate confidence score
        score = (
            len(strong_matches) * weights['strong'] +
            len(moderate_matches) * weights['moderate'] +
            len(context_matches) * weights.get('context', 0.4) +
            len(negative_matches) * weights['negative']
        )

        # Clamp to 0-1
        confidence = max(0.0, min(1.0, score))

        # H2H detection: location + action moderate combination also counts
        # (카운터에서 물건을 들고있으면 h2h로 판단 → Tier2가 현금/카드 판별)
        _LOCATION_KEYWORDS = {'counter', 'cashier', 'checkout', 'front desk', 'reception', 'drawer'}
        _ACTION_KEYWORDS = {'handing', 'holding', 'passing', 'reaching', 'exchanging', 'giving', 'receiving'}
        has_location = bool(_LOCATION_KEYWORDS & set(moderate_matches))
        has_action = bool(_ACTION_KEYWORDS & set(moderate_matches))
        has_h2h = has_location and has_action

        has_signal = len(strong_matches) > 0 or len(context_matches) > 0 or has_h2h
        is_detected = confidence > 0 and has_signal

        # Extract object hints (what Florence-2 thinks the held object is)
        # These don't affect score — they're passed to Tier2 for context
        object_hints_list = keywords.get('object_hints', [])
        object_hints = cls._match_keywords(caption_lower, object_hints_list) if object_hints_list else []

        # Build evidence
        all_matches = strong_matches + context_matches + moderate_matches
        evidence_parts = []
        if all_matches:
            evidence_parts.append(f"Keywords: {', '.join(all_matches[:5])}")
        if object_hints:
            evidence_parts.append(f"Objects: {', '.join(object_hints[:5])}")
        if negative_matches:
            evidence_parts.append(f"Exclusions: {', '.join(negative_matches[:3])}")
        evidence_parts.append(f"Caption: {caption[:150]}")

        return {
            'is_detected': is_detected,
            'confidence': round(confidence, 3),
            'evidence': ' | '.join(evidence_parts),
            'matched_keywords': all_matches,
            'object_hints': object_hints,
            'exclusion_match': ', '.join(negative_matches) if negative_matches else None
        }

    @classmethod
    def analyze_multi(cls, captions: List[str], scenario_type: ScenarioType) -> Dict:
        """
        Analyze multiple captions (from the same video) with cross-frame signal accumulation.

        Different frames may capture different moments of the same event:
        - Frame A: "standing in front of a counter" (moderate: counter)
        - Frame B: "holding a piece of paper" (context: holding a piece of paper)
        Combined: counter + holding paper = stronger signal than either alone.

        Logic:
        - Union of unique keywords across all frames (no double-counting)
        - Score based on combined unique keywords
        - Negative keywords also unioned (any frame with "phone" counts once)
        - Per-frame results also returned for detail logging
        """
        keywords = cls.KEYWORDS.get(scenario_type, {})
        weights = keywords.get('weights', {
            'strong': 0.3, 'moderate': 0.15, 'context': 0.4, 'negative': -0.2
        })

        # Accumulate unique keywords across all frames
        all_strong = set()
        all_moderate = set()
        all_context = set()
        all_negative = set()
        all_object_hints = set()
        per_frame = []

        object_hints_list = keywords.get('object_hints', [])

        for caption in captions:
            frame_result = cls.analyze(caption, scenario_type)
            per_frame.append(frame_result)

            caption_lower = caption.lower()
            strong = cls._match_keywords(caption_lower, keywords.get('strong_positive', []))
            moderate = cls._match_keywords(caption_lower, keywords.get('moderate_positive', []))
            context = cls._match_keywords(caption_lower, keywords.get('context_phrases', []))
            negative = cls._match_keywords(caption_lower, keywords.get('negative', []))

            # Collect object hints
            if object_hints_list:
                hints = cls._match_keywords(caption_lower, object_hints_list)
                all_object_hints.update(hints)

            # Apply neutralization
            neutralizing = keywords.get('neutralizing_phrases', {})
            if neutralizing:
                neutralized = []
                for kw in strong:
                    if kw in neutralizing:
                        safe_phrases = cls._match_keywords(caption_lower, neutralizing[kw])
                        if safe_phrases:
                            masked = caption_lower
                            for phrase in safe_phrases:
                                masked = masked.replace(phrase, ' ' * len(phrase))
                            pattern = cls._get_pattern(kw)
                            if not pattern.search(masked):
                                neutralized.append(kw)
                for kw in neutralized:
                    strong.remove(kw)

            all_strong.update(strong)
            all_moderate.update(moderate)
            all_context.update(context)
            all_negative.update(negative)

        # Calculate video-level score from unique keywords
        score = (
            len(all_strong) * weights['strong'] +
            len(all_moderate) * weights['moderate'] +
            len(all_context) * weights.get('context', 0.4) +
            len(all_negative) * weights['negative']
        )
        confidence = max(0.0, min(1.0, score))

        # H2H: location + action across frames
        _LOCATION_KEYWORDS = {'counter', 'cashier', 'checkout', 'front desk', 'reception', 'drawer'}
        _ACTION_KEYWORDS = {'handing', 'holding', 'passing', 'reaching', 'exchanging', 'giving', 'receiving'}
        has_location = bool(_LOCATION_KEYWORDS & all_moderate)
        has_action = bool(_ACTION_KEYWORDS & all_moderate)
        has_h2h = has_location and has_action

        has_signal = len(all_strong) > 0 or len(all_context) > 0 or has_h2h
        is_detected = confidence > 0 and has_signal

        all_matches = sorted(all_strong) + sorted(all_context) + sorted(all_moderate)
        return {
            'is_detected': is_detected,
            'confidence': round(confidence, 3),
            'matched_keywords': all_matches,
            'strong': sorted(all_strong),
            'context': sorted(all_context),
            'moderate': sorted(all_moderate),
            'negative': sorted(all_negative),
            'object_hints': sorted(all_object_hints),
            'per_frame': per_frame,
            'num_frames': len(captions),
        }


class CashScenario(BaseScenario):
    """Cash transaction detection scenario"""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.CASH

    def process(self, frame: np.ndarray, vlm_adapter: Any) -> ScenarioResult:
        import time

        start_time = time.time()

        # Florence-2: Generate detailed caption (not free-form prompt)
        caption = vlm_adapter.infer(frame, "")

        inference_time = (time.time() - start_time) * 1000

        # Analyze caption with scenario-specific keywords
        parsed = CaptionAnalyzer.analyze(caption, self.scenario_type)
        is_detected = parsed['is_detected']

        result = ScenarioResult(
            scenario_type=self.scenario_type,
            is_detected=is_detected and parsed.get('confidence', 0) >= self.confidence_threshold,
            confidence=parsed.get('confidence', 0.0),
            evidence=parsed.get('evidence', ''),
            exclusion_match=parsed.get('exclusion_match'),
            zone=self.zone,
            raw_response=caption,
            inference_time_ms=inference_time
        )

        self.update_stats(result)
        return result


class ViolenceScenario(BaseScenario):
    """Violence detection scenario"""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.VIOLENCE

    def process(self, frame: np.ndarray, vlm_adapter: Any) -> ScenarioResult:
        import time

        start_time = time.time()
        caption = vlm_adapter.infer(frame, "")
        inference_time = (time.time() - start_time) * 1000

        parsed = CaptionAnalyzer.analyze(caption, self.scenario_type)
        is_detected = parsed['is_detected']

        result = ScenarioResult(
            scenario_type=self.scenario_type,
            is_detected=is_detected and parsed.get('confidence', 0) >= self.confidence_threshold,
            confidence=parsed.get('confidence', 0.0),
            evidence=parsed.get('evidence', ''),
            exclusion_match=parsed.get('exclusion_match'),
            zone=self.zone,
            raw_response=caption,
            inference_time_ms=inference_time
        )

        self.update_stats(result)
        return result


class FireScenario(BaseScenario):
    """Fire detection scenario"""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.FIRE

    def process(self, frame: np.ndarray, vlm_adapter: Any) -> ScenarioResult:
        import time

        start_time = time.time()
        caption = vlm_adapter.infer(frame, "")
        inference_time = (time.time() - start_time) * 1000

        parsed = CaptionAnalyzer.analyze(caption, self.scenario_type)
        is_detected = parsed['is_detected']

        result = ScenarioResult(
            scenario_type=self.scenario_type,
            is_detected=is_detected and parsed.get('confidence', 0) >= self.confidence_threshold,
            confidence=parsed.get('confidence', 0.0),
            evidence=parsed.get('evidence', ''),
            exclusion_match=parsed.get('exclusion_match'),
            zone=self.zone,
            raw_response=caption,
            inference_time_ms=inference_time
        )

        self.update_stats(result)
        return result


# Scenario factory
SCENARIO_CLASSES = {
    ScenarioType.CASH: CashScenario,
    ScenarioType.VIOLENCE: ViolenceScenario,
    ScenarioType.FIRE: FireScenario,
}


def create_scenario(scenario_type: ScenarioType, config: Dict = None) -> BaseScenario:
    """
    Create a scenario agent instance.

    Args:
        scenario_type: Type of scenario
        config: Configuration dictionary

    Returns:
        Scenario agent instance
    """
    scenario_class = SCENARIO_CLASSES.get(scenario_type)
    if not scenario_class:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return scenario_class(config)
