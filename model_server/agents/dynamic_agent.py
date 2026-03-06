"""
Dynamic Agent — Unified two-tier (Florence → Gemini) detection pipeline.

Fixes from stub version:
1. Uses florence_adapter.infer() (not caption())
2. CaptionAnalyzer.analyze() is called correctly as a static method
3. Uncertainty Gate added: confidence + stability + event priority → decide Tier2
4. Real GeminiValidator call path wired in
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List

from model_server.scenarios.base_scenario import CaptionAnalyzer
from model_server.scenarios import ScenarioType
from model_server.adapters.florence_adapter import FlorenceAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Uncertainty Gate – ported from EpisodeManager.should_send_to_tier2 +
#                     EvidenceRouter._score_actions_heuristic core logic
# ---------------------------------------------------------------------------

class UncertaintyGate:
    """
    Decides whether a Tier-1 result is confident enough to accept directly
    or should be escalated to Tier-2 (Gemini).

    Ported thresholds from the existing pipeline:
    - Fire / Violence → always escalate (safety-critical)
    - Cash with high confidence AND high stability → skip Tier-2
    - Everything else → escalate
    """

    # Thresholds (mirror EpisodeManager.should_send_to_tier2 defaults)
    STABILITY_SKIP = 0.90
    CONFIDENCE_SKIP = 0.85

    # Per-scenario Tier-2 confidence thresholds (from EvidenceRouter config)
    TIER2_THRESHOLDS = {
        'fire':     0.60,
        'violence': 0.70,
        'cash':     0.55,
    }

    @classmethod
    def should_escalate(
        cls,
        scenario_name: str,
        tier1_result: Dict[str, Any],
        stability: float = 0.0,
    ) -> bool:
        """
        Returns True if the result should go to Tier-2 for validation.

        Args:
            scenario_name: 'cash', 'fire', 'violence'
            tier1_result:  dict from CaptionAnalyzer.analyze()
            stability:     label stability score (0–1), 0 if single-frame
        """
        # Safety-critical events → always escalate
        if scenario_name in ('fire', 'violence'):
            return True

        confidence = float(tier1_result.get('confidence', 0.0))
        is_detected = bool(tier1_result.get('is_detected', False))

        # Nothing detected → no point escalating
        if not is_detected:
            return False

        # High confidence + high stability → safe to skip Tier-2
        if confidence >= cls.CONFIDENCE_SKIP and stability >= cls.STABILITY_SKIP:
            return False

        # Below the scenario threshold → not worth sending
        threshold = cls.TIER2_THRESHOLDS.get(scenario_name, 0.55)
        if confidence < threshold:
            return False

        # Anything in the "uncertain middle" → escalate
        return True


# ---------------------------------------------------------------------------
# DynamicAgent
# ---------------------------------------------------------------------------

class DynamicAgent:
    """
    Data-driven agent that ties Florence-2 (Tier 1) and Gemini (Tier 2)
    together based on a markdown prompt file containing scenario rules.

    Data flow (matches legacy pipeline):
        Frame → Florence.infer() → CaptionAnalyzer.analyze()
                    │
                    ├─ confident → save event directly
                    └─ uncertain → GeminiValidator.validate_event_evidence()
                                        → final decision
    """

    def __init__(self, scenario_name: str, prompts_dir: str = "prompts"):
        self.scenario_name = scenario_name
        self.prompts_dir = prompts_dir
        self.prompt_path = os.path.join(prompts_dir, f"{scenario_name}.md")
        self.prompt_config = self._load_md_prompt()
        self._prompt_mtime: float = self._get_file_mtime()

        # Map to the enum used by CaptionAnalyzer keyword engine
        self.scenario_type = ScenarioType[scenario_name.upper()]

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _load_md_prompt(self) -> str:
        """Loads the scenario rules from the markdown file."""
        if not os.path.exists(self.prompt_path):
            logger.warning(f"Prompt file not found: {self.prompt_path}. Using empty.")
            return ""
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_file_mtime(self) -> float:
        """Get last modification time of the prompt file."""
        if os.path.exists(self.prompt_path):
            return os.path.getmtime(self.prompt_path)
        return 0.0

    def _check_prompt_reload(self) -> None:
        """Hot-reload prompt if .md file has been modified (e.g. by RuleUpdater)."""
        current_mtime = self._get_file_mtime()
        if current_mtime > self._prompt_mtime:
            old = self.prompt_config[:60] if self.prompt_config else '(empty)'
            self.prompt_config = self._load_md_prompt()
            self._prompt_mtime = current_mtime
            logger.info(
                f"[DynamicAgent:{self.scenario_name}] Prompt hot-reloaded. "
                f"mtime={current_mtime:.0f}"
            )

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(
        self,
        frame: Any,
        florence_adapter: FlorenceAdapter,
        gemini_validator: Any = None,
        *,
        stability: float = 0.0,
        episode_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the two-tier agent pipeline.

        Tier 1: Florence caption → CaptionAnalyzer keyword matching
        Tier 2: Gemini validation (only if Uncertainty Gate fires)

        Args:
            frame:              BGR np.ndarray from OpenCV
            florence_adapter:   Initialised FlorenceAdapter
            gemini_validator:   Optional GeminiValidator (Tier 2)
            stability:          Label stability score (0–1) from EpisodeManager
            episode_metadata:   Extra metadata from the episode (e.g. h2h_peak)

        Returns:
            dict with keys: is_detected, confidence, evidence,
                            tier, scenario, matched_keywords, ...
        """
        t0 = time.time()

        # Hot-reload prompt if modified
        self._check_prompt_reload()

        # ----- Tier 1: Florence-2 caption -----
        try:
            caption = florence_adapter.infer(frame, "")   # ← correct call
        except Exception as e:
            logger.error(f"Florence-2 inference failed: {e}")
            return self._error_result("tier1_inference_failed", str(e))

        # Caption → keyword analysis
        tier1_result = CaptionAnalyzer.analyze(caption, self.scenario_type)
        tier1_result['caption'] = caption
        tier1_result['tier'] = 1
        tier1_result['scenario'] = self.scenario_name
        tier1_result['tier1_time_ms'] = round((time.time() - t0) * 1000, 1)

        # ----- Uncertainty Gate -----
        needs_tier2 = UncertaintyGate.should_escalate(
            self.scenario_name,
            tier1_result,
            stability=stability,
        )

        if not needs_tier2:
            tier1_result['needs_validation'] = False
            tier1_result['router_action'] = 'SKIP'
            return tier1_result

        # ----- Tier 2: Gemini validation -----
        if gemini_validator is None:
            logger.warning("Tier-2 escalation requested but no gemini_validator provided.")
            tier1_result['needs_validation'] = True
            tier1_result['router_action'] = 'GEMINI_SKIPPED_NO_VALIDATOR'
            return tier1_result

        logger.info(
            f"[{self.scenario_name}] Uncertainty Gate fired "
            f"(conf={tier1_result.get('confidence', 0):.2f}, "
            f"stab={stability:.2f}). Sending to Gemini."
        )

        try:
            t1 = time.time()
            # Build a lightweight evidence packet dict for GeminiValidator
            evidence = self._build_evidence_packet(tier1_result, episode_metadata)

            is_valid, gem_conf, reason, corrected_type = (
                gemini_validator.validate_event_evidence(
                    packet=evidence,
                    mode="hybrid",
                    frame=frame,
                )
            )

            tier2_time_ms = round((time.time() - t1) * 1000, 1)

            return {
                'is_detected': is_valid,
                'confidence': gem_conf,
                'evidence': reason,
                'tier': 2,
                'scenario': self.scenario_name,
                'corrected_event_type': corrected_type,
                'needs_validation': False,
                'router_action': 'GEMINI_IMG',
                'tier1_result': tier1_result,
                'tier1_time_ms': tier1_result.get('tier1_time_ms', 0),
                'tier2_time_ms': tier2_time_ms,
                'total_time_ms': round((time.time() - t0) * 1000, 1),
            }

        except Exception as e:
            logger.error(f"Gemini validation failed: {e}")
            # On Gemini failure → fallback to Tier-1 result (graceful degradation)
            tier1_result['needs_validation'] = True
            tier1_result['router_action'] = 'GEMINI_ERROR'
            tier1_result['gemini_error'] = str(e)
            return tier1_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_evidence_packet(
        self,
        tier1_result: Dict[str, Any],
        episode_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a dict matching the EvidencePacket schema expected by GeminiValidator."""
        meta = episode_metadata or {}
        return {
            'event_type': self.scenario_name,
            'tier1_confidence': tier1_result.get('confidence', 0.0),
            'stability_score': meta.get('stability', 0.0),
            'detection_count': meta.get('detection_count', 1),
            'router_action': 'GEMINI_IMG',
            'router_reason': 'uncertainty_gate',
            'florence_signals': {
                'matched_keywords': tier1_result.get('matched_keywords', []),
                'object_hints': tier1_result.get('object_hints', []),
                'exclusion_match': tier1_result.get('exclusion_match', []),
                'global_keywords': [],
            },
            'focus_hints': [],
        }

    @staticmethod
    def _error_result(code: str, detail: str = "") -> Dict[str, Any]:
        return {
            'is_detected': False,
            'confidence': 0.0,
            'evidence': f"Error [{code}]: {detail}",
            'tier': 0,
            'scenario': '',
            'needs_validation': False,
            'router_action': 'ERROR',
        }
