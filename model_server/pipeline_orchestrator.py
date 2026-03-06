"""
Scenario Orchestrator - Multi-Scenario Parallel Inference

Implements EVA Q2E 5-stage pipeline:
1. Classification: Determine frame state
2. Decomposition: Split into individual scenarios
3. Enrichment: Inject scenario-specific prompts
4. Clustering: Group by zones
5. Parallel Inference: Run scenarios concurrently

Reference: https://mellerikat.com/blog_tech/Research/multi_scenario
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base_detector import Detection
from .scenarios import ScenarioType, get_scenario_prompt
from .scenarios.base_scenario import (
    BaseScenario,
    ScenarioResult,
    CashScenario,
    ViolenceScenario,
    FireScenario,
    create_scenario
)
from .adapters.base_adapter import BaseVLMAdapter
from .logger import VLMLogger


@dataclass
class OrchestratorConfig:
    """Configuration for ScenarioOrchestrator"""
    # Enabled scenarios
    detect_cash: bool = True
    detect_violence: bool = True
    detect_fire: bool = True

    # Parallel execution
    max_workers: int = 3
    inference_timeout: float = 10.0  # seconds

    # Confidence thresholds
    cash_threshold: float = 0.6
    violence_threshold: float = 0.5
    fire_threshold: float = 0.4

    # Zone settings
    cashier_zone: Optional[List] = None
    drawer_zone: Optional[List] = None

    # Sampling (burst mode)
    in_burst_mode: bool = False

    # Cash-specific routing helpers
    cash_dual_path_enabled: bool = True
    cash_global_assist_threshold: float = 0.30


@dataclass
class OrchestratorResult:
    """Result from orchestrator processing"""
    detections: List[Detection]
    scenario_results: Dict[str, ScenarioResult]
    total_inference_time_ms: float
    frame_timestamp: datetime
    in_burst_mode: bool = False
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'scenario_results': {
                k: v.to_dict() for k, v in self.scenario_results.items()
            },
            'total_inference_time_ms': self.total_inference_time_ms,
            'frame_timestamp': self.frame_timestamp.isoformat(),
            'in_burst_mode': self.in_burst_mode,
            'metadata': self.metadata
        }


class ScenarioOrchestrator:
    """
    Orchestrates multi-scenario detection with parallel VLM inference.

    Key features:
    - EVA Q2E 5-stage pipeline
    - Parallel scenario execution with ThreadPoolExecutor
    - Zone-based cropping for cash detection
    - Automatic threshold-based filtering
    """

    def __init__(
        self,
        vlm_adapter: BaseVLMAdapter,
        config: OrchestratorConfig = None,
        logger: VLMLogger = None
    ):
        """
        Initialize orchestrator.

        Args:
            vlm_adapter: VLM adapter instance (e.g., FlorenceAdapter)
            config: Orchestrator configuration
            logger: VLM logger instance for structured logging
        """
        self.vlm = vlm_adapter
        self.config = config or OrchestratorConfig()
        self.logger = logger

        # Create scenario agents
        self.scenarios: Dict[str, BaseScenario] = {}
        self._init_scenarios()

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Statistics
        self.total_frames = 0
        self.total_detections = 0

        print(f"[Orchestrator] Initialized with {len(self.scenarios)} scenarios")
        print(f"[Orchestrator] Max workers: {self.config.max_workers}")
        if self.logger:
            print(f"[Orchestrator] Logging to: {self.logger.session_dir}")

    def _init_scenarios(self):
        """Initialize scenario agents based on config"""
        if self.config.detect_cash:
            self.scenarios['cash'] = create_scenario(
                ScenarioType.CASH,
                {
                    'confidence_threshold': self.config.cash_threshold,
                    'zone': 'cashier'
                }
            )

        if self.config.detect_violence:
            self.scenarios['violence'] = create_scenario(
                ScenarioType.VIOLENCE,
                {'confidence_threshold': self.config.violence_threshold}
            )

        if self.config.detect_fire:
            self.scenarios['fire'] = create_scenario(
                ScenarioType.FIRE,
                {'confidence_threshold': self.config.fire_threshold}
            )

    def process_frame(
        self,
        frame: np.ndarray,
        zones: Optional[Dict[str, List]] = None
    ) -> OrchestratorResult:
        """
        Process a frame through all enabled scenarios.

        EVA 5-Stage Pipeline:
        1. Classification: Check if frame needs processing
        2. Decomposition: Prepare individual scenario inputs
        3. Enrichment: Add zone context to prompts
        4. Clustering: Group scenarios (currently all parallel)
        5. Parallel Inference: Execute all scenarios concurrently

        Args:
            frame: BGR frame from OpenCV
            zones: Optional zone polygons {'cashier': [...], 'drawer': [...]}

        Returns:
            OrchestratorResult with all detections
        """
        start_time = time.time()
        self.total_frames += 1

        # Merge zones from config and parameter
        effective_zones = {
            'cashier': zones.get('cashier') if zones else self.config.cashier_zone,
            'drawer': zones.get('drawer') if zones else self.config.drawer_zone
        }

        # Stage 1-4: Prepare scenario inputs
        scenario_inputs = self._prepare_scenario_inputs(frame, effective_zones)

        # Stage 5: Parallel inference
        scenario_results = self._run_parallel_inference(scenario_inputs)

        # Convert to detections
        detections = self._results_to_detections(scenario_results)

        total_time = (time.time() - start_time) * 1000
        self.total_detections += len(detections)

        # === Logging ===
        if self.logger:
            # Log each agent result individually
            for name, result in scenario_results.items():
                self.logger.log_agent_inference(
                    scenario_type=name,
                    frame_idx=self.total_frames,
                    result=result.to_dict()
                )

            # Log unified orchestrator decision
            self.logger.log_orchestrator_frame(
                frame_idx=self.total_frames,
                scenario_results={
                    k: v.to_dict() for k, v in scenario_results.items()
                },
                detections=[d.to_dict() for d in detections],
                total_inference_ms=total_time,
                in_burst_mode=self.config.in_burst_mode
            )

        return OrchestratorResult(
            detections=detections,
            scenario_results=scenario_results,
            total_inference_time_ms=total_time,
            frame_timestamp=datetime.now(),
            in_burst_mode=self.config.in_burst_mode,
            metadata={
                'zones_used': list(effective_zones.keys()),
                'scenarios_run': list(scenario_results.keys())
            }
        )

    def _prepare_scenario_inputs(
        self,
        frame: np.ndarray,
        zones: Dict[str, Optional[List]]
    ) -> List[Dict]:
        """
        Prepare inputs for each scenario (Stages 1-4).

        Returns list of {'scenario': name, 'frame': image, 'prompt': str}
        """
        inputs = []

        for name, scenario in self.scenarios.items():
            # Determine which frame/crop to use
            if name == 'cash' and zones.get('cashier'):
                # Crop to cashier zone for cash detection
                cropped, bbox = self.vlm.crop_zone(frame, zones['cashier'])
                input_frame = cropped
                zone_context = 'cashier'
                global_frame = frame if self.config.cash_dual_path_enabled else None
            else:
                # Use full frame for violence/fire
                input_frame = frame
                zone_context = 'full'
                global_frame = None

            # Get enriched prompt with zone context
            prompt = get_scenario_prompt(
                scenario.scenario_type,
                zone=zone_context
            )

            inputs.append({
                'scenario_name': name,
                'scenario': scenario,
                'frame': input_frame,
                'global_frame': global_frame,
                'prompt': prompt,
                'zone': zone_context,
                'cash_dual_path': bool(self.config.cash_dual_path_enabled and name == 'cash' and global_frame is not None),
            })

        return inputs

    def _run_parallel_inference(
        self,
        inputs: List[Dict]
    ) -> Dict[str, ScenarioResult]:
        """
        Run all scenarios in parallel (Stage 5).

        Uses ThreadPoolExecutor for concurrent VLM calls.
        """
        results = {}
        futures = {}

        # Submit all scenarios
        for inp in inputs:
            future = self.executor.submit(
                self._run_single_scenario,
                inp['scenario'],
                inp['frame'],
                inp['prompt'],
                inp['zone'],
                inp.get('global_frame'),
                inp.get('cash_dual_path', False),
            )
            futures[future] = inp['scenario_name']

        # Collect results with timeout
        for future in as_completed(futures, timeout=self.config.inference_timeout + 5):
            scenario_name = futures[future]
            try:
                result = future.result(timeout=self.config.inference_timeout)
                results[scenario_name] = result
            except TimeoutError:
                print(f"[Orchestrator] Timeout for scenario: {scenario_name}")
                results[scenario_name] = ScenarioResult(
                    scenario_type=self.scenarios[scenario_name].scenario_type,
                    is_detected=False,
                    confidence=0.0,
                    evidence="Inference timeout",
                    inference_time_ms=self.config.inference_timeout * 1000
                )
            except Exception as e:
                print(f"[Orchestrator] Error in {scenario_name}: {e}")
                results[scenario_name] = ScenarioResult(
                    scenario_type=self.scenarios[scenario_name].scenario_type,
                    is_detected=False,
                    confidence=0.0,
                    evidence=f"Error: {str(e)}"
                )

        return results

    def _run_single_scenario(
        self,
        scenario: BaseScenario,
        frame: np.ndarray,
        prompt: str,
        zone: str,
        global_frame: np.ndarray = None,
        cash_dual_path: bool = False,
        shared_caption: str = None,
        global_caption: str = None,
    ) -> ScenarioResult:
        """
        Run a single scenario inference.

        If shared_caption is provided, skip VLM inference and analyze
        the pre-generated caption directly (caption sharing optimization).
        """
        from .scenarios.base_scenario import CaptionAnalyzer

        def _as_signal_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, list):
                candidates = value
            elif isinstance(value, (tuple, set)):
                candidates = list(value)
            else:
                candidates = [value]
            out: List[str] = []
            for item in candidates:
                s = str(item).strip()
                if s:
                    out.append(s)
            return out

        start_time = time.time()

        if shared_caption:
            # Caption already generated - just analyze with keywords
            caption = shared_caption
        else:
            # Generate caption via VLM
            caption = self.vlm.infer(frame, "")

        inference_time = (time.time() - start_time) * 1000

        # Analyze caption with scenario-specific keywords
        parsed = CaptionAnalyzer.analyze(caption, scenario.scenario_type)
        is_detected = parsed['is_detected']
        confidence = parsed.get('confidence', 0.0)
        evidence = parsed.get('evidence', '')
        exclusion_match = parsed.get('exclusion_match')
        raw_response = caption
        result_zone = zone
        florence_signals: Dict[str, Any] = {
            'matched_keywords': _as_signal_list(parsed.get('matched_keywords')),
            'object_hints': _as_signal_list(parsed.get('object_hints')),
            'exclusion_match': _as_signal_list(exclusion_match),
            'global_keywords': [],
        }
        metadata: Dict[str, Any] = {
            'florence_signals': florence_signals
        }

        # Cash dual-path:
        # 1) ROI path remains primary
        # 2) If ROI is weak/miss, promote candidate when global handover signal is strong
        if scenario.scenario_type == ScenarioType.CASH and cash_dual_path:
            if global_caption is None and global_frame is not None:
                global_caption = self.vlm.infer(global_frame, "")
            parsed_global = (
                CaptionAnalyzer.analyze(global_caption, scenario.scenario_type)
                if global_caption else
                {
                    'is_detected': False,
                    'confidence': 0.0,
                    'evidence': '',
                    'matched_keywords': [],
                }
            )

            roi_detected = bool(is_detected and confidence >= scenario.confidence_threshold)
            global_conf = float(parsed_global.get('confidence', 0.0) or 0.0)
            global_detected = bool(
                parsed_global.get('is_detected')
                and global_conf >= float(self.config.cash_global_assist_threshold)
            )

            cash_path = 'roi'
            if roi_detected and global_detected:
                cash_path = 'both'
            elif not roi_detected and global_detected:
                cash_path = 'global_assist'

            if cash_path == 'global_assist':
                is_detected = True
                confidence = max(confidence, global_conf, float(scenario.confidence_threshold))
                evidence_parts = []
                if evidence:
                    evidence_parts.append(f"ROI:{evidence}")
                if parsed_global.get('evidence'):
                    evidence_parts.append(f"GLOBAL:{parsed_global['evidence']}")
                evidence = " | ".join(evidence_parts) or parsed_global.get('evidence', '')
                exclusion_match = parsed_global.get('exclusion_match')
                result_zone = 'global_assist'
            elif cash_path == 'both':
                is_detected = True
                confidence = max(confidence, global_conf)
                if parsed_global.get('evidence'):
                    evidence = f"{evidence} | GLOBAL:{parsed_global['evidence']}".strip()
                result_zone = 'both'
            else:
                is_detected = bool(roi_detected)

            global_keywords = _as_signal_list(parsed_global.get('matched_keywords'))
            metadata.update({
                'cash_path': cash_path,
                'roi_confidence': float(parsed.get('confidence', 0.0) or 0.0),
                'global_handover_score': global_conf,
                'global_keywords': global_keywords[:8],
            })
            florence_signals['global_keywords'] = global_keywords
            if global_caption:
                raw_response = f"[ROI]\n{caption}\n\n[GLOBAL]\n{global_caption}"

        florence_signals['exclusion_match'] = _as_signal_list(exclusion_match)

        return ScenarioResult(
            scenario_type=scenario.scenario_type,
            is_detected=is_detected and confidence >= scenario.confidence_threshold,
            confidence=confidence,
            evidence=evidence,
            exclusion_match=exclusion_match,
            zone=result_zone,
            raw_response=raw_response,
            inference_time_ms=inference_time,
            metadata=metadata,
        )

    def _results_to_detections(
        self,
        results: Dict[str, ScenarioResult]
    ) -> List[Detection]:
        """
        Convert scenario results to Detection objects.

        Only includes positive detections above threshold.
        """
        detections = []

        for name, result in results.items():
            if result.is_detected:
                detection = result.to_detection()
                detections.append(detection)

        return detections

    def process_frame_sequential(
        self,
        frame: np.ndarray,
        zones: Optional[Dict[str, List]] = None
    ) -> OrchestratorResult:
        """
        Process frame sequentially with caption sharing optimization.

        Generates caption ONCE, then all scenarios analyze the same caption.
        This saves 2x VLM inference time (1 call instead of 3).
        """
        start_time = time.time()
        self.total_frames += 1

        effective_zones = {
            'cashier': zones.get('cashier') if zones else self.config.cashier_zone,
            'drawer': zones.get('drawer') if zones else self.config.drawer_zone
        }

        scenario_inputs = self._prepare_scenario_inputs(frame, effective_zones)

        # Generate caption ONCE for full frame
        caption_time = time.time()
        shared_caption = self.vlm.infer(frame, "")
        caption_ms = (time.time() - caption_time) * 1000

        # Analyze with each scenario (CPU-only keyword matching, very fast)
        results = {}
        for inp in scenario_inputs:
            # For cash with cashier zone: generate separate cropped caption
            if inp['scenario_name'] == 'cash' and effective_zones.get('cashier'):
                cropped_caption = self.vlm.infer(inp['frame'], "")
                result = self._run_single_scenario(
                    inp['scenario'], inp['frame'],
                    inp['prompt'], inp['zone'],
                    inp.get('global_frame'),
                    inp.get('cash_dual_path', False),
                    shared_caption=cropped_caption,
                    global_caption=shared_caption if inp.get('cash_dual_path', False) else None,
                )
            else:
                result = self._run_single_scenario(
                    inp['scenario'], inp['frame'],
                    inp['prompt'], inp['zone'],
                    inp.get('global_frame'),
                    inp.get('cash_dual_path', False),
                    shared_caption=shared_caption
                )
            results[inp['scenario_name']] = result

        detections = self._results_to_detections(results)
        total_time = (time.time() - start_time) * 1000

        # === Logging ===
        if self.logger:
            for name, result in results.items():
                self.logger.log_agent_inference(
                    scenario_type=name,
                    frame_idx=self.total_frames,
                    result=result.to_dict()
                )

            self.logger.log_orchestrator_frame(
                frame_idx=self.total_frames,
                scenario_results={
                    k: v.to_dict() for k, v in results.items()
                },
                detections=[d.to_dict() for d in detections],
                total_inference_ms=total_time,
                in_burst_mode=self.config.in_burst_mode
            )

        return OrchestratorResult(
            detections=detections,
            scenario_results=results,
            total_inference_time_ms=total_time,
            frame_timestamp=datetime.now(),
            in_burst_mode=self.config.in_burst_mode,
            metadata={'caption_ms': caption_ms, 'shared_caption': shared_caption[:200]}
        )

    def set_burst_mode(self, enabled: bool):
        """Enable or disable burst mode"""
        self.config.in_burst_mode = enabled

    def update_zones(
        self,
        cashier_zone: Optional[List] = None,
        drawer_zone: Optional[List] = None
    ):
        """Update zone configurations"""
        if cashier_zone is not None:
            self.config.cashier_zone = cashier_zone
        if drawer_zone is not None:
            self.config.drawer_zone = drawer_zone

    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        scenario_stats = {
            name: scenario.get_stats()
            for name, scenario in self.scenarios.items()
        }

        return {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'detection_rate': (
                self.total_detections / self.total_frames
                if self.total_frames > 0 else 0
            ),
            'scenarios': scenario_stats,
            'vlm_stats': self.vlm.get_stats() if self.vlm else {}
        }

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)
        if self.vlm:
            self.vlm.cleanup()


def create_orchestrator(
    vlm_config: Dict = None,
    orchestrator_config: Dict = None
) -> ScenarioOrchestrator:
    """
    Factory function to create a complete orchestrator.

    Args:
        vlm_config: Configuration for VLM adapter
        orchestrator_config: Configuration for orchestrator

    Returns:
        Initialized ScenarioOrchestrator
    """
    from .adapters.florence_adapter import FlorenceAdapter

    # Create VLM adapter
    vlm = FlorenceAdapter(vlm_config or {})
    vlm.initialize()

    # Create orchestrator config
    config = OrchestratorConfig()
    if orchestrator_config:
        for key, value in orchestrator_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return ScenarioOrchestrator(vlm, config)
