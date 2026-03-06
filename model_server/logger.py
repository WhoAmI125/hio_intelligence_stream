"""
VLM Detection Logger

Structured logging system for multi-scenario VLM detection.
Saves 3 types of logs:

1. Agent Logs (per scenario):
   - logs/agent_cash.jsonl
   - logs/agent_violence.jsonl
   - logs/agent_fire.jsonl
   Each VLM inference recorded as one JSONL line.

2. Orchestrator Log (unified):
   - logs/orchestrator.jsonl
   All agent results combined per-frame.
   Final decision, Tier2 routing.

3. Episode Log (lifecycle):
   - logs/episodes.jsonl
   State transitions: IDLE -> ACTIVE -> VALIDATING -> DONE.
   Tier2 decision reasons, validation results.

Format: JSONL (one JSON per line)
- Real-time tail: tail -f logs/orchestrator.jsonl
- Analysis: pandas.read_json('logs/orchestrator.jsonl', lines=True)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading


class VLMLogger:
    """
    Structured logger for VLM detection pipeline.

    Creates separate JSONL log files for each component.
    Thread-safe for concurrent scenario writing.
    """

    def __init__(self, log_dir: str = None, camera_id: str = "unknown"):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            camera_id: Camera identifier for log filenames
        """
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(__file__), '..', 'logs'
            )

        self.log_dir = Path(log_dir)
        # Path arithmetic requires string-like components; allow int camera ids safely.
        self.camera_id = str(camera_id)
        self._lock = threading.Lock()

        # Create dated subdirectory
        today = datetime.now().strftime("%Y%m%d")
        self.session_dir = self.log_dir / today / self.camera_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Log file handles (lazy open)
        self._files: Dict[str, Any] = {}

        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._write_session_start()

    def _get_file(self, name: str):
        """Get or create file handle (thread-safe)"""
        if name not in self._files:
            path = self.session_dir / f"{name}.jsonl"
            self._files[name] = open(path, 'a', encoding='utf-8')
        return self._files[name]

    def _write_line(self, filename: str, data: Dict):
        """Write a single JSONL line (thread-safe)"""
        with self._lock:
            f = self._get_file(filename)
            line = json.dumps(data, ensure_ascii=False, default=str)
            f.write(line + '\n')
            f.flush()

    def _write_session_start(self):
        """Write session start marker to all logs"""
        meta = {
            'event': 'session_start',
            'session_id': self.session_id,
            'camera_id': self.camera_id,
            'timestamp': datetime.now().isoformat(),
            'log_dir': str(self.session_dir)
        }
        self._write_line('orchestrator', meta)

    # ================================================================
    # Agent Logs (per scenario)
    # ================================================================

    def log_agent_inference(
        self,
        scenario_type: str,
        frame_idx: int,
        result: Dict,
        prompt_used: str = None
    ):
        """
        Log individual agent (scenario) inference.

        Written to: agent_{scenario_type}.jsonl

        Args:
            scenario_type: 'cash', 'violence', 'fire'
            frame_idx: Frame number
            result: ScenarioResult as dict
        """
        entry = {
            'ts': datetime.now().isoformat(),
            'frame': frame_idx,
            'scenario': scenario_type,
            'detected': result.get('is_detected', False),
            'confidence': result.get('confidence', 0.0),
            'evidence': result.get('evidence', ''),
            'exclusion': result.get('exclusion_match'),
            'zone': result.get('zone'),
            'inference_ms': result.get('inference_time_ms', 0),
            'raw_response': result.get('raw_response', '')[:500],  # truncate
        }

        if prompt_used:
            entry['prompt_hash'] = hash(prompt_used) % 10**8  # compact ref

        self._write_line(f'agent_{scenario_type}', entry)

    # ================================================================
    # Orchestrator Log (unified per-frame)
    # ================================================================

    def log_orchestrator_frame(
        self,
        frame_idx: int,
        scenario_results: Dict[str, Dict],
        detections: List[Dict],
        total_inference_ms: float,
        in_burst_mode: bool = False
    ):
        """
        Log unified orchestrator decision per frame.

        Written to: orchestrator.jsonl

        Args:
            frame_idx: Frame number
            scenario_results: All scenario results for this frame
            detections: Final detection list
            total_inference_ms: Total processing time
            in_burst_mode: Whether in burst mode
        """
        # Compact scenario summary
        scenario_summary = {}
        for name, result in scenario_results.items():
            scenario_summary[name] = {
                'd': result.get('is_detected', False),  # detected
                'c': round(result.get('confidence', 0), 3),  # confidence
                'ms': round(result.get('inference_time_ms', 0), 1),
            }

        # Compact detection summary
        detection_summary = [
            {
                'label': d.get('label', ''),
                'conf': round(d.get('confidence', 0), 3),
                'evidence': d.get('metadata', {}).get('evidence', '')[:100]
            }
            for d in detections
        ]

        entry = {
            'ts': datetime.now().isoformat(),
            'frame': frame_idx,
            'event': 'frame_processed',
            'agents': scenario_summary,
            'detections': detection_summary,
            'detection_count': len(detections),
            'total_ms': round(total_inference_ms, 1),
            'burst': in_burst_mode,
        }

        self._write_line('orchestrator', entry)

    # ================================================================
    # Episode Log (lifecycle)
    # ================================================================

    def log_episode_transition(
        self,
        episode_id: str,
        event_type: str,
        from_state: str,
        to_state: str,
        trigger: str = '',
        metadata: Dict = None
    ):
        """
        Log episode state transition.

        Written to: episodes.jsonl

        Args:
            episode_id: Episode identifier
            event_type: 'cash', 'violence', 'fire'
            from_state: Previous state
            to_state: New state
            trigger: What caused the transition
            metadata: Additional context
        """
        entry = {
            'ts': datetime.now().isoformat(),
            'event': 'state_transition',
            'episode_id': episode_id,
            'event_type': event_type,
            'from': from_state,
            'to': to_state,
            'trigger': trigger,
        }

        if metadata:
            entry['meta'] = metadata

        self._write_line('episodes', entry)

    def log_tier2_decision(
        self,
        episode_id: str,
        event_type: str,
        should_route: bool,
        reason: str,
        confidence: float,
        stability: float
    ):
        """
        Log Tier2 routing decision.

        Written to: episodes.jsonl
        """
        entry = {
            'ts': datetime.now().isoformat(),
            'event': 'tier2_decision',
            'episode_id': episode_id,
            'event_type': event_type,
            'route_to_tier2': should_route,
            'reason': reason,
            'confidence': round(confidence, 3),
            'stability': round(stability, 3),
        }

        self._write_line('episodes', entry)

    def log_tier2_result(
        self,
        episode_id: str,
        event_type: str,
        validated: bool,
        gemini_confidence: float,
        gemini_reason: str,
        response_time_ms: float
    ):
        """
        Log Tier2 (Gemini) validation result.

        Written to: episodes.jsonl
        """
        entry = {
            'ts': datetime.now().isoformat(),
            'event': 'tier2_result',
            'episode_id': episode_id,
            'event_type': event_type,
            'validated': validated,
            'gemini_conf': round(gemini_confidence, 3),
            'gemini_reason': gemini_reason[:200],
            'response_ms': round(response_time_ms, 1),
        }

        self._write_line('episodes', entry)

    def log_router_decision(
        self,
        episode_id: str,
        event_type: str,
        action: str,
        reason: str,
        q_scores: Dict[str, float],
        state_features: Dict[str, Any],
        selected_mode: str = None,
        video_window_sec: Optional[List[float]] = None
    ):
        """
        Log Stage1 router decision for value-learning dataset generation.

        Written to: episodes.jsonl
        """
        entry = {
            'ts': datetime.now().isoformat(),
            'event': 'router_decision',
            'episode_id': episode_id,
            'event_type': event_type,
            'router_action': action,
            'router_reason': reason,
            'router_q': q_scores or {},
            'state': state_features or {},
        }

        if selected_mode:
            entry['selected_mode'] = selected_mode
        if video_window_sec:
            entry['video_window_sec'] = video_window_sec

        self._write_line('episodes', entry)

    def log_human_feedback(
        self,
        episode_id: str,
        final_label: str,
        note: str = '',
        evidence_span: Optional[List[float]] = None
    ):
        """
        Log optional human review feedback to support Stage2/Stage3 datasets.

        Written to: episodes.jsonl
        """
        entry = {
            'ts': datetime.now().isoformat(),
            'event': 'human_feedback',
            'episode_id': episode_id,
            'final_label': final_label,
            'note': note[:500] if note else '',
        }
        if evidence_span:
            entry['evidence_span'] = evidence_span

        self._write_line('episodes', entry)

    # ================================================================
    # Summary & Cleanup
    # ================================================================

    def log_session_summary(self, stats: Dict):
        """Log session end summary"""
        entry = {
            'ts': datetime.now().isoformat(),
            'event': 'session_end',
            'session_id': self.session_id,
            'stats': stats
        }
        self._write_line('orchestrator', entry)

    def get_log_paths(self) -> Dict[str, str]:
        """Get all log file paths"""
        return {
            'orchestrator': str(self.session_dir / 'orchestrator.jsonl'),
            'agent_cash': str(self.session_dir / 'agent_cash.jsonl'),
            'agent_violence': str(self.session_dir / 'agent_violence.jsonl'),
            'agent_fire': str(self.session_dir / 'agent_fire.jsonl'),
            'episodes': str(self.session_dir / 'episodes.jsonl'),
        }

    def close(self):
        """Close all file handles"""
        if not hasattr(self, "_files"):
            return
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files.clear()

    def __del__(self):
        self.close()
