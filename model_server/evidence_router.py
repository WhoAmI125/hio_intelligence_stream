"""
Evidence Router - Uncertainty Gate and Tier2 Routing

Routes uncertain or high-priority episodes to Tier2 (Gemini) for validation.
Creates evidence packets with keyframes and metadata.

Key responsibilities:
- Implement uncertainty gate logic
- Create evidence packets for Tier2
- Handle routing priority (FIRE > VIOLENCE > CASH)

Reference: CCTV_Florence_Tiered_Architecture_Guide.md Section 7-8
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from pathlib import Path
import json
import os
import time
import uuid
import hashlib
import numpy as np

from .episode_manager import Episode, EpisodeState

try:
    from .policy.config import default_calibration, load_calibration_from_config
    from .policy.state_features import StateFeatureBuilder
    from .policy.critic import DualHeadCritic
    from .policy.decision_layer import DecisionLayer
    from .policy.router_steps import append_router_step, build_router_step
    POLICY_RUNTIME_AVAILABLE = True
except Exception:
    default_calibration = None
    load_calibration_from_config = None
    StateFeatureBuilder = None
    DualHeadCritic = None
    DecisionLayer = None
    append_router_step = None
    build_router_step = None
    POLICY_RUNTIME_AVAILABLE = False


@dataclass
class EvidencePacket:
    """
    Evidence packet for Tier2 validation.

    Contains keyframes, ROI crops, and metadata for Gemini validation.
    """

    episode_id: str
    camera_id: int
    event_type: str
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "v1.1"

    # Keyframes
    global_keyframes: List[np.ndarray] = field(default_factory=list)  # 6-12 frames
    cashier_roi_frames: List[np.ndarray] = field(default_factory=list)  # 8-16 frames
    drawer_roi_frames: List[np.ndarray] = field(default_factory=list)  # 2-8 frames
    global_keyframe_ts: List[float] = field(default_factory=list)
    cashier_roi_ts: List[float] = field(default_factory=list)
    drawer_roi_ts: List[float] = field(default_factory=list)
    anchor_mono_ts: Optional[float] = None
    clip_sec_used: Optional[int] = None
    clip_window_start_mono_ts: Optional[float] = None
    clip_window_end_mono_ts: Optional[float] = None

    # Metadata
    tier1_confidence: float = 0.0
    stability_score: float = 0.0
    detection_count: int = 0
    duration_seconds: float = 0.0

    # Timestamps
    start_ts: datetime = None
    end_ts: datetime = None
    created_at: datetime = field(default_factory=datetime.now)

    # Additional context
    zones_metadata: Dict = field(default_factory=dict)
    tier1_scores: List[float] = field(default_factory=list)

    # Stage1 routing/evidence metadata
    selected_mode: str = "image"  # image | video
    router_action: str = "GEMINI_IMG"
    router_reason: str = ""
    router_q: Dict[str, float] = field(default_factory=dict)
    t_peak_sec: Optional[float] = None
    video_window_sec: Optional[List[float]] = None
    focus_hints: List[str] = field(default_factory=list)
    florence_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary (without image data)."""
        return {
            'packet_id': self.packet_id,
            'schema_version': self.schema_version,
            'episode_id': self.episode_id,
            'camera_id': self.camera_id,
            'event_type': self.event_type,
            'global_keyframes_count': len(self.global_keyframes),
            'cashier_roi_count': len(self.cashier_roi_frames),
            'drawer_roi_count': len(self.drawer_roi_frames),
            'global_keyframe_ts': self.global_keyframe_ts,
            'cashier_roi_ts': self.cashier_roi_ts,
            'drawer_roi_ts': self.drawer_roi_ts,
            'anchor_mono_ts': self.anchor_mono_ts,
            'clip_sec_used': self.clip_sec_used,
            'clip_window_start_mono_ts': self.clip_window_start_mono_ts,
            'clip_window_end_mono_ts': self.clip_window_end_mono_ts,
            'tier1_confidence': self.tier1_confidence,
            'stability_score': self.stability_score,
            'detection_count': self.detection_count,
            'duration_seconds': self.duration_seconds,
            'start_ts': self.start_ts.isoformat() if self.start_ts else None,
            'end_ts': self.end_ts.isoformat() if self.end_ts else None,
            'created_at': self.created_at.isoformat(),
            'selected_mode': self.selected_mode,
            'router_action': self.router_action,
            'router_reason': self.router_reason,
            'router_q': self.router_q,
            't_peak_sec': self.t_peak_sec,
            'video_window_sec': self.video_window_sec,
            'focus_hints': self.focus_hints,
            'florence_signals': self.florence_signals,
        }


class EvidenceRouter:
    """
    Routes episodes to Tier2 based on uncertainty and priority.

    Implements:
    - Uncertainty gate logic
    - Priority-based routing (FIRE > VIOLENCE > CASH)
    - Evidence packet creation
    - Stage1 action policy (SKIP/GEMINI_IMG/GEMINI_VIDEO/HUMAN_QUEUE)
    """

    # Default confidence thresholds for Tier2 routing
    # Below these thresholds -> always send to Tier2
    DEFAULT_THRESHOLDS = {
        'fire': 0.60,
        'violence': 0.70,
        'cash': 0.80,
    }

    # Priority order (higher priority events processed first)
    PRIORITY_ORDER = ['fire', 'violence', 'cash']

    # Stage1 action space
    ACTION_SKIP = 'SKIP'
    ACTION_GEMINI_IMG = 'GEMINI_IMG'
    ACTION_GEMINI_VIDEO = 'GEMINI_VIDEO'
    ACTION_HUMAN_QUEUE = 'HUMAN_QUEUE'

    ACTIONS = (
        ACTION_SKIP,
        ACTION_GEMINI_IMG,
        ACTION_GEMINI_VIDEO,
        ACTION_HUMAN_QUEUE,
    )
    TIER2_ACTIONS = {ACTION_GEMINI_IMG, ACTION_GEMINI_VIDEO}
    DEFAULT_POLICY_FEATURES = [
        'avg_conf',
        'max_conf',
        'stability',
        'detection_count',
        'duration_sec',
        'h2h_conf_peak',
        'drawer_evidence',
        'cashier_zone_used',
        'drawer_zone_used',
        'contamination',
        'uncertainty',
        'cash_path',
        'global_handover_score',
        'is_cash',
        'is_violence',
        'is_fire',
    ]
    REQUIRED_STATE_FIELDS = {
        'event_type': '',
        'avg_conf': 0.0,
        'stability': 0.0,
        'detection_count': 0,
        'duration_sec': 0.0,
        'uncertainty': 1.0,
    }

    def __init__(self, config: Dict = None):
        """
        Initialize Evidence Router.

        Args:
            config: Configuration with keys:
                - fire_conf_tier2: Threshold for fire (default: 0.60)
                - violence_conf_tier2: Threshold for violence (default: 0.70)
                - cash_conf_tier2: Threshold for cash (default: 0.80)
                - stability_threshold: Min stability to skip Tier2 (default: 0.6)
                - max_keyframes: Max keyframes per packet (default: 12)
                - max_roi_frames: Max ROI frames per packet (default: 16)
        """
        self.config = config or {}

        # Thresholds
        self.thresholds = {
            'fire': self.config.get('fire_conf_tier2', self.DEFAULT_THRESHOLDS['fire']),
            'violence': self.config.get('violence_conf_tier2', self.DEFAULT_THRESHOLDS['violence']),
            'cash': self.config.get('cash_conf_tier2', self.DEFAULT_THRESHOLDS['cash']),
        }

        self.stability_threshold = float(self.config.get('stability_threshold', 0.6))
        self.max_keyframes = int(self.config.get('max_keyframes', 12))
        self.max_roi_frames = int(self.config.get('max_roi_frames', 16))

        # Stage1 value-policy settings
        self.router_margin = float(self.config.get('router_margin', 0.08))
        self.gemini_target_ratio = float(self.config.get('gemini_target_ratio', 0.40))
        self.gemini_ratio_penalty = float(self.config.get('gemini_ratio_penalty', 0.6))
        self.hard_tier2_events = set(self.config.get('hard_tier2_events', ['fire', 'violence']))
        self.hard_tier2_max_conf = float(self.config.get('hard_tier2_max_conf', 0.95))
        self.human_queue_fallback_to_gemini = bool(
            self.config.get('human_queue_fallback_to_gemini', True)
        )

        self.action_costs = {
            self.ACTION_SKIP: float(self.config.get('cost_skip', 0.0)),
            self.ACTION_GEMINI_IMG: float(self.config.get('cost_gemini_img', 0.10)),
            self.ACTION_GEMINI_VIDEO: float(self.config.get('cost_gemini_video', 0.20)),
            self.ACTION_HUMAN_QUEUE: float(self.config.get('cost_human_queue', 0.35)),
        }
        self.risk_weights = {
            'fire': float(self.config.get('risk_weight_fire', 2.0)),
            'violence': float(self.config.get('risk_weight_violence', 1.5)),
            'threat': float(self.config.get('risk_weight_threat', 1.5)),
            'cash': float(self.config.get('risk_weight_cash', 1.0)),
        }

        self.video_clip_seconds = float(self.config.get('video_clip_seconds', 10.0))
        self.video_clip_pre_seconds = float(self.config.get('video_clip_pre_seconds', 2.0))

        # Pending packets queue
        self.pending_queue: deque = deque(maxlen=10)

        # Statistics
        self.total_routed = 0
        self.routed_by_type: Dict[str, int] = {'fire': 0, 'violence': 0, 'cash': 0}
        self.total_decisions = 0
        self.action_counts: Dict[str, int] = {a: 0 for a in self.ACTIONS}

        # Optional learned router policy (trained from exported dataset)
        self.policy_path = str(
            self.config.get('router_policy_path')
            or os.getenv('VLM_ROUTER_POLICY_PATH', '')
            or ''
        ).strip()
        self.policy_blend = float(
            self.config.get('router_policy_blend', os.getenv('VLM_ROUTER_POLICY_BLEND', '0.7'))
        )
        self.policy_model: Optional[Dict[str, Any]] = None
        self.policy_feature_names: List[str] = list(self.DEFAULT_POLICY_FEATURES)
        self.policy_load_error: Optional[str] = None
        if self.policy_path:
            self.load_policy(self.policy_path)

        # Stage2 critic/decision runtime (shadow by default).
        self.critic_enabled = self._as_bool(
            self.config.get('critic_enabled', os.getenv('VLM_CRITIC_ENABLED', False)),
            default=False,
        )
        self.critic_shadow_mode = self._as_bool(
            self.config.get('critic_shadow_mode', os.getenv('VLM_CRITIC_SHADOW_MODE', True)),
            default=True,
        )
        self.router_steps_path = str(
            self.config.get('router_steps_path')
            or os.getenv('VLM_ROUTER_STEPS_PATH', '')
            or ''
        ).strip()
        self.camera_id = str(
            self.config.get('camera_id')
            or os.getenv('VLM_CAMERA_ID', '')
            or ''
        ).strip()
        self.critic_rollout_mode = str(
            self.config.get('critic_rollout_mode')
            or os.getenv('VLM_CRITIC_ROLLOUT_MODE', 'shadow')
            or 'shadow'
        ).strip().lower()
        self.critic_canary_pct = self._as_float(
            self.config.get('critic_canary_pct', os.getenv('VLM_CRITIC_CANARY_PCT', 10.0)),
            10.0,
        )
        raw_canary = self.config.get('critic_canary_cameras', os.getenv('VLM_CRITIC_CANARY_CAMERAS', ''))
        if isinstance(raw_canary, str):
            self.critic_canary_cameras = {
                x.strip() for x in raw_canary.split(',') if str(x).strip()
            }
        elif isinstance(raw_canary, (list, tuple, set)):
            self.critic_canary_cameras = {str(x).strip() for x in raw_canary if str(x).strip()}
        else:
            self.critic_canary_cameras = set()

        self.critic = None
        self.critic_decision_layer = None
        self.critic_feature_builder = None
        self.critic_load_error: Optional[str] = None
        self.router_step_counts: Dict[str, int] = {'decision': 0, 'outcome': 0, 'feedback': 0}
        self.last_router_step_ts: Optional[str] = None
        self.router_step_total: int = 0
        self.router_step_empty_state_count: int = 0
        self.router_step_missing_required_total: int = 0
        self._decision_event_ids: set = set()
        self._outcome_event_ids: set = set()
        self._feedback_event_ids: set = set()

        if POLICY_RUNTIME_AVAILABLE:
            try:
                calibration_raw = self.config.get('critic_calibration', {})
                if isinstance(calibration_raw, str):
                    try:
                        calibration_raw = json.loads(calibration_raw)
                    except Exception:
                        calibration_raw = {}
                calibration = (
                    load_calibration_from_config(calibration_raw)
                    if callable(load_calibration_from_config)
                    else default_calibration()
                )
                self.critic_feature_builder = StateFeatureBuilder()
                self.critic_decision_layer = DecisionLayer(calibration=calibration)
                self.critic = DualHeadCritic(
                    bundle_path=str(
                        self.config.get('critic_bundle_path')
                        or os.getenv('VLM_CRITIC_BUNDLE_PATH', '')
                        or ''
                    ),
                    skip_model_path=str(
                        self.config.get('critic_skip_model_path')
                        or os.getenv('VLM_CRITIC_SKIP_MODEL_PATH', '')
                        or ''
                    ),
                    need_video_model_path=str(
                        self.config.get('critic_video_model_path')
                        or os.getenv('VLM_CRITIC_VIDEO_MODEL_PATH', '')
                        or ''
                    ),
                )
            except Exception as e:
                self.critic = None
                self.critic_decision_layer = None
                self.critic_feature_builder = None
                self.critic_load_error = str(e)

    def _risk_weight_for_event(self, event_type: str) -> float:
        et = str(event_type or '').strip().lower()
        if 'fire' in et:
            return float(self.risk_weights.get('fire', 2.0))
        if 'violence' in et:
            return float(self.risk_weights.get('violence', 1.5))
        if 'threat' in et:
            return float(self.risk_weights.get('threat', 1.5))
        if 'cash' in et:
            return float(self.risk_weights.get('cash', 1.0))
        return float(self.risk_weights.get(et, 1.0))

    def _action_cost(self, action_taken: str) -> float:
        return float(self.action_costs.get(str(action_taken or '').strip(), 0.0))

    def _compute_reward(
        self,
        *,
        event_type: str,
        action_taken: str,
        base_score: float,
    ) -> Tuple[float, float, float]:
        risk_weight = self._risk_weight_for_event(event_type)
        action_cost = self._action_cost(action_taken)
        reward = (risk_weight * float(base_score)) - action_cost
        return float(reward), float(action_cost), float(risk_weight)

    @staticmethod
    def _stable_bucket_percent(key: str) -> float:
        src = str(key or "__default__").encode("utf-8", errors="ignore")
        h = hashlib.md5(src).hexdigest()[:8]
        return (int(h, 16) % 10000) / 100.0

    def _critic_live_enabled(self) -> bool:
        if not bool(self.critic_enabled):
            return False
        if bool(self.critic_shadow_mode):
            return False
        mode = str(self.critic_rollout_mode or "shadow").strip().lower()
        if mode == "live":
            return True
        if mode == "canary":
            cam = str(self.camera_id or "").strip()
            if cam and cam in self.critic_canary_cameras:
                return True
            pct = max(0.0, min(100.0, float(self.critic_canary_pct)))
            if pct <= 0.0:
                return False
            return self._stable_bucket_percent(cam or "__default__") < pct
        return False

    def _normalize_state_features(
        self,
        state: Optional[Dict[str, Any]],
        event_type: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, int]:
        src = dict(state or {})
        if event_type and not src.get('event_type'):
            src['event_type'] = str(event_type).strip().lower()

        missing = 0
        for key, default in self.REQUIRED_STATE_FIELDS.items():
            if key not in src or src.get(key) is None:
                src[key] = default
                missing += 1

        # Normalize numeric required fields.
        for key in ('avg_conf', 'stability', 'duration_sec', 'uncertainty'):
            src[key] = self._as_float(src.get(key, 0.0), 0.0)
        src['detection_count'] = int(round(self._as_float(src.get('detection_count', 0), 0.0)))
        src['event_type'] = str(src.get('event_type', '')).strip().lower()

        flag = 'complete' if missing == 0 else 'fallback_minimal'
        src['state_quality_flag'] = flag
        return src, flag, missing

    def _gemini_ratio(self) -> float:
        """Current ratio of Gemini actions among router decisions."""
        if self.total_decisions <= 0:
            return 0.0
        gemini_calls = (
            self.action_counts.get(self.ACTION_GEMINI_IMG, 0)
            + self.action_counts.get(self.ACTION_GEMINI_VIDEO, 0)
        )
        return gemini_calls / max(1, self.total_decisions)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        s = str(value).strip().lower()
        if s in {'1', 'true', 'yes', 'on'}:
            return True
        if s in {'0', 'false', 'no', 'off'}:
            return False
        return bool(default)

    @staticmethod
    def _normalize_signal_items(values: Any, max_items: int = 8) -> List[str]:
        if values is None:
            return []
        if isinstance(values, list):
            raw_items = values
        elif isinstance(values, (tuple, set)):
            raw_items = list(values)
        else:
            raw_items = [values]

        out: List[str] = []
        seen = set()
        for item in raw_items:
            s = str(item).strip().lower()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_items:
                break
        return out

    def _extract_florence_signals(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        raw = metadata.get('florence_signals') if isinstance(metadata, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        return {
            'matched_keywords': self._normalize_signal_items(raw.get('matched_keywords')),
            'object_hints': self._normalize_signal_items(raw.get('object_hints')),
            'exclusion_match': self._normalize_signal_items(raw.get('exclusion_match')),
            'global_keywords': self._normalize_signal_items(raw.get('global_keywords')),
        }

    def load_policy(self, policy_path: str) -> bool:
        """
        Load a learned router policy JSON file.

        Expected structure:
            {
              "feature_names": [...],
              "normalization": {"mean":[...], "std":[...]},
              "action_models": {
                "SKIP": {"weights":[...], "bias": 0.0},
                ...
              }
            }
        """
        self.policy_load_error = None
        try:
            p = Path(policy_path)
            if not p.exists():
                self.policy_model = None
                self.policy_load_error = f"Policy file not found: {p}"
                return False

            data = json.loads(p.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                self.policy_model = None
                self.policy_load_error = "Invalid policy format (not object)"
                return False

            action_models = data.get('action_models')
            if not isinstance(action_models, dict) or not action_models:
                self.policy_model = None
                self.policy_load_error = "Invalid policy format: action_models missing"
                return False

            feature_names = data.get('feature_names') or self.DEFAULT_POLICY_FEATURES
            if not isinstance(feature_names, list) or not feature_names:
                feature_names = self.DEFAULT_POLICY_FEATURES

            self.policy_feature_names = [str(x) for x in feature_names]
            self.policy_model = data
            self.policy_path = str(p)
            return True
        except Exception as e:
            self.policy_model = None
            self.policy_load_error = str(e)
            return False

    def _state_to_feature_vector(
        self,
        state: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        names = feature_names or self.policy_feature_names or self.DEFAULT_POLICY_FEATURES
        event_type = str(state.get('event_type', '')).lower()
        values: List[float] = []
        for name in names:
            if name == 'is_cash':
                values.append(1.0 if event_type == 'cash' else 0.0)
                continue
            if name == 'is_violence':
                values.append(1.0 if event_type == 'violence' else 0.0)
                continue
            if name == 'is_fire':
                values.append(1.0 if event_type == 'fire' else 0.0)
                continue

            v = state.get(name, 0.0)
            if isinstance(v, bool):
                values.append(1.0 if v else 0.0)
            else:
                values.append(self._as_float(v, 0.0))
        return np.array(values, dtype=np.float32)

    def _score_actions_heuristic(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Deterministic fallback scoring used when no learned policy is available.
        """
        event_type = state['event_type']
        avg_conf = state['avg_conf']
        stability = state['stability']
        uncertainty = state['uncertainty']

        q_skip = (avg_conf * 1.1) + (stability * 0.9) - (uncertainty * 1.0)
        q_img = 0.40 + (uncertainty * 0.95)
        q_video = 0.38 + (uncertainty * 1.05)
        q_human = (uncertainty * 0.70)

        # Event-type priors
        if event_type == 'cash':
            q_video += 0.20
            if not state['drawer_evidence']:
                q_video += 0.20
                q_skip -= 0.20
            if state['h2h_conf_peak'] < 0.45:
                q_video += 0.10
            if state['contamination']:
                q_img += 0.10
                q_video += 0.10
                q_skip -= 0.15
        elif event_type in ('fire', 'violence'):
            q_img += 0.20
            q_video += 0.25
            q_skip -= 0.45
            if avg_conf < 0.60:
                q_human += 0.30

        # Very stable/high confidence episodes should bias toward SKIP
        if avg_conf > 0.92 and stability > 0.92:
            q_skip += 0.35

        # Cost penalty
        q_skip -= self.action_costs[self.ACTION_SKIP]
        q_img -= self.action_costs[self.ACTION_GEMINI_IMG]
        q_video -= self.action_costs[self.ACTION_GEMINI_VIDEO]
        q_human -= self.action_costs[self.ACTION_HUMAN_QUEUE]

        # Soft penalty if Gemini ratio grows too high
        if self.total_decisions >= 10:
            overflow = max(0.0, self._gemini_ratio() - self.gemini_target_ratio)
            if overflow > 0:
                penalty = overflow * self.gemini_ratio_penalty
                q_img -= penalty
                q_video -= penalty * 1.10

        return {
            self.ACTION_SKIP: float(q_skip),
            self.ACTION_GEMINI_IMG: float(q_img),
            self.ACTION_GEMINI_VIDEO: float(q_video),
            self.ACTION_HUMAN_QUEUE: float(q_human),
        }

    def _score_actions_learned(self, state: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Score actions from the loaded learned policy model.
        """
        if not self.policy_model:
            return None
        try:
            feature_names = self.policy_model.get('feature_names') or self.policy_feature_names
            x = self._state_to_feature_vector(state, feature_names)

            norm = self.policy_model.get('normalization') or {}
            mean = np.array(norm.get('mean') or [0.0] * len(x), dtype=np.float32)
            std = np.array(norm.get('std') or [1.0] * len(x), dtype=np.float32)
            if mean.shape[0] != x.shape[0] or std.shape[0] != x.shape[0]:
                return None
            std = np.where(np.abs(std) < 1e-6, 1.0, std)
            xn = (x - mean) / std

            out: Dict[str, float] = {}
            action_models = self.policy_model.get('action_models') or {}
            for action in self.ACTIONS:
                m = action_models.get(action)
                if not isinstance(m, dict):
                    continue
                w = np.array(m.get('weights') or [], dtype=np.float32)
                b = self._as_float(m.get('bias', 0.0), 0.0)
                if w.shape[0] != xn.shape[0]:
                    continue
                out[action] = float(np.dot(xn, w) + b)
            return out if out else None
        except Exception:
            return None

    def extract_state_features(self, episode: Episode) -> Dict[str, Any]:
        """
        Extract Stage1 state features from the episode.
        """
        metadata = episode.metadata or {}
        keyword_flags = metadata.get('keyword_flags') or []
        if not isinstance(keyword_flags, list):
            keyword_flags = [str(keyword_flags)]

        avg_conf = float(episode.get_average_confidence())
        max_conf = float(episode.get_max_confidence())
        stability = float(episode.get_stability_score())

        h2h_peak = metadata.get('h2h_conf_peak')
        if h2h_peak is None:
            h2h_peak = metadata.get('handover_score_peak')
        h2h_peak = float(h2h_peak) if isinstance(h2h_peak, (int, float)) else 0.0

        drawer_evidence = bool(
            metadata.get('drawer_detected')
            or metadata.get('drawer_signal_peak')
            or metadata.get('drawer_activity')
        )
        cashier_zone_used = bool(
            metadata.get('cashier_zone_used')
            or metadata.get('cashier_zone')
            or metadata.get('used_cashier_zone')
        )
        drawer_zone_used = bool(
            metadata.get('drawer_zone_used')
            or metadata.get('drawer_zone')
            or metadata.get('used_drawer_zone')
        )

        contamination = any(
            'cash_register' in str(flag).lower() or 'contamination' in str(flag).lower()
            for flag in keyword_flags
        )
        cash_path_raw = str(metadata.get('cash_path', '')).strip().lower()
        cash_path_score = {
            'roi': 1.0,
            'global_assist': 0.6,
            'both': 1.2,
        }.get(cash_path_raw, 0.0)
        global_handover_score = self._as_float(metadata.get('global_handover_score', 0.0), 0.0)

        uncertainty = max(0.0, 1.0 - avg_conf) + max(0.0, 1.0 - stability) * 0.8
        if contamination:
            uncertainty += 0.15
        if episode.event_type == 'cash' and not drawer_evidence:
            uncertainty += 0.20

        return {
            'event_type': episode.event_type,
            'avg_conf': avg_conf,
            'max_conf': max_conf,
            'stability': stability,
            'detection_count': int(episode.detection_count),
            'duration_sec': float(episode.get_duration_seconds()),
            'h2h_conf_peak': h2h_peak,
            'drawer_evidence': drawer_evidence,
            'cashier_zone_used': cashier_zone_used,
            'drawer_zone_used': drawer_zone_used,
            'contamination': contamination,
            'uncertainty': uncertainty,
            'cash_path': cash_path_score,
            'global_handover_score': global_handover_score,
        }

    def _evaluate_critic(
        self,
        *,
        episode: Episode,
        state: Dict[str, Any],
        rule_q: Dict[str, float],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'available': False,
            'enabled': bool(self.critic_enabled),
            'shadow_mode': bool(self.critic_shadow_mode),
            'prediction': None,
            'decision': None,
            'error': '',
        }
        if not (self.critic and self.critic_feature_builder and self.critic_decision_layer):
            if self.critic_load_error:
                out['error'] = str(self.critic_load_error)
            return out

        try:
            feature_map = self.critic_feature_builder.build(
                state=state,
                event_type=str(state.get('event_type') or episode.event_type),
                rule_q=rule_q,
            )
            pred = self.critic.predict(feature_map)
            out['prediction'] = pred.to_dict()
            if not pred.available:
                out['error'] = str(pred.detail or 'critic_prediction_unavailable')
                return out

            skip_safe = bool(state.get('stability', 0.0) >= 0.35 and state.get('avg_conf', 0.0) >= 0.45)
            decision = self.critic_decision_layer.decide(
                event_type=str(state.get('event_type') or episode.event_type),
                pred=pred,
                rule_q=rule_q,
                hard_flags={'skip_safe': skip_safe},
            )
            out['decision'] = decision.to_dict()
            out['available'] = True
            return out
        except Exception as e:
            out['error'] = str(e)
            return out

    def _log_router_step(
        self,
        *,
        record_type: str,
        event_id: Optional[str],
        episode: Episode,
        state: Dict[str, Any],
        action_taken: str,
        router_reason: str,
        router_q: Dict[str, float],
        critic_payload: Dict[str, Any],
        clip_sec_used: Optional[int] = None,
        gemini_result: Optional[Dict[str, Any]] = None,
        human_feedback: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        reward_source: Optional[str] = None,
        cost_applied: Optional[float] = None,
        risk_weight: Optional[float] = None,
        anchor_mono_ts: Optional[float] = None,
    ) -> None:
        if not self.router_steps_path or not callable(build_router_step) or not callable(append_router_step):
            return
        norm_state, state_quality_flag, missing_required = self._normalize_state_features(
            state,
            event_type=str(getattr(episode, 'event_type', '') or ''),
        )
        if anchor_mono_ts is None:
            try:
                _, ts_points = self._split_evidence_points(episode.evidence_frames)
                anchor_mono_ts = float(ts_points[-1]) if ts_points else None
            except Exception:
                anchor_mono_ts = None

        try:
            eid = str(event_id or f"rs_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}")
            camera_id_val = str(getattr(episode, 'camera_id', '') or self.camera_id or "")
            record = build_router_step(
                record_type=str(record_type or 'decision'),
                event_id=eid,
                episode_id=str(getattr(episode, 'episode_id', '') or ''),
                camera_id=camera_id_val,
                event_type=str(getattr(episode, 'event_type', '') or ''),
                state_features=norm_state,
                state_quality_flag=state_quality_flag,
                action_taken=str(action_taken or ''),
                clip_sec_used=int(round(float(clip_sec_used or self.video_clip_seconds or 10.0))),
                anchor_mono_ts=anchor_mono_ts,
                router_reason=router_reason,
                router_q=router_q,
                critic=critic_payload,
                gemini_result=(gemini_result if isinstance(gemini_result, dict) else None),
                human_feedback=(human_feedback if isinstance(human_feedback, dict) else None),
                reward=reward,
                reward_source=reward_source,
                cost_applied=cost_applied,
                risk_weight=risk_weight,
            )
            append_router_step(self.router_steps_path, record)
            rt = str(record_type or 'decision').strip().lower()
            if rt in self.router_step_counts:
                self.router_step_counts[rt] = int(self.router_step_counts.get(rt, 0)) + 1
            self.router_step_total += 1
            if len(norm_state) <= 1:
                self.router_step_empty_state_count += 1
            if int(missing_required) > 0:
                self.router_step_missing_required_total += int(missing_required)
            if rt == 'decision':
                self._decision_event_ids.add(eid)
            elif rt == 'outcome':
                self._outcome_event_ids.add(eid)
            elif rt == 'feedback':
                self._feedback_event_ids.add(eid)
            self.last_router_step_ts = datetime.now().isoformat()
        except Exception:
            # Router step logging must never break runtime.
            return

    def log_decision_step(
        self,
        *,
        event_id: str,
        episode_id: str,
        event_type: str,
        action_taken: str = ACTION_SKIP,
        state_features: Optional[Dict[str, Any]] = None,
        router_reason: str = "prevalidate_fallback",
        router_q: Optional[Dict[str, float]] = None,
        critic_payload: Optional[Dict[str, Any]] = None,
        clip_sec_used: Optional[int] = None,
        anchor_mono_ts: Optional[float] = None,
        camera_id: Optional[str] = None,
    ) -> None:
        if not self.router_steps_path:
            return
        ep = Episode(
            episode_id=str(episode_id or ""),
            event_type=str(event_type or ""),
            camera_id=0,
        )
        if camera_id:
            try:
                ep.camera_id = str(camera_id)
            except Exception:
                pass
        self._log_router_step(
            record_type='decision',
            event_id=event_id,
            episode=ep,
            state=(state_features or {}),
            action_taken=action_taken,
            router_reason=router_reason,
            router_q=(router_q or {}),
            critic_payload=(critic_payload or {}),
            clip_sec_used=clip_sec_used,
            anchor_mono_ts=anchor_mono_ts,
        )

    def log_outcome_step(
        self,
        *,
        event_id: str,
        episode_id: str,
        camera_id: Optional[str] = None,
        event_type: str,
        action_taken: str,
        state_features: Optional[Dict[str, Any]] = None,
        router_reason: str = "",
        router_q: Optional[Dict[str, float]] = None,
        critic_payload: Optional[Dict[str, Any]] = None,
        gemini_result: Optional[Dict[str, Any]] = None,
        clip_sec_used: Optional[int] = None,
        anchor_mono_ts: Optional[float] = None,
    ) -> None:
        if not self.router_steps_path:
            return
        norm_state, _, _ = self._normalize_state_features(state_features, event_type=event_type)
        ev_id = str(event_id or "").strip()
        if ev_id and ev_id not in self._decision_event_ids:
            self.log_decision_step(
                event_id=ev_id,
                episode_id=episode_id,
                camera_id=camera_id,
                event_type=event_type,
                action_taken=action_taken or self.ACTION_SKIP,
                state_features=norm_state,
                router_reason='prevalidate_fallback:auto_decision_for_chain_guard',
                router_q=(router_q or {}),
                critic_payload=(critic_payload or {}),
                clip_sec_used=clip_sec_used,
                anchor_mono_ts=anchor_mono_ts,
            )

        base_score = 0.0
        validated = None
        if isinstance(gemini_result, dict):
            validated = gemini_result.get('validated')
        if isinstance(validated, bool):
            base_score = 0.6 if bool(validated) else -0.6

        reward, cost_applied, risk_weight = self._compute_reward(
            event_type=event_type,
            action_taken=action_taken,
            base_score=base_score,
        )

        ep = Episode(
            episode_id=str(episode_id or ""),
            event_type=str(event_type or ""),
            camera_id=0,
        )
        if camera_id:
            try:
                ep.camera_id = str(camera_id)
            except Exception:
                pass
        self._log_router_step(
            record_type='outcome',
            event_id=event_id,
            episode=ep,
            state=norm_state,
            action_taken=action_taken,
            router_reason=router_reason,
            router_q=(router_q or {}),
            critic_payload=(critic_payload or {}),
            clip_sec_used=clip_sec_used,
            gemini_result=gemini_result,
            reward=reward,
            reward_source='gemini_weak',
            cost_applied=cost_applied,
            risk_weight=risk_weight,
            anchor_mono_ts=anchor_mono_ts,
        )

    def log_feedback_step(
        self,
        *,
        event_id: str,
        episode_id: str,
        camera_id: Optional[str] = None,
        event_type: str,
        action_taken: str,
        state_features: Optional[Dict[str, Any]] = None,
        router_reason: str = "",
        router_q: Optional[Dict[str, float]] = None,
        critic_payload: Optional[Dict[str, Any]] = None,
        human_feedback: Optional[Dict[str, Any]] = None,
        clip_sec_used: Optional[int] = None,
        anchor_mono_ts: Optional[float] = None,
    ) -> None:
        if not self.router_steps_path:
            return
        norm_state, _, _ = self._normalize_state_features(state_features, event_type=event_type)

        decision = ''
        if isinstance(human_feedback, dict):
            decision = str(human_feedback.get('decision') or '').strip().lower()

        if decision == 'accept':
            base_score = 1.0
        elif decision == 'decline':
            base_score = -1.0
        elif decision == 'unsure':
            base_score = 0.0
        else:
            base_score = 0.0

        reward, cost_applied, risk_weight = self._compute_reward(
            event_type=event_type,
            action_taken=action_taken,
            base_score=base_score,
        )

        ep = Episode(
            episode_id=str(episode_id or ""),
            event_type=str(event_type or ""),
            camera_id=0,
        )
        if camera_id:
            try:
                ep.camera_id = str(camera_id)
            except Exception:
                pass
        self._log_router_step(
            record_type='feedback',
            event_id=event_id,
            episode=ep,
            state=norm_state,
            action_taken=action_taken,
            router_reason=router_reason,
            router_q=(router_q or {}),
            critic_payload=(critic_payload or {}),
            clip_sec_used=clip_sec_used,
            human_feedback=human_feedback,
            reward=reward,
            reward_source='human_strong',
            cost_applied=cost_applied,
            risk_weight=risk_weight,
            anchor_mono_ts=anchor_mono_ts,
        )

    def score_actions(self, episode: Episode, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Score Stage1 actions with heuristic and optional learned policy blend.
        """
        heuristic_q = self._score_actions_heuristic(state)
        learned_q = self._score_actions_learned(state)
        if not learned_q:
            return {k: round(float(v), 4) for k, v in heuristic_q.items()}

        # Blend learned Q with heuristic Q for safe rollout.
        blend = min(1.0, max(0.0, float(self.policy_blend)))
        out: Dict[str, float] = {}
        for action in self.ACTIONS:
            h = float(heuristic_q.get(action, 0.0))
            l = float(learned_q.get(action, h))
            out[action] = round((1.0 - blend) * h + blend * l, 4)
        return out

    def select_action(
        self,
        episode: Episode,
        record_decision: bool = True,
        log_step: bool = False,
        step_event_id: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, float], Dict[str, Any]]:
        """
        Select Stage1 action (SKIP/GEMINI_IMG/GEMINI_VIDEO/HUMAN_QUEUE).

        Returns:
            (action, reason, q_scores, state_features)
        """
        if episode.state != EpisodeState.VALIDATING:
            state = self.extract_state_features(episode)
            state, _, _ = self._normalize_state_features(state, event_type=episode.event_type)
            q = {a: 0.0 for a in self.ACTIONS}
            if log_step:
                self._log_router_step(
                    record_type='decision',
                    event_id=step_event_id,
                    episode=episode,
                    state=state,
                    action_taken=self.ACTION_SKIP,
                    router_reason='prevalidate_fallback:episode_not_validating',
                    router_q=q,
                    critic_payload={},
                )
            return self.ACTION_SKIP, 'Episode not in VALIDATING state', q, state

        if episode.tier2_sent:
            state = self.extract_state_features(episode)
            state, _, _ = self._normalize_state_features(state, event_type=episode.event_type)
            q = {a: 0.0 for a in self.ACTIONS}
            if log_step:
                self._log_router_step(
                    record_type='decision',
                    event_id=step_event_id,
                    episode=episode,
                    state=state,
                    action_taken=self.ACTION_SKIP,
                    router_reason='prevalidate_fallback:already_sent_to_tier2',
                    router_q=q,
                    critic_payload={},
                )
            return self.ACTION_SKIP, 'Already sent to Tier2', q, state

        state = self.extract_state_features(episode)
        state, _, _ = self._normalize_state_features(state, event_type=episode.event_type)
        q = self.score_actions(episode, state)
        q_source = 'learned+heuristic' if self.policy_model else 'heuristic'

        event_type = state['event_type']
        avg_conf = state['avg_conf']
        baseline_reason = ""

        # Hard risk rule
        if event_type in self.hard_tier2_events and avg_conf < self.hard_tier2_max_conf:
            action = self.ACTION_GEMINI_VIDEO
            baseline_reason = (
                f'Hard-risk escalation for {event_type}: '
                f'conf={avg_conf:.2f} < {self.hard_tier2_max_conf:.2f}'
            )
        else:
            action = max(self.ACTIONS, key=lambda x: q.get(x, -9999.0))
            baseline_reason = f'max_q_action[{q_source}]'

        # Margin gate to suppress unnecessary Gemini calls
        if action in self.TIER2_ACTIONS:
            margin = q.get(action, 0.0) - q.get(self.ACTION_SKIP, 0.0)
            if margin < self.router_margin and event_type not in self.hard_tier2_events:
                action = self.ACTION_SKIP
                baseline_reason = (
                    f'Margin gate: best_gemini_margin={margin:.3f} '
                    f'< router_margin={self.router_margin:.3f}'
                )

        reason = baseline_reason

        # Critic + DecisionLayer (shadow by default)
        critic_payload = self._evaluate_critic(
            episode=episode,
            state=state,
            rule_q=q,
        )
        try:
            state['critic'] = critic_payload
        except Exception:
            pass

        if critic_payload.get('available') and isinstance(critic_payload.get('decision'), dict):
            c_action = str(critic_payload['decision'].get('action') or '').strip()
            c_reason = str(critic_payload['decision'].get('reason') or 'critic_decision')
            if c_action:
                if self._critic_live_enabled():
                    action = c_action
                    reason = f'critic_live:{c_reason}'
                else:
                    reason = f'{reason} | critic_shadow:{c_action}'
        else:
            c_err = str(critic_payload.get('error') or '').strip()
            if c_err:
                reason = f'{reason} | critic_unavailable:{c_err}'

        # Runtime compatibility: fallback HUMAN_QUEUE to Gemini video
        if action == self.ACTION_HUMAN_QUEUE and self.human_queue_fallback_to_gemini:
            action = self.ACTION_GEMINI_VIDEO
            reason = 'HUMAN_QUEUE fallback to GEMINI_VIDEO (runtime compatibility)'

        # Runtime policy: image path disabled, enforce video-only validation path.
        if action == self.ACTION_GEMINI_IMG:
            action = self.ACTION_GEMINI_VIDEO
            reason = f'{reason} | video_only_runtime: GEMINI_IMG->GEMINI_VIDEO'

        if log_step:
            self._log_router_step(
                record_type='decision',
                event_id=step_event_id,
                episode=episode,
                state=state,
                action_taken=action,
                router_reason=reason,
                router_q=q,
                critic_payload=critic_payload,
            )

        if record_decision:
            self.total_decisions += 1
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

        return action, reason, q, state

    def should_route_to_tier2(self, episode: Episode) -> Tuple[bool, str]:
        """
        Backward-compatible bool API for existing pipelines.

        Internally this maps to select_action().
        """
        action, reason, q, _ = self.select_action(episode, record_decision=False)
        should_route = action in self.TIER2_ACTIONS
        q_short = ', '.join([f'{k}={v:.2f}' for k, v in q.items()])
        return should_route, f'{action}: {reason} | {q_short}'

    @staticmethod
    def _split_evidence_points(evidence_frames: List[Any]) -> Tuple[List[int], List[float]]:
        """
        Parse mixed evidence point format.

        Supports both legacy int frame indices and new (frame_idx, mono_ts) tuples.
        """
        frame_indices: List[int] = []
        frame_mono_ts: List[float] = []
        for point in (evidence_frames or []):
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                try:
                    frame_indices.append(int(point[0]))
                except Exception:
                    pass
                try:
                    frame_mono_ts.append(float(point[1]))
                except Exception:
                    pass
                continue
            try:
                frame_indices.append(int(point))
            except Exception:
                continue
        return frame_indices, frame_mono_ts

    @staticmethod
    def _normalize_frame_entries(
        frame_buffer: List[np.ndarray],
        frame_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize input into [{'frame': np.ndarray, 'mono_ts': Optional[float]}]."""
        src = frame_entries if frame_entries is not None else frame_buffer
        out: List[Dict[str, Any]] = []
        for item in (src or []):
            if isinstance(item, dict) and 'frame' in item:
                ts = item.get('mono_ts')
                try:
                    mono_ts = float(ts) if ts is not None else None
                except Exception:
                    mono_ts = None
                out.append({'frame': item.get('frame'), 'mono_ts': mono_ts})
            else:
                out.append({'frame': item, 'mono_ts': None})
        return out

    @staticmethod
    def _sample_entries_evenly(entries: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if not entries or count <= 0:
            return []
        if len(entries) <= count:
            return list(entries)
        idx = np.linspace(0, len(entries) - 1, num=count, dtype=np.int32)
        return [entries[int(i)] for i in idx]

    @staticmethod
    def _window_entries_by_ts(
        entries: List[Dict[str, Any]],
        t_start: float,
        t_end: float,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            ts = e.get('mono_ts')
            if ts is None:
                continue
            try:
                ts_f = float(ts)
            except Exception:
                continue
            if t_start <= ts_f <= t_end:
                out.append(e)
        return out

    def _select_entries_for_packet(
        self,
        entries: List[Dict[str, Any]],
        evidence_indices: List[int],
        count: int,
        anchor_mono_ts: Optional[float] = None,
        clip_seconds: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Select entries with ordered fallback: anchor -> legacy indices -> uniform."""
        if not entries or count <= 0:
            return [], 'empty'

        clip_sec = float(clip_seconds or self.video_clip_seconds or 6.0)
        has_ts = any(e.get('mono_ts') is not None for e in entries if isinstance(e, dict))
        min_required = min(len(entries), max(4, min(8, int(count))))

        if anchor_mono_ts is not None and has_ts:
            anchor = float(anchor_mono_ts)
            half = clip_sec / 2.0
            symmetric = self._window_entries_by_ts(entries, anchor - half, anchor + half)
            if len(symmetric) >= min_required:
                return self._sample_entries_evenly(symmetric, count), 'anchor_symmetric'

            backward = self._window_entries_by_ts(entries, anchor - clip_sec, anchor + 0.5)
            if len(backward) >= min_required:
                return self._sample_entries_evenly(backward, count), 'anchor_backward'

            if backward:
                return self._sample_entries_evenly(backward, count), 'anchor_backward_sparse'
            if symmetric:
                return self._sample_entries_evenly(symmetric, count), 'anchor_symmetric_sparse'

        n = len(entries)
        if evidence_indices and n > 0:
            valid_indices: List[int] = []
            seen = set()
            for raw_idx in evidence_indices:
                try:
                    idx = int(raw_idx) % n
                except Exception:
                    continue
                if idx not in seen:
                    seen.add(idx)
                    valid_indices.append(idx)

            if valid_indices:
                step = max(1, len(valid_indices) // max(1, count))
                sampled = valid_indices[::step][:count]
                selected = [entries[i] for i in sampled if 0 <= i < n]
                if selected:
                    return selected, 'evidence_indices'

        return self._sample_entries_evenly(entries, count), 'uniform'

    @staticmethod
    def _entries_to_frames_ts(entries: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[float]]:
        frames: List[np.ndarray] = []
        ts_list: List[float] = []
        for e in entries:
            frame = e.get('frame') if isinstance(e, dict) else None
            if frame is None:
                continue
            frames.append(frame.copy() if hasattr(frame, 'copy') else frame)
            ts = e.get('mono_ts') if isinstance(e, dict) else None
            if ts is not None:
                try:
                    ts_list.append(float(ts))
                except Exception:
                    pass
        return frames, ts_list

    @staticmethod
    def _crop_zone_from_entries(
        entries: List[Dict[str, Any]],
        zone_polygon: List,
    ) -> Tuple[List[np.ndarray], List[float]]:
        if not entries or not zone_polygon:
            return [], []

        points = np.array(zone_polygon)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)

        padding = 20
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)

        out_frames: List[np.ndarray] = []
        out_ts: List[float] = []
        for entry in entries:
            frame = entry.get('frame') if isinstance(entry, dict) else None
            if frame is None:
                continue
            h, w = frame.shape[:2]
            x2_bounded = min(w, int(x2) + padding)
            y2_bounded = min(h, int(y2) + padding)
            roi = frame[y1:y2_bounded, x1:x2_bounded].copy()
            if roi.size <= 0:
                continue
            out_frames.append(roi)
            ts = entry.get('mono_ts') if isinstance(entry, dict) else None
            if ts is not None:
                try:
                    out_ts.append(float(ts))
                except Exception:
                    pass
        return out_frames, out_ts

    def _estimate_t_peak_sec(self, episode: Episode, stream_fps: Optional[float]) -> float:
        """Estimate activity peak time in seconds from episode evidence."""
        frame_indices, _ = self._split_evidence_points(episode.evidence_frames)
        if frame_indices:
            if episode.confidence_history:
                n = min(len(frame_indices), len(episode.confidence_history))
                conf = np.array(episode.confidence_history[:n], dtype=np.float32)
                peak_pos = int(np.argmax(conf)) if n > 0 else len(frame_indices) // 2
                peak_frame = int(frame_indices[min(peak_pos, len(frame_indices) - 1)])
            else:
                peak_frame = int(frame_indices[len(frame_indices) // 2])

            if stream_fps and stream_fps > 0:
                return max(0.0, peak_frame / float(stream_fps))

        duration = max(0.0, float(episode.get_duration_seconds()))
        return duration * 0.5

    def _build_video_window(
        self,
        episode: Episode,
        stream_fps: Optional[float] = None,
        clip_seconds: Optional[float] = None,
    ) -> Tuple[float, List[float]]:
        """Build [start, end] seconds around estimated peak."""
        total_sec = float(clip_seconds or self.video_clip_seconds)
        pre_sec = min(float(self.video_clip_pre_seconds), max(0.5, total_sec - 0.5))
        t_peak = self._estimate_t_peak_sec(episode, stream_fps)

        start = max(0.0, t_peak - pre_sec)
        end = start + total_sec

        # Clamp by episode duration only when source FPS is unknown.
        if not stream_fps:
            duration = float(episode.get_duration_seconds())
            if duration > 0:
                end = min(end, duration)
                if end <= start:
                    end = min(duration, start + 1.0)

        return round(t_peak, 3), [round(start, 3), round(end, 3)]

    def create_evidence_packet(
        self,
        episode: Episode,
        frame_buffer: List[np.ndarray],
        frame_entries: Optional[List[Dict[str, Any]]] = None,
        anchor_mono_ts: Optional[float] = None,
        zones: Dict[str, List] = None,
        router_action: Optional[str] = None,
        router_reason: str = '',
        router_q: Optional[Dict[str, float]] = None,
        stream_fps: Optional[float] = None,
        clip_seconds: Optional[float] = None,
    ) -> EvidencePacket:
        """
        Create evidence packet for Tier2 validation.

        Args:
            episode: Episode to create packet for
            frame_buffer: Ring buffer of recent frames
            frame_entries: Optional ring buffer entries with {'frame','mono_ts'}
            anchor_mono_ts: Optional detection anchor timestamp for aligned sampling
            zones: Zone polygons for ROI extraction
            router_action: Optional pre-selected router action
            router_reason: Routing reason string
            router_q: Optional action score dictionary
            stream_fps: Optional source FPS to compute video window
            clip_seconds: Optional clip length override

        Returns:
            EvidencePacket ready for Tier2
        """
        zones = zones or {}

        if router_action is None:
            inferred_action, inferred_reason, inferred_q, _ = self.select_action(
                episode, record_decision=False
            )
            router_action = inferred_action
            if not router_reason:
                router_reason = inferred_reason
            if router_q is None:
                router_q = inferred_q

        selected_mode = 'video'
        resolved_clip_sec = int(round(float(clip_seconds or self.video_clip_seconds or 10.0)))
        clip_window_start_mono_ts = None
        clip_window_end_mono_ts = None
        if anchor_mono_ts is not None:
            try:
                half = float(resolved_clip_sec) / 2.0
                clip_window_start_mono_ts = float(anchor_mono_ts) - half
                clip_window_end_mono_ts = float(anchor_mono_ts) + half
            except Exception:
                clip_window_start_mono_ts = None
                clip_window_end_mono_ts = None

        t_peak_sec = None
        video_window_sec = None
        t_peak_sec, video_window_sec = self._build_video_window(
            episode,
            stream_fps=stream_fps,
            clip_seconds=resolved_clip_sec,
        )

        evidence_indices, evidence_mono_ts = self._split_evidence_points(episode.evidence_frames)
        if anchor_mono_ts is None and evidence_mono_ts:
            anchor_mono_ts = evidence_mono_ts[-1]

        normalized_entries = self._normalize_frame_entries(
            frame_buffer=frame_buffer,
            frame_entries=frame_entries,
        )

        # Sample keyframes
        global_count = min(self.max_keyframes, 12 if selected_mode == 'video' else 8)
        global_entries, global_sampling_source = self._select_entries_for_packet(
            normalized_entries,
            evidence_indices,
            count=global_count,
            anchor_mono_ts=anchor_mono_ts,
            clip_seconds=resolved_clip_sec,
        )
        global_keyframes, global_keyframe_ts = self._entries_to_frames_ts(global_entries)

        # Extract ROI frames if zones available
        cashier_roi = []
        cashier_roi_ts: List[float] = []
        drawer_roi = []
        drawer_roi_ts: List[float] = []
        cashier_count = min(self.max_roi_frames, 16 if selected_mode == 'video' else 12)
        drawer_count = min(self.max_roi_frames // 2, 8 if selected_mode == 'video' else 4)

        if zones.get('cashier'):
            cashier_entries, _ = self._select_entries_for_packet(
                normalized_entries,
                evidence_indices,
                count=cashier_count,
                anchor_mono_ts=anchor_mono_ts,
                clip_seconds=resolved_clip_sec,
            )
            cashier_roi, cashier_roi_ts = self._crop_zone_from_entries(
                cashier_entries,
                zones['cashier'],
            )

        if zones.get('drawer'):
            drawer_entries, _ = self._select_entries_for_packet(
                normalized_entries,
                evidence_indices,
                count=drawer_count,
                anchor_mono_ts=anchor_mono_ts,
                clip_seconds=resolved_clip_sec,
            )
            drawer_roi, drawer_roi_ts = self._crop_zone_from_entries(
                drawer_entries,
                zones['drawer'],
            )

        focus_hints: List[str] = []
        if episode.event_type == 'cash':
            cash_path = str((episode.metadata or {}).get('cash_path', '')).strip().lower()
            if cash_path in {'roi', 'global_assist', 'both'}:
                focus_hints.append(f'cash_path:{cash_path}')
            if cashier_roi:
                focus_hints.append('focus:cashier_roi')
            if drawer_roi:
                focus_hints.append('focus:drawer_roi')
            if global_keyframes:
                focus_hints.append('focus:global_handover')
        elif episode.event_type in {'violence', 'fire'}:
            focus_hints.append('focus:global_scene')
        if video_window_sec:
            focus_hints.append(f"span:{video_window_sec[0]}-{video_window_sec[1]}s")
        focus_hints.append(f"clip_sec:{resolved_clip_sec}")
        focus_hints.append(f"packet_sampling:{global_sampling_source}")

        florence_signals = self._extract_florence_signals(episode.metadata or {})

        packet = EvidencePacket(
            episode_id=episode.episode_id,
            camera_id=episode.camera_id,
            event_type=episode.event_type,
            global_keyframes=global_keyframes,
            cashier_roi_frames=cashier_roi,
            drawer_roi_frames=drawer_roi,
            global_keyframe_ts=global_keyframe_ts,
            cashier_roi_ts=cashier_roi_ts,
            drawer_roi_ts=drawer_roi_ts,
            anchor_mono_ts=anchor_mono_ts,
            clip_sec_used=resolved_clip_sec,
            clip_window_start_mono_ts=clip_window_start_mono_ts,
            clip_window_end_mono_ts=clip_window_end_mono_ts,
            tier1_confidence=episode.get_average_confidence(),
            stability_score=episode.get_stability_score(),
            detection_count=episode.detection_count,
            duration_seconds=episode.get_duration_seconds(),
            start_ts=episode.start_ts,
            end_ts=episode.last_ts,
            tier1_scores=list(episode.confidence_history[-10:]),
            zones_metadata={
                'cashier_zone': zones.get('cashier'),
                'drawer_zone': zones.get('drawer'),
            },
            selected_mode=selected_mode,
            router_action=router_action,
            router_reason=router_reason,
            router_q=router_q or {},
            t_peak_sec=t_peak_sec,
            video_window_sec=video_window_sec,
            focus_hints=focus_hints,
            florence_signals=florence_signals,
        )

        # Update statistics
        if router_action in self.TIER2_ACTIONS:
            self.total_routed += 1
            self.routed_by_type[episode.event_type] = (
                self.routed_by_type.get(episode.event_type, 0) + 1
            )

        return packet

    def _sample_keyframes(
        self,
        frame_buffer: List[np.ndarray],
        evidence_indices: List[int],
        count: int = 8,
    ) -> List[np.ndarray]:
        """
        Sample keyframes from frame buffer.

        Strategy:
        1. Prefer frames at evidence indices
        2. Evenly distribute across episode duration
        3. Include start and end frames
        """
        if not frame_buffer:
            return []

        buffer_len = len(frame_buffer)
        keyframes = []

        # If we have evidence indices, use them
        if evidence_indices:
            valid_indices = [
                i % buffer_len
                for i in evidence_indices
                if 0 <= i % buffer_len < buffer_len
            ]

            if valid_indices:
                step = max(1, len(valid_indices) // max(1, count))
                sampled = valid_indices[::step][:count]

                for idx in sampled:
                    if 0 <= idx < buffer_len:
                        keyframes.append(frame_buffer[idx].copy())

        # If not enough keyframes, sample from buffer
        if len(keyframes) < count and buffer_len > 0:
            step = max(1, buffer_len // max(1, count))
            for i in range(0, buffer_len, step):
                if len(keyframes) >= count:
                    break
                keyframes.append(frame_buffer[i].copy())

        return keyframes[:count]

    def _extract_zone_frames(
        self,
        frame_buffer: List[np.ndarray],
        zone_polygon: List,
        evidence_indices: List[int],
        count: int = 12,
    ) -> List[np.ndarray]:
        """
        Extract and crop frames to zone ROI.

        Args:
            frame_buffer: Frame buffer
            zone_polygon: Zone polygon points
            evidence_indices: Indices of evidence frames
            count: Number of frames to extract

        Returns:
            List of cropped frame ROIs
        """
        if not frame_buffer or not zone_polygon:
            return []

        points = np.array(zone_polygon)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)

        padding = 20
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)

        roi_frames = []
        buffer_len = len(frame_buffer)

        indices_to_use = evidence_indices if evidence_indices else list(range(buffer_len))
        step = max(1, len(indices_to_use) // max(1, count))

        for i in indices_to_use[::step][:count]:
            idx = i % buffer_len
            if 0 <= idx < buffer_len:
                frame = frame_buffer[idx]
                h, w = frame.shape[:2]

                x2_bounded = min(w, int(x2) + padding)
                y2_bounded = min(h, int(y2) + padding)

                roi = frame[y1:y2_bounded, x1:x2_bounded].copy()
                if roi.size > 0:
                    roi_frames.append(roi)

        return roi_frames

    def get_next_packet(self) -> Optional[EvidencePacket]:
        """Get next pending packet (priority order)."""
        if not self.pending_queue:
            return None

        sorted_queue = sorted(
            self.pending_queue,
            key=lambda p: self.PRIORITY_ORDER.index(p.event_type)
            if p.event_type in self.PRIORITY_ORDER else 999,
        )

        if sorted_queue:
            packet = sorted_queue[0]
            self.pending_queue.remove(packet)
            return packet

        return None

    def add_to_queue(self, packet: EvidencePacket):
        """Add packet to pending queue."""
        self.pending_queue.append(packet)

    def get_stats(self) -> Dict:
        """Get router statistics."""
        critic_info = {}
        try:
            critic_info = self.critic.info() if self.critic is not None else {}
        except Exception:
            critic_info = {}
        outcomes = max(1, len(self._outcome_event_ids))
        matched_decision_outcome = len(self._outcome_event_ids & self._decision_event_ids)
        feedback_on_outcome = len(self._outcome_event_ids & self._feedback_event_ids)
        chain_completeness = matched_decision_outcome / float(outcomes)
        feedback_coverage = feedback_on_outcome / float(outcomes)
        empty_state_ratio = (
            float(self.router_step_empty_state_count) / float(max(1, self.router_step_total))
        )
        unmatched_event_id_ratio = max(0.0, 1.0 - chain_completeness)
        return {
            'total_routed': self.total_routed,
            'routed_by_type': self.routed_by_type,
            'pending_queue_size': len(self.pending_queue),
            'thresholds': self.thresholds,
            'total_decisions': self.total_decisions,
            'action_counts': self.action_counts,
            'gemini_ratio': round(self._gemini_ratio(), 4),
            'gemini_target_ratio': self.gemini_target_ratio,
            'router_margin': self.router_margin,
            'policy_loaded': bool(self.policy_model),
            'policy_path': self.policy_path,
            'policy_blend': self.policy_blend,
            'policy_load_error': self.policy_load_error,
            'critic_enabled': bool(self.critic_enabled),
            'critic_shadow_mode': bool(self.critic_shadow_mode),
            'critic_rollout_mode': self.critic_rollout_mode,
            'critic_canary_pct': float(self.critic_canary_pct),
            'critic_live_effective': bool(self._critic_live_enabled()),
            'critic': critic_info,
            'router_steps_path': self.router_steps_path,
            'router_step_counts': dict(self.router_step_counts),
            'router_step_total': int(self.router_step_total),
            'last_router_step_ts': self.last_router_step_ts,
            'chain_completeness': round(float(chain_completeness), 4),
            'feedback_coverage': round(float(feedback_coverage), 4),
            'empty_state_ratio': round(float(empty_state_ratio), 4),
            'unmatched_event_id_ratio': round(float(unmatched_event_id_ratio), 4),
            'required_state_missing_total': int(self.router_step_missing_required_total),
        }


def create_router(config: Dict = None) -> EvidenceRouter:
    """Factory function to create router."""
    return EvidenceRouter(config)
