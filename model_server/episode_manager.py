"""
Episode Manager - Temporal State Management for Detection Episodes

Manages detection episodes across frames:
- Accumulates evidence over time for stable detection
- Tracks episode lifecycle: IDLE -> ACTIVE -> VALIDATING -> DONE
- Implements label stability scoring to reduce flicker
- Manages cooldown periods to prevent duplicate events

Reference: CCTV_Florence_Tiered_Architecture_Guide.md Section 6
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .base_detector import Detection

EvidencePoint = Union[int, Tuple[int, float]]


class EpisodeState(Enum):
    """Episode lifecycle states"""
    IDLE = "idle"           # No activity detected
    ACTIVE = "active"       # Potential event detected, accumulating evidence
    VALIDATING = "validating"  # Enough evidence, awaiting Tier2 validation
    DONE = "done"           # Episode completed (validated or rejected)


@dataclass
class Episode:
    """
    Single detection episode spanning multiple frames.

    An episode represents a potential event (cash transaction, violence, etc.)
    that is tracked across frames to accumulate evidence and ensure stability.
    """
    episode_id: str
    camera_id: int
    event_type: str  # 'cash' | 'violence' | 'fire'
    state: EpisodeState = EpisodeState.IDLE

    # Timestamps
    start_ts: datetime = field(default_factory=datetime.now)
    last_ts: datetime = field(default_factory=datetime.now)
    cooldown_until: Optional[datetime] = None

    # Detection history (sliding window)
    label_history: deque = field(default_factory=lambda: deque(maxlen=10))
    confidence_history: List[float] = field(default_factory=list)

    # Evidence collection
    evidence_frames: List[EvidencePoint] = field(default_factory=list)  # int | (frame_idx, mono_ts)
    keyframe_indices: List[int] = field(default_factory=list)  # Best frames
    detection_count: int = 0

    # Validation
    tier2_sent: bool = False
    tier2_result: Optional[Dict] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def add_detection(
        self,
        label: str,
        confidence: float,
        frame_idx: int,
        is_keyframe: bool = False,
        frame_mono_ts: Optional[float] = None,
    ):
        """Add a detection to the episode history"""
        self.label_history.append((label, confidence, frame_idx))
        self.confidence_history.append(confidence)
        if frame_mono_ts is not None:
            self.evidence_frames.append((int(frame_idx), float(frame_mono_ts)))
        else:
            self.evidence_frames.append(int(frame_idx))
        self.detection_count += 1
        self.last_ts = datetime.now()

        if is_keyframe:
            self.keyframe_indices.append(frame_idx)

        # Keep confidence history bounded
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]

    def get_stability_score(self) -> float:
        """
        Calculate label stability score (0-1).

        Higher score means more consistent detections.
        Used to determine if episode is stable enough for validation.
        """
        if len(self.label_history) < 3:
            return 0.0

        labels = [label for label, conf, idx in self.label_history]
        if not labels:
            return 0.0

        most_common = max(set(labels), key=labels.count)
        return labels.count(most_common) / len(labels)

    def get_average_confidence(self) -> float:
        """Get average confidence across all detections"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def get_max_confidence(self) -> float:
        """Get maximum confidence seen"""
        if not self.confidence_history:
            return 0.0
        return max(self.confidence_history)

    def get_duration_seconds(self) -> float:
        """Get episode duration in seconds"""
        return (self.last_ts - self.start_ts).total_seconds()

    def is_in_cooldown(self) -> bool:
        """Check if episode is in cooldown period"""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def set_cooldown(self, seconds: float):
        """Set cooldown period"""
        self.cooldown_until = datetime.now() + timedelta(seconds=seconds)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'episode_id': self.episode_id,
            'camera_id': self.camera_id,
            'event_type': self.event_type,
            'state': self.state.value,
            'start_ts': self.start_ts.isoformat(),
            'last_ts': self.last_ts.isoformat(),
            'detection_count': self.detection_count,
            'stability_score': self.get_stability_score(),
            'avg_confidence': self.get_average_confidence(),
            'max_confidence': self.get_max_confidence(),
            'duration_seconds': self.get_duration_seconds(),
            'in_cooldown': self.is_in_cooldown(),
            'tier2_sent': self.tier2_sent,
            'keyframe_count': len(self.keyframe_indices)
        }


class EpisodeManager:
    """
    Manages detection episodes for a camera.

    Handles:
    - Episode creation and lifecycle
    - State transitions (IDLE -> ACTIVE -> VALIDATING -> DONE)
    - Evidence accumulation
    - Cooldown management
    - Tier2 validation triggering
    """

    def __init__(self, camera_id: int, config: Dict = None):
        """
        Initialize Episode Manager.

        Args:
            camera_id: Camera identifier
            config: Configuration with keys:
                - min_detections_for_active: Min detections to enter ACTIVE (default: 2)
                - stability_threshold: Min stability for VALIDATING (default: 0.6)
                - confidence_threshold: Min confidence for VALIDATING (default: 0.7)
                - episode_timeout_seconds: Max episode duration (default: 30)
                - cooldown_seconds: Cooldown after DONE (default: 60)
                - max_episodes_per_type: Max concurrent episodes per type (default: 3)
        """
        self.camera_id = camera_id
        self.config = config or {}

        # Configuration
        self.min_detections = self.config.get('min_detections_for_active', 2)
        self.stability_threshold = self.config.get('stability_threshold', 0.6)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.episode_timeout = self.config.get('episode_timeout_seconds', 30)
        self.cooldown_seconds = self.config.get('cooldown_seconds', 60)
        self.max_episodes = self.config.get('max_episodes_per_type', 3)

        # Episode storage by event type
        self.episodes: Dict[str, List[Episode]] = {
            'cash': [],
            'violence': [],
            'fire': []
        }

        # Statistics
        self.total_episodes_created = 0
        self.total_episodes_validated = 0

    def update(
        self,
        detection: Detection,
        frame_idx: int,
        frame_mono_ts: Optional[float] = None,
    ) -> Optional[Episode]:
        """
        Update episode state with new detection.

        Args:
            detection: Detection object
            frame_idx: Current frame index
            frame_mono_ts: Optional monotonic timestamp for frame-aligned evidence

        Returns:
            Updated or created Episode, or None if in cooldown
        """
        event_type = detection.metadata.get('event_type', detection.label.lower())

        # Normalize event type
        if event_type not in self.episodes:
            if 'cash' in event_type.lower():
                event_type = 'cash'
            elif 'violence' in event_type.lower():
                event_type = 'violence'
            elif 'fire' in event_type.lower() or 'smoke' in event_type.lower():
                event_type = 'fire'
            else:
                event_type = 'cash'  # Default

        # Get or create episode
        episode = self._get_active_episode(event_type)

        if episode is None:
            # Check cooldown for this type
            if self._is_type_in_cooldown(event_type):
                return None

            # Create new episode
            episode = self._create_episode(event_type)
            self.episodes[event_type].append(episode)

        # Add detection
        is_keyframe = detection.confidence > 0.8
        episode.add_detection(
            label=detection.label,
            confidence=detection.confidence,
            frame_idx=frame_idx,
            is_keyframe=is_keyframe,
            frame_mono_ts=frame_mono_ts,
        )

        # Update state
        self._update_state(episode)

        return episode

    def _get_active_episode(self, event_type: str) -> Optional[Episode]:
        """Get active episode for event type"""
        for episode in self.episodes.get(event_type, []):
            if episode.state in [EpisodeState.IDLE, EpisodeState.ACTIVE]:
                # Check timeout
                if episode.get_duration_seconds() > self.episode_timeout:
                    episode.state = EpisodeState.DONE
                    episode.set_cooldown(self.cooldown_seconds)
                    continue
                return episode
        return None

    def _is_type_in_cooldown(self, event_type: str) -> bool:
        """Check if any episode of this type is in cooldown"""
        for episode in self.episodes.get(event_type, []):
            if episode.is_in_cooldown():
                return True
        return False

    def _create_episode(self, event_type: str) -> Episode:
        """Create new episode"""
        # Cleanup old episodes if at limit
        type_episodes = self.episodes.get(event_type, [])
        if len(type_episodes) >= self.max_episodes:
            # Remove oldest completed episode
            done_episodes = [e for e in type_episodes if e.state == EpisodeState.DONE]
            if done_episodes:
                oldest = min(done_episodes, key=lambda e: e.start_ts)
                type_episodes.remove(oldest)

        episode = Episode(
            episode_id=f"{self.camera_id}_{event_type}_{uuid.uuid4().hex[:8]}",
            camera_id=self.camera_id,
            event_type=event_type
        )

        self.total_episodes_created += 1
        return episode

    def _update_state(self, episode: Episode):
        """Update episode state based on accumulated evidence"""
        if episode.state == EpisodeState.DONE:
            return

        if episode.state == EpisodeState.IDLE:
            # IDLE -> ACTIVE: Need minimum detections
            if episode.detection_count >= self.min_detections:
                episode.state = EpisodeState.ACTIVE
                print(f"[EpisodeManager] Episode {episode.episode_id} -> ACTIVE")

        elif episode.state == EpisodeState.ACTIVE:
            # ACTIVE -> VALIDATING: Need stable + confident
            stability = episode.get_stability_score()
            avg_conf = episode.get_average_confidence()

            if stability >= self.stability_threshold and avg_conf >= self.confidence_threshold:
                episode.state = EpisodeState.VALIDATING
                print(f"[EpisodeManager] Episode {episode.episode_id} -> VALIDATING "
                      f"(stability={stability:.2f}, conf={avg_conf:.2f})")

    def should_send_to_tier2(self, episode: Episode) -> bool:
        """
        Determine if episode should be sent to Tier2 (Gemini).

        Based on uncertainty gate logic:
        - Low confidence -> Tier2
        - Label instability -> Tier2
        - High priority events (fire/violence) -> Tier2
        """
        if episode.state != EpisodeState.VALIDATING:
            return False

        if episode.tier2_sent:
            return False

        # High priority events always go to Tier2
        if episode.event_type in ['fire', 'violence']:
            return True

        # For cash: send if uncertain
        stability = episode.get_stability_score()
        avg_conf = episode.get_average_confidence()

        # High confidence + stable = can skip Tier2
        if stability > 0.9 and avg_conf > 0.85:
            return False

        return True

    def mark_tier2_sent(self, episode: Episode):
        """Mark episode as sent to Tier2"""
        episode.tier2_sent = True

    def complete_episode(
        self,
        episode: Episode,
        validated: bool,
        tier2_result: Dict = None
    ):
        """
        Complete an episode after Tier2 validation.

        Args:
            episode: Episode to complete
            validated: Whether event was validated
            tier2_result: Optional result from Tier2
        """
        episode.state = EpisodeState.DONE
        episode.tier2_result = tier2_result
        episode.set_cooldown(self.cooldown_seconds)

        if validated:
            self.total_episodes_validated += 1
            print(f"[EpisodeManager] Episode {episode.episode_id} validated!")
        else:
            print(f"[EpisodeManager] Episode {episode.episode_id} rejected")

    def get_validating_episodes(self) -> List[Episode]:
        """Get all episodes in VALIDATING state"""
        result = []
        for type_episodes in self.episodes.values():
            for episode in type_episodes:
                if episode.state == EpisodeState.VALIDATING:
                    result.append(episode)
        return result

    def get_active_episodes(self) -> List[Episode]:
        """Get all active episodes (IDLE, ACTIVE, or VALIDATING)"""
        result = []
        for type_episodes in self.episodes.values():
            for episode in type_episodes:
                if episode.state not in [EpisodeState.DONE]:
                    result.append(episode)
        return result

    def cleanup_old_episodes(self):
        """Remove old completed episodes"""
        for event_type in self.episodes:
            self.episodes[event_type] = [
                e for e in self.episodes[event_type]
                if e.state != EpisodeState.DONE or e.is_in_cooldown()
            ]

    def get_stats(self) -> Dict:
        """Get manager statistics"""
        active_counts = {}
        for event_type, episodes in self.episodes.items():
            active_counts[event_type] = len([
                e for e in episodes if e.state != EpisodeState.DONE
            ])

        return {
            'camera_id': self.camera_id,
            'total_episodes_created': self.total_episodes_created,
            'total_episodes_validated': self.total_episodes_validated,
            'active_episodes': active_counts,
            'validation_rate': (
                self.total_episodes_validated / self.total_episodes_created
                if self.total_episodes_created > 0 else 0
            )
        }
