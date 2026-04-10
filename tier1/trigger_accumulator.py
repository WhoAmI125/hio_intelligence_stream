"""Trigger Accumulator — require repeated triggers within a time window."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import config

logger = logging.getLogger(__name__)


class TriggerAccumulator:
    """Prevent single-frame false positives by requiring N triggers in a window."""

    def __init__(self):
        self._triggers: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._cooldowns: dict[tuple[str, str], float] = {}

        self.thresholds = {
            "cash": {"count": 2, "window": 30},
            "fire": {"count": 2, "window": 15},
            "violence": {"count": 2, "window": 10},
        }
        self.cooldown_seconds = config.TRIGGER_COOLDOWN_SECONDS

    def add(self, cam_id: str, scenario: str, timestamp: float | None = None) -> bool:
        """Add a trigger event. Returns True if accumulation threshold is met.

        Args:
            cam_id: Camera identifier
            scenario: "cash", "fire", or "violence"
            timestamp: Monotonic time (defaults to now)

        Returns:
            True if Tier 2 should be invoked
        """
        ts = timestamp or time.monotonic()
        key = (cam_id, scenario)

        # Check cooldown
        if key in self._cooldowns:
            if ts - self._cooldowns[key] < self.cooldown_seconds:
                return False

        # Evict old triggers
        window = self.thresholds[scenario]["window"]
        self._triggers[key] = [t for t in self._triggers[key] if ts - t < window]

        # Add new trigger
        self._triggers[key].append(ts)

        # Check threshold
        required = self.thresholds[scenario]["count"]
        if len(self._triggers[key]) >= required:
            self._cooldowns[key] = ts
            self._triggers[key] = []
            logger.info(
                "[%s] %s trigger accumulated (%d in %ds) → Tier 2",
                cam_id,
                scenario,
                required,
                window,
            )
            return True

        return False

    def reset(self, cam_id: str | None = None) -> None:
        """Reset triggers, optionally for a specific camera only."""
        if cam_id:
            keys_to_remove = [k for k in self._triggers if k[0] == cam_id]
            for k in keys_to_remove:
                del self._triggers[k]
                self._cooldowns.pop(k, None)
        else:
            self._triggers.clear()
            self._cooldowns.clear()
