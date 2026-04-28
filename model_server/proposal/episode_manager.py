"""Episode-level stabilization for v3 proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_server import config


@dataclass
class EpisodePolicy:
    min_hits: int
    min_duration_sec: float
    high_confidence: float


class EpisodeManager:
    """Merge repeated frame proposals into one stable event emission."""

    POLICIES = {
        "cash": EpisodePolicy(min_hits=2, min_duration_sec=1.0, high_confidence=0.85),
        "fire": EpisodePolicy(min_hits=2, min_duration_sec=1.0, high_confidence=0.80),
        "violence": EpisodePolicy(min_hits=3, min_duration_sec=2.0, high_confidence=0.90),
    }

    def __init__(self) -> None:
        self.cooldown_sec = float(getattr(config, "V3_EPISODE_COOLDOWN_SEC", 20.0))
        self.max_gap_sec = float(getattr(config, "V3_EPISODE_MAX_GAP_SEC", 6.0))
        self._episodes: dict[tuple[str, str], dict[str, Any]] = {}

    def update(
        self,
        *,
        camera_id: str,
        scenario: str,
        result: dict[str, Any],
        now_mono: float,
    ) -> dict[str, Any]:
        scenario = str(scenario).lower()
        if scenario not in self.POLICIES:
            return result

        out = dict(result)
        meta = dict(out.get("metadata", {}) or {})
        policy = self.POLICIES[scenario]
        key = (str(camera_id), scenario)
        detected = bool(out.get("is_detected"))
        now_mono = float(now_mono)
        episode = self._episodes.get(key)

        if not detected:
            if episode and (now_mono - float(episode.get("last_seen", now_mono))) > self.max_gap_sec:
                self._episodes.pop(key, None)
            return out

        if episode is None or (now_mono - float(episode.get("last_seen", now_mono))) > self.max_gap_sec:
            episode = {
                "first_seen": now_mono,
                "last_seen": now_mono,
                "hits": 0,
                "last_emitted": 0.0,
                "max_confidence": 0.0,
            }
            self._episodes[key] = episode

        episode["hits"] = int(episode.get("hits", 0)) + 1
        episode["last_seen"] = now_mono
        episode["max_confidence"] = max(float(episode.get("max_confidence", 0.0)), float(out.get("confidence", 0.0) or 0.0))
        duration = max(0.0, now_mono - float(episode.get("first_seen", now_mono)))
        already_emitted = bool(episode.get("emitted", False))
        recently_emitted = (now_mono - float(episode.get("last_emitted", 0.0))) < self.cooldown_sec
        stable = (
            int(episode["hits"]) >= policy.min_hits
            and duration >= policy.min_duration_sec
        ) or float(episode["max_confidence"]) >= policy.high_confidence

        meta["episode"] = {
            "hits": int(episode["hits"]),
            "duration_sec": round(duration, 2),
            "stable": bool(stable),
            "already_emitted": bool(already_emitted),
            "recently_emitted": bool(recently_emitted),
            "max_confidence": round(float(episode["max_confidence"]), 3),
            "policy": {
                "min_hits": policy.min_hits,
                "min_duration_sec": policy.min_duration_sec,
                "high_confidence": policy.high_confidence,
                "cooldown_sec": self.cooldown_sec,
                "max_gap_sec": self.max_gap_sec,
            },
        }

        # Key change: one emission per "episode". An episode ends when the
        # scenario goes non-detected for more than max_gap_sec; only then can a
        # new episode form and emit. This breaks the 20-second metronome where
        # continuous conditions (e.g. staff standing at counter) kept re-emitting.
        if already_emitted:
            out["is_detected"] = False
            meta["episode"]["suppressed"] = "same_episode_already_emitted"
        elif not stable:
            out["is_detected"] = False
            meta["episode"]["suppressed"] = "warming_up"
        elif recently_emitted:
            out["is_detected"] = False
            meta["episode"]["suppressed"] = "cooldown"
        else:
            episode["last_emitted"] = now_mono
            episode["emitted"] = True
            meta["episode"]["emitted"] = True

        out["metadata"] = meta
        return out

    def filter_results(
        self,
        *,
        camera_id: str,
        scenario_results: dict[str, dict[str, Any]],
        now_mono: float,
    ) -> dict[str, dict[str, Any]]:
        return {
            name: self.update(camera_id=camera_id, scenario=name, result=row, now_mono=now_mono)
            for name, row in scenario_results.items()
        }
