"""
Shadow Agent — Asynchronous background evaluation and feedback collection.

Ported from the existing ShadowLoopRuntime + FeedbackBuffer design:
- Runs in a daemon thread with an asyncio-safe queue
- Evaluates events without blocking the real-time pipeline
- Collects results into FeedbackBuffer for periodic batch processing
- Feeds RuleUpdater / CriticTrainer when buffer is full

Data flow:
    Main thread → queue.put(event)
    Shadow thread → dequeue → evaluate → buffer.add()
                                              ↓  (batch threshold)
                                         CriticTrainer.train()
                                         RuleUpdater.apply_feedback()
"""

import logging
import threading
import queue
import time
import json
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeedbackBuffer — accumulates evaluation results for batch processing
# ---------------------------------------------------------------------------

class FeedbackBuffer:
    """
    Thread-safe buffer that accumulates shadow evaluation results.
    When the buffer reaches `batch_size`, it triggers a flush callback.
    """

    def __init__(
        self,
        batch_size: int = 50,
        flush_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        persist_dir: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.flush_callback = flush_callback
        self.persist_dir = persist_dir
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_flushed = 0

        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

    def add(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.batch_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Must be called while holding self._lock."""
        if not self._buffer:
            return

        batch = list(self._buffer)
        self._buffer.clear()
        self._total_flushed += len(batch)

        # Persist to disk (append-only JSONL)
        if self.persist_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.persist_dir, f"feedback_{ts}.jsonl")
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    for rec in batch:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                logger.info(f"[FeedbackBuffer] Persisted {len(batch)} records → {path}")
            except Exception as e:
                logger.error(f"[FeedbackBuffer] Persist failed: {e}")

        # Fire callback (e.g. CriticTrainer.train)
        if self.flush_callback:
            try:
                self.flush_callback(batch)
            except Exception as e:
                logger.error(f"[FeedbackBuffer] Flush callback error: {e}")

    def flush(self) -> None:
        """Force-flush the current buffer (e.g. on shutdown)."""
        with self._lock:
            self._flush_locked()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        return self._total_flushed


# ---------------------------------------------------------------------------
# ShadowAgent — daemon thread that evaluates events in the background
# ---------------------------------------------------------------------------

class ShadowAgent:
    """
    Background shadow agent that evaluates detection events asynchronously.

    It does NOT block the main real-time pipeline. Events are pushed onto a
    queue by the main thread; the shadow thread pops them and:

    1. (optionally) re-runs Gemini with a "shadow prompt" variant
    2. Compares Tier-1 vs Tier-2 result
    3. Stores the comparison into FeedbackBuffer
    4. When the buffer is full → triggers CriticTrainer / RuleUpdater

    Lifecycle:
        agent = ShadowAgent(...)
        agent.start()         # spawns daemon thread
        agent.enqueue(event)  # non-blocking
        agent.stop()          # graceful shutdown
    """

    def __init__(
        self,
        scenario_name: str,
        gemini_validator: Any = None,
        critic_trainer: Any = None,
        rule_updater: Any = None,
        *,
        batch_size: int = 50,
        persist_dir: str = "data/shadow_feedback",
        max_queue_size: int = 200,
        prompts_dir: str = "",
    ):
        self.scenario_name = scenario_name
        self.gemini_validator = gemini_validator
        self.critic_trainer = critic_trainer
        self.rule_updater = rule_updater
        self.prompts_dir = prompts_dir

        # Load shadow-specific prompt ({scenario}_shadow.md)
        self.shadow_prompt = self._load_shadow_prompt()
        self._shadow_prompt_mtime: float = 0.0
        self._update_prompt_mtime()

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Wire up feedback buffer with batch callback
        self.feedback_buffer = FeedbackBuffer(
            batch_size=batch_size,
            flush_callback=self._on_batch_ready,
            persist_dir=persist_dir,
        )

        # Stats
        self._events_processed = 0
        self._events_dropped = 0
        self._disagreements = 0
        self._agreements = 0

    # ------------------------------------------------------------------
    # Shadow prompt loading + hot-reload
    # ------------------------------------------------------------------

    def _get_shadow_prompt_path(self) -> str:
        return os.path.join(self.prompts_dir, f"{self.scenario_name}_shadow.md")

    def _load_shadow_prompt(self) -> str:
        path = self._get_shadow_prompt_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            logger.info(f"[ShadowAgent:{self.scenario_name}] Loaded shadow prompt: {path}")
            return prompt
        logger.warning(f"[ShadowAgent:{self.scenario_name}] Shadow prompt not found: {path}")
        return ""

    def _update_prompt_mtime(self) -> None:
        path = self._get_shadow_prompt_path()
        if os.path.exists(path):
            self._shadow_prompt_mtime = os.path.getmtime(path)

    def _check_prompt_reload(self) -> None:
        """Hot-reload shadow prompt if the .md file has been modified."""
        path = self._get_shadow_prompt_path()
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if mtime > self._shadow_prompt_mtime:
                self.shadow_prompt = self._load_shadow_prompt()
                self._shadow_prompt_mtime = mtime
                logger.info(f"[ShadowAgent:{self.scenario_name}] Shadow prompt hot-reloaded.")

    # ------------------------------------------------------------------
    # Thread control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"shadow-{self.scenario_name}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[ShadowAgent:{self.scenario_name}] Started background thread.")

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.feedback_buffer.flush()
        logger.info(
            f"[ShadowAgent:{self.scenario_name}] Stopped. "
            f"Processed={self._events_processed}, Dropped={self._events_dropped}"
        )

    # ------------------------------------------------------------------
    # Enqueue (called from main pipeline thread – non-blocking)
    # ------------------------------------------------------------------

    def enqueue(self, event_data: Dict[str, Any]) -> bool:
        """
        Push an event onto the shadow evaluation queue.

        Returns True if enqueued, False if queue is full (event dropped).
        """
        try:
            self._queue.put_nowait(event_data)
            return True
        except queue.Full:
            self._events_dropped += 1
            logger.warning(
                f"[ShadowAgent:{self.scenario_name}] Queue full — event dropped "
                f"(total dropped: {self._events_dropped})"
            )
            return False

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        logger.info(f"[ShadowAgent:{self.scenario_name}] Loop started.")
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self._evaluate(event)
                self.feedback_buffer.add(result)
                self._events_processed += 1
            except Exception as e:
                logger.error(f"[ShadowAgent:{self.scenario_name}] Eval error: {e}")

        logger.info(f"[ShadowAgent:{self.scenario_name}] Loop exited.")

    # ------------------------------------------------------------------
    # Core evaluation logic
    # ------------------------------------------------------------------

    def _evaluate(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shadow-evaluate a single event.

        Steps:
        1. Hot-reload shadow prompt if .md file changed
        2. If Gemini is available → re-run validation with SHADOW prompt
        3. Compare shadow result with original Tier-1 result
        4. Track agreement/disagreement stats
        5. Return comparison record for the feedback buffer
        """
        # Hot-reload check
        self._check_prompt_reload()

        tier1 = event.get('tier1_result', event)
        event_id = event.get('event_id', f"shadow_{int(time.time()*1000)}")

        # Also capture human feedback if present
        human_fb = event.get('human_feedback', None)

        record = {
            'event_id': event_id,
            'scenario': self.scenario_name,
            'timestamp': datetime.now().isoformat(),
            'tier1_detected': tier1.get('is_detected', False),
            'tier1_confidence': tier1.get('confidence', 0.0),
            'tier1_keywords': tier1.get('matched_keywords', []),
            'human_feedback': human_fb,
        }

        # Shadow Gemini re-verification (using shadow-specific prompt)
        if self.gemini_validator and event.get('frame') is not None:
            try:
                evidence_packet = {
                    'event_type': self.scenario_name,
                    'tier1_confidence': tier1.get('confidence', 0.0),
                    'florence_signals': {
                        'matched_keywords': tier1.get('matched_keywords', []),
                        'object_hints': tier1.get('object_hints', []),
                    },
                    # Pass the shadow prompt so Gemini uses a different perspective
                    'shadow_prompt': self.shadow_prompt,
                }
                is_valid, conf, reason, corrected = (
                    self.gemini_validator.validate_event_evidence(
                        packet=evidence_packet,
                        mode="hybrid",
                        frame=event['frame'],
                    )
                )
                record['shadow_gemini_valid'] = is_valid
                record['shadow_gemini_conf'] = conf
                record['shadow_gemini_reason'] = reason
                record['agreement'] = (
                    is_valid == tier1.get('is_detected', False)
                )

                # Track stats
                if record['agreement']:
                    self._agreements += 1
                else:
                    self._disagreements += 1

            except Exception as e:
                record['shadow_gemini_error'] = str(e)
                record['agreement'] = None

        # If human feedback is present, use that as ground truth
        elif human_fb is not None:
            fb_decision = human_fb if isinstance(human_fb, str) else str(human_fb)
            tier1_correct = (
                (fb_decision == 'accept' and tier1.get('is_detected', False)) or
                (fb_decision == 'decline' and not tier1.get('is_detected', False))
            )
            record['agreement'] = tier1_correct
            record['feedback_source'] = 'human'
            if tier1_correct:
                self._agreements += 1
            else:
                self._disagreements += 1
        else:
            record['shadow_gemini_valid'] = None
            record['agreement'] = None

        return record

    # ------------------------------------------------------------------
    # Batch callback — fires when FeedbackBuffer threshold is reached
    # ------------------------------------------------------------------

    def _on_batch_ready(self, batch: List[Dict[str, Any]]) -> None:
        """
        Called by FeedbackBuffer when batch_size records have accumulated.

        Triggers:
        1. CriticTrainer incremental update (LightGBM refit on new data)
        2. RuleUpdater prompt refinement (if disagreement rate is high)
        """
        logger.info(
            f"[ShadowAgent:{self.scenario_name}] Batch ready: {len(batch)} records"
        )

        # 1. Critic training
        if self.critic_trainer:
            try:
                self.critic_trainer.train(batch)
                logger.info(f"[ShadowAgent] CriticTrainer updated with {len(batch)} records.")
            except Exception as e:
                logger.error(f"[ShadowAgent] CriticTrainer error: {e}")

        # 2. Rule update — only if disagreement rate > threshold
        if self.rule_updater:
            agreements = [r.get('agreement') for r in batch if r.get('agreement') is not None]
            if agreements:
                agree_rate = sum(1 for a in agreements if a) / len(agreements)
                disagree_rate = 1.0 - agree_rate

                if disagree_rate > 0.30:  # >30% disagreement → refine rules
                    logger.warning(
                        f"[ShadowAgent:{self.scenario_name}] "
                        f"High disagreement ({disagree_rate:.0%}). Triggering RuleUpdater."
                    )
                    # Extract sample disagreements for rule refinement
                    disagreements = [
                        r for r in batch
                        if r.get('agreement') is False
                    ][:5]  # max 5 examples

                    feedback_summary = "; ".join(
                        f"T1={r.get('tier1_detected')}/T2={r.get('shadow_gemini_valid')} "
                        f"reason={r.get('shadow_gemini_reason', '')[:100]}"
                        for r in disagreements
                    )
                    try:
                        # Only update shadow prompt (operational prompt is human-managed)
                        self.rule_updater.apply_feedback_to_rules(
                            f"{self.scenario_name}_shadow",
                            feedback_summary,
                        )
                        logger.info(
                            f"[ShadowAgent:{self.scenario_name}] "
                            f"RuleUpdater applied to both operational and shadow prompts."
                        )
                    except Exception as e:
                        logger.error(f"[ShadowAgent] RuleUpdater error: {e}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        total = self._agreements + self._disagreements
        return {
            'scenario': self.scenario_name,
            'running': self._running,
            'events_processed': self._events_processed,
            'events_dropped': self._events_dropped,
            'agreements': self._agreements,
            'disagreements': self._disagreements,
            'disagree_rate': round(self._disagreements / total, 2) if total > 0 else 0.0,
            'queue_size': self._queue.qsize(),
            'buffer_pending': self.feedback_buffer.pending,
            'buffer_total_flushed': self.feedback_buffer.total_flushed,
            'shadow_prompt_loaded': bool(self.shadow_prompt),
        }
