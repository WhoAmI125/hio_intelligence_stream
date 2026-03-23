# VLM New Deploy Inference Refactor Guide

## Goal

This document defines the exact refactor to fix the current multi-camera inference bottleneck in `vlm_new_deploy`.

Target changes:

1. Replace the current per-camera background inference loop + global `_inference_lock` model with a central `inference queue + N workers`.
2. Stop running the cash ROI secondary caption on every frame. Run it only when the global caption indicates that cash analysis is worth the extra cost.

This guide is intentionally implementation-oriented. It is written so the server team can apply it directly.

---

## Why the current code is slow

### Current bottleneck 1: per-camera threads still serialize on one global lock

Today each camera starts its own inference thread in `model_server/vlm_api.py`, but all Florence inference is still guarded by one process-wide `_inference_lock`.

Current behavior:

- Camera A thread enters `_inference_loop()`
- Camera B thread enters `_inference_loop()`
- Camera C thread enters `_inference_loop()`
- Only one thread can actually call Florence because all of them wait on `_inference_lock`

That means the system behaves like a single shared inference lane, but with unfair thread contention and extra scheduling overhead.

Relevant current code:

- `model_server/vlm_api.py`
  - `_inference_lock = threading.Lock()`
  - `_inference_threads[camera_id] = threading.Thread(...)`
  - `with _inference_lock: orch_result = srv.pipeline_orchestrator.process_frame_sequential(...)`

### Current bottleneck 2: cash path can do 2 Florence calls per inference cycle

`process_frame_sequential()` already shares one full-frame caption, but if the cashier ROI exists it still calls Florence again on the ROI every cycle.

Current behavior:

- 1 full-frame Florence caption
- 1 ROI Florence caption for cash
- CPU keyword matching for all scenarios

On 3 cameras, that often becomes:

- 3 cameras x 2 Florence calls
- serialized by `_inference_lock`
- visible cadence drops to roughly seconds instead of near-base-fps

### Current bottleneck 3: heavy default Florence settings

Current defaults are expensive:

- `FLORENCE_MODEL=microsoft/Florence-2-large`
- `input_size=(448, 448)`
- `max_tokens=512`
- `num_beams=3`

Even after the scheduler refactor, these settings still matter.

---

## Refactor summary

Do not replace `StreamManager`. It is already better than the RTSP handling in `AI_CCTV_final`.

Keep:

- `StreamManager`
- `EvidenceRouter`
- `GeminiValidator`
- `LocalStorage`
- existing event save and flush flow

Replace:

- per-camera `_inference_loop` threads
- global `_inference_lock`
- unconditional cash ROI secondary caption

With:

1. A central `InferenceScheduler`
2. One or two global Florence workers
3. A single reusable `_run_inference_once(camera_id, frame, state, now_ts)` function
4. An orchestrator option to conditionally skip cash ROI secondary caption

---

## New target architecture

```text
RTSP Camera Streams
   -> StreamManager
   -> latest frame per camera
   -> InferenceScheduler dispatcher
   -> central job queue
   -> 1 or 2 inference workers
   -> PipelineOrchestrator
      -> full-frame caption always
      -> cash ROI caption only when needed
   -> EvidenceRouter
   -> GeminiValidator
   -> LocalStorage / events / clips
```

Key effect:

- camera fairness improves
- only one scheduling layer decides who gets processed next
- Florence concurrency is explicit and bounded by worker count
- ROI second-pass cost is paid only when it has value

---

## File-level change list

### Add

- `model_server/inference_scheduler.py`

### Modify

- `model_server/config.py`
- `model_server/main.py`
- `model_server/vlm_api.py`
- `model_server/pipeline_orchestrator.py`
- `.env.example`

### Optional but recommended

- `.env`

---

## Step 1: add `model_server/inference_scheduler.py`

Create a new file:

```python
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("model_server.inference_scheduler")


@dataclass
class InferenceJob:
    camera_id: str
    run_id: int
    frame: Any
    enqueued_at: float


class InferenceScheduler:
    """
    Central scheduler for all cameras.

    Responsibilities:
    - Fair camera polling
    - Enqueue at most one pending job per camera
    - Run inference through a small bounded worker pool
    """

    def __init__(
        self,
        *,
        stream_manager: Any,
        get_state: Callable[[str], Dict[str, Any]],
        process_fn: Callable[[str, Any, Dict[str, Any], float], None],
        workers: int = 1,
        dispatcher_sleep_sec: float = 0.02,
        queue_size: int = 128,
        active_burst_sec: float = 3.0,
        active_burst_fps: float = 3.0,
    ):
        self.stream_manager = stream_manager
        self.get_state = get_state
        self.process_fn = process_fn
        self.workers = max(1, int(workers))
        self.dispatcher_sleep_sec = max(0.005, float(dispatcher_sleep_sec))
        self.active_burst_sec = max(0.5, float(active_burst_sec))
        self.active_burst_fps = max(0.5, float(active_burst_fps))

        self._queue: queue.Queue[InferenceJob] = queue.Queue(maxsize=max(8, int(queue_size)))
        self._stop_event = threading.Event()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._worker_threads: list[threading.Thread] = []
        self._camera_ids: set[str] = set()
        self._lock = threading.RLock()

        self._camera_runtime: Dict[str, Dict[str, Any]] = {}

    def start(self) -> None:
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            return

        self._stop_event.clear()

        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="inference-dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()

        self._worker_threads = []
        for idx in range(self.workers):
            th = threading.Thread(
                target=self._worker_loop,
                name=f"inference-worker-{idx}",
                daemon=True,
            )
            th.start()
            self._worker_threads.append(th)

        logger.info(
            "[InferenceScheduler] started with workers=%d queue_size=%d",
            self.workers,
            self._queue.maxsize,
        )

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop_event.set()

        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=timeout_sec)

        for th in self._worker_threads:
            if th.is_alive():
                th.join(timeout=timeout_sec)

        logger.info("[InferenceScheduler] stopped")

    def register_camera(self, camera_id: str) -> None:
        with self._lock:
            self._camera_ids.add(str(camera_id))
            self._camera_runtime.setdefault(
                str(camera_id),
                {   
                    "last_submit_ts": 0.0,
                    "last_finish_ts": 0.0,
                    "pending": False,
                    "inflight": False,
                    "last_active_ts": 0.0,
                    "jobs_enqueued": 0,
                    "jobs_completed": 0,
                    "jobs_dropped": 0,
                },
            )

    def unregister_camera(self, camera_id: str) -> None:
        with self._lock:
            self._camera_ids.discard(str(camera_id))
            runtime = self._camera_runtime.get(str(camera_id))
            if runtime:
                runtime["pending"] = False
                runtime["inflight"] = False

    def mark_camera_active(self, camera_id: str) -> None:
        with self._lock:
            runtime = self._camera_runtime.setdefault(str(camera_id), {})
            runtime["last_active_ts"] = time.time()

    def get_metrics(self, camera_id: str) -> Dict[str, Any]:
        with self._lock:
            runtime = dict(self._camera_runtime.get(str(camera_id), {}))
        runtime["queue_size"] = self._queue.qsize()
        runtime["worker_count"] = self.workers
        return runtime

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()

            with self._lock:
                camera_ids = list(self._camera_ids)

            for camera_id in camera_ids:
                state = self.get_state(camera_id)
                if not bool(state.get("running")):
                    continue

                run_id = int(state.get("run_id", 0))
                runtime = self._camera_runtime.setdefault(camera_id, {})
                if runtime.get("pending") or runtime.get("inflight"):
                    continue

                base_fps = max(float(state.get("base_fps", 1.5) or 1.5), 0.5)
                target_fps = base_fps

                last_active_ts = float(runtime.get("last_active_ts", 0.0) or 0.0)
                if now - last_active_ts <= self.active_burst_sec:
                    target_fps = max(base_fps, self.active_burst_fps)

                interval = 1.0 / max(target_fps, 0.5)
                last_submit_ts = float(runtime.get("last_submit_ts", 0.0) or 0.0)
                if now - last_submit_ts < interval:
                    continue

                frame = self.stream_manager.get_frame(camera_id) if self.stream_manager else None
                if frame is None:
                    state["last_frame_age_sec"] = 999.0
                    continue

                job = InferenceJob(
                    camera_id=camera_id,
                    run_id=run_id,
                    frame=frame,
                    enqueued_at=now,
                )

                try:
                    self._queue.put_nowait(job)
                    runtime["pending"] = True
                    runtime["last_submit_ts"] = now
                    runtime["jobs_enqueued"] = int(runtime.get("jobs_enqueued", 0)) + 1
                except queue.Full:
                    runtime["jobs_dropped"] = int(runtime.get("jobs_dropped", 0)) + 1

            time.sleep(self.dispatcher_sleep_sec)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            state = self.get_state(job.camera_id)
            runtime = self._camera_runtime.setdefault(job.camera_id, {})

            if not bool(state.get("running")) or int(state.get("run_id", 0)) != job.run_id:
                runtime["pending"] = False
                self._queue.task_done()
                continue

            runtime["pending"] = False
            runtime["inflight"] = True
            started_at = time.time()

            try:
                self.process_fn(job.camera_id, job.frame, state, started_at)
                runtime["jobs_completed"] = int(runtime.get("jobs_completed", 0)) + 1
                runtime["last_finish_ts"] = time.time()
            except Exception as e:
                state["last_error"] = f"Inference worker error: {e}"
                logger.exception("[InferenceScheduler] worker error camera=%s", job.camera_id)
            finally:
                runtime["inflight"] = False
                self._queue.task_done()
```

### Why this file matters

This gives the project a single fair scheduling point.

It also removes the need for `_inference_lock`.

If `workers=1`, the model still runs one inference at a time, but:

- no camera thread contention
- no duplicate scheduling races
- clean backpressure
- predictable fairness

If GPU headroom exists, `workers=2` can be tested later.

---

## Step 2: modify `model_server/config.py`

Add the following environment-backed settings:

```python
INFERENCE_WORKERS = _env_int("INFERENCE_WORKERS", 1)
INFERENCE_QUEUE_SIZE = _env_int("INFERENCE_QUEUE_SIZE", 128)
INFERENCE_ACTIVE_BURST_SEC = _env_float("INFERENCE_ACTIVE_BURST_SEC", 3.0)
INFERENCE_ACTIVE_BURST_FPS = _env_float("INFERENCE_ACTIVE_BURST_FPS", 3.0)

CASH_ROI_SECONDARY_CAPTION = _env_bool("CASH_ROI_SECONDARY_CAPTION", True)
CASH_ROI_SECONDARY_MIN_GLOBAL_CONF = _env_float("CASH_ROI_SECONDARY_MIN_GLOBAL_CONF", 0.45)
FLORENCE_MAX_TOKENS = _env_int("FLORENCE_MAX_TOKENS", 128)
FLORENCE_NUM_BEAMS = _env_int("FLORENCE_NUM_BEAMS", 1)
FLORENCE_INPUT_SIZE = _env_int("FLORENCE_INPUT_SIZE", 384)
```

Important note:

- `FLORENCE_INPUT_SIZE` already exists in some setups. If so, keep one source of truth.
- If the project already defines `_env_int` and `_env_bool`, reuse them.

### Recommended production defaults

For the first rollout:

```env
INFERENCE_WORKERS=1
INFERENCE_QUEUE_SIZE=128
INFERENCE_ACTIVE_BURST_SEC=3.0
INFERENCE_ACTIVE_BURST_FPS=3.0

CASH_ROI_SECONDARY_CAPTION=true
CASH_ROI_SECONDARY_MIN_GLOBAL_CONF=0.45

FLORENCE_MODEL=microsoft/Florence-2-base
FLORENCE_INPUT_SIZE=384
FLORENCE_MAX_TOKENS=128
FLORENCE_NUM_BEAMS=1
```

If the server is already barely fitting `large`, move to `base` first. That alone can make a major difference.

---

## Step 3: modify `model_server/main.py`

### Add global scheduler state

Near the current global variables:

```python
inference_scheduler = None
```

### Initialize the Florence adapter with lighter decode settings

Change the adapter creation block to pass `max_tokens`.

```python
florence_adapter = create_florence_adapter({
    "model": config.FLORENCE_MODEL,
    "backend": config.FLORENCE_BACKEND,
    "device": config.FLORENCE_DEVICE,
    "input_size": (config.FLORENCE_INPUT_SIZE, config.FLORENCE_INPUT_SIZE),
    "cache_dir": str(config.MODELS_DIR),
    "lora_enabled": config.LORA_ENABLED,
    "lora_adapter_path": config.LORA_ADAPTER_PATH,
    "max_tokens": config.FLORENCE_MAX_TOKENS,
})
```

### Pass cash ROI gating settings into the orchestrator

Replace orchestrator config creation with:

```python
orchestrator_cfg = OrchestratorConfig(
    detect_cash=True,
    detect_violence=True,
    detect_fire=True,
    cash_threshold=float(config.CASH_THRESHOLD),
    violence_threshold=float(config.VIOLENCE_THRESHOLD),
    fire_threshold=float(config.FIRE_THRESHOLD),
    cash_dual_path_enabled=True,
    cash_global_assist_threshold=0.30,
    use_cash_roi_secondary_caption=bool(config.CASH_ROI_SECONDARY_CAPTION),
    cash_roi_secondary_min_global_conf=float(config.CASH_ROI_SECONDARY_MIN_GLOBAL_CONF),
)
```

### Create the scheduler after stream manager and orchestrator exist

Add this after `evidence_router` / `gemini_validator` initialization, before startup completes:

```python
from model_server.inference_scheduler import InferenceScheduler
import model_server.vlm_api as legacy_vlm_api

inference_scheduler = InferenceScheduler(
    stream_manager=stream_manager,
    get_state=legacy_vlm_api._get_or_create_state,
    process_fn=legacy_vlm_api._run_inference_once,
    workers=int(config.INFERENCE_WORKERS),
    queue_size=int(config.INFERENCE_QUEUE_SIZE),
    active_burst_sec=float(config.INFERENCE_ACTIVE_BURST_SEC),
    active_burst_fps=float(config.INFERENCE_ACTIVE_BURST_FPS),
)
inference_scheduler.start()
logger.info(
    "InferenceScheduler initialized (workers=%d)",
    int(config.INFERENCE_WORKERS),
)
```

### Stop the scheduler on shutdown

Add to shutdown:

```python
if inference_scheduler:
    inference_scheduler.stop()
```

---

## Step 4: modify `model_server/pipeline_orchestrator.py`

This is where the second major win happens.

### Extend `OrchestratorConfig`

Add fields:

```python
use_cash_roi_secondary_caption: bool = True
cash_roi_secondary_min_global_conf: float = 0.45
```

### Add a helper to decide whether ROI second pass is worth it

Insert this method inside `ScenarioOrchestrator`:

```python
def _should_run_cash_roi_secondary(
    self,
    cash_global_result: ScenarioResult,
) -> bool:
    if not bool(self.config.use_cash_roi_secondary_caption):
        return False

    if bool(cash_global_result.is_detected):
        return True

    if float(cash_global_result.confidence or 0.0) >= float(self.config.cash_roi_secondary_min_global_conf):
        return True

    metadata = cash_global_result.metadata or {}
    florence_signals = metadata.get("florence_signals", {}) if isinstance(metadata, dict) else {}
    matched = florence_signals.get("matched_keywords", []) if isinstance(florence_signals, dict) else []
    object_hints = florence_signals.get("object_hints", []) if isinstance(florence_signals, dict) else []
    global_keywords = florence_signals.get("global_keywords", []) if isinstance(florence_signals, dict) else []

    return bool(matched or object_hints or global_keywords)
```

### Replace `process_frame_sequential()` with gated ROI logic

Replace the current implementation with:

```python
def process_frame_sequential(
    self,
    frame: np.ndarray,
    zones: Optional[Dict[str, List]] = None
) -> OrchestratorResult:
    start_time = time.time()
    self.total_frames += 1

    effective_zones = {
        'cashier': zones.get('cashier') if zones else self.config.cashier_zone,
        'drawer': zones.get('drawer') if zones else self.config.drawer_zone
    }

    scenario_inputs = self._prepare_scenario_inputs(frame, effective_zones)

    caption_time = time.time()
    shared_caption = self.vlm.infer(frame, "", max_new_tokens=self.vlm.max_tokens, num_beams=1)
    caption_ms = (time.time() - caption_time) * 1000

    results = {}
    for inp in scenario_inputs:
        if inp['scenario_name'] != 'cash':
            result = self._run_single_scenario(
                inp['scenario'], inp['frame'],
                inp['prompt'], inp['zone'],
                inp.get('global_frame'),
                inp.get('cash_dual_path', False),
                shared_caption=shared_caption
            )
            results[inp['scenario_name']] = result
            continue

        has_cashier_zone = bool(effective_zones.get('cashier'))

        cash_global_result = self._run_single_scenario(
            inp['scenario'],
            frame,
            inp['prompt'],
            'full',
            global_frame=None,
            cash_dual_path=False,
            shared_caption=shared_caption,
            global_caption=None,
        )

        if not has_cashier_zone:
            results['cash'] = cash_global_result
            continue

        if not self._should_run_cash_roi_secondary(cash_global_result):
            results['cash'] = cash_global_result
            continue

        cropped_caption = self.vlm.infer(
            inp['frame'],
            "",
            max_new_tokens=self.vlm.max_tokens,
            num_beams=1,
        )

        cash_roi_result = self._run_single_scenario(
            inp['scenario'], inp['frame'],
            inp['prompt'], inp['zone'],
            inp.get('global_frame'),
            inp.get('cash_dual_path', False),
            shared_caption=cropped_caption,
            global_caption=shared_caption,
        )
        results['cash'] = cash_roi_result

    detections = self._results_to_detections(results)
    total_time = (time.time() - start_time) * 1000

    if self.logger:
        for name, result in results.items():
            self.logger.log_agent_inference(
                scenario_type=name,
                frame_idx=self.total_frames,
                result=result.to_dict()
            )

        self.logger.log_orchestrator_frame(
            frame_idx=self.total_frames,
            scenario_results={k: v.to_dict() for k, v in results.items()},
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
        metadata={
            'caption_ms': caption_ms,
            'shared_caption': shared_caption[:200],
            'cash_roi_secondary_enabled': bool(self.config.use_cash_roi_secondary_caption),
        }
    )
```

### Why this change is important

Before:

- full-frame caption always
- cash ROI caption always if cashier zone exists

After:

- full-frame caption always
- cash ROI caption only when:
  - global cash already looks promising, or
  - matched cash-like signals exist

This keeps cash quality while eliminating the wasteful second caption on most negative frames.

---

## Step 5: refactor `model_server/vlm_api.py`

This is the largest change.

### Remove old model-level concurrency controls

Delete or deprecate:

```python
_inference_threads: dict[str, threading.Thread] = {}
_inference_lock = threading.Lock()
```

Keep `_worker_locks` only if you still want per-camera state mutation safety.

### Extend per-camera state

In `_get_or_create_state()`, add:

```python
"scheduler": {
    "pending": False,
    "inflight": False,
    "jobs_enqueued": 0,
    "jobs_completed": 0,
    "jobs_dropped": 0,
},
"last_inference_started_at": None,
"last_inference_finished_at": None,
```

### Add a reusable one-shot inference function

Do not keep the old `_inference_loop()` as the main execution path.

Add this new function:

```python
def _run_inference_once(camera_id: str, frame, state: dict[str, Any], started_at: float) -> None:
    srv = _get_server_modules()

    from model_server.scenarios.base_scenario import CaptionAnalyzer
    from model_server.scenarios import ScenarioType

    now = started_at
    state["frame_count"] += 1
    state["last_frame_age_sec"] = 0.0
    prev_started = state.get("last_inference_started_at")
    if prev_started:
        state["current_fps"] = 1.0 / max(now - float(prev_started), 0.001)
    state["last_inference_started_at"] = now

    cash_zone_applied = False
    cash_zone_bbox = None
    scenario_results: dict[str, dict[str, Any]] = {}
    full_caption = ""
    cash_caption = ""

    if getattr(srv, "pipeline_orchestrator", None) is not None:
        try:
            zones = {
                "cashier": state.get("cashier_zone", []),
                "drawer": state.get("drawer_zone", []),
            }

            orch_result = srv.pipeline_orchestrator.process_frame_sequential(frame, zones=zones)
            scenario_results = {name: sr.to_dict() for name, sr in orch_result.scenario_results.items()}
            full_caption = orch_result.metadata.get("shared_caption", "")
            cash_zone_applied = len(zones["cashier"]) >= 3

            state["last_vlm"] = {
                "scenario_results": scenario_results,
                "total_inference_time_ms": orch_result.total_inference_time_ms,
                "cashier_zone_applied": cash_zone_applied,
                "cashier_zone_points": len(zones["cashier"]),
                "shared_caption": full_caption,
                "cash_caption": (scenario_results.get("cash", {}) or {}).get("raw_response", ""),
                "source": "orchestrator",
            }
        except Exception as e:
            state["last_error"] = f"Orchestrator error: {e}"
            logger.exception("[VLM API] Orchestrator error for %s", camera_id)
            return
    elif getattr(srv, "florence_adapter", None) is not None:
        try:
            full_caption = srv.florence_adapter.infer(frame, "", max_new_tokens=server_config.FLORENCE_MAX_TOKENS, num_beams=1)
            cash_caption = full_caption
            cashier_zone = state.get("cashier_zone", []) or []
            if len(cashier_zone) >= 3:
                cropped, bbox = srv.florence_adapter.crop_zone(frame, cashier_zone)
                if cropped is not None and getattr(cropped, "size", 0) > 0:
                    cash_caption = srv.florence_adapter.infer(cropped, "", max_new_tokens=server_config.FLORENCE_MAX_TOKENS, num_beams=1)
                    cash_zone_applied = True
                    cash_zone_bbox = [int(v) for v in bbox]
        except Exception as e:
            state["last_error"] = f"Florence error: {e}"
            logger.exception("[VLM API] Florence fallback error for %s", camera_id)
            return

        for scenario_name in ["cash", "fire", "violence"]:
            t0 = time.time()
            try:
                scenario_type = ScenarioType[scenario_name.upper()]
                if scenario_name == "cash" and cash_zone_applied:
                    scenario_caption = f"[ROI]\\n{cash_caption}\\n\\n[GLOBAL]\\n{full_caption}"
                else:
                    scenario_caption = full_caption

                result = CaptionAnalyzer.analyze(scenario_caption, scenario_type)
                result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
                result["raw_response"] = scenario_caption
                result["scenario_type"] = scenario_name
                result["zone"] = "cashier" if (scenario_name == "cash" and cash_zone_applied) else "full"
                if scenario_name == "cash" and cash_zone_applied and cash_zone_bbox is not None:
                    result["zone_bbox"] = cash_zone_bbox
                scenario_results[scenario_name] = result
            except Exception as e:
                scenario_results[scenario_name] = {
                    "error": str(e),
                    "is_detected": False,
                    "confidence": 0.0,
                    "zone": "cashier" if (scenario_name == "cash" and cash_zone_applied) else "full",
                }

        state["last_vlm"] = {
            "scenario_results": scenario_results,
            "total_inference_time_ms": round(sum(r.get("inference_time_ms", 0) for r in scenario_results.values()), 1),
            "cashier_zone_applied": cash_zone_applied,
            "cashier_zone_points": len(state.get("cashier_zone", []) or []),
            "shared_caption": full_caption,
            "cash_caption": cash_caption,
            "source": "fallback",
        }
    else:
        if state["frame_count"] == 1:
            state["last_error"] = "Florence-2 not loaded. Caption analysis only."
        return

    # Keep the rest of the current event pipeline unchanged:
    # - rolling logs
    # - event creation
    # - EvidenceRouter
    # - Gemini validation
    # - clip save
    # - thumbnail save
    # - LoRA collection
    # - burst trigger
    #
    # IMPORTANT:
    # Copy the current logic from the existing _inference_loop()
    # starting at the "Keep a rolling server-side log..." section
    # and reuse it here with minimal edits.

    state["last_inference_finished_at"] = time.time()
```

### Important implementation note

Do not rewrite the event pipeline logic from scratch.

Take the current code from the existing `_inference_loop()` and move the body after the orchestrator call into `_run_inference_once()`.

Only the scheduling and invocation model should change.

### Replace start flow

In `vlm_start()`:

- keep RTSP validation
- keep duplicate RTSP protection
- keep stream startup
- remove per-camera thread creation
- register the camera with the central scheduler

Replace the tail of the function with:

```python
with worker_lock:
    state["running"] = True
    state["status"] = "running"
    state["last_error"] = ""
    state["server_start_time"] = datetime.now().isoformat()
    state["frame_count"] = 0
    state["run_id"] += 1

if getattr(srv, "inference_scheduler", None) is not None:
    srv.inference_scheduler.register_camera(camera_id)

logger.info(f"[VLM API] Started: {rtsp_url} for camera {camera_id}")
return {"success": True, "camera_id": camera_id}
```

### Replace stop flow

In `vlm_stop()`:

- stop setting/joining per-camera inference thread
- unregister the camera from scheduler

Use:

```python
with worker_lock:
    state["running"] = False
    state["status"] = "stopping"
    state["run_id"] += 1

if getattr(srv, "inference_scheduler", None) is not None:
    srv.inference_scheduler.unregister_camera(camera_id)

stream_stopped = True
if srv.stream_manager:
    try:
        stream_stopped = bool(srv.stream_manager.remove_camera(camera_id))
    except Exception:
        stream_stopped = False

with worker_lock:
    state["status"] = "stopped" if stream_stopped else "error"
    if stream_stopped:
        state["last_error"] = ""

return {
    "success": stream_stopped,
    "stream_stopped": stream_stopped,
}
```

### Update status endpoint to expose scheduler metrics

Inside `vlm_status()`:

```python
scheduler_metrics = {}
if getattr(srv, "inference_scheduler", None) is not None:
    try:
        scheduler_metrics = srv.inference_scheduler.get_metrics(camera_id)
    except Exception:
        scheduler_metrics = {}
```

Then return:

```python
"scheduler": scheduler_metrics,
```

### Remove old `_inference_loop()` after migration

Once the one-shot function is wired to the scheduler, delete `_inference_loop()` or keep it only temporarily behind a feature flag.

Do not keep both active.

---

## Step 6: update `.env.example`

Add:

```env
INFERENCE_WORKERS=1
INFERENCE_QUEUE_SIZE=128
INFERENCE_ACTIVE_BURST_SEC=3.0
INFERENCE_ACTIVE_BURST_FPS=3.0

CASH_ROI_SECONDARY_CAPTION=true
CASH_ROI_SECONDARY_MIN_GLOBAL_CONF=0.45

FLORENCE_MAX_TOKENS=128
FLORENCE_NUM_BEAMS=1
```

If there is no explicit `FLORENCE_INPUT_SIZE`, add:

```env
FLORENCE_INPUT_SIZE=384
```

---

## Recommended rollout order

### Phase 1: safe low-risk speed win

Apply first:

1. `FLORENCE_NUM_BEAMS=1`
2. `FLORENCE_MAX_TOKENS=128`
3. `FLORENCE_INPUT_SIZE=384`
4. `microsoft/Florence-2-base`

This is the easiest immediate speed win.

### Phase 2: scheduler migration

Apply next:

1. add `InferenceScheduler`
2. move to `_run_inference_once()`
3. remove `_inference_lock`
4. remove per-camera inference thread creation

### Phase 3: ROI gating

Apply after scheduler is stable:

1. add `use_cash_roi_secondary_caption`
2. gate ROI second pass on global confidence/signals

This sequencing reduces rollout risk.

---

## Expected gains

These are realistic directional gains, not guaranteed exact numbers.

### Scheduler refactor

Expected benefits:

- fairer processing across cameras
- no per-camera thread contention around one global lock
- less wasted wakeup/spin behavior
- more stable multi-camera cadence

### ROI gating

If cashier ROI exists for all cameras and most frames are negative:

- Florence calls for cash path can drop from 2 per cycle to near 1 per cycle on most frames
- cash-heavy scenes still get ROI second pass when needed

### Combined effect

For a 3-camera single-GPU deployment, this refactor should usually move the system from:

- "serialized and bursty, often seconds between useful updates"

to:

- "steady queue-driven cadence"
- "more predictable latency"
- "significantly fewer unnecessary ROI caption calls"

---

## Validation checklist

After applying the refactor, verify the following:

1. `GET /api/vlm/status/?camera_id=...` returns `scheduler` metrics.
2. `scheduler.jobs_enqueued` increases steadily.
3. `scheduler.jobs_dropped` stays near zero during normal load.
4. `current_fps` no longer collapses as sharply when adding camera 2 and 3.
5. `recent_inference_logs` still populate.
6. `recent_events` still populate.
7. Tier-2 validation still runs normally.
8. clip save and thumbnail save still work.
9. stopping one camera no longer impacts the others.

### Must-check regression

Cash detection quality must be rechecked after ROI gating.

Specifically test:

- obvious cash handover
- weak small-bill handover
- no cash but hand movement near cashier
- customer standing idle at register

If cash misses increase, lower:

```env
CASH_ROI_SECONDARY_MIN_GLOBAL_CONF=0.35
```

---

## Recommended production values for first deployment

If the server is single GPU and currently overloaded:

```env
FLORENCE_MODEL=microsoft/Florence-2-base
FLORENCE_DEVICE=cuda
FLORENCE_INPUT_SIZE=384
FLORENCE_MAX_TOKENS=128
FLORENCE_NUM_BEAMS=1

BASE_FPS=1.5
BURST_FPS=4.0

INFERENCE_WORKERS=1
INFERENCE_QUEUE_SIZE=128
INFERENCE_ACTIVE_BURST_SEC=3.0
INFERENCE_ACTIVE_BURST_FPS=3.0

CASH_ROI_SECONDARY_CAPTION=true
CASH_ROI_SECONDARY_MIN_GLOBAL_CONF=0.45
```

If this is stable and GPU headroom exists, only then test:

```env
INFERENCE_WORKERS=2
```

Do not start with 2 workers unless the server is known to have enough VRAM and the Florence model remains stable under concurrent CUDA use.

---

## Final implementation note

The correct design is:

- RTSP remains camera-local
- inference scheduling becomes global
- expensive ROI second pass becomes conditional

That is the right combination of `vlm_new_deploy` strengths and `AI_CCTV_final` strengths.

Do not import `AI_CCTV_final` RTSP code into this project. Import its inference gating idea only.
