"""
Florence-2 Adapter with OpenVINO Backend

Provides CPU-optimized inference for Florence-2 VLM using OpenVINO.
Falls back to PyTorch if OpenVINO is not available.

Reference:
- OpenVINO Florence-2: https://docs.openvino.ai/2024/notebooks/florence2-with-output.html
- Florence-2: https://huggingface.co/microsoft/Florence-2-base
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base_adapter import BaseVLMAdapter, VLMInferenceResult


class _BatchRequest:
    __slots__ = ("image", "task", "text_input", "kwargs", "event", "result", "error")

    def __init__(self, image, task, text_input, kwargs):
        self.image = image
        self.task = task
        self.text_input = text_input
        self.kwargs = kwargs
        self.event = threading.Event()
        self.result = None
        self.error = None


class FlorenceAdapter(BaseVLMAdapter):
    """
    Florence-2 adapter with OpenVINO backend for CPU-optimized inference.

    Supports:
    - OpenVINO backend (recommended for CPU)
    - PyTorch backend (fallback)
    - Zone-based image cropping
    - Scenario-specific prompts
    """

    # Supported models
    MODELS = {
        'florence-2-base': 'microsoft/Florence-2-base',
        'florence-2-large': 'microsoft/Florence-2-large',
    }

    def __init__(self, config: Dict = None):
        """
        Initialize Florence adapter.

        Args:
            config: Configuration with keys:
                - model: 'florence-2-base' or 'florence-2-large'
                - backend: 'openvino', 'onnx', or 'pytorch'
                - device: 'cpu' (GPU not yet supported for OpenVINO)
                - input_size: (width, height), default (448, 448)
                - cache_dir: Directory to cache models
                - openvino_model_path: Path to pre-converted OpenVINO model
        """
        super().__init__(config)

        # Florence-specific settings
        self.cache_dir = self.config.get('cache_dir', './model_cache')
        self.openvino_model_path = self.config.get('openvino_model_path', None)

        # Model components
        self.vision_encoder = None
        self.text_decoder = None
        self.tokenizer = None

        # Caption task profile for Florence-2.
        detail = str(self.config.get('caption_detail', 'more')).strip().lower()
        task_map = {
            'basic': '<CAPTION>',
            'caption': '<CAPTION>',
            'detailed': '<DETAILED_CAPTION>',
            'more': '<MORE_DETAILED_CAPTION>',
        }
        self.caption_task = task_map.get(detail, '<MORE_DETAILED_CAPTION>')
        self.num_beams = max(1, int(self.config.get('num_beams', 3)))

        # Micro-batcher (opt-in via FLORENCE_BATCH_SIZE)
        self._batch_size = max(1, int(os.getenv('FLORENCE_BATCH_SIZE', '1')))
        self._batch_wait_ms = max(0, int(os.getenv('FLORENCE_BATCH_WAIT_MS', '40')))
        self._batch_queue: "queue.Queue[_BatchRequest]" = queue.Queue()
        self._batch_thread: Optional[threading.Thread] = None
        self._batch_shutdown = threading.Event()

    def _batch_loop(self) -> None:
        """Background thread that coalesces concurrent infer() calls into batches."""
        wait_s = self._batch_wait_ms / 1000.0
        while not self._batch_shutdown.is_set():
            try:
                first = self._batch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            batch: List[_BatchRequest] = [first]
            if self._batch_size > 1 and wait_s > 0:
                deadline = time.monotonic() + wait_s
                while len(batch) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self._batch_queue.get(timeout=remaining))
                    except queue.Empty:
                        break
            try:
                results = self._run_task_batch(batch)
                for req, res in zip(batch, results):
                    req.result = res
                    req.event.set()
            except Exception as exc:  # pragma: no cover - runtime safety
                for req in batch:
                    req.error = exc
                    req.event.set()

    def _start_batcher(self) -> None:
        if self._batch_size <= 1:
            return
        if self._batch_thread and self._batch_thread.is_alive():
            return
        self._batch_shutdown.clear()
        self._batch_thread = threading.Thread(
            target=self._batch_loop, name='florence-batcher', daemon=True
        )
        self._batch_thread.start()
        print(f"[FlorenceAdapter] micro-batcher started (size={self._batch_size}, wait={self._batch_wait_ms}ms)")

    def initialize(self) -> bool:
        """
        Initialize Florence-2 model.

        Tries OpenVINO first, falls back to PyTorch.
        """
        print(f"\n{'='*50}")
        print(f"Initializing Florence-2 Adapter")
        print(f"Model: {self.model_name}")
        print(f"Backend: {self.backend}")
        print(f"Device: {self.device}")
        print(f"{'='*50}")

        try:
            if self.backend == 'openvino':
                success = self._init_openvino()
            elif self.backend == 'onnx':
                success = self._init_onnx()
            else:
                success = self._init_pytorch()

            if success:
                self.is_initialized = True
                print(f"[FlorenceAdapter] Initialization successful!")
            else:
                print(f"[FlorenceAdapter] Initialization failed, trying fallback...")
                success = self._init_pytorch()
                if success:
                    self.is_initialized = True
                    self.backend = 'pytorch'
                    print(f"[FlorenceAdapter] Fallback to PyTorch successful!")

            print(f"{'='*50}\n")
            return self.is_initialized

        except Exception as e:
            print(f"[FlorenceAdapter] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_openvino(self) -> bool:
        """Initialize with OpenVINO backend"""
        try:
            # Check if OpenVINO is available
            import openvino as ov
            print(f"[FlorenceAdapter] OpenVINO version: {ov.__version__}")

            # Try to load pre-converted model or convert on-the-fly
            if self.openvino_model_path and Path(self.openvino_model_path).exists():
                return self._load_openvino_model(self.openvino_model_path)
            else:
                print("[FlorenceAdapter] OpenVINO model not found, will use PyTorch")
                return False

        except ImportError:
            print("[FlorenceAdapter] OpenVINO not installed")
            return False
        except Exception as e:
            print(f"[FlorenceAdapter] OpenVINO init error: {e}")
            return False

    def _load_openvino_model(self, model_path: str) -> bool:
        """Load pre-converted OpenVINO model"""
        try:
            import openvino as ov

            core = ov.Core()

            # Load vision encoder and text decoder
            vision_path = Path(model_path) / "vision_encoder.xml"
            decoder_path = Path(model_path) / "text_decoder.xml"

            if vision_path.exists() and decoder_path.exists():
                self.vision_encoder = core.compile_model(str(vision_path), "CPU")
                self.text_decoder = core.compile_model(str(decoder_path), "CPU")

                # Load tokenizer
                AutoProcessor, _ = self._import_hf_auto_classes()
                hf_model = self.MODELS.get(self.model_name, self.model_name)
                self.processor = AutoProcessor.from_pretrained(
                    hf_model,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
                )

                print(f"[FlorenceAdapter] Loaded OpenVINO model from {model_path}")
                return True

            print(f"[FlorenceAdapter] OpenVINO model files not found in {model_path}")
            return False

        except Exception as e:
            print(f"[FlorenceAdapter] OpenVINO model loading error: {e}")
            return False

    def _init_onnx(self) -> bool:
        """Initialize with ONNX Runtime backend"""
        # TODO: Implement ONNX Runtime backend
        print("[FlorenceAdapter] ONNX backend not yet implemented")
        return False

    def _init_pytorch(self) -> bool:
        """Initialize with PyTorch backend (fallback)"""
        try:
            import torch
            AutoProcessor, AutoModelForCausalLM = self._import_hf_auto_classes()

            hf_model = self.MODELS.get(self.model_name, self.model_name)

            print(f"[FlorenceAdapter] Loading PyTorch model: {hf_model}")

            # Set device (allow 'auto' to pick CUDA when available)
            req = str(self.device).lower().strip() if self.device is not None else 'auto'
            use_cuda = (req in ('cuda', 'auto')) and torch.cuda.is_available()
            if req == 'cuda' and not use_cuda:
                raise RuntimeError(
                    "FLORENCE_DEVICE=cuda but CUDA is not available. "
                    "Install CUDA-enabled torch and verify NVIDIA driver."
                )
            if use_cuda:
                device = torch.device('cuda')
                dtype = torch.float16
                self.device = 'cuda'
            else:
                device = torch.device('cpu')
                dtype = torch.float32
                self.device = 'cpu'

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                hf_model,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=self.cache_dir
            ).to(device)

            self.model.eval()

            # Load LoRA adapter if available
            self._try_load_lora()

            # Optimize for CPU if needed
            if self.device == 'cpu':
                # Set thread count for CPU
                cpu_count = os.cpu_count() or 4
                torch.set_num_threads(min(cpu_count, 4))
                print(f"[FlorenceAdapter] Using {torch.get_num_threads()} CPU threads")

            # torch.compile for 20-40% faster inference (one-time ~30-60s warmup cost).
            # Opt-out with FLORENCE_COMPILE=0 in .env.
            if use_cuda and os.getenv('FLORENCE_COMPILE', '1') != '0':
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("[FlorenceAdapter] torch.compile enabled (mode=reduce-overhead)")
                except Exception as exc:
                    print(f"[FlorenceAdapter] torch.compile skipped: {exc}")

            # Start micro-batcher if enabled
            self._start_batcher()

            print(f"[FlorenceAdapter] PyTorch model loaded on {self.device}")
            return True

        except ImportError as e:
            print(f"[FlorenceAdapter] Missing dependency: {e}")
            return False

    def _try_load_lora(self) -> None:
        """Try to load LoRA adapter weights if available."""
        lora_path = self.config.get('lora_adapter_path', '')
        lora_enabled = self.config.get('lora_enabled', False)

        if not lora_enabled or not lora_path:
            return

        from pathlib import Path
        adapter_config = Path(lora_path) / "adapter_config.json"
        if not adapter_config.exists():
            print(f"[FlorenceAdapter] No LoRA adapter found at {lora_path}")
            return

        try:
            from peft import PeftModel
            print(f"[FlorenceAdapter] Loading LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()
            print(f"[FlorenceAdapter] LoRA adapter loaded successfully!")
        except ImportError:
            print("[FlorenceAdapter] peft not installed. pip install peft")
        except Exception as e:
            print(f"[FlorenceAdapter] LoRA loading failed (using base model): {e}")

    def _import_hf_auto_classes(self):
        """
        Import HF auto classes with fallback paths for edge-case packaging issues.
        """
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            return AutoProcessor, AutoModelForCausalLM
        except Exception:
            # Fallback: import directly from auto modules.
            from transformers.models.auto.processing_auto import AutoProcessor
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
            return AutoProcessor, AutoModelForCausalLM
        except Exception as e:
            print(f"[FlorenceAdapter] PyTorch init error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def infer(
        self,
        image: np.ndarray,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Run Florence-2 inference.

        Args:
            image: BGR image from OpenCV
            prompt: Text prompt (will be combined with task prefix)
            **kwargs: Additional parameters:
                - max_new_tokens: Maximum tokens to generate
                - num_beams: Beam search beams
                - do_sample: Whether to sample

        Returns:
            Model's text response
        """
        if not self.is_initialized:
            return "Error: Model not initialized"

        if self.backend == 'openvino':
            return self._infer_openvino(image, prompt, **kwargs)
        else:
            return self._infer_pytorch(image, prompt, **kwargs)

    def _infer_openvino(self, image: np.ndarray, prompt: str, **kwargs) -> str:
        """Run inference with OpenVINO backend"""
        # TODO: Implement OpenVINO inference
        # For now, fall back to PyTorch
        return self._infer_pytorch(image, prompt, **kwargs)

    def _run_task_batch(self, requests: List[_BatchRequest]) -> List[dict]:
        """
        Batched Florence-2 forward pass. Keeps greedy beams=1 for deterministic output.
        Assumes all requests share the same task token (validated below).
        Falls back to per-request single inference if tasks differ.
        """
        import torch
        from PIL import Image

        if not requests:
            return []

        # If heterogeneous tasks in batch, fall back to per-request (rare in this repo).
        first_task = requests[0].task
        heterogeneous = any(r.task != first_task or r.text_input for r in requests[1:])
        if heterogeneous or len(requests) == 1:
            return [self._run_task_single(r.image, r.task, r.text_input, **r.kwargs) for r in requests]

        pil_images = [Image.fromarray(self.preprocess_image(r.image)) for r in requests]
        prompts = [first_task] * len(requests)

        inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = int(requests[0].kwargs.get('max_new_tokens', self.max_tokens))
        num_beams = int(requests[0].kwargs.get('num_beams', self.num_beams))
        if num_beams < 1:
            num_beams = 1

        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else None
        with torch.inference_mode():
            if autocast_ctx:
                with autocast_ctx:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        do_sample=False,
                    )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                )

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        results: List[dict] = []
        for text, pil in zip(generated_texts, pil_images):
            parsed = self.processor.post_process_generation(
                text, task=first_task, image_size=(pil.width, pil.height)
            )
            results.append(parsed)
        return results

    def _run_task_single(self, image: np.ndarray, task: str, text_input: Optional[str] = None, **kwargs) -> dict:
        """Original single-image path. Preserved for heterogeneous-task fallback."""
        import torch
        from PIL import Image

        image_rgb = self.preprocess_image(image)
        pil_image = Image.fromarray(image_rgb)
        prompt = f"{task}{text_input}" if text_input else task

        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = kwargs.get('max_new_tokens', self.max_tokens)
        num_beams = int(kwargs.get('num_beams', self.num_beams))
        if num_beams < 1:
            num_beams = 1

        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else None
        with torch.inference_mode():
            if autocast_ctx:
                with autocast_ctx:
                    generated_ids = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        num_beams=num_beams, do_sample=False,
                    )
            else:
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    num_beams=num_beams, do_sample=False,
                )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            generated_text, task=task, image_size=(pil_image.width, pil_image.height)
        )

    def _run_task(self, image: np.ndarray, task: str, text_input: str = None, **kwargs) -> dict:
        """
        Run a Florence-2 task.

        If FLORENCE_BATCH_SIZE > 1, concurrent calls are coalesced by the background
        batcher into a single forward pass. Callers are blocked until their result
        is ready (typical wait: up to FLORENCE_BATCH_WAIT_MS).

        Florence-2 uses fixed task tokens, not free-form prompts.
        Supported tasks:
            <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>
            <OD>, <DENSE_REGION_CAPTION>, <REGION_PROPOSAL>
            <OCR>, <OCR_WITH_REGION>
            <CAPTION_TO_PHRASE_GROUNDING> (+ text_input)
            <OPEN_VOCABULARY_DETECTION> (+ text_input, large model only)
        """
        if self._batch_size > 1 and self._batch_thread and self._batch_thread.is_alive() and not text_input:
            req = _BatchRequest(image, task, text_input, kwargs)
            self._batch_queue.put(req)
            req.event.wait()
            if req.error:
                raise req.error
            return req.result

        import torch
        from PIL import Image

        image_rgb = self.preprocess_image(image)
        pil_image = Image.fromarray(image_rgb)

        # Build prompt: task token only, or task + text for grounding
        if text_input:
            prompt = f"{task}{text_input}"
        else:
            prompt = task

        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = kwargs.get('max_new_tokens', self.max_tokens)
        num_beams = int(kwargs.get('num_beams', self.num_beams))
        if num_beams < 1:
            num_beams = 1

        device_type = device.type
        # CUDA: use autocast for speed/memory; CPU: keep default.
        if device_type == 'cuda':
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            autocast_ctx = None

        with torch.inference_mode():
            if autocast_ctx:
                with autocast_ctx:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        do_sample=False
                    )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False
                )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process with Florence-2's built-in parser
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height)
        )

        return parsed

    def _infer_pytorch(self, image: np.ndarray, prompt: str, **kwargs) -> str:
        """
        Run inference with PyTorch backend.

        Strategy for scenario detection:
        1. Generate caption (task is configurable by caption_detail)
        2. Return caption text for scenario keyword matching
        """
        # Generate scene description using configured caption detail.
        result = self._run_task(image, self.caption_task, **kwargs)

        # Extract caption text
        caption = result.get(self.caption_task, "")
        if not caption:
            # Fallback
            caption = str(result)

        return caption

    def caption(self, image: np.ndarray, detail: str = "more") -> str:
        """
        Generate image caption.

        Args:
            image: BGR image
            detail: 'basic', 'detailed', or 'more'

        Returns:
            Caption text
        """
        task_map = {
            'basic': '<CAPTION>',
            'detailed': '<DETAILED_CAPTION>',
            'more': '<MORE_DETAILED_CAPTION>'
        }
        task = task_map.get(detail, '<MORE_DETAILED_CAPTION>')
        result = self._run_task(image, task)
        return result.get(task, "")

    def detect_objects(self, image: np.ndarray) -> dict:
        """
        Run object detection.

        Returns:
            Dict with 'bboxes' and 'labels'
        """
        result = self._run_task(image, "<OD>")
        return result.get("<OD>", {})

    def ground_phrase(self, image: np.ndarray, phrase: str) -> dict:
        """
        Find locations of a phrase in the image.

        Args:
            image: BGR image
            phrase: Text to find (e.g., "cash money bills")

        Returns:
            Dict with 'bboxes' and 'labels'
        """
        result = self._run_task(image, "<CAPTION_TO_PHRASE_GROUNDING>", phrase)
        return result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        return response

    def infer_scenario(
        self,
        image: np.ndarray,
        scenario_prompt: str,
        zone_polygon: Optional[List] = None,
        **kwargs
    ) -> str:
        """
        Run scenario-specific inference with optional zone cropping.

        Args:
            image: BGR image
            scenario_prompt: Scenario-specific prompt
            zone_polygon: Optional zone to crop

        Returns:
            Model response
        """
        # Crop to zone if provided
        if zone_polygon:
            cropped, bbox = self.crop_zone(image, zone_polygon)
            image = cropped

        return self.infer(image, scenario_prompt, **kwargs)

    def batch_infer_scenarios(
        self,
        image: np.ndarray,
        scenarios: List[Dict],
        **kwargs
    ) -> List[str]:
        """
        Run multiple scenarios on same image.

        Note: This is sequential. For parallel processing,
        use ScenarioOrchestrator with ThreadPoolExecutor.

        Args:
            image: BGR image
            scenarios: List of {'prompt': str, 'zone': optional polygon}

        Returns:
            List of responses
        """
        results = []

        for scenario in scenarios:
            prompt = scenario.get('prompt', '')
            zone = scenario.get('zone', None)

            response = self.infer_scenario(image, prompt, zone, **kwargs)
            results.append(response)

        return results


# Factory function
def create_florence_adapter(config: Dict = None) -> FlorenceAdapter:
    """
    Create a Florence adapter instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized FlorenceAdapter
    """
    adapter = FlorenceAdapter(config)
    if not adapter.initialize():
        raise RuntimeError("FlorenceAdapter initialization failed.")
    return adapter
