"""Explicit Intel XPU reference backend; model files are supplied by the operator."""
from __future__ import annotations
import base64
import contextlib
import io
import queue
import threading
from pathlib import Path
from typing import Iterator


class XPUBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        max_pixels: int = 4_000_000,
        max_input_tokens: int = 16384,
        max_output_tokens: int = 12384,
        attention_backend: str = "eager",
        normalization: str = "original",
    ):
        if attention_backend not in {"eager", "sdpa", "hybrid"}:
            raise ValueError("Attention backend must be eager, sdpa or hybrid")
        if normalization not in {"original", "gain-only"}:
            raise ValueError("Normalization must be original or gain-only")
        self.attention_backend = attention_backend
        self.normalization = normalization
        self.model_path = Path(model_path)
        self.max_pixels = max_pixels
        self.max_input_tokens = max_input_tokens
        self.max_prefill_tokens = 4096
        self.max_output_tokens = max_output_tokens
        self.model = None
        self.processor = None
        self._gain_cache = None
        self._runtime_prepared = False
        self._device_identity = None
        self._model_verification = None

    def health(self) -> dict:
        memory = None
        if self._runtime_prepared:
            import torch
            if torch.xpu.is_initialized():
                memory = {"allocated_bytes": torch.xpu.memory_allocated(), "reserved_bytes": torch.xpu.memory_reserved(), "peak_allocated_bytes": torch.xpu.max_memory_allocated()}
        return {
            "backend": "transformers-xpu",
            "loaded": self.model is not None,
            "state": "ready" if self.model is not None else "cold",
            "model": "datalab-to/chandra-ocr-2",
            "revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
            "attention_backend": self.attention_backend,
            "normalization": self.normalization,
            "gain_cache": {
                "active": self._gain_cache is not None,
                "modules": self._gain_cache.module_count if self._gain_cache else 0,
                "bytes": self._gain_cache.gain_bytes if self._gain_cache else 0,
            },
            "max_prefill_tokens": self.max_prefill_tokens,
            "processor_size": {"shortest_edge": 3136, "longest_edge": 3145728},
            "device": "xpu:0",
            "dtype": "bfloat16",
            "device_identity": self._device_identity,
            "model_verification": self._model_verification,
            "torch_xpu_memory": memory,
        }

    def load(self) -> None:
        if self.model is not None:
            return
        from .model_identity import verify
        self._model_verification = verify(self.model_path)
        from .bootstrap import prepare_runtime
        prepare_runtime()
        self._runtime_prepared = True
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.xpu.is_available():
            raise RuntimeError("Intel XPU unavailable; CPU fallback is disabled")
        if torch.xpu.device_count() != 1:
            raise RuntimeError("Expected one unambiguous Level Zero GPU")
        properties = torch.xpu.get_device_properties(0)
        if properties.device_id != 0x56A0 or properties.is_integrated_gpu or "Level-Zero" not in properties.platform_name:
            raise RuntimeError("Selected XPU is not the qualified discrete Arc A770 via Level Zero")
        self._device_identity = {"name": properties.name, "device_id": properties.device_id, "uuid": str(properties.uuid), "platform": properties.platform_name}
        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        processor.image_processor.size = {"shortest_edge": 3136, "longest_edge": 3145728}
        from transformers import modeling_utils
        # Arc rejects the loader's single model-sized reservation. Individual
        # parameter allocations retain the exact weights and arithmetic.
        warmup = modeling_utils.caching_allocator_warmup
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        model = gain_cache = None
        try:
            try:
                model = AutoModelForImageTextToText.from_pretrained(self.model_path, local_files_only=True, dtype=torch.bfloat16, device_map={"": "xpu:0"}, attn_implementation={"vision_config": "sdpa", "text_config": "eager"} if self.attention_backend == "hybrid" else self.attention_backend)
            finally:
                modeling_utils.caching_allocator_warmup = warmup
            if self.attention_backend == "hybrid":
                if model.config.vision_config._attn_implementation != "sdpa" or model.config.text_config._attn_implementation != "eager":
                    raise RuntimeError("Hybrid attention configuration was not applied")
            model.eval()
            wrong = {str(p.device) for p in model.parameters() if p.device.type != "xpu"}
            if wrong:
                raise RuntimeError(f"Model parameters escaped XPU: {wrong}")
            if self.normalization == "gain-only":
                from .norm_gain import install_gain_cache
                gain_cache = install_gain_cache(model)
            self.model, self.processor, self._gain_cache = model, processor, gain_cache
        except BaseException:
            if gain_cache is not None:
                gain_cache.close()
            model = processor = gain_cache = None
            self.unload()
            raise

    def unload(self) -> None:
        """Release resident weights at idle; the next request performs a cold load."""
        if self._gain_cache is not None:
            self._gain_cache.close()
            self._gain_cache = None
        self.model = None
        self.processor = None
        if not self._runtime_prepared:
            return
        import gc
        import torch
        gc.collect()
        if torch.xpu.is_initialized():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()

    def generate(self, request: dict, cancel: threading.Event) -> Iterator[str | dict]:
        from PIL import Image
        content = request["messages"][0]["content"]
        text = next(x["text"] for x in content if x["type"] == "text")
        url = next(x["image_url"]["url"] for x in content if x["type"] == "image_url")
        header, payload = url.split(",", 1)
        if header not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
            raise ValueError("Only PNG/JPEG/WebP data URLs are supported")
        data = base64.b64decode(payload, validate=True)
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > self.max_pixels:
                    raise ValueError("Image exceeds configured pixel limit")
                image.load()
                rgb = image.convert("RGB")
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("Invalid or oversized image payload") from error
        tokens = request.get("max_tokens", self.max_output_tokens)
        if not 1 <= tokens <= self.max_output_tokens:
            raise ValueError("Output token limit exceeds runtime allowance")
        if cancel.is_set():
            return
        self.load()
        import torch
        from transformers import TextIteratorStreamer
        conversation = {"role": "user", "content": [{"type": "image", "image": rgb}, {"type": "text", "text": text}]}
        inputs = self.processor.apply_chat_template([conversation], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        if self.attention_backend == "eager" and "image_grid_thw" in inputs and int(inputs.image_grid_thw.prod(-1).sum()) > 4096:
            raise ValueError("Vision input exceeds eager attention memory allowance; qualified SDPA is required")
        prompt_tokens = inputs.input_ids.shape[-1]
        if prompt_tokens > self.max_prefill_tokens:
            raise ValueError("Processed prompt exceeds 4096-token prefill allowance; use a shorter prompt or smaller image")
        if prompt_tokens + tokens > self.max_input_tokens:
            raise ValueError("Input plus requested output exceeds configured context limit")
        inputs = inputs.to("xpu:0")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        end = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if end is not None and end not in eos:
            eos.append(end)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.2, clean_up_tokenization_spaces=False)
        result = {}
        def run():
            try:
                attention_context = contextlib.nullcontext()
                if self.attention_backend in {"sdpa", "hybrid"}:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    attention_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.OVERRIDEABLE])
                with torch.inference_mode(), attention_context:
                    from transformers import StoppingCriteria, StoppingCriteriaList
                    class Cancelled(StoppingCriteria):
                        def __call__(self, input_ids, scores, **kwargs):
                            return cancel.is_set()
                    result["ids"] = self.model.generate(**inputs, max_new_tokens=tokens, do_sample=False, eos_token_id=eos, streamer=streamer, stopping_criteria=StoppingCriteriaList([Cancelled()]))
                torch.xpu.synchronize()
            except BaseException as error:
                result["error"] = error
        thread = threading.Thread(target=run, name="chandra-xpu-generation", daemon=False)
        thread.start()
        try:
            while True:
                try:
                    text_delta = next(streamer)
                except StopIteration:
                    break
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue
                if not cancel.is_set() and text_delta:
                    yield text_delta
        except BaseException:
            cancel.set()
            raise
        finally:
            thread.join()
        if "error" in result:
            raise result["error"]
        if cancel.is_set():
            return
        output_ids = result["ids"][0, prompt_tokens:]
        count = len(output_ids)
        yield {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count, "total_tokens": prompt_tokens + count}, "finish_reason": "stop" if count and int(output_ids[-1]) in eos else "length"}
