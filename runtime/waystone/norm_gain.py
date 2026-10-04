"""Explicit, model-owned gain cache; unused by the serving backend by default."""
from __future__ import annotations

import hashlib
import inspect
import types

GRAPH_SHA256 = "d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939"


class GainCache:
    """Own per-instance overrides and restore them before releasing model storage."""

    def __init__(self):
        self._entries = []
        self.closed = False
        self.module_count = 0
        self.gain_bytes = 0

    def close(self):
        if self.closed:
            return
        for module, had_forward, previous in reversed(self._entries):
            if had_forward:
                module.forward = previous
            else:
                delattr(module, "forward")
        self._entries.clear()
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Gain cache is already closed")
        return self

    def __exit__(self, *exception):
        self.close()


def _forward(gain, weight, version):
    # Closure owns only the per-module gain/weight; never a model or handle.
    import torch

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Gain cache requires eval inference")
        if self.weight is not weight or self.weight._version != version:
            raise RuntimeError("Frozen norm weight changed; gain cache is invalid")
        if x.device != weight.device or x.dtype != torch.bfloat16 or x.shape[-1] != weight.numel():
            raise RuntimeError("Unexpected norm input device, dtype or width")
        return (self._norm(x.float()) * gain).type_as(x)

    return forward


def _install(model, norm_class, expected_names, *, device_type):
    """Internal CPU-test seam; production callers use install_gain_cache only."""
    import torch

    names = tuple(expected_names)
    modules = [(name, module) for name, module in model.named_modules() if type(module) is norm_class]
    if len(set(names)) != len(names) or {name for name, _ in modules} != set(names):
        raise RuntimeError("Unexpected zero-centered norm inventory")
    if model.training or any(module.training for _, module in modules):
        raise RuntimeError("Gain cache requires a model in eval mode")
    validated = []
    for _, module in modules:
        weight = module.weight
        if weight.device.type != device_type or weight.dtype != torch.bfloat16 or weight.ndim != 1:
            raise RuntimeError("Gain cache requires unchanged BF16 norm weights on the selected device")
        # Inference tensors without version tracking cannot support this guard.
        validated.append((module, weight, weight._version))
    owner = GainCache()
    try:
        for module, weight, version in validated:
            with torch.inference_mode():
                gain = 1.0 + weight.float()
            previous = module.__dict__.get("forward")
            had_forward = "forward" in module.__dict__
            replacement = types.MethodType(_forward(gain, weight, version), module)
            owner._entries.append((module, had_forward, previous))
            module.forward = replacement
            owner.module_count += 1
            owner.gain_bytes += gain.numel() * gain.element_size()
        return owner
    except BaseException:
        owner.close()
        raise


def install_gain_cache(model):
    """Opt-in only for the pinned XPU graph; no gated norm or class-global patch."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as graph

    with open(inspect.getfile(graph), "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != GRAPH_SHA256 or torch.__version__ != "2.14.1+xpu":
        raise RuntimeError("Unqualified graph or PyTorch runtime for gain cache")
    names = []
    for index in range(32):
        prefix = f"model.language_model.layers.{index}"
        names.extend((prefix + ".input_layernorm", prefix + ".post_attention_layernorm"))
        if model.config.text_config.layer_types[index] == "full_attention":
            names.extend((prefix + ".self_attn.q_norm", prefix + ".self_attn.k_norm"))
    names.append("model.language_model.norm")
    if len(names) != 81:
        raise RuntimeError("Unexpected pinned text layer layout")
    return _install(model, graph.Qwen3_5RMSNorm, names, device_type="xpu")
