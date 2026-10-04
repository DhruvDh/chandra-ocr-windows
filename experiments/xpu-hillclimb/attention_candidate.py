"""Scoped experimental text SDPA using the unchanged eager graph's explicit mask.

Load hybrid (vision SDPA/text eager), then enter masked_text_sdpa(). No model,
device, processor, positions, cache, or serving configuration is changed here.
"""
from contextlib import contextmanager
import hashlib
import inspect


@contextmanager
def masked_text_sdpa(*, mask_kind="boolean", expected_graph_sha256=None, fixture=None):
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as graph
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if mask_kind not in {"boolean", "additive"}:
        raise ValueError("mask_kind must be boolean or additive")
    source = inspect.getfile(graph.Qwen3_5Attention)
    with open(source, "rb") as stream:
        graph_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_graph_sha256 is None or graph_hash != expected_graph_sha256:
        raise RuntimeError("Supply the exact approved installed Qwen3.5 source hash")
    original = graph.eager_attention_forward
    calls = []

    def candidate(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if not isinstance(module, graph.Qwen3_5Attention):
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if module.training or dropout != 0.0 or not module.is_causal:
            raise RuntimeError("Only evaluation causal full-text attention is qualified for this experiment")
        if kwargs.get("output_attentions", False):
            raise RuntimeError("Candidate does not return attention weights")
        if module.config._attn_implementation != "eager":
            raise RuntimeError("Keep text configuration eager so upstream constructs its explicit causal mask")
        if any(t.device.type != "xpu" or t.dtype != torch.bfloat16 for t in (query, key, value)):
            raise RuntimeError("Candidate requires XPU BF16 Q/K/V")
        if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
            raise RuntimeError("Unexpected Q/K/V layout")
        if attention_mask is None or attention_mask.ndim != 4:
            raise RuntimeError("Explicit upstream four-dimensional causal mask is mandatory")
        if attention_mask.device != query.device or attention_mask.shape[-2:] != (query.shape[-2], key.shape[-2]):
            raise RuntimeError("Mask length/device mismatch; no mask slicing or inferred cache alignment")
        if attention_mask.shape[0] not in (1, query.shape[0]) or attention_mask.shape[1] not in (1, query.shape[1]):
            raise RuntimeError("Mask cannot broadcast to attention heads/batch")
        mask = attention_mask
        if mask_kind == "boolean":
            # Eager adds masks. SDPA boolean True means allowed. Reject biases
            # rather than dropping them; this scalar check intentionally costs a
            # synchronization and must be included in performance measurement.
            if not mask.is_floating_point():
                raise RuntimeError("Eager graph must supply an additive floating mask")
            binary = (mask == 0) | (mask == torch.finfo(mask.dtype).min) | torch.isneginf(mask)
            if not bool(binary.all()):
                raise RuntimeError("Cannot convert an additive attention bias into a boolean mask")
            mask = mask == 0
        elif not mask.is_floating_point():
            raise RuntimeError("Additive candidate requires a floating mask")
        if fixture is not None:
            fixture(module, query, key, value, attention_mask, mask)
        groups = module.num_key_value_groups
        key = graph.repeat_kv(key, groups)
        value = graph.repeat_kv(value, groups)
        # Explicit mask carries all causality/padding/packing. is_causal=True
        # would add upper-left alignment and is forbidden for cached decoding.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.OVERRIDEABLE]):
            output = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scaling,
            )
        calls.append({"layer": module.layer_idx, "query_length": query.shape[-2], "kv_length": key.shape[-2], "mask_kind": mask_kind})
        return output.transpose(1, 2).contiguous(), None

    receipt = {"candidate": "explicit-eager-mask-text-sdpa-v1", "graph_sha256": graph_hash, "mask_kind": mask_kind, "math_backend": "disabled", "calls": calls}
    graph.eager_attention_forward = candidate
    try:
        yield receipt
    finally:
        graph.eager_attention_forward = original
