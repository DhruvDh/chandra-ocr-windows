"""Disabled, source-only vision Q slicing through the pinned stock SDPA function.

Import is stdlib-only. There is no CLI, hardware admission, model loading, or
serving integration. CPU checks do not qualify XPU arithmetic or TDR safety.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import threading

IDENTITY = "vision-stock-sdpa-query-chunk32-fenced-source-v2"
GRAPH_SHA256 = "d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939"
SDPA_SHA256 = "cc0ddaf52ca0fb84670154bed07860917f999a8cec047263a145a6be28ab8aad"


def _chunk_sdpa(torch, prior, module, query, key, value, chunk, call_kwargs, completed=None):
    """Retain full K/V and stock scaling; assemble the original [B,Q,H,D] order."""
    batch, heads, q_length, _ = query.shape
    output = torch.empty((batch, q_length, heads, value.shape[-1]),
                         dtype=query.dtype, device=query.device)
    calls = 0
    for start in range(0, q_length, chunk):
        stop = min(start + chunk, q_length)
        part, weights = prior(module, query[..., start:stop, :], key, value, **call_kwargs)
        if (weights is not None or tuple(part.shape) != (batch, stop - start, heads, value.shape[-1])
                or part.dtype != output.dtype or part.device != output.device):
            raise RuntimeError("Pinned stock vision SDPA output contract changed")
        output[:, start:stop] = part
        # Finish this stock attention call and its output copy before submitting
        # another query chunk. This cannot bound one kernel's execution time.
        torch.xpu.synchronize(query.device)
        calls += 1
        if completed is not None:
            completed(stop - start)
    return (output, None), calls


def _dispatch(graph, prior, chunk, record):
    def dispatch(module, query, key, value, attention_mask, dropout=0.0,
                 scaling=None, is_causal=None, position_bias=None, **kwargs):
        if record["active"] is not True or record["failed"]:
            raise RuntimeError("Closed or failed vision-query scope; stale dispatch refused")
        call_kwargs = dict(attention_mask=attention_mask, dropout=dropout, scaling=scaling,
                           is_causal=is_causal, position_bias=position_bias, **kwargs)
        if type(module) is not graph.Qwen3_5VisionAttention:
            record["forwarded_nonvision"] += 1
            return prior(module, query, key, value, **call_kwargs)
        tensors = (query, key, value)
        if (record["in_call"] or record["attempted_vision"] >= 24
                or record["owner_thread"] not in (None, threading.get_ident())
                or module.training or graph.torch.is_grad_enabled()
                or module.config._attn_implementation != "sdpa"
                or module.is_causal is not False or is_causal is not False
                or module.num_key_value_groups != 1 or dropout != 0.0
                or attention_mask is not None or position_bias is not None
                or kwargs.get("output_attentions", False)
                or getattr(module.config, "output_attentions", False)
                or any(t.ndim != 4 or t.device.type != "xpu" or t.dtype != graph.torch.bfloat16 for t in tensors)
                or any(t.device != query.device for t in tensors)
                or str(query.device) != "xpu:0" or any(t.stride(-1) != 1 for t in tensors)
                or query.shape[0] != 1 or query.shape[-2] < 1
                or key.shape[-2] < 1 or key.shape[-2] > 13464 or query.shape[-2] > 13464
                or query.shape[1] != 16 or query.shape[-1] != 64 or key.shape != value.shape
                or query.shape[-2] != key.shape[-2]
                or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]):
            raise RuntimeError("Unqualified vision SDPA geometry or semantics; refused")
        record["attempted_vision"] += 1
        record["owner_thread"] = threading.get_ident()
        record["in_call"] = True
        def completed(rows):
            record["stock_sdpa_calls"] += 1
            record["completed_chunk_fences"] += 1
            record["maximum_query_rows"] = max(record["maximum_query_rows"], rows)
        try:
            result, calls = _chunk_sdpa(graph.torch, prior, module, query, key, value,
                                        chunk, call_kwargs, completed)
        except BaseException:
            record["failed"] = True
            raise
        finally:
            record["in_call"] = False
        record["intercepted_vision"] += 1
        record["geometries"].append([query.shape[-2], key.shape[-2], calls])
        return result
    return dispatch


@contextmanager
def scoped_candidate(graph, *, enabled=False, attention_collection_excluded=False, chunk=32):
    """Opt-in only; caller still needs separately reviewed hardware authority."""
    if enabled is not True or attention_collection_excluded is not True:
        raise RuntimeError("Disabled source candidate; explicit scope and observer exclusion required")
    if type(chunk) is not int or not 1 <= chunk <= 32:
        raise RuntimeError("Fixed bounded query chunk of 1..32 required")
    if hashlib.sha256(Path(graph.__file__).read_bytes()).hexdigest() != GRAPH_SHA256:
        raise RuntimeError("Pinned Qwen3.5 graph source required")
    original = graph.ALL_ATTENTION_FUNCTIONS
    prior = original["sdpa"]
    source = inspect.getsourcefile(prior)
    if (prior.__name__ != "sdpa_attention_forward" or source is None
            or hashlib.sha256(Path(source).read_bytes()).hexdigest() != SDPA_SHA256):
        raise RuntimeError("Unwrapped pinned stock SDPA integration required")
    record = {"identity": IDENTITY, "enabled_by_explicit_caller": True, "query_chunk": chunk,
              "intercepted_vision": 0, "stock_sdpa_calls": 0, "forwarded_nonvision": 0,
              "attempted_vision": 0, "completed_chunk_fences": 0,
              "active": True, "failed": False, "in_call": False, "owner_thread": None,
              "maximum_query_rows": 0, "geometries": [], "restored": False,
              "full_key_value_retained": True, "synchronize_after_each_chunk": True,
              "gpu_qualified": False, "cpu_numerical_tested": False, "tdr_safe": False}
    # Use a graph-local instance, preserving every entering interface and any
    # local overrides. Do not mutate the shared global registration dictionary.
    replacement = type(original)()
    for name in original:
        replacement[name] = original[name]
    replacement["sdpa"] = _dispatch(graph, prior, chunk, record)
    graph.ALL_ATTENTION_FUNCTIONS = replacement
    try:
        yield record
    finally:
        record["active"] = False
        if graph.ALL_ATTENTION_FUNCTIONS is not replacement:
            record["ownership_lost"] = True
            raise RuntimeError("Graph interface ownership lost; foreign interface preserved")
        graph.ALL_ATTENTION_FUNCTIONS = original
        record["restored"] = graph.ALL_ATTENTION_FUNCTIONS is original
