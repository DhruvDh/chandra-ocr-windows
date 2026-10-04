"""Inactive stock-op query-axis chunk experiment; no serving entrypoint."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

IDENTITY = 'text-eager-query-chunk256-v1'
GRAPH_SHA256 = 'd0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939'
CHUNK = 256


def chunked_attention(graph, module, query, key, value, attention_mask, scaling):
    """Same full-key reduction per query; changed GEMM outer geometry is unqualified."""
    torch = graph.torch
    key_states = graph.repeat_kv(key, module.num_key_value_groups)
    value_states = graph.repeat_kv(value, module.num_key_value_groups)
    output = torch.empty((*query.shape[:-1], value_states.shape[-1]),
                         dtype=query.dtype, device=query.device)
    for start in range(0, query.shape[-2], CHUNK):
        stop = min(start + CHUNK, query.shape[-2])
        scores = torch.matmul(query[..., start:stop, :], key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[..., start:stop, :]
            scores = scores + mask
        probabilities_fp32 = graph.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        del scores
        probabilities = probabilities_fp32.to(query.dtype)
        del probabilities_fp32
        probabilities = graph.nn.functional.dropout(probabilities, p=0.0, training=False)
        output[..., start:stop, :] = torch.matmul(probabilities, value_states)
        del probabilities
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def scoped_candidate(graph, *, attention_collection_excluded):
    """Caller must independently attest no attention collection/weight observer.

    Exact graph, text class, BF16 eval/no-grad and multi-query only. Single-token
    decode and unsupported masks retain the entering stock/qualified function.
    """
    if attention_collection_excluded is not True:
        raise RuntimeError('Explicit exclusion of attention collection required')
    if hashlib.sha256(Path(graph.__file__).read_bytes()).hexdigest() != GRAPH_SHA256:
        raise RuntimeError('Pinned installed graph required')
    original = graph.eager_attention_forward
    record = {'candidate_identity': IDENTITY, 'chunk_queries': CHUNK,
              'intercepted': 0, 'forwarded': 0, 'chunks': 0, 'shapes': {}, 'layers': {}, 'restored': False}

    def dispatch(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        q, k = query.shape[-2], key.shape[-2]
        mask_supported = (attention_mask is None or
                          (attention_mask.ndim >= 2 and attention_mask.shape[-2] in (1, q)
                           and attention_mask.shape[-1] in (1, k)))
        if (type(module) is graph.Qwen3_5Attention and not module.training
                and not graph.torch.is_grad_enabled() and q > 1 and dropout == 0.0
                and query.dtype == key.dtype == value.dtype == graph.torch.bfloat16
                and (kwargs.get('output_attentions') is None or kwargs.get('output_attentions') is False)
                and getattr(module.config, 'output_attentions', False) is False
                and mask_supported):
            result = chunked_attention(graph, module, query, key, value, attention_mask, scaling)
            record['intercepted'] += 1
            record['chunks'] += (q + CHUNK - 1) // CHUNK
            shape = f'{q}:{k}'
            record['shapes'][shape] = record['shapes'].get(shape, 0) + 1
            layer = str(module.layer_idx)
            row = record['layers'].setdefault(layer, {})
            row[shape] = row.get(shape, 0) + 1
            return result
        record['forwarded'] += 1
        return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

    graph.eager_attention_forward = dispatch
    try:
        yield record
    finally:
        graph.eager_attention_forward = original
        record['restored'] = graph.eager_attention_forward is original
