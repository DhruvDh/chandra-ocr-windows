"""Offline partial roofline estimates. Standard library only; never loads a model."""
import argparse
import hashlib
import json
from pathlib import Path

PROJECTIONS = {'text_mlp', 'text_linear_attention', 'text_full_attention', 'vocabulary_head', 'vision_blocks'}
DIMENSIONS = {'weight_bytes', 'full_attention_layers', 'query_heads', 'kv_heads', 'attention_head_dim',
              'linear_layers', 'linear_value_heads', 'linear_key_dim', 'linear_value_dim',
              'recurrent_state_bytes', 'vision_layers', 'vision_heads', 'vision_head_dim',
              'vocabulary_positions_per_prefill', 'output_allowance'}
# TextModel::newRequest/forward: FP32 caches, fixed capacity, serial B1, 64-token chunks.
NATIVE_FLOAT_BYTES = 4
NATIVE_CONTEXT_TOKENS = 16384
NATIVE_PREFILL_CHUNK_TOKENS = 64
NATIVE_CACHE_BYTES = {
    'kv': 8 * 2 * NATIVE_CONTEXT_TOKENS * 1024 * NATIVE_FLOAT_BYTES,
    'recurrent': 24 * 32 * 128 * 128 * NATIVE_FLOAT_BYTES,
    'convolution_history': 24 * 8192 * 4 * NATIVE_FLOAT_BYTES,
}


def positive(value, name):
    if type(value) is not int or not 0 < value <= 10**12:
        raise ValueError(name + ' must be a positive bounded integer')
    return value


def build(model, batch, prefix, patches, completion, vision_segments=None, *, queue_concurrency=1):
    """Return hypothetical model-row estimates and source-defined native B1 geometry."""
    if model.get('schema') != 'chandra.roofline-inputs.v1' or model.get('precision') != 'bfloat16':
        raise ValueError('Expected BF16 model assumption schema')
    if model.get('assumptions', {}).get('multiply_add_flops') != 2 or model['assumptions'].get('shared_projection_weight_reads_per_batch') != 1:
        raise ValueError('Expected two FLOPs per multiply-add and ideal single weight read')
    weights, dims = model['projection_elements'], model['dimensions']
    if set(weights) != PROJECTIONS or set(dims) != DIMENSIONS:
        raise ValueError('Incomplete or unexpected model dimensions/projections')
    for name, value in {**weights, **dims}.items():
        positive(value, name)
    for name, value in [('batch', batch), ('prefix', prefix), ('patches', patches), ('completion', completion),
                        ('queue_concurrency', queue_concurrency)]:
        positive(value, name)
    if dims['weight_bytes'] != 2 or dims['recurrent_state_bytes'] != 4:
        raise ValueError('BF16 weights and FP32 recurrent state required')
    if dims['kv_heads'] > dims['query_heads'] or dims['query_heads'] % dims['kv_heads']:
        raise ValueError('Invalid grouped-query head dimensions')
    segments = [patches] if vision_segments is None else list(vision_segments)
    if not segments or any(positive(n, 'vision segment') != n for n in segments) or sum(segments) != patches:
        raise ValueError('Positive per-page vision segments must sum to patch count')
    text = sum(weights[n] for n in ('text_mlp', 'text_linear_attention', 'text_full_attention'))
    dense = text + weights['vocabulary_head']
    kv = dims['full_attention_layers'] * 2 * dims['kv_heads'] * dims['attention_head_dim'] * NATIVE_FLOAT_BYTES
    recurrent = dims['linear_layers'] * dims['linear_value_heads'] * dims['linear_key_dim'] * dims['linear_value_dim'] * dims['recurrent_state_bytes']
    full_chunks, tail_tokens = divmod(prefix, NATIVE_PREFILL_CHUNK_TOKENS)
    chunk_count = full_chunks + bool(tail_tokens)
    text_weight_bytes = dims['weight_bytes'] * text
    head_weight_bytes = dims['weight_bytes'] * weights['vocabulary_head']
    # Prefill produces completion token one. Cached forwards consume tokens 1..D.
    cached_forwards = completion - 1
    decode_traffic = {
        'shared_projection_weight_reads': cached_forwards * dims['weight_bytes'] * dense,
        'attention_kv_reads': batch * kv * (cached_forwards * prefix + cached_forwards * (cached_forwards + 1) // 2),
        'new_kv_writes': batch * kv * cached_forwards,
        'recurrent_matrix_reads_and_writes': 2 * batch * recurrent * cached_forwards,
    }
    def estimate(value, tag):
        return {'value': value, 'status': tag}
    return {
        'schema': 'chandra.roofline-estimates.v1', 'model_revision': model['model_revision'],
        'precision': model['precision'],
        'precision_scope': 'stored_projection_weights; native KV and recurrent storage are FP32',
        'scenario': {'batch': batch, 'prefix_tokens': prefix, 'vision_patches_per_page': patches,
                     'completion_tokens_per_page': completion, 'vision_segments_per_page': segments,
                     'identical_row_shapes_assumed': True,
                     'batch_semantics': 'hypothetical_model_rows_in_one_invocation',
                     'matches_current_native_model_batch': batch == 1,
                     'client_queue_concurrency': queue_concurrency},
        'native_engine': {'execution': 'serial_B1', 'model_rows_per_invocation': 1,
                          'active_gpu_request_slots': 1, 'waiting_request_slots': 1,
                          'queue_concurrency_changes_model_batch_or_weight_reuse': False},
        'source_sha256': dict(model['source_sha256']),
        'empirical_roofs': {'effective_dram_bytes_per_second': None,
                           'precision_shape_compatible_flops_per_second': None,
                           'ordinary_launch_seconds_per_call': None},
        'conditional_cached_decode_dram_scenario': {
            'status': 'conditional_logical_traffic_assumed_to_stream_through_dram',
            'cached_forwards': cached_forwards,
            'prefill_produces_first_completion_token': True,
            'first_attention_context_tokens': prefix + 1 if cached_forwards else None,
            'last_attention_context_tokens': prefix + cached_forwards if cached_forwards else None,
            'bytes_per_batch': {**decode_traffic, 'subtotal': sum(decode_traffic.values())},
            'bytes_per_completed_page': {**{k: v / batch for k, v in decode_traffic.items()},
                                         'subtotal': sum(decode_traffic.values()) / batch},
            'other_traffic_bytes': None,
            'assumptions': [
                'Batch is hypothetical model rows with equal P/G and no padding waste; the current native engine implements only B1.',
                'One shared projection-weight read per cached forward; batch>1 weight amortization requires an unimplemented engine batch.',
                'Client queue concurrency does not change these byte or FLOP estimates.',
                'FP32 KV attention reads include the newly appended token, plus one separate FP32 KV write.',
                'Each cached forward reads and writes the full FP32 recurrent matrices once.',
                'Cache reuse/fusion can reduce DRAM traffic; copies/rereads can increase it. This is not a proven DRAM lower bound.',
                'Excludes convolution history, other activations/scratch, nonlinear work, prefill, vision and host costs.',
                'A zero cached-decode subtotal at completion=1 still requires prefill and vision work.',
            ],
        },
        'phases': {
            'vision': {'projection_flops': estimate(2 * batch * patches * weights['vision_blocks'], 'shape_estimate'),
                       'attention_flops': estimate(4 * batch * dims['vision_layers'] * dims['vision_heads'] * dims['vision_head_dim'] * sum(n*n for n in segments), 'shape_estimate'),
                       'projection_weight_bytes_once': estimate(dims['weight_bytes'] * weights['vision_blocks'], 'ideal_traffic_estimate')},
            'text_prefill': {'projection_flops': estimate(2 * batch * (prefix * text + dims['vocabulary_positions_per_prefill'] * weights['vocabulary_head']), 'shape_estimate'),
                             'full_attention_flops': estimate(4 * batch * dims['full_attention_layers'] * dims['query_heads'] * dims['attention_head_dim'] * prefix**2, 'ideal_whole_prefix_shape_estimate'),
                             'projection_weight_bytes_once': estimate(dims['weight_bytes'] * dense, 'ideal_whole_prefix_single_weight_sweep'),
                             'one_layer_softmax_pair_peak_bytes': estimate(6 * batch * dims['query_heads'] * prefix**2, 'ideal_whole_prefix_softmax_pair_lifetime_estimate'),
                             'native_chunked_prefill_per_request': {
                                 'model_rows': 1, 'chunk_tokens': NATIVE_PREFILL_CHUNK_TOKENS,
                                 'chunk_count': chunk_count, 'full_chunks': full_chunks,
                                 'last_chunk_tokens': tail_tokens or NATIVE_PREFILL_CHUNK_TOKENS,
                                 'text_projection_weight_bytes_per_chunk': estimate(text_weight_bytes, 'logical_weight_operand_extent'),
                                 'vocabulary_weight_bytes_once': estimate(head_weight_bytes, 'logical_weight_operand_extent'),
                                 'projection_weight_sweep_bytes': estimate(chunk_count * text_weight_bytes + head_weight_bytes, 'conditional_one_weight_sweep_per_chunk_not_measured_dram'),
                                 'dram_traffic_bytes': None,
                                 'assumptions': [
                                     'forward visits every text projection in each successive chunk, then invokes the vocabulary head once for the last row.',
                                     'Sweep bytes count distinct projection operands once per source chunk; shader row groups can request weights repeatedly.',
                                     'Inter-group/inter-chunk cache reuse can reduce off-chip traffic. Required operands are not measured DRAM bytes.',
                                 ],
                             }},
            'decode_step': {'projection_flops': estimate(2 * batch * dense, 'shape_estimate'),
                            'projection_weight_bytes_once': estimate(dims['weight_bytes'] * dense, 'ideal_traffic_estimate'),
                            'full_attention_flops_at_first_cached_token': estimate(4 * batch * dims['full_attention_layers'] * dims['query_heads'] * dims['attention_head_dim'] * (prefix+1), 'shape_estimate'),
                            'ideal_projection_flop_per_byte': estimate(2 * batch / dims['weight_bytes'], 'ideal_intensity_estimate')}
        },
        'capacity_terms': {'recurrent_state_bytes_per_page': estimate(recurrent, 'storage_estimate'),
                           'kv_bytes_per_page_per_token': estimate(kv, 'storage_estimate'),
                           'recurrent_plus_final_kv_bytes': estimate(batch * (recurrent + kv * (prefix+cached_forwards)), 'logical_occupied_storage_not_resident_allocation'),
                           'recurrent_plus_allowance_kv_bytes': estimate(batch * (recurrent + kv * (prefix+dims['output_allowance']-1)), 'logical_occupied_storage_not_resident_allocation'),
                           'native_request_resident_cache': {
                               'status': 'source_defined_fixed_pinned_native_B1_allocation',
                               'storage': 'four-byte FP32 floats through raw R32_TYPELESS views',
                               'context_capacity_tokens': NATIVE_CONTEXT_TOKENS,
                               'kv_layers': 8, 'kv_buffers_per_layer': 2, 'kv_elements_per_buffer': NATIVE_CONTEXT_TOKENS * 1024,
                               'linear_layers': 24, 'convolution_history_elements_per_layer': 8192 * 4,
                               'recurrent_elements_per_layer': 32 * 128 * 128,
                               'bytes_per_request': {**NATIVE_CACHE_BYTES, 'total': sum(NATIVE_CACHE_BYTES.values())},
                               'independent_of_actual_prefix_completion_and_client_queue_concurrency': True,
                               'other_resident_and_temporary_bytes': None,
                           },
                           'total_peak_process_bytes': None, 'admissible_batch_size': None},
        'measured_timing': {'page_seconds': None, 'completed_pages_per_second': None},
        'analytical_hardware_ceiling': {'seconds': None, 'effective_clock_hz': None,
                                       'bandwidth_bytes_per_second': None,
                                       'compatible_shader_flops_per_second': None,
                                       'xmx_peak_assumed': False},
        'sustained_completed_pages_per_second': None,
        'limits': [model['assumptions']['scope'],
                   'Ideal traffic is not measured DRAM traffic; dense FLOPs omit recurrence and other work.',
                   'Whole-prefix attention/softmax estimates describe an ideal alternate schedule, not the current chunked native allocation or timing.',
                   'Occupied KV terms omit convolution history and do not describe newRequest allocation, which reserves all 16384 tokens immediately.',
                   'No hardware clock, bandwidth, XMX peak or driver compatibility is assumed. BF16 weight storage does not specify shader compute throughput.',
                   'Roofline time max(F/C,Q/BW) needs compatible empirical roofs and actual traffic. Null fields prohibit a numeric hardware ceiling.',
                   'This script cannot validate a supplied hardware calibration or authorize a batch.']
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('roofline-inputs.json'))
    for name in ('batch', 'prefix', 'patches', 'completion'):
        parser.add_argument('--'+name, type=int, help='Hypothetical model rows; native engine supports only B1' if name == 'batch' else None)
    parser.add_argument('--queue-concurrency', type=int, default=1, help='Client queue demand only; does not create an engine batch or weight reuse')
    parser.add_argument('--vision-segments', help='Comma-separated positive segment lengths per page')
    args = parser.parse_args()
    raw = args.inputs.read_bytes()
    model = json.loads(raw)
    values = {name: getattr(args, name) if getattr(args, name) is not None else model['defaults'][name]
              for name in ('batch', 'prefix', 'patches', 'completion')}
    try:
        segments = None if args.vision_segments is None else [int(n) for n in args.vision_segments.split(',')]
        result = build(model, **values, vision_segments=segments, queue_concurrency=args.queue_concurrency)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result['model_assumptions_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
