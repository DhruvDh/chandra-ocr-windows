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


def positive(value, name):
    if type(value) is not int or not 0 < value <= 10**12:
        raise ValueError(name + ' must be a positive bounded integer')
    return value


def build(model, batch, prefix, patches, completion, vision_segments=None):
    """Return estimates for equal-shaped rows, not performance or memory admission."""
    if model.get('schema') != 'chandra.roofline-inputs.v1' or model.get('precision') != 'bfloat16':
        raise ValueError('Expected BF16 model assumption schema')
    if model.get('assumptions', {}).get('multiply_add_flops') != 2 or model['assumptions'].get('shared_projection_weight_reads_per_batch') != 1:
        raise ValueError('Expected two FLOPs per multiply-add and ideal single weight read')
    weights, dims = model['projection_elements'], model['dimensions']
    if set(weights) != PROJECTIONS or set(dims) != DIMENSIONS:
        raise ValueError('Incomplete or unexpected model dimensions/projections')
    for name, value in {**weights, **dims}.items():
        positive(value, name)
    for name, value in [('batch', batch), ('prefix', prefix), ('patches', patches), ('completion', completion)]:
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
    kv = dims['full_attention_layers'] * 2 * dims['kv_heads'] * dims['attention_head_dim'] * dims['weight_bytes']
    recurrent = dims['linear_layers'] * dims['linear_value_heads'] * dims['linear_key_dim'] * dims['linear_value_dim'] * dims['recurrent_state_bytes']
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
        'scenario': {'batch': batch, 'prefix_tokens': prefix, 'vision_patches_per_page': patches,
                     'completion_tokens_per_page': completion, 'vision_segments_per_page': segments,
                     'identical_row_shapes_assumed': True},
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
                'Batch means active rows in one actual model invocation; equal P/G, no draining or padding waste.',
                'One shared projection-weight read per cached forward.',
                'KV attention reads include the newly appended token, plus one separate KV write.',
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
                             'full_attention_flops': estimate(4 * batch * dims['full_attention_layers'] * dims['query_heads'] * dims['attention_head_dim'] * prefix**2, 'shape_estimate'),
                             'projection_weight_bytes_once': estimate(dims['weight_bytes'] * dense, 'ideal_traffic_estimate'),
                             'one_layer_softmax_pair_peak_bytes': estimate(6 * batch * dims['query_heads'] * prefix**2, 'lifetime_shape_estimate')},
            'decode_step': {'projection_flops': estimate(2 * batch * dense, 'shape_estimate'),
                            'projection_weight_bytes_once': estimate(dims['weight_bytes'] * dense, 'ideal_traffic_estimate'),
                            'full_attention_flops_at_first_cached_token': estimate(4 * batch * dims['full_attention_layers'] * dims['query_heads'] * dims['attention_head_dim'] * (prefix+1), 'shape_estimate'),
                            'ideal_projection_flop_per_byte': estimate(2 * batch / dims['weight_bytes'], 'ideal_intensity_estimate')}
        },
        'capacity_terms': {'recurrent_state_bytes_per_page': estimate(recurrent, 'storage_estimate'),
                           'kv_bytes_per_page_per_token': estimate(kv, 'storage_estimate'),
                           'recurrent_plus_final_kv_bytes': estimate(batch * (recurrent + kv * (prefix+completion)), 'storage_estimate'),
                           'recurrent_plus_allowance_kv_bytes': estimate(batch * (recurrent + kv * (prefix+dims['output_allowance'])), 'storage_estimate'),
                           'total_peak_process_bytes': None, 'admissible_batch_size': None},
        'sustained_completed_pages_per_second': None,
        'limits': [model['assumptions']['scope'],
                   'Ideal traffic is not measured DRAM traffic; dense FLOPs omit recurrence and other work.',
                   'Softmax pair assumes BF16-score/FP32-softmax lifetime variant, excludes mask/workspaces.',
                   'Stored KV excludes expanded heads and concatenation copies. Padding and finished rows can waste work.',
                   'Roofline time max(F/C,Q/BW) needs compatible empirical roofs and actual traffic. Null fields prohibit a numeric hardware ceiling.',
                   'This script cannot validate a supplied hardware calibration or authorize a batch.']
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path(__file__).with_name('roofline-inputs.json'))
    for name in ('batch', 'prefix', 'patches', 'completion'):
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--vision-segments', help='Comma-separated positive segment lengths per page')
    args = parser.parse_args()
    raw = args.inputs.read_bytes()
    model = json.loads(raw)
    values = {name: getattr(args, name) if getattr(args, name) is not None else model['defaults'][name]
              for name in ('batch', 'prefix', 'patches', 'completion')}
    try:
        segments = None if args.vision_segments is None else [int(n) for n in args.vision_segments.split(',')]
        result = build(model, **values, vision_segments=segments)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result['model_assumptions_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
