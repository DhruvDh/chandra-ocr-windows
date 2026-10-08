"""Offline dimensional checks; no device or performance assertions."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.roofline import build

MODEL = json.loads(Path(__file__).with_name('roofline-inputs.json').read_text())


def test_batch_shares_ideal_weights_and_scales_work_and_state():
    one = build(MODEL, 1, 801, 12096, 395)
    fifteen = build(MODEL, 15, 801, 12096, 395)
    for phase in ('vision', 'text_prefill', 'decode_step'):
        assert fifteen['phases'][phase]['projection_flops']['value'] == 15 * one['phases'][phase]['projection_flops']['value']
        assert fifteen['phases'][phase]['projection_weight_bytes_once'] == one['phases'][phase]['projection_weight_bytes_once']
    assert fifteen['capacity_terms']['recurrent_plus_final_kv_bytes']['value'] == 15 * one['capacity_terms']['recurrent_plus_final_kv_bytes']['value']
    assert fifteen['phases']['decode_step']['ideal_projection_flop_per_byte']['value'] == 15
    assert all(value is None for value in fifteen['empirical_roofs'].values())
    assert fifteen['sustained_completed_pages_per_second'] is None
    assert fifteen['capacity_terms']['admissible_batch_size'] is None


def test_persistent_state_and_kv_dimensions():
    result = build(MODEL, 3, 801, 12096, 395)
    capacity = result['capacity_terms']
    assert capacity['recurrent_state_bytes_per_page']['value'] == 50331648
    assert capacity['kv_bytes_per_page_per_token']['value'] == 65536
    assert capacity['recurrent_plus_final_kv_bytes']['value'] == 3 * (50331648 + 65536 * 1195)
    assert result['phases']['decode_step']['projection_flops']['value'] == 3 * 8409579520


def test_vision_segments_square_each_page_segment_not_concatenated_batch():
    full = build(MODEL, 2, 801, 12096, 395)
    split = build(MODEL, 2, 801, 12096, 395, [6048, 6048])
    assert split['phases']['vision']['attention_flops']['value'] * 2 == full['phases']['vision']['attention_flops']['value']
    assert split['phases']['vision']['projection_flops'] == full['phases']['vision']['projection_flops']
    single = build(MODEL, 1, 801, 12096, 395)
    assert full['phases']['vision']['attention_flops']['value'] == 2 * single['phases']['vision']['attention_flops']['value']


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2', 10**13])
def test_invalid_scenario_dimensions(value):
    with pytest.raises(ValueError):
        build(MODEL, value, 801, 12096, 395)


@pytest.mark.parametrize('segments', [[], [0, 12096], [12095], [True, 12095], [-1, 12097]])
def test_invalid_segmentation(segments):
    with pytest.raises(ValueError):
        build(MODEL, 1, 801, 12096, 395, segments)


def test_invalid_model_dimensions_and_precision():
    for section, key, value in [('dimensions', 'kv_heads', 3), ('dimensions', 'weight_bytes', 4),
                                ('dimensions', 'linear_key_dim', 0), ('projection_elements', 'text_mlp', True)]:
        model = copy.deepcopy(MODEL)
        model[section][key] = value
        with pytest.raises(ValueError):
            build(model, 1, 801, 12096, 395)


@pytest.mark.parametrize('batch,prefix,completion', [(1, 1, 1), (2, 3, 2), (3, 5, 7)])
def test_cached_decode_closed_form_matches_explicit_forward_sum(batch, prefix, completion):
    model = copy.deepcopy(MODEL)
    # Small independent dimensions make the per-forward accounting inspectable.
    model['projection_elements'] = {name: i + 1 for i, name in enumerate(model['projection_elements'])}
    dims = model['dimensions']
    dims.update(full_attention_layers=2, query_heads=4, kv_heads=2, attention_head_dim=3,
                linear_layers=2, linear_value_heads=3, linear_key_dim=2, linear_value_dim=3)
    result = build(model, batch, prefix, 4, completion)
    scenario = result['conditional_cached_decode_dram_scenario']
    weights = 2 * sum(model['projection_elements'][name] for name in (
        'text_mlp', 'text_linear_attention', 'text_full_attention', 'vocabulary_head'))
    kv = 2 * 2 * 2 * 3 * 4
    recurrent = 2 * 3 * 2 * 3 * 4
    expected = dict(shared_projection_weight_reads=0, attention_kv_reads=0,
                    new_kv_writes=0, recurrent_matrix_reads_and_writes=0)
    for context in range(prefix + 1, prefix + completion):
        expected['shared_projection_weight_reads'] += weights
        expected['attention_kv_reads'] += batch * kv * context
        expected['new_kv_writes'] += batch * kv
        expected['recurrent_matrix_reads_and_writes'] += 2 * batch * recurrent
    assert scenario['bytes_per_batch'] == {**expected, 'subtotal': sum(expected.values())}
    assert scenario['bytes_per_completed_page'] == {**{k: v / batch for k, v in expected.items()},
                                                     'subtotal': sum(expected.values()) / batch}
    assert scenario['cached_forwards'] == completion - 1
    assert scenario['first_attention_context_tokens'] == (prefix + 1 if completion > 1 else None)
    assert scenario['last_attention_context_tokens'] == (prefix + completion - 1 if completion > 1 else None)
    assert scenario['other_traffic_bytes'] is None


def test_single_completion_token_has_no_cached_forward_but_prefill_work_remains():
    result = build(MODEL, 15, 3617, 12096, 1)
    assert all(value == 0 for value in result['conditional_cached_decode_dram_scenario']['bytes_per_batch'].values())
    assert result['phases']['text_prefill']['projection_flops']['value'] > 0
    assert result['phases']['vision']['projection_flops']['value'] > 0


def test_batch_amortizes_only_shared_decode_weights_per_page():
    one = build(MODEL, 1, 3617, 12096, 397)['conditional_cached_decode_dram_scenario']
    fifteen = build(MODEL, 15, 3617, 12096, 397)['conditional_cached_decode_dram_scenario']
    assert fifteen['bytes_per_batch']['shared_projection_weight_reads'] == one['bytes_per_batch']['shared_projection_weight_reads']
    assert fifteen['bytes_per_completed_page']['shared_projection_weight_reads'] * 15 == one['bytes_per_completed_page']['shared_projection_weight_reads']
    for term in ('attention_kv_reads', 'new_kv_writes', 'recurrent_matrix_reads_and_writes'):
        assert fifteen['bytes_per_completed_page'][term] == one['bytes_per_completed_page'][term]
    assert fifteen['bytes_per_completed_page']['subtotal'] == 360922349568


@pytest.mark.parametrize('field', ['batch', 'prefix', 'patches', 'completion'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5, '2', 10**13])
def test_decode_scenario_rejects_invalid_inputs(field, value):
    args = dict(batch=2, prefix=3, patches=4, completion=5)
    args[field] = value
    with pytest.raises(ValueError):
        build(MODEL, **args)
