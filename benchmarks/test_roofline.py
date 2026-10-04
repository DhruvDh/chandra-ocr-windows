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
    assert capacity['kv_bytes_per_page_per_token']['value'] == 32768
    assert capacity['recurrent_plus_final_kv_bytes']['value'] == 3 * (50331648 + 32768 * 1196)
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
