"""Source-derived native accounting checks; stdlib only, no device or speed claims."""
import json
from pathlib import Path
import unittest

from benchmarks.roofline import build

MODEL = json.loads(Path(__file__).with_name('roofline-inputs.json').read_text())


class NativeRooflineTests(unittest.TestCase):
    def scenario(self, **overrides):
        args = dict(batch=1, prefix=3617, patches=12096, completion=397)
        args.update(overrides)
        return build(MODEL, **args)

    def test_four_byte_native_cache_and_fixed_request_composition(self):
        result = self.scenario()
        capacity = result['capacity_terms']
        self.assertEqual(capacity['kv_bytes_per_page_per_token']['value'], 65536)
        native = capacity['native_request_resident_cache']
        # Independent arithmetic from the buffers in TextModel::newRequest.
        expected = dict(kv=8 * 2 * 16384 * 1024 * 4,
                        recurrent=24 * 32 * 128 * 128 * 4,
                        convolution_history=24 * 8192 * 4 * 4)
        self.assertEqual(native['bytes_per_request'], {**expected, 'total': 1127219200})
        self.assertEqual(sum(expected.values()), 1127219200)
        self.assertEqual(native['kv_elements_per_buffer'], 16777216)
        small = self.scenario(prefix=1, completion=1, queue_concurrency=15)
        self.assertEqual(small['capacity_terms']['native_request_resident_cache'], native)

    def test_original_prompt_and_chunk_boundaries(self):
        for prefix, count, full, last in [(1, 1, 0, 1), (63, 1, 0, 63), (64, 1, 1, 64),
                                         (65, 2, 1, 1), (128, 2, 2, 64), (3617, 57, 56, 33)]:
            with self.subTest(prefix=prefix):
                chunk = self.scenario(prefix=prefix)['phases']['text_prefill']['native_chunked_prefill_per_request']
                self.assertEqual((chunk['chunk_count'], chunk['full_chunks'], chunk['last_chunk_tokens']), (count, full, last))
                self.assertEqual(sum(min(64, prefix - start) for start in range(0, prefix, 64)), prefix)

    def test_whole_prefix_bound_and_repeated_source_sweeps_stay_distinct(self):
        prefill = self.scenario()['phases']['text_prefill']
        ideal = prefill['projection_weight_bytes_once']
        native = prefill['native_chunked_prefill_per_request']
        self.assertEqual(ideal, dict(value=8409579520, status='ideal_whole_prefix_single_weight_sweep'))
        self.assertEqual(native['text_projection_weight_bytes_per_chunk']['value'], 7138181120)
        self.assertEqual(native['vocabulary_weight_bytes_once']['value'], 1271398400)
        # 57 passes over the text projections; one final-row head, not 57 heads.
        sweep = native['projection_weight_sweep_bytes']
        self.assertEqual(sweep['value'], 408147722240)
        self.assertEqual(sweep['status'], 'conditional_one_weight_sweep_per_chunk_not_measured_dram')
        self.assertIsNone(native['dram_traffic_bytes'])
        self.assertIn('ideal_whole_prefix', prefill['one_layer_softmax_pair_peak_bytes']['status'])

    def test_queue_demand_does_not_amortize_weights_or_make_model_rows(self):
        one, queued = self.scenario(), self.scenario(queue_concurrency=15)
        for field in ('phases', 'capacity_terms', 'conditional_cached_decode_dram_scenario', 'native_engine'):
            self.assertEqual(queued[field], one[field])
        self.assertEqual(queued['scenario']['client_queue_concurrency'], 15)
        self.assertEqual(queued['native_engine']['model_rows_per_invocation'], 1)
        self.assertFalse(queued['native_engine']['queue_concurrency_changes_model_batch_or_weight_reuse'])
        hypothetical = self.scenario(batch=15)
        self.assertFalse(hypothetical['scenario']['matches_current_native_model_batch'])
        self.assertEqual(hypothetical['scenario']['batch_semantics'], 'hypothetical_model_rows_in_one_invocation')
        self.assertEqual(hypothetical['native_engine']['execution'], 'serial_B1')
        self.assertEqual(hypothetical['phases']['text_prefill']['native_chunked_prefill_per_request'], one['phases']['text_prefill']['native_chunked_prefill_per_request'])

    def test_consumed_cache_length_excludes_last_predicted_token(self):
        result = self.scenario()
        last_context = result['conditional_cached_decode_dram_scenario']['last_attention_context_tokens']
        self.assertEqual(last_context, 4013)
        self.assertEqual(result['capacity_terms']['recurrent_plus_final_kv_bytes']['value'], 313327616)
        self.assertEqual(result['capacity_terms']['recurrent_plus_allowance_kv_bytes']['value'], 1098907648)

    def test_measurements_and_hardware_ceiling_remain_unknown(self):
        result = self.scenario()
        self.assertTrue(all(value is None for value in result['empirical_roofs'].values()))
        self.assertTrue(all(value is None for value in result['measured_timing'].values()))
        ceiling = result['analytical_hardware_ceiling']
        self.assertFalse(ceiling['xmx_peak_assumed'])
        self.assertTrue(all(value is None for key, value in ceiling.items() if key != 'xmx_peak_assumed'))
        self.assertIsNone(result['sustained_completed_pages_per_second'])
        self.assertIsNone(result['capacity_terms']['admissible_batch_size'])
        for value in (0, -1, True, 1.5, '2', 10**13):
            with self.subTest(queue_concurrency=value), self.assertRaises(ValueError):
                self.scenario(queue_concurrency=value)


if __name__ == '__main__':
    unittest.main()
