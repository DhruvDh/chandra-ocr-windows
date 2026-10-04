"""Small stdlib checks only; never hashes a real trajectory or imports Torch."""
import json
import hashlib
import copy
import stat
import types
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
from verification import compare_gain_exact as reader


class ExactTests(unittest.TestCase):
    def state(self, position=800):
        return {key: {'shape': shape, 'dtype': dtype, 'finite': True, 'sha256': 'a' * 64}
                for key, (shape, dtype) in reader.cache_specs(position).items()}

    def test_inventory_and_position_shape(self):
        specs = reader.cache_specs(1194)
        self.assertEqual(len(specs), 64)
        self.assertEqual(specs['31.keys.0'], ([1, 4, 1195, 256], 'torch.bfloat16'))
        self.assertEqual(len(reader.norm_names()), 81)
        failures = []
        reader.validate_states([self.state()], [800], failures, 'test')
        self.assertEqual(failures, [])

    def test_missing_reordered_extra_keys_fail(self):
        for change in ('missing', 'reordered', 'extra'):
            state = self.state()
            if change == 'missing': state.pop('0.conv_states.0')
            if change == 'reordered': state = dict(reversed(list(state.items())))
            if change == 'extra': state['extra'] = {}
            failures = []
            reader.validate_states([state], [800], failures, 'test')
            self.assertTrue(failures)

    def test_shape_dtype_finite_digest_fail(self):
        for field, value in [('shape', [1, 8192, 5]), ('dtype', 'torch.float32'), ('finite', 1), ('sha256', 'bad')]:
            state = self.state(); state['0.conv_states.0'][field] = value
            failures = []
            reader.validate_states([state], [800], failures, 'test')
            self.assertTrue(failures)

    def test_duplicate_commitment_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.json'
            for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError): reader.read_json(path)
            path.write_text('{}')
            with self.assertRaises(ValueError): reader.read_json(path, '0' * 64)

    def test_exact_preserves_types(self):
        self.assertFalse(reader.exact({'finite': 1}, {'finite': True}))
        self.assertFalse(reader.exact([800.0], [800]))
        self.assertTrue(reader.exact({'b': 2, 'a': 1}, {'a': 1, 'b': 2}))

    def integration(self, mutation=None):
        source = {key: 'b' * 64 for key in ('norm_export.py', 'norm_candidate.py', 'runtime/waystone/norm_gain.py', 'experiments/xpu-hillclimb/production_gain_adapter.py')}
        adapter = {key: source[key] for key in ('runtime/waystone/norm_gain.py', 'experiments/xpu-hillclimb/production_gain_adapter.py')}
        manifest = {'schema': 'chandra.production.gain.exact-manifest.v1', 'norm_source_sha256': source, 'adapter_source_sha256': adapter}
        baseline = {'schema_version': 1, 'context': {'positions': list(range(800, 1195)), 'prefix_token_ids': [1] * 801, 'target_token_ids': [2] * 394},
                    'inputs': {}, 'runtime': {}, 'response_sha256': 'c' * 64,
                    'cache_equivalence': {'finite': True, 'argmax_equal': True},
                    'states': [self.state(position) for position in range(800, 1195)],
                    'logits': {'shape': [395, 248320], 'dtype': 'float32', 'byte_order': 'little', 'file': 'logits.f32', 'sha256': 'a' * 64}}
        candidate = copy.deepcopy(baseline)
        receipt = {'source_sha256': adapter, 'identity': 'production-gain-v1', 'mode': 'production-gain', 'module_count': 81,
                   'gain_bytes': 681984, 'restored': True, 'calls': {key: 396 for key in reader.norm_names()},
                   'install_allocated_delta_bytes': 681984, 'close_released_allocated_bytes': 681984, 'close_releases_exact_gain_bytes': True,
                   'memory_gate': {'expected_gain_bytes': 681984, 'allocator_rounding_allowance_bytes': 0}}
        record = {'mode': 'production-gain', 'candidate_identity': 'production-gain-v1', 'helper_revision': 2,
                  'source_sha256': source.copy(), 'receipts': [receipt]}
        if mutation: mutation(candidate, record)
        def read(path, commitment=None):
            if Path(path).name == 'norm-experiment.json': return record
            if Path(path).parent.name == 'baseline':
                self.assertEqual(commitment, reader.BASELINE_SHA256)
                return baseline
            return candidate
        with patch.object(reader, 'read_json', side_effect=read), patch.object(reader, 'digest_file', return_value='a' * 64) as hashing, patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'stat', return_value=types.SimpleNamespace(st_size=395 * 248320 * 4, st_mode=stat.S_IFREG)):
            result = reader.verify('baseline', 'candidate', manifest)
            hashing.assert_called_once_with(Path('candidate/logits.f32'), 395 * 248320 * 4)
            return result

    def test_integration_success(self):
        result = self.integration()
        self.assertTrue(result['passed'], result['failures'])
        self.assertEqual(result['observed_cache_records'], 25280)

    def test_integration_source_counts_restoration_rows(self):
        mutations = [lambda meta, record: record['source_sha256'].update({'norm_export.py': 'd' * 64}),
                     lambda meta, record: record['receipts'][0]['calls'].update({reader.norm_names()[0]: 395}),
                     lambda meta, record: record['receipts'][0].update({'restored': False}),
                     lambda meta, record: meta['states'].pop()]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result = self.integration(mutation)
                self.assertFalse(result['passed'])
                self.assertTrue(result['failures'])
        self.assertEqual(result['observed_cache_positions'], 394)
        self.assertEqual(result['expected_cache_records'], 25280)

    def test_actual_altered_truncated_and_growing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'payload'; path.write_bytes(b'abc')
            expected = reader.digest_file(path, 3)
            path.write_bytes(b'abd')
            self.assertNotEqual(reader.digest_file(path, 3), expected)
            for content in (b'ab', b'abcd'):
                path.write_bytes(content)
                with self.assertRaises(ValueError): reader.digest_file(path, 3)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'FIFO creation unavailable on this platform')
    def test_fifo_json_and_payload_fail_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fifo'; os.mkfifo(path)
            with self.assertRaises(ValueError): reader.read_json(path)
            with self.assertRaises(ValueError): reader.digest_file(path, 3)

    def test_cli_retains_missing_evidence_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = root / 'manifest.json'; output = root / 'failure.json'
            raw = b'{}'; manifest.write_bytes(raw)
            argv = ['compare', '--baseline', str(root / 'missing'), '--candidate', str(root / 'candidate'),
                    '--manifest', str(manifest), '--manifest-sha256', hashlib.sha256(raw).hexdigest(), '--output', str(output)]
            with patch('sys.argv', argv): self.assertEqual(reader.main(), 1)
            self.assertFalse(json.loads(output.read_text())['passed'])
            self.assertTrue(json.loads(output.read_text())['failures'])

    def test_streaming_digest_small_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bytes'; path.write_bytes(b'abc')
            self.assertEqual(reader.digest_file(path), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__ == '__main__': unittest.main()
