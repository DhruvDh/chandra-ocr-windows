"""Stdlib fake-model scope tests; imports no Torch or production runtime."""
import importlib.util
from pathlib import Path
import types
import tempfile
import json
import unittest

SPEC = importlib.util.spec_from_file_location('production_gain_adapter', Path(__file__).with_name('production_gain_adapter.py'))
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class Norm:
    def forward(self, x):
        return x + 1


class ScopeTests(unittest.TestCase):
    def exercise(self, production, fail=False, body_bytes=0):
        modules = [(str(i), Norm()) for i in range(81)]
        # Also test an existing direct override, not only class-bound methods.
        override = types.MethodType(lambda self, x: x + 1, modules[0][1])
        modules[0][1].forward = override
        memory = {'allocated_bytes': 1000}
        def snapshot(): return dict(memory)
        def installer(model):
            prior = [(m, 'forward' in m.__dict__, m.__dict__.get('forward')) for _, m in modules]
            for _, m in modules:
                previous = m.forward
                m.forward = types.MethodType(lambda self, x, _previous=previous: _previous(x), m)
            memory['allocated_bytes'] += adapter.EXPECTED_GAIN_BYTES
            class Owner:
                module_count = 81
                gain_bytes = adapter.EXPECTED_GAIN_BYTES
                def close(self):
                    for m, had, previous in prior:
                        if had: m.forward = previous
                        else: del m.forward
                    memory['allocated_bytes'] -= self.gain_bytes
            return Owner()
        receipt = None
        try:
            with adapter._scope(modules, installer, None, lambda: None, snapshot, production=production) as receipt:
                for _, module in modules:
                    self.assertEqual(module.forward(7), 8)
                memory['allocated_bytes'] += body_bytes
                if fail: raise ValueError('intentional fake-body failure')
        except ValueError:
            self.assertTrue(fail)
        self.assertTrue(receipt['restored'])
        self.assertTrue(receipt['close_releases_exact_gain_bytes'])
        self.assertEqual(receipt['install_allocated_delta_bytes'], adapter.EXPECTED_GAIN_BYTES if production else 0)
        self.assertEqual(set(receipt['calls'].values()), {1})
        self.assertIs(modules[0][1].__dict__['forward'], override)
        self.assertTrue(all('forward' not in m.__dict__ for _, m in modules[1:]))
        self.assertEqual(memory['allocated_bytes'], 1000 + body_bytes)

    def test_original(self): self.exercise(False)
    def test_production(self): self.exercise(True)
    def test_body_failure_restores_original(self): self.exercise(False, True)
    def test_body_failure_restores_production(self): self.exercise(True, True)
    def test_body_allocation_survives_gain_close(self): self.exercise(True, body_bytes=512)
    def test_original_body_allocation_survives_close(self): self.exercise(False, body_bytes=512)
    def test_post_warmup_drift_rejected(self):
        receipt = {'memory_before_install': {'allocated_bytes': 1512}, 'memory_after_close': {'allocated_bytes': 1512}}
        self.assertTrue(adapter.steady_allocation_gate(receipt, 1512)['post_close_equal'])
        receipt['memory_after_close']['allocated_bytes'] += 512
        with self.assertRaises(RuntimeError): adapter.steady_allocation_gate(receipt, 1512)
        receipt['memory_after_close']['allocated_bytes'] = 1512
        receipt['memory_before_install']['allocated_bytes'] += 512
        with self.assertRaises(RuntimeError): adapter.steady_allocation_gate(receipt, 1512)
    def test_reference_ids_unavailable_are_not_acceptance(self):
        result = adapter.token_reference_comparison([1, 2], 2, [2], None)
        self.assertFalse(result['available'])
        self.assertIsNone(result['exact_equal'])
    def test_reference_exact_and_mismatch(self):
        reference = {'token_ids': [1, 2], 'terminal_id': 2, 'stop_ids': [2]}
        self.assertTrue(adapter.token_reference_comparison([1, 2], 2, [2], reference)['exact_equal'])
        self.assertFalse(adapter.token_reference_comparison([3, 2], 2, [2], reference)['exact_equal'])
        self.assertFalse(adapter.token_reference_comparison([1, 2], 2, [2, 4], reference)['exact_equal'])
    def test_failed_token_evidence_persisted_before_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failed.json'
            reference = {'token_ids': [1, 2], 'terminal_id': 2, 'stop_ids': [2]}
            with self.assertRaises(RuntimeError):
                evidence = adapter.persist_token_evidence(path, [3, 2], 2, [2], 'decoded', 'streamed', reference)
                if any(evidence['failure_flags'].values()): raise RuntimeError('reject after persistence')
            retained = json.loads(path.read_text())
            self.assertEqual(retained['token_ids'], [3, 2])
            self.assertEqual(retained['decoded_content'], 'decoded')
            self.assertFalse(retained['complete_decode_equals_stream'])
            self.assertFalse(retained['retained_token_comparison']['exact_equal'])
            self.assertTrue(all(retained['failure_flags'].values()))
    def test_wrong_inventory_rejected(self):
        with self.assertRaises(RuntimeError):
            with adapter._scope([], None, None, lambda: None, lambda: {'allocated_bytes': 0}, production=False): pass


if __name__ == '__main__': unittest.main()
