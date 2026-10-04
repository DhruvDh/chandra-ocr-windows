"""Small CPU object-lifetime regression; no XPU allocator or model claim.

Run only in verification/cpu/.venv with an explicit owned timeout. The boundary
wrapper observes entry to post-normalization multiplication without retaining
its normalization input or changing arithmetic.
"""
import os
import importlib.metadata
if importlib.metadata.version("torch") != "2.14.1+cpu":
    raise RuntimeError("Lifetime test requires the matched CPU-only Torch runtime")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import types
import unittest
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.waystone.norm_gain import _install

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.125, -0.25, 0.5, -0.75], dtype=torch.bfloat16))
        self.eps = 1e-6

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * (1.0 + self.weight.float())).type_as(x)


def legacy_forward(self, x):
    # Exact lifetime-relevant sequence in the frozen experimental candidate.
    fp32 = x.float()
    normalized = self._norm(fp32)
    return (normalized * (1.0 + self.weight.float())).type_as(x)


class MultiplyBoundary:
    def __init__(self, output, converted_ref, observations):
        self.output = output
        self.converted_ref = converted_ref
        self.observations = observations

    def __mul__(self, gain):
        self.observations.append(self.converted_ref() is not None)
        return self.output * gain


def observe_conversion(norm):
    observations = []
    # Capture the original unbound implementation, never a converted tensor.
    implementation = Norm._norm
    def observed(self, x):
        converted_ref = weakref.ref(x)
        output = implementation(self, x)
        return MultiplyBoundary(output, converted_ref, observations)
    norm._norm = types.MethodType(observed, norm)
    return observations


class LifetimeTests(unittest.TestCase):
    def test_conversion_lifetime_at_gain_boundary(self):
        x = torch.linspace(-8, 8, 200).reshape(50, 4).to(torch.bfloat16)
        records = {}
        outputs = {}
        for mode in ('baseline', 'production', 'legacy'):
            norm = Norm().eval()
            observations = observe_conversion(norm)
            model = torch.nn.ModuleDict({'norm': norm}).eval()
            owner = None
            if mode == 'production':
                owner = _install(model, Norm, ('norm',), device_type='cpu')
            elif mode == 'legacy':
                norm.forward = types.MethodType(legacy_forward, norm)
            try:
                with torch.inference_mode():
                    outputs[mode] = norm(x)
            finally:
                if owner is not None:
                    owner.close()
            self.assertEqual(len(observations), 1)
            records[mode] = observations[0]
        self.assertEqual(records, {'baseline': False, 'production': False, 'legacy': True})
        self.assertTrue(torch.equal(outputs['baseline'], outputs['production']))
        self.assertTrue(torch.equal(outputs['baseline'], outputs['legacy']))
        print({'converted_input_alive_at_gain_multiply': records,
               'all_outputs_bit_equal': True,
               'scope': 'CPython CPU tensor object lifetime; not XPU allocation or peak memory'})

    def test_uninstrumented_exact_arithmetic(self):
        norm = Norm().eval()
        model = torch.nn.ModuleDict({'norm': norm}).eval()
        for x in (torch.zeros(2, 4, dtype=torch.bfloat16),
                  torch.tensor([[1, -2, 3, -4], [0.001, 128, -64, 0.5]], dtype=torch.bfloat16)):
            with torch.inference_mode():
                expected = norm(x)
                owner = _install(model, Norm, ('norm',), device_type='cpu')
                try:
                    actual = norm(x)
                finally:
                    owner.close()
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(actual.dtype, torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
