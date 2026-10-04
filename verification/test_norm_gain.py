"""CPU-only ownership/arithmetic checks; run in verification/cpu/.venv."""
import gc
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.waystone.norm_gain import _install, install_gain_cache


class Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.125, -0.25, 0.5, -0.75], dtype=torch.bfloat16))
        self.eps = 1e-6

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self._norm(x.float()) * (1.0 + self.weight.float())).type_as(x)


class Gated(Norm):
    pass


def model():
    return torch.nn.ModuleDict({"first": Norm(), "second": Norm(), "gated": Gated()}).eval()


def install(m):
    return _install(m, Norm, ("first", "second"), device_type="cpu")


class GainTests(unittest.TestCase):
    def test_exact_arithmetic_and_exclusion(self):
        m = model()
        x = torch.linspace(-8, 8, 200).reshape(50, 4).to(torch.bfloat16)
        with torch.inference_mode():
            expected = m["first"](x)
            with install(m) as owner:
                self.assertTrue(torch.equal(expected, m["first"](x)))
                self.assertEqual(owner.module_count, 2)
                self.assertEqual(owner.gain_bytes, 32)
                self.assertNotIn("forward", m["gated"].__dict__)
        self.assertNotIn("forward", m["first"].__dict__)
        owner.close()

    def test_mutation_replacement_and_training_rejected(self):
        for action in ("mutate", "replace", "train"):
            m = model()
            with install(m):
                with torch.no_grad():
                    if action == "mutate":
                        m["first"].weight.add_(1)
                    elif action == "replace":
                        m["first"].weight = torch.nn.Parameter(m["first"].weight.clone())
                    else:
                        m["first"].train()
                with torch.inference_mode(), self.assertRaises(RuntimeError):
                    m["first"](torch.ones(1, 4, dtype=torch.bfloat16))

    def test_preexisting_override_restored(self):
        m = model()
        original = types.MethodType(lambda self, x: x, m["first"])
        m["first"].forward = original
        owner = install(m)
        owner.close()
        self.assertIs(m["first"].forward, original)

    def test_partial_installation_restored(self):
        m = model()
        import runtime.waystone.norm_gain as module
        factory = module._forward
        calls = []
        def fail_second(*args):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("injected construction failure")
            return factory(*args)
        with patch.object(module, "_forward", fail_second), self.assertRaises(RuntimeError):
            install(m)
        self.assertTrue(all("forward" not in norm.__dict__ for norm in m.values()))

    def test_close_releases_model_and_gains(self):
        m = model()
        owner = install(m)
        model_ref = weakref.ref(m)
        module_ref = weakref.ref(m["first"])
        function = m["first"].forward.__func__
        captured = dict(zip(function.__code__.co_freevars, function.__closure__))
        gain_ref = weakref.ref(captured["gain"].cell_contents)
        del function, captured
        owner.close()
        del m
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(module_ref())
        self.assertIsNone(gain_ref())
        self.assertEqual(owner._entries, [])

    def test_inventory_and_weight_validation_are_transactional(self):
        m = model()
        with self.assertRaises(RuntimeError):
            _install(m, Norm, ("first",), device_type="cpu")
        m["second"].weight = torch.nn.Parameter(m["second"].weight.float())
        with self.assertRaises(RuntimeError):
            install(m)
        self.assertTrue(all("forward" not in norm.__dict__ for norm in m.values()))

    def test_invalid_execution_inputs_rejected(self):
        m = model()
        with install(m):
            with self.assertRaises(RuntimeError):
                m["first"](torch.ones(1, 4, dtype=torch.bfloat16))
            with torch.inference_mode():
                for x in (torch.ones(1, 4), torch.ones(1, 3, dtype=torch.bfloat16)):
                    with self.assertRaises(RuntimeError):
                        m["first"](x)

    def test_production_guard_rejects_cpu(self):
        with self.assertRaises(RuntimeError):
            install_gain_cache(model())


if __name__ == "__main__":
    unittest.main()
