"""Small real CPU algebra and metadata-only XPU contract/lifecycle checks."""
import hashlib
import json
from pathlib import Path
import sys
import transformers
import threading
from types import SimpleNamespace
import unittest

import vision_query_chunk as candidate

import torch
if torch.__version__ != "2.14.1+cpu":
    raise RuntimeError("Qualified CPU-only Torch 2.14.1 runtime required")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.utils.generic import GeneralInterface

GRAPH_SOURCE = Path(transformers.__file__).resolve().parent / "models/qwen3_5/modeling_qwen3_5.py"
MEASUREMENTS = []
EXACT_OUTPUTS = []


class CPUFences:
    def __init__(self):
        self.devices = []
        self.fail_at = None
    def synchronize(self, device):
        assert device.type == "cpu"
        self.devices.append(device)
        if self.fail_at == len(self.devices):
            raise RuntimeError("synthetic fence failure")


class CPUProxy:
    def __init__(self):
        self.xpu = CPUFences()
    def __getattr__(self, name):
        return getattr(torch, name)


def cpu_inputs(dtype, length=65, heads=16, dim=64):
    generator = torch.Generator(device="cpu").manual_seed(4731)
    # QKV projection-like gaps: final dimension contiguous, other strides differ
    # from a contiguous [B,H,N,D] tensor. These are synthetic, not model inputs.
    packed = torch.randn((1, length, 3, heads, dim), generator=generator,
                         device="cpu", dtype=torch.float32).to(dtype)
    return tuple(packed[:, :, i].permute(0, 2, 1, 3) for i in range(3))


def call_kwargs(scale=0.125):
    return dict(attention_mask=None, dropout=0.0, scaling=scale, is_causal=False,
                position_bias=None, output_attentions=False)


MODULE = SimpleNamespace(is_causal=False, num_key_value_groups=1)


class AlgebraChecks(unittest.TestCase):
    def test_chunked_stock_math_and_independent_double_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for chunk in (1, 7, 32):
                with self.subTest(dtype=dtype, chunk=chunk), torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
                    query, key, value = cpu_inputs(dtype)
                    frozen = [t.clone() for t in (query, key, value)]
                    kwargs = call_kwargs(0.37)
                    proxy = CPUProxy()
                    observed = []
                    def prior(module, part, k, v, **passed):
                        self.assertIs(module, MODULE)
                        self.assertIs(k, key)
                        self.assertIs(v, value)
                        self.assertEqual(passed, kwargs)
                        self.assertEqual(part.stride(), query.stride())
                        self.assertEqual(part.dtype, query.dtype)
                        self.assertEqual(part.device, query.device)
                        self.assertEqual(part.untyped_storage().data_ptr(), query.untyped_storage().data_ptr())
                        observed.append((part.shape[-2], part.storage_offset()))
                        return sdpa_attention_forward(module, part, k, v, **passed)
                    (actual, weights), calls = candidate._chunk_sdpa(proxy, prior, MODULE,
                        query, key, value, chunk, kwargs)
                    expected, _ = sdpa_attention_forward(MODULE, query, key, value, **kwargs)
                    reference = ((query.double() @ key.double().transpose(-2, -1)) * kwargs["scaling"])
                    reference = (reference.softmax(-1) @ value.double()).transpose(1, 2).contiguous()
                    maximum = float((actual.double() - expected.double()).abs().max())
                    reference_error = float((actual.double() - reference).abs().max())
                    baseline_error = float((expected.double() - reference).abs().max())
                    baseline_rmse = float(((expected.double() - reference).square().mean()).sqrt())
                    chunk_rmse = float(((actual.double() - reference).square().mean()).sqrt())
                    exact = torch.equal(actual, expected)
                    EXACT_OUTPUTS.append(exact)
                    torch.testing.assert_close(actual.double(), reference,
                        rtol=2e-5 if dtype == torch.float32 else 0.012,
                        atol=2e-6 if dtype == torch.float32 else 0.008)
                    torch.testing.assert_close(expected.double(), reference,
                        rtol=2e-5 if dtype == torch.float32 else 0.012,
                        atol=2e-6 if dtype == torch.float32 else 0.008)
                    self.assertIsNone(weights)
                    self.assertEqual(tuple(actual.shape), (1, 65, 16, 64))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(calls, (65 + chunk - 1) // chunk)
                    self.assertEqual(len(proxy.xpu.devices), calls)
                    self.assertEqual([x[0] for x in observed], [min(chunk, 65 - s) for s in range(0, 65, chunk)])
                    self.assertEqual([x[1] for x in observed], [query.storage_offset() + s * query.stride(-2) for s in range(0, 65, chunk)])
                    self.assertTrue(all(torch.equal(t, f) for t, f in zip((query, key, value), frozen)))
                    MEASUREMENTS.append({"dtype": str(dtype), "N": 65, "H": 16, "D": 64,
                        "chunk": chunk, "calls": calls, "stock_math_exact": exact,
                        "different_elements": int(torch.count_nonzero(actual != expected)),
                        "max_abs_stock_math_error": maximum, "max_abs_independent_float64_error": reference_error,
                        "stock_max_abs_independent_float64_error": baseline_error,
                        "stock_rmse_independent_float64": baseline_rmse,
                        "chunk_rmse_independent_float64": chunk_rmse})

    def test_query_row_permutation_equivariance(self):
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            query, key, value = cpu_inputs(torch.float32, length=37, heads=2, dim=8)
            order = torch.arange(36, -1, -1, device="cpu")
            baseline, _ = sdpa_attention_forward(MODULE, query, key, value, **call_kwargs())
            (permuted, _), _ = candidate._chunk_sdpa(CPUProxy(), sdpa_attention_forward, MODULE,
                query[:, :, order], key, value, 7, call_kwargs())
            reference = ((query[:, :, order].double() @ key.double().transpose(-2, -1)) * 0.125)
            reference = (reference.softmax(-1) @ value.double()).transpose(1, 2).contiguous()
            torch.testing.assert_close(permuted.double(), reference, rtol=2e-5, atol=2e-6)
            # Preserve the bitwise finding independently of the algebra check.
            EXACT_OUTPUTS.append(torch.equal(permuted, baseline[:, order]))

    def test_none_scale_retains_stock_default(self):
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            query, key, value = cpu_inputs(torch.float32, length=9, heads=2, dim=8)
            (actual, _), _ = candidate._chunk_sdpa(CPUProxy(), sdpa_attention_forward, MODULE,
                query, key, value, 7, call_kwargs(None))
            expected, _ = sdpa_attention_forward(MODULE, query, key, value, **call_kwargs(None))
            reference = (query.double() @ key.double().transpose(-2, -1)) / (query.shape[-1] ** 0.5)
            reference = (reference.softmax(-1) @ value.double()).transpose(1, 2).contiguous()
            torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=2e-6)
            EXACT_OUTPUTS.append(torch.equal(actual, expected))

    def test_output_contract_and_fence_failure_no_result(self):
        query, key, value = cpu_inputs(torch.float32, length=9, heads=2, dim=8)
        bad_outputs = [torch.empty((1, 7, 2, 7), device="cpu"),
                       torch.empty((1, 7, 2, 8), dtype=torch.bfloat16, device="cpu")]
        for output in bad_outputs:
            proxy = CPUProxy()
            with self.subTest(shape=tuple(output.shape), dtype=output.dtype), self.assertRaisesRegex(RuntimeError, "output contract"):
                candidate._chunk_sdpa(proxy, lambda *a, **k: (output, None), MODULE,
                                       query, key, value, 7, call_kwargs())
            self.assertEqual(proxy.xpu.devices, [])
        proxy = CPUProxy()
        proxy.xpu.fail_at = 2
        completed = []
        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]), self.assertRaisesRegex(RuntimeError, "fence failure"):
            candidate._chunk_sdpa(proxy, sdpa_attention_forward, MODULE,
                query, key, value, 7, call_kwargs(), completed.append)
        self.assertEqual(completed, [7])
        self.assertEqual(len(proxy.xpu.devices), 2)


class Device:
    def __init__(self, name="xpu:0"):
        self.name = name
        self.type = name.split(":")[0]
    def __str__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name


class MetaTensor:
    def __init__(self, shape=(1, 16, 65, 64), dtype="bf16", device=None, strides=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or Device()
        self.strides = strides or (16 * 65 * 64, 64, 16 * 64, 1)
        self.writes = []
    def stride(self, index=None):
        return self.strides if index is None else self.strides[index]
    def __getitem__(self, key):
        start, stop = key[-2].start, key[-2].stop
        return MetaTensor(self.shape[:-2] + (stop - start, self.shape[-1]), self.dtype,
                          self.device, self.strides)
    def __setitem__(self, key, value):
        self.writes.append((key[1].start, key[1].stop, value))


class Vision:
    def __init__(self):
        self.training = False
        self.is_causal = False
        self.num_key_value_groups = 1
        self.config = SimpleNamespace(_attn_implementation="sdpa", output_attentions=False)


def record():
    return dict(active=True, failed=False, in_call=False, owner_thread=None,
        attempted_vision=0, intercepted_vision=0, stock_sdpa_calls=0,
        completed_chunk_fences=0, forwarded_nonvision=0, maximum_query_rows=0, geometries=[])


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.calls, self.fences, self.outputs = [], [], []
        self.grad = False
        def empty(shape, dtype, device):
            output = MetaTensor(shape, dtype, device)
            self.outputs.append(output)
            return output
        self.graph = SimpleNamespace(Qwen3_5VisionAttention=Vision,
            torch=SimpleNamespace(bfloat16="bf16", is_grad_enabled=lambda: self.grad,
                empty=empty, xpu=SimpleNamespace(synchronize=self.fences.append)))
        self.module = Vision()
        self.query, self.key, self.value = MetaTensor(), MetaTensor(), MetaTensor()
        self.record = record()
        def prior(module, query, key, value, **kwargs):
            self.calls.append((module, query, key, value, kwargs))
            return MetaTensor((1, query.shape[-2], 16, 64), query.dtype, query.device), None
        self.prior = prior
        self.dispatch = candidate._dispatch(self.graph, prior, 32, self.record)
    def invoke(self, **changes):
        kwargs = call_kwargs()
        kwargs.update(changes)
        return self.dispatch(self.module, self.query, self.key, self.value, **kwargs)
    def test_valid_dispatch_order_flags_and_fences(self):
        output, weights = self.invoke()
        self.assertIsNone(weights)
        self.assertEqual([r[1].shape[-2] for r in self.calls], [32, 32, 1])
        self.assertTrue(all(r[2] is self.key and r[3] is self.value for r in self.calls))
        self.assertTrue(all(r[4] == call_kwargs() for r in self.calls))
        self.assertEqual([(r[0], r[1]) for r in output.writes], [(0, 32), (32, 64), (64, 65)])
        self.assertEqual(self.fences, [self.query.device] * 3)
        self.assertEqual(self.record["stock_sdpa_calls"], 3)
        self.assertEqual(self.record["completed_chunk_fences"], 3)
        self.assertEqual(self.record["geometries"], [[65, 65, 3]])
    def test_semantics_and_geometry_guards_before_allocation(self):
        mutations = [
            lambda: setattr(self.module, "training", True),
            lambda: setattr(self.module, "is_causal", True),
            lambda: setattr(self.module, "num_key_value_groups", 2),
            lambda: setattr(self.module.config, "_attn_implementation", "eager"),
            lambda: setattr(self.module.config, "output_attentions", True),
            lambda: setattr(self, "grad", True),
            lambda: setattr(self.query, "dtype", "fp32"),
            lambda: setattr(self.query, "device", Device("cpu")),
            lambda: setattr(self.query, "device", Device("xpu:1")),
            lambda: setattr(self.value, "device", Device("xpu:1")),
            lambda: setattr(self.query, "strides", (66560, 64, 1024, 2)),
            lambda: setattr(self.query, "shape", (1, 16, 64, 64)),
            lambda: setattr(self.query, "shape", (2, 16, 65, 64)),
            lambda: setattr(self.key, "shape", (1, 16, 13465, 64)),
            lambda: setattr(self.value, "shape", (1, 16, 65, 63)),
            lambda: setattr(self.query, "ndim", 3),
            lambda: self.record.update(in_call=True),
            lambda: self.record.update(attempted_vision=24),
            lambda: self.record.update(owner_thread=threading.get_ident() + 1),
        ]
        for index, mutate in enumerate(mutations):
            self.setUp()
            mutate()
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, "Unqualified"):
                self.invoke()
            self.assertEqual(self.calls, [])
            self.assertEqual(self.outputs, [])
            self.assertEqual(self.fences, [])
        for changes in (dict(attention_mask=object()), dict(dropout=0.1), dict(is_causal=True),
                        dict(is_causal=None), dict(position_bias=object()), dict(output_attentions=True)):
            self.setUp()
            with self.subTest(changes=str(changes)), self.assertRaisesRegex(RuntimeError, "Unqualified"):
                self.invoke(**changes)
            self.assertEqual(self.outputs, [])
    def test_closed_or_failed_scope_refuses_stale_dispatch(self):
        for changes in (dict(active=False), dict(failed=True)):
            self.record.update(changes)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                self.invoke()
            self.assertEqual(self.calls, [])
            self.record.update(active=True, failed=False)
    def test_nonvision_forwarded_unchanged(self):
        other = object()
        result = self.dispatch(other, self.query, self.key, self.value, **call_kwargs())
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.calls[0][0], other)
        self.assertIs(self.calls[0][1], self.query)
        self.assertEqual(self.calls[0][4], call_kwargs())
        self.assertEqual(self.outputs, [])
        self.assertEqual(self.fences, [])
        self.assertEqual(self.record["forwarded_nonvision"], 1)
        self.assertIsNone(result[1])
    def test_fence_failure_marks_scope_failed_and_refuses_reuse(self):
        def fail(device):
            raise RuntimeError("metadata fake device loss")
        self.graph.torch.xpu.synchronize = fail
        with self.assertRaisesRegex(RuntimeError, "device loss"):
            self.invoke()
        self.assertTrue(self.record["failed"])
        self.assertFalse(self.record["in_call"])
        self.assertEqual(self.record["completed_chunk_fences"], 0)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.invoke()
        self.assertEqual(len(self.calls), 1)


class Interface(GeneralInterface):
    _global_mapping = {"sdpa": sdpa_attention_forward}


class ScopeChecks(unittest.TestCase):
    def setUp(self):
        self.original = Interface()
        self.local = object()
        self.original["private-existing"] = self.local
        self.global_before = dict(Interface._global_mapping)
        self.graph = SimpleNamespace(__file__=str(GRAPH_SOURCE), ALL_ATTENTION_FUNCTIONS=self.original,
                                     Qwen3_5VisionAttention=Vision, torch=CPUProxy())
    def scope(self, **changes):
        kwargs = dict(enabled=True, attention_collection_excluded=True, chunk=32)
        kwargs.update(changes)
        return candidate.scoped_candidate(self.graph, **kwargs)
    def test_registry_restoration_and_stale_capture(self):
        with self.scope() as receipt:
            replacement = self.graph.ALL_ATTENTION_FUNCTIONS
            captured = replacement["sdpa"]
            self.assertIs(replacement["private-existing"], self.local)
            self.assertIsNot(replacement, self.original)
            self.assertEqual(Interface._global_mapping, self.global_before)
        self.assertIs(self.graph.ALL_ATTENTION_FUNCTIONS, self.original)
        self.assertTrue(receipt["restored"])
        self.assertFalse(receipt["active"])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            captured(object(), None, None, None, attention_mask=None)
    def test_body_exception_restores(self):
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with self.scope() as receipt:
                raise RuntimeError("body failed")
        self.assertTrue(receipt["restored"])
        self.assertIs(self.graph.ALL_ATTENTION_FUNCTIONS, self.original)
    def test_nested_scope_rejected_without_disturbing_owner(self):
        with self.scope() as receipt:
            owner = self.graph.ALL_ATTENTION_FUNCTIONS
            with self.assertRaisesRegex(RuntimeError, "Unwrapped"):
                with self.scope():
                    self.fail("nested scope entered")
            self.assertIs(self.graph.ALL_ATTENTION_FUNCTIONS, owner)
        self.assertTrue(receipt["restored"])
    def test_foreign_owner_preserved_and_reported(self):
        foreign = Interface()
        with self.assertRaisesRegex(RuntimeError, "ownership lost"):
            with self.scope() as receipt:
                self.graph.ALL_ATTENTION_FUNCTIONS = foreign
        self.assertIs(self.graph.ALL_ATTENTION_FUNCTIONS, foreign)
        self.assertTrue(receipt["ownership_lost"])
        self.assertFalse(receipt["restored"])
        self.assertFalse(receipt["active"])
    def test_disabled_bounds_and_observer_exclusion(self):
        for changes in (dict(enabled=False), dict(attention_collection_excluded=False),
                        dict(chunk=0), dict(chunk=33), dict(chunk=True), dict(chunk=3.0)):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                with self.scope(**changes):
                    self.fail("invalid scope entered")
            self.assertIs(self.graph.ALL_ATTENTION_FUNCTIONS, self.original)
    def test_graph_or_stock_source_change_refused(self):
        self.graph.__file__ = __file__
        with self.assertRaisesRegex(RuntimeError, "Pinned Qwen"):
            with self.scope():
                self.fail("source mismatch entered")
        self.graph.__file__ = str(GRAPH_SOURCE)
        self.original["sdpa"] = lambda *a, **k: None
        with self.assertRaisesRegex(RuntimeError, "Unwrapped"):
            with self.scope():
                self.fail("wrapped stock entered")
        self.assertIs(self.graph.ALL_ATTENTION_FUNCTIONS, self.original)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    receipt = {"scope": "Small synthetic real CPU tensors, fake XPU descriptors/fences and registry lifecycle; no GPU or model loading",
        "python": sys.version, "torch": torch.__version__, "torch_commit": torch.version.git_version,
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "passed": result.wasSuccessful(), "real_cpu_sdpa_backend": "MATH only",
        "cpu_algebra_and_contract_passed": result.wasSuccessful(),
        "stock_math_bitwise_gate_passed": bool(EXACT_OUTPUTS) and all(EXACT_OUTPUTS),
        "historical_exact_comparison": "See benchmarks/evidence/hillclimb-2026-10-04.json",
        "float64_reference_bounds_unchanged_from_first_suite": {"fp32_rtol": 2e-5, "fp32_atol": 2e-6, "bf16_rtol": 0.012, "bf16_atol": 0.008},
        "maximum_real_shape": [1, 16, 65, 64], "measurements": MEASUREMENTS,
        "GPU_qualified": False, "native_extent_real_tensor_tested": False,
        "candidate_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()}
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
