"""CPU-only numerical, source, lifetime and restoration checks."""
import ast
import importlib.util
from pathlib import Path
import types
import unittest
import weakref
from unittest.mock import patch
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lifetime", HERE / "softmax_lifetime_candidate.py")
lifetime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifetime)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        path = Path(os.environ["PINNED_GRAPH_SOURCE"])
        namespace = {"torch": torch, "nn": torch.nn}
        tree = ast.parse(path.read_bytes())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("repeat_kv", "eager_attention_forward")]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec", flags=16777216), namespace)
        class Text:
            training = False
            num_key_value_groups = 2
        cls.graph = types.SimpleNamespace(__file__=str(path), Qwen3_5Attention=Text, **namespace)

    def test_exact_cpu_and_lifetime(self):
        torch.manual_seed(814)
        original = self.graph.eager_attention_forward
        candidate = lifetime.build_candidate(self.graph)
        module = self.graph.Qwen3_5Attention()
        for length in (1, 17):
            q = torch.randn(1, 4, length, 8).to(torch.bfloat16)
            k = torch.randn(1, 2, 17, 8).to(torch.bfloat16)
            v = torch.randn_like(k)
            for mask in (None, torch.zeros(1, 1, length, 17)):
                expected = original(module, q, k, v, mask, 0.125)
                saved = torch.nn.functional.softmax
                references = []
                def observe(scores, **kwargs):
                    references.append(weakref.ref(scores))
                    return saved(scores, **kwargs)
                actual = candidate(module, q, k, v, mask, 0.125)
                for a, b in zip(expected, actual):
                    self.assertTrue(torch.equal(a, b))
                # Intercept cast to confirm old score dies before BF16 conversion.
                old_to = torch.Tensor.to
                def cast(tensor, *args, **kwargs):
                    if references and tensor.dtype == torch.float32:
                        self.assertIsNone(references[-1]())
                    return old_to(tensor, *args, **kwargs)
                with patch.object(torch.nn.functional, "softmax", observe), patch.object(torch.Tensor, "to", cast):
                    candidate(module, q, k, v, mask, 0.125)

    def test_scope_restored_and_guarded(self):
        original = self.graph.eager_attention_forward
        q = torch.ones(1, 4, 1, 8, dtype=torch.bfloat16)
        k = torch.ones(1, 2, 1, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "test exit"):
            with lifetime.scoped_candidate(self.graph) as counts, torch.no_grad():
                self.graph.eager_attention_forward(self.graph.Qwen3_5Attention(), q, k, k, None, 1.0)
                self.assertEqual(counts["intercepted"], 1)
                with torch.enable_grad():
                    self.graph.eager_attention_forward(self.graph.Qwen3_5Attention(), q, k, k, None, 1.0)
                self.assertEqual(counts["forwarded"], 1)
                raise RuntimeError("test exit")
        self.assertIs(self.graph.eager_attention_forward, original)

    def test_source_mismatch_rejected(self):
        graph = types.SimpleNamespace(__file__=str(HERE / "softmax_lifetime_candidate.py"))
        with self.assertRaisesRegex(RuntimeError, "source differs"):
            lifetime.build_candidate(graph)

if __name__ == "__main__":
    unittest.main()
