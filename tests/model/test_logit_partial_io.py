"""Exercise exporter I/O without importing Torch or allocating a model."""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np


source = Path(__file__).resolve().parents[2] / "verification/export_logits.py"
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "append_logit_row")
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
append_logit_row = namespace["append_logit_row"]


class PartialLogitIO(unittest.TestCase):
    def test_complete_bytes_equal_old_restack(self):
        rows = np.array([[0.0, -0.0, 1.25], [-12.5, 2**-20, 32768.0]], dtype=">f4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.f32"
            for row in rows:
                append_logit_row(path, row)
            self.assertEqual(path.read_bytes(), np.stack(rows).astype("<f4").tobytes())

    def test_interruption_after_closed_row_preserves_prefix(self):
        rows = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.f32"
            try:
                for row in rows[:2]:
                    append_logit_row(path, row)
                raise RuntimeError("Interrupted before next forward")
            except RuntimeError:
                pass
            self.assertEqual(path.read_bytes(), np.stack(rows[:2]).astype("<f4").tobytes())
            self.assertEqual(path.stat().st_size, 2 * 3 * 4)


if __name__ == "__main__":
    unittest.main()
