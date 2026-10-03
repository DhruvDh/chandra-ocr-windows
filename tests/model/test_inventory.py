import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("inventory", Path(__file__).resolve().parents[2] / "scripts/model_inventory.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryTest(unittest.TestCase):
    def write(self, root, header, data):
        raw = json.dumps(header).encode()
        path = Path(root) / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
        return path

    def test_tied_storage_requires_bytes_not_config(self):
        with tempfile.TemporaryDirectory() as root:
            names = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
            header = {name: {"dtype": "BF16", "shape": [1, 2], "data_offsets": [i * 4, i * 4 + 4]} for i, name in enumerate(names)}
            result = inventory.inspect_weights(self.write(root, header, b"abcdabcd"))
            self.assertTrue(result["tied_embeddings"]["equal"])
            self.assertEqual(result["resident_weight_bytes_if_shared"], 4)
            result = inventory.inspect_weights(self.write(root, header, b"abcdabce"))
            self.assertFalse(result["tied_embeddings"]["equal"])
            self.assertEqual(result["resident_weight_bytes_if_shared"], 8)

    def test_overlap_and_extra_data_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            header = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                      "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}}
            with self.assertRaises(ValueError):
                inventory.inspect_weights(self.write(root, header, b"abcdef"))
            with self.assertRaises(ValueError):
                inventory.inspect_weights(self.write(root, {"a": header["a"]}, b"abcdef"))

    def test_integrity_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write(root, {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}, b"abcd")
            manifest = {"model": "test", "revision": "test", "files": [{"file": path.name, "size": path.stat().st_size, "sha256": inventory.digest(path)}]}
            self.assertTrue(inventory.verify_model(Path(root), manifest)["integrity_passed"])
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"z")
            with self.assertRaises(ValueError):
                inventory.verify_model(Path(root), manifest)


if __name__ == "__main__":
    unittest.main()
