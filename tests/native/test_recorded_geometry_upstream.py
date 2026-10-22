"""Compare the native recorded-geometry indexing with the pinned upstream Qwen3.5 source on the CPU.

Run with the pinned CPU environment (torch 2.14.1+cpu, transformers 5.18.0), for example
`/venv/bin/python -I tests/native/test_recorded_geometry_upstream.py`; it skips elsewhere. It builds
recorded_geometry_test.cpp against the translated runtime HLSL (see test_exact_input_correctness.py) and checks,
at grid [1,126,96] and 3,617 prompt rows:
  * native vision positions equal transformers.vision_utils.get_vision_position_ids;
  * native FP32 cos/sin (host libm, not HLSL sin/cos) differ from Qwen3_5VisionRotaryEmbedding by at most an
    absolute 2^-23 at every element; this uniform absolute gate is not a ULP bound or ULP measurement;
  * the emulated vision_position.hlsl interpolation of a synthetic BF16 table is bit-identical to
    get_vision_interpolation_indices_and_weights followed by the model's embedding, weighted sum and BF16 cast;
  * the recorded prompt positions and RoPE delta equal Qwen3_5Model.get_rope_index, cached decode positions equal
    compute_3d_position_ids, every merged image row's four patch positions are its text M-RoPE block, and
    masked_scatter places image feature k on prompt row 4 + k as the native merge does.
Only synthetic tensors and model metadata are used; no weights are loaded and no model is executed.
"""
import hashlib
import json
import sys
import tempfile
import types
import unittest
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_exact_input_correctness as base  # noqa: E402

PINNED = {"torch": "2.14.1+cpu", "transformers": "5.18.0"}
SOURCE_SHA256 = {"models/qwen3_5/modeling_qwen3_5.py": "d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939",
                 "vision_utils.py": "bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356"}
# vision_config of the pinned config.json (SHA-256 e26f17b70463de21fd68f1ef6d8f67f8e33e65d55a7a6895242130a18db60587).
VISION_CONFIG = {"deepstack_visual_indexes": [], "depth": 24, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1024, "in_channels": 3,
                 "initializer_range": 0.02, "intermediate_size": 4096, "model_type": "qwen3_5", "num_heads": 16,
                 "num_position_embeddings": 2304, "out_hidden_size": 2560, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}
IMAGE_TOKEN, PROMPT, PATCHES, MERGED, FIRST_IMAGE = 248056, 3617, 12096, 3024, 4


def pinned():
    try:
        versions = {name: metadata.version(name) for name in PINNED}
    except metadata.PackageNotFoundError:
        return False
    if versions != PINNED:
        return False
    import transformers
    root = Path(transformers.__file__).parent
    return all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in SOURCE_SHA256.items())


@unittest.skipUnless(base.COMPILER and pinned(), "pinned torch/transformers source and a host C++17 compiler required")
class RecordedGeometryUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        torch.set_num_threads(1)
        cls.np, cls.torch = np, torch
        cls.temporary = tempfile.TemporaryDirectory(prefix="chandra-recorded-upstream-")
        root = Path(cls.temporary.name)
        _, binaries = base.build_all(root, ("recorded-test",))
        positions = base.recorded_positions()
        assert hashlib.sha256(positions).hexdigest() == base.RECORDED_POSITIONS_SHA256
        (root / "positions.i64").write_bytes(positions)
        cls.out = root / "recorded"
        cls.out.mkdir()
        cls.report = base.run(binaries["recorded-test"], root / "positions.i64", cls.out)
        cls.positions = np.frombuffer(positions, "<i8").reshape(3, 1, PROMPT)
        cls.grid = torch.tensor([[1, 126, 96]])
        cls.native_positions = np.fromfile(cls.out / "vision_positions.u32", "<u4").reshape(PATCHES, 4)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_vision_positions_and_rotary(self):
        from transformers import vision_utils
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        np = self.np
        upstream = vision_utils.get_vision_position_ids(self.grid, 2).numpy()
        self.assertTrue(np.array_equal(self.native_positions[:, :2], upstream))
        self.assertTrue((self.native_positions[:, 2:] == [126, 96]).all())
        config = Qwen3_5VisionConfig(**VISION_CONFIG)
        self.assertEqual(config.rope_parameters, {"rope_theta": 10000.0, "rope_type": "axial"})
        cos, sin = m.Qwen3_5VisionRotaryEmbedding(config)(self.torch.zeros(1), self.torch.from_numpy(upstream))
        native = np.fromfile(self.out / "rotary_cos_sin.f32", "<f4").reshape(PATCHES, 128)
        # Uniform absolute gate, not a ULP metric: 2^-23 is two FP32 ulps only for magnitudes in [0.5, 1).
        self.assertLessEqual(float(np.abs(native[:, :64] - cos.numpy()).max()), 2 ** -23)
        self.assertLessEqual(float(np.abs(native[:, 64:] - sin.numpy()).max()), 2 ** -23)

    def test_learned_position_interpolation_is_bit_identical(self):
        from transformers import vision_utils
        np, torch = self.np, self.torch
        words = np.fromfile(self.out / "pos_embed_table.bf16x2", "<u4")
        bits = (np.stack([words & 0xFFFF, words >> 16], 1).reshape(-1)[:2304 * 1024].astype(np.uint32) << 16)
        table = torch.from_numpy(bits.view(np.float32).reshape(2304, 1024).copy()).to(torch.bfloat16)
        embedding = torch.nn.Embedding(2304, 1024, _weight=table)
        indices, weights = vision_utils.get_vision_interpolation_indices_and_weights(
            self.grid, 48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        with torch.no_grad():  # As Qwen3_5VisionModel.forward: (pos_embed(idx) * w).sum(1), then the BF16 activation cast.
            reference = (embedding(indices) * weights[:, :, None]).sum(1).to(torch.bfloat16).float().numpy()
        native = np.fromfile(self.out / "position_embedding.f32", "<f4").reshape(PATCHES, 1024)
        self.assertEqual(int((native != reference).sum()), 0)
        self.assertGreater(float(np.abs(reference).max()), 0.5)

    def test_text_positions_decode_and_multimodal_alignment(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        np, torch = self.np, self.torch
        ids = torch.full((1, PROMPT), 198, dtype=torch.long)
        ids[0, FIRST_IMAGE:FIRST_IMAGE + MERGED] = IMAGE_TOKEN
        types_ = (ids == IMAGE_TOKEN).long()
        fake = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2)), rope_deltas=None)
        fake.get_vision_position_ids = types.MethodType(m.Qwen3_5Model.get_vision_position_ids, fake)
        fake.get_rope_index = types.MethodType(m.Qwen3_5Model.get_rope_index, fake)
        positions, delta = fake.get_rope_index(ids, types_, image_grid_thw=self.grid, attention_mask=torch.ones_like(ids))
        self.assertTrue(np.array_equal(positions.numpy(), self.positions))
        self.assertEqual(int(delta), 656 - PROMPT)
        fake.rope_deltas = delta
        for step in range(2):  # Cached decode: inference_core.cpp uses maximumPosition + index + 1 on all axes.
            cache = types.SimpleNamespace(get_seq_length=lambda n=PROMPT + step: n)
            decode = m.Qwen3_5Model.compute_3d_position_ids(fake, input_ids=None, inputs_embeds=torch.zeros(1, 1, 2560),
                                                            image_grid_thw=None, past_key_values=cache, mm_token_type_ids=None)
            self.assertEqual(decode.reshape(-1).tolist(), [655 + step + 1] * 3)
        # Merged row k (patches 4k..4k+3, block-major) is the 2x2 block at its prompt row's (h, w) M-RoPE coordinate.
        block = self.native_positions[:, :2].reshape(MERGED, 4, 2) // 2
        self.assertTrue((block == block[:, :1, :]).all())
        image = self.positions[1:, 0, FIRST_IMAGE:FIRST_IMAGE + MERGED].T - FIRST_IMAGE
        self.assertTrue(np.array_equal(block[:, 0, :], image))
        features = torch.arange(MERGED, dtype=torch.float32)[:, None].expand(MERGED, 2)
        merged = torch.full((1, PROMPT, 2), -1.0).masked_scatter((ids == IMAGE_TOKEN)[..., None].expand(1, PROMPT, 2), features)
        self.assertTrue(torch.equal(merged[0, FIRST_IMAGE:FIRST_IMAGE + MERGED, 0], torch.arange(MERGED, dtype=torch.float32)))
        self.assertEqual(self.report["merge_emulation"]["image_rows"], [FIRST_IMAGE, FIRST_IMAGE + MERGED - 1])


if __name__ == "__main__":
    unittest.main()
