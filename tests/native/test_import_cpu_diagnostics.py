"""Standard-library checks of the CPU teacher-export importer on small synthetic exports.

No model, device, Torch or network. Each fixture is a twelve-row prepared input with one 2x2 merged image and a retained-export directory shaped like verification/export_logits.py output; expected positions, prefixes and phases are written out by hand here rather than taken from the importer.
"""
from array import array
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("import_cpu_diagnostics", ROOT / "scripts/native/import_cpu_diagnostics.py")
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)
cd = imp.cd
V = cd.VOCABULARY
ROW = 4 * V
IMG = 248056
PROMPT = [248045, 846, 198, 248053, IMG, IMG, IMG, IMG, 248054, 74455, 198, 271]
P0 = len(PROMPT)
GRID = [1, 4, 4]
# Hand-derived pinned-source positions: four text rows, a 2x2 merged image at t=4, then text from 4 + max(2, 2).
AXES = [[0, 1, 2, 3, 4, 4, 4, 4, 6, 7, 8, 9], [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9]]
NEXT = 10  # Next decode position; rope delta NEXT - P0 = -2.
TEACHER = [2539, 795, 1402, 1944, 420]
PINS = imp.packager.PROCESSOR_PINS
PROFILE = {"size": {"shortest_edge": 3136, "longest_edge": 3145728}}
PIXELS = struct.pack("<1536f", *range(1536)) * 16
MIB = 1024 * 1024
# A runtime string the importer repeats in the plan, every record and the report; each é takes six bytes once JSON-escaped.
LARGE = "a" * 400000 + "é" * 50000


def i64(values):
    return struct.pack(f"<{len(values)}q", *values)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(manifest_sha, prompt, rows, n):
    text = f"chandra.directcompute.consumed-prefix.v1\ninput_manifest_sha256 {manifest_sha}\nprompt_rows {prompt}\n"
    text += "".join(f"{i} {r[0]} {r[1]} {r[2]} {r[3]}\n" for i, r in enumerate(rows[:n]))
    return sha(text.encode())


def expected_rows(extra=()):
    prompt = [[t, AXES[0][i], AXES[1][i], AXES[2][i]] for i, t in enumerate(PROMPT)]
    return prompt + [[t, NEXT + j, NEXT + j, NEXT + j] for j, t in enumerate(extra)]


def finite_row(seed):
    """Random words with exponent bit 30 cleared: always finite, including zeros, subnormals and non-BF16 values."""
    words = array(cd.WORD, random.Random(seed).randbytes(ROW))
    for i in range(len(words)):
        words[i] &= 0xBFFFFFFF
    if sys.byteorder != "little":
        words.byteswap()
    return words.tobytes()


def snapshot(length):
    record = lambda shape: {"shape": shape, "dtype": "torch.float32", "finite": True, "sha256": "a" * 64}
    return {"0.conv_states.0": record([1, 8192, 4]), "0.recurrent_states.0": record([1, 32, 128, 128]),
            "3.keys.0": record([1, 4, length, 256]), "3.values.0": record([1, 4, length, 256])}


class Fixture:
    def __init__(self, root):
        self.root, self.count = root, 0

    def path(self, name):
        self.count += 1
        return self.root / f"{name}{self.count}"

    def manifest(self, extra=(), mutate=None):
        directory = self.path("input")
        directory.mkdir()
        ids = PROMPT + list(extra)
        n = len(ids)
        axes = [a + list(range(NEXT, NEXT + len(extra))) for a in AXES]
        tensors = {"input_ids": (i64(ids), [1, n], "I64"), "attention_mask": (i64([1] * n), [1, n], "I64"),
                   "mm_token_type_ids": (i64([int(t == IMG) for t in ids]), [1, n], "I64"),
                   "pixel_values": (PIXELS, [16, 1536], "F32"), "image_grid_thw": (i64(GRID), [1, 3], "I64"),
                   "position_ids": (i64(sum(axes, [])), [3, 1, n], "I64"), "text_position_ids": (i64(list(range(n))), [1, n], "I64"),
                   "rope_deltas": (i64([NEXT + len(extra) - n]), [1, 1], "I64")}
        manifest = {"schema": "chandra.directcompute.input.v1", "model": "datalab-to/chandra-ocr-2", "revision": imp.PIN,
                    "processor_files": dict(PINS), "processor_profile": "northstone-serving", "processor_kwargs": PROFILE,
                    "prompt_tokens": n, "image_token_id": IMG, "image_token_positions": [4, 5, 6, 7], "image_grid_thw": [GRID],
                    "image": {"file": "input.png", "sha256": sha(b"synthetic png")}, "prompt": {"file": "prompt.txt", "sha256": sha(b"prompt")},
                    "positions": {"origin": "source-derived single-image B1 unpadded specialization", "source_sha256": imp.packager.ROPE_SOURCE_SHA,
                                  "next_decode_position": NEXT + len(extra), "captured_from_amd": False}, "tensors": {}}
        (directory / "input.png").write_bytes(b"synthetic png")
        (directory / "prompt.txt").write_bytes(b"prompt")
        for name, (data, shape, dtype) in tensors.items():  # Valid native descriptors, contiguous strides included.
            (directory / f"{name}.raw").write_bytes(data)
            strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
            manifest["tensors"][name] = {"file": f"{name}.raw", "sha256": sha(data), "bytes": len(data), "dtype": dtype,
                                         "shape": shape, "strides": strides, "byte_order": "little"}
        if mutate:
            mutate(manifest, directory)
        data = json.dumps(manifest, indent=1).encode()
        (directory / "input-manifest.json").write_bytes(data)
        return directory / "input-manifest.json", sha(data)

    def export(self, route, teacher=TEACHER, rows=None, mutate=None, logits=None, partial=False, admission=True):
        directory = self.path("export")
        directory.mkdir()
        context = {"vocabulary_size": V, "model_revision": imp.PIN, "tokenizer_sha256": PINS["tokenizer.json"],
                   "config_sha256": PINS["config.json"], "image_sha256": sha(b"synthetic png"), "prompt_sha256": sha(b"prompt"),
                   "processor_profile": PROFILE}
        inputs = {"input_ids": i64(PROMPT), "attention_mask": i64([1] * P0), "mm_token_type_ids": i64([int(t == IMG) for t in PROMPT]),
                  "pixel_values": PIXELS, "image_grid_thw": i64(GRID)}
        shapes = {"pixel_values": [16, 1536], "image_grid_thw": [1, 3]}
        inputs = {k: {"shape": shapes.get(k, [1, P0]), "dtype": "torch.float32" if k == "pixel_values" else "torch.int64",
                      "finite": True, "sha256": sha(v)} for k, v in inputs.items()}
        runtime = {"torch": "2.14.1+cpu", "transformers": "5.18.0", "python": "3.12.14", "device": "cpu", "dtype": "torch.float32",
                   "threads": 8, "attention": "eager", "model_source_sha256": imp.packager.ROPE_SOURCE_SHA, "script_sha256": "5" * 64}
        if route == imp.CACHED:
            rows = list(range(P0 - 1, P0 + len(teacher))) if rows is None else rows
            context.update(positions=rows, prefix_token_ids=list(PROMPT), target_token_ids=list(teacher))
            states = [snapshot(P0 + i) for i in range(len(teacher) + 1)]
        else:
            rows = list(range(P0 - 1, P0 - 1 + len(teacher))) if rows is None else rows
            context.update(positions=rows, prefix_token_ids=list(PROMPT), target_token_ids=[teacher[r - P0 + 1] for r in rows],
                           teacher_token_ids=list(teacher))
            states = [snapshot(P0 + len(teacher))]
        payload = logits if logits is not None else b"".join(finite_row(i) for i in range(len(rows)))
        meta = {"schema_version": 1, "context": context, "inputs": inputs, "runtime": runtime, "response_sha256": "6" * 64,
                "states": states, "logits": {"file": "logits.f32", "shape": [len(rows), V], "dtype": "float32", "byte_order": "little",
                                             "sha256": sha(payload)}, "elapsed_seconds": 1.0, "peak_rss_kib": 1}
        if route == imp.CACHED:
            meta["cache_equivalence"] = {"max_abs": 0.0, "rmse": 0.0, "argmax_equal": True, "finite": True}
        if mutate:
            mutate(meta)
        if admission:  # Written from the final metadata, as the producer does before loading the model.
            early = dict(meta["context"])
            if route == imp.MONOLITHIC:
                early.update(positions=list(range(P0 - 1, P0 + 2)), target_token_ids=list(teacher[:2]))
            early.pop("teacher_token_ids", None)
            (directory / "admission.json").write_text(json.dumps({"schema_version": 1, "context": early, "inputs": meta["inputs"],
                                                                  "runtime": meta["runtime"], "response_sha256": meta["response_sha256"], "states": []}))
        (directory / "logits.f32").write_bytes(payload)
        if partial:
            (directory / "logits.partial.f32").write_bytes(payload if partial is True else partial)
        data = json.dumps(meta, indent=2).encode()
        (directory / "metadata.json").write_bytes(data)
        return directory, sha(data), payload


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chandra-cpu-import-")
        self.root = Path(os.path.realpath(self.temporary.name))
        self.fx = Fixture(self.root)
        self.manifest, self.manifest_sha = self.fx.manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def run_import(self, export, meta_sha, route, rows, manifest=None, attest=True, output=None, extra=(), limits=()):
        output = output or self.fx.path("out")
        manifest, manifest_sha = manifest or (self.manifest, self.manifest_sha)
        bounds = dict({"--max-input-bytes": 64 * MIB, "--max-rows": 64, "--max-output-bytes": 64 * MIB, "--deadline-seconds": 600}, **dict(limits))
        args = ["--export-dir", export, "--metadata-sha256", meta_sha, "--input-manifest", manifest, "--input-sha256", manifest_sha,
                "--route", route, "--rows", rows, "--output", output]
        args += [str(v) for item in bounds.items() for v in item]
        if attest:
            args += ["--attest-model-sha256", imp.MODEL_SHA256, "--attestation-basis", "synthetic fixture"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = imp.main([str(a) for a in list(args) + list(extra)])
        return code, json.loads(out.getvalue()), Path(output)

    def refused(self, fragment, *args, **options):
        code, report, output = self.run_import(*args, **options)
        self.assertEqual((code, report["status"]), (2, "refused"), report)
        self.assertIn(fragment, report["error"])
        self.assertFalse(os.path.lexists(output))
        return report

    @staticmethod
    def written(output):
        """Regular-file bytes actually on disk under an output directory, measured independently of the importer."""
        return sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file() and not path.is_symlink())

    def test_cached_export_keeps_exact_words_phases_and_teacher_conditioning(self):
        export, meta_sha, payload = self.fx.export(imp.CACHED, teacher=TEACHER[:3])
        code, report, output = self.run_import(export, meta_sha, imp.CACHED, "0,1,2,3")
        self.assertEqual((code, report["status"], report["error"]), (0, "imported", None))
        self.assertEqual(sorted(os.listdir(output)), ["import-report.json", "reference"])
        dump = cd.load_dump(output / "reference")
        rows = expected_rows(TEACHER[:3])
        self.assertEqual(len(dump["history"].rows), P0 + 3)
        self.assertEqual(dump["history"].rows, rows)
        for source in range(4):
            key = "text.logits/i-/prefill/s-/r0/c-" if source == 0 else f"text.logits/i-/decode/s{source - 1}/r0/c-"
            record = dump["records"][key]
            # Byte-for-byte equality with the source row, not merely equal floats.
            self.assertEqual((output / "reference" / record["file"]).read_bytes(), payload[source * ROW:(source + 1) * ROW])
            self.assertEqual(record["bf16_rounding"], "none")
            absolute, n = P0 - 1 + source, P0 + source
            coordinate = record["coordinates"][0]
            self.assertEqual((coordinate["absolute_row"], coordinate["token_id"], coordinate["position"]),
                             (absolute, rows[absolute][0], rows[absolute][1:]))
            self.assertEqual(record["conditioning"], {"prefix_rows": n, "prefix_sha256": canonical(self.manifest_sha, P0, rows, n)})
            self.assertEqual(record["produces_generated_index"], source)
            self.assertEqual(record["reference_provenance"]["position_origin"], "source_derived")
            if source == 0:
                self.assertEqual((record["phase"], record["decode"], coordinate["source"]), ("prefill", None, "text"))
            else:
                self.assertEqual((record["phase"], record["decode_step"]), ("decode", source - 1))
                self.assertEqual(record["decode"], {"step": source - 1, "consumed_generated_index": source - 1,
                                                    "consumed_token_id": TEACHER[source - 1], "logits_produce_generated_index": source})
                self.assertNotIn("source", coordinate)
            self.assertEqual(record["cache"]["cache_length_after"], P0 + source)
        header = dump["header"]
        self.assertFalse(header["producer"]["kind"].startswith("cpu_"))
        self.assertEqual((dump["model"]["model_sha256_origin"], dump["model"]["config_sha256_origin"]), ("owner_attestation", "producer_metadata"))
        self.assertEqual(header["plan"]["position_origin"], "source_derived")
        self.assertEqual(report["output"]["comparator_load"]["records"], 4)
        self.assertEqual(report["source_export"]["recorded_cache_lengths"], [P0, P0 + 1, P0 + 2, P0 + 3])
        self.assertFalse(any(report["claims"].values()))
        self.assertIn("model_safetensors_sha256", [m["commitment"] for m in report["missing_producer_commitments"]])

    def test_sparse_rows_and_earlier_history_divergence(self):
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:4])
        code, report, output = self.run_import(export, meta_sha, imp.CACHED, "0,3")
        self.assertEqual(code, 0)
        reference = cd.load_dump(output / "reference")
        self.assertEqual(sorted(reference["records"]), ["text.logits/i-/decode/s2/r0/c-", "text.logits/i-/prefill/s-/r0/c-"])
        self.assertEqual(len(reference["history"].rows), P0 + 4)  # Unselected teacher rows still condition later rows.

        def native(generated, name):
            spike = array("f", bytes(ROW))
            spike[7] = 4.0
            prefill = {"stage": "text.logits", "phase": "prefill", "logical_shape": [1, V], "selected_rows": [0], "payload_shape": [1, V],
                       "complete_tensor": True, "produces_generated_index": 0, "values": spike,
                       "coordinates": [{"absolute_row": P0 - 1, "call_row": P0 - 1, "chunk_index": 0, "chunk_row": P0 - 1,
                                        "token_id": PROMPT[-1], "position": [9, 9, 9], "source": "text"}]}
            records = [prefill]
            for step in (0, 2):
                records.append({"stage": "text.logits", "phase": "decode", "decode_step": step, "logical_shape": [1, V], "selected_rows": [0],
                                "payload_shape": [1, V], "complete_tensor": True, "produces_generated_index": step + 1, "values": spike,
                                "decode": {"step": step, "consumed_generated_index": step, "consumed_token_id": generated[step],
                                           "logits_produce_generated_index": step + 1},
                                "coordinates": [{"absolute_row": P0 + step, "call_row": 0, "chunk_index": 0, "chunk_row": 0,
                                                 "token_id": generated[step], "position": [NEXT + step] * 3}]})
            model = {"model_sha256": imp.MODEL_SHA256, "config_sha256": PINS["config.json"], "model_bytes": imp.MODEL_BYTES}
            return cd.load_dump(cd.write_dump(self.root / name, {"model_revision": imp.PIN, "input_manifest_sha256": self.manifest_sha},
                                              model, {"kind": "directcompute_native"}, records,
                                              conditioning=expected_rows(generated), prompt_rows=P0))

        # Native generated index 1 differs from the teacher; index 2 coincides again but its conditioning does not.
        report = cd.compare(native([TEACHER[0], 9999, TEACHER[2]], "diverged"), reference, {})
        status = {row["key"]: (row["status"], row.get("reason")) for row in report["records"]}
        self.assertEqual(status["text.logits/i-/prefill/s-/r0/c-"][0], "no_tolerance")
        self.assertEqual(status["text.logits/i-/decode/s0/r0/c-"], ("incomparable", "reference record absent"))
        self.assertEqual(status["text.logits/i-/decode/s2/r0/c-"], ("incomparable", "consumed prefix differs"))
        self.assertEqual((report["conditioning"]["first_divergent_row"], report["conditioning"]["first_divergent_generated_index"]), (P0 + 1, 1))
        self.assertFalse(report["cpu_oracle_agreement"])
        same = cd.compare(native(TEACHER[:3], "same"), reference, {})
        self.assertEqual({row["key"]: row["status"] for row in same["records"]}["text.logits/i-/decode/s2/r0/c-"], "no_tolerance")

    def test_monolithic_rows_stay_prefill(self):
        export, meta_sha, payload = self.fx.export(imp.MONOLITHIC)
        code, report, output = self.run_import(export, meta_sha, imp.MONOLITHIC, "0")
        self.assertEqual(code, 0)
        dump = cd.load_dump(output / "reference")
        self.assertEqual(list(dump["records"]), ["text.logits/i-/prefill/s-/r0/c-"])
        self.assertEqual(dump["history"].rows, expected_rows(TEACHER))  # The whole monolithic call's consumed rows.
        self.assertEqual(dump["records"]["text.logits/i-/prefill/s-/r0/c-"]["cache"]["call_tokens"], P0 + len(TEACHER))
        # A later teacher-forced row is never relabelled as a cached decode step.
        message = self.refused("would otherwise call it a cached decode step", export, meta_sha, imp.MONOLITHIC, "0,2")["error"]
        self.assertIn(f"exactly those {P0 + 2} rows (the {P0} processor rows followed by the first 2 teacher tokens)", message)
        # With an input whose prompt is exactly its consumed prefix, row 2 is that input's last prefill row.
        extended = self.fx.manifest(extra=TEACHER[:2])
        code, report, output = self.run_import(export, meta_sha, imp.MONOLITHIC, "2", manifest=extended)
        self.assertEqual(code, 0, report)
        record = cd.load_dump(output / "reference")["records"]["text.logits/i-/prefill/s-/r0/c-"]
        self.assertEqual((record["phase"], record["produces_generated_index"], record["coordinates"][0]["absolute_row"]), ("prefill", 0, P0 + 1))
        self.assertEqual(record["coordinates"][0]["position"], [NEXT + 1] * 3)
        self.assertEqual(record["reference_provenance"]["response_generated_index"], 2)
        self.assertEqual(record["conditioning"]["prefix_sha256"], canonical(extended[1], P0 + 2, expected_rows(TEACHER), P0 + 2))
        self.assertEqual((output / "reference" / record["file"]).read_bytes(), payload[2 * ROW:3 * ROW])
        # Against that longer prompt, row 0's shorter prefix is refused too rather than joined.
        self.refused(f"is not the manifest's {P0 + 2} prompt rows", export, meta_sha, imp.MONOLITHIC, "0", manifest=extended)
        # Routes are checked against the producer-recorded call structure.
        self.refused("contradicting the cached-decode route", export, meta_sha, imp.CACHED, "0")
        cached, cached_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:2])
        self.refused("contradicting the monolithic route", cached, cached_sha, imp.MONOLITHIC, "0")
        self.refused("cached export prefilled exactly", cached, cached_sha, imp.CACHED, "0", manifest=extended)

    def test_mismatched_commitments_are_refused(self):
        def lengths(meta):
            for name in ("3.keys.0", "3.values.0"):
                meta["states"][1][name] = dict(meta["states"][1][name], shape=[1, 4, 99, 256])

        cases = [
            ("profile", lambda m: m["context"].update(processor_profile={})),
            ("Tokenizer", lambda m: m["context"].update(tokenizer_sha256="7" * 64)),
            ("modeling source", lambda m: m["runtime"].update(model_source_sha256="7" * 64)),
            ("pixel_values differs", lambda m: m["inputs"]["pixel_values"].update(sha256="7" * 64)),
            ("Only unrestricted CPU FP32", lambda m: m["runtime"].update(dtype="torch.bfloat16")),
            ("logits shape", lambda m: m["logits"].update(shape=[3, V - 1])),
            ("Image commitment", lambda m: m["context"].update(image_sha256="7" * 64)),
            ("unexpected ['greedy']", lambda m: m.update(greedy={})),
            ("prompt tokens differ", lambda m: m["context"]["prefix_token_ids"].__setitem__(1, 847)),
            ("cache lengths", lengths),
        ]
        for fragment, mutate in cases:
            with self.subTest(fragment):
                export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:2], mutate=mutate)
                self.refused(fragment, export, meta_sha, imp.CACHED, "0")
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:2])
        admission = json.loads((export / "admission.json").read_text())
        (export / "admission.json").write_text(json.dumps(dict(admission, response_sha256="8" * 64)))
        self.refused("admission.json contradicts", export, meta_sha, imp.CACHED, "0")
        export, meta_sha, payload = self.fx.export(imp.CACHED, teacher=TEACHER[:2])
        self.refused("--metadata-sha256", export, "0" * 64, imp.CACHED, "0")
        self.refused("--input-sha256", export, meta_sha, imp.CACHED, "0", manifest=(self.manifest, "0" * 64))
        self.refused("names weights other than", export, meta_sha, imp.CACHED, "0", attest=False,
                     extra=["--attest-model-sha256", "1" * 64, "--attestation-basis", "x"])
        self.refused("records no safetensors SHA-256", export, meta_sha, imp.CACHED, "0", attest=False)
        tampered = bytearray(payload)
        tampered[ROW + 5] ^= 1
        (export / "logits.f32").write_bytes(tampered)
        self.refused("logits.f32 SHA-256 differs", export, meta_sha, imp.CACHED, "0")
        bad_positions = self.fx.manifest(mutate=lambda m, d: self._retensor(m, d, "position_ids", i64(sum(AXES, [])[:-1] + [10])))
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:2])
        self.refused("pinned-source derivation", export, meta_sha, imp.CACHED, "0", manifest=bad_positions)

    @staticmethod
    def _retensor(manifest, directory, name, data):
        (directory / f"{name}.raw").write_bytes(data)
        manifest["tensors"][name].update(sha256=sha(data), bytes=len(data))

    def test_nonfinite_words_are_refused_or_preserved_unchanged(self):
        special = array(cd.WORD, [0x7F800001, 0xFFC00001, 0x7FC12345, 0x7F800000, 0x80000000, 0x00000001, 0x7F7FFFFF, 0x3F800001])
        words = array(cd.WORD, bytes(ROW))
        words[100:108] = special
        if sys.byteorder != "little":
            words.byteswap()
        payload = finite_row(1) + words.tobytes()
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1], logits=payload)
        self.refused("nonfinite words {1: 4}", export, meta_sha, imp.CACHED, "0,1")
        code, report, output = self.run_import(export, meta_sha, imp.CACHED, "0,1", extra=["--nonfinite", "preserve"])
        self.assertEqual(code, 0, report)
        record = cd.load_dump(output / "reference")["records"]["text.logits/i-/decode/s0/r0/c-"]
        self.assertEqual((output / "reference" / record["file"]).read_bytes(), payload[ROW:])
        self.assertEqual(record["observed"]["nonfinite"], 4)
        self.assertTrue(any("preserved unchanged" in text for text in report["limitations"]))

    def test_written_payload_must_equal_its_source_row(self):
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
        original = imp.native_order

        def altered(data):  # One changed word, as an unfaithful copy would produce.
            values = original(data)
            values[3] = -values[3] if values[3] else 1.0
            return values

        imp.native_order = altered
        try:
            code, report, output = self.run_import(export, meta_sha, imp.CACHED, "0,1")
        finally:
            imp.native_order = original
        self.assertEqual((code, report["status"]), (1, "failed"))
        self.assertIn("not byte-identical", report["error"])
        self.assertEqual(sorted(os.listdir(output)), ["import-report.json", "unverified-reference"])

    def test_file_system_and_ceiling_refusals(self):
        export, meta_sha, payload = self.fx.export(imp.CACHED, teacher=TEACHER[:2])
        args = (export, meta_sha, imp.CACHED, "0,1")
        (export / "notes.txt").write_text("x")
        self.refused("no teacher export writes", *args)
        (export / "notes.txt").unlink()
        (export / "logits.f32").write_bytes(payload[:-4])
        self.refused("truncated", *args)
        (export / "logits.f32").write_bytes(payload + b"\0\0\0\0")
        self.refused("exceeds its", *args)
        (export / "logits.f32").unlink()
        (self.root / "elsewhere.f32").write_bytes(payload)
        os.symlink(self.root / "elsewhere.f32", export / "logits.f32")
        self.refused("regular non-symlink", *args)
        (export / "logits.f32").unlink()
        (export / "logits.f32").write_bytes(payload)
        link = self.root / "linked-export"
        os.symlink(export, link)
        self.refused("must not traverse a symlink", link, meta_sha, imp.CACHED, "0")
        (export / "logits.partial.f32").write_bytes(payload[::-1])
        self.refused("logits.partial.f32 differs", *args)
        (export / "logits.partial.f32").unlink()
        monolithic, mono_sha, _ = self.fx.export(imp.MONOLITHIC, partial=True)
        self.refused("contradicts the monolithic route", monolithic, mono_sha, imp.MONOLITHIC, "0")
        self.refused("sorted ascending", export, meta_sha, imp.CACHED, "1,0")
        self.refused("outside the export", export, meta_sha, imp.CACHED, "0,3")
        self.refused("exceed --max-rows", *args, limits={"--max-rows": 1})
        self.refused("--max-input-bytes", *args, limits={"--max-input-bytes": 3 * ROW})
        self.refused("--max-output-bytes", *args, limits={"--max-output-bytes": ROW})
        self.refused("--max-output-bytes must be within", *args, limits={"--max-output-bytes": 512 * MIB})
        self.refused("absolute normalized", "relative/export", meta_sha, imp.CACHED, "0")
        os.symlink(self.root, self.root / "linked-root")
        self.refused("must not traverse a symlink", *args, output=self.root / "linked-root" / "out")

    def test_fresh_output_and_monotonic_deadline(self):
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:2])
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("owner state")
        code, report, _ = self.run_import(export, meta_sha, imp.CACHED, "0", output=existing)
        self.assertEqual((code, report["status"]), (2, "refused"))
        self.assertEqual(os.listdir(existing), ["keep.txt"])
        self.assertEqual((existing / "keep.txt").read_text(), "owner state")
        for seconds in (0, 7201, "nan"):
            self.refused("--deadline-seconds", export, meta_sha, imp.CACHED, "0", limits={"--deadline-seconds": seconds})
        ticks = itertools.count()
        imp.clock = lambda: float(next(ticks))  # Every clock reading advances one second.
        try:
            self.refused("monotonic deadline expired", export, meta_sha, imp.CACHED, "0", limits={"--deadline-seconds": 20})
        finally:
            imp.clock = time.monotonic
        code, report, output = self.run_import(export, meta_sha, imp.CACHED, "0,1,2", attest=False, extra=["--dry-run"])
        self.assertEqual((code, report["status"], report["conversion_admissible"]), (3, "dry_run", False))
        self.assertEqual(os.listdir(output), ["import-report.json"])
        self.assertEqual([r["prefix_rows"] for r in report["records"]], [P0, P0 + 1, P0 + 2])
        self.assertLessEqual(report["resources"]["input_bytes_read"], 64 * MIB)

    def large_export(self):
        """A cached export whose runtime carries LARGE, and the exact forecast a roomy dry run reports for it."""
        export, meta_sha, payload = self.fx.export(imp.CACHED, teacher=TEACHER[:2], mutate=lambda m: m["runtime"].update(attention=LARGE))
        args = (export, meta_sha, imp.CACHED, "0,2")
        code, report, output = self.run_import(*args, output=self.root / "ceiling-0", extra=["--dry-run"],
                                               limits={"--max-output-bytes": 9999999})
        self.assertEqual((code, report["status"], report["conversion_admissible"]), (0, "dry_run", True), report)
        self.assertEqual(os.listdir(output), ["import-report.json"])
        self.assertEqual(self.written(output), report["resources"]["output_bytes"])
        self.assertLessEqual(self.written(output), report["forecast"]["report_bytes_bound"])
        # Output names of one length and seven-digit ceilings keep every argv, and so every report, the same size.
        self.assertEqual({len(str(v)) for v in (report["forecast"]["report_bytes_bound"] - 1, report["forecast"]["conversion_output_bytes_bound"] - 1)}, {7})
        return args, payload, report["forecast"]

    def test_large_metadata_output_is_bounded_before_any_write(self):
        args, payload, forecast = self.large_export()
        dry, conversion, reference = forecast["report_bytes_bound"], forecast["conversion_output_bytes_bound"], forecast["reference_bytes"]
        # At exactly its own bound a dry run's report fits, but it reports the conversion inadmissible; one byte less is refused.
        code, report, output = self.run_import(*args, output=self.root / "ceiling-1", extra=["--dry-run"], limits={"--max-output-bytes": dry})
        self.assertEqual((code, report["conversion_admissible"], report["forecast"]["conversion_within_max_output_bytes"]), (3, False, False))
        self.assertEqual(self.written(output), report["resources"]["output_bytes"])
        self.assertLessEqual(self.written(output), dry)
        self.refused("dry-run report bound", *args, output=self.root / "ceiling-2", extra=["--dry-run"], limits={"--max-output-bytes": dry - 1})
        # A conversion is refused one byte below the dry run's forecast and imports at exactly it.
        self.refused(f"Forecast output of {conversion} bytes", *args, output=self.root / "ceiling-3", limits={"--max-output-bytes": conversion - 1})
        code, report, output = self.run_import(*args, output=self.root / "ceiling-4", limits={"--max-output-bytes": conversion})
        self.assertEqual((code, report["status"]), (0, "imported"), report)
        self.assertEqual(report["forecast"]["output_bytes_bound"], conversion)
        progress = (output / "reference" / "progress.jsonl").read_bytes()
        self.assertEqual(progress.count(b"a" * 400000), 4)  # Plan producer, plan limitations and both records' producers.
        self.assertEqual(self.written(output / "reference"), reference)  # The dump, repeated metadata included, is forecast exactly.
        self.assertEqual(hashlib.sha256(progress).hexdigest(), report["output"]["comparator_load"]["progress_sha256"])
        self.assertEqual(self.written(output), report["resources"]["output_bytes"])
        self.assertLessEqual(self.written(output), conversion)
        self.assertEqual(json.loads((output / "import-report.json").read_bytes()), report)
        dump = cd.load_dump(output / "reference")
        for source, key in ((0, "text.logits/i-/prefill/s-/r0/c-"), (2, "text.logits/i-/decode/s1/r0/c-")):
            self.assertEqual((output / "reference" / dump["records"][key]["file"]).read_bytes(), payload[source * ROW:(source + 1) * ROW])
        self.assertEqual(dump["header"]["producer"]["runtime"]["attention"], LARGE)

    def test_failure_routes_stay_within_the_ceiling_and_never_promote_without_a_report(self):
        args, _, forecast = self.large_export()
        ceiling = forecast["conversion_output_bytes_bound"]  # Every route below runs at exactly the forecast ceiling.

        def attempt(name, **substitutes):
            saved = {key: getattr(imp, key) for key in substitutes}
            for key, value in substitutes.items():
                setattr(imp, key, value)
            try:
                code, report, output = self.run_import(*args, output=self.root / f"ceiling-{name}", limits={"--max-output-bytes": ceiling})
            finally:
                for key, value in saved.items():
                    setattr(imp, key, value)
            self.assertEqual((code, report["status"], report["output"]["reference"]), (1, "failed", None), report)
            self.assertFalse(os.path.lexists(output / "reference"))
            self.assertLessEqual(self.written(output), ceiling)
            return report, output

        # A verification failure keeps the complete staged dump and its failure receipt, together within the ceiling.
        original = imp.native_order

        def altered(data):
            values = original(data)
            values[3] = -values[3] if values[3] else 1.0
            return values

        report, output = attempt("5", native_order=altered)
        self.assertIn("not byte-identical", report["error"])
        self.assertEqual(sorted(os.listdir(output)), ["import-report.json", "unverified-reference"])
        self.assertEqual(self.written(output), report["resources"]["output_bytes"])
        self.assertEqual(self.written(output / "unverified-reference"), forecast["reference_bytes"])

        # Promotion fails after the acceptance report is durable; the report is rewritten in place, its long message bounded to fit.
        def failing_rename(source, target):
            raise OSError(28, "injected rename failure " + "x" * 100000)

        report, output = attempt("6", promote=failing_rename)
        self.assertTrue(report["error"].endswith(" [truncated]") and "injected rename failure" in report["error"])
        self.assertLessEqual(len(json.dumps(report["error"])), imp.ERROR_LIMIT)
        self.assertEqual(json.loads((output / "import-report.json").read_bytes()), report)
        self.assertEqual(self.written(output), report["resources"]["output_bytes"])
        self.assertTrue(report["output"]["comparator_load"]["authenticated"])  # Verified, yet never renamed to reference/.

        # If the acceptance report cannot be written nothing is promoted; the failure receipt replaces it.
        real, calls = imp.write_report, []

        def first_write_fails(*values):
            calls.append(values)
            if len(calls) == 1:
                raise OSError(5, "injected report failure")
            return real(*values)

        report, output = attempt("7", write_report=first_write_fails)
        self.assertIn("injected report failure", report["error"])
        self.assertEqual(sorted(os.listdir(output)), ["import-report.json", "unverified-reference"])

        # With no report writable at all, standard output carries the failure and still no reference/ exists.
        def every_write_fails(*values):
            raise OSError(5, "injected report failure")

        report, output = attempt("8", write_report=every_write_fails)
        self.assertIn("import-report.json could not be written", report["error"])
        self.assertEqual(os.listdir(output), ["unverified-reference"])

    def test_position_tensors_must_have_the_exact_native_abi(self):
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
        n = P0

        def declared(key, **fields):
            return self.fx.manifest(mutate=lambda m, d: m["tensors"][key].update(fields))

        def stored(key, data, **fields):
            return self.fx.manifest(mutate=lambda m, d: (self._retensor(m, d, key, data), m["tensors"][key].update(fields)))

        def unstrided(m, d):
            del m["tensors"]["rope_deltas"]["strides"]

        cases = [
            # The reviewed probe: unchanged causal-order bytes declared as F32 [1, 2 * prompt].
            ("text_position_ids must be I64 [1, 12]", declared("text_position_ids", dtype="F32", shape=[1, 2 * n], strides=[2 * n, 1])),
            ("text_position_ids must be I64 [1, 12]", declared("text_position_ids", shape=[n, 1], strides=[1, 1])),
            ("text_position_ids must be I64 [1, 12]", declared("text_position_ids", shape=[1, 1, n], strides=[n, n, 1])),
            ("position_ids must be I64 [3, 1, 12]", declared("position_ids", dtype="F32", shape=[3, 2, n], strides=[2 * n, n, 1])),
            ("position_ids must be I64 [3, 1, 12]", declared("position_ids", shape=[3, n], strides=[n, 1])),
            ("position_ids must be I64 [3, 1, 12]", declared("position_ids", shape=[1, 3, n], strides=[3 * n, n, 1])),
            ("rope_deltas must be I64 [1, 1]", declared("rope_deltas", dtype="F32", shape=[1, 2], strides=[2, 1])),
            ("rope_deltas must be I64 [1, 1]", declared("rope_deltas", shape=[1], strides=[1])),
            # The reviewed probe: four validly committed bytes declared F32 [1, 1].
            ("rope_deltas must be I64 [1, 1]", stored("rope_deltas", b"\0" * 4, dtype="F32")),
            # Byte extents that are not whole I64 values under the right declaration.
            ("rope_deltas bytes differ from its I64 shape", stored("rope_deltas", b"\0" * 4)),
            ("text_position_ids bytes differ from its I64 shape", stored("text_position_ids", i64(range(n))[:-4])),
            ("position_ids must declare the contiguous element strides [12, 12, 1]", declared("position_ids", strides=[1, 12, 3])),
            ("rope_deltas must declare the contiguous element strides [1, 1]", self.fx.manifest(mutate=unstrided)),
            ("input_ids must be I64 [1, 12]", declared("input_ids", dtype=["I64"])),
            ("pixel_values must be F32 [1..32768, 1536]", declared("pixel_values", shape=[32, 768], strides=[768, 1])),
        ]
        for fragment, manifest in cases:
            with self.subTest(fragment):
                self.refused(fragment, export, meta_sha, imp.CACHED, "0", manifest=manifest)
        # The optional causal text positions may be absent; present, they must be exact.
        code, report, _ = self.run_import(export, meta_sha, imp.CACHED, "0",
                                          manifest=self.fx.manifest(mutate=lambda m, d: m["tensors"].pop("text_position_ids")))
        self.assertEqual((code, report["status"]), (0, "imported"), report)

    def test_descriptor_and_metadata_integers_must_be_json_integers(self):
        # Python equality admits 96.0, 9.6e1 or true for an integer; the native unsignedNumber reader refuses each of them.
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
        n = P0

        def declared(key, **fields):
            return self.fx.manifest(mutate=lambda m, d: m["tensors"][key].update(fields))

        def literal(key, field, text):
            """A raw JSON number literal, such as an exponent, that json.dumps never writes for the fixture."""
            path, _ = self.fx.manifest(mutate=lambda m, d: m["tensors"][key].update({field: "@literal@"}))
            data = path.read_bytes()
            self.assertEqual(data.count(b'"@literal@"'), 1)
            path.write_bytes(data.replace(b'"@literal@"', text.encode()))
            return path, sha(path.read_bytes())

        def strides_type(key, strides, rank, shown):
            return (f"{key} must declare the contiguous element strides {strides} the native input ABI requires as {rank} JSON integers "
                    f"below 2**64; the manifest declares {shown}")

        bytes_type = "bytes must be a JSON integer from 0 to 268435456 under the native input ABI; the manifest declares"
        cases = [
            # The three reviewed probes, each of which imported and promoted reference/ before this repair.
            (f"text_position_ids {bytes_type} 96.0", declared("text_position_ids", bytes=96.0)),
            (strides_type("text_position_ids", [n, 1], 2, "[12.0, 1.0]"), declared("text_position_ids", strides=[12.0, 1.0])),
            (strides_type("rope_deltas", [1, 1], 2, "[True, True]"), declared("rope_deltas", strides=[True, True])),
            # Raw exponent literals, which Python also decodes to floats equal to the expected integers.
            (f"input_ids {bytes_type} 96.0", literal("input_ids", "bytes", "9.6e1")),
            (strides_type("mm_token_type_ids", [n, 1], 2, "[12.0, 1]"), literal("mm_token_type_ids", "strides", "[1.2e1, 1]")),
            (strides_type("position_ids", [n, n, 1], 3, "[12, 12, 1.0]"), literal("position_ids", "strides", "[12, 12, 1E0]")),
            # Already unequal, now refused by type or by the native range before the extent comparison.
            (f"rope_deltas {bytes_type} True", declared("rope_deltas", bytes=True)),
            (f"rope_deltas {bytes_type} {2 ** 64}", declared("rope_deltas", bytes=2 ** 64)),
            (f"pixel_values {bytes_type} {256 * MIB + 8}", declared("pixel_values", bytes=256 * MIB + 8)),
            (strides_type("rope_deltas", [1, 1], 2, "[-1, 1]"), declared("rope_deltas", strides=[-1, 1])),
            (strides_type("rope_deltas", [1, 1], 2, f"[{2 ** 64}, 1]"), declared("rope_deltas", strides=[2 ** 64, 1])),
            (strides_type("rope_deltas", [1, 1], 2, "[1]"), declared("rope_deltas", strides=[1])),
        ]
        # Every exact ABI role: float bytes, float strides and a Boolean for its unit innermost stride.
        for key, record in json.loads(self.manifest.read_bytes())["tensors"].items():
            size, strides, rank = record["bytes"], record["strides"], len(record["strides"])
            cases += [(f"{key} {bytes_type} {float(size)}", declared(key, bytes=float(size))),
                      (strides_type(key, strides, rank, [float(v) for v in strides]), declared(key, strides=[float(v) for v in strides])),
                      (strides_type(key, strides, rank, strides[:-1] + [True]), declared(key, strides=strides[:-1] + [True]))]
        # Integer metadata compared by value: native reads the scalars with unsignedNumber, and its == never equates true with 1.
        scalars = "image_token_id and positions.next_decode_position must be JSON integers"
        cases += [(scalars, self.fx.manifest(mutate=lambda m, d: m.update(image_token_id=float(IMG)))),
                  (scalars, self.fx.manifest(mutate=lambda m, d: m["positions"].update(next_decode_position=float(NEXT)))),
                  ("image_grid_thw rows must be a list of nonnegative integers",
                   self.fx.manifest(mutate=lambda m, d: m.update(image_grid_thw=[[True] + GRID[1:]]))),
                  ("image_grid_thw rows must be a list of nonnegative integers",
                   self.fx.manifest(mutate=lambda m, d: m.update(image_grid_thw=[[float(v) for v in GRID]]))),
                  ("image_token_positions must be a list of nonnegative integers",
                   self.fx.manifest(mutate=lambda m, d: m.update(image_token_positions=[4.0, 5.0, 6.0, 7.0])))]
        self.assertEqual(len(cases), 41)
        for fragment, manifest in cases:
            with self.subTest(fragment):
                self.refused(fragment, export, meta_sha, imp.CACHED, "0", manifest=manifest)
                self.refused(fragment, export, meta_sha, imp.CACHED, "0", manifest=manifest, extra=["--dry-run"])
        # The same descriptors with exact JSON integers still import, and an integer literal through the same rewrite is a control.
        code, report, output = self.run_import(export, meta_sha, imp.CACHED, "0")
        self.assertEqual((code, report["status"]), (0, "imported"), report)
        self.assertEqual(self.written(output), report["resources"]["output_bytes"])
        code, report, _ = self.run_import(export, meta_sha, imp.CACHED, "0", manifest=literal("input_ids", "bytes", "96"))
        self.assertEqual((code, report["status"]), (0, "imported"), report)

    def test_overflowing_numbers_and_unrepresentable_json_are_refused_before_output(self):
        def replaced(directory, name, old, new):
            data = (directory / name).read_bytes()
            self.assertEqual(data.count(old), 1)
            (directory / name).write_bytes(data.replace(old, new))

        cases = [
            ("the number under 'elapsed_seconds' overflows", '"elapsed_seconds": 1.0', '"elapsed_seconds": 1e999'),  # Never used.
            ("the number under 'max_abs' overflows", '"max_abs": 0.0', '"max_abs": 1e999'),  # Copied into the report.
            ("the number under 'threads' overflows", '"threads": 8', '"threads": -1e999'),  # Copied into the plan and every record.
            ("JSON cannot be represented: ValueError", '"peak_rss_kib": 1', '"peak_rss_kib": ' + "9" * 5000),
            ("JSON nesting exceeds 64 levels", '"rmse": 0.0', '"rmse": ' + "[" * 70 + "]" * 70),
            ("JSON cannot be represented: RecursionError", '"rmse": 0.0', '"rmse": ' + "[" * 200000 + "]" * 200000),
        ]
        for fragment, old, new in cases:
            with self.subTest(fragment):
                export, _, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
                replaced(export, "metadata.json", old.encode(), new.encode())
                meta_sha = sha((export / "metadata.json").read_bytes())
                self.refused(f"metadata.json: {fragment}", export, meta_sha, imp.CACHED, "0")
                self.refused(f"metadata.json: {fragment}", export, meta_sha, imp.CACHED, "0", extra=["--dry-run"])
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
        replaced(export, "admission.json", b'"threads": 8', b'"threads": 1e999')
        self.refused("admission.json: the number under 'threads' overflows", export, meta_sha, imp.CACHED, "0")
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
        manifest, _ = self.fx.manifest()
        replaced(manifest.parent, manifest.name, b'"captured_from_amd": false', b'"captured_from_amd": 1e999')
        self.refused("input manifest: the number under 'captured_from_amd' overflows", export, meta_sha, imp.CACHED, "0",
                     manifest=(manifest, sha(manifest.read_bytes())))

    def test_malformed_arguments_names_and_writer_are_refused_as_json(self):
        export, meta_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1])
        self.refused("argument --max-rows: invalid int value", export, meta_sha, imp.CACHED, "0", limits={"--max-rows": "many"})
        self.refused("unrecognized arguments: --dry", export, meta_sha, imp.CACHED, "0", extra=["--dry"])  # No abbreviations.
        self.refused("comma-separated source row indices", export, meta_sha, imp.CACHED, "1" * 5000)
        superscript = lambda m: m["states"][0].update({"².keys.0": m["states"][0]["3.keys.0"]})  # str.isdigit() accepts "²".
        named, named_sha, _ = self.fx.export(imp.CACHED, teacher=TEACHER[:1], mutate=superscript)
        self.refused("names an unknown cache tensor", named, named_sha, imp.CACHED, "0")
        # The forecast reproduces one exact comparator writer; with any other, nothing is written.
        pinned, imp.WRITER_SHA256 = imp.WRITER_SHA256, "0" * 64
        try:
            self.refused("is not the writer whose exact output this importer forecasts", export, meta_sha, imp.CACHED, "0")
        finally:
            imp.WRITER_SHA256 = pinned


if __name__ == "__main__":
    unittest.main()
