"""Compile and run the portable native diagnostic checks on the CPU with a fake recording Device.

Builds the real operators.cpp, text_model.cpp, vision_model.cpp and diagnostics.cpp with the host C++
compiler (CXX, g++ or clang++). No D3D11, GPU, model bytes or Windows build is exercised; those remain
root-owned. The predecessor comparison uses Git metadata for the snapshot commit named below.
"""
from array import array
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "ChandraNative/runtime"
# Snapshot of the native runtime immediately before the diagnostic hooks were added.
PREDECESSOR = "888ab33"
PREDECESSOR_FILES = ("api.h", "text_model.h", "text_model.cpp", "vision_model.h", "vision_model.cpp", "operators.cpp")
spec = importlib.util.spec_from_file_location("compare_diagnostics", ROOT / "scripts/native/compare_diagnostics.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
COMPILER = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
FLAGS = ["-std=c++17", "-O1", "-Wall", "-Wextra"]


def build(output, sources, cwd):
    flags = FLAGS + (["-Wno-dangling-reference"] if "g++" in Path(COMPILER).name else [])
    result = subprocess.run([COMPILER, *flags, "-o", str(output), *map(str, sources)], cwd=cwd,
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise AssertionError(f"compile failed:\n{result.stderr[-4000:]}")
    return output


@unittest.skipUnless(COMPILER and os.name == "posix", "host C++17 compiler and POSIX file sink required")
class NativeDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="chandra-native-diagnostics-")
        cls.root = Path(cls.temporary.name)
        graph = [RUNTIME / name for name in ("operators.cpp", "text_model.cpp", "vision_model.cpp")]
        cls.test = build(cls.root / "diagnostics-test", [RUNTIME / "diagnostics_test.cpp", RUNTIME / "diagnostics.cpp", *graph], cls.root)
        cls.trace = build(cls.root / "diagnostics-trace", [RUNTIME / "diagnostics_trace.cpp", *graph], cls.root)
        cls.dump, cls.sparse, cls.early, cls.failed = (cls.root / name for name in ("dump", "sparse", "early-stop", "failed"))
        cls.result = subprocess.run([str(cls.test), str(cls.dump), str(cls.sparse), str(cls.early), str(cls.failed)],
                                    capture_output=True, text=True, timeout=600)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cpp_unit_tests_pass_and_write_complete_dump(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        summary = json.loads(self.result.stdout)
        self.assertEqual((summary["dump_complete"], summary["written_records"], summary["qualified"]), (True, 265, False))
        loaded = cd.load_dump(self.dump)  # Independent hashlib authentication of every payload.
        self.assertEqual((len(loaded["records"]), loaded["progress_sha256"]), (265, summary["progress_sha256"]))
        for path in self.dump.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path.name)
        stages = {}
        for record in loaded["records"].values():
            stages[record["stage"]] = stages.get(record["stage"], 0) + 1
        self.assertEqual(stages, {"vision.patch_embedding": 2, "vision.position_added": 2, "vision.block_output": 48,
                                  "vision.merger_output": 2, "text.merged_embedding": 4, "text.layer_output": 4 * 32 + 2 * 32,
                                  "text.final_norm": 4 + 2, "text.logits": 3, "text.gdn_conv_state": 3, "text.gdn_recurrent_state": 3})
        rows = {(r["layer"], r["selected_rows"][0]): r for r in loaded["records"].values()
                if r["stage"] == "text.layer_output" and r["phase"] == "prefill"}
        image = rows[(0, 8)]["coordinates"][0]
        self.assertEqual((image["token_id"], image["source"], image["chunk_index"], image["position"]), (248056, "vision", 0, [5, 6, 6]))
        last = rows[(31, 129)]
        self.assertEqual((last["coordinates"][0]["chunk_index"], last["cache"]["cache_length_after"], last["layer_kind"]), (2, 130, "full_attention"))
        progress = [json.loads(line) for line in (self.dump / "progress.jsonl").read_text().splitlines()]
        self.assertEqual([line["name"] for line in progress if line["event"] == "boundary"], ["fake_device_created"])
        # The comparator recomputed the native plan SHA-256, planned keys and every prefix digest; the consumed
        # history is the 130-row prompt followed by decode tokens 5 and 7.
        history = loaded["history"]
        self.assertEqual((history.prompt_rows, len(history.rows), history.rows[130:]), (130, 132, [[5, 128, 128, 128], [7, 129, 129, 129]]))
        self.assertEqual((summary["consumed_rows"], summary["consumed_prefix_sha256"]), (132, history.digests[132]))
        self.assertEqual(loaded["policy_violations"], [])

    def reference(self, candidate, name, history=None, transform=None, producer=None):
        """A float64 copy of the candidate's records written as an independent reference dump."""
        records = []
        for record in candidate["records"].values():
            spec = {k: record[k] for k in cd.IDENTITY + ("coordinates", "bf16_rounding")}
            spec.update(values=list(cd.values(candidate, record)), storage_dtype="float64")
            records.append(transform(spec) if transform else spec)
        header = candidate["header"]["commitments"]
        rows = candidate["history"].rows if history is None else history
        return cd.write_dump(self.root / name, {k: header[k] for k in ("model_revision", "input_manifest_sha256")}, candidate["model"],
                             producer or {"kind": "cpu_fp64_oracle", "test": "copied fake values"}, records,
                             conditioning=rows, prompt_rows=candidate["history"].prompt_rows)

    def test_dump_compares_with_independent_float64_reference(self):
        candidate = cd.load_dump(self.dump)
        records, perturbed = [], None
        for record in candidate["records"].values():
            spec = {k: record[k] for k in cd.IDENTITY + ("coordinates", "cache", "bf16_rounding")}
            values = list(cd.values(candidate, record))
            if perturbed is None and record["stage"] == "text.final_norm":
                bits = array("I", array("f", values[:1]).tobytes())[0] + 0x10000  # One BF16 step away from zero.
                values[0] = array("f", array("I", [bits]).tobytes())[0]
                perturbed = record["key"]
            spec.update(values=values, storage_dtype="float64")
            records.append(spec)
        self.assertIsNotNone(perturbed)
        header = candidate["header"]["commitments"]
        reference = cd.write_dump(self.root / "reference", {k: header[k] for k in ("model_revision", "input_manifest_sha256")},
                                  candidate["model"], {"kind": "cpu_fp64_oracle", "test": "copied fake values"}, records,
                                  conditioning=candidate["history"].rows, prompt_rows=candidate["history"].prompt_rows)
        report = cd.compare(candidate, cd.load_dump(reference), {"*": {"max_bf16_ulp": 0, "max_abs": 1.0}})
        outside = [row["key"] for row in report["records"] if row["status"] != "within_tolerance"]
        self.assertEqual(outside, [perturbed])
        self.assertEqual(report["counts"], {"within_tolerance": 264, "outside_tolerance": 1})
        report = cd.compare(candidate, cd.load_dump(reference), {"*": {"max_bf16_ulp": 1, "max_abs": 1.0}})
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"], report["amd_ocr_qualified"]), ("within_tolerance", True, False))

    def test_monolithic_reference_joins_tiled_native_rows_and_state(self):
        # A reference that ran the prompt as one call has no tile coordinates and different completed cache
        # lengths and state call rows; joins depend only on the causal prefix.
        candidate = cd.load_dump(self.dump)

        def monolithic(spec):
            for field in ("call_row", "chunk_index", "chunk_row"):
                spec["coordinates"][0].pop(field, None)
            prefix = spec["coordinates"][0].get("last_absolute_row", spec["coordinates"][0].get("absolute_row", 0)) + 1
            spec["cache"] = {"cache_length_after": max(prefix, 130), "call_first_row": 0, "call_row_count": prefix}
            return spec

        report = cd.compare(candidate, cd.load_dump(self.reference(candidate, "monolithic", transform=monolithic)), {"*": {"max_abs": 0}})
        self.assertEqual((report["verdict"], report["counts"], report["cpu_oracle_agreement"]), ("within_tolerance", {"within_tolerance": 265}, True))
        tiled = [r for r in candidate["records"].values() if r["stage"] == "text.gdn_recurrent_state" and r["cache_length"] == 130]
        self.assertEqual((tiled[0]["cache"]["call_first_row"], tiled[0]["cache"]["call_row_count"]), (128, 2))

    def test_sparse_native_decode_commits_to_unselected_history(self):
        # Only decode step 1 is captured; step 0 consumed token 5 without any selected record.
        candidate = cd.load_dump(self.sparse)
        self.assertEqual((len(candidate["records"]), candidate["history"].rows[130][0], candidate["dump_complete"]), (14, 5, True))
        same = cd.compare(candidate, cd.load_dump(self.reference(candidate, "sparse-same")), {"*": {"max_abs": 0}})
        self.assertEqual((same["verdict"], same["cpu_oracle_agreement"]), ("within_tolerance", True))
        divergent = [list(row) for row in candidate["history"].rows]
        divergent[130][0] = 6  # A reference whose free-running step 0 chose a different token.
        report = cd.compare(candidate, cd.load_dump(self.reference(candidate, "sparse-divergent", divergent)), {"*": {"max_abs": 0}})
        statuses = {(r["key"].split("/")[2], r["status"], r.get("reason")) for r in report["records"]}
        self.assertEqual(statuses, {("vision", "within_tolerance", None), ("prefill", "within_tolerance", None),
                                    ("decode", "incomparable", "consumed prefix differs")})
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"], report["conditioning"]["first_divergent_generated_index"]),
                         ("incomparable", False, 0))

    def test_native_early_stop_and_failed_dumps_are_never_complete(self):
        complete = cd.load_dump(self.sparse)
        reference = cd.load_dump(self.reference(complete, "partial-reference"))
        for directory, status, written in ((self.early, "completed", 9), (self.failed, "failed", 59)):
            with self.assertRaises(cd.Refusal):
                cd.load_dump(directory)
            loaded = cd.load_dump(directory, allow_incomplete=True)
            self.assertEqual((loaded["run_status"], loaded["dump_complete"], len(loaded["records"])), (status, False, written))
        early = cd.load_dump(self.early, allow_incomplete=True)
        self.assertEqual((early["finished"]["consumed_rows"], early["finished"]["error"], early["incomplete_record"]), (131, None, None))
        report = cd.compare(early, reference, {"*": {"max_abs": 0}})
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("within_tolerance_incomplete_dump", False))
        failed = cd.load_dump(self.failed, allow_incomplete=True)
        self.assertEqual((failed["incomplete_record"], failed["finished"]["error"]), ("text.layer_output/i19/prefill/s-/r8/c-", "injected exclusive-create failure"))

    def test_existing_dump_directory_is_refused(self):
        existing = self.root / "existing"
        existing.mkdir()
        result = subprocess.run([str(self.test), str(existing)], capture_output=True, text=True, timeout=600)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Fresh diagnostic directory required", result.stderr)
        self.assertEqual(list(existing.iterdir()), [])

    def test_disabled_path_trace_matches_predecessor(self):
        sources = {}
        for name in PREDECESSOR_FILES:
            shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{PREDECESSOR}:ChandraNative/runtime/{name}"],
                                   capture_output=True, timeout=60)
            if shown.returncode:
                self.skipTest(f"predecessor {PREDECESSOR} unavailable: {shown.stderr.decode(errors='replace').strip()}")
            sources[name] = shown.stdout
        old = self.root / "predecessor"
        old.mkdir()
        for name, data in sources.items():
            (old / name).write_bytes(data)
        for name in ("diagnostics_fake_device.h", "diagnostics_trace.cpp"):
            shutil.copy(RUNTIME / name, old / name)
        old_trace = build(self.root / "predecessor-trace", [old / "diagnostics_trace.cpp", old / "operators.cpp", old / "text_model.cpp", old / "vision_model.cpp"], old)
        for binary, output in ((old_trace, "old.trace"), (self.trace, "new.trace")):
            result = subprocess.run([str(binary), str(self.root / output)], capture_output=True, text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr)
        old_lines = (self.root / "old.trace").read_text().splitlines()
        new_lines = (self.root / "new.trace").read_text().splitlines()
        self.assertGreater(len(old_lines), 20000)
        first = next((i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b), None)
        self.assertIsNone(first, f"first differing trace event {first}")
        self.assertEqual(len(old_lines), len(new_lines))
        kinds = {}
        for line in new_lines:
            kinds[line.split()[0]] = kinds.get(line.split()[0], 0) + 1
        self.assertEqual(kinds["read"], 3)  # Only the scenario's own logits readbacks; no diagnostic reads when disabled.


if __name__ == "__main__":
    unittest.main()
