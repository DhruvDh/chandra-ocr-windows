"""Build the portable diagnostic serialization regression with each host compiler and verify every dump it writes.

ChandraNative/runtime/diagnostics_serialization_test.cpp is standard C++17, so the same source builds with GCC, Clang
and MSVC. This harness builds it with g++ and clang++ when present (never MSVC), checks its exit status and pinned
progress digests, loads every written dump with the unchanged strict parser in scripts/native/compare_diagnostics.py,
and independently fixes the exact JSON type of every field of every progress line. Set
CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT to the output directory of a separately built binary (for example MSVC) to
verify that output with the same checks. No GPU, D3D11, model bytes or numerical claim is exercised.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "ChandraNative/runtime"
# Recorder immediately before explicit one-element coordinate construction; MSVC Build09 compiled this source.
PREDECESSOR = "caefe03"
spec = importlib.util.spec_from_file_location("compare_diagnostics", ROOT / "scripts/native/compare_diagnostics.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
# Identical for GCC 16.2.1 and Clang 23.1.1 with both the fixed and the predecessor recorder; pinned in the C++ test too.
EXPECTED_PROGRESS = {
    "complete": "5a1a1768a541e09f2f0067adc26f3c5485777a89a74a83dd4e3318138631c29b",
    "early_stop": "3463b7484bc79222d1295a792b45f951e42e3a01f886ac72a442e9eea7083564",
    "failed": "3a2574561ec61ef5a29c58e4a3fb04aee25a038cfa3f2af2d02ab61d3963fe03",
}
# Scenario: (run_status, dump_complete, written records).
SCENARIOS = {"complete": ("completed", True, 105), "early_stop": ("completed", False, 225), "failed": ("failed", False, 103)}
SOURCES = ("diagnostics_serialization_test.cpp", "operators.cpp", "text_model.cpp", "vision_model.cpp")
FLAGS = ["-std=c++17", "-O1", "-Wall", "-Wextra"]
# Stages whose coordinates MSVC Build09 wrote as a bare object: every site that brace-wrapped one Json value.
BUILD09_BARE = ("vision.patch_embedding", "vision.position_added", "vision.block_output", "text.layer_output",
                "text.final_norm", "text.logits")


def compilers():
    found = []
    for name in (os.environ.get("CXX"), "g++", "clang++"):
        path = name and shutil.which(name)
        if path and os.path.realpath(path) not in [os.path.realpath(p) for p in found]:
            found.append(path)
    return found if os.name == "posix" else []


def build(compiler, output, recorder, cwd):
    flags = FLAGS + (["-Wno-dangling-reference"] if "g++" in Path(compiler).name else [])
    sources = [RUNTIME / SOURCES[0], recorder, *(RUNTIME / name for name in SOURCES[1:])]
    result = subprocess.run([compiler, *flags, f"-I{RUNTIME}", "-o", str(output), *map(str, sources)], cwd=cwd,
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise AssertionError(f"{compiler} compile failed:\n{result.stderr[-4000:]}")
    return output


def run(binary, output):
    return subprocess.run([str(binary), str(output)], capture_output=True, text=True, timeout=600)


def natural(value):
    return type(value) is int and value >= 0


def digest(value):
    return isinstance(value, str) and cd.DIGEST.fullmatch(value) is not None


def position(value):
    return isinstance(value, list) and len(value) == 3 and all(natural(v) for v in value)


RECORD_FIELDS = {"event", "schema", "sequence", "file", "key", "stage", "component", "phase", "layer", "layer_kind",
                 "vision_block", "decode_step", "cache_length", "logical_shape", "selected_axis", "selected_rows",
                 "payload_shape", "complete_tensor", "storage_dtype", "byte_order", "bf16_rounding", "payload_bytes",
                 "payload_sha256", "observed", "coordinates", "cache", "decode", "conditioning",
                 "produces_generated_index", "commitments", "producer", "qualified"}
PATCH = {"patch_row", "frame", "grid_row", "grid_col", "merged_row"}
ROW = {"absolute_row", "call_row", "chunk_index", "chunk_row", "position", "token_id"}
COORDINATES = {"vision.patch_embedding": PATCH, "vision.position_added": PATCH, "vision.block_output": PATCH,
               "vision.merger_output": {"merged_row", "prompt_row"},
               "text.merged_embedding": {"absolute_row", "token_id", "source", "merged_row", "position"},
               "text.layer_output": ROW, "text.final_norm": ROW, "text.logits": ROW,
               "text.gdn_conv_state": {"last_absolute_row", "last_position"},
               "text.gdn_recurrent_state": {"last_absolute_row", "last_position"}}
CACHE = {"cache_length_after", "cache_length_source", "request_tokens_before_call", "generated_before_call",
         "call_tokens", "call_first_row", "call_row_count"}
DECODE = {"step", "consumed_generated_index", "consumed_token_id", "logits_produce_generated_index"}
TERMINAL = {"event", "run_status", "dump_complete", "planned_records", "written_records", "missing_records",
            "incomplete_record", "payload_bytes", "readbacks", "readback_bytes", "error", "consumed_rows",
            "consumed_prefix_sha256", "qualified"}


def record_errors(r):
    """Exact field set and JSON type of one record line, independent of compare_diagnostics.py."""
    if set(r) != RECORD_FIELDS:
        return [f"record fields differ: {sorted(set(r) ^ RECORD_FIELDS)}"]
    errors = []

    def need(ok, what):
        if not ok:
            errors.append(what)

    stage, phase = r["stage"], r["phase"]
    if stage not in COORDINATES or not isinstance(phase, str):
        return [f"unknown stage {stage!r} or phase"]
    text, state = stage.startswith("text."), stage in cd.STATE_STAGES
    causal, decode = stage in cd.CONDITIONED, phase == "decode"
    need(phase in (("prefill", "decode") if causal else ("merge",) if stage == "text.merged_embedding" else ("vision",)), "phase")
    need(r["schema"] == cd.RECORD_SCHEMA and natural(r["sequence"]) and isinstance(r["file"], str) and isinstance(r["key"], str), "identity types")
    need(r["component"] == ("text" if text else "vision") and r["qualified"] is False, "component or qualification")
    if stage == "text.layer_output" or state:
        need(natural(r["layer"]) and r["layer"] < 32
             and r["layer_kind"] == ("full_attention" if r["layer"] % 4 == 3 else "gated_delta_net"), "layer and layer_kind")
    else:
        need(r["layer"] is None and r["layer_kind"] is None, "layer and layer_kind must be null")
    need(natural(r["vision_block"]) if stage == "vision.block_output" else r["vision_block"] is None, "vision_block")
    need(natural(r["decode_step"]) if decode else r["decode_step"] is None, "decode_step")
    need(natural(r["cache_length"]) if state else r["cache_length"] is None, "cache_length")
    selected, logical, payload = r["selected_rows"], r["logical_shape"], r["payload_shape"]
    need(r["selected_axis"] == 0 and natural(r["selected_axis"]), "selected_axis")
    if stage == "text.gdn_conv_state":
        need(selected == "all" and payload == logical, "conv state selects all rows")
    else:
        need(isinstance(selected, list) and len(selected) == 1 and natural(selected[0]), "selected_rows must be a one-element integer array")
        need(isinstance(logical, list) and payload == [1] + logical[1:], "payload_shape")
    need(isinstance(logical, list) and logical and all(natural(v) and v > 0 for v in logical), "logical_shape")
    need(type(r["complete_tensor"]) is bool and r["storage_dtype"] == "float32" and r["byte_order"] == "little", "storage types")
    need(r["bf16_rounding"] == (cd.STATE_POLICY if state else cd.BF16_POLICY), "bf16_rounding")
    need(natural(r["payload_bytes"]) and digest(r["payload_sha256"]), "payload bytes and digest")
    observed = r["observed"]
    need(isinstance(observed, dict) and set(observed) == {"nonfinite", "finite_non_bf16_representable"}
         and all(natural(v) for v in observed.values()), "observed")
    coordinates = r["coordinates"]
    if not (isinstance(coordinates, list) and len(coordinates) == 1 and isinstance(coordinates[0], dict)):
        return errors + ["coordinates must be an array of exactly one object"]
    c = coordinates[0]
    expected = COORDINATES[stage] | ({"source"} if stage in cd.ROW_STAGES and not decode else set())
    need(set(c) == expected, f"coordinate fields differ: {sorted(set(c) ^ expected)}")
    if set(c) == expected:
        for name, value in c.items():
            if name in ("position", "last_position"):
                need(position(value), f"coordinate {name}")
            elif name == "source":
                need(value in ("vision", "text"), "coordinate source")
            elif name == "merged_row" and stage == "text.merged_embedding":
                need(natural(value) if c["source"] == "vision" else value is None, "embedding merged_row")
            else:
                need(natural(value), f"coordinate {name}")
    if causal:
        cache, conditioning = r["cache"], r["conditioning"]
        need(isinstance(cache, dict) and set(cache) == CACHE
             and all(natural(v) for k, v in cache.items() if k != "cache_length_source")
             and cache["cache_length_source"] == {"text.final_norm": "last_layer", "text.logits": "request"}.get(stage, "layer"), "cache")
        need(isinstance(conditioning, dict) and set(conditioning) == {"prefix_rows", "prefix_sha256"}
             and natural(conditioning["prefix_rows"]) and digest(conditioning["prefix_sha256"]), "conditioning")
    else:
        need(r["cache"] is None and r["conditioning"] is None, "vision and merge cache/conditioning must be null")
    need(isinstance(r["decode"], dict) and set(r["decode"]) == DECODE and all(natural(v) for v in r["decode"].values())
         and r["decode"]["step"] == r["decode_step"] if decode else r["decode"] is None, "decode")
    need(natural(r["produces_generated_index"]) if stage == "text.logits" else r["produces_generated_index"] is None, "produces_generated_index")
    need(isinstance(r["commitments"], dict) and isinstance(r["producer"], dict), "commitments and producer objects")
    return errors


def line_errors(line):
    event = line.get("event")
    if event == "record":
        return record_errors(line)
    shapes = {
        "plan": ({"event", "schema", "plan", "plan_sha256", "commitments", "producer", "qualified", "conditioning", "scope"},
                 lambda j: j["schema"] == cd.PROGRESS_SCHEMA and isinstance(j["plan"], dict) and digest(j["plan_sha256"])
                 and isinstance(j["commitments"], dict) and isinstance(j["producer"], dict) and j["qualified"] is False
                 and j["conditioning"] == {"schema": cd.PREFIX_SCHEMA, "prompt_rows": j["conditioning"].get("prompt_rows")}
                 and natural(j["conditioning"]["prompt_rows"]) and isinstance(j["scope"], str)),
        "model_authenticated": ({"event", "model", "device"},
                                lambda j: isinstance(j["model"], dict) and set(j["model"]) == {"revision", "model_sha256", "config_sha256", "model_bytes"}
                                and all(isinstance(j["model"][k], str) for k in ("revision", "model_sha256", "config_sha256"))
                                and natural(j["model"]["model_bytes"]) and isinstance(j["device"], dict)),
        "boundary": ({"event", "name", "completed_records", "payload_bytes"},
                     lambda j: isinstance(j["name"], str) and natural(j["completed_records"]) and natural(j["payload_bytes"])),
        "record_started": ({"event", "sequence", "file", "key"},
                           lambda j: natural(j["sequence"]) and isinstance(j["file"], str) and isinstance(j["key"], str)),
        "conditioning": ({"event", "first_row", "rows", "consumed_rows", "prefix_sha256"},
                         lambda j: natural(j["first_row"]) and natural(j["consumed_rows"]) and digest(j["prefix_sha256"])
                         and isinstance(j["rows"], list) and j["rows"]
                         and all(isinstance(row, list) and len(row) == 4 and all(natural(v) for v in row) for row in j["rows"])),
        "finished": (TERMINAL,
                     lambda j: j["run_status"] in ("completed", "failed") and type(j["dump_complete"]) is bool and j["qualified"] is False
                     and all(natural(j[k]) for k in ("planned_records", "written_records", "payload_bytes", "readbacks", "readback_bytes", "consumed_rows"))
                     and isinstance(j["missing_records"], list) and all(isinstance(k, str) for k in j["missing_records"])
                     and (j["incomplete_record"] is None or isinstance(j["incomplete_record"], str))
                     and (j["error"] is None if j["run_status"] == "completed" else isinstance(j["error"], str) and j["error"])
                     and digest(j["consumed_prefix_sha256"])),
    }
    if event not in shapes:
        return [f"unknown event {event!r}"]
    fields, valid = shapes[event]
    if set(line) != fields:
        return [f"{event} fields differ: {sorted(set(line) ^ fields)}"]
    return [] if valid(line) else [f"{event} field types differ"]


def schema_errors(progress):
    lines = [json.loads(text) for text in progress.read_text(encoding="utf-8").splitlines()]
    return [(number, error) for number, line in enumerate(lines, start=1) for error in line_errors(line)]


class Verification:
    """Checks shared by host-compiled and externally supplied output directories."""

    def verify_output(self, output):
        for name, (status, complete, written) in SCENARIOS.items():
            with self.subTest(output=str(output), dump=name):
                directory = output / name
                if complete:
                    loaded = cd.load_dump(directory)
                else:
                    with self.assertRaises(cd.Refusal):
                        cd.load_dump(directory)
                    loaded = cd.load_dump(directory, allow_incomplete=True)
                self.assertEqual((loaded["run_status"], loaded["dump_complete"], len(loaded["records"])), (status, complete, written))
                self.assertEqual(loaded["progress_sha256"], EXPECTED_PROGRESS[name])
                self.assertEqual(loaded["policy_violations"], [])
                observed = [r["observed"] for r in loaded["records"].values()]
                self.assertTrue(any(o["nonfinite"] for o in observed))
                # Only explicit-plan dumps select FP32 GDN state, whose payloads end with a finite non-BF16 word.
                self.assertEqual(any(o["finite_non_bf16_representable"] for o in observed), name != "early_stop")
                self.assertEqual(schema_errors(directory / "progress.jsonl"), [])
                for record in loaded["records"].values():
                    self.assertIsInstance(record["coordinates"], list)
                    self.assertEqual(len(record["coordinates"]), 1)
                    self.assertIsInstance(record["coordinates"][0], dict)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["failures"], summary["gpu_or_d3d11_exercised"], summary["nlohmann_json"]), (0, False, "3.12.0"))
        self.assertEqual({name: d["expected_progress_sha256"] for name, d in summary["dumps"].items()}, EXPECTED_PROGRESS)
        self.assertTrue(all(d["progress_matches"] and d["schema_errors"] == 0 for d in summary["dumps"].values()))
        self.assertEqual(summary["schema_negative_controls_refused"], 8)
        return summary


def files(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "summary.json"}


@unittest.skipUnless(compilers(), "host C++17 compiler on POSIX required")
class HostCompiledSerializationTests(unittest.TestCase, Verification):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="chandra-diagnostic-serialization-")
        cls.root = Path(cls.temporary.name)
        cls.outputs, cls.results = {}, {}
        for index, compiler in enumerate(compilers()):
            binary = build(compiler, cls.root / f"serialization-{index}", RUNTIME / "diagnostics.cpp", cls.root)
            cls.outputs[compiler] = cls.root / f"output-{index}"
            cls.results[compiler] = run(binary, cls.outputs[compiler])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_every_host_compiler_passes_with_pinned_progress(self):
        for compiler, result in self.results.items():
            with self.subTest(compiler=compiler):
                self.assertEqual(result.returncode, 0, result.stderr)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["single_json_brace_assignment"], {"lvalue": "array", "prvalue": "array"})
                self.assertGreater(summary["schema_lines_checked"], 850)

    def test_every_dump_roundtrips_the_unchanged_strict_parser_with_exact_types(self):
        for compiler, output in self.outputs.items():
            with self.subTest(compiler=compiler):
                self.verify_output(output)

    def test_fixed_recorder_writes_the_predecessor_bytes_on_conforming_compilers(self):
        # Where {value} already built a one-element array, the explicit construction changes no byte.
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{PREDECESSOR}:ChandraNative/runtime/diagnostics.cpp"],
                               capture_output=True, timeout=60)
        if shown.returncode:
            self.skipTest(f"predecessor {PREDECESSOR} unavailable: {shown.stderr.decode(errors='replace').strip()}")
        old = self.root / "predecessor"
        old.mkdir(exist_ok=True)
        (old / "diagnostics.cpp").write_bytes(shown.stdout)
        expected = files(next(iter(self.outputs.values())))
        self.assertEqual(len(expected), 3 + 105 + 225 + 103)
        for index, (compiler, output) in enumerate(self.outputs.items()):
            with self.subTest(compiler=compiler):
                self.assertEqual(files(output), expected)
                binary = build(compiler, self.root / f"predecessor-{index}", old / "diagnostics.cpp", old)
                previous = self.root / f"predecessor-output-{index}"
                result = run(binary, previous)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(files(previous), expected)

    def test_build09_bare_coordinate_objects_are_refused_at_line_6(self):
        # Synthetic counterexample with the shape MSVC Build09 wrote: every brace-wrapped coordinate became its object.
        source = next(iter(self.outputs.values())) / "complete"
        bare = self.root / "build09-shape"
        shutil.copytree(source, bare)
        lines, changed = [], []
        for number, text in enumerate((source / "progress.jsonl").read_text(encoding="utf-8").splitlines(), start=1):
            line = json.loads(text)
            if line["event"] == "record" and line["stage"] in BUILD09_BARE:
                line["coordinates"] = line["coordinates"][0]
                text = json.dumps(line, sort_keys=True, separators=(",", ":"))
                changed.append(number)
            lines.append(text)
        (bare / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual((len(changed), changed[0]), (86, 6))
        with self.assertRaises(cd.Refusal) as refusal:
            cd.load_dump(bare)
        self.assertEqual(str(refusal.exception), "progress line 6: one coordinate object per record required")
        flagged = sorted({number for number, _ in schema_errors(bare / "progress.jsonl")})
        self.assertEqual(flagged, changed)


@unittest.skipUnless(os.environ.get("CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT"), "set CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT to verify external output")
class ExternalSerializationOutputTests(unittest.TestCase, Verification):
    def test_external_output_matches_the_pinned_schema_and_bytes(self):
        output = Path(os.environ["CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT"])
        summary = self.verify_output(output)
        self.assertTrue(re.fullmatch(r"(gcc|clang|msvc) .+", summary["compiler"]), summary["compiler"])


if __name__ == "__main__":
    unittest.main()
