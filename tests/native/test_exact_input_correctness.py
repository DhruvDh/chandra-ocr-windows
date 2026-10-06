"""CPU checks for the exact recorded-input correctness repair (docs/directcompute-exact-input-correctness.md).

Translates the unmodified runtime HLSL into C++ (tests/native/hlsl_cpu.py), builds three drivers with the host
C++17 compiler and the unchanged graph sources, and checks:
  * the 2c44182 text attention softmax has a groupshared data race that legal schedules turn into wrong
    probabilities, while the repaired shader is race-free, schedule-independent and within BF16 rounding of a
    float64 oracle; every other barrier shader is a race-free negative control; the race detector itself flags
    synthetic hazards; the fast linear/GEMV replica equals the emulated HLSL bit for bit;
  * at the exact recorded geometry (grid [1,126,96], 3,617 rows, positions hashed against the recorded input)
    the graph's tiles, chunks, offsets, cache lengths and RoPE coordinates are complete, its tracked allocation
    trajectory equals the failed Build16 run phase by phase, and the merge maps every row;
  * executed text layers are chunk invariant, prefill equals cached decode, state is initialized and isolated,
    and the repaired attention layer is schedule independent while the 2c44182 one is not;
  * the model-free GPU regression driver (text_softmax_regression.h) uses production's shader name, cbuffer
    words, groups and plane size at all 17 planned calls, its oracle accepts the translated repaired shader
    under every schedule and rejects the translated 2c44182 control under the hazardous ascending schedule.
No D3D11, GPU, model weight or trained value is used. The fiber emulator and the recorded file driver are
Linux/POSIX CPU seams, not MSVC or GPU tests. test_recorded_geometry_upstream.py compares the emitted vision
geometry with the pinned Transformers source.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "ChandraNative/runtime"
SHADERS = ROOT / "ChandraNative/shaders/runtime"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hlsl_cpu  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare_diagnostics", ROOT / "scripts/native/compare_diagnostics.py")
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)

COMPILER = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
FLAGS = ["-std=c++17", "-O3", "-ffp-contract=off", "-Wall", "-Wextra"]


def compiler_family(compiler):
    """Classify by the compiler's own predefined macros, never by its file name: clang++ defines both __clang__
    and __GNUC__, GCC defines __GNUC__ only. Anything else is "other" and receives no family-specific option."""
    result = subprocess.run([compiler, "-x", "c++", "-dM", "-E", os.devnull], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise AssertionError(f"{compiler} cannot report its predefined macros:\n{result.stderr[-2000:]}")
    macros = {line.split()[1] for line in result.stdout.splitlines() if line.startswith("#define ")}
    return "clang" if "__clang__" in macros else "gcc" if "__GNUC__" in macros else "other"


def compiler_flags(family):
    # GCC-only: silences GCC's -Wdangling-reference false positive in the unchanged vision_model.cpp. Clang
    # rejects the option name with -Wunknown-warning-option, which the warning-free build gate would report.
    return FLAGS + (["-Wno-dangling-reference"] if family == "gcc" else [])
SOURCE_COMMIT = "2c441820e7b58b2641d3fa4da602c2e4146ce679"
ORIGINAL_SOFTMAX_SHA256 = "5b6c0445dcebe2acdc8104bf1c0fd2b37e1c05665d30d960ca4a0e83d6432369"
REPAIRED_SOFTMAX_SHA256 = "283d3356ea3c882517e96ddff26706bae21bfe3cd04a523fc065e9aa8183169e"
# derived-position_ids.raw of input manifest 0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae.
RECORDED_POSITIONS_SHA256 = "698cadcea94a3ce34a9e33848f591dfcea1786ef8dd32a6db707bd6c087c493a"
# Source-controlled byte copy of the 2c44182 shader that the GPU regression driver dispatches as its control.
CONTROL_SHADER = ROOT / "ChandraNative/shaders/control/text_attention_softmax_2c44182.hlsl"
# planJson() of text_softmax_regression.h; the Windows driver prints the same hash in its inactive mode.
REGRESSION_PLAN_SHA256 = "e19e8f0a82b201e769e2d60069e485dca4e9374d4b92cdf5f29717d83575d213"
# memory_observations of the failed normal-allowance Build16 result.json (SHA-256
# da3b30b2fe94f5263a30fb36e23b52ef5b3dc7898d14c740cd7a6fce35155ed0); the 14-token point records the same values.
FAILED_RUN_PHASES = [
    ("complete_vision_forward", 9159042048, 9369868288), ("text_embedding", 9196094596, 9369868288),
    ("ordered_multimodal_merge", 9233118208, 9369868288), ("text_request_cache_creation", 10242788352, 10242788352),
    ("text_prefill", 10243791872, 10311350016), ("greedy_cached_decode_and_token_readback", 10206753792, 10311350016),
    ("request_retirement_and_drain", 9078531072, 10311350016)]
RESIDENT_MODEL_BYTES = 9078531072
PLAN = ROOT / "verification/native/recorded-native-0-plan.json"
PLAN_FILE_SHA256 = "c37044ddf7ae8becd8628dd5dc515c4d54df4c5670a24c25323e50b1e8b8d1ba"
RESOLVED_PLAN_SHA256 = "0e4832ac763820defb1743fcdd02fb0e6b457d9e7895cf7df2b37ac50726f47e"
CONTROLS = {
    "control/uav_race.hlsl": "RWByteAddressBuffer y:register(u0);\n[numthreads(4,1,1)] void main(uint3 t:SV_DispatchThreadID){ y.Store(0,t.x); }\n",
    "control/uninitialized.hlsl": ("ByteAddressBuffer x:register(t0); RWByteAddressBuffer y:register(u0);\ngroupshared float s[64];\n"
                                   "[numthreads(64,1,1)] void main(uint lane:SV_GroupIndex){ float v=s[lane]; y.Store(lane*4,x.Load(lane*4)+asuint(v)); }\n"),
}


def recorded_positions(prefix=4, grid=(1, 126, 96), merge=2, suffix=589):
    """Three-axis prompt positions of the recorded request, in get_rope_index order: 4 text rows, the 63x48
    merged image block at temporal 4, then 589 text rows from 4 + max(63, 48)."""
    t, h, w = [], [], []
    for i in range(prefix):
        t.append(i), h.append(i), w.append(i)
    gh, gw = grid[1] // merge, grid[2] // merge
    for r in range(gh):
        for c in range(gw):
            t.append(prefix), h.append(prefix + r), w.append(prefix + c)
    start = prefix + max(gh, gw)
    for i in range(suffix):
        t.append(start + i), h.append(start + i), w.append(start + i)
    return struct.pack(f"<{3 * len(t)}q", *(t + h + w))


def original_softmax():
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{SOURCE_COMMIT}:ChandraNative/shaders/runtime/text_attention_softmax.hlsl"],
                           capture_output=True, timeout=60)
    if shown.returncode:
        raise unittest.SkipTest(f"source commit {SOURCE_COMMIT} unavailable")
    if hashlib.sha256(shown.stdout).hexdigest() != ORIGINAL_SOFTMAX_SHA256:
        raise AssertionError("2c44182 text_attention_softmax.hlsl bytes differ from the recorded hash")
    return shown.stdout


TARGETS = {"shader-test": ["shader_emulation_test.cpp", "operators.cpp"],
           "text-state-test": ["text_state_test.cpp", "operators.cpp", "text_model.cpp"],
           "recorded-test": ["recorded_geometry_test.cpp", "operators.cpp", "text_model.cpp", "vision_model.cpp", "diagnostics.cpp"],
           "regression-emulation-test": ["text_softmax_regression_emulation_test.cpp", "operators.cpp", "text_model.cpp"]}


def build_all(root, targets=tuple(TARGETS)):
    """Generate the translated shader header and build the selected drivers in `root`."""
    (root / "original").mkdir()
    (root / "control").mkdir()
    (root / "original/text_attention_softmax.hlsl").write_bytes(original_softmax())
    for name, text in CONTROLS.items():
        (root / name).write_text(text)
    shaders = [(f"runtime/{p.name}", p) for p in sorted(SHADERS.glob("*.hlsl")) if p.name != "vision_common.hlsl"]
    extra = [(name, root / name) for name in ["original/text_attention_softmax.hlsl", *CONTROLS]]
    extra.append(("control/text_attention_softmax_2c44182.hlsl", CONTROL_SHADER))
    digests = hlsl_cpu.generate(shaders, root / "shaders.generated.h", extra)
    flags = compiler_flags(compiler_family(COMPILER))
    binaries = {}
    for name in targets:
        sources = TARGETS[name]
        result = subprocess.run([COMPILER, *flags, "-I", str(root), "-I", str(RUNTIME), "-o", str(root / name),
                                 *[str(RUNTIME / s) for s in sources]], capture_output=True, text=True, timeout=900)
        if result.returncode or "warning" in result.stderr:
            raise AssertionError(f"{name} build failed or warned:\n{result.stderr[-4000:]}")
        binaries[name] = root / name
    return digests, binaries


def run(binary, *args, selector=None, timeout=900):
    env = {k: v for k, v in os.environ.items() if k != "CHANDRA_EXPERIMENTAL_GEMV_B1"}
    if selector:
        env["CHANDRA_EXPERIMENTAL_GEMV_B1"] = selector
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode:
        raise AssertionError(f"{binary.name} {' '.join(map(str, args))} failed:\n{result.stderr[-4000:]}\n{result.stdout[-2000:]}")
    return json.loads(result.stdout)


@unittest.skipUnless(COMPILER and os.name == "posix", "host C++17 compiler required")
class CompilerClassificationTests(unittest.TestCase):
    def test_family_comes_from_predefined_macros(self):
        # Review defect: a "g++" substring test also matched "clang++" and gave Clang the GCC-only option.
        self.assertNotIn("-Wno-dangling-reference", compiler_flags("clang"))
        self.assertNotIn("-Wno-dangling-reference", compiler_flags("other"))
        self.assertIn("-Wno-dangling-reference", compiler_flags("gcc"))
        for name, family in (("g++", "gcc"), ("clang++", "clang")):
            path = shutil.which(name)
            if path:
                self.assertEqual(compiler_family(path), family, name)
        self.assertIn(compiler_family(COMPILER), ("gcc", "clang"))


@unittest.skipUnless(COMPILER and os.name == "posix", "host C++17 compiler and POSIX fibers required")
class ExactInputCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="chandra-exact-input-")
        cls.root = Path(cls.temporary.name)
        cls.digests, cls.binaries = build_all(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_shader_hashes(self):
        self.assertEqual(self.digests["runtime/text_attention_softmax.hlsl"], REPAIRED_SOFTMAX_SHA256)
        self.assertEqual(self.digests["original/text_attention_softmax.hlsl"], ORIGINAL_SOFTMAX_SHA256)
        text = (SHADERS / "text_attention_softmax.hlsl").read_text()
        self.assertIn("m=tmp[0];GroupMemoryBarrierWithGroupSync();float total=0;", text)
        # The GPU driver's control file is byte-identical to the shader at the pinned source commit.
        self.assertEqual(self.digests["control/text_attention_softmax_2c44182.hlsl"], ORIGINAL_SOFTMAX_SHA256)
        self.assertEqual(CONTROL_SHADER.read_bytes(), original_softmax())

    def test_text_softmax_race_and_repair(self):
        cases = run(self.binaries["shader-test"], "softmax")["cases"]
        self.assertEqual(len(cases), 27)
        for case in cases:
            self.assertEqual(case["repaired_findings"], 0)
            self.assertLessEqual(case["repaired_max_rel"], 1.01 / 256)
            self.assertTrue(set(case["original_findings"]) <= {"groupshared_race_write_read", "groupshared_race_read_write"})
            self.assertGreater(sum(case["original_findings"].values()), 0)
        wrong = [c for c in cases if c["order"] == "ascending" and c["count"] > 1]
        self.assertEqual(len(wrong), 8)
        for case in wrong:
            self.assertGreater(case["original_words_differing_from_repaired"], 0)
            self.assertGreater(case["original_max_rel"], 0.5)
        recorded = [c for c in cases if (c["base"], c["rows"]) == (3584, 33)]
        self.assertEqual({c["count"] for c in recorded}, {3617})

    def test_negative_and_detector_controls(self):
        cases = run(self.binaries["shader-test"], "controls")["cases"]
        controls = [c for c in cases if c["case"] == "negative_control"]
        self.assertEqual({c["shader"] for c in controls}, {
            "runtime/vision_attention_softmax.hlsl", "runtime/vision_attention_scores.hlsl", "runtime/norm.hlsl",
            "runtime/text_gdn_gate.hlsl", "runtime/text_gdn_prepare.hlsl"})
        self.assertTrue(all(c["findings"] == 0 for c in controls))
        detector = next(c for c in cases if c["case"] == "detector_controls")
        self.assertGreater(detector["uav_race"].get("uav_race_write_write", 0), 0)
        self.assertEqual(detector["uninitialized"], {"groupshared_uninitialized_read": 64, "uninitialized_read": 64})

    def test_fast_linear_replica_equals_emulated_hlsl(self):
        for selector, route in ((None, False), ("ordered", True)):
            case = run(self.binaries["shader-test"], "linear", selector=selector)["cases"][0]
            self.assertEqual((case["gemv_route_for_rows_1"], case["compared_words"]), (route, 78340))

    def test_recorded_geometry_trace_and_phase_bytes(self):
        positions = recorded_positions()
        self.assertEqual(hashlib.sha256(positions).hexdigest(), RECORDED_POSITIONS_SHA256)
        path = self.root / "positions.i64"
        path.write_bytes(positions)
        output = self.root / "recorded"
        output.mkdir()
        report = run(self.binaries["recorded-test"], path, output)
        self.assertEqual(report["failures"], 0)
        phases = [(p["name"], p["tracked_live"], p["tracked_peak"]) for p in report["phases"]]
        self.assertEqual(phases[0], ("resident_model", RESIDENT_MODEL_BYTES, RESIDENT_MODEL_BYTES))
        self.assertEqual(phases[1:], FAILED_RUN_PHASES)
        vision, text = report["vision_trace"], report["text_trace"]
        self.assertEqual((vision["query_tiles_per_block"], vision["key_tiles"], vision["merger_tile_first_rows"]),
                         (378, 95, [0, 512, 1024, 1536, 2048, 2560]))
        self.assertLessEqual(max(vision["maximum_group_dimension"], text["maximum_group_dimension"]), 65535)
        self.assertEqual((text["prefill_tiles"], text["last_tile_rows"], text["attention_calls"], text["delta_calls"]), (57, 33, 8 * 59, 24 * 59))
        self.assertEqual((text["rope_coordinates_checked"], text["cache_tokens_after_decode"]), (8 * 3 * 3619, 3619))
        self.assertEqual(report["merge_emulation"]["mismatches"], 0)
        self.assertEqual(report["vision_emulation"]["findings"], 0)
        sizes = {p.name: p.stat().st_size for p in output.iterdir()}
        self.assertEqual(sizes, {"vision_positions.u32": 12096 * 16, "rotary_cos_sin.f32": 12096 * 512,
                                 "position_embedding.f32": 12096 * 4096, "pos_embed_table.bf16x2": 2304 * 2048})

    def test_recorded_plan_dry_run_executes_and_compares(self):
        # The root-owned diagnostic plan, executed by the unchanged recorder over the stubbed recorded-geometry replay
        # with the 14-token cap: every planned record is written, the strict loader accepts it and two runs compare
        # exactly. Payloads are stub zeros, so this checks the plan/recorder/comparator path, not numerics.
        self.assertEqual(hashlib.sha256(PLAN.read_bytes()).hexdigest(), PLAN_FILE_SHA256)
        path = self.root / "positions.i64"
        path.write_bytes(recorded_positions())
        dumps = []
        for name in ("dry-a", "dry-b"):
            report = run(self.binaries["recorded-test"], path, self.root, PLAN, self.root / name, 14)
            summary = report["dry_run"]
            self.assertEqual((summary["plan_sha256"], summary["planned_records"], summary["written_records"], summary["dump_complete"]),
                             (RESOLVED_PLAN_SHA256, 1023, 1023, True))
            self.assertEqual((summary["consumed_rows"], summary["payload_bytes"], summary["readback_bytes"]), (3617 + 13, 24387584, 1552353280))
            dumps.append(cd.load_dump(self.root / name))
        forecast = dumps[0]["header"]["plan"]["forecast"]
        self.assertEqual((forecast["records"], forecast["readback_bytes"]), (1023, 1552353280))
        self.assertLessEqual(forecast["readback_bytes"], 8 << 30)
        self.assertEqual(forecast["largest_readback_bytes"], 12096 * 1024 * 4)  # One complete vision hidden buffer, below 128 MiB.
        stages = {}
        for record in dumps[0]["records"].values():
            stages[record["stage"]] = stages.get(record["stage"], 0) + 1
        self.assertEqual(stages, {"vision.patch_embedding": 7, "vision.position_added": 7, "vision.block_output": 24 * 7, "vision.merger_output": 7,
                                  "text.merged_embedding": 11, "text.layer_output": 32 * 11 + 32 * 13, "text.final_norm": 11 + 13,
                                  "text.logits": 1 + 13, "text.gdn_conv_state": 5, "text.gdn_recurrent_state": 2 * 5 + 2})
        report = cd.compare(dumps[0], dumps[1], {"*": {"max_abs": 0.0, "require_argmax_match": True}})
        self.assertEqual((report["verdict"], report["counts"], report["conditioning"]["common_rows_identical"]),
                         ("within_tolerance", {"within_tolerance": 1023}, True))
        self.assertEqual((report["cpu_oracle_agreement"], report["native_numerically_qualified"]), (False, False))

    def text_state(self, part, selector):
        report = run(self.binaries["text-state-test"], part, selector=selector, timeout=1800)
        self.assertEqual(report["failures"], 0)
        return report["cases"]

    def check_layer(self, cases, layer):
        chunks = [c for c in cases if c["case"] == "chunk_invariance"]
        self.assertEqual([c["calls"] for c in chunks], ["64,64,12", "12,64,64", "35,35,35,35", "1,63,64,12", "140x1"])
        self.assertTrue(all(c["differing_output_words"] == 0 and c["differing_state_words"] == 0 and c["findings"] == 0 for c in chunks))
        initial = next(c for c in cases if c["case"] == "initial_state_sensitivity")
        unzeroed = next(c for c in cases if c["case"] == "unzeroed_cache")
        isolation = next(c for c in cases if c["case"] == "request_isolation")
        self.assertEqual((isolation["differing_a"], isolation["differing_b"]), (0, 0))
        if layer % 4 == 3:
            self.assertEqual((initial["differing_output_words"], unzeroed["uninitialized_reads"]), (0, 0))
            position = next(c for c in cases if c["case"] == "position_sensitivity")
            self.assertEqual(position["differing_before_row_100"], 0)
            self.assertGreater(position["differing_output_words"], 0)
        else:
            self.assertGreater(initial["differing_output_words"], 0)
            self.assertGreater(unzeroed["uninitialized_reads"], 0)

    def test_gdn_layer_state_controls_on_both_decode_routes(self):
        for selector in (None, "ordered"):
            self.check_layer(self.text_state("gdn", selector), 0)

    def test_attention_layer_state_controls_on_both_decode_routes(self):
        for selector in (None, "ordered"):
            self.check_layer(self.text_state("attention", selector), 3)

    def test_attention_layer_schedule_determinism(self):
        cases = {c["softmax"]: c for c in self.text_state("schedule", "ordered")}
        repaired, original = cases["repaired"], cases["original_2c44182"]
        self.assertEqual((repaired["race_findings"], repaired["ascending_vs_descending_differing_words"], repaired["shuffled_vs_descending_differing_words"]), (0, 0, 0))
        self.assertGreater(original["race_findings"], 0)
        self.assertGreater(original["ascending_vs_descending_differing_words"] + original["shuffled_vs_descending_differing_words"], 0)

    def test_regression_driver_on_translated_hlsl(self):
        report = run(self.binaries["regression-emulation-test"], timeout=1800)
        self.assertEqual((report["failures"], report["report"]["plan_sha256"]), (0, REGRESSION_PLAN_SHA256))
        transcript, numerics = report["report"]["transcript"], report["report"]["numerics"]
        self.assertEqual(len(transcript), 17)
        self.assertTrue(all(row["equal_to_regression"] for row in transcript))
        tail = next(row for row in transcript if row["case"] == "recorded_tail_c3617")
        self.assertEqual((tail["production_cbuffer"], tail["production_groups"], tail["production_score_words"]),
                         ([33, 16, 0, 0, 3584, 3617, 0, 0], [528, 1, 1], 528 * 3617))
        self.assertEqual(len(numerics), 17 * 6)
        for row in numerics:
            repaired = row["shader"] == "runtime/text_attention_softmax.hlsl"
            if repaired:
                self.assertEqual((row["oracle_passed"], row["findings"], row["equal_to_repaired_ascending"]), (True, 0, True), row)
            else:
                self.assertGreater(row["race_findings"], 0)
            if not repaired and row["order"] == "ascending":
                self.assertEqual(row["oracle_passed"], row["count"] < 2, row)
            if not repaired and row["order"] == "descending":
                self.assertEqual((row["oracle_passed"], row["equal_to_repaired_ascending"]), (True, True), row)
        rejected = {row["case"] for row in numerics if not row["oracle_passed"]}
        self.assertEqual(len(rejected), 16)  # Every case but the single-key one, under the ascending control schedule.

    def test_translator_refuses_unsupported_constructs(self):
        base = "RWByteAddressBuffer y:register(u0);\n[numthreads(1,1,1)] void main(uint3 t:SV_DispatchThreadID){ %s }\n"
        for body in ("InterlockedAdd(t.x,1,t.y);", "AllMemoryBarrierWithGroupSync();", "double d=1;"):
            with self.assertRaises(ValueError):
                hlsl_cpu.translate(base % body, "S", "control/refused.hlsl")
        with self.assertRaises(ValueError):
            hlsl_cpu.translate("RWByteAddressBuffer y:register(t0);\n[numthreads(1,1,1)] void main(uint i:SV_GroupIndex){}\n", "S", "x")


if __name__ == "__main__":
    unittest.main()
