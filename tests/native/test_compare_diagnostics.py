"""Standard-library checks of diagnostic dump authentication, refusal and numerical comparison.

Small synthetic dumps only: no model, device, Torch or network. Values are chosen so every metric has
an independently hand-computed expectation.
"""
from array import array
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compare_diagnostics", ROOT / "scripts/native/compare_diagnostics.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

COMMITMENTS = {"model_revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09", "input_manifest_sha256": "1" * 64}
MODEL = {"model_sha256": "2" * 64, "config_sha256": "3" * 64, "model_bytes": 10591220088}
NATIVE = {"kind": "directcompute_native", "executable_sha256": "4" * 64}
ORACLE = {"kind": "cpu_fp64_oracle", "source": "independent reference"}
PROMPT = 801


def history_for(records, overrides=None):
    """Consumed rows [token_id, t, h, w] that agree with the records: prompt rows default to token 11 and
    position [row]*3, generated rows to token 7; overrides {absolute_row: row} model a different history."""
    rows = [[11, r, r, r] for r in range(PROMPT)]

    def extend(n):
        while len(rows) < n:
            rows.append([7] + [len(rows)] * 3)

    for record in records:
        coordinate = record["coordinates"][0]
        if record["stage"] in cd.STATE_STAGES:
            extend(record["cache_length"])
            rows[record["cache_length"] - 1][1:] = coordinate["last_position"]
        elif record["stage"] in cd.ROW_STAGES:
            extend(coordinate["absolute_row"] + 1)
            rows[coordinate["absolute_row"]] = [coordinate["token_id"]] + coordinate["position"]
    for row, value in (overrides or {}).items():
        extend(row + 1)
        rows[row] = value
    return rows


def hidden(values, row=2, layer=5, token=11, **extra):
    record = {"stage": "text.layer_output", "phase": "prefill", "layer": layer, "logical_shape": [801, len(values)],
              "selected_rows": [row], "payload_shape": [1, len(values)], "complete_tensor": False,
              "coordinates": [{"absolute_row": row, "token_id": token, "position": [row, row, row], "chunk_index": row // 64}],
              "cache": {"cache_length_after": 64}, "values": values}
    record.update(extra)
    return record


def logits(values, step=None, consumed=None):
    absolute = 800 if step is None else 801 + step
    record = {"stage": "text.logits", "phase": "decode" if step is not None else "prefill", "decode_step": step,
              "logical_shape": [1, cd.VOCABULARY], "selected_rows": [0], "payload_shape": [1, cd.VOCABULARY],
              "complete_tensor": True, "produces_generated_index": 0 if step is None else step + 1,
              "coordinates": [{"absolute_row": absolute, "token_id": consumed or 7, "position": [absolute] * 3}],
              "values": values}
    if step is not None:
        record["decode"] = {"step": step, "consumed_generated_index": step, "consumed_token_id": consumed,
                            "logits_produce_generated_index": step + 1}
    return record


def vocabulary(spikes):
    row = array("f", bytes(4 * cd.VOCABULARY))
    for index, value in spikes.items():
        row[index] = value
    return row


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chandra-diagnostic-compare-")
        self.root = Path(self.temporary.name)
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def dump(self, records, producer=NATIVE, commitments=COMMITMENTS, model=MODEL, run_completed=True, rows=None, **options):
        self.count += 1
        return cd.write_dump(self.root / f"dump{self.count}", commitments, model, producer, records, run_completed=run_completed,
                             conditioning=history_for(records) if rows is None else rows, prompt_rows=PROMPT, **options)

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cd.main([str(a) for a in args])
        return code, json.loads(out.getvalue())

    def progress(self, directory):
        return [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]

    def rewrite(self, directory, lines):
        path = directory / "progress.jsonl"
        path.chmod(0o600)
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    def test_hand_computed_metrics_and_bf16_ulps(self):
        candidate = self.dump([hidden([1.0, 2.0, 3.0, 4.0]), hidden([1.0, -1.0, 0.0, 0.0], row=3)])
        reference = self.dump([hidden([1.0, 2.0, 3.0, 5.0]), hidden([1.0078125, -1.0078125, -0.0, 0.0], row=3)], producer=ORACLE)
        report = cd.compare(cd.load_dump(candidate), cd.load_dump(reference), {})
        first, second = (row["metrics"] for row in report["records"])
        self.assertEqual((first["max_abs"], first["max_abs_index"], first["exact_equal"]), (1.0, 3, 3))
        self.assertAlmostEqual(first["max_rel"], 0.2)
        self.assertAlmostEqual(first["rmse"], 0.5)
        self.assertAlmostEqual(first["cosine"], 34 / math.sqrt(30 * 39))
        # BF16 4.0 is 0x4080 and 5.0 is 0x40a0: 32 representable steps apart.
        self.assertEqual(first["bf16"], {"applicable": True, "max_ulp": 32, "exact": 3, "one_ulp": 0, "two_ulp": 0, "over_two_ulp": 1})
        # +/-1.0078125 are one BF16 step from +/-1; signed zeros compare equal.
        self.assertEqual((second["bf16"]["max_ulp"], second["bf16"]["one_ulp"], second["exact_equal"]), (1, 2, 2))
        self.assertEqual(second["zero_reference_elements"], 2)
        self.assertEqual(report["verdict"], "no_verdict")
        self.assertTrue(report["reference_independent"])
        self.assertFalse(report["cpu_oracle_agreement"])
        self.assertFalse(report["amd_ocr_qualified"])

    def test_explicit_tolerances_including_zero(self):
        candidate = self.dump([hidden([1.0, 2.0, 3.0, 4.0])])
        reference = self.dump([hidden([1.0, 2.0, 3.0, 4.03125])], producer=ORACLE)  # One BF16 step at 4.0.
        self.assertEqual(self.run_cli(candidate, reference)[0], 3)
        code, report = self.run_cli(candidate, reference, "--max-abs", "0")
        self.assertEqual((code, report["records"][0]["failed_criteria"]), (1, ["max_abs"]))
        code, report = self.run_cli(candidate, reference, "--max-bf16-ulp", "1", "--min-cosine", "0.9999")
        self.assertEqual((code, report["verdict"], report["cpu_oracle_agreement"]), (0, "within_tolerance", True))
        self.assertEqual(self.run_cli(candidate, reference, "--max-bf16-ulp", "0")[0], 1)
        code, report = self.run_cli(candidate, reference, "--max-abs", "0", "--stage-tolerance", "text.layer_output:max_abs=0.05")
        self.assertEqual(code, 0)
        self.assertEqual(self.run_cli(candidate, reference, "--max-abs", "nan")[0], 2)

    def test_relative_error_and_bf16_applicability(self):
        candidate = self.dump([hidden([0.5, 1.0]), hidden([0.1, 0.0], row=4, bf16_rounding=cd.STATE_POLICY)])
        reference = self.dump([hidden([0.0, 1.0]), hidden([0.1, 0.0], row=4, bf16_rounding=cd.STATE_POLICY, storage_dtype="float64")], producer=ORACLE)
        code, report = self.run_cli(candidate, reference, "--max-rel", "0.5", "--max-bf16-ulp", "0")
        first, second = report["records"]
        self.assertEqual(first["metrics"]["zero_reference_nonzero_candidate"], 1)
        self.assertEqual(first["failed_criteria"], ["max_rel", "max_bf16_ulp"])
        # FP32 state is exempt from BF16 ULP; float32(0.1) differs from float64 0.1 only by its FP32 rounding.
        self.assertFalse(second["metrics"]["bf16"]["applicable"])
        self.assertAlmostEqual(second["metrics"]["max_abs"], abs(array("f", [0.1])[0] - 0.1))
        self.assertEqual((second["status"], second["failed_criteria"]), ("within_tolerance", []))
        self.assertEqual(code, 1)
        # A declared BF16 boundary that is not BF16 representable fails a BF16 criterion instead of being skipped,
        # and fails as an invalid candidate whatever the tolerance; an invalid reference is incomparable.
        bad = self.dump([hidden([0.1, 1.0])])
        report = self.run_cli(bad, self.dump([hidden([0.10009765625, 1.0])], producer=ORACLE), "--max-bf16-ulp", "4")[1]
        self.assertEqual(report["records"][0]["failed_criteria"], ["max_bf16_ulp"])
        self.assertEqual((report["records"][0]["status"], report["verdict"]), ("invalid_candidate", "invalid_candidate"))
        self.assertEqual(self.run_cli(bad, bad, "--max-bf16-ulp", "4")[1]["records"][0]["status"], "incomparable")

    def test_complete_vocabulary_top_tokens_and_ties(self):
        candidate = self.dump([logits(vocabulary({7: 10.0, 3: 9.0})), logits(vocabulary({5: 2.0, 9: 2.0}), step=0, consumed=7)])
        reference = self.dump([logits(vocabulary({7: 10.0, 3: 10.5})), logits(vocabulary({5: 2.0, 9: 2.0}), step=0, consumed=7)], producer=ORACLE)
        code, report = self.run_cli(candidate, reference, "--require-argmax-match")
        first, second = (row["metrics"]["top_tokens"] for row in report["records"])
        self.assertEqual((first["candidate_top"], first["reference_top"], first["argmax_agree"]), (7, 3, False))
        self.assertEqual((first["candidate_margin"], first["reference_margin"]), (1.0, 0.5))
        self.assertEqual((first["candidate_rank_of_reference_top"], first["reference_rank_of_candidate_top"]), (1, 1))
        self.assertFalse(first["reference_margin_exceeds_twice_max_abs_error"])
        self.assertEqual((second["candidate_top"], second["candidate_max_ties"], second["candidate_margin"]), (5, 2, 0.0))
        self.assertEqual(report["records"][0]["failed_criteria"], ["require_argmax_match"])
        self.assertEqual(report["records"][1]["status"], "within_tolerance")
        self.assertEqual(code, 1)

    def test_incompatible_commitments_are_refused(self):
        candidate = self.dump([hidden([1.0])])
        for change in ({"commitments": dict(COMMITMENTS, input_manifest_sha256="9" * 64)},
                       {"model": dict(MODEL, model_sha256="8" * 64)},
                       {"commitments": dict(COMMITMENTS, model_revision="0" * 40)}):
            reference = self.dump([hidden([1.0])], producer=ORACLE, **change)
            code, report = self.run_cli(candidate, reference, "--max-abs", "1")
            self.assertEqual((code, report["verdict"]), (2, "refused"))
            self.assertIn("Incompatible", report["error"])

    def test_incomparable_coordinates_and_missing_reference(self):
        candidate = self.dump([logits(vocabulary({1: 1.0}), step=2, consumed=1944), hidden([1.0], row=9, token=4), hidden([1.0], row=10)])
        reference = self.dump([logits(vocabulary({1: 1.0}), step=2, consumed=420), hidden([1.0], row=9, token=5)], producer=ORACLE)
        code, report = self.run_cli(candidate, reference, "--max-abs", "1")
        self.assertEqual((code, report["verdict"]), (1, "incomparable"))
        reasons = [row.get("reason") for row in report["records"]]
        self.assertEqual(reasons, ["decode differs", "coordinate 0 token_id differs", "reference record absent"])
        # Producer tiling metadata is not identity: a reference without chunk fields still pairs.
        plain = self.dump([hidden([1.0], row=10, coordinates=[{"absolute_row": 10, "token_id": 11, "position": [10, 10, 10]}])], producer=ORACLE)
        self.assertEqual(self.run_cli(self.dump([hidden([1.0], row=10)]), plain, "--max-abs", "0")[0], 0)

    def test_corrupt_or_incomplete_payloads_are_refused(self):
        def refused(directory, fragment, allow=False):
            with self.assertRaises(cd.Refusal) as caught:
                cd.load_dump(directory, allow)
            self.assertIn(fragment, str(caught.exception))

        changed = self.dump([hidden([1.0, 2.0])])
        payload = changed / self.progress(changed)[3]["file"]  # Lines: plan, model, conditioning, start, record, finished.
        payload.chmod(0o600)
        payload.write_bytes(array("f", [1.0, 3.0]).tobytes())
        refused(changed, "SHA-256 differs")
        payload.write_bytes(array("f", [1.0]).tobytes())
        refused(changed, "size differs")
        payload.unlink()
        refused(changed, "missing")
        target = self.root / "outside.f32"
        target.write_bytes(array("f", [1.0, 2.0]).tobytes())
        payload.symlink_to(target)
        refused(changed, "regular non-symlink")

        for mutate, fragment in (
            (lambda lines: [line.update(file="../escape.f32") for line in lines[3:5]], "unsafe payload filename"),
            (lambda lines: lines[4].update(payload_shape=[2, 2]), "payload shape"),
            (lambda lines: lines[4].update(selected_rows=[801]), "selected rows"),
            (lambda lines: lines[4].update(qualified=True), "cannot claim qualification"),
            (lambda lines: lines[4]["commitments"].update(input_manifest_sha256="7" * 64), "commitments differ"),
            (lambda lines: lines[4].update(sequence=4), "matching start line"),
            (lambda lines: lines.insert(5, {"event": "surprise"}), "unknown progress event"),
            (lambda lines: lines[5].update(written_records=2), "record count"),
        ):
            directory = self.dump([hidden([1.0, 2.0])])
            lines = self.progress(directory)
            mutate(lines)
            self.rewrite(directory, lines)
            refused(directory, fragment)

        duplicate = self.dump([hidden([1.0])])
        text = (duplicate / "progress.jsonl").read_text().replace('"event": "finished"', '"event": "finished", "event": "finished"')
        (duplicate / "progress.jsonl").chmod(0o600)
        (duplicate / "progress.jsonl").write_text(text)
        refused(duplicate, "duplicate JSON key")
        nan = self.dump([hidden([1.0])])
        (nan / "progress.jsonl").chmod(0o600)
        (nan / "progress.jsonl").write_text((nan / "progress.jsonl").read_text().replace('"model_bytes": 10591220088', '"model_bytes": NaN', 1))
        refused(nan, "nonfinite JSON")
        orphan = self.dump([hidden([1.0])])
        (orphan / "99999.extra.f32").write_bytes(b"0000")
        refused(orphan, "unreferenced files")

    def test_unfinished_and_failed_dumps_stay_unqualified(self):
        unfinished = self.dump([hidden([1.0]), hidden([2.0], row=3)])
        lines = self.progress(unfinished)
        # Simulate a run retired after record 0 completed and record 1 started: no record or finished line follows.
        self.rewrite(unfinished, lines[:6])
        with self.assertRaises(cd.Refusal):
            cd.load_dump(unfinished)
        loaded = cd.load_dump(unfinished, allow_incomplete=True)
        self.assertEqual((loaded["run_status"], loaded["dump_complete"], len(loaded["records"])), ("unfinished", False, 1))
        self.assertEqual(loaded["incomplete_record"], lines[5]["key"])
        self.assertEqual(loaded["orphans"], [lines[5]["file"]])
        reference = self.dump([hidden([1.0]), hidden([2.0], row=3)], producer=ORACLE)
        self.assertEqual(self.run_cli(unfinished, reference, "--max-abs", "0")[0], 2)
        code, report = self.run_cli(unfinished, reference, "--max-abs", "0", "--allow-incomplete")
        self.assertEqual((code, report["verdict"], report["cpu_oracle_agreement"]), (3, "within_tolerance_incomplete_dump", False))
        failed = self.dump([hidden([1.0])], run_completed=False)
        self.assertEqual(cd.load_dump(failed, allow_incomplete=True)["run_status"], "failed")
        with self.assertRaises(cd.Refusal):
            cd.load_dump(failed)

    def test_report_output_is_fresh_and_native_reference_is_not_an_oracle(self):
        candidate = self.dump([hidden([1.0])])
        twin = self.dump([hidden([1.0])])
        output = self.root / "report.json"
        code, report = self.run_cli(candidate, twin, "--max-abs", "0", "--output", output)
        self.assertEqual((code, report["reference_kind"], report["cpu_oracle_agreement"]), (0, "directcompute_native", False))
        self.assertFalse(report["reference_independent"])
        self.assertEqual(json.loads(output.read_text())["verdict"], "within_tolerance")
        self.assertEqual(self.run_cli(candidate, twin, "--max-abs", "0", "--output", output)[0], 2)
        self.assertEqual(oct(os.stat(candidate / "progress.jsonl").st_mode & 0o777), "0o600")



def review_hidden(step=None, token=7, cache_length=64, value=1.0, stage="text.layer_output"):
    """The finished review's synthetic hidden record, in the v2 record format."""
    absolute = 0 if step is None else PROMPT + step
    record = {"stage": stage, "layer": 0 if stage == "text.layer_output" else None,
              "phase": "prefill" if step is None else "decode", "decode_step": step,
              "logical_shape": [PROMPT if step is None else 1, 1], "selected_rows": [0], "payload_shape": [1, 1],
              "complete_tensor": step is not None,
              "coordinates": [{"absolute_row": absolute, "token_id": token, "position": [absolute] * 3}],
              "cache": {"cache_length_after": cache_length}, "values": [value]}
    if step is not None:
        record["decode"] = {"step": step, "consumed_generated_index": step, "consumed_token_id": token,
                            "logits_produce_generated_index": step + 1}
    return record


def review_state(first, length=130, stage="text.gdn_recurrent_state", value=1.0):
    """Carried GDN state after `length` consumed rows; `first` is the producing call's first row (provenance)."""
    decode = length > PROMPT
    record = {"stage": stage, "layer": 0, "phase": "decode" if decode else "prefill",
              "decode_step": length - PROMPT - 1 if decode else None, "cache_length": length,
              "bf16_rounding": cd.STATE_POLICY, "coordinates": [{"last_absolute_row": length - 1, "last_position": [127, 127, 127]}],
              "cache": {"cache_length_after": length, "request_tokens_before_call": 0 if first < PROMPT else first,
                        "call_first_row": first if first < PROMPT else 0, "call_row_count": length - first}}
    if stage == "text.gdn_conv_state":
        record.update(logical_shape=[8192, 4], selected_rows="all", payload_shape=[8192, 4], complete_tensor=True, values=[value] * 32768)
    else:
        record.update(logical_shape=[32, 128, 128], selected_rows=[0], payload_shape=[1, 128, 128], complete_tensor=False, values=[value] * 16384)
    if decode:
        step = length - PROMPT - 1
        record["decode"] = {"step": step, "consumed_generated_index": step, "consumed_token_id": 7,
                            "logits_produce_generated_index": step + 1}
    return record


class ReviewRegressionTests(unittest.TestCase):
    """Regression expectations for the finished review's six counterexamples and its B1-B4 repair criteria."""
    setUp, tearDown, dump, run_cli, progress, rewrite = (ComparatorTests.setUp, ComparatorTests.tearDown, ComparatorTests.dump,
                                                         ComparatorTests.run_cli, ComparatorTests.progress, ComparatorTests.rewrite)

    def compare(self, candidate, reference, tolerances=None, c_rows=None, r_rows=None, allow=False, **options):
        c = self.dump(candidate, rows=c_rows, **options.get("candidate", {}))
        r = self.dump(reference, producer=ORACLE, rows=r_rows, **options.get("reference", {}))
        return cd.compare(cd.load_dump(c, allow), cd.load_dump(r, allow), {"*": {"max_abs": 0}} if tolerances is None else tolerances)

    def statuses(self, report):
        return [(row["status"], row.get("reason")) for row in report["records"]]

    def mutated(self, records, mutate, rows=None):
        directory = self.dump(records, rows=rows)
        lines = self.progress(directory)
        mutate(lines)
        self.rewrite(directory, lines)
        return directory

    def refused(self, directory, fragment, allow=True):
        with self.assertRaises(cd.Refusal) as caught:
            cd.load_dump(directory, allow)
        self.assertIn(fragment, str(caught.exception))

    # B1: every decode record is bound to its complete consumed prefix.
    def test_b1_divergent_history_then_same_current_token(self):
        report = self.compare([review_hidden(0, 5, 802), review_hidden(1, 7, 803)],
                              [review_hidden(0, 6, 802), review_hidden(1, 7, 803)])
        self.assertEqual(self.statuses(report), [("incomparable", "decode differs"), ("incomparable", "consumed prefix differs")])
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("incomparable", False))
        self.assertEqual((report["conditioning"]["first_divergent_row"], report["conditioning"]["first_divergent_generated_index"]), (801, 0))

    def test_b1_sparse_decode_commits_to_unselected_history(self):
        # Only step 1 is captured; the dumps still carry the unselected step-0 rows they actually consumed.
        sparse = [review_hidden(1, 7, 803), review_hidden(1, 7, 803, stage="text.final_norm")]
        report = self.compare(sparse, sparse, c_rows=history_for(sparse, {801: [5, 801, 801, 801]}),
                              r_rows=history_for(sparse, {801: [6, 801, 801, 801]}))
        self.assertEqual(self.statuses(report), [("incomparable", "consumed prefix differs")] * 2)
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("incomparable", False))
        self.assertEqual(report["conditioning"]["first_divergent_generated_index"], 0)
        # Identical complete histories make the same sparse records comparable.
        same = history_for(sparse, {801: [5, 801, 801, 801]})
        report = self.compare(sparse, sparse, c_rows=same, r_rows=same)
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"], report["conditioning"]["common_rows_identical"]),
                         ("within_tolerance", True, True))
        # A later divergence leaves earlier records comparable: causality, not whole-run equality.
        later = history_for(sparse, {801: [5, 801, 801, 801], 803: [9, 803, 803, 803]})
        self.assertEqual(self.compare(sparse, sparse, c_rows=same, r_rows=later)["verdict"], "within_tolerance")
        # A writer cannot commit a record to rows it never consumed.
        with self.assertRaisesRegex(ValueError, "not covered"):
            self.dump(sparse, rows=history_for([review_hidden(0, 5, 802)]))
        with self.assertRaisesRegex(ValueError, "consumed conditioning rows"):
            cd.write_dump(self.root / "no-history", COMMITMENTS, MODEL, ORACLE, sparse)

    def test_b1_absent_or_contradictory_prefix_commitments_are_refused(self):
        records = [review_hidden(0, 5, 802), review_hidden(1, 7, 803)]
        rows = history_for(records)
        digests = cd.History(COMMITMENTS["input_manifest_sha256"], PROMPT)
        for row in rows:
            digests.append(row)
        # Lines: plan, model, conditioning, start 0, record 0, start 1, record 1, finished.
        for mutate, fragment in (
            (lambda lines: lines[6].update(conditioning=None), "lacks its consumed-prefix commitment"),
            (lambda lines: lines[6]["conditioning"].update(prefix_sha256="0" * 64), "consumed-prefix SHA-256 differs"),
            (lambda lines: lines[6]["conditioning"].update(prefix_rows=802, prefix_sha256=digests.digests[802]), "phase or decode step differs"),
            (lambda lines: lines[6]["conditioning"].update(prefix_rows=900), "not covered"),
            (lambda lines: lines[2]["rows"][801].__setitem__(0, 6), "conditioning prefix SHA-256 differs from its rows"),
            (lambda lines: lines[2].update(first_row=1), "contiguously"),
            (lambda lines: lines.insert(7, lines.pop(2)), "not covered"),
            (lambda lines: lines[6]["decode"].update(consumed_token_id=8), "decode metadata differs"),
            (lambda lines: lines[6]["coordinates"][0].update(position=[1, 1, 1]), "token or position differs"),
            (lambda lines: lines[0].update(conditioning=None), "without a declared consumed-prefix schema"),
            (lambda lines: lines[7].update(consumed_prefix_sha256="0" * 64), "consumed-prefix SHA-256 differs from the conditioning"),
            (lambda lines: lines[7].update(consumed_rows=802), "precede the dump's conditioning rows"),
        ):
            self.refused(self.mutated(records, mutate, rows), fragment)
        # Version-1 dumps carry no prefix commitments and are refused outright.
        self.refused(self.mutated(records, lambda lines: lines[0].update(schema="chandra.directcompute.diagnostic-progress.v1")), "v2 plan line")

    # B2: causal values join by semantic prefix, not by the producer's completed tile or last call.
    def test_b2_causal_rows_are_independent_of_completed_tiles(self):
        tiled = [review_hidden(cache_length=64), review_hidden(cache_length=64, stage="text.final_norm")]
        for record in tiled:  # Native tile provenance: completed 64-row tile 0 of an 801-row call.
            record["coordinates"][0].update(call_row=0, chunk_index=0, chunk_row=0)
            record["cache"].update(call_first_row=0, call_row_count=64)
        monolithic = [review_hidden(cache_length=801), review_hidden(cache_length=801, stage="text.final_norm")]
        report = self.compare(tiled, monolithic)
        self.assertEqual(self.statuses(report), [("within_tolerance", None)] * 2)
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("within_tolerance", True))
        # Different causal endpoint or history is never joined.
        report = self.compare([hidden([1.0], row=4)], [hidden([1.0], row=5)])
        self.assertEqual(self.statuses(report), [("incomparable", "reference record absent")])
        rows = [hidden([1.0], row=2), hidden([1.0], row=5)]
        report = self.compare(rows, rows, r_rows=history_for(rows, {3: [12, 3, 3, 3]}))
        self.assertEqual(self.statuses(report), [("within_tolerance", None), ("incomparable", "consumed prefix differs")])

    def test_b2_carried_state_is_independent_of_the_last_tile(self):
        for stage in ("text.gdn_recurrent_state", "text.gdn_conv_state"):
            report = self.compare([review_state(128, stage=stage)], [review_state(0, stage=stage)])
            self.assertEqual(self.statuses(report), [("within_tolerance", None)])
            self.assertEqual(report["cpu_oracle_agreement"], True)
        # Decode-phase state after a different earlier generated token is incomparable.
        state = [review_state(PROMPT + 1, length=PROMPT + 2)]
        report = self.compare(state, state, r_rows=history_for(state, {PROMPT: [6, PROMPT, PROMPT, PROMPT]}))
        self.assertEqual(self.statuses(report), [("incomparable", "consumed prefix differs")])
        # Different prefix endpoints stay distinct records.
        report = self.compare([review_state(64, length=128)], [review_state(0, length=130)])
        self.assertEqual(self.statuses(report), [("incomparable", "reference record absent")])
        # A state whose declared endpoint contradicts its prefix is refused.
        self.refused(self.mutated([review_state(128)], lambda lines: lines[4]["coordinates"][0].update(last_absolute_row=128)),
                     "state endpoint differs")

    # B3: declared BF16 boundaries are validated per producer, independent of tolerances.
    def test_b3_invalid_declared_bf16_boundaries(self):
        invalid, rounded = [review_hidden(value=0.1)], [review_hidden(value=0.10009765625)]
        unrounded = [dict(review_hidden(value=0.1), bf16_rounding="none", storage_dtype="float64")]
        for reference in (rounded, unrounded):
            for tolerances in ({"*": {"max_abs": 0}}, {"*": {"max_abs": 1}}, {}):
                report = self.compare(invalid, reference, tolerances)
                self.assertEqual((report["records"][0]["status"], report["verdict"], report["cpu_oracle_agreement"]),
                                 ("invalid_candidate", "invalid_candidate", False))
                self.assertEqual(report["candidate_policy_violations"], [report["records"][0]["key"]])
                self.assertEqual(report["records"][0]["policy"]["candidate"]["finite_non_bf16_representable"], 1)
                self.assertIn("max_abs", report["records"][0]["metrics"])  # The invalid payload is still reported, not rounded.
        # The review's example: both producers break the declared boundary.
        report = self.compare(invalid, invalid)
        self.assertEqual(self.statuses(report), [("incomparable", "reference declares a BF16 boundary holding finite non-BF16 values")])
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("incomparable", False))
        # An invalid reference is incomparable; a valid candidate against an explicit FP64 reference is ordinary.
        self.assertEqual(self.compare(rounded, invalid)["records"][0]["status"], "incomparable")
        report = self.compare(rounded, unrounded, {"*": {"max_abs": 1e-3, "max_bf16_ulp": 0}})
        row = report["records"][0]
        self.assertEqual((row["status"], row["metrics"]["bf16"]["applicable"], report["cpu_oracle_agreement"]), ("within_tolerance", False, True))
        self.assertEqual(self.compare(rounded, unrounded, {"*": {"max_bf16_ulp": 0}})["verdict"], "no_verdict")
        # CLI exit codes: invalid candidates fail with or without tolerances.
        c, r = self.dump(invalid), self.dump(rounded, producer=ORACLE)
        self.assertEqual(self.run_cli(c, r)[0], 1)
        self.assertEqual(self.run_cli(c, r, "--max-abs", "1")[0], 1)
        # Declared observation counts must match the payload.
        self.refused(self.mutated(invalid, lambda lines: lines[4]["observed"].update(finite_non_bf16_representable=0)),
                     "observed counts differ")

    # B4: terminal metadata must agree with the plan and parsed progress.
    def test_b4_contradictory_terminal_metadata_is_refused(self):
        records = [review_hidden()]
        review = lambda lines: lines[-1].update(planned_records=2, missing_records=["text.layer_output/i1/prefill/s-/r0/c-"],
                                                incomplete_record="unfinished-key", payload_bytes=4, error="failed write")
        key = cd.record_key(review_hidden())
        self.assertEqual(key, "text.layer_output/i0/prefill/s-/r0/c-")
        for mutate, fragment in (
            (review, "planned record count"),
            (lambda lines: lines[-1].update(planned_records=2), "planned record count"),
            (lambda lines: lines[-1].update(missing_records=[key]), "missing records differ"),
            (lambda lines: lines[-1].update(incomplete_record=key), "incomplete record differs"),
            (lambda lines: lines[-1].update(error="failed write"), "completed run names no error"),
            (lambda lines: lines[-1].update(payload_bytes=8), "payload bytes differ"),
            (lambda lines: lines[-1].update(run_status="failed", error="x"), "dump_complete contradicts"),
            (lambda lines: lines[-1].update(dump_complete=False), "dump_complete contradicts"),
            (lambda lines: lines[-1].update(run_status="failed"), "failed run names one"),
            (lambda lines: lines[-1].update(written_records=0), "record count differs"),
            (lambda lines: lines[-1].update(extra=True), "terminal contract"),
            (lambda lines: lines[-1].pop("missing_records"), "terminal contract"),
            (lambda lines: lines[-1].update(qualified=True), "malformed"),
            (lambda lines: lines[-1].update(readbacks=1, readback_bytes=4), "Readback counts without a readback forecast"),
        ):
            self.refused(self.mutated(records, mutate), fragment)

    def test_b4_plan_commitment_and_forecast_counts(self):
        records = [hidden([1.0])]

        def replan(change):
            def mutate(lines):
                change(lines[0]["plan"])
                digest = cd.canonical_sha256(lines[0]["plan"])
                for line in lines:
                    if "commitments" in line:
                        line["commitments"]["plan_sha256"] = digest
                lines[0]["plan_sha256"] = digest
            return mutate

        self.refused(self.mutated(records, lambda lines: lines[0]["plan"].update(note="edited")), "canonical resolved plan")
        self.refused(self.mutated(records, lambda lines: lines[0]["commitments"].update(plan_sha256="0" * 64)), "different plan_sha256")
        self.refused(self.mutated(records, replan(lambda plan: plan["forecast"].update(records=2))), "equal to its record count")
        self.refused(self.mutated(records, replan(lambda plan: plan["forecast"].update(keys=["a"]))), "unplanned")
        self.refused(self.mutated(records, replan(lambda plan: plan["forecast"]["keys"].insert(0, "z/extra"))), "sorted, unique")
        self.refused(self.mutated(records, replan(lambda plan: plan.update(schema="other"))), "Unknown plan schema")
        self.refused(self.mutated(records, lambda lines: lines[3].update(key="text.layer_output/i5/prefill/s-/r3/c-")), "unplanned")
        self.refused(self.mutated(records, lambda lines: lines[4].update(key="text.layer_output/i5/prefill/s-/r3/c-")), "matching start line")
        self.refused(self.mutated(records, lambda lines: [line.update(key="text.layer_output/i5/prefill/s-/r3/c-") for line in lines[3:5]]),
                     "unplanned")
        # A native plan's forecast keys must be exactly what its resolved selections enumerate, so an unreached
        # requested scope cannot be dropped from the key list to claim completeness.
        native = {"schema": cd.NATIVE_PLAN_SCHEMA, "byte_limit": 1 << 20,
                  "vision": {"rows": [], "blocks": [], "patch_embedding": False, "position_added": False},
                  "merger": {"rows": []}, "embedding": {"rows": []},
                  "prefill": {"rows": [2], "layers": [5], "final_norm": False, "logits": False},
                  "decode": {"steps": [], "layers": [], "final_norm": False, "logits": False}, "gdn_state": []}

        def nativize(change):
            def apply(plan):
                keys = plan["forecast"]
                plan.clear()
                plan.update(json.loads(json.dumps(native)), forecast=keys)
                change(plan)
            return replan(apply)

        loaded = cd.load_dump(self.mutated(records, nativize(lambda plan: None)))
        self.assertEqual(loaded["dump_complete"], True)
        self.refused(self.mutated(records, nativize(lambda plan: plan["decode"].update(steps=[0], logits=True))), "resolved selections")
        self.refused(self.mutated(records, nativize(lambda plan: plan["prefill"].update(rows=[2, 3]))), "resolved selections")
        self.refused(self.mutated(records, nativize(lambda plan: plan.update(byte_limit=3))), "byte limit")  # One 4-byte payload.

    def test_b4_genuine_partial_dumps_stay_readable_but_never_complete(self):
        records = [review_hidden(0, 5, 802)]
        reference = self.dump(records + [review_hidden(1, 7, 803)], producer=ORACLE)
        # Early stop: the run completed, planned step-1 records were never reached.
        unreached = cd.record_key(review_hidden(1, 7, 803))
        early = self.dump(records, planned_keys=[unreached])
        self.refused(early, "completed and incomplete", allow=False)
        loaded = cd.load_dump(early, True)
        self.assertEqual((loaded["run_status"], loaded["dump_complete"], loaded["finished"]["missing_records"]), ("completed", False, [unreached]))
        report = cd.compare(loaded, cd.load_dump(reference), {"*": {"max_abs": 0}})
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("within_tolerance_incomplete_dump", False))
        # Failed record: started, never written, named by the failed terminal line.
        failed = self.dump(records, run_completed=False, planned_keys=[unreached], error="injected write failure")
        lines = self.progress(failed)
        lines.insert(-1, {"event": "record_started", "sequence": 1, "file": "00001.text.layer_output.f32", "key": unreached})
        lines[-1]["incomplete_record"] = unreached
        self.rewrite(failed, lines)
        loaded = cd.load_dump(failed, True)
        self.assertEqual((loaded["run_status"], loaded["dump_complete"], loaded["incomplete_record"]), ("failed", False, unreached))
        report = cd.compare(loaded, cd.load_dump(reference), {"*": {"max_abs": 0}})
        self.assertEqual((report["verdict"], report["cpu_oracle_agreement"]), ("within_tolerance_incomplete_dump", False))
        self.refused(failed, "failed and incomplete", allow=False)
        # The same failed line without its unfinished start contradicts the progress.
        lines.pop(-2)
        self.rewrite(failed, lines)
        self.refused(failed, "incomplete record differs")

if __name__ == "__main__":
    unittest.main()
