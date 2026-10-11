"""CPU-only contracts for the experimental native route; no worker, model or shader runs."""
import asyncio
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from runtime.native import config as C, protocol as P
from tests.native.native_fakes import config
from chandra_service.service import Job, Worker


class ExperimentalRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-route-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.config = config(Path(self.temporary.name))

    def selected(self, **changes):
        values = dict(experimental_b2_decode=True, gemv_b1_selection="parallel32",
                      experimental_unified_parallel32_b2=True,
                      experimental_b2_mix_input_projections=True)
        values.update(changes)
        return dataclasses.replace(self.config, **values)

    def test_default_route_does_not_enable_experiments(self):
        self.assertFalse(self.config.experimental_b2_decode)
        self.assertFalse(self.config.experimental_b2_mix_input_projections)
        self.assertFalse(any(a.startswith("--experimental-") for a in self.config.argv(Path("snapshot"))))
        self.assertEqual(self.config.environment({"KEEP": "yes", "CHANDRA_EXPERIMENTAL_GEMV_B1": "parallel32",
                                                 "chandra_unknown": "yes"}), {"KEEP": "yes"})

    def test_explicit_mix_route_sets_only_requested_selectors(self):
        selected = self.selected(drain_wait_selection="fence_event", experimental_attention_values_grid=True)
        argv = selected.argv(Path("snapshot"))
        for flag in ("--experimental-padded32", "--experimental-b2-decode",
                     "--experimental-unified-parallel32-b2", "--experimental-b2-mix-input-projections",
                     "--experimental-attention-values-grid"):
            self.assertEqual(argv.count(flag), 1)
        self.assertNotIn("--experimental-ordered2-b2", argv)
        self.assertTrue(selected.expectation().experimental_b2_mix_input_projections)
        self.assertEqual(selected.environment({"KEEP": "yes", "CHANDRA_OTHER": "discard"}),
                         {"KEEP": "yes", "CHANDRA_EXPERIMENTAL_GEMV_B1": "parallel32",
                          "CHANDRA_EXPERIMENTAL_DRAIN_WAIT": "fence_event"})
        self.assertEqual(selected.job_limits(), self.config.job_limits())

    def test_mix_requires_typed_unified_two_row_selection(self):
        for changes in ({"experimental_b2_decode": False},
                        {"experimental_unified_parallel32_b2": False},
                        {"experimental_ordered2_b2": True},
                        {"gemv_b1_selection": "ordered"},
                        {"experimental_b2_mix_input_projections": 1}):
            with self.subTest(changes=changes), self.assertRaises(C.ConfigurationError):
                self.selected(**changes)

    def test_mix_protocol_refuses_mutated_arithmetic_or_unselected_mode(self):
        value = dict(P.COHORT_ARITHMETIC,
                     schema="chandra.directcompute.worker-unified-parallel32-b2-mix-input-selection.v1",
                     B1_selector="parallel32", parallel32_excluded=False,
                     unified_parallel32=dict(P.UNIFIED_PARALLEL32_ARITHMETIC, scope=P.MIX_INPUT_SCOPE),
                     mix_input_projections=copy.deepcopy(P.MIX_INPUT_PROJECTIONS))
        P.check_cohort_arithmetic(value, False, False, True, True)
        for mutate in (lambda v: v.update(B1_selector="ordered"),
                       lambda v: v.update(parallel32_excluded=0),
                       lambda v: v["mix_input_projections"].update(unexpected=True)):
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.assertRaises(P.ProtocolViolation):
                P.check_cohort_arithmetic(changed, False, False, True, True)
        with self.assertRaises(P.ProtocolViolation):
            P.check_cohort_arithmetic(value, False, False, False, True)

    def test_cohort_events_require_opt_in_and_ordered_sequences(self):
        line = json.dumps({"schema": P.EVENT_SCHEMA, "seq": 0, "event": "cohort_started"}).encode("utf-8")
        with self.assertRaises(P.ProtocolViolation):
            P.EventStream().accept(line)
        stream = P.EventStream(True)
        self.assertEqual(stream.accept(line)["event"], "cohort_started")
        with self.assertRaises(P.ProtocolViolation):
            stream.accept(line)
        with self.assertRaises(P.ProtocolViolation):
            P.EventStream(True).accept(json.dumps({"schema": P.EVENT_SCHEMA, "seq": True, "event": "cohort_started"}).encode("utf-8"))
        with self.assertRaises(P.ProtocolViolation):
            P.EventStream(True).accept(b'{"schema":"x","seq":0,"seq":1,"event":"ready"}')

    def test_owner_schema_refuses_invalid_mix_combinations(self):
        self.assertEqual(P.cohort_owner_schema(False, True, True, True),
                         "chandra.directcompute.worker-unified-parallel32-b2-mix-input-attention-values-grid-page-owner.v1")
        for flags in ((False, False, False, True), (True, False, True, True)):
            with self.assertRaises(P.ProtocolViolation):
                P.cohort_owner_schema(*flags)


class ServiceAssociationTests(unittest.TestCase):
    def test_existing_backend_keeps_one_execution_slot(self):
        class Backend:
            pass
        worker = Worker(Backend(), 0)
        self.assertEqual(worker.execution_slots, 1)
        self.assertIsInstance(worker.lock, asyncio.Lock)
        for parallel in (True, 0, 3):
            backend = Backend()
            backend.parallel_requests = parallel
            with self.assertRaises(ValueError):
                Worker(backend, 0)

    def test_response_association_is_submitted_once_on_failure(self):
        class Backend:
            parallel_requests = 2
            calls = []
            def generate_associated(self, payload, cancel, response_id):
                self.calls.append(response_id)
                raise RuntimeError("synthetic failure after the one submission")
        backend = Backend()
        worker = Worker(backend, 1)
        job = Job({"max_tokens": 12384}, response_id="chatcmpl-synthetic")
        with self.assertLogs(level="ERROR"):
            worker.generate(job)
        self.assertEqual(backend.calls, ["chatcmpl-synthetic"])
        self.assertTrue(job.failure)
        self.assertEqual(worker.execution_slots, 2)

    def test_idle_guard_holds_and_releases_both_slots(self):
        class Backend:
            parallel_requests = 2
        worker = Worker(Backend(), 1)
        async def exercise():
            async with worker._all_slots():
                self.assertTrue(worker.lock.locked())
            self.assertFalse(worker.lock.locked())
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
