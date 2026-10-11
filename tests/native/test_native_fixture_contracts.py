
"""CPU-only fake protocol and descriptor ownership controls; no native worker or model runs."""
import copy
import os
from pathlib import Path
import tempfile
import threading
import unittest

from runtime.native import NativeBackend
from runtime.native import protocol as P
from tests.native.native_fakes import PINNED, FakeProcessor, FakeTransport, config, drain_wait, payload


class DrainObservationTests(unittest.TestCase):
    def test_query_fixture_is_settled_and_accepted(self):
        P.check_drain_wait(drain_wait(), "query_sleep", settled=True)
        P.check_resident_drain_wait(self.result(), "query_sleep")

    def result(self):
        return {"memory_observations": [{"phase": "request_retirement_and_drain",
                "observed": {"drain_wait": drain_wait(), "cpu_timing": {
                    "schema": "chandra.directcompute.cpu_timing.v1", "enabled": False,
                    "environment_variable": "CHANDRA_NATIVE_CPU_TIMING"}}}]}

    def test_missing_wrong_type_route_and_outstanding_waits_are_refused(self):
        for field, value in (("schema", None), ("route", "fence_event"), ("calls", True),
                             ("calls", 1), ("completed_calls", -1), ("poisoned", True),
                             ("registered_event_outstanding", True), ("pending_storage_references", 1)):
            changed = drain_wait()
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(P.ProtocolViolation):
                P.check_drain_wait(changed, "query_sleep", settled=True)
        with self.assertRaises(P.ProtocolViolation):
            P.check_drain_wait(None, "query_sleep", settled=True)

    def test_resident_receipt_requires_the_actual_final_drain_and_disabled_timing(self):
        cases = []
        for mutate in (lambda v: v.clear(),
                       lambda v: v.update(memory_observations=[]),
                       lambda v: v["memory_observations"][-1].update(phase="decode_step"),
                       lambda v: v["memory_observations"][-1]["observed"].pop("drain_wait"),
                       lambda v: v["memory_observations"][-1]["observed"]["cpu_timing"].update(enabled=True),
                       lambda v: v["memory_observations"][-1]["observed"]["drain_wait"]["pending_copy_owner"].update(active=True),
                       lambda v: v["memory_observations"][-1]["observed"]["drain_wait"]["pending_copy_owner"].update(staging_bytes=4)):
            result = copy.deepcopy(self.result())
            mutate(result)
            cases.append(result)
        for result in cases:
            with self.subTest(result=result), self.assertRaises(P.ProtocolViolation):
                P.check_resident_drain_wait(result, "query_sleep")


class FakePipeOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-pipe-owner-")
        self.addCleanup(self.temporary.cleanup)
        self.counter = 0

    def backend(self):
        self.counter += 1
        root = Path(self.temporary.name) / str(self.counter)
        root.mkdir()
        transport = FakeTransport()
        backend = NativeBackend(config(root), transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.retire, backend, transport)
        return backend, transport

    def retire(self, backend, transport):
        transport.unkillable = False
        backend.close()
        for worker in transport.workers:
            worker.kill()
            worker.thread.join(2)
            if worker.reader_thread is not None and worker.reader_thread.ident is not None:
                worker.reader_thread.join(2)
            self.assertFalse(worker.thread.is_alive())
            self.assertTrue(worker.reader_thread is None or not worker.reader_thread.is_alive())
            worker.close_pipes()

    def test_repeated_closure_releases_real_fds_without_closing_reused_numbers(self):
        for _ in range(20):
            backend, transport = self.backend()
            list(backend.generate(payload(), threading.Event()))
            worker, process = transport.workers[0], transport.launches[0]["process"]
            owned = tuple(worker.pipe_fds.values())
            self.assertEqual(len(owned), 6)
            self.assertTrue(backend.close())
            self.assertEqual(worker.pipe_fds, {})
            for fd in owned:
                with self.assertRaises(OSError):
                    os.fstat(fd)
            sentinel = os.open(os.devnull, os.O_RDONLY)
            try:
                if sentinel != owned[0]:
                    os.dup2(sentinel, owned[0])
                    os.close(sentinel)
                    sentinel = owned[0]
                self.assertIn(sentinel, owned)
                self.assertTrue(process.confirm_closed()["handles_closed"])
                self.assertTrue(process.confirm_closed()["handles_closed"])
                os.fstat(sentinel)  # A repeated confirmation must not retry an old descriptor number.
            finally:
                os.close(sentinel)

    def test_unconfirmed_worker_keeps_its_real_descriptors(self):
        backend, transport = self.backend()
        list(backend.generate(payload(), threading.Event()))
        worker, process = transport.workers[0], transport.launches[0]["process"]
        owned = tuple(worker.pipe_fds.values())
        transport.unkillable = True
        self.assertIsNone(process.confirm_closed())
        self.assertEqual(len(worker.pipe_fds), 6)
        for fd in owned:
            os.fstat(fd)
        transport.unkillable = False
        self.assertTrue(backend.close())
        self.assertEqual(worker.pipe_fds, {})


if __name__ == "__main__":
    unittest.main()
