"""Offline C2/C4 and explicit Q8 C6 sessions: slots and shared worker retirement."""

import asyncio
import threading
import time
import unittest

from chandra_service.service import Worker
from runtime.llama.backend import LlamaFailure
from runtime.llama.worker import AdmissionRefused
from tests.llama.test_backend import GIB, Harness, request

Q8_C6 = {"engine": "sycl", "sycl_kv_cache_type": "q8_0",
         "admission_physical_bytes": 21 * GIB // 4, "admission_commit_bytes": 49 * GIB // 4}


class ParallelBackendTests(unittest.TestCase):
    def harness(self, parallel=2, **profile):
        h = Harness(parallel, **profile)
        h.stream_mode = "held"
        self.addCleanup(h.close)
        return h

    def until(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("fake parallel condition did not arrive")
            time.sleep(0.01)

    def blocked(self, h, name):
        self.until(lambda: h.record(name) is not None and h.record(name).blocked.is_set())
        return h.record(name)

    def finished(self, result):
        self.assertTrue(result.done.wait(3))
        result.thread.join(1)
        self.assertFalse(result.thread.is_alive())

    def test_two_four_and_explicit_q8_six_requests_share_distinct_slots(self):
        for parallel, profile in ((2, {}), (4, {}), (6, Q8_C6)):
            with self.subTest(parallel=parallel):
                h = self.harness(parallel, **profile)
                results = [h.collect(request(f"page-{index}")) for index in range(parallel)]
                records = [self.blocked(h, f"page-{index}") for index in range(parallel)]
                self.assertEqual((len(h.workers), len(h.clients), len(h.posts)), (1, 1, parallel))
                self.assertEqual({record.slot for record in records}, set(range(parallel)))
                self.assertTrue(all(not result.done.is_set() for result in results))
                for record in records:
                    record.release.set()
                for result in results:
                    self.finished(result)
                    self.assertIsNone(result.error)
                    self.assertIsInstance(result.items[-1], dict)
                self.assertTrue(all(record.post_done_slot_checks > 0 for record in records))

    def test_concurrent_callers_share_startup_before_any_private_client_exists(self):
        h = self.harness(4)
        h.start_allowed.clear()
        results = [h.collect(request("first"))]
        self.assertTrue(h.start_entered.wait(2))
        results.extend(h.collect(request(f"peer-{index}")) for index in range(3))
        time.sleep(0.1)
        self.assertEqual((len(h.workers), len(h.clients), len(h.posts)), (1, 0, 0))
        h.start_allowed.set()
        names = ("first", "peer-0", "peer-1", "peer-2")
        records = [self.blocked(h, name) for name in names]
        self.assertEqual((len(h.workers), len(h.clients)), (1, 1))
        for record in records:
            record.release.set()
        for result in results:
            self.finished(result)
            self.assertIsNone(result.error)

    def test_completed_call_cancel_and_generator_close_do_not_abort_a_busy_peer(self):
        h = self.harness()
        peer = h.collect(request("peer"))
        record = self.blocked(h, "peer")
        h.modes["completed"] = "normal"
        cancel = threading.Event()
        completed = h.backend.generate(request("completed"), cancel)
        self.addCleanup(completed.close)
        self.assertEqual(next(completed), "<p>ok</p>")
        self.assertEqual(next(completed)["finish_reason"], "stop")
        # The generator is paused after successful metadata, before its final release.
        cancel.set()
        self.assertFalse(h.workers[0].stopped.wait(0.2))
        self.assertFalse(peer.done.is_set())
        self.assertFalse(peer.cancel.is_set())
        completed.close()
        self.assertFalse(h.workers[0].stopped.wait(0.2))
        self.assertFalse(peer.done.is_set())
        self.assertIsNone(h.backend.health()["error"])
        record.release.set()
        self.finished(peer)
        self.assertIsNone(peer.error)

    def test_shared_startup_admission_refusal_is_retryable_for_all_joined_callers(self):
        h = self.harness()
        h.start_allowed.clear()
        first = h.collect(request("first"))
        self.assertTrue(h.start_entered.wait(2))
        peer = h.collect(request("peer"))
        self.until(lambda: h.preparations == 2)
        worker = h.workers[0]
        worker.allow_retire.clear()
        h.memory["available_physical"] = 10 * GIB  # Above the live floor, but post-auth forecast admission refuses.
        h.start_allowed.set()
        self.assertTrue(worker.retiring.wait(2))
        self.assertFalse(first.done.is_set())
        self.assertFalse(peer.done.is_set())
        self.assertIsNone(h.backend.health()["error"])
        self.assertEqual((len(h.workers), len(h.clients), len(h.posts)), (1, 0, 0))
        h.memory["available_physical"] = 20 * GIB
        worker.allow_retire.set()
        for result in (first, peer):
            self.finished(result)
            self.assertIsInstance(result.error, AdmissionRefused)
        self.assertIsNone(h.backend.health()["error"])
        h.modes["recovered"] = "normal"
        recovered = h.collect(request("recovered"))
        self.finished(recovered)
        self.assertIsNone(recovered.error)
        self.assertEqual((len(h.workers), len(h.posts)), (2, 1))

    def test_cancel_during_shared_startup_waits_for_start_and_confirmed_retirement(self):
        h = self.harness()
        h.start_allowed.clear()
        first = h.collect(request("first"))
        self.assertTrue(h.start_entered.wait(2))
        peer = h.collect(request("peer"))
        self.until(lambda: h.preparations == 2)
        worker = h.workers[0]
        worker.allow_retire.clear()
        first.cancel.set()
        self.assertTrue(worker.stopped.wait(2))
        self.assertFalse(worker.retiring.is_set())
        self.assertFalse(first.done.is_set())
        self.assertFalse(peer.done.is_set())
        self.assertEqual((len(h.clients), len(h.posts)), (0, 0))
        h.start_allowed.set()
        self.assertTrue(worker.retiring.wait(2))
        self.assertFalse(first.done.is_set())
        self.assertFalse(peer.done.is_set())
        self.assertTrue(h.backend.health()["owned_worker_pending"])
        worker.allow_retire.set()
        for result in (first, peer):
            self.finished(result)
            self.assertIsInstance(result.error, LlamaFailure)
        self.assertFalse(h.backend.health()["owned_worker_pending"])
        self.assertIsNone(h.backend.health()["error"])

    def test_done_waits_for_own_idle_slot_then_replacement_uses_only_that_slot(self):
        h = self.harness()
        first, peer = h.collect(request("first")), h.collect(request("peer"))
        a, b = self.blocked(h, "first"), self.blocked(h, "peer")
        h.slot_releases[a.slot].clear()
        a.release.set()
        self.assertTrue(a.done.wait(2))
        self.assertTrue(h.slot_waiting.wait(2))
        self.assertFalse(first.done.is_set())
        replacement = h.collect(request("replacement"))
        self.assertFalse(replacement.done.wait(0.1))
        self.assertEqual(len(h.posts), 2)
        h.slot_releases[a.slot].set()
        self.finished(first)
        self.assertIsNone(first.error)
        self.assertFalse(peer.done.is_set())
        c = self.blocked(h, "replacement")
        self.assertEqual(c.slot, a.slot)
        self.assertNotEqual(c.slot, b.slot)
        c.release.set()
        self.finished(replacement)
        self.assertIsNone(replacement.error)
        self.assertFalse(peer.done.is_set())
        b.release.set()
        self.finished(peer)
        self.assertIsNone(peer.error)
        self.assertTrue(all(record.post_done_slot_checks > 0 for record in (a, b, c)))

    def test_cancel_retires_peers_and_retains_the_old_client_until_confirmed(self):
        h = self.harness()
        first, peer = h.collect(request("first")), h.collect(request("peer"))
        self.blocked(h, "first")
        self.blocked(h, "peer")
        worker, client = h.workers[0], h.clients[0]
        worker.allow_retire.clear()
        first.cancel.set()
        self.assertTrue(worker.retiring.wait(2))
        h.modes["during-stop"] = "normal"
        late = h.collect(request("during-stop"))
        self.assertFalse(first.done.is_set())
        self.assertFalse(peer.done.is_set())
        self.assertFalse(client.is_closed)
        time.sleep(0.1)
        self.assertEqual((len(h.workers), len(h.posts)), (1, 2))
        worker.allow_retire.set()
        self.finished(first)
        self.finished(peer)
        self.assertIsInstance(peer.error, LlamaFailure)
        self.assertTrue(client.is_closed)
        self.finished(late)  # It may refuse the stopping session or join a fresh one after retirement.
        h.modes["later"] = "normal"
        restarted = h.collect(request("later"))
        self.finished(restarted)
        self.assertIsNone(restarted.error)
        self.assertEqual(len(h.workers), 2)
        names = [body["messages"][0]["content"][0]["text"] for body in h.posts]
        self.assertEqual((names.count("first"), names.count("peer"), names.count("later")), (1, 1, 1))

    def test_http_failure_retires_a_running_peer_and_never_replays_either_post(self):
        h = self.harness()
        peer = h.collect(request("peer"))
        self.blocked(h, "peer")
        worker = h.workers[0]
        worker.allow_retire.clear()
        h.statuses["bad"] = 500
        bad = h.collect(request("bad"))
        self.assertTrue(worker.retiring.wait(2))
        self.assertFalse(peer.done.is_set())
        self.assertFalse(bad.done.is_set())
        self.assertEqual(len(h.posts), 2)
        worker.allow_retire.set()
        for result in (peer, bad):
            self.finished(result)
            self.assertIsInstance(result.error, LlamaFailure)
        refused = h.collect(request("after-fault"))
        self.finished(refused)
        self.assertIsInstance(refused.error, LlamaFailure)
        self.assertEqual((len(h.workers), len(h.posts)), (1, 2))

    def test_control_fault_first_observed_during_cancel_retirement_latches_restart_closed(self):
        h = self.harness()
        first, peer = h.collect(request("first")), h.collect(request("peer"))
        self.blocked(h, "first")
        self.blocked(h, "peer")
        worker = h.workers[0]
        worker.allow_retire.clear()
        retire = worker.retire

        def fault_during_retirement():
            receipt = retire()
            receipt["fault"] = "retirement: RuntimeError: fake transient control failure, cleanup now confirmed"
            return receipt

        worker.retire = fault_during_retirement
        first.cancel.set()
        self.assertTrue(worker.retiring.wait(2))
        self.assertTrue(all(not result.done.is_set() for result in (first, peer)))
        worker.allow_retire.set()
        for result in (first, peer):
            self.finished(result)
            self.assertIsInstance(result.error, LlamaFailure)
        self.assertTrue(worker.closed)
        self.assertIsNotNone(h.backend.health()["error"])
        refused = h.collect(request("after-retirement-fault"))
        self.finished(refused)
        self.assertIsInstance(refused.error, LlamaFailure)
        self.assertEqual((len(h.workers), len(h.posts)), (1, 2))

    def test_guard_trip_holds_all_affected_callers_until_shared_retirement(self):
        h = self.harness(4)
        results = [h.collect(request(f"page-{index}")) for index in range(4)]
        for index in range(4):
            self.blocked(h, f"page-{index}")
        worker = h.workers[0]
        worker.allow_retire.clear()
        h.memory["available_commit"] = 4 * GIB
        self.assertTrue(worker.retiring.wait(2))
        self.assertTrue(all(not result.done.is_set() for result in results))
        self.assertTrue(h.backend.health()["owned_worker_pending"])
        worker.allow_retire.set()
        for result in results:
            self.finished(result)
            self.assertIsInstance(result.error, LlamaFailure)
        self.assertIsNotNone(h.backend.health()["error"])
        self.assertEqual(len(h.posts), 4)

    def test_context_refusal_releases_only_the_unsubmitted_reservation(self):
        h = self.harness()
        peer = h.collect(request("peer"))
        a = self.blocked(h, "peer")
        h.token_counts["too-long"] = 4000
        refused = h.collect(request("too-long"))
        self.finished(refused)
        self.assertIsInstance(refused.error, ValueError)
        self.assertEqual(len(h.posts), 1)
        self.assertFalse(h.workers[0].stopped.is_set())
        replacement = h.collect(request("replacement"))
        b = self.blocked(h, "replacement")
        self.assertNotEqual(a.slot, b.slot)
        self.assertEqual(len(h.workers), 1)
        for record in (a, b):
            record.release.set()
        for result in (peer, replacement):
            self.finished(result)
            self.assertIsNone(result.error)

    def test_one_wrong_slot_context_refuses_the_complete_profile_before_generation(self):
        h = self.harness(4)
        h.slot_contexts[2] = 8192
        result = h.collect(request("page"))
        self.finished(result)
        self.assertIsInstance(result.error, LlamaFailure)
        self.assertEqual(h.posts, [])
        self.assertTrue(h.workers[0].closed)


class ParallelServiceCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def until(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("fake parallel service condition did not arrive")
            await asyncio.sleep(0.01)

    async def test_q8_six_admitted_seventh_refused_and_cancellation_never_replays(self):
        h = Harness(6, **Q8_C6)
        h.stream_mode = "held"
        self.addAsyncCleanup(asyncio.to_thread, h.close)
        service = Worker(h.backend, queue_limit=5, inference_seconds=5)
        jobs = [service.submit(request(f"page-{index}")) for index in range(6)]
        self.assertTrue(all(job is not None for job in jobs))
        await self.until(lambda: len(h.records) == 6 and all(record.blocked.is_set() for record in h.records))
        self.assertEqual({record.slot for record in h.records}, set(range(6)))
        self.assertIsNone(service.submit(request("seventh")))
        worker = h.workers[0]
        worker.allow_retire.clear()
        jobs[0].cancel.set()
        await self.until(worker.retiring.is_set)
        self.assertTrue(all(not job.done.is_set() for job in jobs))
        self.assertIsNone(service.submit(request("during-retirement")))
        worker.allow_retire.set()
        await self.until(lambda: not service.jobs and all(job.done.is_set() for job in jobs))
        self.assertTrue(worker.closed)
        self.assertEqual((len(h.workers), len(h.posts)), (1, 6))
        self.assertTrue(all(not job.metadata for job in jobs))

    async def test_canceled_peer_holds_all_four_service_jobs_until_shared_retirement(self):
        h = Harness(4)
        h.stream_mode = "held"
        self.addAsyncCleanup(asyncio.to_thread, h.close)
        service = Worker(h.backend, queue_limit=3, inference_seconds=5)
        jobs = [service.submit(request(f"page-{index}")) for index in range(4)]
        self.assertTrue(all(job is not None for job in jobs))
        await self.until(lambda: len(h.records) == 4 and all(record.blocked.is_set() for record in h.records))
        worker = h.workers[0]
        worker.allow_retire.clear()
        jobs[0].cancel.set()
        await self.until(worker.retiring.is_set)
        self.assertTrue(all(not job.done.is_set() for job in jobs))
        self.assertEqual(len(service.jobs), 4)
        self.assertIsNone(service.submit(request("extra")))
        worker.allow_retire.set()
        await self.until(lambda: not service.jobs and all(job.done.is_set() for job in jobs))
        self.assertTrue(worker.closed)
        self.assertEqual(len(h.posts), 4)
        self.assertTrue(all(not job.metadata for job in jobs))
