"""Finite offline adapter checks with fake Job ownership and in-memory HTTP only."""

import asyncio
import copy
import json
import threading
import time
from types import SimpleNamespace
import unittest

import httpx

from chandra_service.service import Worker, create_app
from runtime.llama.backend import LlamaBackend, LlamaFailure
from runtime.llama.worker import AdmissionRefused

GIB = 1 << 30


def request(name=None):
    return {"model": "chandra", "max_tokens": 12384, "temperature": 0, "top_p": 0.1,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ocr_layout" if name is None else name},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}]}


def frame(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class FakeWorker:
    def __init__(self, config):
        self.alias = None
        self.stopped, self.retiring, self.allow_retire = (threading.Event() for _ in range(3))
        self.allow_retire.set()
        self.closed = False
        self.start_entered, self.start_allowed = threading.Event(), threading.Event()
        self.start_allowed.set()

    def start(self, port, key, alias, admit=None):
        self.start_entered.set()
        self.start_allowed.wait()
        if admit:
            admit()
        self.alias = alias
        return self

    def request_stop(self):
        self.stopped.set()

    def alive(self):
        return not self.stopped.is_set() and not self.closed

    def retire(self):
        self.retiring.set()
        self.allow_retire.wait()
        self.closed = True
        return self.snapshot()

    def snapshot(self):
        return {"closed": self.closed, "fault": None}


class GatedStream(httpx.SyncByteStream):
    def __init__(self, harness, record):
        self.harness, self.record, self.alias = harness, record, record.worker.alias

    def event(self, choices, **extra):
        return frame({"model": self.alias, "object": "chat.completion.chunk", "choices": choices, **extra})

    def __iter__(self):
        h = self.harness
        record = self.record
        content = self.event([{"index": 0, "delta": {"content": "<p>ok</p>"}, "finish_reason": None}])
        if record.mode == "after-content":
            yield content  # A short delta must reach the caller before the later read blocks.
        if record.mode in ("blocked", "after-content", "held"):
            h.stream_blocked.set()
            record.blocked.set()
            deadline = time.monotonic() + 5
            while not record.worker.stopped.is_set():
                if record.mode == "held" and record.release.wait(0.01):
                    break
                if record.mode != "held":
                    record.worker.stopped.wait(0.01)
                if time.monotonic() >= deadline:
                    raise AssertionError("fake blocked stream was never stopped or released")
            if record.worker.stopped.is_set():
                raise httpx.ReadError("fake connection ended after owned worker stop")
        yield content
        yield self.event([{"index": 0, "delta": {}, "finish_reason": "stop"}])
        yield self.event([], usage={"prompt_tokens": record.input_tokens, "completion_tokens": 557,
                                    "total_tokens": record.input_tokens + 557})
        record.done.set()
        yield b"data: [DONE]\n\n"


class Collection:
    def __init__(self, backend, payload=None, cancel=None):
        self.items, self.error = [], None
        self.done, self.content = threading.Event(), threading.Event()
        self.cancel = cancel or threading.Event()

        def run():
            try:
                for item in backend.generate(payload or request(), self.cancel):
                    self.items.append(item)
                    if isinstance(item, str) and item:
                        self.content.set()
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, name="fake-adapter-request")
        self.thread.start()


class Harness:
    def __init__(self, parallel=1):
        self.workers, self.clients, self.calls, self.collections = [], [], [], []
        self.mutex = threading.RLock()
        self.records, self.active_slots = [], {}
        self.modes, self.statuses, self.token_counts = {}, {}, {}
        self.preparations = 0
        self.input_tokens, self.post_status, self.stream_mode = 3617, 200, "normal"
        self.memory = {"available_physical": 20 * GIB, "available_commit": 30 * GIB}
        self.stream_blocked, self.slot_waiting, self.slot_release = (threading.Event() for _ in range(3))
        self.slot_release.set()
        self.slot_releases = [threading.Event() for _ in range(parallel)]
        for gate in self.slot_releases:
            gate.set()
        self.slot_contexts = [16384] * parallel
        self.start_entered, self.start_allowed = threading.Event(), threading.Event()
        self.start_allowed.set()
        self.config = SimpleNamespace(startup_seconds=5, request_seconds=5, max_image_pixels=6_291_456,
                                      admission_physical_bytes=6 * GIB, admission_commit_bytes=11 * GIB,
                                      parallel_requests=parallel, engine="vulkan")
        self.backend = LlamaBackend(self.config, worker_factory=self.worker_factory,
                                    client_factory=self.client_factory, prepare=self.prepare,
                                    memory=lambda: dict(self.memory), port_factory=lambda: 18003)

    @property
    def posts(self):
        with self.mutex:
            return [body for method, path, body in self.calls if method == "POST" and path == "/v1/chat/completions"]

    def record(self, name):
        with self.mutex:
            return next((item for item in self.records if item.name == name), None)

    def worker_factory(self, config):
        worker = FakeWorker(config)
        worker.start_entered, worker.start_allowed = self.start_entered, self.start_allowed
        self.workers.append(worker)
        return worker

    def client_factory(self, **options):
        worker = self.workers[-1]
        client = httpx.Client(transport=httpx.MockTransport(lambda incoming: self.handle(incoming, worker)), **options)
        self.clients.append(client)
        return client

    def prepare(self, payload, maximum):
        self.preparations += 1
        return copy.deepcopy(payload), {"fake": True}

    def handle(self, incoming, worker):
        body = json.loads(incoming.content) if incoming.content else None
        path = incoming.url.path
        with self.mutex:
            self.calls.append((incoming.method, path, body))
        alias = worker.alias
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": alias}]})
        if path == "/slots":
            slots = []
            with self.mutex:
                for record in self.records:
                    if record.worker is worker and record.thread_id == threading.get_ident() and record.done.is_set():
                        record.post_done_slot_checks += 1
                for slot, context in enumerate(self.slot_contexts):
                    active = self.active_slots.get((worker, slot))
                    if active is not None and active.done.is_set():
                        if self.slot_release.is_set() and self.slot_releases[slot].is_set():
                            del self.active_slots[(worker, slot)]
                            active = None
                        else:
                            self.slot_waiting.set()
                    slots.append({"id": slot, "n_ctx": context, "is_processing": active is not None})
            return httpx.Response(200, json=slots)
        if path == "/v1/chat/completions/input_tokens":
            name = body["messages"][0]["content"][0]["text"]
            return httpx.Response(200, json={"input_tokens": self.token_counts.get(name, self.input_tokens)})
        if path == "/v1/chat/completions":
            name = body["messages"][0]["content"][0]["text"]
            slot = body.get("id_slot")
            assert type(slot) is int and 0 <= slot < self.config.parallel_requests and body.get("n") == 1
            status = self.statuses.get(name, self.post_status)
            if status != 200:
                return httpx.Response(status, json={"error": "fake backend failure"})
            record = SimpleNamespace(name=name, worker=worker, slot=slot, blocked=threading.Event(),
                                     release=threading.Event(), done=threading.Event(), post_done_slot_checks=0,
                                     thread_id=threading.get_ident(),
                                     mode=self.modes.get(name, self.stream_mode), input_tokens=self.token_counts.get(name, self.input_tokens))
            with self.mutex:
                assert (worker, slot) not in self.active_slots, "two live requests were assigned the same slot"
                self.active_slots[(worker, slot)] = record
                self.records.append(record)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=GatedStream(self, record))
        raise AssertionError(f"unexpected fake endpoint: {path}")

    def collect(self, payload=None, cancel=None):
        result = Collection(self.backend, payload, cancel)
        self.collections.append(result)
        return result

    def close(self):
        self.start_allowed.set()
        self.slot_release.set()
        for gate in self.slot_releases:
            gate.set()
        for collection in self.collections:
            collection.cancel.set()
        for worker in self.workers:
            worker.request_stop()
            worker.allow_retire.set()
        for collection in self.collections:
            collection.thread.join(3)
            if collection.thread.is_alive():
                raise AssertionError("fake request retained a test thread")
        self.backend.close()


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.close)

    def finished(self, collection):
        self.assertTrue(collection.done.wait(2), "fake generation did not finish")
        collection.thread.join(1)
        self.assertFalse(collection.thread.is_alive())

    def assert_retirement_retains_request(self, result):
        worker = self.h.workers[-1]
        self.assertTrue(worker.retiring.wait(2))
        self.assertFalse(result.done.is_set())
        self.assertTrue(self.h.backend.health()["owned_worker_pending"])
        self.assertFalse(any(isinstance(item, dict) for item in result.items))
        worker.allow_retire.set()
        self.finished(result)
        self.assertTrue(worker.closed)
        self.assertFalse(self.h.backend.health()["owned_worker_pending"])

    def test_construction_health_and_two_requests_share_one_startup(self):
        self.assertEqual(self.h.backend.health()["backend"], "llama-vulkan-candidate")
        self.assertEqual(self.h.backend.health()["state"], "asleep")
        self.assertEqual((self.h.workers, self.h.clients, self.h.calls), ([], [], []))
        for _ in range(2):
            result = self.h.collect()
            self.finished(result)
            self.assertIsNone(result.error)
            self.assertEqual(result.items[0], "<p>ok</p>")
            self.assertEqual(result.items[-1]["usage"], {"prompt_tokens": 3617, "completion_tokens": 557, "total_tokens": 4174})
        self.assertEqual((len(self.h.workers), len(self.h.posts), self.h.preparations), (1, 2, 2))
        self.assertEqual(self.h.backend.health()["state"], "awake")
        for payload in self.h.posts:
            self.assertEqual(payload["model"], self.h.workers[0].alias)
            self.assertFalse(payload["cache_prompt"])
            self.assertEqual((payload["max_tokens"], payload["temperature"], payload["top_p"]), (12384, 0, 0.1))

    def test_sycl_health_remains_passive_and_unqualified(self):
        self.h.config.engine = "sycl"
        health = self.h.backend.health()
        self.assertEqual((health["backend"], health["phase"], health["qualified"]),
                         ("llama-sycl-candidate", "asleep", False))
        self.assertEqual((self.h.workers, self.h.clients, self.h.calls), ([], [], []))
        result = self.h.collect()
        self.finished(result)
        self.assertIsNone(result.error)
        self.assertEqual(result.items[-1]["finish_reason"], "stop")
        self.assertFalse(self.h.backend.health()["qualified"])
        self.assertEqual(self.h.preparations, 1)

    def test_final_metadata_waits_for_the_observed_idle_slot(self):
        self.h.slot_release.clear()
        first = self.h.collect()
        self.assertTrue(self.h.slot_waiting.wait(2))
        self.assertFalse(first.done.is_set())
        self.assertFalse(any(isinstance(item, dict) for item in first.items))
        second = self.h.collect()
        self.assertFalse(second.done.wait(0.1))
        self.assertEqual(len(self.h.posts), 1)
        self.h.slot_release.set()
        self.finished(first)
        self.finished(second)
        self.assertIsNone(first.error)
        self.assertIsNone(second.error)
        self.assertEqual(len(self.h.posts), 2)

    def test_cancel_during_blocked_read_waits_for_retirement(self):
        self.h.stream_mode = "blocked"
        result = self.h.collect()
        self.assertTrue(self.h.stream_blocked.wait(2))
        self.assertEqual(self.h.backend.health()["state"], "awake")
        self.assertEqual(self.h.backend.health()["phase"], "busy")
        self.h.workers[-1].allow_retire.clear()
        result.cancel.set()
        self.assert_retirement_retains_request(result)
        self.assertEqual(len(self.h.posts), 1)

    def test_cancel_after_content_waits_for_retirement(self):
        self.h.stream_mode = "after-content"
        result = self.h.collect()
        self.assertTrue(result.content.wait(2))
        self.assertTrue(self.h.stream_blocked.wait(2))
        self.h.workers[-1].allow_retire.clear()
        result.cancel.set()
        self.assert_retirement_retains_request(result)
        self.assertEqual(result.items, ["<p>ok</p>"])

    def test_live_memory_floor_trip_stops_and_retires_before_return(self):
        self.h.stream_mode = "blocked"
        result = self.h.collect()
        self.assertTrue(self.h.stream_blocked.wait(2))
        self.h.workers[-1].allow_retire.clear()
        self.h.memory["available_physical"] = 4 * GIB
        self.assert_retirement_retains_request(result)
        self.assertIsNotNone(result.error)
        self.assertIsNotNone(self.h.backend.health()["error"])

    def test_http_failure_is_not_replayed_or_classified_as_client_input(self):
        self.h.post_status = 500
        result = self.h.collect()
        self.finished(result)
        self.assertIsInstance(result.error, LlamaFailure)
        self.assertEqual(len(self.h.posts), 1)
        again = self.h.collect()
        self.finished(again)
        self.assertIsInstance(again.error, LlamaFailure)
        self.assertEqual(len(self.h.posts), 1)
        self.assertEqual(len(self.h.workers), 1)

    def test_full_output_context_allowance_refused_before_generation_post(self):
        self.h.input_tokens = 4000  # 4000 + unchanged 12384 reaches the fixed context limit.
        result = self.h.collect()
        self.finished(result)
        self.assertIsInstance(result.error, ValueError)
        self.assertEqual(self.h.posts, [])
        self.assertFalse(self.h.workers[0].closed)
        self.h.input_tokens = 3617
        recovered = self.h.collect()
        self.finished(recovered)
        self.assertIsNone(recovered.error)
        self.assertEqual((len(self.h.workers), len(self.h.posts)), (1, 1))

    def test_prelaunch_admission_refusal_can_recover_without_replay(self):
        self.h.memory["available_physical"] = 10 * GIB  # Forecast leaves exactly the reserve, so admission refuses.
        result = self.h.collect()
        self.finished(result)
        self.assertIsInstance(result.error, AdmissionRefused)
        self.assertEqual(self.h.workers, [])
        self.assertIsNone(self.h.backend.health()["error"])
        self.h.memory["available_physical"] = 20 * GIB
        result = self.h.collect()
        self.finished(result)
        self.assertIsNone(result.error)
        self.assertEqual(len(self.h.posts), 1)

    def test_unload_and_close_wait_for_confirmed_worker_retirement(self):
        result = self.h.collect()
        self.finished(result)
        for operation in (self.h.backend.unload, self.h.backend.close):
            worker = self.h.workers[-1]
            worker.allow_retire.clear()
            done = threading.Event()

            def run():
                try:
                    operation()
                finally:
                    done.set()

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(worker.retiring.wait(2))
            self.assertFalse(done.is_set())
            self.assertTrue(self.h.backend.health()["owned_worker_pending"])
            worker.allow_retire.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(self.h.backend.health()["owned_worker_pending"])
            if operation == self.h.backend.unload:
                result = self.h.collect()
                self.finished(result)
                self.assertIsNone(result.error)
        self.assertEqual(self.h.backend.health()["state"], "offline")
        self.assertEqual(self.h.backend.health()["phase"], "closed")


class ServiceCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("fake service condition did not arrive")
            await asyncio.sleep(0.01)

    async def test_passive_routes_do_not_construct_the_worker(self):
        h = Harness()
        self.addAsyncCleanup(asyncio.to_thread, h.close)
        app = create_app(h.backend, queue_limit=0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            self.assertEqual((await client.get("/health")).status_code, 200)
            self.assertEqual((await client.get("/v1/models")).json()["data"][0]["id"], "chandra")
        self.assertEqual((h.workers, h.clients, h.calls, h.preparations), ([], [], [], 0))

    async def test_cancel_keeps_service_capacity_until_retirement(self):
        h = Harness()
        self.addAsyncCleanup(asyncio.to_thread, h.close)
        h.stream_mode = "blocked"
        service = Worker(h.backend, queue_limit=0, inference_seconds=5)
        job = service.submit(request())
        self.assertIsNotNone(job)
        await self.until(h.stream_blocked.is_set)
        worker = h.workers[-1]
        worker.allow_retire.clear()
        job.cancel.set()
        await self.until(worker.retiring.is_set)
        self.assertFalse(job.done.is_set())
        self.assertEqual(len(service.jobs), 1)
        self.assertIsNone(service.submit(request()))
        worker.allow_retire.set()
        await self.until(lambda: job.done.is_set() and not service.jobs)
        self.assertTrue(worker.closed)
        self.assertFalse(h.backend.health()["owned_worker_pending"])
