"""CPU lifecycle tests for the native endpoint bridge through the real chandra_service app.

Fake lifecycle evidence only: FakeTransport/FakeWorker speak the worker protocol over OS pipes and
FakeProcessor is a byte-level stand-in. No pinned processor, tokenizer, worker build, Job or GPU runs
here; those are root-owned acceptance steps in docs/directcompute-native-endpoint.md.
"""
import asyncio
import collections
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import httpx

from runtime.native import NativeBackend, NativeFailure
from runtime.native import backend as native_backend
from runtime.native import protocol as P
from runtime.native.app import create_native_app
from runtime.native.prepare import RequestRefused
from tests.native.native_fakes import (PINNED, FakeProcessor, FakeTransport, Script, config, decode, payload, png)

ROOT = Path(__file__).resolve().parents[2]
UNICODE = [104, 1000, 1001, 108, 108, 111, 32, 1002, 1003, 1004, 1005, 248046]  # "héllo 😀 表\n"


def eventually(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    raise AssertionError("condition not reached within its bound")


class Harness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="chandra-native-endpoint-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.backend_count = 0

    def backend(self, script=None, processor=None, **overrides):
        self.backend_count += 1
        self.root = Path(self.directory.name) / f"backend-{self.backend_count}"
        self.root.mkdir()  # Distinct backends must never share still-owned shader/input roots.
        self.transport = FakeTransport(script or Script())
        self.processor = processor or FakeProcessor()
        self.config = config(self.root, **overrides)
        backend = NativeBackend(self.config, transport=self.transport, processor=self.processor, pinned=PINNED)
        self.addCleanup(self._close, backend, self.transport)
        return backend

    def _close(self, backend, transport):
        session = backend._session or backend._finalizing or (backend._cleanup_owner and backend._cleanup_owner.session)
        backend.close()
        transport.unkillable = False  # A test's failed-closure control ends at teardown; no production behavior changes.
        for launch in transport.launches:
            launch["process"].worker.kill()
            launch["process"].worker.thread.join(3)
            self.assertFalse(launch["process"].worker.thread.is_alive())
            reader = launch["process"].worker.reader_thread
            if reader is not None and reader.ident is not None:
                reader.join(3)
                self.assertFalse(reader.is_alive())
        if session is not None:
            for thread in session.threads:
                if thread.ident is not None:
                    thread.join(3)
                    self.assertFalse(thread.is_alive(), thread.name)

    def app(self, backend, **options):
        app = create_native_app(backend, **options)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(client.aclose)
        return app, client

    async def until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("condition not reached within its bound")

    def generate(self, backend, request=None, cancel=None):
        cancel = cancel or threading.Event()
        items = list(backend.generate(request or payload(), cancel))
        return "".join(i for i in items if isinstance(i, str)), [i for i in items if isinstance(i, dict)]


class DefaultPathTests(Harness):
    def test_import_starts_nothing_and_loads_no_runtime(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import runtime.native, runtime.native.app, chandra_service.__main__; "
                "print(sorted(m for m in ('torch', 'transformers', 'PIL', 'tokenizers', 'msvcrt') if m in sys.modules))")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    async def test_health_and_models_never_start_or_renew(self):
        backend = self.backend()
        app, client = self.app(backend)
        for _ in range(3):
            health = (await client.get("/health")).json()
            self.assertEqual((await client.get("/v1/models")).json()["data"][0]["id"], "chandra")
        self.assertEqual(health["state"], "cold")
        self.assertFalse(health["loaded"])
        self.assertEqual((health["qualified_OCR"], health["performance_claim"]), (False, False))
        self.assertEqual(self.transport.launches, [])
        self.assertEqual(self.processor.calls, 0)
        before = app.state.worker.last_activity
        await client.get("/health")
        self.assertEqual(app.state.worker.last_activity, before)

    def test_production_transport_refuses_without_windows_job(self):
        backend = NativeBackend(config(self.root), processor=FakeProcessor(), pinned=PINNED)
        with self.assertRaisesRegex(NativeFailure, "Windows Job"):
            self.generate(backend)
        self.assertEqual(backend.health()["state"], "failed")
        self.assertEqual(list((self.root / "inputs").iterdir()), [])
        with self.assertRaisesRegex(NativeFailure, "restart"):
            self.generate(backend)


class RequestTests(Harness):
    async def test_two_requests_share_one_startup_and_report_producer_facts(self):
        backend = self.backend(Script([[72, 105, 248046], [79, 75, 248044]]))
        _, client = self.app(backend)
        first = (await client.post("/v1/chat/completions", json=payload())).json()
        second = (await client.post("/v1/chat/completions", json=payload("second prompt"))).json()
        self.assertEqual(first["choices"][0]["message"]["content"], "Hi")
        self.assertEqual(second["choices"][0]["message"]["content"], "OK")
        self.assertEqual(first["choices"][0]["finish_reason"], "stop")
        prompt = len(self.transport.workers[0].packages[0]["manifest"]["image_token_positions"]) + 4 + len("ocr_layout") + 1
        self.assertEqual(first["usage"], {"prompt_tokens": prompt, "completion_tokens": 3, "total_tokens": prompt + 3})
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(len(self.transport.submits()), 2)
        health = (await client.get("/health")).json()
        self.assertEqual((health["state"], health["loaded"]), ("ready", True))
        self.assertEqual(health["worker"]["device_identity"]["pci_bdf"], "03:00.0")
        self.assertEqual(health["last_request"]["stop_token_id"], 248044)
        # Packages are removed after the terminal; only the live worker's held shader snapshot remains until confirmed close.
        snapshot = health["worker"]["shader_source"]["snapshot"]
        self.assertEqual([p.name for p in (self.root / "inputs").iterdir()], [snapshot])
        self.assertEqual(len(list((self.root / "outputs").iterdir())), 2)
        self.assertTrue(backend.close())
        self.assertEqual(list((self.root / "inputs").iterdir()), [])

    def test_concurrent_callers_share_a_single_startup(self):
        script = Script([[72, 248046]])
        gate = script.gate("0:0")
        backend = self.backend(script)
        results, errors = [], []

        def run():
            try:
                results.append(self.generate(backend))
            except Exception as error:  # pragma: no cover - reported below
                errors.append(error)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        eventually(lambda: len(self.transport.submits()) == 2)
        gate.set()
        for thread in threads:
            thread.join(10)
        self.assertEqual(errors, [])
        self.assertEqual([r[0] for r in results], ["H", "H"])
        self.assertEqual(len(self.transport.launches), 1)

    async def test_overflow_is_refused_without_submission(self):
        script = Script()
        gate = script.gate("0:0")
        backend = self.backend(script)
        app, client = self.app(backend, queue_limit=0)
        first = asyncio.create_task(client.post("/v1/chat/completions", json=payload()))
        await self.until(lambda: len(self.transport.submits()) == 1)
        overflow = await client.post("/v1/chat/completions", json=payload())
        self.assertEqual(overflow.status_code, 503)
        self.assertEqual((await client.get("/health")).json()["pending_or_active"], 1)
        gate.set()
        self.assertEqual((await first).status_code, 200)
        self.assertEqual(len(self.transport.submits()), 1)

    async def test_unicode_streaming_and_nonstreaming_agree(self):
        backend = self.backend(Script([UNICODE]))
        _, client = self.app(backend)
        expected = decode(UNICODE)
        self.assertEqual(expected, "héllo 😀 表\n")
        whole = (await client.post("/v1/chat/completions", json=payload())).json()
        self.assertEqual(whole["choices"][0]["message"]["content"], expected)
        streamed = await client.post("/v1/chat/completions", json=payload(stream=True, stream_options={"include_usage": True}))
        chunks = [json.loads(line[6:]) for line in streamed.text.split("\n\n") if line.startswith("data: {")]
        deltas = [c["choices"][0]["delta"].get("content") for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]
        self.assertEqual("".join(deltas), expected)
        self.assertTrue(all("�" not in delta for delta in deltas))
        self.assertGreater(len(deltas), 3)
        self.assertEqual(chunks[-1]["usage"], whole["usage"])
        self.assertTrue(streamed.text.endswith("data: [DONE]\n\n"))

    def test_length_finish_uses_the_full_allowance(self):
        backend = self.backend(Script([[65] * P.OUTPUT_ALLOWANCE]))
        text, metadata = self.generate(backend)
        self.assertEqual(text, "A" * P.OUTPUT_ALLOWANCE)
        self.assertEqual(metadata[0]["finish_reason"], "length")
        self.assertEqual(metadata[0]["usage"]["completion_tokens"], P.OUTPUT_ALLOWANCE)

    async def test_unsupported_requests_are_refused_before_submission(self):
        backend = self.backend(max_patch_rows=16)
        _, client = self.app(backend)
        swapped = payload()
        swapped["messages"][0]["content"].reverse()
        cases = [payload(max_tokens=100), swapped, payload(image=png(200, 200)), payload(image=png(fmt="JPEG")),
                 payload(image=png(frames=2)), payload(image=b"not an image")]
        for body in cases:
            response = await client.post("/v1/chat/completions", json=body)
            self.assertEqual(response.status_code, 400, response.text)
        long_prompt = self.backend(processor=FakeProcessor(prompt_padding=4000))
        with self.assertRaisesRegex(RequestRefused, "12384-token allowance"):
            self.generate(long_prompt)
        tight = self.backend(max_image_pixels=100)
        with self.assertRaisesRegex(RequestRefused, "pixel limit"):
            self.generate(tight)
        self.assertEqual(self.transport.launches, [])
        self.assertEqual(list((self.root / "inputs").iterdir()), [])

    def test_package_is_the_native_manifest_abi_and_authenticated(self):
        from scripts.native.prepare_input import positions
        backend = self.backend()
        self.generate(backend)
        package = self.transport.workers[0].packages[0]
        manifest = package["manifest"]
        self.assertEqual(package["files"], sorted(["attention_mask.raw", "chandra-caller-hf.py", "derived-position_ids.raw", "derived-rope_deltas.raw",
                                                   "derived-text_position_ids.raw", "generation_config.json", "image_grid_thw.raw", "input-manifest.json",
                                                   "input.png", "input_ids.raw", "mm_token_type_ids.raw", "pixel_values.raw", "prompt.txt",
                                                   "rendered_prompt.txt", "tokenizer_config.json"]))
        self.assertEqual(manifest["processor_profile"], "northstone-serving")
        self.assertEqual(manifest["processor_kwargs"], {"size": {"shortest_edge": 3136, "longest_edge": 3145728}})
        self.assertEqual(manifest["generation"]["stop_token_ids"], [248044, 248046])
        self.assertEqual(manifest["image"]["dimensions"], [64, 64])
        self.assertEqual(manifest["tensors"]["pixel_values"]["dtype"], "F32")
        ids = list(json.loads(json.dumps(manifest["image_token_positions"])))
        self.assertTrue(ids)
        n = manifest["prompt_tokens"]
        types = [1 if i in manifest["image_token_positions"] else 0 for i in range(n)]
        _, delta = positions([248056 if t else 1 for t in types], types, [1] * n, manifest["image_grid_thw"][0])
        self.assertEqual(manifest["positions"]["next_decode_position"], n + delta)
        snapshot = backend.health()["worker"]["shader_source"]["snapshot"]
        self.assertEqual([p.name for p in (self.root / "inputs").iterdir()], [snapshot])  # Request input is gone; the live worker's held snapshot is not.
        self.assertTrue(backend.close())
        self.assertEqual(list((self.root / "inputs").iterdir()), [])


class CancellationTests(Harness):
    async def test_disconnect_holds_capacity_until_released_then_survives(self):
        script = Script([[72, 105, 33, 248046], [79, 75, 248046]], token_delay=0.01)
        gate = script.gate("0:2")
        backend = self.backend(script)
        app, client = self.app(backend, queue_limit=0)
        disconnected = asyncio.Event()
        sent = []

        async def receive():
            if not sent:
                sent.append(1)
                return {"type": "http.request", "body": json.dumps(payload(stream=True)).encode(), "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and b'"content":"i"' in message.get("body", b""):
                disconnected.set()

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1), "server": ("test", 80)}
        request = asyncio.create_task(app(scope, receive, send))
        worker = await asyncio.to_thread(lambda: eventually(lambda: self.transport.workers and any(
            m["type"] == "cancel" for m in self.transport.workers[0].received)) and self.transport.workers[0])
        await asyncio.wait_for(request, 5)
        self.assertEqual((await client.get("/health")).json()["pending_or_active"], 1)
        self.assertEqual((await client.post("/v1/chat/completions", json=payload())).status_code, 503)
        gate.set()
        await self.until(lambda: not app.state.worker.jobs)
        survivor = await client.post("/v1/chat/completions", json=payload())
        self.assertEqual(survivor.json()["choices"][0]["message"]["content"], "OK")
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual([m["type"] for m in worker.received if m["type"] != "lease"], ["submit", "cancel", "submit"])

    async def test_timeout_holds_capacity_until_cancellation_is_released(self):
        script = Script([[72, 248046], [79, 75, 248046]], ignore_cancel=True)
        gate = script.gate("0:0")
        backend = self.backend(script)
        app, client = self.app(backend, queue_limit=0, inference_seconds=0.3)
        response = await client.post("/v1/chat/completions", json=payload())
        self.assertEqual(response.status_code, 504)
        await self.until(lambda: any(m["type"] == "cancel" for m in self.transport.workers[0].received))
        await asyncio.sleep(0.3)
        self.assertEqual((await client.get("/health")).json()["pending_or_active"], 1)
        script.options["ignore_cancel"] = False
        gate.set()
        await self.until(lambda: not app.state.worker.jobs)
        self.assertEqual((await client.post("/v1/chat/completions", json=payload())).status_code, 200)
        self.assertEqual(len(self.transport.launches), 1)

    def test_unconfirmed_cancellation_retires_the_whole_worker(self):
        script = Script(ignore_cancel=True)
        gate = script.gate("0:0")
        backend = self.backend(script, cancel_grace_seconds=0.3)
        cancel, items = threading.Event(), []

        def run():
            try:
                items.extend(backend.generate(payload(), cancel))
            except NativeFailure as error:
                items.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        eventually(lambda: len(self.transport.submits()) == 1)
        cancel.set()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(items[-1], NativeFailure)
        self.assertEqual(self.transport.launches[0]["process"].terminations, 1)
        self.assertEqual(backend.health()["state"], "failed")
        self.assertTrue(backend.health()["retirements"][-1]["confirmed"])
        self.assertFalse(gate.is_set())

    def test_unclean_cancellation_is_never_a_survivor(self):
        script = Script([[72, 105, 248046]], unclean_cancel=True)
        script.gate("0:1")
        backend = self.backend(script)
        cancel = threading.Event()
        threading.Thread(target=lambda: (eventually(lambda: self.transport.submits()), time.sleep(0.1), cancel.set())).start()
        with self.assertRaisesRegex(NativeFailure, "without proving release"):
            self.generate(backend, cancel=cancel)
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertTrue(health["retirements"][0]["confirmed"])
        with self.assertRaises(NativeFailure):
            self.generate(backend)
        self.assertEqual(len(self.transport.launches), 1)


class FailureTests(Harness):
    def assert_retired_without_replay(self, backend):
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertTrue(health["retirements"] and health["retirements"][-1]["confirmed"])
        self.assertEqual(len(self.transport.submits()), 1)
        with self.assertRaises(NativeFailure):
            self.generate(backend)
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(len(self.transport.submits()), 1)

    def test_malformed_or_inconsistent_channel_retires_the_worker(self):
        for failure in ("malformed", "duplicate_key", "seq_gap", "oversized", "index_skip", "wrong_stop"):
            with self.subTest(failure=failure):
                backend = self.backend(Script([[72, 105, 248046]], fail={0: (failure, 1)}))
                with self.assertRaises(NativeFailure):
                    self.generate(backend)
                self.assertGreaterEqual(self.transport.launches[0]["process"].terminations, 1)
                self.assert_retired_without_replay(backend)

    async def test_worker_crash_fails_without_replay(self):
        backend = self.backend(Script([[72, 105, 33, 248046]], fail={0: ("crash", 2)}))
        _, client = self.app(backend)
        streamed = await client.post("/v1/chat/completions", json=payload(stream=True))
        self.assertIn("backend_failure", streamed.text)
        self.assertNotIn("[DONE]", streamed.text)
        self.assertIn('"content":"H"', streamed.text)
        self.assertIn('"content":"i"', streamed.text)
        self.assertEqual((await client.post("/v1/chat/completions", json=payload())).status_code, 502)
        self.assert_retired_without_replay(backend)

    def test_poisoned_worker_is_retired_and_not_reused(self):
        backend = self.backend(Script([[72, 105, 248046]], fail={0: ("poison", 1)}))
        with self.assertRaisesRegex(NativeFailure, "without proving release"):
            self.generate(backend)
        self.assertEqual(backend.health()["retirements"][-1]["worker_exit_code"], 2)
        self.assert_retired_without_replay(backend)

    def test_receipts_are_authenticated_before_success(self):
        for failure in ("wrong_sha", "result_mismatch", "journal_mismatch"):
            with self.subTest(failure=failure):
                backend = self.backend(Script([[72, 105, 248046]], fail={0: (failure, None)}))
                items = []
                with self.assertRaisesRegex(NativeFailure, "receipt|result|journal"):
                    for item in backend.generate(payload(), threading.Event()):
                        items.append(item)
                self.assertFalse(any(isinstance(i, dict) for i in items))

    def test_ready_commitments_are_enforced(self):
        overrides = {"device_identity.pci_bdf": "04:00.0", "device_identity.device_id": 0x5690, "job.kill_on_job_close": False,
                     "job.process_memory_limit_bytes": None, "job.breakaway_allowed": True, "gemv_b1_selection": "ordered",
                     "resident_model.model_sha256": "0" * 64, "model.normal_output_allowance": 4096, "lease_ms": 2000,
                     "qualified_OCR": True, "model.caller_stop_token_ids": [248046]}
        for path, value in overrides.items():
            with self.subTest(path=path):
                backend = self.backend(Script(ready_override={path: value}))
                with self.assertRaisesRegex(NativeFailure, "Ready commitments refused"):
                    self.generate(backend)
                self.assertEqual(self.transport.submits(), [])
                worker = self.transport.workers[0]
                self.assertEqual(worker.exit_code, 0)
                self.assertEqual(backend.health()["state"], "failed")

    def test_startup_failure_and_hang_are_bounded(self):
        backend = self.backend(Script(startup_fatal=True))
        with self.assertRaisesRegex(NativeFailure, "refused startup"):
            self.generate(backend)
        backend = self.backend(Script(startup_hang=True), startup_seconds=0.3)
        started = time.monotonic()
        with self.assertRaisesRegex(NativeFailure, "startup bound"):
            self.generate(backend)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(self.transport.launches[0]["process"].terminations, 1)
        self.assertEqual(self.transport.submits(), [])

    def test_unkillable_worker_keeps_capacity_and_reports_unconfirmed(self):
        backend = self.backend(Script(startup_hang=True), startup_seconds=0.2, terminate_seconds=0.2)
        self.transport.unkillable = True
        with self.assertRaises(NativeFailure):
            self.generate(backend)
        eventually(lambda: backend.health()["worker"] and backend.health()["state"] == "retirement_unconfirmed")
        self.transport.unkillable = False
        self.transport.workers[0].kill()
        eventually(lambda: backend.health()["state"] == "failed" and backend.health()["retirements"])


class LifecycleTests(Harness):
    async def test_idle_retirement_reports_cold_only_after_closure_and_restarts_fresh(self):
        backend = self.backend(Script([[72, 248046]]))
        app, client = self.app(backend, idle_seconds=0.2)
        async with app.router.lifespan_context(app):
            self.assertEqual((await client.post("/v1/chat/completions", json=payload())).status_code, 200)
            worker = self.transport.workers[0]
            await self.until(lambda: worker.exit_code is not None, 10)
            await self.until(lambda: backend.health()["state"] == "cold", 5)
            health = (await client.get("/health")).json()
            self.assertEqual(health["retirements"][-1]["reason"], "idle_unload")
            self.assertTrue(health["retirements"][-1]["clean"])
            self.assertEqual(worker.exit_code, 0)
            self.assertIn({"schema": P.REQUEST_SCHEMA, "type": "shutdown"}, worker.received)
            self.assertGreaterEqual(self.processor.unloaded, 1)
            for _ in range(3):
                await client.get("/health")
            self.assertEqual(len(self.transport.launches), 1)
            self.assertEqual((await client.post("/v1/chat/completions", json=payload())).status_code, 200)
            self.assertEqual(len(self.transport.launches), 2)

    async def test_lease_is_maintained_during_long_requests(self):
        script = Script([[72, 105, 248046]])
        gate = script.gate("0:1")
        backend = self.backend(script, lease_ms=1000)
        _, client = self.app(backend)
        task = asyncio.create_task(client.post("/v1/chat/completions", json=payload()))
        await asyncio.sleep(2.0)
        gate.set()
        self.assertEqual((await task).status_code, 200)
        worker = self.transport.workers[0]
        self.assertGreaterEqual(sum(m["type"] == "lease" for m in worker.received), 3)
        self.assertIsNone(worker.closing)

    async def test_service_shutdown_closes_the_worker_cleanly(self):
        backend = self.backend()
        app, client = self.app(backend)
        async with app.router.lifespan_context(app):
            self.assertEqual((await client.post("/v1/chat/completions", json=payload())).status_code, 200)
        health = backend.health()
        self.assertEqual(health["state"], "closed")
        self.assertEqual(health["retirements"][-1]["reason"], "service_shutdown")
        self.assertTrue(health["retirements"][-1]["clean"])
        self.assertEqual(self.transport.workers[0].exit_code, 0)
        self.assertEqual(self.transport.launches[0]["process"].terminations, 0)

    def test_launch_uses_explicit_argv_environment_and_limits(self):
        backend = self.backend(gemv_b1_selection="ordered")
        import os
        os.environ["CHANDRA_NATIVE_CPU_TIMING"] = "1"
        self.addCleanup(os.environ.pop, "CHANDRA_NATIVE_CPU_TIMING", None)
        self.generate(backend)
        launch = self.transport.launches[0]
        self.assertEqual(launch["argv"][:2], [str(self.config.executable), "--execute"])
        self.assertIn("--lease-ms", launch["argv"])
        self.assertEqual(launch["env"].get("CHANDRA_EXPERIMENTAL_GEMV_B1"), "ordered")
        self.assertNotIn("CHANDRA_NATIVE_CPU_TIMING", launch["env"])
        self.assertEqual((launch["limits"].process_memory, launch["limits"].job_memory, launch["limits"].active_processes), (12 << 30, 13 << 30, 1))

    def test_changed_executable_is_refused_before_launch(self):
        backend = self.backend()
        self.config.executable.write_bytes(b"substituted build")
        with self.assertRaisesRegex(NativeFailure, "launch refused"):
            self.generate(backend)
        self.assertEqual(self.transport.launches, [])


if __name__ == "__main__":
    unittest.main()


class FailingStore(collections.deque):
    """Retirement receipt storage that raises MemoryError for its first `failures` appends."""

    def __init__(self, failures):
        super().__init__((), 8)
        self.failures, self.attempts = failures, 0

    def append(self, item):
        self.attempts += 1
        if self.failures > 0:
            self.failures -= 1
            raise MemoryError("injected receipt storage failure")
        super().append(item)


class PublicationFailureTests(Harness):
    """R4: a failed evidence publication never reports cold, never loses the closure receipt and never replays."""

    def started(self, script=None):
        backend = self.backend(script or Script([[72, 248046]]))
        self.assertEqual(self.generate(backend)[0], "H")
        process = self.transport.launches[0]["process"]
        confirmations = []
        original = process.confirm_closed

        def confirm():
            closure = original()
            if closure is not None:
                confirmations.append(closure)
            return closure
        process.confirm_closed = confirm
        return backend, process, confirmations

    def assert_owned_and_failed(self, backend, process, store):
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertEqual(health["failure"], native_backend.PUBLICATION_FAILED)
        self.assertFalse(health["loaded"])
        self.assertEqual(health["requests_in_backend"], 0)
        self.assertEqual(health["retirements"], [])
        worker = health["worker"]
        self.assertEqual(worker["pid"], process.pid)
        self.assertFalse(worker["published"])
        self.assertIn("MemoryError", worker["publication_error"])
        self.assertTrue(worker["retirement"]["confirmed"])
        self.assertEqual(worker["retirement"]["closure"]["job_active_processes"], 0)
        self.assertIn("MemoryError", worker["retirement"]["publication_error"])
        self.assertFalse(worker["retirement"]["request_replayed"])
        self.assertGreater(store.attempts, 0)
        return health

    def test_persistent_store_failure_at_idle_retirement_keeps_failed_ownership(self):
        backend, process, confirmations = self.started()
        store = backend._retirements = FailingStore(10 ** 9)
        with self.assertRaisesRegex(NativeFailure, "unpublished"):
            backend.unload()
        self.assertEqual(self.processor.unloaded, 0)  # Not cold: the processor stays until the evidence is accounted for.
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(self.transport.holds[0].releases, 1)
        attempts = store.attempts
        self.assert_owned_and_failed(backend, process, store)
        self.assertGreater(store.attempts, attempts)  # Health retried adoption and failed again without crashing.
        with self.assertRaisesRegex(NativeFailure, "failed earlier"):
            self.generate(backend)
        backend.unload()  # A later unload neither starts, clears nor reports cold.
        self.assert_owned_and_failed(backend, process, store)
        started = time.monotonic()
        self.assertFalse(backend.close())  # Bounded, and honest that the evidence is still unpublished.
        self.assertLess(time.monotonic() - started, 5)
        self.assert_owned_and_failed(backend, process, store)
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(len(self.transport.submits()), 1)
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(self.transport.holds[0].releases, 1)

    def test_transient_store_failure_publishes_a_failed_receipt_never_cold(self):
        backend, process, confirmations = self.started()
        store = backend._retirements = FailingStore(1)
        backend.unload()  # The second attempt stores the receipt; the backend stays failed.
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertEqual(health["failure"], native_backend.PUBLICATION_FAILED)
        self.assertIsNone(health["worker"])
        self.assertEqual(len(health["retirements"]), 1)
        receipt = health["retirements"][0]
        self.assertEqual(receipt["reason"], "idle_unload")
        self.assertTrue(receipt["confirmed"])
        self.assertIn("MemoryError", receipt["publication_error"])
        self.assertEqual(store.attempts, 2)
        self.assertEqual(len(confirmations), 1)
        with self.assertRaisesRegex(NativeFailure, "failed earlier"):
            self.generate(backend)
        self.assertTrue(backend.close())
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(len(self.transport.submits()), 1)

    def test_store_failure_while_a_caller_waits_fails_the_caller_before_closure(self):
        script = Script([[72, 105, 248046]])
        script.gate("0:1")
        backend = self.backend(script)
        store = backend._retirements = FailingStore(10 ** 9)
        seen = {}

        def run():
            try:
                self.generate(backend)
            except NativeFailure as error:
                seen["error"] = str(error)
                seen["failed"] = backend._failed  # Installed before the caller could observe closure.
        caller = threading.Thread(target=run)
        caller.start()
        eventually(lambda: self.transport.launches and len(self.transport.submits()) == 1)
        eventually(lambda: backend.health()["state"] == "running")
        process = self.transport.launches[0]["process"]
        process.worker.kill(crashed=True)
        caller.join(15)
        self.assertFalse(caller.is_alive())
        self.assertEqual(seen["failed"], native_backend.PUBLICATION_FAILED)
        health = self.assert_owned_and_failed(backend, process, store)
        self.assertFalse(health["worker"]["retirement"]["clean"])
        self.assertEqual(len(self.transport.submits()), 1)  # Never replayed.
        with self.assertRaisesRegex(NativeFailure, "failed earlier"):
            self.generate(backend)
        self.assertEqual(len(self.transport.launches), 1)

    def test_failure_after_the_closure_receipt_and_before_publication_reuses_that_receipt(self):
        backend, process, confirmations = self.started()
        original, calls = native_backend.LOG.log, []

        def failing_log(level, message, *args, **kwargs):
            if message.startswith("Native worker retired") and not calls:
                calls.append(1)
                raise MemoryError("injected failure after the closure receipt")
            return original(level, message, *args, **kwargs)
        native_backend.LOG.log = failing_log
        self.addCleanup(setattr, native_backend.LOG, "log", original)
        started = time.monotonic()
        backend.unload()  # Bounded: the supervisor retries with the cached receipt instead of hanging or dying.
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(calls, [1])
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertIn("without a clean drained exit", health["failure"])
        self.assertIn("supervisor failed: MemoryError", backend._retirements[0]["failure"])
        self.assertIsNone(health["worker"])
        receipt = health["retirements"][0]
        self.assertTrue(receipt["confirmed"])
        self.assertFalse(receipt["clean"])
        self.assertEqual(len(confirmations), 1)  # The closure receipt was neither lost nor re-confirmed.
        self.assertEqual(self.transport.holds[0].releases, 1)  # Custody was retired exactly once.
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(len(self.transport.submits()), 1)

    def test_failure_after_publication_while_waking_callers_stays_failed_and_refuses_admission(self):
        backend, process, confirmations = self.started()
        session = backend._session

        class WakeupFailure(dict):
            armed = failed = False

            def values(self):
                if self.armed and not self.failed:
                    self.failed = True
                    raise MemoryError("injected post-publication wakeup allocation failure")
                return super().values()
        requests = session.requests = WakeupFailure(session.requests)
        original = session.on_closed

        def publish_then_arm(closing, evidence):
            original(closing, evidence)
            requests.armed = True
        session.on_closed = publish_then_arm
        backend.unload()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (not requests.failed or session.threads[-1].is_alive()):
            time.sleep(0.02)
        self.assertTrue(requests.failed)
        self.assertFalse(session.threads[-1].is_alive())
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertTrue(health["failure"])
        self.assertTrue(any(not receipt["clean"] for receipt in health["retirements"]))
        self.assertTrue(health["retirements"][0]["clean"])  # The confirmed clean closure receipt is not obscured.
        with self.assertRaises(NativeFailure):
            list(backend.generate(payload(), threading.Event()))
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(self.transport.holds[0].releases, 1)
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))

    # ----- R4 finalization ownership: the backend owns a published session until its wake-up work is accounted for -----
    def arm_wakeup(self, session, block=None, then=None):
        """After the clean publication, the first wake-up list allocation raises MemoryError, or waits on `block`."""
        class Wakeup(dict):
            armed = failed = False

            def values(self):
                if self.armed and not self.failed:
                    self.failed = True
                    if block is None:
                        raise MemoryError("injected post-publication wakeup allocation failure")
                    block[0].set()
                    block[1].wait(10)
                return super().values()
        requests = session.requests = Wakeup(session.requests)
        original = session.on_closed

        def publish_then_arm(closing, evidence):
            original(closing, evidence)
            requests.armed = True
            if then is not None:
                then()
        session.on_closed = publish_then_arm
        return requests

    def hold_recovery(self, session, requests):
        """Deterministic interval: the supervisor's first poll after the wake-up failure waits here, holding no lock."""
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original = session.process.poll

        def poll():
            if requests.failed and not entered.is_set():
                entered.set()
                release.wait(10)
            return original()
        session.process.poll = poll
        return entered, release

    def assert_failed_during_finalization(self, backend, process, failure):
        health = backend.health()
        self.assertEqual(health["state"], "failed", "a failure known to the session must be the backend's before any retry")
        self.assertEqual(health["failure"], native_backend.PUBLICATION_FAILED)
        self.assertFalse(health["loaded"])
        self.assertIsNone(health["worker"])
        self.assertEqual(health["finalization"], {"pid": process.pid, "finalized": False, "failure": failure})
        self.assertEqual(len(health["retirements"]), 1)  # The original confirmed clean receipt stays observable.
        self.assertTrue(health["retirements"][0]["clean"])
        self.assertTrue(health["retirements"][0]["confirmed"])
        with self.assertRaisesRegex(NativeFailure, "failed earlier"):
            self.generate(backend, payload("explicit new request while the post-publication failure is unrecovered"))
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))
        return health

    def assert_recovered_failed(self, backend, session, confirmations):
        eventually(lambda: not session.threads[-1].is_alive(), timeout=10)
        self.assertTrue(session.finalized)
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertIsNone(health["finalization"])
        self.assertTrue(health["retirements"][0]["clean"])
        self.assertTrue(health["retirements"][0]["confirmed"])
        with self.assertRaisesRegex(NativeFailure, "failed earlier"):
            self.generate(backend)
        self.assertTrue(backend.close())
        self.assertEqual(len(confirmations), 1)  # Closure confirmed and custody retired once, never repeated by the retry.
        self.assertEqual(self.transport.holds[0].releases, 1)
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))
        return health

    def test_ungated_wakeup_failure_refuses_admission_inside_the_ordinary_retry_interval(self):
        backend, process, confirmations = self.started()
        session = backend._session
        self.arm_wakeup(session)
        backend.unload()
        eventually(lambda: session.failure == "Native supervisor failed: MemoryError")
        recovering = session.threads[-1].is_alive() and not session.finalized
        self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
        self.assertTrue(recovering and session.threads[-1].is_alive(), "observed before the supervisor's one-second retry")
        health = self.assert_recovered_failed(backend, session, confirmations)
        self.assertFalse(health["retirements"][-1]["clean"])
        self.assertIn("after its retirement was published", health["failure"])

    def test_failure_interval_is_owned_while_supervisor_recovery_is_held(self):
        backend, process, confirmations = self.started()
        session = backend._session
        requests = self.arm_wakeup(session)
        entered, release = self.hold_recovery(session, requests)
        backend.unload()
        eventually(lambda: session.failure is not None)
        self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
        self.assertTrue(entered.wait(5))  # The retry is now in progress and cannot finish until released.
        self.assertTrue(session.threads[-1].is_alive())
        self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
        backend.unload()  # Neither starts, clears nor reports cold.
        self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
        release.set()
        health = self.assert_recovered_failed(backend, session, confirmations)
        self.assertFalse(health["retirements"][-1]["clean"])

    def test_failure_handler_holding_the_session_condition_neither_hides_failure_nor_blocks_health(self):
        backend, process, confirmations = self.started()
        session = backend._session
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original_fail = session._fail_locked

        def fail_then_hold(reason):
            original_fail(reason)
            if reason == "Native supervisor failed: MemoryError":
                entered.set()
                release.wait(10)
        session._fail_locked = fail_then_hold
        self.arm_wakeup(session)
        unloader = threading.Thread(target=backend.unload)  # It may wait on the session condition the handler holds.
        unloader.start()
        self.assertTrue(entered.wait(5))
        started = time.monotonic()
        self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(release.is_set())
        release.set()
        unloader.join(10)
        self.assertFalse(unloader.is_alive())
        self.assert_recovered_failed(backend, session, confirmations)

    def test_new_request_waits_for_clean_finalization_then_starts_a_fresh_worker(self):
        backend, process, confirmations = self.started()
        session = backend._session
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.arm_wakeup(session, block=(entered, release))
        unloader = threading.Thread(target=backend.unload)
        unloader.start()
        self.assertTrue(entered.wait(5))  # Published; its wake-up work is in progress under the session condition.
        health = backend.health()
        self.assertEqual(health["state"], "retiring")
        self.assertIsNone(health["failure"])
        self.assertEqual(health["finalization"], {"pid": process.pid, "finalized": False, "failure": None})
        waiting, original_wait, results = threading.Event(), session.wait_finalized, {}

        def wait_finalized(seconds):
            waiting.set()
            return original_wait(seconds)
        session.wait_finalized = wait_finalized
        caller = threading.Thread(target=lambda: results.update(output=self.generate(backend)[0]))
        caller.start()
        self.assertTrue(waiting.wait(5))
        self.assertEqual(len(self.transport.launches), 1)  # No second worker while the first is finalizing.
        release.set()
        caller.join(15)
        unloader.join(10)
        self.assertFalse(caller.is_alive() or unloader.is_alive())
        self.assertEqual(results["output"], "H")
        health = backend.health()
        self.assertIsNone(health["failure"])
        self.assertIsNone(health["finalization"])
        self.assertTrue(health["retirements"][0]["clean"])
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (2, 2))
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(self.transport.holds[0].releases, 1)

    # ----- R4 bounded finalization waits: a held Session.cond never extends a deadline -----
    def test_contended_idle_tick_and_health_do_not_wait_or_retire_an_active_successor(self):
        backend, process, confirmations = self.started()
        session = backend._session
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        seen = {}

        def hold():
            with session.cond:
                entered.set()
                if not release.wait(18):
                    raise AssertionError("finite current-session condition gate expired")
        holder = threading.Thread(target=hold)
        unloader = threading.Thread(target=backend.unload)
        reader = threading.Thread(target=lambda: seen.update(health=backend.health()))
        try:
            holder.start()
            self.assertTrue(entered.wait(5))
            unloader.start()
            reader.start()
            unloader.join(.5)
            reader.join(.5)
            self.assertFalse(unloader.is_alive() or reader.is_alive())
            self.assert_condition_held(session)
            self.assertTrue(backend._lock.acquire(blocking=False))
            backend._lock.release()
            health = seen["health"]
            self.assertEqual((health["state"], health["loaded"]), ("busy", False))
            self.assertTrue(health["worker"]["snapshot_contended"])
            self.assertIsNone(health["worker"]["requests"])
            self.assertIsNone(session.retire_reason)
            self.assertEqual((len(confirmations), self.transport.holds[0].releases), (0, 0))
            self.assertEqual(self.processor.unloaded, 0)
        finally:
            release.set()
            self.join_all(holder, unloader, reader)
        self.assertEqual(self.generate(backend, payload("explicit successor after skipped idle tick"))[0], "H")
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 2))
        backend.unload()
        self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))

    def test_unpublished_failure_handler_cannot_block_idle_health_or_close(self):
        backend, process, confirmations = self.started()
        session = backend._session
        store = backend._retirements = FailingStore(10 ** 9)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original_fail, original_callback = session._fail_locked, session.on_closed
        requests = self.arm_wakeup(session)

        def publish_then_arm_even_on_failure(closing, evidence):
            try:
                original_callback(closing, evidence)
            finally:
                requests.armed = True
        session.on_closed = publish_then_arm_even_on_failure

        def fail_then_hold(reason):
            original_fail(reason)
            if reason == "Native supervisor failed: MemoryError":
                entered.set()
                if not release.wait(18):
                    raise AssertionError("finite unpublished-failure condition gate expired")
        session._fail_locked = fail_then_hold
        seen = {}

        def unload():
            try:
                backend.unload()
            except NativeFailure as error:
                seen["original_unload"] = str(error)
        first = threading.Thread(target=unload)
        second = threading.Thread(target=backend.unload)
        reader = threading.Thread(target=lambda: seen.update(health=backend.health()))
        closer = threading.Thread(target=lambda: seen.update(closed=backend.close()))
        try:
            first.start()
            self.assertTrue(entered.wait(5))
            first.join(2)
            self.assertFalse(first.is_alive())
            self.assertIn("unpublished", seen["original_unload"])
            original_receipt = session.closure
            for thread in (second, reader, closer):
                thread.start()
                thread.join(.5)
                self.assertFalse(thread.is_alive(), "held unpublished failure handler must not strand backend management")
            self.assert_condition_held(session)
            self.assertFalse(seen["closed"])
            health = seen["health"]
            self.assertEqual((health["state"], health["failure"]), ("failed", native_backend.PUBLICATION_FAILED))
            self.assertTrue(health["worker"]["snapshot_contended"])
            self.assertTrue(health["worker"]["retirement"]["confirmed"])
            self.assertIs(backend._session, session)
            self.assertFalse(session.published)
            self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))
            self.assertEqual(self.processor.unloaded, 0)
            with self.assertRaisesRegex(NativeFailure, "closed"):
                self.generate(backend)
            # Adoption can win while the failure handler still holds cond; it transfers, rather than drops, ownership.
            store.failures = 0
            health = backend.health()
            self.assertIsNone(health["worker"])
            self.assertEqual(health["finalization"]["failure"], "Native supervisor failed: MemoryError")
            self.assertIs(backend._finalizing, session)
            self.assertFalse(backend.close())
            self.assertIs(session.closure, original_receipt)
        finally:
            release.set()
            self.join_all(first, second, reader, closer)
        eventually(lambda: session.finalized)
        self.assertTrue(backend.close())
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))
        self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))

    def test_publication_race_keeps_old_owner_until_new_explicit_request_can_launch(self):
        backend, process, confirmations = self.started()
        session = backend._session
        entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original_callback, original_wait = session.on_closed, session.wait_closed
        seen = {}

        def hold_publication(closing, evidence):
            entered.set()
            if not release.wait(18):
                raise AssertionError("finite publication gate expired")
            original_callback(closing, evidence)
        session.on_closed = hold_publication

        def wait_closed(seconds=None):
            if threading.current_thread() is caller:
                waiting.set()
            return original_wait(seconds)
        session.wait_closed = wait_closed
        unloader = threading.Thread(target=backend.unload)
        caller = threading.Thread(target=lambda: seen.update(output=self.generate(backend, payload("explicit publication successor"))[0]))
        try:
            unloader.start()
            self.assertTrue(entered.wait(5))
            caller.start()
            self.assertTrue(waiting.wait(5))
            health = backend.health()
            self.assertEqual(health["state"], "retiring")
            self.assertIs(backend._session, session)
            self.assertFalse(session.closed or session.published)
            self.assertTrue(caller.is_alive())
            self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))
            self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))
        finally:
            release.set()
            self.join_all(unloader, caller)
        self.assertEqual(seen["output"], "H")
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (2, 2))
        self.assertTrue(backend.health()["retirements"][0]["confirmed"])

    def test_cancellation_with_failed_closure_keeps_capacity_inputs_and_custody_during_contention(self):
        script = Script(ignore_cancel=True)
        native_gate = script.gate("0:0")
        backend = self.backend(script, cancel_grace_seconds=.2, terminate_seconds=.2, shutdown_seconds=.2)
        cancel, closure_allowed, closure_failed = threading.Event(), threading.Event(), threading.Event()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.addCleanup(closure_allowed.set)
        self.addCleanup(native_gate.set)
        seen = {}

        def generate():
            try:
                self.generate(backend, cancel=cancel)
            except NativeFailure as error:
                seen["error"] = str(error)
        caller = threading.Thread(target=generate)
        holder = reader = None
        try:
            caller.start()
            eventually(lambda: len(self.transport.submits()) == 1)
            session, process = backend._session, self.transport.launches[0]["process"]
            original_confirm = process.confirm_closed

            def confirm():
                if not closure_allowed.is_set():
                    closure_failed.set()
                    raise OSError("injected unaccounted process/Job closure")
                return original_confirm()
            process.confirm_closed = confirm
            cancel.set()
            self.assertTrue(closure_failed.wait(5))

            def hold():
                with session.cond:
                    entered.set()
                    if not release.wait(18):
                        raise AssertionError("finite failed-closure condition gate expired")
            holder = threading.Thread(target=hold)
            holder.start()
            self.assertTrue(entered.wait(5))
            reader = threading.Thread(target=lambda: seen.update(health=backend.health()))
            reader.start()
            reader.join(.5)
            self.assertFalse(reader.is_alive())
            backend.unload()  # An active cancellation still owns capacity; an idle tick must not release anything.
            self.assertTrue(caller.is_alive())
            self.assertEqual(backend._active, 1)
            self.assertIs(backend._session, session)
            self.assertFalse(session.closed)
            self.assertIsNone(session.closure)
            self.assertEqual(self.transport.holds[0].releases, 0)
            self.assertTrue(list(Path(backend.config.input_root).glob("p-*")))
            self.assertTrue(seen["health"]["worker"]["snapshot_contended"])
            self.assertFalse(seen["health"]["loaded"])
            self.assertFalse(session.wait_closed(.15))
            self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))
        finally:
            release.set()
            closure_allowed.set()
            self.join_all(holder, reader, caller)
        self.assertIn("not replayed", seen["error"])
        self.assertEqual(backend._active, 0)
        self.assertEqual(self.transport.holds[0].releases, 1)
        self.assertEqual(backend.health()["state"], "failed")
        self.assertEqual(list(Path(backend.config.input_root).glob("p-*")), [])

    def assert_condition_held(self, session):
        """The gate thread still owns Session.cond: this thread cannot take it."""
        self.assertFalse(session.cond.acquire(timeout=0))

    def join_all(self, *threads):
        for thread in threads:
            if thread is not None and thread.ident is not None:
                thread.join(15)
        self.assertFalse(any(thread is not None and thread.is_alive() for thread in threads))

    def test_finalization_waits_obey_their_deadlines_while_clean_wakeup_holds_the_condition(self):
        backend, process, confirmations = self.started()
        session = backend._session
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.arm_wakeup(session, block=(entered, release))
        original_wait, timed, results = session.wait_finalized, {}, {}

        def short_admission_wait(seconds):  # The caller's own bound, shortened so the test observes it.
            started = time.monotonic()
            outcome = original_wait(0.2)
            results.setdefault("admission_wait", (outcome, time.monotonic() - started))
            return outcome
        unloader = threading.Thread(target=backend.unload)
        waiter = refused = caller = None
        try:
            unloader.start()
            self.assertTrue(entered.wait(5))  # Published, receipt stored; its wake-up step holds the condition.

            def timed_wait():
                started = time.monotonic()
                timed["finalized"] = original_wait(0.15)
                timed["closed"] = session.wait_closed(0.15)
                timed["elapsed"] = time.monotonic() - started
            waiter = threading.Thread(target=timed_wait)
            waiter.start()
            waiter.join(1.0)
            self.assertFalse(waiter.is_alive(), "wait_finalized(0.15) must return while Session.cond is still held")
            self.assert_condition_held(session)
            self.assertEqual((timed["finalized"], timed["closed"]), (False, True))
            self.assertGreaterEqual(timed["elapsed"], 0.15)
            self.assertLess(timed["elapsed"], 0.6)
            session.wait_finalized = short_admission_wait
            refused = threading.Thread(target=lambda: results.update(refused=self._refuse(backend)))
            refused.start()
            refused.join(1.5)
            self.assertFalse(refused.is_alive(), "admission must return within its finalization bound")
            self.assert_condition_held(session)
            self.assertEqual(results["refused"], "Native worker retirement is still finalizing; no new worker was started")
            self.assertFalse(results["admission_wait"][0])
            self.assertLess(results["admission_wait"][1], 0.6)
            health = backend.health()
            self.assertEqual(health["state"], "retiring")  # Truthful while unresolved: neither cold nor failed.
            self.assertIsNone(health["failure"])
            self.assertEqual(health["finalization"], {"pid": process.pid, "finalized": False, "failure": None})
            self.assertTrue(health["retirements"][0]["clean"])
            self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))
            session.wait_finalized = original_wait
            caller = threading.Thread(target=lambda: results.update(output=self.generate(backend)[0]))
            caller.start()
            release.set()
            self.join_all(unloader, waiter, refused, caller)
        finally:
            release.set()
            self.join_all(unloader, waiter, refused, caller)
        self.assertTrue(session.finalized)
        self.assertEqual(results["output"], "H")  # The explicit new request starts one fresh worker after finalization.
        health = backend.health()
        self.assertIsNone(health["failure"])
        self.assertIsNone(health["finalization"])
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (2, 2))
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(self.transport.holds[0].releases, 1)

    def _refuse(self, backend):
        try:
            self.generate(backend, payload("explicit new request during a held clean finalization"))
        except NativeFailure as error:
            return str(error)
        return "admitted"

    def test_close_returns_false_within_its_bound_while_a_failure_handler_holds_the_condition(self):
        backend, process, confirmations = self.started()
        session = backend._session
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original_fail = session._fail_locked

        def fail_then_hold(reason):
            original_fail(reason)
            if reason == "Native supervisor failed: MemoryError":
                entered.set()
                release.wait(15)
        session._fail_locked = fail_then_hold
        self.arm_wakeup(session)
        unloader = threading.Thread(target=backend.unload)
        closer, seen = None, {}

        def close():
            started = time.monotonic()
            seen["returned"] = backend.close()
            seen["elapsed"] = time.monotonic() - started
        try:
            unloader.start()
            self.assertTrue(entered.wait(5))  # Published; the failure handler records MemoryError and keeps the condition.
            self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
            closer = threading.Thread(target=close)
            closer.start()
            closer.join(2.0)  # The configured bound is 17 s; a held condition must not be needed to return False.
            self.assertFalse(closer.is_alive(), "close must not wait to acquire the condition the failure handler holds")
            self.assert_condition_held(session)
            self.assertIs(seen["returned"], False)
            self.assertLess(seen["elapsed"], 1.0)
            health = backend.health()
            self.assertEqual(health["state"], "failed")
            self.assertEqual(health["failure"], native_backend.PUBLICATION_FAILED)
            self.assertEqual(health["finalization"], {"pid": process.pid, "finalized": False, "failure": "Native supervisor failed: MemoryError"})
            self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))
            release.set()
            self.join_all(unloader, closer)
        finally:
            release.set()
            self.join_all(unloader, closer)
        eventually(lambda: not session.threads[-1].is_alive(), timeout=10)
        self.assertTrue(session.finalized)
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertIsNone(health["finalization"])
        self.assertTrue(health["retirements"][0]["clean"])  # The original confirmed clean receipt stays first.
        self.assertTrue(backend.close())  # Finalization now finished; closure and custody were not repeated.
        self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))
        self.assertEqual((len(self.transport.launches), len(self.transport.submits())), (1, 1))

    def test_repeated_supplementary_append_failures_keep_the_preallocated_failure(self):
        backend, process, confirmations = self.started()
        session = backend._session

        class FailAfterFirst(collections.deque):
            attempts = 0

            def append(self, receipt):
                self.attempts += 1
                if self.attempts > 1:
                    raise MemoryError("injected supplementary receipt append failure")
                super().append(receipt)
        store = backend._retirements = FailAfterFirst((), 8)
        requests = self.arm_wakeup(session)
        entered, release = self.hold_recovery(session, requests)
        backend.unload()
        self.assertTrue(entered.wait(5))
        self.assert_failed_during_finalization(backend, process, "Native supervisor failed: MemoryError")
        release.set()
        eventually(lambda: not session.threads[-1].is_alive(), timeout=10)
        for _ in range(20):
            backend._on_closed(session, session.evidence)
            health = backend.health()
            self.assertEqual(health["state"], "failed")
            self.assertEqual(health["failure"], native_backend.PUBLICATION_FAILED)
            self.assertEqual(len(health["retirements"]), 1)
        self.assertEqual(store.attempts, 22)
        self.assert_recovered_failed(backend, session, confirmations)

    def test_message_construction_failure_keeps_the_preallocated_failure(self):
        backend, process, confirmations = self.started()
        session = backend._session
        backend.unload()
        self.assertEqual(backend.health()["state"], "cold")

        class Unprintable:
            def __str__(self):
                raise MemoryError("injected failure message allocation failure")
        backend._on_closed(session, {"expected": False, "unusable": Unprintable()})
        self.assertEqual(backend.health()["failure"], native_backend.PUBLICATION_FAILED)
        with self.assertRaisesRegex(NativeFailure, "failed earlier"):
            self.generate(backend)
        self.assertEqual((len(self.transport.launches), len(confirmations), self.transport.holds[0].releases), (1, 1, 1))

    def test_supervisor_lost_before_finishing_finalization_is_failed_without_a_retry(self):
        backend, process, confirmations = self.started()
        session = backend._session
        stopped = []
        self.addCleanup(setattr, threading, "excepthook", threading.excepthook)
        threading.excepthook = lambda args: stopped.append(args.exc_type)
        original_clock = session.clock

        def lose_supervisor():
            def clock():
                if threading.current_thread() is session.threads[-1]:
                    raise MemoryError("injected failure while recording the supervisor error")
                return original_clock()
            session.clock = clock
        self.arm_wakeup(session, then=lose_supervisor)
        backend.unload()
        eventually(lambda: not session.threads[-1].is_alive())
        self.assertEqual(stopped, [MemoryError])
        self.assertIsNone(session.failure)  # The session never recorded it; the backend still must not report cold.
        self.assert_failed_during_finalization(backend, process, None)
        self.assertFalse(backend.close())  # Bounded and honest: the wake-up work never finished.
        self.assertEqual((len(confirmations), self.transport.holds[0].releases), (1, 1))
