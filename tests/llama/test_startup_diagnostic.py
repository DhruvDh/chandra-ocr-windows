"""Real private-file persistence with fake Jobs/pipes and in-memory HTTP; no Windows/model."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from runtime.llama.startup_diagnostic import FILE_NAME, MAX_BYTES, StartupDiagnostic, StartupDiagnosticError
from runtime.llama.worker import AdmissionRefused, OwnedWorker, TAIL_BYTES
from runtime.native.winjob import ContainmentError
from tests.llama.test_worker import CleanupOwner, Config, Job, Transport
from tests.llama.test_backend import Harness, request


class StartupEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / FILE_NAME
        self.diagnostic = StartupDiagnostic(self.path)
        self.events = []
        self.job = Job(self.events)
        self.config = Config(self.events)
        self.transport = Transport(self.events, self.job)
        self.worker = OwnedWorker(self.config, self.transport, diagnostic=self.diagnostic)
        self.addCleanup(self.close_worker)

    def close_worker(self):
        self.job.finish()
        self.worker.retire()

    def read(self):
        raw = self.path.read_bytes()
        self.assertLessEqual(len(raw), MAX_BYTES)
        self.assertEqual(raw.count(b"\n"), 1, "embedded output must stay JSON escaped")
        return json.loads(raw)

    def test_atomic_bounded_startup_tails_survive_object_loss_without_credentials(self):
        key = "PRIVATE-API-KEY-0123456789"
        self.job.data["stdout"] = b"discarded\n" + b"x" * (TAIL_BYTES * 2) + b"\nlast out\x00\n"
        # A key split across the existing 8192-byte pipe reads must not leak a prefix.
        self.job.data["stderr"] = b"first\n" + b"y" * 8170 + key.encode() + b"\nlast err\n"
        samples, write = [], self.diagnostic._write
        def retained():
            write()
            samples.append(self.path.read_bytes())
        with patch.object(self.diagnostic, "_write", side_effect=retained):
            self.worker.start(18003, key, "PRIVATE-ALIAS")
            self.job.finish()
            self.worker.retire()
        saved = self.read()
        self.assertEqual((saved["startup_stage"], saved["retirement_stage"], saved["native_pid"]),
                         ("native_started", "closed", 1234))
        self.assertTrue(saved["closed"] and saved["closure"]["handles_closed"] and saved["held_files_released"])
        self.assertIn("last out\0", saved["stdout_tail"])
        self.assertIn("last err", saved["stderr_tail"])
        self.assertTrue(all(key[:12].encode() not in raw and b"PRIVATE-ALIAS" not in raw for raw in samples))
        self.assertNotIn("api-key", self.path.read_text())
        self.assertEqual(set(p.name for p in self.path.parent.iterdir()), {FILE_NAME})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.worker._tails = {"stdout": b"", "stderr": b""}  # Evidence no longer depends on RAM tails.
        self.assertEqual(self.read(), saved)

    def test_authentication_and_second_admission_are_distinct_and_refusal_never_launches(self):
        self.config.auth_hook = lambda: self.assertEqual(self.read()["startup_stage"], "authenticate_inputs")
        def refuse():
            self.assertEqual(self.read()["startup_stage"], "second_admission")
            raise AdmissionRefused("secret error text must not be dumped")
        with self.assertRaises(AdmissionRefused):
            self.worker.start(18003, "key", "alias", admit=refuse)
        self.worker.retire()
        saved = self.read()
        self.assertEqual(saved["startup_stage"], "second_admission")
        self.assertIsNone(saved["native_pid"])
        self.assertEqual(self.transport.launches, 0)
        self.assertEqual(self.transport.held.releases, 1)
        self.assertNotIn("secret", self.path.read_text())

    def test_stop_while_authentication_blocks_is_durable_before_start_returns(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        self.config.auth_hook = lambda: (entered.set(), release.wait(3))
        def start():
            try:
                self.worker.start(18003, "key", "alias")
            except RuntimeError as error:
                errors.append(error)
        starter = threading.Thread(target=start)
        retire = threading.Thread(target=self.worker.retire)
        starter.start()
        try:
            self.assertTrue(entered.wait(1))
            self.worker.request_stop()
            # The checkpoint is readable even before blocked startup/retirement can complete.
            self.assertEqual(self.read()["startup_stage"], "authenticate_inputs")
            retire.start()
            release.set()
            starter.join(2)
            retire.join(2)
            self.assertFalse(starter.is_alive() or retire.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertTrue(self.read()["closed"])
            self.assertEqual(self.transport.launches, 0)
        finally:
            release.set()
            starter.join(3)
            if retire.ident is not None:
                retire.join(3)

    def test_postlaunch_sink_failure_stops_then_drains_and_releases_actual_owner(self):
        record = self.diagnostic.record
        def fail(snapshot, **options):
            if options.get("stage") == "native_started":
                raise StartupDiagnosticError("injected private sink failure")
            return record(snapshot, **options)
        with patch.object(self.diagnostic, "record", side_effect=fail):
            with self.assertRaises(StartupDiagnosticError):
                self.worker.start(18003, "key", "alias")
        self.assertEqual(self.transport.launches, 1)
        self.assertTrue(self.worker._stop.is_set())
        self.assertEqual(self.transport.held.releases, 0)
        self.job.finish()
        self.worker.retire()
        saved = self.read()
        self.assertEqual(saved["native_pid"], 1234)
        self.assertTrue(saved["closed"] and saved["fault_present"])
        self.assertEqual(self.job.observed_eof, {"stdout", "stderr"})
        self.assertEqual(self.transport.held.releases, 1)

    def test_sink_failure_before_launch_preserves_prior_record_and_cleanup(self):
        before = self.path.read_bytes()
        with patch("runtime.llama.startup_diagnostic.os.replace", side_effect=OSError("secret path")):
            with self.assertRaisesRegex(StartupDiagnosticError, "write failed"):
                self.worker.start(18003, "key", "alias")
            with self.assertLogs("chandra.llama.startup", level="ERROR"):
                self.worker.retire()
        self.assertEqual(self.transport.launches, 0)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.diagnostic.temp.exists())

    def test_partial_launch_pid_survives_completed_cleanup_and_resets_for_next_generation(self):
        for pending, pid in ((False, 777), (True, 888)):
            with self.subTest(pending=pending), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / FILE_NAME
                diagnostic = StartupDiagnostic(path)
                owner = CleanupOwner() if pending else None
                if owner:
                    owner.pid = pid
                    owner.allow.set()
                transport = Transport([], Job([]))
                transport.failure = ContainmentError("secret exception", cleanup_owner=owner,
                    evidence={"pid": pid, "child_created": True, "child_retired": not pending,
                              "handles_closed": not pending, "private": "secret payload"})
                worker = OwnedWorker(Config([]), transport, diagnostic=diagnostic)
                with self.assertRaises(ContainmentError):
                    worker.start(18003, "key", "alias")
                worker.retire()
                saved = json.loads(path.read_bytes())
                self.assertEqual(saved["native_pid"], pid)
                self.assertTrue(saved["closed"])
                self.assertEqual(saved["launch_evidence"]["child_retired"], not pending)
                if pending:
                    self.assertTrue(saved["cleanup"]["handles_closed"])
                self.assertNotIn("secret", path.read_text())
                diagnostic.begin("b" * 64, "new-key", "new-alias")
                diagnostic.record({"pid": None}, stage="hold_inputs")
                self.assertIsNone(json.loads(path.read_bytes())["native_pid"])

    def test_first_failed_closure_and_successful_retry_are_both_durable(self):
        self.job.close_allowed.clear()
        retry = self.job.retry_cleanup
        def close_retry():
            self.job.close_allowed.set()
            return retry()
        self.job.retry_cleanup = close_retry
        self.worker.start(18003, "key", "alias")
        self.job.finish()
        self.worker.retire()
        saved = self.read()
        self.assertFalse(saved["closure"]["handles_closed"])
        self.assertTrue(saved["cleanup"]["handles_closed"])
        self.assertTrue(saved["closed"] and saved["held_files_released"])

    def test_frozen_startup_excerpt_cannot_capture_later_request_or_response(self):
        self.diagnostic.begin("a" * 64, "private-key", "alias")
        snapshot = {"pid": 1234, "stdout_tail": "startup\n", "stderr_tail": "private-key\n"}
        self.diagnostic.record(snapshot, stage="engine_ready", freeze=True)
        self.diagnostic.record({**snapshot, "stdout_tail": "PRIVATE REQUEST\n", "stderr_tail": "PRIVATE RESPONSE\n"},
                               retirement="closed")
        saved = self.read()
        self.assertEqual(saved["stdout_tail"], "startup\n")
        self.assertEqual(saved["stderr_tail"], "<redacted>\n")
        self.assertTrue(saved["native_output_frozen"])
        self.assertNotIn("PRIVATE", self.path.read_text())

    def test_existing_record_and_reparse_parent_are_never_replaced(self):
        before = self.path.read_bytes()
        with self.assertRaises(StartupDiagnosticError):
            StartupDiagnostic(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        linked = self.path.parent / "linked"
        linked.symlink_to(self.path.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            StartupDiagnostic(linked / FILE_NAME)


class FreezeBeforePostTests(unittest.TestCase):
    def test_backend_freezes_after_readiness_and_before_any_native_post(self):
        harness = Harness()
        self.addCleanup(harness.close)
        stages = []
        factory, handle = harness.backend._worker_factory, harness.handle
        def worker(config):
            result = factory(config)
            result.diagnostic_record = lambda **options: stages.append(options)
            return result
        def incoming(message, owner):
            if message.method == "POST":
                self.assertTrue(stages and stages[-1].get("freeze"), "native POST preceded privacy freeze")
                self.assertEqual(stages[-1]["stage"], "engine_ready")
            return handle(message, owner)
        harness.backend._worker_factory = worker
        harness.backend._startup_diagnostic = True
        harness.handle = incoming
        result = list(harness.backend.generate(request(), threading.Event()))
        self.assertTrue(result)
        self.assertEqual(stages, [{"stage": "engine_readiness"}, {"stage": "engine_ready", "freeze": True}])
