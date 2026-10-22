"""Offline ownership checks: fake Jobs, pipes and file holds; no process, network or Win32 API."""

from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from runtime.llama.worker import AdmissionRefused, OwnedWorker, READ_BYTES, TAIL_BYTES
from runtime.native.winjob import ContainmentError, JobLimits


class Held:
    def __init__(self, events):
        self.events, self.releases = events, 0
        self.allow = threading.Event()
        self.allow.set()

    def release(self):
        self.events.append("release")
        self.releases += 1
        return {"released_all": self.allow.is_set()}


class Job:
    pid = 1234

    def __init__(self, events):
        self.events = events
        self.exited, self.empty, self.close_allowed = (threading.Event() for _ in range(3))
        self.empty.set()
        self.close_allowed.set()
        self.eof = {name: threading.Event() for name in ("stdout", "stderr")}
        self.read_started = {name: threading.Event() for name in self.eof}
        self.observed_eof = set()
        self.data = {name: b"" for name in self.eof}
        self.sizes = []
        self.confirmations = self.retries = self.terminations = 0
        self.stdin_closed = False

    def read(self, name, size):
        self.read_started[name].set()
        self.sizes.append(size)
        if self.data[name]:
            chunk, self.data[name] = self.data[name][:size], self.data[name][size:]
            return chunk
        self.eof[name].wait()
        self.observed_eof.add(name)
        return b""

    def read_stdout(self, size):
        return self.read("stdout", size)

    def read_stderr(self, size):
        return self.read("stderr", size)

    def close_stdin(self):
        self.stdin_closed = True

    def poll(self):
        return 0 if self.exited.is_set() else None

    def terminate(self):
        self.terminations += 1

    def confirm_closed(self):
        self.confirmations += 1
        assert self.exited.is_set() and self.observed_eof == {"stdout", "stderr"}
        if not self.empty.is_set():
            return None
        self.events.append("confirm")
        return {"handles_closed": self.close_allowed.is_set(), "ambiguous_descriptor_numbers": {},
                "job_active_processes": 0, "process_exit_code": 0}

    def retry_cleanup(self):
        self.retries += 1
        return {"handles_closed": self.close_allowed.is_set()}

    def finish(self):
        self.exited.set()
        for gate in self.eof.values():
            gate.set()


class CleanupOwner:
    def __init__(self):
        self.allow = threading.Event()
        self.pending = True
        self.retries = 0

    def retry_cleanup(self):
        self.retries += 1
        self.pending = not self.allow.is_set()
        return {"handles_closed": not self.pending}


class Transport:
    def __init__(self, events, job):
        self.events, self.job = events, job
        self.held, self.launches = Held(events), 0
        self.failure = self.hold_failure = None

    def hold(self, files):
        self.events.append("hold")
        if self.hold_failure:
            raise self.hold_failure
        return self.held

    def launch(self, argv, environment, cwd, limits):
        self.events.append("launch")
        self.launches += 1
        if self.failure:
            raise self.failure
        return self.job


class Config:
    runtime_dir = Path("/fake/runtime")
    engine = "vulkan"
    config_sha256 = "a" * 64

    def __init__(self, events):
        self.events, self.auth_hook = events, None

    def files(self):
        return [self.runtime_dir / "llama-server.exe", Path("/fake/model.gguf")]

    def authenticate(self):
        self.events.append("authenticate")
        if self.auth_hook:
            self.auth_hook()

    def argv(self, port, key, alias):
        return [str(self.files()[0]), "--port", str(port), "--api-key", key, "--alias", alias]

    def environment(self):
        return {"SystemRoot": "C:\\Windows"}

    def job_limits(self):
        return JobLimits(1024, 2048, 1)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.job = Job(self.events)
        self.config = Config(self.events)
        self.transport = Transport(self.events, self.job)
        self.worker = OwnedWorker(self.config, self.transport)
        self.retiring = None
        self.addCleanup(self.finish)

    def finish(self):
        self.job.finish()
        self.job.empty.set()
        self.job.close_allowed.set()
        self.transport.held.allow.set()
        for error in (self.transport.failure, self.transport.hold_failure):
            if error is not None and getattr(error, "cleanup_owner", None):
                error.cleanup_owner.allow.set()
        if self.retiring:
            self.retiring.join(3)
            self.assertFalse(self.retiring.is_alive(), "fake retirement failed to drain")
        else:
            self.worker.retire()

    def start(self):
        self.worker.start(18003, "private-key", "chandra")
        for gate in self.job.read_started.values():
            self.assertTrue(gate.wait(1))

    def retire_in_background(self):
        self.retiring = threading.Thread(target=self.worker.retire)
        self.retiring.start()
        self.assertTrue(self.worker._stop.wait(1))

    def test_status_is_passive_and_start_holds_before_authentication(self):
        self.assertFalse(self.worker.alive())
        self.assertFalse(self.worker.snapshot()["started"])
        self.assertEqual(self.events, [])
        self.start()
        self.assertEqual(self.events[:3], ["hold", "authenticate", "launch"])
        self.assertTrue(self.job.stdin_closed)
        self.assertTrue(self.worker.alive())
        with self.assertRaises(RuntimeError):
            self.worker.start(18003, "key", "chandra")

    def test_sycl_identity_is_retained_without_polling_or_launching(self):
        self.config.engine = "sycl"
        before = self.worker.snapshot()
        self.assertEqual((before["engine"], before["config_sha256"]), ("sycl", self.config.config_sha256))
        self.assertEqual(self.events, [])
        self.start()
        self.job.finish()
        closed = self.worker.retire()
        self.assertTrue(closed["closed"])
        self.assertEqual((closed["engine"], closed["config_sha256"]), (before["engine"], before["config_sha256"]))

    def test_process_exit_and_both_drains_precede_handle_and_file_release(self):
        self.start()
        self.retire_in_background()
        self.assertEqual(self.transport.held.releases, 0)
        self.job.exited.set()
        self.job.eof["stdout"].set()
        self.retiring.join(0.1)
        self.assertTrue(self.retiring.is_alive())
        self.assertEqual((self.job.confirmations, self.transport.held.releases), (0, 0))
        self.job.eof["stderr"].set()
        self.retiring.join(1)
        self.assertFalse(self.retiring.is_alive())
        self.assertEqual(self.events[-2:], ["confirm", "release"])
        self.assertTrue(self.worker.snapshot()["closed"])

    def test_empty_job_confirmation_is_required_after_process_exit(self):
        self.start()
        self.job.empty.clear()
        self.job.finish()
        self.retire_in_background()
        self.retiring.join(0.1)
        self.assertTrue(self.retiring.is_alive())
        self.assertTrue(self.worker.snapshot()["pending"])
        self.assertEqual(self.transport.held.releases, 0)
        self.job.empty.set()
        self.retiring.join(1)
        self.assertFalse(self.retiring.is_alive())

    def test_partial_launch_failure_keeps_cleanup_owner_and_files(self):
        owner = CleanupOwner()
        self.transport.failure = ContainmentError("partial launch", cleanup_owner=owner)
        with self.assertRaises(ContainmentError):
            self.worker.start(18003, "key", "chandra")
        self.retire_in_background()
        self.retiring.join(0.1)
        self.assertTrue(self.retiring.is_alive())
        self.assertIs(self.worker._cleanup_owner, owner)
        self.assertEqual(self.transport.held.releases, 0)
        self.assertIn("startup", self.worker.snapshot()["fault"])
        owner.allow.set()
        self.retiring.join(1)
        self.assertFalse(self.retiring.is_alive())
        self.assertEqual(self.transport.held.releases, 1)

    def test_partial_hold_failure_is_retained(self):
        owner = CleanupOwner()
        self.transport.hold_failure = ContainmentError("partial hold", cleanup_owner=owner)
        with self.assertRaises(ContainmentError):
            self.worker.start(18003, "key", "chandra")
        self.retire_in_background()
        self.retiring.join(0.1)
        self.assertTrue(self.retiring.is_alive())
        self.assertEqual(self.transport.launches, 0)
        owner.allow.set()
        self.retiring.join(1)
        self.assertTrue(self.worker.snapshot()["closed"])

    def test_cancellation_during_authentication_prevents_launch(self):
        self.config.auth_hook = self.worker.request_stop
        with self.assertRaises(RuntimeError):
            self.worker.start(18003, "key", "chandra")
        self.assertEqual(self.transport.launches, 0)
        self.assertEqual(self.transport.held.releases, 0)
        self.worker.retire()
        self.assertEqual(self.transport.held.releases, 1)

    def test_retire_waits_for_start_to_clear_its_initially_set_barrier(self):
        clearing, allow_clear, holding, allow_hold, retiring_at_state = (threading.Event() for _ in range(5))
        original_clear, original_hold = self.worker._startup_done.clear, self.transport.hold
        original_lock, test = self.worker._state_lock, self
        errors = []

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread() is test.retiring:
                    retiring_at_state.set()
                original_lock.acquire()

            def __exit__(self, *args):
                original_lock.release()

        def delayed_clear():
            clearing.set()  # start has checked stop and still owns the state lock here.
            allow_clear.wait()
            original_clear()

        def delayed_hold(files):
            holding.set()
            allow_hold.wait()
            return original_hold(files)

        def start():
            try:
                self.worker.start(18003, "key", "chandra")
            except RuntimeError as error:
                errors.append(error)

        starter = threading.Thread(target=start)
        with patch.object(self.worker._startup_done, "clear", delayed_clear), \
                patch.object(self.transport, "hold", delayed_hold), \
                patch.object(self.worker, "_state_lock", ObservedLock()):
            try:
                starter.start()
                self.assertTrue(clearing.wait(1))
                self.retire_in_background()
                # Retirement has reached its state-lock observation while clear is paused.
                # The old implementation reached this lock only after passing the set barrier.
                self.assertTrue(retiring_at_state.wait(1))
                allow_clear.set()
                self.assertTrue(holding.wait(1))
                self.retiring.join(0.1)
                self.assertTrue(self.retiring.is_alive())
                self.assertFalse(self.worker._closed)
                self.assertEqual(self.transport.held.releases, 0)
                allow_hold.set()
                starter.join(1)
                self.retiring.join(1)
                self.assertFalse(starter.is_alive())
                self.assertFalse(self.retiring.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertEqual((self.transport.launches, self.transport.held.releases), (0, 1))
                self.assertTrue(self.worker.snapshot()["closed"])
                self.assertIsNone(self.worker.snapshot()["fault"])
            finally:
                allow_clear.set()
                allow_hold.set()
                starter.join(3)
                if self.retiring:
                    self.retiring.join(3)

    def test_post_authentication_admission_refusal_prevents_launch(self):
        def refuse():
            self.events.append("admit")
            raise AdmissionRefused("fake live memory reserve refusal")

        with self.assertRaisesRegex(AdmissionRefused, "reserve refusal"):
            self.worker.start(18003, "key", "chandra", admit=refuse)
        self.assertEqual(self.events, ["hold", "authenticate", "admit"])
        self.assertEqual(self.transport.launches, 0)
        self.assertEqual(self.transport.held.releases, 0)
        self.assertIsNone(self.worker.snapshot()["fault"])
        self.assertTrue(self.worker.retire()["closed"])
        self.assertEqual(self.transport.held.releases, 1)
        self.assertIsNone(self.worker.snapshot()["fault"])

    def test_both_tails_are_bounded_and_drained_without_serial_deadlock(self):
        self.job.data["stdout"] = b"x" * (TAIL_BYTES * 3) + b"out"
        self.job.data["stderr"] = b"y" * (TAIL_BYTES * 3) + b"err"
        self.start()
        self.job.finish()
        status = self.worker.retire()
        self.assertEqual(status["stdout_tail"], "x" * (TAIL_BYTES - 3) + "out")
        self.assertEqual(status["stderr_tail"], "y" * (TAIL_BYTES - 3) + "err")
        self.assertTrue(all(size == READ_BYTES for size in self.job.sizes))

    def test_failed_second_reader_start_keeps_and_drains_both_pipes(self):
        original = threading.Thread.start

        def start(thread):
            if thread.name == "llama-stderr":
                raise RuntimeError("fake reader start failure")
            return original(thread)

        with patch.object(threading.Thread, "start", start):
            with self.assertRaises(RuntimeError):
                self.worker.start(18003, "key", "chandra")
        self.assertEqual(self.transport.held.releases, 0)
        self.job.finish()
        status = self.worker.retire()
        self.assertTrue(status["closed"])
        self.assertEqual(self.job.observed_eof, {"stdout", "stderr"})
        self.assertEqual(self.events[-2:], ["confirm", "release"])

    def test_reader_failure_requests_stop_and_is_reported(self):
        def fail(size):
            raise OSError("fake read failure")

        self.job.read_stdout = fail
        try:
            self.worker.start(18003, "key", "chandra")
        except RuntimeError:
            pass  # The reader may request stop before start's final cancellation check.
        self.assertTrue(self.worker._drained["stdout"].wait(1))
        self.assertTrue(self.worker.snapshot()["stop_requested"])
        self.assertIn("stdout drain", " ".join(self.worker.snapshot()["faults"]))
        # The failed reader ended; it did not reach EOF, so this fake explicitly records
        # termination of that read path rather than making its closure assertion reject it.
        self.job.observed_eof.add("stdout")
        self.job.finish()
        self.assertTrue(self.worker.retire()["closed"])

    def test_raw_handle_and_file_release_failures_hold_capacity(self):
        self.start()
        self.job.close_allowed.clear()
        self.transport.held.allow.clear()
        self.job.finish()
        self.retire_in_background()
        self.retiring.join(0.1)
        self.assertTrue(self.retiring.is_alive())
        self.assertEqual(self.transport.held.releases, 0)
        self.job.close_allowed.set()
        self.retiring.join(0.1)
        self.assertTrue(self.retiring.is_alive())
        self.assertFalse(self.worker.snapshot()["closed"])
        self.assertIsNotNone(self.worker.snapshot()["fault"])
        self.transport.held.allow.set()
        self.retiring.join(1)
        self.assertFalse(self.retiring.is_alive())
        self.assertTrue(self.worker.snapshot()["closed"])
