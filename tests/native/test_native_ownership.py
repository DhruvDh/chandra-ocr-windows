"""Finite fake checks for global source work and recoverable native startup/custody ownership.

No Win32 object, native process, listener, model or accelerator runs here. Failure controls are fake API calls.
"""
import ctypes
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from runtime.native import NativeBackend, NativeFailure
from runtime.native import backend as B, config as C, shaders as S, winjob as W
from tests.native.native_fakes import PINNED, FakeProcessor, FakeTransport, config, payload
from tests.native.test_native_config_and_job import FakeKernel32, FakeOsf


class CheckedAPI(FakeKernel32):
    """Retains each fake raw handle until a successful CloseHandle; allows one targeted call to fail."""
    def __init__(self, fail_call=None, ordinal=1):
        super().__init__()
        self.fail_call, self.ordinal, self.seen = fail_call, ordinal, {}
        self.live, self.close_attempts = set(), []
        self.close_failures, self.timeout = set(), False

    def _handle(self):
        handle = super()._handle()
        self.live.add(handle)
        return handle

    def _record(self, name, result=1):
        self.calls.append(name)
        self.seen[name] = self.seen.get(name, 0) + 1
        return 0 if name == self.fail_call and self.seen[name] == self.ordinal else result

    def CreateJobObjectW(self, attributes, name):
        return self._handle() if self._record("CreateJobObjectW") else 0

    def CreatePipe(self, read, write, security, size):
        if not self._record("CreatePipe"):
            return 0
        read._obj.value, write._obj.value = self._handle(), self._handle()
        return 1

    def CreateProcessW(self, *args):
        if not self._record("CreateProcessW"):
            return 0
        info = args[-1]._obj
        info.hProcess, info.hThread, info.dwProcessId = self._handle(), self._handle(), 4242
        return 1

    def CreateFileW(self, *args):
        return self._handle() if self._record("CreateFileW") else W.INVALID_HANDLE_VALUE

    def InitializeProcThreadAttributeList(self, buffer, count, flags, size):
        if buffer is None:
            size._obj.value = W.MAX_ATTRIBUTE_BYTES + 1 if self.fail_call == "attribute_size" else 48
            self.error = W.ERROR_INSUFFICIENT_BUFFER
            return 0
        return self._record("InitializeProcThreadAttributeList")

    def ResumeThread(self, handle):
        self.calls.append("ResumeThread")
        return 0 if self.fail_call == "ResumeThread" else 1

    def GetExitCodeProcess(self, handle, code):
        code._obj.value = 0
        return self._record("GetExitCodeProcess")

    def WaitForSingleObject(self, handle, ms):
        self.calls.append("WaitForSingleObject")
        return W.WAIT_TIMEOUT if self.timeout else W.WAIT_OBJECT_0

    def CloseHandle(self, handle):
        self.close_attempts.append(handle)
        ok = self._record("CloseHandle") and handle not in self.close_failures
        if ok:
            assert handle in self.live, "closed an unowned or previously closed handle"
            self.live.remove(handle)
        return int(bool(ok))


class CheckedCRT(FakeOsf):
    def __init__(self, api, fail_at=None):
        self.api, self.fail_at, self.calls, self.fds = api, fail_at, 0, []

    def open_osfhandle(self, handle, flags):
        self.calls += 1
        if self.calls == self.fail_at:
            raise OSError("injected CRT adoption failure")
        fd = super().open_osfhandle(handle, flags)
        self.api.live.remove(handle)  # CRT ownership transferred; its descriptor now owns the fake resource.
        self.fds.append(fd)
        return fd

    def assert_closed(self, test):
        for fd in self.fds:
            with test.assertRaises(OSError):
                os.fstat(fd)


class SourceWorkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native-owned-source-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shaders, self.inputs = self.root / "shaders", self.root / "inputs"
        self.shaders.mkdir()
        self.inputs.mkdir()
        (self.shaders / "a.hlsl").write_bytes(b"// admitted source\n")

    def test_original_302_directory_counterexample_is_globally_refused(self):
        for branch in ("branch_a", "branch_b"):
            (self.shaders / branch).mkdir()
            for i in range(150):
                (self.shaders / branch / f"d{i:03}").mkdir()
        seen = []
        original = os.lstat
        def observe(path):
            seen.append(str(path))
            return original(path)
        with patch.object(S.os, "lstat", observe), self.assertRaisesRegex(S.ShaderSourceRefused, "global directory"):
            S.read_tree(self.shaders)
        self.assertLessEqual(len(seen), S.MAX_DIRECTORIES + S.MAX_FILES + 2)

    def test_enumeration_stops_at_the_first_excess_entry(self):
        class Entry:
            name = "a"
        reads = []
        class Entries:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                for _ in range(10000):
                    reads.append(1)
                    yield Entry()
        with patch.object(S.os, "scandir", return_value=Entries()), self.assertRaises(S.ShaderSourceRefused):
            S.directory_names(self.shaders, 3)
        self.assertEqual(len(reads), 4)

    def test_lexical_include_and_line_work_refusals_preserve_authentication(self):
        for content in (b'#include "../x.hlsl"\n', b'#include NAME\n', b'#include "missing.hlsl"\n', b'#line 1\n',
                        b'#inc\\\nlude "b.hlsl"\n', b'??=include "b.hlsl"\n', b'\x00', b'\n' * S.MAX_SOURCE_LINES):
            with self.subTest(content=content[:40]):
                (self.shaders / "a.hlsl").write_bytes(content)
                with self.assertRaises(S.ShaderSourceRefused):
                    S.read_tree(self.shaders)
        (self.shaders / "a.hlsl").write_bytes(b'#include "b.hlsl"\n')
        (self.shaders / "b.hlsl").write_bytes(b'#include "a.hlsl"\n')
        with self.assertRaisesRegex(S.ShaderSourceRefused, "cycle"):
            S.read_tree(self.shaders)
        (self.shaders / "b.hlsl").write_bytes(b"// leaf\n")
        tree = S.read_tree(self.shaders)
        self.assertEqual(tree.includes, 1)
        with self.assertRaisesRegex(S.ShaderSourceRefused, "differs"):
            S.verified_tree(self.shaders, "0" * 64)

    def test_fanout_expansion_and_total_entry_budgets_are_independent(self):
        (self.shaders / "a.hlsl").write_bytes(b'#include "b.hlsl"\n' * 3)
        (self.shaders / "b.hlsl").write_bytes(b'#include "c.hlsl"\n' * 3)
        (self.shaders / "c.hlsl").write_bytes(b'// leaf\n')
        with patch.object(S, "MAX_INCLUDE_EXPANSIONS", 10), self.assertRaisesRegex(S.ShaderSourceRefused, "expansion"):
            S.read_tree(self.shaders)
        with patch.object(S, "MAX_ENTRIES", 2), self.assertRaisesRegex(S.ShaderSourceRefused, "entry budget"):
            S.read_tree(self.shaders)

    def test_real_shader_digest_and_snapshot_substitution(self):
        root = Path(__file__).resolve().parents[2] / "ChandraNative/shaders"
        tree = S.read_tree(root)
        self.assertEqual((len(tree.files), tree.total_bytes, tree.includes, tree.tree_sha256),
                         (38, 60285, 3, "5438acdb33fbc1ba2d3a19b31d328e2ab3d05df011b42811108b26434cab89c1"))
        snapshot = S.Snapshot.create(self.inputs, tree)
        try:
            self.assertEqual(snapshot.verify().tree_sha256, tree.tree_sha256)
            path = snapshot.files()[0]
            os.chmod(path, 0o600)
            path.write_bytes(b"// substituted\n")
            with self.assertRaisesRegex(S.ShaderSourceRefused, "differs"):
                snapshot.verify()
        finally:
            self.assertIsNone(snapshot.remove())

    def test_failed_snapshot_removal_is_unclean_retained_and_recoverable(self):
        snapshot = S.Snapshot.create(self.inputs, S.read_tree(self.shaders))
        transport = FakeTransport()
        held = transport.hold(snapshot.files())
        custody = S.SourceCustody(snapshot, held)
        with patch.object(snapshot, "remove", return_value="injected removal failure"):
            receipt = custody.retire()
        self.assertFalse(receipt["clean"])
        self.assertFalse(receipt["removed"])
        self.assertTrue(custody.pending)
        self.assertTrue(snapshot.directory.exists())
        self.assertIs(custody.retire(), receipt)
        self.assertEqual(held.releases, 1)
        with self.assertRaisesRegex(S.ShaderSourceRefused, "prior shader snapshot"):
            S.Snapshot.create(self.inputs, snapshot.tree)
        self.assertTrue(custody.retry_cleanup()["removed"])
        self.assertFalse(custody.pending)
        self.assertFalse(receipt["clean"])  # Recovery is a new disposition, never retrospective clean retirement.

    def test_failed_creation_retains_snapshot_owner_and_cleanup_has_global_bound(self):
        tree = S.read_tree(self.shaders)
        with patch.object(S.Snapshot, "verify", side_effect=S.ShaderSourceRefused("injected write verification failure")), \
                patch.object(S.Snapshot, "remove", return_value="injected removal failure"):
            with self.assertRaises(S.ShaderSourceRefused) as error:
                S.Snapshot.create(self.inputs, tree)
        snapshot = error.exception.cleanup_snapshot
        self.assertEqual(snapshot.creation_removal_error, "injected removal failure")
        (snapshot.shaders / "extra").mkdir()
        for i in range(S.MAX_DIRECTORIES):
            (snapshot.shaders / "extra" / f"d{i}").mkdir()
        refusal = snapshot.remove()
        self.assertIn("global directory", refusal)
        self.assertTrue(snapshot.files()[0].exists())  # Planning refusal happens before the first unlink.


class HandleOwnerTests(unittest.TestCase):
    LIMITS = W.JobLimits(12 << 30, 13 << 30, 1)

    def test_failed_held_close_and_failed_open_keep_values_and_refuse_replacement(self):
        api = CheckedAPI("CreateFileW", 2)
        api.close_failures.add(101)
        transport = W.WindowsJobTransport(api, CheckedCRT(api))
        with self.assertRaises(W.ContainmentError) as error:
            transport.hold([Path("a.hlsl"), Path("b.hlsl")])
        owner = error.exception.cleanup_owner
        self.assertEqual(owner._handles, [("a.hlsl", 101)])
        self.assertEqual(api.close_attempts, [101])
        with self.assertRaisesRegex(W.ContainmentError, "still owns"):
            transport.hold([Path("c.hlsl")])
        api.close_failures.clear()
        self.assertTrue(transport.retry_cleanup()["released_all"])
        self.assertFalse(owner.pending)
        self.assertFalse(api.live)

    def test_startup_failure_stages_close_or_retain_every_owner(self):
        stages = [("CreateJobObjectW", 1), ("SetInformationJobObject", 1), ("QueryInformationJobObject", 1),
                  *(('CreatePipe', i) for i in range(1, 4)), *(('SetHandleInformation', i) for i in range(1, 4)),
                  ("attribute_size", 1), ("InitializeProcThreadAttributeList", 1), ("UpdateProcThreadAttribute", 1),
                  ("CreateProcessW", 1), *(('CloseHandle', i) for i in range(1, 4)), ("GetPriorityClass", 1),
                  ("AssignProcessToJobObject", 1), ("IsProcessInJob", 1), ("ResumeThread", 1), ("CloseHandle", 4)]
        for stage, ordinal in stages:
            with self.subTest(stage=stage, ordinal=ordinal):
                api = CheckedAPI(stage, ordinal)
                crt = CheckedCRT(api)
                transport = W.WindowsJobTransport(api, crt)
                with self.assertRaises(W.ContainmentError) as error:
                    transport.launch(["C:\\worker.exe", "--execute"], {"SystemRoot": "C:\\Windows"}, "C:\\", self.LIMITS)
                evidence = error.exception.evidence
                self.assertTrue(evidence["child_retired"])
                if stage == "CloseHandle":
                    self.assertFalse(evidence["handles_closed"])
                    self.assertIs(error.exception.cleanup_owner, transport._pending)
                    self.assertEqual(len(api.live), 1)
                    failed = next(iter(api.live))
                    self.assertEqual(api.close_attempts.count(failed), 1)  # No hidden retry in launch cleanup.
                    self.assertTrue(transport.retry_cleanup()["handles_closed"])
                self.assertFalse(api.live)
                crt.assert_closed(self)
                if stage == "InitializeProcThreadAttributeList":
                    self.assertNotIn("DeleteProcThreadAttributeList", api.calls)
                    self.assertEqual(evidence["attribute_list"], "not_initialized")
                if stage != "ResumeThread" and not (stage == "CloseHandle" and ordinal == 4):
                    self.assertNotIn("ResumeThread", api.calls)
        for failure in (1, 2, 3):
            with self.subTest(adoption=failure):
                api = CheckedAPI()
                crt = CheckedCRT(api, failure)
                with self.assertRaises(W.ContainmentError):
                    W.WindowsJobTransport(api, crt).launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
                self.assertFalse(api.live)
                crt.assert_closed(self)

    def test_unobserved_exit_retains_process_and_job_control_until_explicit_retry(self):
        api = CheckedAPI("ResumeThread")
        api.timeout = True
        transport = W.WindowsJobTransport(api, CheckedCRT(api))
        with self.assertRaises(W.ContainmentError) as error:
            transport.launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        owner, receipt = error.exception.cleanup_owner, error.exception.evidence
        self.assertTrue(error.exception.child_may_remain)
        self.assertEqual(set(owner.handles), {"process", "job"})
        self.assertFalse(receipt["handles_closed"])
        with self.assertRaisesRegex(W.ContainmentError, "still owns"):
            transport.launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        api.timeout = False
        self.assertTrue(transport.retry_cleanup()["child_retired"])
        self.assertFalse(api.live)
        self.assertFalse(receipt["child_retired"])

    def test_job_process_failed_handles_remain_recoverable_and_original_receipt_immutable(self):
        api = CheckedAPI()
        crt = CheckedCRT(api)
        process = W.WindowsJobTransport(api, crt).launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        api.close_failures.update((process.process, process.job))
        receipt = process.confirm_closed()
        self.assertTrue(process.retired)
        self.assertFalse(process.closed)
        self.assertEqual(len(api.live), 2)
        self.assertIs(process.confirm_closed(), receipt)
        api.close_failures.clear()
        self.assertTrue(process.retry_cleanup()["handles_closed"])
        self.assertTrue(process.closed)
        self.assertFalse(receipt["handles_closed"])
        self.assertFalse(api.live)
        crt.assert_closed(self)

    def test_failed_successor_construction_keeps_launch_ownership_for_cleanup(self):
        api = CheckedAPI()
        crt = CheckedCRT(api)
        with patch.object(W, "JobProcess", side_effect=MemoryError("injected successor construction failure")):
            with self.assertRaises(W.ContainmentError) as error:
                W.WindowsJobTransport(api, crt).launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        self.assertTrue(error.exception.evidence["resumed"])
        self.assertTrue(error.exception.evidence["child_retired"])
        self.assertTrue(error.exception.evidence["handles_closed"])
        self.assertFalse(api.live)
        crt.assert_closed(self)

    def test_cleanup_failure_controls_retain_the_process_and_job(self):
        for contained, failure in ((False, "TerminateProcess"), (True, "TerminateJobObject"),
                                   (False, "wait_error"), (True, "QueryInformationJobObject")):
            with self.subTest(contained=contained, failure=failure):
                api = CheckedAPI(failure)
                owner = W._Launch(api, CheckedCRT(api))
                owner.own("job", api._handle(), "fake")
                owner.own("process", api._handle(), "fake")
                owner.pid, owner.contained = 4242, contained
                original_wait = api.WaitForSingleObject
                if failure == "wait_error":
                    api.WaitForSingleObject = lambda handle, ms: 0xffffffff
                elif failure.startswith("Terminate"):
                    api.timeout = True
                receipt = owner.cleanup()
                self.assertFalse(receipt["child_retired"])
                self.assertFalse(receipt["handles_closed"])
                self.assertEqual(set(owner.handles), {"process", "job"})
                api.timeout, api.fail_call, api.WaitForSingleObject = False, None, original_wait
                self.assertTrue(owner.retry_cleanup()["handles_closed"])
                self.assertFalse(api.live)
                self.assertFalse(receipt["child_retired"])

    def test_ambiguous_crt_close_records_number_and_never_retries_it(self):
        api = CheckedAPI()
        crt = CheckedCRT(api)
        process = W.WindowsJobTransport(api, crt).launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        failed = process._descriptors["stdout"]
        original_close = os.close
        attempts = []
        def close(fd):
            attempts.append(fd)
            if fd == failed:
                raise OSError(5, "injected ambiguous CRT close")
            original_close(fd)
        try:
            with patch.object(W.os, "close", close):
                receipt = process.confirm_closed()
                self.assertFalse(receipt["handles_closed"])
                self.assertEqual(receipt["ambiguous_descriptor_numbers"], {"stdout": failed})
                self.assertFalse(process.retry_cleanup()["handles_closed"])
                self.assertEqual(attempts.count(failed), 1)
                self.assertIs(process.confirm_closed(), receipt)
        finally:
            original_close(failed)  # Test owner knows the injected close did not execute; production cannot assume that.
        self.assertFalse(api.live)
        crt.assert_closed(self)

    def test_total_resource_bounds_refuse_before_win32_allocation(self):
        api = CheckedAPI()
        transport = W.WindowsJobTransport(api, CheckedCRT(api))
        for argv, env, limits in [(["a"] * 65, {}, self.LIMITS), (["a" * 32768], {}, self.LIMITS),
                                  (["a"], {str(i): "x" for i in range(65)}, self.LIMITS),
                                  (["a"], {"x": "v" * 32768}, self.LIMITS), (["a"], {}, W.JobLimits(1, 2, 1, 2))]:
            with self.subTest(argv=len(argv), env=len(env), active=limits.active_processes), self.assertRaises(W.ContainmentError):
                transport.launch(argv, env, "C:\\", limits)
        with self.assertRaises(W.ContainmentError):
            transport.hold([Path("a.hlsl")] * 258)
        self.assertEqual(api.calls, [])


class BackendCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native-backend-owner-")
        self.addCleanup(self.tmp.cleanup)
        self.cfg = config(self.tmp.name, shutdown_seconds=.2, terminate_seconds=.2)

    def join_fake(self, backend, transport):
        session = backend._session or backend._finalizing or (backend._cleanup_owner and backend._cleanup_owner.session)
        for worker in transport.workers:
            worker.kill()
        backend.close()
        if session is not None:
            for thread in session.threads:
                if thread.ident is not None:
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
        for worker in transport.workers:
            worker.thread.join(2)
            self.assertFalse(worker.thread.is_alive())
            if worker.reader_thread is not None:
                worker.reader_thread.join(2)
                self.assertFalse(worker.reader_thread.is_alive())

    def test_exited_fake_worker_teardown_stops_and_joins_its_reader(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        list(backend.generate(payload(), threading.Event()))
        worker = transport.workers[0]
        backend.unload()
        self.assertEqual(worker.exit_code, 0)
        worker.kill()
        worker.reader_thread.join(2)
        self.assertFalse(worker.reader_thread.is_alive())

    def test_idle_failed_removal_retains_owner_failed_health_and_refuses_accumulation(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        list(backend.generate(payload(), threading.Event()))
        session = backend._session
        snapshot = session.custody.snapshot
        with patch.object(snapshot, "remove", return_value="injected removal failure"):
            backend.unload()
        self.assertTrue(session.wait_finalized(2))
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertFalse(health["retirements"][-1]["clean"])
        self.assertFalse(health["retirements"][-1]["expected"])
        self.assertTrue(health["retained_cleanup"]["pending"])
        self.assertIs(backend._cleanup_owner.custody, session.custody)
        self.assertFalse(backend.close())
        with self.assertRaisesRegex(NativeFailure, "closed|failed earlier"):
            list(backend.generate(payload(), threading.Event()))
        self.assertEqual(len(transport.launches), 1)
        self.assertFalse(backend.cleanup_retained()["pending"])
        self.assertFalse(snapshot.directory.exists())
        self.assertTrue(backend.close())
        self.assertFalse(health["retirements"][-1]["clean"])

    def test_failed_snapshot_creation_retains_backend_owner_and_original_disposition(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        with patch.object(S.Snapshot, "verify", side_effect=S.ShaderSourceRefused("injected creation failure")), \
                patch.object(S.Snapshot, "remove", return_value="injected removal failure"):
            with self.assertRaises(NativeFailure):
                list(backend.generate(payload(), threading.Event()))
        owner = backend._cleanup_owner
        self.assertIsNotNone(owner.snapshot)
        self.assertTrue(owner.snapshot.directory.exists())
        self.assertEqual(transport.holds, [])
        self.assertFalse(backend.close())
        original = backend.health()["launch_failure"]
        self.assertFalse(backend.cleanup_retained()["pending"])
        self.assertEqual(original["disposition"]["snapshot_removal"], "injected removal failure")
        self.assertEqual(transport.launches, [])

    def test_failed_custody_handle_release_retains_snapshot_until_explicit_recovery(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        list(backend.generate(payload(), threading.Event()))
        session = backend._session
        snapshot = session.custody.snapshot
        transport.release_fails = True
        backend.unload()
        self.assertTrue(session.wait_finalized(2))
        health = backend.health()
        self.assertEqual(health["state"], "failed")
        self.assertFalse(health["retirements"][-1]["shader_source"]["held"]["released_all"])
        self.assertTrue(snapshot.files()[0].exists())
        self.assertIs(backend._cleanup_owner.session, session)
        self.assertFalse(backend.close())
        transport.release_fails = False
        self.assertFalse(backend.cleanup_retained()["pending"])
        self.assertFalse(snapshot.directory.exists())
        self.assertEqual(session.custody.held.releases, 2)
        self.assertFalse(health["retirements"][-1]["clean"])

    def test_retired_failed_process_handles_keep_a_backend_owner_after_finalization(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        list(backend.generate(payload(), threading.Event()))
        session = backend._session
        transport.handle_close_fails = True
        backend.unload()
        self.assertTrue(session.wait_finalized(2))
        health = backend.health()
        self.assertIsNone(backend._finalizing)
        self.assertIs(backend._cleanup_owner.process, session.process)
        self.assertFalse(health["retirements"][-1]["handles_closed"])
        self.assertFalse(backend.close())
        transport.handle_close_fails = False
        self.assertFalse(backend.cleanup_retained()["pending"])
        self.assertFalse(health["retirements"][-1]["handles_closed"])

    def test_snapshot_mutation_refuses_the_next_submit_and_does_not_replay(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        list(backend.generate(payload(), threading.Event()))
        session = backend._session
        path = session.custody.snapshot.files()[0]
        os.chmod(path, 0o600)
        path.write_bytes(b"// substituted source\n")
        with self.assertRaisesRegex(NativeFailure, "shader|source|custody"):
            list(backend.generate(payload(), threading.Event()))
        self.assertTrue(session.wait_closed(2))
        self.assertEqual(len(transport.submits()), 1)
        self.assertEqual(len(transport.launches), 1)
        self.assertEqual(backend.health()["state"], "failed")

    def test_failed_launch_control_keeps_snapshot_and_file_handles_until_child_retirement(self):
        api = CheckedAPI("ResumeThread")
        api.timeout = True
        transport = W.WindowsJobTransport(api, CheckedCRT(api))
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        with self.assertRaises(NativeFailure):
            list(backend.generate(payload(), threading.Event()))
        owner = backend._cleanup_owner
        self.assertTrue(owner.snapshot.directory.exists())
        self.assertTrue(owner.held.pending)
        self.assertEqual(set(owner.control.handles), {"process", "job"})
        self.assertFalse(backend.close())
        original = backend.health()["launch_failure"]
        api.timeout = False
        self.assertFalse(backend.cleanup_retained()["pending"])
        self.assertFalse(api.live)
        self.assertFalse(owner.snapshot.directory.exists())
        self.assertFalse(original["cleanup"]["child_retired"])
        self.assertEqual(list(self.cfg.input_root.glob("p-*")), [])  # No submit occurred; prepared input was safe to discard.

    def test_partial_channel_start_failure_closes_native_owner_and_joins_started_thread(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        original = threading.Thread.start
        def start(thread):
            if thread.name == "chandra-native-stderr":
                raise RuntimeError("injected Thread.start failure")
            return original(thread)
        with patch.object(threading.Thread, "start", start), self.assertRaises(NativeFailure):
            list(backend.generate(payload(), threading.Event()))
        # Immediate cleanup may race the reader's condition; an explicit retained retry accounts for that owner.
        if backend._cleanup_owner is not None:
            time.sleep(.05)
            self.assertFalse(backend.cleanup_retained()["pending"])
        self.assertTrue(backend.close())
        self.assertEqual(len(transport.launches), 1)
        self.assertEqual(transport.submits(), [])
        self.assertEqual(list(self.cfg.input_root.iterdir()), [])

    def test_documented_config_names_every_required_field(self):
        import json
        guide = (Path(__file__).resolve().parents[2] / "docs/directcompute-native-endpoint.md").read_text()
        example = json.loads(guide.split("```json\n", 1)[1].split("```", 1)[0])
        self.assertEqual(C.REQUIRED - set(example), set())

    def test_retained_owner_allocation_failure_uses_existing_atomic_publication_failure(self):
        transport = FakeTransport()
        backend = NativeBackend(self.cfg, transport=transport, processor=FakeProcessor(), pinned=PINNED)
        self.addCleanup(self.join_fake, backend, transport)
        list(backend.generate(payload(), threading.Event()))
        session = backend._session
        transport.handle_close_fails = True
        with patch.object(B, "_CleanupOwner", side_effect=MemoryError("injected cleanup owner allocation failure")):
            with self.assertRaisesRegex(NativeFailure, "unpublished"):
                backend.unload()
            health = backend.health()
            self.assertEqual(health["state"], "failed")
            self.assertEqual(health["failure"], B.PUBLICATION_FAILED)
            self.assertEqual(health["retirements"], [])
            self.assertIs(backend._session, session)
            self.assertFalse(session.published)
            receipt = session.closure
            self.assertFalse(backend.close())
        backend.health()
        self.assertIs(backend._cleanup_owner.process, session.process)
        self.assertIs(session.closure, receipt)
        self.assertEqual(session.custody.held.releases, 1)
        self.assertEqual(len(transport.launches), 1)
        transport.handle_close_fails = False
        self.assertFalse(backend.cleanup_retained()["pending"])


if __name__ == "__main__":
    unittest.main()
