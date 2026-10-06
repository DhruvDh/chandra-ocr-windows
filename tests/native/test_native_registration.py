"""One-shot allocation controls at acquired-resource seams; only fake Win32 objects and local CRT test descriptors."""
import errno
import os
import unittest
from unittest.mock import patch

from runtime.native import winjob as W
from tests.native.test_native_ownership import CheckedAPI, CheckedCRT


class FailInsert(dict):
    def __init__(self, key):
        super().__init__()
        self.key, self.failed = key, False

    def __setitem__(self, key, value):
        if key == self.key and not self.failed:
            self.failed = True
            raise MemoryError("one-shot registration failure: " + key)
        super().__setitem__(key, value)


class RegistrationTests(unittest.TestCase):
    LIMITS = W.JobLimits(12 << 30, 13 << 30, 1)

    def owner_fault(self, table, key):
        original, owners = W._Launch, []
        class Injected(original):
            def __init__(self, *args):
                super().__init__(*args)
                setattr(self, table, FailInsert(key))
                owners.append(self)
        return patch.object(W, "_Launch", Injected), owners

    def transport(self, api=None):
        api = CheckedAPI() if api is None else api
        crt = CheckedCRT(api)
        return W.WindowsJobTransport(api, crt), api, crt

    def launch_failure(self, transport):
        with self.assertRaises(W.ContainmentError) as caught:
            transport.launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        return caught.exception

    def dispose_fakes(self, api, crt):
        # Test-only final disposition if an assertion fails; production never searches for unregistered handles.
        api.close_failures.clear()
        for handle in list(api.live):
            api.CloseHandle(handle)
        for fd in crt.fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def assert_slot_inventory(self, owner, api, crt):
        raw = {slot.get() for slot in owner.raw.values() if slot.get() is not None}
        self.assertEqual(api.live, raw)
        adopted = {slot.number for slot in owner.adopted.values() if slot.number is not None}
        for fd in crt.fds:
            try:
                os.fstat(fd)
            except OSError:
                continue
            self.assertIn(fd, adopted)

    def test_each_raw_registry_failure_closes_both_outputs_and_never_resumes(self):
        labels = ("job", "stdin_read", "stdin_write", "stdout_read", "stdout_write", "stderr_read",
                  "stderr_write", "process", "thread")
        for label in labels:
            with self.subTest(label=label):
                transport, api, crt = self.transport()
                self.addCleanup(self.dispose_fakes, api, crt)
                injection, owners = self.owner_fault("handles", label)
                with injection:
                    error = self.launch_failure(transport)
                owner = owners[0]
                self.assertTrue(owner.handles.failed)
                self.assertTrue(error.evidence["handles_closed"])
                self.assertTrue(error.evidence["child_retired"])
                self.assertFalse(owner.pending)
                self.assertIsNone(transport._pending)
                self.assertFalse(api.live)
                self.assertNotIn("ResumeThread", api.calls)
                self.assertEqual(len(api.close_attempts), len(set(api.close_attempts)))
                crt.assert_closed(self)

    def test_each_crt_registry_failure_owns_adopted_number_and_does_not_close_raw_alias(self):
        for label in ("stdin_write", "stdout_read", "stderr_read"):
            with self.subTest(label=label):
                transport, api, crt = self.transport()
                self.addCleanup(self.dispose_fakes, api, crt)
                injection, owners = self.owner_fault("descriptors", label)
                with injection:
                    error = self.launch_failure(transport)
                self.assertTrue(owners[0].descriptors.failed)
                self.assertTrue(error.evidence["handles_closed"])
                self.assertTrue(error.evidence["child_retired"])
                self.assertIsNone(transport._pending)
                self.assertFalse(api.live)
                self.assertNotIn("ResumeThread", api.calls)
                crt.assert_closed(self)

    def test_unregistered_pipe_failed_close_retains_slot_until_explicit_recovery(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        api.close_failures.add(103)  # Second stdin pipe output, before its registry insertion.
        injection, owners = self.owner_fault("handles", "stdin_write")
        with injection:
            error = self.launch_failure(transport)
        owner, receipt = owners[0], error.evidence
        self.assertIs(transport._pending, owner)
        self.assertFalse(receipt["handles_closed"])
        self.assertEqual(receipt["retained_handles"], ["stdin_write"])
        self.assert_slot_inventory(owner, api, crt)
        self.assertEqual(api.close_attempts.count(103), 1)
        with self.assertRaisesRegex(W.ContainmentError, "still owns"):
            transport.hold(["C:\\shader.hlsl"])
        api.close_failures.clear()
        self.assertTrue(transport.retry_cleanup()["handles_closed"])
        self.assertFalse(receipt["handles_closed"])
        self.assertFalse(api.live)
        self.assertIsNone(transport._pending)

    def test_unregistered_process_and_thread_remain_controlled_when_retirement_is_unknown(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        api.timeout = True
        injection, owners = self.owner_fault("handles", "process")
        with injection:
            error = self.launch_failure(transport)
        owner, receipt = owners[0], error.evidence
        self.assertTrue(error.child_may_remain)
        self.assertFalse(receipt["handles_closed"])
        self.assertFalse(receipt["child_retired"])
        self.assertEqual(set(receipt["retained_handles"]), {"job", "process"})
        self.assert_slot_inventory(owner, api, crt)
        self.assertIn("TerminateProcess", api.calls)
        self.assertNotIn("ResumeThread", api.calls)
        self.assertIs(transport._pending, owner)
        api.timeout = False
        self.assertTrue(transport.retry_cleanup()["handles_closed"])
        self.assertFalse(receipt["handles_closed"])
        self.assertFalse(api.live)

    def test_unregistered_process_failed_raw_closes_retain_both_control_slots(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        api.close_failures.update((101, 108))  # Job and process returned by the fake API.
        injection, owners = self.owner_fault("handles", "process")
        with injection:
            error = self.launch_failure(transport)
        self.assertTrue(error.evidence["child_retired"])
        self.assertFalse(error.evidence["handles_closed"])
        self.assert_slot_inventory(owners[0], api, crt)
        self.assertEqual(api.live, {101, 108})
        api.close_failures.clear()
        self.assertTrue(transport.retry_cleanup()["handles_closed"])
        self.assertFalse(error.evidence["handles_closed"])

    def test_failed_pid_display_assignment_does_not_claim_no_child_or_false_retirement(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        api.timeout = True
        original, owners = W._Launch, []
        class Injected(original):
            def __init__(self, *args):
                super().__init__(*args)
                self.failed = False
                owners.append(self)
            def __setattr__(self, name, value):
                if name == "pid" and value == 4242 and not self.failed:
                    self.failed = True
                    raise MemoryError("PID display assignment")
                super().__setattr__(name, value)
        with patch.object(W, "_Launch", Injected):
            error = self.launch_failure(transport)
        self.assertIsNone(owners[0].pid)
        self.assertEqual(error.evidence["pid"], 4242)
        self.assertTrue(error.evidence["child_created"])
        self.assertFalse(error.evidence["child_retired"])
        self.assertFalse(error.evidence["handles_closed"])
        self.assertTrue(error.child_may_remain)
        self.assertIs(transport._pending, owners[0])
        self.assert_slot_inventory(owners[0], api, crt)
        api.timeout = False
        self.assertTrue(transport.retry_cleanup()["handles_closed"])
        self.assert_slot_inventory(owners[0], api, crt)
        self.assertFalse(api.live)
        self.assertFalse(error.evidence["handles_closed"])

    def test_each_held_display_label_is_prepared_before_any_file_acquisition(self):
        for ordinal in (1, 2, 3):
            with self.subTest(ordinal=ordinal):
                transport, api, crt = self.transport()
                original, calls = W.Path, []
                def name(path):
                    calls.append(path)
                    if len(calls) == ordinal:
                        raise MemoryError("held display name")
                    return original(path)
                with patch.object(W, "Path", name), self.assertRaises(MemoryError):
                    transport.hold(["a.hlsl", "b.hlsl", "c.hlsl"])
                self.assertFalse(api.live)
                self.assertNotIn("CreateFileW", api.calls)
                self.assertIsNone(transport._pending)

    def test_generic_exception_after_held_acquisition_retains_failed_close_owner(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        api.close_failures.add(101)
        original = W.HeldFiles.acquire
        def fail(owner):
            original(owner)
            raise MemoryError("after file acquisition")
        with patch.object(W.HeldFiles, "acquire", fail), self.assertRaises(MemoryError) as caught:
            transport.hold(["a.hlsl", "b.hlsl"])
        owner, receipt = transport._pending, caught.exception.evidence
        self.assertIs(caught.exception.cleanup_owner, owner)
        self.assertEqual(owner._handles, [("a.hlsl", 101)])
        self.assertFalse(receipt["held"]["released_all"])
        self.assertEqual((receipt["held"]["held"], receipt["held"]["released"], receipt["held"]["retained"]), (2, 1, 1))
        self.assertEqual(api.close_attempts.count(101), 1)
        api.close_failures.clear()
        self.assertTrue(transport.retry_cleanup()["released_all"])
        self.assertFalse(receipt["held"]["released_all"])
        self.assertFalse(api.live)

    def test_unregistered_held_handle_has_fixed_owner_before_display_list_append(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        api.close_failures.add(101)
        original = W.HeldFiles
        class Entries(list):
            def append(self, value):
                raise MemoryError("held handle display insertion")
        class Injected(original):
            def __init__(self, *args):
                super().__init__(*args)
                self._handles = Entries()
        with patch.object(W, "HeldFiles", Injected), self.assertRaises(MemoryError) as caught:
            transport.hold(["a.hlsl"])
        owner = transport._pending
        self.assertIs(caught.exception.cleanup_owner, owner)
        self.assertEqual(owner._handles, [])  # Registry failed; the slot owns the acquired raw resource.
        self.assertEqual({slot.get() for _, _, slot in owner._slots}, api.live)
        self.assertFalse(caught.exception.evidence["held"]["released_all"])
        self.assertEqual(caught.exception.evidence["held"]["held"], 1)
        api.close_failures.clear()
        self.assertTrue(transport.retry_cleanup()["released_all"])
        self.assertFalse(api.live)

    def test_post_close_receipt_allocation_failure_does_not_close_raw_handle_twice(self):
        transport, api, crt = self.transport()
        self.addCleanup(self.dispose_fakes, api, crt)
        injection, owners = self.owner_fault("outcomes", "stdin_read")
        with injection:
            error = self.launch_failure(transport)
        self.assertTrue(owners[0].outcomes.failed)
        self.assertTrue(error.evidence["handles_closed"])
        self.assertFalse(api.live)
        self.assertEqual(len(api.close_attempts), len(set(api.close_attempts)))
        self.assertNotIn("ResumeThread", api.calls)

    def test_post_crt_close_allocation_failure_keeps_remaining_owner_and_does_not_retry_closed_fd(self):
        transport, api, crt = self.transport(CheckedAPI("ResumeThread"))
        self.addCleanup(self.dispose_fakes, api, crt)
        injection, owners = self.owner_fault("outcomes", "stdin_write")
        original, attempts = os.close, []
        def close(fd):
            attempts.append(fd)
            return original(fd)
        with injection, patch.object(W.os, "close", close):
            error = self.launch_failure(transport)
            first = crt.fds[0]
            self.assertIs(transport._pending, owners[0])
            self.assertFalse(error.evidence["handles_closed"])
            self.assert_slot_inventory(owners[0], api, crt)
            self.assertTrue(transport.retry_cleanup()["handles_closed"])
            self.assertEqual(attempts.count(first), 1)
        self.assertFalse(error.evidence["handles_closed"])
        self.assertFalse(api.live)
        crt.assert_closed(self)

    def test_ambiguous_unregistered_crt_number_is_never_retried_after_reuse(self):
        transport, api, crt = self.transport()
        injection, owners = self.owner_fault("descriptors", "stdin_write")
        original, attempts, guards = os.close, [], []
        def close(fd):
            attempts.append(fd)
            original(fd)
            guard = os.open(os.devnull, os.O_RDONLY)
            guards.append(guard)
            self.assertEqual(guard, fd)
            raise OSError(errno.EBADF, "unknown disposition after descriptor reuse")
        try:
            with injection, patch.object(W.os, "close", close):
                error = self.launch_failure(transport)
                receipt = error.evidence
                self.assertTrue(receipt["child_retired"])
                self.assertFalse(receipt["handles_closed"])
                self.assertEqual(receipt["ambiguous_descriptor_numbers"], {"stdin_write": guards[0]})
                self.assertIs(transport._pending, owners[0])
                self.assertFalse(api.live)
                self.assertFalse(transport.retry_cleanup()["handles_closed"])
                self.assertEqual(attempts, [guards[0]])
                os.fstat(guards[0])  # Still owned by the new consumer; operator cleanup must not close it.
                self.assertNotIn(103, api.close_attempts)  # The CRT adopted this raw pipe end.
        finally:
            for guard in guards:
                original(guard)
            for handle in list(api.live):
                api.close_failures.clear()
                api.CloseHandle(handle)

    def test_ordinary_launch_and_held_files_transfer_then_close_all_resources(self):
        transport, api, crt = self.transport()
        held = transport.hold(["a.hlsl", "b.hlsl"])
        self.assertTrue(held.release()["released_all"])
        process = transport.launch(["C:\\worker.exe"], {}, "C:\\", self.LIMITS)
        self.assertEqual(api.live, {process.job, process.process})
        self.assertEqual(api.calls.count("ResumeThread"), 1)
        self.assertTrue(process.confirm_closed()["handles_closed"])
        self.assertFalse(api.live)
        self.assertIsNone(transport._pending)
        crt.assert_closed(self)


if __name__ == "__main__":
    unittest.main()
