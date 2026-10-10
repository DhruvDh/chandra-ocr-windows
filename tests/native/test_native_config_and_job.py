"""Explicit configuration refusals, Windows Job launcher call order (fake kernel32) and launcher mode.

The fake kernel32 exercises only the checked call sequence and structure layouts on Linux; actual Job,
pipe, CreateProcess and ResumeThread behaviour remains a root-owned Windows check.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime.native import config as C
from runtime.native import winjob as W
from tests.native.native_fakes import shader_tree

PINS = None


def write_config(root, **changes):
    values = {"schema": C.SCHEMA, "activate": C.ACTIVATION, "executable": str(root / "bin/chandra-worker.exe"),
              "executable_sha256": hashlib.sha256(b"worker").hexdigest(), "model_dir": str(root / "model"),
              "shader_root": str(root / "shaders"), "input_root": str(root / "inputs"), "output_root": str(root / "outputs"),
              "caller_source": str(root / "bin/hf.py"), "pci": "03:00.0", "luid": "00000000:0000a1b2", "gemv_b1_selection": "predecessor",
              "lease_ms": 60000, "process_memory_limit_bytes": 12 << 30, "job_memory_limit_bytes": 13 << 30, "max_patch_rows": 832,
              "shader_tree_sha256": shader_tree(root / "shaders"), "worker_cpu": 0}
    values.update(changes)
    values = {k: v for k, v in values.items() if v is not None}
    data = json.dumps(values).encode()
    path = root / ("config-%d.json" % len(list(root.glob("config-*.json"))))
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chandra-native-config-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for name in ("bin", "model", "shaders", "inputs", "outputs"):
            (self.root / name).mkdir()
        (self.root / "bin/chandra-worker.exe").write_bytes(b"worker")
        (self.root / "bin/hf.py").write_bytes(b"x" * C.CALLER_SOURCE_BYTES)

    def refused(self, pattern, **changes):
        path, digest = write_config(self.root, **changes)
        with self.assertRaisesRegex(C.ConfigurationError, pattern):
            C.load_config(path, digest)

    def test_explicit_refusals(self):
        path, digest = write_config(self.root)
        with self.assertRaisesRegex(C.ConfigurationError, "SHA-256 differs"):
            C.load_config(path, "0" * 64)
        self.refused("activation marker", activate="yes")
        self.refused("Unknown configuration fields", device="auto")
        self.refused("Missing configuration fields: pci", pci=None)
        self.refused("absolute", model_dir="model")
        self.refused("normalized", model_dir=str(self.root / "model") + "/")
        self.refused("normalized", model_dir=str(self.root / "bin/../model"))
        self.refused("missing component", shader_root=str(self.root / "absent"))
        self.refused("distinct and not nested", output_root=str(self.root / "inputs"))
        (self.root / "inputs/nested").mkdir()
        self.refused("distinct and not nested", output_root=str(self.root / "inputs/nested"))
        self.refused("pci must be", pci="0000:03:00.0")
        self.refused("luid must be", luid="0x1234")
        self.refused("gemv_b1_selection", gemv_b1_selection="1")
        self.refused("lease_ms", lease_ms=999)
        self.refused("job_memory_limit_bytes", job_memory_limit_bytes=1 << 30)
        self.refused("process_memory_limit_bytes", process_memory_limit_bytes=0)
        self.refused("max_patch_rows", max_patch_rows=12289)
        self.refused("Nonfinite", startup_seconds=float("inf"))
        self.refused("startup_seconds", startup_seconds=0)
        self.refused("processor_dll_bootstrap", processor_dll_bootstrap="yes")
        self.refused("executable hash differs", executable_sha256="0" * 64)
        self.refused("not the pinned Chandra", )
        os.symlink(self.root / "model", self.root / "model-link")
        self.refused("symlink", model_dir=str(self.root / "model-link"))

    def test_pinned_processor_files_are_required(self):
        from scripts.native.prepare_input import CALLER_SOURCE_SHA
        caller = self.real_caller()
        path, digest = write_config(self.root, caller_source=str(caller))
        with self.assertRaisesRegex(C.ConfigurationError, "missing component|Pinned processor"):
            C.load_config(path, digest)
        self.assertEqual(hashlib.sha256(caller.read_bytes()).hexdigest(), CALLER_SOURCE_SHA)

    def real_caller(self):
        """The recorded public chandra-ocr 0.2.0 generate_hf source, when the project environment provides it."""
        try:
            import chandra.model as model
        except ImportError:
            self.skipTest("chandra-ocr 0.2.0 is not installed in this environment")
        return Path(model.__file__).with_name("hf.py")

    def test_serving_profile_is_the_established_one(self):
        from benchmarks.export_processor import PROFILES
        from runtime.native.prepare import PROFILE, PROFILE_NAME
        self.assertEqual(PROFILE[PROFILE_NAME], PROFILES["northstone-serving"])
        self.assertEqual(PROFILE[PROFILE_NAME], {"size": {"shortest_edge": 3136, "longest_edge": 3145728}})


class FakeKernel32:
    """Records checked calls; simulates a Job whose limits stick unless told otherwise."""

    def __init__(self, fail=None, resume=1, sticky=True):
        self.calls, self.fail, self.resume, self.sticky = [], fail, resume, sticky
        self.limits = None
        self.next = 100
        self.error = 0

    def last_error(self):
        return self.error

    def _handle(self):
        self.next += 1
        return self.next

    def __getattr__(self, name):
        raise AssertionError("unexpected Win32 call " + name)

    def _record(self, name, result=1):
        self.calls.append(name)
        return 0 if self.fail == name else result

    def CreateJobObjectW(self, attributes, name):
        self.calls.append("CreateJobObjectW")
        assert attributes is None and name is None
        return self._handle()

    def SetInformationJobObject(self, job, kind, info, size):
        info = info._obj
        assert kind == W.JobObjectExtendedLimitInformation and size == ctypes.sizeof(W.JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
        if self.sticky:
            self.limits = (info.BasicLimitInformation.LimitFlags, info.BasicLimitInformation.ActiveProcessLimit, info.ProcessMemoryLimit, info.JobMemoryLimit,
                           info.BasicLimitInformation.Affinity)
        return self._record("SetInformationJobObject")

    def GetPriorityClass(self, process):
        return self._record("GetPriorityClass", W.BELOW_NORMAL_PRIORITY_CLASS)

    def QueryInformationJobObject(self, job, kind, info, size, returned):
        info = info._obj
        if kind == W.JobObjectExtendedLimitInformation and self.limits:
            flags, active, process, total, affinity = self.limits
            info.BasicLimitInformation.LimitFlags, info.BasicLimitInformation.ActiveProcessLimit = flags, active
            info.BasicLimitInformation.Affinity = affinity
            info.ProcessMemoryLimit, info.JobMemoryLimit = process, total
        return self._record("QueryInformationJobObject")

    def CreatePipe(self, read, write, security, size):
        read._obj.value, write._obj.value = self._handle(), self._handle()
        assert security._obj.bInheritHandle == 0
        return self._record("CreatePipe")

    def SetHandleInformation(self, handle, mask, flags):
        return self._record("SetHandleInformation")

    def InitializeProcThreadAttributeList(self, buffer, count, flags, size):
        self.calls.append("InitializeProcThreadAttributeList")
        if buffer is None:
            size._obj.value = 48
            self.error = W.ERROR_INSUFFICIENT_BUFFER
            return 0
        return 1

    def UpdateProcThreadAttribute(self, buffer, flags, attribute, value, size, previous, returned):
        assert attribute == W.PROC_THREAD_ATTRIBUTE_HANDLE_LIST and size == 3 * ctypes.sizeof(ctypes.c_void_p)
        return self._record("UpdateProcThreadAttribute")

    def DeleteProcThreadAttributeList(self, buffer):
        self.calls.append("DeleteProcThreadAttributeList")

    def CreateProcessW(self, application, commandline, pa, ta, inherit, flags, env, cwd, startup, info):
        assert flags & W.CREATE_SUSPENDED and not flags & 0x01000000  # Suspended and never breakaway.
        assert startup._obj.StartupInfo.cb == ctypes.sizeof(W.STARTUPINFOEXW)
        info._obj.hProcess, info._obj.hThread, info._obj.dwProcessId = self._handle(), self._handle(), 4242
        return self._record("CreateProcessW")

    def AssignProcessToJobObject(self, job, process):
        return self._record("AssignProcessToJobObject")

    def IsProcessInJob(self, process, job, result):
        result._obj.value = 1
        return self._record("IsProcessInJob")

    def ResumeThread(self, thread):
        self.calls.append("ResumeThread")
        return self.resume

    def TerminateProcess(self, process, code):
        return self._record("TerminateProcess")

    def TerminateJobObject(self, job, code):
        return self._record("TerminateJobObject")

    def WaitForSingleObject(self, handle, ms):
        self.calls.append("WaitForSingleObject")
        return W.WAIT_OBJECT_0

    def CloseHandle(self, handle):
        self.calls.append("CloseHandle")
        return 1


class FakeOsf:
    def open_osfhandle(self, handle, flags):
        return os.open(os.devnull, os.O_RDWR)


class WindowsJobTests(unittest.TestCase):
    LIMITS = W.JobLimits(12 << 30, 13 << 30, 1)

    def launch(self, api):
        return W.WindowsJobTransport(api=api, osf=FakeOsf()).launch(["C:\\w\\chandra-worker.exe", "--execute"], {"SystemRoot": "C:\\Windows"}, "C:\\w", self.LIMITS)

    def test_x64_structure_layouts(self):
        sizes = {W.IO_COUNTERS: 48, W.JOBOBJECT_BASIC_LIMIT_INFORMATION: 64, W.JOBOBJECT_EXTENDED_LIMIT_INFORMATION: 144,
                 W.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION: 48, W.STARTUPINFOW: 104, W.STARTUPINFOEXW: 112, W.PROCESS_INFORMATION: 24,
                 W.SECURITY_ATTRIBUTES: 24}
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("x64 layout check")
        for structure, size in sizes.items():
            self.assertEqual(ctypes.sizeof(structure), size, structure.__name__)

    def test_suspended_child_is_admitted_before_resume(self):
        api = FakeKernel32()
        process = self.launch(api)
        order = [c for c in api.calls if c in ("CreateJobObjectW", "SetInformationJobObject", "QueryInformationJobObject", "CreateProcessW",
                                              "AssignProcessToJobObject", "IsProcessInJob", "ResumeThread")]
        self.assertEqual(order, ["CreateJobObjectW", "SetInformationJobObject", "QueryInformationJobObject", "CreateProcessW",
                                 "AssignProcessToJobObject", "IsProcessInJob", "ResumeThread"])
        flags, active, memory, job, affinity = api.limits
        self.assertEqual(flags & W.LIMIT_FLAGS, W.LIMIT_FLAGS)
        self.assertFalse(flags & (W.JOB_OBJECT_LIMIT_BREAKAWAY_OK | W.JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK))
        self.assertEqual((active, memory, job, affinity), (1, 12 << 30, 13 << 30, 1))
        self.assertEqual(process.pid, 4242)
        process.close_stdin()

    def test_any_failed_admission_step_refuses_without_resume(self):
        for failure in ("SetInformationJobObject", "CreatePipe", "CreateProcessW", "AssignProcessToJobObject", "IsProcessInJob"):
            with self.subTest(failure=failure):
                api = FakeKernel32(fail=failure)
                with self.assertRaises(W.ContainmentError):
                    self.launch(api)
                self.assertNotIn("ResumeThread", api.calls)
                self.assertIn("CloseHandle", api.calls)
                if failure in ("AssignProcessToJobObject", "IsProcessInJob"):
                    self.assertIn("TerminateProcess" if failure == "AssignProcessToJobObject" else "TerminateJobObject", api.calls)

    def test_limits_that_do_not_stick_or_a_bad_resume_are_refused(self):
        with self.assertRaisesRegex(W.ContainmentError, "did not take effect"):
            self.launch(FakeKernel32(sticky=False))
        api = FakeKernel32(resume=0)
        with self.assertRaisesRegex(W.ContainmentError, "ResumeThread"):
            self.launch(api)
        self.assertIn("TerminateJobObject", api.calls)

    def test_unsupported_host_and_unsafe_arguments(self):
        if os.name != "nt":
            with self.assertRaisesRegex(W.ContainmentError, "unsupported"):
                W.kernel32()
        with self.assertRaises(W.ContainmentError):
            W.command_line(["C:\\w\\worker.exe", 'x" --plan'])
        with self.assertRaises(W.ContainmentError):
            W.WindowsJobTransport(api=FakeKernel32(), osf=FakeOsf()).launch(["C:\\w.exe"], {}, "C:\\", W.JobLimits(0, 0, 1))
        self.assertEqual(W.environment_block({"b": "2", "A": "1"}), "A=1\0b=2\0\0")


class LauncherTests(unittest.TestCase):
    def test_native_mode_requires_explicit_configuration_and_refuses_unsupported_hosts(self):
        from chandra_service.__main__ import main
        cases = [["native"], ["native", "--native-config", "x.json"], ["native", "--native-config", "x.json", "--native-config-sha256", "0" * 64, "--model-path", "m"],
                 ["native", "--native-config", "/nonexistent/x.json", "--native-config-sha256", "0" * 64]]
        for argv in cases:
            with self.subTest(argv=argv), patch("sys.argv", ["chandra_service", *argv]), patch("chandra_service.__main__.socket.socket") as sockets:
                with self.assertRaises(SystemExit):
                    main()
                sockets.assert_not_called()


if __name__ == "__main__":
    unittest.main()
