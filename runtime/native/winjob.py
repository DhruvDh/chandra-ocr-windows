"""Windows Job containment for the native worker: the only production transport.

The child is created suspended at below-normal priority with exactly three inherited pipe handles, assigned to
a fresh unnamed, non-inheritable Job with kill-on-close, finite process and job memory limits, one active
process, one logical CPU and no breakaway, verified inside that Job, and only then resumed. Every Win32 call
is checked, including every handle closure: a failed closure is reported as failed and its handle retained,
never claimed closed. There is no subprocess fallback: an unsupported host or any failed step refuses
execution. If this process dies the OS closes the Job handle and kill-on-close retires the worker.
"""
import ctypes
from ctypes import c_int32, c_int64, c_size_t, c_uint16, c_uint32, c_uint64, c_void_p, c_wchar_p
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import subprocess
import sys
import threading

DWORD, WORD, BOOL, HANDLE = c_uint32, c_uint16, c_int32, c_void_p
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x8
JOB_OBJECT_LIMIT_AFFINITY = 0x10
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x400
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x800
JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
LIMIT_FLAGS = (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
               | JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | JOB_OBJECT_LIMIT_AFFINITY)
BREAKAWAY_FLAGS = JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
JobObjectBasicAccountingInformation = 1
JobObjectExtendedLimitInformation = 9
CREATE_SUSPENDED = 0x4
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
CREATE_UNICODE_ENVIRONMENT = 0x400
EXTENDED_STARTUPINFO_PRESENT = 0x80000
CREATE_NO_WINDOW = 0x08000000
CREATION_FLAGS = CREATE_SUSPENDED | BELOW_NORMAL_PRIORITY_CLASS | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW
STARTF_USESTDHANDLES = 0x100
HANDLE_FLAG_INHERIT = 0x1
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x20002
ERROR_INSUFFICIENT_BUFFER = 122
WAIT_OBJECT_0, WAIT_TIMEOUT = 0, 0x102
STILL_ACTIVE = 259
TERMINATION_CODE = 0xC0DE0001
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
INVALID_HANDLE_VALUE = c_void_p(-1).value
CLEANUP_WAIT_MS = 5000
MAX_HELD_FILES = 257  # The executable and the admitted shader tree.
MAX_ENV_ENTRIES = 64
MAX_ENV_CHARS = 32767
MAX_COMMAND_CHARS = 32767
MAX_ARGUMENTS = 64
MAX_ATTRIBUTE_BYTES = 65536


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                              "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", c_int64), ("PerJobUserTimeLimit", c_int64), ("LimitFlags", DWORD),
                ("MinimumWorkingSetSize", c_size_t), ("MaximumWorkingSetSize", c_size_t), ("ActiveProcessLimit", DWORD),
                ("Affinity", c_size_t), ("PriorityClass", DWORD), ("SchedulingClass", DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", c_size_t), ("JobMemoryLimit", c_size_t),
                ("PeakProcessMemoryUsed", c_size_t), ("PeakJobMemoryUsed", c_size_t)]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [("TotalUserTime", c_int64), ("TotalKernelTime", c_int64), ("ThisPeriodTotalUserTime", c_int64),
                ("ThisPeriodTotalKernelTime", c_int64), ("TotalPageFaultCount", DWORD), ("TotalProcesses", DWORD),
                ("ActiveProcesses", DWORD), ("TotalTerminatedProcesses", DWORD)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("nLength", DWORD), ("lpSecurityDescriptor", c_void_p), ("bInheritHandle", BOOL)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", DWORD), ("lpReserved", c_wchar_p), ("lpDesktop", c_wchar_p), ("lpTitle", c_wchar_p),
                ("dwX", DWORD), ("dwY", DWORD), ("dwXSize", DWORD), ("dwYSize", DWORD), ("dwXCountChars", DWORD),
                ("dwYCountChars", DWORD), ("dwFillAttribute", DWORD), ("dwFlags", DWORD), ("wShowWindow", WORD),
                ("cbReserved2", WORD), ("lpReserved2", c_void_p), ("hStdInput", HANDLE), ("hStdOutput", HANDLE), ("hStdError", HANDLE)]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE), ("dwProcessId", DWORD), ("dwThreadId", DWORD)]


SIGNATURES = {
    "CreateJobObjectW": ([c_void_p, c_wchar_p], HANDLE),
    "SetInformationJobObject": ([HANDLE, c_int32, c_void_p, DWORD], BOOL),
    "QueryInformationJobObject": ([HANDLE, c_int32, c_void_p, DWORD, c_void_p], BOOL),
    "CreatePipe": ([c_void_p, c_void_p, c_void_p, DWORD], BOOL),
    "SetHandleInformation": ([HANDLE, DWORD, DWORD], BOOL),
    "InitializeProcThreadAttributeList": ([c_void_p, DWORD, DWORD, c_void_p], BOOL),
    "UpdateProcThreadAttribute": ([c_void_p, DWORD, c_size_t, c_void_p, c_size_t, c_void_p, c_void_p], BOOL),
    "DeleteProcThreadAttributeList": ([c_void_p], None),
    "CreateProcessW": ([c_wchar_p, c_wchar_p, c_void_p, c_void_p, BOOL, DWORD, c_void_p, c_wchar_p, c_void_p, c_void_p], BOOL),
    "GetPriorityClass": ([HANDLE], DWORD),
    "AssignProcessToJobObject": ([HANDLE, HANDLE], BOOL),
    "IsProcessInJob": ([HANDLE, HANDLE, c_void_p], BOOL),
    "ResumeThread": ([HANDLE], DWORD),
    "TerminateProcess": ([HANDLE, c_uint32], BOOL),
    "TerminateJobObject": ([HANDLE, c_uint32], BOOL),
    "WaitForSingleObject": ([HANDLE, DWORD], DWORD),
    "GetExitCodeProcess": ([HANDLE, c_void_p], BOOL),
    "CreateFileW": ([c_wchar_p, DWORD, DWORD, c_void_p, DWORD, DWORD, HANDLE], HANDLE),
    "CloseHandle": ([HANDLE], BOOL),
}


class ContainmentError(RuntimeError):
    """Execution refused: the worker could not be admitted into an owned finite Job.

    evidence records the checked cleanup; child_may_remain is true when a created child was not observed to exit.
    """

    def __init__(self, message, evidence=None, cleanup_owner=None):
        super().__init__(message)
        self.evidence = evidence
        self.child_may_remain = bool(evidence and evidence.get("child_retired") is False)
        self.cleanup_owner = cleanup_owner  # Strong, recoverable ownership; evidence alone cannot close a handle.


@dataclass(frozen=True)
class JobLimits:
    process_memory: int
    job_memory: int
    affinity: int  # Exactly one logical CPU.
    active_processes: int = 1


def last_error(api):
    """Win32 last error of the calling thread (the injected API supplies it on non-Windows test hosts)."""
    reader = getattr(api, "last_error", None)
    return reader() if reader is not None else ctypes.get_last_error()


def kernel32():
    if sys.platform != "win32":
        raise ContainmentError("Native execution requires a Windows Job; this host is unsupported and no uncontained fallback exists")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, (arguments, result) in SIGNATURES.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api


def environment_block(env):
    if len(env) > MAX_ENV_ENTRIES:
        raise ContainmentError("Environment exceeds its entry bound")
    total = 1
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ContainmentError("Invalid environment entry")
        total += len(key) + len(value) + 2
        if total > MAX_ENV_CHARS:
            raise ContainmentError("Environment exceeds its total character bound")
        if not key or "=" in key[1:] or "\0" in key or "\0" in value:
            raise ContainmentError("Invalid environment entry")
    return "".join(f"{k}={v}\0" for k, v in sorted(env.items(), key=lambda item: item[0].upper())) + "\0"


def command_line(argv):
    if not 1 <= len(argv) <= MAX_ARGUMENTS:
        raise ContainmentError("Worker argument count exceeds its bound")
    total = 0
    for argument in argv:
        if not isinstance(argument, str):
            raise ContainmentError("Worker arguments must be strings")
        total += len(argument) + 3  # Spaces, quotes and a trailing slash can expand quoting.
        if total > MAX_COMMAND_CHARS:
            raise ContainmentError("Worker command line exceeds its character bound")
        if not argument or any(c in argument for c in '"\0\r\n'):
            raise ContainmentError("Worker arguments must be nonempty and contain no quotes or control characters")
    command = subprocess.list2cmdline(argv)
    if len(command) >= MAX_COMMAND_CHARS:
        raise ContainmentError("Worker command line exceeds its character bound")
    return command


def valid_handle(handle):
    return handle not in (None, 0, INVALID_HANDLE_VALUE)


class JobProcess:
    """A resumed worker inside its owned Job. It owns the Job and process handles and the three pipe descriptors."""

    def __init__(self, api, job, process, pid, descriptors, limits):
        self.api, self.job, self.process, self.pid, self.limits = api, job, process, pid, limits
        self._descriptors = dict(zip(("stdin", "stdout", "stderr"), descriptors))
        self._descriptor_outcomes = {}
        self._ambiguous_descriptors = {}  # Failed close numbers are evidence, never automatically reused as handles.
        self._lock = threading.Lock()  # A descriptor number is closed once, never after it could be reused.
        self.exit_code = None
        self.terminated = False
        self.retired = False  # Process exit and an empty Job were observed.
        self.closed = False   # Retired, and every owned handle and descriptor closed successfully.
        self._evidence = None
        self.cleanup = None

    @property
    def cleanup_pending(self):
        return not self.closed

    def retry_cleanup(self):
        """Operator-only retry after affirmative retirement; preserve the first closure receipt.

        Failed CRT close is ambiguous: descriptor numbers can be reused, so they are never retried.
        A raw Win32 handle is retained on failure and can be retried while this owner is kept alive.
        """
        if not self.retired:
            return self.confirm_closed()
        with self._lock:
            handles = {name: self._close_handle(name) for name in ("process", "job")}
            closed = all(v == "closed" for v in (*handles.values(), *self._descriptor_outcomes.values()))
            self.cleanup = {"handles": handles, "descriptors": dict(self._descriptor_outcomes), "handles_closed": closed,
                            "ambiguous_descriptor_numbers": dict(self._ambiguous_descriptors),
                            "disposition": "released" if closed else "retained: operator disposition required"}
            self.closed = closed
            return self.cleanup

    def _descriptor(self, name):
        descriptor = self._descriptors.get(name)
        if descriptor is None:
            raise OSError(errno.EBADF, f"worker {name} descriptor is closed")
        return descriptor

    def write(self, data):
        view = memoryview(data)
        while view:
            view = view[os.write(self._descriptor("stdin"), view):]

    def close_stdin(self):
        outcome = self._close_descriptor("stdin")
        if outcome != "closed":
            raise OSError(errno.EIO, "worker stdin " + outcome)

    def read_stdout(self, size):
        return os.read(self._descriptor("stdout"), size)

    def read_stderr(self, size):
        return os.read(self._descriptor("stderr"), size)

    def _close_descriptor(self, name):
        with self._lock:
            descriptor = self._descriptors.get(name)
            if descriptor is None:
                return self._descriptor_outcomes.get(name, "closed")
            self._descriptors[name] = None
            try:
                os.close(descriptor)
                outcome = "closed"
            except OSError as error:
                outcome = f"close_failed: errno {error.errno}"
                self._ambiguous_descriptors[name] = descriptor
            self._descriptor_outcomes[name] = outcome
            return outcome

    def _close_handle(self, name):
        handle = getattr(self, name)
        if handle is None:
            return "closed"
        if self.api.CloseHandle(handle):
            setattr(self, name, None)
            return "closed"
        return f"close_failed: Win32 error {last_error(self.api)}"  # Retained; never closed twice or claimed closed.

    def poll(self):
        if self.exit_code is None and self.process:
            state = self.api.WaitForSingleObject(self.process, 0)
            if state == WAIT_OBJECT_0:
                code = DWORD()
                if not self.api.GetExitCodeProcess(self.process, ctypes.byref(code)) or code.value == STILL_ACTIVE:
                    raise ContainmentError(f"GetExitCodeProcess failed: {last_error(self.api)}")
                self.exit_code = code.value
            elif state != WAIT_TIMEOUT:
                raise ContainmentError(f"WaitForSingleObject failed: {last_error(self.api)}")
        return self.exit_code

    def terminate(self):
        """Enforced whole-process retirement: terminate every process in the owned Job."""
        if self.retired:
            return
        if not self.api.TerminateJobObject(self.job, TERMINATION_CODE):
            raise ContainmentError(f"TerminateJobObject failed: {last_error(self.api)}")
        self.terminated = True

    def confirm_closed(self):
        """Retirement evidence once the process exited and the Job reports no active process; then release ownership.

        Returns None while the worker may still run. Each descriptor and handle closure is checked and reported;
        handles_closed is true only when every one of them succeeded.
        """
        if self._evidence is not None:
            return self._evidence
        if self.poll() is None:
            return None
        accounting, limits = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(), JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if not self.api.QueryInformationJobObject(self.job, JobObjectBasicAccountingInformation, ctypes.byref(accounting), ctypes.sizeof(accounting), None) \
                or not self.api.QueryInformationJobObject(self.job, JobObjectExtendedLimitInformation, ctypes.byref(limits), ctypes.sizeof(limits), None):
            raise ContainmentError(f"QueryInformationJobObject failed: {last_error(self.api)}")
        if accounting.ActiveProcesses != 0:
            return None
        self.retired = True
        descriptors = {name: self._close_descriptor(name) for name in ("stdin", "stdout", "stderr")}
        handles = {name: self._close_handle(name) for name in ("process", "job")}
        closed = all(outcome == "closed" for outcome in (*descriptors.values(), *handles.values()))
        self._evidence = {"process_exit_code": self.exit_code, "job_active_processes": accounting.ActiveProcesses,
                          "job_total_processes": accounting.TotalProcesses, "job_terminated_processes": accounting.TotalTerminatedProcesses,
                          "terminated_by_parent": self.terminated, "peak_process_memory_bytes": limits.PeakProcessMemoryUsed,
                          "peak_job_memory_bytes": limits.PeakJobMemoryUsed, "process_memory_limit_bytes": limits.ProcessMemoryLimit,
                          "job_memory_limit_bytes": limits.JobMemoryLimit, "descriptors": descriptors, "handles": handles,
                          "handles_closed": closed}
        self._evidence["ambiguous_descriptor_numbers"] = dict(self._ambiguous_descriptors)
        self.closed = closed
        return self._evidence


class _HandleSlot:
    """Allocated before acquisition; API output cells remain owned even if the display registry fails."""

    __slots__ = ("value", "output", "field", "enabled", "close_failed", "descriptor", "acquired")

    def __init__(self, output=None, field=None):
        self.value, self.output, self.field = None, output, field
        self.enabled, self.close_failed, self.descriptor = True, False, None
        self.acquired = False

    def get(self):
        if not self.enabled or (self.descriptor is not None and
                                (self.descriptor.number is not None or self.descriptor.ambiguous is not None)):
            return None  # open_osfhandle has transferred this raw resource to the CRT.
        value = self.value
        if self.output is not None:
            value = getattr(self.output, self.field) if self.field else self.output.value
        return value if valid_handle(value) else None


class _DescriptorSlot:
    """The actual adopted number is saved before any fallible registry change; ambiguous numbers are never retried."""

    __slots__ = ("number", "ambiguous")

    def __init__(self):
        self.number, self.ambiguous = None, None


class HeldFiles:
    """Read handles without write or delete sharing: while held, no other opener can change, replace or delete a file."""

    description = "Windows read handles without write or delete sharing"

    def __init__(self, api, paths):
        self.api, self.failures, self._handles = api, [], []  # Display registry, never the acquired-resource authority.
        if len(paths) > MAX_HELD_FILES:
            raise ContainmentError("Held source files exceed their total handle bound")
        paths = [str(path) for path in paths]
        if any(len(path) > 1024 for path in paths):
            raise ContainmentError("Held source path exceeds its length bound")
        # Every label, path and output slot is prepared before the first CreateFileW. The transport owns us before acquire.
        self._slots = [(path, Path(path).name, _HandleSlot()) for path in paths]

    def acquire(self):
        for path, name, slot in self._slots:
            slot.value = self.api.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, None)
            if slot.get() is None:
                raise ContainmentError(f"CreateFileW could not hold {name}: Win32 error {last_error(self.api)}")
            slot.acquired = True
            self._handles.append((name, slot.value))

    @property
    def count(self):
        return sum(slot.acquired for _, _, slot in self._slots)

    @property
    def pending(self):
        return any(slot.get() is not None for _, _, slot in self._slots)

    def release(self):
        """Try each currently owned handle once; retain failed raw values for explicit operator recovery."""
        failures = []
        for _, name, slot in reversed(self._slots):
            handle = slot.get()
            if handle is None:
                continue
            slot.acquired = True  # Also accounts for an exception before acquired-file bookkeeping completed.
            if not self.api.CloseHandle(handle):
                failures.append(f"{name}: Win32 error {last_error(self.api)}")
            else:
                slot.enabled = False  # Record affirmative closure before allocating receipt metadata.
                for index in range(len(self._handles) - 1, -1, -1):
                    if self._handles[index][1] == handle:
                        del self._handles[index]
                        break
        self.failures = failures[:8]
        retained = sum(slot.get() is not None for _, _, slot in self._slots)
        return {"held": self.count, "released": self.count - retained, "released_all": not retained,
                "retained": retained, "failures": list(self.failures), "by": self.description}

    retry_cleanup = release


class _Launch:
    """Every handle, descriptor and attribute list one launch owns, with the checked outcome of releasing each."""

    def __init__(self, api, osf):
        self.api, self.osf = api, osf
        self.handles = {}       # Display registry; fixed slots below are authoritative cleanup owners.
        self.descriptors = {}
        self.raw = {label: _HandleSlot() for label in ("job", "stdin_read", "stdin_write", "stdout_read",
                                                      "stdout_write", "stderr_read", "stderr_write", "process", "thread")}
        self.adopted = {label: _DescriptorSlot() for label in ("stdin_write", "stdout_read", "stderr_read")}
        for label, slot in self.adopted.items():
            self.raw[label].descriptor = slot
        self.pipes = [("stdin_read", "stdin_write", "stdin_read"),
                      ("stdout_read", "stdout_write", "stdout_write"), ("stderr_read", "stderr_write", "stderr_write")]
        for read, write, _ in self.pipes:
            self.raw[read].output, self.raw[write].output = HANDLE(), HANDLE()
        self.process_info = PROCESS_INFORMATION()
        self.raw["process"].output, self.raw["process"].field = self.process_info, "hProcess"
        self.raw["thread"].output, self.raw["thread"].field = self.process_info, "hThread"
        self.ambiguous_descriptors = {}
        self.outcomes = {}
        self.attributes = None  # Set only once InitializeProcThreadAttributeList succeeded.
        self.attribute_list = "not_initialized"
        self.pid = None
        self.created = False
        self.contained = False
        self.resumed = False
        self.child_retired = False
        self.receipt = None
        self.cleanup_receipt = None

    def check(self, ok, call):
        if not ok:
            raise ContainmentError(f"{call} failed: Win32 error {last_error(self.api)}")

    def own(self, label, handle, call):
        slot = self.raw[label]
        if slot.output is None or not valid_handle(getattr(slot.output, slot.field) if slot.field else slot.output.value):
            slot.value, slot.output = handle, None  # Also supports owners populated directly by finite fake controls.
        self.check(valid_handle(handle), call)
        self.handles[label] = handle
        return handle

    def close(self, label, retry=False):
        slot = self.raw[label]
        if not retry and slot.close_failed:
            return False  # A launch failure must not hide a failed call by silently retrying it in its exception handler.
        handle = slot.get()
        if handle is None:
            return True
        if self.api.CloseHandle(handle):
            slot.enabled, slot.close_failed = False, False
            self.handles.pop(label, None)
            self.outcomes[label] = "closed"
            return True
        slot.close_failed = True
        self.outcomes[label] = f"close_failed: Win32 error {last_error(self.api)}"  # Not retried and not claimed closed.
        return False

    def adopt(self, label, flags):
        """Hand one parent pipe end to a CRT descriptor; from then on only os.close may release it."""
        raw, adopted = self.raw[label], self.adopted[label]
        adopted.number = self.osf.open_osfhandle(raw.get(), flags)
        # The saved descriptor already suppresses raw closure, even before this flag or display-table update.
        raw.enabled = False
        self.handles.pop(label, None)
        self.descriptors[label] = adopted.number
        return adopted.number

    def delete_attributes(self):
        if self.attributes is not None:
            self.api.DeleteProcThreadAttributeList(self.attributes)
            self.attributes = None
            self.attribute_list = "deleted"

    def transfer(self, limits):
        """Construct the successor before dropping this owner's handles; a failed construction still has cleanup."""
        labels = ("stdin_write", "stdout_read", "stderr_read")
        descriptors = [self.adopted[label].number for label in labels]
        successor = JobProcess(self.api, self.raw["job"].get(), self.raw["process"].get(), self.pid, descriptors, limits)
        # Complete fallible bookkeeping before the fixed-slot handoff. Failed successor construction leaves us owning all.
        del self.handles["job"], self.handles["process"]
        for label in labels:
            del self.descriptors[label]
        self.raw["job"].enabled, self.raw["process"].enabled = False, False
        for label in labels:
            self.adopted[label].number = None
        return successor

    @property
    def pending(self):
        return bool(any(slot.get() is not None for slot in self.raw.values()) or
                    any(slot.number is not None or slot.ambiguous is not None for slot in self.adopted.values()) or
                    ((self.created or self.pid is not None) and not self.child_retired))

    def retry_cleanup(self):
        return self.cleanup(retry=True)

    def cleanup(self, retry=False):
        """Retire any created child, then release everything still owned; returns bounded evidence of each outcome."""
        api = self.api
        process = self.raw["process"].get()
        child_created = self.created or self.pid is not None or process is not None or self.raw["thread"].get() is not None
        evidence = {"pid": self.pid if self.pid is not None else self.process_info.dwProcessId or None,
                    "child_created": child_created, "contained": self.contained,
                    "resumed": self.resumed, "terminate": None, "wait": None, "child_exited": None}

        def wait():
            state = api.WaitForSingleObject(process, CLEANUP_WAIT_MS)
            evidence["wait"] = "exited" if state == WAIT_OBJECT_0 else "timeout" if state == WAIT_TIMEOUT else f"failed: Win32 error {last_error(api)}"
            evidence["child_exited"] = state == WAIT_OBJECT_0

        if process is not None and not self.child_retired:
            if self.contained:
                ok, call = api.TerminateJobObject(self.raw["job"].get(), TERMINATION_CODE), "TerminateJobObject"
            else:
                ok, call = api.TerminateProcess(process, TERMINATION_CODE), "TerminateProcess"  # Never resumed: no worker code ran.
            evidence["terminate"] = f"{call}: ok" if ok else f"{call}: failed, Win32 error {last_error(api)}"
            wait()
            # A contained launch must also observe an empty Job, not just its original process exit.
            if evidence["child_exited"] and self.contained:
                accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
                empty = api.QueryInformationJobObject(self.raw["job"].get(), JobObjectBasicAccountingInformation,
                                                      ctypes.byref(accounting), ctypes.sizeof(accounting), None)
                evidence["job_active_processes"] = accounting.ActiveProcesses if empty else None
                evidence["child_exited"] = bool(empty and accounting.ActiveProcesses == 0)
            self.child_retired = evidence["child_exited"] is True
        for label, slot in self.adopted.items():
            descriptor = slot.number
            if descriptor is None:
                continue
            # Consume once before calling os.close; a failed/unknown close cannot be retried after number reuse.
            slot.ambiguous, slot.number = descriptor, None
            try:
                os.close(descriptor)
                slot.ambiguous = None
                self.outcomes[label] = "closed"
            except OSError as error:
                self.outcomes[label] = f"close_failed: errno {error.errno}"
                self.ambiguous_descriptors[label] = descriptor
            self.descriptors.pop(label, None)
        for label in self.raw:
            if label in ("job", "process"):
                continue
            self.close(label, retry=retry)
        # Keep the process and Job handles until retirement is affirmative. They are the recovery control path.
        if not child_created or self.child_retired:
            for label in ("process", "job"):
                self.close(label, retry=retry)
        self.delete_attributes()
        evidence["child_retired"] = not evidence["child_created"] or self.child_retired
        evidence["handles"] = dict(self.outcomes)
        evidence["retained_handles"] = [label for label, slot in self.raw.items() if slot.get() is not None]
        evidence["ambiguous_descriptor_numbers"] = {label: slot.ambiguous for label, slot in self.adopted.items() if slot.ambiguous is not None}
        evidence["handles_closed"] = not self.pending
        evidence["attribute_list"] = self.attribute_list
        if retry:
            self.cleanup_receipt = evidence
        else:
            self.receipt = evidence
        return evidence


def summary(evidence):
    """One bounded line for logs and health: what was retired and what could not be released."""
    failed = sorted(label for label, outcome in evidence["handles"].items() if outcome != "closed")
    child = ("no child created" if not evidence["child_created"] else
             f"child {evidence['pid']} exited" if evidence["child_retired"] else f"child {evidence['pid']} NOT observed to exit; it may remain")
    return f"{child}; terminate {evidence['terminate'] or 'not needed'}; failed closures: {', '.join(failed) or 'none'}"


class WindowsJobTransport:
    """Launch exactly one suspended worker into a fresh finite kill-on-close Job, verify it, then resume."""

    def __init__(self, api=None, osf=None):
        self.api = api if api is not None else kernel32()
        if osf is None:
            import msvcrt as osf
        self.osf = osf
        self._pending = None  # At most one failed startup owner; no new resources until its explicit disposition.

    def _admit(self):
        if self._pending is not None and not self._pending.pending:
            self._pending = None  # Explicit cleanup on the same retained owner may have completed outside this facade.
        if self._pending is not None:
            raise ContainmentError("A failed native startup still owns resources; operator cleanup is required",
                                   cleanup_owner=self._pending)

    def retry_cleanup(self):
        owner = self._pending
        if owner is None:
            return None
        result = owner.retry_cleanup()
        if not owner.pending:
            self._pending = None
        return result

    def hold(self, paths):
        self._admit()
        owned = HeldFiles(self.api, paths)  # Metadata preparation owns no native resource.
        self._pending = owned  # Reachable before any CreateFileW or fallible acquired-resource bookkeeping.
        try:
            owned.acquire()
        except BaseException as error:
            try:
                evidence = {"held": owned.release(), "child_retired": True}
            except Exception:
                evidence = {"held": {"released_all": False}, "child_retired": True}
            if not owned.pending:
                self._pending = None
            error.cleanup_owner, error.evidence = self._pending, evidence
            raise
        self._pending = None
        return owned

    def launch(self, argv, env, cwd, limits):
        api = self.api
        self._admit()
        if not (type(limits.process_memory) is int and type(limits.job_memory) is int and 0 < limits.process_memory <= limits.job_memory <= 1 << 40):
            raise ContainmentError("Finite process and job memory limits are required")
        if not (type(limits.affinity) is int and 0 < limits.affinity < 1 << 64 and limits.affinity & (limits.affinity - 1) == 0):
            raise ContainmentError("Exactly one logical CPU is required")
        if type(limits.active_processes) is not int or limits.active_processes != 1:
            raise ContainmentError("Exactly one active process is required")
        commandline = ctypes.create_unicode_buffer(command_line(argv))
        block = ctypes.create_unicode_buffer(environment_block(env))
        owned = _Launch(api, self.osf)
        try:
            owned.raw["job"].value = api.CreateJobObjectW(None, None)  # Store before fallible registry insertion.
            job = owned.own("job", owned.raw["job"].value, "CreateJobObjectW")  # Unnamed; default security is not inheritable.
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = LIMIT_FLAGS
            info.BasicLimitInformation.ActiveProcessLimit = limits.active_processes
            info.BasicLimitInformation.Affinity = limits.affinity
            info.ProcessMemoryLimit, info.JobMemoryLimit = limits.process_memory, limits.job_memory
            owned.check(api.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)), "SetInformationJobObject")
            observed = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            owned.check(api.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(observed), ctypes.sizeof(observed), None), "QueryInformationJobObject")
            flags = observed.BasicLimitInformation.LimitFlags
            if (flags & LIMIT_FLAGS) != LIMIT_FLAGS or flags & BREAKAWAY_FLAGS \
                    or observed.ProcessMemoryLimit != limits.process_memory or observed.JobMemoryLimit != limits.job_memory \
                    or observed.BasicLimitInformation.ActiveProcessLimit != limits.active_processes \
                    or observed.BasicLimitInformation.Affinity != limits.affinity:
                raise ContainmentError("Job limits did not take effect exactly")
            security = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, 0)
            children = []
            for read_label, write_label, child in owned.pipes:
                read, write = owned.raw[read_label].output, owned.raw[write_label].output
                owned.check(api.CreatePipe(ctypes.byref(read), ctypes.byref(write), ctypes.byref(security), 65536), "CreatePipe")
                owned.own(read_label, read.value, "CreatePipe")
                owned.own(write_label, write.value, "CreatePipe")
                owned.check(api.SetHandleInformation(owned.handles[child], HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT), "SetHandleInformation")
                children.append(child)
            size = c_size_t(0)
            if api.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size)) or last_error(api) != ERROR_INSUFFICIENT_BUFFER:
                raise ContainmentError("InitializeProcThreadAttributeList size query failed")
            if not 0 < size.value <= MAX_ATTRIBUTE_BYTES:
                raise ContainmentError("Attribute list exceeds its finite allocation bound")
            attributes = ctypes.create_string_buffer(size.value)
            owned.check(api.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)), "InitializeProcThreadAttributeList")
            owned.attributes, owned.attribute_list = attributes, "initialized"  # Only an initialized list is ever deleted.
            inherit = (HANDLE * 3)(*(owned.handles[child] for child in children))  # Only these three handles reach the child.
            owned.check(api.UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, ctypes.sizeof(inherit), None, None), "UpdateProcThreadAttribute")
            startup = STARTUPINFOEXW()
            startup.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = (owned.handles[c] for c in children)
            startup.lpAttributeList = ctypes.cast(attributes, c_void_p)
            process_info = owned.process_info  # Both output handles have fixed owners before CreateProcessW writes either.
            created = api.CreateProcessW(argv[0], commandline, None, None, 1, CREATION_FLAGS, block, str(cwd),
                                         ctypes.byref(startup), ctypes.byref(process_info))
            owned.created = bool(created)  # Child existence remains owned even if PID/handle display registration fails.
            owned.check(created, "CreateProcessW")
            owned.pid = process_info.dwProcessId
            owned.own("process", process_info.hProcess, "CreateProcessW")
            owned.own("thread", process_info.hThread, "CreateProcessW")
            owned.delete_attributes()
            for child in children:  # The child owns its copies; keeping ours would hide EOF.
                if not owned.close(child):
                    raise ContainmentError(f"CloseHandle of the parent's copy of {child} failed: {owned.outcomes[child]}")
            priority = api.GetPriorityClass(owned.handles["process"])
            owned.check(priority, "GetPriorityClass")
            if priority != BELOW_NORMAL_PRIORITY_CLASS:
                raise ContainmentError(f"Worker priority class is {priority:#x}, not below normal")
            # The child has not executed one instruction yet; refuse it unless it is inside the owned Job.
            owned.check(api.AssignProcessToJobObject(job, owned.handles["process"]), "AssignProcessToJobObject")
            owned.contained = True  # Successful assignment already creates Job ownership, even if its verification fails.
            inside = BOOL(0)
            owned.check(api.IsProcessInJob(owned.handles["process"], job, ctypes.byref(inside)), "IsProcessInJob")
            if not inside.value:
                raise ContainmentError("Worker is not inside the owned Job")
            for label, mode in (("stdin_write", os.O_WRONLY), ("stdout_read", os.O_RDONLY), ("stderr_read", os.O_RDONLY)):
                owned.adopt(label, mode)
            previous = api.ResumeThread(owned.handles["thread"])
            owned.resumed = previous in (0, 1)  # The worker thread may now be running.
            if previous != 1:
                raise ContainmentError(f"ResumeThread returned {previous}; expected one prior suspension")
            if not owned.close("thread"):
                raise ContainmentError(f"CloseHandle of the worker thread failed: {owned.outcomes['thread']}")
            return owned.transfer(limits)
        except BaseException as error:
            self._pending = owned  # Install ownership before cleanup, including a cleanup routine that itself throws.
            try:
                evidence = owned.cleanup()
            except Exception as cleanup_error:
                evidence = {"pid": owned.pid or owned.process_info.dwProcessId or None,
                            "child_created": owned.created or owned.pid is not None, "child_retired": owned.child_retired,
                            "contained": owned.contained, "terminate": "cleanup raised " + type(cleanup_error).__name__,
                            "handles": dict(owned.outcomes), "handles_closed": False,
                            "retained_handles": [label for label, slot in owned.raw.items() if slot.get() is not None]}
                owned.receipt = evidence
            if not owned.pending:
                self._pending = None
            error.cleanup_owner = owned if owned.pending else None
            error.evidence = evidence
            error.child_may_remain = evidence["child_retired"] is False
            if not isinstance(error, Exception):
                raise
            raise ContainmentError(f"{type(error).__name__ if not isinstance(error, ContainmentError) else 'Launch refused'}: "
                                   f"{str(error)[:256]}; cleanup: {summary(evidence)}", evidence, self._pending) from error
        finally:
            owned.delete_attributes()
