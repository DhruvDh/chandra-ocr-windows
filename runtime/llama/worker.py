"""One opt-in llama-server Job owner. Import and status reads start nothing."""

import copy
import logging
import threading
import time

from runtime.native.winjob import WindowsJobTransport

TAIL_BYTES = 64 * 1024
READ_BYTES = 8192
POLL_SECONDS = 0.05


class AdmissionRefused(RuntimeError):
    """Expected pre-launch headroom refusal; cleanup is required, control is not faulty."""


class _StartupCanceled(RuntimeError):
    """A requested stop observed at a startup barrier, rather than a control fault."""


class OwnedWorker:
    """Keep the authenticated files and every cleanup owner until affirmative retirement.

    The backend supplies request/startup deadlines and calls retire on every terminal path.
    retire deliberately keeps waiting on uncertain cleanup; it never returns service capacity
    while a child, reader, control handle or held file might remain owned.
    """

    def __init__(self, config, transport=None, *, diagnostic=None):
        self.config, self._transport = config, transport
        self._diagnostic = diagnostic
        self._diagnostic_launch_evidence = {}
        self._tail_clipped = {name: False for name in ("stdout", "stderr")}
        self._state_lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._retire_lock = threading.Lock()
        self._startup_done = threading.Event()
        self._startup_done.set()
        self._stop = threading.Event()
        self._started = self._closed = self._termination_sent = False
        self._held = self._process = self._cleanup_owner = None
        self._readers = {name: None for name in ("stdout", "stderr")}
        self._drained = {name: threading.Event() for name in self._readers}
        self._tails = {name: b"" for name in self._readers}
        self._faults = []
        self._exit_code = self._closure = self._cleanup = self._file_release = None

    def diagnostic_record(self, *, stage=None, retirement=None, freeze=False, required=True):
        if self._diagnostic is None:
            return
        snapshot = self.snapshot()
        snapshot["tail_clipped"] = dict(self._tail_clipped)
        snapshot["partial_launch_pending"] = self._cleanup_owner is not None
        snapshot["launch_evidence"] = self._diagnostic_launch_evidence
        if snapshot["pid"] is None and self._cleanup_owner is not None:
            snapshot["pid"] = (getattr(self._cleanup_owner, "pid", None) or
                               getattr(getattr(self._cleanup_owner, "process_info", None), "dwProcessId", None))
        if snapshot["pid"] is None:
            snapshot["pid"] = self._diagnostic_launch_evidence.get("pid")
        try:
            self._diagnostic.record(snapshot, stage=stage, retirement=retirement, freeze=freeze)
        except Exception:
            if required:
                raise
            # Retirement and the original failure must survive diagnostic sink failure.
            try:
                logging.getLogger("chandra.llama.startup").error("Private startup evidence write failed during owned retirement")
            except BaseException:
                pass

    def _fault(self, operation, error):
        message = f"{operation}: {type(error).__name__}: {str(error)[:256]}"
        with self._state_lock:
            if message not in self._faults:
                self._faults = (self._faults + [message])[-8:]

    def start(self, port, api_key, alias, admit=None):
        with self._state_lock:
            if self._started or self._closed or self._stop.is_set():
                raise RuntimeError("llama worker startup is one-shot and was already started or stopped")
            self._started = True
            self._startup_done.clear()
        try:
            if self._diagnostic is not None:
                self._diagnostic.begin(self.config.config_sha256, api_key, alias)
            self.diagnostic_record(stage="hold_inputs")
            if self._transport is None:
                self._transport = WindowsJobTransport()
            self._held = self._transport.hold(self.config.files())
            self.diagnostic_record(stage="authenticate_inputs")
            self.config.authenticate()  # Hash/inventory checks occur under the read-only file holds.
            with self._control_lock:
                if self._stop.is_set():
                    raise _StartupCanceled("llama worker startup was canceled before launch")
                if admit is not None:
                    self.diagnostic_record(stage="second_admission")
                    admit()
                if self._stop.is_set():
                    raise _StartupCanceled("llama worker startup was canceled before launch")
                if self._diagnostic is not None:
                    self.diagnostic_record(stage="launch_native")
                    if self._stop.is_set():
                        raise _StartupCanceled("llama worker startup was canceled before launch")
                self._process = self._transport.launch(
                    self.config.argv(port, api_key, alias), self.config.environment(),
                    self.config.runtime_dir, self.config.job_limits())
            self.diagnostic_record(stage="native_started")
            # Once transport launch is admitted it owns the child, even if stop races with ResumeThread.
            if self._stop.is_set():
                self.request_stop()
            for name in self._readers:
                thread = threading.Thread(target=self._drain, args=(name,), name=f"llama-{name}", daemon=True)
                self._readers[name] = thread  # Retain even a Thread.start failure after partial startup.
                thread.start()
            self._process.close_stdin()
            if self._stop.is_set():
                raise _StartupCanceled("llama worker startup was canceled")
            return self
        except BaseException as error:
            self._cleanup_owner = getattr(error, "cleanup_owner", None)
            evidence = getattr(error, "evidence", None)
            if self._diagnostic is not None and type(evidence) is dict:
                self._diagnostic_launch_evidence = {key: evidence.get(key)
                    for key in ("pid", "handles_closed", "child_created", "child_retired", "child_exited", "contained", "resumed")
                    if type(evidence.get(key)) in (int, bool)}
            if not isinstance(error, (AdmissionRefused, _StartupCanceled)):
                self._fault("startup", error)
            self.request_stop()
            self.diagnostic_record(retirement="startup_failed", required=False)
            raise
        finally:
            self._startup_done.set()

    def _drain(self, name):
        try:
            reader = getattr(self._process, "read_" + name)
            while True:
                chunk = reader(READ_BYTES)
                if not chunk:
                    break
                with self._state_lock:
                    self._tail_clipped[name] |= len(self._tails[name]) + len(chunk) > TAIL_BYTES
                    self._tails[name] = (self._tails[name] + chunk)[-TAIL_BYTES:]
                try:
                    self.diagnostic_record()
                except Exception as error:
                    self._fault("startup diagnostic", error)
                    self.request_stop()  # Keep draining the owned pipes through termination/EOF.
        except BaseException as error:
            self._fault(name + " drain", error)
            self.request_stop()
        finally:
            self._drained[name].set()

    def request_stop(self):
        """Record cancellation immediately; never wait behind startup or cleanup."""
        self._stop.set()
        if not self._control_lock.acquire(blocking=False):
            return
        try:
            process = self._process
            if process is not None and not self._closed and not self._termination_sent:
                try:
                    process.terminate()
                    self._termination_sent = True
                except Exception as error:
                    self._fault("terminate", error)
        finally:
            self._control_lock.release()

    def alive(self):
        """A non-waiting poll only; failure is visible and conservatively retains ownership."""
        process = self._process
        if process is None or self._closed:
            return False
        if not self._control_lock.acquire(blocking=False):
            return True
        try:
            code = process.poll()
            with self._state_lock:
                self._exit_code = code
            return code is None
        except Exception as error:
            self._fault("poll", error)
            self._stop.set()
            return True
        finally:
            self._control_lock.release()

    def _retire_process(self):
        process = self._process
        if process is None:
            return True
        self.request_stop()
        with self._control_lock:
            code = process.poll()
        with self._state_lock:
            self._exit_code = code
        if code is None:
            return False
        for name, thread in self._readers.items():
            if thread is None or thread.ident is None:
                if not self._drained[name].is_set():
                    self._drain(name)  # A failed Thread.start still leaves this pipe owned; child has exited.
            else:
                thread.join(POLL_SECONDS)
        if any(not self._drained[name].is_set() or thread is not None and thread.is_alive()
               for name, thread in self._readers.items()):
            return False
        with self._control_lock:
            if self._closure is None:
                closure = process.confirm_closed()
                with self._state_lock:
                    self._closure = copy.deepcopy(closure)
            else:
                closure = self._closure
            if closure is None:
                return False
            if closure.get("handles_closed") is True:
                return True
            # The transport retries retained raw handles only. Ambiguous CRT descriptor numbers
            # must never be closed again; retain this owner rather than pretending closure.
            if closure.get("ambiguous_descriptor_numbers"):
                self._fault("retirement", RuntimeError("ambiguous descriptor closure requires operator disposition"))
                return False
            cleanup = process.retry_cleanup()
            with self._state_lock:
                self._cleanup = copy.deepcopy(cleanup)
            if cleanup.get("handles_closed") is True:
                return True
            self._fault("retirement", RuntimeError("process/Job handles remain owned"))
            return False

    def retire(self):
        """Block until startup, Job, drains, control resources and held files have retired."""
        self.request_stop()
        if not self._closed:
            self.diagnostic_record(retirement="retirement_requested", required=False)
        with self._retire_lock:
            with self._state_lock:
                started = self._started  # start checks stop and clears the barrier under this same lock.
            if started:
                self._startup_done.wait()
            while not self._closed:
                try:
                    control = self._cleanup_owner
                    if control is not None:
                        cleanup = control.retry_cleanup()
                        with self._state_lock:
                            self._cleanup = copy.deepcopy(cleanup)
                        if control.pending:
                            self._fault("startup cleanup", RuntimeError("partial startup resources remain owned"))
                            time.sleep(POLL_SECONDS)
                            continue
                        self._cleanup_owner = None
                    if not self._retire_process():
                        self.diagnostic_record(retirement="waiting_native_closure", required=False)
                        time.sleep(POLL_SECONDS)
                        continue
                    if self._held is not None:
                        self.diagnostic_record(retirement="releasing_inputs", required=False)
                        release = self._held.release()
                        with self._state_lock:
                            self._file_release = copy.deepcopy(release)
                        if release.get("released_all") is not True:
                            self._fault("file release", RuntimeError("authenticated file handles remain owned"))
                            time.sleep(POLL_SECONDS)
                            continue
                        self._held = None
                    with self._state_lock:
                        self._closed = True
                    self.diagnostic_record(retirement="closed", required=False)
                except Exception as error:
                    self._fault("retirement", error)
                    time.sleep(POLL_SECONDS)
            return self.snapshot()

    def snapshot(self):
        """Copy bounded status only: no polling, authentication, cleanup or launch."""
        with self._state_lock:
            result = {"engine": self.config.engine, "config_sha256": self.config.config_sha256,
                      "started": self._started, "starting": not self._startup_done.is_set(),
                      "stop_requested": self._stop.is_set(), "closed": self._closed,
                      "pending": self._started and not self._closed,
                      "pid": getattr(self._process, "pid", None), "exit_code": self._exit_code,
                      "fault": self._faults[-1] if self._faults else None, "faults": list(self._faults),
                      "closure": copy.deepcopy(self._closure), "cleanup": copy.deepcopy(self._cleanup),
                      "file_release": copy.deepcopy(self._file_release)}
            tails = dict(self._tails)
        result.update({name + "_tail": tail.decode("utf-8", errors="replace") for name, tail in tails.items()})
        return result
