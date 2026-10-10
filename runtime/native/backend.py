"""DirectCompute native Backend for chandra_service: one contained resident worker, never replayed.

Construction, import, health and model listings start nothing. The first request prepares its input on
the CPU, then joins a single shared startup of chandra-worker.exe inside an owned Windows Job. Every
event is validated and correlated; capacity is held until the worker reports terminal released:true or
whole-process retirement is confirmed. A worker that fails, crashes, desynchronizes or ends uncleanly
is retired as a whole and the backend refuses further work until the service is restarted.

A session publishes its retirement evidence to the backend before it reports closed, so no caller,
health reader or capacity release can observe closure without the matching failure and receipt. Lock
order: a backend lock holder may only try Session.cond without waiting; no session lock holder takes the
backend lock. Publication runs with no session lock held. Any exception in a channel thread fails and retires the worker.

Publication also makes the backend owner of the session's finalization (waking its callers) until the
session marks it finished as its last, allocation-free step. Until then no new worker is admitted, and a
failure the session records meanwhile is the backend's failure immediately, not after a supervisor retry.

Closure and finalization waits never take Session.cond: they read the session's one-way flags and wake on an
advisory event whose lock no caller code holds, so a thread holding the condition (a wake-up step or a failure
handler) cannot extend their deadline through initial acquisition, waiting or reacquisition.
"""
import collections
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import stat
import threading
import time

from . import protocol as P
from . import shaders
from .decode import IncrementalDecoder
from .prepare import Pinned, PinnedProcessor, RequestRefused, discard, prepare

LOG = logging.getLogger("chandra.native")
STDERR_TAIL = 16384
RESULT_CAP = 16 * 1024 * 1024
JOURNAL_CAP = 16 * 1024 * 1024
WRITE_QUEUE = 64
RECENT = 64
WAIT_POLL = 0.05  # Lock-free waiters re-read the session's flags at least this often, even without a wake-up.


PUBLICATION_FAILED = "Worker retirement evidence could not be published; the backend retains the closed session and its receipt; restart the service"


class NativeFailure(RuntimeError):
    """Internal failure; never a ValueError, so the service reports 502 rather than a client error."""


class _Request:
    def __init__(self, ident, output, prepared):
        self.id, self.output, self.prepared = ident, output, prepared
        self.state = "submitted"
        self.items = collections.deque()
        self.tokens = []
        self.terminal = None
        self.rejection = None
        self.cancel_sent = False
        self.cancel_deadline = None
        self.source_checked = False
        self.source_stages = []


class _CleanupOwner:
    """One failed startup or retired session's resources; the backend keeps this owner until explicit disposal.

    No retry occurs in health, unload or close. Operator cleanup never reopens admission or changes the original receipt.
    """

    def __init__(self, config):
        self.config = config
        self.snapshot = self.held = self.process = self.session = self.control = self.custody = None
        self.result = None
        self.pending = True
        self.started_failed = False
        self.unknown_child = False

    def identity(self):
        return {"pending": self.pending, "snapshot": self.snapshot.name if self.snapshot else None,
                "pid": getattr(self.process, "pid", getattr(self.control, "pid", None)), "disposition": self.result,
                "operator_action": "cleanup_retained(); admission remains failed; restart only after review"}

    def dispose(self, retry=False):
        result = {"process": None, "startup_control": None, "held": None, "snapshot_removal": None, "threads_joined": None}
        safe = not self.unknown_child
        if self.control is not None:
            if retry:
                result["startup_control"] = self.control.retry_cleanup()
            else:
                result["startup_control"] = getattr(self.control, "receipt", None)
            safe = not self.control.pending
        if self.process is not None:
            try:
                if self.started_failed and not getattr(self.process, "retired", False):
                    self.process.terminate()
                closure = self.process.retry_cleanup() if retry and hasattr(self.process, "retry_cleanup") else self.process.confirm_closed()
                result["process"] = closure
                safe = safe and closure is not None and closure.get("handles_closed") is True
            except Exception as error:
                result["process"] = {"error": f"{type(error).__name__}: {str(error)[:200]}"}
                safe = False
        # Neither files nor snapshot are released while a child or its controlling handles may still be owned.
        if safe and self.custody is not None:
            result["custody"] = self.custody.retry_cleanup() if retry else self.custody.retire()
            safe = not self.custody.pending
        elif safe:
            if self.held is not None:
                try:
                    result["held"] = self.held.release()
                    safe = result["held"].get("released_all") is True
                except Exception as error:
                    result["held"] = {"released_all": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
                    safe = False
            if safe and self.snapshot is not None:
                creation_error = getattr(self.snapshot, "creation_removal_error", None)
                result["snapshot_removal"] = self.snapshot.remove() if retry or creation_error is None else creation_error
                safe = result["snapshot_removal"] is None
        if self.started_failed and self.session is not None:
            # A failed Thread.start may leave channel threads but no supervisor. Wake/join them only after closure.
            if safe and self.session.cond.acquire(timeout=0):
                try:
                    self.session.closed = self.session.finalized = True
                    self.session.cond.notify_all()
                finally:
                    self.session.cond.release()
                deadline = time.monotonic() + self.config.terminate_seconds
                for thread in self.session.threads:
                    if thread.ident is not None:
                        thread.join(max(0.0, deadline - time.monotonic()))
            result["threads_joined"] = not any(t.is_alive() for t in self.session.threads)
            safe = safe and result["threads_joined"]
        self.pending = not safe
        result["pending"] = self.pending
        self.result = result
        return result


class Session:
    """One worker process. Reader, stderr, writer and supervisor threads never block on a client."""

    def __init__(self, process, config, expectation, on_closed, custody=None, clock=time.monotonic):
        self.process, self.config, self.expectation, self.on_closed, self.clock = process, config, expectation, on_closed, clock
        self.custody = custody       # Host custody of the shader snapshot this worker compiles from.
        self.cond = threading.Condition()
        self.stream = P.EventStream()
        self.ready = None
        self.resident = None
        self.startup_failure = None
        self.requests = {}
        self.recent = collections.OrderedDict()
        self.failure = None          # Channel, protocol, crash or contract failure: never reuse.
        self.unusable = None         # Reason no new request may be admitted.
        self.worker_closing = None
        self.exit_event = None
        self.eof = False
        self.reader_done = False     # No further event will be read, at EOF or after a reader failure.
        self.exited_at = None
        self.retire_reason = None
        self.retire_force = False
        self.shutdown_sent = False
        self.retire_deadline = None
        self.terminated_at = None
        self.terminate_error = None
        self.exit_code = None
        self.closed = False          # Set only after the retirement evidence was offered to on_closed.
        self.published = False       # Set by the backend, under its own lock, only after it holds the evidence.
        self.finalized = False       # Set last, after closure wake-up work; until then the backend owns this finalization.
        self.settled = threading.Event()  # Advisory wake-up for lock-free waiters, set after finalized.
        self.publication_error = None
        self.closure = None          # Cached receipts: a retried closure never re-confirms or re-retires custody.
        self.custody_result = None
        self.unconfirmed = False
        self.evidence = None
        self.thread_errors = []
        self.events_seen = 0
        self.counts = collections.Counter()
        self.writes = collections.deque()
        self.write_started = None
        self.last_write = clock()
        self.stderr = collections.deque()
        self.stderr_bytes = 0
        self.stderr_total = 0
        self.threads = [threading.Thread(target=self._guard("event reader", self._read, self._drain_stdout), name="chandra-native-events", daemon=True),
                        threading.Thread(target=self._guard("stderr reader", self._read_stderr), name="chandra-native-stderr", daemon=True),
                        threading.Thread(target=self._guard("stdin writer", self._write), name="chandra-native-stdin", daemon=True),
                        threading.Thread(target=self._supervise, name="chandra-native-supervisor", daemon=True)]

    def start(self):
        for thread in self.threads:
            thread.start()

    # ----- channel threads -----
    def _guard(self, name, body, after_failure=None):
        """Run one channel thread; any exception fails and retires the worker instead of leaving a dead channel usable."""
        def run():
            try:
                body()
            except BaseException as error:
                with self.cond:
                    self.thread_errors = (self.thread_errors + [f"{name}: {type(error).__name__}: {str(error)[:200]}"])[-8:]
                    self._fail_locked(f"Native {name} thread failed: {type(error).__name__}")
                if after_failure is not None:
                    after_failure()
        return run

    def _drain_stdout(self):
        """After a reader failure keep stdout drained, so a worker being retired never blocks writing an event."""
        try:
            while self.process.read_stdout(65536):
                pass
        except Exception:
            pass
        finally:
            with self.cond:
                self.reader_done = True
                self.cond.notify_all()

    def _read(self):
        buffer = bytearray()
        while True:
            try:
                chunk = self.process.read_stdout(65536)
            except OSError:
                chunk = b""
            if not chunk:
                with self.cond:
                    self.eof = True
                    self.reader_done = True
                    if buffer and not self.failure:
                        self._fail_locked("Event channel ended inside an event")
                    if self.exit_event is None and not self.failure:
                        self._fail_locked("Event channel closed before the worker's exit event")
                    self.cond.notify_all()
                return
            buffer += chunk
            while True:
                end = buffer.find(b"\n")
                if end < 0:
                    break
                line = bytes(buffer[:end])
                del buffer[:end + 1]
                self._accept(line)
            if len(buffer) >= P.MAX_EVENT_BYTES:
                with self.cond:
                    self._fail_locked("Event exceeds 65536 bytes")
                buffer.clear()  # Keep draining so a dying worker never blocks on its stdout.

    def _accept(self, line):
        with self.cond:
            if self.failure:
                return  # The channel is already untrustworthy; drain without acting.
            try:
                event = self.stream.accept(line)
                self.events_seen += 1
                self._dispatch(event)
            except P.ProtocolViolation as error:
                self._fail_locked(str(error))
            except Exception as error:  # An unforeseen shape is still a channel failure, never a dead reader.
                self._fail_locked("Event handling failed: " + type(error).__name__)
            self.cond.notify_all()

    def _read_stderr(self):
        while True:
            try:
                chunk = self.process.read_stderr(4096)
            except OSError:
                chunk = b""
            if not chunk:
                return
            with self.cond:
                self.stderr.append(chunk)
                self.stderr_bytes += len(chunk)
                self.stderr_total += len(chunk)
                while self.stderr_bytes > STDERR_TAIL:
                    self.stderr_bytes -= len(self.stderr.popleft())

    def _write(self):
        while True:
            with self.cond:
                while not self.writes and not self.closed:
                    self.cond.wait(0.5)
                if self.closed:
                    return
                line = self.writes.popleft()
                if line is None:
                    self.write_started = None
                    break
                self.write_started = self.clock()
            try:
                self.process.write(line)
            except (OSError, ValueError) as error:
                with self.cond:
                    self.write_started = None
                    if not self.failure and self.exit_code is None and self.exit_event is None:
                        self._fail_locked("Worker stdin write failed: " + type(error).__name__)
                    self.cond.notify_all()
                return
            with self.cond:
                self.write_started = None
                self.last_write = self.clock()
        try:
            self.process.close_stdin()
        except OSError:
            pass

    def _send_locked(self, line):
        if len(self.writes) >= WRITE_QUEUE:
            self._fail_locked("Worker stdin queue exceeded its bound")
            return False
        self.writes.append(line)
        self.cond.notify_all()
        return True

    # ----- event correlation (reader thread, under cond) -----
    def _dispatch(self, event):
        kind = event["event"]
        if self.ready is None and self.startup_failure is None:
            if kind == "ready":
                try:
                    self.resident = P.check_ready(event, self.expectation)
                except P.ProtocolViolation as error:
                    self.startup_failure = "Ready commitments refused: " + str(error)
                    self.unusable = self.startup_failure
                    self._retire_locked("ready_refused", force=False)
                    return
                self.ready = {k: event.get(k) for k in ("device_identity", "resident_model", "job", "lease_ms", "gemv_b1_selection")}
                return
            if kind in ("fatal", "exit"):
                self.startup_failure = "Worker refused startup: " + str(event.get("stage") or event.get("reason"))
                self.unusable = self.startup_failure
                if kind == "exit":
                    self.exit_event = self._exit_summary(event)
                return
            raise P.ProtocolViolation("Event before ready")
        if self.ready is None:
            if kind == "exit":
                self.exit_event = self._exit_summary(event)
                return
            if kind in ("closing", "fatal"):
                return
            raise P.ProtocolViolation("Unexpected event after a refused startup")
        if self.exit_event is not None:
            raise P.ProtocolViolation("Event after exit")
        ident = event.get("id")
        request = self.requests.get(ident) if isinstance(ident, str) else None
        if kind == "admitted":
            P.require(request is not None and request.state == "submitted", "Admission for an unknown or already admitted request")
            P.require(event.get("output") == request.output and event.get("input_sha256") == request.prepared.manifest_sha256
                      and event.get("diagnostic_token_cap") is None and event.get("output_allowance") == P.OUTPUT_ALLOWANCE
                      and event.get("slot") in ("active", "waiting") and P.integer(event.get("request_index"), 1), "Admission commitments differ")
            request.state = "admitted"
        elif kind == "rejected":
            submitted = event.get("submitted_id")
            request = self.requests.get(submitted) if P.text(submitted) else None
            P.require(request is not None and request.state == "submitted" and P.text(event.get("reason")), "Rejection for an unknown request")
            request.state, request.rejection = "rejected", event["reason"]
            request.items.append(("rejected", event["reason"]))
            del self.requests[request.id]
        elif kind == "started":
            P.require(request is not None and request.state == "admitted", "Start for a request that is not admitted")
            P.require(not any(r.state == "started" for r in self.requests.values()), "Two requests started at once")
            request.state = "started"
        elif kind == "phase":
            P.require(request is not None and request.state == "started" and isinstance(event.get("name"), str), "Phase outside an active request")
        elif kind == "token":
            P.require(request is not None and request.state == "started", "Token outside an active request")
            token, index, stop = event.get("token_id"), event.get("index"), event.get("stop")
            P.require(P.exact(index, len(request.tokens)) and P.integer(token) and token < P.VOCABULARY and P.boolean(stop)
                      and stop == (token in P.STOP_TOKEN_IDS), "Token index, ID or stop flag is inconsistent")
            P.require(not (request.tokens and request.tokens[-1] in P.STOP_TOKEN_IDS) and len(request.tokens) < P.OUTPUT_ALLOWANCE,
                      "Token after a stop or beyond the normal allowance")
            request.tokens.append(token)
            request.items.append(("token", token))
        elif kind == "cancel_requested":
            P.require(request is not None and request.state in ("admitted", "started") and event.get("released") is False
                      and event.get("state") in ("waiting", "admitted", "active") and isinstance(event.get("origin"), str), "Inconsistent cancellation acknowledgement")
            if event["origin"] == "client":
                P.require(request.cancel_sent, "Client cancellation that this backend never sent")
        elif kind == "cancel_rejected":
            submitted = event.get("submitted_id")
            P.require(event.get("reason") == "already_terminal" and P.text(submitted) and self.recent.get(submitted) is True,
                      "Cancellation refused unexpectedly")
        elif kind == "terminal":
            P.require(request is not None and request.state in ("admitted", "started"), "Terminal for an unknown or finished request")
            survivable, status = P.check_terminal(event, request.tokens, self.resident)
            request.state, request.terminal = "terminal", event
            request.items.append(("terminal", event))
            self.counts[status] += 1
            del self.requests[request.id]
            self.recent[request.id] = request.cancel_sent
            while len(self.recent) > RECENT:
                self.recent.popitem(last=False)
            if not survivable:
                self.unusable = "Worker did not prove clean release of request " + request.id
                self._retire_locked("unclean_request_release", force=False)
        elif kind == "closing":
            P.require(isinstance(event.get("reason"), str) and event.get("accepting") is False, "Malformed closing event")
            self.worker_closing = event["reason"]
            if self.retire_reason is None:
                self.unusable = "Worker closed itself: " + event["reason"]
                self._retire_locked("worker_closed:" + event["reason"], force=False)
        elif kind == "exit":
            self.exit_event = self._exit_summary(event)
        else:
            raise P.ProtocolViolation("Unexpected worker event: " + kind)

    @staticmethod
    def _exit_summary(event):
        keys = ("reason", "clean", "exit_code", "model_released", "drained", "tracked_buffer_bytes_after_release", "device_destroyed",
                "admitted_total", "terminal_counts", "rejected_total", "protocol_errors", "request_replayed")
        return {k: event.get(k) for k in keys}

    def _fail_locked(self, reason):
        if self.failure:
            return
        self.failure = reason[:512]
        self.unusable = self.unusable or self.failure
        LOG.error("Native worker channel failure: %s", self.failure)
        self._retire_locked("channel_failure", force=True)

    def _retire_locked(self, reason, force):
        if self.retire_reason is None:
            self.retire_reason = reason
        self.retire_force = self.retire_force or force
        self.unusable = self.unusable or ("Retiring: " + reason)
        self.cond.notify_all()

    # ----- supervisor: lease, write stalls, unexpected exit and bounded retirement -----
    def _supervise(self):
        """The supervisor alone confirms closure, so it never dies: an error fails the worker and supervision goes on."""
        failing_since = None
        while True:
            try:
                if self._supervise_once():
                    return
                failing_since = None
                time.sleep(0.05 if not self.unconfirmed else 1.0)
            except Exception as error:
                now = self.clock()
                with self.cond:
                    if failing_since is None:
                        failing_since = now
                        self.thread_errors = (self.thread_errors + [f"supervisor: {type(error).__name__}: {str(error)[:200]}"])[-8:]
                        LOG.error("Native supervisor error: %s: %s", type(error).__name__, str(error)[:200])
                    self._fail_locked("Native supervisor failed: " + type(error).__name__)
                    if now - failing_since > self.config.terminate_seconds and not self.unconfirmed:
                        self.unconfirmed = True
                        LOG.error("Native worker retirement is unconfirmed after supervisor errors; capacity remains held")
                time.sleep(1.0)

    def _supervise_once(self):
        lease_interval = max(0.25, self.config.lease_ms / 4000)
        stall = max(1.0, self.config.lease_ms / 1000)
        terminate = False
        with self.cond:
            now = self.clock()
            if not self.threads[0].is_alive() and not self.reader_done:
                self._fail_locked("Event reader stopped unexpectedly")
            if self.write_started is not None and now - self.write_started > stall:
                self._fail_locked("Worker stdin write stalled beyond the lease")
            if self.ready is not None and self.retire_reason is None and not self.failure and now - self.last_write >= lease_interval:
                if self._send_locked(P.message("lease")):
                    self.last_write = now
            if self.retire_reason is not None:
                if self.retire_deadline is None:
                    self.retire_deadline = now + (0 if self.retire_force else self.config.shutdown_seconds)
                    if not self.retire_force and not self.failure and self.exit_code is None:
                        self.shutdown_sent = self._send_locked(P.message("shutdown"))
                        self.writes.append(None)  # EOF after shutdown also retires the worker.
                if self.exit_code is None and self.terminated_at is None and (self.retire_force or now >= self.retire_deadline):
                    terminate = True
                    self.terminated_at = now
            # Unconfirmed: terminated, or exited on its own, and still no confirmed closure after terminate_seconds.
            since = self.terminated_at if self.terminated_at is not None else self.exited_at
            if since is not None and now - since > self.config.terminate_seconds and not self.unconfirmed:
                self.unconfirmed = True
                LOG.error("Native worker retirement is unconfirmed after %.0f s; capacity remains held", self.config.terminate_seconds)
        if terminate:
            try:
                self.process.terminate()
            except Exception as error:  # Retained as evidence; polling continues.
                with self.cond:
                    self.terminate_error = type(error).__name__ + ": " + str(error)[:256]
        try:
            code = self.process.poll()
        except Exception as error:
            code = None
            with self.cond:
                self.terminate_error = "poll: " + str(error)[:256]
        with self.cond:
            if code is not None and self.exit_code is None:
                self.exit_code = code
                self.exited_at = self.clock()
                if self.retire_reason is None:
                    self._fail_locked(f"Worker process exited unexpectedly with code {code}")
                self.cond.notify_all()
        return code is not None and self._close_if_confirmed()

    def _retire_custody(self):
        if self.custody is None:
            return None
        try:
            return self.custody.retire()
        except Exception as error:
            return {"clean": False, "verification_error": "custody retirement failed: " + type(error).__name__}

    def _close_if_confirmed(self):
        deadline = self.clock() + 5
        with self.cond:  # Let the reader consume the final events before judging the exit.
            while not self.reader_done and self.clock() < deadline:
                self.cond.wait(0.05)
        if self.closure is None:
            try:
                closure = self.process.confirm_closed()
            except Exception as error:
                with self.cond:
                    self.terminate_error = "confirm: " + str(error)[:256]
                return False
            if closure is None:
                return False
            self.closure = closure
            self.custody_result = self._retire_custody()  # Verify, release the held files and remove the snapshot; no lock held.
        closure, custody = self.closure, self.custody_result
        with self.cond:
            exit_event = self.exit_event or {}
            handles_closed = closure.get("handles_closed") is True
            custody_clean = custody is None or custody.get("clean") is True
            clean = (not self.failure and self.terminated_at is None and self.exit_code == 0 and exit_event.get("clean") is True
                     and P.exact(exit_event.get("tracked_buffer_bytes_after_release"), 0) and handles_closed and custody_clean)
            # Only idle unload and service shutdown are ordinary; any other retirement leaves the backend failed.
            expected = clean and self.retire_reason in ("idle_unload", "service_shutdown") and self.unusable == "Retiring: " + self.retire_reason
            if not expected and not clean:
                problems = []
                if not handles_closed:
                    failed = {k: v for k, v in {**closure.get("descriptors", {}), **closure.get("handles", {})}.items() if v != "closed"}
                    problems.append("ownership cleanup failed: " + json.dumps(failed or {"handles_closed": closure.get("handles_closed")}))
                if not custody_clean:
                    problems.append("shader custody at retirement failed: " + json.dumps(
                        {k: custody.get(k) for k in ("verification_error", "held")}, default=str)[:300])
                self.unusable = "Worker ended without a clean drained exit (" + str(self.unusable) + ")" + "".join("; " + p for p in problems)
            self.evidence = {"pid": self.process.pid, "reason": self.retire_reason, "clean": clean, "expected": expected, "failure": self.failure,
                             "unusable": None if expected else str(self.unusable)[:1024], "worker_exit_code": self.exit_code, "exit_event": self.exit_event,
                             "terminated": self.terminated_at is not None, "terminate_error": self.terminate_error,
                             "closure": closure, "confirmed": True, "handles_closed": handles_closed, "shader_source": custody,
                             "events": self.events_seen, "terminal_counts": dict(self.counts), "thread_errors": list(self.thread_errors),
                             "stderr_bytes": self.stderr_total, "request_replayed": False}
            stderr = b"".join(self.stderr)[-4096:].decode("utf-8", "replace")
        LOG.log(logging.INFO if clean else logging.ERROR, "Native worker retired: %s stderr_tail=%r", json.dumps(self.evidence, default=str), stderr)
        try:
            self.on_closed(self, self.evidence)  # The backend holds the failure and receipt before anyone sees closure.
        except Exception as error:
            # The backend installed its failed state and kept owning this session before the exception left it.
            self.publication_error = "publication failed"
            try:
                self.publication_error = f"{type(error).__name__}: {str(error)[:200]}"
                LOG.error("Native retirement publication failed: %s; the backend retains this closed session", self.publication_error)
            except Exception:
                pass
        with self.cond:
            self.closed = True  # Set before any allocation so a later failure cannot strand callers or the supervisor.
            try:
                if self.publication_error:
                    self.evidence["publication_error"] = self.publication_error
                for request in list(self.requests.values()):
                    request.items.append(("closed", None))
            finally:
                self.cond.notify_all()
            self.finalized = True  # Last and allocation-free: only now may the backend release its finalization ownership.
        try:
            self.settled.set()  # Advisory only: waiters also poll the flags, so a failure here cannot strand them.
        except Exception:
            pass
        return True

    # ----- API used by the backend -----
    def wait_ready(self, seconds):
        deadline = self.clock() + seconds
        with self.cond:
            while self.ready is None and self.startup_failure is None and not self.failure and not self.closed:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    self._retire_locked("startup_timeout", force=True)
                    return "Worker was not ready within the startup bound"
                self.cond.wait(min(remaining, 0.25))
            if self.ready is not None and not self.failure:
                return None
            if self.startup_failure is None and self.exit_event is not None and self.ready is None:
                self.startup_failure = "Worker exited before ready"
            reason = self.startup_failure or self.failure or "Worker closed before ready"
            self._retire_locked("startup_failed", force=self.exit_event is None and self.failure is not None)
            return reason

    def _await(self, done, seconds):
        """Lock-free: never takes self.cond, so a thread holding it cannot extend `seconds`; `done` reads one-way flags."""
        deadline = None if seconds is None else self.clock() + seconds
        while not done():
            remaining = WAIT_POLL if deadline is None else min(deadline - self.clock(), WAIT_POLL)
            if remaining <= 0:
                return done()
            self.settled.wait(remaining)  # Its private lock is only held inside Event methods, never across caller code.
        return True

    def wait_closed(self, seconds=None):
        return self._await(lambda: self.closed, seconds)

    def wait_finalized(self, seconds):
        """Bounded: until wake-up work is finished, or a recorded failure or stopped supervisor shows it is not."""
        return self._await(lambda: self.finalized or self.failure is not None or not self.threads[-1].is_alive(), seconds)

    def usable(self):
        with self.cond:
            return self.ready is not None and not self.unusable and not self.closed and not self.reader_done

    def try_idle_retire(self):
        """Called under the backend lock: atomically retire an idle usable worker, or skip this contended tick."""
        if not self.cond.acquire(blocking=False):
            return False
        try:
            if self.ready is None or self.unusable or self.closed or self.reader_done:
                return False
            self._retire_locked("idle_unload", force=False)
            return True
        finally:
            self.cond.release()

    def retire(self, reason, force=False, seconds=None):
        """With `seconds`, return False rather than wait longer for a condition another thread holds."""
        if not self.cond.acquire(timeout=-1 if seconds is None else max(seconds, 0)):
            return False
        try:
            self._retire_locked(reason, force)
        finally:
            self.cond.release()
        return True

    def fail(self, reason):
        """An integrity failure found outside the channel: never reuse this worker, retire it now."""
        with self.cond:
            self._fail_locked(reason)

    def submit(self, prepared):
        request = _Request("r-" + secrets.token_hex(16), "o-" + secrets.token_hex(16), prepared)
        line = P.message("submit", id=request.id, model="chandra", input_manifest=prepared.manifest,
                         input_sha256=prepared.manifest_sha256, output=request.output)
        with self.cond:
            if self.ready is None or self.unusable or self.closed:
                raise NativeFailure("Native worker is not accepting requests")
            self.requests[request.id] = request
            if not self._send_locked(line):
                raise NativeFailure("Native worker stdin is saturated")
            deadline = self.clock() + self.config.admission_seconds
            while request.state == "submitted" and not self.failure and not self.closed:
                if self.clock() >= deadline:
                    self._fail_locked("Worker did not answer a submission within the admission bound")
                    break
                self.cond.wait(0.1)
        return request

    def next_item(self, request, cancel):
        """Block for the next ("token"|"terminal"|"rejected"|"closed", value); sends cancel at most once."""
        with self.cond:
            while not request.items:
                if self.closed:
                    return ("closed", None)
                now = self.clock()
                if cancel.is_set() and not request.cancel_sent and request.terminal is None and not self.failure and self.exit_code is None:
                    if self._send_locked(P.message("cancel", id=request.id)):
                        request.cancel_sent = True
                        request.cancel_deadline = now + self.config.cancel_grace_seconds
                if request.cancel_deadline is not None and now > request.cancel_deadline and request.terminal is None:
                    self._fail_locked("Cancellation was not confirmed within the grace bound")
                    request.cancel_deadline = None
                self.cond.wait(0.1)
            return request.items.popleft()

    def snapshot(self):
        """Health never waits for cond; a contended snapshot exposes flags and the retained receipt, not mutable summaries."""
        acquired = self.cond.acquire(blocking=False)
        try:
            state = ("closed" if self.closed else "retirement_unconfirmed" if self.unconfirmed else "retiring" if self.retire_reason
                     else "starting" if self.ready is None else "busy" if not acquired else "running" if self.requests else "ready")
            return {"state": state, "pid": self.process.pid, "ready": self.ready, "requests": len(self.requests) if acquired else None,
                    "terminal_counts": dict(self.counts) if acquired else None, "unusable": self.unusable, "retire_reason": self.retire_reason,
                    "failure": self.failure, "thread_errors": list(self.thread_errors) if acquired else None,
                    "shader_source": self.custody.identity() if acquired and self.custody is not None else None,
                    "published": self.published, "publication_error": self.publication_error,
                    "retirement": dict(self.evidence) if self.closed and self.evidence is not None else None,
                    "snapshot_contended": not acquired}
        finally:
            if acquired:
                self.cond.release()


class _Startup:
    def __init__(self):
        self.done = threading.Event()
        self.session = None
        self.error = None


class NativeBackend:
    """Backend protocol implementation; see chandra_service/protocol.py."""

    def __init__(self, config, *, transport=None, processor=None, pinned=None, expectation=None):
        self.config = config
        self._transport = transport
        self._processor = processor if processor is not None else PinnedProcessor(config.model_dir, getattr(config, "processor_dll_bootstrap", False))
        self._pinned = pinned
        self._expectation = expectation if expectation is not None else config.expectation()
        self._lock = threading.Lock()
        self._session = None
        self._startup = None
        self._finalizing = None   # A published session whose closure wake-up is unfinished: no admission until it resolves.
        self._failed = None
        self._closed = False
        self._active = 0
        self._retirements = collections.deque(maxlen=8)
        self._launch_failure = None
        self._cleanup_owner = None  # One strong owner, even after a failed launch or receipt publication detaches a session.
        self._cleanup_lock = threading.Lock()  # Serializes explicit operator recovery; never used by health/close.
        self._outputs = collections.deque()
        self._last_request = None
        self._counts = collections.Counter()

    # ----- Backend protocol -----
    def health(self):
        with self._lock:
            self._adopt_locked()
            session, failed, closed, starting, finishing = self._session, self._failed, self._closed, self._startup is not None, self._finalizing
            info = {"backend": "directcompute-native", "model": P.MODEL, "revision": P.REVISION, "context_limit": P.CONTEXT_LIMIT,
                    "normal_output_allowance": P.OUTPUT_ALLOWANCE, "stop_token_ids": list(P.STOP_TOKEN_IDS),
                    "processor_profile": "northstone-serving", "admitted_geometry": {
                        "max_patch_rows": self.config.max_patch_rows, "max_prompt_tokens": P.CONTEXT_LIMIT - P.OUTPUT_ALLOWANCE,
                        "max_image_pixels": self.config.max_image_pixels},
                    "gemv_b1_selection": self.config.gemv_b1_selection, "worker_cpu": self.config.worker_cpu,
                    "shader_source": {"tree_sha256": self.config.shader_tree_sha256, "canonical": shaders.CANONICAL, "custody": "host",
                                      "native_compile_evidence": None},
                    "requests_in_backend": self._active, "request_counts": dict(self._counts), "last_request": self._last_request,
                    "retirements": [{k: r.get(k) for k in ("reason", "clean", "expected", "worker_exit_code", "terminated", "confirmed",
                                                            "handles_closed", "closure", "shader_source", "publication_error")}
                                    for r in self._retirements],
                    "launch_failure": self._launch_failure,
                    "retained_cleanup": self._cleanup_owner.identity() if self._cleanup_owner is not None else None,
                    "finalization": finishing and {"pid": finishing.process.pid, "finalized": finishing.finalized, "failure": finishing.failure},
                    "qualified_full_graph": False, "qualified_OCR": False, "performance_claim": False}
        worker = session.snapshot() if session is not None else None
        if worker is not None and worker["state"] == "closed":
            state = "failed" if failed else "retiring"  # A closed but still owned session means its publication failed.
        elif worker is not None and worker["state"] in ("retiring", "retirement_unconfirmed", "starting"):
            state = worker["state"]  # Explicit until the evidence is published.
        elif failed:
            state = "failed"
        elif worker is not None:
            state = "closing" if closed else worker["state"]
        else:
            state = "retiring" if finishing else "closed" if closed else "starting" if starting else "cold"
        info.update(state=state, loaded=state in ("ready", "running"), failure=failed or (worker or {}).get("failure"), worker=worker and {
            "pid": worker["pid"], "requests": worker["requests"], "terminal_counts": worker["terminal_counts"],
            "failure": worker["failure"], "unusable": worker["unusable"], "retire_reason": worker["retire_reason"],
            "thread_errors": worker["thread_errors"], "shader_source": worker["shader_source"],
            "device_identity": (worker["ready"] or {}).get("device_identity"), "job": (worker["ready"] or {}).get("job"),
            "resident_model": (worker["ready"] or {}).get("resident_model"), "published": worker["published"],
            "publication_error": worker["publication_error"], "retirement": worker["retirement"],
            "snapshot_contended": worker["snapshot_contended"]})
        return info

    def generate(self, request, cancel):
        with self._lock:
            self._adopt_locked()
            if self._closed:
                raise NativeFailure("Native backend is closed")
            if self._failed:
                raise NativeFailure("Native backend failed earlier; review evidence and restart the service")
            self._active += 1
        prepared = session = handle = None
        try:
            if self._pinned is None:
                self._pinned = Pinned.load(self.config.caller_source, self.config.model_dir)
            prepared = prepare(request, self._processor, self._pinned, self.config.input_root, self.config.max_image_pixels, self.config.max_patch_rows)
            if cancel.is_set():
                return
            session = self._ensure_session()
            if cancel.is_set():
                return
            self._verify_source(session, "before_submit")
            handle = session.submit(prepared)
            handle.source_stages.append("before_submit")
            yield from self._stream(session, handle, prepared, cancel)
        except (RequestRefused, NativeFailure):
            raise
        except Exception as error:
            raise NativeFailure("Native backend failure: " + type(error).__name__) from error
        finally:
            self._finish(session, handle, prepared, cancel)

    def unload(self):
        """Idle retirement of the whole worker; cold only after confirmed process/Job closure."""
        with self._lock:
            self._adopt_locked()
            session = self._session
            if self._active or self._startup is not None:
                return
            if session is None:
                self._processor_unload()
                return
            if not session.try_idle_retire():
                return
        if not session.wait_closed(self.config.shutdown_seconds + self.config.terminate_seconds + 10):
            raise NativeFailure("Idle retirement is unconfirmed; worker evidence is retained in health and logs")
        with self._lock:
            self._adopt_locked()
            if self._session is session:
                raise NativeFailure("Idle retirement evidence is unpublished; the backend retains the closed session and is failed")
        self._processor_unload()

    def close(self):
        """Service shutdown: stop admission, retire the worker and wait for bounded confirmation.

        After the startup wait, retirement and finalization share one bound that no Session.cond holder can extend."""
        with self._lock:
            self._closed = True
            startup = self._startup
        if startup is not None:
            startup.done.wait(self.config.startup_seconds + self.config.terminate_seconds + 10)
        deadline = time.monotonic() + self.config.shutdown_seconds + self.config.terminate_seconds + 10
        with self._lock:
            self._adopt_locked()
            session, finishing = self._session, self._finalizing
        if session is not None:
            if not session.closed and not session.retire("service_shutdown", seconds=max(0.0, deadline - time.monotonic())):
                LOG.error("Native worker condition stayed held at service shutdown; the Job handle closes with this process")
                return False
            if not session.wait_closed(max(0.0, deadline - time.monotonic())):
                LOG.error("Native worker retirement unconfirmed at service shutdown; the Job handle closes with this process")
                return False
            with self._lock:
                self._adopt_locked()
                if self._session is session:
                    LOG.error("Native worker closed but its retirement evidence is unpublished; the backend retains it at shutdown")
                    return False
                finishing = self._finalizing
        if finishing is not None:
            finishing.wait_finalized(max(0.0, deadline - time.monotonic()))
            with self._lock:
                self._adopt_locked()
                if self._finalizing is finishing:
                    LOG.error("Native worker closure wake-up is unfinished at service shutdown; the backend retains its failure")
                    return False
        with self._lock:
            return self._cleanup_owner is None or not self._cleanup_owner.pending

    def cleanup_retained(self):
        """Root/operator-only cleanup after admission failed or closed; no HTTP route, request replay or state reset.

        This is an explicit finite retry of resources that retain recoverable raw handles or snapshot paths.
        Ambiguous failed CRT descriptor closes require operator disposition, not a blind retry of a reused number.
        """
        if not self._cleanup_lock.acquire(blocking=False):
            raise NativeFailure("Native cleanup is already in progress")
        try:
            with self._lock:
                if self._startup is not None or self._active or self._session is not None or self._finalizing is not None:
                    raise NativeFailure("Native startup, requests or finalization still own resources")
                owner = self._cleanup_owner
            if owner is None:
                return {"pending": False}
            result = owner.dispose(retry=True)
            with self._lock:
                if not owner.pending and self._cleanup_owner is owner:
                    self._cleanup_owner = None
            return result
        finally:
            self._cleanup_lock.release()

    # ----- internals -----
    def _processor_unload(self):
        unload = getattr(self._processor, "unload", None)
        if unload is not None:
            unload()

    def _launch(self):
        """Verify the source and build, give the worker a held verified snapshot, then launch it in an owned Job."""
        config = self.config
        transport = self._transport
        if transport is None:
            from .winjob import WindowsJobTransport
            transport = self._transport = WindowsJobTransport()  # Refuses on unsupported hosts; never a fallback.
        tree = shaders.verified_tree(config.shader_root, config.shader_tree_sha256)  # The configured source, re-read now.
        owner = _CleanupOwner(config)
        try:
            snapshot = shaders.Snapshot.create(config.input_root, tree, on_created=lambda snapshot: setattr(owner, "snapshot", snapshot))
            owner.held = transport.hold([config.executable, *snapshot.files()])
            config.verify_executable()  # After the hold, so the hashed image is the one the launch maps.
            snapshot.verify()           # After the hold, so the verified bytes are the ones the worker can compile.
            owner.process = transport.launch(config.argv(snapshot.shaders), config.environment(), snapshot.cwd, config.job_limits())
            owner.custody = shaders.SourceCustody(snapshot, owner.held)
            session = owner.session = Session(owner.process, config, self._expectation, self._on_closed, custody=owner.custody)
            session.start()
        except BaseException as error:
            control = getattr(error, "cleanup_owner", None)
            owner.unknown_child = bool(getattr(error, "child_may_remain", False) and control is None)
            if control is not None:
                if hasattr(control, "_handles"):  # A failed HeldFiles construction, before any child exists.
                    owner.held = control
                    # It already made its one automatic release attempt; retain the failed values until operator retry.
                    owner.control = control
                else:
                    owner.control = control
            owner.started_failed = owner.process is not None
            try:
                disposition = owner.dispose()
            except Exception as cleanup_error:
                disposition = {"pending": True, "error": f"{type(cleanup_error).__name__}: {str(cleanup_error)[:200]}"}
            with self._lock:
                self._cleanup_owner = owner if owner.pending else None
                self._launch_failure = {"error": f"{type(error).__name__}: {str(error)[:512]}", "cleanup": getattr(error, "evidence", None),
                                        "snapshot": owner.snapshot.name if owner.snapshot else None, "disposition": disposition}
            raise
        return session

    def _ensure_session(self):
        while True:
            with self._lock:
                self._adopt_locked()
                if self._closed:
                    raise NativeFailure("Native backend is closing")
                if self._failed:
                    raise NativeFailure("Native backend failed earlier; review evidence and restart the service")
                session, startup, finishing = self._session, self._startup, self._finalizing
                starter = False
                if session is None and startup is None and finishing is None:
                    startup = self._startup = _Startup()
                    starter = True
            if session is not None:
                if session.usable():
                    return session
                session.wait_closed()  # Never start a second worker before the first is confirmed closed and published.
                continue
            if finishing is not None:  # Nor before its published closure has finished waking callers, or failed.
                if not finishing.wait_finalized(self.config.shutdown_seconds + self.config.terminate_seconds + 10):
                    raise NativeFailure("Native worker retirement is still finalizing; no new worker was started")
                continue
            if not starter:
                startup.done.wait()
                if startup.error:
                    raise NativeFailure(startup.error)
                continue
            try:
                launched = self._launch()
            except Exception as error:
                with self._lock:
                    self._failed = "Native worker launch refused: " + str(error)[:512]
                    self._startup = None
                    startup.error = self._failed
                startup.done.set()
                raise NativeFailure(startup.error) from error
            with self._lock:
                self._session = launched
            reason = launched.wait_ready(self.config.startup_seconds)
            if reason is not None:
                launched.wait_closed(self.config.shutdown_seconds + self.config.terminate_seconds + 10)
                with self._lock:
                    self._failed = reason[:512]  # The specific startup reason outranks the generic closure summary.
                    self._startup = None
                    startup.error = self._failed
                startup.done.set()
                raise NativeFailure(startup.error)
            with self._lock:
                self._startup = None
                startup.session = launched
            startup.done.set()
            return launched

    def _on_closed(self, session, evidence):
        """Called by the session's supervisor with no session lock held, before the session reports closed."""
        with self._lock:
            self._publish_locked(session, evidence)

    def _publish_locked(self, session, evidence):
        """Atomic: either the receipt is stored and the session released, or the backend is failed and still owns the session."""
        if session.published:
            # A later supervisor failure (e.g. while waking callers after publication) re-enters here; keep it owned.
            if not evidence.get("expected"):
                self._failed = self._failed or PUBLICATION_FAILED
                try:
                    if evidence is not self._retirements[-1]:
                        self._retirements.append(evidence)
                    self._failed = "Worker failed after its retirement was published; " + str(evidence.get("unusable"))[:400]
                except Exception:
                    pass
            return
        try:
            if evidence.get("publication_error"):
                reason = "Worker retirement publication failed (" + str(evidence["publication_error"]) + "); " + str(evidence.get("unusable"))
            else:
                reason = None if evidence.get("expected") else (evidence.get("unusable") or "Worker ended uncleanly")
            reason = str(reason)[:512] if reason else None
            owner = None
            if evidence.get("handles_closed") is not True or (session.custody is not None and session.custody.pending):
                owner = _CleanupOwner(self.config)
                owner.session, owner.process, owner.custody = session, session.process, session.custody
                owner.snapshot = session.custody.snapshot if session.custody else None
                owner.result = {"retirement": evidence, "pending": True}
            self._retirements.append(evidence)
        except BaseException:
            self._failed = self._failed or PUBLICATION_FAILED  # A preallocated string: installing it cannot fail.
            raise
        self._finalizing = session  # Owned with the receipt until the session's wake-up work is accounted for.
        if owner is not None:
            self._cleanup_owner = owner
        session.published = True
        if reason and not self._failed:
            self._failed = reason
        if self._session is session:
            self._session = None

    def _adopt_locked(self):
        """Publication fallback: retry a closed, unpublished session; on failure keep owning it with the backend failed."""
        session = self._session
        if session is not None and session.closed and not session.published:
            try:
                self._publish_locked(session, session.evidence)
            except Exception:
                self._failed = self._failed or PUBLICATION_FAILED
        if self._session is session and session is not None and session.published:
            self._session = None
        finishing = self._finalizing
        if finishing is not None:
            # Lock-free reads of monotonic flags in this order (finalized, failure, supervisor liveness, finalized), so
            # neither health nor admission waits behind a session condition. A failure the session has recorded, or a
            # supervisor that stopped before finishing, is the backend's failure now; installing it cannot fail.
            finalized = finishing.finalized
            if finishing.failure is not None or not (finalized or finishing.threads[-1].is_alive() or finishing.finalized):
                self._failed = self._failed or PUBLICATION_FAILED
            if finalized:
                self._finalizing = None

    def _verify_source(self, session, stage, handle=None):
        """Re-verify the worker's shader snapshot; a mismatch retires the worker and refuses this request's result."""
        if handle is not None:
            handle.source_checked = True
        custody = session.custody
        try:
            if custody is None:
                raise shaders.ShaderSourceRefused("Native worker has no verified shader custody")
            custody.verify(stage)
        except (shaders.ShaderSourceRefused, OSError) as error:
            session.fail(f"Shader snapshot verification failed {stage}: {str(error)[:200]}")
            raise NativeFailure(f"Native shader snapshot changed {stage}; the worker is retired and no result is returned") from None
        if handle is not None:
            handle.source_stages.append(stage)

    def _stream(self, session, handle, prepared, cancel):
        decoder = IncrementalDecoder(self._processor.decode)
        while True:
            kind, value = session.next_item(handle, cancel)
            if kind == "token":
                delta = decoder.push(value)
                if delta:
                    yield delta
            elif kind == "rejected":
                raise NativeFailure("Native worker refused admission: " + value)
            elif kind == "closed":
                raise NativeFailure("Native worker ended before this request's terminal; it is not replayed")
            else:
                break
        terminal = value
        if not terminal["released"]:
            session.wait_closed()  # Capacity stays held until whole-process retirement is confirmed.
            raise NativeFailure("Native request ended without proving release; worker retired")
        self._verify_source(session, "after_terminal", handle)  # Before any success metadata or survivor reuse.
        status = terminal["status"]
        if status == "canceled":
            if handle.cancel_sent and cancel.is_set():
                return
            raise NativeFailure("Native worker canceled the request: " + str(terminal.get("cancel_origin")))
        if status != "completed":
            raise NativeFailure("Native request failed: " + str(terminal.get("failure_class")))
        self._authenticate(session, handle, prepared, terminal)
        rest = decoder.finish()
        if rest:
            yield rest
        completion, prompt = len(handle.tokens), len(prepared.prompt_ids)
        finish = "stop" if terminal["stop_reason"] == "stop_token" else "length"
        with self._lock:
            self._last_request = {"status": status, "stop_reason": terminal["stop_reason"], "stop_token_id": terminal["stop_token_id"],
                                  "prompt_tokens": prompt, "completion_tokens": completion, "patch_rows": prepared.patch_rows,
                                  "result_sha256": terminal["result_sha256"],
                                  "shader_source": {"tree_sha256": session.custody.snapshot.tree.tree_sha256, "snapshot": session.custody.snapshot.name,
                                                    "host_verified": list(handle.source_stages), "native_compile_evidence": None}}
        yield {"usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}, "finish_reason": finish}

    def _read_owned(self, directory, name, cap, expected=None):
        path = directory / name
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
            raise NativeFailure("Native receipt is not a bounded regular file")
        with open(path, "rb") as source:
            data = source.read(cap + 1)
        if len(data) > cap or (expected is not None and hashlib.sha256(data).hexdigest() != expected):
            raise NativeFailure("Native receipt failed authentication")
        return data

    def _authenticate(self, session, handle, prepared, terminal):
        """result.json must hash to the terminal's commitment and agree with every streamed fact."""
        directory = Path(self.config.output_root) / handle.output
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise NativeFailure("Native output is not a plain directory")
        try:
            result = P.strict_json(self._read_owned(directory, "result.json", RESULT_CAP, terminal["result_sha256"]))
            worker = result.get("worker") or {}
            checks = [
                result.get("schema") == "chandra.directcompute.result.v1", result.get("mode") == "full_page_generation",
                result.get("passed") is True, result.get("input_manifest_sha256") == prepared.manifest_sha256,
                result.get("model_revision") == P.REVISION, result.get("prompt_tokens") == len(prepared.prompt_ids),
                result.get("prompt_token_ids") == list(prepared.prompt_ids), result.get("generated_token_ids") == handle.tokens,
                result.get("generated_tokens_including_stop") == len(handle.tokens), result.get("stop_reason") == terminal["stop_reason"],
                result.get("stop_token_id") == terminal["stop_token_id"], result.get("stop_token_ids") == list(prepared.stop_ids),
                result.get("context_limit") == P.CONTEXT_LIMIT, result.get("normal_output_allowance") == P.OUTPUT_ALLOWANCE,
                result.get("diagnostic_token_cap") is None, result.get("request_replayed") is False,
                result.get("request_buffers_released") is True, result.get("qualified_OCR") is False,
                worker.get("request_id") == handle.id, worker.get("terminal_status") == "completed", worker.get("request_replayed") is False,
                (result.get("model_provenance") or {}).get("model_sha256") == self._expectation.model_sha256,
                result.get("device_identity") == session.ready["device_identity"]]
            if not all(checks):
                raise NativeFailure("Native result.json disagrees with the streamed request")
            journal = self._read_owned(directory, "generated-tokens.jsonl", JOURNAL_CAP).split(b"\n")
            if journal[-1] != b"" or len(journal) - 1 != len(handle.tokens):
                raise NativeFailure("Native token journal length differs")
            for index, (line, token) in enumerate(zip(journal, handle.tokens)):
                row = P.strict_json(line)
                if not P.exact(row.get("generated_index"), index) or not P.exact(row.get("token_id"), token) \
                        or row.get("stop") is not (token in P.STOP_TOKEN_IDS):
                    raise NativeFailure("Native token journal differs from token events")
        except (OSError, P.ProtocolViolation) as error:
            raise NativeFailure("Native receipt unreadable: " + type(error).__name__) from None

    def _finish(self, session, handle, prepared, cancel):
        """Hold until the request is terminal or the whole worker is confirmed closed; then release files."""
        try:
            if session is not None and handle is not None and handle.terminal is None and handle.state != "rejected":
                cancel.set()
                while handle.terminal is None:
                    kind, _ = session.next_item(handle, cancel)
                    if kind == "closed":
                        break
            if handle is not None and handle.terminal is not None:
                if not handle.terminal["released"]:
                    session.wait_closed()
                elif not handle.source_checked:
                    try:  # A canceled survivor keeps its compiled shaders; verify before it may serve again.
                        self._verify_source(session, "after_terminal", handle)
                    except NativeFailure as error:
                        LOG.error("%s", error)
        finally:
            with self._lock:
                self._active -= 1
                status = handle.terminal["status"] if handle is not None and handle.terminal is not None else "not_completed"
                self._counts[status] += 1
            if prepared is not None:
                try:
                    discard(self.config.input_root, prepared)
                except OSError:
                    LOG.error("Could not remove a task-owned native input package")
            if handle is not None:
                self._retain_output(handle.output)

    def _retain_output(self, name):
        with self._lock:
            self._outputs.append(name)
            stale = []
            while len(self._outputs) > self.config.retained_outputs:
                stale.append(self._outputs.popleft())
        for old in stale:
            path = Path(self.config.output_root) / old
            try:
                if old.startswith("o-") and os.path.lexists(path) and stat.S_ISDIR(os.lstat(path).st_mode):
                    shutil.rmtree(path)
            except OSError:
                LOG.error("Could not prune a task-owned native output directory")
