"""Bounded slots on one exclusively owned llama.cpp b11429 server.

Each successful call observes its own slot idle after DONE. A shared failure retires the
whole Job before affected calls return capacity. Submitted requests are never replayed.
"""
from dataclasses import dataclass, field
import json
import secrets
import socket
import threading
import time

import httpx

from chandra_service.validation import validate, validate_completion_metadata
from .config import CONTEXT, validate_kv_profile
from .memory import check_memory, observe_memory
from .worker import AdmissionRefused, OwnedWorker


class LlamaFailure(RuntimeError):
    pass


class RequestRefused(ValueError):
    pass


def _prepare(payload, maximum):
    from runtime.sycl import prepare_request
    return prepare_request(payload, max_image_pixels=maximum)


def _private_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _events(response):
    pending = b""
    total = 0
    for chunk in response.iter_raw():
        total += len(chunk)
        if total > 16 * 1024 * 1024:
            raise LlamaFailure("Engine response exceeded its bound")
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if len(line) > 262144:
                raise LlamaFailure("Engine event exceeded its bound")
            line = line.rstrip(b"\r")
            if not line or line.startswith(b":"):
                continue
            if not line.startswith(b"data: "):
                raise LlamaFailure("Unexpected engine event")
            value = line[6:]
            if value == b"[DONE]":
                yield None
                return
            try:
                yield json.loads(value)
            except (ValueError, UnicodeDecodeError) as error:
                raise LlamaFailure("Invalid engine event JSON") from error
        if len(pending) > 262144:
            raise LlamaFailure("Engine event exceeded its bound")
    raise LlamaFailure("Engine stream ended without DONE")


@dataclass(eq=False)
class _Session:
    worker: object
    alias: str
    startup_deadline: float
    client: object = None
    watcher: object = None
    ready: bool = False
    stopping: bool = False
    reason: str = ""
    refusal: bool = False
    start_done: threading.Event = field(default_factory=threading.Event)
    watch_stop: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)
    retire_lock: object = field(default_factory=threading.Lock)


@dataclass(eq=False)
class _Call:
    slot: int
    cancel: threading.Event
    deadline: float
    session: _Session | None = None
    submitted: bool = False
    completed: bool = False


class LlamaBackend:
    def __init__(self, config, *, worker_factory=OwnedWorker, client_factory=httpx.Client,
                 prepare=_prepare, memory=observe_memory, port_factory=_private_port):
        validate_kv_profile(config.engine, config.parallel_requests, getattr(config, "sycl_kv_cache_type", "bf16"),
                            config.admission_physical_bytes, config.admission_commit_bytes,
                            batch_size=getattr(config, "sycl_batch_size", 128),
                            host_cpu_count=getattr(config, "sycl_host_cpu_count", 1),
                            worker_cpu=getattr(config, "worker_cpu", None),
                            vision_leaf=getattr(config, "sycl_vision_leaf_sha256", None),
                            cache_dir=getattr(config, "sycl_cache_dir", None),
                            decode_graph=getattr(config, "sycl_decode_graph", False),
                            model_sha256=getattr(getattr(config, "model", None), "sha256", None),
                            runtime_files=getattr(config, "runtime_files", ()))
        self.config, self.parallel_requests = config, config.parallel_requests
        self._worker_factory, self._client_factory = worker_factory, client_factory
        self._prepare, self._memory, self._port_factory = prepare, memory, port_factory
        self._condition = threading.Condition()
        self._active, self._free = {}, set(range(self.parallel_requests))
        self._session, self._failure, self.last_retirement = None, None, None
        self._closing, self._maintenance = False, False

    def health(self):
        with self._condition:
            session = self._session
            phase = ("retiring" if session and session.stopping and not session.closed.is_set() else
                     "closed" if self._closing else "failed" if self._failure else
                     "asleep" if session is None or session.closed.is_set() else
                     "starting" if not session.ready else "busy" if self._active else "awake")
            state = ("failed" if self._failure else "offline" if self._closing else
                     "awake" if phase in ("awake", "busy") else "degraded" if phase == "retiring" else "asleep")
            return {"state": state, "phase": phase, "backend": f"llama-{self.config.engine}-candidate",
                    "loaded": phase in ("awake", "busy"), "capacity": self.parallel_requests,
                    "qualified": False, "error": self._failure,
                    "owned_worker_pending": session is not None and not session.worker.snapshot()["closed"]}

    def _admit(self):
        try:
            check_memory(self._memory(), self.config.admission_physical_bytes, self.config.admission_commit_bytes)
        except Exception as error:
            raise AdmissionRefused("Windows headroom admission failed before model launch") from error

    def _reserve(self, cancel):
        with self._condition:
            while True:
                if cancel.is_set() or self._closing:
                    raise LlamaFailure("Request canceled before reservation")
                if self._failure:
                    raise LlamaFailure(self._failure)
                if self._free and not self._maintenance and not (self._session and self._session.stopping):
                    slot = min(self._free)
                    self._free.remove(slot)
                    call = _Call(slot, cancel, time.monotonic() + self.config.startup_seconds, self._session)
                    self._active[slot] = call
                    return call
                self._condition.wait(0.1)

    def _check(self, call):
        with self._condition:
            session = call.session
            if session and session.stopping:
                error = AdmissionRefused if session.refusal else LlamaFailure
                raise error(session.reason)
            if call.cancel.is_set() or self._closing:
                raise LlamaFailure("Request canceled")
            if self._failure:
                raise LlamaFailure(self._failure)

    def _abort(self, session, reason, *, fatal=False, refusal=False):
        with self._condition:
            if not session.stopping:
                session.stopping, session.reason, session.refusal = True, reason, refusal
                if fatal:
                    self._failure = reason
                # Unblock service output backpressure too; each peer still raises its failure.
                for call in self._active.values():
                    if call.session is session and not call.completed:
                        call.cancel.set()
                self._condition.notify_all()
        session.worker.request_stop()

    def _watch(self, session):
        while not session.watch_stop.wait(0.1):
            memory_error, reason, fatal = False, None, True
            try:
                check_memory(self._memory())
            except Exception:
                memory_error = True
            with self._condition:
                stopping = session.stopping
                if not stopping:
                    calls = [call for call in self._active.values() if call.session is session and not call.completed]
                    canceled = self._closing or any(call.cancel.is_set() for call in calls)
                    expired = (not session.ready and time.monotonic() >= session.startup_deadline or
                               any(time.monotonic() >= call.deadline for call in calls))
                    if memory_error:
                        reason = "Windows memory observation or live 4 GiB reserve failed"
                    elif canceled:
                        reason, fatal = "Owned engine canceled; unfinished peers were aborted", False
                    elif expired:
                        reason = "Engine deadline expired"
                    else:
                        try:
                            if session.ready and not session.worker.alive():
                                reason = "Owned engine exited unexpectedly"
                            elif session.worker.snapshot()["fault"] is not None:
                                reason = "Owned engine control or channel failed"
                        except Exception:
                            reason = "Owned engine control or channel failed"
                    if reason:
                        self._abort(session, reason, fatal=fatal)
            if stopping or reason:
                self._retire(session)
                return

    def _json(self, session, method, path, payload=None, timeout=None):
        with session.client.stream(method, path, json=payload, timeout=timeout or self.config.request_seconds) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes(chunk_size=8192):
                data.extend(chunk)
                if len(data) > 1024 * 1024:
                    raise LlamaFailure("Engine control response exceeded its bound")
            try:
                return json.loads(data)
            except (ValueError, UnicodeDecodeError) as error:
                raise LlamaFailure("Invalid engine control JSON") from error

    def _idle_slot(self, session, call, slot=None):
        # A fresh query AFTER this call's DONE is required: /slots is not a FIFO POST barrier.
        while True:
            self._check(call)
            slots = self._json(session, "GET", "/slots", timeout=1)
            if (not isinstance(slots, list) or len(slots) != self.parallel_requests or
                    any(not isinstance(row, dict) or type(row.get("id")) is not int or
                        type(row.get("n_ctx")) is not int or row["n_ctx"] != CONTEXT or
                        type(row.get("is_processing")) is not bool for row in slots) or
                    {row["id"] for row in slots} != set(range(self.parallel_requests))):
                raise LlamaFailure("Owned engine slot IDs or full per-slot contexts differ")
            if not any(row["is_processing"] for row in slots if slot is None or row["id"] == slot):
                return
            time.sleep(0.05)

    def _start(self, call):
        with self._condition:
            self._check(call)
            starter = self._session is None
            if starter:
                self._admit()  # Refuse before even constructing an owner; retry needs a new request.
                session = _Session(self._worker_factory(self.config), "chandra-owned-" + secrets.token_hex(16),
                                   time.monotonic() + self.config.startup_seconds)
                self._session = session
                for current in self._active.values():
                    current.session = session
            else:
                session = self._session
                call.session = session
        if starter:
            try:
                session.watcher = threading.Thread(target=self._watch, args=(session,), name="llama-owner-watch", daemon=True)
                session.watcher.start()
                key, port = secrets.token_hex(32), self._port_factory()
                session.worker.start(port, key, session.alias, admit=self._admit)
                self._check(call)
                session.client = self._client_factory(base_url=f"http://127.0.0.1:{port}",
                    headers={"Authorization": "Bearer " + key, "Accept-Encoding": "identity"}, trust_env=False,
                    follow_redirects=False, timeout=httpx.Timeout(self.config.request_seconds, connect=1, pool=1))
                while True:
                    self._check(call)
                    if not session.worker.alive():
                        raise LlamaFailure("Owned engine exited during startup")
                    try:
                        models = self._json(session, "GET", "/v1/models", timeout=0.5)
                        if [item["id"] for item in models["data"]] != [session.alias]:
                            raise LlamaFailure("Private engine identity differs")
                        self._idle_slot(session, call)
                        with self._condition:
                            self._check(call)
                            session.ready = True
                        break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        time.sleep(0.1)
                    except httpx.HTTPStatusError as error:
                        if error.response.status_code != 503:
                            raise
                        time.sleep(0.1)
            except AdmissionRefused:
                self._abort(session, "Windows headroom admission refused before launch", refusal=True)
                raise
            except BaseException:
                self._abort(session, "Owned engine startup failed", fatal=not (call.cancel.is_set() or self._closing))
                raise
            finally:
                session.start_done.set()  # Retirement must not race later startup client/handle publication.
        else:
            while not session.start_done.wait(0.1):
                self._check(call)
        self._check(call)
        if not session.ready:
            raise LlamaFailure("Shared engine startup did not complete")
        with self._condition:
            call.deadline = time.monotonic() + self.config.request_seconds
        return session

    def _retire(self, session):
        session.worker.request_stop()
        session.start_done.wait()
        with session.retire_lock:
            if not session.closed.is_set():
                self.last_retirement = session.worker.retire()
                if self.last_retirement.get("fault") is not None:
                    with self._condition:
                        self._failure = self._failure or "Owned engine control or cleanup failed; restart the adapter before retrying"
                session.watch_stop.set()
                if session.client is not None:
                    session.client.close()
                session.closed.set()
        # Never join while holding retire_lock: the watcher can be another retirement caller.
        watcher = session.watcher
        if watcher is not None and watcher.ident is not None and watcher is not threading.current_thread():
            watcher.join()
        with self._condition:
            self._detach_idle(session)
            self._condition.notify_all()

    def _detach_idle(self, session):
        if self._session is session and session.closed.is_set() and not any(call.session is session for call in self._active.values()):
            self._session = None

    def _release(self, call):
        while True:
            with self._condition:
                session = call.session
                if not (session and session.stopping and not session.closed.is_set()):
                    del self._active[call.slot]
                    self._free.add(call.slot)
                    if session:
                        self._detach_idle(session)
                    self._condition.notify_all()
                    return
            self._retire(session)

    def _completion(self, session, call, payload, request, count):
        finish, usage, chars, done = None, None, 0, False
        call.submitted = True
        with session.client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity") != "identity":
                raise LlamaFailure("Unexpected compressed engine stream")
            for event in _events(response):
                self._check(call)
                if event is None:
                    done = True
                    break
                if not isinstance(event, dict) or event.get("model") != session.alias or event.get("object") != "chat.completion.chunk":
                    raise LlamaFailure("Invalid engine response identity or type")
                choices = event.get("choices")
                if not isinstance(choices, list) or len(choices) > 1:
                    raise LlamaFailure("Unexpected engine choices")
                if event.get("usage") is not None:
                    if usage is not None or choices or finish is None:
                        raise LlamaFailure("Unexpected engine usage ordering")
                    usage = event["usage"]
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    if choice.get("index") != 0 or finish is not None or not isinstance(delta, dict):
                        raise LlamaFailure("Unexpected engine delta ordering")
                    if delta.get("reasoning_content") or delta.get("tool_calls"):
                        raise LlamaFailure("Unexpected reasoning or tool output")
                    content = delta.get("content")
                    if content is not None:
                        if not isinstance(content, str):
                            raise LlamaFailure("Invalid engine content")
                        chars += len(content)
                        if content:
                            yield content
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
        if not done or not isinstance(usage, dict) or usage.get("prompt_tokens") != count:
            raise LlamaFailure("Engine stream or prompt accounting is incomplete")
        metadata = {"finish_reason": finish, "usage": {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
        validate_completion_metadata(metadata, request, chars)
        return metadata

    def generate(self, request, cancel):
        call = self._reserve(cancel)
        try:
            self._check(call)
            try:
                payload, _ = self._prepare(validate(request), self.config.max_image_pixels)
            except ValueError as error:
                raise RequestRefused(str(error)) from error
            self._check(call)
            session = self._start(call)
            # b11429 silently wraps invalid id_slot values, so only our bounded reservation is sent.
            payload.update(model=session.alias, id_slot=call.slot, stream=True, stream_options={"include_usage": True}, n=1,
                           cache_prompt=False, n_cache_reuse=0, reasoning_effort="none",
                           chat_template_kwargs={"enable_thinking": False})
            count = self._json(session, "POST", "/v1/chat/completions/input_tokens", payload).get("input_tokens")
            if type(count) is not int or count <= 0:
                raise LlamaFailure("Invalid engine token count")
            if count + payload["max_tokens"] >= CONTEXT:
                raise RequestRefused("Prompt and requested output do not fit the complete 16384-token context")
            self._check(call)
            metadata = yield from self._completion(session, call, payload, request, count)
            self._idle_slot(session, call, call.slot)
            with self._condition:
                self._check(call)
                call.completed = True
            yield metadata
        except RequestRefused:
            # Preparation and input_tokens create no completion task; peers keep their slots.
            raise
        except AdmissionRefused:
            if call.session:
                self._abort(call.session, "Windows headroom admission refused before launch", refusal=True)
            raise
        except BaseException as error:
            canceled = cancel.is_set() or self._closing or isinstance(error, GeneratorExit)
            if call.session and not call.completed:
                self._abort(call.session, "Owned engine canceled; unfinished peers were aborted" if canceled else
                            "Owned engine request failed; restart the adapter before retrying", fatal=not canceled)
            elif not canceled:
                with self._condition:
                    self._failure = "Owned engine preparation or startup failed"
            if isinstance(error, (LlamaFailure, GeneratorExit, KeyboardInterrupt, SystemExit)):
                raise
            raise LlamaFailure("Owned engine request failed") from error
        finally:
            self._release(call)  # A stopping shared session remains charged until confirmed retirement.

    def unload(self):
        with self._condition:
            while self._maintenance:
                self._condition.wait()
            self._maintenance = True
            while self._active:
                self._condition.wait()
            session = self._session
        try:
            if session:
                self._abort(session, "Idle worker unload")
                self._retire(session)
        finally:
            with self._condition:
                self._maintenance = False
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closing = True
            for call in self._active.values():
                call.cancel.set()
            session = self._session
            self._condition.notify_all()
        if session:
            self._abort(session, "Adapter shutdown")
            self._retire(session)
        with self._condition:
            while self._active:
                self._condition.wait()
