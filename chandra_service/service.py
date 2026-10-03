"""Single-worker inference service; disconnect signals cancellation, never frees a running worker."""
import asyncio
import contextlib
import json
import logging
import traceback
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .validation import read_payload, validate_completion_metadata


def error(message, status=503, kind="service_unavailable"):
    return JSONResponse({"error": {"message": message[:512], "type": kind}}, status_code=status)

class OwnedStream(StreamingResponse):
    def __init__(self, *args, cancel, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel = cancel

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cancel.set()


@dataclass
class Job:
    payload: dict
    cancel: threading.Event = field(default_factory=threading.Event)
    output: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=16))
    done: threading.Event = field(default_factory=threading.Event)
    failure: bool = False
    metadata: dict = field(default_factory=dict)
    deadline: float = 0
    timed_out: bool = False
    invalid: bool = False
    output_chars: int = 0

class Worker:
    def __init__(self, backend, queue_limit, inference_seconds=1800):
        if type(queue_limit) is not int or queue_limit < 0:
            raise ValueError("queue_limit must be a nonnegative integer")
        if inference_seconds <= 0:
            raise ValueError("inference_seconds must be positive")
        self.inference_seconds = inference_seconds
        self.backend, self.limit = backend, queue_limit + 1
        self.lock = asyncio.Lock()
        self.jobs = set()
        self.active_jobs = []
        self.closing = False
        self.last_error = None
        self.last_activity = time.monotonic()

    def submit(self, payload):
        if self.closing or len(self.jobs) >= self.limit:
            return None
        job = Job(payload, deadline=time.monotonic() + self.inference_seconds)
        self.active_jobs.append(job)
        task = asyncio.create_task(self.run(job))
        self.jobs.add(task)
        task.add_done_callback(self.jobs.discard)
        return job

    async def run(self, job):
        try:
            async with self.lock:
                if not job.cancel.is_set():
                    await asyncio.to_thread(self.generate, job)
        finally:
            job.done.set()
            self.active_jobs.remove(job)
            self.last_activity = time.monotonic()

    def generate(self, job):
        try:
            self.last_error = None
            for item in self.backend.generate(job.payload, job.cancel):
                if isinstance(item, dict):
                    if job.metadata:
                        raise RuntimeError("Duplicate backend completion metadata")
                    validate_completion_metadata(item, job.payload, job.output_chars)
                    job.metadata = {"usage": dict(item["usage"]), "finish_reason": item["finish_reason"]}
                    continue
                if not isinstance(item, str):
                    raise TypeError("Backend delta must be string or final metadata dictionary")
                if job.metadata:
                    raise RuntimeError("Content after backend completion metadata")
                job.output_chars += len(item)
                while not job.cancel.is_set():
                    try:
                        job.output.put(item, timeout=.1)
                        break
                    except queue.Full:
                        pass
            if not job.cancel.is_set() and not job.metadata:
                raise RuntimeError("Backend ended without completion metadata")
        except ValueError:
            job.invalid = not job.output_chars and not job.metadata
            job.failure = True
        except Exception as exc:
            logging.getLogger("chandra.service").error("Backend %s at %s", type(exc).__name__, traceback.extract_tb(exc.__traceback__))
            # Do not expose internal paths, model inputs or backend exception text.
            self.last_error = "Inference failed; inspect backend logs before resubmitting."
            job.failure = True

    async def idle(self, seconds):
        while True:
            await asyncio.sleep(min(seconds, 5))
            async with self.lock:
                if not self.jobs and time.monotonic() - self.last_activity >= seconds and hasattr(self.backend, "unload"):
                    try:
                        await asyncio.to_thread(self.backend.unload)
                    except Exception:
                        self.last_error = "Idle unload failed; inspect backend logs."
                    self.last_activity = time.monotonic()

    async def close(self):
        self.closing = True
        # Cancellation is cooperative. Waiting here retains ownership until actual stop.
        for job in getattr(self, "active_jobs", []):
            job.cancel.set()
        if self.jobs:
            _, pending = await asyncio.wait(self.jobs, timeout=30)
            if pending:
                logging.getLogger("chandra.service").error("Shutdown cancellation exceeded 30s; %d workers still own generation", len(pending))

async def deltas(job, request):
    try:
        while not job.done.is_set() or not job.output.empty():
            if time.monotonic() >= job.deadline:
                job.timed_out = True
                return
            if await request.is_disconnected():
                return
            try:
                yield job.output.get_nowait()
            except queue.Empty:
                await asyncio.sleep(.02)
    finally:
        job.cancel.set()


def create_app(backend, queue_limit=2, inference_seconds=1800, idle_seconds=300):
    worker = Worker(backend, queue_limit, inference_seconds)
    if idle_seconds <= 0:
        raise ValueError("idle_seconds must be positive")
    @asynccontextmanager
    async def lifespan(app):
        idle = asyncio.create_task(worker.idle(idle_seconds))
        try:
            yield
        finally:
            idle.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await idle
            await worker.close()
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.worker = worker

    @app.get("/health")
    async def health():
        try:
            info = backend.health()
        except Exception:
            info = {"state": "degraded", "error": "Backend status unavailable; inspect backend logs."}
        return {**info, "pending_or_active": len(worker.jobs), "capacity": worker.limit,
                "accepting": not worker.closing and len(worker.jobs) < worker.limit,
                "last_error": worker.last_error}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "chandra", "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        try:
            payload = await read_payload(request)
        except OverflowError as exc:
            return error(str(exc), 413, "invalid_request_error")
        except ValueError as exc:
            return error(str(exc), 400, "invalid_request_error")
        job = worker.submit(payload)
        if job is None:
            return error("OCR queue is full or service is stopping; retry later.")
        ident = "chatcmpl-" + uuid.uuid4().hex
        base = {"id": ident, "created": int(time.time()), "model": "chandra"}
        if payload["stream"]:
            async def stream():
                def event(value):
                    return "data: " + json.dumps(value, separators=(",", ":")) + "\n\n"
                try:
                    yield event({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                    async for delta in deltas(job, request):
                        yield event({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]})
                    if job.timed_out:
                        yield event({"error": {"type": "inference_timeout", "message": "Inference deadline exceeded; cancellation requested. Capacity remains held until generation stops."}})
                    elif job.failure or (job.done.is_set() and not job.metadata):
                        yield event({"error": {"type": "backend_failure", "message": "Inference failed; inspect backend logs. Request was not replayed."}})
                    elif job.done.is_set():
                        yield event({**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": job.metadata["finish_reason"]}]})
                        if payload.get("stream_options", {}).get("include_usage") and "usage" in job.metadata:
                            yield event({**base, "object": "chat.completion.chunk", "choices": [], "usage": job.metadata["usage"]})
                        yield "data: [DONE]\n\n"
                finally:
                    job.cancel.set()
            return OwnedStream(stream(), cancel=job.cancel, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        try:
            parts = [delta async for delta in deltas(job, request)]
            if job.timed_out:
                return error("Inference deadline exceeded; cancellation requested. Capacity remains held until generation stops.", 504, "inference_timeout")
            if not job.done.is_set():
                return error("Client disconnected; cancellation requested.", 499, "client_disconnected")
            if job.invalid:
                return error("Backend rejected the image or prompt; check the configured pixel, input, and output token limits.", 400, "invalid_request_error")
            if job.failure or not job.metadata:
                return error("Inference failed; inspect backend logs. Request was not replayed.", 502, "backend_failure")
            result = {**base, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": job.metadata["finish_reason"]}]}
            result["usage"] = job.metadata["usage"]
            return result
        finally:
            job.cancel.set()
    return app
