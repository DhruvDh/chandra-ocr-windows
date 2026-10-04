"""Allowlisted, single-submission router. Upstream is drained after client cancellation."""
import asyncio
import ipaddress
import json
import logging
import math
import time
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from .service import error, OwnedStream
from .validation import read_payload

@dataclass
class Endpoint:
    name: str
    url: str
    capacity: int = 1
    queue_limit: int = 0
    auto_queue_limit: int = field(default=0, kw_only=True)
    expected_page_seconds: float = 1
    active: int = 0
    state: str = "unverified"
    retry_at: float = 0
    uncertain: bool = False
    remote_active: int = 0
    health_epoch: int = 0

    def __post_init__(self):
        parsed = urlsplit(self.url)
        try:
            port = parsed.port
        except ValueError:
            raise ValueError("Backend origin port must be between 1 and 65535") from None
        if parsed.netloc.endswith(":") or (port is not None and not 1 <= port <= 65535):
            raise ValueError("Backend origin port must be between 1 and 65535")
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Backend URL must be a private HTTP(S) origin without credentials or path")
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            if parsed.hostname != "localhost":
                raise ValueError("Use a loopback or private numeric backend address") from None
        else:
            if not (address.is_private or address.is_loopback) or address.is_unspecified:
                raise ValueError("Backend address must be loopback or private")
        if type(self.capacity) is not int or self.capacity < 1:
            raise ValueError("Backend capacity must be positive")
        if type(self.queue_limit) is not int or self.queue_limit < 0:
            raise ValueError("Backend queue_limit must be a nonnegative integer")
        if type(self.auto_queue_limit) is not int or not 0 <= self.auto_queue_limit <= self.queue_limit:
            raise ValueError("Backend auto_queue_limit must be an integer between zero and queue_limit")
        if (isinstance(self.expected_page_seconds, bool) or not isinstance(self.expected_page_seconds, (int, float))
                or not math.isfinite(self.expected_page_seconds) or self.expected_page_seconds <= 0):
            raise ValueError("expected_page_seconds must be a positive finite number")
        self.url = self.url.rstrip("/")


def create_router(config, client=None):
    """config: {'backends': {'northstone': {'url': ..., 'capacity':1}, ...}, cooldown_seconds:15}."""
    endpoints = [Endpoint(name, **value) for name, value in config["backends"].items()]
    if not endpoints or any(e.name not in {"northstone", "waystone"} for e in endpoints):
        raise ValueError("Configure northstone and/or waystone backends")
    cooldown = config.get("cooldown_seconds", 15)
    if (isinstance(cooldown, bool) or not isinstance(cooldown, (int, float))
            or not math.isfinite(cooldown) or cooldown < 0):
        raise ValueError("cooldown_seconds must be a finite nonnegative number")
    inference_seconds = config.get("inference_seconds", 1800)
    if (isinstance(inference_seconds, bool) or not isinstance(inference_seconds, (int, float))
            or not math.isfinite(inference_seconds) or inference_seconds <= 0):
        raise ValueError("inference_seconds must be a positive finite number")
    owned = client is None
    client = client or httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(1800, connect=5))
    tasks = set()
    selection_counter = 0
    lock = asyncio.Lock()

    def quarantine(endpoint, state="degraded", uncertain=False):
        # Invalidate health observations already in flight when ownership becomes
        # uncertain. A later response cannot undo this failure or its cooldown.
        endpoint.health_epoch += 1
        endpoint.state = state
        endpoint.uncertain = endpoint.uncertain or uncertain
        endpoint.retry_at = time.monotonic() + cooldown

    async def probe(endpoint):
        if time.monotonic() < endpoint.retry_at:
            return
        epoch = endpoint.health_epoch
        try:
            response = await client.get(endpoint.url + "/health", timeout=3)
            response.raise_for_status()
            info = response.json()
            if epoch != endpoint.health_epoch or time.monotonic() < endpoint.retry_at:
                return
            state = info.get("state", "unverified")
            if info.get("accepting") is False:
                endpoint.state = "degraded"
                return
            if state in {"failed", "degraded", "offline", "unverified"}:
                endpoint.state = state
                return
            # Existing NorthStone gateway exposes cold/asleep/awake; cold is routable.
            if state not in {"ready", "cold", "asleep", "awake", "loaded", "unloaded"}:
                endpoint.state = "unverified"
                return
            remote = info.get("pending_or_active")
            if type(remote) is not int or remote < 0:
                endpoint.state = "unverified"
                return
            endpoint.remote_active = remote
            if endpoint.uncertain and (remote or endpoint.active):
                endpoint.state = "degraded"
                return
            endpoint.uncertain = False
            endpoint.state = state
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            if epoch == endpoint.health_epoch and time.monotonic() >= endpoint.retry_at:
                quarantine(endpoint, "offline")

    @asynccontextmanager
    async def lifespan(app):
        yield
        # Producers hold capacity until remote stream actually finishes, including shutdown.
        pending = set()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=30)
        if pending:
            logging.getLogger("chandra.router").error("Shutdown drain exceeded 30s; %d requests remain in flight", len(pending))
        if owned and not pending:
            await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.endpoints = endpoints

    @app.get("/health")
    async def health():
        await asyncio.gather(*(probe(e) for e in endpoints))
        statuses = {e.name: {"state": e.state, "active": e.active, "remote_active": e.remote_active,
                               "capacity": e.capacity, "queue_limit": e.queue_limit, "auto_queue_limit": e.auto_queue_limit, "expected_page_seconds": e.expected_page_seconds, "uncertain": e.uncertain} for e in endpoints}
        good = [e for e in endpoints if e.state not in {"unverified", "offline", "failed", "degraded"}]
        return {"state": "ready" if len(good) == len(endpoints) else "degraded" if good else "offline", "backends": statuses}

    @app.get("/v1/models")
    @app.get("/backends/{backend}/v1/models")
    async def models(backend: str | None = None):
        if backend is not None and backend not in {e.name for e in endpoints}:
            return error("Unknown or unconfigured backend.", 404, "invalid_request_error")
        return {"object": "list", "data": [{"id": "chandra", "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    @app.post("/backends/{backend}/v1/chat/completions")
    async def completions(request: Request, backend: str | None = None):
        nonlocal selection_counter
        try:
            payload = await read_payload(request)
        except OverflowError as exc:
            return error(str(exc), 413, "invalid_request_error")
        except ValueError as exc:
            return error(str(exc), 400, "invalid_request_error")
        if backend is not None and backend not in {e.name for e in endpoints}:
            return error("Unknown or unconfigured backend.", 404, "invalid_request_error")
        header_selection = request.headers.get("x-chandra-backend")
        if backend is not None and header_selection not in {None, backend}:
            return error("Backend path and X-Chandra-Backend disagree.", 400, "invalid_request_error")
        selection = backend or header_selection or "auto"
        if selection not in {"auto", "northstone", "waystone"}:
            return error("X-Chandra-Backend must be auto, northstone, or waystone.", 400, "invalid_request_error")
        candidates = [e for e in endpoints if selection == "auto" or e.name == selection]
        await asyncio.gather(*(probe(e) for e in candidates))
        async with lock:
            ready = [e for e in candidates if e.state not in {"offline", "failed", "degraded", "unverified"}
                     and max(e.active, e.remote_active) < e.capacity + (e.queue_limit if selection != "auto" else e.auto_queue_limit)]
            if not ready:
                return error("Selected OCR backend is unavailable or at capacity; inspect /health and retry later.")
            # Static estimated finish cost: ((observed occupancy + 1) / capacity)
            # * configured page seconds. Calibrate with representative corpus medians;
            # this cannot predict page difficulty, cold-load time, or remaining work
            # in an active request or discrete execution waves. Queued estimates
            # are approximations, not promised completion times. It ranks eligible backends, with
            # round-robin ties; explicit selection still selects its named backend.
            def finish_cost(e):
                return ((max(e.active, e.remote_active) + 1) / e.capacity) * e.expected_page_seconds
            minimum = min(finish_cost(e) for e in ready)
            tied = [e for e in ready if finish_cost(e) == minimum]
            endpoint = tied[selection_counter % len(tied)]
            selection_counter += 1
            endpoint.active += 1
        output = asyncio.Queue(maxsize=16)
        abandoned = asyncio.Event()
        done = asyncio.Event()
        result = {"status": 502, "content_type": "application/json", "failure": False}

        async def produce():
            try:
                # One POST only. No redirects, retry, fallback, or replay after submission.
                async with asyncio.timeout(inference_seconds), client.stream("POST", endpoint.url + "/v1/chat/completions", json=payload) as response:
                    result.update(status=response.status_code, content_type=response.headers.get("content-type", "application/json"))
                    if response.status_code >= 500:
                        quarantine(endpoint)
                    async for chunk in response.aiter_bytes():
                        if abandoned.is_set():
                            continue
                        while not abandoned.is_set():
                            try:
                                await asyncio.wait_for(output.put(chunk), .1)
                                break
                            except asyncio.TimeoutError:
                                pass
            except (Exception, asyncio.CancelledError):
                result["failure"] = True
                quarantine(endpoint, uncertain=True)
            finally:
                endpoint.active -= 1
                done.set()

        task = asyncio.create_task(produce())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

        async def chunks():
            try:
                while not done.is_set() or not output.empty():
                    if await request.is_disconnected():
                        return
                    try:
                        yield output.get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(.02)
            finally:
                abandoned.set()
        if payload["stream"]:
            # Wait for response headers/first data so upstream errors keep their HTTP status.
            try:
                while not done.is_set() and output.empty():
                    if await request.is_disconnected():
                        abandoned.set()
                        return error("Client disconnected; upstream is draining.", 499)
                    await asyncio.sleep(.02)
            except asyncio.CancelledError:
                abandoned.set()
                raise
            if result["failure"] and output.empty():
                return error("Backend transport failed. Request was not replayed; inspect backend before resubmitting.", 502, "backend_failure")
            async def stream():
                async for chunk in chunks():
                    yield chunk
                if result["failure"]:
                    yield b'data: {"error":{"type":"backend_failure","message":"Backend stream failed. Request was not replayed."}}\n\n'
            return OwnedStream(stream(), cancel=abandoned, status_code=result["status"], media_type=result["content_type"], headers={"X-Chandra-Backend": endpoint.name})
        data = bytearray()
        async for chunk in chunks():
            data.extend(chunk)
            if len(data) > 32 * 1024 * 1024:
                abandoned.set()
                return error("Backend response exceeds 32 MiB limit; upstream is draining.", 502, "backend_failure")
        if result["failure"]:
            return error("Backend transport failed. Request was not replayed; inspect backend before resubmitting.", 502, "backend_failure")
        if not done.is_set():
            return error("Client disconnected; upstream is draining.", 499)
        return Response(bytes(data), status_code=result["status"], media_type=result["content_type"], headers={"X-Chandra-Backend": endpoint.name})
    return app
