"""Test-only fakes for the native endpoint. Nothing here is reachable from production code.

FakeWorker speaks protocol version 1 over real OS pipes with the resident worker's admission, ordering,
cancellation, shutdown and receipt rules, and authenticates each submitted package's manifest and raw
tensors. It executes no model and compiles no shader: tokens are scripted. FakeProcessor is a deterministic
byte-level stand-in for the pinned processor/tokenizer, and SHADERS is a synthetic HLSL tree, so these are
lifecycle tests, never processor, shader or native acceptance.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import select
import string
import struct
import threading
import time

from runtime.native import protocol as P
from runtime.native import winjob as W
from runtime.native.config import NativeConfig
from runtime.native.prepare import Pinned, Processed, RawTensor
from runtime.native.shaders import read_tree

STOPS = [248044, 248046]
CALLER = (b"# Synthetic test stand-in for the pinned Chandra caller source; not the original file.\n" + b"#" * 4000)[:3177] + b"\n"
GENERATION = json.dumps({"eos_token_id": 248044}).encode()
TOKENIZER = json.dumps({"added_tokens_decoder": {"248046": {"content": "<|im_end|>"}, "248044": {"content": "<|endoftext|>"}}}).encode()
PINNED = Pinned.from_bytes(CALLER, GENERATION, TOKENIZER)
PCI, LUID = "03:00.0", "00000000:0000a1b2"
RESIDENT = 10_000_000_000
IDENTITY = {"dxgi_index": 0, "description": "Intel(R) Arc(TM) A770 Graphics", "vendor_id": 0x8086, "device_id": 0x56A0,
            "luid": LUID, "pci_bdf": PCI, "pci_bus": 3, "pci_device": 0, "pci_function": 0}
# Synthetic stand-ins with the reviewed tree's shape: runtime shaders, one shared include and a root-level file.
SHADERS = {"bf16.hlsl": b"// Synthetic test stand-in; not reviewed arithmetic.\n",
           "runtime/linear.hlsl": b"// Synthetic test stand-in.\n[numthreads(1, 1, 1)] void main() {}\n",
           "runtime/vision_common.hlsl": b"// Synthetic shared definitions.\n#define TEST_ROWS 4\n",
           "runtime/vision_rope.hlsl": b'// Synthetic shader using the common header.\n#include "vision_common.hlsl"\n[numthreads(1, 1, 1)] void main() {}\n'}
# Byte-level vocabulary: IDs below 256 are single bytes; these split multi-byte characters across tokens.
PIECES = {1000: "é".encode()[:1], 1001: "é".encode()[1:], 1002: "😀".encode()[:2], 1003: "😀".encode()[2:],
          1004: " 表".encode(), 1005: "\n".encode()}


def piece(token):
    if token in STOPS:
        return b""
    if token < 256:
        return bytes([token])
    return PIECES.get(token, b"?")


def decode(ids):
    return b"".join(piece(t) for t in ids).decode("utf-8", "replace")


def text_tokens(text):
    return list(text.encode("utf-8"))


class FakeProcessor:
    """Deterministic stand-in with the pinned processor's tensor inventory and a byte-level tokenizer."""

    runtime = {"fake_processor": True}

    def __init__(self, prompt_padding=0):
        self.calls = 0
        self.prompt_padding = prompt_padding
        self.unloaded = 0

    def process(self, image, text):
        self.calls += 1
        w, h = image.size
        gh, gw = 2 * max(1, math.ceil(h / 32)), 2 * max(1, math.ceil(w / 32))
        ids = [60, 105, 109, 62] + [P.IMAGE_TOKEN] * (gh * gw // 4) + text_tokens(text)[:64] + [10] * self.prompt_padding + [62]
        n, patches = len(ids), gh * gw
        def i64(values, shape):
            return RawTensor("I64", shape, struct.pack("<%dq" % len(values), *values))
        pixels = struct.pack("<%df" % (patches * 1536), *([0.25] * (patches * 1536)))
        tensors = {"input_ids": i64(ids, (1, n)), "attention_mask": i64([1] * n, (1, n)),
                   "mm_token_type_ids": i64([int(t == P.IMAGE_TOKEN) for t in ids], (1, n)),
                   "image_grid_thw": i64([1, gh, gw], (1, 3)), "pixel_values": RawTensor("F32", (patches, 1536), pixels)}
        return Processed("<|im_start|>user\n<image>" + text + "<|im_end|>\n<|im_start|>assistant\n", tensors)

    def decode(self, ids):
        return decode(ids)

    def unload(self):
        self.unloaded += 1


def png(width=64, height=64, fmt="PNG", frames=1):
    from PIL import Image
    buffer = io.BytesIO()
    images = [Image.new("RGB", (width, height), (10 * i, 120, 200)) for i in range(frames)]
    images[0].save(buffer, format=fmt, save_all=frames > 1, append_images=images[1:])
    return buffer.getvalue()


def payload(text="ocr_layout", stream=False, image=None, fmt="png", **extra):
    import base64
    url = f"data:image/{fmt};base64," + base64.b64encode(image or png()).decode()
    return {"model": "chandra", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text}]}], "temperature": 0, "top_p": 0.1, "max_tokens": 12384, "stream": stream, **extra}


def shader_tree(root, files=None):
    """Write the synthetic tree (once) and return its canonical digest."""
    for relative, data in (files or SHADERS).items():
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    return read_tree(root).tree_sha256


def config(root, **overrides):
    root = Path(root)
    for name in ("model", "shaders", "inputs", "outputs", "bin"):
        (root / name).mkdir(exist_ok=True)
    executable = root / "bin/chandra-worker.exe"
    if not executable.exists():
        executable.write_bytes(b"fake worker image for hash checks only")
    values = dict(config_sha256="0" * 64, executable=executable, executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  model_dir=root / "model", shader_root=root / "shaders", shader_tree_sha256=shader_tree(root / "shaders"),
                  input_root=root / "inputs", output_root=root / "outputs",
                  caller_source=root / "bin/caller.py", pci=PCI, luid=LUID, gemv_b1_selection="predecessor", lease_ms=1000,
                  process_memory_limit_bytes=12 << 30, job_memory_limit_bytes=13 << 30, worker_cpu=0, max_patch_rows=64, max_image_pixels=4_000_000,
                  startup_seconds=10, admission_seconds=5, cancel_grace_seconds=5, shutdown_seconds=5, terminate_seconds=2, retained_outputs=4)
    values.update(overrides)
    return NativeConfig(**values)


class Script:
    """Per-worker behavior. tokens: list of token lists, one per admitted request (the last repeats)."""

    def __init__(self, tokens=None, **options):
        self.tokens = tokens or [[72, 105, 248046]]
        self.options = options
        self.gates = {}

    def gate(self, name):
        return self.gates.setdefault(name, threading.Event())

    def get(self, key, default=None):
        return self.options.get(key, default)


class FakeWorker:
    def __init__(self, script, argv, env, limits):
        self.script, self.argv, self.env, self.limits = script, argv, env, limits
        options = dict(zip(argv[2::2], argv[3::2]))
        self.input_root, self.output_root = Path(options["--input-root"]), Path(options["--output-root"])
        self.pci, self.luid, self.lease_ms = options["--pci"], options["--luid"], int(options["--lease-ms"])
        self.stdin_r, self.stdin_w = os.pipe()
        self.stdout_r, self.stdout_w = os.pipe()
        self.stderr_r, self.stderr_w = os.pipe()
        self.lock = threading.Condition()
        self.seq = 0
        self.killed = False
        self.closing = None
        self.active = self.waiting = None
        self.admitted, self.received = set(), []
        self.exit_code = None
        self.lease_deadline = None
        self.packages = []
        self.counter = 0
        self.reader_thread = None
        self.thread = threading.Thread(target=self.main, daemon=True)
        self.thread.start()

    # ----- channel -----
    def inject(self, value, ident=""):
        """Emit one deliberately malformed event; it consumes a sequence number like any other.

        value is raw bytes (string.Template fields $seq, $id and $schema), a dict of fields placed after schema
        and seq so it may override them, or a callable taking the request ID and returning such a dict.
        """
        if callable(value):
            value = value(ident)
        if isinstance(value, bytes):
            raw = string.Template(value.decode()).substitute(seq=self.seq, id=ident, schema=P.EVENT_SCHEMA).encode() + b"\n"
        else:
            raw = (json.dumps({"schema": P.EVENT_SCHEMA, "seq": self.seq, **value}) + "\n").encode()
        self.seq += 1
        self.emit(None, raw=raw)

    def emit(self, event, raw=None):
        if self.killed:
            return
        if raw is None:
            event = {**event, "schema": P.EVENT_SCHEMA, "seq": self.seq}
            self.seq += 1
            raw = (json.dumps(event, ensure_ascii=False) + "\n").encode()
        try:
            os.write(self.stdout_w, raw)
        except OSError:
            pass

    def main(self):
        os.write(self.stderr_w, b"chandra-worker: fake startup\n")
        if self.script.get("startup_fatal"):
            self.emit({"event": "fatal", "stage": "startup", "error": "injected", "device_created": True, "model_loaded": False})
            self.emit({"event": "exit", "reason": "startup_failed", "clean": False, "exit_code": 1})
            return self.finish(1)
        if self.script.get("startup_hang"):
            while not self.killed:
                time.sleep(0.02)
            return
        ready = {"event": "ready", "mode": "execute", "accepting": True,
                 "protocol": {"request_schema": P.REQUEST_SCHEMA, "max_line_bytes": 4096, "max_event_bytes": 65536, "active_slots": 1, "waiting_slots": 1},
                 "model": {"alias": "chandra", "revision": P.REVISION, "context_limit": 16384, "normal_output_allowance": 12384, "caller_stop_token_ids": STOPS},
                 "resident_model": {"revision": P.REVISION, "model_sha256": P.MODEL_SHA256, "config_sha256": P.CONFIG_SHA256, "tie_byte_equality": True,
                                    "full_graph_requested": True, "resident_tracked_bytes": RESIDENT, "imports": 1},
                 "device_identity": dict(IDENTITY, pci_bdf=self.pci, luid=self.luid), "device_created": True, "model_loaded": True,
                 "gemv_b1_selection": "ordered" if self.env.get("CHANDRA_EXPERIMENTAL_GEMV_B1") == "ordered" else "predecessor",
                 "job": {"observed": True, "in_job": True, "limits_observed": True, "limit_flags": W.LIMIT_FLAGS, "kill_on_job_close": True,
                         "breakaway_allowed": False, "process_memory_limit_bytes": self.limits.process_memory,
                         "job_memory_limit_bytes": self.limits.job_memory, "active_process_limit": self.limits.active_processes},
                 "lease_ms": self.lease_ms, "qualified_full_graph": False, "qualified_OCR": False, "performance_claim": False}
        for path, value in (self.script.get("ready_override") or {}).items():
            target = ready
            keys = path.split(".")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
        with self.lock:
            self.renew()
            self.emit(ready)
        self.reader_thread = threading.Thread(target=self.read, daemon=True)
        self.reader_thread.start()
        self.serve()

    def renew(self):
        self.lease_deadline = time.monotonic() + self.lease_ms / 1000

    def read(self):
        buffer = b""
        while not self.killed:
            ready, _, _ = select.select([self.stdin_r], [], [], 0.02)
            if not ready:
                continue
            try:
                chunk = os.read(self.stdin_r, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                with self.lock:
                    self.close("input_eof")
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                with self.lock:
                    self.handle(json.loads(line))

    def handle(self, message):
        assert message["schema"] == P.REQUEST_SCHEMA
        self.received.append(message)
        if not self.closing:
            self.renew()
        kind = message["type"]
        if kind == "lease":
            return
        if kind == "shutdown":
            return self.close("shutdown_requested")
        if kind == "cancel":
            ident = message["id"]
            for slot in ("waiting", "active"):
                request = getattr(self, slot)
                if request and request["id"] == ident:
                    request["cancel"] = "client"
                    self.emit({"event": "cancel_requested", "id": ident, "state": "waiting" if slot == "waiting" else "active",
                               "origin": "client", "released": False})
                    if slot == "waiting":
                        self.waiting = None
                        self.terminal(request, "canceled", dispatched=False)
                    self.lock.notify_all()
                    return
            return self.emit({"event": "cancel_rejected", "submitted_id": ident, "reason": "already_terminal" if ident in self.admitted else "unknown_id"})
        if self.closing:
            return self.emit({"event": "rejected", "submitted_id": message["id"], "reason": "closing", "detail": "closing"})
        if self.active and self.waiting:
            return self.emit({"event": "rejected", "submitted_id": message["id"], "reason": "busy", "detail": "busy"})
        output = self.output_root / message["output"]
        os.mkdir(output)
        self.admitted.add(message["id"])
        request = {"id": message["id"], "message": message, "output": output, "cancel": None, "index": len(self.admitted)}
        slot = "waiting" if self.active else "active"
        setattr(self, slot, request)
        self.emit({"event": "admitted", "id": message["id"], "slot": slot, "request_index": request["index"], "output": message["output"],
                   "input_sha256": message["input_sha256"], "diagnostic_token_cap": None, "output_allowance": 12384})
        self.lock.notify_all()

    def close(self, reason):
        if self.closing:
            return
        self.closing = reason
        self.emit({"event": "closing", "reason": reason, "accepting": False})
        if self.waiting:
            request, self.waiting = self.waiting, None
            request["cancel"] = reason
            self.terminal(request, "canceled", dispatched=False)
        if self.active and not self.active["cancel"]:
            self.active["cancel"] = reason
            self.emit({"event": "cancel_requested", "id": self.active["id"], "state": "active", "origin": reason, "released": False})
        self.lock.notify_all()

    # ----- device thread -----
    def serve(self):
        while True:
            with self.lock:
                while not self.active and not self.closing and not self.killed:
                    if time.monotonic() > self.lease_deadline:
                        self.close("lease_expired")
                        break
                    self.lock.wait(0.02)
                if self.killed:
                    return
                if not self.active:
                    break
                request = self.active
            self.run(request)
            with self.lock:
                self.active, self.waiting = self.waiting, None
        code = 0 if self.closing in ("input_eof", "shutdown_requested") else 3
        if self.script.get("poisoned"):
            code = 2
        with self.lock:
            self.emit({"event": "exit", "reason": self.closing, "clean": code != 2, "exit_code": code, "model_released": True, "drained": True,
                       "tracked_buffer_bytes_after_release": 0 if code != 2 else 4096, "device_created": True, "device_destroyed": True,
                       "admitted_total": len(self.admitted), "terminal_counts": {}, "rejected_total": 0, "protocol_errors": 0, "request_replayed": False})
        self.finish(code)

    def finish(self, code):
        self.exit_code = code
        for fd in (self.stdout_w, self.stderr_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def authenticate(self, request):
        """The subset of the native authenticateInput rules a package writer can get wrong."""
        message = request["message"]
        path = self.input_root / message["input_manifest"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == message["input_sha256"], "manifest hash"
        manifest = json.loads(data)
        root = path.parent
        assert manifest["schema"] == "chandra.directcompute.input.v1" and manifest["revision"] == P.REVISION
        generation = manifest["generation"]
        assert generation["context_limit"] == 16384 and generation["max_output_tokens"] == 12384
        provenance = generation["stop_token_provenance"]
        for name, key in (("chandra-caller-hf.py", "caller_source_sha256"), ("generation_config.json", "generation_config_sha256"),
                          ("tokenizer_config.json", "tokenizer_config_sha256")):
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == provenance[key], name
        assert sorted(generation["stop_token_ids"]) == STOPS == sorted(provenance["caller_stop_token_ids"])
        values = {}
        for name, record in manifest["tensors"].items():
            raw = (root / record["file"]).read_bytes()
            assert len(raw) == record["bytes"] == math.prod(record["shape"]) * (8 if record["dtype"] == "I64" else 4), name
            assert hashlib.sha256(raw).hexdigest() == record["sha256"] and record["byte_order"] == "little", name
            if record["dtype"] == "I64":
                values[name] = [v[0] for v in struct.iter_unpack("<q", raw)]
        n = manifest["prompt_tokens"]
        ids = values["input_ids"]
        assert len(ids) == n and n + 12384 <= 16384
        assert manifest["image_token_positions"] == [i for i, t in enumerate(ids) if t == P.IMAGE_TOKEN]
        positions = values["position_ids"]
        assert len(positions) == 3 * n and max(positions) + 1 == manifest["positions"]["next_decode_position"]
        assert values["text_position_ids"] == list(range(n)) and values["attention_mask"] == [1] * n
        assert n + values["rope_deltas"][0] == max(positions) + 1
        self.packages.append({"manifest": manifest, "files": sorted(p.name for p in root.iterdir())})
        return manifest, ids

    def run(self, request):
        with self.lock:
            self.emit({"event": "started", "id": request["id"], "request_index": request["index"], "mode": "execute"})
        manifest, ids = self.authenticate(request)
        number = self.counter
        self.counter += 1
        tokens = self.script.tokens[min(number, len(self.script.tokens) - 1)]
        failure = (self.script.get("fail") or {}).get(number)
        generated, journal = [], []
        for index, token in enumerate(tokens):
            gate = self.script.gates.get(f"{number}:{index}")
            if gate is not None:
                Path(str(request["output"]) + ".parked").touch()
                while not gate.wait(0.02):
                    if self.killed or (request["cancel"] and not self.script.get("ignore_cancel")):
                        break
            if self.killed:
                return
            with self.lock:
                if request["cancel"] and not self.script.get("ignore_cancel"):
                    clean = not self.script.get("unclean_cancel")
                    return self.terminal(request, "canceled", generated=generated, released=clean, drained=clean)
            if failure == ("crash", index):
                self.kill(crashed=True)
                return
            if failure == ("poison", index):
                with self.lock:  # As worker.cpp: clear both slots before closing, so no cancellation follows the terminal.
                    self.terminal(request, "failed", generated=generated, failure_class="device_or_graph_failure", poisoned=True, released=False, drained=False)
                    self.script.options["poisoned"] = True
                    queued, self.waiting, self.active = self.waiting, None, None
                    self.close("worker_poisoned")
                    if queued:
                        self.terminal(queued, "failed", dispatched=False, failure_class="worker_poisoned")
                return
            row = {"generated_index": index, "token_id": token, "stop": token in STOPS, "best_logit": 1.0, "runner_up_logit": 0.5,
                   "top_margin": 0.5, "maximum_tie_count": 1}
            journal.append(json.dumps(row) + "\n")
            generated.append(token)
            event = {"event": "token", "id": request["id"], "index": index, "token_id": token, "stop": token in STOPS,
                     "best_logit": 1.0, "runner_up_logit": 0.5, "top_margin": 0.5, "maximum_tie_count": 1}
            injected = (self.script.get("inject") or {}).get((number, index))
            with self.lock:
                if injected is not None:
                    self.inject(injected, request["id"])
                elif failure == ("malformed", index):
                    self.emit(None, raw=b"{not json}\n")
                elif failure == ("duplicate_key", index):
                    self.emit(None, raw=b'{"schema":"%s","schema":"x"}\n' % P.EVENT_SCHEMA.encode())
                elif failure == ("seq_gap", index):
                    self.seq += 1
                    self.emit(event)
                elif failure == ("oversized", index):
                    self.emit(None, raw=b"x" * 70000 + b"\n")
                elif failure == ("index_skip", index):
                    self.emit(dict(event, index=index + 1))
                elif failure == ("wrong_stop", index):
                    self.emit(dict(event, stop=not event["stop"]))
                else:
                    self.emit(event)
            if self.script.get("token_delay"):
                time.sleep(self.script.get("token_delay"))
            if token in STOPS:
                break
        (request["output"] / "generated-tokens.jsonl").write_text("".join(journal), encoding="utf-8")
        stopped = bool(generated) and generated[-1] in STOPS
        result = {"schema": "chandra.directcompute.result.v1", "mode": "full_page_generation", "passed": True,
                  "input_manifest_sha256": request["message"]["input_sha256"], "input_metadata": manifest, "model_revision": P.REVISION,
                  "prompt_tokens": len(ids), "prompt_token_ids": ids, "context_limit": 16384, "normal_output_allowance": 12384,
                  "diagnostic_token_cap": None, "stop_token_ids": manifest["generation"]["stop_token_ids"], "qualified_full_graph": False,
                  "qualified_OCR": False, "performance_claim": False, "request_replayed": False, "device_identity": dict(IDENTITY, pci_bdf=self.pci, luid=self.luid),
                  "model_provenance": {"model_sha256": P.MODEL_SHA256}, "generated_token_ids": generated, "generated_tokens_including_stop": len(generated),
                  "stop_reason": "stop_token" if stopped else "normal_output_length", "stop_token_id": generated[-1] if stopped else None,
                  "request_buffers_released": True, "tracked_buffer_bytes_after": RESIDENT, "resident_model_bytes": RESIDENT,
                  "worker": {"request_id": request["id"], "terminal_status": "completed", "request_replayed": False}}
        if failure == ("result_mismatch", None):
            result["generated_token_ids"] = generated[:-1]
        data = (json.dumps(result) + "\n").encode()
        (request["output"] / "result.json").write_bytes(data)
        if failure == ("journal_mismatch", None):
            (request["output"] / "generated-tokens.jsonl").write_text("".join(journal[:-1]), encoding="utf-8")
        digest = hashlib.sha256(data).hexdigest() if failure != ("wrong_sha", None) else "f" * 64
        with self.lock:
            self.terminal(request, "completed", generated=generated, result_sha=digest,
                          stop_reason="stop_token" if stopped else "normal_output_length", stop_token=generated[-1] if stopped else None)

    def terminal(self, request, status, generated=(), dispatched=True, released=True, drained=True, poisoned=False,
                 failure_class=None, result_sha=None, stop_reason=None, stop_token=None):
        self.emit({"event": "terminal", "id": request["id"], "status": status, "failure_class": failure_class, "error": None,
                   "dispatched": dispatched, "cancel_origin": request["cancel"], "cancel_boundary": "decode_step" if status == "canceled" else None,
                   "generated_tokens": len(generated), "stop_reason": stop_reason, "stop_token_id": stop_token, "request_retired": drained or not dispatched,
                   "drained": drained if dispatched else False, "tracked_buffer_bytes_after": RESIDENT if dispatched and released else None,
                   "resident_model_bytes": RESIDENT, "released": released, "worker_poisoned": poisoned, "request_replayed": False,
                   "result_file": "result.json" if result_sha else None, "result_sha256": result_sha,
                   "qualified_full_graph": False, "qualified_OCR": False, "performance_claim": False})

    def kill(self, crashed=False):
        with self.lock:
            self.killed = True
            self.lock.notify_all()
            if self.exit_code is not None:
                return  # Its reader must still stop even when process exit preceded test teardown.
        self.finish(0xC0000005 if crashed else 0xC0DE0001)


class FakeProcess:
    def __init__(self, worker, transport):
        self.worker, self.transport, self.pid = worker, transport, 4000 + len(transport.launches)
        self.terminations = 0

    def write(self, data):
        if self.worker.exit_code is not None:
            raise BrokenPipeError()
        os.write(self.worker.stdin_w, data)

    def close_stdin(self):
        try:
            os.close(self.worker.stdin_w)
        except OSError:
            pass

    def read_stdout(self, size):
        return os.read(self.worker.stdout_r, size)

    def read_stderr(self, size):
        return os.read(self.worker.stderr_r, size)

    def poll(self):
        return self.worker.exit_code

    def terminate(self):
        self.terminations += 1
        if not self.transport.unkillable:
            self.worker.kill()

    def confirm_closed(self):
        if self.worker.exit_code is None or self.transport.unkillable:
            return None
        handles = {"process": "closed", "job": "close_failed: Win32 error 6" if self.transport.handle_close_fails else "closed"}
        return {"process_exit_code": self.worker.exit_code, "job_active_processes": 0, "terminated_by_parent": self.terminations > 0,
                "descriptors": {"stdin": "closed", "stdout": "closed", "stderr": "closed"}, "handles": handles,
                "handles_closed": not self.transport.handle_close_fails, "fake": True}


class FakeHeld:
    """Records the paths a launch asked to hold; it provides no immutability."""

    description = "test fake: records held paths, provides no immutability"

    def __init__(self, paths, transport):
        self.paths, self.transport = [Path(p) for p in paths], transport
        self.releases = 0

    def release(self):
        self.releases += 1
        failed = self.transport.release_fails
        return {"held": len(self.paths), "released": 0 if failed else len(self.paths), "released_all": not failed,
                "failures": ["fake: Win32 error 6"] if failed else [], "by": self.description}


class FakeTransport:
    """Explicitly injected by tests; production constructs WindowsJobTransport only."""

    def __init__(self, script=None):
        self.script = script or Script()
        self.launches = []
        self.holds = []
        self.unkillable = False
        self.handle_close_fails = False
        self.release_fails = False
        self.before_launch = None

    def hold(self, paths):
        held = FakeHeld(paths, self)
        self.holds.append(held)
        return held

    def launch(self, argv, env, cwd, limits):
        if self.before_launch is not None:
            self.before_launch(argv, cwd)
        worker = FakeWorker(self.script, argv, env, limits)
        process = FakeProcess(worker, self)
        self.launches.append({"argv": argv, "env": env, "cwd": cwd, "limits": limits, "process": process})
        return process

    @property
    def workers(self):
        return [launch["process"].worker for launch in self.launches]

    def submits(self):
        return [m for w in self.workers for m in w.received if m["type"] == "submit"]
