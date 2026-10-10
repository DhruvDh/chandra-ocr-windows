"""Bounded parsing and validation of the resident worker's newline-JSON protocol (stdlib only).

Mirrors docs/directcompute-resident-worker.md protocol version 1. Every event is checked for size,
schema, strictly increasing gapless seq, known type and field types before the backend acts on it.
Every check validates a value's JSON type before comparing, hashing, sorting or looking it up, so a
well-formed object with a wrong-typed field is a ProtocolViolation rather than a TypeError.
"""
import json
import math
import re

REQUEST_SCHEMA = "chandra.directcompute.worker-request.v1"
EVENT_SCHEMA = "chandra.directcompute.worker-event.v1"
MAX_REQUEST_LINE_BYTES = 4096
MAX_EVENT_BYTES = 65536
MAX_INTEGER_DIGITS = 20  # The worker's widest values are uint64.
MODEL = "datalab-to/chandra-ocr-2"
REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
MODEL_SHA256 = "0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847"
CONFIG_SHA256 = "e26f17b70463de21fd68f1ef6d8f67f8e33e65d55a7a6895242130a18db60587"
CONTEXT_LIMIT = 16384
OUTPUT_ALLOWANCE = 12384
STOP_TOKEN_IDS = (248044, 248046)
VOCABULARY = 248320
IMAGE_TOKEN = 248056
ARC_A770 = {"vendor_id": 0x8086, "device_id": 0x56A0}
EVENTS = {"ready", "admitted", "rejected", "started", "phase", "token", "cancel_requested", "cancel_rejected", "terminal",
          "status", "protocol_error", "closing", "exit", "fatal", "inactive", "internal_error"}
SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")
SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,62}[A-Za-z0-9_-]")
HEX64 = re.compile(r"[0-9a-f]{64}")


class ProtocolViolation(RuntimeError):
    """The worker channel is untrustworthy; the whole worker must be retired, never reused."""


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolViolation("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ProtocolViolation("Nonfinite JSON number")


def _integer(text):
    if len(text.lstrip("-")) > MAX_INTEGER_DIGITS:
        raise ProtocolViolation("JSON integer exceeds 20 digits")
    return int(text)


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolViolation("Nonfinite JSON number")  # 1e999 overflows to infinity.
    return value


def strict_json(data):
    """Strict UTF-8 JSON object: no duplicate keys, NaN, Infinity, overflowing float or unbounded integer."""
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite,
                           parse_int=_integer, parse_float=_float)
    except ProtocolViolation:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError) as error:
        raise ProtocolViolation("Malformed JSON: " + type(error).__name__) from None
    if not isinstance(value, dict):
        raise ProtocolViolation("Expected a JSON object")
    return value


def message(kind, **fields):
    """One canonical request line; refuses anything the worker would treat as a protocol error."""
    if kind not in {"submit", "cancel", "lease", "shutdown"}:
        raise ValueError("Unsupported worker message type")
    line = (json.dumps({"schema": REQUEST_SCHEMA, "type": kind, **fields}, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    if len(line) > MAX_REQUEST_LINE_BYTES:
        raise ValueError("Worker request line exceeds 4096 bytes")
    return line


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def boolean(value):
    return type(value) is bool


def exact(value, expected):
    """Equality that also requires the JSON type, so true never passes for 1."""
    return type(value) is type(expected) and value == expected


def text(value):
    return type(value) is str


class EventStream:
    """Validates framing and ordering; content consistency is checked by the session."""

    def __init__(self):
        self.next_seq = 0

    def accept(self, line):
        if len(line) + 1 > MAX_EVENT_BYTES:
            raise ProtocolViolation("Event exceeds 65536 bytes")
        event = strict_json(line)
        if not exact(event.get("schema"), EVENT_SCHEMA):
            raise ProtocolViolation("Unexpected event schema")
        if not exact(event.get("seq"), self.next_seq):
            raise ProtocolViolation("Event sequence gap, repeat, reordering or non-integer seq")
        kind = event.get("event")
        if not text(kind) or kind not in EVENTS:
            raise ProtocolViolation("Unknown or non-string event type")
        self.next_seq += 1
        return event


def require(condition, reason):
    if not condition:
        raise ProtocolViolation(reason)


def check_ready(event, expected):
    """Verify every ready commitment against the explicit configuration; expected is a ReadyExpectation."""
    require(exact(event.get("mode"), "execute") and event.get("accepting") is True, "Worker is not an accepting execute worker")
    protocol = event.get("protocol")
    require(isinstance(protocol, dict) and exact(protocol.get("request_schema"), REQUEST_SCHEMA)
            and exact(protocol.get("max_line_bytes"), MAX_REQUEST_LINE_BYTES) and exact(protocol.get("max_event_bytes"), MAX_EVENT_BYTES)
            and exact(protocol.get("active_slots"), 1) and exact(protocol.get("waiting_slots"), 1), "Worker protocol limits differ")
    model = event.get("model")
    stops = model.get("caller_stop_token_ids") if isinstance(model, dict) else None
    require(isinstance(model, dict) and exact(model.get("alias"), "chandra") and exact(model.get("revision"), REVISION)
            and exact(model.get("context_limit"), CONTEXT_LIMIT) and exact(model.get("normal_output_allowance"), OUTPUT_ALLOWANCE)
            and isinstance(stops, list) and all(integer(stop) for stop in stops) and sorted(stops) == list(STOP_TOKEN_IDS),
            "Worker model/context/allowance/stop commitments differ")
    resident = event.get("resident_model")
    require(isinstance(resident, dict) and exact(resident.get("revision"), REVISION) and exact(resident.get("model_sha256"), expected.model_sha256)
            and exact(resident.get("config_sha256"), expected.config_sha256) and resident.get("tie_byte_equality") is True
            and resident.get("full_graph_requested") is True and exact(resident.get("imports"), 1)
            and integer(resident.get("resident_tracked_bytes"), 1), "Resident model commitments differ")
    require(event.get("device_created") is True and event.get("model_loaded") is True, "Worker reports no resident Device/model")
    identity = event.get("device_identity")
    require(isinstance(identity, dict) and exact(identity.get("pci_bdf"), expected.pci) and exact(identity.get("luid"), expected.luid)
            and all(exact(identity.get(key), value) for key, value in expected.device.items()), "Device PCI/LUID/adapter identity differs")
    require(exact(event.get("gemv_b1_selection"), expected.selection), "GEMV selector differs from configuration")
    job = event.get("job")
    flags = job.get("limit_flags") if isinstance(job, dict) else None
    require(isinstance(job, dict) and job.get("observed") is True and job.get("in_job") is True and job.get("limits_observed") is True
            and job.get("kill_on_job_close") is True and job.get("breakaway_allowed") is False
            and exact(job.get("process_memory_limit_bytes"), expected.process_memory_limit)
            and exact(job.get("job_memory_limit_bytes"), expected.job_memory_limit)
            and integer(flags) and flags & expected.job_limit_flags == expected.job_limit_flags
            and not flags & expected.job_breakaway_flags, "Observed Job limits differ from the launched Job")
    if expected.active_process_limit is not None:
        require(exact(job.get("active_process_limit"), expected.active_process_limit), "Observed Job process limit differs")
    require(exact(event.get("lease_ms"), expected.lease_ms), "Worker lease differs from configuration")
    require(event.get("qualified_full_graph") is False and event.get("qualified_OCR") is False and event.get("performance_claim") is False,
            "Worker made an unexpected qualification claim")
    return resident["resident_tracked_bytes"]


TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def check_terminal(event, tokens, resident_bytes):
    """Token/receipt consistency of one terminal; returns (survivable, outcome) or raises ProtocolViolation."""
    status = event.get("status")
    require(text(status) and status in TERMINAL_STATUSES, "Unexpected terminal status")
    flags = ("dispatched", "request_retired", "drained", "released", "worker_poisoned")
    require(all(boolean(event.get(key)) for key in flags) and event.get("request_replayed") is False, "Malformed terminal flags")
    require(event.get("qualified_full_graph") is False and event.get("qualified_OCR") is False and event.get("performance_claim") is False,
            "Terminal made an unexpected qualification claim")
    generated = event.get("generated_tokens")
    # A failed token-journal write can leave one generated ID without its event; every other status is exact.
    require(integer(generated) and (generated == len(tokens) or (status == "failed" and generated == len(tokens) + 1)),
            "Terminal token count differs from ordered token events")
    stop_reason, stop_token = event.get("stop_reason"), event.get("stop_token_id")
    stopped = bool(tokens) and tokens[-1] in STOP_TOKEN_IDS
    clean_release = event["released"] and not event["worker_poisoned"] and (not event["dispatched"] or (
        event["drained"] and event["request_retired"] and exact(event.get("tracked_buffer_bytes_after"), resident_bytes)
        and exact(event.get("resident_model_bytes"), resident_bytes)))
    if status == "completed":
        require(clean_release and event["dispatched"] and event.get("failure_class") is None, "Completed request did not retire cleanly")
        require(event.get("result_file") == "result.json" and isinstance(event.get("result_sha256"), str) and HEX64.fullmatch(event["result_sha256"]),
                "Completed request lacks an authenticated receipt")
        if stop_reason == "stop_token":
            require(stopped and exact(stop_token, tokens[-1]), "Stop receipt differs from the final token")
        else:
            require(stop_reason == "normal_output_length" and not stopped and stop_token is None and len(tokens) == OUTPUT_ALLOWANCE,
                    "Length receipt differs from the normal allowance")
    elif status == "failed":
        require(text(event.get("failure_class")), "Failed terminal lacks a failure class")
    return clean_release, status
