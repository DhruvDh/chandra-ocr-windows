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
    if kind not in {"submit", "submit_cohort", "cancel", "lease", "shutdown"}:
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

    def __init__(self, experimental_b2=False):
        self.next_seq = 0
        self.events = EVENTS | {"cohort_started", "cohort_terminal"} if experimental_b2 else EVENTS

    def accept(self, line):
        if len(line) + 1 > MAX_EVENT_BYTES:
            raise ProtocolViolation("Event exceeds 65536 bytes")
        event = strict_json(line)
        if not exact(event.get("schema"), EVENT_SCHEMA):
            raise ProtocolViolation("Unexpected event schema")
        if not exact(event.get("seq"), self.next_seq):
            raise ProtocolViolation("Event sequence gap, repeat, reordering or non-integer seq")
        kind = event.get("event")
        if not text(kind) or kind not in self.events:
            raise ProtocolViolation("Unknown or non-string event type")
        self.next_seq += 1
        return event


def require(condition, reason):
    if not condition:
        raise ProtocolViolation(reason)


def check_drain_wait(value, selection, settled=False):
    """Authenticate the native Device route; host configuration never supplies native evidence."""
    require(type(selection) is str and selection in ("query_sleep", "fence_event"), "Unexpected configured drain wait selection")
    require(isinstance(value, dict) and exact(value.get("schema"), "chandra.directcompute.experimental_drain_wait.v1")
            and exact(value.get("environment_variable"), "CHANDRA_EXPERIMENTAL_DRAIN_WAIT")
            and exact(value.get("route"), selection)
            and exact(value.get("support"), "Device5_Context4_fence_created" if selection == "fence_event" else "not_queried")
            and exact(value.get("maximum_block_slice_milliseconds"), 10), "Native drain wait route/support differs")
    require(all(value.get(k) is False for k in ("poisoned", "retirement_required", "registered_event_outstanding")),
            "Native drain wait is poisoned or has an outstanding registration")
    counters = ("calls", "completed_calls", "failed_calls", "signals", "event_registrations", "completion_checks",
                "blocking_wait_calls", "requested_wait_milliseconds", "last_issued_value")
    require(all(integer(value.get(k)) and value[k] <= 0xffffffffffffffff for k in counters),
            "Native drain wait counters are missing or overflowed")
    if selection == "fence_event":
        require(value["calls"] > 0 and value["calls"] == value["completed_calls"] == value["signals"] == value["last_issued_value"]
                and value["failed_calls"] == 0 and value["event_registrations"] <= value["calls"]
                and value["completion_checks"] >= value["calls"] and exact(value.get("last_result"), "completed"),
                "Native fence waits did not complete cleanly")
    else:
        require(all(value[k] == 0 for k in counters) and value.get("last_result") is None,
                "Native query route reports unexpected fence work")
    if settled:
        copy = value.get("pending_copy_owner")
        require(exact(value.get("pending_storage_references"), 0) and isinstance(copy, dict)
                and copy.get("active") is False and copy.get("submitted") is False and exact(copy.get("staging_bytes"), 0),
                "Native drain wait still owns request storage")


def check_resident_drain_wait(result, selection):
    """Check every native page observation and its final resident request drain, without rewriting it."""
    rows = result.get("memory_observations")
    require(isinstance(rows, list) and bool(rows), "Native page lacks drain wait observations")
    for row in rows:
        require(isinstance(row, dict) and text(row.get("phase")) and isinstance(row.get("observed"), dict),
                "Native page has malformed memory observations")
        observed = row["observed"]
        check_drain_wait(observed.get("drain_wait"), selection)
        require(exact(observed.get("cpu_timing"), {"schema": "chandra.directcompute.cpu_timing.v1", "enabled": False,
                "environment_variable": "CHANDRA_NATIVE_CPU_TIMING"}) and observed["cpu_timing"].get("enabled") is False,
                "Resident native CPU timing is unexpectedly enabled")
    require(exact(rows[-1]["phase"], "request_retirement_and_drain"), "Native page lacks final resident request drain")
    check_drain_wait(rows[-1]["observed"]["drain_wait"], selection, settled=True)


def check_ready(event, expected):
    """Verify every ready commitment against the explicit configuration; expected is a ReadyExpectation."""
    require(exact(event.get("mode"), "execute") and event.get("accepting") is True, "Worker is not an accepting execute worker")
    cohort = getattr(expected, "experimental_b2_decode", False)
    ordered2 = getattr(expected, "experimental_ordered2_b2", False)
    grid = getattr(expected, "experimental_attention_values_grid", False)
    unified = getattr(expected, "experimental_unified_parallel32_b2", False)
    mix = getattr(expected, "experimental_b2_mix_input_projections", False)
    require(type(mix) is bool and (not mix or (cohort and unified and not ordered2)), "Invalid mix-input worker mode")
    require(type(unified) is bool and (not unified or (cohort and not ordered2 and expected.selection == "parallel32")), "Invalid unified parallel32 worker mode")
    require(type(grid) is bool and (not grid or expected.selection == ("parallel32" if unified else "ordered")), "Invalid attention-grid worker mode")
    check_attention_values_grid(event.get("experimental_attention_values"), grid)
    require(grid or "experimental_attention_values" not in event, "Unselected grid metadata present")
    require(type(ordered2) is bool and (not ordered2 or cohort), "Invalid configured ordered2 worker mode")
    if cohort:
        require(exact(expected.selection, "parallel32" if unified else "ordered"), "Experimental B2 selector differs")
        check_cohort_arithmetic(event.get("cohort_arithmetic"), ordered2, grid, unified, mix)
        require(exact(event.get("cohort_cancel_scope"), "whole_cohort_only"), "Cohort cancellation scope differs")
    else:
        require("cohort_arithmetic" not in event and "cohort_cancel_scope" not in event, "Unexpected experimental worker mode")
    protocol = event.get("protocol")
    require(isinstance(protocol, dict) and exact(protocol.get("request_schema"), REQUEST_SCHEMA)
            and exact(protocol.get("max_line_bytes"), MAX_REQUEST_LINE_BYTES) and exact(protocol.get("max_event_bytes"), MAX_EVENT_BYTES)
            and exact(protocol.get("active_slots"), 2 if cohort else 1) and exact(protocol.get("waiting_slots"), 0 if cohort else 1), "Worker protocol limits differ")
    if unified:
        require(exact(protocol.get("execution"), MIX_INPUT_EXECUTION if mix else UNIFIED_EXECUTION), "Unified execution declaration differs")
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
    check_drain_wait(event.get("drain_wait"), expected.drain_wait_selection, settled=True)
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


COHORT_ARITHMETIC = {"schema": "chandra.directcompute.worker-ordered-padded32-b2-selection.v1",
    "explicit_B2": True, "max_decode_slots": 2, "B1_selector": "ordered",
    "multirow_shader": "runtime/linear_gemm_padded32.hlsl",
    "multirow_shader_sha256": "93206bb0eae75929f86a81775360245705f6e3655639c054f912b4247f2a4d75",
    "fallback_shader": "runtime/linear.hlsl",
    "fallback_shader_sha256": "8fa9fb68f69b35a812f4a5821db11148e665611b0b5f560a5a0186ac13343d5f",
    "shader_bytes_authenticated_before_Device": True, "unsupported_multirow_shapes_use_original_fallback": True,
    "parallel32_excluded": True, "trained_numerical_OCR_PPS_acceptance": False}


ORDERED2_ARITHMETIC = {
    "explicit_cli_opt_in": True, "cli_flag": "--experimental-ordered2-b2",
    "shader": "runtime/linear_gemm_ordered2.hlsl",
    "shader_source_sha256": "df14d59cbea50c41f150e259d9453b92486637d0b589354b88527c9ab8448ea2",
    "rows": 2, "output_columns_per_group": 8, "tile_K": 512, "groupshared_bytes": 20520,
    "scope": "Only explicitly marked genuine two-active-row shared FFN/head calls; vision/prefill/mix/survivor unchanged",
    "arithmetic": "One precise FP32 chain from +0 in ascending k; separate product then add, then bias and optional BF16 RNE; no reassociation",
    "unsupported_requested_shape_refused_before_allocation": True, "shader_bytes_authenticated_before_Device": True}
ORDERED2_DISPATCH_SCOPE = "Successful Device::dispatch returns across this whole cohort; repeated in both page reports; not GPU completion/numerical proof"


ATTENTION_VALUES_GRID = {'schema': 'chandra.directcompute.attention-values-tile-grid-selection.v1', 'selector': 'tile-grid-3d', 'explicit_cli_opt_in': True, 'cli_flag': '--experimental-attention-values-grid', 'shader': 'runtime/text_attention_values_tile_grid.hlsl', 'shader_source_sha256': 'bd9a31e1e09c264ec34684dfb592e679383d5e55073fd0242568081773acbe5e', 'serial_shader': 'runtime/text_attention_values.hlsl', 'serial_shader_source_sha256': '32128314fc79d6b1fb9f37bb2097c004aa754ad1c905cda35366a3c51427f8b6', 'reduce_shader': 'runtime/text_attention_reduce.hlsl', 'reduce_shader_source_sha256': 'f0548298b831ab6d47adf4d0ebde9f6209d49134ca1aaddfbf60102db9c99e26', 'groups': '16,rows,ceil(totalKeys/128)', 'tile_axis': 'SV_GroupID.z', 'keys_per_group': 128, 'supported_rows': [1, 64], 'maximum_total_keys': 16384, 'partial_layout': '((tile*rows+row)*16+head)*256+channel', 'arithmetic': 'Original ascending k within each 128-key tile; original ascending tile reduction and BF16 RNE', 'operation_order_changed': False, 'precision_changed': False, 'other_operator_selectors_changed': False, 'scope': 'Explicit pre-model source-pinned selection; compiled arithmetic and trained equivalence remain unqualified', 'trained_numerics_accepted': False, 'complete_OCR_or_PPS_accepted': False}
COMBINED_ORDERED2_SCOPE = "Only explicitly marked genuine two-active-row shared FFN/head calls; original B1 and survivor linear arithmetic retained; attention-values separately selected by --experimental-attention-values-grid"

def check_attention_values_grid(value, selected=False):
    require(type(selected) is bool, "Attention-grid selector must be boolean")
    if not selected:
        require(value is None, "Unexpected attention-grid commitment")
        return
    require(isinstance(value, dict) and set(value) == set(ATTENTION_VALUES_GRID)
            and all(exact(value.get(k), v) for k, v in ATTENTION_VALUES_GRID.items() if k != "supported_rows")
            and type(value.get("supported_rows")) is list and len(value["supported_rows"]) == 2
            and all(exact(x, y) for x, y in zip(value["supported_rows"], ATTENTION_VALUES_GRID["supported_rows"])), "Exact attention-grid source/arithmetic commitment differs")

def cohort_result_schema(ordered2=False, grid=False):
    return ("chandra.directcompute.worker-ordered2-b2-attention-values-grid-page-result.v1" if ordered2 else "chandra.directcompute.worker-b2-attention-values-grid-page-result.v1") if grid else ("chandra.directcompute.worker-ordered2-b2-page-result.v1" if ordered2 else "chandra.directcompute.worker-b2-page-result.v1")

def cohort_owner_schema(ordered2=False, grid=False):
    return "chandra.directcompute.worker-ordered2-b2-attention-values-grid-page-owner.v1" if ordered2 and grid else ("chandra.directcompute.worker-ordered2-b2-page-owner.v1" if ordered2 else "chandra.directcompute.worker-b2-page-owner.v1")

def cohort_mode(ordered2=False, grid=False):
    return ("resident_ordered_B1_ordered2_B2_attention_values_tile_grid_page" if ordered2 else "resident_ordered_B1_padded32_B2_attention_values_tile_grid_page") if grid else ("resident_ordered_B1_ordered2_B2_page" if ordered2 else "resident_ordered_B1_padded32_B2_page")

def check_cohort_arithmetic(value, experimental_ordered2_b2=False, experimental_attention_values_grid=False):
    require(type(experimental_ordered2_b2) is bool, "Configured ordered2 selector must be boolean")
    require(type(experimental_attention_values_grid) is bool, "Attention-grid selector must be boolean")
    expected = COHORT_ARITHMETIC
    if experimental_ordered2_b2:
        expected = dict(COHORT_ARITHMETIC, schema="chandra.directcompute.worker-ordered2-padded32-b2-selection.v1",
                        ordered2=dict(ORDERED2_ARITHMETIC, scope=COMBINED_ORDERED2_SCOPE) if experimental_attention_values_grid else ORDERED2_ARITHMETIC)
    require(isinstance(value, dict) and set(value) == set(expected)
            and all(exact(value.get(k), v) for k, v in expected.items() if k != "ordered2"),
            "Cohort arithmetic/shader commitments differ")
    if experimental_ordered2_b2:
        observed = value.get("ordered2")
        require(isinstance(observed, dict) and set(observed) == set(ORDERED2_ARITHMETIC)
                and all(exact(observed.get(k), v) for k, v in expected["ordered2"].items()),
                "Native ordered2 commitment types or values differ")


def check_ordered2_dispatches(result, experimental_ordered2_b2, page_count):
    """Native count is an observation, never GPU completion or numerical acceptance."""
    require(type(experimental_ordered2_b2) is bool, "Configured ordered2 selector must be boolean")
    if not experimental_ordered2_b2:
        require("ordered2_dispatches" not in result, "Default worker unexpectedly reports ordered2 dispatches")
        return
    value = result.get("ordered2_dispatches")
    require(isinstance(value, dict) and set(value) == {"count", "scope"}
            and integer(value.get("count")) and value["count"] <= 0xffffffffffffffff
            and exact(value.get("scope"), ORDERED2_DISPATCH_SCOPE), "Native ordered2 dispatch observation differs")
    require(page_count in (1, 2) and type(page_count) is int and (page_count != 1 or value["count"] == 0),
            "Singleton must retain the original single-row path")

UNIFIED_PARALLEL32_ARITHMETIC = {
    "explicit_cli_opt_in": True, "cli_flag": "--experimental-unified-parallel32-b2",
    "captured_and_live_B1_selector": "parallel32", "shader": "runtime/linear_gemm_parallel32_b2.hlsl",
    "shader_source_sha256": "57d88c96ea7e913e374fb830d0b84788d23a34471f27cbc7f13f770a35c3f97e",
    "single_row_shader": "runtime/linear_gemv_parallel32.hlsl",
    "single_row_shader_sha256": "8d3a4293f5c96ab5e689c70e2168bb6a570b17e998ab26437118d18d1f6ca847",
    "rows": 2, "output_columns_per_group": 8, "lanes_per_output": 32, "tile_K": 512, "groupshared_bytes": 20512,
    "arithmetic": "Per-row +0; k%32 ascending across TILE512; separate precise mul/add; tree16,8,4,2,1; bias once; optional BF16 RNE once",
    "scope": "B1 mix/singleton/survivor use unchanged parallel32; only explicitly marked two-active-row FFN/head share weights; dense prefill unchanged",
    "unsupported_requested_shape_refused_before_allocation": True, "shader_bytes_authenticated_before_Device": True}
UNIFIED_DISPATCH_SCOPE = "Successful dispatch submissions across this cohort; repeated in both page reports; not completion, numerical or PPS proof"
UNIFIED_EXECUTION = "Explicit static B2 unified parallel32: unchanged serial vision/dense prefill; per-row fixed-tree B1 mix/singleton/survivor and shared-weight fixed-tree two-row FFN/head"
_original_cohort_result_schema = cohort_result_schema
_original_cohort_owner_schema = cohort_owner_schema
_original_cohort_mode = cohort_mode
_original_check_cohort_arithmetic = check_cohort_arithmetic
def _unified_selection(ordered2, grid, unified):
    require(all(type(x) is bool for x in (ordered2, grid, unified)), "Typed arithmetic selectors required")
    require(not unified or not ordered2, "Unified parallel32 and ordered2 are mutually exclusive")
def cohort_result_schema(ordered2=False, grid=False, unified=False):
    _unified_selection(ordered2,grid,unified)
    if unified:
        return "chandra.directcompute.worker-unified-parallel32-b2-attention-values-grid-page-result.v1" if grid else "chandra.directcompute.worker-unified-parallel32-b2-page-result.v1"
    return _original_cohort_result_schema(ordered2,grid)
def cohort_owner_schema(ordered2=False, grid=False, unified=False):
    _unified_selection(ordered2,grid,unified)
    if unified:
        return "chandra.directcompute.worker-unified-parallel32-b2-attention-values-grid-page-owner.v1" if grid else "chandra.directcompute.worker-unified-parallel32-b2-page-owner.v1"
    return _original_cohort_owner_schema(ordered2,grid)
def cohort_mode(ordered2=False, grid=False, unified=False):
    _unified_selection(ordered2,grid,unified)
    if unified:
        return "resident_unified_parallel32_B2_attention_values_tile_grid_page" if grid else "resident_unified_parallel32_B2_page"
    return _original_cohort_mode(ordered2,grid)
def check_cohort_arithmetic(value, experimental_ordered2_b2=False, experimental_attention_values_grid=False, experimental_unified_parallel32_b2=False):
    _unified_selection(experimental_ordered2_b2,experimental_attention_values_grid,experimental_unified_parallel32_b2)
    if not experimental_unified_parallel32_b2:
        return _original_check_cohort_arithmetic(value,experimental_ordered2_b2,experimental_attention_values_grid)
    expected=dict(COHORT_ARITHMETIC,schema="chandra.directcompute.worker-unified-parallel32-b2-selection.v1",
                  B1_selector="parallel32",parallel32_excluded=False,unified_parallel32=UNIFIED_PARALLEL32_ARITHMETIC)
    require(type(value) is dict and set(value)==set(expected) and
            all(exact(value.get(k),v) for k,v in expected.items() if k!="unified_parallel32"),
            "Unified parallel32 cohort commitments differ")
    observed=value.get("unified_parallel32")
    require(type(observed) is dict and set(observed)==set(UNIFIED_PARALLEL32_ARITHMETIC) and
            all(exact(observed.get(k),v) for k,v in UNIFIED_PARALLEL32_ARITHMETIC.items()),
            "Unified parallel32 shader/arithmetic commitment types or values differ")
def check_unified_parallel32_dispatches(result, selected, page_count):
    require(type(selected) is bool, "Unified parallel32 selector must be boolean")
    if not selected:
        require("unified_parallel32_B2_dispatches" not in result, "Unselected unified dispatch counter")
        return
    require("ordered2_dispatches" not in result, "Mutually exclusive ordered2 counter present")
    value=result.get("unified_parallel32_B2_dispatches")
    require(type(value) is dict and set(value)=={"count","scope"} and integer(value.get("count")) and
            value["count"]<=0xffffffffffffffff and exact(value.get("scope"),UNIFIED_DISPATCH_SCOPE),
            "Unified parallel32 dispatch counter differs")
    require(type(page_count) is int and page_count in (1,2) and (page_count!=1 or value["count"]==0),
            "Singleton must use the explicit B1 parallel32 path")

# The extra native selector is default-off and leaves all prior signatures' defaults unchanged.
MIX_INPUT_EXECUTION = "Explicit static B2 unified parallel32 with two-active-row mix-input and FFN/head sharing; serial vision/prefill, independent state, original B1 singleton/survivor and K4096 mix output"
MIX_INPUT_SCOPE = "Explicit two-active-row mix-input projections and FFN/head share weights; B1/singleton/survivor, K4096 mix output and dense prefill unchanged"
MIX_INPUT_PROJECTIONS = {
    "explicit_cli_opt_in": True, "cli_flag": "--experimental-b2-mix-input-projections",
    "requires_unified_parallel32": True, "max_active_rows": 2, "input_width": 2560,
    "input_norm": "Each slot keeps its original one-row input_layernorm and BF16 boundary before raw gather",
    "attention": ["q_proj", "k_proj", "v_proj"], "GDN": ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"],
    "row_mapping": "Raw FP32 gather/scatter in admitted slot order; separate positions, cache and recurrent state",
    "arithmetic": "Existing unified per-row k%32/TILE512/tree16,8,4,2,1; bias absent; BF16 RNE once",
    "singleton_and_survivor": "Unchanged original advance and B1 parallel32",
    "default_off": True, "trained_numerical_OCR_PPS_acceptance": False}
_mix_parent_result_schema = cohort_result_schema
_mix_parent_owner_schema = cohort_owner_schema
_mix_parent_mode = cohort_mode
_mix_parent_arithmetic = check_cohort_arithmetic
def _mix_selection(ordered2, grid, unified, mix):
    _unified_selection(ordered2,grid,unified)
    require(type(mix) is bool and (not mix or unified), "Mix-input requires typed explicit unified selection")
def _mix_json_exact(value, expected):
    if type(value) is not type(expected):return False
    if type(expected) is dict:return set(value)==set(expected) and all(_mix_json_exact(value[k],v) for k,v in expected.items())
    if type(expected) is list:return len(value)==len(expected) and all(_mix_json_exact(a,b) for a,b in zip(value,expected))
    return exact(value,expected)
def cohort_result_schema(ordered2=False, grid=False, unified=False, mix=False):
    _mix_selection(ordered2,grid,unified,mix)
    if mix:
        return "chandra.directcompute.worker-unified-parallel32-b2-mix-input-attention-values-grid-page-result.v1" if grid else "chandra.directcompute.worker-unified-parallel32-b2-mix-input-page-result.v1"
    return _mix_parent_result_schema(ordered2,grid,unified)
def cohort_owner_schema(ordered2=False, grid=False, unified=False, mix=False):
    _mix_selection(ordered2,grid,unified,mix)
    if mix:
        return "chandra.directcompute.worker-unified-parallel32-b2-mix-input-attention-values-grid-page-owner.v1" if grid else "chandra.directcompute.worker-unified-parallel32-b2-mix-input-page-owner.v1"
    return _mix_parent_owner_schema(ordered2,grid,unified)
def cohort_mode(ordered2=False, grid=False, unified=False, mix=False):
    _mix_selection(ordered2,grid,unified,mix)
    if mix:
        return "resident_unified_parallel32_B2_mix_input_attention_values_tile_grid_page" if grid else "resident_unified_parallel32_B2_mix_input_page"
    return _mix_parent_mode(ordered2,grid,unified)
def check_cohort_arithmetic(value, experimental_ordered2_b2=False, experimental_attention_values_grid=False, experimental_unified_parallel32_b2=False, experimental_b2_mix_input_projections=False):
    _mix_selection(experimental_ordered2_b2,experimental_attention_values_grid,experimental_unified_parallel32_b2,experimental_b2_mix_input_projections)
    if not experimental_b2_mix_input_projections:
        return _mix_parent_arithmetic(value,experimental_ordered2_b2,experimental_attention_values_grid,experimental_unified_parallel32_b2)
    expected=dict(COHORT_ARITHMETIC,schema="chandra.directcompute.worker-unified-parallel32-b2-mix-input-selection.v1",
        B1_selector="parallel32",parallel32_excluded=False,
        unified_parallel32=dict(UNIFIED_PARALLEL32_ARITHMETIC,scope=MIX_INPUT_SCOPE),
        mix_input_projections=MIX_INPUT_PROJECTIONS)
    require(_mix_json_exact(value,expected), "Exact mix-input own-row shader/arithmetic commitments differ")

