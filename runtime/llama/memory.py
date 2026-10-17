"""Host headroom, independent of the observing process's nested Job commit limit."""
import ctypes
from functools import lru_cache
import json
import logging
import math
import sys

FLOOR = 4 << 30
ADMISSION_FAILURE_LOGGER = "chandra.llama.admission"
ADMISSION_FAILURE_MAX_BYTES = 4096
_MISSING = object()


class _AdmissionFailureHandler(logging.StreamHandler):
    def handleError(self, record):
        # StreamHandler otherwise prints a chained exception/raw message to stderr.
        pass


_ADMISSION_LOGGER = logging.getLogger(ADMISSION_FAILURE_LOGGER)
_ADMISSION_LOGGER.addHandler(_AdmissionFailureHandler())
_ADMISSION_LOGGER.setLevel(logging.WARNING)
_ADMISSION_LOGGER.propagate = False


class _Status(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ("total_physical", "available_physical", "total_commit",
                                            "available_commit", "total_virtual", "available_virtual", "extended_virtual")]


class _Performance(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in ("commit_total", "commit_limit", "commit_peak", "physical_total",
            "physical_available", "system_cache", "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size")
    ] + [(name, ctypes.c_uint32) for name in ("handle_count", "process_count", "thread_count")]


@lru_cache(maxsize=1)
def _memory_calls():
    if sys.platform != "win32":
        raise RuntimeError("The llama candidate requires Windows memory observation")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    status, performance = kernel.GlobalMemoryStatusEx, kernel.K32GetPerformanceInfo
    status.argtypes, status.restype = [ctypes.POINTER(_Status)], ctypes.c_int
    performance.argtypes, performance.restype = [ctypes.POINTER(_Performance), ctypes.c_uint32], ctypes.c_int
    return status, performance


def observe_memory():
    """System commit is (CommitLimit - CommitTotal) * PageSize; retain caller commit only as a diagnostic."""
    status = _Status()
    status.length = ctypes.sizeof(status)
    performance = _Performance()
    performance.cb = ctypes.sizeof(performance)
    status_call, performance_call = _memory_calls()
    if not status_call(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    if not performance_call(ctypes.byref(performance), ctypes.sizeof(performance)):
        raise OSError(ctypes.get_last_error(), "K32GetPerformanceInfo failed")
    if not performance.page_size or performance.commit_limit < performance.commit_total:
        raise RuntimeError("Invalid system commit observation")
    return {"available_physical": status.available_physical,
            "available_commit": (performance.commit_limit - performance.commit_total) * performance.page_size,
            "reported_process_commit": status.available_commit,
            "commit_metric": "system_commit_limit_minus_total"}


def check_memory(snapshot, physical=0, commit=0):
    for key, forecast in (("available_physical", physical), ("available_commit", commit)):
        if type(snapshot.get(key)) is not int or snapshot[key] - forecast <= FLOOR:
            raise RuntimeError("Windows headroom admission or live 4 GiB reserve was not satisfied")


def _diagnostic_number(value):
    """Bound values without coercing invalid guard inputs or rendering arbitrary objects."""
    kind, status, number = "other", "not_integer", None
    if value is _MISSING:
        kind, status = "missing", "unavailable"
    elif value is None:
        kind = "none"
    elif type(value) is bool:
        kind, number = "bool", value
    elif type(value) is int:
        kind = "int"
        if -(2**64 - 1) <= value <= 2**64 - 1:
            status, number = "finite", value
        else:
            status = "out_of_range"
    elif type(value) is float:
        kind = "float"
        if math.isfinite(value):
            number = value
        else:
            status = "nonfinite"
    return {"kind": kind, "status": status, "value": number}


def _diagnostic_cause(error):
    category = "other_exception"
    for error_type, label in ((OSError, "os_error"), (MemoryError, "memory_error"),
                              (TypeError, "type_error"), (ValueError, "value_error"),
                              (AttributeError, "attribute_error"), (RuntimeError, "runtime_error")):
        if isinstance(error, error_type):
            category = label
            break
    args = error.args
    message_index = 1 if isinstance(error, OSError) else 0
    message = args[message_index] if type(args) is tuple and len(args) > message_index else None
    operation = "unknown"
    if type(message) is str and len(message) <= 128:
        operation = {
            "GlobalMemoryStatusEx failed": "global_memory_status",
            "K32GetPerformanceInfo failed": "k32_performance",
            "Invalid system commit observation": "system_commit_validation",
            "The llama candidate requires Windows memory observation": "unsupported_platform",
            "Windows headroom admission or live 4 GiB reserve was not satisfied": "reserve_or_counter_check",
        }.get(message, "unknown")
    return {"category": category, "operation": operation,
            "errno": _diagnostic_number(getattr(error, "errno", _MISSING)),
            "winerror": _diagnostic_number(getattr(error, "winerror", _MISSING))}


def log_admission_failure(config, stage, snapshot, forecasts, error):
    """One private record from the failed admission's existing observation; never resample."""
    checked = forecasts is not None
    if forecasts is None:
        forecasts = (config.admission_physical_bytes, config.admission_commit_bytes)
    observed = stage == "check"
    counters = snapshot if observed and type(snapshot) is dict else {}
    digest = getattr(config, "config_sha256", None)
    if not (type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
        digest = None
    checks = {}
    for key, forecast in zip(("available_physical", "available_commit"), forecasts):
        available = dict.get(counters, key, _MISSING)
        threshold = forecast + FLOOR if type(forecast) is int and -(2**64 - 1) <= forecast <= 2**64 - 1 - FLOOR else _MISSING
        comparison = "unavailable"
        if observed and type(snapshot) is dict:
            if type(available) is not int:
                comparison = "invalid_counter"
            elif threshold is not _MISSING:
                comparison = "reserve_satisfied" if available > threshold else "reserve_not_satisfied"
        checks[key] = {"available": _diagnostic_number(available), "forecast": _diagnostic_number(forecast),
                       "strict_threshold": _diagnostic_number(threshold), "diagnostic_comparison": comparison}
    metric = dict.get(counters, "commit_metric", _MISSING)
    metric_status = ("expected" if type(metric) is str and metric == "system_commit_limit_minus_total" else
                     "missing" if metric is _MISSING else "unrecognized")
    event = {"schema": "llama-admission-failure-v1", "config_sha256": digest, "stage": stage,
             "snapshot_status": "dict" if observed and type(snapshot) is dict else "unsupported" if observed else "not_observed",
             "forecasts_scope": "checked" if checked else "configured_not_checked", "floor_bytes": FLOOR,
             "checks": checks, "commit_metric_status": metric_status,
             "reported_process_commit": _diagnostic_number(dict.get(counters, "reported_process_commit", _MISSING)),
             "cause": _diagnostic_cause(error)}
    payload = json.dumps(event, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(payload) > ADMISSION_FAILURE_MAX_BYTES:
        raise ValueError("Admission diagnostic exceeded its fixed bound")
    _ADMISSION_LOGGER.warning("%s", payload)
