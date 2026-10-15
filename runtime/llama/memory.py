"""Host headroom, independent of the observing process's nested Job commit limit."""
import ctypes
from functools import lru_cache
import sys

FLOOR = 4 << 30


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
