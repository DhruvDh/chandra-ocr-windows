"""Offline Windows-counter fakes; no DLL, process, device or memory reservation is used."""

import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from runtime.llama import memory as M

GIB = 1 << 30


class MemoryObservationTests(unittest.TestCase):
    def setUp(self):
        self.physical, self.process_commit = 20 * GIB, GIB
        self.page_size = 4096
        self.commit_limit, self.commit_total = 32 * GIB // 4096, 8 * GIB // 4096
        self.status_ok = self.performance_ok = True
        self.calls = []

    def status(self, pointer):
        self.calls.append("status")
        value = pointer._obj
        self.assertIsInstance(value, M._Status)
        self.assertEqual(value.length, ctypes.sizeof(M._Status))
        value.available_physical, value.available_commit = self.physical, self.process_commit
        return int(self.status_ok)

    def performance(self, pointer, size):
        self.calls.append("performance")
        value = pointer._obj
        self.assertIsInstance(value, M._Performance)
        self.assertEqual(size, ctypes.sizeof(M._Performance))
        value.cb = size
        value.commit_total, value.commit_limit, value.page_size = self.commit_total, self.commit_limit, self.page_size
        return int(self.performance_ok)

    def observe(self):
        with patch.object(M, "_memory_calls", return_value=(self.status, self.performance)), \
                patch.object(M.ctypes, "get_last_error", return_value=5, create=True):
            return M.observe_memory()

    def test_low_process_counter_does_not_hide_machine_commit_headroom(self):
        observed = self.observe()
        self.assertEqual(observed["available_physical"], 20 * GIB)
        self.assertEqual(observed["available_commit"], 24 * GIB)
        self.assertEqual(observed["reported_process_commit"], GIB)
        self.assertEqual(observed["commit_metric"], "system_commit_limit_minus_total")
        self.assertEqual(self.calls, ["status", "performance"])
        M.check_memory(observed, physical=6 * GIB, commit=11 * GIB)

    def test_high_process_counter_cannot_admit_low_machine_commit(self):
        self.process_commit = 30 * GIB
        self.commit_total = 28 * GIB // self.page_size
        observed = self.observe()
        self.assertEqual(observed["available_commit"], 4 * GIB)
        self.assertEqual(observed["reported_process_commit"], 30 * GIB)
        with self.assertRaises(RuntimeError):
            M.check_memory(observed, physical=6 * GIB, commit=11 * GIB)
        with self.assertRaises(RuntimeError):
            M.check_memory(observed)

    def test_physical_reserve_remains_strict_with_healthy_system_commit(self):
        self.physical = M.FLOOR
        with self.assertRaises(RuntimeError):
            M.check_memory(self.observe())
        self.physical += 1
        M.check_memory(self.observe())
        self.physical = M.FLOOR + 6 * GIB
        with self.assertRaises(RuntimeError):
            M.check_memory(self.observe(), physical=6 * GIB)

    def test_failed_status_api_raises_without_querying_performance(self):
        self.status_ok = False
        with self.assertRaises(OSError) as error:
            self.observe()
        self.assertEqual(error.exception.errno, 5)
        self.assertEqual(self.calls, ["status"])

    def test_failed_performance_api_raises_instead_of_using_process_counter(self):
        self.performance_ok = False
        with self.assertRaises(OSError) as error:
            self.observe()
        self.assertEqual(error.exception.errno, 5)
        self.assertEqual(self.calls, ["status", "performance"])

    def test_invalid_system_counter_relationships_are_refused(self):
        self.page_size = 0
        with self.assertRaises(RuntimeError):
            self.observe()
        self.page_size = 4096
        self.commit_total = self.commit_limit + 1
        with self.assertRaises(RuntimeError):
            self.observe()

    def test_api_resolution_is_lazy_and_cached(self):
        status, performance = Mock(), Mock()
        api = SimpleNamespace(GlobalMemoryStatusEx=status, K32GetPerformanceInfo=performance)
        M._memory_calls.cache_clear()
        try:
            with patch.object(M.sys, "platform", "win32"), \
                    patch.object(M.ctypes, "WinDLL", return_value=api, create=True) as library:
                M._Status()
                M._Performance()
                library.assert_not_called()
                self.assertEqual(M._memory_calls(), (status, performance))
                self.assertEqual(M._memory_calls(), (status, performance))
                library.assert_called_once()
                status.assert_not_called()
                performance.assert_not_called()
        finally:
            M._memory_calls.cache_clear()

    def test_non_windows_resolution_refuses_without_loading_a_dll(self):
        M._memory_calls.cache_clear()
        try:
            with patch.object(M.sys, "platform", "linux"), \
                    patch.object(M.ctypes, "WinDLL", create=True) as library:
                with self.assertRaises(RuntimeError):
                    M.observe_memory()
                library.assert_not_called()
        finally:
            M._memory_calls.cache_clear()
