"""Admission diagnostics use offline counters/fake ownership, never Windows or model work."""

import asyncio
from contextlib import redirect_stderr
import io
import json
import logging
import threading
import unittest
from unittest.mock import Mock, patch

import httpx

from runtime.llama import memory as M
from runtime.llama.app import create_llama_app
from runtime.llama.worker import AdmissionRefused
from tests.llama.test_backend import Harness, request

GIB = 1 << 30


class AdmissionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(config_sha256="a" * 64)
        self.addCleanup(self.h.close)
        self.logger = logging.getLogger(M.ADMISSION_FAILURE_LOGGER)

    def refused(self, observed, expected_cause=None):
        reader = Mock(side_effect=observed) if isinstance(observed, BaseException) else Mock(return_value=observed)
        with patch.object(self.h.backend, "_memory", reader), self.assertLogs(self.logger, level="WARNING") as logs:
            with self.assertRaises(AdmissionRefused) as caught:
                self.h.backend._admit()
        reader.assert_called_once_with()
        self.assertEqual(len(logs.records), 1)
        payload = logs.records[0].getMessage()
        self.assertLessEqual(len(payload.encode("ascii")), M.ADMISSION_FAILURE_MAX_BYTES)
        event = json.loads(payload, parse_constant=lambda value: self.fail("nonfinite JSON value"))
        self.assertEqual(event["schema"], "llama-admission-failure-v1")
        self.assertEqual(event["floor_bytes"], M.FLOOR)
        if expected_cause is not None:
            self.assertIs(caught.exception.__cause__, expected_cause)
        self.assertEqual(str(caught.exception), "Windows headroom admission failed before model launch")
        return event, caught.exception

    def test_success_observes_once_and_does_not_log(self):
        reader = Mock(return_value={"available_physical": 10 * GIB + 1, "available_commit": 15 * GIB + 1})
        with patch.object(self.h.backend, "_memory", reader), patch.object(self.logger, "warning") as warning:
            self.h.backend._admit()
        reader.assert_called_once_with()
        warning.assert_not_called()
        self.assertEqual(self.h.workers, [])

    def test_both_forecast_reserves_remain_strict_at_each_boundary(self):
        for key, forecast in (("available_physical", 6 * GIB), ("available_commit", 11 * GIB)):
            for offset in (-1, 0, 1):
                with self.subTest(key=key, offset=offset):
                    snapshot = {"available_physical": 20 * GIB, "available_commit": 30 * GIB,
                                "reported_process_commit": GIB, "commit_metric": "system_commit_limit_minus_total"}
                    snapshot[key] = forecast + M.FLOOR + offset
                    if offset > 0:
                        reader = Mock(return_value=snapshot)
                        with patch.object(self.h.backend, "_memory", reader), patch.object(self.logger, "warning") as warning:
                            self.h.backend._admit()
                        reader.assert_called_once_with()
                        warning.assert_not_called()
                    else:
                        event, error = self.refused(snapshot)
                        self.assertIsInstance(error.__cause__, RuntimeError)
                        self.assertEqual(event["stage"], "check")
                        self.assertEqual(event["forecasts_scope"], "checked")
                        self.assertEqual(event["config_sha256"], "a" * 64)
                        self.assertEqual(event["checks"][key]["available"]["value"], snapshot[key])
                        self.assertEqual(event["checks"][key]["forecast"]["value"], forecast)
                        self.assertEqual(event["checks"][key]["strict_threshold"]["value"], forecast + M.FLOOR)
                        self.assertEqual(event["checks"][key]["diagnostic_comparison"], "reserve_not_satisfied")
                        self.assertEqual(event["commit_metric_status"], "expected")
                        self.assertEqual(event["cause"]["operation"], "reserve_or_counter_check")

    def test_actual_observer_api_failures_keep_operation_and_os_code(self):
        for failed_call, operation in (("status", "global_memory_status"), ("performance", "k32_performance")):
            with self.subTest(failed_call=failed_call):
                calls = []

                def status(pointer):
                    calls.append("status")
                    pointer._obj.available_physical = 20 * GIB
                    return int(failed_call != "status")

                def performance(pointer, size):
                    calls.append("performance")
                    return 0

                reader = Mock(wraps=M.observe_memory)
                with patch.object(M, "_memory_calls", return_value=(status, performance)), \
                        patch.object(M.ctypes, "get_last_error", return_value=5, create=True), \
                        patch.object(self.h.backend, "_memory", reader), self.assertLogs(self.logger, level="WARNING") as logs:
                    with self.assertRaises(AdmissionRefused) as caught:
                        self.h.backend._admit()
                reader.assert_called_once_with()
                self.assertEqual(calls, ["status"] if failed_call == "status" else ["status", "performance"])
                event = json.loads(logs.records[0].getMessage())
                self.assertEqual(len(logs.records), 1)
                self.assertIsInstance(caught.exception.__cause__, OSError)
                self.assertEqual(event["stage"], "observation")
                self.assertEqual(event["snapshot_status"], "not_observed")
                self.assertEqual(event["forecasts_scope"], "configured_not_checked")
                self.assertEqual(event["checks"]["available_commit"]["forecast"]["value"], 11 * GIB)
                self.assertEqual(event["cause"]["category"], "os_error")
                self.assertEqual(event["cause"]["operation"], operation)
                self.assertEqual(event["cause"]["errno"]["value"], 5)

    def test_actual_observer_invalid_system_counter_is_observation_failure(self):
        with patch.object(M, "_memory_calls", return_value=(lambda pointer: 1, lambda pointer, size: 1)):
            reader = Mock(wraps=M.observe_memory)
            with patch.object(self.h.backend, "_memory", reader), self.assertLogs(self.logger, level="WARNING") as logs:
                with self.assertRaises(AdmissionRefused):
                    self.h.backend._admit()
        reader.assert_called_once_with()
        event = json.loads(logs.records[0].getMessage())
        self.assertEqual(event["stage"], "observation")
        self.assertEqual(event["cause"]["operation"], "system_commit_validation")

    def test_invalid_snapshot_values_are_typed_finite_and_redacted(self):
        class Secret:
            def __repr__(self):
                raise AssertionError("arbitrary object must not be rendered")

        for value, kind, status in ((True, "bool", "not_integer"), (1.0, "float", "not_integer"),
                                    (float("nan"), "float", "nonfinite"), (float("inf"), "float", "nonfinite"),
                                    ("SECRET_PAYLOAD_PATH", "other", "not_integer"),
                                    (Secret(), "other", "not_integer"), (None, "none", "not_integer")):
            with self.subTest(kind=kind, status=status):
                event, _ = self.refused({"available_physical": value, "available_commit": 30 * GIB,
                                         "commit_metric": "SECRET_METRIC"})
                number = event["checks"]["available_physical"]["available"]
                self.assertEqual((number["kind"], number["status"]), (kind, status))
                self.assertEqual(event["checks"]["available_physical"]["diagnostic_comparison"], "invalid_counter")
                self.assertEqual(event["commit_metric_status"], "unrecognized")
                self.assertNotIn("SECRET", json.dumps(event))
        for snapshot in (None, [], "SECRET_SNAPSHOT", {}):
            event, _ = self.refused(snapshot)
            self.assertEqual(event["snapshot_status"], "dict" if type(snapshot) is dict else "unsupported")
            self.assertNotIn("SECRET", json.dumps(event))
        event, _ = self.refused({"available_physical": 1 << 20000})
        self.assertEqual(event["checks"]["available_physical"]["available"],
                         {"kind": "int", "status": "out_of_range", "value": None})

    def test_only_safe_cause_classification_and_digest_are_retained(self):
        secret = type("SECRET_EXCEPTION_CLASS", (Exception,), {})("SECRET_MESSAGE_CREDENTIAL")
        self.h.config.config_sha256 = "SECRET_CONFIG_PATH"
        event, _ = self.refused(secret, secret)
        self.assertIsNone(event["config_sha256"])
        self.assertEqual(event["cause"]["category"], "other_exception")
        self.assertNotIn("SECRET", json.dumps(event))
        error = OSError(5, "SECRET_OS_PATH")
        error.winerror = float("nan")
        event, _ = self.refused(error, error)
        self.assertEqual(event["cause"]["operation"], "unknown")
        self.assertEqual(event["cause"]["winerror"]["status"], "nonfinite")
        self.assertNotIn("SECRET", json.dumps(event))

    def test_generate_refuses_before_worker_construction_and_releases_reservation(self):
        reader = Mock(return_value={"available_physical": M.FLOOR, "available_commit": 30 * GIB})
        with patch.object(self.h.backend, "_memory", reader), self.assertLogs(self.logger, level="WARNING") as logs:
            with self.assertRaises(AdmissionRefused) as caught:
                list(self.h.backend.generate(request(), threading.Event()))
        reader.assert_called_once_with()
        self.assertEqual(len(logs.records), 1)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual((self.h.workers, self.h.clients, self.h.calls), ([], [], []))
        self.assertEqual(self.h.backend._free, {0})
        self.assertIsNone(self.h.backend._session)
        self.assertIsNone(self.h.backend.health()["error"])

    def test_diagnostic_logger_or_formatter_failure_cannot_replace_refusal(self):
        for target, method in ((self.logger, "warning"), (M.json, "dumps")):
            with self.subTest(method=method):
                original = RuntimeError("SECRET_ORIGINAL_CAUSE")
                reader = Mock(side_effect=original)
                with patch.object(self.h.backend, "_memory", reader), \
                        patch.object(target, method, side_effect=RuntimeError("SECRET_DIAGNOSTIC_FAILURE")):
                    with self.assertRaises(AdmissionRefused) as caught:
                        self.h.backend._admit()
                reader.assert_called_once_with()
                self.assertIs(caught.exception.__cause__, original)

    def test_real_handler_formatter_failure_never_prints_chained_secret(self):
        class FailingFormatter(logging.Formatter):
            def format(self, record):
                raise RuntimeError("SECRET_FORMATTER_PATH")

        original = OSError(5, "SECRET_OBSERVER_CREDENTIAL")
        reader = Mock(side_effect=original)
        stream, fallback = io.StringIO(), io.StringIO()
        handler = M._AdmissionFailureHandler(stream)
        handler.setFormatter(FailingFormatter())
        with patch.object(self.h.backend, "_memory", reader), patch.object(self.logger, "handlers", [handler]), \
                patch.object(logging, "raiseExceptions", True), redirect_stderr(fallback):
            with self.assertRaises(AdmissionRefused) as caught:
                self.h.backend._admit()
        reader.assert_called_once_with()
        self.assertIs(caught.exception.__cause__, original)
        self.assertEqual((stream.getvalue(), fallback.getvalue()), ("", ""))
        self.assertFalse(self.logger.propagate)

    def test_oversize_diagnostic_is_not_emitted_and_original_refusal_survives(self):
        original = RuntimeError("original")
        reader = Mock(side_effect=original)
        with patch.object(self.h.backend, "_memory", reader), \
                patch.object(M.json, "dumps", return_value="x" * (M.ADMISSION_FAILURE_MAX_BYTES + 1)), \
                patch.object(self.logger, "warning") as warning:
            with self.assertRaises(AdmissionRefused) as caught:
                self.h.backend._admit()
        reader.assert_called_once_with()
        warning.assert_not_called()
        self.assertIs(caught.exception.__cause__, original)


class AdmissionHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_http_failure_remains_generic_502_without_worker_or_replay(self):
        h = Harness(config_sha256="a" * 64)
        self.addAsyncCleanup(asyncio.to_thread, h.close)
        h.memory["available_physical"] = M.FLOOR
        reader = Mock(side_effect=lambda: dict(h.memory))
        app = create_llama_app(h.backend, queue_limit=0, inference_seconds=5, idle_seconds=10)
        with patch.object(h.backend, "_memory", reader), self.assertLogs(M.ADMISSION_FAILURE_LOGGER, level="WARNING") as logs:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    response = await client.post("/v1/chat/completions", json=request())
        reader.assert_called_once_with()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"error": {
            "message": "Inference failed; inspect backend logs. Request was not replayed.", "type": "backend_failure"}})
        self.assertEqual(len(logs.records), 1)
        self.assertEqual((h.workers, h.clients, h.calls), ([], [], []))
