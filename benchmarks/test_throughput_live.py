"""CPU-only fake HTTP tests; no real endpoint, process or hardware access."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
import httpx
try:
    from . import throughput_live as live
    from . import throughput as core
except ImportError:
    import throughput_live as live
    import throughput as core


class Stream(httpx.AsyncByteStream):
    def __init__(self, raw, *, gate=None, close=None, fail=False):
        self.raw, self.gate, self.close, self.fail = raw, gate, close, fail
    async def __aiter__(self):
        if self.gate:
            await self.gate.wait()
        await asyncio.sleep(0)
        yield self.raw
        if self.fail:
            raise httpx.ReadError('ambiguous partial execution')
    async def aclose(self):
        if self.close:
            self.close()
            self.close = None


class LiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        expected = {k: [] for k in ('table', 'numbers', 'text', 'end_markers', 'ordered_text', 'required_labels')}
        fixtures = []
        for name in ('tiny', 'large'):
            image = name.encode()
            (self.root / (name + '.png')).write_bytes(image)
            fixtures.append({'id': name, 'image': name + '.png', 'image_sha256': core.digest(image), 'expected': expected})
        (self.root / 'manifest.json').write_text(json.dumps({'fixtures': fixtures}))
        work, metadata = core.corpus_work(self.root, 2)
        source = Path(live.__file__).resolve().parents[1]
        self.monitor = self.root / 'monitor.json'
        self.admission = {'candidate': live.IDENTITY, 'root_live_throughput_lease': True,
            'lease_id': 'owned-test', 'node_id': 'fake-single-node', 'direct_url': 'http://node.invalid',
            'issued_unix': time.time(), 'expires_unix': time.time() + 120,
            'monitor_path': str(self.monitor), 'expected_worker': {'pid': 123, 'starts': 1, 'start_marker': 'fixed'},
            'expected_health': {'backend': 'fake-stock', 'model': 'frozen-model', 'revision': 'frozen-revision'},
            'node_source_sha256': '0' * 64, 'health_profile': 'waystone', 'initial_loaded': True, 'allow_request_wake': False,
            'source_sha256': {name: core.digest((source / name).read_bytes()) for name in live.SOURCE_FILES},
            'concurrency': [2], 'repeat': 2, 'warmup_cycles': 1, 'planned_pages': len(work),
            'corpus_metadata': metadata, 'request_seconds': .5, 'cell_seconds': 1,
            'response_bytes': 1024, 'work_limit': 20, 'drain_seconds': .08, 'poll_seconds': .01,
            'monitor_max_age_seconds': 5, 'max_output_bytes': 16 * 1024**2, 'max_client_bytes': 32 * 1024**2,
            'drain_contract': 'service-pending-holds-inference', 'drain_contract_source_sha256': '1' * 64}
        self.health = {'backend': 'fake-stock', 'model': 'frozen-model', 'revision': 'frozen-revision',
            'loaded': True, 'state': 'ready', 'pending_or_active': 0, 'capacity': 4,
            'accepting': True, 'worker_pid': 123, 'starts': 1}
        self.witness = {'lease_id': 'owned-test', 'node_id': 'fake-single-node',
            'worker_identity': self.admission['expected_worker'], 'source_sha256': '0' * 64,
            'model_revision': 'frozen-revision', 'model': 'frozen-model', 'backend': 'fake-stock',
            'health_state': 'ready', 'issued_unix': time.time(),
            'allow_admission': True, 'resources_passed': True, 'abort': False}
        self.write_witness()
        self.raw = json.dumps({'choices': [{'message': {'content': '<div data-label="Text" data-bbox="0 0 2 2">ok</div>'},
            'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}).encode()

    def tearDown(self):
        self.temporary.cleanup()

    def write_witness(self):
        self.monitor.write_text(json.dumps(self.witness))

    async def invoke(self, handler, name='out', **kwargs):
        return await live.run(self.admission, 'http://node.invalid', self.root, self.monitor,
            self.root / name, execute=True, http_transport=httpx.MockTransport(handler), **kwargs)

    async def test_bounded_heterogeneous_rolling_success_no_management(self):
        methods = []
        active = set()
        released = asyncio.Event()
        peak = 0
        async def handler(request):
            nonlocal peak
            methods.append((request.method, request.url.path))
            if request.method == 'GET':
                return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
            identifier = request.headers['X-Request-ID']
            self.assertEqual(request.url.path, '/v1/chat/completions')
            active.add(identifier)
            peak = max(peak, len(active))
            self.health['pending_or_active'] = len(active)
            if identifier.endswith('measured-00002'):
                self.assertIn('owned-test-c0-measured-00000', active)
                released.set()
            gate = released if identifier.endswith('measured-00000') else None
            def close():
                active.discard(identifier)
                self.health['pending_or_active'] = len(active)
            return httpx.Response(200, stream=Stream(self.raw, gate=gate, close=close))
        report = await self.invoke(handler)
        self.assertTrue(report['valid'])
        self.assertTrue(report['backend_drain_verified'])
        self.assertFalse(report['hardware_ownership_released'])
        self.assertLessEqual(peak, 2)
        measured = report['configurations'][0]['measured']['result']
        self.assertEqual(measured['completed_correct_pages'], 4)
        self.assertEqual(len([m for m in methods if m[0] == 'POST']), 6)
        self.assertEqual({path for _, path in methods}, {'/health', '/v1/chat/completions'})
        self.assertEqual(len(list((self.root / 'out').glob('*.response.raw'))), 6)

    async def test_fault_is_not_replayed_and_partial_bytes_retained(self):
        posts = []
        self.admission['concurrency'] = [1, 2]
        async def handler(request):
            if request.method == 'GET':
                return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
            posts.append(request.headers['X-Request-ID'])
            return httpx.Response(200, stream=Stream(self.raw[:30], fail=True))
        report = await self.invoke(handler)
        self.assertFalse(report['valid'])
        self.assertEqual(len(posts), 1)
        self.assertEqual(report['unattempted_concurrency'], [2])
        self.assertEqual((self.root / 'out' / (posts[0] + '.response.raw')).read_bytes(), self.raw[:30])
        receipt = json.loads((self.root / 'out' / (posts[0] + '.receipt.json')).read_text())
        self.assertEqual(receipt['transport']['post_attempts'], 1)
        self.assertEqual(receipt['http_status'], 599)

    async def test_monitor_abort_cancels_client_but_busy_backend_not_claimed_drained(self):
        gate = asyncio.Event()
        posts = []
        async def handler(request):
            if request.method == 'GET':
                return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
            posts.append(request.headers['X-Request-ID'])
            self.health['pending_or_active'] = 1
            self.witness['resources_passed'] = False
            self.write_witness()
            return httpx.Response(200, stream=Stream(b'partial', gate=gate))
        report = await self.invoke(handler)
        self.assertFalse(report['valid'])
        self.assertFalse(report['backend_drain_verified'])
        self.assertFalse(report['hardware_ownership_released'])
        self.assertEqual(len(posts), 1)
        self.assertIn('abort', report['configurations'][0]['warmup'])

    async def test_worker_drift_and_router_health_stop_before_post(self):
        for name, changed in (('router', {'backends': {}}), ('worker', {'worker_pid': 999}), ('starts', {'starts': 2})):
            posts = []
            async def handler(request):
                if request.method == 'GET':
                    return httpx.Response(200, stream=Stream(json.dumps(dict(self.health, **changed)).encode()))
                posts.append(request)
                return httpx.Response(200, stream=Stream(self.raw))
            report = await self.invoke(handler, name=name)
            self.assertFalse(report['valid'])
            self.assertFalse(report['backend_drain_verified'])
            self.assertEqual(posts, [])

    async def test_engine_metrics_required_cannot_use_gateway_zero(self):
        self.admission['drain_contract'] = 'external-engine-metrics'
        self.admission['engine_metrics_source'] = 'http://engine.invalid/metrics'
        self.witness['engine_telemetry'] = {'source': 'http://engine.invalid/metrics',
            'running': 1, 'waiting': 0, 'observed_unix': time.time()}
        self.write_witness()
        async def handler(request):
            return httpx.Response(200, stream=Stream(json.dumps(self.health).encode())) if request.method == 'GET' else httpx.Response(200, stream=Stream(self.raw))
        report = await self.invoke(handler)
        self.assertFalse(report['valid'])
        self.assertFalse(report['backend_drain_verified'])
        self.assertIsNone(report['configurations'][0]['measured'])

    async def test_refuses_unexecuted_stale_mismatched_and_unbounded_before_http(self):
        async def forbidden(request):
            self.fail('HTTP admission must not happen')
        with self.assertRaises(live.AdmissionError):
            await live.run(self.admission, 'http://node.invalid', self.root, self.monitor, self.root / 'disabled',
                           http_transport=httpx.MockTransport(forbidden))
        for key, value in (('expires_unix', time.time() + 1), ('issued_unix', time.time() - 301),
                           ('planned_pages', 3), ('max_client_bytes', 1), ('max_output_bytes', 1)):
            original = self.admission[key]
            self.admission[key] = value
            with self.assertRaises(live.AdmissionError):
                await self.invoke(forbidden, name=key)
            self.admission[key] = original
            self.assertFalse((self.root / key).exists())

    async def test_exact_northstone_gateway_profile_requires_external_engine_identity(self):
        self.admission['health_profile'] = 'northstone'
        self.admission['initial_state'] = 'awake'
        self.admission['drain_contract'] = 'external-engine-metrics'
        self.admission['engine_metrics_source'] = 'http://engine.invalid/metrics'
        self.health = {'state': 'awake', 'pending_or_active': 0, 'worker_pid': 123,
            'starts': 1, 'wakes': 0, 'sleeps': 0, 'last_start_seconds': 1.2,
            'last_wake_seconds': None, 'last_error': None}
        self.witness['health_state'] = 'awake'
        self.witness['engine_telemetry'] = {'source': 'http://engine.invalid/metrics',
            'running': 0, 'waiting': 0, 'observed_unix': time.time()}
        self.write_witness()
        posts = []
        async def handler(request):
            if request.method == 'GET':
                return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
            posts.append(request)
            return httpx.Response(200, stream=Stream(self.raw))
        report = await self.invoke(handler)
        self.assertTrue(report['valid'])
        self.assertTrue(report['backend_drain_verified'])
        self.assertEqual(len(posts), 6)
        self.assertNotIn('backend', report['initial_health'])  # Actual gateway shape, not fictional metadata.

    async def test_pre_measured_health_drift_never_posts_measured_work(self):
        for name, changed in (('became-cold', {'state': 'cold', 'loaded': False}),
                              ('worker-replaced', {'worker_pid': 777, 'starts': 2})):
            posts = []
            gets = 0
            async def handler(request):
                nonlocal gets
                if request.method == 'GET':
                    gets += 1
                    # Initial, prewarmup and warmup-drain observations; drift before measured.
                    if gets >= 4:
                        return httpx.Response(200, stream=Stream(json.dumps(dict(self.health, **changed)).encode()))
                    return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
                posts.append(request.headers['X-Request-ID'])
                return httpx.Response(200, stream=Stream(self.raw))
            report = await self.invoke(handler, name=name)
            self.assertFalse(report['valid'])
            self.assertEqual(len(posts), 2)
            self.assertTrue(all('warmup' in identifier for identifier in posts))
            self.assertIsNone(report['configurations'][0]['measured'])

    async def test_caller_cancellation_joins_http_streams_and_retains_ownership(self):
        entered = asyncio.Event()
        closed = asyncio.Event()
        gate = asyncio.Event()
        async def handler(request):
            if request.method == 'GET':
                return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
            entered.set()
            return httpx.Response(200, stream=Stream(self.raw[:10], gate=gate, close=closed.set))
        task = asyncio.create_task(self.invoke(handler))
        await entered.wait()
        await asyncio.sleep(.005)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        report = json.loads((self.root / 'out' / 'summary.json').read_text())
        self.assertFalse(report['valid'])
        self.assertFalse(report['hardware_ownership_released'])
        self.assertIn('cancelled', report['failure'])

    def test_lease_short_of_terminal_drain_rejected(self):
        self.admission['drain_seconds'] = 20
        # Covers older envelope (cells plus two cell drains +10), misses final drain.
        self.admission['expires_unix'] = time.time() + 55
        with self.assertRaisesRegex(live.AdmissionError, 'whole warmup'):
            live.validate_admission(self.admission, 'http://node.invalid', self.root, self.monitor, execute=True)

    def test_cold_wake_scope_is_held_before_http(self):
        for initial_loaded, allow_wake in ((False, True), (True, True), (False, False)):
            self.admission['initial_loaded'] = initial_loaded
            self.admission['allow_request_wake'] = allow_wake
            with self.assertRaisesRegex(live.AdmissionError, 'warm-only'):
                live.validate_admission(self.admission, 'http://node.invalid', self.root, self.monitor, execute=True)

    def test_witness_forecast_rejects_deterministic_late_sample_exhaustion(self):
        self.admission['cell_seconds'] = 100
        self.admission['expires_unix'] = time.time() + 1000
        with self.assertRaisesRegex(live.AdmissionError, '10000 passive witness'):
            live.validate_admission(self.admission, 'http://node.invalid', self.root, self.monitor, execute=True)

    async def test_response_bound_invalidates_and_keeps_only_bounded_prefix(self):
        self.admission['concurrency'] = [1]
        self.admission['response_bytes'] = 20
        posts = []
        async def handler(request):
            if request.method == 'GET':
                return httpx.Response(200, stream=Stream(json.dumps(self.health).encode()))
            posts.append(request.headers['X-Request-ID'])
            return httpx.Response(200, stream=Stream(self.raw))
        report = await self.invoke(handler)
        self.assertFalse(report['valid'])
        self.assertEqual(len(posts), 1)
        self.assertEqual(len((self.root / 'out' / (posts[0] + '.response.raw')).read_bytes()), 20)

if __name__ == '__main__':
    unittest.main()
