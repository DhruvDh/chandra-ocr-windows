"""CPU-only behavioral tests; fake transports never contact an endpoint."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from .throughput import Bounds, Reply, Work, corpus_work, endpoint_validator, measure, replay_cli, digest
except ImportError:
    from throughput import Bounds, Reply, Work, corpus_work, endpoint_validator, measure, replay_cli, digest


def jobs(count=5):
    return tuple(Work(str(i), ('tiny', 'large', 'small')[i % 3], b'{}', '{}') for i in range(count))


def correct(item, reply):
    return {'passed': reply.raw == b'correct', 'failures': [] if reply.raw == b'correct' else ['content']}


class Rolling(unittest.IsolatedAsyncioTestCase):
    async def test_refills_while_slow_peer_remains_and_caps_flight(self):
        slow_release = asyncio.Event()
        third_started = asyncio.Event()
        active = set()
        peak = 0
        seen = []
        async def transport(item, request_id, deadline, response_limit):
            nonlocal peak
            active.add(item.work_id)
            peak = max(peak, len(active))
            seen.append((item.work_id, set(active)))
            if item.work_id == '0':
                await slow_release.wait()
            elif item.work_id == '2':
                self.assertIn('0', active)  # Third starts before original slow peer finishes.
                third_started.set()
            await asyncio.sleep(0)
            active.remove(item.work_id)
            return Reply(b'correct')
        task = asyncio.create_task(measure(jobs(), transport, correct, Bounds(2, 2, 5), request_prefix='rolling'))
        await asyncio.wait_for(third_started.wait(), 1)
        slow_release.set()
        result = await task
        self.assertTrue(result['valid'])
        self.assertEqual(peak, 2)
        self.assertEqual(result['peak_in_flight'], 2)
        self.assertEqual(len(result['records']), 5)
        self.assertEqual([r['work_id'] for r in result['records']], ['0', '1', '2', '3', '4'])
        self.assertEqual(len({r['request_id'] for r in result['records']}), 5)

    async def test_error_stops_admission_but_drains_slow_admitted_peer(self):
        release = asyncio.Event()
        failed = asyncio.Event()
        calls = []
        async def transport(item, request_id, deadline, response_limit):
            calls.append(item.work_id)
            if item.work_id == '0':
                await release.wait()
                return Reply(b'correct')
            failed.set()
            return Reply(b'wrong')
        task = asyncio.create_task(measure(jobs(), transport, correct, Bounds(2, 2, 5), request_prefix='failure'))
        await failed.wait()
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        self.assertEqual(calls, ['0', '1'])
        release.set()
        result = await task
        self.assertFalse(result['valid'])
        self.assertEqual(result['admitted_pages'], 2)
        self.assertEqual(result['completed_requests'], 2)
        self.assertEqual(result['completed_correct_pages'], 1)
        self.assertEqual(result['unadmitted_work_ids'], ['2', '3', '4'])
        self.assertIsNone(result['completed_correct_pages_per_second'])

    async def test_equal_work_mix_and_denominator_includes_final_drain(self):
        work = jobs(6)
        summaries = []
        for concurrency in (1, 3):
            admitted = []
            async def transport(item, request_id, deadline, response_limit):
                admitted.append(item.fixture_id)
                await asyncio.sleep(.01 if item.work_id != '5' else .04)
                return Reply(b'correct')
            result = await measure(work, transport, correct, Bounds(concurrency, 2, 5), request_prefix=f'c{concurrency}')
            summaries.append(result)
            self.assertEqual(admitted, [item.fixture_id for item in work])
            self.assertTrue(result['valid'])
            self.assertGreaterEqual(result['elapsed_seconds'], max(r['finished_seconds'] for r in result['records']))
            self.assertAlmostEqual(result['completed_correct_pages_per_second'], 6 / result['elapsed_seconds'])
        self.assertEqual(summaries[0]['workload_sha256'], summaries[1]['workload_sha256'])
        self.assertEqual(summaries[0]['work_sequence'], summaries[1]['work_sequence'])

    async def test_deadline_and_response_bound_invalidate_without_retry(self):
        cancellations = []
        async def slow(item, request_id, deadline, response_limit):
            try:
                await asyncio.sleep(10)
            finally:
                cancellations.append(request_id)
        result = await measure(jobs(), slow, correct, Bounds(2, .5, .02), request_prefix='deadline')
        self.assertFalse(result['valid'])
        self.assertEqual(result['admitted_pages'], 2)
        self.assertEqual(len(cancellations), 2)
        self.assertEqual(result['completed_requests'], 2)
        retained = []
        async def large(item, request_id, deadline, response_limit):
            return Reply(b'correct' * 10)
        result = await measure(jobs(), large, correct, Bounds(1, 1, 2, response_bytes=5),
                               request_prefix='bytes', on_record=lambda r, raw: retained.append(raw))
        self.assertFalse(result['valid'])
        self.assertEqual(result['admitted_pages'], 1)
        self.assertEqual(retained, [b'corre'])
        self.assertIn('byte limit', result['records'][0]['error'])

    async def test_status_and_receipt_failure_cannot_count_as_improvement(self):
        for status in (500, 200):
            async def transport(item, request_id, deadline, response_limit):
                return Reply(b'correct', status)
            def broken_record(record, raw):
                raise OSError('disk full')
            result = await measure(jobs(), transport, correct, Bounds(1), request_prefix=f's{status}',
                                   on_record=broken_record if status == 200 else None)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['completed_correct_pages_per_second'])
            self.assertEqual(result['admitted_pages'], 1)


    async def test_caller_cancellation_joins_all_local_tasks(self):
        entered = asyncio.Event()
        active = set()
        async def transport(item, request_id, deadline, response_limit):
            active.add(request_id)
            if len(active) == 2:
                entered.set()
            try:
                await asyncio.sleep(10)
            finally:
                active.remove(request_id)
        task = asyncio.create_task(measure(jobs(), transport, correct, Bounds(2), request_prefix='cancel'))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(active, set())  # Client task cleanup only, not a remote assertion.

    async def test_duplicate_work_rejected_before_transport(self):
        calls = []
        async def transport(*args):
            calls.append(args)
            return Reply(b'correct')
        item = jobs(1)[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            await measure((item, item), transport, correct, Bounds(1), request_prefix='duplicate')
        self.assertEqual(calls, [])

    async def test_offline_cli_warmup_is_separate_and_failure_blocks_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {key: [] for key in ('table', 'numbers', 'text', 'end_markers', 'ordered_text', 'required_labels')}
            fixtures = []
            replay = {}
            for fixture_id in ('tiny', 'large'):
                image = fixture_id.encode()
                (root / (fixture_id + '.png')).write_bytes(image)
                fixtures.append({'id': fixture_id, 'image': fixture_id + '.png',
                                 'image_sha256': digest(image), 'expected': expected})
                response = {'choices': [{'message': {'content': '<div data-label="Text" data-bbox="0 0 1 1">ok</div>'},
                                         'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}
                (root / (fixture_id + '.raw')).write_text(json.dumps(response))
                replay[fixture_id] = {'response_file': fixture_id + '.raw', 'delay_seconds': .001}
            (root / 'manifest.json').write_text(json.dumps({'fixtures': fixtures}))
            replay_path = root / 'replay.json'
            replay_path.write_text(json.dumps(replay))
            args = SimpleNamespace(direct_url='http://127.0.0.1:9999/v1', corpus=root, repeat=2,
                replay=replay_path, output=root / 'success', concurrency=[1, 2], warmup_cycles=1,
                request_seconds=1, run_seconds=5, max_response_bytes=1024)
            with patch('socket.create_connection', side_effect=AssertionError('Network forbidden')):
                self.assertEqual(await replay_cli(args), 0)
            summary = json.loads((args.output / 'summary.json').read_text())
            self.assertFalse(summary['hardware_measurement'])
            cells = summary['configurations']
            for cell in cells:
                self.assertEqual(cell['warmup']['completed_requests'], 2)
                self.assertEqual(cell['measured']['completed_requests'], 4)
                self.assertTrue(cell['measured']['client_tasks_drained'])
                self.assertFalse(cell['measured']['backend_drain_verified'])
            self.assertEqual(cells[0]['measured']['workload_sha256'], cells[1]['measured']['workload_sha256'])
            self.assertEqual(len(list(args.output.glob('*.response.raw'))), 12)
            replay['tiny']['http_status'] = 500
            replay_path.write_text(json.dumps(replay))
            args.output = root / 'failed'
            self.assertEqual(await replay_cli(args), 1)
            summary = json.loads((args.output / 'summary.json').read_text())
            self.assertIsNone(summary['configurations'][0]['measured'])
            self.assertEqual(summary['unattempted_concurrency'], [2])
            self.assertEqual(len(list(args.output.glob('*.response.raw'))), 1)
            args.max_replay_raw_bytes = 1
            args.output = root / 'disk-rejected'
            with self.assertRaisesRegex(ValueError, 'disk bound'):
                await replay_cli(args)
            self.assertFalse(args.output.exists())
            args.max_replay_raw_bytes = 512 * 1024 * 1024
            args.warmup_cycles = 0
            args.output = root / 'failed-measured'
            self.assertEqual(await replay_cli(args), 1)
            summary = json.loads((args.output / 'summary.json').read_text())
            self.assertFalse(summary['configurations'][0]['measured']['valid'])
            self.assertEqual(summary['unattempted_concurrency'], [2])
            self.assertEqual(len(summary['configurations']), 1)
            self.assertEqual(len(list(args.output.glob('*.response.raw'))), 1)


class CorpusAndQuality(unittest.TestCase):
    def test_corpus_work_is_frozen_heterogeneous_and_hash_checked(self):
        root = Path(__file__).with_name('inputs-v2')
        work, metadata = corpus_work(root, 2)
        self.assertEqual(len(work), 2 * len(metadata['fixture_order']))
        self.assertEqual([w.fixture_id for w in work], metadata['fixture_order'] * 2)
        self.assertEqual(len({w.work_id for w in work}), len(work))
        self.assertIs(work[0].request, work[len(metadata['fixture_order'])].request)
        self.assertIs(work[0].expected_json, work[len(metadata['fixture_order'])].expected_json)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'bad.png').write_bytes(b'changed')
            (target / 'manifest.json').write_text(json.dumps({'fixtures': [{'id': 'bad', 'image': 'bad.png',
                'image_sha256': '0' * 64, 'expected': {key: [] for key in ('table', 'numbers', 'text', 'end_markers', 'ordered_text', 'required_labels')}}]}))
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                corpus_work(target, 1)
            (target / 'manifest.json').write_text(json.dumps({'fixtures': [{'id': 'other', 'image': 'bad.png',
                'image_sha256': digest(b'changed'), 'expected': {'ordered_blocks': [], 'end_marker': 'END'}}]}))
            with self.assertRaisesRegex(ValueError, 'Unsupported expected schema'):
                corpus_work(target, 1)

    def test_existing_endpoint_quality_gate_used(self):
        expected = {'table': [], 'numbers': ['123'], 'text': ['A'], 'end_markers': ['END'],
                    'ordered_text': ['A', 'END'], 'required_labels': ['Text']}
        item = Work('0', 'tiny', b'{"max_tokens":12384,"stream":false}', json.dumps(expected))
        raw = json.dumps({'choices': [{'message': {'content': '<div data-label="Text" data-bbox="0 0 20 20">A 123 END</div>'},
                                      'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': 10, 'completion_tokens': 12, 'total_tokens': 22}}).encode()
        self.assertTrue(endpoint_validator(item, Reply(raw))['passed'])
        corrupted = json.loads(raw)
        corrupted['choices'][0]['finish_reason'] = 'length'
        self.assertFalse(endpoint_validator(item, Reply(json.dumps(corrupted).encode()))['passed'])

    def test_bounds_and_duplicate_ids_reject_before_transport(self):
        for bounds in (Bounds(0), Bounds(65), Bounds(1, run_seconds=float('nan')), Bounds(1, response_bytes=0)):
            with self.assertRaises(ValueError):
                bounds.check()

if __name__ == '__main__':
    unittest.main()
