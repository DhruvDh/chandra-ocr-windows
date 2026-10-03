"""Acceptance orchestration tests use fakes; never invoke installed units or GPUs."""
import argparse
import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

BENCHMARKS = Path(__file__).resolve().parents[2] / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
import deployed_lifecycle as deployed


def good(backend):
    return {'correctness': {'passed': True}, 'response_headers': {'x-chandra-backend': backend}, 'ttft_seconds': .1,
            'result': {'content': 'retained exact fixture output', 'usage': {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4}, 'finish_reason': 'stop'}}


def router_health(north=0, way=0):
    return {'backends': {'northstone': {'active': north, 'remote_active': north, 'state': 'ready', 'uncertain': False}, 'waystone': {'active': way, 'remote_active': way, 'state': 'ready', 'uncertain': False}}}


class DeployedLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, directory):
        args = argparse.Namespace(output=Path(directory), base_url='http://test/v1', worker_base_url='http://worker/v1',
                                  corpus=BENCHMARKS / 'inputs-v2', request_seconds=1, drain_seconds=1,
                                  overflow_fixture='representative')
        return deployed.Lifecycle(args, None, {'phases': {}})

    async def test_overflow_requires_two_active_north_and_third_waystone(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            started = 0; released = asyncio.Event()
            class Harness:
                async def request(self, fixture, stream=True):
                    nonlocal started
                    started += 1
                    if started <= 2:
                        await released.wait()
                        return good('northstone')
                    released.set()
                    return good('waystone')
                async def health(self): return router_health(north=2 if started >= 2 else 0)
                async def idle(self): return {'passed': True}
            runner.harness = lambda *_: Harness()
            result = await runner.overflow()
            self.assertTrue(result['passed'])
            self.assertEqual(started, 3)
            self.assertTrue(result['observed_northstone_active_two'])

    async def test_crash_kills_only_verified_unit_and_requires_no_done_plus_one_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            killed = asyncio.Event()
            calls = 0
            class Harness:
                output = Path(directory)
                async def request(self, fixture, stream=True, first_content=None):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        first_content.set()
                        await killed.wait()
                        (self.output / 'partial.raw').write_bytes(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\ndata: {"error":{"message":"Request was not replayed"}}\n\n')
                        return {'raw_file': 'partial.raw', 'ttft_seconds': .1, 'correctness': {'passed': False}}
                    return good('waystone')
                async def health(self): return router_health(way=0 if killed.is_set() else 1)
                async def idle(self): return {'passed': True}
            runner.harness = lambda *_: Harness()
            runner.verify_owned_unit = AsyncMock(return_value={'unit': deployed.WORKER_UNIT})
            runner.restart_count = AsyncMock(side_effect=[4, 5, 5, 5])
            runner.worker_health = AsyncMock(return_value={'state': 'cold', 'pending_or_active': 0})
            async def control(*args):
                self.assertEqual(args, ('kill', '--kill-whom=main', '--signal=SIGKILL', deployed.WORKER_UNIT))
                killed.set()
                return ''
            runner.control = AsyncMock(side_effect=control)
            result = await runner.crash()
            self.assertTrue(result['passed'])
            self.assertEqual(calls, 2)
            runner.verify_owned_unit.assert_awaited_once()
            runner.control.assert_awaited_once()
            self.assertEqual(result['restarts_after'], result['restarts_before'] + 1)

    async def test_finished_stream_never_triggers_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            class Harness:
                async def request(self, *args, first_content=None):
                    first_content.set()
                    return good('waystone')
                async def health(self): return router_health()
                async def idle(self): return {'passed': True}
            runner.harness = lambda *_: Harness()
            runner.control = AsyncMock()
            with self.assertRaises(ValueError): await runner.crash()
            runner.control.assert_not_called()

    async def test_two_idle_cycles_poll_health_models_without_extra_posts(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            clock = [0.0]; boundary = [0.0]; posts = 0
            class Harness:
                async def request(self, *args):
                    nonlocal posts
                    posts += 1; boundary[0] = clock[0]
                    return good('waystone')
                async def idle(self): return {'passed': True}
            runner.harness = lambda *_: Harness()
            async def health():
                cold = clock[0] - boundary[0] >= 300
                return {'state': 'cold' if cold else 'ready', 'loaded': not cold, 'pending_or_active': 0,
                        'model_verification': {'files_checked': 9, 'files_hashed': 0},
                        'torch_xpu_memory': {'allocated_bytes': 0 if cold else 10, 'reserved_bytes': 0 if cold else 20}}
            runner.worker_health = AsyncMock(side_effect=health)
            response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'data': [{'id': 'chandra'}]})
            runner.client = SimpleNamespace(get=AsyncMock(return_value=response))
            async def sleep(seconds): clock[0] += seconds
            with patch.object(deployed, 'time', SimpleNamespace(monotonic=lambda: clock[0])), patch.object(deployed.asyncio, 'sleep', side_effect=sleep):
                result = await runner.idle()
            self.assertTrue(result['passed'])
            self.assertEqual(posts, 2)
            self.assertEqual([cycle['unload_seconds'] for cycle in result['cycles']], [300, 300])
            self.assertTrue(deployed.unloaded(result['final_worker_health']))
            self.assertEqual(runner.client.get.await_count, 122)

    def test_unload_requires_observable_zero_allocated_and_reserved_memory(self):
        self.assertFalse(deployed.unloaded({'state': 'cold', 'loaded': False, 'pending_or_active': 0}))
        self.assertFalse(deployed.unloaded({'state': 'cold', 'loaded': False, 'pending_or_active': 0,
                                           'torch_xpu_memory': {'allocated_bytes': 0, 'reserved_bytes': 1}}))


if __name__ == '__main__': unittest.main()
