"""Offline scheduling checks: service admission, execution permits and idle unloading."""

import asyncio
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chandra_service.service import Worker


class GatedBackend:
    def __init__(self, parallel=None):
        if parallel is not None:
            self.parallel_requests = parallel
        self.gates = [threading.Event() for _ in range(8)]
        self.mutex = threading.Lock()
        self.started, self.active, self.peak = set(), 0, 0
        self.unload_entered, self.unload_allowed = threading.Event(), threading.Event()
        self.unload_allowed.set()
        self.active_at_unload = None

    def state(self):
        with self.mutex:
            return set(self.started), self.active, self.peak

    def generate(self, request, cancel):
        ident = request["id"]
        with self.mutex:
            self.started.add(ident)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.gates[ident].wait()
            yield {"finish_reason": "stop", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        finally:
            with self.mutex:
                self.active -= 1

    def unload(self):
        with self.mutex:
            self.active_at_unload = self.active
        self.unload_entered.set()
        self.unload_allowed.wait()


class ParallelCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def until(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("fake scheduling condition did not arrive")
            await asyncio.sleep(0.01)

    def make_worker(self, parallel=None, queue_limit=4):
        backend = GatedBackend(parallel)
        worker = Worker(backend, queue_limit)
        idlers = []

        async def cleanup():
            for gate in backend.gates:
                gate.set()
            backend.unload_allowed.set()
            for task in idlers:
                task.cancel()
            if idlers:
                await asyncio.gather(*idlers, return_exceptions=True)
            await worker.close()

        self.addAsyncCleanup(cleanup)
        return backend, worker, idlers

    def submit(self, worker, ident):
        job = worker.submit({"id": ident, "max_tokens": 1})
        self.assertIsNotNone(job)
        return job

    def test_parallel_count_is_a_bounded_integer(self):
        for value in (1, 2, 4, 64):
            with self.subTest(value=value):
                self.assertEqual(Worker(SimpleNamespace(parallel_requests=value), 0).execution_slots, value)
        for value in (True, False, None, 0, -1, 65, 4.0, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Worker(SimpleNamespace(parallel_requests=value), 0)

    async def test_four_execute_and_fifth_admitted_job_waits_for_a_permit(self):
        backend, worker, _ = self.make_worker(4, queue_limit=4)
        jobs = [self.submit(worker, ident) for ident in range(5)]
        await self.until(lambda: len(backend.state()[0]) == 4)
        self.assertEqual(backend.state(), ({0, 1, 2, 3}, 4, 4))
        self.assertFalse(jobs[4].done.is_set())
        self.assertIsNone(worker.submit({"id": 5, "max_tokens": 1}))
        backend.gates[0].set()
        await self.until(lambda: 4 in backend.state()[0])
        self.assertEqual(backend.state()[2], 4)
        self.assertTrue(jobs[0].done.is_set())

    async def test_zero_queue_admits_only_one_even_with_four_execution_slots(self):
        backend, worker, _ = self.make_worker(4, queue_limit=0)
        first = self.submit(worker, 0)
        self.assertIsNone(worker.submit({"id": 1, "max_tokens": 1}))
        await self.until(lambda: backend.state()[0] == {0})
        self.assertEqual(backend.state()[1:], (1, 1))
        backend.gates[0].set()
        await self.until(lambda: first.done.is_set() and not worker.jobs)
        self.submit(worker, 1)
        await self.until(lambda: 1 in backend.state()[0])
        self.assertEqual(backend.state()[2], 1)

    async def test_default_one_and_existing_two_slot_profiles_keep_their_behavior(self):
        for parallel, expected in ((None, 1), (2, 2)):
            with self.subTest(parallel=parallel):
                backend, worker, _ = self.make_worker(parallel, queue_limit=2)
                jobs = [self.submit(worker, ident) for ident in range(3)]
                await self.until(lambda: len(backend.state()[0]) == expected)
                self.assertEqual(worker.execution_slots, expected)
                self.assertEqual(backend.state()[1:], (expected, expected))
                self.assertFalse(jobs[expected].done.is_set())
                for gate in backend.gates:
                    gate.set()
                await self.until(lambda: not worker.jobs)
                self.assertEqual(backend.state()[0], {0, 1, 2})

    async def test_idle_unload_waits_for_all_workers_and_holds_every_permit(self):
        backend, worker, idlers = self.make_worker(4, queue_limit=3)
        for ident in range(4):
            self.submit(worker, ident)
        await self.until(lambda: len(backend.state()[0]) == 4)
        backend.unload_allowed.clear()
        idle = asyncio.create_task(worker.idle(0.01))
        idlers.append(idle)
        backend.gates[0].set()
        await self.until(lambda: backend.state()[1] == 3)
        await asyncio.sleep(0.03)
        self.assertFalse(backend.unload_entered.is_set())
        for ident in (1, 2, 3):
            backend.gates[ident].set()
        await self.until(backend.unload_entered.is_set)
        self.assertEqual(backend.active_at_unload, 0)
        self.submit(worker, 4)
        await asyncio.sleep(0.03)
        self.assertNotIn(4, backend.state()[0])
        backend.unload_allowed.set()
        await self.until(lambda: 4 in backend.state()[0])

    async def test_idle_ticks_never_take_execution_permits_from_continuous_backlog(self):
        backend, worker, idlers = self.make_worker(4, queue_limit=7)
        for ident in range(8):
            self.submit(worker, ident)
        await self.until(lambda: backend.state()[0] == {0, 1, 2, 3})
        original_sleep, original_acquire = asyncio.sleep, worker.lock.acquire
        sleeps = asyncio.Queue()
        idle_attempts = []
        idle = None

        async def controlled_sleep(seconds):
            if asyncio.current_task() is idle:
                resume = asyncio.get_running_loop().create_future()
                sleeps.put_nowait(resume)
                await resume
            else:
                await original_sleep(seconds)

        async def observed_acquire():
            if asyncio.current_task() is idle:
                idle_attempts.append(len(worker.jobs))
            return await original_acquire()

        with patch.object(asyncio, "sleep", controlled_sleep), \
                patch.object(worker.lock, "acquire", observed_acquire):
            idle = asyncio.create_task(worker.idle(60))
            idlers.append(idle)
            pending_tick = await asyncio.wait_for(sleeps.get(), 1)

            async def tick(recent=False):
                nonlocal pending_tick
                # Expire recency during backlog so the jobs check itself must protect capacity.
                worker.last_activity = time.monotonic() if recent else time.monotonic() - 120
                pending_tick.set_result(None)
                await original_sleep(0)  # The resumed idle task checks state before this task resumes.
                self.assertEqual(idle_attempts, [], "idle attempted the all-slot barrier while work was admitted")
                pending_tick = await asyncio.wait_for(sleeps.get(), 1)

            try:
                for ident in range(4):
                    await tick()
                    backend.gates[ident].set()
                    await self.until(lambda: ident + 4 in backend.state()[0])
                    self.assertEqual(backend.state()[1:], (4, 4))
                    self.assertTrue(worker.jobs)
                await tick()  # Four second-wave calls still own all four permits.
                for ident in range(4, 8):
                    backend.gates[ident].set()
                await self.until(lambda: not worker.jobs)
                await tick(recent=True)  # Recently completed work also avoids needless barrier acquisition.
                self.assertFalse(backend.unload_entered.is_set())
            finally:
                idle.cancel()
                await asyncio.gather(idle, return_exceptions=True)

    async def test_cancel_during_all_slot_acquisition_restores_partial_permits(self):
        _, worker, _ = self.make_worker(4)
        await worker.lock.acquire()  # One running owner leaves exactly three permits available.
        acquired_three = asyncio.Event()
        entered = asyncio.Event()
        original_acquire = worker.lock.acquire
        count = 0
        task = None

        async def observed_acquire():
            nonlocal count
            result = await original_acquire()
            if asyncio.current_task() is task:
                count += 1
                if count == 3:
                    acquired_three.set()
            return result

        async def take_all():
            async with worker._all_slots():
                entered.set()

        with patch.object(worker.lock, "acquire", observed_acquire):
            task = asyncio.create_task(take_all())
            try:
                await asyncio.wait_for(acquired_three.wait(), 1)
                self.assertFalse(entered.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                worker.lock.release()
        # All four permits must now be usable together; a leaked partial acquisition times out.
        await asyncio.wait_for(take_all(), 1)
        self.assertTrue(entered.is_set())
