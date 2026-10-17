"""Offline app lifespan: passive routes neither wake nor retain an idle fake engine."""

import asyncio
import time
import unittest

import httpx

from runtime.llama.app import create_llama_app
from tests.llama.test_backend import Harness, request


class LlamaAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_and_models_do_not_wake_or_extend_idle_lifetime(self):
        h = Harness()
        self.addAsyncCleanup(asyncio.to_thread, h.close)
        idle_seconds = 0.2
        app = create_llama_app(h.backend, queue_limit=0, inference_seconds=5, idle_seconds=idle_seconds)

        async def poll(client):
            health = await client.get("/health")
            models = await client.get("/v1/models")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(models.status_code, 200)
            self.assertEqual(models.json()["data"][0]["id"], "chandra")
            return health.json()

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                for _ in range(3):
                    self.assertEqual((await poll(client))["state"], "asleep")
                    await asyncio.sleep(0.02)
                self.assertEqual((h.workers, h.clients, h.calls, h.preparations), ([], [], [], 0))

                response = await client.post("/v1/chat/completions", json=request())
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["choices"][0]["message"]["content"], "<p>ok</p>")
                self.assertEqual((len(h.workers), len(h.posts)), (1, 1))
                resident = h.workers[0]
                activity = app.state.worker.last_activity
                # Poll much faster than the idle interval. A route that renews activity will
                # prevent unloading and fail this bound, anchored to the completed request.
                deadline = activity + idle_seconds + 3
                while True:
                    health = await poll(client)
                    if resident.closed and health["state"] == "asleep":
                        break
                    self.assertLess(time.monotonic(), deadline, "passive polling retained the idle worker")
                    await asyncio.sleep(0.02)
                self.assertTrue(resident.stopped.is_set())
                self.assertFalse(health["owned_worker_pending"])
                for _ in range(6):
                    health = await poll(client)
                    self.assertEqual(health["state"], "asleep")
                    self.assertFalse(health["owned_worker_pending"])
                    await asyncio.sleep(0.02)
                self.assertEqual((len(h.workers), len(h.posts), h.preparations), (1, 1, 1))

        self.assertTrue(app.state.worker.closing)
        self.assertEqual(h.backend.health()["phase"], "closed")
        self.assertEqual(h.backend.health()["state"], "offline")
        self.assertTrue(resident.closed)
        self.assertTrue(all(client.is_closed for client in h.clients))
