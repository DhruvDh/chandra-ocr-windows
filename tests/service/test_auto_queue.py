"""Bounded opt-in auto admission; event-controlled fake origins, no inference."""
import asyncio
import json
import unittest

import httpx
from starlette.requests import Request

from chandra_service.router import Endpoint, create_router
from tests.service.test_scheduling import request_payload


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.release = asyncio.Event()
        self.posted = asyncio.Event()
        self.posts = []
        self.occupancy = {'127.0.0.1': 2, '127.0.0.2': 0}
        self.accepting = True
        self.fail = False
        async def handle(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'state': 'awake', 'pending_or_active': self.occupancy[request.url.host], 'accepting': self.accepting})
            self.posts.append(request.url.host)
            self.posted.set()
            await self.release.wait()
            if self.fail:
                raise httpx.ReadError('ambiguous execution', request=request)
            return httpx.Response(200, json={'choices': []})
        self.upstream = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(self.upstream.aclose)
        self.clients = []
        self.running = []
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        self.release.set()
        if self.running:
            await asyncio.wait_for(asyncio.gather(*self.running, return_exceptions=True), 3)
        for client in self.clients:
            await client.aclose()

    def router(self, allowance=1, slow_capacity=1):
        app = create_router({'backends': {
            'northstone': {'url': 'http://127.0.0.1', 'capacity': 2, 'queue_limit': 1,
                          'auto_queue_limit': allowance, 'expected_page_seconds': 11},
            'waystone': {'url': 'http://127.0.0.2', 'capacity': slow_capacity, 'expected_page_seconds': 85},
        }}, self.upstream)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test')
        self.clients.append(client)
        return app, client

    def start(self, client, selection='auto'):
        task = asyncio.create_task(client.post('/v1/chat/completions', json=request_payload(), headers={'X-Chandra-Backend': selection}))
        self.running.append(task)
        return task

    async def test_default_zero_preserves_spill(self):
        app, client = self.router(allowance=0)
        task = self.start(client)
        await asyncio.wait_for(self.posted.wait(), 2)
        self.assertEqual(self.posts, ['127.0.0.2'])
        self.release.set()
        self.assertEqual((await asyncio.wait_for(task, 2)).status_code, 200)
        self.assertEqual(Endpoint('northstone', 'http://127.0.0.1').auto_queue_limit, 0)

    async def test_fast_queued_choice_and_health(self):
        app, client = self.router()
        task = self.start(client)
        await asyncio.wait_for(self.posted.wait(), 2)
        self.assertEqual(self.posts, ['127.0.0.1'])
        health = (await client.get('/health')).json()
        self.assertEqual(health['backends']['northstone']['auto_queue_limit'], 1)
        self.assertEqual(health['backends']['northstone']['capacity'], 2)
        self.release.set()
        self.assertEqual((await asyncio.wait_for(task, 2)).headers['X-Chandra-Backend'], 'northstone')

    async def test_external_occupancy_saturates_allowance(self):
        self.occupancy['127.0.0.1'] = 3
        _, client = self.router()
        task = self.start(client)
        await asyncio.wait_for(self.posted.wait(), 2)
        self.assertEqual(self.posts, ['127.0.0.2'])
        self.release.set()
        await asyncio.wait_for(task, 2)

    async def test_explicit_selection_uses_total_queue(self):
        _, client = self.router(allowance=0)
        task = self.start(client, 'northstone')
        await asyncio.wait_for(self.posted.wait(), 2)
        self.assertEqual(self.posts, ['127.0.0.1'])
        self.release.set()
        await asyncio.wait_for(task, 2)

    async def test_concurrent_local_reservations_bounded(self):
        # Health can lag every local submission: atomic local reservations still
        # bound accepted work to execution slots plus one waiting allowance.
        self.occupancy['127.0.0.1'] = 0
        self.occupancy['127.0.0.2'] = 1
        app, client = self.router()
        tasks = [self.start(client) for _ in range(5)]
        for _ in range(100):
            if len(self.posts) == 3 and sum(task.done() for task in tasks) == 2:
                break
            await asyncio.sleep(.01)
        self.assertEqual(self.posts, ['127.0.0.1'] * 3)
        self.assertEqual(app.state.endpoints[0].active, 3)
        self.release.set()
        responses = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertEqual(sorted(response.status_code for response in responses), [200, 200, 200, 503, 503])

    async def test_health_not_accepting_blocks_submission(self):
        self.accepting = False
        _, client = self.router()
        response = await client.post('/v1/chat/completions', json=request_payload())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.posts, [])

    async def test_cancel_retains_reservation_and_drains(self):
        app, _ = self.router()
        disconnected = asyncio.Event()
        body_sent = False
        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {'type': 'http.request', 'body': json.dumps(request_payload()).encode(), 'more_body': False}
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
                           'headers': [], 'query_string': b''}, receive)
        endpoint = next(route.endpoint for route in app.routes
                        if route.path == '/v1/chat/completions')
        task = asyncio.create_task(endpoint(request))
        self.running.append(task)
        await asyncio.wait_for(self.posted.wait(), 2)
        disconnected.set()
        response = await asyncio.wait_for(task, 2)
        self.assertEqual(response.status_code, 499)
        self.assertEqual(app.state.endpoints[0].active, 1)
        self.release.set()
        async def drained():
            while app.state.endpoints[0].active:
                await asyncio.sleep(.01)
        await asyncio.wait_for(drained(), 2)
        self.assertEqual(self.posts, ['127.0.0.1'])

    async def test_transport_failure_does_not_replay(self):
        self.fail = True
        self.release.set()
        app, client = self.router()
        response = await client.post('/v1/chat/completions', json=request_payload())
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.posts, ['127.0.0.1'])
        self.assertTrue(app.state.endpoints[0].uncertain)

    def test_invalid_allowances(self):
        for value in (-1, 2, True, 1.5, None, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Endpoint('northstone', 'http://127.0.0.1', queue_limit=1, auto_queue_limit=value)

    def test_existing_positional_cost_is_preserved(self):
        endpoint = Endpoint('northstone', 'http://127.0.0.1', 2, 1, 11)
        self.assertEqual(endpoint.expected_page_seconds, 11)
        self.assertEqual(endpoint.auto_queue_limit, 0)


if __name__ == '__main__':
    unittest.main()
