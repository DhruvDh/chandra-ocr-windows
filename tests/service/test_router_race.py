"""A stale health response cannot clear transport-failure quarantine."""
import asyncio
import unittest

import httpx

from chandra_service.router import Endpoint, create_router


def payload():
    return {'model': 'chandra', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'ocr_layout'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}},
    ]}]}


class RouterRaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_idle_probe_cannot_clear_new_failure(self):
        pending_health = asyncio.Event()
        release_health = asyncio.Event()
        posts = 0
        health_calls = 0
        async def handle(request):
            nonlocal posts, health_calls
            if request.url.path == '/health':
                health_calls += 1
                if health_calls == 1:
                    pending_health.set()
                    await release_health.wait()
                return httpx.Response(200, json={'state': 'awake', 'pending_or_active': 0})
            posts += 1
            if posts == 1:
                raise httpx.ReadError('ambiguous execution', request=request)
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = create_router({'backends': {'waystone': {'url': 'http://127.0.0.1'}}, 'cooldown_seconds': .1}, upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                old_probe = asyncio.create_task(client.get('/health'))
                await pending_health.wait()
                response = await client.post('/v1/chat/completions', json=payload())
                self.assertEqual(response.status_code, 502)
                self.assertTrue(app.state.endpoints[0].uncertain)
                release_health.set()
                await old_probe
                self.assertTrue(app.state.endpoints[0].uncertain)
                self.assertEqual(app.state.endpoints[0].state, 'degraded')
                response = await client.post('/v1/chat/completions', json=payload())
                self.assertEqual(response.status_code, 503)
                self.assertEqual(posts, 1)
                self.assertEqual(health_calls, 2)
                await asyncio.sleep(.11)
                response = await client.post('/v1/chat/completions', json=payload())
                self.assertEqual(response.status_code, 200)
                self.assertEqual(posts, 2)
                self.assertEqual(health_calls, 3)
                self.assertFalse(app.state.endpoints[0].uncertain)

    def test_invalid_time_bounds_and_ports_rejected_at_construction(self):
        config = {'backends': {'waystone': {'url': 'http://127.0.0.1'}}}
        for field in ['cooldown_seconds', 'inference_seconds']:
            for value in [float('nan'), float('inf'), float('-inf'), -1, True, None, '1']:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    create_router({**config, field: value})
        with self.assertRaises(ValueError):
            create_router({**config, 'inference_seconds': 0})
        for port in ['0', '65536', '-1', 'nope', '']:
            with self.subTest(port=port), self.assertRaises(ValueError):
                Endpoint('waystone', 'http://127.0.0.1:' + port)


if __name__ == '__main__':
    unittest.main()
