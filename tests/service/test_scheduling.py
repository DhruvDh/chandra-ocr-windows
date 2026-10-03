"""Static asymmetric routing costs use mocked endpoints; no model or live calls."""
import unittest

import httpx

from chandra_service.router import Endpoint, create_router


def request_payload():
    return {"model": "chandra", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "ocr_layout"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
    ]}]}


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def route(self, costs, requests=1, occupancy=None, selection=None):
        occupancy = occupancy or {}
        def handle(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={"state": "awake", "pending_or_active": occupancy.get(request.url.host, 0)})
            return httpx.Response(200, json={"choices": []})
        config = {"backends": {
            "northstone": {"url": "http://127.0.0.1", "expected_page_seconds": costs[0]},
            "waystone": {"url": "http://127.0.0.2", "expected_page_seconds": costs[1]},
        }}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = create_router(config, upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                health = (await client.get('/health')).json()
                results = []
                for _ in range(requests):
                    response = await client.post('/v1/chat/completions', json=request_payload(), headers={"X-Chandra-Backend": selection or "auto"})
                    self.assertEqual(response.status_code, 200)
                    results.append(response.headers['X-Chandra-Backend'])
                return results, health

    async def test_unequal_costs_prefer_fast_endpoint(self):
        results, health = await self.route((7.6, 27.6), requests=3)
        self.assertEqual(results, ['northstone'] * 3)
        self.assertEqual(health['backends']['northstone']['expected_page_seconds'], 7.6)
        self.assertEqual(health['backends']['waystone']['expected_page_seconds'], 27.6)

    async def test_fast_full_uses_other_endpoint(self):
        results, _ = await self.route((7.6, 27.6), occupancy={'127.0.0.1': 1})
        self.assertEqual(results, ['waystone'])

    async def test_equal_costs_rotate(self):
        results, _ = await self.route((1, 1), requests=4)
        self.assertEqual(results, ['northstone', 'waystone', 'northstone', 'waystone'])

    async def test_explicit_choice_overrides_cost(self):
        results, _ = await self.route((7.6, 27.6), selection='waystone')
        self.assertEqual(results, ['waystone'])

    def test_cost_validation_and_compatible_default(self):
        self.assertEqual(Endpoint('northstone', 'http://127.0.0.1').expected_page_seconds, 1)
        for cost in [0, -1, float('nan'), float('inf'), float('-inf'), True, '1', None]:
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                Endpoint('northstone', 'http://127.0.0.1', expected_page_seconds=cost)


if __name__ == '__main__':
    unittest.main()
