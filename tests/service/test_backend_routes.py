"""Base-URL selection preserves the ordinary Chandra CLI's request contract."""
import unittest

import httpx

from chandra_service.router import create_router


class BackendRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_base_url_and_passive_models(self):
        calls = []
        def upstream(request):
            calls.append((request.method, request.url.host, request.url.path))
            if request.url.path == "/health":
                return httpx.Response(200, json={"state": "cold", "pending_or_active": 0})
            return httpx.Response(200, json={"choices": []})
        config = {"backends": {"northstone": {"url": "http://127.0.0.1"}, "waystone": {"url": "http://127.0.0.2"}}}
        payload = {"model": "chandra", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ocr_layout"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
        ]}]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as remote:
            app = create_router(config, remote)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                for backend in ("waystone", "northstone"):
                    self.assertEqual((await client.get(f"/backends/{backend}/v1/models")).status_code, 200)
                self.assertEqual(calls, [])
                response = await client.post("/backends/waystone/v1/chat/completions", json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["X-Chandra-Backend"], "waystone")
                self.assertEqual(calls, [("GET", "127.0.0.2", "/health"), ("POST", "127.0.0.2", "/v1/chat/completions")])
                calls.clear()
                response = await client.post("/backends/waystone/v1/chat/completions", json=payload, headers={"X-Chandra-Backend": "northstone"})
                self.assertEqual(response.status_code, 400)
                self.assertEqual((await client.post("/backends/unknown/v1/chat/completions", json=payload)).status_code, 404)
                self.assertEqual((await client.get("/backends/unknown/v1/models")).status_code, 404)
                self.assertEqual((await client.post("/backends/waystone/sleep", json={})).status_code, 404)
                self.assertEqual(calls, [])
