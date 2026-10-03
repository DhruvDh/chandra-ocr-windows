import asyncio
import threading
import unittest

import httpx
from chandra_service.service import Worker, create_app
from chandra_service.router import create_router, Endpoint
from chandra_service.validation import validate


def payload(stream=False):
    return {"model": "chandra", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}, {"type": "text", "text": "ocr_layout"}]}], "temperature": 0, "top_p": .1, "max_tokens": 12384, "stream": stream}

class Fake:
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.stop = threading.Event()
        self.block = False
        self.crash = False
    def health(self):
        return {"state": "cold"}
    def generate(self, request, cancel):
        self.calls += 1
        self.started.set()
        if self.block:
            self.stop.wait(3)
        if self.crash:
            raise RuntimeError("private input and secret path")
        yield "hello"
        yield {"usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}, "finish_reason": "stop"}

class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_models_do_not_load(self):
        backend = Fake()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(backend)), base_url="http://test") as client:
            self.assertEqual((await client.get('/health')).json()['state'], 'cold')
            await client.get('/v1/models')
            self.assertEqual(backend.calls, 0)
            self.assertEqual((await client.post('/wake_up')).status_code, 404)
            result = await client.post('/v1/chat/completions', json=payload())
            self.assertEqual(result.json()['choices'][0]['message']['content'], 'hello')
            self.assertEqual(result.json()['usage']['total_tokens'], 3)

    async def test_cancel_keeps_worker_until_actual_stop(self):
        backend = Fake(); backend.block = True
        worker = Worker(backend, 0)
        job = worker.submit(payload())
        await asyncio.to_thread(backend.started.wait, 1)
        job.cancel.set()
        self.assertIsNone(worker.submit(payload()))
        self.assertFalse(job.done.is_set())
        backend.stop.set()
        await asyncio.gather(*worker.jobs)
        self.assertTrue(job.done.is_set())
        self.assertIsNotNone(worker.submit(payload()))
        await worker.close()

    async def test_bounded_queue_one_inference_and_crash_recovery(self):
        backend = Fake(); backend.block = True
        worker = Worker(backend, 1)
        first = worker.submit(payload()); second = worker.submit(payload())
        await asyncio.to_thread(backend.started.wait, 1)
        self.assertEqual(backend.calls, 1)
        self.assertIsNone(worker.submit(payload()))
        second.cancel.set(); backend.stop.set()
        await asyncio.gather(*worker.jobs)
        self.assertEqual(backend.calls, 1)
        backend.crash = True
        job = worker.submit(payload()); await asyncio.gather(*worker.jobs)
        self.assertTrue(job.failure)
        self.assertNotIn('secret', worker.last_error)
        backend.crash = False
        job = worker.submit(payload()); await asyncio.gather(*worker.jobs)
        self.assertFalse(job.failure)

    async def test_deadline_retains_capacity_and_reports_timeout(self):
        backend = Fake(); backend.block = True
        app = create_app(backend, queue_limit=0, inference_seconds=.05)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json=payload())
            self.assertEqual(response.status_code, 504)
            self.assertEqual((await client.get('/health')).json()['pending_or_active'], 1)
            self.assertEqual((await client.post('/v1/chat/completions', json=payload())).status_code, 503)
            backend.stop.set()
            await asyncio.gather(*app.state.worker.jobs)

    async def test_missing_and_invalid_backend_metadata_fail_closed(self):
        class Incomplete(Fake):
            def __init__(self, metadata): super().__init__(); self.metadata = metadata
            def generate(self, request, cancel):
                self.calls += 1
                yield "plausible OCR"
                if self.metadata is not None: yield self.metadata
        bad = [None, {}, {"finish_reason":"stop"},
               {"finish_reason":"stop","usage":{"prompt_tokens":2,"completion_tokens":True,"total_tokens":3}},
               {"finish_reason":"stop","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":4}},
               {"finish_reason":"length","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}},
               {"finish_reason":"stop","usage":{"prompt_tokens":2,"completion_tokens":0,"total_tokens":2}}]
        for metadata in bad:
            for streaming in (False, True):
                backend = Incomplete(metadata)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(backend)),base_url='http://test') as client:
                    response = await client.post('/v1/chat/completions',json=payload(streaming))
                    if streaming:
                        self.assertIn('backend_failure',response.text)
                        self.assertNotIn('[DONE]',response.text)
                        self.assertNotIn('"finish_reason":"stop"',response.text)
                    else: self.assertEqual(response.status_code,502)
                    self.assertEqual(backend.calls,1)

    async def test_metadata_must_be_final_and_generation_value_error_is_not_bad_request(self):
        class BadOrder(Fake):
            def generate(self, request, cancel):
                yield "hello"
                yield {"finish_reason":"stop","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}
                yield "late"
        class MidDecodeFailure(Fake):
            def generate(self, request, cancel):
                yield "partial"
                raise ValueError('internal generation failure')
        for backend in (BadOrder(),MidDecodeFailure()):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(backend)),base_url='http://test') as client:
                response=await client.post('/v1/chat/completions',json=payload())
                self.assertEqual(response.status_code,502)

    async def test_stream_usage(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(Fake())), base_url='http://test') as client:
            data = payload(True); data['stream_options'] = {'include_usage': True}
            response = await client.post('/v1/chat/completions', json=data)
            self.assertIn('"content":"hello"', response.text)
            self.assertIn('"usage":', response.text)
            self.assertTrue(response.text.endswith('data: [DONE]\n\n'))

class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_and_no_replay(self):
        posts = []
        def transport(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'state': 'cold', 'pending_or_active': 0})
            posts.append(request.url.host)
            if request.url.host == '127.0.0.2':
                raise httpx.ReadError('crashed', request=request)
            return httpx.Response(200, json={'choices': []})
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        app = create_router({'backends': {'northstone': {'url': 'http://127.0.0.1'}, 'waystone': {'url': 'http://127.0.0.2'}}}, upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            await client.get('/health'); await client.get('/v1/models')
            self.assertEqual(posts, [])
            r = await client.post('/v1/chat/completions', json=payload(), headers={'X-Chandra-Backend': 'waystone'})
            self.assertEqual(r.status_code, 502)
            self.assertEqual(posts, ['127.0.0.2'])
            r = await client.post('/v1/chat/completions', json=payload(), headers={'X-Chandra-Backend': 'waystone'})
            self.assertEqual(r.status_code, 503)
            r = await client.post('/v1/chat/completions', json=payload())
            self.assertEqual(r.headers['X-Chandra-Backend'], 'northstone')
            self.assertEqual(posts, ['127.0.0.2', '127.0.0.1'])
            self.assertEqual((await client.post('/sleep')).status_code, 404)
        await upstream.aclose()

    async def test_cancel_drain_holds_capacity(self):
        started = asyncio.Event()
        finish = asyncio.Event()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: first\n\n'
                started.set()
                await finish.wait()
                yield b'data: [DONE]\n\n'
        def transport(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'state': 'cold', 'pending_or_active': 0})
            return httpx.Response(200, stream=Stream(), headers={'content-type': 'text/event-stream'})
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        app = create_router({'backends': {'waystone': {'url': 'http://127.0.0.2'}}}, upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            task = asyncio.create_task(client.post('/v1/chat/completions', json=payload(True)))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(.05)
            self.assertEqual(app.state.endpoints[0].active, 1)
            response = await client.post('/v1/chat/completions', json=payload())
            self.assertEqual(response.status_code, 503)
            finish.set()
            for _ in range(100):
                if app.state.endpoints[0].active == 0: break
                await asyncio.sleep(.01)
            self.assertEqual(app.state.endpoints[0].active, 0)
            try:
                await task
            except asyncio.CancelledError:
                pass
        await upstream.aclose()

    async def test_unexpected_upstream_failure_is_not_partial_success(self):
        posts=[]
        class Broken(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"partial":'
                raise RuntimeError('unexpected decoder failure')
        def transport(request):
            if request.url.path=='/health': return httpx.Response(200,json={'state':'cold','pending_or_active':0})
            posts.append(request.url.host)
            return httpx.Response(200,stream=Broken())
        upstream=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        app=create_router({'backends':{'waystone':{'url':'http://127.0.0.2'}}},upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.post('/v1/chat/completions',json=payload())
            self.assertEqual(response.status_code,502)
            self.assertEqual(posts,['127.0.0.2'])
            self.assertTrue(app.state.endpoints[0].uncertain)
        await upstream.aclose()

    async def test_capacity_selection(self):
        def transport(request):
            if request.url.path == '/health':
                return httpx.Response(200, json={'state': 'awake', 'pending_or_active': 1 if request.url.host == '127.0.0.1' else 0})
            return httpx.Response(200, json={})
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        app = create_router({'backends': {'northstone': {'url': 'http://127.0.0.1'}, 'waystone': {'url': 'http://127.0.0.2'}}}, upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            r = await client.post('/v1/chat/completions', json=payload())
            self.assertEqual(r.headers['X-Chandra-Backend'], 'waystone')
            r = await client.post('/v1/chat/completions', json=payload(), headers={'X-Chandra-Backend': 'northstone'})
            self.assertEqual(r.status_code, 503)
        await upstream.aclose()

class ValidationTests(unittest.TestCase):
    def test_reject_unsupported_and_remote(self):
        for change in [{'n': 2}, {'temperature': .1}, {'max_tokens': 12385}, {'top_p': 0}, {'stream_options': {'other': True}}]:
            with self.assertRaises(ValueError):
                validate({**payload(), **change})
        data = payload(); data['messages'][0]['content'][0]['image_url']['url'] = 'https://example.com/image.png'
        with self.assertRaises(ValueError): validate(data)
        with self.assertRaises(ValueError): Endpoint('waystone', 'http://8.8.8.8')

if __name__ == '__main__': unittest.main()
