import argparse
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import httpx
from live_integration import Harness, execute, occupancy, sha

CORPUS=Path(__file__).with_name('inputs-v2')


def content(fixture):
    expected=fixture['expected']
    text=' '.join(expected['ordered_text'])
    table='<table>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in expected['table'])+'</table>'
    math=''.join('<math>'+v+'</math>' for v in expected.get('equations',[]))
    return '<div data-label="Table" data-bbox="0 0 1000 1000">'+text+table+math+'</div>'

class Stream(httpx.AsyncByteStream):
    def __init__(self, raw, release=None):self.raw=raw;self.release=release;self.closed=False
    async def __aiter__(self):
        boundary=b'\r\n\r\n' if b'\r\n\r\n' in self.raw else b'\n\n'
        yield self.raw.split(boundary,1)[0]+boundary
        if self.release:await self.release.wait()
        yield self.raw.split(boundary,1)[1]
    async def aclose(self):self.closed=True

class LiveTests(unittest.IsolatedAsyncioTestCase):
    def fixtures(self):return json.loads((CORPUS/'manifest.json').read_text())['fixtures']
    def fixture_for_request(self, request):
        data=json.loads(request.content);image=base64.b64decode(data['messages'][0]['content'][0]['image_url']['url'].split(',')[1]);return next(f for f in self.fixtures() if f['image_sha256']==sha(image))
    def response(self, fixture):return dict(choices=[dict(message=dict(content=content(fixture)),finish_reason='stop')],usage=dict(prompt_tokens=801,completion_tokens=397,total_tokens=1198))
    def stream(self, fixture, release=None):
        events=[dict(choices=[dict(index=0,delta=dict(content=content(fixture)),finish_reason=None)]),dict(choices=[dict(index=0,delta={},finish_reason='stop')]),dict(choices=[],usage=dict(prompt_tokens=801,completion_tokens=397,total_tokens=1198))]
        raw=b''.join(b'data: '+json.dumps(e,separators=(',',':')).encode()+b'\r\n\r\n' for e in events)+b'data: [DONE]\r\n\r\n'
        # Preserve CRLF bytes while defining first-event boundary for controlled streaming.
        return raw,Stream(raw,release)
    async def test_five_warm_repeats_are_free_generation_with_metadata(self):
        posts=[]
        async def handler(request):
            if request.method=='GET':return httpx.Response(200,json=dict(pending_or_active=0) if request.url.path=='/health' else dict(data=[dict(id='chandra')]))
            posts.append(request);return httpx.Response(200,json=self.response(self.fixture_for_request(request)))
        factory=httpx.AsyncClient
        with tempfile.TemporaryDirectory() as tmp:
            provenance=Path(tmp)/'provenance.json';provenance.write_text('{}')
            args=argparse.Namespace(output=Path(tmp)/'run',corpus=CORPUS,cli_corpus=Path(__file__).with_name('cli-inputs-v1'),provenance=provenance,base_url='http://mock.invalid/v1',cli_base_url=None,backend='auto',request_seconds=2,drain_seconds=1,budget_seconds=10,stages=['warm'],warm_fixture='tiny',repeat=5)
            with patch('live_integration.httpx.AsyncClient',side_effect=lambda **kw:factory(transport=httpx.MockTransport(handler),**kw)):
                report=await execute(args)
            self.assertTrue(report['passed']);self.assertEqual(len(posts),6);self.assertEqual(report['stages']['warm']['n'],5);self.assertFalse(report['stages']['warm']['performance_promotion'])
    async def test_concurrent_capacity_rejection_is_not_replayed(self):
        release=asyncio.Event();posts=[]
        async def handler(request):
            posts.append(request)
            if len(posts)==1:
                raw,stream=self.stream(self.fixture_for_request(request),release)
                return httpx.Response(200,stream=stream)
            release.set();return httpx.Response(503,json={'error':{'type':'capacity'}})
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                harness=Harness(client,'http://mock.invalid/v1',CORPUS,Path(tmp),request_seconds=2,drain_seconds=1)
                harness.health=AsyncMock(return_value={'pending_or_active':1});harness.idle=AsyncMock(return_value={'passed':True})
                report=await harness.concurrent()
                self.assertTrue(report['passed']);self.assertEqual(len(posts),2);self.assertTrue(report['contender_rejected_at_capacity']);self.assertEqual(report['contender']['fixture'],'small')
    async def test_stream_cancel_retains_partial_and_requires_drain_before_recovery(self):
        streams=[];posts=[]
        async def handler(request):
            posts.append(request)
            fixture=self.fixture_for_request(request)
            if len(posts)==1:
                raw,stream=self.stream(fixture);streams.append(stream);return httpx.Response(200,stream=stream)
            return httpx.Response(200,json=self.response(fixture))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                h=Harness(client,'http://mock.invalid/v1',CORPUS,Path(tmp),request_seconds=2,drain_seconds=1);h.health=AsyncMock(return_value={'pending_or_active':0});h.idle=AsyncMock(return_value={'passed':True})
                report=await h.cancellation();self.assertTrue(report['passed']);self.assertEqual(len(posts),2);self.assertTrue(streams[0].closed);self.assertFalse(report['cancelled_request']['correctness']['passed']);self.assertTrue(report['recovery']['correctness']['passed'])
    async def test_unknown_occupancy_cannot_pass(self):
        self.assertIsNone(occupancy({'state':'ready'},'auto'))
        self.assertEqual(occupancy({'backends':{'waystone':{'active':1,'remote_active':1}}},'waystone'),1)

if __name__=='__main__':unittest.main()
