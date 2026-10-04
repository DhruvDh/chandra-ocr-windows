"""Fake HTTP and synthetic bytes only; no renderer, model or external network."""
import asyncio
import base64
import html
import json
import io
from PIL import Image
from pathlib import Path
import unittest
import httpx
try:
    from . import throughput_first_seen as first
    from . import throughput_live as live
    from . import throughput as core
    from . import test_throughput_live as fixtures
except ImportError:
    import throughput_first_seen as first
    import throughput_live as live
    import throughput as core
    import test_throughput_live as fixtures

GENERATOR = '''def plan(seed,count):
    pages=[]
    for i in range(count):
        title=f"Report {seed}-{i}";middle=f"Measured {i+7} samples at station {seed}-{i}.";end=f"End {seed}-{i}."
        table=[["Site","Count"],[f"S{i}",str(i+7)]]
        blocks=[title,middle," ".join(c for row in table for c in row),end]
        pages.append({"family":("prose","columns","equations","tables","long")[i%5],"width":16,"height":16,"id":f"page-{i}","content_id":f"content-{seed}-{i}","blocks":blocks,
            "expected":{"equations":[],"table":table,"numbers":[],"text":[title],"end_markers":[end],"ordered_text":[title,end],"required_labels":["Table"]},
            "independent_expected":{"ordered_blocks":blocks,"table":table,"end_marker":end,"geometry_reference":[]}})
    return pages
'''

class FirstSeenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.base=fixtures.LiveTests();self.base.setUp();self.root=self.base.root
        self.generator=self.root/'generator.py';self.generator.write_text(GENERATOR)
        namespace={};exec(GENERATOR,namespace);self.plan=namespace['plan'];self.by_image={}
        self.measured=self.make_corpus('measured',10);self.warm=self.make_corpus('warm',5)
        self.adapter=first.FirstSeen(self.warm,self.generator)
        self.base.admission['expected_worker']['starttime']='fixed-starttime'
        self.base.write_witness()
        a=self.base.admission;a.update({'repeat':1,'selected_pages':10,'work_scope':'pilot','adapter_identity':self.adapter.identity,
            'adapter_source_sha256':{n:core.digest((self.adapter.root/n).read_bytes()) for n in self.adapter.source_files},
            'first_seen_root_attestation':{'fresh_identified_worker':True,'no_prior_measured_corpus_use':True,'worker_identity':a['expected_worker']},
            'warmup_corpus':str(self.warm),'generator_path':str(self.generator),'generator_sha256':core.digest(self.generator.read_bytes()),
            'measured_manifest_sha256':core.digest((self.measured/'manifest.json').read_bytes()),
            'warmup_manifest_sha256':core.digest((self.warm/'manifest.json').read_bytes())})
        work,metadata=self.adapter.load(self.measured,a);a['planned_pages']=len(work);a['corpus_metadata']=metadata
    def tearDown(self):self.base.tearDown()
    def make_corpus(self,seed,count):
        path=self.root/seed;path.mkdir();fixtures=self.plan(seed,count)
        for i,f in enumerate(fixtures):
            color=tuple(bytes.fromhex(core.digest((seed+'-'+str(i)).encode()))[:3])
            image_object=Image.new('RGB',(16,16),color);buffer=io.BytesIO();image_object.save(buffer,format='PNG');image=buffer.getvalue();pixels=core.digest(image_object.tobytes());image_object.close()
            f.update({'image':str(i)+'.png','image_sha256':core.digest(image),'pixel_sha256':pixels})
            (path/f['image']).write_bytes(image);self.by_image[image]=f
            self.by_image[image]['_test_seed']=seed
        (path/'manifest.json').write_text(json.dumps({'seed':seed,'generator_sha256':core.digest(self.generator.read_bytes()),'anchor_sha256':{},'fixtures':fixtures}))
        return path
    def response(self,f,omit=False):
        blocks=f['blocks'];parts=[]
        for i,b in enumerate(blocks):
            if omit and i==1:continue
            if i==2:
                content='<table>'+''.join('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row)+'</tr>' for row in f['expected']['table'])+'</table>';label='Table'
            else:content=html.escape(b);label='Text'
            parts.append('<div data-label="'+label+'" data-bbox="0 0 900 900">'+content+'</div>')
        return json.dumps({'choices':[{'message':{'content':''.join(parts)},'finish_reason':'stop'}],'usage':{'prompt_tokens':3629,'completion_tokens':700,'total_tokens':4329}}).encode()
    async def invoke(self,handler):
        return await first.run(self.base.admission,'http://node.invalid',self.measured,self.base.monitor,self.root/'output',
                              warmup_corpus=self.warm,generator=self.generator,execute=True,http_transport=httpx.MockTransport(handler))
    async def test_separate_warmup_full_work_rolling_and_raw_retention(self):
        posted=[];active=set();gate=asyncio.Event();peak=0
        async def handler(request):
            nonlocal peak
            if request.method=='GET':return httpx.Response(200,stream=fixtures.Stream(json.dumps(self.base.health).encode()))
            payload=json.loads(request.content);image=base64.b64decode(payload['messages'][0]['content'][0]['image_url']['url'].split(',')[1]);posted.append(image)
            identifier=request.headers['X-Request-ID'];active.add(identifier);peak=max(peak,len(active));self.base.health['pending_or_active']=len(active)
            if identifier.endswith('measured-00002'):
                self.assertIn('owned-test-c0-measured-00000',active);gate.set()
            def close():active.remove(identifier);self.base.health['pending_or_active']=len(active)
            return httpx.Response(200,stream=fixtures.Stream(self.response(self.by_image[image]),gate=gate if identifier.endswith('measured-00000') else None,close=close))
        report=await self.invoke(handler)
        self.assertTrue(report['valid']);self.assertLessEqual(peak,2)
        self.assertEqual(set(posted[:5]),{image for image,f in self.by_image.items() if f['_test_seed']=='warm'})
        self.assertEqual(set(posted[5:]),{image for image,f in self.by_image.items() if f['_test_seed']=='measured'})
        self.assertEqual(len(posted),15);self.assertEqual(len(set(posted)),15)
        self.assertEqual(report['configurations'][0]['measured']['result']['completed_correct_pages'],10)
        self.assertEqual(len(list((self.root/'output').glob('*.request.json'))),15)
        self.assertEqual(len(list((self.root/'output').glob('*.response.raw'))),15)
    async def test_c16_rolling_refill_and_complete_accounting(self):
        self.measured=self.make_corpus('measured-c16',35)
        a=self.base.admission;a['concurrency']=[16];a['work_limit']=64;a['selected_pages']=30
        a['measured_manifest_sha256']=core.digest((self.measured/'manifest.json').read_bytes())
        work,metadata=self.adapter.load(self.measured,a);a['planned_pages']=30;a['corpus_metadata']=metadata
        active=set();gate=asyncio.Event();peak=0
        async def handler(request):
            nonlocal peak
            if request.method=='GET':return httpx.Response(200,stream=fixtures.Stream(json.dumps(self.base.health).encode()))
            image=base64.b64decode(json.loads(request.content)['messages'][0]['content'][0]['image_url']['url'].split(',')[1])
            identifier=request.headers['X-Request-ID'];active.add(identifier);peak=max(peak,len(active));self.base.health['pending_or_active']=len(active)
            if identifier.endswith('measured-00016'):
                self.assertIn('owned-test-c0-measured-00000',active);gate.set()
            def close():active.remove(identifier);self.base.health['pending_or_active']=len(active)
            return httpx.Response(200,stream=fixtures.Stream(self.response(self.by_image[image]),gate=gate if identifier.endswith('measured-00000') else None,close=close))
        report=await self.invoke(handler);self.assertTrue(report['valid']);self.assertEqual(peak,16)
        self.assertEqual(report['configurations'][0]['measured']['result']['completed_correct_pages'],30)
        self.assertEqual(len(list((self.root/'output').glob('*.response.raw'))),35)

    async def test_omitted_middle_invalidates_and_stops_new_admissions(self):
        self.base.admission['concurrency']=[1];posts=[]
        _,metadata=self.adapter.load(self.measured,self.base.admission)
        self.base.admission['corpus_metadata']=metadata
        async def handler(request):
            if request.method=='GET':return httpx.Response(200,stream=fixtures.Stream(json.dumps(self.base.health).encode()))
            image=base64.b64decode(json.loads(request.content)['messages'][0]['content'][0]['image_url']['url'].split(',')[1]);posts.append(image)
            return httpx.Response(200,stream=fixtures.Stream(self.response(self.by_image[image],omit=self.by_image[image]['_test_seed']=='measured')))
        report=await self.invoke(handler);self.assertFalse(report['valid']);self.assertEqual(len(posts),6);self.assertEqual(report['unattempted_concurrency'],[])
        receipt=json.loads(next((self.root/'output').glob('*measured*.receipt.json')).read_text())
        self.assertTrue(receipt['correctness']['endpoint']['passed']);self.assertFalse(receipt['correctness']['complete']['passed'])
    def test_duplicate_actual_pixels_with_fabricated_hash_rejected(self):
        a=self.base.admission;manifest=json.loads((self.measured/'manifest.json').read_text())
        source=manifest['fixtures'][0];target=manifest['fixtures'][1]
        (self.measured/target['image']).write_bytes((self.measured/source['image']).read_bytes())
        target['image_sha256']=source['image_sha256'];target['pixel_sha256']='f'*64
        (self.measured/'manifest.json').write_text(json.dumps(manifest))
        a['measured_manifest_sha256']=core.digest((self.measured/'manifest.json').read_bytes())
        with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)

    def test_declared_balanced_prefix_fixed_before_admission(self):
        a=self.base.admission;a['selected_pages']=5
        work,metadata=self.adapter.load(self.measured,a)
        self.assertEqual(len(work),5);self.assertEqual(metadata['available_pages'],10)
        self.assertEqual(metadata['fixture_order'],['measured:page-'+str(i) for i in range(5)])
        self.assertEqual(metadata['work_scope'],'pilot')
        for count in (0,6,15,True):
            a['selected_pages']=count
            with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)

    def test_bindings_overlap_and_caps_fail_before_http(self):
        a=self.base.admission
        a['concurrency']=[1,2]
        with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)
        a['concurrency']=[2]
        a['first_seen_root_attestation']['no_prior_measured_corpus_use']=False
        with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)
        a['first_seen_root_attestation']['no_prior_measured_corpus_use']=True
        a['response_bytes']=1024**2+1
        with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)
        a['response_bytes']=1024
        original=a['warmup_manifest_sha256'];a['warmup_manifest_sha256']='0'*64
        with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)
        a['warmup_manifest_sha256']=original
        raw=json.loads((self.warm/'manifest.json').read_text());raw['fixtures'][0]['pixel_sha256']=a['corpus_metadata']['measured_pixel_sha256'][0]
        (self.warm/'manifest.json').write_text(json.dumps(raw));a['warmup_manifest_sha256']=core.digest((self.warm/'manifest.json').read_bytes())
        with self.assertRaises(live.AdmissionError):self.adapter.load(self.measured,a)

if __name__=='__main__':unittest.main()
