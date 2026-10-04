"""Explicit first-seen corpus adapter; no model import, automatic build or live default."""
import argparse
import asyncio
import base64
import importlib.util
import json
import io
from pathlib import Path
try:
    from . import throughput as core
    from . import throughput_live as live
except ImportError:
    import throughput as core
    import throughput_live as live


class FirstSeen:
    identity = 'dense-first-seen-complete-content-v1'
    source_files = ('benchmarks/throughput_first_seen.py', 'benchmarks/qualification-v1/validate.py')

    def __init__(self, warmup_corpus, generator):
        self.warmup_corpus = Path(warmup_corpus).resolve()
        self.generator = Path(generator).resolve()
        self._warmup = (); self._expected = {}
        self.root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('first_seen_complete', self.root/'benchmarks/qualification-v1/validate.py')
        self.complete = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.complete)

    def _corpus(self, path, manifest_sha, generator, prefix, prompt, selected_count=None):
        path = Path(path).resolve()
        with (path/'manifest.json').open('rb') as stream: raw = stream.read(8*1024**2+1)
        if len(raw)>8*1024**2 or core.digest(raw)!=manifest_sha:
            raise live.AdmissionError('Bounded render manifest differs')
        manifest = json.loads(raw); fixtures=manifest['fixtures']
        if not 5<=len(fixtures)<=150 or len(fixtures)%5 or len({f['id'] for f in fixtures})!=len(fixtures):
            raise live.AdmissionError('Unique bounded fixture IDs required')
        if manifest['generator_sha256'] != core.digest(self.generator.read_bytes()):
            raise live.AdmissionError('Render generator commitment differs')
        for name,digest in manifest['anchor_sha256'].items():
            source=(self.root/name).resolve()
            if not source.is_relative_to(self.root) or core.digest(source.read_bytes())!=digest:
                raise live.AdmissionError('Corpus source anchor differs')
        selected_count = len(fixtures) if selected_count is None else selected_count
        if type(selected_count) is not int or not 5 <= selected_count <= len(fixtures) or selected_count % 5:
            raise live.AdmissionError('Declared balanced prefix must be a positive multiple of five within corpus')
        families = ('prose', 'columns', 'equations', 'tables', 'long')
        if any(sum(f.get('family') == family for f in fixtures[:selected_count]) != selected_count//5 for family in families):
            raise live.AdmissionError('Declared prefix is not balanced across five families')
        planned=generator.plan(manifest['seed'],len(fixtures))
        if not isinstance(planned, list) or len(planned) != len(fixtures):
            raise live.AdmissionError('Generator must return one oracle per committed fixture')
        work=[]; pixels=set(); total=0
        for index,(f,expected) in enumerate(zip(fixtures,planned)):
            if not isinstance(expected, dict) or not isinstance(f.get('expected'), dict) or not isinstance(f.get('independent_expected'), dict):
                raise live.AdmissionError('Fresh fixture/oracle objects required')
            required = ('table', 'numbers', 'text', 'end_markers', 'ordered_text', 'required_labels', 'equations')
            independent = f['independent_expected']
            if (any(not isinstance(f['expected'].get(key), list) for key in required)
                    or not isinstance(independent.get('ordered_blocks'), list)
                    or not 1 <= len(independent['ordered_blocks']) <= 256
                    or any(not isinstance(block, str) for block in independent['ordered_blocks'])
                    or not isinstance(independent.get('table'), list)
                    or not isinstance(independent.get('end_marker'), str)
                    or not isinstance(independent.get('geometry_reference'), list)):
                raise live.AdmissionError('Complete bounded endpoint/independent oracle schema required')
            for key,value in expected.items():
                observed=f.get(key)
                if key=='independent_expected':
                    observed=dict(observed);observed['geometry_reference']=[]
                if observed!=value: raise live.AdmissionError('Frozen content oracle differs')
            if index >= selected_count: continue
            if f['pixel_sha256'] in pixels: raise live.AdmissionError('Duplicate corpus pixels')
            pixels.add(f['pixel_sha256'])
            image_path=(path/f['image']).resolve()
            if not image_path.is_relative_to(path):raise live.AdmissionError('Image path escapes corpus')
            with image_path.open('rb') as stream:image=stream.read(16*1024**2+1)
            if len(image)>16*1024**2 or core.digest(image)!=f['image_sha256']:
                raise live.AdmissionError('Image bytes differ or exceed bound')
            from PIL import Image
            with Image.open(io.BytesIO(image)) as decoded:
                if (decoded.format != 'PNG' or decoded.mode != 'RGB'
                        or decoded.size != (f['width'], f['height'])
                        or decoded.width * decoded.height > 3145728
                        or core.digest(decoded.tobytes()) != f['pixel_sha256']):
                    raise live.AdmissionError('Actual PNG/RGB shape or pixel commitment differs')
            payload={'model':'chandra','temperature':0,'top_p':.1,'max_tokens':12384,'stream':False,
                     'messages':[{'role':'user','content':[
                         {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}},
                         {'type':'text','text':prompt}]}]}
            request=json.dumps(payload,separators=(',',':')).encode();total+=len(request)
            if total>128*1024**2:raise live.AdmissionError('Corpus request bytes exceed 128 MiB')
            fixture_id=prefix+':'+f['id']
            self._expected[fixture_id]=f['independent_expected']
            work.append(core.Work(fixture_id,fixture_id,request,json.dumps(f['expected'],sort_keys=True,separators=(',',':'))))
        return tuple(work),pixels,manifest

    def load(self, corpus, admission):
        self._expected={};self._warmup=()
        if len(admission['concurrency']) != 1:
            raise live.AdmissionError('One concurrency per first-seen admission; no measured corpus replay')
        worker = admission['expected_worker']
        if (type(worker.get('pid')) is not int or worker['pid'] <= 0
                or not isinstance(worker.get('starttime'), (str, int)) or not worker['starttime']):
            raise live.AdmissionError('First-seen worker PID and process starttime must be pinned')
        attestation = admission['first_seen_root_attestation']
        if (attestation.get('fresh_identified_worker') is not True
                or attestation.get('no_prior_measured_corpus_use') is not True
                or attestation.get('worker_identity') != admission['expected_worker']):
            raise live.AdmissionError('Root-attested fresh worker/prior-use exclusion required')
        if admission['repeat']!=1 or admission['warmup_cycles']!=1:
            raise live.AdmissionError('First-seen measured pages and separate warmup admitted once only')
        if admission['response_bytes']>1024**2:
            raise live.AdmissionError('Prospective first-seen response cap is 1 MiB; output tokens remain 12384')
        if Path(admission['warmup_corpus']).resolve()!=self.warmup_corpus or Path(admission['generator_path']).resolve()!=self.generator:
            raise live.AdmissionError('Explicit warmup/generator paths differ')
        if core.digest(self.generator.read_bytes())!=admission['generator_sha256']:
            raise live.AdmissionError('Generator source admission differs')
        spec=importlib.util.spec_from_file_location('first_seen_generator',self.generator)
        generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
        prompt=(self.root/'benchmarks/prompt.txt').read_text()
        if admission['work_scope'] not in ('pilot', 'formal'):
            raise live.AdmissionError('Declared pilot/formal scope required before admission')
        work,pixels,manifest=self._corpus(corpus,admission['measured_manifest_sha256'],generator,'measured',prompt,admission['selected_pages'])
        warmup,warm_pixels,warm_manifest=self._corpus(self.warmup_corpus,admission['warmup_manifest_sha256'],generator,'warmup',prompt)
        if len(warmup)!=5 or pixels & warm_pixels:
            raise live.AdmissionError('Exactly five disjoint warmup pages required')
        self._warmup=warmup
        if sum(len(p) for p in self.requests(work))>128*1024**2:
            raise live.AdmissionError('Combined immutable requests exceed 128 MiB')
        metadata={'manifest_sha256':admission['measured_manifest_sha256'],
                  'warmup_manifest_sha256':admission['warmup_manifest_sha256'],
                  'fixture_order':[w.fixture_id for w in work],
                  'warmup_fixture_order':[w.fixture_id for w in warmup],
                  'warmup_maximum_possible_concurrency':min(5, admission['concurrency'][0]),
                  'warmup_scope':'Five pages exercise at most five concurrent requests; larger true batch shapes are not established fully warmed.',
                  'generator_sha256':admission['generator_sha256'],
                  'work_scope':admission['work_scope'], 'selection_policy':'balanced-prefix-v1',
                  'selected_pages':len(work), 'available_pages':len(manifest['fixtures']),
                  'available_fixture_order':[f['id'] for f in manifest['fixtures']],
                  'prompt_sha256':core.digest(prompt.encode()),'repeat':1,
                  'measured_pixel_sha256':sorted(pixels),'warmup_pixel_sha256':sorted(warm_pixels),
                  'first_seen_root_attestation':attestation,
                  'scope':'Client decodes and proves disjoint selected measured/warmup pixels within one admission. Full manifest and selected prefix are committed before requests; unselected image files are not decoded by this client. Prior worker use is root attestation, not remotely proven global history. Fair curve points require separately owned fresh worker lifetimes and same warmup; this client never restarts workers. Text-prefix cache policy does not establish cold vision cache.'}
        return work,metadata

    def requests(self, work):
        return tuple(item.request for item in work)+tuple(item.request for item in self._warmup)

    def warmup(self, work, metadata, cycles):
        if cycles!=1 or len(self._warmup)!=5:raise live.AdmissionError('Separate warmup not prepared')
        return self._warmup

    def validate(self, item, reply):
        endpoint=core.endpoint_validator(item,reply)
        complete=self.complete.check(self._expected[item.fixture_id],json.loads(reply.raw),max_output_tokens=12384)
        return {'passed':endpoint['passed'] and complete['passed'], 'endpoint':endpoint,'complete':complete,
                'validator_identity':self.identity}


async def run(admission, url, corpus, monitor, output, *, warmup_corpus, generator,
              execute=False, http_transport=None):
    adapter=FirstSeen(warmup_corpus,generator)
    return await live.run(admission,url,corpus,monitor,output,execute=execute,
                          http_transport=http_transport,work_adapter=adapter)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('admission','direct-url','corpus','warmup-corpus','generator','monitor','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Live disabled without explicit execute and fresh root admission')
    admission,_=live.read_json(a.admission)
    report=asyncio.run(run(admission,a.direct_url,a.corpus,a.monitor,a.output,
                           warmup_corpus=a.warmup_corpus,generator=a.generator,execute=True))
    return 0 if report['valid'] else 1

if __name__=='__main__':raise SystemExit(main())
