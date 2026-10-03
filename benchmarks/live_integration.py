"""Bounded, explicitly invoked synthetic acceptance; never starts services or replays POSTs."""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time
import httpx
from endpoint import parse_sse, validate
from evaluate import verify_manifest

MAX_RESPONSE_BYTES = 32*1024*1024

def sha(data): return hashlib.sha256(data).hexdigest()

def occupancy(health, backend):
    if 'backends' in health:
        entries=health['backends']
        if backend=='auto': values=list(entries.values())
        else: values=[entries.get(backend,{})]
        if not values or any(type(v.get('active')) is not int or type(v.get('remote_active')) is not int for v in values): return None
        return sum(max(v['active'],v['remote_active']) for v in values)
    value=health.get('pending_or_active')
    return value if type(value) is int and value>=0 else None

class Harness:
    def __init__(self, client, base_url, corpus, output, backend='auto', request_seconds=900, drain_seconds=120):
        self.client=client;self.base=base_url.rstrip('/');self.corpus=corpus;self.output=output;self.backend=backend;self.request_seconds=request_seconds;self.drain_seconds=drain_seconds
        self.manifest=json.loads((corpus/'manifest.json').read_text());self.fixtures={f['id']:f for f in self.manifest['fixtures']};self.prompt=Path(__file__).with_name('prompt.txt').read_text();self.counter=0;self.is_router=False
    def payload(self, fixture, stream=False):
        f=self.fixtures[fixture];path=(self.corpus/f['image']).resolve()
        if not path.is_relative_to(self.corpus.resolve()):raise ValueError('Fixture path escape')
        encoded=path.read_bytes()
        if sha(encoded)!=f['image_sha256']:raise ValueError('Fixture hash mismatch')
        return dict(model='chandra',temperature=0,top_p=.1,max_tokens=12384,stream=stream,**({'stream_options':{'include_usage':True}} if stream else {}),messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(encoded).decode())),dict(type='text',text=self.prompt)])])
    async def health(self):
        r=await self.client.get(self.base.removesuffix('/v1')+'/health',timeout=5);r.raise_for_status();return r.json()
    async def idle(self):
        deadline=time.monotonic()+self.drain_seconds;samples=[]
        while time.monotonic()<deadline:
            h=await self.health();samples.append(dict(seconds=time.monotonic(),health=h))
            if occupancy(h,self.backend)==0:return dict(passed=True,samples=samples)
            await asyncio.sleep(.5)
        return dict(passed=False,samples=samples,failure='drain_deadline_or_unobservable_occupancy')
    async def request(self, fixture, stream=False, cancel=False, first_content=None, admitted=None):
        self.counter+=1;name=f'{self.counter:03d}-{fixture}';request=json.dumps(self.payload(fixture,stream),separators=(',',':')).encode();(self.output/(name+'-request.json')).write_bytes(request)
        raw=bytearray();status=None;headers={};error=None;interrupted=False;ttft=None;cancelled=False;started=time.monotonic();result={}
        try:
            async with asyncio.timeout(self.request_seconds):
                async with self.client.stream('POST',self.base+'/chat/completions',content=request,headers={'Content-Type':'application/json','X-Chandra-Backend':self.backend},timeout=self.request_seconds) as response:
                    status=response.status_code;headers=dict(response.headers)
                    if admitted:admitted.set()
                    if stream and status==200:
                        pending=b''
                        async for chunk in response.aiter_raw():
                            raw.extend(chunk);pending+=chunk
                            if len(raw)>MAX_RESPONSE_BYTES:raise ValueError('Response size bound')
                            while b'\n' in pending:
                                line,pending=pending.split(b'\n',1)
                                if not line.startswith(b'data:'):continue
                                try:event=json.loads(line[5:])
                                except ValueError:continue
                                if any(c.get('delta',{}).get('content') for c in event.get('choices',[])):
                                    if ttft is None:
                                        ttft=time.monotonic()-started
                                        if first_content:first_content.set()
                                    if cancel:cancelled=True;break
                            if cancelled:break
                    else:
                        async for chunk in response.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw)>MAX_RESPONSE_BYTES:raise ValueError('Response size bound')
            if status==200 and not cancelled:
                if stream:result=parse_sse(bytes(raw))
                else:
                    document=json.loads(raw);choice=document['choices'][0];result=dict(content=choice['message']['content'],finish_reason=choice['finish_reason'],usage=document.get('usage'))
        except asyncio.CancelledError:
            error='Orchestrator budget/cancellation interrupted request; backend drain remains unknown';interrupted=True
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            if first_content and not first_content.is_set():first_content.set()
            if admitted and not admitted.is_set():admitted.set()
        correctness=validate(self.fixtures[fixture]['expected'],result)
        if self.is_router and self.backend!='auto' and status==200 and headers.get('x-chandra-backend')!=self.backend:
            correctness['passed']=False;correctness['failures'].append('backend_selection_unverified')
        if status!=200 or error or cancelled:correctness['passed']=False
        report=dict(fixture=fixture,backend_selection_verified=not self.is_router or self.backend=='auto' or headers.get('x-chandra-backend')==self.backend,status=status,response_headers=headers,error=error,elapsed_seconds=time.monotonic()-started,ttft_seconds=ttft,cancelled_after_content=cancelled,cancelled_before_terminal_event=cancelled and b'[DONE]' not in raw,correctness=correctness,result=result,request_sha256=sha(request),response_sha256=sha(raw),raw_file=name+'-response.raw',provenance_verified=False)
        (self.output/(name+'-response.raw')).write_bytes(raw);(self.output/(name+'-run.json')).write_text(json.dumps(report,indent=2)+'\n')
        if interrupted:raise asyncio.CancelledError
        return report
    async def concurrent(self):
        ready=asyncio.Event();primary=asyncio.create_task(self.request('tiny',True,admitted=ready))
        try:
            await ready.wait();health=await self.health();second=await self.request('small');first=await primary;drain=await self.idle()
            rejected=second['status']==503 and not second['result'] and second['error'] is None
            overlap=occupancy(health,self.backend)
            return dict(passed=first['correctness']['passed'] and (second['correctness']['passed'] or rejected) and overlap is not None and overlap>0 and drain['passed'],primary=first,contender=second,health_during_primary=health,observed_overlap=overlap,contender_rejected_at_capacity=rejected,drain=drain,evidence_scope='Two submitted requests, observed service ownership, content isolation; no replay/cross-talk inferred beyond retained results')
        finally:
            if not primary.done():
                primary.cancel()
                await asyncio.gather(primary,return_exceptions=True)
    async def cancellation(self):
        result=await self.request('tiny',True,cancel=True);health=await self.health();drain=await self.idle()
        recovery=await self.request('tiny') if drain['passed'] else None
        return dict(passed=result['status']==200 and result['backend_selection_verified'] and result['cancelled_before_terminal_event'] and drain['passed'] and recovery is not None and recovery['correctness']['passed'],cancelled_request=result,health_after_close=health,drain=drain,recovery=recovery,evidence_scope='Downstream close after actual content, bounded observed drain and subsequent complete OCR; backend logs required to establish exact generation stop or replay count')

async def run_cli_process(command, timeout, output):
    stdout_path=output/'cli-stdout.raw';stderr_path=output/'cli-stderr.raw'
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        process=await asyncio.create_subprocess_exec(*command,stdout=stdout,stderr=stderr,start_new_session=os.name=='posix')
        try:
            await asyncio.wait_for(process.wait(),timeout)
        finally:
            if process.returncode is None:
                if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
                try:await asyncio.wait_for(process.wait(),5)
                except asyncio.TimeoutError:
                    if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                    await process.wait()
    return subprocess.CompletedProcess(command,process.returncode,stdout_path.read_bytes(),stderr_path.read_bytes())


async def execute(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    corpus=args.corpus.resolve();cli_corpus=args.cli_corpus.resolve()
    if not verify_manifest(corpus/'manifest.json')['passed'] or not verify_manifest(cli_corpus/'manifest.json')['passed']:raise ValueError('Corpus commitment mismatch')
    provenance=json.loads(args.provenance.read_text());report=dict(schema='chandra-live-integration-v1',base_url=args.base_url,cli_base_url=args.cli_base_url or args.base_url,backend=args.backend,provenance=provenance,provenance_verified=False,corpus_sha256=sha((corpus/'manifest.json').read_bytes()),stages={},passed=False)
    started=time.monotonic()
    try:
        async with asyncio.timeout(args.budget_seconds), httpx.AsyncClient(trust_env=False,follow_redirects=False) as client:
            harness=Harness(client,args.base_url,corpus,output,args.backend,args.request_seconds,args.drain_seconds)
            health=await harness.health();models=await client.get(args.base_url.rstrip('/')+'/models',timeout=5);models.raise_for_status()
            harness.is_router='backends' in health
            report['initial_health']=health;report['models']=models.json()
            if not any(m.get('id')=='chandra' for m in report['models'].get('data',[])):raise ValueError('Missing chandra model alias')
            if occupancy(health,args.backend)!=0:raise ValueError('Service is occupied or ownership cannot be observed; no inference submitted')
            for stage in args.stages:
                if stage=='cli':
                    command=[sys.executable,str(Path(__file__).with_name('cli_acceptance.py')),'--base-url',args.cli_base_url or args.base_url,'--corpus',str(cli_corpus),'--output',str(output/'cli'),'--timeout',str(args.request_seconds)]
                    remaining=max(1,args.budget_seconds-(time.monotonic()-started))
                    result=await run_cli_process(command,remaining,output)
                    (output/'cli-stdout.raw').write_bytes(result.stdout);(output/'cli-stderr.raw').write_bytes(result.stderr)
                    value=dict(passed=result.returncode==0,returncode=result.returncode,command=command,results=json.loads((output/'cli/runs.json').read_text()) if (output/'cli/runs.json').exists() else None)
                elif stage=='corpus':
                    runs=[await harness.request(f) for f in harness.fixtures];value=dict(passed=all(r['correctness']['passed'] for r in runs),runs=runs)
                elif stage=='warm':
                    warmup=await harness.request(args.warm_fixture)
                    runs=[]
                    if warmup['correctness']['passed']:
                        for _ in range(args.repeat):
                            run=await harness.request(args.warm_fixture);runs.append(run)
                            if not run['correctness']['passed']:break
                    times=[r['elapsed_seconds'] for r in runs];value=dict(passed=len(runs)>=5 and all(r['correctness']['passed'] for r in runs),warmup=warmup,runs=runs,fixture=args.warm_fixture,n=len(runs),median_seconds=statistics.median(times) if times else None,min_seconds=min(times) if times else None,max_seconds=max(times) if times else None,completion_tokens=[r['result'].get('usage',{}).get('completion_tokens') for r in runs],performance_promotion=False)
                elif stage=='concurrency':value=await harness.concurrent()
                else:value=await harness.cancellation()
                report['stages'][stage]=value
                (output/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
                if not value['passed']:break
                drain=await harness.idle()
                if not drain['passed']:raise ValueError('Stage finished but service did not drain')
            report['passed']=len(report['stages'])==len(args.stages) and all(v['passed'] for v in report['stages'].values())
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (output/'integration.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',required=True);p.add_argument('--cli-base-url');p.add_argument('--corpus',type=Path,default=Path(__file__).with_name('inputs-v2'));p.add_argument('--cli-corpus',type=Path,default=Path(__file__).with_name('cli-inputs-v1'));p.add_argument('--output',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True);p.add_argument('--backend',choices=['auto','waystone','northstone'],default='auto');p.add_argument('--stages',nargs='+',choices=['cli','corpus','warm','concurrency','cancel'],default=['cli','corpus','warm','concurrency','cancel']);p.add_argument('--repeat',type=int,default=5);p.add_argument('--warm-fixture',default='tiny');p.add_argument('--request-seconds',type=float,default=900);p.add_argument('--drain-seconds',type=float,default=120);p.add_argument('--budget-seconds',type=float,default=3600);a=p.parse_args()
    if a.repeat<5:p.error('At least five warm repetitions required')
    if len(set(a.stages))!=len(a.stages):p.error('Stages must be unique')
    for url in (a.base_url,a.cli_base_url or a.base_url):
        if not url.startswith(('http://','https://')) or not url.rstrip('/').endswith('/v1'):p.error('Explicit HTTP(S) /v1 base required')
    if min(a.request_seconds,a.drain_seconds,a.budget_seconds)<=0:p.error('Deadlines must be positive')
    result=asyncio.run(execute(a));print(json.dumps({'passed':result['passed'],'stages':list(result['stages']),'error':result.get('error')}));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
