"""Bounded observational proxy for public synthetic Chandra CLI acceptance."""
import argparse
import base64
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import signal
import socket
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

REQUEST_LIMIT = 128 * 1024 * 1024
RESPONSE_LIMIT = 32 * 1024 * 1024

def sha(data): return hashlib.sha256(data).hexdigest()

def image_fact(request):
    from PIL import Image
    payload=json.loads(request)
    parts=payload['messages'][0]['content']
    images=[p['image_url']['url'] for p in parts if p.get('type')=='image_url']
    if len(images)!=1: raise ValueError('Expected one request image')
    header,encoded=images[0].split(',',1)
    if header not in ('data:image/png;base64','data:image/jpeg;base64','data:image/webp;base64'): raise ValueError('Unsupported image data URL')
    data=base64.b64decode(encoded,validate=True)
    with Image.open(io.BytesIO(data)) as image:
        if image.width*image.height > 8_000_000: raise ValueError('Recorder image pixel limit')
        image.load(); rgb=image.convert('RGB')
        return dict(encoded_sha256=sha(data),pixels_sha256=sha(rgb.tobytes()),width=rgb.width,height=rgb.height,mode='RGB',input_mode=image.mode)

def expected_images(corpus, manifest):
    from PIL import Image
    from chandra.model.util import scale_to_fit
    import inspect
    facts=[]
    for fixture in manifest['fixtures']:
        for page in fixture['rendered_pages']:
            path=(corpus/page['image']).resolve()
            if not path.is_relative_to(corpus.resolve()) or sha(path.read_bytes())!=page['image_sha256']: raise ValueError('Frozen page commitment mismatch')
            with Image.open(path) as image:
                rgb=image.convert('RGB')
                if sha(rgb.tobytes())!=page['pixels_sha256']: raise ValueError('Frozen decoded page commitment mismatch')
                image=scale_to_fit(rgb)
                facts.append(dict(fixture_id=fixture['id'],page_num=page['page_num'],pixels_sha256=sha(image.tobytes()),width=image.width,height=image.height,mode='RGB',frozen_pixels_sha256=page['pixels_sha256']))
    source=Path(inspect.getsourcefile(scale_to_fit))
    return facts,dict(path=str(source),sha256=sha(source.read_bytes()),operation='Frozen render -> RGB -> installed Chandra scale_to_fit; association uses decoded pixels, never arrival order')

def observed_response(status, raw, request):
    failures=[]
    try:
        data=json.loads(raw); payload=json.loads(request)
        if status!=200 or data.get('error') is not None: failures.append('http_or_error')
        choices=data.get('choices')
        if not isinstance(choices,list) or len(choices)!=1: raise ValueError('choices')
        choice=choices[0]; content=choice['message']['content']; finish=choice.get('finish_reason'); usage=data.get('usage')
        if not isinstance(content,str) or not content.strip(): failures.append('missing_content')
        if finish!='stop': failures.append('finish_reason')
        if not isinstance(usage,dict): raise ValueError('usage')
        counts=[usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')]
        if not all(type(v) is int and v>0 for v in counts): failures.append('token_counts')
        elif counts[0]+counts[1]!=counts[2] or counts[1]>=payload.get('max_tokens',12384): failures.append('token_accounting_or_truncation')
        return dict(passed=not failures,failures=failures,finish_reason=finish,usage=usage)
    except (ValueError,KeyError,TypeError,AttributeError):
        return dict(passed=False,failures=failures+['invalid_completion_metadata'],finish_reason=None,usage=None)

class Recorder:
    def __init__(self, upstream, output, timeout=900):
        self.upstream=urlsplit(upstream.rstrip('/'))
        if self.upstream.scheme not in ('http','https') or not self.upstream.hostname or self.upstream.username or self.upstream.password or self.upstream.query or self.upstream.fragment or not self.upstream.path.endswith('/v1'): raise ValueError('Use an explicit HTTP(S) /v1 API base without credentials/query/fragment')
        if not math.isfinite(timeout) or timeout<=0: raise ValueError('Positive finite timeout required')
        self.active_connections=set();self.active_sockets=set();self.handlers=0;self.closed=threading.Event()
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.timeout=timeout;self.lock=threading.Lock();self.records=[];self.counter=0;self.slots=threading.BoundedSemaphore(4)
        recorder=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def setup(self):
                super().setup();self.connection.settimeout(min(recorder.timeout,30))
            def log_message(self,*args): pass
            def do_GET(self): self.forward()
            def do_POST(self): self.forward()
            def forward(self):
                if (self.command,self.path) not in (('GET','/v1/models'),('POST','/v1/chat/completions')):
                    self.send_error(404);return
                if not recorder.slots.acquire(blocking=False): self.send_error(503);return
                with recorder.lock:recorder.handlers+=1
                try: recorder.handle(self)
                finally:
                    recorder.slots.release()
                    with recorder.lock:recorder.handlers-=1
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,name='synthetic-cli-recorder',daemon=True)
    @property
    def base_url(self): return f'http://127.0.0.1:{self.server.server_address[1]}/v1'
    def start(self): self.thread.start();return self
    def close(self):
        self.server.shutdown()
        with self.lock: connections=list(self.active_connections);sockets=list(self.active_sockets)
        for channel in sockets:
            try:channel.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        for connection in connections: connection.close()
        self.server.server_close();self.thread.join(timeout=5)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with self.lock:pending=self.handlers
            if not pending:break
            time.sleep(.02)
        self.closed.set()
    def handle(self, handler):
        with self.lock: index=self.counter;self.counter+=1
        name=f'{index:04d}';started=time.monotonic();body=b'';raw=b'';status=502;error=None;headers={};image=None;connection=None;upstream_status=None;captured=bytearray();client_raw=None;client_status=None;channel=None;incoming=bytearray();length=None
        try:
            if handler.headers.get('Transfer-Encoding'): raise ValueError('Chunked recorder requests are unsupported')
            length=int(handler.headers.get('Content-Length','0'))
            if length<0 or length>REQUEST_LIMIT: raise ValueError('Recorder request size limit')
            if handler.command=='POST' and not length: raise ValueError('Missing request body')
            incoming=bytearray()
            while len(incoming)<length:
                remaining=self.timeout-(time.monotonic()-started)
                if remaining<=0:raise TimeoutError('Recorder request deadline')
                handler.connection.settimeout(remaining)
                chunk=handler.rfile.read1(min(65536,length-len(incoming)))
                if not chunk:raise ValueError('Incomplete request body')
                incoming.extend(chunk)
            body=bytes(incoming)
            if handler.command=='POST': image=image_fact(body)
            factory=http.client.HTTPSConnection if self.upstream.scheme=='https' else http.client.HTTPConnection
            connection=factory(self.upstream.hostname,self.upstream.port,timeout=max(.001,self.timeout-(time.monotonic()-started)))
            with self.lock:self.active_connections.add(connection)
            upstream_path=self.upstream.path+handler.path.removeprefix('/v1')
            # One submission only. http.client performs no redirect or retry.
            connection.request(handler.command,upstream_path,body=body if handler.command=='POST' else None,headers={'Content-Type':'application/json','Accept':'application/json','Connection':'close'})
            channel=connection.sock
            with self.lock:self.active_sockets.add(channel)
            response=connection.getresponse();status=response.status;upstream_status=status
            headers={key:response.getheader(key) for key in ('content-type','x-chandra-backend') if response.getheader(key) is not None}
            size=0
            while True:
                remaining=self.timeout-(time.monotonic()-started)
                if remaining<=0: raise TimeoutError('Recorder forwarding deadline')
                if response.fp is None: break
                response.fp.raw._sock.settimeout(remaining)
                chunk=response.read1(65536)
                if not chunk: break
                size+=len(chunk)
                captured.extend(chunk[:max(0,RESPONSE_LIMIT-len(captured))])
                if size>RESPONSE_LIMIT: raise ValueError('Recorder response size limit')
            raw=bytes(captured)
        except Exception as exc:
            if incoming and not body:body=bytes(incoming)
            error=type(exc).__name__+': '+str(exc)
            raw=bytes(captured);client_status=502;client_raw=json.dumps({'error':{'type':'recorder_failure','message':error}}).encode()
            if upstream_status is None:status=502
        finally:
            if connection is not None:
                connection.close()
                with self.lock:
                    self.active_connections.discard(connection);self.active_sockets.discard(channel)
        request_path=self.output/(name+'-request.raw');response_path=self.output/(name+'-response.raw');request_path.write_bytes(body);response_path.write_bytes(raw)
        if client_raw is not None:(self.output/(name+'-delivered.raw')).write_bytes(client_raw)
        record=dict(schema='chandra-cli-http-observation-v1',id=index,method=handler.command,path=handler.path,upstream_path=self.upstream.path+handler.path.removeprefix('/v1'),status=status,upstream_status=upstream_status,delivered_status=client_status or status,error=error,elapsed_seconds=time.monotonic()-started,declared_request_bytes=length,request_complete=(length is not None and len(body)==length),request_sha256=sha(body),response_sha256=sha(raw),response_headers=headers,image=image,request_file=request_path.name,response_file=response_path.name,delivered_sha256=sha(client_raw if client_raw is not None else raw))
        if handler.command=='POST':record['completion']=observed_response(status,raw,body)
        (self.output/(name+'-record.json')).write_text(json.dumps(record,indent=2)+'\n')
        with self.lock:self.records.append(record)
        try:
            delivered=client_raw if client_raw is not None else raw
            handler.send_response(client_status or status)
            for key,value in (headers if client_raw is None else {'content-type':'application/json'}).items():handler.send_header(key,value)
            handler.send_header('Content-Length',str(len(delivered)));handler.send_header('Connection','close');handler.end_headers();handler.wfile.write(delivered)
        except (OSError,TimeoutError):pass
        handler.close_connection=True

def gate(records, expected, cli_reports):
    failures=[];associations=[];seen=set();cli_pages={}
    for report in cli_reports:
        metadata=report.get('metadata') or {}
        for page in metadata.get('pages',[]):
            cli_pages[(report.get('fixture_id'),page.get('page_num'))]=page.get('token_count')
    for record in records:
        if record['method']!='POST':continue
        image=record.get('image');matches=[]
        if image:
            matches=[p for p in expected if all(p[key]==image[key] for key in ('pixels_sha256','width','height','mode'))]
        match=matches[0] if len(matches)==1 else None
        associations.append(dict(record_id=record['id'],matches=[dict(fixture_id=p['fixture_id'],page_num=p['page_num']) for p in matches],association='unique_pixels' if match else 'ambiguous_or_unmatched',completion=record.get('completion')))
        if match:
            key=(match['fixture_id'],match['page_num'])
            if key in seen:failures.append('extra_or_retried_page:'+str(key))
            seen.add(key)
            count=(record.get('completion',{}).get('usage') or {}).get('completion_tokens')
            if cli_pages.get(key)!=count or type(count) is not int:failures.append('cli_actual_token_binding:'+str(key))
        else:failures.append('page_association:'+str(record['id']))
        if record.get('completion',{}).get('passed') is not True or record.get('error'):failures.append('actual_completion:'+str(record['id']))
    wanted={(p['fixture_id'],p['page_num']) for p in expected}
    if seen!=wanted:failures.append('observed_page_coverage')
    if {r.get('fixture_id') for r in cli_reports}!={p['fixture_id'] for p in expected} or len(cli_reports)!=len({p['fixture_id'] for p in expected}):failures.append('cli_fixture_coverage')
    if not cli_reports or any(r.get('correctness',{}).get('passed') is not True for r in cli_reports):failures.append('cli_content_or_metadata')
    return dict(passed=not failures,failures=failures,associations=associations,truncation_verified=not failures,full_ocr_acceptance=not failures,provenance_verified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream-base-url',required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=900);a=p.parse_args()
    if os.name!='posix':p.error('Recorded CLI subprocess-tree cleanup currently requires a POSIX client host; upstream may be native Windows')
    from evaluate import verify_manifest
    corpus=a.corpus.resolve();manifest_path=corpus/'manifest.json'
    if verify_manifest(manifest_path)['passed'] is not True:raise ValueError('Corpus commitments failed')
    root=Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve()!=(root/'.venv').resolve():p.error('Run with repository .venv/bin/python to bind the same CPU client environment as cli_acceptance.py')
    manifest=json.loads(manifest_path.read_text());expected,preprocess=expected_images(corpus,manifest);output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    sources={str(path.relative_to(root)):sha(path.read_bytes()) for directory in ('chandra_service','runtime/waystone') for path in (root/directory).glob('*.py')}
    sources.update({str(path.relative_to(root)):sha(path.read_bytes()) for path in (Path(__file__),root/'benchmarks/cli_acceptance.py')})
    import chandra
    client_root=Path(chandra.__file__).parent
    client_sources={str(path.relative_to(client_root)):sha(path.read_bytes()) for path in (client_root/'model/vllm.py',client_root/'model/util.py',client_root/'input.py',client_root/'settings.py',client_root/'prompts.py')}
    lock_hashes={str(path.relative_to(root)):sha(path.read_bytes()) for path in (root/'uv.lock',root/'runtime/waystone/uv.lock') if path.is_file()}
    recorder=Recorder(a.upstream_base_url,output/'http',a.timeout).start();command=[sys.executable,str(root/'benchmarks/cli_acceptance.py'),'--base-url',recorder.base_url,'--corpus',str(corpus),'--output',str(output/'cli'),'--timeout',str(a.timeout)]
    process=None;error=None;returncode=None
    try:
        process=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        stdout,stderr=process.communicate(timeout=a.timeout*len(manifest['fixtures'])+30);returncode=process.returncode
        (output/'stdout.raw').write_bytes(stdout);(output/'stderr.raw').write_bytes(stderr)
    except (OSError,subprocess.TimeoutExpired) as exc:
        error=type(exc).__name__+': '+str(exc)
        (output/'stdout.raw').write_bytes(getattr(exc,'output',None) or b'')
        (output/'stderr.raw').write_bytes(getattr(exc,'stderr',None) or b'')
    finally:
        if process is not None:
            # This fresh session contains only this recorder's CLI and descendants.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            if process.poll() is None:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait(timeout=5)
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        recorder.close()
    try:reports=json.loads((output/'cli/runs.json').read_text())
    except (OSError,ValueError):reports=[]
    with recorder.lock:records=sorted(recorder.records,key=lambda r:r['id'])
    result=gate(records,expected,reports)
    if recorder.handlers:result['passed']=False;result['failures'].append('recorder_handlers_not_quiescent')
    if error or process is None or returncode!=0:result['passed']=False;result['full_ocr_acceptance']=False;result['truncation_verified']=False;result['failures'].append('cli_process')
    if not result['passed']:result['truncation_verified']=False;result['full_ocr_acceptance']=False
    receipt=dict(schema='chandra-recorded-cli-v1',upstream_base_url=a.upstream_base_url,command=command,returncode=returncode,error=error,corpus_sha256=sha(manifest_path.read_bytes()),expected_request_pixels=expected,preprocess=preprocess,local_candidate_source_sha256=sources,local_client_source_sha256=client_sources,lock_sha256=lock_hashes,remote_runtime_source_identity_verified=False,records=records,cli_reports=reports,correctness=result,scope='Synthetic content+metadata+actual HTTP finish/usage; no claim of hidden holdout or numerical model validation')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
