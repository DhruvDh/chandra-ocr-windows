"""Explicit synthetic endpoint benchmark. No default endpoint and no service lifecycle mutations."""
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import time
import urllib.request
from evaluate import Tables, check_equations

def sha(b): return hashlib.sha256(b).hexdigest()

class Blocks(HTMLParser):
    def __init__(self): super().__init__(); self.blocks=[]
    def handle_starttag(self,tag,attrs):
        if tag=='div': self.blocks.append(dict(attrs))

def validate(expected, result, max_tokens=12384):
    failures=[]; text=result.get('content')
    if not isinstance(text,str) or not text.strip(): return dict(passed=False,failures=['missing_content'])
    if result.get('finish_reason')!='stop': failures.append('finish_reason')
    if result.get('stream_errors'): failures.append('stream_error')
    usage=result.get('usage')
    if not isinstance(usage,dict): failures.append('missing_usage')
    else:
        vals=[usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')]
        if not all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in vals): failures.append('invalid_usage')
        elif vals[0]+vals[1]!=vals[2] or vals[1]>=max_tokens: failures.append('token_accounting_or_truncation')
    parser=Tables(); parser.feed(text)
    if parser.rows!=expected['table']: failures.append('table_cells_and_order')
    for key in ('numbers','text','end_markers'):
        for value in expected[key]:
            if value not in text: failures.append(key+':'+value)
    failures.extend(check_equations(expected.get('equations', []), text))
    positions=[text.find(v) for v in expected['ordered_text']]
    if any(v<0 for v in positions) or positions!=sorted(positions): failures.append('reading_order')
    blocks=Blocks(); blocks.feed(text)
    labels=[b.get('data-label') for b in blocks.blocks]
    for label in expected['required_labels']:
        if label not in labels: failures.append('layout_label:'+label)
    if not blocks.blocks: failures.append('missing_layout')
    for b in blocks.blocks:
        try:
            coords=[float(v) for v in b['data-bbox'].replace(',',' ').split()]
            if len(coords)!=4 or not all(0<=v<=1000 for v in coords) or coords[0]>=coords[2] or coords[1]>=coords[3]: raise ValueError()
        except (KeyError,ValueError): failures.append('invalid_bbox')
    return dict(passed=not failures,failures=failures)

def parse_sse(raw):
    content=[]; finish=None; usage=None; done=False; errors=[]
    for block in raw.replace(b'\r\n',b'\n').split(b'\n\n'):
        lines=[line[5:].lstrip() for line in block.splitlines() if line.startswith(b'data:')]
        if not lines: continue
        value=b'\n'.join(lines)
        if value==b'[DONE]': done=True; continue
        event=json.loads(value)
        if event.get('error') is not None: errors.append(event['error'])
        if event.get('usage') is not None: usage=event['usage']
        for choice in event.get('choices',[]):
            if choice.get('index',0)!=0: continue
            delta=choice.get('delta',{}).get('content')
            if delta: content.append(delta)
            if choice.get('finish_reason') is not None: finish=choice['finish_reason']
    if not done: finish=None
    return dict(content=''.join(content),finish_reason=finish,usage=usage,stream_errors=errors)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base-url',required=True); p.add_argument('--corpus',required=True); p.add_argument('--output',required=True); p.add_argument('--provenance',required=True); p.add_argument('--repeat',type=int,default=5); p.add_argument('--phase',choices=['warm','process-cold','compile-cold','recreated'],default='warm'); p.add_argument('--stream',action='store_true'); p.add_argument('--timeout',type=float,default=900); a=p.parse_args()
    if a.repeat<1: p.error('repeat must be positive')
    root=Path(a.corpus); manifest_path=root/'manifest.json'; manifest=json.loads(manifest_path.read_text()); prompt=Path(__file__).with_name('prompt.txt').read_text(); provenance=json.loads(Path(a.provenance).read_text()); output=Path(a.output); output.mkdir(parents=True,exist_ok=False); runs=[]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for rep in range(a.repeat):
        for fixture in manifest['fixtures']:
            image=(root/fixture['image']).read_bytes()
            if sha(image)!=fixture['image_sha256']: raise ValueError('Corpus image hash mismatch')
            payload=dict(model='chandra',temperature=0,top_p=.1,max_tokens=12384,stream=a.stream,messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image).decode())),dict(type='text',text=prompt)])])
            if a.stream: payload['stream_options']={'include_usage':True}
            request_bytes=json.dumps(payload,separators=(',',':')).encode(); name=f'{rep:02d}-{fixture["id"]}'; (output/(name+'-request.json')).write_bytes(request_bytes); started=time.perf_counter(); raw=b''; error=None; status=None; ttft=None
            try:
                request=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=request_bytes,headers={'Content-Type':'application/json'})
                with opener.open(request,timeout=a.timeout) as response:
                    status=response.status
                    if a.stream:
                        parts=[]
                        for line in response:
                            if ttft is None and line.startswith(b'data:'):
                                try:
                                    event=json.loads(line[5:]); choices=event.get('choices',[])
                                    if any(c.get('delta',{}).get('content') for c in choices): ttft=time.perf_counter()-started
                                except json.JSONDecodeError: pass
                            parts.append(line)
                        raw=b''.join(parts); result=parse_sse(raw)
                    else:
                        raw=response.read(); data=json.loads(raw); choice=data['choices'][0]; result=dict(content=choice['message']['content'],finish_reason=choice['finish_reason'],usage=data.get('usage'))
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc); result={}
                if hasattr(exc,'read'): raw=exc.read()
            elapsed=time.perf_counter()-started; (output/(name+'-response.raw')).write_bytes(raw)
            correctness=validate(fixture['expected'],result)
            run=dict(schema='chandra-endpoint-run-v1',fixture_id=fixture['id'],repeat=rep,phase=(a.phase if rep==0 and fixture is manifest['fixtures'][0] else 'warm'),pages=1,profiled=False,elapsed_seconds=elapsed,ttft_seconds=ttft,http_status=status,error=error,result=result,correctness=correctness,provenance_verified=False,provenance=provenance,corpus_manifest_sha256=sha(manifest_path.read_bytes()),prompt_sha256=sha(prompt.encode()),request_sha256=sha(request_bytes),response_sha256=sha(raw),image_sha256=sha(image))
            if error or status!=200: run['correctness']['passed']=False
            (output/(name+'-run.json')).write_text(json.dumps(run,indent=2)+'\n'); runs.append(run)
    (output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n'); return 0 if all(r['correctness']['passed'] for r in runs) else 1
if __name__=='__main__': raise SystemExit(main())
