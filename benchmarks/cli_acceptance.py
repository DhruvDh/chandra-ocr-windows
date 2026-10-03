"""Explicit public synthetic CLI compatibility run; no default endpoint or lifecycle changes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from evaluate import Tables, check_equations, verify_manifest


def sha(data): return hashlib.sha256(data).hexdigest()

def validate_cli(expected, text, metadata, html_files):
    failures=[]
    if not isinstance(text,str) or not text.strip(): failures.append('missing_markdown')
    if not isinstance(metadata,dict): return dict(passed=False,failures=failures+['missing_metadata'])
    pages=metadata.get('pages',[])
    if not isinstance(pages,list): return dict(passed=False,failures=failures+['invalid_pages'])
    if type(metadata.get('num_pages')) is not int or metadata['num_pages'] != expected['pages'] or len(pages)!=expected['pages']: failures.append('page_count')
    if [p.get('page_num') for p in pages]!=list(range(expected['pages'])): failures.append('page_order')
    if metadata.get('file_name')!=expected['file_name']: failures.append('file_name')
    tokens=[];chunks=[]
    for p in pages:
        count=p.get('token_count'); chunk=p.get('num_chunks'); box=p.get('page_box')
        if type(count) is not int or not 0<count<12384: failures.append('page_tokens')
        else: tokens.append(count)
        if type(chunk) is not int or chunk<=0: failures.append('page_chunks')
        else: chunks.append(chunk)
        if not isinstance(box,list) or len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box) or box[0]>=box[2] or box[1]>=box[3]: failures.append('page_geometry')
    if type(metadata.get('total_token_count')) is not int or metadata['total_token_count']!=sum(tokens): failures.append('token_sum')
    if type(metadata.get('total_chunks')) is not int or metadata['total_chunks']!=sum(chunks): failures.append('chunk_sum')
    if html_files: failures.append('unexpected_html')
    if isinstance(text,str):
        parser=Tables();parser.feed(text)
        if parser.rows!=expected['table']: failures.append('table_cells_and_order')
        for field in ('text','numbers','end_markers'):
            for value in expected.get(field,[]):
                if value not in text: failures.append(field+':'+value)
        failures.extend(check_equations(expected.get('equations',[]),text))
        positions=[text.find(v) for v in expected.get('ordered_text',[])]
        if any(v<0 for v in positions) or positions!=sorted(positions): failures.append('reading_order')
    return dict(passed=not failures,failures=failures,evidence_scope='CLI content and metadata only',finish_reasons=None,truncation_verified=False,full_ocr_acceptance=False)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--base-url',required=True);p.add_argument('--model',default='chandra');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=1800);a=p.parse_args()
    if not a.base_url.startswith(('http://','https://')) or not a.base_url.rstrip('/').endswith('/v1'):p.error('base-url must be an explicit HTTP(S) /v1 API base')
    corpus=a.corpus.resolve();manifest_path=corpus/'manifest.json'
    if not verify_manifest(manifest_path)['passed']:raise ValueError('Corpus commitment mismatch')
    manifest=json.loads(manifest_path.read_text());output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];reports=[]
    for f in manifest['fixtures']:
        if f['id'] not in ('native','scanned'):raise ValueError('Unsupported fixture identifier')
        pdf=(corpus/f['pdf']).resolve()
        if not pdf.is_relative_to(corpus) or sha(pdf.read_bytes())!=f['pdf_sha256']:raise ValueError('PDF commitment mismatch')
        run=output/f['id'];run.mkdir();state=run/'state';state.mkdir();ocr=run/'ocr'
        env={k:os.environ[k] for k in ('PATH','LANG','LC_ALL','SYSTEMROOT') if k in os.environ}
        env.update(HOME=str(state),XDG_CONFIG_HOME=str(state/'config'),XDG_DATA_HOME=str(state/'data'),XDG_CACHE_HOME=str(state/'cache'),HF_HOME=str(state/'huggingface'),UV_CACHE_DIR=str(state/'uv'),UV_PROJECT_ENVIRONMENT=str(root/'.venv'),VLLM_API_BASE=a.base_url,VLLM_MODEL_NAME=a.model)
        command=['uv','run','--no-sync','chandra',str(pdf),str(ocr),'--method','vllm','--batch-size','2','--no-html']
        started=time.perf_counter();error=None;returncode=None
        try:
            result=subprocess.run(command,cwd=root,env=env,capture_output=True,timeout=a.timeout);stdout=result.stdout;stderr=result.stderr;returncode=result.returncode
        except OSError as exc:
            stdout=b'';stderr=b'';error=type(exc).__name__+': '+str(exc)
        except subprocess.TimeoutExpired as exc:
            stdout=exc.stdout or b'';stderr=exc.stderr or b'';error='CLI timeout; submitted endpoint work may still be active'
        elapsed=time.perf_counter()-started;(run/'stdout.raw').write_bytes(stdout);(run/'stderr.raw').write_bytes(stderr)
        folder=ocr/Path(f['pdf']).stem;markdown=folder/(Path(f['pdf']).stem+'.md');metadata_path=folder/(Path(f['pdf']).stem+'_metadata.json');metadata=None
        try:
            text=markdown.read_text();metadata=json.loads(metadata_path.read_text())
        except (OSError,ValueError) as exc:text='';error=error or type(exc).__name__+': '+str(exc)
        correctness=validate_cli(f['expected'],text,metadata,list(ocr.rglob('*.html')) if ocr.exists() else [])
        if returncode!=0 or error:correctness['passed']=False;correctness['failures'].append('cli_process_failure')
        outputs=[dict(path=str(x.relative_to(run)),sha256=sha(x.read_bytes())) for x in ocr.rglob('*') if x.is_file()] if ocr.exists() else []
        report=dict(schema='chandra-cli-acceptance-v1',fixture_id=f['id'],command=command,environment={k:env[k] for k in ('VLLM_API_BASE','VLLM_MODEL_NAME')},corpus_sha256=sha(manifest_path.read_bytes()),pdf_sha256=sha((corpus/f['pdf']).read_bytes()),elapsed_seconds=elapsed,returncode=returncode,error=error,metadata=metadata,correctness=correctness,provenance_verified=False,output_files=outputs,stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),finish_reasons=None,raw_endpoint_responses_available=False)
        (run/'run.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
    (output/'runs.json').write_text(json.dumps(reports,indent=2)+'\n');return 0 if all(r['correctness']['passed'] for r in reports) else 1
if __name__=='__main__':raise SystemExit(main())
