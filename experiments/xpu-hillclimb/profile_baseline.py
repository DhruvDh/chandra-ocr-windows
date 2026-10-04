"""Bounded full-generation profiling of the qualified hybrid XPU backend."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--image', default='benchmarks/inputs-v2/representative.png')
    p.add_argument('--prompt', default='benchmarks/prompt.txt')
    p.add_argument('--manifest', default='benchmarks/inputs-v2/manifest.json')
    p.add_argument('--output', required=True)
    p.add_argument('--cancel-file', help='Owned supervisor touches this file to cancel generation')
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--timings', type=int, default=1)
    p.add_argument('--profile-wait', type=int, default=0)
    p.add_argument('--profile-active', type=int, default=9)
    p.add_argument('--max-tokens', type=int, default=12384)
    a = p.parse_args()
    if min(a.warmups, a.timings, a.profile_wait) < 0 or a.profile_active < 1:
        p.error('Invalid run or profile-window counts')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    from runtime.waystone import XPUBackend
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function, schedule
    if not hasattr(ProfilerActivity, 'XPU'):
        raise RuntimeError('This runtime cannot record XPU events; no CPU-only substitute')
    image = Path(a.image).read_bytes()
    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    matches = [f for f in manifest['fixtures'] if f['image_sha256'] == hashlib.sha256(image).hexdigest()]
    if len(matches) != 1:
        raise RuntimeError('Input must match exactly one frozen synthetic fixture')
    fixture = matches[0]
    sys.path.insert(0, str(ROOT / 'benchmarks'))
    from endpoint import validate
    prompt = Path(a.prompt).read_text(encoding='utf-8')
    request = {'model': 'chandra', 'temperature': 0, 'max_tokens': a.max_tokens,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url':
                       {'url': 'data:image/png;base64,' + base64.b64encode(image).decode()}}]}]}
    backend = XPUBackend(a.model, attention_backend='hybrid')
    summary = {'schema': 'chandra.xpu.profile.v1', 'attention': 'hybrid',
               'profile_window': {'wait_forwards': a.profile_wait, 'active_forwards': a.profile_active},
               'image_sha256': hashlib.sha256(image).hexdigest(), 'prompt_sha256': digest(a.prompt),
               'manifest_sha256': digest(a.manifest), 'fixture': fixture['id'],
               'request_max_tokens': a.max_tokens,
               'source_sha256': {str(f.relative_to(ROOT)): digest(f) for f in [
                   Path(__file__), ROOT / 'runtime/waystone/backend.py',
                   ROOT / 'runtime/waystone/bootstrap.py', ROOT / 'runtime/waystone/model_identity.py']},
               'runs': [], 'performance_promotion': False}
    def save():
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    def run(label):
        torch.xpu.synchronize()
        torch.xpu.reset_peak_memory_stats()
        start = time.perf_counter()
        chunks, first, final = [], None, None
        cancel = threading.Event()
        print(json.dumps({'phase': 'begin', 'run': label}), flush=True)
        with (out / (label + '.stream.jsonl')).open('w', encoding='utf-8') as events:
            for part in backend.generate(request, cancel):
                if a.cancel_file and Path(a.cancel_file).exists():
                    cancel.set()
                elapsed = time.perf_counter() - start
                events.write(json.dumps({'elapsed': elapsed, 'part': part}) + '\n')
                events.flush()
                if isinstance(part, str):
                    chunks.append(part)
                    if first is None:
                        first = elapsed
                else:
                    final = part
        if final is None:
            raise RuntimeError('Missing final generation metadata')
        response = {'model': 'chandra', 'choices': [{'index': 0, 'message': {
            'role': 'assistant', 'content': ''.join(chunks)}, 'finish_reason': final['finish_reason']}],
            'usage': final['usage']}
        path = out / (label + '.response.json')
        path.write_text(json.dumps(response, indent=2), encoding='utf-8')
        row = {'run': label, 'elapsed_seconds': time.perf_counter() - start,
               'first_text_seconds': first, **final, 'response_sha256': digest(path),
               'correctness': validate(fixture['expected'], {'content': ''.join(chunks), **final}, a.max_tokens),
               'memory': backend.health()['torch_xpu_memory']}
        summary['runs'].append(row)
        save()
        print(json.dumps({'phase': 'complete', **row}), flush=True)
        if not row['correctness']['passed']:
            raise RuntimeError('Complete synthetic OCR gate failed; response retained')
    try:
        start = time.perf_counter()
        backend.load()
        summary.update(load_seconds=time.perf_counter() - start, health=backend.health(),
                       torch=torch.__version__)
        import transformers
        summary['transformers'] = transformers.__version__
        import inspect
        graph = Path(inspect.getfile(type(backend.model)))
        summary['model_graph_sha256'] = digest(graph)
        save()
        for i in range(a.warmups):
            run(f'warmup-{i}')
        for i in range(a.timings):
            run(f'timing-{i}')
        original_generate = backend.model.generate
        forwards = []
        def profiled_generate(*args, **kwargs):
            # The serving backend generates on a dedicated thread. Enter the
            # profiler there so thread-local CPU annotations cover actual work.
            scopes = []
            def before(module, args, kwargs):
                ids = kwargs.get('input_ids')
                if ids is None and args:
                    ids = args[0]
                tokens = int(ids.shape[-1]) if ids is not None else None
                phase = 'prefill' if tokens and tokens > 1 else 'decode'
                scope = record_function('chandra::' + phase)
                scope.__enter__()
                scopes.append((scope, time.perf_counter(), tokens, phase))
            def after(module, args, kwargs, result):
                scope, start, tokens, phase = scopes.pop()
                forwards.append({'index': len(forwards), 'phase': phase, 'input_tokens': tokens,
                                 'host_forward_seconds': time.perf_counter() - start})
                scope.__exit__(None, None, None)
                prof.step()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                         record_shapes=True, profile_memory=False, with_stack=False,
                         schedule=schedule(wait=a.profile_wait, warmup=0,
                                           active=a.profile_active, repeat=1),
                         on_trace_ready=lambda prof: prof.export_chrome_trace(str(out / 'trace.json'))) as prof:
                pre = backend.model.register_forward_pre_hook(before, with_kwargs=True)
                post = backend.model.register_forward_hook(after, with_kwargs=True)
                try:
                    return original_generate(*args, **kwargs)
                finally:
                    pre.remove()
                    post.remove()
            # Exiting the profile records the selected window only.
        backend.model.generate = profiled_generate
        try:
            run('profile')
        finally:
            backend.model.generate = original_generate
        (out / 'forwards.json').write_text(json.dumps(forwards, indent=2), encoding='utf-8')
        summary['trace_sha256'] = digest(out / 'trace.json')
        save()
    finally:
        # A bound-method local otherwise keeps all weights alive across unload.
        if 'original_generate' in locals():
            del original_generate
        backend.unload()
        print(json.dumps({'phase': 'unloaded', 'memory': backend.health()['torch_xpu_memory']}), flush=True)


if __name__ == '__main__':
    main()
