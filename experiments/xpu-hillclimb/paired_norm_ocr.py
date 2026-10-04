"""Prepared only: one owned loaded model, two warmups, one or two paired trials.

Requires explicit root GPU lease, outer Windows Job Object and <=600s deadline.
No serving changes. All output is private. The measured request includes input
decode, processor/H2D, generation thread, streaming, buffered logging, thread
join and final scalar readback. Load, patch setup and full-token readback/checks
are excluded. Production mode is revision 6; previous immutable trial plans remain preserved.
"""
import argparse
import base64
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'benchmarks'))
from endpoint import validate
from runtime.waystone import XPUBackend
from norm_candidate import candidate, GRAPH_SHA256
from verification.compare_ocr import compare


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--candidate', choices=['norm-gain', 'production-gain'], default='norm-gain'); p.add_argument('--model', required=True); p.add_argument('--image', required=True); p.add_argument('--prompt', required=True); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True); p.add_argument('--qualified-reference', required=True); p.add_argument('--qualified-provenance', required=True); p.add_argument('--pairs', type=int, choices=[1, 2], default=2); p.add_argument('--start-gain', action='store_true'); a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    import torch
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as graph
    from PIL import Image
    if sha(inspect.getfile(graph)) != GRAPH_SHA256 or torch.__version__ != '2.14.1+xpu' or transformers.__version__ != '5.18.0': raise RuntimeError('Unqualified runtime or graph')
    manifest = json.loads(Path(a.manifest).read_text()); fixture = next(f for f in manifest['fixtures'] if f['image_sha256'] == sha(a.image))
    if fixture['id'] != 'representative' or manifest['schema'] != 'chandra-synthetic-corpus-v1': raise RuntimeError('Paired timing requires frozen representative fixture')
    prompt = Path(a.prompt).read_text(); request = {'model': 'chandra', 'temperature': 0, 'max_tokens': 12384, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(Path(a.image).read_bytes()).decode()}}]}]}
    report = {'schema': 'chandra.paired.norm.ocr.v1', 'helper_revision': 6 if a.candidate == 'production-gain' else 3, 'candidate': a.candidate, 'candidate_identity': 'production-gain-v1' if a.candidate == 'production-gain' else 'legacy-norm-gain', 'candidate_qualified': False, 'performance_promotion': False, 'request_max_tokens': 12384, 'image_sha256': sha(a.image), 'prompt_sha256': sha(a.prompt), 'manifest_sha256': sha(a.manifest), 'graph_sha256': GRAPH_SHA256, 'pairs_requested': a.pairs, 'runs': [], 'timing_scope': 'Elapsed backend request includes PNG/base64 decode, processor, H2D, generation thread, streaming, buffered event logging, thread join and final terminal scalar readback. Model load, candidate patch/gain setup, full-token CPU readback/decode and correctness checks excluded. Two warmups precede timing; no compiler.'}
    report['source_sha256'] = {name: sha(ROOT / name) for name in ['experiments/xpu-hillclimb/paired_norm_ocr.py', 'experiments/xpu-hillclimb/norm_candidate.py', 'runtime/waystone/backend.py', 'runtime/waystone/bootstrap.py', 'runtime/waystone/model_identity.py', 'benchmarks/endpoint.py', 'benchmarks/evaluate.py', 'verification/compare_ocr.py', 'runtime/waystone/uv.lock', 'provenance/model.json']}
    if a.candidate == 'production-gain':
        from production_gain_adapter import EXPECTED_GAIN_BYTES, production_gain_scope, counted_original_scope, steady_allocation_gate
        for name in ['runtime/waystone/norm_gain.py', 'experiments/xpu-hillclimb/production_gain_adapter.py']:
            report['source_sha256'][name] = sha(ROOT / name)
        report['prospective_memory_gate'] = {'gain_bytes': EXPECTED_GAIN_BYTES, 'install_allocated_delta_exact': True, 'request_peak_maximum': 'paired original peak plus 681984 bytes', 'live_current_delta_exact': True, 'no_accumulation': True, 'after_close_warmed_reference_equal': True, 'unload_zero': True, 'rounding_allowance_bytes': 0}
        report['instrumentation'] = 'Same 81 per-instance call-count wrappers and counter updates on both original and production variants, included in measured requests; setup/synchronized snapshots/token readback excluded.'
        report['staging_requirement'] = 'Stage actual pinned production norm_gain.py and adapter alongside historical qualified backend snapshot. Historical anchor hash checks remain strict; no unreviewed current backend hash exemption.'
    qualified = json.loads(Path(a.qualified_reference).read_text())
    anchor = json.loads(Path(a.qualified_provenance).read_text())
    for field in ['image_sha256', 'prompt_sha256', 'manifest_sha256', 'request_max_tokens']:
        if anchor[field] != report[field]: raise RuntimeError('Qualified anchor request identity differs: ' + field)
    if sha(a.qualified_reference) != anchor['response_sha256']: raise RuntimeError('Qualified raw response hash differs')
    if anchor['model_revision'] != 'af93b47dba1b47b6640c86ccf487ed2260ab9a09' or anchor['attention'] != 'hybrid' or anchor['precision'] != 'bfloat16': raise RuntimeError('Qualified anchor model profile differs')
    for name, digest in anchor['qualified_source_sha256'].items():
        if report['source_sha256'][name] != digest: raise RuntimeError('Qualified source identity differs: ' + name)
    report['qualified_anchor'] = {'response_sha256': sha(a.qualified_reference), 'provenance_sha256': sha(a.qualified_provenance), 'historical_generated_ids_recorded': False, 'scope': 'Exact historical content/HTML/usage/stop/geometry anchor; historical generated IDs and terminal scalar were not recorded. Newly captured IDs/EOS are checked against pinned stops and identically across all pairs. No historical IDs invented.'}
    def save(): (out / 'summary.json').write_text(json.dumps(report, indent=2))
    backend = XPUBackend(a.model, attention_backend='hybrid'); original_generate = None; captured = {}; reference_path = None; reference_ids = None
    try:
        start = time.perf_counter(); backend.load(); report['load_seconds'] = time.perf_counter() - start; report['loaded_health'] = backend.health()
        with Image.open(a.image) as image: rgb = image.convert('RGB')
        with torch.inference_mode(): inputs = backend.processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image', 'image': rgb}, {'type': 'text', 'text': prompt}]}], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        prefix = inputs['input_ids'][0].tolist(); report['processed_inputs'] = {name: {'shape': list(t.shape), 'dtype': str(t.dtype), 'sha256': hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()} for name, t in inputs.items()}; del inputs
        norms = [(n, m, m.forward) for n, m in backend.model.named_modules() if type(m) is graph.Qwen3_5RMSNorm]; assert len(norms) == 81
        original_generate = backend.model.generate
        def retain(*args, **kwargs):
            if captured or kwargs.get('past_key_values') is not None: raise RuntimeError('Unexpected retained generation state')
            captured['fresh_request'] = kwargs.get('past_key_values') is None
            captured['thread'] = threading.current_thread().name
            captured['ids'] = original_generate(*args, **kwargs)
            return captured['ids']
        backend.model.generate = retain
        baseline_current = None
        def readback(final, chunks):
            if final is None or 'ids' not in captured: raise RuntimeError('Missing complete token/final result')
            ids = captured.pop('ids')[0].cpu().tolist(); thread = captured.pop('thread'); fresh = captured.pop('fresh_request')
            if thread != 'chandra-xpu-generation' or not fresh or ids[:len(prefix)] != prefix: raise RuntimeError('Actual thread or fresh input prefix differs')
            generated = ids[len(prefix):]; stop = backend.model.generation_config.eos_token_id; stop = [stop] if isinstance(stop, int) else list(stop); stop.append(backend.processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
            if not generated or len(generated) != final['usage']['completion_tokens'] or final['usage']['prompt_tokens'] != len(prefix) or generated[-1] not in stop: raise RuntimeError('Terminal tokens disagree')
            decoded = backend.processor.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if decoded != ''.join(chunks): raise RuntimeError('Complete token decode differs from streamed content')
            return generated, thread, fresh, stop
        def run(label, mode, measured, pair=None):
            nonlocal reference_path, reference_ids, baseline_current
            if any(m.forward != original for _, m, original in norms): raise RuntimeError('Norm restoration failed before request')
            torch.xpu.synchronize(); setup = time.perf_counter()
            if a.candidate == 'production-gain':
                scope = production_gain_scope(backend.model) if mode == 'production-gain' else counted_original_scope(backend.model)
            else:
                scope = candidate(backend.model, mode='gain-only') if mode == 'norm-gain' else contextlib.nullcontext(None)
            with scope as receipt:
                torch.xpu.synchronize(); setup = time.perf_counter() - setup; torch.xpu.reset_peak_memory_stats(); chunks = []; final = None; first = None; start = time.perf_counter()
                with (out / (label + '.stream.jsonl')).open('w') as events:
                    try:
                        for part in backend.generate(request, threading.Event()):
                            elapsed = time.perf_counter() - start; events.write(json.dumps({'elapsed_seconds': elapsed, 'part': part}) + '\n')
                            if isinstance(part, str):
                                chunks.append(part)
                                if first is None: first = elapsed
                            else: final = part
                    finally:
                        elapsed = time.perf_counter() - start; (out / (label + '.partial.txt')).write_text(''.join(chunks))
                if a.candidate == 'production-gain': torch.xpu.synchronize()
                memory_inside_scope = backend.health()['torch_xpu_memory']
                if a.candidate == 'production-gain':
                    generated, thread, fresh, stop = readback(final, chunks)
            if a.candidate != 'production-gain':
                generated, thread, fresh, stop = readback(final, chunks)
            restored = all(m.forward == original for _, m, original in norms)
            if not restored or (receipt and not receipt['restored']): raise RuntimeError('Candidate context restoration failed')
            response = {'model': 'chandra', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(chunks)}, 'finish_reason': final['finish_reason']}], 'usage': final['usage']}; path = out / (label + '.response.json'); path.write_text(json.dumps(response, indent=2)); (out / (label + '.tokens.json')).write_text(json.dumps({'token_ids': generated, 'terminal_id': generated[-1], 'stop_ids': stop}))
            correctness = validate(fixture['expected'], {'content': ''.join(chunks), **final}, 12384)
            interception = receipt is None or (len(receipt['calls']) == 81 and all(v == len(generated) for v in receipt['calls'].values()))
            row = {'run': label, 'variant': mode, 'pair': pair, 'measured_warm_run': measured, 'elapsed_seconds': elapsed, 'first_text_seconds': first, 'candidate_setup_seconds': setup, 'correctness': correctness, 'response_sha256': sha(path), 'usage': final['usage'], 'finish_reason': final['finish_reason'], 'terminal_id': generated[-1], 'pinned_stop_ids': stop, 'complete_decode_equals_stream': True, 'decode_options': {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}, 'actual_generation_thread': thread, 'fresh_generation_call_without_past_key_values': fresh, 'norm_scope_restored': restored, 'norm_receipt': receipt, 'interception_passed': interception, 'memory': backend.health()['torch_xpu_memory']}
            row['memory_inside_scope'] = memory_inside_scope
            if a.candidate == 'production-gain':
                row['original_steady_allocated_bytes'] = baseline_current
                row['memory_gate'] = {'install_delta_exact': receipt['install_allocated_delta_bytes'] == receipt['gain_bytes'], 'close_releases_exact_gain_bytes': receipt['close_releases_exact_gain_bytes'], 'steady_reference_policy': 'First correctness-accepted original warmup establishes post-close reference; every later pre-install and post-close allocation must equal it.', 'peak_and_live_pair_checks': 'Checked after both pair observations; warmup checked against original warmup.'}
            row['memory_boundaries'] = {'memory_inside_scope': 'After generation timer stops, before norm context exit and full generated-token CPU readback; live gains retained for gain variants, same point for hybrid. Allocator peak since this request reset.', 'memory': ('After excluded full-token readback/decode, then restoration' if a.candidate == 'production-gain' else 'After restoration, then excluded full-token readback/decode') + (', output writes and correctness checks; peak includes teardown since adapter reset before scope close. Request peak is memory_inside_scope; allocated bytes exclude gains.' if a.candidate == 'production-gain' else ', output writes and correctness checks; request allocator peak. Allocated bytes can exclude gains.')}
            if reference_path is None:
                assert mode == 'hybrid'
                row['historical_qualified_comparison'] = compare(a.qualified_reference, path)
                row['historical_usage_and_stop_equal'] = qualified['usage'] == final['usage'] and qualified['choices'][0]['finish_reason'] == final['finish_reason']
                if not row['historical_qualified_comparison']['exact_html_equal'] or not row['historical_usage_and_stop_equal']: raise RuntimeError('First hybrid warmup differs from independent qualified anchor')
                reference_path = path; reference_ids = generated
            comparison = compare(reference_path, path); row['comparison_to_first_hybrid_warmup'] = comparison; row['exact_generated_token_ids_equal'] = generated == reference_ids
            report['runs'].append(row); save(); print(json.dumps({'run': label, 'variant': mode, 'elapsed_seconds': elapsed, 'correctness': correctness, 'interception_passed': interception}), flush=True)
            if not correctness['passed'] or not interception or not row['exact_generated_token_ids_equal'] or not comparison['exact_html_equal']: raise RuntimeError('Paired content/tokens/interception differs; review preserved geometry before timing promotion')
            if a.candidate == 'production-gain':
                if baseline_current is None:
                    if label != 'warmup-hybrid' or mode != 'hybrid' or measured: raise RuntimeError('Steady reference must follow first accepted original warmup')
                    baseline_current = receipt['memory_after_close']['allocated_bytes']
                    row['memory_gate']['first_original_warmup_allocation_delta_bytes'] = baseline_current - receipt['memory_before_install']['allocated_bytes']
                    report['steady_allocation_reference'] = {'allocated_bytes': baseline_current, 'established_after': label, 'scope': 'After correctness-accepted excluded original warmup and scope close; body allocation changes retained, not attributed to gains.'}
                else:
                    row['memory_gate']['steady_allocation'] = steady_allocation_gate(receipt, baseline_current)
                row['original_steady_allocated_bytes'] = baseline_current
                save()

        def memory_pair(original, trial):
            delta = trial['memory_inside_scope']['allocated_bytes'] - original['memory_inside_scope']['allocated_bytes']
            peak = trial['memory_inside_scope']['peak_allocated_bytes'] - original['memory_inside_scope']['peak_allocated_bytes']
            result = {'live_allocated_delta_bytes': delta, 'peak_allocated_delta_bytes': peak, 'expected_gain_bytes': EXPECTED_GAIN_BYTES, 'live_current_exact': delta == EXPECTED_GAIN_BYTES, 'peak_within_original_plus_gains': peak <= EXPECTED_GAIN_BYTES, 'rounding_allowance_bytes': 0}
            trial['memory_gate']['paired_request'] = result; save()
            if not result['live_current_exact'] or not result['peak_within_original_plus_gains']: raise RuntimeError('Prospective production memory gate failed; keep receipts and stop')
        run('warmup-hybrid', 'hybrid', False); run('warmup-gain', a.candidate, False)
        if a.candidate == 'production-gain': memory_pair(report['runs'][-2], report['runs'][-1])
        for index in range(a.pairs):
            modes = [a.candidate, 'hybrid'] if bool(index % 2) != a.start_gain else ['hybrid', a.candidate]
            for mode in modes: run(f'pair-{index}-{mode}', mode, True, index)
            if a.candidate == 'production-gain':
                pair_rows = {r['variant']: r for r in report['runs'] if r['measured_warm_run'] and r['pair'] == index}
                memory_pair(pair_rows['hybrid'], pair_rows[a.candidate])
    except BaseException as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}; raise
    finally:
        if original_generate and backend.model is not None: backend.model.generate = original_generate
        original_generate = None; captured.clear()
        # Do not retain loaded module/weight references through final unload.
        norms = []; backend.unload()
        if torch.xpu.is_initialized():
            import gc
            gc.collect(); torch.xpu.synchronize(); torch.xpu.empty_cache()
        report['unloaded_health'] = backend.health(); report['unload_memory'] = {'allocated_bytes': torch.xpu.memory_allocated(), 'reserved_bytes': torch.xpu.memory_reserved()} if torch.xpu.is_initialized() else {'allocated_bytes': 0, 'reserved_bytes': 0}; report['unload_zero_memory'] = all(v == 0 for v in report['unload_memory'].values()); save()
        if not report['unload_zero_memory']: raise RuntimeError('Owned process retained XPU allocation after unload')


if __name__ == '__main__': main()
