"""Source-only prospective production-gain experiment scope, identity v1.

Calls the actual runtime.waystone.norm_gain.install_gain_cache; duplicates no
normalization arithmetic. Stage that pinned production file and this adapter
alongside the historical qualified backend snapshot for anchored experiments.
Do not bypass the historical backend hash check to admit an unreviewed backend.
Original and production timing variants use identical per-instance call-count
wrappers. Their overhead belongs to measured request time. Setup, synchronized
allocator snapshots and full-token readback/checks do not. Numerical/full OCR
qualification must precede a separate five-pair prospective timing experiment.
Close must release exactly the owned gains, independently of body allocations.
The first accepted excluded original warmup establishes the fixed post-close
steady reference; all later pre-install/post-close allocations must equal it.
The legacy candidate and its incomplete two-pair trial remain unqualified.

Importing this module uses only stdlib. Runtime access occurs only when a public
scope is executed under a future explicit root lease. Private _scope is a
stdlib fake-model test seam, not an alternate production implementation.
"""
import contextlib
import hashlib
from pathlib import Path
import types

IDENTITY = 'production-gain-v1'
PRODUCTION_SHA256 = '31044117fbe5d02399f2cb9cf484c0fc67cc7666c4a042d8b379328820d9c02f'
EXPECTED_GAIN_BYTES = 681984


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _allocated(snapshot):
    return snapshot['allocated_bytes']


@contextlib.contextmanager
def _scope(modules, installer, model, synchronize, snapshot, *, production, reset_peak=None):
    """Equal counting instrumentation; fake dependencies permit stdlib tests."""
    if len(modules) != 81 or len({name for name, _ in modules}) != 81:
        raise RuntimeError('Expected exactly 81 distinct norm modules')
    prior = [(module, 'forward' in module.__dict__, module.__dict__.get('forward')) for _, module in modules]
    receipt = {'identity': IDENTITY if production else 'counted-original-v1', 'mode': 'production-gain' if production else 'hybrid', 'module_count': 81, 'calls': {name: 0 for name, _ in modules}, 'gain_bytes': 0, 'restored': False, 'memory_gate': {'expected_gain_bytes': EXPECTED_GAIN_BYTES if production else 0, 'allocator_rounding_allowance_bytes': 0, 'scope': 'Synchronized Torch allocated bytes, not driver VRAM. Request peak compared separately against paired original peak plus gain bytes.'}}
    owner = None; wrapped = []; original = None; counted = None
    try:
        synchronize(); receipt['memory_before_install'] = snapshot()
        if reset_peak is not None: reset_peak()
        if production:
            owner = installer(model)
            receipt['gain_bytes'] = owner.gain_bytes
            if owner.module_count != 81 or owner.gain_bytes != EXPECTED_GAIN_BYTES:
                raise RuntimeError('Production norm inventory or logical gain bytes differs')
        for name, module in modules:
            original = module.forward
            wrapped.append((module, original))
            def counted(self, x, *, _original=original, _name=name):
                receipt['calls'][_name] += 1
                return _original(x)
            module.forward = types.MethodType(counted, module)
        original = None; counted = None
        synchronize(); receipt['memory_after_install'] = snapshot()
        delta = _allocated(receipt['memory_after_install']) - _allocated(receipt['memory_before_install'])
        receipt['install_allocated_delta_bytes'] = delta
        if delta != receipt['gain_bytes']:
            raise RuntimeError('Synchronized installed allocation differs; no rounding allowance')
        yield receipt
    finally:
        original = None; counted = None
        synchronize(); receipt['memory_before_close'] = snapshot()
        if reset_peak is not None: reset_peak()
        # Restore direct instance attribute semantics before production owner
        # removes its overrides and cached gain closures.
        if owner is not None:
            for module, installed in reversed(wrapped):
                module.forward = installed
            installed = None; wrapped.clear()
            owner.close()
        else:
            for module, had_forward, previous in reversed(prior):
                if had_forward:
                    module.forward = previous
                elif 'forward' in module.__dict__:
                    delattr(module, 'forward')
            wrapped.clear()
        receipt['restored'] = all(('forward' in module.__dict__) == had and (not had or module.__dict__.get('forward') is previous) for module, had, previous in prior)
        synchronize(); receipt['memory_after_close'] = snapshot()
        receipt['close_released_allocated_bytes'] = _allocated(receipt['memory_before_close']) - _allocated(receipt['memory_after_close'])
        receipt['close_releases_exact_gain_bytes'] = receipt['close_released_allocated_bytes'] == receipt['gain_bytes']
        modules = []; prior = []; owner = None
        if not receipt['restored']:
            raise RuntimeError('Original per-instance norm state not restored')
        if not receipt['close_releases_exact_gain_bytes']:
            raise RuntimeError('Scope close did not release exactly its owned gain bytes')


def _runtime_dependencies(model):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as graph
    modules = [(name, mod) for name, mod in model.named_modules() if type(mod) is graph.Qwen3_5RMSNorm]
    def snapshot():
        return {'allocated_bytes': torch.xpu.memory_allocated(), 'reserved_bytes': torch.xpu.memory_reserved(), 'peak_allocated_bytes': torch.xpu.max_memory_allocated()}
    return modules, torch.xpu.synchronize, snapshot, torch.xpu.reset_peak_memory_stats


@contextlib.contextmanager
def production_gain_scope(model):
    from runtime.waystone import norm_gain
    if sha(norm_gain.__file__) != PRODUCTION_SHA256:
        raise RuntimeError('Production module differs from prospective frozen identity')
    modules, synchronize, snapshot, reset_peak = _runtime_dependencies(model)
    with _scope(modules, norm_gain.install_gain_cache, model, synchronize, snapshot, production=True, reset_peak=reset_peak) as receipt:
        receipt['source_sha256'] = {'runtime/waystone/norm_gain.py': sha(norm_gain.__file__), 'experiments/xpu-hillclimb/production_gain_adapter.py': sha(__file__)}
        yield receipt


@contextlib.contextmanager
def counted_original_scope(model):
    modules, synchronize, snapshot, reset_peak = _runtime_dependencies(model)
    with _scope(modules, None, model, synchronize, snapshot, production=False, reset_peak=reset_peak) as receipt:
        receipt['source_sha256'] = {'experiments/xpu-hillclimb/production_gain_adapter.py': sha(__file__)}
        yield receipt


def steady_allocation_gate(receipt, warmed_reference):
    """Body allocations are distinct from owned gains; reject later drift."""
    result = {'warmed_reference_allocated_bytes': warmed_reference,
              'pre_install_equal': _allocated(receipt['memory_before_install']) == warmed_reference,
              'post_close_equal': _allocated(receipt['memory_after_close']) == warmed_reference}
    if not all(result[k] for k in ('pre_install_equal', 'post_close_equal')):
        raise RuntimeError('Allocation drift from accepted first original warmup')
    return result


def token_reference_comparison(generated, terminal, stops, reference):
    """Unavailable historical IDs remain unavailable; present IDs are exact."""
    if reference is None:
        return {'available': False, 'exact_equal': None, 'scope': 'Historical full token IDs unavailable; not an acceptance claim.'}
    return {'available': True, 'exact_equal': generated == reference['token_ids'] and terminal == reference['terminal_id'] and stops == reference['stop_ids']}


def persist_token_evidence(path, generated, terminal, stops, decoded, streamed, reference):
    """Persist complete failure evidence before caller enforces acceptance."""
    import json
    comparison = token_reference_comparison(generated, terminal, stops, reference)
    decode_equal = decoded == streamed
    evidence = {'token_ids': generated, 'terminal_id': terminal, 'stop_ids': stops,
                'decoded_content': decoded, 'streamed_content': streamed,
                'complete_decode_equals_stream': decode_equal,
                'retained_token_comparison': comparison,
                'failure_flags': {'decode_mismatch': not decode_equal,
                                  'retained_tokens_mismatch': comparison['available'] and not comparison['exact_equal']}}
    Path(path).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    return evidence
