"""Opt-in compiler-free zero-centered RMSNorm experiments; never serving defaults."""
from __future__ import annotations
import contextlib
import hashlib
import inspect
import types

GRAPH_SHA256 = 'd0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939'


def check_graph():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as graph
    path = inspect.getfile(graph)
    with open(path, 'rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != GRAPH_SHA256:
        raise RuntimeError(f'Unqualified installed graph source: {actual}')
    if torch.__version__ != '2.14.1+xpu':
        raise RuntimeError(f'Unqualified PyTorch runtime: {torch.__version__}')
    return torch, graph


@contextlib.contextmanager
def candidate(model, *, mode='gain-only'):
    """Patch only zero-centered norms in an already loaded, frozen eval model.

    gain-only keeps the exact original normalization arithmetic. fused replaces
    only the unweighted FP32 normalization with native XPU RMSNorm; gain and the
    BF16 output boundary remain separate. Gated RMSNorm is never intercepted.
    """
    if mode not in {'gain-only', 'fused'}:
        raise ValueError('Choose gain-only or fused explicitly')
    torch, graph = check_graph()
    if mode == 'fused' and not torch._C._dispatch_has_kernel_for_dispatch_key('aten::_fused_rms_norm', 'XPU'):
        raise RuntimeError('Locked runtime lacks native fused RMSNorm XPU kernel')
    modules = [(name, mod) for name, mod in model.named_modules() if type(mod) is graph.Qwen3_5RMSNorm]
    if len(modules) != 81:
        raise RuntimeError(f'Expected all 81 pinned zero-centered norms, found {len(modules)}')
    if model.training or any(mod.training for _, mod in modules):
        raise RuntimeError('Candidate requires frozen model.eval() inference')
    saved = []
    receipt = {'mode': mode, 'graph_sha256': GRAPH_SHA256, 'module_count': len(modules),
               'calls': {}, 'gain_bytes': 0, 'restored': False}
    try:
        for name, mod in modules:
            weight = mod.weight
            if weight.device.type != 'xpu' or weight.dtype != torch.bfloat16:
                raise RuntimeError('Candidate requires unchanged BF16 XPU weights')
            version = weight._version
            # Creation itself must be explicit inference arithmetic; no model
            # parameter is replaced, mutated, quantized or copied to CPU.
            with torch.inference_mode():
                gain = 1.0 + weight.float()
            receipt['gain_bytes'] += gain.numel() * gain.element_size()
            receipt['calls'][name] = 0
            original = mod.forward
            def forward(self, x, *, _gain=gain, _weight=weight, _version=version, _name=name):
                if self.training or torch.is_grad_enabled():
                    raise RuntimeError('Norm candidate requires inference_mode and eval')
                if self.weight is not _weight or self.weight._version != _version:
                    raise RuntimeError('Frozen norm weight changed; cached gain invalid')
                if x.device != _weight.device or x.dtype != torch.bfloat16:
                    raise RuntimeError('Unexpected norm input device or dtype')
                if x.shape[-1] != _weight.numel():
                    raise RuntimeError('Unexpected norm input width')
                receipt['calls'][_name] += 1
                fp32 = x.float()
                if mode == 'gain-only':
                    normalized = self._norm(fp32)
                else:
                    # Passing no gain preserves the original post-normalization
                    # multiplication and its separate FP32 rounding boundary.
                    normalized = torch.rms_norm(fp32, (_weight.numel(),), weight=None, eps=self.eps)
                return (normalized * _gain).type_as(x)
            saved.append((mod, original))
            mod.forward = types.MethodType(forward, mod)
        yield receipt
    finally:
        for mod, original in reversed(saved):
            mod.forward = original
        saved.clear()
        receipt['restored'] = True


@contextlib.contextmanager
def capture_trained_norms(model, output, forward_index, *, max_rows=16):
    """Capture every applicable norm on actual prefill and first cached forward.

    forward_index is a callable maintained by a hook on the complete model;
    return 0 during prefill and 1 during the first cached decoding forward.
    The original norm implementation computes all model outputs unchanged.
    """
    import json
    from pathlib import Path
    torch, graph = check_graph()
    if max_rows < 1:
        raise ValueError('max_rows must be positive')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    modules = [(name, mod) for name, mod in model.named_modules() if type(mod) is graph.Qwen3_5RMSNorm]
    if len(modules) != 81 or model.training:
        raise RuntimeError('Expected frozen pinned model and all 81 zero-centered norms')
    receipt = {'schema': 'chandra.trained.norm.v1', 'graph_sha256': GRAPH_SHA256,
               'module_count': len(modules), 'fixtures': [], 'restored': False}
    saved, seen = [], set()
    def tensor_file(tensor, name):
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        (out / name).write_bytes(raw)
        return {'file': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    try:
        for ordinal, (name, mod) in enumerate(modules):
            original = mod.forward
            def forward(self, x, *, _original=original, _ordinal=ordinal, _name=name):
                result = _original(x)
                index = forward_index()
                key = (_ordinal, index)
                if index in {0, 1} and key not in seen:
                    seen.add(key)
                    flat = x.reshape(-1, x.shape[-1])
                    rows = torch.linspace(0, flat.shape[0] - 1, min(max_rows, flat.shape[0]),
                                          dtype=torch.long, device='cpu').unique()
                    # Explicit capture copies are instrumentation, never timings.
                    indices = rows.to(x.device)
                    prefix = f'norm-{_ordinal:03d}-forward-{index}'
                    with torch.inference_mode():
                        gain = 1.0 + self.weight.float()
                    receipt['fixtures'].append({'module': _name, 'forward_index': index,
                        'original_shape': list(x.shape), 'rows': rows.tolist(), 'eps': self.eps,
                        'input': tensor_file(flat.index_select(0, indices), prefix + '.input.bf16'),
                        'weight': tensor_file(self.weight, prefix + '.weight.bf16'),
                        'gain': tensor_file(gain, prefix + '.gain.f32'),
                        'reference_output': tensor_file(result.reshape(-1, result.shape[-1]).index_select(0, indices),
                                                        prefix + '.output.bf16')})
                    (out / 'manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
                return result
            saved.append((mod, original))
            mod.forward = types.MethodType(forward, mod)
        yield receipt
    finally:
        for mod, original in reversed(saved):
            mod.forward = original
        receipt['restored'] = True
        receipt['coverage_complete'] = len(seen) == 162
        (out / 'manifest.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
