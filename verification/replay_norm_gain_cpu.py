"""Bounded CPU replay of production gain ownership, not XPU qualification.

Run only with the coordinator's CPU test lease in verification/cpu/.venv.
All payload commitments are verified before importing Torch or computing.
"""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import weakref

MANIFEST_SHA256 = '3ad7aabcf4ab9ce21e0d21d248cb2d7e145b98ab230a211df2e7097b90cd46be'
GRAPH_SHA256 = 'd0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939'
REPO = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory():
    names = []
    for index in range(32):
        prefix = f'model.language_model.layers.{index}'
        names.extend((prefix + '.input_layernorm', prefix + '.post_attention_layernorm'))
        if index % 4 == 3:
            names.extend((prefix + '.self_attn.q_norm', prefix + '.self_attn.k_norm'))
    return set(names + ['model.language_model.norm'])


def payload_path(root, record):
    name = record['file']
    if not isinstance(name, str) or Path(name).name != name or Path(name).is_absolute():
        raise ValueError('Fixture path must be a direct child filename')
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError('Fixture symlink or escaped path rejected')
    return path


def payload(root, record):
    path = payload_path(root, record)
    if path.stat().st_size != record['bytes']:
        raise ValueError('Fixture size mismatch: ' + path.name)
    raw = path.read_bytes()
    if digest(raw) != record['sha256']:
        raise ValueError('Fixture hash mismatch: ' + path.name)
    return raw


def verify(manifest_path):
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 1048576:
        raise ValueError('Invalid manifest path or size')
    raw = manifest_path.read_bytes()
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError('Pinned manifest commitment mismatch')
    manifest = json.loads(raw)
    fixtures = manifest['fixtures']
    if (manifest['schema'] != 'chandra.trained.norm.v1' or manifest['graph_sha256'] != GRAPH_SHA256
            or manifest['module_count'] != 81 or not manifest['coverage_complete']
            or not manifest['restored'] or len(fixtures) != 162):
        raise ValueError('Incomplete or unpinned trained capture')
    expected = {(name, step) for name in inventory() for step in (0, 1)}
    if {(f['module'], f['forward_index']) for f in fixtures} != expected:
        raise ValueError('Require each pinned norm at prefill and first cached step exactly once')
    total_bytes = 0
    for fixture in fixtures:
        width = 256 if fixture['module'].endswith(('q_norm', 'k_norm')) else 2560
        shape = fixture['input']['shape']
        if (len(shape) != 2 or type(shape[0]) is not int or not 1 <= shape[0] <= 16
                or shape[1] != width or fixture['eps'] != 1e-6):
            raise ValueError('Unexpected bounded shape or epsilon')
        if len(fixture['rows']) != shape[0] or len(set(fixture['rows'])) != shape[0]:
            raise ValueError('Invalid sampled rows')
        for key in ('input', 'weight', 'gain', 'reference_output'):
            record = fixture[key]
            wanted_shape = [width] if key in ('weight', 'gain') else shape
            wanted_dtype = 'torch.float32' if key == 'gain' else 'torch.bfloat16'
            wanted_bytes = width * (1 if key in ('weight', 'gain') else shape[0]) * (4 if key == 'gain' else 2)
            if record['shape'] != wanted_shape or record['dtype'] != wanted_dtype or record['bytes'] != wanted_bytes:
                raise ValueError('Invalid tensor metadata')
            payload(manifest_path.parent, record)
            total_bytes += wanted_bytes
    if total_bytes > 32 * 1024 * 1024:
        raise ValueError('Fixture byte allowance exceeded')
    return manifest, total_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.absolute()
    manifest, verified_bytes = verify(manifest_path)
    if args.output.exists():
        raise ValueError('Output must be new; historical evidence is never overwritten')
    if importlib.metadata.version('torch') != '2.14.1+cpu':
        raise RuntimeError('Require the locked CPU Torch package 2.14.1+cpu')
    if importlib.metadata.version('transformers') != '5.18.0':
        raise RuntimeError('Require locked Transformers 5.18.0 metadata')
    graph_path = Path(importlib.metadata.distribution('transformers').locate_file(
        'transformers/models/qwen3_5/modeling_qwen3_5.py'))
    graph_raw = graph_path.read_bytes()
    installed_graph_sha = digest(graph_raw)
    if installed_graph_sha != GRAPH_SHA256:
        raise RuntimeError('Installed graph commitment mismatch')
    def norm_ast(tree, class_name):
        cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name)
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_norm')
        return ast.dump(method, include_attributes=False)
    if norm_ast(ast.parse(graph_raw), 'Qwen3_5RMSNorm') != norm_ast(ast.parse(Path(__file__).read_text()), 'Norm'):
        raise RuntimeError('Replay _norm AST differs from pinned graph equation')
    import torch
    if torch.__version__ != '2.14.1+cpu':
        raise RuntimeError('Imported Torch differs from locked CPU package metadata')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(REPO))
    from runtime.waystone.norm_gain import _install

    class Norm(torch.nn.Module):
        def __init__(self, weight, eps):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.eps = eps

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            return (self._norm(x.float()) * (1.0 + self.weight.float())).type_as(x)

    def tensor(record):
        dtype = torch.float32 if record['dtype'] == 'torch.float32' else torch.bfloat16
        return torch.frombuffer(bytearray(payload(manifest_path.parent, record)), dtype=dtype).reshape(record['shape']).clone()

    def raw(value):
        return value.detach().contiguous().view(torch.uint8).numpy().tobytes()

    def compare(left, right):
        bits = left.view(torch.int16) != right.view(torch.int16)
        error = left.float() - right.float()
        return {'bit_equal': not bool(bits.any()), 'different_elements': int(bits.sum()),
                'elements': left.numel(), 'max_abs': float(error.abs().max()),
                'left_sha256': digest(raw(left)), 'right_sha256': digest(raw(right))}

    started = time.monotonic()
    results = []
    for fixture in manifest['fixtures']:
        if time.monotonic() - started > 120:
            raise TimeoutError('Bounded CPU replay exceeded 120 seconds')
        x, weight, captured_gain, captured_output = [tensor(fixture[key]) for key in ('input', 'weight', 'gain', 'reference_output')]
        if not all(bool(torch.isfinite(value).all()) for value in (x, weight, captured_gain, captured_output)):
            raise ValueError('Nonfinite fixture rejected')
        model = torch.nn.ModuleDict({'norm': Norm(weight, fixture['eps'])}).eval()
        norm = model['norm']
        model_ref, norm_ref = weakref.ref(model), weakref.ref(norm)
        with torch.inference_mode():
            ordinary = norm(x)
        owner = _install(model, Norm, ('norm',), device_type='cpu')
        try:
            closure = dict(zip(norm.forward.__func__.__code__.co_freevars, norm.forward.__func__.__closure__))
            gain = closure['gain'].cell_contents
            gain_ref = weakref.ref(gain)
            gain_equal = raw(gain) == raw(captured_gain)
            gain_sha = digest(raw(gain))
            owner_valid = owner.module_count == 1 and owner.gain_bytes == weight.numel() * 4
            with torch.inference_mode():
                production = norm(x)
            arithmetic = compare(production, ordinary)
            xpu = compare(production, captured_output)
            del gain, closure
        finally:
            owner.close()
        restored = 'forward' not in norm.__dict__ and owner.closed and not owner._entries
        owner.close()
        del model, norm
        gc.collect()
        released = model_ref() is None and norm_ref() is None and gain_ref() is None
        results.append({'module': fixture['module'], 'forward_index': fixture['forward_index'],
            'production_vs_ordinary_cpu': arithmetic, 'production_cpu_vs_captured_xpu': xpu,
            'gain_fp32_captured_bytes_equal': gain_equal, 'gain_sha256': gain_sha,
            'owner_inventory_bytes_valid': owner_valid, 'restored': restored, 'ownership_released': released})
        del x, weight, captured_gain, captured_output, ordinary, production, owner
    passed = all(r['production_vs_ordinary_cpu']['bit_equal'] and r['gain_fp32_captured_bytes_equal']
                 and r['owner_inventory_bytes_valid'] and r['restored'] and r['ownership_released'] for r in results)
    receipt = {'schema': 'chandra.production.norm.cpu-replay.v1', 'passed': passed,
        'scope': 'Production _install CPU seam only; captured XPU comparison is descriptive and permits reduction differences. No GPU or full-model qualification.',
        'manifest_sha256': MANIFEST_SHA256, 'graph_sha256': GRAPH_SHA256,
        'installed_graph_sha256': installed_graph_sha, 'transformers_metadata_version': '5.18.0',
        'source_sha256': digest(Path(__file__).read_bytes()),
        'production_source_sha256': digest((REPO / 'runtime/waystone/norm_gain.py').read_bytes()),
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
                    'interop_threads': torch.get_num_interop_threads()},
        'verified_payload_bytes': verified_bytes, 'fixture_count': len(results), 'module_count': 81,
        'max_rows': 16, 'elapsed_seconds': time.monotonic() - started,
        'captured_xpu_bit_equal_fixtures': sum(r['production_cpu_vs_captured_xpu']['bit_equal'] for r in results),
        'captured_xpu_max_abs': max(r['production_cpu_vs_captured_xpu']['max_abs'] for r in results),
        'fixtures': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as destination:
        json.dump(receipt, destination, indent=2)
        destination.write('\n')
    print(json.dumps({'passed': passed, 'fixtures': len(results), 'output': str(args.output)}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
