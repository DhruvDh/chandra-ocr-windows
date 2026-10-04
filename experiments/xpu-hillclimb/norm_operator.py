"""Bounded trained-input RMSNorm replay; explicit XPU lease required."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from norm_cpu_reference import read_tensor
from norm_candidate import check_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['gain-only', 'fused'], required=True)
    args = parser.parse_args()
    directory = Path(args.fixtures)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('coverage_complete') or len(manifest['fixtures']) != 162:
        raise RuntimeError('Incomplete trained fixture coverage')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    torch, _ = check_graph()
    if not torch._C._dispatch_has_kernel_for_dispatch_key('aten::_fused_rms_norm', 'XPU'):
        raise RuntimeError('Native XPU RMSNorm kernel unavailable')
    rows = []
    receipt = {'schema': 'chandra.norm.xpu-operator.v1', 'mode': args.mode, 'fixtures': rows,
               'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                   Path(__file__), Path(__file__).with_name('norm_cpu_reference.py'),
                   Path(__file__).with_name('norm_candidate.py')]},
               'scope': 'Bounded selected trained rows; full original tensor layouts require model-level gates'}
    with torch.inference_mode():
        for index, fixture in enumerate(manifest['fixtures']):
            x = torch.from_numpy(read_tensor(directory, fixture['input']).copy()).to('xpu:0', torch.bfloat16)
            weight = torch.from_numpy(read_tensor(directory, fixture['weight']).copy()).to('xpu:0', torch.bfloat16)
            recorded_gain = read_tensor(directory, fixture['gain'])
            reference = read_tensor(directory, fixture['reference_output'])
            gain = 1.0 + weight.float()
            gain_host = gain.cpu().numpy()
            gain_equal = np.array_equal(gain_host.view(np.uint32), recorded_gain.view(np.uint32))
            fp32 = x.float()
            if args.mode == 'gain-only':
                normalized = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + fixture['eps'])
            else:
                normalized = torch.rms_norm(fp32, (x.shape[-1],), weight=None, eps=fixture['eps'])
            result = (normalized * gain).to(torch.bfloat16).float().cpu().numpy()
            torch.xpu.synchronize()
            if not np.isfinite(result).all():
                raise RuntimeError('Nonfinite operator output')
            error = result.astype(np.float64) - reference.astype(np.float64)
            raw = result.astype('<f4').tobytes()
            file = f'candidate-{index:03d}.f32'
            (output / file).write_bytes(raw)
            rows.append({'module': fixture['module'], 'forward_index': fixture['forward_index'],
                         'gain_bit_equal': gain_equal, 'output_bit_equal': bool(np.array_equal(result, reference)),
                         'max_abs': float(np.max(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error * error))),
                         'file': file, 'sha256': hashlib.sha256(raw).hexdigest()})
            (output / 'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    receipt.update(gain_bit_equal_all=all(row['gain_bit_equal'] for row in rows),
                   output_bit_equal_all=all(row['output_bit_equal'] for row in rows))
    (output / 'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'mode': args.mode, 'fixtures': len(rows),
                      'gain_bit_equal_all': receipt['gain_bit_equal_all'],
                      'output_bit_equal_all': receipt['output_bit_equal_all'],
                      'maximum_abs_error': max(row['max_abs'] for row in rows)}), flush=True)
    if not receipt['gain_bit_equal_all'] or (args.mode == 'gain-only' and not receipt['output_bit_equal_all']):
        raise RuntimeError('Exact-arithmetic operator gate failed')


if __name__ == '__main__':
    main()
