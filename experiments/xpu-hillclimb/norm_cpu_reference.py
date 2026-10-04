"""Independent bounded FP32 NumPy oracle for actual trained norm captures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read_tensor(directory, record):
    raw = (directory / record['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['sha256'] or len(raw) != record['bytes']:
        raise RuntimeError('Fixture payload commitment mismatch')
    if record['dtype'] == 'torch.bfloat16':
        values = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    elif record['dtype'] == 'torch.float32':
        values = np.frombuffer(raw, dtype='<f4')
    else:
        raise RuntimeError('Unsupported fixture dtype')
    return values.reshape(record['shape'])


def bf16_round(x):
    # Round finite FP32 to BF16, ties to even, independently of PyTorch.
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) >> 16
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.fixtures)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('coverage_complete') or len(manifest['fixtures']) != 162:
        raise RuntimeError('Require all 81 trained norms at prefill and first cached decode')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, fixture in enumerate(manifest['fixtures']):
        x = read_tensor(directory, fixture['input'])
        weight = read_tensor(directory, fixture['weight'])
        gain = read_tensor(directory, fixture['gain'])
        reference = read_tensor(directory, fixture['reference_output'])
        recomputed_gain = np.add(weight, np.float32(1), dtype=np.float32)
        gain_equal = np.array_equal(gain.view(np.uint32), recomputed_gain.view(np.uint32))
        variance = np.mean(np.multiply(x, x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
        inv_rms = np.reciprocal(np.sqrt(np.add(variance, np.float32(fixture['eps']), dtype=np.float32)))
        normalized = np.multiply(x, inv_rms, dtype=np.float32)
        result = bf16_round(np.multiply(normalized, recomputed_gain, dtype=np.float32))
        if not np.isfinite(result).all() or not np.isfinite(reference).all():
            raise RuntimeError('Nonfinite trained norm fixture or oracle result')
        error = result.astype(np.float64) - reference.astype(np.float64)
        raw = result.astype('<f4').tobytes()
        filename = f'oracle-{index:03d}.f32'
        (output / filename).write_bytes(raw)
        results.append({'module': fixture['module'], 'forward_index': fixture['forward_index'],
                        'gain_bit_equal': gain_equal, 'output_equal_fraction': float(np.mean(result == reference)),
                        'max_abs': float(np.max(np.abs(error))), 'rms': float(np.sqrt(np.mean(error * error))),
                        'file': filename, 'sha256': hashlib.sha256(raw).hexdigest()})
    receipt = {'schema': 'chandra.norm.cpu-oracle.v1', 'numpy': np.__version__,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
               'gain_bit_equal_all': all(row['gain_bit_equal'] for row in results), 'fixtures': results,
               'scope': 'Independent FP32 row normalization followed by BF16 rounding; no full-model claim'}
    (output / 'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'fixtures': len(results), 'gain_bit_equal_all': receipt['gain_bit_equal_all'],
                      'maximum_output_abs_error': max(row['max_abs'] for row in results)}))
    if not receipt['gain_bit_equal_all']:
        raise RuntimeError('Precomputed gain is not bit-equivalent to CPU recomputation')


if __name__ == '__main__':
    main()
