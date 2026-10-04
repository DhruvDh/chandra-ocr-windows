"""Exact production-gain screen; cache equality uses exporter digest receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import os

BASELINE_SHA256 = '9b3f3ce5bc73eea10310c76ea9ced7504488621dccf594b84d61f2903b421a3d'
HEX = re.compile(r'[0-9a-f]{64}\Z')
JSON_LIMIT = 16 * 1024 * 1024


def regular_stream(path):
    # Nonblocking open prevents a raced-in FIFO from hanging before fstat.
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('Opened evidence must be regular')
    return os.fdopen(fd, 'rb')


def digest_file(path, expected_bytes=None):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Payload must be a regular file')
    if expected_bytes is None:
        expected_bytes = path.stat().st_size
    h = hashlib.sha256()
    with regular_stream(path) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Opened payload must be regular')
        remaining = expected_bytes
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError('Truncated payload')
            remaining -= len(block); h.update(block)
        if stream.read(1):
            raise ValueError('Trailing payload bytes')
    return h.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path, commitment=None):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > JSON_LIMIT:
        raise ValueError('Invalid or oversized JSON file')
    with regular_stream(path) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Opened JSON must be regular')
        raw = stream.read(JSON_LIMIT + 1)
    if len(raw) > JSON_LIMIT:
        raise ValueError('JSON allowance exceeded')
    if commitment is not None and (not HEX.fullmatch(commitment) or hashlib.sha256(raw).hexdigest() != commitment):
        raise ValueError('JSON commitment differs: ' + path.name)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def exact(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def cache_specs(position):
    result = {}
    for layer in range(32):
        fields = ('keys', 'values') if layer % 4 == 3 else ('conv_states', 'recurrent_states')
        for field in fields:
            shape = [1, 4, position + 1, 256] if field in ('keys', 'values') else [1, 8192, 4] if field == 'conv_states' else [1, 32, 128, 128]
            result[f'{layer}.{field}.0'] = (shape, 'torch.float32' if field == 'recurrent_states' else 'torch.bfloat16')
    return result


def validate_states(states, positions, failures, label):
    if not isinstance(states, list) or len(states) != len(positions):
        failures.append(label + ': cache position coverage'); return
    for row, (state, position) in enumerate(zip(states, positions)):
        specs = cache_specs(position)
        if not isinstance(state, dict) or list(state) != list(specs):
            failures.append(f'{label}: row {row} position {position} cache keys/order'); continue
        for key, (shape, dtype) in specs.items():
            record = state[key]
            if (not isinstance(record, dict) or set(record) != {'shape', 'dtype', 'finite', 'sha256'}
                    or record.get('shape') != shape or any(type(v) is not int for v in record.get('shape', []))
                    or record.get('dtype') != dtype or record.get('finite') is not True
                    or not isinstance(record.get('sha256'), str) or not HEX.fullmatch(record['sha256'])):
                failures.append(f'{label}: row {row} position {position} invalid {key}')


def norm_names():
    names = []
    for layer in range(32):
        prefix = f'model.language_model.layers.{layer}'
        names += [prefix + '.input_layernorm', prefix + '.post_attention_layernorm']
        if layer % 4 == 3:
            names += [prefix + '.self_attn.q_norm', prefix + '.self_attn.k_norm']
    return names + ['model.language_model.norm']


def verify(baseline, candidate, manifest):
    failures = []
    def require(ok, reason):
        if not ok: failures.append(reason)
    require(manifest.get('schema') == 'chandra.production.gain.exact-manifest.v1', 'Frozen manifest schema')
    base = read_json(Path(baseline) / 'metadata.json', BASELINE_SHA256)
    meta = read_json(Path(candidate) / 'metadata.json')
    positions = list(range(800, 1195))
    for label, value in [('baseline', base), ('candidate', meta)]:
        require(value.get('schema_version') == 1, label + ': metadata schema')
        context = value['context']
        require(exact(context['positions'], positions) and len(context['prefix_token_ids']) == 801 and len(context['target_token_ids']) == 394, label + ': conditioned positions')
        validate_states(value['states'], positions, failures, label)
        require(exact(value['logits']['shape'], [395, 248320]) and value['logits']['dtype'] == 'float32' and value['logits']['byte_order'] == 'little' and value['logits']['file'] == 'logits.f32', label + ': logits layout')
    for key in ('context', 'inputs', 'runtime', 'response_sha256', 'cache_equivalence'):
        require(exact(meta[key], base[key]), 'Exact metadata differs: ' + key)
    require(meta['cache_equivalence'].get('finite') is True and meta['cache_equivalence'].get('argmax_equal') is True, 'Final cache check finite/decision')
    for row, (left, right) in enumerate(zip(base['states'], meta['states'])):
        for key in left:
            if not exact(right.get(key), left[key]): failures.append(f'Cache differs: row {row} position {positions[row]} {key}')
    payload = Path(candidate) / 'logits.f32'
    if payload.is_symlink() or payload.stat().st_size != 395 * 248320 * 4:
        raise ValueError('Candidate payload bytes/path')
    actual = digest_file(payload, 395 * 248320 * 4)
    require(actual == meta['logits']['sha256'] == base['logits']['sha256'], 'Candidate full logits digest differs')
    record = read_json(Path(candidate) / 'norm-experiment.json')
    require(record.get('mode') == 'production-gain' and record.get('candidate_identity') == 'production-gain-v1' and record.get('helper_revision') == 2, 'Production exporter identity')
    expected = manifest['norm_source_sha256']; adapter_sources = manifest['adapter_source_sha256']
    require(set(expected) == {'norm_export.py', 'norm_candidate.py', 'runtime/waystone/norm_gain.py', 'experiments/xpu-hillclimb/production_gain_adapter.py'} and all(isinstance(v, str) and HEX.fullmatch(v) for v in expected.values()), 'Frozen norm source inventory')
    require(set(adapter_sources) == {'runtime/waystone/norm_gain.py', 'experiments/xpu-hillclimb/production_gain_adapter.py'} and all(isinstance(v, str) and HEX.fullmatch(v) for v in adapter_sources.values()), 'Frozen adapter source inventory')
    require(all(expected.get(k) == v for k, v in adapter_sources.items()), 'Frozen source maps inconsistent')
    require(record.get('source_sha256') == expected, 'Production wrapper source commitments differ')
    receipts = record.get('receipts', [])
    require(len(receipts) == 1, 'Exactly one production receipt')
    if len(receipts) == 1:
        receipt = receipts[0]
        require(receipt.get('source_sha256') == adapter_sources, 'Adapter executed source differs')
        require(receipt.get('identity') == 'production-gain-v1' and receipt.get('mode') == 'production-gain' and receipt.get('module_count') == 81 and receipt.get('gain_bytes') == 681984 and receipt.get('restored') is True, 'Production scope inventory/restoration')
        calls = receipt.get('calls', {})
        require(set(calls) == set(norm_names()) and all(type(v) is int and v == 396 for v in calls.values()), 'Production norm coverage/counts')
        require(all(type(receipt.get(k)) is int for k in ('module_count', 'gain_bytes', 'install_allocated_delta_bytes', 'close_released_allocated_bytes')), 'Production integer receipt fields')
        require(receipt.get('memory_gate', {}).get('expected_gain_bytes') == 681984 and receipt.get('memory_gate', {}).get('allocator_rounding_allowance_bytes') == 0, 'Production zero-slack memory policy')
        require(receipt.get('install_allocated_delta_bytes') == 681984 and receipt.get('close_released_allocated_bytes') == 681984 and receipt.get('close_releases_exact_gain_bytes') is True, 'Production allocation ownership')
    return {'passed': not failures, 'failures': failures, 'expected_logit_rows': 395, 'expected_cache_records': 25280, 'observed_logit_shape': meta['logits']['shape'], 'observed_cache_positions': len(meta['states']), 'observed_cache_records': sum(len(row) for row in meta['states'] if isinstance(row, dict)), 'candidate_payload_sha256': actual, 'scope': 'Exact logit payload and exporter cache digest receipts; no raw cache replay, GPU attestation or performance promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'manifest', 'manifest-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise ValueError('Failure report must be new')
    try:
        manifest = read_json(args.manifest, args.manifest_sha256)
        result = verify(args.baseline, args.candidate, manifest)
    except Exception as error:
        result = {'passed': False, 'failures': [type(error).__name__ + ': ' + str(error)], 'scope': 'Fail closed; unavailable or malformed evidence.', 'expected_logit_rows': 395, 'expected_cache_records': 25280, 'observed_logit_shape': None, 'observed_cache_positions': None, 'observed_cache_records': None}
    result['manifest_sha256'] = args.manifest_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
