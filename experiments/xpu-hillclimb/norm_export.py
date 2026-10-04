"""Scoped norm experiment around the existing complete numerical exporter."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from norm_candidate import candidate, capture_trained_norms, check_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norm-mode', choices=['gain-only', 'fused', 'capture', 'production-gain'], required=True)
    parser.add_argument('--norm-fixtures')
    args, export_args = parser.parse_known_args()
    if args.norm_mode == 'capture' and not args.norm_fixtures:
        parser.error('Capture requires a new --norm-fixtures directory')
    if '--output' not in export_args or '--attention' not in export_args:
        parser.error('Supply explicit numerical --output and --attention hybrid')
    if export_args[export_args.index('--attention') + 1] != 'hybrid':
        parser.error('This experiment starts from the qualified hybrid graph only')
    output = Path(export_args[export_args.index('--output') + 1])
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    _, graph = check_graph()
    cls = graph.Qwen3_5ForConditionalGeneration
    original = cls.from_pretrained
    previous = cls.__dict__.get('from_pretrained')
    receipts = []
    def retain_record():
        record = {'mode': args.norm_mode, 'source_sha256': {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in [
                Path(__file__), Path(__file__).with_name('norm_candidate.py')]},
            'candidate_identity': 'production-gain-v1' if args.norm_mode == 'production-gain' else args.norm_mode, 'helper_revision': 2, 'receipts': receipts, 'scope': 'Numerical experiment only; no serving or performance promotion'}
        if args.norm_mode == 'production-gain':
            for name in ['runtime/waystone/norm_gain.py', 'experiments/xpu-hillclimb/production_gain_adapter.py']:
                record['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if output.exists():
            (output / 'norm-experiment.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
    with contextlib.ExitStack() as stack:
        stack.callback(retain_record)
        def load(inner_cls, *positional, **kwargs):
            model = original(*positional, **kwargs).eval()
            if receipts:
                raise RuntimeError('Expected one explicitly scoped model load')
            if args.norm_mode == 'capture':
                index = [-1]
                def before(module, positional, keyword):
                    index[0] += 1
                hook = model.register_forward_pre_hook(before, with_kwargs=True)
                stack.callback(hook.remove)
                receipt = stack.enter_context(capture_trained_norms(model, args.norm_fixtures, lambda: index[0]))
            elif args.norm_mode == 'production-gain':
                from production_gain_adapter import production_gain_scope
                receipt = stack.enter_context(production_gain_scope(model))
            else:
                receipt = stack.enter_context(candidate(model, mode=args.norm_mode))
            receipts.append(receipt)
            return model
        cls.from_pretrained = classmethod(load)
        try:
            sys.argv = [str(ROOT / 'scripts/waystone/run_export.py'),
                        str(ROOT / 'verification/export_logits.py'), *export_args]
            runpy.run_path(str(ROOT / 'scripts/waystone/run_export.py'), run_name='__main__')
        finally:
            if previous is None:
                delattr(cls, 'from_pretrained')
            else:
                cls.from_pretrained = previous
    if not receipts:
        raise RuntimeError('No candidate model was loaded')


if __name__ == '__main__':
    main()
