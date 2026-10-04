"""Run the existing exact-input exporter under the isolated candidate scope."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

from attention_candidate import masked_text_sdpa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--mask-kind", choices=["boolean", "additive"], default="boolean")
    parser.add_argument("--fixture-limit", type=int, default=0, help="Save this many distinct trained attention shapes from the first full-attention layer")
    parser.add_argument("export_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.fixture_limit <= 16:
        parser.error("fixture-limit must be 0..16")
    remainder = args.export_args
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if "--attention" not in remainder or remainder[remainder.index("--attention") + 1] != "hybrid":
        parser.error("Pass --attention hybrid to keep upstream eager-mask construction")
    if "--device" not in remainder or remainder[remainder.index("--device") + 1] != "xpu":
        parser.error("Pass --device xpu; no CPU fallback")
    output = Path(remainder[remainder.index("--output") + 1])
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    wrapper = root / "scripts/waystone/run_export.py"
    exporter = root / "verification/export_logits.py"
    sys.argv = [str(wrapper), str(exporter), *remainder]
    fixtures = []
    seen = set()
    first_layer = None
    fixture_bytes = 0

    def capture(module, query, key, value, additive, effective):
        nonlocal first_layer, fixture_bytes
        if first_layer is None:
            first_layer = module.layer_idx
        shape = (query.shape[-2], key.shape[-2])
        if module.layer_idx != first_layer or shape in seen or len(fixtures) >= args.fixture_limit:
            return
        from safetensors.torch import save_file
        path = output / f"trained-attention-{len(fixtures):02d}.safetensors"
        tensors = {
            "query": query, "key": key, "value": value,
            "additive_mask": additive, "effective_mask": effective,
        }
        byte_count = sum(t.numel() * t.element_size() for t in tensors.values())
        if query.shape[0] != 1 or shape[0] > 4096 or shape[1] > 16384 or byte_count > 64 * 1024**2 or fixture_bytes + byte_count > 128 * 1024**2:
            raise RuntimeError("Trained fixture capture shape/128MiB aggregate allowance breached")
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}, str(path))
        fixture_bytes += byte_count
        fixtures.append({"file": path.name, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(), "tensor_bytes": byte_count, "layer": first_layer, "query_length": shape[0], "kv_length": shape[1], "scale": module.scaling, "kv_groups": module.num_key_value_groups})
        seen.add(shape)

    with masked_text_sdpa(mask_kind=args.mask_kind, expected_graph_sha256=args.graph_sha256, fixture=capture if args.fixture_limit else None) as receipt:
        try:
            runpy.run_path(str(wrapper), run_name="__main__")
        finally:
            receipt["source_sha256"] = {
                path.name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                for path in (Path(__file__), Path(__file__).with_name("attention_candidate.py"))
            }
            receipt["trained_fixtures"] = fixtures
            if output.exists():
                (output / "attention-candidate.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if not receipt["calls"]:
        raise RuntimeError("No text attention call was intercepted")


if __name__ == "__main__":
    main()
