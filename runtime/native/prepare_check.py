"""Root acceptance of the real request-preparation path: pinned CPU processor only, no worker, Job or GPU.

Prepares one image and prompt exactly as the endpoint would into a fresh directory and optionally compares
every raw tensor commitment with a retained input-manifest.json (for example the closed tiny package).
"""
import argparse
import base64
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--image", type=Path, required=True, help="PNG, JPEG or WebP request image (already client-scaled)")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; created exclusively")
    parser.add_argument("--compare-manifest", type=Path)
    args = parser.parse_args()
    from .config import load_config
    from .prepare import Pinned, PinnedProcessor, accelerators_initialized, prepare
    config = load_config(args.config, args.config_sha256)
    suffix = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}[args.image.suffix.lower()]
    request = {"max_tokens": 12384, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{suffix};base64," + base64.b64encode(args.image.read_bytes()).decode()}},
        {"type": "text", "text": args.prompt.read_text(encoding="utf-8")}]}]}
    args.output.mkdir(mode=0o700)
    processor = PinnedProcessor(config.model_dir, config.processor_dll_bootstrap)
    prepared = prepare(request, processor, Pinned.load(config.caller_source, config.model_dir), args.output, config.max_image_pixels, config.max_patch_rows)
    manifest = json.loads((args.output / prepared.manifest).read_bytes())
    summary = {"manifest_sha256": prepared.manifest_sha256, "prompt_tokens": len(prepared.prompt_ids), "patch_rows": prepared.patch_rows,
               "image_grid_thw": list(prepared.grid), "image": manifest["image"], "runtime": manifest["runtime"],
               "tensors": {k: v["sha256"] for k, v in manifest["tensors"].items()}, "accelerators_initialized": accelerators_initialized(processor._torch),
               "worker_started": False, "native_qualification": False}
    if args.compare_manifest:
        retained = json.loads(args.compare_manifest.read_bytes())
        summary["compared"] = {k: retained["tensors"].get(k, {}).get("sha256") == v["sha256"] for k, v in manifest["tensors"].items()}
        summary["compared"]["prompt_tokens"] = retained.get("prompt_tokens") == len(prepared.prompt_ids)
        summary["compared"]["pixels_sha256"] = retained.get("image", {}).get("pixels_sha256") == manifest["image"]["pixels_sha256"]
        summary["compared"]["processor_profile"] = retained.get("processor_profile") == manifest["processor_profile"]
    print(json.dumps(summary, indent=2))
    return 0 if all(summary.get("compared", {}).values()) and not any(summary["accelerators_initialized"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
