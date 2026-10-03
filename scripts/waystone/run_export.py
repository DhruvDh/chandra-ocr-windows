"""Run the shared numerical exporter with the verified Windows XPU loader."""
import hashlib
import json
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone.bootstrap import prepare_runtime
prepare_runtime()
import torch
from transformers import modeling_utils
script = Path(sys.argv[1])
args = sys.argv[2:]
output = Path(args[args.index("--output") + 1])
warmup = modeling_utils.caching_allocator_warmup
modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
try:
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")
finally:
    modeling_utils.caching_allocator_warmup = warmup
record = {"torch": torch.__version__, "device": torch.xpu.get_device_name(0), "precision": "bfloat16", "allocator_warmup": "disabled_scoped", "bootstrap_sha256": hashlib.file_digest((Path(__file__).resolve().parents[2] / "runtime/waystone/bootstrap.py").open("rb"), "sha256").hexdigest(), "wrapper_sha256": hashlib.file_digest(Path(__file__).open("rb"), "sha256").hexdigest(), "memory": {"allocated": torch.xpu.memory_allocated(), "reserved": torch.xpu.memory_reserved(), "peak": torch.xpu.max_memory_allocated()}}
(output / "windows-runtime.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record), flush=True)
