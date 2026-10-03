"""Run exact synthetic corpus pixels through the direct XPU reference adapter."""
import argparse, base64, hashlib, json, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone import XPUBackend
from runtime.waystone.bootstrap import prepare_runtime
prepare_runtime()
import torch

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--inputs", required=True)
p.add_argument("--prompt", required=True)
p.add_argument("--output", required=True)
p.add_argument("--attention", choices=["eager", "sdpa", "hybrid"], default="sdpa")
p.add_argument("--case", nargs="+", default=["tiny", "small", "representative"])
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
prompt = Path(a.prompt).read_text(encoding="utf-8")
backend = XPUBackend(a.model, attention_backend=a.attention)
t = time.perf_counter(); backend.load(); load_seconds = time.perf_counter() - t
summary = {"load_seconds": load_seconds, "model_verification": backend.health()["model_verification"], "attention": a.attention, "torch": torch.__version__, "cases": [], "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.file_digest(path.open("rb"), "sha256").hexdigest() for path in [Path(__file__), Path(__file__).resolve().parents[2] / "runtime/waystone/backend.py", Path(__file__).resolve().parents[2] / "runtime/waystone/bootstrap.py"]}}
try:
    for case in a.case:
        raw = (Path(a.inputs) / (case + ".png")).read_bytes()
        request = {"model": "chandra", "temperature": 0, "max_tokens": 12384, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}]}]}
        torch.xpu.reset_peak_memory_stats(); start = time.perf_counter(); ttft = None; text = []; final = None; chunks = 0
        print(json.dumps({"phase": "begin", "case": case}), flush=True)
        with (out / (case + ".stream.jsonl")).open("w", encoding="utf-8") as events:
            for part in backend.generate(request, threading.Event()):
                elapsed = time.perf_counter() - start
                events.write(json.dumps({"elapsed": elapsed, "part": part}) + "\n"); events.flush()
                if isinstance(part, str):
                    text.append(part); chunks += 1
                    if ttft is None:
                        ttft = elapsed
                        print(json.dumps({"phase": "first_text", "case": case, "ttft": ttft}), flush=True)
                    elif chunks % 50 == 0:
                        print(json.dumps({"phase": "progress", "case": case, "chunks": chunks, "elapsed": elapsed}), flush=True)
                else:
                    final = part
        if final is None:
            raise RuntimeError("Generation ended without final usage/finish metadata")
        response = {"model": "chandra", "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)}, "finish_reason": final["finish_reason"]}], "usage": final["usage"]}
        (out / (case + ".response.json")).write_text(json.dumps(response, indent=2), encoding="utf-8")
        row = {"case": case, "image_sha256": hashlib.sha256(raw).hexdigest(), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "ttft_seconds": ttft, "elapsed_seconds": time.perf_counter() - start, **final, "memory": {"allocated": torch.xpu.memory_allocated(), "reserved": torch.xpu.memory_reserved(), "peak": torch.xpu.max_memory_allocated()}}
        summary["cases"].append(row)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "complete", **row}), flush=True)
finally:
    backend.unload()
    print(json.dumps({"phase": "unloaded", "allocated": torch.xpu.memory_allocated(), "reserved": torch.xpu.memory_reserved()}), flush=True)
