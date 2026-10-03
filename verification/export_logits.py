"""Exact-input full-vocabulary teacher-forced numerical export (CPU FP32 or XPU BF16)."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:
    resource = None
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
PROFILE = {"size": {"shortest_edge": 3136, "longest_edge": 3145728}}

def peak_rss_kib():
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize // 1024

def cpu_memory(admission=False):
    if platform.system() != "Linux":
        return None
    def fields(path):
        return {line.split(":",1)[0]:int(line.split()[1]) for line in Path(path).read_text().splitlines() if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()}
    system = fields("/proc/meminfo")
    process = fields("/proc/self/smaps_rollup")
    result = {"available_kib":system["MemAvailable"],"rss_kib":process["Rss"],"anonymous_kib":process["Anonymous"],"file_pss_kib":process["Pss_File"],"swap_kib":process["Swap"]}
    gib = 1024**2
    if result["available_kib"] < (30 if admission else 4)*gib or result["rss_kib"] > 32*gib or result["anonymous_kib"] > 24*gib:
        raise RuntimeError(f"CPU numerical run memory budget breached: {result}")
    return result

def sha(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()

def tensor_record(t):
    c = t.detach().cpu().contiguous()
    return {"shape": list(c.shape), "dtype": str(c.dtype), "finite": bool(torch.isfinite(c).all()), "sha256": hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()}

def states(cache):
    records = {}
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            values = list(value.values()) if isinstance(value,dict) else value if isinstance(value, (list, tuple)) else [value]
            for j, tensor in enumerate(values):
                if isinstance(tensor, torch.Tensor):
                    records[f"{i}.{name}.{j}"] = tensor_record(tensor)
    return records

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--response", required=True, help="North OpenAI response JSON or plain UTF-8 text")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--no-cache-check", action="store_true")
    p.add_argument("--activation-fixture", action="store_true", help="Export first trained linear-attention prefill inputs/output/state")
    p.add_argument("--full-teacher", action="store_true", help="One full-response forward, selected full-vocabulary rows across response")
    p.add_argument("--all-teacher", action="store_true", help="Every full-response conditional vocabulary row")
    p.add_argument("--attention", choices=["eager","sdpa"],default="eager")
    p.add_argument("--precision", choices=["float32","bfloat16"], help="Explicit precision; defaults CPU FP32 / XPU BF16")
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=False)
    modeldir = Path(a.model)
    dtype = getattr(torch,a.precision) if a.precision else torch.float32 if a.device == "cpu" else torch.bfloat16
    processor = AutoProcessor.from_pretrained(modeldir, local_files_only=True)
    processor.image_processor.size = PROFILE["size"]
    prompt = Path(a.prompt).read_text()
    image = Image.open(a.image).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    response = Path(a.response).read_text()
    try:
        response = json.loads(response)["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    targets = processor.tokenizer.encode(response, add_special_tokens=False)[:a.steps]
    prefix = inputs["input_ids"][0].tolist()
    meta = {"schema_version": 1, "context": {"vocabulary_size": 248320, "model_revision": REVISION, "tokenizer_sha256": sha(modeldir / "tokenizer.json"), "config_sha256": sha(modeldir / "config.json"), "image_sha256": sha(a.image), "prompt_sha256": sha(a.prompt), "processor_profile": PROFILE, "positions": list(range(len(prefix)-1, len(prefix)+len(targets))), "prefix_token_ids": prefix, "target_token_ids": targets}, "inputs": {k: tensor_record(v) for k,v in inputs.items()}, "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(), "device": a.device, "dtype": str(dtype), "threads": 8, "attention": a.attention, "model_source_sha256": sha(inspect.getfile(Qwen3_5ForConditionalGeneration)), "script_sha256": sha(__file__)}, "response_sha256": sha(a.response), "states": []}
    (outdir / "admission.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"phase": "prepared", "input_shapes": {k:list(v.shape) for k,v in inputs.items()}, "targets": targets}), flush=True)
    start = time.monotonic()
    if a.device == "cpu":
        meta["memory_admission"] = cpu_memory(admission=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(modeldir, dtype=dtype, device_map=a.device, local_files_only=True, attn_implementation=a.attention).eval()
    if a.device == "cpu":
        meta["memory_after_load"] = cpu_memory()
    if model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr():
        raise RuntimeError("Expected tied embedding/head storage")
    inputs = {k:v.to(a.device) for k,v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    if a.full_teacher or a.all_teacher:
        teacher = processor.tokenizer.encode(response, add_special_tokens=False)
        offsets = list(range(len(teacher))) if a.all_teacher else sorted({i for i in [0,7,31,63,127,191,255,319,len(teacher)-1] if i < len(teacher)})
        selected = [len(prefix)-1+i for i in offsets]
        full = dict(inputs)
        full["input_ids"] = torch.cat([inputs["input_ids"],torch.tensor([teacher],device=a.device)],dim=1)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        full["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"],torch.zeros((1,len(teacher)),dtype=inputs["mm_token_type_ids"].dtype,device=a.device)],dim=1)
        with torch.inference_mode():
            result = model(**full,use_cache=True,logits_to_keep=torch.tensor(selected,device=a.device))
        rows = result.logits[0].float().cpu()
        if not torch.isfinite(rows).all():
            raise RuntimeError("Nonfinite sparse teacher logits")
        meta["context"]["positions"] = selected
        meta["context"]["teacher_token_ids"] = teacher
        meta["context"]["target_token_ids"] = [teacher[i] for i in offsets]
        meta["states"] = [states(result.past_key_values)]
        rows.numpy().astype("<f4").tofile(outdir / "logits.f32")
        meta["logits"] = {"file":"logits.f32","shape":list(rows.shape),"dtype":"float32","byte_order":"little","sha256":sha(outdir / "logits.f32")}
        meta["elapsed_seconds"] = time.monotonic()-start
        meta["peak_rss_kib"] = peak_rss_kib()
        if a.device == "cpu":
            meta["memory_completion"] = cpu_memory()
        (outdir / "metadata.json").write_text(json.dumps(meta,indent=2))
        print(json.dumps({"phase":"complete_sparse","elapsed_seconds":meta["elapsed_seconds"],"peak_rss_kib":meta["peak_rss_kib"],"teacher_tokens":len(teacher),"offsets":offsets}),flush=True)
        return
    rows = []
    if a.activation_fixture:
        import transformers.models.qwen3_5.modeling_qwen3_5 as graph
        original = graph.torch_chunk_gated_delta_rule
        captured = False
        def capture(query, key, value, **kwargs):
            nonlocal captured
            result = original(query, key, value, **kwargs)
            if not captured:
                captured = True
                if kwargs.get("initial_state") is not None or not kwargs.get("use_qk_l2norm_in_kernel"):
                    raise RuntimeError("Fixture requires zero-state normalized prefill")
                tensors = {"q": graph.l2norm(query.float(), dim=-1, eps=1e-6) * query.shape[-1]**-0.5, "k": graph.l2norm(key.float(),dim=-1,eps=1e-6), "v":value.float(), "g":kwargs["g"].float(), "beta":kwargs["beta"].float(), "expected_output":result[0].float(), "expected_state":result[1].float()}
                records = {}
                offset = 0
                with open(outdir / "activation.f32", "wb") as payload:
                    for name, tensor in tensors.items():
                        tensor = tensor.squeeze(0).cpu().contiguous()
                        data = tensor.numpy().astype("<f4").tobytes()
                        payload.write(data)
                        records[name] = {"offset_bytes":offset,"byte_length":len(data),"shape":list(tensor.shape),"sha256":hashlib.sha256(data).hexdigest()}
                        offset += len(data)
                fixture = {"schema":"chandra.native.delta.v1","dtype":"float32","byte_order":"little","payload_file":"activation.f32","payload_bytes":offset,"payload_sha256":sha(outdir / "activation.f32"),"tensors":records,"model_revision":REVISION,"source":{"layer":0,"function":"torch_chunk_gated_delta_rule","model_source_sha256":meta["runtime"]["model_source_sha256"]},"semantics":{"q":"l2_normalized_scaled_sqrt_keydim","k":"l2_normalized","g":"negative_log_decay","beta":"post_sigmoid","heads":"expanded_value_heads","initial_state":"zeros","layout":"THD"},"context":meta["context"]}
                (outdir / "activation.json").write_text(json.dumps(fixture,indent=2))
            return result
        graph.torch_chunk_gated_delta_rule = capture
    with torch.inference_mode():
        result = model(**inputs, use_cache=True, logits_to_keep=1)
        for position in range(len(targets)+1):
            logits = result.logits[0,-1].float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite logits")
            rows.append(logits)
            if a.device == "cpu":
                meta.setdefault("memory_positions",[]).append(cpu_memory())
            torch.stack(rows).numpy().astype("<f4").tofile(outdir / "logits.partial.f32")
            meta["states"].append(states(result.past_key_values))
            print(json.dumps({"phase":"position", "position":position, "argmax":int(logits.argmax()), "elapsed_seconds":time.monotonic()-start, "peak_rss_kib":peak_rss_kib()}), flush=True)
            if position < len(targets):
                next_ids = torch.tensor([[targets[position]]], device=a.device)
                extended = torch.cat([inputs["input_ids"], torch.tensor([targets[:position+1]], device=a.device)],dim=1)
                mask = torch.ones_like(extended)
                positions = torch.full((3,1,1), len(prefix)+position,device=a.device,dtype=torch.long) + model.model.rope_deltas
                result = model(input_ids=next_ids, attention_mask=mask, position_ids=positions, past_key_values=result.past_key_values, use_cache=True, logits_to_keep=1)
        if a.no_cache_check:
            uncached = dict(inputs)
            uncached["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([targets],device=a.device)], dim=1)
            uncached["attention_mask"] = torch.ones_like(uncached["input_ids"])
            if "mm_token_type_ids" in uncached:
                uncached["mm_token_type_ids"] = torch.cat([uncached["mm_token_type_ids"],torch.zeros((1,len(targets)),dtype=uncached["mm_token_type_ids"].dtype,device=a.device)],dim=1)
            final = model(**uncached, use_cache=False, logits_to_keep=1).logits[0,-1].float().cpu()
            meta["cache_equivalence"] = {"max_abs":float((final-rows[-1]).abs().max()), "rmse":float((final-rows[-1]).square().mean().sqrt()), "argmax_equal":bool(final.argmax()==rows[-1].argmax()), "finite":bool(torch.isfinite(final).all())}
    torch.stack(rows).numpy().astype("<f4").tofile(outdir / "logits.f32")
    meta["logits"] = {"file":"logits.f32", "shape":[len(rows),248320], "dtype":"float32", "byte_order":"little", "sha256":sha(outdir / "logits.f32")}
    meta["elapsed_seconds"] = time.monotonic()-start
    meta["peak_rss_kib"] = peak_rss_kib()
    (outdir / "metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps({"phase":"complete", "elapsed_seconds":meta["elapsed_seconds"], "peak_rss_kib":meta["peak_rss_kib"]}),flush=True)

if __name__ == "__main__":
    main()
