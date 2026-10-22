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
import stat
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
PROFILE = {"size": {"shortest_edge": 3136, "longest_edge": 3145728}}
MODEL = "datalab-to/chandra-ocr-2"
INVENTORY = Path(__file__).resolve().parents[1] / "provenance/model.json"
INVENTORY_SHA256 = "5cb10cc7ecd5fa31055ec9f18da92fd8bb0bb0c75803c095f336428fa84e2ee4"  # provenance/model.json (eol=lf): the --custody trust root
EXPORTER, HELPER = "verification/export_logits.py", "verification/cpu_custody.py"
MODULE_CODE = sys._getframe(0).f_code if hasattr(sys, "_getframe") else None  # This module as Python compiled it; --custody binds it to retained bytes.

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

def append_logit_row(path, row):
    """Close each appended little-endian FP32 row before the next forward."""
    with open(path, "ab") as stream:
        row.astype("<f4").tofile(stream)

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

def read_helper(path):
    """The custody helper's bytes, read before any custody code exists: a regular file of at most 4 MiB, opened without following a final link or blocking on a FIFO."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOCTTY", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
            raise OSError(f"{path} is not a regular file of at most 4 MiB")
        data = os.read(fd, info.st_size + 1)
    finally:
        os.close(fd)
    if len(data) != info.st_size:
        raise OSError(f"{path} changed while it was read")
    return data

def open_custody(a):
    """--custody: execute the exact helper bytes it records and refuse, before any output exists, a run the checked route cannot cover."""
    def refused(error):
        print(json.dumps({"phase": "custody_refused", "error": error, "output_created": False}), flush=True)
        raise SystemExit(2)
    here = Path(__file__).resolve().parent
    helper_path = here / "cpu_custody.py"
    try:
        helper_bytes = read_helper(helper_path)
        helper = types.ModuleType("chandra_cpu_custody")
        helper.__file__ = str(helper_path)
        exec(compile(helper_bytes, str(helper_path), "exec", dont_inherit=True), helper.__dict__)
    except (OSError, SyntaxError, ValueError) as error:
        refused(f"The custody helper cannot be read and compiled: {type(error).__name__}: {error}")
    try:
        helper.require(a.device == "cpu" and a.precision in (None, "float32"), "--custody covers CPU FP32 exports only")
        helper.require(not a.activation_fixture, "--custody does not cover --activation-fixture, whose activation.json is accepted before the post-load pass")
        helper.require(not a.greedy_max_tokens, "--custody covers teacher-logit exports; --greedy-max-tokens free generation is outside the checked route")
        exporter_path = Path(__file__).resolve()
        snapshot = helper.read_regular(str(exporter_path), helper.PRODUCER_LIMIT, f"The exporter {exporter_path}")
        producer = {EXPORTER: (exporter_path, snapshot), HELPER: (helper_path, helper_bytes)}
        session = helper.Session(a.model, INVENTORY, a.output, producer, 1800.0 if a.custody_hash_seconds is None else a.custody_hash_seconds,
                                 model=MODEL, revision=REVISION, inventory_sha256=INVENTORY_SHA256, inventory_name="provenance/model.json")
    except helper.CustodyError as error:
        refused(str(error))
    return helper, session

def bootstrap(helper, custody):
    """--custody: require the retained exporter and helper bytes to compile to the code this process is already executing, then return a fresh module executed from the retained exporter compilation, which performs the export."""
    code = custody.bind_execution({EXPORTER: MODULE_CODE, HELPER: helper.MODULE_CODE})[EXPORTER]
    retained = types.ModuleType("chandra_export_logits_retained")
    retained.__file__ = custody.producer[EXPORTER]["path"]
    exec(code, retained.__dict__)
    return retained

def torch_libraries():
    return [Path(torch.__file__).parent / "lib"]

def computation_classes(processor):
    """Classes whose module sources configure, model, process and tokenize this export."""
    classes = {"model": Qwen3_5ForConditionalGeneration, "model_config": getattr(Qwen3_5ForConditionalGeneration, "config_class", None), "auto_processor": AutoProcessor,
               "processor": type(processor), "tokenizer": type(processor.tokenizer), "image_processor": type(processor.image_processor),
               "video_processor": type(getattr(processor, "video_processor", None))}
    def source(c):
        try:
            return inspect.getsourcefile(c)
        except TypeError:  # A class without Python source is named with a null source rather than aborting the run.
            return None
    return {label: (f"{c.__module__}.{c.__qualname__}", source(c)) for label, c in classes.items() if c not in (None, type(None))}

def write_metadata(outdir, meta, custody, payloads, model):
    """Final metadata; under --custody only after the post-load pass matches, through the session's acceptance."""
    if custody is None:
        (outdir / "metadata.json").write_text(json.dumps(meta,indent=2))
        return
    reported = {"model.name_or_path": str(getattr(model, "name_or_path", None)), "model.config._name_or_path": str(getattr(model.config, "_name_or_path", None)), "model.dtype": str(getattr(model, "dtype", None))}
    custody.accept(outdir, meta, payloads, custody.loaded_sources(sys.modules, torch_libraries()), reported)
    print(json.dumps({"phase": "custody_accepted", "metadata_sha256": custody.metadata_sha256, "receipt_sha256": custody.receipt_sha256, "evidence": custody.evidence}), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--response", required=True, help="North OpenAI response JSON or plain UTF-8 text")
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--check-cache-equivalence", "--no-cache-check", dest="no_cache_check", action="store_true",help="Compare final cached logits with a complete uncached forward")
    p.add_argument("--activation-fixture", action="store_true", help="Export first trained linear-attention prefill inputs/output/state")
    p.add_argument("--full-teacher", action="store_true", help="One full-response forward, selected full-vocabulary rows across response")
    p.add_argument("--all-teacher", action="store_true", help="Every full-response conditional vocabulary row")
    p.add_argument("--attention", choices=["eager","sdpa","hybrid"],default="eager")
    p.add_argument("--greedy-max-tokens", type=int, help="Independent complete greedy response, explicit bounded output allowance")
    p.add_argument("--precision", choices=["float32","bfloat16"], help="Explicit precision; defaults CPU FP32 / XPU BF16")
    p.add_argument("--custody", action="store_true", help="Checked route: verify --model against provenance/model.json before loading and again before metadata.json; CPU FP32 only; evidence in <output>.custody")
    p.add_argument("--custody-hash-seconds", type=float, help="Monotonic deadline of each custody hashing pass (default 1800, at most 3600)")
    a = p.parse_args()
    if a.custody_hash_seconds is not None and not a.custody:
        p.error("--custody-hash-seconds requires --custody")
    if not a.custody:
        return run(a, None)
    helper, custody = open_custody(a)
    try:
        custody.begin()
        bootstrap(helper, custody).run(a, custody)
    except BaseException as error:
        custody.refuse(error)
        if isinstance(error, helper.CustodyError):
            print(json.dumps({"phase": "custody_refused", "error": str(error), "evidence": custody.evidence, "metadata_written": os.path.lexists(Path(a.output) / "metadata.json")}), flush=True)
            raise SystemExit(2)
        raise

def run(a, custody):
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=False)
    modeldir = Path(a.model)
    dtype = getattr(torch,a.precision) if a.precision else torch.float32 if a.device == "cpu" else torch.bfloat16
    attention = {"vision_config":"sdpa","text_config":"eager"} if a.attention=="hybrid" else a.attention
    load = custody.load if custody else lambda role, loader, path, **kwargs: loader(path, **kwargs)
    if custody:
        before = custody.verify_before_load()
        custody.declare("processor", AutoProcessor.from_pretrained, modeldir, local_files_only=True)
        custody.declare("model", Qwen3_5ForConditionalGeneration.from_pretrained, modeldir, dtype=dtype, device_map=a.device, local_files_only=True, attn_implementation=attention)
        print(json.dumps({"phase": "custody_verified_before_load", "members": len(before["members"]), "bytes_hashed": before["bytes_hashed"], "elapsed_seconds": before["elapsed_seconds"]}), flush=True)
    processor = load("processor", AutoProcessor.from_pretrained, modeldir, local_files_only=True)
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
    digest = custody.file_sha256 if custody else sha  # Under --custody, the same digests read only regular files, without blocking and within the deadline.
    payload_digest = (lambda name: custody.payload_sha256(outdir, name)) if custody else lambda name: sha(outdir / name)  # Under --custody, read as acceptance reads payloads: in the admitted --output directory, never through a link.
    meta = {"schema_version": 1, "context": {"vocabulary_size": 248320, "model_revision": REVISION, "tokenizer_sha256": digest(modeldir / "tokenizer.json"), "config_sha256": digest(modeldir / "config.json"), "image_sha256": digest(a.image), "prompt_sha256": digest(a.prompt), "processor_profile": PROFILE, "positions": list(range(len(prefix)-1, len(prefix)+len(targets))), "prefix_token_ids": prefix, "target_token_ids": targets}, "inputs": {k: tensor_record(v) for k,v in inputs.items()}, "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(), "device": a.device, "dtype": str(dtype), "threads": 8, "attention": a.attention, "model_source_sha256": digest(inspect.getfile(Qwen3_5ForConditionalGeneration)), "script_sha256": digest(__file__)}, "response_sha256": digest(a.response), "states": []}
    if custody:
        custody.require_digest("tokenizer.json", meta["context"]["tokenizer_sha256"], "context tokenizer_sha256")
        custody.require_digest("config.json", meta["context"]["config_sha256"], "context config_sha256")
        custody.require_producer(EXPORTER, meta["runtime"]["script_sha256"], "runtime script_sha256")
        custody.record_sources(custody.loaded_sources(sys.modules, torch_libraries()), computation_classes(processor))
        custody.require_source("model", meta["runtime"]["model_source_sha256"], "runtime model_source_sha256")
        meta["runtime"]["custody"] = custody.admission_record()
        custody.write_admission(outdir, meta)
    else:
        (outdir / "admission.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"phase": "prepared", "input_shapes": {k:list(v.shape) for k,v in inputs.items()}, "targets": targets}), flush=True)
    start = time.monotonic()
    if a.device == "cpu":
        meta["memory_admission"] = cpu_memory(admission=True)
    model = load("model", Qwen3_5ForConditionalGeneration.from_pretrained, modeldir, dtype=dtype, device_map=a.device, local_files_only=True, attn_implementation=attention).eval()
    meta["runtime"]["attention_config"] = {"vision":model.config.vision_config._attn_implementation,"text":model.config.text_config._attn_implementation}
    expected_attention = {"vision":"sdpa","text":"eager"} if a.attention=="hybrid" else {"vision":a.attention,"text":a.attention}
    if meta["runtime"]["attention_config"] != expected_attention:
        raise RuntimeError("Attention configuration does not match explicit requested graph")
    if a.device == "cpu":
        meta["memory_after_load"] = cpu_memory()
    if model.lm_head.weight.data_ptr() != model.model.language_model.embed_tokens.weight.data_ptr():
        raise RuntimeError("Expected tied embedding/head storage")
    inputs = {k:v.to(a.device) for k,v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    if a.greedy_max_tokens:
        stop_ids = sorted({model.generation_config.eos_token_id,processor.tokenizer.convert_tokens_to_ids("<|im_end|>")})
        class Receipt:
            def __init__(self):
                self.ids=[]
                self.first=True
            def put(self,value):
                if self.first:
                    self.first=False
                    return
                self.ids.extend(value.detach().cpu().reshape(-1).tolist())
                if len(self.ids)%32==0:
                    self.save()
                    if a.device=="cpu":cpu_memory()
                    print(json.dumps({"phase":"greedy_progress","tokens":len(self.ids),"elapsed_seconds":time.monotonic()-start}),flush=True)
            def save(self):
                (outdir/"greedy.partial.txt").write_text(processor.tokenizer.decode(self.ids,skip_special_tokens=False,clean_up_tokenization_spaces=False))
                (outdir/"greedy.partial.ids.json").write_text(json.dumps(self.ids))
            def end(self):
                self.save()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=a.greedy_max_tokens, do_sample=False, use_cache=True,eos_token_id=stop_ids,streamer=Receipt())
        generated_ids = generated[0,len(prefix):].cpu().tolist()
        text = processor.tokenizer.decode(generated_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        (outdir / "greedy.txt").write_text(text)
        meta["greedy"] = {"token_ids":generated_ids,"max_new_tokens":a.greedy_max_tokens,"checkpoint_eos_token_id":model.generation_config.eos_token_id,"stop_token_ids":stop_ids,"ended_with_eos":bool(generated_ids and generated_ids[-1] in stop_ids),"text_sha256":sha(outdir / "greedy.txt")}
        meta["elapsed_seconds"] = time.monotonic()-start
        meta["peak_rss_kib"] = peak_rss_kib()
        if a.device == "cpu":
            meta["memory_completion"] = cpu_memory()
        write_metadata(outdir, meta, custody, {"greedy.txt": meta["greedy"]["text_sha256"]}, model)
        print(json.dumps({"phase":"complete_greedy","elapsed_seconds":meta["elapsed_seconds"],"generated_tokens":len(generated_ids),"ended_with_eos":meta["greedy"]["ended_with_eos"]}),flush=True)
        return
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
        meta["logits"] = {"file":"logits.f32","shape":list(rows.shape),"dtype":"float32","byte_order":"little","sha256":payload_digest("logits.f32")}
        meta["elapsed_seconds"] = time.monotonic()-start
        meta["peak_rss_kib"] = peak_rss_kib()
        if a.device == "cpu":
            meta["memory_completion"] = cpu_memory()
        write_metadata(outdir, meta, custody, {"logits.f32": meta["logits"]["sha256"]}, model)
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
            append_logit_row(outdir / "logits.partial.f32", logits.numpy())
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
    meta["logits"] = {"file":"logits.f32", "shape":[len(rows),248320], "dtype":"float32", "byte_order":"little", "sha256":payload_digest("logits.f32")}
    meta["elapsed_seconds"] = time.monotonic()-start
    meta["peak_rss_kib"] = peak_rss_kib()
    write_metadata(outdir, meta, custody, {"logits.f32": meta["logits"]["sha256"], "logits.partial.f32": meta["logits"]["sha256"]}, model)
    print(json.dumps({"phase":"complete", "elapsed_seconds":meta["elapsed_seconds"], "peak_rss_kib":meta["peak_rss_kib"]}),flush=True)

if __name__ == "__main__":
    main()
