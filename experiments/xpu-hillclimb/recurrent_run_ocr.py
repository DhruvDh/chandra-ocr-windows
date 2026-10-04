"""Prepared complete-OCR and unprofiled timing harness; candidates unqualified.

Run only under an explicit exclusive GPU lease and the outer run_channel Windows
Job Object. Each invocation owns one model and one selected implementation.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import ExitStack
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GRAPH_HASH = "d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939"
MODES = ("hybrid", "recurrent", "norm-gain", "norm-fused", "recurrent+norm-gain", "recurrent+norm-fused")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=MODES)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compile-cache", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timings", type=int, default=1)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--baseline-response", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()
    uses_recurrent = args.candidate.startswith("recurrent")
    if min(args.warmups, args.timings) < 0 or args.warmups + args.timings < 1 or (uses_recurrent and args.warmups < 1):
        parser.error("Require at least one complete request and at least one recurrent warmup")
    if bool(args.baseline_summary) != bool(args.baseline_response):
        parser.error("Supply both baseline summary and response, or neither")
    if uses_recurrent != bool(args.compile_cache):
        parser.error("Supply a new project-owned --compile-cache exactly for recurrent modes")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.compile_cache:
        args.compile_cache = args.compile_cache.resolve()
        args.compile_cache.mkdir(parents=True, exist_ok=False)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.compile_cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(args.compile_cache / "triton")
    from runtime.waystone import XPUBackend
    from runtime.waystone.bootstrap import prepare_runtime
    compiler_environment = dict(os.environ)
    prepare_runtime()
    import torch
    import transformers
    from PIL import Image
    from transformers.models.qwen3_5 import modeling_qwen3_5 as graph
    if sha(inspect.getfile(graph)) != GRAPH_HASH:
        raise RuntimeError("Installed graph differs from pinned qualified source")
    if torch.__version__ != "2.14.1+xpu" or transformers.__version__ != "5.18.0":
        raise RuntimeError("Installed runtime differs from pinned runtime")
    summary = {"schema": "chandra.xpu.ocr-candidate.v1", "candidate": args.candidate,
               "candidate_qualified": False, "performance_promotion": False,
               "request_max_tokens": 12384, "graph_sha256": GRAPH_HASH,
               "image_sha256": sha(args.image), "prompt_sha256": sha(args.prompt),
               "manifest_sha256": sha(args.manifest), "runs": [],
               "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "python": platform.python_version()}, "source_sha256": {}}
    paths = [Path(__file__), ROOT / "runtime/waystone/backend.py", ROOT / "runtime/waystone/bootstrap.py",
             ROOT / "runtime/waystone/model_identity.py", ROOT / "benchmarks/endpoint.py",
             ROOT / "benchmarks/evaluate.py", ROOT / "verification/compare_ocr.py"]
    paths.extend([ROOT / "provenance/model.json", ROOT / "runtime/waystone/uv.lock"])
    if uses_recurrent:
        paths.append(Path(__file__).with_name("recurrent_candidate.py"))
    if "norm-" in args.candidate:
        paths.append(Path(__file__).with_name("norm_candidate.py"))
    summary["source_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in paths}

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fixtures = [fixture for fixture in manifest["fixtures"] if fixture["image_sha256"] == summary["image_sha256"]]
    if len(fixtures) != 1:
        raise RuntimeError("Image must match one exact frozen manifest fixture")
    fixture = fixtures[0]
    summary["fixture"] = fixture["id"]
    schema = manifest.get("schema")
    qualification = None
    if schema == "chandra-synthetic-corpus-v1":
        validator_path = ROOT / "benchmarks/endpoint.py"
    elif schema == "chandra-independent-synthetic-qualification-v1":
        validator_path = ROOT / "benchmarks/qualification-v1/validate.py"
        spec = importlib.util.spec_from_file_location("chandra_frozen_qualification", validator_path)
        qualification = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qualification)
    else:
        raise RuntimeError("Unsupported frozen manifest schema; no arbitrary validator import")
    summary["validator"] = {"manifest_schema": schema, "source": str(validator_path.relative_to(ROOT)),
                            "sha256": sha(validator_path)}
    summary["timing_scope"] = "Unprofiled backend generation with identical buffered client event logging; token readback/content checks excluded; not zero instrumentation"
    image_bytes = args.image.read_bytes()
    prompt = args.prompt.read_text(encoding="utf-8")
    request = {"model": "chandra", "temperature": 0, "max_tokens": 12384,
               "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode()}}]}]}
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from endpoint import validate
    from verification.compare_ocr import compare
    baseline_response = None
    if args.baseline_summary:
        reference = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        for field in ("image_sha256", "prompt_sha256", "manifest_sha256", "request_max_tokens"):
            if reference.get(field) != summary[field]:
                raise RuntimeError(f"Baseline request provenance differs: {field}")
        if reference.get("candidate") != "hybrid" or reference.get("graph_sha256") != GRAPH_HASH:
            raise RuntimeError("Comparison requires this harness's unchanged pinned hybrid baseline")
        if reference.get("validator") != summary["validator"]:
            raise RuntimeError("Baseline validator/schema identity differs")
        baseline_response = json.loads(args.baseline_response.read_text(encoding="utf-8"))
        if not any(row.get("response_sha256") == sha(args.baseline_response) and row.get("correctness", {}).get("passed")
                   for row in reference["runs"]):
            raise RuntimeError("Baseline response lacks a successful committed run")
        summary["comparison_reference"] = {"summary_sha256": sha(args.baseline_summary),
                                           "response_sha256": sha(args.baseline_response)}
    backend = XPUBackend(args.model, attention_backend="hybrid")
    captured = {}
    original_generate = None
    recurrent_receipt = norm_receipt = None
    try:
        if uses_recurrent:
            import triton
            import triton.runtime.driver
            import triton.backends.intel.driver as intel_driver
            start = time.perf_counter()
            compiler_cwd = args.compile_cache / "compiler-host"
            compiler_cwd.mkdir()
            original_subprocess_run = subprocess.run
            def compiler_run(*positional, **keywords):
                command = positional[0] if positional else keywords.get("args")
                if isinstance(command, (list, tuple)) and command and Path(command[0]).name.lower() in {"cl.exe", "link.exe"}:
                    keywords = dict(keywords, cwd=str(compiler_cwd), env=compiler_environment)
                return original_subprocess_run(*positional, **keywords)
            subprocess.run = compiler_run
            try:
                summary["triton_target"] = repr(triton.runtime.driver.active.get_current_target())
            finally:
                subprocess.run = original_subprocess_run
            summary["helper_initialization_seconds"] = time.perf_counter() - start
            summary["runtime"]["triton"] = triton.__version__
            driver_path = Path(inspect.getfile(intel_driver))
            summary["runtime"]["triton_intel_source_sha256"] = {name: sha(driver_path.parent / name)
                for name in ("driver.py", "driver.c", "arch_parser.c", "extension_utils.py")}
            summary["compile_cache"] = str(args.compile_cache)
        start = time.perf_counter()
        backend.load()
        summary["load_seconds"] = time.perf_counter() - start
        summary["loaded_health"] = backend.health()
        with Image.open(args.image) as image:
            rgb = image.convert("RGB")
        with torch.inference_mode():
            inputs = backend.processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image", "image": rgb}, {"type": "text", "text": prompt}]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
        summary["processed_inputs"] = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                for name, tensor in inputs.items()}
        expected_prefix = inputs["input_ids"][0].tolist()
        if args.baseline_summary and reference.get("processed_inputs") != summary["processed_inputs"]:
            raise RuntimeError("Baseline processed tensor identity differs")
        del inputs
        original_generate = backend.model.generate

        def retained_generate(*positional, **keywords):
            result = original_generate(*positional, **keywords)
            # Preserve the result reference only. No extra synchronization or
            # tensor readback enters the measured backend generation interval.
            captured["ids"] = result
            return result

        backend.model.generate = retained_generate
        with ExitStack() as scopes:
            torch.xpu.synchronize()
            setup_start = time.perf_counter()
            if uses_recurrent:
                from recurrent_candidate import compiled_recurrent
                recurrent_receipt = scopes.enter_context(compiled_recurrent(expected_graph_sha256=GRAPH_HASH))
            if "norm-" in args.candidate:
                from norm_candidate import candidate
                mode = "fused" if args.candidate.endswith("norm-fused") else "gain-only"
                norm_receipt = scopes.enter_context(candidate(backend.model, mode=mode))
            torch.xpu.synchronize()
            summary["candidate_setup_seconds"] = time.perf_counter() - setup_start
            save()

            def run(label, measured):
                before_recurrent = recurrent_receipt["calls"] if recurrent_receipt else 0
                before_norm = dict(norm_receipt["calls"]) if norm_receipt else {}
                torch.xpu.synchronize()
                torch.xpu.reset_peak_memory_stats()
                cancel = threading.Event()
                chunks, final, first = [], None, None
                start = time.perf_counter()
                with (args.output / (label + ".stream.jsonl")).open("w", encoding="utf-8") as events:
                    try:
                        for part in backend.generate(request, cancel):
                            elapsed = time.perf_counter() - start
                            events.write(json.dumps({"elapsed_seconds": elapsed, "part": part}) + "\n")
                            if args.cancel_file and args.cancel_file.exists():
                                cancel.set()
                            if isinstance(part, str):
                                chunks.append(part)
                                if first is None:
                                    first = elapsed
                            else:
                                final = part
                    finally:
                        elapsed = time.perf_counter() - start
                        (args.output / (label + ".partial.txt")).write_text("".join(chunks), encoding="utf-8")
                if final is None or cancel.is_set():
                    raise RuntimeError("Missing complete final metadata or cancelled generation")
                ids = captured.pop("ids")
                if ids.ndim != 2 or ids.shape[0] != 1:
                    raise RuntimeError("Unexpected full generation token layout")
                all_ids = ids[0].cpu().tolist()
                if final["usage"]["prompt_tokens"] != len(expected_prefix) or all_ids[:len(expected_prefix)] != expected_prefix:
                    raise RuntimeError("Actual generation prefix differs from frozen processed input")
                generated = all_ids[len(expected_prefix):]
                del ids
                eos = backend.model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else list(eos)
                eos.append(backend.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if len(generated) != final["usage"]["completion_tokens"] or not generated or generated[-1] not in eos:
                    raise RuntimeError("Complete generated token/terminal metadata disagree")
                response = {"model": "chandra", "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "".join(chunks)}, "finish_reason": final["finish_reason"]}],
                    "usage": final["usage"]}
                path = args.output / (label + ".response.json")
                path.write_text(json.dumps(response, indent=2), encoding="utf-8")
                (args.output / (label + ".tokens.json")).write_text(json.dumps({"token_ids": generated,
                    "terminal_id": generated[-1], "stop_ids": eos}), encoding="utf-8")
                recurrent_calls = recurrent_receipt["calls"] - before_recurrent if recurrent_receipt else 0
                norm_calls = {name: count - before_norm[name] for name, count in norm_receipt["calls"].items()} if norm_receipt else {}
                expected_recurrent = 24 * (len(generated) - 1) if recurrent_receipt else 0
                interception_ok = recurrent_calls == expected_recurrent and all(count == len(generated) for count in norm_calls.values())
                row = {"run": label, "measured_warm_run": measured, "elapsed_seconds": elapsed,
                       "first_text_seconds": first, **final, "response_sha256": sha(path),
                       "correctness": qualification.check(fixture["expected"], response, 12384) if qualification else validate(fixture["expected"], {"content": "".join(chunks), **final}, 12384),
                       "interception": {"passed": interception_ok, "recurrent_calls": recurrent_calls,
                           "expected_recurrent_calls": expected_recurrent, "norm_calls": norm_calls,
                           "expected_calls_per_norm": len(generated) if norm_receipt else 0},
                       "memory": backend.health()["torch_xpu_memory"], "comparable_with_baseline": False}
                if baseline_response:
                    row["baseline_comparison"] = compare(args.baseline_response, path)
                    if qualification:
                        row["independent_content_and_geometry_comparison"] = qualification.compare(fixture["expected"], baseline_response, response, 12384)
                    row["same_usage_and_stop"] = baseline_response["usage"] == final["usage"] and baseline_response["choices"][0]["finish_reason"] == final["finish_reason"]
                    row["comparable_with_baseline"] = row["same_usage_and_stop"] and row["baseline_comparison"]["exact_html_equal"] and row["correctness"]["passed"] and interception_ok
                summary["runs"].append(row)
                save()
                print(json.dumps({"phase": "complete", "run": label, "elapsed_seconds": elapsed,
                                  "usage": final["usage"], "interception_passed": interception_ok}), flush=True)
                if not row["correctness"]["passed"] or not interception_ok:
                    raise RuntimeError("Complete OCR or exact interception gate failed; raw run retained")

            for index in range(args.warmups):
                run(f"warmup-{index}", False)
            for index in range(args.timings):
                run(f"timing-{index}", args.warmups > 0 or index > 0)
        summary["candidate_receipts"] = {"recurrent": recurrent_receipt, "norm": norm_receipt}
    except BaseException as error:
        summary["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        summary["candidate_receipts"] = {"recurrent": recurrent_receipt, "norm": norm_receipt}
        if original_generate is not None and backend.model is not None:
            backend.model.generate = original_generate
        original_generate = None
        captured.clear()
        backend.unload()
        if torch.xpu.is_initialized():
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        summary["unloaded_health"] = backend.health()
        memory = {"allocated_bytes": torch.xpu.memory_allocated(), "reserved_bytes": torch.xpu.memory_reserved()} if torch.xpu.is_initialized() else {"allocated_bytes": 0, "reserved_bytes": 0}
        summary["unload_memory"] = memory
        summary["unload_zero_memory"] = memory is not None and memory["allocated_bytes"] == 0 and memory["reserved_bytes"] == 0
        save()
        if not summary["unload_zero_memory"]:
            raise RuntimeError("Candidate context unload retained GPU memory")


if __name__ == "__main__":
    main()
