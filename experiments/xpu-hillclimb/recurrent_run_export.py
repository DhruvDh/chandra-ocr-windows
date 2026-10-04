"""Run exact-input exporter with isolated compiled recurrent cached decode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from recurrent_candidate import compiled_recurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--compile-cache", required=True, type=Path,
                        help="New project-owned local cache directory, preferably runtime AppData NTFS on Windows")
    parser.add_argument("--fixture-limit", type=int, default=24)
    parser.add_argument("--fixture-positions", help="Comma-separated zero-based cached decode positions; all 24 layers at each")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--compiler-diagnostic", action="store_true")
    parser.add_argument("--isolate-compiler-host", action="store_true",
                        help="Run MSVC children from owned C cache with pre-bootstrap vcvars PATH")
    parser.add_argument("--compiler-separate-stderr", action="store_true")
    parser.add_argument("--compiler-clean-worker", action="store_true")
    parser.add_argument("--preinitialize-triton", action="store_true")
    parser.add_argument("--replay-fixture", type=Path)
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("export_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    fixture_positions = set(map(int, args.fixture_positions.split(","))) if args.fixture_positions else None
    if fixture_positions is not None and any(position < 0 for position in fixture_positions):
        parser.error("Fixture positions must be nonnegative")
    if bool(args.policy) != bool(args.policy_sha256):
        parser.error("Supply both policy file and predeclared SHA256")
    if args.policy and hashlib.sha256(args.policy.read_bytes()).hexdigest() != args.policy_sha256:
        parser.error("Policy differs from its predeclared hash")
    cache = args.compile_cache.resolve()
    cache.mkdir(parents=True, exist_ok=False)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    if args.compiler_diagnostic or args.isolate_compiler_host:
        os.environ["VERBOSE"] = "1"
        compiler_environment = dict(os.environ)
        compiler_cwd = cache / "compiler-host"
        compiler_cwd.mkdir()
        subprocess_run = subprocess.run
        def diagnostic_run(*positional, **keywords):
            command = positional[0] if positional else keywords.get("args")
            if (args.isolate_compiler_host and isinstance(command, (list, tuple)) and command
                    and Path(command[0]).name.lower() in {"cl.exe", "link.exe"}):
                keywords = dict(keywords, cwd=str(compiler_cwd), env=compiler_environment)
                if args.compiler_separate_stderr:
                    keywords["stderr"] = subprocess.PIPE
                source = next((Path(item) for item in command if isinstance(item, str)
                               and item.endswith((".cpp", ".c")) and Path(item).is_file()), None)
                if source is not None:
                    shutil.copy2(source, compiler_cwd / source.name)
                    replay = [item.replace(str(source.parent), str(compiler_cwd)) for item in command]
                    (compiler_cwd / "command.json").write_text(json.dumps(replay, indent=2))
                print(json.dumps({"compiler_launch": {"cwd": keywords.get("cwd"),
                                  "stdout": keywords.get("stdout"), "stderr": keywords.get("stderr"),
                                  "stdin": keywords.get("stdin"), "close_fds": keywords.get("close_fds")}}),
                      file=sys.stderr, flush=True)
                if args.compiler_clean_worker:
                    request = compiler_cwd / "worker-request.json"
                    response = compiler_cwd / "worker-response.json"
                    request.write_text(json.dumps({"command": command, "cwd": str(compiler_cwd)}))
                    worker_command = [sys.executable, str(Path(__file__).with_name("recurrent_compiler_worker.py")),
                                      str(request), str(response)]
                    result = subprocess_run(worker_command, cwd=str(compiler_cwd), env=compiler_environment,
                                            capture_output=True, stdin=subprocess.DEVNULL)
                    actual_code = json.loads(response.read_text())["returncode"] if response.exists() else result.returncode
                    if actual_code:
                        raise subprocess.CalledProcessError(actual_code, command, result.stdout, result.stderr)
                    return subprocess.CompletedProcess(command, actual_code, result.stdout, result.stderr)
            try:
                return subprocess_run(*positional, **keywords)
            except subprocess.CalledProcessError as error:
                print(json.dumps({"subprocess_failure": {"command": error.cmd, "returncode": error.returncode,
                                  "stdout": repr(error.stdout), "stderr": repr(error.stderr)}}), file=sys.stderr, flush=True)
                raise
        subprocess.run = diagnostic_run
    remainder = args.export_args
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not args.replay_fixture:
        for flag, required in (("--attention", "hybrid"), ("--device", "xpu")):
            if flag not in remainder or remainder[remainder.index(flag) + 1] != required:
                parser.error(f"Pass {flag} {required}")
    elif not args.fixture_sha256 or not args.replay_output or remainder:
        parser.error("Replay requires fixture SHA256 and a new --replay-output, without exporter arguments")
    if "--all-teacher" in remainder or "--full-teacher" in remainder:
        parser.error("Full-sequence teacher forcing bypasses recurrent decode; use --steps 394")
    output = args.replay_output if args.replay_fixture else Path(remainder[remainder.index("--output") + 1])
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    preinitialized_target = None
    if args.preinitialize_triton:
        import torch
        import triton.runtime.driver
        preinitialized_target = repr(triton.runtime.driver.active.get_current_target())
        print(json.dumps({"phase": "triton_preinitialized", "target": preinitialized_target}), flush=True)
    wrapper = root / "scripts/waystone/run_export.py"
    exporter = root / "verification/export_logits.py"
    sys.argv = [str(wrapper), str(exporter), *remainder]
    fixtures = []
    capture_calls = 0

    def capture(arguments, fallback, compiled):
        nonlocal capture_calls
        call_ordinal = capture_calls
        capture_calls += 1
        decode_position, layer_ordinal = divmod(call_ordinal, 24)
        if fixture_positions is not None and decode_position not in fixture_positions:
            return
        if len(fixtures) >= args.fixture_limit:
            return
        import torch
        from safetensors.torch import save_file
        names = ("query", "key", "value", "g", "beta", "initial_state")
        original_inputs = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in zip(names, arguments[:6])}
        reference_output, reference_state = fallback(*arguments)
        tensors = dict(original_inputs)
        tensors.update(reference_output=reference_output.detach().cpu().contiguous(),
                       reference_state=reference_state.detach().cpu().contiguous())
        reference_path = output / f"trained-recurrent-{len(fixtures):03d}-reference.safetensors"
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}, str(reference_path))
        candidate_output, candidate_state = compiled(*arguments)
        torch.xpu.synchronize()
        for name, tensor in zip(names, arguments[:6]):
            if not torch.equal(original_inputs[name], tensor.detach().cpu()):
                raise RuntimeError(f"Recurrent candidate mutated input {name}")
        for actual, reference in ((candidate_output, reference_output), (candidate_state, reference_state)):
            if actual.shape != reference.shape or actual.dtype != reference.dtype:
                raise RuntimeError("Compiled recurrent shape/dtype changed")
        tensors.update(candidate_output=candidate_output, candidate_state=candidate_state)
        host = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
        if not all(bool(torch.isfinite(tensor).all()) for tensor in host.values()):
            raise RuntimeError("Nonfinite trained recurrent differential")
        metrics = {}
        for name in ("output", "state"):
            difference = host[f"candidate_{name}"].float() - host[f"reference_{name}"].float()
            metrics[name] = {"max_abs": float(difference.abs().max()),
                             "rmse": float(difference.square().mean().sqrt()),
                             "reference_max_abs": float(host[f"reference_{name}"].float().abs().max()),
                             "reference_rms": float(host[f"reference_{name}"].float().square().mean().sqrt()),
                             "bit_equal": bool(torch.equal(host[f"candidate_{name}"], host[f"reference_{name}"]))}
            metrics[name]["normalized_rmse"] = metrics[name]["rmse"] / max(metrics[name]["reference_rms"], 1e-30)
        path = output / f"trained-recurrent-{len(fixtures):03d}.safetensors"
        save_file(host, str(path))
        fixtures.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "ordinal": len(fixtures), "metrics": metrics,
                         "call_ordinal": call_ordinal, "decode_position": decode_position,
                         "linear_layer_ordinal": layer_ordinal,
                         "inputs_unchanged": True,
                         "tensor_layouts": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                                            for name, tensor in host.items()},
                         "semantics": "raw-expanded-heads; normalized-in-FP32; initialized-trained-FP32-state"})

    with compiled_recurrent(expected_graph_sha256=args.graph_sha256,
                            fixture=capture if args.fixture_limit else None) as receipt:
        try:
            if args.replay_fixture:
                import torch
                from safetensors.torch import load_file
                if hashlib.sha256(args.replay_fixture.read_bytes()).hexdigest() != args.fixture_sha256:
                    raise RuntimeError("Trained fixture differs from supplied hash")
                output.mkdir(parents=True, exist_ok=False)
                host = load_file(str(args.replay_fixture))
                tensors = {name: tensor.to("xpu:0") for name, tensor in host.items()}
                import transformers.models.qwen3_5.modeling_qwen3_5 as graph
                with torch.inference_mode():
                    graph.torch_recurrent_gated_delta_rule(*(tensors[name] for name in
                        ("query", "key", "value", "g", "beta", "initial_state")),
                        output_final_state=True, use_qk_l2norm_in_kernel=True)
                receipt["replayed_fixture_sha256"] = args.fixture_sha256
            else:
                runpy.run_path(str(wrapper), run_name="__main__")
        finally:
            receipt["trained_fixtures"] = fixtures
            receipt["fixture_positions"] = sorted(fixture_positions) if fixture_positions is not None else None
            receipt["policy_sha256"] = args.policy_sha256
            receipt["compile_cache"] = str(cache)
            receipt["compiler_host_isolated"] = args.isolate_compiler_host
            receipt["triton_preinitialized_target"] = preinitialized_target
            receipt["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in (Path(__file__), Path(__file__).with_name("recurrent_candidate.py"))}
            if output.exists():
                (output / "recurrent-candidate.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if not receipt["calls"]:
        raise RuntimeError("No trained recurrent cached decode was intercepted")


if __name__ == "__main__":
    main()
