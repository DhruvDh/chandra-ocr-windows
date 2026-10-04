"""Trained-fixture XPU graph contingency; no model load or serving patch.

Execute ONLY inside the coordinator's authenticated owned run_channel Job Object
and an exclusive GPU lease, with graph_deadline.py as the inner deadline.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--torch-graphs-sha256", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 2 <= len(args.fixture) <= 4:
        parser.error("Supply two to four distinct trained fixtures from one layer")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"candidate": "unchanged-eager-recurrent-xpu-graph-capability-v1", "state": "running", "cases": [], "source_sha256": sha(__file__), "acceptance": "bitwise output AND FP32 state; no timing or model acceptance"}
    receipt = args.output / "graph-capability.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from runtime.waystone.bootstrap import prepare_runtime
    prepare_runtime()
    import torch
    from safetensors.torch import load_file
    import transformers.models.qwen3_5.modeling_qwen3_5 as graph
    from recurrent_candidate import undecorated_fallback
    fallback, graph_hash, function_hash = undecorated_fallback(graph)
    if graph_hash != args.graph_sha256 or sha(inspect.getfile(torch.xpu.XPUGraph)) != args.torch_graphs_sha256:
        raise RuntimeError("Pinned installed model/runtime graph source mismatch")
    if torch.__version__ != "2.14.1+xpu" or torch.version.git_version != "5c4886908584029761b579af026dcfb627c84070":
        raise RuntimeError("Pinned Windows Torch wheel differs")
    if "SYCL_COMPILER_VERSION=20260100" not in torch.__config__.show():
        raise RuntimeError("Expected native-recording build missing")
    names = ("query", "key", "value", "g", "beta", "initial_state")
    host = []
    total_bytes = 0
    signatures = None
    for filename, digest in args.fixture:
        path = Path(filename)
        if path.stat().st_size > 16 * 1024**2 or sha(path) != digest:
            raise RuntimeError("Trained fixture size/hash mismatch")
        tensors = load_file(str(path), device="cpu")
        tensors = {name: tensors[name] for name in (*names, "reference_output", "reference_state")}
        total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
        shape = {n: (tuple(tensors[n].shape), str(tensors[n].dtype)) for n in tensors}
        if signatures is not None and signatures != shape:
            raise RuntimeError("Distinct fixture shapes require separate graphs")
        signatures = shape
        if any(tensors[n].dtype != torch.bfloat16 for n in names[:3]) or tensors["initial_state"].dtype != torch.float32 or tensors["reference_state"].dtype != torch.float32:
            raise RuntimeError("BF16 inputs/FP32 persistent state required")
        if any(tensors[n].ndim != 4 or tensors[n].shape[:2] != (1, 1) for n in names[:3]) or tensors["initial_state"].ndim != 4 or total_bytes > 64 * 1024**2:
            raise RuntimeError("Single-token batch-one / 64MiB fixture allowance breached")
        if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise RuntimeError("Nonfinite fixture")
        host.append(tensors)
    if len({digest for _, digest in args.fixture}) != len(host):
        raise RuntimeError("Changed-input test requires distinct fixtures")
    report.update(graph_sha256=graph_hash, torch_graphs_sha256=args.torch_graphs_sha256, fallback_helper_sha256=sha(inspect.getfile(undecorated_fallback)), bootstrap_sha256=sha(inspect.getfile(prepare_runtime)), function_ast_sha256=function_hash, fixture_sha256=[digest for _, digest in args.fixture], fixture_bytes=total_bytes, torch=torch.__version__, torch_commit=torch.version.git_version)
    save()
    # Everything above this line is CPU/source validation. Allocation begins
    # only when this explicitly leased probe is executed, never on import.
    if torch.xpu.device_count() != 1:
        raise RuntimeError("Expected one unambiguous XPU")
    properties = torch.xpu.get_device_properties(0)
    if properties.device_id != 0x56A0 or properties.is_integrated_gpu or "Level-Zero" not in properties.platform_name or str(properties.uuid) != args.device_uuid:
        raise RuntimeError("Device differs from leased discrete A770")
    report["device"] = {"name": properties.name, "uuid": str(properties.uuid), "device_id": properties.device_id, "platform": properties.platform_name}
    save()
    xpu_graph = None
    static = outputs = device = stream = None

    def bits(tensor):
        return tensor.detach().cpu().contiguous().view(torch.uint8)

    def exact(actual, expected):
        return actual.shape == expected.shape and actual.dtype == expected.dtype and bool(torch.equal(bits(actual), bits(expected)))

    try:
        with torch.inference_mode():
            device = [{n: tensors[n].to("xpu:0") for n in names} for tensors in host]
            static = [device[0][n].clone() for n in names]
            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(stream):
                for _ in range(3):
                    warm = fallback(*static, output_final_state=True, use_qk_l2norm_in_kernel=True)
                del warm
            torch.xpu.synchronize()
            xpu_graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(xpu_graph, stream=stream):
                outputs = fallback(*static, output_final_state=True, use_qk_l2norm_in_kernel=True)
            torch.xpu.synchronize()
            report["capture_completed"] = True
            save()

            def run(label, index, initial=None, frozen_reference=False):
                inputs = [device[index][n] for n in names]
                if initial is not None:
                    inputs[-1] = initial
                before = [bits(t).clone() for t in inputs]
                with torch.xpu.stream(stream):
                    eager = fallback(*inputs, output_final_state=True, use_qk_l2norm_in_kernel=True)
                    for buffer, value in zip(static, inputs):
                        buffer.copy_(value)
                    xpu_graph.replay()
                    actual = tuple(value.clone() for value in outputs)
                torch.xpu.synchronize()
                measurements = {n: {"bit_equal": exact(a, e), "max_abs": float((a.float() - e.float()).abs().max()), "dtype": str(a.dtype)} for n, a, e in zip(("output", "state"), actual, eager)}
                unchanged = all(torch.equal(old, bits(t)) for old, t in zip(before, inputs))
                frozen = None if not frozen_reference else exact(eager[0], host[index]["reference_output"]) and exact(eager[1], host[index]["reference_state"])
                report["cases"].append({"case": label, "fixture": index, "metrics": measurements, "inputs_unchanged": bool(unchanged), "frozen_eager_reference_equal": frozen})
                save()
                if not unchanged or not all(v["bit_equal"] for v in measurements.values()) or frozen is False or actual[1].dtype != torch.float32:
                    raise RuntimeError("Exact eager output/state/input criterion failed")
                return actual

            first = run("fixture-0", 0, frozen_reference=True)
            repeated = run("repeat-identical", 0, frozen_reference=True)
            if not all(exact(a, b) for a, b in zip(first, repeated)):
                raise RuntimeError("Repeated replay differs")
            for index in range(1, len(host)):
                run("changed-trained-input", index, frozen_reference=True)
            chain = first[1]
            for step in range(8):
                chain = run(f"chain-{step}", step % len(host), initial=chain)[1]
            reset = run("page-reset-original-trained-state", 0, frozen_reference=True)
            if not all(exact(a, b) for a, b in zip(first, reset)):
                raise RuntimeError("Page reset retains previous trajectory")
            report["state"] = "passed_capability_only"
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        # Never release pointer-backed storage while a replay is still running.
        try:
            torch.xpu.synchronize()
            if xpu_graph is not None:
                xpu_graph.reset()
            outputs = static = device = stream = xpu_graph = None
            first = repeated = chain = reset = None
            gc.collect()
            torch.xpu.empty_cache()
            report["cleanup"] = "synchronized; graph reset; owned buffers released; empty_cache"
        except BaseException as error:
            report.update(state="failed", cleanup=f"failed: {type(error).__name__}: {error}; outer owned channel must terminate this process")
            save()
            raise
        save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
