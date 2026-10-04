"""Source-identical, single-token compiled recurrent fallback experiment.

Nothing is compiled or allocated until the scope receives a cached XPU call.
Inductor arithmetic must pass trained and complete-model differential gates.
"""
import ast
from contextlib import contextmanager
import hashlib
import inspect


def undecorated_fallback(graph):
    source = inspect.getfile(graph.Qwen3_5GatedDeltaNet)
    with open(source, "rb") as stream:
        raw = stream.read()
    tree = ast.parse(raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "torch_recurrent_gated_delta_rule")
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    namespace = dict(vars(graph))
    exec(compile(ast.fix_missing_locations(module), source, "exec"), namespace)
    return namespace[function.name], hashlib.sha256(raw).hexdigest(), hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()


@contextmanager
def compiled_recurrent(*, expected_graph_sha256, fixture=None, compile_function=None):
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as graph
    fallback, graph_hash, function_hash = undecorated_fallback(graph)
    if graph_hash != expected_graph_sha256:
        raise RuntimeError("Installed Qwen3.5 source differs from approved hash")
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("Compiler error suppression must be disabled")
    # Static decode shape only. fullgraph raises on a graph break; failed
    # compilation must remain a failed candidate rather than an eager timing.
    compiler = compile_function or torch.compile
    compiled = compiler(fallback, backend="inductor", fullgraph=True, dynamic=False)
    original = graph.torch_recurrent_gated_delta_rule
    receipt = {"candidate": "source-identical-compiled-recurrent-decode-v1",
               "undecorated_function_ast_sha256": function_hash,
               "graph_sha256": graph_hash, "backend": "inductor", "fullgraph": True,
               "dynamic": False, "calls": 0, "shapes": [], "state_dtype": "float32"}
    shapes = set()

    def candidate(query, key, value, g, beta, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if key.shape[1] != 1 or initial_state is None:
            return original(query, key, value, g=g, beta=beta,
                            initial_state=initial_state, output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        if any(t.device.type != "xpu" for t in (query, key, value, g, beta, initial_state)):
            raise RuntimeError("Cached candidate requires XPU tensors; no CPU fallback")
        if any(t.dtype != torch.bfloat16 for t in (query, key, value)) or initial_state.dtype != torch.float32:
            raise RuntimeError("Candidate requires BF16 Q/K/V and unchanged FP32 state")
        if not output_final_state or not use_qk_l2norm_in_kernel:
            raise RuntimeError("Unexpected trained decode semantics")
        if kwargs.get("cu_seqlens") is not None:
            raise RuntimeError("Packed recurrence is outside the isolated candidate")
        # Source fallback ignores extra kwargs. Do not make them compiler guards.
        arguments = (query, key, value, g, beta, initial_state, True, True)
        if fixture is not None:
            fixture(arguments, fallback, compiled)
        result = compiled(*arguments)
        if result[1].dtype != torch.float32 or result[0].dtype != query.dtype:
            raise RuntimeError("Compiled recurrent output/state precision changed")
        shape = (tuple(query.shape), tuple(value.shape), tuple(initial_state.shape))
        if shape not in shapes:
            shapes.add(shape)
            receipt["shapes"].append(shape)
        receipt["calls"] += 1
        return result

    graph.torch_recurrent_gated_delta_rule = candidate
    try:
        yield receipt
    finally:
        graph.torch_recurrent_gated_delta_rule = original
