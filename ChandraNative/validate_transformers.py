"""CPU-only cross-check against an installed Transformers source, without importing its model module."""
import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("modeling_source", type=Path)
    args = parser.parse_args()
    source = args.modeling_source.read_bytes()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "torch_recurrent_gated_delta_rule")
    function.decorator_list = []  # Exercise the source fallback, not a hub-selected fused kernel.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), "installed_transformers_reference", "exec"), namespace)
    raw = args.fixture.read_bytes()
    tokens, heads, keys, values = struct.unpack_from("<4I", raw)
    offset = 16

    def take(count):
        nonlocal offset
        result = torch.frombuffer(bytearray(raw[offset:offset + count * 4]), dtype=torch.float32).clone()
        offset += count * 4
        return result

    q = take(tokens * heads * keys).reshape(1, tokens, heads, keys)
    k = take(tokens * heads * keys).reshape_as(q)
    v = take(tokens * heads * values).reshape(1, tokens, heads, values)
    g = take(tokens * heads).reshape(1, tokens, heads)
    beta = take(tokens * heads).reshape_as(g)
    expected_state = take(heads * keys * values).reshape(1, heads, keys, values)
    expected_output = take(tokens * heads * values).reshape_as(v)
    if offset != len(raw):
        raise ValueError("unexpected fixture payload")
    # C++ fixture Q is already scaled; installed source always applies this scale.
    output, state = namespace[function.name](q * math.sqrt(keys), k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=False)
    errors = {"state_max_abs": float((state - expected_state).abs().max()), "output_max_abs": float((output - expected_output).abs().max())}
    passed = all(math.isfinite(value) and value < 2e-5 for value in errors.values())
    print(json.dumps({"test": "installed_transformers_cpu_reference", "passed": passed, "tokens": tokens, "heads": heads, "keys": keys, "values": values, "torch": torch.__version__, "source_sha256": hashlib.sha256(source).hexdigest(), "fixture_sha256": hashlib.sha256(raw).hexdigest(), **errors}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
