"""Exercise the actual native importer with valid and malformed tiny payloads; no GPU."""
import argparse
import copy
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checker", type=Path)
    args = parser.parse_args()
    names = ["q", "k", "v", "g", "beta", "expected_output", "expected_state"]
    values = [0.125, 1, 2, 0, 0.5, 0.125, 1]
    payload = struct.pack("<7f", *values)
    metadata = {
        "schema": "chandra.native.delta.v1", "dtype": "float32", "byte_order": "little",
        "payload_file": "fixture.bin", "payload_bytes": len(payload), "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "model_revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09", "source": {"function": "hand_calculated_scalar"},
        "semantics": {"q": "l2_normalized_scaled_sqrt_keydim", "k": "l2_normalized", "g": "negative_log_decay", "beta": "post_sigmoid", "heads": "expanded_value_heads", "initial_state": "zeros", "layout": "THD"},
        "tensors": {name: {"shape": [1, 1] if name in ("g", "beta") else [1, 1, 1], "offset_bytes": i * 4, "byte_length": 4, "sha256": hashlib.sha256(payload[i*4:i*4+4]).hexdigest()} for i, name in enumerate(names)},
    }
    cases = []

    def mutated(name, change, data=payload):
        m = copy.deepcopy(metadata)
        change(m)
        cases.append((name, json.dumps(m), data, False))

    cases.append(("valid_scalar", json.dumps(metadata), payload, True))
    mutated("payload_hash_mismatch", lambda m: m.update(payload_sha256="0"*64))
    mutated("tensor_hash_mismatch", lambda m: m["tensors"]["q"].update(sha256="0"*64))
    mutated("overlap", lambda m: m["tensors"]["k"].update(offset_bytes=0))
    mutated("zero_dimension", lambda m: m["tensors"]["q"].update(shape=[0, 1, 1]))
    mutated("fractional_dimension", lambda m: m["tensors"]["q"].update(shape=[1.0, 1, 1]))
    mutated("shape_overflow", lambda m: m["tensors"]["q"].update(shape=[4096, 4096, 4096]))
    mutated("wrong_semantics", lambda m: m["semantics"].update(q="raw_projection"))
    mutated("wrong_model", lambda m: m.update(model_revision="0"*40))
    mutated("path_escape", lambda m: m.update(payload_file="../fixture.bin"))
    mutated("unknown_root_field", lambda m: m.update(unrecognized=True))
    mutated("truncated_payload", lambda m: None, payload[:-1])
    cases.append(("duplicate_json_key", json.dumps(metadata)[:-1] + ',"dtype":"float32"}', payload, False))
    for name, index, value in [("nonfinite", 0, float("nan")), ("positive_decay", 3, 0.1), ("invalid_beta", 4, 1.1)]:
        updated = values.copy(); updated[index] = value
        data = struct.pack("<7f", *updated)

        def change(m, data=data, index=index):
            m["payload_sha256"] = hashlib.sha256(data).hexdigest()
            m["tensors"][names[index]]["sha256"] = hashlib.sha256(data[index*4:index*4+4]).hexdigest()

        mutated(name, change, data)
    failures = []
    with tempfile.TemporaryDirectory(prefix="chandra-native-format-") as temporary:
        root = Path(temporary)
        for name, text, data, expected in cases:
            (root / "fixture.json").write_text(text)
            (root / "fixture.bin").write_bytes(data)
            result = subprocess.run([str(args.checker), str(root / "fixture.json")], capture_output=True, text=True)
            if (result.returncode == 0) != expected:
                failures.append({"case": name, "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(json.dumps({"test": "native_fixture_format", "cases": len(cases), "passed": not failures, "failures": failures}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
