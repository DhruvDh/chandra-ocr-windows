"""Verify the pinned download and inspect safetensors without importing a GPU runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path: Path, start: int = 0, length: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        while length is None or length > 0:
            block = stream.read(8 * 1024 * 1024 if length is None else min(length, 8 * 1024 * 1024))
            if not block:
                if length:
                    raise ValueError(f"Unexpected EOF: {path.name}")
                break
            h.update(block)
            if length is not None:
                length -= len(block)
    return h.hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def inspect_weights(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing safetensors header length")
        header_length = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_length <= min(size - 8, 100 * 1024 * 1024):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(stream.read(header_length), object_pairs_hook=unique_object)
    data_start = 8 + header_length
    tensors = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if record["dtype"] != "BF16":
            raise ValueError(f"Pinned Chandra inventory expects BF16: {name}")
        shape = record["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid shape: {name}")
        start, end = record["data_offsets"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= size - data_start:
            raise ValueError(f"Invalid offsets: {name}")
        nbytes = math.prod(shape) * 2
        if end - start != nbytes:
            raise ValueError(f"Shape/byte mismatch: {name}")
        tensors[name] = {"shape": shape, "dtype": "BF16", "bytes": nbytes, "offsets": [start, end]}
    cursor = 0
    for name, tensor in sorted(tensors.items(), key=lambda item: item[1]["offsets"]):
        start, end = tensor["offsets"]
        if start != cursor:
            raise ValueError(f"Overlapping or missing tensor data before {name}")
        cursor = end
    if cursor != size - data_start:
        raise ValueError("Unaccounted safetensors payload")
    tied_names = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    tied = None
    if all(name in tensors for name in tied_names):
        hashes = []
        for name in tied_names:
            tensor = tensors[name]
            hashes.append(digest(path, data_start + tensor["offsets"][0], tensor["bytes"]))
        tied = {
            "names": tied_names,
            "sha256": hashes,
            "equal": hashes[0] == hashes[1] and tensors[tied_names[0]]["shape"] == tensors[tied_names[1]]["shape"],
        }
    payload_bytes = size - data_start
    saved_bytes = tensors[tied_names[0]]["bytes"] if tied and tied["equal"] else 0
    mtp_bytes = sum(t["bytes"] for name, t in tensors.items() if name.startswith("mtp.") or ".mtp." in name)
    return {
        "tensor_count": len(tensors),
        "payload_bytes": payload_bytes,
        "tied_embeddings": tied,
        "resident_weight_bytes_if_shared": payload_bytes - saved_bytes,
        "mtp_bytes": mtp_bytes,
        "mtp_omission_validated": False,
        "tensors": tensors,
    }


def verify_model(root: Path, manifest: dict) -> dict:
    verified = []
    for record in manifest["files"]:
        relative = Path(record["file"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes model root")
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError(f"Model integrity failed: {relative}")
        verified.append(record)
    if not any(record["file"] == "model.safetensors" for record in verified):
        raise ValueError("Manifest did not cover model.safetensors")
    return {"schema_version": 1, "model": manifest["model"], "revision": manifest["revision"],
            "integrity_passed": True, "files": verified, "weights": inspect_weights(root / "model.safetensors")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "provenance/model.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_model(args.model_directory, json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_passed": True, "tensor_count": result["weights"]["tensor_count"],
                      "tied_embeddings_equal": result["weights"]["tied_embeddings"]["equal"]}))


if __name__ == "__main__":
    main()
