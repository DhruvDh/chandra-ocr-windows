"""Package one retained CPU processor export for native qualification (stdlib only).

Does not tokenize, decode PNG pixels, load weights or execute an inference runtime.
The original corpus/export commitments identify decoded RGB pixels. Positions are
derived from the pinned Transformers 5.18 source, not captured AMD runtime data.
Output is exclusive, bounded to 128 MiB and 30 seconds, and remains unqualified.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct
import time

PIN = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
ROPE_SOURCE_SHA = "d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939"
CALLER_SOURCE_SHA = "185ef7cbff08dea1a196f7fd96993f596d5943b90c2edcdaba001f0487cd1921"
CAP = 128 * 1024 * 1024
PROFILE = {"checkpoint-default": {}, "northstone-serving": {
    "size": {"shortest_edge": 3136, "longest_edge": 3145728}}}
PROCESSOR_PINS = {
    "chat_template.jinja": "0d158f349ca965f7eea9db0eb45cd177b85bb0e4ae05dcdd0f060da8f7d41812",
    "config.json": "e26f17b70463de21fd68f1ef6d8f67f8e33e65d55a7a6895242130a18db60587",
    "generation_config.json": "0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e",
    "preprocessor_config.json": "957eb01d1ea45341a92d543daec95857a7cbeff5803834bc0603b27ba7b41b3f",
    "processor_config.json": "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b",
    "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    "video_preprocessor_config.json": "de7ba2c4528aa3c92754dc61ae83f1871369cbf7ab4298dcaa99c2f5a7c80848",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def finite_json(data):
    return json.loads(data, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: require(False, "Nonfinite JSON"))


def regular(path):
    require(path.is_file() and not path.is_symlink(), "Expected regular nonsymlink file")
    return path


def child(root, name):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            and name not in (".", ".."), "Unsafe asset filename")
    return regular(root / name)


def small(path, expected=None):
    regular(path)
    require(path.stat().st_size <= 1024 * 1024, "Metadata exceeds 1 MiB")
    data = path.read_bytes()
    if expected is not None:
        require(digest(data) == expected, "Metadata hash mismatch")
    return data


def int64_values(path, record):
    require(record["dtype"] == "torch.int64" and record["bytes"] <= 131072,
            "Unsupported integer tensor")
    data = small(path)
    require(len(data) == record["bytes"] and digest(data) == record["sha256"],
            "Integer tensor identity mismatch")
    return [value[0] for value in struct.iter_unpack("<q", data)]


def positions(ids, types, mask, grid):
    """Exact single still image, unpadded B1 specialization of get_rope_index."""
    require(0 < len(ids) <= 4000 and len(types) == len(mask) == len(ids), "Token geometry")
    require(all(value == 1 for value in mask), "Padding is outside this packet")
    require(len(grid) == 3 and grid[0] == 1 and all(type(v) is int and v > 0 for v in grid),
            "Only one still image is supported")
    require(grid[1] % 2 == grid[2] % 2 == 0, "Spatial merge divisibility")
    require(all(0 <= token < 248320 for token in ids), "Token outside pinned vocabulary")
    require(all(t in (0, 1) and (t == 1) == (token == 248056)
                for token, t in zip(ids, types)), "Image token/type disagreement")
    axes = [[], [], []]
    current = images = 0
    for modality, group in itertools.groupby(types):
        count = sum(1 for _ in group)
        if modality == 0:
            for axis in axes:
                axis.extend(range(current, current + count))
            current += count
        else:
            images += 1
            height, width = grid[1] // 2, grid[2] // 2
            require(images == 1 and count == height * width, "Image span/grid disagreement")
            for h in range(height):
                for w in range(width):
                    axes[0].append(current)
                    axes[1].append(current + h)
                    axes[2].append(current + w)
            current += max(height, width)
    require(images == 1 and all(len(axis) == len(ids) for axis in axes), "Position geometry")
    delta = max(max(axis) for axis in axes) + 1 - len(ids)
    return axes, delta


def tensor_record(filename, data, shape):
    stride, strides = 1, []
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return dict(file=filename, sha256=digest(data), bytes=len(data),
                dtype="I64", shape=shape, strides=strides, byte_order="little")


def package(args):
    deadline = time.monotonic() + 30

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Original 30 second packaging deadline")

    report_data = small(args.processor_report, args.report_sha256)
    report = finite_json(report_data)
    corpus_data = small(args.corpus / "manifest.json", report["corpus_sha256"])
    corpus = finite_json(corpus_data)
    prompt_data = small(args.prompt, report["prompt_sha256"])
    require(report["schema"] == "chandra-processor-reference-v1"
            and report["model"] == "datalab-to/chandra-ocr-2" and report["revision"] == PIN
            and report["device"] == "cpu" and report["processor_files"] == PROCESSOR_PINS,
            "Unpinned CPU processor export")
    require(report.get("processor_profile", "checkpoint-default") == args.profile
            and report.get("processor_kwargs", {}) == PROFILE[args.profile], "Processor profile mismatch")
    require(report["runtime"]["transformers"] == "5.18.0", "Position source version mismatch")
    small(args.rope_source, ROPE_SOURCE_SHA)  # Read source bytes only; never import it.
    caller = small(args.caller_source, CALLER_SOURCE_SHA)
    generation_data = small(args.generation_config, PROCESSOR_PINS["generation_config.json"])
    tokenizer_data = small(args.tokenizer_config, PROCESSOR_PINS["tokenizer_config.json"])
    generation, tokenizer = finite_json(generation_data), finite_json(tokenizer_data)
    # Exact specialization of installed Chandra generate_hf; pinned bytes bind its branches.
    eos = generation["eos_token_id"]
    stop_ids = [eos] if type(eos) is int else list(eos)
    turn_ends = [int(k) for k, v in tokenizer["added_tokens_decoder"].items()
                 if v["content"] == "<|im_end|>"]
    require(len(turn_ends) == 1, "Missing or duplicate turn-end token")
    if turn_ends[0] not in stop_ids:
        stop_ids.append(turn_ends[0])
    require(stop_ids == [248044, 248046], "Unexpected pinned caller stops")
    selected = [f for f in report["fixtures"] if f["fixture_id"] == args.fixture]
    expected = [f for f in corpus["fixtures"] if f["id"] == args.fixture]
    require(len(selected) == len(expected) == 1, "Missing or duplicate fixture")
    fixture, original = selected[0], expected[0]
    require(fixture["encoded_sha256"] == original["image_sha256"]
            and fixture["pixels_sha256"] == original["pixels_sha256"]
            and fixture["dimensions"] == [original["width"], original["height"]]
            and original["mode"] == "RGB" and fixture["image_token_id"] == 248056,
            "Original pixel commitment mismatch")
    rendered = fixture["rendered_prompt"].encode("utf-8")
    require(digest(rendered) == fixture["rendered_prompt_sha256"], "Rendered prompt mismatch")
    tensors = fixture["tensors"]
    require(set(tensors) == {"input_ids", "attention_mask", "mm_token_type_ids",
                             "pixel_values", "image_grid_thw"}, "Unexpected tensor inventory")
    reserved = {"input.png", "prompt.txt", "rendered_prompt.txt", "input-manifest.json",
                "processor-reference.json", "corpus-manifest.json", "chandra-caller-hf.py",
                "generation_config.json", "tokenizer_config.json", "derived-position_ids.raw",
                "derived-text_position_ids.raw", "derived-rope_deltas.raw"}
    filenames = [record["file"] for record in tensors.values()]
    require(len(set(filenames)) == len(filenames) and not (set(filenames) & reserved),
            "Duplicate or reserved tensor filename")
    report_root = args.processor_report.parent
    paths = {}
    for key, record in tensors.items():
        shape = record["shape"]
        require(shape and all(type(d) is int and 0 < d <= 16384 for d in shape), "Tensor shape")
        item_size = {"torch.int64": 8, "torch.float32": 4}.get(record["dtype"])
        require(item_size is not None and record["byte_order"] == "little"
                and type(record["bytes"]) is int
                and math.prod(shape) * item_size == record["bytes"], "Tensor byte geometry")
        stride, wanted = 1, []
        for dim in reversed(shape):
            wanted.insert(0, stride)
            stride *= dim
        require(record["strides"] == wanted, "Only contiguous export is supported")
        paths[key] = child(report_root, record["file"])
        require(paths[key].stat().st_size == record["bytes"], "Tensor length mismatch")
    ids = int64_values(paths["input_ids"], tensors["input_ids"])
    mask = int64_values(paths["attention_mask"], tensors["attention_mask"])
    types = int64_values(paths["mm_token_type_ids"], tensors["mm_token_type_ids"])
    grid = int64_values(paths["image_grid_thw"], tensors["image_grid_thw"])
    n = len(ids)
    require(all(tensors[k]["shape"] == [1, n] for k in ("input_ids", "attention_mask", "mm_token_type_ids"))
            and tensors["image_grid_thw"]["shape"] == [1, 3]
            and tensors["pixel_values"]["dtype"] == "torch.float32"
            and tensors["pixel_values"]["shape"] == [math.prod(grid), 1536], "Pinned tensor layout")
    require(ids == fixture["prompt_token_ids"] and [grid] == fixture["image_grid_thw"]
            and [i for i, token in enumerate(ids) if token == 248056] == fixture["image_token_positions"],
            "Export token/grid disagreement")
    axes, delta = positions(ids, types, mask, grid)
    added = {
        "position_ids": (struct.pack("<" + "q" * (3 * n), *sum(axes, [])), [3, 1, n]),
        "text_position_ids": (struct.pack("<" + "q" * n, *range(n)), [1, n]),
        "rope_deltas": (struct.pack("<q", delta), [1, 1]),
    }
    image = child(args.corpus, original["image"])
    total = sum(r["bytes"] for r in tensors.values()) + image.stat().st_size
    total += len(report_data) + len(corpus_data) + len(prompt_data) + len(rendered)
    total += len(caller) + len(generation_data) + len(tokenizer_data)
    total += sum(len(data) for data, _ in added.values())
    require(total + 65536 <= CAP, "Package exceeds 128 MiB forecast")
    check_time()
    args.output.mkdir(parents=True, exist_ok=False)

    def copy(source, name, size, sha):
        check_time()
        hasher, count = hashlib.sha256(), 0
        with regular(source).open("rb") as src, (args.output / name).open("xb") as dst:
            while chunk := src.read(1024 * 1024):
                check_time()
                count += len(chunk)
                require(count <= size, "Asset grew during packaging")
                hasher.update(chunk)
                dst.write(chunk)
        require(count == size and hasher.hexdigest() == sha, "Asset hash mismatch")

    for key, record in tensors.items():
        copy(paths[key], record["file"], record["bytes"], record["sha256"])
    copy(image, "input.png", image.stat().st_size, original["image_sha256"])
    output_tensors = {key: dict(record, source_dtype=record["dtype"],
                               dtype={"torch.int64": "I64", "torch.float32": "F32"}[record["dtype"]])
                      for key, record in tensors.items()}
    for key, (data, shape) in added.items():
        filename = "derived-" + key + ".raw"
        output_tensors[key] = tensor_record(filename, data, shape)
        (args.output / filename).write_bytes(data)
    for name, data in (("processor-reference.json", report_data), ("corpus-manifest.json", corpus_data),
                       ("prompt.txt", prompt_data), ("rendered_prompt.txt", rendered),
                       ("chandra-caller-hf.py", caller), ("generation_config.json", generation_data),
                       ("tokenizer_config.json", tokenizer_data)):
        check_time()
        with (args.output / name).open("xb") as dst:
            dst.write(data)
    manifest = dict(schema="chandra.directcompute.input.v1", model=report["model"], revision=PIN,
                    processor_report_sha256=digest(report_data), corpus_sha256=digest(corpus_data),
                    processor_files=PROCESSOR_PINS, processor_profile=args.profile,
                    processor_kwargs=PROFILE[args.profile], runtime=report["runtime"],
                    fixture_id=args.fixture, prompt_tokens=n, image_token_id=248056,
                    image_token_positions=fixture["image_token_positions"], image_grid_thw=[grid],
                    image=dict(file="input.png", sha256=original["image_sha256"],
                               pixels_sha256=original["pixels_sha256"], dimensions=fixture["dimensions"], mode="RGB"),
                    prompt=dict(file="prompt.txt", sha256=digest(prompt_data)),
                    rendered_prompt=dict(file="rendered_prompt.txt", sha256=digest(rendered)),
                    tensors=output_tensors, expected_ocr=original["expected"],
                    generation=dict(context_limit=16384, max_output_tokens=12384, stop_token_ids=stop_ids,
                                    stop_token_provenance=dict(caller_source_file="chandra-caller-hf.py",
                                        caller_source_sha256=CALLER_SOURCE_SHA, caller_stop_token_ids=stop_ids,
                                        caller="Installed Chandra 0.2.0 generate_hf; source extraction only",
                                        generation_config_sha256=digest(generation_data),
                                        tokenizer_config_sha256=digest(tokenizer_data),
                                        amd_runtime_policy_verified=False)),
                    positions=dict(origin="source-derived single-image B1 unpadded specialization",
                                   source_sha256=ROPE_SOURCE_SHA, spatial_merge_size=2,
                                   next_decode_position=n + delta, captured_from_amd=False),
                    decoded_pixels_verification="Original corpus and CPU exporter commitments agree; PNG decoding is not repeated here",
                    forecast_bytes=total + 65536, provenance_verified=False, native_qualification=False,
                    reference_eos_token_ids_available=False,
                    scope="Input packaging only; no AMD internal tensor equality, model numerics, OCR or speed qualification")
    data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= 65536, "Manifest exceeds reserved allowance")
    check_time()
    with (args.output / "input-manifest.json").open("xb") as dst:
        dst.write(data)
    return dict(manifest_sha256=digest(data), bytes_forecast=total + 65536,
                native_qualification=False, provenance_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--rope-source", type=Path, required=True)
    parser.add_argument("--caller-source", type=Path, required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--fixture", choices=("tiny", "small", "representative"), required=True)
    parser.add_argument("--profile", choices=PROFILE, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch("[0-9a-f]{64}", args.report_sha256), "Expected SHA-256")
    print(json.dumps(package(args), allow_nan=False))


if __name__ == "__main__":
    main()
