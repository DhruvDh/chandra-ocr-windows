"""Import selected rows of a retained Chandra CPU FP32 teacher export as a diagnostic v2 reference (standard library only).

The source is a verification/export_logits.py output directory (metadata.json plus little-endian FP32 logits.f32); the result is a dump that scripts/native/compare_diagnostics.py loads. Each selected complete 248,320-value row is copied word for word from the stream whose SHA-256 the producer metadata names: no arithmetic, BF16 rounding, NaN repair or tolerance fitting, and no BF16 boundary is declared.

Records keep the phase the producer executed. A cached export (prefill, then one teacher token per advance) yields its prefill row and decode steps. A monolithic teacher-forced forward yields prefill rows only, and a row is admissible only as the last prompt row of an input manifest whose prompt is exactly that row's consumed prefix. Conditioning is the manifest's prompt rows followed by the producer's own recorded teacher rows, never free-generated history.

Three-axis positions were never captured by these producers. They are the prepared input's pinned-source derivation, re-derived here and bound to the producer's modeling-source hash, and every output labels them source-derived. The producer metadata records no safetensors digest, so a conversion requires an explicit weights attestation that the output records as such.

The output is a fresh directory holding import-report.json and, unless --dry-run, the verified dump in reference/. Every byte any outcome can write there is forecast from the exact serialized dump and a bound on the serialized report before the directory is created. Nothing here imports Torch, loads a model, uses a GPU or qualifies CPU, native or AMD OCR results.
"""
import argparse
from array import array
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
try:
    import resource
except ImportError:  # Windows has no getrusage; peak RSS is then reported as unavailable.
    resource = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _sibling(name):
    spec = importlib.util.spec_from_file_location(f"chandra_native_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cd = _sibling("compare_diagnostics")  # Dump format, strict JSON, write_dump and load_dump; used read-only.
packager = _sibling("prepare_input")  # Pinned processor hashes and the B1 position derivation; used read-only.
Refusal, require = cd.Refusal, cd.require

REPORT_SCHEMA = "chandra.directcompute.cpu-reference-import.v1"
PRODUCER_KIND = "transformers_cpu_fp32_teacher_export"  # Deliberately not "cpu_": an imported graph export is no oracle.
PIN = packager.PIN
MODEL_SHA256 = "0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847"  # provenance/model.json model.safetensors
MODEL_BYTES = 10591220088
VOCABULARY = cd.VOCABULARY
ROW_BYTES = 4 * VOCABULARY
IMAGE_TOKEN = 248056
CACHED, MONOLITHIC = "cached-teacher-decode", "monolithic-teacher-prefill"
MIB = 1024 * 1024
HARD_ROWS, HARD_OUTPUT, HARD_INPUT, HARD_DEADLINE = 64, 256 * MIB, 16 * 1024 * MIB, 7200.0
METADATA_LIMIT, MANIFEST_LIMIT, TENSOR_LIMIT, SOURCE_LIMIT = 8 * MIB, MIB, 256 * MIB, 4 * MIB
JSON_DEPTH = 64  # Owner JSON nesting bound; producer metadata nests five levels.
ERROR_LIMIT = 4096  # JSON-encoded bytes of the failure message a report may record.
WIDEST_FLOAT, WIDEST_INT = -sys.float_info.max, -2 ** 63  # Longest JSON forms of a float and of a 64-bit integer.
# SHA-256 of the compare_diagnostics.write_dump source whose exact output forecast_dump reproduces before any write.
WRITER_SHA256 = "d689a4e04308c39c63ebca06c80ba771f4ade4d158f87562d7135831d347a10d"
EVIDENCE = "verification/evidence/numerical-2026-10-03.json"
# Files verification/export_logits.py writes into a teacher export directory; anything else is refused.
EXPORT_ENTRIES = {"metadata.json", "logits.f32", "admission.json", "logits.partial.f32", "activation.json", "activation.f32"}
REQUIRED_FIELDS = {"schema_version", "context", "inputs", "runtime", "response_sha256", "states", "logits"}
OPTIONAL_FIELDS = {"elapsed_seconds", "peak_rss_kib", "memory_admission", "memory_after_load", "memory_completion",
                   "memory_positions", "cache_equivalence"}
CONTEXT_FIELDS = {"vocabulary_size", "model_revision", "tokenizer_sha256", "config_sha256", "image_sha256", "prompt_sha256",
                  "processor_profile", "positions", "prefix_token_ids", "target_token_ids"}
PROCESSOR_TENSORS = {"input_ids": ("torch.int64", "I64"), "attention_mask": ("torch.int64", "I64"),
                     "mm_token_type_ids": ("torch.int64", "I64"), "pixel_values": ("torch.float32", "F32"),
                     "image_grid_thw": ("torch.int64", "I64")}
MANIFEST_TENSORS = set(PROCESSOR_TENSORS) | {"position_ids", "text_position_ids", "rope_deltas"}
ITEM_BYTES = {"I64": 8, "F32": 4}
# Exact native input ABI of each role (ChandraNative/runtime/inference.cpp input()): dtype and shape, "P" for the prompt rows,
# None for the 1..32768 original patch rows. Only a single still image is imported, so the grid is [1, 3].
NATIVE_ABI = {"input_ids": ("I64", [1, "P"]), "attention_mask": ("I64", [1, "P"]), "mm_token_type_ids": ("I64", [1, "P"]),
              "text_position_ids": ("I64", [1, "P"]), "position_ids": ("I64", [3, 1, "P"]), "rope_deltas": ("I64", [1, 1]),
              "image_grid_thw": ("I64", [1, 3]), "pixel_values": ("F32", [None, 1536])}
PATCH_ROWS = 32768
# inference.cpp reads descriptor bytes and strides with unsignedNumber, which admits only JSON integers 0..2**64-1 (nlohmann reads a
# larger literal as a float), and bounds each descriptor's byte extent by its 256 MiB inputByteLimit.
NATIVE_UNSIGNED, NATIVE_TENSOR_BYTES = 2 ** 64, 256 * MIB
HEX = re.compile(r"[0-9a-f]{64}")
SAFE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
clock = time.monotonic  # One monotonic clock for the whole command; tests may substitute it.


class Failure(Exception):
    """The import failed after its fresh output directory was created."""


class Deadline:
    def __init__(self, start, seconds):
        self.seconds, self.end = seconds, start + seconds

    def check(self, stage):
        require(clock() < self.end, f"The {self.seconds:g}-second monotonic deadline expired during {stage}")


class Budget:
    """Every byte read from owner inputs counts against --max-input-bytes."""

    def __init__(self, limit):
        self.limit, self.used = limit, 0

    def take(self, count, what):
        self.used += count
        require(self.used <= self.limit, f"Reading {what} exceeds --max-input-bytes {self.limit}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def hex64(value, what):
    require(isinstance(value, str) and HEX.fullmatch(value), f"{what} must be a lowercase SHA-256")
    return value


def naturals(value, what, bound=None):
    require(isinstance(value, list) and all(cd.natural(v) and (bound is None or v < bound) for v in value),
            f"{what} must be a list of nonnegative integers" + ("" if bound is None else f" below {bound}"))
    return value


def absolute(text, what):
    """An absolute, normalized path none of whose existing components is a symlink."""
    require(isinstance(text, str) and os.path.isabs(text) and os.path.normpath(text) == text,
            f"{what} must be an absolute normalized path")
    require(os.path.realpath(text) == text, f"{what} must not traverse a symlink: {text}")
    return Path(text)


def open_regular(path, limit, what):
    """Open a regular file without following a final symlink; refuse it beyond limit bytes."""
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        raise Refusal(f"{what} is missing: {path}") from None
    require(stat.S_ISREG(before.st_mode), f"{what} must be a regular non-symlink file: {path}")
    require(before.st_size <= limit, f"{what} exceeds its {limit}-byte bound: {path}")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    except OSError as error:
        raise Refusal(f"{what} cannot be opened without following links: {error.strerror}") from None
    handle = os.fdopen(fd, "rb", buffering=0)
    after = os.fstat(handle.fileno())
    if not (stat.S_ISREG(after.st_mode) and (after.st_dev, after.st_ino, after.st_size) == (before.st_dev, before.st_ino, before.st_size)):
        handle.close()
        raise Refusal(f"{what} changed while it was opened: {path}")
    return handle, before.st_size


def read_file(path, limit, what, budget, deadline):
    handle, size = open_regular(path, limit, what)
    with handle:
        budget.take(size, what)
        chunks, total = [], 0
        while True:
            deadline.check(f"reading {what}")
            chunk = handle.read(min(MIB, size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            require(total <= size, f"{what} grew while it was read: {path}")
            chunks.append(chunk)
    require(total == size, f"{what} is truncated: {path}")
    return b"".join(chunks)


def load_json(data, what):
    """Strict JSON (no duplicate keys or NaN/Infinity tokens) whose every number is finite and whose nesting is bounded."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise Refusal(f"{what} is not UTF-8") from None
    try:
        value = cd.strict_json(text, what)
    except (ValueError, RecursionError) as error:  # Integers beyond Python's digit limit, or nesting beyond the decoder's stack.
        raise Refusal(f"{what}: JSON cannot be represented: {type(error).__name__}") from None
    finite_json(value, what)
    return value


def finite_json(value, what):
    """Python decodes an overflowing literal such as 1e999 to infinity, which strict_json's token check never sees."""
    stack = [(value, 0, None)]
    while stack:
        item, depth, key = stack.pop()
        if isinstance(item, float):
            require(math.isfinite(item), f"{what}: the number under {key!r:.80} overflows to a nonfinite value")
        elif isinstance(item, (dict, list)):
            require(depth < JSON_DEPTH, f"{what}: JSON nesting exceeds {JSON_DEPTH} levels")
            stack.extend((v, depth + 1, k) for k, v in (item.items() if isinstance(item, dict) else enumerate(item)))


def int64s(data):
    require(len(data) % 8 == 0, "An I64 tensor's bytes are not a whole number of 8-byte values")
    values = array("q")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


# ---------------------------------------------------------------- producer metadata and route

def check_tensor_records(records, what):
    require(isinstance(records, dict) and records, f"{what} must be a non-empty object")
    for name, record in records.items():
        require(isinstance(record, dict) and set(record) == {"shape", "dtype", "finite", "sha256"}, f"{what} {name} is malformed")
        naturals(record["shape"], f"{what} {name} shape")
        require(isinstance(record["dtype"], str) and type(record["finite"]) is bool, f"{what} {name} is malformed")
        hex64(record["sha256"], f"{what} {name} sha256")
        require(record["finite"], f"{what} {name} is recorded nonfinite by its producer")


def key_lengths(snapshot, index):
    """Full-attention cache length the producer recorded after one model call."""
    lengths = set()
    for name, record in snapshot.items():
        parts = name.split(".")
        require(re.fullmatch(r"[0-9]{1,6}\.(keys|values|conv_states|recurrent_states)\.[0-9]{1,6}", name),
                f"producer state {index} names an unknown cache tensor {name!r:.120}")
        if parts[1] in ("keys", "values"):
            require(len(record["shape"]) == 4 and int(parts[0]) % 4 == 3, f"producer state {index} {name} is not a full-attention cache")
            lengths.add(record["shape"][2])
    require(len(lengths) == 1, f"producer state {index} has no single recorded full-attention cache length")
    return lengths.pop()


def check_metadata(meta, route):
    """Validate producer metadata and the attested route against its recorded call structure."""
    fields = set(meta)
    require(REQUIRED_FIELDS <= fields and fields <= REQUIRED_FIELDS | OPTIONAL_FIELDS,
            f"Producer metadata fields differ from a teacher export: unexpected {sorted(fields - REQUIRED_FIELDS - OPTIONAL_FIELDS)}, "
            f"missing {sorted(REQUIRED_FIELDS - fields)}")
    require(type(meta["schema_version"]) is int and meta["schema_version"] == 1, "Producer metadata schema_version must be 1")
    context, runtime, logits = meta["context"], meta["runtime"], meta["logits"]
    require(isinstance(context, dict) and CONTEXT_FIELDS <= set(context) <= CONTEXT_FIELDS | {"teacher_token_ids"},
            "Producer context fields differ from a teacher export")
    require(context["vocabulary_size"] == VOCABULARY and context["model_revision"] == PIN,
            "Producer vocabulary or model revision differs from the pinned checkpoint")
    for name in ("tokenizer_sha256", "config_sha256", "image_sha256", "prompt_sha256"):
        hex64(context[name], f"context {name}")
    hex64(meta["response_sha256"], "response_sha256")
    require(isinstance(context["processor_profile"], dict), "Producer processor profile must be an object")
    prefix = naturals(context["prefix_token_ids"], "prefix_token_ids", VOCABULARY)
    targets = naturals(context["target_token_ids"], "target_token_ids", VOCABULARY)
    rows = naturals(context["positions"], "context positions")
    require(prefix, "Producer prefix is empty")
    require(isinstance(meta["inputs"], dict) and set(meta["inputs"]) == set(PROCESSOR_TENSORS),
            "Producer input tensors differ from the processor's five tensors")
    check_tensor_records(meta["inputs"], "producer input")
    for name, (dtype, _) in PROCESSOR_TENSORS.items():
        require(meta["inputs"][name]["dtype"] == dtype, f"producer input {name} dtype differs")
    require(isinstance(runtime, dict) and runtime.get("device") == "cpu" and runtime.get("dtype") == "torch.float32",
            "Only unrestricted CPU FP32 exports are imported")
    require(runtime.get("transformers") == "5.18.0" and isinstance(runtime.get("torch"), str),
            "Producer Transformers version differs from the pinned position source")
    require(runtime.get("model_source_sha256") == packager.ROPE_SOURCE_SHA,
            "Producer modeling source differs from the pinned source that derives input positions")
    hex64(runtime.get("script_sha256"), "runtime script_sha256")
    require(isinstance(logits, dict) and set(logits) == {"file", "shape", "dtype", "byte_order", "sha256"}
            and logits["file"] == "logits.f32" and logits["dtype"] == "float32" and logits["byte_order"] == "little",
            "Producer logits record must name little-endian float32 logits.f32")
    shape = naturals(logits["shape"], "logits shape")
    require(len(shape) == 2 and shape[0] > 0 and shape[1] == VOCABULARY, "Producer logits shape must be [rows, 248320]")
    hex64(logits["sha256"], "logits sha256")
    require(len(rows) == shape[0], "Producer row positions differ from the logits row count")
    states = meta["states"]
    require(isinstance(states, list) and states, "Producer cache snapshots are missing")
    for index, snapshot in enumerate(states):
        check_tensor_records(snapshot, f"producer state {index}")
    lengths = [key_lengths(snapshot, index) for index, snapshot in enumerate(states)]
    p = len(prefix)
    if route == CACHED:
        require("teacher_token_ids" not in context, "Metadata records a full teacher forward, contradicting the cached-decode route")
        require(targets and rows == list(range(p - 1, p + len(targets))),
                "Cached-decode rows must be the last prompt row followed by one row per teacher target")
        require(lengths == [p + i for i in range(len(targets) + 1)],
                f"Recorded cache lengths {lengths} contradict a prefill followed by single-token cached advances")
        teacher = targets
    else:
        require(route == MONOLITHIC, f"Unknown route {route!r}")
        require("teacher_token_ids" in context, "Metadata records no teacher-forced forward, contradicting the monolithic route")
        teacher = naturals(context["teacher_token_ids"], "teacher_token_ids", VOCABULARY)
        require(teacher, "A monolithic teacher forward must record its teacher tokens")
        require(rows == sorted(set(rows)) and all(p - 1 <= r < p - 1 + len(teacher) for r in rows),
                "Monolithic rows must be distinct ascending rows of the teacher-forced sequence")
        require(targets == [teacher[r - p + 1] for r in rows], "Recorded targets differ from the teacher tokens at the recorded rows")
        require(lengths == [p + len(teacher)], f"Recorded cache lengths {lengths} contradict one forward over prompt and teacher rows")
        require("cache_equivalence" not in meta and "memory_positions" not in meta,
                "Metadata records cached-advance products, contradicting the monolithic route")
    if "cache_equivalence" in meta:
        require(isinstance(meta["cache_equivalence"], dict), "cache_equivalence must be an object")
    return {"prefix": prefix, "teacher": teacher, "rows": rows, "shape": shape, "lengths": lengths}


def check_admission(admission, meta, route):
    """admission.json is the same metadata written before the model ran; it must not contradict metadata.json."""
    require(set(admission) == {"schema_version", "context", "inputs", "runtime", "response_sha256", "states"}
            and admission["states"] == [], "admission.json fields differ from the producer's admission record")
    runtime = {k: v for k, v in meta["runtime"].items() if k != "attention_config"}  # Added only after loading.
    later = {"positions", "target_token_ids", "teacher_token_ids"} if route == MONOLITHIC else set()
    same = lambda c: {k: v for k, v in c.items() if k not in later} if isinstance(c, dict) else c
    require(admission["schema_version"] == meta["schema_version"] and admission["inputs"] == meta["inputs"]
            and admission["runtime"] == runtime and admission["response_sha256"] == meta["response_sha256"]
            and same(admission["context"]) == same(meta["context"]),
            "admission.json contradicts metadata.json")
    if route == MONOLITHIC:
        early = admission["context"].get("target_token_ids")
        require(isinstance(early, list) and early == meta["context"]["teacher_token_ids"][:len(early)],
                "admission.json targets are not the leading teacher tokens")


# ---------------------------------------------------------------- prepared input and association

def load_manifest(path, expected, budget, deadline):
    data = read_file(path, MANIFEST_LIMIT, "input manifest", budget, deadline)
    require(sha256(data) == expected, "Input manifest SHA-256 differs from --input-sha256")
    m = load_json(data, "input manifest")
    require(m.get("schema") == "chandra.directcompute.input.v1" and m.get("model") == "datalab-to/chandra-ocr-2"
            and m.get("revision") == PIN, "Input manifest is not a pinned chandra.directcompute.input.v1 manifest")
    prompt = m.get("prompt_tokens")
    require(cd.natural(prompt) and 0 < prompt <= 4000, "Input manifest prompt_tokens is invalid")
    tensors = m.get("tensors")
    require(isinstance(tensors, dict) and set(tensors) <= MANIFEST_TENSORS
            and set(tensors) >= MANIFEST_TENSORS - {"text_position_ids"}, "Input manifest tensor inventory differs")
    # Integers associate() compares by value, where Python equality admits 10.0 or true. Native reads image_token_id and
    # next_decode_position with unsignedNumber, and its nlohmann == on the two lists never equates true with 1.
    origin, grid = m.get("positions"), m.get("image_grid_thw")
    require(cd.natural(m.get("image_token_id")) and (not isinstance(origin, dict) or cd.natural(origin.get("next_decode_position"))),
            "Input manifest image_token_id and positions.next_decode_position must be JSON integers")
    naturals(m.get("image_token_positions"), "Input manifest image_token_positions")
    require(isinstance(grid, list), "Input manifest image_grid_thw must be a list")
    for row in grid:
        naturals(row, "Input manifest image_grid_thw rows")
    root, raw, names = path.parent, {}, set()

    def child(name, what):
        require(isinstance(name, str) and SAFE.fullmatch(name) and ".." not in name and name not in names, f"{what} has an unsafe or repeated filename")
        names.add(name)
        return root / name

    for key, record in tensors.items():
        # Each role's exact native dtype, rank, shape, byte extent and contiguous strides, before any of its bytes are read or decoded.
        require(isinstance(record, dict) and record.get("byte_order") == "little", f"input tensor {key} storage is unsupported")
        dtype, dims = NATIVE_ABI[key]
        shape = naturals(record.get("shape"), f"input tensor {key} shape")
        wanted = [prompt if dim == "P" else dim for dim in dims]
        require(record.get("dtype") == dtype and len(shape) == len(wanted)
                and all(0 < have <= PATCH_ROWS if want is None else have == want for have, want in zip(shape, wanted)),
                f"input tensor {key} must be {dtype} [{', '.join('1..32768' if w is None else str(w) for w in wanted)}] under the native "
                f"input ABI; the manifest declares {record.get('dtype')!r:.40} {shape!s:.80}")
        count, strides = 1, []
        for dim in reversed(shape):
            strides.insert(0, count)
            count *= dim
        # Python equality admits 96.0 or true for an integer; the native reader refuses both, so exact types and ranges come first.
        size, declared = record.get("bytes"), record.get("strides")
        require(cd.natural(size) and size <= NATIVE_TENSOR_BYTES, f"input tensor {key} bytes must be a JSON integer from 0 to "
                f"{NATIVE_TENSOR_BYTES} under the native input ABI; the manifest declares {size!r:.40}")
        require(size == count * ITEM_BYTES[dtype], f"input tensor {key} bytes differ from its {dtype} shape")
        require(isinstance(declared, list) and len(declared) == len(shape) and all(cd.natural(v) and v < NATIVE_UNSIGNED for v in declared),
                f"input tensor {key} must declare the contiguous element strides {strides} the native input ABI requires as "
                f"{len(shape)} JSON integers below 2**64; the manifest declares {declared!r:.80}")
        require(declared == strides, f"input tensor {key} must declare the contiguous element strides {strides} the native input ABI requires")
        data = read_file(child(record.get("file"), f"input tensor {key}"), TENSOR_LIMIT, f"input tensor {key}", budget, deadline)
        require(len(data) == record["bytes"] and sha256(data) == hex64(record.get("sha256"), f"input tensor {key} sha256"),
                f"input tensor {key} differs from its manifest commitment")
        raw[key] = data
    for name in ("image", "prompt"):
        entry = m.get(name)
        require(isinstance(entry, dict), f"Input manifest {name} commitment missing")
        data = read_file(child(entry.get("file"), f"input {name}"), TENSOR_LIMIT, f"input {name}", budget, deadline)
        require(sha256(data) == hex64(entry.get("sha256"), f"input {name} sha256"), f"input {name} file differs from its manifest commitment")
    return m, raw


def associate(meta, checked, m, raw, route):
    """Prove the producer consumed this prepared input's prompt; return prompt rows, positions and the consumed sequence."""
    context, inputs = meta["context"], meta["inputs"]
    pins = packager.PROCESSOR_PINS
    files = m.get("processor_files") if isinstance(m.get("processor_files"), dict) else {}
    require(context["tokenizer_sha256"] == files.get("tokenizer.json") == pins["tokenizer.json"], "Tokenizer commitment differs")
    require(context["config_sha256"] == files.get("config.json") == pins["config.json"], "Model config commitment differs")
    require(context["image_sha256"] == m["image"]["sha256"], "Image commitment differs")
    require(context["prompt_sha256"] == m["prompt"]["sha256"], "Prompt commitment differs")
    require(context["processor_profile"] == m.get("processor_kwargs"), "Processor profile differs")
    origin = m.get("positions")
    require(isinstance(origin, dict) and origin.get("source_sha256") == meta["runtime"]["model_source_sha256"],
            "Input positions were derived from a different modeling source than the producer imported")
    prefix, teacher = checked["prefix"], checked["teacher"]
    p0, prompt = len(prefix), m["prompt_tokens"]
    sequence = prefix + teacher
    if route == CACHED:
        require(prompt == p0, f"A cached export prefilled exactly {p0} processor rows; the manifest declares {prompt}")
    else:
        require(p0 <= prompt <= len(sequence), f"The manifest prompt of {prompt} rows is not a prefix of the producer's {len(sequence)} consumed rows")
    tensors = m["tensors"]
    for name, (_, dtype) in PROCESSOR_TENSORS.items():
        record, mine = inputs[name], tensors[name]
        require(mine["dtype"] == dtype, f"input tensor {name} dtype differs from the producer's")
        if name in ("pixel_values", "image_grid_thw"):
            require(mine["shape"] == record["shape"] and mine["sha256"] == record["sha256"], f"input tensor {name} differs from the producer's")
        else:
            # Rows beyond the processor prompt must be exactly the producer's appended teacher rows (checked below).
            require(record["shape"] == [1, p0] and mine["shape"] == [1, prompt] and sha256(raw[name][:8 * p0]) == record["sha256"],
                    f"input tensor {name} differs from the producer's processor tensor")
    ids, mask, types = int64s(raw["input_ids"]), int64s(raw["attention_mask"]), int64s(raw["mm_token_type_ids"])
    require(ids == sequence[:prompt], "Manifest prompt tokens differ from the producer's consumed tokens")
    require(all(v == 1 for v in mask[p0:]) and all(v == 0 for v in types[p0:]),
            "Manifest rows beyond the processor prompt differ from the producer's appended teacher rows")
    grid = int64s(raw["image_grid_thw"])
    try:
        axes, delta = packager.positions(ids, types, mask, grid)
    except ValueError as error:
        raise Refusal(f"Manifest prompt does not satisfy the pinned-source position derivation: {error}") from None
    require(int64s(raw["position_ids"]) == axes[0] + axes[1] + axes[2] and tensors["position_ids"]["shape"] == [3, 1, prompt],
            "Manifest three-axis positions differ from the pinned-source derivation")
    require(int64s(raw["rope_deltas"]) == [delta] and tensors["rope_deltas"]["shape"] == [1, 1], "Manifest rope delta differs from the derivation")
    if "text_position_ids" in raw:
        require(int64s(raw["text_position_ids"]) == list(range(prompt)), "Manifest text positions differ from causal row order")
    following = prompt + delta
    require(following == max(max(axis) for axis in axes) + 1 == origin.get("next_decode_position"),
            "Manifest next decode position differs from the derived positions")
    require(m.get("image_token_id") == IMAGE_TOKEN and m.get("image_token_positions") == [i for i, t in enumerate(ids) if t == IMAGE_TOKEN]
            and m.get("image_grid_thw") == [grid], "Manifest image token rows or grid differ from its tensors")
    conditioning = [[ids[r], axes[0][r], axes[1][r], axes[2][r]] for r in range(prompt)]
    conditioning += [[sequence[r]] + [following + r - prompt] * 3 for r in range(prompt, len(sequence))]
    return {"prompt": prompt, "processor_prompt": p0, "ids": ids, "conditioning": conditioning, "next_position": following,
            "rope_delta": delta, "grid": grid}


# ---------------------------------------------------------------- records

def prefix_digests(manifest_sha, prompt, rows, wanted):
    """Canonical consumed-prefix SHA-256 for each requested prefix length, computed independently of the comparator."""
    hasher = hashlib.sha256(f"{cd.PREFIX_SCHEMA}\ninput_manifest_sha256 {manifest_sha}\nprompt_rows {prompt}\n".encode())
    result = {}
    for index, row in enumerate(rows):
        hasher.update(f"{index} {row[0]} {row[1]} {row[2]} {row[3]}\n".encode())
        if index + 1 in wanted:
            result[index + 1] = hasher.hexdigest()
    return result


def plan_records(route, checked, bound, selected):
    """Record specifications for the selected source rows; inadmissible rows are refused with what root must obtain."""
    prompt, conditioning, lengths = bound["prompt"], bound["conditioning"], checked["lengths"]
    p0, specs, refused = bound["processor_prompt"], [], []
    for source in selected:
        absolute = checked["rows"][source]
        n = absolute + 1
        row = conditioning[absolute]
        coordinate = {"absolute_row": absolute, "token_id": row[0], "position": row[1:]}
        spec = {"stage": "text.logits", "logical_shape": [1, VOCABULARY], "selected_rows": [0], "payload_shape": [1, VOCABULARY],
                "complete_tensor": True, "bf16_rounding": "none"}
        if route == CACHED and source > 0:
            step = source - 1
            spec.update(phase="decode", decode_step=step, produces_generated_index=step + 1,
                        decode={"step": step, "consumed_generated_index": step, "consumed_token_id": row[0],
                                "logits_produce_generated_index": step + 1},
                        cache={"cache_length_after": lengths[source], "cache_length_source": "producer_recorded_full_attention_keys",
                               "request_tokens_before_call": absolute, "generated_before_call": step, "call_tokens": 1,
                               "call_first_row": 0, "call_row_count": 1})
        elif n == prompt:
            coordinate["source"] = "vision" if row[0] == IMAGE_TOKEN else "text"
            call = lengths[0]
            spec.update(phase="prefill", decode_step=None, produces_generated_index=0, decode=None,
                        cache={"cache_length_after": call, "cache_length_source": "producer_recorded_full_attention_keys",
                               "request_tokens_before_call": 0, "generated_before_call": 0, "call_tokens": call,
                               "call_first_row": absolute, "call_row_count": 1})
        else:
            refused.append((source, absolute))
            continue
        spec["coordinates"] = [coordinate]
        spec["reference_provenance"] = {"source_row": source, "source_byte_offset": source * ROW_BYTES, "producer_absolute_row": absolute,
                                        "response_generated_index": absolute - (p0 - 1), "position_origin": "source_derived",
                                        "phase_origin": f"{route} route checked against producer-recorded cache lengths"}
        spec["prefix_rows"] = n
        specs.append(spec)
    if refused:
        source, absolute = refused[0]
        teacher_rows = absolute + 1 - p0
        cached = f" or a cached teacher-decode export reaching decode step {teacher_rows - 1}" if teacher_rows > 0 else ""
        raise Refusal(
            f"Monolithic teacher-forced rows {[s for s, _ in refused][:8]} are prefill values whose consumed prefix is not the manifest's "
            f"{prompt} prompt rows (source row {source} at absolute row {absolute} consumes {absolute + 1} rows). A prefill logits row joins "
            "only as the last row of a prompt that is exactly its consumed prefix; the v2 format would otherwise call it a cached decode step, "
            f"which it is not, so it is refused. To compare source row {source}, root must obtain an input manifest whose prompt is exactly "
            f"those {absolute + 1} rows (the {p0} processor rows followed by the first {teacher_rows} teacher tokens) and a native prefill "
            f"on it{cached}.")
    return specs


def key_of(spec):
    return cd.key_text(spec["stage"], None, spec["phase"], spec["decode_step"], 0, None)


# ---------------------------------------------------------------- import

def check_arguments(args):
    for name in ("metadata_sha256", "input_sha256"):
        hex64(getattr(args, name), f"--{name.replace('_', '-')}")
    require(0 < args.max_rows <= HARD_ROWS, f"--max-rows must be within 1..{HARD_ROWS}")
    require(0 < args.max_output_bytes <= HARD_OUTPUT, f"--max-output-bytes must be within 1..{HARD_OUTPUT}")
    require(0 < args.max_input_bytes <= HARD_INPUT, f"--max-input-bytes must be within 1..{HARD_INPUT}")
    require(re.fullmatch(r"[0-9]{1,9}(,[0-9]{1,9})*", args.rows or ""), "--rows must be comma-separated source row indices")
    rows = [int(v) for v in args.rows.split(",")]
    require(rows == sorted(set(rows)), "--rows must be sorted ascending without repeats")
    require(len(rows) <= args.max_rows, f"{len(rows)} selected rows exceed --max-rows {args.max_rows}")
    if args.attest_model_sha256 is not None:
        require(args.attest_model_sha256 == MODEL_SHA256,
                "--attest-model-sha256 names weights other than the pinned provenance/model.json model.safetensors")
        basis = args.attestation_basis
        require(isinstance(basis, str) and 0 < len(basis) <= 1000 and "\n" not in basis and "\r" not in basis,
                "--attest-model-sha256 requires a one-line --attestation-basis of at most 1000 characters")
    else:
        require(args.attestation_basis is None, "--attestation-basis is only meaningful with --attest-model-sha256")
    export, manifest = absolute(args.export_dir, "--export-dir"), absolute(args.input_manifest, "--input-manifest")
    output = absolute(args.output, "--output")
    require(os.path.isdir(output.parent), "--output parent must be an existing directory")
    require(not os.path.lexists(output), "--output must name a fresh path; existing output is never overwritten")
    source = absolute(args.producer_source, "--producer-source") if args.producer_source else None
    return rows, export, manifest, output, source


def curated_record(metadata_sha):
    """Name of the tracked curated evidence entry for this exact metadata, if any (informational only)."""
    try:
        evidence = load_json((ROOT / EVIDENCE).read_bytes(), EVIDENCE)
        for section in ("exports", "cache_checks"):
            for entry in evidence.get(section, []):
                if entry.get("metadata_sha256") == metadata_sha:
                    return {"record": EVIDENCE, "section": section, "run_id": entry.get("run_id")}
    except (OSError, Refusal, AttributeError, TypeError):
        pass
    return None


def stream_logits(path, rows, selected, budget, deadline):
    """Hash the complete payload once and keep only the selected complete rows from the same stream."""
    handle, size = open_regular(path, rows * ROW_BYTES, "logits payload")
    kept, hasher = {}, hashlib.sha256()
    with handle:
        require(size == rows * ROW_BYTES, f"logits payload is truncated: {size} bytes where its declared shape needs {rows * ROW_BYTES}")
        buffer = bytearray(ROW_BYTES)
        view = memoryview(buffer)
        for index in range(rows):
            deadline.check("logits streaming")
            filled = 0
            while filled < ROW_BYTES:
                count = handle.readinto(view[filled:])
                require(count, f"logits payload is truncated at row {index}")
                filled += count
            budget.take(ROW_BYTES, "logits payload")
            hasher.update(view)
            if index in selected:
                kept[index] = bytes(buffer)
        require(handle.read(1) == b"", "logits payload grew beyond its declared shape")
    return hasher.hexdigest(), kept


def nonfinite_words(data):
    words = array(cd.WORD)
    words.frombytes(data)
    if sys.byteorder != "little":
        words.byteswap()
    return sum(1 for w in words if (w & 0x7F800000) == 0x7F800000)


def native_order(data):
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def exact_copy_probe():
    """write_dump copies an array('f') initializer; prove that copy keeps signaling/quiet NaN payloads, signed zero and subnormals."""
    words = array(cd.WORD, [0x7F800001, 0xFFC00001, 0x7FC12345, 0x80000000, 0x00000001, 0x7F800000, 0xFF7FFFFF, 0x3F800001])
    probe = array("f")
    probe.frombytes(words.tobytes())
    require(array("f", probe).tobytes() == probe.tobytes(), "This Python does not copy float32 arrays word for word")


def peak_rss_kib():
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // 1024 if sys.platform == "darwin" else peak


def written_bytes(root, skip=None):
    """Regular-file bytes under root, the quantity --max-output-bytes bounds; directory metadata is the filesystem's."""
    total, pending = 0, [root]
    while pending:
        with os.scandir(pending.pop()) as listing:
            for entry in listing:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif entry.is_file(follow_symlinks=False) and Path(entry.path) != skip:
                    total += entry.stat(follow_symlinks=False).st_size
    return total


def sync_directory(path):
    """Best-effort durability of a directory's entries; Windows has no directory fsync."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


promote = os.rename  # The single step that makes reference/ exist; tests may substitute it.


def prepare(args, start):
    """Everything before output creation: authentication, association, selection and payload extraction."""
    deadline = Deadline(start, args.deadline_seconds)
    require(0 < args.deadline_seconds <= HARD_DEADLINE, f"--deadline-seconds must be within (0, {HARD_DEADLINE:g}]")
    selected, export, manifest_path, output, source_path = check_arguments(args)
    budget = Budget(args.max_input_bytes)
    deadline.check("argument validation")
    require(os.path.isdir(export) and not os.path.islink(export), "--export-dir must be a directory")
    entries = {}
    with os.scandir(export) as listing:
        for entry in listing:
            require(entry.name in EXPORT_ENTRIES, f"Export directory holds an entry no teacher export writes: {entry.name!r:.120}")
            require(entry.is_file(follow_symlinks=False), f"Export entry must be a regular non-symlink file: {entry.name}")
            entries[entry.name] = entry.stat(follow_symlinks=False).st_size
    require({"metadata.json", "logits.f32"} <= set(entries), "Export directory lacks metadata.json or logits.f32")
    metadata_bytes = read_file(export / "metadata.json", METADATA_LIMIT, "metadata.json", budget, deadline)
    require(sha256(metadata_bytes) == args.metadata_sha256, "metadata.json SHA-256 differs from --metadata-sha256")
    meta = load_json(metadata_bytes, "metadata.json")
    checked = check_metadata(meta, args.route)
    rows_total = checked["shape"][0]
    require(all(r < rows_total for r in selected), f"--rows names a row outside the export's {rows_total} logits rows")
    if "admission.json" in entries:
        check_admission(load_json(read_file(export / "admission.json", METADATA_LIMIT, "admission.json", budget, deadline), "admission.json"),
                        meta, args.route)
    partial = "logits.partial.f32" in entries
    require(not (partial and args.route == MONOLITHIC), "logits.partial.f32 is written only by cached advances; it contradicts the monolithic route")
    m, raw = load_manifest(manifest_path, args.input_sha256, budget, deadline)
    bound = associate(meta, checked, m, raw, args.route)
    specs = plan_records(args.route, checked, bound, selected)
    digests = prefix_digests(args.input_sha256, bound["prompt"], bound["conditioning"],
                             {s["prefix_rows"] for s in specs} | {len(bound["conditioning"])})
    producer_source = None
    if source_path is not None:
        data = read_file(source_path, SOURCE_LIMIT, "--producer-source", budget, deadline)
        require(sha256(data) == meta["runtime"]["script_sha256"], "--producer-source differs from the producer's recorded script_sha256")
        producer_source = sha256(data)
    attested = args.attest_model_sha256 is not None
    require(attested or args.dry_run,
            "The CPU producer metadata records no safetensors SHA-256, so its weights cannot be bound to the native model commitment. "
            f"Pass --attest-model-sha256 {MODEL_SHA256} with --attestation-basis only if root holds evidence that this export loaded "
            "the pinned checkpoint; the output then records an owner attestation, not a producer commitment. --dry-run reports without it.")
    big = rows_total * ROW_BYTES * (2 if partial else 1)
    require(budget.used + big <= budget.limit, f"Streaming {big} payload bytes would exceed --max-input-bytes {budget.limit}")
    payload = len(specs) * ROW_BYTES  # The exact forecast follows streaming; payloads alone already refuse an impossible conversion.
    require(args.dry_run or payload <= args.max_output_bytes,
            f"The {payload} selected payload bytes alone exceed --max-output-bytes {args.max_output_bytes}")
    deadline.check("authentication")
    digest, kept = stream_logits(export / "logits.f32", rows_total, set(selected), budget, deadline)
    require(digest == meta["logits"]["sha256"], "logits.f32 SHA-256 differs from the producer metadata")
    if partial:
        partial_digest, _ = stream_logits(export / "logits.partial.f32", rows_total, set(), budget, deadline)
        require(partial_digest == digest, "logits.partial.f32 differs from logits.f32; the export is contradictory")
    nonfinite = {}
    for spec in specs:
        data = kept[spec["reference_provenance"]["source_row"]]
        spec["source_bytes"], spec["source_sha256"] = data, sha256(data)
        count = nonfinite_words(data)
        if count:
            nonfinite[spec["reference_provenance"]["source_row"]] = count
    require(not nonfinite or args.nonfinite == "preserve",
            f"Selected rows hold nonfinite words {nonfinite}; the exporter refuses nonfinite logits, so this contradicts its contract. "
            "--nonfinite preserve keeps such words unchanged as unqualified evidence; nothing is repaired.")
    exact_copy_probe()
    deadline.check("payload authentication")
    return {"deadline": deadline, "budget": budget, "export": export, "entries": entries, "output": output, "meta": meta,
            "metadata_bytes": len(metadata_bytes), "checked": checked, "manifest": m, "bound": bound, "specs": specs,
            "digests": digests, "selected": selected, "attested": attested, "producer_source": producer_source,
            "partial": partial, "nonfinite": nonfinite}


def describe(args, s):
    """Commitments, producer identity, plan description and the honest limits of this import."""
    meta, checked, bound = s["meta"], s["checked"], s["bound"]
    runtime = meta["runtime"]
    route_text = ("one prefill call over the processor prompt followed by one cached advance per recorded teacher token"
                  if args.route == CACHED else
                  f"one teacher-forced forward over {len(checked['prefix'])} prompt and {len(checked['teacher'])} teacher rows; every row is prefill")
    missing = [
        {"commitment": "model_safetensors_sha256", "producer_recorded": False,
         "resolution": "owner attestation recorded" if s["attested"] else "absent; conversion refused until attested"},
        {"commitment": "producer_script_bytes", "producer_recorded": "sha256 only",
         "resolution": "supplied bytes match script_sha256" if s["producer_source"] else
                       "bytes not in this repository; route checked against recorded cache lengths only"},
        {"commitment": "three_axis_positions", "producer_recorded": False,
         "resolution": "source-derived from the prepared input; modeling-source SHA-256 equals the producer's"},
        {"commitment": "processor_template_and_preprocessor_files", "producer_recorded": False,
         "resolution": "consumed processor tensors are hash-identical to the manifest's"},
        {"commitment": "response_bytes", "producer_recorded": "sha256 only",
         "resolution": "consumed teacher token IDs are recorded directly and used without retokenization"},
    ]
    limitations = [
        f"Route {args.route}: {route_text}. Producer-recorded full-attention cache lengths {checked['lengths'][:12]} confirm the call structure; "
        "the producer source itself was " + ("matched by hash but not executed or parsed." if s["producer_source"] else "not available for inspection."),
        f"Three-axis positions are source-derived, not captured: prompt rows from the prepared input's pinned-source derivation, re-derived "
        f"here, and teacher rows at next decode position {bound['next_position']} plus their offset.",
        "Weights: " + (f"owner attestation of the pinned safetensors SHA-256; basis: {args.attestation_basis}" if s["attested"] else
                       "no commitment; the producer metadata records none."),
        f"Values are unrestricted CPU FP32 logits from Torch {runtime['torch']} ({runtime.get('attention', 'unrecorded')} attention, "
        f"{runtime.get('threads', 'unrecorded')} threads); no BF16 boundary is declared. They are a retained reference graph, not a "
        "correctness oracle or the CPU higher-precision operator oracle.",
        "Conditioning rows after the prompt are the producer's recorded teacher tokens, not native or CPU free-generated history.",
        "Only complete full-vocabulary logits rows exist in this export; no hidden state, cache payload or vision value is imported.",
    ]
    if args.route == MONOLITHIC:
        limitations.append("Monolithic rows remain prefill and join only prefill values of an input whose prompt is exactly their consumed prefix; "
                           "they never pair with cached decode steps.")
    if s["nonfinite"]:
        limitations.append(f"Nonfinite words were preserved unchanged in source rows {sorted(s['nonfinite'])} under --nonfinite preserve.")
    commitments = {"model_revision": PIN, "input_manifest_sha256": args.input_sha256,
                   "input": {"prompt_tokens": bound["prompt"], "image_grid_thw": [bound["grid"]], "next_decode_position": bound["next_position"]},
                   "source_export": {"metadata_sha256": args.metadata_sha256, "logits_sha256": meta["logits"]["sha256"],
                                     "logits_shape": checked["shape"], "route": args.route}}
    model = {"config_sha256": meta["context"]["config_sha256"], "config_sha256_origin": "producer_metadata"}
    if s["attested"]:
        model.update(model_sha256=MODEL_SHA256, model_bytes=MODEL_BYTES, model_sha256_origin="owner_attestation",
                     attestation_basis=args.attestation_basis)
    producer = {"kind": PRODUCER_KIND, "route": args.route, "runtime": runtime, "response_sha256": meta["response_sha256"],
                "importer_sha256": sha256(Path(__file__).read_bytes()),
                "comparator_sha256": sha256((HERE / "compare_diagnostics.py").read_bytes()),
                "packager_sha256": sha256((HERE / "prepare_input.py").read_bytes())}
    description = {"source": "imported_cpu_teacher_export", "route": args.route, "selected_source_rows": s["selected"],
                   "response_generated_indices": [spec["reference_provenance"]["response_generated_index"] for spec in s["specs"]],
                   "position_origin": "source_derived", "bf16_boundary": False, "nonfinite_policy": args.nonfinite,
                   "missing_producer_commitments": missing, "limitations": limitations}
    return commitments, model, producer, description, missing, limitations


def record_summary(spec, digests):
    return {"key": key_of(spec), "source_row": spec["reference_provenance"]["source_row"],
            "absolute_row": spec["coordinates"][0]["absolute_row"], "phase": spec["phase"], "decode_step": spec["decode_step"],
            "produces_generated_index": spec["produces_generated_index"],
            "response_generated_index": spec["reference_provenance"]["response_generated_index"],
            "token_id": spec["coordinates"][0]["token_id"], "position": spec["coordinates"][0]["position"],
            "payload_sha256": spec["source_sha256"], "nonfinite": nonfinite_words(spec["source_bytes"]),
            "prefix_rows": spec["prefix_rows"], "prefix_sha256": digests[spec["prefix_rows"]]}


# ---------------------------------------------------------------- output forecast

def dump_specs(s):
    """The record specifications write_dump receives, without their payload values."""
    return [{k: v for k, v in spec.items() if k not in ("source_bytes", "source_sha256", "prefix_rows")} for spec in s["specs"]]


def forecast_dump(s, commitments, model, producer, description):
    """Exact bytes and progress SHA-256 of the dump write_dump writes for these records, computed before anything is written.

    This reproduces write_dump line for line (its source SHA-256 is pinned as WRITER_SHA256) with the comparator's own
    record_key, observe and canonical_sha256; the written dump must then equal it exactly.
    """
    bound, digests = s["bound"], s["digests"]
    rows = bound["conditioning"]
    full = dict(commitments)
    full.update({k: model[k] for k in ("model_sha256", "config_sha256", "model_bytes") if k in model})
    records = []
    for sequence, (spec, values) in enumerate(zip(s["specs"], dump_specs(s))):
        record = {"event": "record", "schema": cd.RECORD_SCHEMA, "selected_axis": 0, "layer": None, "vision_block": None,
                  "decode_step": None, "cache_length": None, "decode": None, "produces_generated_index": None,
                  "cache": None, "bf16_rounding": cd.BF16_POLICY, "conditioning": None}
        record.update(values)
        record["conditioning"] = {"prefix_rows": spec["prefix_rows"], "prefix_sha256": digests[spec["prefix_rows"]]}
        record.update(sequence=sequence, file=f"{sequence:05d}.{record['stage']}.f32", key=cd.record_key(record),
                      storage_dtype="float32", byte_order="little", payload_bytes=ROW_BYTES, payload_sha256=spec["source_sha256"],
                      observed=cd.observe({"storage_dtype": "float32"}, spec["source_bytes"]), qualified=False)
        records.append(record)
    keys = sorted(record["key"] for record in records)
    plan = dict(description, schema=cd.REFERENCE_PLAN_SCHEMA, forecast={"records": len(keys), "keys": keys})
    full.setdefault("plan_sha256", cd.canonical_sha256(plan))
    lines = [{"event": "plan", "schema": cd.PROGRESS_SCHEMA, "plan": plan, "plan_sha256": full["plan_sha256"], "commitments": full,
              "producer": producer, "qualified": False, "conditioning": {"schema": cd.PREFIX_SCHEMA, "prompt_rows": bound["prompt"]}},
             {"event": "model_authenticated", "model": dict(model, revision=commitments["model_revision"])},
             {"event": "conditioning", "first_row": 0, "rows": rows, "consumed_rows": len(rows), "prefix_sha256": digests[len(rows)]}]
    for record in records:
        record.update(commitments=full, producer=producer)
        lines += [{"event": "record_started", "sequence": record["sequence"], "file": record["file"], "key": record["key"]}, record]
    lines.append({"event": "finished", "run_status": "completed", "dump_complete": True, "planned_records": len(keys),
                  "written_records": len(records), "missing_records": [], "incomplete_record": None,
                  "payload_bytes": len(records) * ROW_BYTES, "error": None, "consumed_rows": len(rows),
                  "consumed_prefix_sha256": digests[len(rows)], "qualified": False})
    size, hasher = 0, hashlib.sha256()
    for line in lines:  # One line at a time: metadata repeated in the plan and every record is counted, never accumulated.
        data = (json.dumps(line, allow_nan=False) + "\n").encode()
        size += len(data)
        hasher.update(data)
    return {"bytes": size + len(records) * ROW_BYTES, "progress_bytes": size, "progress_sha256": hasher.hexdigest(),
            "plan_sha256": full["plan_sha256"]}


def serialize(report):
    """The report's exact bytes: indented, ASCII-escaped JSON that refuses nonfinite numbers."""
    return (json.dumps(report, indent=1, allow_nan=False) + "\n").encode("ascii")


def bounded(text, limit=ERROR_LIMIT):
    """text, shortened so that its JSON string form takes at most limit bytes."""
    if len(json.dumps(text)) <= limit:
        return text
    text, low, high = text[:limit], 0, min(len(text), limit)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(text[:middle] + " [truncated]")) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low] + " [truncated]"


def report_template(args, s, argv, described, dump, dry_run):
    """Every report field that is settled before output creation; status, error, output and late resources follow."""
    commitments, model, producer, description, missing, limitations = described
    meta, checked, bound = s["meta"], s["checked"], s["bound"]
    teacher_rows = len(bound["conditioning"]) - bound["prompt"]
    return {
        "schema": REPORT_SCHEMA, "status": "dry_run" if dry_run else "failed", "dry_run": dry_run,
        "conversion_admissible": s["attested"], "error": None,
        "command": {"argv": argv, "importer_sha256": producer["importer_sha256"], "comparator_sha256": producer["comparator_sha256"],
                    "packager_sha256": producer["packager_sha256"]},
        "source_export": {"metadata_sha256": args.metadata_sha256, "metadata_bytes": s["metadata_bytes"], "logits": meta["logits"],
                          "logits_bytes": checked["shape"][0] * ROW_BYTES, "runtime": meta["runtime"], "route": args.route,
                          "recorded_cache_lengths": checked["lengths"], "entries": s["entries"],
                          "partial_logits_identical": True if s["partial"] else None,
                          "cache_equivalence": meta.get("cache_equivalence"), "curated_evidence": curated_record(args.metadata_sha256),
                          "producer_source_sha256_verified": s["producer_source"]},
        "input": {"manifest_sha256": args.input_sha256, "prompt_rows": bound["prompt"], "processor_prompt_rows": bound["processor_prompt"],
                  "positions": {"origin": "source_derived", "manifest_origin": s["manifest"]["positions"].get("origin"),
                                "source_sha256": s["manifest"]["positions"]["source_sha256"], "rope_delta": bound["rope_delta"],
                                "next_decode_position": bound["next_position"], "rederived_and_equal": True}},
        "conditioning": {"prompt_rows": bound["prompt"], "consumed_rows": len(bound["conditioning"]), "teacher_rows_after_prompt": teacher_rows,
                         "consumed_prefix_sha256": s["digests"][len(bound["conditioning"])], "source": "manifest prompt rows then producer-recorded teacher tokens"},
        "records": [record_summary(spec, s["digests"]) for spec in s["specs"]],
        "attestations": ([{"commitment": "model_safetensors_sha256", "value": MODEL_SHA256, "basis": args.attestation_basis,
                           "origin": "owner command-line attestation"}] if s["attested"] else []),
        "missing_producer_commitments": missing, "limitations": limitations,
        "output": {"reference": None, "comparator_load": None},
        "claims": {"cpu_reference_qualified": False, "native_numerically_qualified": False, "amd_ocr_qualified": False,
                   "throughput_measured": False},
        "forecast": {"counted": "regular-file bytes under --output", "max_output_bytes": args.max_output_bytes,
                     "reference_bytes": dump["bytes"], "report_bytes_bound": None, "output_bytes_bound": None,
                     "conversion_output_bytes_bound": None, "conversion_within_max_output_bytes": None},
        "resources": {"elapsed_seconds": None, "deadline_seconds": args.deadline_seconds, "input_bytes_read": s["budget"].used,
                      "max_input_bytes": args.max_input_bytes, "max_output_bytes": args.max_output_bytes, "peak_rss_kib": None,
                      "cpu_seconds": None, "output_bytes": None},
    }


def report_bound(report, load):
    """Bytes of the report with every field settled after output creation at its longest serialized form.

    JSON length is the sum of independent field lengths, so this bounds every version of the report any outcome writes: an
    imported, dry-run or failed status, a failure message within ERROR_LIMIT, the forecast comparator load, and any elapsed
    time, CPU time, peak RSS or output byte count.
    """
    resources = dict(report["resources"], elapsed_seconds=WIDEST_FLOAT, cpu_seconds=WIDEST_FLOAT, peak_rss_kib=WIDEST_INT,
                     output_bytes=WIDEST_INT)
    forecast = dict(report["forecast"], report_bytes_bound=WIDEST_INT, output_bytes_bound=WIDEST_INT,
                    conversion_output_bytes_bound=WIDEST_INT, conversion_within_max_output_bytes=False)
    output = report["output"] if report["dry_run"] else {"reference": "reference", "comparator_load": load}
    return len(serialize(dict(report, status="imported", conversion_admissible=False, error="x" * (ERROR_LIMIT - 2),
                              output=output, forecast=forecast, resources=resources)))


def plan_output(args, s, argv):
    """Exact dump forecast and report bound for every outcome, refused here, before output creation, beyond the ceiling."""
    try:
        writer = sha256(inspect.getsource(cd.write_dump).encode())
    except (OSError, TypeError):
        writer = None
    require(writer == WRITER_SHA256, "compare_diagnostics.write_dump is not the writer whose exact output this importer forecasts; "
            "forecast_dump and WRITER_SHA256 must follow it before any output can be bounded")
    described = describe(args, s)
    dump = forecast_dump(s, *described[:4])
    rows = len(s["bound"]["conditioning"])
    load = {"loader": "compare_diagnostics.load_dump", "authenticated": True, "dump_complete": True, "run_status": "completed",
            "records": len(s["specs"]), "progress_sha256": dump["progress_sha256"], "plan_sha256": dump["plan_sha256"],
            "conditioning_rows": rows, "consumed_prefix_sha256": s["digests"][rows], "unreferenced_files": []}
    ceiling = args.max_output_bytes
    conversion = report_template(args, s, [a for a in argv if a != "--dry-run"], described, dump, False)
    conversion_total = dump["bytes"] + report_bound(conversion, load)
    fits = conversion_total <= ceiling
    if args.dry_run:  # A dry run writes only its report; whether the conversion would fit is part of its admissibility.
        report = report_template(args, s, argv, described, dump, True)
        report["conversion_admissible"] = s["attested"] and fits
        total = report_bytes = report_bound(report, None)
        require(total <= ceiling, f"The dry-run report bound of {total} bytes exceeds --max-output-bytes {ceiling}")
    else:
        report, total, report_bytes = conversion, conversion_total, conversion_total - dump["bytes"]
        require(fits, f"Forecast output of {total} bytes ({dump['bytes']} exact reference bytes and a {report_bytes}-byte report bound) "
                f"exceeds --max-output-bytes {ceiling}")
    report["forecast"].update(report_bytes_bound=report_bytes, output_bytes_bound=total, conversion_output_bytes_bound=conversion_total,
                              conversion_within_max_output_bytes=fits)
    s["deadline"].check("output forecast")
    return {"report": report, "described": described, "dump": dump, "load": load}


# ---------------------------------------------------------------- output

def write_reference(s, staging, out):
    """Write with the comparator's own writer, authenticate it with the comparator's loader, compare every payload byte and
    require the dump to be exactly the forecast that bounded the output."""
    commitments, model, producer, description = out["described"][:4]
    specs = dump_specs(s)
    for spec, values in zip(s["specs"], specs):
        values["values"] = native_order(spec["source_bytes"])
    try:
        cd.write_dump(staging, commitments, model, producer, specs, plan=description,
                      conditioning=s["bound"]["conditioning"], prompt_rows=s["bound"]["prompt"])
    except ValueError as error:
        raise Failure(f"write_dump refused the reference: {error}") from None
    s["deadline"].check("reference writing")
    dump = cd.load_dump(staging)
    expected = {key_of(spec): spec for spec in s["specs"]}
    if not (dump["dump_complete"] and set(dump["records"]) == set(expected)):
        raise Failure("The written reference is incomplete or names different records")
    for key, record in dump["records"].items():
        s["deadline"].check("reference verification")
        spec = expected[key]
        handle, _ = open_regular(staging / record["file"], ROW_BYTES, "written payload")
        with handle:
            written = handle.read()
        if written != spec["source_bytes"] or record["payload_sha256"] != spec["source_sha256"]:
            raise Failure(f"{key}: written payload is not byte-identical to source row {spec['reference_provenance']['source_row']}")
        if record["conditioning"] != {"prefix_rows": spec["prefix_rows"], "prefix_sha256": s["digests"][spec["prefix_rows"]]}:
            raise Failure(f"{key}: consumed-prefix commitment differs from the independently computed canonical prefix")
    total = len(s["bound"]["conditioning"])
    if len(dump["history"].rows) != total or dump["history"].digests[-1] != s["digests"][total]:
        raise Failure("The written conditioning rows differ from the producer's consumed rows")
    load = {"loader": "compare_diagnostics.load_dump", "authenticated": True, "dump_complete": True, "run_status": dump["run_status"],
            "records": len(dump["records"]), "progress_sha256": dump["progress_sha256"], "plan_sha256": dump["header"]["plan_sha256"],
            "conditioning_rows": total, "consumed_prefix_sha256": dump["history"].digests[-1], "unreferenced_files": dump["orphans"]}
    size = written_bytes(staging)
    if load != out["load"] or size != out["dump"]["bytes"]:
        raise Failure(f"The written dump ({size} bytes, progress SHA-256 {load['progress_sha256']}) differs from the exact forecast "
                      f"({out['dump']['bytes']} bytes, {out['dump']['progress_sha256']}) that bounded --max-output-bytes")
    return load


def write_report(args, report, output, start):
    """Serialize the report with its own exact output_bytes, check the ceiling, then write it durably over any partial copy."""
    path = output / "import-report.json"
    resources = report["resources"]
    resources.update(elapsed_seconds=clock() - start, peak_rss_kib=peak_rss_kib(), cpu_seconds=time.process_time(), output_bytes=None)
    base = written_bytes(output, skip=path)
    data = serialize(report)
    for _ in range(8):  # output_bytes counts this report too; its digit count settles within a few passes.
        if resources["output_bytes"] == base + len(data):
            break
        resources["output_bytes"] = base + len(data)
        data = serialize(report)
    require(resources["output_bytes"] == base + len(data) <= args.max_output_bytes,
            f"Writing {base + len(data)} output bytes would exceed --max-output-bytes {args.max_output_bytes}")
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags | (os.O_TRUNC if os.path.lexists(path) else os.O_CREAT | os.O_EXCL), 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(output)
    return data.decode("ascii")


def finish(args, report, output, start, code, error=None):
    """Write the report of an outcome that promotes nothing; if even that fails, standard output carries it."""
    if error is not None:
        report["status"], report["error"] = "failed", bounded(str(error))
    try:
        text = write_report(args, report, output, start)
    except (Refusal, OSError) as failure:
        prior = f" after: {report['error']}" if report["error"] else ""
        report["status"], report["error"] = "failed", bounded(f"import-report.json could not be written ({failure}){prior}")
        text, code = serialize(report).decode("ascii"), 1
    print(text, end="")
    return code


def emit(args, s, out, start):
    """Write the outcome. The acceptance report is durable before reference/ exists, so a reference never appears without it."""
    report, output = out["report"], s["output"]
    staging = output / "unverified-reference"
    try:
        if not args.dry_run:
            report["output"]["comparator_load"] = write_reference(s, staging, out)
        s["deadline"].check("completion")
    except (Refusal, Failure, OSError) as error:
        return finish(args, report, output, start, 1, error)
    if args.dry_run:
        return finish(args, report, output, start, 0 if report["conversion_admissible"] else 3)
    report["status"], report["output"]["reference"] = "imported", "reference"
    try:
        text = write_report(args, report, output, start)
        s["deadline"].check("promotion")  # Only an in-deadline, verified dump whose report is durable is renamed to reference/.
        promote(staging, output / "reference")
    except (Refusal, OSError) as error:
        report["output"]["reference"] = None
        return finish(args, report, output, start, 1, error)
    sync_directory(output)
    print(text, end="")
    return 0


class Arguments(argparse.ArgumentParser):
    """Usage errors are refusals like any other: JSON on standard output, exit code 2 and no output."""

    def error(self, message):
        raise Refusal(f"{self.prog}: {message}")


def refuse(message):
    print(json.dumps({"schema": REPORT_SCHEMA, "status": "refused", "error": bounded(message), "native_numerically_qualified": False,
                      "amd_ocr_qualified": False}))
    return 2


def main(argv=None):
    start = clock()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = Arguments(description=__doc__.splitlines()[0], allow_abbrev=False)  # Exact option names keep argv forecasts exact.
    parser.add_argument("--export-dir", required=True, help="absolute retained export directory (metadata.json, logits.f32)")
    parser.add_argument("--metadata-sha256", required=True, help="expected SHA-256 of metadata.json")
    parser.add_argument("--input-manifest", required=True, help="absolute chandra.directcompute.input.v1 manifest")
    parser.add_argument("--input-sha256", required=True, help="expected SHA-256 of the input manifest")
    parser.add_argument("--route", required=True, choices=(CACHED, MONOLITHIC), help="owner-attested producer route, checked against metadata")
    parser.add_argument("--rows", required=True, help="sorted comma-separated source logits row indices")
    parser.add_argument("--attest-model-sha256", help="owner attestation that the producer loaded the pinned safetensors")
    parser.add_argument("--attestation-basis", help="one-line evidence for the weights attestation, recorded verbatim")
    parser.add_argument("--producer-source", help="absolute copy of the exact producer script, checked against script_sha256")
    parser.add_argument("--nonfinite", choices=("refuse", "preserve"), default="refuse")
    parser.add_argument("--dry-run", action="store_true", help="authenticate and report without writing a reference dump")
    parser.add_argument("--max-input-bytes", type=int, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True, help="ceiling on every regular-file byte written under --output")
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--output", required=True, help="absolute fresh output directory; never overwritten")
    try:
        args = parser.parse_args(argv)
        state = prepare(args, start)
        out = plan_output(args, state, argv)
        os.mkdir(state["output"], 0o700)
    except (Refusal, OSError) as error:
        return refuse(str(error) if isinstance(error, Refusal) else f"{type(error).__name__}: {error}")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError, OverflowError) as error:
        # A backstop: malformed input that no explicit check names is still refused before any output exists.
        return refuse(f"Input validation raised {type(error).__name__} ({error}); nothing was written")
    return emit(args, state, out, start)


if __name__ == "__main__":
    sys.exit(main())
