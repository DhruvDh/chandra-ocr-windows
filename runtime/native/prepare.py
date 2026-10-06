"""Production request preparation for the native worker: one RGB image and prompt to the input.v1 manifest ABI.

The pinned CPU processor/tokenizer (local files only, never weights or a GPU) produces the original
tensors exactly as benchmarks/export_processor.py does under the established serving profile. Position
rules come from scripts/native/prepare_input.py; nothing here re-derives them. Packages are written to a
fresh private staging directory, re-read and authenticated, then promoted exclusively.
"""
import base64
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
import struct
import sys
import threading
import warnings

from scripts.native.prepare_input import (CALLER_SOURCE_SHA, PIN, PROCESSOR_PINS, PROFILE, ROPE_SOURCE_SHA,
                                          positions, tensor_record)

from .protocol import CONTEXT_LIMIT, IMAGE_TOKEN, MODEL, OUTPUT_ALLOWANCE, STOP_TOKEN_IDS

PROFILE_NAME = "northstone-serving"
PACKAGE_CAP = 128 * 1024 * 1024
MANIFEST_CAP = 1024 * 1024
MEDIA = {"data:image/png;base64": ("PNG", "input.png"), "data:image/jpeg;base64": ("JPEG", "input.jpg"),
         "data:image/webp;base64": ("WEBP", "input.webp")}
PROCESSOR_TENSORS = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
TRANSFORMERS = "5.18.0"
ROPE_SOURCE = "transformers/models/qwen3_5/modeling_qwen3_5.py"


class RequestRefused(ValueError):
    """A supported-request violation found before any GPU submission; the service reports 400."""


def refuse(condition, message):
    if not condition:
        raise RequestRefused(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RawTensor:
    dtype: str  # "I64" or "F32", little-endian, contiguous
    shape: tuple
    data: bytes

    def values(self):
        if self.dtype != "I64" or len(self.data) != 8 * math.prod(self.shape):
            raise RuntimeError("Processor integer tensor ABI differs")
        return [v[0] for v in struct.iter_unpack("<q", self.data)]


@dataclass(frozen=True)
class Processed:
    rendered: str
    tensors: dict


@dataclass(frozen=True)
class Pinned:
    """Authenticated stop provenance copied into every package for the worker's own re-authentication."""
    caller: bytes
    generation_config: bytes
    tokenizer_config: bytes
    stop_ids: tuple = field(default=())

    @classmethod
    def from_bytes(cls, caller, generation_config, tokenizer_config):
        generation, tokenizer = json.loads(generation_config), json.loads(tokenizer_config)
        eos = generation["eos_token_id"]
        stops = [eos] if type(eos) is int else list(eos)
        ends = [int(k) for k, v in tokenizer["added_tokens_decoder"].items() if v["content"] == "<|im_end|>"]
        if len(ends) != 1:
            raise RuntimeError("Missing or duplicate turn-end token")
        if ends[0] not in stops:
            stops.append(ends[0])  # Exact specialization of the installed Chandra generate_hf caller.
        if sorted(stops) != list(STOP_TOKEN_IDS):
            raise RuntimeError("Pinned caller stops differ from 248044/248046")
        return cls(caller, generation_config, tokenizer_config, tuple(stops))

    @classmethod
    def load(cls, caller_source, model_dir):
        caller = Path(caller_source).read_bytes()
        generation = (Path(model_dir) / "generation_config.json").read_bytes()
        tokenizer = (Path(model_dir) / "tokenizer_config.json").read_bytes()
        if (digest(caller), digest(generation), digest(tokenizer)) != (
                CALLER_SOURCE_SHA, PROCESSOR_PINS["generation_config.json"], PROCESSOR_PINS["tokenizer_config.json"]):
            raise RuntimeError("Pinned caller/generation/tokenizer provenance differs")
        return cls.from_bytes(caller, generation, tokenizer)


@dataclass(frozen=True)
class Prepared:
    package: str
    manifest: str
    manifest_sha256: str
    prompt_ids: tuple
    patch_rows: int
    grid: tuple
    stop_ids: tuple
    package_bytes: int


def read_request(payload, max_pixels):
    """Exact Chandra client request shape: one image then one text part, normal allowance, still image."""
    from PIL import Image
    refuse(payload.get("max_tokens", OUTPUT_ALLOWANCE) == OUTPUT_ALLOWANCE,
           "The native endpoint serves only the normal 12384-token output allowance; the worker's shorter cap is diagnostic-only")
    content = payload["messages"][0]["content"]
    refuse([part.get("type") for part in content] == ["image_url", "text"],
           "Supply the image_url part before the text part, as the Chandra client does; parts are never reordered")
    text = content[1]["text"]
    header, encoded = content[0]["image_url"]["url"].split(",", 1)
    refuse(header in MEDIA, "Only PNG, JPEG or WebP data URLs are supported")
    fmt, filename = MEDIA[header]
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise RequestRefused("Invalid image base64") from None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                refuse(image.format == fmt, "Image bytes do not match the declared data URL media type")
                refuse(getattr(image, "n_frames", 1) == 1, "Only a single still image is supported")
                refuse(image.width * image.height <= max_pixels, "Image exceeds the configured decoded pixel limit")
                image.load()
                rgb = image.convert("RGB")
    except RequestRefused:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise RequestRefused("Invalid or oversized image payload") from None
    return text, data, filename, rgb


def check_processed(processed, max_patch_rows):
    """Geometry/context admission; every refusal precedes packaging and submission."""
    tensors = processed.tensors
    if set(tensors) != PROCESSOR_TENSORS or not isinstance(processed.rendered, str):
        raise RuntimeError("Pinned processor returned an unexpected tensor inventory")
    ids = tensors["input_ids"].values()
    n = len(ids)
    mask, types, grid = tensors["attention_mask"].values(), tensors["mm_token_type_ids"].values(), tensors["image_grid_thw"].values()
    pixels = tensors["pixel_values"]
    if any(tensors[k].shape != (1, n) for k in ("input_ids", "attention_mask", "mm_token_type_ids")) or tensors["image_grid_thw"].shape != (1, 3) \
            or pixels.dtype != "F32" or len(pixels.shape) != 2 or pixels.shape[1] != 1536 or len(pixels.data) != 4 * math.prod(pixels.shape):
        raise RuntimeError("Pinned processor tensor layout differs from the native ABI")
    refuse(pixels.shape[0] == math.prod(grid) and len(grid) == 3 and grid[0] == 1, "Only one still image is supported")
    refuse(pixels.shape[0] <= max_patch_rows, f"Processed image has {pixels.shape[0]} patch rows; this endpoint currently admits at most {max_patch_rows}")
    refuse(n + OUTPUT_ALLOWANCE <= CONTEXT_LIMIT,
           f"Processed prompt has {n} tokens; it must leave the full 12384-token allowance inside the 16384-token context")
    try:
        axes, delta = positions(ids, types, mask, grid)
    except ValueError as error:
        raise RequestRefused("Processed geometry is outside the native single-image B1 contract: " + str(error)) from None
    refuse(max(max(axis) for axis in axes) + OUTPUT_ALLOWANCE < CONTEXT_LIMIT, "Prompt positions must preserve the full decode allowance")
    return ids, axes, delta, tuple(grid)


def _plain_dir(path):
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise RuntimeError("Expected a plain private directory")


def package(input_root, pinned, processed, ids, axes, delta, grid, text, image_bytes, image_name, rgb, runtime):
    """Write one complete package into a fresh staging directory, authenticate it and promote it exclusively."""
    n, tensors = len(ids), processed.tensors
    pixels = rgb.tobytes()
    files = {}
    records = {}
    for key in sorted(PROCESSOR_TENSORS):
        raw = tensors[key]
        name = key + ".raw"
        files[name] = raw.data
        records[key] = dict(tensor_record(name, raw.data, list(raw.shape)), dtype=raw.dtype,
                            source_dtype={"I64": "torch.int64", "F32": "torch.float32"}[raw.dtype])
    derived = {"position_ids": (struct.pack("<" + "q" * (3 * n), *sum(axes, [])), [3, 1, n]),
               "text_position_ids": (struct.pack("<" + "q" * n, *range(n)), [1, n]),
               "rope_deltas": (struct.pack("<q", delta), [1, 1])}
    for key, (data, shape) in derived.items():
        name = "derived-" + key + ".raw"
        files[name] = data
        records[key] = tensor_record(name, data, shape)
    prompt = text.encode("utf-8")
    rendered = processed.rendered.encode("utf-8")
    files.update({image_name: image_bytes, "prompt.txt": prompt, "rendered_prompt.txt": rendered, "chandra-caller-hf.py": pinned.caller,
                  "generation_config.json": pinned.generation_config, "tokenizer_config.json": pinned.tokenizer_config})
    stops = list(pinned.stop_ids)
    manifest = dict(
        schema="chandra.directcompute.input.v1", model=MODEL, revision=PIN, source="chandra.native-endpoint.request.v1",
        processor_files=PROCESSOR_PINS, processor_profile=PROFILE_NAME, processor_kwargs=PROFILE[PROFILE_NAME], runtime=runtime,
        prompt_tokens=n, image_token_id=IMAGE_TOKEN, image_token_positions=[i for i, t in enumerate(ids) if t == IMAGE_TOKEN],
        image_grid_thw=[list(grid)],
        image=dict(file=image_name, sha256=digest(image_bytes), pixels_sha256=digest(pixels), dimensions=list(rgb.size), mode="RGB"),
        prompt=dict(file="prompt.txt", sha256=digest(prompt)), rendered_prompt=dict(file="rendered_prompt.txt", sha256=digest(rendered)),
        tensors=records,
        generation=dict(context_limit=CONTEXT_LIMIT, max_output_tokens=OUTPUT_ALLOWANCE, stop_token_ids=stops,
                        stop_token_provenance=dict(caller_source_file="chandra-caller-hf.py", caller_source_sha256=digest(pinned.caller),
                                                   caller_stop_token_ids=stops, caller="Installed Chandra 0.2.0 generate_hf; source extraction only",
                                                   generation_config_sha256=digest(pinned.generation_config),
                                                   tokenizer_config_sha256=digest(pinned.tokenizer_config), amd_runtime_policy_verified=False)),
        positions=dict(origin="source-derived single-image B1 unpadded specialization", source_sha256=ROPE_SOURCE_SHA,
                       spatial_merge_size=2, next_decode_position=n + delta, captured_from_amd=False),
        decoded_pixels_verification="Endpoint decoded the request once to RGB; pixels_sha256 commits the exact pixels given to the pinned CPU processor",
        provenance_verified=False, native_qualification=False,
        scope="Endpoint request packaging only; no numerical, OCR or speed qualification")
    data = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MANIFEST_CAP:
        raise RuntimeError("Manifest exceeds the native 1 MiB bound")
    files["input-manifest.json"] = data
    total = sum(len(v) for v in files.values())
    refuse(total <= PACKAGE_CAP, "Prepared native package exceeds 128 MiB")
    _plain_dir(input_root)
    name = "p-" + secrets.token_hex(16)
    staging, final = Path(input_root) / ("s-" + secrets.token_hex(16)), Path(input_root) / name
    os.mkdir(staging, 0o700)  # Exclusive; never an existing directory.
    try:
        for filename, content in files.items():
            with open(staging / filename, "xb") as output:
                output.write(content)
        # Authenticate exactly what the worker will read before it can be named.
        if sorted(os.listdir(staging)) != sorted(files):
            raise RuntimeError("Staging directory contents differ from the package")
        for filename, content in files.items():
            with open(staging / filename, "rb") as source:
                hasher, size = hashlib.sha256(), 0
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    hasher.update(chunk)
            if size != len(content) or hasher.hexdigest() != digest(content):
                raise RuntimeError("Package file failed authentication after write")
        if os.path.lexists(final):
            raise RuntimeError("Package name collision")
        os.rename(staging, final)  # Windows refuses an existing target; the random name makes POSIX collisions refused above.
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Prepared(name, name + "/input-manifest.json", digest(data), tuple(ids), tensors["pixel_values"].shape[0], grid, tuple(stops), total)


def prepare(payload, processor, pinned, input_root, max_pixels, max_patch_rows):
    text, data, image_name, rgb = read_request(payload, max_pixels)
    pixels = digest(rgb.tobytes())
    processed = processor.process(rgb, text)
    if digest(rgb.tobytes()) != pixels:
        raise RuntimeError("Processor modified the decoded RGB image")
    ids, axes, delta, grid = check_processed(processed, max_patch_rows)
    return package(input_root, pinned, processed, ids, axes, delta, grid, text, data, image_name, rgb, processor.runtime)


def discard(input_root, prepared):
    """Remove one task-owned package after the worker can no longer read it."""
    path = Path(input_root) / prepared.package
    if prepared.package.startswith("p-") and os.path.lexists(path):
        _plain_dir(path)
        shutil.rmtree(path)


def accelerators_initialized(torch):
    states = {}
    for name in ("cuda", "xpu"):
        check = getattr(getattr(torch, name, None), "is_initialized", None)
        if check is not None:
            states[name] = bool(check())
    return states


class PinnedProcessor:
    """CPU-only pinned processor and tokenizer from local files; it never loads weights or selects a device."""

    def __init__(self, model_dir, dll_bootstrap=False):
        self.model_dir = Path(model_dir)
        self.dll_bootstrap = dll_bootstrap
        self._processor = None
        self._torch = None
        self.runtime = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._processor is not None:
                return self._processor
            for name, expected in PROCESSOR_PINS.items():
                path = self.model_dir / name
                if os.path.islink(path) or digest(path.read_bytes()) != expected:
                    raise RuntimeError("Pinned processor/tokenizer file differs: " + name)
            if sys.byteorder != "little":
                raise RuntimeError("The native raw tensor ABI requires a little-endian host")
            if importlib.metadata.version("transformers") != TRANSFORMERS:
                raise RuntimeError("Position derivation is pinned to Transformers 5.18.0")
            source = importlib.metadata.distribution("transformers").locate_file(ROPE_SOURCE)
            if digest(Path(source).read_bytes()) != ROPE_SOURCE_SHA:  # Bytes only; the modeling module is never imported.
                raise RuntimeError("Installed Qwen3.5 position source differs from the pinned source")
            if self.dll_bootstrap and sys.platform == "win32":
                from runtime.waystone.bootstrap import prepare_runtime
                prepare_runtime()  # Loads the locked environment's DLLs by absolute path; no device is queried.
            import torch
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
            self._require_cpu(torch)
            self.runtime = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "pillow", "tokenizers")}
            self._processor, self._torch = processor, torch
            return processor

    @staticmethod
    def _require_cpu(torch):
        initialized = [name for name, value in accelerators_initialized(torch).items() if value]
        if initialized:
            raise RuntimeError("CPU request preparation initialized an accelerator: " + ", ".join(initialized))

    def process(self, image, text):
        processor = self._load()
        torch = self._torch
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[rendered], images=[image], return_tensors="pt", **PROFILE[PROFILE_NAME])
        tensors = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in (torch.int64, torch.float32):
                raise RuntimeError("Expected a CPU int64/float32 tensor: " + key)
            if value.dtype == torch.float32 and not bool(torch.isfinite(value).all()):
                raise RequestRefused("Processed pixels are not finite")
            raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
            tensors[key] = RawTensor("I64" if value.dtype == torch.int64 else "F32", tuple(value.shape), raw)
        self._require_cpu(torch)
        return Processed(rendered, tensors)

    def decode(self, ids):
        return self._load().tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def unload(self):
        with self._lock:
            self._processor = None
