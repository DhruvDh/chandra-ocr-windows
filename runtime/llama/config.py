"""Explicit configuration for the unpromoted, pinned llama.cpp adapters."""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import stat

ACTIVATION = "evaluate-llama-b11429-vulkan"
SYCL_ACTIVATION = "evaluate-llama-b11429-sycl"
ENGINES = {ACTIVATION: "vulkan", SYCL_ACTIVATION: "sycl"}
CONTEXT = 16384
OUTPUT = 12384
GIB = 1 << 30
SYCL_DECODE_GRAPH_LEAF_SHA256 = "4e4445190ab5b7686a0990c11aa8faf8f0c27bbf2b9f445430193b0aaa446234"
SYCL_DECODE_GRAPH_MODEL_SHA256 = "74699f738159fc227e8153e5345cc36795aa81fe726ff4746fee8f7b7efe7f6d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_kv_profile(engine, slots, kv_type, physical, commit, *, batch_size=128,
                        host_cpu_count=1, worker_cpu=None, vision_leaf=None, cache_dir=None,
                        decode_graph=False, model_sha256=None, runtime_files=()):
    require(engine in ("vulkan", "sycl"), "Invalid llama engine")
    require(isinstance(kv_type, str) and kv_type in ("bf16", "q8_0"), "Invalid sycl_kv_cache_type")
    require(engine == "sycl" or kv_type == "bf16", "Q8 KV requires SYCL")
    require(type(slots) is int and slots in (1, 2, 4, 5, 6), "Invalid parallel_requests")
    require(type(batch_size) is int and batch_size in (128, 256), "Invalid sycl_batch_size")
    require(batch_size == 128 or (engine == "sycl" and slots == 5), "Batch 256 requires the explicit SYCL Q8 five-slot profile")
    if slots in (5, 6):
        require(engine == "sycl" and kv_type == "q8_0", "Five or six slots require explicit SYCL Q8 KV")
        require(type(physical) is int and 21 * GIB // 4 <= physical <= 32 * GIB
                and type(commit) is int and 49 * GIB // 4 <= commit <= 32 * GIB,
                "Five or six slots require explicit admission forecasts >= 5.25 GiB physical and 12.25 GiB commit")
    if slots == 5:
        require(batch_size == 256, "Five slots require explicit sycl_batch_size 256")
        require(type(host_cpu_count) is int and host_cpu_count == 4
                and type(worker_cpu) is int and worker_cpu == 0,
                "Five slots require the explicit four-CPU SYCL profile")
        require(isinstance(vision_leaf, str) and re.fullmatch(r"[0-9a-f]{64}", vision_leaf)
                and cache_dir is not None, "Five slots require explicit fused vision and persistent cache")
    require(type(decode_graph) is bool, "Invalid sycl_decode_graph boolean")
    if decode_graph:
        require(engine == "sycl" and slots == 5 and batch_size == 256 and kv_type == "q8_0",
                "Decode graph requires the explicit SYCL Q8 five-slot batch-256 profile")
        require(isinstance(cache_dir, Path), "Decode graph requires an explicit persistent cache path")
        require(vision_leaf == SYCL_DECODE_GRAPH_LEAF_SHA256 and model_sha256 == SYCL_DECODE_GRAPH_MODEL_SHA256,
                "Decode graph requires the pinned graph leaf and evaluated Q8 text model")
        require(isinstance(runtime_files, (tuple, list))
                and any(isinstance(item, Identity) and item.path.name.casefold() == "ggml-sycl.dll"
                        and item.sha256 == SYCL_DECODE_GRAPH_LEAF_SHA256 for item in runtime_files),
                "Decode graph leaf must match the authenticated runtime inventory")


def plain_path(value):
    require(isinstance(value, str) and len(value) <= 1024 and not any(c in value for c in '\0\r\n"'), "Invalid path")
    path = Path(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts, "Paths must be absolute and normalized")
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400, "Reparse and symlink paths are refused")
    return path


@dataclass(frozen=True)
class Identity:
    path: Path
    size: int
    sha256: str

    def authenticate(self):
        require(plain_path(str(self.path)).is_file(), "An authenticated input is not a regular file")
        require(self.path.stat().st_size == self.size, f"Size differs for {self.path.name}")
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            while block := source.read(1 << 20):
                digest.update(block)
        require(digest.hexdigest() == self.sha256, f"SHA-256 differs for {self.path.name}")


def identity(raw, maximum):
    require(isinstance(raw, dict) and set(raw) == {"path", "size", "sha256"}, "Invalid file identity")
    require(type(raw["size"]) is int and 0 < raw["size"] <= maximum, "Invalid file size")
    require(isinstance(raw["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", raw["sha256"]), "Invalid SHA-256")
    path = plain_path(raw["path"])
    require(path.is_file(), "An input is not a regular file")
    return Identity(path, raw["size"], raw["sha256"])


@dataclass(frozen=True)
class LlamaConfig:
    config_sha256: str
    runtime_dir: Path
    runtime_files: tuple[Identity, ...]
    model: Identity
    projector: Identity
    template: Identity
    device_index: int
    worker_cpu: int
    startup_seconds: int
    request_seconds: int
    max_image_pixels: int
    parallel_requests: int
    admission_physical_bytes: int = 6 * GIB
    admission_commit_bytes: int = 11 * GIB
    engine: str = "vulkan"
    sycl_cache_dir: Path | None = None
    sycl_vision_leaf_sha256: str | None = None
    sycl_host_cpu_count: int = 1
    sycl_kv_cache_type: str = "bf16"
    sycl_batch_size: int = 128
    sycl_decode_graph: bool = False

    def __post_init__(self):
        require(type(self.sycl_decode_graph) is bool, "Invalid sycl_decode_graph boolean")
        if self.sycl_decode_graph:
            self._validate_profile()

    def _validate_profile(self):
        validate_kv_profile(self.engine, self.parallel_requests, self.sycl_kv_cache_type,
                            self.admission_physical_bytes, self.admission_commit_bytes,
                            batch_size=self.sycl_batch_size, host_cpu_count=self.sycl_host_cpu_count,
                            worker_cpu=self.worker_cpu, vision_leaf=self.sycl_vision_leaf_sha256,
                            cache_dir=self.sycl_cache_dir, decode_graph=self.sycl_decode_graph,
                            model_sha256=getattr(self.model, "sha256", None), runtime_files=self.runtime_files)

    def files(self):
        return [item.path for item in (*self.runtime_files, self.model, self.projector, self.template)]

    def authenticate(self):
        if self.sycl_decode_graph:
            self._validate_profile()
        if self.sycl_cache_dir is not None:
            require(plain_path(str(self.sycl_cache_dir)).is_dir(), "SYCL cache directory is required")
        expected = {item.path.name.casefold() for item in self.runtime_files}
        actual = {path.name.casefold() for path in self.runtime_dir.iterdir() if path.suffix.lower() in (".exe", ".dll")}
        require(actual == expected, "Runtime executable/DLL inventory differs")
        for item in (*self.runtime_files, self.model, self.projector, self.template):
            item.authenticate()

    def job_limits(self):
        from runtime.native.winjob import JobLimits
        require(self.sycl_host_cpu_count == 1 or (self.engine == "sycl" and self.sycl_host_cpu_count == 4 and self.worker_cpu == 0),
                "Four-CPU affinity requires the explicit SYCL profile")
        affinity = 15 if self.sycl_host_cpu_count == 4 else 1 << self.worker_cpu
        return JobLimits(21 * GIB // 2, 43 * GIB // 4, affinity)

    def environment(self):
        require(self.engine in ("vulkan", "sycl"), "Invalid llama engine")
        if self.sycl_decode_graph:
            self._validate_profile()
        # Do not inherit undocumented arithmetic, backend selection or executable search overrides.
        allowed = {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "HOMEDRIVE", "HOMEPATH"}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        system = next((value for key, value in env.items() if key.upper() == "SYSTEMROOT"), None)
        require(system is not None, "Windows SYSTEMROOT is required")
        env.update(PATH=str(self.runtime_dir) + os.pathsep + str(Path(system) / "System32"), OMP_NUM_THREADS="1")
        if self.engine == "sycl":
            vision = "1" if self.sycl_vision_leaf_sha256 is not None else "0"
            env.update(ONEAPI_DEVICE_SELECTOR=f"level_zero:{self.device_index}", MKL_NUM_THREADS="1",
                       GGML_SYCL_FA_ONEDNN_UNMASKED=vision, GGML_SYCL_FA_ONEDNN_DG2_D64=vision,
                       ZES_ENABLE_SYSMAN="0", UR_L0_ENABLE_SYSMAN_ENV_DEFAULT="0", UR_L0_ENABLE_ZESINIT_DEFAULT="0")
            if self.sycl_decode_graph:
                env.update(GGML_SYCL_ENABLE_GRAPH="0", GGML_SYCL_DECODE_GRAPH="3")
            if self.sycl_cache_dir is not None:
                require(plain_path(str(self.sycl_cache_dir)).is_dir(), "SYCL cache directory is required")
                env.update(SYCL_CACHE_PERSISTENT="1", SYCL_CACHE_TRACE="1", SYCL_CACHE_MAX_SIZE=str(GIB),
                           SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE="0", SYCL_CACHE_DIR=str(self.sycl_cache_dir))
        else:
            env["GGML_VK_VISIBLE_DEVICES"] = str(self.device_index)
        return env

    def argv(self, port, api_key, alias):
        self._validate_profile()
        device = "SYCL0" if self.engine == "sycl" else "Vulkan0"
        return [str(self.runtime_dir / "llama-server.exe"),
                # Device parsing initializes SYCL: its one-time graph settings need INFO logging already enabled.
                *(["--log-verbosity", "4"] if self.sycl_decode_graph else []),
                "--model", str(self.model.path), "--mmproj", str(self.projector.path),
                "--device", device, "--mmproj-device", device,
                *(["--tensor-split", "1"] if self.engine == "sycl" else []), "--n-gpu-layers", "99",
                "--fit", "off", "--load-mode", "none", "--ctx-size", str(CONTEXT * self.parallel_requests), "--n-predict", str(OUTPUT),
                *(["--no-kv-unified"] if self.parallel_requests > 1 else []),
                "--image-max-tokens", "3072", "--batch-size", str(self.sycl_batch_size), "--ubatch-size", str(self.sycl_batch_size), "--parallel", str(self.parallel_requests),
                "--threads", "1", "--threads-batch", "1", "--flash-attn", "on",
                "--cache-type-k", self.sycl_kv_cache_type, "--cache-type-v", self.sycl_kv_cache_type,
                "--cache-ram", "0", "--ctx-checkpoints", "0", "--cache-reuse", "0", "--no-cache-prompt",
                "--no-cache-idle-slots", "--no-context-shift", "--no-warmup", "--jinja",
                "--chat-template-file", str(self.template.path), "--alias", alias, "--api-key", api_key,
                "--host", "127.0.0.1", "--port", str(port), "--slots"]


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate configuration field")
        result[key] = value
    return result


def load_config(path, expected_sha256):
    """Authenticate only the small configuration here; models are read on first demand."""
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256), "Configuration SHA-256 is required")
    with Path(path).open("rb") as source:
        data = source.read(65537)
    require(len(data) <= 65536 and hashlib.sha256(data).hexdigest() == expected_sha256, "Configuration size or SHA-256 differs")
    raw = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    require(isinstance(raw, dict) and isinstance(raw.get("activate"), str)
            and raw["activate"] in ENGINES, "Explicit llama candidate configuration is required")
    engine = ENGINES[raw["activate"]]
    device_field = "sycl_device_index" if engine == "sycl" else "device_index"
    fields = {"activate", "runtime_dir", "runtime_files", "model", "projector", "template", device_field, "worker_cpu", "startup_seconds", "request_seconds", "max_image_pixels", "parallel_requests"}
    optional = {"admission_physical_bytes", "admission_commit_bytes"}
    if engine == "sycl":
        optional |= {"sycl_cache_dir", "sycl_vision_leaf_sha256", "sycl_host_cpu_count", "sycl_kv_cache_type", "sycl_batch_size", "sycl_decode_graph"}
    require(fields <= set(raw) <= fields | optional, "Explicit engine-specific llama configuration is required")
    runtime = plain_path(raw["runtime_dir"])
    require(runtime.is_dir(), "Runtime directory is required")
    require(isinstance(raw["runtime_files"], list) and 1 <= len(raw["runtime_files"]) <= 250, "Runtime file inventory is required")
    runtime_files = tuple(identity(value, 2 * GIB) for value in raw["runtime_files"])
    require(all(item.path.parent == runtime and item.path.suffix.lower() in (".exe", ".dll") for item in runtime_files), "Runtime identities must name adjacent executable/DLL files")
    names = {item.path.name.casefold() for item in runtime_files}
    require(len(names) == len(runtime_files) and "llama-server.exe" in names, "Runtime inventory must uniquely include llama-server.exe")
    if engine == "sycl":
        require("ggml-sycl.dll" in names, "SYCL runtime inventory must include ggml-sycl.dll")
    cache = plain_path(raw["sycl_cache_dir"]) if "sycl_cache_dir" in raw else None
    if cache is not None:
        require(cache.is_dir(), "SYCL cache directory is required")
    vision = raw.get("sycl_vision_leaf_sha256")
    if "sycl_vision_leaf_sha256" in raw:
        require(isinstance(vision, str) and re.fullmatch(r"[0-9a-f]{64}", vision), "Invalid SYCL vision leaf SHA-256")
        require(any(item.path.name.casefold() == "ggml-sycl.dll" and item.sha256 == vision for item in runtime_files),
                "SYCL vision leaf SHA-256 must match the authenticated runtime inventory")
    host_cpus = raw.get("sycl_host_cpu_count", 1)
    if "sycl_host_cpu_count" in raw:
        require(type(host_cpus) is int and host_cpus == 4 and type(raw["worker_cpu"]) is int and raw["worker_cpu"] == 0,
                "SYCL host CPU profile requires count 4 and worker_cpu 0")
    model, projector, template = (identity(raw[name], maximum) for name, maximum in (("model", 12 * GIB), ("projector", 2 * GIB), ("template", 1 << 20)))
    require(len(set(item.path for item in (*runtime_files, model, projector, template))) == len(runtime_files) + 3, "Input identities must be distinct")
    for name, low, high in ((device_field, 0, 15), ("worker_cpu", 0, 63), ("startup_seconds", 1, 1800), ("request_seconds", 1, 1800), ("max_image_pixels", 1, 6_291_456)):
        require(type(raw[name]) is int and low <= raw[name] <= high, f"Invalid {name}")
    kv_type = raw.get("sycl_kv_cache_type", "bf16")
    batch_size = raw.get("sycl_batch_size", 128)
    decode_graph = raw.get("sycl_decode_graph", False)
    # Pass raw forecasts so C5/C6 cannot inherit a missing field's default.
    validate_kv_profile(engine, raw["parallel_requests"], kv_type,
                        raw.get("admission_physical_bytes"), raw.get("admission_commit_bytes"),
                        batch_size=batch_size, host_cpu_count=host_cpus, worker_cpu=raw["worker_cpu"],
                        vision_leaf=vision, cache_dir=cache, decode_graph=decode_graph,
                        model_sha256=model.sha256, runtime_files=runtime_files)
    forecasts = [raw.get("admission_physical_bytes", 6 * GIB), raw.get("admission_commit_bytes", 11 * GIB)]
    require(all(type(value) is int and GIB <= value <= 32 * GIB for value in forecasts), "Invalid admission forecasts")
    return LlamaConfig(expected_sha256, runtime, runtime_files, model, projector, template,
                       *(raw[name] for name in (device_field, "worker_cpu", "startup_seconds", "request_seconds", "max_image_pixels", "parallel_requests")), *forecasts,
                       engine=engine, sycl_cache_dir=cache, sycl_vision_leaf_sha256=vision, sycl_host_cpu_count=host_cpus,
                       sycl_kv_cache_type=kv_type, sycl_batch_size=batch_size, sycl_decode_graph=decode_graph)
