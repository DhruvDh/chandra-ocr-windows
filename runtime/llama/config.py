"""Explicit configuration for the unpromoted, pinned llama.cpp Vulkan adapter."""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import stat

ACTIVATION = "evaluate-llama-b11429-vulkan"
CONTEXT = 16384
OUTPUT = 12384
GIB = 1 << 30


def require(condition, message):
    if not condition:
        raise ValueError(message)


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

    def files(self):
        return [item.path for item in (*self.runtime_files, self.model, self.projector, self.template)]

    def authenticate(self):
        expected = {item.path.name.casefold() for item in self.runtime_files}
        actual = {path.name.casefold() for path in self.runtime_dir.iterdir() if path.suffix.lower() in (".exe", ".dll")}
        require(actual == expected, "Runtime executable/DLL inventory differs")
        for item in (*self.runtime_files, self.model, self.projector, self.template):
            item.authenticate()

    def job_limits(self):
        from runtime.native.winjob import JobLimits
        return JobLimits(21 * GIB // 2, 43 * GIB // 4, 1 << self.worker_cpu)

    def environment(self):
        # Do not inherit undocumented arithmetic, backend selection or executable search overrides.
        allowed = {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "HOMEDRIVE", "HOMEPATH"}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        system = next((value for key, value in env.items() if key.upper() == "SYSTEMROOT"), None)
        require(system is not None, "Windows SYSTEMROOT is required")
        env.update(PATH=str(self.runtime_dir) + os.pathsep + str(Path(system) / "System32"),
                   OMP_NUM_THREADS="1", GGML_VK_VISIBLE_DEVICES=str(self.device_index))
        return env

    def argv(self, port, api_key, alias):
        require(type(self.parallel_requests) is int and self.parallel_requests in (1, 2, 4), "Invalid parallel_requests")
        return [str(self.runtime_dir / "llama-server.exe"),
                "--model", str(self.model.path), "--mmproj", str(self.projector.path),
                "--device", "Vulkan0", "--mmproj-device", "Vulkan0", "--n-gpu-layers", "99",
                "--fit", "off", "--load-mode", "none", "--ctx-size", str(CONTEXT * self.parallel_requests), "--n-predict", str(OUTPUT),
                *(["--no-kv-unified"] if self.parallel_requests > 1 else []),
                "--image-max-tokens", "3072", "--batch-size", "128", "--ubatch-size", "128", "--parallel", str(self.parallel_requests),
                "--threads", "1", "--threads-batch", "1", "--flash-attn", "on", "--cache-type-k", "bf16", "--cache-type-v", "bf16",
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
    fields = {"activate", "runtime_dir", "runtime_files", "model", "projector", "template", "device_index", "worker_cpu", "startup_seconds", "request_seconds", "max_image_pixels", "parallel_requests"}
    optional = {"admission_physical_bytes", "admission_commit_bytes"}
    require(isinstance(raw, dict) and fields <= set(raw) <= fields | optional and raw["activate"] == ACTIVATION, "Explicit llama candidate configuration is required")
    runtime = plain_path(raw["runtime_dir"])
    require(runtime.is_dir(), "Runtime directory is required")
    require(isinstance(raw["runtime_files"], list) and 1 <= len(raw["runtime_files"]) <= 250, "Runtime file inventory is required")
    runtime_files = tuple(identity(value, 2 * GIB) for value in raw["runtime_files"])
    require(all(item.path.parent == runtime and item.path.suffix.lower() in (".exe", ".dll") for item in runtime_files), "Runtime identities must name adjacent executable/DLL files")
    names = {item.path.name.casefold() for item in runtime_files}
    require(len(names) == len(runtime_files) and "llama-server.exe" in names, "Runtime inventory must uniquely include llama-server.exe")
    model, projector, template = (identity(raw[name], maximum) for name, maximum in (("model", 12 * GIB), ("projector", 2 * GIB), ("template", 1 << 20)))
    require(len(set(item.path for item in (*runtime_files, model, projector, template))) == len(runtime_files) + 3, "Input identities must be distinct")
    for name, low, high in (("device_index", 0, 15), ("worker_cpu", 0, 63), ("startup_seconds", 1, 1800), ("request_seconds", 1, 1800), ("max_image_pixels", 1, 6_291_456)):
        require(type(raw[name]) is int and low <= raw[name] <= high, f"Invalid {name}")
    require(type(raw["parallel_requests"]) is int and raw["parallel_requests"] in (1, 2, 4), "Invalid parallel_requests")
    forecasts = [raw.get("admission_physical_bytes", 6 * GIB), raw.get("admission_commit_bytes", 11 * GIB)]
    require(all(type(value) is int and GIB <= value <= 32 * GIB for value in forecasts), "Invalid admission forecasts")
    return LlamaConfig(expected_sha256, runtime, runtime_files, model, projector, template,
                       *(raw[name] for name in ("device_index", "worker_cpu", "startup_seconds", "request_seconds", "max_image_pixels", "parallel_requests")), *forecasts)
