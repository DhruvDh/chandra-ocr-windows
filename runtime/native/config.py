"""Explicit native endpoint configuration; loading it hashes files but starts no process or Device.

The JSON file must be authenticated by its SHA-256, name every path absolutely and carry the explicit
activation marker. Nothing is defaulted that selects hardware, a worker build, its HLSL source or a limit.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
import re
import stat

from .protocol import CONFIG_SHA256, MODEL_SHA256, ARC_A770
from .shaders import HEX64, ShaderSourceRefused, verified_tree, directory_names, MAX_WORK_ROOT_ENTRIES
from .winjob import BREAKAWAY_FLAGS, LIMIT_FLAGS, JobLimits

SCHEMA = "chandra.native-endpoint.config.v1"
ACTIVATION = "serve-unqualified-directcompute-native-chandra"
CALLER_SOURCE_BYTES = 3178
MODEL_BYTES = 10591220088
MAX_CONFIG_BYTES = 65536
# The serving profile bounds processed pixels at 3,145,728, i.e. 12,288 16x16 patch rows.
SERVING_PATCH_CEILING = 12288
REQUIRED = {"schema", "activate", "executable", "executable_sha256", "model_dir", "shader_root", "shader_tree_sha256", "input_root",
            "output_root", "caller_source", "pci", "luid", "gemv_b1_selection", "lease_ms", "process_memory_limit_bytes",
            "job_memory_limit_bytes", "worker_cpu", "max_patch_rows"}
OPTIONAL = {"max_image_pixels": 4_000_000, "startup_seconds": 900, "admission_seconds": 60, "cancel_grace_seconds": 300,
            "shutdown_seconds": 120, "terminate_seconds": 30, "retained_outputs": 16, "processor_dll_bootstrap": False}
REPARSE = 0x400


class ConfigurationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ConfigurationError(message)


def sha256_file(path, limit=None):
    hasher, total = hashlib.sha256(), 0
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            require(limit is None or total <= limit, f"{path.name} exceeds its size bound")
            hasher.update(chunk)
    return hasher.hexdigest(), total


def plain_path(value, name, kind):
    """Absolute, normalized, existing, with no symlink or reparse component from the root down."""
    require(isinstance(value, str) and value and "\0" not in value and len(value) <= 1024, f"{name} must be a path string")
    path = Path(value)
    require(path.is_absolute(), f"{name} must be absolute")
    require(not any(part in (".", "..") for part in PurePath(value).parts) and str(path) == value.rstrip("/\\") == value,
            f"{name} must be normalized without trailing separators")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError:
            raise ConfigurationError(f"{name} has a missing component") from None
        require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & REPARSE),
                f"{name} has a symlink or reparse component")
    info = os.lstat(path)
    require(stat.S_ISDIR(info.st_mode) if kind == "dir" else stat.S_ISREG(info.st_mode), f"{name} must be an existing plain {kind}")
    return path


def nested(a, b):
    a, b = os.path.normcase(str(a)).rstrip("/\\"), os.path.normcase(str(b)).rstrip("/\\")
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


@dataclass(frozen=True)
class ReadyExpectation:
    """Commitments the worker's ready event must report exactly."""
    pci: str
    luid: str
    selection: str
    lease_ms: int
    process_memory_limit: int
    job_memory_limit: int
    model_sha256: str = MODEL_SHA256
    config_sha256: str = CONFIG_SHA256
    device: tuple = tuple(ARC_A770.items())
    active_process_limit: int | None = 1
    job_limit_flags: int = LIMIT_FLAGS
    job_breakaway_flags: int = BREAKAWAY_FLAGS

    def __post_init__(self):
        object.__setattr__(self, "device", dict(self.device))


@dataclass(frozen=True)
class NativeConfig:
    config_sha256: str
    executable: Path
    executable_sha256: str
    model_dir: Path
    shader_root: Path
    shader_tree_sha256: str
    input_root: Path
    output_root: Path
    caller_source: Path
    pci: str
    luid: str
    gemv_b1_selection: str
    lease_ms: int
    process_memory_limit_bytes: int
    job_memory_limit_bytes: int
    max_patch_rows: int
    max_image_pixels: int
    startup_seconds: float
    admission_seconds: float
    cancel_grace_seconds: float
    shutdown_seconds: float
    terminate_seconds: float
    retained_outputs: int
    worker_cpu: int
    processor_dll_bootstrap: bool = False

    def expectation(self):
        return ReadyExpectation(self.pci, self.luid, self.gemv_b1_selection, self.lease_ms,
                                self.process_memory_limit_bytes, self.job_memory_limit_bytes)

    def argv(self, shader_snapshot):
        """The worker compiles from the task-owned verified snapshot, never from the configured shader_root."""
        return [str(self.executable), "--execute", "--model-dir", str(self.model_dir), "--shader-root", str(shader_snapshot),
                "--input-root", str(self.input_root), "--output-root", str(self.output_root), "--pci", self.pci,
                "--luid", self.luid, "--lease-ms", str(self.lease_ms)]

    def job_limits(self):
        return JobLimits(self.process_memory_limit_bytes, self.job_memory_limit_bytes, 1 << self.worker_cpu)

    def environment(self, base=None):
        """Inherited environment minus every CHANDRA_* opt-in; the GEMV selector is set only explicitly."""
        env = {k: v for k, v in (os.environ if base is None else base).items() if not k.upper().startswith("CHANDRA_")}
        if self.gemv_b1_selection == "ordered":
            env["CHANDRA_EXPERIMENTAL_GEMV_B1"] = "ordered"
        return env

    def verify_executable(self):
        """Re-hash immediately before every launch; a changed build is refused, never substituted.

        Windows loads a DLL from the executable's directory before the system copy, so a d3dcompiler_47.dll placed
        beside the worker would compile the HLSL; such a directory is refused.
        """
        executable = plain_path(str(self.executable), "executable", "file")
        try:
            libraries = [name for name in directory_names(executable.parent, MAX_WORK_ROOT_ENTRIES) if name.lower().endswith(".dll")]
        except ShaderSourceRefused as error:
            raise ConfigurationError("Executable directory refused: " + str(error)) from None
        require(not libraries, "Executable directory holds DLLs Windows would load first: " + ", ".join(libraries[:4]))
        digest, _ = sha256_file(executable, 64 * 1024 * 1024)
        require(digest == self.executable_sha256, "Native worker executable hash differs from configuration")


def _number(value, name, low, high, kind=int):
    ok = type(value) is int if kind is int else (type(value) in (int, float) and value == value)
    require(ok and low <= value <= high, f"{name} must be {kind.__name__} in [{low}, {high}]")
    return value


def _flag(value, name):
    require(type(value) is bool, f"{name} must be boolean")
    return value


def parse(data, digest):
    from scripts.native.prepare_input import CALLER_SOURCE_SHA, PROCESSOR_PINS
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=lambda pairs: _pairs(pairs),
                         parse_constant=lambda value: require(False, "Nonfinite configuration number"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ConfigurationError("Configuration must be strict UTF-8 JSON") from None
    require(isinstance(raw, dict), "Configuration must be a JSON object")
    require(not set(raw) - REQUIRED - set(OPTIONAL), "Unknown configuration fields: " + ", ".join(sorted(set(raw) - REQUIRED - set(OPTIONAL))))
    require(REQUIRED <= set(raw), "Missing configuration fields: " + ", ".join(sorted(REQUIRED - set(raw))))
    require(raw["schema"] == SCHEMA, "Unexpected configuration schema")
    require(raw["activate"] == ACTIVATION, "Native endpoint activation marker is absent")
    values = {**OPTIONAL, **raw}
    paths = {name: plain_path(values[name], name, "dir") for name in ("model_dir", "shader_root", "input_root", "output_root")}
    executable = plain_path(values["executable"], "executable", "file")
    caller = plain_path(values["caller_source"], "caller_source", "file")
    require(isinstance(values["executable_sha256"], str) and re.fullmatch("[0-9a-f]{64}", values["executable_sha256"]), "executable_sha256 must be lowercase SHA-256")
    require(isinstance(values["shader_tree_sha256"], str) and HEX64.fullmatch(values["shader_tree_sha256"]), "shader_tree_sha256 must be lowercase SHA-256")
    for a, b in (("input_root", "output_root"), ("input_root", "model_dir"), ("output_root", "model_dir"),
                 ("input_root", "shader_root"), ("output_root", "shader_root")):
        require(not nested(paths[a], paths[b]), f"{a} and {b} must be distinct and not nested")
    require(isinstance(values["pci"], str) and re.fullmatch(r"[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", values["pci"]), "pci must be the adapter's lowercase BB:DD.F")
    require(isinstance(values["luid"], str) and re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{8}", values["luid"]), "luid must be lowercase HHHHHHHH:LLLLLLLL")
    require(values["gemv_b1_selection"] in ("predecessor", "ordered"), "gemv_b1_selection must be predecessor or ordered")
    lease = _number(values["lease_ms"], "lease_ms", 1000, 600000)
    process_limit = _number(values["process_memory_limit_bytes"], "process_memory_limit_bytes", 1 << 30, 1 << 40)
    job_limit = _number(values["job_memory_limit_bytes"], "job_memory_limit_bytes", process_limit, 1 << 40)
    config = NativeConfig(
        config_sha256=digest, executable=executable, executable_sha256=values["executable_sha256"], caller_source=caller,
        shader_tree_sha256=values["shader_tree_sha256"], worker_cpu=_number(values["worker_cpu"], "worker_cpu", 0, 63),
        pci=values["pci"], luid=values["luid"], gemv_b1_selection=values["gemv_b1_selection"], lease_ms=lease,
        process_memory_limit_bytes=process_limit, job_memory_limit_bytes=job_limit,
        max_patch_rows=_number(values["max_patch_rows"], "max_patch_rows", 4, SERVING_PATCH_CEILING),
        max_image_pixels=_number(values["max_image_pixels"], "max_image_pixels", 1, 6_291_456),
        startup_seconds=_number(values["startup_seconds"], "startup_seconds", 1, 3600, float),
        admission_seconds=_number(values["admission_seconds"], "admission_seconds", 1, 600, float),
        cancel_grace_seconds=_number(values["cancel_grace_seconds"], "cancel_grace_seconds", 1, 3600, float),
        shutdown_seconds=_number(values["shutdown_seconds"], "shutdown_seconds", 1, 3600, float),
        terminate_seconds=_number(values["terminate_seconds"], "terminate_seconds", 1, 600, float),
        retained_outputs=_number(values["retained_outputs"], "retained_outputs", 1, 1024),
        processor_dll_bootstrap=_flag(values["processor_dll_bootstrap"], "processor_dll_bootstrap"), **paths)
    config.verify_executable()
    try:
        verified_tree(config.shader_root, config.shader_tree_sha256)
    except (ShaderSourceRefused, OSError) as error:
        raise ConfigurationError(f"shader_root refused: {error}") from None
    digest_caller, size = sha256_file(caller, CALLER_SOURCE_BYTES)
    require(size == CALLER_SOURCE_BYTES and digest_caller == CALLER_SOURCE_SHA, "caller_source is not the pinned Chandra 0.2.0 generate_hf source")
    for name, expected in PROCESSOR_PINS.items():
        found, _ = sha256_file(plain_path(str(paths["model_dir"] / name), name, "file"), 64 * 1024 * 1024)
        require(found == expected, f"Pinned processor/tokenizer file differs: {name}")
    weights = plain_path(str(paths["model_dir"] / "model.safetensors"), "model.safetensors", "file")
    require(weights.stat().st_size == MODEL_BYTES, "model.safetensors size differs from the pinned checkpoint; the worker authenticates its hash")
    return config


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate configuration key")
        result[key] = value
    return result


def load_config(path, expected_sha256):
    """Read the configuration file once, authenticate its bytes, then validate every commitment."""
    require(isinstance(expected_sha256, str) and re.fullmatch("[0-9a-f]{64}", expected_sha256), "Expected configuration SHA-256 is required")
    path = plain_path(str(path), "configuration", "file")
    with open(path, "rb") as source:
        data = source.read(MAX_CONFIG_BYTES + 1)
    require(len(data) <= MAX_CONFIG_BYTES, "Configuration exceeds 64 KiB")
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "Configuration SHA-256 differs")
    return parse(data, expected_sha256)
