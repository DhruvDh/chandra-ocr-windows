"""Explicit DirectCompute native endpoint backend; importing this package starts no process, Device or model."""
from .backend import NativeBackend, NativeFailure
from .config import ConfigurationError, NativeConfig, load_config

__all__ = ["NativeBackend", "NativeFailure", "ConfigurationError", "NativeConfig", "load_config"]
