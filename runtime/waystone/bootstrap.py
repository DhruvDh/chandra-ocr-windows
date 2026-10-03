"""Load this locked environment's DLLs by absolute path before PyTorch.

Windows can reject dependency-name searches that encounter untrusted mount points.
No DLL bytes, drive trust, machine settings, or global PATH are modified.
"""
import ctypes
import os
import sys
from pathlib import Path

_handles = []
_directories = []
_ready = False


def prepare_runtime() -> None:
    global _ready
    if _ready or sys.platform != "win32":
        return
    roots = [Path(sys.prefix) / "Library/bin", Path(sys.prefix) / "Lib/site-packages/torch/lib", Path(sys.base_prefix)]
    files = {p.name.lower(): p for root in roots for p in root.glob("*.dll")}
    required = ["vcruntime140.dll", "libiomp5md.dll", "uv.dll", "ur_win_proxy_loader.dll", "libmmd.dll", "sycl9.dll", "pti_view-1.dll", "vcruntime140_1.dll", "c10.dll", "torch_cpu.dll", "shm.dll", "python312.dll", "c10_xpu.dll", "mkl_core.3.dll", "mkl_sycl_blas.6.dll", "mkl_sycl_dft.6.dll", "mkl_sycl_lapack.6.dll", "torch_xpu.dll", "torch_python.dll"]
    missing = [name for name in required if name not in files]
    if missing:
        raise RuntimeError(f"Locked XPU runtime DLLs missing: {missing}")
    os.environ["PATH"] = os.pathsep.join(str(root) for root in roots) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    _directories.extend(os.add_dll_directory(str(root)) for root in roots)
    _handles.extend(ctypes.WinDLL(str(files[name])) for name in required)
    _ready = True
