"""Bounded MSVC host diagnostics; no model load or inference."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--cwd", required=True, type=Path)
p.add_argument("--command-json", type=Path)
a = p.parse_args()
a.cwd.mkdir(parents=True, exist_ok=False)
environment = dict(os.environ)
compiler = shutil.which("cl.EXE")
source = a.cwd / "compiler-smoke.cpp"
source.write_text("int main() { return 0; }\n")
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetDllDirectoryW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p]
kernel.GetDllDirectoryW.restype = ctypes.c_uint
kernel.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
kernel.SetDllDirectoryW.restype = ctypes.c_int

def dll_directory():
    buffer = ctypes.create_unicode_buffer(32768)
    kernel.GetDllDirectoryW(len(buffer), buffer)
    return buffer.value

def check(phase):
    result = subprocess.run([compiler, "/?"], cwd=a.cwd, env=environment, capture_output=True)
    print(json.dumps({"phase": phase, "compiler": compiler, "dll_directory": dll_directory(),
                      "returncode": result.returncode, "stdout": result.stdout.decode(errors="replace")[:1000],
                      "stderr": result.stderr.decode(errors="replace")[:1000]}), flush=True)
    build = subprocess.run([compiler, "/nologo", str(source), "/Fo" + str(a.cwd / "smoke.obj"),
                            "/Fe" + str(a.cwd / "smoke.exe")], cwd=a.cwd, env=environment, capture_output=True)
    print(json.dumps({"phase": phase + "_real_compile", "returncode": build.returncode,
                      "stdout": build.stdout.decode(errors="replace")[:2000],
                      "stderr": build.stderr.decode(errors="replace")[:2000]}), flush=True)
    if a.command_json:
        exact = subprocess.run(json.loads(a.command_json.read_text()), cwd=a.cwd, env=environment, capture_output=True)
        print(json.dumps({"phase": phase + "_exact_helper", "returncode": exact.returncode,
                          "stdout": exact.stdout.decode(errors="replace")[:5000],
                          "stderr": exact.stderr.decode(errors="replace")[:5000]}), flush=True)

check("before_runtime")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone.bootstrap import prepare_runtime
prepare_runtime()
check("after_bootstrap")
import torch
check("after_torch_import")
if a.command_json:
    tensor = torch.zeros(1, device="xpu")
    torch.xpu.synchronize()
    check("after_xpu_initialization")
saved = dll_directory()
kernel.SetDllDirectoryW(None)
try:
    check("after_reset_dll_directory")
finally:
    kernel.SetDllDirectoryW(saved or None)
