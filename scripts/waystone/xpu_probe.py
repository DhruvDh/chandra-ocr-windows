"""Bounded XPU numerical prerequisites; no model downloads or CPU fallback."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone.bootstrap import prepare_runtime
prepare_runtime()
import torch
import torch.nn.functional as F

assert torch.xpu.is_available(), "Intel XPU unavailable"
torch.manual_seed(731)
results = {"torch": torch.__version__, "device": torch.xpu.get_device_name(0), "cases": []}
for dtype in (torch.float32, torch.bfloat16):
    a = torch.randn(16, 32).to(dtype)
    b = torch.randn(32, 16).to(dtype)
    gpu = (a.to("xpu") @ b.to("xpu")).cpu()
    ref = a.float() @ b.float()
    error = (gpu.float() - ref).abs().max().item()
    assert torch.isfinite(gpu).all()
    assert error < (0.001 if dtype == torch.float32 else 0.2), error
    x = a.to("xpu")
    rms = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    conv = F.conv1d(x.reshape(1, 16, 32), torch.ones(16, 1, 4, device="xpu", dtype=dtype), groups=16)
    q = torch.randn(1, 2, 8, 16, device="xpu", dtype=dtype)
    attn = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert all(torch.isfinite(t).all() for t in (rms, conv, attn))
    results["cases"].append({"dtype": str(dtype), "matmul_max_abs_error": error, "rmsnorm": True, "depthwise_conv": True, "sdpa": True})
torch.xpu.synchronize()
results["memory"] = {"allocated": torch.xpu.memory_allocated(), "reserved": torch.xpu.memory_reserved(), "peak": torch.xpu.max_memory_allocated()}
print(json.dumps(results, indent=2))
