"""A bounded synthetic Chandra request; writes evidence outside tracked source."""
import argparse, base64, hashlib, io, json, sys, threading, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone.bootstrap import prepare_runtime
prepare_runtime()
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.waystone import XPUBackend
from chandra.prompts import PROMPT_MAPPING

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
image = Image.new("RGB", (512, 256), "white")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default(size=28)
draw.text((24, 32), "CHANDRA TEST", fill="black", font=font)
draw.text((24, 90), "Total: 123.45", fill="black", font=font)
draw.text((24, 148), "Alpha beta 2026", fill="black", font=font)
b = io.BytesIO(); image.save(b, format="PNG"); raw = b.getvalue()
request = {"model": "chandra", "temperature": 0, "max_tokens": 256, "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT_MAPPING["ocr_layout"]}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}]}]}
backend = XPUBackend(a.model)
t = time.perf_counter(); backend.load(); load_seconds = time.perf_counter() - t
torch.xpu.reset_peak_memory_stats(); t = time.perf_counter()
parts = list(backend.generate(request, threading.Event()))
result = {"torch": torch.__version__, "device": torch.xpu.get_device_name(0), "image_sha256": hashlib.sha256(raw).hexdigest(), "load_seconds": load_seconds, "generate_seconds": time.perf_counter() - t, "parts": parts, "memory": {"allocated": torch.xpu.memory_allocated(), "reserved": torch.xpu.memory_reserved(), "peak": torch.xpu.max_memory_allocated()}}
Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
assert "123.45" in "".join(x for x in parts if isinstance(x, str)), "Synthetic amount missing"
