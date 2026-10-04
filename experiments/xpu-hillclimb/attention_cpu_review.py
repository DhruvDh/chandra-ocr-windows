"""CPU-only review of trained fixtures and nine vocabulary rows; no model load."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--numerics", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
run, numerics = Path(args.run), Path(args.numerics)
torch.set_num_threads(8)
torch.set_num_interop_threads(1)


def metrics(a, b):
    a, b = a.float(), b.float()
    diff = a - b
    return {"finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "max_abs": float(diff.abs().max()), "rmse": float(diff.square().mean().sqrt()), "reference_max_abs": float(a.abs().max())}


report = {"scope": "Actual trained post-MRoPE/cache fixture CPU operator comparison; no XPU replay or promotion", "torch": torch.__version__, "original_gate": "FAILED: do not advance", "trained_operators": [], "vocabulary_rows": {}}
receipt = json.loads((run / "attention-candidate.json").read_text())
for desc in receipt["trained_fixtures"]:
    path = run / desc["file"]
    if hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != desc["sha256"] or path.stat().st_size > 64 * 1024**2:
        raise RuntimeError("Fixture hash/size mismatch")
    tensors = load_file(str(path), device="cpu")
    q = tensors["query"]
    k = tensors["key"].repeat_interleave(desc["kv_groups"], dim=1)
    v = tensors["value"].repeat_interleave(desc["kv_groups"], dim=1)
    mask = tensors["additive_mask"]
    oracle = ((q.float() @ k.float().transpose(-1, -2)) * desc["scale"] + mask.float()).softmax(-1) @ v.float()
    eager = ((q @ k.transpose(-1, -2)) * desc["scale"] + mask).softmax(-1, dtype=torch.float32).to(q.dtype) @ v
    with sdpa_kernel([SDPBackend.MATH]):
        boolean = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=tensors["effective_mask"], is_causal=False, dropout_p=0, scale=desc["scale"])
        additive = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, dropout_p=0, scale=desc["scale"])
    report["trained_operators"].append({**desc, "cpu_fp32_vs_eager_bf16": metrics(oracle, eager), "cpu_fp32_vs_masked_sdpa_bf16": metrics(oracle, boolean), "boolean_vs_additive_sdpa": metrics(boolean, additive)})


def export(path):
    meta = json.loads((path / "metadata.json").read_text())
    desc = meta["logits"]
    payload = path / desc["file"]
    if hashlib.file_digest(payload.open("rb"), "sha256").hexdigest() != desc["sha256"]:
        raise RuntimeError("Vocabulary payload hash mismatch")
    return meta, np.memmap(payload, dtype="<f4", mode="r", shape=tuple(desc["shape"]))


cm, candidate = export(run)
for name in ("xpu-cached-hybrid-02", "cpu214-bf16-sdpa-cached-01", "cpu214-fp32-tiny-all-01"):
    rm, reference = export(numerics / name)
    if cm["inputs"] != rm["inputs"] or cm["context"]["prefix_token_ids"] != rm["context"]["prefix_token_ids"]:
        raise RuntimeError("Processed inputs/prefix differ")
    indices = {p: i for i, p in enumerate(rm["context"]["positions"])}
    rows = []
    for i, position in enumerate(cm["context"]["positions"]):
        a = np.asarray(reference[indices[position]], dtype=np.float64)
        b = np.asarray(candidate[i], dtype=np.float64)
        if i < len(cm["context"]["target_token_ids"]) and cm["context"]["target_token_ids"][i] != rm["context"]["target_token_ids"][indices[position]]:
            raise RuntimeError("Conditional teacher targets differ")
        al = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
        bl = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
        top = np.argsort(a)[-10:][::-1]
        # BF16 spacing changes with magnitude: calculate per-logit units using
        # nextafter on exact BF16-rounded reference values, including top ten.
        base = torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
        spacing = (torch.nextafter(base, torch.full_like(base, float("inf"))).float() - base.float()).numpy()
        rows.append({"position": position, "argmax_equal": bool(a.argmax() == b.argmax()), "max_abs": float(np.abs(a - b).max()), "rmse": float(np.sqrt(np.mean((a - b)**2))), "kl_reference_candidate": float((np.exp(al) * (al - bl)).sum()), "fraction_logits_abs_delta_above_0_0625": float((np.abs(a - b) > .0625).mean()), "top10_max_abs": float(np.abs(a[top] - b[top]).max()), "top10_max_bf16_spacing_units": float(np.max(np.abs(a[top] - b[top]) / spacing[top]))})
    report["vocabulary_rows"][name] = {"runtime": rm["runtime"], "cache_equivalence": rm.get("cache_equivalence"), "rows": rows}
hm, hybrid = export(numerics / "xpu-cached-hybrid-02")
fm, fp32 = export(numerics / "cpu214-fp32-tiny-all-01")
if cm["context"] != hm["context"] or cm["inputs"] != hm["inputs"] or cm["inputs"] != fm["inputs"]:
    raise RuntimeError("Side-by-side conditional identity mismatch")
fp32_indices = {p: i for i, p in enumerate(fm["context"]["positions"])}
side_by_side = []
for i, position in enumerate(cm["context"]["positions"]):
    reference = np.asarray(fp32[fp32_indices[position]], dtype=np.float64)
    al = reference - (reference.max() + np.log(np.exp(reference - reference.max()).sum()))
    row = {"position": position}
    for label, values in (("hybrid", hybrid), ("candidate", candidate)):
        actual = np.asarray(values[i], dtype=np.float64)
        bl = actual - (actual.max() + np.log(np.exp(actual - actual.max()).sum()))
        row[label] = {"argmax_equal": bool(reference.argmax() == actual.argmax()), "max_abs": float(np.abs(reference - actual).max()), "rmse": float(np.sqrt(np.mean((reference - actual)**2))), "kl_reference_actual": float((np.exp(al) * (al - bl)).sum())}
    side_by_side.append(row)
report["same_positions_cpu_fp32_comparison"] = {"reference_runtime": fm["runtime"], "hybrid_runtime": hm["runtime"], "rows": side_by_side}
Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"fixtures": len(report["trained_operators"]), "comparators": list(report["vocabulary_rows"])}))
