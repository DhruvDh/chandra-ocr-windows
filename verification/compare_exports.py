"""Compare complete-vocabulary binary exports without rounding logits through JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def load(directory):
    root = Path(directory)
    meta = json.loads((root / "metadata.json").read_text())
    desc = meta["logits"]
    path = root / desc["file"]
    if path.stat().st_size != int(np.prod(desc["shape"]))*4:
        raise ValueError("Logits payload length mismatch")
    if hashlib.file_digest(open(path,"rb"),"sha256").hexdigest() != desc["sha256"]:
        raise ValueError("Logits payload hash mismatch")
    rows = np.memmap(path,dtype="<f4",mode="r",shape=tuple(desc["shape"]))
    if rows.shape[1] != meta["context"]["vocabulary_size"] or not np.isfinite(rows).all():
        raise ValueError("Incomplete vocabulary or nonfinite logits")
    return meta, rows

def compare(reference,candidate):
    rm, r = load(reference)
    cm, c = load(candidate)
    if rm["context"] != cm["context"] or rm["inputs"] != cm["inputs"]:
        raise ValueError("Conditional distribution or processed inputs differ")
    if r.shape != c.shape:
        raise ValueError("Different exported row shapes")
    rows = []
    for i,(a,b) in enumerate(zip(r,c)):
        a, b = a.astype(np.float64), b.astype(np.float64)
        al = a - (a.max() + np.log(np.exp(a-a.max()).sum()))
        bl = b - (b.max() + np.log(np.exp(b-b.max()).sum()))
        target = rm["context"]["target_token_ids"][i] if i < len(rm["context"]["target_token_ids"]) else None
        margin = np.partition(a,-2)[-2:]
        rows.append({"position":rm["context"]["positions"][i],"max_abs":float(np.abs(a-b).max()),"rmse":float(np.sqrt(np.mean((a-b)**2))),"centered_rmse":float(np.sqrt(np.mean(((a-a.mean())-(b-b.mean()))**2))),"centered_max_abs":float(np.abs((a-a.mean())-(b-b.mean())).max()),"reference_argmax":int(a.argmax()),"candidate_argmax":int(b.argmax()),"argmax_equal":bool(a.argmax()==b.argmax()),"kl_reference_candidate":float((np.exp(al)*(al-bl)).sum()),"reference_top1_top2_margin":float(margin.max()-margin.min()),"target_token_id":target,"target_log_probability_error":float(abs(al[target]-bl[target])) if target is not None else None})
    return {"identity_equal":True,"reference_runtime":rm["runtime"],"candidate_runtime":cm["runtime"],"rows":rows,"argmax_agreement":sum(row["argmax_equal"] for row in rows)/len(rows),"max_abs":max(row["max_abs"] for row in rows),"max_kl":max(row["kl_reference_candidate"] for row in rows),"acceptance":"measurements only; tolerance and complete OCR gates are separately required"}

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--output")
    args=p.parse_args()
    result=json.dumps(compare(args.reference,args.candidate),indent=2)
    if args.output:
        Path(args.output).write_text(result)
    else:
        print(result)
