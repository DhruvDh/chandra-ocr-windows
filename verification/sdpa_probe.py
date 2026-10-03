"""Bounded Qwen-shaped BF16 SDPA/GQA probe with frozen inputs and FP32 eager oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import save_file

def digest(path):
    return hashlib.file_digest(open(path,"rb"),"sha256").hexdigest()

def metrics(a,b):
    a,b=a.float().cpu(),b.float().cpu()
    return {"shape":list(a.shape),"finite":bool(torch.isfinite(a).all() and torch.isfinite(b).all()),"max_abs":float((a-b).abs().max()),"rmse":float((a-b).square().mean().sqrt()),"reference_abs_max":float(a.abs().max())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["cpu","xpu"],default="xpu")
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    rng=np.random.Generator(np.random.PCG64(73021))
    arrays={"query":torch.from_numpy(rng.standard_normal((1,16,801,256),dtype=np.float32)),"key":torch.from_numpy(rng.standard_normal((1,4,809,256),dtype=np.float32)),"value":torch.from_numpy(rng.standard_normal((1,4,809,256),dtype=np.float32))}
    # Freeze the exact BF16-rounded inputs used on both CPU and XPU.
    arrays={k:v.to(torch.bfloat16).contiguous() for k,v in arrays.items()}
    save_file(arrays,str(out/"inputs.safetensors"))
    report={"schema":"chandra.sdpa.probe.v1","torch":torch.__version__,"python":platform.python_version(),"device":a.device,"dtype":"bfloat16","seed":73021,"rng":"numpy.PCG64","inputs_sha256":digest(out/"inputs.safetensors"),"script_sha256":digest(__file__),"heads":{"query":16,"key_value":4,"head_dim":256},"strict_backends":["FLASH_ATTENTION","EFFICIENT_ATTENTION","OVERRIDEABLE"],"cases":[]}
    cases=[(1,801,"none",False),(1,809,"none",False),(1,809,"explicit",False),(1,809,"none",True),(801,801,"none",True),(801,809,"explicit",False)]
    for i,(qlen,kvlen,mask_kind,causal) in enumerate(cases):
        q=arrays["query"][:,:,:qlen].contiguous()
        k=arrays["key"][:,:,:kvlen].contiguous()
        v=arrays["value"][:,:,:kvlen].contiguous()
        mask=None
        if mask_kind=="explicit":
            mask=torch.arange(kvlen)[None,:] <= (torch.arange(qlen)[:,None]+kvlen-qlen)
        kc,vc=k.repeat_interleave(4,dim=1).float(),v.repeat_interleave(4,dim=1).float()
        scores=q.float()@kc.transpose(-1,-2)*256**-0.5
        oracle_mask=mask
        if causal:
            oracle_mask=torch.arange(kvlen)[None,:] <= torch.arange(qlen)[:,None]
        if oracle_mask is not None:
            scores=scores.masked_fill(~oracle_mask,float("-inf"))
        expected=scores.softmax(-1)@vc
        expected.numpy().astype("<f4").tofile(out/f"case-{i}-cpu-fp32.f32")
        record={"query_length":qlen,"kv_length":kvlen,"mask":mask_kind,"is_causal":causal,"oracle_sha256":digest(out/f"case-{i}-cpu-fp32.f32"),"variants":{}}
        outputs={}
        for variant,gqa in [("gqa",True),("manual_repeat",False)]:
            try:
                qd=q.to(a.device);kd=k.to(a.device);vd=v.to(a.device)
                if not gqa:
                    kd=kd.repeat_interleave(4,dim=1);vd=vd.repeat_interleave(4,dim=1)
                md=mask.to(a.device) if mask is not None else None
                if a.device=="xpu":
                    context=sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.OVERRIDEABLE])
                else:
                    context=sdpa_kernel([SDPBackend.MATH])
                with torch.inference_mode(),context:
                    actual=torch.nn.functional.scaled_dot_product_attention(qd,kd,vd,attn_mask=md,is_causal=causal,enable_gqa=gqa,dropout_p=0.0,scale=256**-0.5).float().cpu()
                actual.numpy().astype("<f4").tofile(out/f"case-{i}-{variant}.f32")
                outputs[variant]=actual
                record["variants"][variant]={**metrics(expected,actual),"payload_sha256":digest(out/f"case-{i}-{variant}.f32")}
            except Exception as exc:
                record["variants"][variant]={"error":type(exc).__name__+": "+str(exc)}
        if len(outputs)==2:
            record["gqa_vs_manual_repeat"]=metrics(outputs["gqa"],outputs["manual_repeat"])
        report["cases"].append(record)
        (out/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps({"case":i,**record}),flush=True)
    report["scope"]="Synthetic operator evidence; no full-model or generation acceptance implied. is_causal=True,q_len=1 intentionally uses upper-left causal alignment."
    (out/"report.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
