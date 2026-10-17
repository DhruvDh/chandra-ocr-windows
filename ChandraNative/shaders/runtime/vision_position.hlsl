// New code, MPL-2.0. Transformers 5.18 Qwen3.5: bilinear, align_corners=True.
#include "vision_common.hlsl"
ByteAddressBuffer positions:register(t0); ByteAddressBuffer table:register(t1);
RWByteAddressBuffer result:register(u0);
cbuffer Params:register(b0) { uint count,first,width,pad; }
[numthreads(256,1,1)] void main(uint3 t:SV_DispatchThreadID) {
    if(t.x>=count)return;
    uint i=first+t.x,row=i/1024,c=i%1024;
    uint4 p=positions.Load4(row*16); // Integer h-coordinate,w-coordinate,h-size,w-size.
    // Multiplication then division matches the helper's FP32 tensor operation ordering.
    precise float hs=float(p.x)*47.0; hs=hs/float(max(p.z-1,1));
    precise float ws=float(p.y)*47.0; ws=ws/float(max(p.w-1,1));
    float hf=floor(hs),wf=floor(ws);
    uint h0=min(uint(hf),47),h1=min(uint(hf)+1,47);
    uint w0=min(uint(wf),47),w1=min(uint(wf)+1,47);
    precise float a0=max(1.0-abs(hs-hf),0.0),a1=max(1.0-abs(hs-hf-1.0),0.0);
    precise float b0=max(1.0-abs(ws-wf),0.0),b1=max(1.0-abs(ws-wf-1.0),0.0);
    precise float v0=vision_weight(table,(h0*48+w0)*1024+c)*(a0*b0);
    precise float v1=vision_weight(table,(h0*48+w1)*1024+c)*(a0*b1);
    precise float v2=vision_weight(table,(h1*48+w0)*1024+c)*(a1*b0);
    precise float v3=vision_weight(table,(h1*48+w1)*1024+c)*(a1*b1);
    precise float sum=((v0+v1)+v2)+v3;
    result.Store(i*4,asuint(vision_bf16(sum)));
}
