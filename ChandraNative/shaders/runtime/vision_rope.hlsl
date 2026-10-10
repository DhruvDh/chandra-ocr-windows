// New code, MPL-2.0. Axial vision RoPE: 16 H + 16 W frequencies, duplicated to 64.
#include "vision_common.hlsl"
ByteAddressBuffer qkv:register(t0); ByteAddressBuffer rotary:register(t1);
RWByteAddressBuffer q:register(u0); RWByteAddressBuffer k:register(u1); RWByteAddressBuffer v:register(u2);
cbuffer Params:register(b0) { uint rows,rowFirst,count,pad; }
[numthreads(256,1,1)] void main(uint3 t:SV_DispatchThreadID) {
    if(t.x>=count)return;
    uint r=t.x/1024,c=t.x%1024,dim=c%64;
    float cs=asfloat(rotary.Load(((rowFirst+r)*128+dim)*4));
    float sn=asfloat(rotary.Load(((rowFirst+r)*128+64+dim)*4));
    uint other=dim<32?c+32:c-32;
    precise float qrot=asfloat(qkv.Load((r*3072+other)*4))*(dim<32?-1.0:1.0);
    precise float krot=asfloat(qkv.Load((r*3072+1024+other)*4))*(dim<32?-1.0:1.0);
    precise float qvalue=asfloat(qkv.Load((r*3072+c)*4))*cs+qrot*sn;
    precise float kvalue=asfloat(qkv.Load((r*3072+1024+c)*4))*cs+krot*sn;
    uint outputAddress=((rowFirst+r)*1024+c)*4;
    q.Store(outputAddress,asuint(vision_bf16(qvalue))); k.Store(outputAddress,asuint(vision_bf16(kvalue)));
    v.Store(outputAddress,qkv.Load((r*3072+2048+c)*4));
}
