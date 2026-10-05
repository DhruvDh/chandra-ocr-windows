// New code, MPL-2.0. <=256 partials/output element; final SDPA boundary BF16.
#include "vision_common.hlsl"
ByteAddressBuffer partials:register(t0); RWByteAddressBuffer result:register(u0);
cbuffer Params:register(b0) { uint count,rowFirst,tileCount,pad; }
[numthreads(256,1,1)] void main(uint3 t:SV_DispatchThreadID) {
    if(t.x>=count)return;precise float sum=0.0;
    for(uint tile=0;tile<tileCount;tile++)sum+=asfloat(partials.Load((tile*count+t.x)*4));
    result.Store((rowFirst*1024+t.x)*4,asuint(vision_bf16(sum)));
}
