// New code, MPL-2.0. Model computes FP32 cos/sin once before blocks; duplicate [H16,W16] to full64.
ByteAddressBuffer positions:register(t0); ByteAddressBuffer inverse:register(t1);
RWByteAddressBuffer rotary:register(u0);
cbuffer Params:register(b0) { uint count,pad0,pad1,pad2; }
[numthreads(256,1,1)] void main(uint3 t:SV_DispatchThreadID) {
    if(t.x>=count)return;uint row=t.x/32,dim=t.x%32;
    uint coordinate=positions.Load(row*16+(dim<16?0:4));
    precise float angle=float(coordinate)*asfloat(inverse.Load((dim%16)*4));
    float cs=cos(angle),sn=sin(angle);
    rotary.Store((row*128+dim)*4,asuint(cs));rotary.Store((row*128+dim+32)*4,asuint(cs));
    rotary.Store((row*128+64+dim)*4,asuint(sn));rotary.Store((row*128+96+dim)*4,asuint(sn));
}
