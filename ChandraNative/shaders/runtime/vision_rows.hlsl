// New code, MPL-2.0. Bounded gather/scatter; no numerical operation.
ByteAddressBuffer source:register(t0); RWByteAddressBuffer destination:register(u0);
cbuffer Params:register(b0) { uint count,inputOffset,outputOffset,pad; }
[numthreads(256,1,1)] void main(uint3 t:SV_DispatchThreadID) {
    if(t.x<count) destination.Store((outputOffset+t.x)*4,source.Load((inputOffset+t.x)*4));
}
