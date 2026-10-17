// New code, MPL-2.0. Qwen3.5 axial head_dim64: spatial_dim32, theta10000, sixteen FP32 inverse frequencies.
RWByteAddressBuffer inverse:register(u0);
[numthreads(16,1,1)] void main(uint lane:SV_GroupIndex) {
    precise float value=1.0/pow(10000.0,float(2*lane)/32.0);
    inverse.Store(lane*4,asuint(value));
}
