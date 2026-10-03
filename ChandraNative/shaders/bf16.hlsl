// New ChandraNative code, MPL-2.0. BF16 storage is not R16_FLOAT.
Buffer<uint> packed : register(t0);
RWBuffer<float> output : register(u0);
cbuffer Params : register(b0) { uint count; uint3 reserved; };
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < count) {
        uint bits = (packed[tid.x / 2] >> ((tid.x & 1) * 16)) & 0xffff;
        output[tid.x] = asfloat(bits << 16);
    }
}
