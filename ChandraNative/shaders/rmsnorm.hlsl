// New ChandraNative code, MPL-2.0. FP32 sum and output; epsilon is configurable.
// Gated RMSNorm here is weight * rms(x) * silu(z); weight is the effective scale.
Buffer<float> input : register(t0);
Buffer<float> weight : register(t1);
Buffer<float> gate : register(t2);
RWBuffer<float> output : register(u0);
cbuffer Params : register(b0) { uint width; uint rows; uint gated; float epsilon; };
groupshared float sums[256];
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    uint base = group.x * width;
    float sum = 0;
    for (uint i = thread; i < width; i += 256) sum += input[base+i] * input[base+i];
    sums[thread] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride; stride >>= 1) {
        if (thread < stride) sums[thread] += sums[thread+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    float scale = rsqrt(sums[0] / width + epsilon);
    for (uint j = thread; j < width; j += 256) {
        float y = input[base+j] * scale * weight[j];
        if (gated) { float z=gate[base+j]; y *= z / (1+exp(-z)); }
        output[base+j] = y;
    }
}
