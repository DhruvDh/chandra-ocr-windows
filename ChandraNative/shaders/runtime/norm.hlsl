// New code, MPL-2.0. FP32 reductions, explicit effective weight and BF16 boundary.
// RMS: x * rsqrt(mean(x*x)+eps) * (onePlus ? 1+w : w).
// LayerNorm: biased centered variance; scale and bias before the final boundary.
// SM5 rsqrt has implementation error; this is not a bitwise PyTorch claim.
ByteAddressBuffer input : register(t0);
ByteAddressBuffer weight : register(t1);
ByteAddressBuffer bias : register(t2);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint width, rows, rowFirst, flags;
    float epsilon; uint layerNorm, reserved0, reserved1;
};
float packed(ByteAddressBuffer source, uint index, bool bf16) {
    if (!bf16) return asfloat(source.Load(index * 4));
    uint word = source.Load((index >> 1) * 4);
    return asfloat(((word >> ((index & 1) * 16)) & 0xffff) << 16);
}
float bf16_rne(float value) {
    uint bits = asuint(value);
    if ((bits & 0x7fffffff) > 0x7f800000) return asfloat((bits & 0xffff0000) | 0x00400000);
    return asfloat((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}
groupshared float partial[256];
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint row = rowFirst + group.x;
    // Host dispatches exactly rows groups: no divergent early return at barriers.
    uint base = row * width;
    precise float sum = 0.0;
    for (uint j = lane; j < width; j += 256) {
        float x = asfloat(input.Load((base+j)*4));
        precise float term = layerNorm != 0 ? x : x*x;
        sum = sum + term;
    }
    partial[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (lane < stride) partial[lane] = partial[lane] + partial[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    precise float mean = layerNorm != 0 ? partial[0] / float(width) : 0.0;
    GroupMemoryBarrierWithGroupSync(); // Read mean before the centered pass overwrites partial.
    if (layerNorm != 0) {
        sum = 0.0;
        for (uint j = lane; j < width; j += 256) {
            precise float centered = asfloat(input.Load((base+j)*4)) - mean;
            precise float squared = centered * centered;
            sum = sum + squared;
        }
        partial[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128; stride != 0; stride >>= 1) {
            if (lane < stride) partial[lane] = partial[lane] + partial[lane+stride];
            GroupMemoryBarrierWithGroupSync();
        }
    }
    precise float variance = partial[0] / float(width);
    precise float scale = rsqrt(variance + epsilon);
    for (uint j = lane; j < width; j += 256) {
        precise float x = asfloat(input.Load((base+j)*4)) - mean;
        precise float gain = packed(weight,j,(flags & 1) != 0);
        if ((flags & 4) != 0) gain = 1.0 + gain;
        precise float y = x * scale;
        y = y * gain;
        if (layerNorm != 0) y = y + packed(bias,j,(flags & 8) != 0);
        if ((flags & 2) != 0) y = bf16_rne(y);
        output.Store((base+j)*4,asuint(y));
    }
}
