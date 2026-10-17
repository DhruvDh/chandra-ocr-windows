// New code, MPL-2.0. Inactive GEMM candidate; see docs/directcompute-gemm-candidates.md.
// Same cbuffer, bindings, 16x16 group, packed BF16 addressing, bias, rounding and store as
// linear.hlsl. Each output keeps the identical precise FP32 product-then-sum chain from +0 in
// ascending k. Only staging changes: K64 tiles, rows padded to 65 words, staging lanes walk k.
ByteAddressBuffer input : register(t0);
ByteAddressBuffer weight : register(t1);
ByteAddressBuffer bias : register(t2);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint inputWidth, outputWidth, batchRows, firstOutput;
    uint shardRows, rowFirst, flags, weightRowFirst;
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
#define TILE_K 64
#define PARTS 4
// Odd row stride: lanes reading one k from 16 distinct rows touch 16 distinct words modulo any
// power-of-two bank count (a structural hypothesis about A770 SLM, not a measured fact).
groupshared float tileInput[16][TILE_K + 1];
groupshared float tileWeight[16][TILE_K + 1];
[numthreads(16,16,1)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
    uint localOut = group.x * 16 + lane.x;
    uint localRow = group.y * 16 + lane.y;
    uint row = rowFirst + localRow;
    // Staging: lane.y picks the tile row of both operands and lane.x + 16 * part its k, so
    // adjacent lanes load adjacent k of one input row or one weight row.
    uint stageOut = group.x * 16 + lane.y;
    bool bf16Weight = (flags & 1) != 0;
    precise float sum = 0.0;
    for (uint begin = 0; begin < inputWidth; begin += TILE_K) {
        // All lanes participate in barriers, including ragged rows/output columns.
        float a[PARTS], w[PARTS];
        [unroll] for (uint part = 0; part < PARTS; ++part) {
            uint k = begin + lane.x + part * 16;
            a[part] = (localRow < batchRows && k < inputWidth)
                ? asfloat(input.Load((row * inputWidth + k) * 4)) : 0.0;
            w[part] = (stageOut < shardRows && k < inputWidth)
                ? packed(weight, (weightRowFirst + stageOut) * inputWidth + k, bf16Weight) : 0.0;
        }
        [unroll] for (uint store = 0; store < PARTS; ++store) {
            tileInput[lane.y][lane.x + store * 16] = a[store];
            tileWeight[lane.y][lane.x + store * 16] = w[store];
        }
        GroupMemoryBarrierWithGroupSync();
        uint n = min(TILE_K, inputWidth - begin);
        for (uint k = 0; k < n; ++k) {
            precise float product = tileInput[lane.y][k] * tileWeight[lane.x][k];
            sum = sum + product;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (localRow < batchRows && localOut < shardRows) {
        uint column = firstOutput + localOut;
        if ((flags & 4) != 0) sum = sum + packed(bias, column, (flags & 8) != 0);
        if ((flags & 2) != 0) sum = bf16_rne(sum);
        output.Store((row * outputWidth + column) * 4, asuint(sum));
    }
}
