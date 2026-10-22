// New code, MPL-2.0. FP32 dot in ascending k order; no FP16 arithmetic.
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
groupshared float tileInput[16][32];
groupshared float tileWeight[16][32];
[numthreads(16,16,1)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
    uint localOut = group.x * 16 + lane.x;
    uint localRow = group.y * 16 + lane.y;
    uint row = rowFirst + localRow;
    precise float sum = 0.0;
    for (uint begin = 0; begin < inputWidth; begin += 32) {
        // All lanes participate in barriers, including ragged rows/output columns.
        for (uint part = 0; part < 2; ++part) {
            uint ik = begin + lane.x + part * 16;
            tileInput[lane.y][lane.x + part * 16] = (localRow < batchRows && ik < inputWidth)
                ? asfloat(input.Load((row * inputWidth + ik) * 4)) : 0.0;
            uint wk = begin + lane.y + part * 16;
            tileWeight[lane.x][lane.y + part * 16] = (localOut < shardRows && wk < inputWidth)
                ? packed(weight, (weightRowFirst + localOut) * inputWidth + wk, (flags & 1) != 0) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        uint n = min(32, inputWidth - begin);
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
