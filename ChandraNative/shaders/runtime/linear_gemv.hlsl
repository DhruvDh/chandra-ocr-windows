// New code, MPL-2.0. Experimental opt-in batch-row-1 projection (CHANDRA_EXPERIMENTAL_GEMV_B1=ordered).
// Each output keeps linear.hlsl's operation sequence: one FP32 sum from +0, then sum = sum + a[k]*w[k]
// in ascending k, then bias, then optional BF16 RNE. Only memory staging differs: 32 lanes per output
// row load consecutive words of one K tile, and one barrier pair serves 512 K instead of 32 K.
// No FP16 arithmetic, split-K, reassociation, transposition or prepacking.
ByteAddressBuffer input : register(t0);
ByteAddressBuffer weight : register(t1);
ByteAddressBuffer bias : register(t2);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint inputWidth, outputWidth, batchRows, firstOutput;
    uint shardRows, rowFirst, flags, weightRowFirst;
};
// Host contract: batchRows == 1 and rowFirst == 0; only activation row 0 is read and written.
#define ROWS 8
#define LANES 32
#define THREADS 256
#define TILE 512
#define STRIDE 513
// Maximum staged words per lane: FP32 TILE/LANES = 16; packed BF16 needs ceil((TILE+1)/2/LANES) = 9.
#define STAGED 16
groupshared float tileWeight[ROWS * STRIDE];
groupshared float tileInput[TILE];
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
[numthreads(THREADS,1,1)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
    uint thread = lane.x;
    uint stageRow = thread / LANES, stageLane = thread % LANES;
    uint stageOut = group.x * ROWS + stageRow;
    uint localOut = group.x * ROWS + thread;
    bool owner = thread < ROWS && localOut < shardRows;
    bool bf16 = (flags & 1) != 0;
    precise float sum = 0.0;
    for (uint begin = 0; begin < inputWidth; begin += TILE) {
        uint n = min(TILE, inputWidth - begin);
        // Element index of this tile's first weight in the shard; host validation keeps every
        // touched word below the shard's uint32 byte range.
        uint first = (weightRowFirst + stageOut) * inputWidth + begin;
        uint parity = bf16 ? (first & 1) : 0;
        uint base = bf16 ? (first >> 1) : first;
        uint count = stageOut < shardRows ? (bf16 ? ((parity + n + 1) >> 1) : n) : 0;
        // Issue all global loads before any groupshared store so they can be in flight together.
        uint staged[STAGED];
        [unroll] for (uint i = 0; i < STAGED; ++i) {
            uint j = stageLane + i * LANES;
            staged[i] = 0;
            if (j < count) staged[i] = weight.Load((base + j) * 4);
        }
        [unroll] for (uint m = 0; m < TILE / THREADS; ++m) {
            uint index = thread + m * THREADS;
            if (index < n) tileInput[index] = asfloat(input.Load((begin + index) * 4));
        }
        [unroll] for (uint s = 0; s < STAGED; ++s) {
            uint j = stageLane + s * LANES;
            if (j < count) {
                uint row = stageRow * STRIDE;
                if (!bf16) {
                    tileWeight[row + j] = asfloat(staged[s]);
                } else {
                    // Word j holds tile positions 2j-parity (low half) and 2j+1-parity (high half).
                    if (2 * j >= parity) tileWeight[row + 2 * j - parity] = asfloat(staged[s] << 16);
                    if (2 * j + 1 - parity < n) tileWeight[row + 2 * j + 1 - parity] = asfloat(staged[s] & 0xffff0000);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (owner) {
            uint row = thread * STRIDE;
            uint k = 0;
            for (; k + 4 <= n; k += 4) {
                float a0 = tileInput[k], a1 = tileInput[k + 1], a2 = tileInput[k + 2], a3 = tileInput[k + 3];
                float w0 = tileWeight[row + k], w1 = tileWeight[row + k + 1];
                float w2 = tileWeight[row + k + 2], w3 = tileWeight[row + k + 3];
                precise float p0 = a0 * w0;
                sum = sum + p0;
                precise float p1 = a1 * w1;
                sum = sum + p1;
                precise float p2 = a2 * w2;
                sum = sum + p2;
                precise float p3 = a3 * w3;
                sum = sum + p3;
            }
            for (; k < n; ++k) {
                precise float product = tileInput[k] * tileWeight[row + k];
                sum = sum + product;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (owner) {
        uint column = firstOutput + localOut;
        if ((flags & 4) != 0) sum = sum + packed(bias, column, (flags & 8) != 0);
        if ((flags & 2) != 0) sum = bf16_rne(sum);
        output.Store(column * 4, asuint(sum));
    }
}
