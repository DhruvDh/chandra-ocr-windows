// New code, MPL-2.0. Inactive unified parallel32 genuine-B2 projection candidate.
// Explicit worker flag --experimental-unified-parallel32-b2 plus captured/live B1=parallel32.
// REASSOCIATED FP32: lane l accumulates k = l (mod 32), ascending across all tiles, from +0.
// A fixed tree reduces lane pairs at offsets 16,8,4,2,1, then adds bias and optionally BF16 RNE.
// This is not bit-equal to ordered8 or linear.hlsl. No FMA: precise products and sums are separate.
// Each real row independently reproduces linear_gemv_parallel32.hlsl; weight staging is shared.
// Exact host ABI: batchRows==2, rowFirst selects two contiguous real activation/output rows.
ByteAddressBuffer input : register(t0);
ByteAddressBuffer weight : register(t1);
ByteAddressBuffer bias : register(t2);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint inputWidth, outputWidth, batchRows, firstOutput;
    uint shardRows, rowFirst, flags, weightRowFirst;
};
// Host validates complete buffers/shards and <=1024 outputs before submission; no padded input row.
#define ROWS 8
#define LANES 32
#define THREADS 256
#define TILE 512
#define STRIDE 513
#define STAGED 16
groupshared float tileWeight[ROWS * STRIDE];
groupshared float tileInput[2 * TILE];
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
    bool activeRow = stageOut < shardRows;
    bool bf16 = (flags & 1) != 0;
    precise float sum0 = 0.0;
    precise float sum1 = 0.0;
    for (uint begin = 0; begin < inputWidth; begin += TILE) {
        uint n = min(TILE, inputWidth - begin);
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
            if (index < n) {
                tileInput[index] = asfloat(input.Load((rowFirst * inputWidth + begin + index) * 4));
                tileInput[TILE + index] = asfloat(input.Load(((rowFirst + 1) * inputWidth + begin + index) * 4));
            }
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
        if (activeRow) {
            uint row = stageRow * STRIDE;
            for (uint k = stageLane; k < n; k += LANES) {
                precise float product0 = tileInput[k] * tileWeight[row + k];
                sum0 = sum0 + product0;
                precise float product1 = tileInput[TILE + k] * tileWeight[row + k];
                sum1 = sum1 + product1;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // All tile reads completed. The two input regions now hold disjoint per-row partials.
    tileInput[thread] = sum0;
    tileInput[TILE + thread] = sum1;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = LANES / 2; offset > 0; offset >>= 1) {
        if (activeRow && stageLane < offset) {
            precise float other0 = tileInput[thread + offset];
            sum0 = sum0 + other0;
            tileInput[thread] = sum0;
            precise float other1 = tileInput[TILE + thread + offset];
            sum1 = sum1 + other1;
            tileInput[TILE + thread] = sum1;
        }
        // Unconditional participation, including padded rows; writers and readers are disjoint
        // within a stage. The next stage cannot read its predecessor's outputs until this barrier.
        GroupMemoryBarrierWithGroupSync();
    }
    if (activeRow && stageLane == 0) {
        uint column = firstOutput + stageOut;
        if ((flags & 4) != 0) {
            precise float b = packed(bias, column, (flags & 8) != 0);
            sum0 = sum0 + b;
            sum1 = sum1 + b;
        }
        if ((flags & 2) != 0) {
            sum0 = bf16_rne(sum0);
            sum1 = bf16_rne(sum1);
        }
        output.Store((rowFirst * outputWidth + column) * 4, asuint(sum0));
        output.Store(((rowFirst + 1) * outputWidth + column) * 4, asuint(sum1));
    }
}
