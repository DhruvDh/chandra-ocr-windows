// New code, MPL-2.0. Experimental opt-in batch-row-1 projection (CHANDRA_EXPERIMENTAL_GEMV_B1=ordered64).
// Arithmetic is linear_gemv.hlsl's: one FP32 sum from +0, then sum = sum + a[k]*w[k] in ascending k,
// then bias, then optional BF16 RNE. Only the schedule differs: ROWS owner lanes per group, a TILE-element
// K tile, and row-major staging in which consecutive threads load consecutive words of one weight row.
// No FP16 arithmetic, split-K, reassociation, transposition or prepacking. docs/directcompute-gemv-occupancy.md
ByteAddressBuffer input : register(t0);
ByteAddressBuffer weight : register(t1);
ByteAddressBuffer bias : register(t2);
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) {
    uint inputWidth, outputWidth, batchRows, firstOutput;
    uint shardRows, rowFirst, flags, weightRowFirst;
};
// Host contract: batchRows == 1 and rowFirst == 0; only activation row 0 is read and written.
#define ROWS 64
#define THREADS 256
#define TILE 64
#define TILE_SHIFT 6
#define STRIDE 65
// Row-major word slots per thread: FP32 ROWS*TILE/THREADS = 16; packed BF16 needs half of them, plus
// slot SLOTS for the extra word of an odd-parity full tile.
#define SLOTS 16
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
struct Slot { uint row, word, parity, base, count; };
// Slot i < SLOTS holds word f % width of tile row f / width, f = thread + i * THREADS, where width is
// TILE / 2 packed BF16 words or TILE FP32 words. Slot SLOTS holds packed-BF16 word TILE / 2 of row
// `thread`. Rows at or beyond ROWS, or beyond the shard, have count 0. Host validation keeps every
// touched word below the shard's uint32 byte range.
Slot stage(uint groupFirst, uint thread, uint i, uint begin, uint n, bool bf16) {
    Slot s;
    uint shift = bf16 ? TILE_SHIFT - 1 : TILE_SHIFT;
    uint f = thread + i * THREADS;
    s.row = i < SLOTS ? f >> shift : (bf16 && thread < ROWS ? thread : ROWS);
    s.word = i < SLOTS ? (f & ((1u << shift) - 1)) : TILE / 2;
    uint stageOut = groupFirst + s.row;
    uint first = (weightRowFirst + stageOut) * inputWidth + begin;
    s.parity = bf16 ? (first & 1) : 0;
    s.base = bf16 ? (first >> 1) : first;
    s.count = s.row < ROWS && stageOut < shardRows ? (bf16 ? ((s.parity + n + 1) >> 1) : n) : 0;
    return s;
}
[numthreads(THREADS,1,1)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
    uint thread = lane.x;
    uint groupFirst = group.x * ROWS;
    uint localOut = groupFirst + thread;
    bool owner = thread < ROWS && localOut < shardRows;
    bool bf16 = (flags & 1) != 0;
    precise float sum = 0.0;
    // Every thread runs the same trip count and reaches both barriers of every tile.
    for (uint begin = 0; begin < inputWidth; begin += TILE) {
        uint n = min(TILE, inputWidth - begin);
        // Issue all global loads before any groupshared store so they can be in flight together.
        uint staged[SLOTS + 1];
        [unroll] for (uint i = 0; i <= SLOTS; ++i) {
            Slot s = stage(groupFirst, thread, i, begin, n, bf16);
            staged[i] = 0;
            if (s.word < s.count) staged[i] = weight.Load((s.base + s.word) * 4);
        }
        float a = 0.0;
        if (thread < n) a = asfloat(input.Load((begin + thread) * 4));
        [unroll] for (uint m = 0; m <= SLOTS; ++m) {
            Slot s = stage(groupFirst, thread, m, begin, n, bf16);
            if (s.word < s.count) {
                uint row = s.row * STRIDE;
                if (!bf16) {
                    tileWeight[row + s.word] = asfloat(staged[m]);
                } else {
                    // Word j holds tile positions 2j-parity (low half) and 2j+1-parity (high half).
                    if (2 * s.word >= s.parity) tileWeight[row + 2 * s.word - s.parity] = asfloat(staged[m] << 16);
                    if (2 * s.word + 1 - s.parity < n) tileWeight[row + 2 * s.word + 1 - s.parity] = asfloat(staged[m] & 0xffff0000);
                }
            }
        }
        if (thread < n) tileInput[thread] = a;
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
