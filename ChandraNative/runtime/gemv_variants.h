// New ChandraNative code, MPL-2.0. Portable selector and dispatch geometry for the experimental
// batch-row-1 GEMV routes; no D3D or device code. Used by operators.cpp, the model-free calibration
// and the CPU tests. See docs/directcompute-gemv-occupancy.md.
#pragma once
#include <cstdint>
#include <cstring>

namespace chandra::gemv_variants {
// CHANDRA_EXPERIMENTAL_GEMV_B1: unset or "0" is the predecessor (linear.hlsl for every batch);
// each other route replaces only batch-row-1 calls. Matching is exact and case-sensitive.
enum class Route { predecessor, ordered, ordered32, ordered64, parallel32, invalid };

struct Shape {
    Route route; const char* selector; const char* shader;
    uint32_t outputsPerGroup, threads, tile, stride;
    constexpr uint32_t groups(uint32_t outputs) const { return (outputs + outputsPerGroup - 1) / outputsPerGroup; }
    constexpr uint32_t tiles(uint32_t inputWidth) const { return (inputWidth + tile - 1) / tile; }
    // Padded FP32 weight rows plus the FP32 input tile.
    constexpr uint32_t groupsharedBytes() const { return (outputsPerGroup * stride + tile) * 4; }
};
// "ordered" is the immutable linear_gemv.hlsl of commit 0935926. ordered32/ordered64 keep its
// arithmetic and differ in owners per group, K tile and row-major staging (staging:: below).
inline constexpr Shape shapes[] = {
    {Route::ordered, "ordered", "runtime/linear_gemv.hlsl", 8, 256, 512, 513},
    {Route::ordered32, "ordered32", "runtime/linear_gemv_ordered32.hlsl", 32, 256, 128, 129},
    {Route::ordered64, "ordered64", "runtime/linear_gemv_ordered64.hlsl", 64, 256, 64, 65},
    // Opt-in reassociation: the ordered8 staging/geometry is unchanged, but all 32 lanes per row
    // accumulate disjoint ascending-K subsequences and reduce through a fixed five-stage tree.
    {Route::parallel32, "parallel32", "runtime/linear_gemv_parallel32.hlsl", 8, 256, 512, 513},
};
constexpr uint32_t shapeCount = sizeof(shapes) / sizeof(shapes[0]);

inline Route parse(const char* value) {
    if (!value || std::strcmp(value, "0") == 0) return Route::predecessor;
    for (const Shape& s : shapes) if (std::strcmp(value, s.selector) == 0) return s.route;
    return Route::invalid;
}
// Null for the predecessor and invalid routes.
inline const Shape* shape(Route route) {
    for (const Shape& s : shapes) if (s.route == route) return &s;
    return nullptr;
}
inline const Shape* shape(const char* selector) { return selector ? shape(parse(selector)) : nullptr; }

// Index algebra of the row-major variants (ordered32/ordered64), stated once for the CPU model and
// tests; operators.cpp does not use it and the HLSL restates it. Per K tile [begin, begin + n),
// slot i < slots of a thread holds word f % width of tile row f / width, f = thread + i * threads,
// where width is tile / 2 packed BF16 words or tile FP32 words. Slot `slots` holds packed-BF16 word
// tile / 2 of row `thread`, present only for odd start parity on a full tile. Rows at or beyond
// outputsPerGroup and rows beyond the shard have count 0, so they never load or store.
namespace staging {
constexpr uint32_t slots = 16; // FP32: outputsPerGroup * tile / threads; packed BF16 uses half.
constexpr uint32_t tileShift(uint32_t tile) { return tile == 128 ? 7 : tile == 64 ? 6 : 0; }
constexpr bool rowMajor(const Shape& s) {
    return s.route != Route::ordered && s.threads == 256 && s.stride == s.tile + 1 && tileShift(s.tile) &&
           s.tile <= s.threads && s.outputsPerGroup * s.tile == slots * s.threads && s.outputsPerGroup <= s.threads;
}
struct Slot { uint32_t row, word, parity, base, count; };
// Mirrors stage() in linear_gemv_ordered32.hlsl and linear_gemv_ordered64.hlsl with uint32 wrap.
inline Slot stage(const Shape& s, uint32_t inputWidth, uint32_t shardRows, uint32_t weightRowFirst,
                  uint32_t groupFirst, uint32_t thread, uint32_t i, uint32_t begin, uint32_t n, bool bf16) {
    Slot result{};
    const uint32_t shift = bf16 ? tileShift(s.tile) - 1 : tileShift(s.tile);
    const uint32_t f = thread + i * s.threads;
    result.row = i < slots ? f >> shift : (bf16 && thread < s.outputsPerGroup ? thread : s.outputsPerGroup);
    result.word = i < slots ? (f & ((1u << shift) - 1)) : s.tile / 2;
    const uint32_t stageOut = groupFirst + result.row;
    const uint32_t first = (weightRowFirst + stageOut) * inputWidth + begin;
    result.parity = bf16 ? (first & 1) : 0;
    result.base = bf16 ? (first >> 1) : first;
    result.count = result.row < s.outputsPerGroup && stageOut < shardRows ? (bf16 ? ((result.parity + n + 1) >> 1) : n) : 0;
    return result;
}
} // namespace staging

static_assert(shapes[0].route == Route::ordered && shapes[0].groupsharedBytes() == 18464, "ordered8 geometry is immutable");
static_assert(shapes[3].groupsharedBytes() == shapes[0].groupsharedBytes() &&
              shapes[3].outputsPerGroup == shapes[0].outputsPerGroup && shapes[3].threads == shapes[0].threads &&
              shapes[3].tile == shapes[0].tile && shapes[3].stride == shapes[0].stride,
              "parallel32 reuses ordered8 geometry and retired input-tile reduction storage");
static_assert(staging::rowMajor(shapes[1]) && staging::rowMajor(shapes[2]), "row-major variant geometry");
static_assert(shapes[1].groupsharedBytes() == 17024 && shapes[2].groupsharedBytes() == 16896, "variant groupshared bytes");
static_assert(shapes[0].groupsharedBytes() < 32768 && shapes[1].groupsharedBytes() < 32768 && shapes[2].groupsharedBytes() < 32768,
              "cs_5_0 groupshared limit");
} // namespace chandra::gemv_variants
