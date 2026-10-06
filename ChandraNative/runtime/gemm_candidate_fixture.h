// New ChandraNative code, MPL-2.0. Portable plan, public fixtures, exact integer oracle and
// admission rules for the model-free GEMM candidate calibration. Windows entry:
// gemm_candidate_calibration.cpp. CPU checks: gemm_candidate_fixture_test.cpp. Nothing here
// creates a device, compiles or executes HLSL or reads model assets; run() drives a Device it is
// given. See docs/directcompute-gemm-candidates.md.
#pragma once
#include "api.h"
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace chandra::gemm_candidates {
using Json = nlohmann::json;

constexpr const char* schema = "private.chandra.directcompute.gemm-candidate-calibration.v1";
struct Kernel { const char* shader; const char* file; uint32_t tileK, rowStride; bool candidate; };
// Index 0 is the unchanged production predecessor; candidates must match it and the oracle.
constexpr std::array<Kernel, 3> kernels{{
    {"runtime/linear.hlsl", "linear.hlsl", 32, 32, false},
    {"runtime/linear_gemm_padded32.hlsl", "linear_gemm_padded32.hlsl", 32, 33, true},
    {"runtime/linear_gemm_padded64.hlsl", "linear_gemm_padded64.hlsl", 64, 65, true}}};
// operators.cpp bounds one linear.hlsl dispatch to <=32 rows, <=1024 outputs and K <= 9216.
constexpr uint32_t maximumInputWidth = 9216, maximumBatchRows = 32, maximumDispatchOutputs = 1024;
constexpr uint64_t maximumDispatchTerms = uint64_t(maximumBatchRows) * maximumDispatchOutputs * maximumInputWidth;
constexpr double dispatchLimitMilliseconds = 100.0, admissionSafetyFactor = 2.0;
constexpr uint32_t deadlineSeconds = 120, timingRounds = 3;
constexpr uint64_t maximumBufferBytes = 128ull << 20;  // device.cpp per-buffer limit
constexpr uint64_t deviceCaseCapBytes = 128ull << 20;  // one case's tracked buffers plus its readback staging copy
constexpr uint64_t jobMemoryCapBytes = 1ull << 30;     // the external Job memory limit must be set and at most this
constexpr uint64_t receiptCapBytes = 8ull << 20;
constexpr size_t reportedMismatches = 16;
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24, "IEEE binary32 required");
#if defined(FLT_EVAL_METHOD)
static_assert(FLT_EVAL_METHOD == 0, "Each binary32 CPU cross-check operation must round to binary32");
#endif

inline void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }  // no per-call allocation
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
struct Deadline {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    void check() const {
        require(std::chrono::steady_clock::now() - start < std::chrono::seconds(deadlineSeconds),
                "120-second source deadline exceeded; retire the externally owned Job");
    }
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};
inline uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result; }
inline float fromBits(uint32_t value) { float result; std::memcpy(&result, &value, sizeof(result)); return result; }
// The shaders' bf16_rne bit formula; the oracle derives BF16 independently (bf16BitsOfUnits).
inline uint32_t roundedBF16Bits(uint32_t value) {
    if ((value & 0x7fffffffu) > 0x7f800000u) return (value & 0xffff0000u) | 0x00400000u;
    return (value + 0x7fffu + ((value >> 16) & 1u)) & 0xffff0000u;
}
inline uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31);
}
inline uint64_t stream(uint64_t domain, uint64_t a, uint64_t b) { return splitmix(splitmix(domain ^ a) ^ b); }

// ---- Exact dyadic arithmetic. Every oracle quantity is an int64 multiple of 2^-42. ----
constexpr int unitExponent = -42;
constexpr uint64_t magnitudeLimit = 1ull << 61;
inline int bitLength(uint64_t value) {
#if defined(_MSC_VER)
    unsigned long index = 0; return _BitScanReverse64(&index, value) ? int(index) + 1 : 0;
#elif defined(__GNUC__)
    return value ? 64 - __builtin_clzll(value) : 0;
#else
    int length = 0; while (value) { ++length; value >>= 1; } return length;
#endif
}
inline uint64_t magnitude(int64_t value) { return value < 0 ? uint64_t(0) - uint64_t(value) : uint64_t(value); }
inline int64_t withSign(bool negative, uint64_t value) { return negative ? -int64_t(value) : int64_t(value); }
// Round to `keep` significant bits, ties to even: binary32 (24) or BF16 (8) RNE of a normal value.
inline int64_t roundSignificant(int64_t value, int keep) {
    const uint64_t m = magnitude(value);
    const int length = bitLength(m);
    if (length <= keep) return value;
    const int drop = length - keep;
    uint64_t kept = m >> drop;
    const uint64_t rest = m & ((1ull << drop) - 1), half = 1ull << (drop - 1);
    if (rest > half || (rest == half && (kept & 1u))) ++kept;
    return withSign(value < 0, kept << drop);
}
// Exact binary32 storage bits of units * 2^-42; the value must already be binary32 and normal.
// Zero is +0: an ascending sum from +0 can only produce -0 if every addend were -0 and it began at -0.
inline uint32_t fp32BitsOfUnits(int64_t units) {
    if (!units) return 0;
    const uint64_t m = magnitude(units);
    const int length = bitLength(m);
    require(m < magnitudeLimit && roundSignificant(units, 24) == units, "Oracle value is not an exact binary32 value");
    const int biased = length - 1 + unitExponent + 127;
    require(biased >= 1 && biased <= 254, "Oracle value outside the normal binary32 range");
    const uint64_t significand = length > 24 ? m >> (length - 24) : m << (24 - length);
    return (units < 0 ? 0x80000000u : 0u) | (uint32_t(biased) << 23) | uint32_t(significand & 0x7fffffu);
}
inline uint32_t bf16BitsOfUnits(int64_t units) { return fp32BitsOfUnits(roundSignificant(units, 8)); }

// ---- Public generators. Every operand is integer * 2^unit, a pure function of its indices. ----
enum class Generator { dyadic, orderFp32Input, orderBf16Input };
enum class BiasKind { none, bf16, fp32 };
struct Units { int input, weight, bias; };
struct Magnitudes { uint64_t input, weight, bias; };
inline const char* generatorName(Generator g) {
    return g == Generator::dyadic ? "splitmix_dyadic_v1" : g == Generator::orderFp32Input ? "order_fp32_input_bf16_weight_v1"
                                                                                         : "order_bf16_input_fp32_weight_v1";
}
inline const char* biasName(BiasKind b) { return b == BiasKind::none ? "none" : b == BiasKind::bf16 ? "bf16" : "fp32"; }
inline Units units(Generator g) {
    if (g == Generator::dyadic) return {-5, -5, -5};
    if (g == Generator::orderFp32Input) return {-27, -15, -27};
    return {-11, -31, -27};
}
// Exclusive integer magnitude bounds, used by the static overflow/exactness analysis.
inline Magnitudes magnitudes(Generator g) {
    if (g == Generator::dyadic) return {17, 17, 17};
    if (g == Generator::orderFp32Input) return {1ull << 29, 1ull << 15, 1ull << 33};
    return {1ull << 13, 1ull << 31, 1ull << 33};
}
constexpr uint64_t inputDomain = 0x676d6d2d696e7075ull, weightDomain = 0x676d6d2d77656967ull, biasDomain = 0x676d6d2d62696173ull;
// splitmix_dyadic_v1: integers in [-16,16] over 32. Hashing, unlike a mod-33 linear form that
// repeats every 33 rows or outputs, makes a wrong row, column or k read a different value except
// by 1-in-33 per-element chance, so whole-dot addressing errors cannot cancel out.
inline int64_t dyadicInteger(uint64_t r) { return int64_t(r % 33) - 16; }
// order generators: 1/64 exact zeros (bits 0-5); otherwise a full significand of the stated
// width (bits 6-28), exponent (bits 32-39) and sign (bit 63). 24 x 8 significant bits make
// 32-bit products, so binary32 products round as well as sums.
inline int64_t significandOperand(uint64_t r, int width, uint32_t exponents) {
    if ((r & 63u) == 0) return 0;
    const uint64_t significand = (1ull << (width - 1)) | ((r >> 6) & ((1ull << (width - 1)) - 1));
    return withSign((r >> 63) != 0, significand << (((r >> 32) & 0xffu) % exponents));
}
inline int64_t inputInteger(Generator g, uint32_t row, uint32_t k) {
    const uint64_t r = stream(inputDomain + uint64_t(g), row, k);
    if (g == Generator::dyadic) return dyadicInteger(r);
    return significandOperand(r, g == Generator::orderFp32Input ? 24 : 8, 6);    // [2^-4, 2^2)
}
inline int64_t weightInteger(Generator g, uint32_t output, uint32_t k) {
    const uint64_t r = stream(weightDomain + uint64_t(g), output, k);
    if (g == Generator::dyadic) return dyadicInteger(r);
    return significandOperand(r, g == Generator::orderFp32Input ? 8 : 24, 8);    // [2^-8, 1)
}
inline int64_t biasInteger(Generator g, BiasKind b, uint32_t output) {
    const uint64_t r = stream(biasDomain + uint64_t(g), output, uint64_t(b));
    if (g == Generator::dyadic) return dyadicInteger(r);
    if ((r & 63u) == 0) return 0;
    if (b == BiasKind::fp32) return significandOperand(r | 1u, 24, 10);       // m * 2^(e-27): [2^-4, 2^6)
    return significandOperand(r | 1u, 8, 10) * (int64_t(1) << 16);          // m * 2^(e-11), in 2^-27 units
}
inline int64_t toUnits(int64_t integer, int unit) {
    const int shift = unit - unitExponent;
    require(shift >= 0 && shift < 62 && bitLength(magnitude(integer)) + shift <= 61, "Operand exceeds the oracle unit range");
    return integer * (int64_t(1) << shift);
}
// Storage bits; BF16 operands must have zero low halves and every operand is zero or normal.
inline uint32_t storageBits(int64_t integer, int unit, bool bf16) {
    const uint32_t value = fp32BitsOfUnits(toUnits(integer, unit));
    require(!bf16 || (value & 0xffffu) == 0, "Generated BF16 operand is not exactly BF16");
    return value;
}

// ---- Cases. Geometry follows operators.cpp: firstOutput = shardFirstRow + weightRowFirst. ----
struct CaseSpec {
    const char* name; Generator generator; bool bf16Weights; BiasKind bias; uint32_t inputWidth;
    uint32_t totalRows, rowFirst, batchRows;                     // input/output buffer rows; dispatched rows
    uint32_t outputWidth, shardFirstRow, shardTotalRows, weightRowFirst, shardRows;
    uint32_t guardWords; bool timingShape;
    uint32_t firstOutput() const { return shardFirstRow + weightRowFirst; }
};
inline std::vector<CaseSpec> plan() {
    using G = Generator; using B = BiasKind;
    return {
        {"k1_single_term_b1_offsets", G::dyadic, true, B::bf16, 1, 3, 2, 1, 7, 1, 5, 2, 3, 3, false},
        {"k37_odd_bf16_ragged_offsets_bf16_bias", G::dyadic, true, B::bf16, 37, 23, 3, 19, 61, 5, 55, 7, 45, 5, false},
        {"k67_odd_fp32_weights_fp32_bias", G::dyadic, false, B::fp32, 67, 9, 1, 7, 44, 9, 33, 8, 23, 4, false},
        {"k96_k64_tail_bf16_fp32_bias", G::dyadic, true, B::fp32, 96, 32, 0, 32, 50, 0, 50, 16, 33, 2, false},
        {"k1001_odd_second_row_chunk", G::dyadic, true, B::none, 1001, 64, 32, 31, 1030, 24, 1003, 1, 1000, 7, false},
        {"order_k2560_fp32_input_bf16_weight_bf16_bias", G::orderFp32Input, true, B::bf16, 2560, 40, 5, 32, 400, 64, 320, 3, 300, 3, false},
        {"order_k4096_bf16_input_fp32_weight_fp32_bias", G::orderBf16Input, false, B::fp32, 4096, 20, 2, 16, 1545, 512, 1031, 5, 1024, 1, false},
        {"b1_k9216_n1024", G::dyadic, true, B::none, 9216, 2, 1, 1, 1024, 0, 1024, 0, 1024, 1, false},
        {"k9216_n1024_r16", G::dyadic, true, B::none, 9216, 17, 1, 16, 1027, 0, 1025, 1, 1024, 1, true},
        {"k9216_n1024_r32_second_chunks", G::dyadic, true, B::none, 9216, 64, 32, 32, 2560, 0, 2560, 1024, 1024, 1, true},
        {"order_k9216_n1024_r32_bf16_bias", G::orderFp32Input, true, B::bf16, 9216, 32, 0, 32, 1024, 0, 1024, 0, 1024, 1, false}};
}
struct LinearParams {
    uint32_t inputWidth, outputWidth, batchRows, firstOutput;
    uint32_t shardRows, rowFirst, flags, weightRowFirst;
};
static_assert(sizeof(LinearParams) == 32, "linear.hlsl cbuffer ABI");
inline LinearParams params(const CaseSpec& s, bool roundOutputBF16) {
    const uint32_t flags = uint32_t(s.bf16Weights) | (uint32_t(roundOutputBF16) << 1) |
                           (uint32_t(s.bias != BiasKind::none) << 2) | (uint32_t(s.bias == BiasKind::bf16) << 3);
    return {s.inputWidth, s.outputWidth, s.batchRows, s.firstOutput(), s.shardRows, s.rowFirst, flags, s.weightRowFirst};
}
inline std::array<uint32_t, 8> paramWords(const LinearParams& p) {
    return {p.inputWidth, p.outputWidth, p.batchRows, p.firstOutput, p.shardRows, p.rowFirst, p.flags, p.weightRowFirst};
}
inline std::array<uint32_t, 3> groups(const CaseSpec& s) { return {(s.shardRows + 15) / 16, (s.batchRows + 15) / 16, 1}; }
// Cost proxy: every group runs 256 lanes through all K whether or not its rows/columns are ragged.
inline uint64_t work(const CaseSpec& s) { auto g = groups(s); return uint64_t(g[0]) * g[1] * s.inputWidth; }
struct Words { uint64_t input, weight, bias, output; };
inline Words words(const CaseSpec& s) {
    const uint64_t weightElements = uint64_t(s.shardTotalRows) * s.inputWidth;
    return {uint64_t(s.totalRows) * s.inputWidth, s.bf16Weights ? (weightElements + 1) / 2 : weightElements,
            s.bias == BiasKind::none ? 0 : s.bias == BiasKind::bf16 ? (uint64_t(s.outputWidth) + 1) / 2 : s.outputWidth,
            uint64_t(s.totalRows) * s.outputWidth + s.guardWords};
}
inline uint64_t trackedBytes(const CaseSpec& s) { auto w = words(s); return 4 * (w.input + w.weight + w.bias + w.output); }

// Refuses before any allocation or dispatch; returns the derived bounds that justify admission.
inline Json admit(const CaseSpec& s) {
    const Units u = units(s.generator);
    const Magnitudes m = magnitudes(s.generator);
    require(s.inputWidth >= 1 && s.inputWidth <= maximumInputWidth, "Input width outside 1..9216");
    require(s.batchRows >= 1 && s.batchRows <= maximumBatchRows, "Batch rows outside 1..32");
    require(s.shardRows >= 1 && s.shardRows <= maximumDispatchOutputs, "Dispatch outputs outside 1..1024");
    require(uint64_t(s.rowFirst) + s.batchRows <= s.totalRows, "Dispatched rows exceed the activation rows");
    require(uint64_t(s.weightRowFirst) + s.shardRows <= s.shardTotalRows, "Dispatched outputs exceed the weight shard");
    require(uint64_t(s.shardFirstRow) + s.shardTotalRows <= s.outputWidth, "Weight shard exceeds the output width");
    require(uint64_t(s.firstOutput()) + s.shardRows <= s.outputWidth, "Dispatched columns exceed the output width");
    require(s.generator == Generator::dyadic || s.bf16Weights == (s.generator == Generator::orderFp32Input),
            "Order generator weight significand differs from the weight storage type");
    const Words w = words(s);
    for (uint64_t count : {w.input, w.weight, w.output})
        require(count >= 1 && count * 4 <= maximumBufferBytes, "Buffer outside one bounded raw D3D11 buffer");
    require(w.bias * 4 <= maximumBufferBytes, "Bias buffer exceeds one bounded raw D3D11 buffer");
    const uint64_t forecast = trackedBytes(s), staging = 4 * w.output;
    require(forecast + staging <= deviceCaseCapBytes, "Case tracked bytes plus readback staging exceed the device cap");
    const auto g = groups(s);
    require(g[0] <= 65535 && g[1] <= 65535, "Dispatch groups exceed the SM5 limit");
    const uint64_t terms = uint64_t(s.batchRows) * s.shardRows * s.inputWidth;
    require(terms <= maximumDispatchTerms, "Dispatch exceeds the production per-call term bound");
    // Exactness: products are integer * 2^(input+weight) and must fit the 2^-42 grid.
    const int productShift = u.input + u.weight - unitExponent;
    require(productShift >= 0 && u.bias >= unitExponent, "Generator units are finer than the oracle grid");
    const int productBits = bitLength(m.input - 1) + bitLength(m.weight - 1) + productShift;
    const int biasBits = s.bias == BiasKind::none ? 0 : bitLength(m.bias - 1) + (u.bias - unitExponent);
    require(productBits <= 60 && biasBits <= 60, "A product or bias could exceed the int64 oracle grid");
    // Each binary32 rounding grows a magnitude by at most 2^-24 relative, so after K steps every
    // partial sum is below (K * 2^productBits + 2^biasBits) * (1 + 2^-24)^K < 2^61.
    const long double growth = std::pow(1.0L + std::ldexp(1.0L, -24), (long double)s.inputWidth);
    const long double bound = (std::ldexp((long double)s.inputWidth, productBits) + (biasBits ? std::ldexp(1.0L, biasBits) : 0.0L)) * growth;
    require(bound < std::ldexp(1.0L, 61), "Partial sums could exceed the int64 oracle grid");
    // Every nonzero quantity is a multiple of 2^-42 >= FLT_MIN and below 2^19 < FLT_MAX, so no
    // operand, product or sum is subnormal or overflows and D3D11 denormal flushing cannot apply.
    const bool orderInsensitive = s.generator == Generator::dyadic &&
        uint64_t(s.inputWidth) * 256 + (s.bias == BiasKind::none ? 0 : 512) < (1ull << 24);
    require(s.generator != Generator::dyadic || orderInsensitive, "Dyadic case is not exact in binary32 for every order");
    return {{"groups", g}, {"work_group_k", work(s)}, {"terms", terms},
            {"raw_params", paramWords(params(s, false))}, {"rounded_params", paramWords(params(s, true))},
            {"words", {{"input", w.input}, {"weight", w.weight}, {"bias", w.bias}, {"output", w.output}}},
            {"tracked_bytes", forecast}, {"readback_staging_bytes", staging},
            {"oracle_product_bits_bound", productBits}, {"oracle_bias_bits_bound", biasBits},
            {"oracle_partial_sum_bound_log2", double(std::log2(bound))}, {"oracle_grid_log2", unitExponent},
            {"exact_for_every_summation_order", orderInsensitive}};
}

// ---- Uploaded operand words (input rows are absolute; shard row r is logical row shardFirstRow + r). ----
inline std::vector<uint32_t> inputWords(const CaseSpec& s) {
    const Units u = units(s.generator);
    std::vector<uint32_t> result(size_t(s.totalRows) * s.inputWidth);
    for (uint32_t row = 0; row < s.totalRows; ++row)
        for (uint32_t k = 0; k < s.inputWidth; ++k)
            result[size_t(row) * s.inputWidth + k] = storageBits(inputInteger(s.generator, row, k), u.input, s.generator != Generator::orderFp32Input);
    return result;
}
inline std::vector<uint32_t> weightWords(const CaseSpec& s, const Deadline& deadline) {
    const Units u = units(s.generator);
    std::vector<uint32_t> result(size_t(words(s).weight), 0);
    for (uint32_t r = 0; r < s.shardTotalRows; ++r) {
        if (!(r % 64)) deadline.check();
        for (uint32_t k = 0; k < s.inputWidth; ++k) {
            const uint32_t value = storageBits(weightInteger(s.generator, s.shardFirstRow + r, k), u.weight, s.bf16Weights);
            const size_t index = size_t(r) * s.inputWidth + k;
            if (s.bf16Weights) result[index / 2] |= (value >> 16) << (16 * (index % 2));
            else result[index] = value;
        }
    }
    return result;
}
inline std::vector<uint32_t> biasWords(const CaseSpec& s) {
    std::vector<uint32_t> result(size_t(words(s).bias), 0);
    for (uint32_t c = 0; c < s.outputWidth && s.bias != BiasKind::none; ++c) {
        const uint32_t value = storageBits(biasInteger(s.generator, s.bias, c), units(s.generator).bias, s.bias == BiasKind::bf16);
        if (s.bias == BiasKind::bf16) result[c / 2] |= (value >> 16) << (16 * (c % 2));
        else result[c] = value;
    }
    return result;
}
// Signalling/quiet NaN words with a per-dispatch salt; every expected output is finite.
inline uint32_t sentinel(uint64_t index, uint32_t salt) {
    return 0x7f800001u | uint32_t(splitmix(0x73656e74696e656cull ^ (uint64_t(salt) << 40) ^ index) & 0x807ffffeu);
}
inline std::vector<uint32_t> sentinelImage(const CaseSpec& s, uint32_t salt) {
    std::vector<uint32_t> result(size_t(words(s).output));
    for (size_t i = 0; i < result.size(); ++i) result[i] = sentinel(i, salt);
    return result;
}
inline bool inRegion(const CaseSpec& s, uint64_t index) {
    if (index >= uint64_t(s.totalRows) * s.outputWidth) return false;
    const uint64_t row = index / s.outputWidth, column = index % s.outputWidth;
    return row >= s.rowFirst && row < uint64_t(s.rowFirst) + s.batchRows &&
           column >= s.firstOutput() && column < uint64_t(s.firstOutput()) + s.shardRows;
}

// ---- Independent oracle: exact integer model of the ascending binary32 sequence. ----
// For each output: s = +0; for k ascending: p = rn24(a*w); s = rn24(s + p); then rn24(s + bias).
// Raw outputs are the binary32 bits of s; rounded outputs are rn8(s) derived on the integer, and
// cross-checked against the bit formula and against a separate CPU binary32 loop.
struct Expectation {
    std::vector<uint32_t> raw, rounded; // [batchRows][shardRows] region, row-major
    Json analysis;
};
inline Expectation oracle(const CaseSpec& s, const Deadline& deadline, uint32_t observationRows) {
    const Units u = units(s.generator);
    const int64_t productScale = int64_t(1) << (u.input + u.weight - unitExponent);
    std::vector<int64_t> a(size_t(s.batchRows) * s.inputWidth), w(s.inputWidth), p(s.inputWidth), exact(s.inputWidth);
    std::vector<float> af(a.size()), wf(s.inputWidth);
    for (uint32_t r = 0; r < s.batchRows; ++r)
        for (uint32_t k = 0; k < s.inputWidth; ++k) {
            const int64_t value = inputInteger(s.generator, s.rowFirst + r, k);
            a[size_t(r) * s.inputWidth + k] = value;
            af[size_t(r) * s.inputWidth + k] = fromBits(storageBits(value, u.input, false));
        }
    Expectation e; e.raw.resize(size_t(s.batchRows) * s.shardRows); e.rounded.resize(e.raw.size());
    uint64_t productRoundings = 0, sumRoundings = 0, tiesDown = 0, tiesUp = 0, bf16Changed = 0, zeros = 0, negative = 0;
    uint64_t observed = 0, fusedDiffers = 0, descendingDiffers = 0, blockedDiffers = 0, exactDiffers = 0;
    int maximumLength = 0;
    for (uint32_t j = 0; j < s.shardRows; ++j) {
        if (!(j % 16)) deadline.check();
        const uint32_t column = s.firstOutput() + j;
        for (uint32_t k = 0; k < s.inputWidth; ++k) {
            w[k] = weightInteger(s.generator, column, k);
            wf[k] = fromBits(storageBits(w[k], u.weight, s.bf16Weights));
        }
        const int64_t bias = s.bias == BiasKind::none ? 0 : toUnits(biasInteger(s.generator, s.bias, column), u.bias);
        const float biasFloat = fromBits(fp32BitsOfUnits(bias));
        for (uint32_t r = 0; r < s.batchRows; ++r) {
            const int64_t* row = &a[size_t(r) * s.inputWidth];
            const float* rowFloat = &af[size_t(r) * s.inputWidth];
            int64_t sum = 0; float sumFloat = 0.0f;
            for (uint32_t k = 0; k < s.inputWidth; ++k) {
                const int64_t unrounded = row[k] * w[k] * productScale, product = roundSignificant(unrounded, 24);
                const int64_t next = sum + product;
                sum = roundSignificant(next, 24);
                require(magnitude(sum) < magnitudeLimit, "Oracle partial sum left the proven range");
                productRoundings += product != unrounded; sumRoundings += sum != next;
                const float productFloat = rowFloat[k] * wf[k];
                sumFloat = sumFloat + productFloat;
            }
            if (s.bias != BiasKind::none) {
                const int64_t next = sum + bias;
                sum = roundSignificant(next, 24); sumRoundings += sum != next;
                sumFloat = sumFloat + biasFloat;
            }
            maximumLength = std::max(maximumLength, bitLength(magnitude(sum)));
            const uint32_t raw = fp32BitsOfUnits(sum), rounded = bf16BitsOfUnits(sum);
            require(bits(sumFloat) == raw, "CPU binary32 ascending loop differs from the integer oracle");
            require(roundedBF16Bits(raw) == rounded, "BF16 bit formula differs from the integer RNE oracle");
            const size_t index = size_t(r) * s.shardRows + j;
            e.raw[index] = raw; e.rounded[index] = rounded;
            zeros += sum == 0; negative += sum < 0; bf16Changed += rounded != raw;
            if ((raw & 0xffffu) == 0x8000u) ++((raw & 0x10000u) ? tiesUp : tiesDown);
            if (r >= observationRows) continue;
            // Alternative orders: fused (unrounded products), descending k, 32-wide split-K blocks
            // summed ascending, and one rounding of the exact dot. Counts only; never expectations.
            ++observed;
            int64_t fused = 0, descending = 0, blocked = 0, exactSum = 0;
            for (uint32_t k = 0; k < s.inputWidth; ++k) { exact[k] = row[k] * w[k] * productScale; p[k] = roundSignificant(exact[k], 24); }
            for (uint32_t k = 0; k < s.inputWidth; ++k) { fused = roundSignificant(fused + exact[k], 24); exactSum += exact[k]; }
            for (uint32_t k = s.inputWidth; k-- > 0;) descending = roundSignificant(descending + p[k], 24);
            for (uint32_t begin = 0; begin < s.inputWidth; begin += 32) {
                int64_t block = 0;
                for (uint32_t k = begin; k < std::min(begin + 32, s.inputWidth); ++k) block = roundSignificant(block + p[k], 24);
                blocked = roundSignificant(blocked + block, 24);
            }
            auto finish = [&](int64_t value) { return s.bias == BiasKind::none ? value : roundSignificant(value + bias, 24); };
            fusedDiffers += fp32BitsOfUnits(finish(fused)) != raw;
            descendingDiffers += fp32BitsOfUnits(finish(descending)) != raw;
            blockedDiffers += fp32BitsOfUnits(finish(blocked)) != raw;
            exactDiffers += fp32BitsOfUnits(roundSignificant(exactSum + bias, 24)) != raw;
        }
    }
    e.analysis = {{"outputs", e.raw.size()}, {"product_rounding_events", productRoundings}, {"sum_rounding_events", sumRoundings},
                  {"bf16_ties_round_down_to_even", tiesDown}, {"bf16_ties_round_up_to_even", tiesUp},
                  {"bf16_changed_outputs", bf16Changed}, {"zero_outputs", zeros}, {"negative_outputs", negative},
                  {"maximum_output_bits_in_2^-42_units", maximumLength},
                  {"cpu_binary32_loop_matches", true}, {"bf16_formula_matches_integer_rne", true},
                  {"alternative_order_observation", {{"outputs", observed}, {"fused_products_differ", fusedDiffers},
                      {"descending_k_differ", descendingDiffers}, {"split_k32_blocks_differ", blockedDiffers},
                      {"single_rounding_of_exact_dot_differ", exactDiffers}}}};
    return e;
}
inline std::vector<uint32_t> expectedImage(const CaseSpec& s, const std::vector<uint32_t>& region, uint32_t salt) {
    require(region.size() == size_t(s.batchRows) * s.shardRows, "Expected region shape differs");
    auto image = sentinelImage(s, salt);
    for (uint32_t r = 0; r < s.batchRows; ++r)
        for (uint32_t j = 0; j < s.shardRows; ++j)
            image[size_t(s.rowFirst + r) * s.outputWidth + s.firstOutput() + j] = region[size_t(r) * s.shardRows + j];
    return image;
}
inline std::vector<uint32_t> regionOf(const CaseSpec& s, const std::vector<uint32_t>& image) {
    require(image.size() == words(s).output, "Output image size differs");
    std::vector<uint32_t> region(size_t(s.batchRows) * s.shardRows);
    for (uint32_t r = 0; r < s.batchRows; ++r)
        for (uint32_t j = 0; j < s.shardRows; ++j)
            region[size_t(r) * s.shardRows + j] = image[size_t(s.rowFirst + r) * s.outputWidth + s.firstOutput() + j];
    return region;
}

// ---- Completed-dispatch checks. ----
// Every word: computed region against the oracle, everything else against this dispatch's sentinel.
inline Json compareImage(const CaseSpec& s, const std::vector<uint32_t>& observed, const std::vector<uint32_t>& expected) {
    require(observed.size() == expected.size() && expected.size() == words(s).output, "Incomplete output readback");
    uint64_t regionMismatches = 0, sentinelMismatches = 0, regionWords = 0;
    Json first = Json::array();
    for (size_t i = 0; i < expected.size(); ++i) {
        const bool region = inRegion(s, i);
        regionWords += region;
        if (observed[i] == expected[i]) continue;
        ++(region ? regionMismatches : sentinelMismatches);
        if (first.size() < reportedMismatches)
            first.push_back({{"index", i}, {"region", region}, {"expected", expected[i]}, {"observed", observed[i]}});
    }
    require(regionWords == uint64_t(s.batchRows) * s.shardRows, "Region accounting differs");
    return {{"words", expected.size()}, {"region_words", regionWords}, {"region_mismatches", regionMismatches},
            {"sentinel_mismatches", sentinelMismatches}, {"first_mismatches", std::move(first)},
            {"passed", regionMismatches == 0 && sentinelMismatches == 0}};
}
// One dispatch per Device profile window: exactly one completed, nondisjoint record of the
// expected shader and groups, finite, nonnegative and strictly below 100 ms.
inline double checkProfile(const Json& profile, const char* shader, const std::array<uint32_t, 3>& expectedGroups) {
    require(profile.is_object() && profile.contains("disjoint") && profile.at("disjoint").is_boolean() &&
            profile.at("disjoint") == false && profile.contains("frequency") && profile.at("frequency").is_number_unsigned() &&
            profile.at("frequency").get<uint64_t>() > 0, "Invalid or disjoint GPU timing window");
    require(profile.contains("dispatches") && profile.at("dispatches").is_array() && profile.at("dispatches").size() == 1,
            "Profile window must hold exactly one completed dispatch");
    const Json& d = profile.at("dispatches").front();
    require(d.is_object() && d.contains("shader") && d.at("shader") == shader, "Unexpected profiled shader");
    require(d.contains("groups") && d.at("groups") == Json(expectedGroups), "Unexpected profiled dispatch groups");
    require(d.contains("gpu_milliseconds") && d.at("gpu_milliseconds").is_number(), "Profiled dispatch duration missing");
    const double ms = d.at("gpu_milliseconds").get<double>();
    require(std::isfinite(ms) && ms >= 0.0, "Nonfinite or negative completed dispatch duration");
    require(ms < dispatchLimitMilliseconds, "A completed dispatch reached 100 ms; stop before advancing");
    return ms;
}
// Before a larger case: each kernel's slowest observed dispatch, scaled by the work ratio and a
// 2x margin, must forecast strictly below 100 ms. Returns the forecasts for the receipt.
inline Json admitNext(const std::array<double, 3>& slowest, uint64_t observedWork, uint64_t nextWork) {
    require(observedWork > 0 && nextWork > 0, "Work proxy must be positive");
    Json forecasts = Json::array();
    for (size_t i = 0; i < kernels.size(); ++i) {
        require(std::isfinite(slowest[i]) && slowest[i] >= 0.0, "Admission needs a completed observation for every kernel");
        const double forecast = admissionSafetyFactor * slowest[i] * double(nextWork) / double(observedWork);
        forecasts.push_back({{"shader", kernels[i].shader}, {"observed_maximum_ms", slowest[i]}, {"forecast_ms", forecast}});
        require(forecast < dispatchLimitMilliseconds, std::string("Forecast for ") + kernels[i].shader + " reaches 100 ms; stop before the larger case");
    }
    return forecasts;
}

// ---- Optional counterbalanced timing schedule, only after every correctness case passed. ----
// Six orders of {production, padded32, padded64}: each kernel holds each position twice per block.
constexpr std::array<std::array<uint32_t, 3>, 6> timingOrders{{{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {0, 2, 1}, {1, 0, 2}}};
struct TimingSlot { uint32_t round, shape, block, position, kernel; };
inline std::vector<TimingSlot> timingSchedule(uint32_t rounds, uint32_t shapes) {
    std::vector<TimingSlot> slots;
    for (uint32_t round = 0; round < rounds; ++round)
        for (uint32_t i = 0; i < shapes; ++i) {
            const uint32_t shape = (i + round) % shapes;  // alternate which shape runs first
            for (uint32_t block = 0; block < timingOrders.size(); ++block) {
                const auto& order = timingOrders[(block + round) % timingOrders.size()];
                for (uint32_t position = 0; position < 3; ++position) slots.push_back({round, shape, block, position, order[position]});
            }
        }
    return slots;
}
inline Json timingSummary(const std::vector<double>& samples) {
    require(!samples.empty(), "No timing samples");
    auto sorted = samples; std::sort(sorted.begin(), sorted.end());
    double total = 0; for (double v : sorted) total += v;
    const size_t n = sorted.size();
    const double median = n % 2 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    return {{"samples", n}, {"minimum_ms", sorted.front()}, {"median_ms", median}, {"mean_ms", total / double(n)}, {"maximum_ms", sorted.back()}};
}

// ---- Strict arguments, identity and external Job admission (portable parts). ----
enum class PlanSelector { correctness, correctnessThenTiming };
struct Arguments { bool execute = false; PlanSelector plan = PlanSelector::correctness; bool planGiven = false;
                   std::wstring shaderRoot, pci, luid, output; };
inline std::string narrow(const std::wstring& text) {
    std::string result;
    for (wchar_t c : text) result.push_back(c > 31 && c < 127 ? char(c) : '?');
    return result;
}
inline bool hexDigits(const std::wstring& text, size_t first, size_t count) {
    for (size_t i = first; i < first + count; ++i)
        if (!((text[i] >= L'0' && text[i] <= L'9') || (text[i] >= L'a' && text[i] <= L'f'))) return false;
    return true;
}
// adapter_identity.h formats "%02x:%02x.%x" (device <= 31, function <= 7) and "%08x:%08x".
inline bool validPci(const std::wstring& t) {
    return t.size() == 7 && hexDigits(t, 0, 2) && t[2] == L':' && hexDigits(t, 3, 2) && t[5] == L'.' && t[6] >= L'0' && t[6] <= L'7' &&
           (t[3] == L'0' || t[3] == L'1');
}
inline bool validLuid(const std::wstring& t) { return t.size() == 17 && hexDigits(t, 0, 8) && t[8] == L':' && hexDigits(t, 9, 8); }
inline Arguments parseArguments(const std::vector<std::wstring>& args) {
    Arguments a; bool seen[5] = {};
    for (size_t i = 0; i < args.size(); ++i) {
        const std::wstring& name = args[i];
        if (name == L"--execute") { require(!a.execute, "Duplicate --execute"); a.execute = true; continue; }
        static const wchar_t* names[5] = {L"--plan", L"--shader-root", L"--pci", L"--luid", L"--output"};
        size_t which = 5;
        for (size_t n = 0; n < 5; ++n) if (name == names[n]) which = n;
        require(which < 5, "Unknown calibration option: " + narrow(name));
        require(!seen[which], "Duplicate calibration option: " + narrow(name));
        require(i + 1 < args.size(), "Option value absent: " + narrow(name));
        const std::wstring& value = args[++i];
        require(!value.empty() && value.rfind(L"--", 0) != 0 && value.find(L'\0') == std::wstring::npos,
                "Option value empty or another option: " + narrow(name));
        seen[which] = true;
        if (which == 0) {
            require(value == L"correctness" || value == L"correctness-then-timing",
                    "--plan must be exactly correctness or correctness-then-timing");
            a.plan = value == L"correctness" ? PlanSelector::correctness : PlanSelector::correctnessThenTiming; a.planGiven = true;
        } else if (which == 1) a.shaderRoot = value;
        else if (which == 2) { require(validPci(value), "--pci must be lowercase bb:dd.f"); a.pci = value; }
        else if (which == 3) { require(validLuid(value), "--luid must be lowercase hhhhhhhh:hhhhhhhh"); a.luid = value; }
        else a.output = value;
    }
    if (a.execute) require(seen[0] && seen[1] && seen[2] && seen[3] && seen[4],
                           "--execute requires --plan, --shader-root, --pci, --luid and --output");
    return a;
}
// Device already selects only Intel A770 adapters by exact PCI/LUID; this re-checks its report.
inline void authenticateIdentity(const Json& identity, const std::string& pci, const std::string& luid) {
    require(identity.is_object() && identity.contains("pci_bdf") && identity.at("pci_bdf") == pci &&
            identity.contains("luid") && identity.at("luid") == luid, "Created device PCI/LUID differs from the request");
    require(identity.contains("vendor_id") && identity.at("vendor_id") == 0x8086u, "Created device is not an Intel adapter");
    require(identity.contains("description") && identity.at("description").is_string() &&
            identity.at("description").get<std::string>().find("A770") != std::string::npos, "Created device is not an Arc A770");
}
// winnt.h JOB_OBJECT_LIMIT_* values; the Windows entry static_asserts equality.
constexpr uint32_t jobLimitProcessTime = 0x2, jobLimitJobTime = 0x4, jobLimitProcessMemory = 0x100,
                   jobLimitJobMemory = 0x200, jobLimitKillOnJobClose = 0x2000;
struct JobLimits { bool inJob = false; uint32_t flags = 0; uint64_t processMemory = 0, jobMemory = 0; };
inline Json admitJob(const JobLimits& j) {
    require(j.inJob, "Run inside the externally owned Job; refusing to execute");
    require((j.flags & jobLimitKillOnJobClose) != 0, "External Job must kill on close");
    const bool process = (j.flags & jobLimitProcessMemory) != 0, job = (j.flags & jobLimitJobMemory) != 0;
    require(process || job, "External Job must bound memory");
    require((!process || (j.processMemory > 0 && j.processMemory <= jobMemoryCapBytes)) &&
            (!job || (j.jobMemory > 0 && j.jobMemory <= jobMemoryCapBytes)), "External Job memory limit exceeds 1 GiB");
    return {{"in_job", true}, {"limit_flags", j.flags}, {"kill_on_close", true},
            {"process_memory_limit", process ? Json(j.processMemory) : Json(nullptr)},
            {"job_memory_limit", job ? Json(j.jobMemory) : Json(nullptr)},
            {"cpu_time_limited", (j.flags & (jobLimitProcessTime | jobLimitJobTime)) != 0}};
}

// ---- Run loop over a caller-owned Device (D3D11 on Windows; a control-flow stand-in in CPU tests). ----
// Hooks keep Win32 out of this header: SHA-256 of raw bytes and a bounded receipt checkpoint.
struct Hooks { std::function<std::string(const void*, size_t)> sha256; std::function<void(const Json&)> checkpoint; };
inline std::string digest(const Hooks& h, const std::vector<uint32_t>& w) { return h.sha256(w.data(), w.size() * 4); }
struct Operands { chandra::dc::Buffer input, weight, bias, output; Json hashes; };
inline Operands upload(chandra::dc::Device& device, const CaseSpec& s, const Hooks& hooks, const Deadline& deadline) {
    Operands o;
    const auto in = inputWords(s), wt = weightWords(s, deadline), bs = biasWords(s), first = sentinelImage(s, 0);
    o.hashes = {{"input_words", in.size()}, {"input_sha256", digest(hooks, in)}, {"weight_words", wt.size()},
                {"weight_sha256", digest(hooks, wt)}, {"bias_words", bs.size()}, {"bias_sha256", bs.empty() ? Json(nullptr) : Json(digest(hooks, bs))}};
    o.input = device.words(uint32_t(in.size()), in.data());
    o.weight = device.words(uint32_t(wt.size()), wt.data());
    o.weight.packedBF16 = s.bf16Weights; o.weight.logicalElements = uint64_t(s.shardTotalRows) * s.inputWidth;
    if (s.bias != BiasKind::none) {
        o.bias = device.words(uint32_t(bs.size()), bs.data());
        o.bias.packedBF16 = s.bias == BiasKind::bf16; o.bias.logicalElements = s.outputWidth;
    }
    o.output = device.words(uint32_t(first.size()), first.data());
    deadline.check();
    return o;
}
struct Ledger { uint32_t salt = 0; uint64_t dispatches = 0; };
// One dispatch: fresh salted sentinels, its own profile window, then a complete readback compared
// with the oracle image and (candidates) the predecessor's region. `record` already sits in the
// receipt tree, so a stop keeps everything observed so far. Returns whether it passed; no rerun.
inline bool dispatchOnce(chandra::dc::Device& device, const CaseSpec& s, Operands& o, size_t kernel, bool rounded,
                         const std::vector<uint32_t>& region, const std::vector<uint32_t>* predecessor,
                         std::vector<uint32_t>* observedRegion, Ledger& ledger, const Hooks& hooks, const Deadline& deadline,
                         double& milliseconds, Json& record) {
    deadline.check();
    const uint32_t salt = ++ledger.salt;
    const auto sentinels = sentinelImage(s, salt), expected = expectedImage(s, region, salt);
    const LinearParams p = params(s, rounded);
    const auto g = groups(s);
    record["kernel"] = kernels[kernel].shader; record["call"] = rounded ? "bf16_rounded" : "fp32_raw";
    record["params"] = paramWords(p); record["groups"] = g; record["salt"] = salt;
    record["expected_image_sha256"] = digest(hooks, expected); record["passed"] = false; record["submitted"] = false;
    device.upload(o.output, sentinels.data(), uint32_t(sentinels.size() * 4));
    device.beginProfile();
    const chandra::dc::Buffer* bias = s.bias == BiasKind::none ? &o.weight : &o.bias;  // unread without flag 4, as in operators.cpp
    device.dispatch(kernels[kernel].shader, {&o.input, &o.weight, bias}, {&o.output}, &p, uint32_t(sizeof(p)), g[0], g[1], g[2]);
    ++ledger.dispatches; record["submitted"] = true;
    const Json profile = Json::parse(device.finishProfile());
    record["gpu_profile"] = profile;
    milliseconds = checkProfile(profile, kernels[kernel].shader, g);
    record["gpu_milliseconds"] = milliseconds;
    const auto observed = device.readWords(o.output);
    record["observed_image_sha256"] = digest(hooks, observed);
    record["comparison"] = compareImage(s, observed, expected);
    const auto mine = regionOf(s, observed);
    record["observed_region_sha256"] = digest(hooks, mine);
    if (predecessor) record["matches_predecessor_region"] = mine == *predecessor;
    if (observedRegion) *observedRegion = mine;
    const bool passed = record.at("comparison").at("passed") == true && (!predecessor || mine == *predecessor);
    record["passed"] = passed;
    return passed;
}
// Oracle first, then two calls (FP32 raw, BF16 rounded), each production -> padded32 -> padded64.
inline void runCase(chandra::dc::Device& device, const CaseSpec& s, Json& phase, Ledger& ledger, const Hooks& hooks,
                    const Deadline& deadline, std::array<double, 3>& slowest, std::vector<uint32_t>& retainedRounded) {
    const double started = deadline.seconds();
    phase["admission"] = admit(s);
    const auto expectation = oracle(s, deadline, std::min<uint32_t>(s.batchRows, 2));
    phase["oracle"] = expectation.analysis;
    phase["expected_raw_region_sha256"] = digest(hooks, expectation.raw);
    phase["expected_rounded_region_sha256"] = digest(hooks, expectation.rounded);
    phase["oracle_seconds"] = deadline.seconds() - started;
    slowest = {0.0, 0.0, 0.0};
    {
        Operands o = upload(device, s, hooks, deadline);
        phase["operands"] = o.hashes;
        require(device.trackedBufferBytes() == trackedBytes(s), "Tracked device bytes differ from the case forecast");
        phase["memory_before"] = Json::parse(device.memoryJson());
        phase["dispatches"] = Json::array();
        for (bool rounded : {false, true}) {
            const auto& region = rounded ? expectation.rounded : expectation.raw;
            std::vector<uint32_t> predecessor;
            for (size_t kernel = 0; kernel < kernels.size(); ++kernel) {
                double ms = 0;
                phase["dispatches"].push_back(Json::object());
                const bool passed = dispatchOnce(device, s, o, kernel, rounded, region, kernel ? &predecessor : nullptr,
                                                 kernel ? nullptr : &predecessor, ledger, hooks, deadline, ms, phase["dispatches"].back());
                require(passed, std::string(kernels[kernel].shader) + " output differs from the oracle, its sentinels or the predecessor; stop");
                slowest[kernel] = std::max(slowest[kernel], ms);
            }
            if (rounded) retainedRounded = predecessor;
        }
        phase["memory_after"] = Json::parse(device.memoryJson());
    }
    device.drain(); deadline.check();
    require(device.trackedBufferBytes() == 0, "Completed case buffers were not released");
    phase["slowest_gpu_milliseconds"] = slowest;
    phase["elapsed_seconds"] = deadline.seconds() - started;
    phase["passed"] = true;
}
struct TimingShape { const CaseSpec* spec; std::array<double, 3> slowest; std::vector<uint32_t> rounded; Json hashes; };
// Optional: only after every correctness case passed. Reuses the retained oracle/predecessor
// regions, regenerates the same operands (hash-checked) and checks every timed dispatch in full.
inline void timing(chandra::dc::Device& device, Json& report, Ledger& ledger, const Hooks& hooks, const Deadline& deadline,
                   const std::vector<TimingShape>& shapes) {
    Json& t = report["timing"];
    require(!shapes.empty(), "No timing shapes in the plan");
    uint64_t total = 0; for (const auto& shape : shapes) total += trackedBytes(*shape.spec) + 4 * words(*shape.spec).output;
    require(total <= deviceCaseCapBytes, "Timing shapes exceed the device cap together");
    for (const auto& shape : shapes) t["admission"].push_back(admitNext(shape.slowest, 1, 1));
    std::vector<Operands> operands; uint64_t tracked = 0;
    for (const auto& shape : shapes) {
        operands.push_back(upload(device, *shape.spec, hooks, deadline)); tracked += trackedBytes(*shape.spec);
        require(operands.back().hashes == shape.hashes, "Regenerated timing operands differ from the correctness operands");
    }
    require(device.trackedBufferBytes() == tracked, "Tracked device bytes differ from the timing forecast");
    t["memory_before"] = Json::parse(device.memoryJson());
    const auto slots = timingSchedule(timingRounds, uint32_t(shapes.size()));
    t["planned_dispatches"] = slots.size();
    t["samples"] = Json::array();
    std::vector<std::array<std::vector<double>, 3>> samples(shapes.size());
    for (const auto& slot : slots) {
        double ms = 0;
        const auto& shape = shapes[slot.shape];
        t["samples"].push_back({{"round", slot.round}, {"shape", shape.spec->name}, {"block", slot.block}, {"position", slot.position}});
        const bool passed = dispatchOnce(device, *shape.spec, operands[slot.shape], slot.kernel, true, shape.rounded,
                                         slot.kernel ? &shape.rounded : nullptr, nullptr, ledger, hooks, deadline, ms, t["samples"].back());
        require(passed, "Timing dispatch output differs from the oracle, its sentinels or the predecessor; stop");
        samples[slot.shape][slot.kernel].push_back(ms);
    }
    t["memory_after"] = Json::parse(device.memoryJson());
    operands.clear(); device.drain();
    require(device.trackedBufferBytes() == 0, "Timing buffers were not released");
    t["summary"] = Json::array();
    for (size_t i = 0; i < shapes.size(); ++i) {
        Json shape = {{"shape", shapes[i].spec->name}, {"kernels", Json::array()}};
        const double base = timingSummary(samples[i][0]).at("median_ms").get<double>();
        for (size_t k = 0; k < kernels.size(); ++k) {
            Json summary = timingSummary(samples[i][k]);
            summary["shader"] = kernels[k].shader;
            summary["median_ratio_to_predecessor"] = base > 0 ? Json(summary.at("median_ms").get<double>() / base) : Json(nullptr);
            shape["kernels"].push_back(std::move(summary));
        }
        t["summary"].push_back(std::move(shape));
    }
    t["scope"] = "Synthetic repeated dispatches of two public shapes; not model, page, OCR or sustained-throughput acceptance";
}
inline Json caseHeader(const CaseSpec& s) {
    return {{"name", s.name}, {"generator", generatorName(s.generator)}, {"bf16_weights", s.bf16Weights}, {"bias", biasName(s.bias)},
            {"input_width", s.inputWidth}, {"total_rows", s.totalRows}, {"row_first", s.rowFirst}, {"batch_rows", s.batchRows},
            {"output_width", s.outputWidth}, {"shard_first_row", s.shardFirstRow}, {"shard_total_rows", s.shardTotalRows},
            {"weight_row_first", s.weightRowFirst}, {"first_output", s.firstOutput()}, {"shard_rows", s.shardRows},
            {"guard_words", s.guardWords}, {"passed", false}};
}
// Cases in plan order; before each larger case the growth forecast must admit every kernel.
inline void run(chandra::dc::Device& device, Json& report, const Hooks& hooks, PlanSelector selector, const Deadline& deadline,
                const std::vector<CaseSpec>& cases) {
    Ledger ledger;
    std::array<double, 3> previous{}; uint64_t previousWork = 0;
    std::vector<TimingShape> shapes;
    report["cases"] = Json::array();
    try {
        for (const auto& s : cases) {
            report["cases"].push_back(caseHeader(s));
            Json& phase = report["cases"].back();
            if (previousWork) phase["growth_admission"] = admitNext(previous, previousWork, work(s));
            std::array<double, 3> slowest{}; std::vector<uint32_t> rounded;
            runCase(device, s, phase, ledger, hooks, deadline, slowest, rounded);
            report["dispatches_submitted"] = ledger.dispatches;
            hooks.checkpoint(report);
            previous = slowest; previousWork = work(s);
            if (s.timingShape) shapes.push_back({&s, slowest, std::move(rounded), phase.at("operands")});
        }
        report["correctness_passed"] = true;
        if (selector == PlanSelector::correctnessThenTiming) timing(device, report, ledger, hooks, deadline, shapes);
    } catch (...) {
        report["dispatches_submitted"] = ledger.dispatches;
        throw;
    }
    report["dispatches_submitted"] = ledger.dispatches;
    deadline.check();
}
inline Json initialReport(const std::vector<CaseSpec>& cases) {
    Json k = Json::array();
    for (const auto& kernel : kernels)
        k.push_back({{"shader", kernel.shader}, {"tile_k", kernel.tileK}, {"groupshared_row_stride_words", kernel.rowStride},
                     {"role", kernel.candidate ? "inactive candidate" : "unchanged production predecessor"}});
    Json planned = Json::array(); uint64_t timingShapes = 0;
    for (const auto& s : cases) { planned.push_back({{"name", s.name}, {"admission", admit(s)}}); timingShapes += s.timingShape; }
    return {{"schema", schema}, {"state", "PREPARING"}, {"native_executed", false}, {"passed", false},
            {"candidates_selectable_by_serving", false}, {"full_model_accepted", false}, {"trained_numerics_accepted", false},
            {"ocr_accepted", false}, {"end_to_end_performance_accepted", false}, {"page_throughput_accepted", false},
            {"automatic_rerun", false}, {"fallback", false}, {"kernels", k},
            {"dispatch_limit_milliseconds", dispatchLimitMilliseconds}, {"admission_safety_factor", admissionSafetyFactor},
            {"source_deadline_seconds", deadlineSeconds}, {"device_case_cap_bytes", deviceCaseCapBytes},
            {"required_external_job", {{"kill_on_close", true}, {"memory_limit_at_most_bytes", jobMemoryCapBytes}}},
            {"required_external_cpu", "One core, below normal priority"},
            {"correctness_dispatches", cases.size() * 2 * kernels.size()},
            {"timing_dispatches_if_selected", timingSchedule(timingRounds, uint32_t(timingShapes)).size()},
            {"plan", planned}, {"cases", Json::array()}};
}
} // namespace chandra::gemm_candidates
