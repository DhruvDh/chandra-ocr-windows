// New ChandraNative code, MPL-2.0. Portable core of the public, model-free multiwave regression for
// runtime/text_attention_softmax.hlsl (docs/directcompute-exact-input-correctness.md). It dispatches the
// actual shader through the unchanged production Device with production cbuffer words and groups, checks
// every output word against an independent binary64 normalizer with prospective BF16 gates, and runs the
// byte-identical 2c44182 original as a non-gating comparison control. Windows host and wmain:
// text_softmax_regression.cpp. CPU tests: text_softmax_regression_test.cpp (fake Device) and
// text_softmax_regression_emulation_test.cpp (translated HLSL). No model, weight, pixel or private asset is
// read. Root owns the external kill-on-close Job, CPU placement, device admission, build and execution.
#pragma once
#include "api.h"
#include "vision_dispatch_calibration.h" // Reused: SHA-256, pins, strict arguments, receipt excerpts, D3D11 screens.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::text_softmax_regression {
namespace vc = chandra::vision_calibration;
using Json = nlohmann::json;
using chandra::dc::Buffer;
using chandra::dc::Device;

constexpr const char* schema = "private.chandra.directcompute.text-softmax-regression.v1";
constexpr const char* generatorId = "public_lowbias32_dyadic_causal_text_scores_v1";
constexpr const char* repairedShader = "runtime/text_attention_softmax.hlsl";
constexpr const char* controlShader = "control/text_attention_softmax_2c44182.hlsl";
// Production constants: text_model.cpp mix() (16 query heads per row, <=64 rows per call, 16,384-token
// context) and text_attention_softmax.hlsl (128 lanes, groupshared tmp[128], one group per query-head row).
constexpr uint32_t lanes = 128, heads = 16, maximumCallRows = 64, maximumContext = 16384;
constexpr uint32_t maskedScoreBits = 0xff7fffffu; // -3.402823466e38, text_attention_scores.hlsl for key > base + row.
constexpr uint32_t sentinelBits = vc::sentinelBits; // Guard words after the score plane; no stage writes them.
constexpr uint32_t guardWords = lanes;
constexpr uint32_t repairedRepetitions = 3, controlRepetitions = 3;
constexpr uint32_t maximumGroups = maximumCallRows * heads;            // 1,024 groups, far below 65,535.
constexpr uint64_t maximumDispatchWords = 1ull << 22;                  // Score words per dispatch.
constexpr uint32_t maximumKeysPerLane = maximumContext / lanes;        // 128 iterations of each lane loop.
constexpr double dispatchLimitMilliseconds = 100.0;                    // Exclusive per-dispatch GPU gate.
constexpr uint32_t deadlineSeconds = 120;
constexpr uint64_t maximumBufferBytes = 128ull << 20;                  // device.cpp per-buffer limit.
constexpr uint64_t deviceAllocationCap = 32ull << 20;                  // One score buffer plus its staging copy.
constexpr uint64_t hostVectorCap = 128ull << 20;                       // Upload, oracle, readbacks, one control copy.
constexpr uint64_t receiptCap = 8ull << 20;
constexpr uint64_t jobMemoryCap = 1ull << 30;                          // External Job memory limit must be set and <= this.
constexpr uint64_t processCommitCap = 768ull << 20, jobCommitCap = 1024ull << 20; // Proposed external caps.
constexpr size_t reportedViolations = 16;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE binary32 required");
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53, "Independent binary64 oracle required");

struct FrozenSource { const char* relative; uint64_t bytes; const char* sha256; const char* role; };
constexpr std::array<FrozenSource, 2> frozenShaders{{
    {repairedShader, 1308, "283d3356ea3c882517e96ddff26706bae21bfe3cd04a523fc065e9aa8183169e",
     "repaired production shader under test: barrier after m=tmp[0]"},
    {controlShader, 1146, "5b6c0445dcebe2acdc8104bf1c0fd2b37e1c05665d30d960ca4a0e83d6432369",
     "byte-identical 2c44182 original; non-gating comparison control"}}};

inline void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
// Cancellation is a distinct, fail-closed outcome: no further dispatch, one bounded drain, no rerun.
struct Cancelled : std::runtime_error { using std::runtime_error::runtime_error; };

// Checked integer arithmetic for every size derived from a case.
inline uint64_t checkedAdd(uint64_t a, uint64_t b, const char* what) {
    require(b <= std::numeric_limits<uint64_t>::max() - a, std::string("Integer overflow: ") + what);
    return a + b;
}
inline uint64_t checkedMul(uint64_t a, uint64_t b, const char* what) {
    require(a == 0 || b <= std::numeric_limits<uint64_t>::max() / a, std::string("Integer overflow: ") + what);
    return a * b;
}
inline uint32_t checked32(uint64_t value, const char* what) {
    require(value <= std::numeric_limits<uint32_t>::max(), std::string("Value exceeds 32 bits: ") + what);
    return uint32_t(value);
}

// ---------------------------------------------------------------------------------------------------
// Explicit finite plan. Each case is one production attention call: `rows` query rows after `base` cached
// tokens, count = base + rows keys, 16 * rows groups, and row r admits keys 0..base + r.
struct CaseSpec { const char* name; uint32_t base, rows; const char* purpose; };
inline const std::vector<CaseSpec>& cases() {
    static const std::vector<CaseSpec> list = {
        {"single_key_decode_c1", 0, 1, "One key: every probability is exactly 1"},
        {"operator_fixture_like_c5", 2, 3,
         "Ragged rows of 3-5 keys: only lanes 0-4 own keys, all in the first wave of any width >= 8, so the shared-maximum hazard cannot change this output"},
        {"first_prefill_tile_c64", 0, 64, "First 64-row prefill tile: ragged causal rows of 1-64 keys"},
        {"decode_c127", 126, 1, "127 keys: lane 127 owns none"},
        {"decode_c128", 127, 1, "128 keys: every lane owns exactly one"},
        {"decode_c129", 128, 1, "129 keys: lane 0 owns two"},
        {"ragged_tail_c129", 96, 33, "33-row tail whose ragged rows of 97-129 keys cross 127, 128 and 129"},
        {"second_prefill_tile_c128", 64, 64, "Second 64-row tile: ragged rows of 65-128 keys"},
        {"tiny_prompt_tail_c801", 768, 33, "Last 33-row prefill tile of the original tiny page's 801-token prompt"},
        {"recorded_full_tile_c3584", 3520, 64, "Recorded page's last full 64-row prefill tile"},
        {"recorded_tail_c3617", 3584, 33, "Recorded page's 33-row causal prefill tail at 3,617 keys"},
        {"recorded_last_row_c3617", 3616, 1, "All 3,617 recorded prompt keys in one row"},
        {"recorded_decode0_c3618", 3617, 1, "First cached decode after the 3,617-token prompt"},
        {"recorded_decode4_c3622", 3621, 1, "Decode step 4: consumes generated token 4 and produces generated index 5"},
        {"normal_allowance_last_c16000", 15999, 1,
         "Largest key count of a 3,617-token prompt within the 12,384-token normal allowance (3,617 + 12,384 - 1)"},
        {"context_bound_tail_c16384", 16382, 2, "Two ragged rows of 16,383 and 16,384 keys at the context bound"},
        {"context_bound_decode_c16384", 16383, 1, "Maximum cached key count of the 16,384-token context"}};
    return list;
}
// Production cbuffer of text_model.cpp mix(): p={n,16,0,0,c.length,c.length+n,0,0}; the softmax reads count.
struct TextParams { uint32_t rows, width, offset, stride, base, count, mode, pad; };
static_assert(sizeof(TextParams) == 32, "text attention cbuffer ABI");
struct Geometry {
    uint32_t base = 0, rows = 0, count = 0, groups = 0, planeWords = 0, bufferWords = 0;
    uint32_t minimumValid = 0, maximumValid = 0, keysPerLane = 0;
    uint64_t maskedWords = 0, bufferBytes = 0, deviceBytes = 0, hostBytes = 0;
};
// Sizes are derived and refused here, before any allocation; every product and sum is checked.
inline Geometry geometry(const CaseSpec& s) {
    Geometry g; g.base = s.base; g.rows = s.rows;
    require(s.rows >= 1 && s.rows <= maximumCallRows, "Case rows must be 1..64, one production prefill/decode call");
    const uint64_t count = checkedAdd(s.base, s.rows, "base + rows");
    require(count <= maximumContext, "Case key count exceeds the 16,384-token context bound");
    g.count = uint32_t(count);
    const uint64_t groups = checkedMul(s.rows, heads, "rows * 16");
    require(groups <= maximumGroups, "Case groups exceed 1,024");
    g.groups = uint32_t(groups);
    const uint64_t plane = checkedMul(groups, count, "groups * count");
    require(plane <= maximumDispatchWords, "Case score plane exceeds the per-dispatch word cap");
    g.planeWords = checked32(plane, "score plane words");
    g.bufferWords = checked32(checkedAdd(plane, guardWords, "plane + guard"), "buffer words");
    g.bufferBytes = checkedMul(g.bufferWords, 4, "buffer bytes");
    require(g.bufferBytes <= maximumBufferBytes, "Case buffer exceeds the 128 MiB Device limit");
    g.deviceBytes = checkedMul(g.bufferBytes, 2, "buffer plus staging bytes");
    require(g.deviceBytes <= deviceAllocationCap, "Case buffer plus staging exceeds the 32 MiB device cap");
    // Upload vector, two held readbacks, one sensitivity-control copy, the 6-byte/word oracle and row scratch.
    const uint64_t vectors = checkedMul(g.bufferBytes, 4, "host word vectors");
    const uint64_t oracle = checkedMul(plane, 6, "oracle bytes");
    const uint64_t scratch = checkedMul(count, 4 * sizeof(double), "row scratch");
    g.hostBytes = checkedAdd(checkedAdd(vectors, oracle, "host bytes"), scratch, "host bytes");
    require(g.hostBytes <= hostVectorCap, "Case host vectors exceed the 128 MiB host cap");
    g.minimumValid = s.base + 1; g.maximumValid = g.count;
    g.keysPerLane = (g.count + lanes - 1) / lanes;
    require(g.keysPerLane <= maximumKeysPerLane, "Lane loop exceeds 128 iterations");
    uint64_t valid = 0;
    for (uint32_t r = 0; r < s.rows; ++r) valid = checkedAdd(valid, checkedMul(heads, uint64_t(s.base) + r + 1, "valid"), "valid");
    g.maskedWords = plane - valid;
    return g;
}
inline TextParams params(const Geometry& g) { return {g.rows, heads, 0, 0, g.base, g.count, 0, 0}; }
inline uint32_t validKeys(const Geometry& g, uint32_t group) { return g.base + group / heads + 1; }
// Waves of `width` lanes that contain at least one lane owning a key. The hazard can change an output
// only if a lane outside lane 0's wave owns a key, i.e. this is at least two.
inline uint32_t wavesWithKeys(uint32_t count, uint32_t width) { return (std::min(count, lanes) + width - 1) / width; }

// ---------------------------------------------------------------------------------------------------
// Public generator v1 (lowbias32 via vc::mix). Valid scores are j/16 with |j| <= 255: BF16-exact values
// times 1/16, like text_attention_scores.hlsl's bf(q.k)*0.0625. Row kinds cycle so every case, including
// 16-group decode cases, has all three. Spike rows put their maximum on the last valid key, key 0, the
// last lane or a hashed key, which places the maximum in different lanes and waves.
enum class RowKind : uint8_t { spread = 0, uniform = 1, spike = 2 };
constexpr uint32_t tagSpread = 0x53707264u, tagUniform = 0x556e6966u, tagSpikeKey = 0x53706b4bu, tagSpikeValue = 0x53706b56u,
                   tagSpikeRest = 0x53706b52u;
inline RowKind rowKind(uint32_t caseIndex, uint32_t group) { return RowKind((group + caseIndex) % 3u); }
inline const char* kindName(RowKind k) { return k == RowKind::spread ? "spread" : k == RowKind::uniform ? "uniform" : "spike"; }
inline uint32_t spikeKey(uint32_t caseIndex, uint32_t group, uint32_t valid) {
    switch ((group / 3u) % 4u) {
    case 0: return valid - 1;
    case 1: return 0;
    case 2: return std::min(lanes - 1, valid - 1);
    default: return vc::mix(tagSpikeKey, caseIndex, group) % valid;
    }
}
// Integer j of a valid key's score j/16.
inline int scoreInteger(uint32_t caseIndex, uint32_t group, uint32_t key, uint32_t valid) {
    switch (rowKind(caseIndex, group)) {
    case RowKind::spread: return int(vc::mix(tagSpread ^ caseIndex, group, key) % 256u) - 128;
    case RowKind::uniform: return int(vc::mix(tagUniform, caseIndex, group) % 255u) - 127;
    default:
        return key == spikeKey(caseIndex, group, valid) ? 128 + int(vc::mix(tagSpikeValue, caseIndex, group) % 64u)
                                                        : -255 + int(vc::mix(tagSpikeRest ^ caseIndex, group, key) % 64u);
    }
}
inline uint32_t scoreWord(uint32_t caseIndex, const Geometry& g, uint32_t group, uint32_t key) {
    const uint32_t valid = validKeys(g, group);
    return key < valid ? vc::bits(float(scoreInteger(caseIndex, group, key, valid)) / 16.0f) : maskedScoreBits;
}
struct Deadline;
inline void periodic(const Deadline* deadline, uint64_t i);
// Score plane in [group][key] order followed by the sentinel guard, exactly the buffer the call uploads.
inline std::vector<uint32_t> uploadWords(uint32_t caseIndex, const Geometry& g, const Deadline* deadline = nullptr) {
    std::vector<uint32_t> words(g.bufferWords, sentinelBits);
    for (uint32_t group = 0; group < g.groups; ++group) {
        periodic(deadline, group);
        for (uint32_t key = 0; key < g.count; ++key) words[size_t(group) * g.count + key] = scoreWord(caseIndex, g, group, key);
    }
    return words;
}

// ---------------------------------------------------------------------------------------------------
// One absolute deadline from receipt creation; never reset, extended or restarted. `now` is injectable so
// CPU tests can expire it at exact boundaries; the Windows host passes steady_clock::now.
using Clock = std::chrono::steady_clock;
struct Deadline {
    std::function<Clock::time_point()> now = [] { return Clock::now(); };
    Clock::time_point start = Clock::now();
    std::function<bool()> cancelled = [] { return false; };
    Deadline() = default;
    Deadline(std::function<Clock::time_point()> clockSource, std::function<bool()> cancelSource)
        : now(std::move(clockSource)), start(now()), cancelled(std::move(cancelSource)) {}
    bool expired() const { return now() - start >= std::chrono::seconds(deadlineSeconds); }
    double seconds() const { return std::chrono::duration<double>(now() - start).count(); }
    // Every admission boundary: cancellation first, then the deadline. Nothing is dispatched after either.
    void check(const char* boundary = "host_work") const {
        if (cancelled()) throw Cancelled(std::string("Cancellation requested; stopped at ") + boundary + "; no further dispatch");
        if (expired())
            throw std::runtime_error(std::string("120-second source deadline exceeded at ") + boundary +
                                     "; no further dispatch; retire the externally owned Job");
    }
};
inline void periodic(const Deadline* deadline, uint64_t i) { if (deadline && !(i % 64)) deadline->check("host_work"); }

// ---------------------------------------------------------------------------------------------------
// Independent normalizer and prospective gates (derivation in the guide). The oracle uses only the
// generator's integers and binary64 exp/division with a compensated sum; it never reads device output.
// Named assumptions, shared with the closed vision calibration: FP32 add/mul/sub round to nearest even;
// exp lowers to base-2 exp with relative error <= 2^-21 (vc::exp2RelativeError); FP32 division is within
// 2.5 ULP (D3D11). The argument bound below covers both a mul-then-exp2 and a fused or unfused mad lowering
// of exp(s - m), and the sum bound holds for any association inside each lane's loop.
constexpr double log2e = 1.4426950408889634;
constexpr double oracleMargin = vc::oracleRelativeMargin; // 2^-36, far below every gate.
inline double argumentErrorBound(double absX, double absS, double absM) {
    return log2e * (absX + 2 * absS + 2 * absM) * 4 * vc::unitRoundoff;
}
inline double exponentialTermBound(double absX, double absS, double absM) {
    const double delta = argumentErrorBound(absX, absS, absM);
    require(delta <= 0x1p-10, "Exponential argument bound outside the derived regime");
    return std::exp2(delta) * (1 + vc::exp2RelativeError) - 1;
}
// Shader order: each lane sums ceil(count/128) terms from +0, then a seven-level tree (vc helper, 128 lanes).
inline double probabilityBound(double termBound, double rowTermBound, uint32_t count) {
    return vc::probabilityRelativeBound(termBound, rowTermBound, count) + oracleMargin;
}
// Round a positive binary64 value to the BF16 grid (8 significant bits), nearest-even, exactly.
inline uint32_t bf16NearestBits(double v) {
    require(std::isfinite(v) && v >= 0x1p-126 && v < 0x1p127, "Oracle probability outside the normal BF16 range");
    uint64_t u; std::memcpy(&u, &v, sizeof(u));
    constexpr unsigned drop = 52 - 7;
    const uint64_t remainder = u & ((uint64_t(1) << drop) - 1), half = uint64_t(1) << (drop - 1);
    u -= remainder;
    if (remainder > half || (remainder == half && ((u >> drop) & 1u))) u += uint64_t(1) << drop;
    double rounded; std::memcpy(&rounded, &u, sizeof(rounded));
    const float f = float(rounded);
    require(double(f) == rounded && (vc::bits(f) & 0xffffu) == 0, "BF16 rounding left the float grid");
    return vc::bits(f);
}
// BF16 word of any binary64 model value (control models only), with the shader bf()'s rounding contract:
// nearest-even to 8 significant bits with an unbounded exponent, overflowing to infinity only when that
// rounded magnitude reaches 2^128. So [2^127, 2^128 - 2^119) stays finite and the tie 2^128 - 2^119 goes
// to even, which is infinity. Magnitudes below 2^-126 flush to a signed zero (D3D11 denormal flushing);
// NaN becomes the canonical quiet NaN. Unlike bf16NearestBits, it never refuses a value.
inline uint32_t bf16ModelBits(double v) {
    if (!(v == v)) return 0x7fc00000u;
    const uint32_t sign = std::signbit(v) ? 0x80000000u : 0u;
    const double a = std::fabs(v);
    if (a < 0x1p-126) return sign;
    if (a < 0x1p127) return sign | bf16NearestBits(a);
    if (std::isinf(a)) return sign | 0x7f800000u;
    uint64_t u; std::memcpy(&u, &a, sizeof(u)); // Same exact binary64 rounding as bf16NearestBits.
    constexpr unsigned drop = 52 - 7;
    const uint64_t remainder = u & ((uint64_t(1) << drop) - 1), half = uint64_t(1) << (drop - 1);
    u -= remainder;
    if (remainder > half || (remainder == half && ((u >> drop) & 1u))) u += uint64_t(1) << drop;
    double rounded; std::memcpy(&rounded, &u, sizeof(rounded));
    if (rounded >= 0x1p128) return sign | 0x7f800000u;
    return sign | vc::bits(float(rounded));
}
struct Oracle {
    // High halves of the admissible BF16 interval [low, high] and of the correctly rounded binary64
    // probability, per valid word; masked words must be exactly +0 and keep zero entries here.
    std::vector<uint16_t> low, high, nearest;
    std::vector<double> rowBound;
    double maximumBound = 0, maximumTermBound = 0, minimumProbability = 1;
    uint64_t validWords = 0, twoValueWords = 0, wideWords = 0;
};
// Binary64 probabilities of one row from the generator's integers; returns the row maximum's integer.
inline int rowProbabilities(uint32_t caseIndex, const Geometry& g, uint32_t group, std::vector<int>& j, std::vector<double>& e,
                            std::vector<double>& p) {
    const uint32_t valid = validKeys(g, group);
    j.resize(valid); e.resize(valid); p.resize(valid);
    int maximum = std::numeric_limits<int>::min();
    for (uint32_t k = 0; k < valid; ++k) {
        j[k] = scoreInteger(caseIndex, group, k, valid);
        require(j[k] >= -255 && j[k] <= 255, "Generated score is not a BF16-exact j/16");
        maximum = std::max(maximum, j[k]);
    }
    double sum = 0, compensation = 0; // Neumaier compensated binary64 sum.
    for (uint32_t k = 0; k < valid; ++k) {
        e[k] = std::exp(double(j[k] - maximum) / 16.0);
        const double next = sum + e[k];
        compensation += std::fabs(sum) >= e[k] ? (sum - next) + e[k] : (e[k] - next) + sum;
        sum = next;
    }
    sum += compensation;
    for (uint32_t k = 0; k < valid; ++k) p[k] = e[k] / sum;
    return maximum;
}
inline Oracle buildOracle(uint32_t caseIndex, const Geometry& g, const Deadline* deadline = nullptr) {
    Oracle o; o.low.assign(g.planeWords, 0); o.high.assign(g.planeWords, 0); o.nearest.assign(g.planeWords, 0);
    o.rowBound.assign(g.groups, 0.0);
    std::vector<int> j; std::vector<double> e, p, t;
    for (uint32_t group = 0; group < g.groups; ++group) {
        periodic(deadline, group);
        const uint32_t valid = validKeys(g, group);
        const int maximum = rowProbabilities(caseIndex, g, group, j, e, p);
        t.resize(valid); double rowTerm = 0;
        for (uint32_t k = 0; k < valid; ++k) {
            // |x|, |s| and |m| are exact: x = (j - jmax)/16 is computed exactly in FP32 by the shader.
            t[k] = exponentialTermBound(double(maximum - j[k]) / 16.0, std::fabs(double(j[k])) / 16.0, std::fabs(double(maximum)) / 16.0);
            rowTerm = std::max(rowTerm, t[k]);
        }
        o.maximumTermBound = std::max(o.maximumTermBound, rowTerm);
        for (uint32_t k = 0; k < valid; ++k) {
            const double bound = probabilityBound(t[k], rowTerm, g.count);
            const size_t index = size_t(group) * g.count + k;
            const uint32_t low = bf16NearestBits(p[k] * (1 - bound)), high = bf16NearestBits(p[k] * (1 + bound));
            o.low[index] = uint16_t(low >> 16); o.high[index] = uint16_t(high >> 16);
            o.nearest[index] = uint16_t(bf16NearestBits(p[k]) >> 16);
            o.rowBound[group] = std::max(o.rowBound[group], bound);
            o.maximumBound = std::max(o.maximumBound, bound);
            o.minimumProbability = std::min(o.minimumProbability, p[k]);
            const uint32_t admissible = uint32_t(o.high[index] - o.low[index]) + 1;
            o.twoValueWords += admissible == 2 ? 1u : 0u; o.wideWords += admissible > 2 ? 1u : 0u;
        }
        o.validWords += valid;
    }
    // A gate admitting more than two adjacent BF16 values would be too broad for this derivation.
    require(o.wideWords == 0, "Oracle gate admits more than two BF16 values for some word");
    return o;
}

// Every word of the buffer: valid words in their admissible BF16 interval, finite, BF16-stored and
// positive; masked words exactly +0; guard words unchanged; rows normalized within the BF16 bound;
// uniform rows bit-identical across their valid keys.
inline Json checkOutput(uint32_t caseIndex, const Geometry& g, const Oracle& o, const std::vector<uint32_t>& observed,
                        const Deadline* deadline = nullptr) {
    require(observed.size() == g.bufferWords, "Readback size differs from the case buffer");
    uint64_t gate = 0, nonfinite = 0, storage = 0, masked = 0, guard = 0, normalization = 0, uniform = 0;
    uint64_t equalNearest = 0, admittedNotNearest = 0, admittedInTwoValueGate = 0;
    double worstNormalization = 0, worstNormalizationRatio = 0;
    Json first = Json::array();
    auto violation = [&](const char* kind, uint64_t index, uint32_t word) {
        if (first.size() >= reportedViolations) return;
        Json v = {{"kind", kind}, {"index", index}, {"observed_bits", word}};
        if (index < g.planeWords) {
            const uint32_t group = uint32_t(index / g.count), key = uint32_t(index % g.count);
            v["group"] = group; v["key"] = key; v["row_kind"] = kindName(rowKind(caseIndex, group));
            if (key < validKeys(g, group)) {
                v["admissible_bits"] = {uint32_t(o.low[index]) << 16, uint32_t(o.high[index]) << 16};
                v["nearest_bits"] = uint32_t(o.nearest[index]) << 16;
            }
        }
        first.push_back(std::move(v));
    };
    for (uint32_t group = 0; group < g.groups; ++group) {
        periodic(deadline, group);
        const uint32_t valid = validKeys(g, group);
        const size_t row = size_t(group) * g.count;
        double sum = 0; bool finiteRow = true;
        for (uint32_t key = 0; key < g.count; ++key) {
            const size_t index = row + key; const uint32_t word = observed[index];
            if (key >= valid) {
                if (word != 0) { ++masked; violation("masked_word_not_positive_zero", index, word); }
                continue;
            }
            if ((word & 0x7f800000u) == 0x7f800000u) { ++nonfinite; finiteRow = false; violation("nonfinite", index, word); continue; }
            if (word & 0xffffu) { ++storage; violation("not_bf16_storage", index, word); }
            const uint32_t half = word >> 16;
            if (word >> 31 || half < o.low[index] || half > o.high[index]) { ++gate; violation("outside_admissible_bf16_interval", index, word); }
            else if (half == o.nearest[index]) ++equalNearest;
            else ++admittedNotNearest;
            admittedInTwoValueGate += o.low[index] != o.high[index] ? 1u : 0u;
            sum += double(vc::fromBits(word));
        }
        if (finiteRow) {
            // Oracle-free: sum |bf16(y_k) - p_k| <= 2^-8 (1 + B) + B, plus binary64 summation of the row.
            const double bound = 0x1p-8 * (1 + o.rowBound[group]) + o.rowBound[group] + double(valid) * 0x1p-52;
            const double error = std::fabs(sum - 1.0);
            worstNormalization = std::max(worstNormalization, error);
            worstNormalizationRatio = std::max(worstNormalizationRatio, error / bound);
            if (!(error <= bound)) { ++normalization; violation("row_normalization", row, observed[row]); }
        }
        if (rowKind(caseIndex, group) == RowKind::uniform)
            for (uint32_t key = 1; key < valid; ++key)
                if (observed[row + key] != observed[row]) { ++uniform; violation("uniform_row_not_bit_identical", row + key, observed[row + key]); break; }
    }
    for (uint32_t i = g.planeWords; i < g.bufferWords; ++i)
        if (observed[i] != sentinelBits) { ++guard; violation("guard_word_changed", i, observed[i]); }
    const bool passed = gate + nonfinite + storage + masked + guard + normalization + uniform == 0;
    return {{"passed", passed}, {"words_checked", observed.size()}, {"valid_words", o.validWords}, {"masked_words", g.maskedWords},
            {"guard_words", guardWords}, {"outside_admissible_interval", gate}, {"nonfinite", nonfinite}, {"not_bf16_storage", storage},
            {"masked_not_positive_zero", masked}, {"guard_changed", guard}, {"rows_failing_normalization", normalization},
            {"uniform_rows_not_identical", uniform}, {"equal_to_correctly_rounded_binary64", equalNearest},
            {"admitted_adjacent_to_correctly_rounded_binary64", admittedNotNearest},
            {"valid_words_with_two_value_gate", admittedInTwoValueGate}, {"maximum_row_normalization_error", worstNormalization},
            {"maximum_row_normalization_error_to_bound_ratio", worstNormalizationRatio}, {"first_violations", first}};
}

// ---------------------------------------------------------------------------------------------------
// Sensitivity controls: deliberately wrong outputs derived from an accepted readback (or from binary64
// models of the hazard) must be refused by the same check. A control that would be accepted fails the run.
struct ControlModel { bool changed = false; std::vector<uint32_t> words; };
// Original shader with lane-granular waves of `width` executed in ascending order after the maximum
// reduction: lane 0's wave stores tmp[0]=its exp partial before later waves read the maximum, so those waves
// use lane 0's partial as m. Binary64 model; BF16 rounding as stored.
inline ControlModel raceModel(uint32_t caseIndex, const Geometry& g, const std::vector<uint32_t>& accepted, uint32_t width) {
    ControlModel model{false, accepted};
    std::vector<int> j; std::vector<double> e, p;
    for (uint32_t group = 0; group < g.groups; ++group) {
        const uint32_t valid = validKeys(g, group);
        const int maximum = rowProbabilities(caseIndex, g, group, j, e, p);
        if (valid <= width) continue; // Every key-owning lane is in lane 0's wave.
        double lane0 = 0;
        for (uint32_t k = 0; k < valid; k += lanes) lane0 += e[k];
        double sum = 0; std::vector<double> wrong(valid);
        for (uint32_t k = 0; k < valid; ++k) {
            const double m = (k % lanes) < width ? double(maximum) / 16.0 : lane0;
            wrong[k] = std::exp(double(j[k]) / 16.0 - m); sum += wrong[k];
        }
        for (uint32_t k = 0; k < valid; ++k) model.words[size_t(group) * g.count + k] = bf16ModelBits(wrong[k] / sum);
    }
    for (uint32_t i = 0; i < g.planeWords; ++i) model.changed = model.changed || model.words[i] != accepted[i];
    return model;
}
// Lane 127's exp partial omitted from the denominator.
inline ControlModel droppedLaneModel(uint32_t caseIndex, const Geometry& g, const std::vector<uint32_t>& accepted) {
    ControlModel model{false, accepted};
    std::vector<int> j; std::vector<double> e, p;
    for (uint32_t group = 0; group < g.groups; ++group) {
        const uint32_t valid = validKeys(g, group);
        if (valid < lanes) continue;
        rowProbabilities(caseIndex, g, group, j, e, p);
        double sum = 0, dropped = 0;
        for (uint32_t k = 0; k < valid; ++k) { sum += e[k]; dropped += k % lanes == lanes - 1 ? e[k] : 0.0; }
        for (uint32_t k = 0; k < valid; ++k) model.words[size_t(group) * g.count + k] = bf16ModelBits(e[k] / (sum - dropped));
    }
    for (uint32_t i = 0; i < g.planeWords; ++i) model.changed = model.changed || model.words[i] != accepted[i];
    return model;
}
inline Json sensitivityControls(uint32_t caseIndex, const Geometry& g, const Oracle& o, const std::vector<uint32_t>& accepted,
                                const Deadline* deadline = nullptr) {
    Json list = Json::array(); bool allRefused = true;
    auto judge = [&](const char* name, bool applicable, const std::vector<uint32_t>* words, const char* note) {
        Json entry = {{"name", name}, {"applicable", applicable}, {"note", note}};
        if (applicable) {
            if (deadline) deadline->check("sensitivity_control");
            const Json check = checkOutput(caseIndex, g, o, *words, deadline);
            const bool refused = check.at("passed") == false;
            entry["refused"] = refused; allRefused = allRefused && refused;
            if (!check.at("first_violations").empty()) entry["first_violation_kind"] = check.at("first_violations").front().at("kind");
        }
        list.push_back(std::move(entry));
    };
    // Valid word of the longest row with the largest probability, and a masked word if any exist.
    const uint32_t longest = g.groups - 1; const size_t row = size_t(longest) * g.count;
    size_t target = row;
    for (uint32_t k = 1; k < validKeys(g, longest); ++k) if (o.nearest[row + k] > o.nearest[target]) target = row + k;
    std::vector<uint32_t> copy = accepted;
    auto reset = [&] { copy = accepted; };
    copy[target] = (uint32_t(o.high[target]) << 16) + 0x10000u;
    judge("next_bf16_above_admissible_interval", true, &copy, "One valid word one BF16 step above its gate");
    reset(); const bool lowerable = o.low[target] > 0x0080u;
    if (lowerable) copy[target] = (uint32_t(o.low[target]) << 16) - 0x10000u;
    judge("next_bf16_below_admissible_interval", lowerable, &copy, "One valid word one BF16 step below its gate");
    reset(); copy[target] = 0x7fc00000u;
    judge("nan_word", true, &copy, "One valid word replaced by a quiet NaN");
    reset(); copy[target] = accepted[target] | 1u;
    judge("non_bf16_storage", true, &copy, "Low 16 bits of one valid word set");
    reset(); size_t maskedIndex = 0; bool hasMasked = false;
    for (uint32_t group = 0; group < g.groups && !hasMasked; ++group)
        if (validKeys(g, group) < g.count) { maskedIndex = size_t(group) * g.count + validKeys(g, group); hasMasked = true; }
    if (hasMasked) copy[maskedIndex] = 0x00800000u;
    judge("masked_word_smallest_normal", hasMasked, &copy, "One causally masked word set to the smallest normal BF16");
    reset(); copy[g.planeWords] = 0;
    judge("guard_word_overwritten", true, &copy, "First guard word after the score plane overwritten");
    reset();
    for (uint32_t k = 0; k < validKeys(g, longest); ++k) copy[row + k] = (uint32_t(o.nearest[row + k]) << 16) + 0x10000u;
    judge("longest_row_raised_one_bf16_step", true, &copy, "Every valid word of the longest row one BF16 step above nearest");
    std::vector<uint32_t>().swap(copy);
    for (uint32_t width : {32u, 16u}) {
        const ControlModel race = raceModel(caseIndex, g, accepted, width);
        judge(width == 32 ? "original_hazard_model_wave32_ascending" : "original_hazard_model_wave16_ascending", race.changed, &race.words,
              "Binary64 model of the 2c44182 shader when lane 0's wave overwrites tmp[0] before later waves read the maximum; "
              "not applicable when every key-owning lane is in lane 0's wave");
    }
    const ControlModel dropped = droppedLaneModel(caseIndex, g, accepted);
    judge("lane127_partial_dropped", dropped.changed, &dropped.words, "Binary64 model omitting lane 127's exp partial from the denominator");
    return {{"controls", list}, {"all_applicable_refused", allRefused}};
}

// ---------------------------------------------------------------------------------------------------
// Plan, identities and admission.
inline Json caseJson(size_t index) {
    const CaseSpec& s = cases().at(index); const Geometry g = geometry(s); const TextParams p = params(g);
    return {{"index", index}, {"name", s.name}, {"purpose", s.purpose}, {"base", g.base}, {"rows", g.rows}, {"count", g.count},
            {"groups", std::array<uint32_t, 3>{{g.groups, 1u, 1u}}}, {"cbuffer_words", {p.rows, p.width, p.offset, p.stride, p.base, p.count, p.mode, p.pad}},
            {"valid_keys_per_row", {g.minimumValid, g.maximumValid}}, {"masked_words", g.maskedWords}, {"plane_words", g.planeWords},
            {"guard_words", guardWords}, {"buffer_bytes", g.bufferBytes}, {"device_plus_staging_bytes", g.deviceBytes},
            {"host_forecast_bytes", g.hostBytes}, {"maximum_keys_per_lane", g.keysPerLane}, {"lanes_owning_keys", std::min(g.count, lanes)},
            {"waves_with_key_owning_lanes", {{"8", wavesWithKeys(g.count, 8)}, {"16", wavesWithKeys(g.count, 16)},
                                             {"32", wavesWithKeys(g.count, 32)}, {"64", wavesWithKeys(g.count, 64)}}},
            {"repaired_repetitions", repairedRepetitions}, {"control_repetitions", controlRepetitions}};
}
inline Json planJson() {
    Json list = Json::array(); uint64_t words = 0, device = 0, host = 0;
    for (size_t i = 0; i < cases().size(); ++i) {
        list.push_back(caseJson(i)); const Geometry g = geometry(cases()[i]);
        words += g.planeWords; device = std::max(device, g.deviceBytes); host = std::max(host, g.hostBytes);
    }
    Json frozen = Json::array();
    for (const auto& f : frozenShaders) frozen.push_back({{"path", f.relative}, {"bytes", f.bytes}, {"sha256", f.sha256}, {"role", f.role}});
    const uint64_t dispatches = uint64_t(cases().size()) * (repairedRepetitions + controlRepetitions);
    return {{"schema", schema}, {"generator", generatorId}, {"cases", list}, {"dispatches", dispatches},
            {"repaired_dispatches", uint64_t(cases().size()) * repairedRepetitions},
            {"control_dispatches", uint64_t(cases().size()) * controlRepetitions}, {"plane_words_per_pass", words},
            {"maximum_device_plus_staging_bytes", device}, {"maximum_host_forecast_bytes", host},
            {"caps", {{"device_allocation_bytes", deviceAllocationCap}, {"host_vector_bytes", hostVectorCap}, {"receipt_bytes", receiptCap},
                      {"dispatch_words", maximumDispatchWords}, {"groups", maximumGroups}, {"keys_per_lane", maximumKeysPerLane},
                      {"context_keys", maximumContext}}},
            {"frozen_shaders", frozen},
            {"gates", {{"dispatch_limit_milliseconds_exclusive", dispatchLimitMilliseconds}, {"source_deadline_seconds", deadlineSeconds},
                       {"unit_roundoff", vc::unitRoundoff}, {"exp2_relative_error_assumed", vc::exp2RelativeError},
                       {"division_relative_error_d3d11", vc::divisionRelativeError}, {"oracle_relative_margin", oracleMargin},
                       {"argument_bound", "log2(e) (|x| + 2|s| + 2|m|) 4u"},
                       {"normalization_bound", "2^-8 (1 + B_row) + B_row + valid 2^-52"}}},
            {"original_control_gating", false}};
}
inline std::string planSha256() { const std::string text = planJson().dump(); return vc::sha256Bytes(text.data(), text.size()); }
inline Json fileCommitment(const std::filesystem::path& path, uint64_t cap) {
    require(std::filesystem::is_regular_file(path), "Committed file absent: " + path.u8string());
    const uint64_t size = std::filesystem::file_size(path);
    require(size > 0 && size <= cap, "Committed file empty or above its size bound: " + path.u8string());
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Committed file unreadable: " + path.u8string());
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    require(data.size() == size, "Committed file changed while it was read: " + path.u8string());
    return {{"path", path.u8string()}, {"bytes", data.size()}, {"sha256", vc::sha256Bytes(data.data(), data.size())}};
}
// Hashes the exact bytes the Device will compile; any difference refuses before Device creation (and fails
// the run when re-checked after the last dispatch).
inline Json verifyFrozenShaders(const std::filesystem::path& shaderRoot) {
    Json result = Json::array();
    for (const auto& f : frozenShaders) {
        const auto path = shaderRoot / std::filesystem::path(f.relative);
        require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == f.bytes,
                std::string("Frozen shader absent or resized: ") + f.relative);
        Json commitment = fileCommitment(path, f.bytes);
        commitment["relative"] = f.relative; commitment["frozen_sha256"] = f.sha256;
        require(commitment.at("sha256") == f.sha256, std::string("Shader bytes differ from the frozen source: ") + f.relative);
        result.push_back(std::move(commitment));
    }
    return result;
}
// Device selects only Intel A770 adapters by exact PCI/LUID; this re-checks the created device's report.
inline void authenticateIdentity(const Json& identity, const std::string& pci, const std::string& luid) {
    require(identity.is_object() && identity.contains("pci_bdf") && identity.at("pci_bdf") == pci && identity.contains("luid") &&
            identity.at("luid") == luid, "Created device PCI/LUID differs from the request or is missing");
    require(identity.contains("vendor_id") && identity.at("vendor_id") == 0x8086u, "Created device is not an Intel adapter");
    require(identity.contains("description") && identity.at("description").is_string() &&
            identity.at("description").get<std::string>().find("A770") != std::string::npos, "Created device is not an Arc A770");
}
// winnt.h JOB_OBJECT_LIMIT_* values; the Windows host static_asserts equality.
constexpr uint32_t jobLimitProcessTime = 0x2, jobLimitJobTime = 0x4, jobLimitProcessMemory = 0x100, jobLimitJobMemory = 0x200,
                   jobLimitKillOnJobClose = 0x2000;
struct JobLimits { bool inJob = false; uint32_t flags = 0; uint64_t processMemory = 0, jobMemory = 0; };
// Verifies the root-owned Job; the driver never creates, joins or modifies a Job.
inline Json admitJob(const JobLimits& j) {
    require(j.inJob, "Run inside the externally owned Job; refusing to execute");
    require((j.flags & jobLimitKillOnJobClose) != 0, "External Job must kill on close");
    const bool process = (j.flags & jobLimitProcessMemory) != 0, job = (j.flags & jobLimitJobMemory) != 0;
    require(process || job, "External Job must bound memory");
    require((!process || (j.processMemory > 0 && j.processMemory <= jobMemoryCap)) && (!job || (j.jobMemory > 0 && j.jobMemory <= jobMemoryCap)),
            "External Job memory limit exceeds 1 GiB");
    return {{"in_job", true}, {"created_by_this_process", false}, {"limit_flags", j.flags}, {"kill_on_close", true},
            {"process_memory_limit", process ? Json(j.processMemory) : Json(nullptr)},
            {"job_memory_limit", job ? Json(j.jobMemory) : Json(nullptr)},
            {"cpu_time_limited", (j.flags & (jobLimitProcessTime | jobLimitJobTime)) != 0}};
}
// Output must be absolute, fresh, in an existing directory and outside the authenticated shader root.
inline void admitPaths(const std::filesystem::path& root, const std::filesystem::path& output) {
    require(root.is_absolute() && std::filesystem::is_directory(root), "Absolute existing shader root required");
    require(output.is_absolute() && output.has_filename() && std::filesystem::is_directory(output.parent_path()),
            "Absolute fresh receipt path in an existing directory required");
    const auto canonicalRoot = std::filesystem::weakly_canonical(root), canonicalOutput = std::filesystem::weakly_canonical(output);
    const auto mismatch = std::mismatch(canonicalRoot.begin(), canonicalRoot.end(), canonicalOutput.begin(), canonicalOutput.end());
    require(mismatch.first != canonicalRoot.end(), "Receipt path must lie outside the authenticated shader root");
    require(!std::filesystem::exists(std::filesystem::symlink_status(output)), "Receipt path already exists; a fresh path is required");
}

// ---------------------------------------------------------------------------------------------------
// Bounded receipt built from the vision calibration's streaming serializer and excerpts, with this schema. An oversized
// report becomes a failing RECEIPT_OVERSIZE summary of bounded fields, never truncated JSON.
inline std::string oversizeReceipt(const Json& report, uint64_t fullBytes) {
    auto field = [&](const char* key) -> Json {
        if (!report.is_object() || !report.contains(key)) return {{"present", false}};
        Json digest = vc::fieldDigest(report.at(key)); digest["present"] = true; return digest;
    };
    auto scalar = [&](const char* key) -> Json {
        if (!report.is_object() || !report.contains(key)) return nullptr;
        const Json& v = report.at(key); return v.is_boolean() || v.is_number_integer() ? v : Json(nullptr);
    };
    Json drain = nullptr;
    if (report.is_object() && report.contains("failure_drain") && report.at("failure_drain").is_object()) {
        const Json& d = report.at("failure_drain");
        drain = {{"completed", d.contains("completed") && d.at("completed").is_boolean() ? d.at("completed") : Json(nullptr)},
                 {"tracked_after", d.contains("tracked_after") && d.at("tracked_after").is_number_integer() ? d.at("tracked_after") : Json(nullptr)}};
    }
    const Json summary = {
        {"schema", schema}, {"state", "RECEIPT_OVERSIZE"}, {"passed", false}, {"retirement_required", true}, {"detail_lost", true},
        {"error", "Receipt of " + std::to_string(fullBytes) + " bytes exceeds the " + std::to_string(receiptCap) +
                  "-byte cap; detail outside this bounded summary is lost; retire the externally owned Job"},
        {"full_receipt_bytes", fullBytes}, {"receipt_cap_bytes", receiptCap}, {"original_state", field("state")},
        {"original_error", field("error")}, {"device_created", scalar("device_created")},
        {"dispatches_submitted", scalar("dispatches_submitted")}, {"dispatches_completed_and_measured", scalar("dispatches_completed_and_measured")},
        {"failure_drain", drain}};
    std::string text;
    if (vc::receipt_detail::serialize(summary, &text, receiptCap - 1) > receiptCap - 1)
        return "{\"schema\":\"" + std::string(schema) + "\",\"state\":\"RECEIPT_OVERSIZE\",\"passed\":false,\"detail_lost\":true}\n";
    return text + "\n";
}
struct BoundedReceipt { std::string text; bool complete = false; };
inline BoundedReceipt boundedReceipt(const Json& report) {
    BoundedReceipt r;
    const uint64_t full = vc::receipt_detail::serialize(report, &r.text, receiptCap - 1) + 1;
    if (full <= receiptCap) { r.text.push_back('\n'); r.complete = true; return r; }
    try { r.text = oversizeReceipt(report, full); } catch (const std::exception&) { r.text = "{\"state\":\"RECEIPT_OVERSIZE\",\"passed\":false}\n"; }
    return r;
}

// ---------------------------------------------------------------------------------------------------
// Execution over a caller-owned Device: D3D11 on Windows, a CPU fake or translated-HLSL emulator in tests.
using Persist = std::function<void()>;
// Submission and completion are counted in the report the moment they happen, so every receipt, including
// one written from a failure path, states exactly how many dispatches reached the Device.
inline void countDispatch(Json& report, const char* field) { report[field] = report.at(field).get<uint64_t>() + 1; }
inline double profileCheck(const Json& profile, const char* shader, uint32_t groups) {
    require(profile.is_object() && profile.contains("disjoint") && profile.at("disjoint") == false && profile.contains("frequency") &&
            profile.at("frequency").is_number_unsigned() && profile.at("frequency").get<uint64_t>() > 0,
            "Invalid or disjoint GPU timing window; admission fails");
    require(profile.contains("dispatches") && profile.at("dispatches").is_array() && profile.at("dispatches").size() == 1,
            "Exactly one measured dispatch required per window; admission fails");
    const Json& d = profile.at("dispatches").front();
    require(d.is_object() && d.contains("shader") && d.at("shader") == shader && d.contains("groups") && d.at("groups") == Json(std::array<uint32_t, 3>{{groups, 1u, 1u}}),
            "Measured dispatch shader or groups differ from the planned call");
    require(d.contains("gpu_milliseconds") && d.at("gpu_milliseconds").is_number(), "Missing GPU duration; admission fails");
    const double ms = d.at("gpu_milliseconds").get<double>();
    require(std::isfinite(ms) && ms >= 0.0, "Invalid completed dispatch duration; admission fails");
    return ms;
}
// One fresh buffer, one profiled dispatch, one readback, then release, drain and zero tracked bytes. The
// same absolute deadline and cancellation are checked after every call that can block before submission.
inline std::vector<uint32_t> dispatchOnce(Device& device, const Geometry& g, const std::vector<uint32_t>& upload, const char* shader,
                                          Json& report, Json& record, const Deadline& deadline, const Persist& persist) {
    record["dispatch_call_started"] = false; record["submitted"] = false; record["completed"] = false;
    deadline.check("before_allocation");
    std::vector<uint32_t> observed;
    {
        Buffer scores = device.words(g.bufferWords, upload.data());
        require(device.trackedBufferBytes() == g.bufferBytes && g.deviceBytes <= deviceAllocationCap, "Tracked allocation differs from the case forecast");
        record["dispatch_call_started"] = true; persist();
        deadline.check("after_receipt_persistence");
        device.beginProfile();
        deadline.check("after_profile_preparation");
        const TextParams p = params(g);
        device.dispatch(shader, {}, {&scores}, &p, sizeof(p), g.groups);
        record["submitted"] = true; countDispatch(report, "dispatches_submitted");
        record["gpu_profile"] = Json::parse(device.finishProfile());
        const double ms = profileCheck(record.at("gpu_profile"), shader, g.groups);
        record["completed"] = true; record["gpu_milliseconds"] = ms; countDispatch(report, "dispatches_completed_and_measured");
        record["strictly_below_gate"] = ms < dispatchLimitMilliseconds;
        require(ms < dispatchLimitMilliseconds, "Dispatch completed at or above the 100 ms gate; no further dispatch");
        observed = device.readWords(scores);
        require(observed.size() == g.bufferWords, "Readback size differs from the allocated buffer");
    }
    device.drain();
    record["tracked_after_release"] = device.trackedBufferBytes();
    require(device.trackedBufferBytes() == 0, "Case buffer was not released after drain");
    record["observed_sha256"] = vc::sha256Words(observed);
    return observed;
}
// Repaired pass (gating), then the original control pass (recorded, never gating). Lifecycle, timing,
// tracked-byte and sensitivity failures stop either pass at once.
inline void run(Device& device, Json& report, const Deadline& deadline, const Persist& persist) {
    require(device.trackedBufferBytes() == 0, "Regression requires a device with no owned buffers");
    require(report.at("dispatches_submitted") == 0 && report.at("dispatches_completed_and_measured") == 0, "Fresh report required");
    report["cases"] = Json::array(); report["original_control"] = Json::array();
    const Persist& persisted = persist;
    std::vector<std::string> repairedHashes;
    report["phase"] = "repaired";
    for (size_t i = 0; i < cases().size(); ++i) {
        deadline.check("case_admission");
        const uint32_t index = uint32_t(i); const Geometry g = geometry(cases()[i]);
        report["cases"].push_back(caseJson(i)); Json& c = report["cases"].back();
        c["passed"] = false; c["repetitions"] = Json::array(); persisted();
        const std::vector<uint32_t> upload = uploadWords(index, g, &deadline);
        c["input_sha256"] = vc::sha256Words(upload);
        const auto oracleStart = deadline.now();
        const Oracle oracle = buildOracle(index, g, &deadline);
        c["oracle"] = {{"seconds", std::chrono::duration<double>(deadline.now() - oracleStart).count()}, {"valid_words", oracle.validWords},
                       {"maximum_relative_bound", oracle.maximumBound}, {"maximum_exponential_term_bound", oracle.maximumTermBound},
                       {"minimum_probability", oracle.minimumProbability}, {"two_value_gate_words", oracle.twoValueWords}};
        std::vector<uint32_t> first;
        for (uint32_t rep = 0; rep < repairedRepetitions; ++rep) {
            c["repetitions"].push_back(Json::object()); Json& r = c["repetitions"].back();
            std::vector<uint32_t> observed = dispatchOnce(device, g, upload, repairedShader, report, r, deadline, persisted);
            r["check"] = checkOutput(index, g, oracle, observed, &deadline);
            if (rep) r["bit_identical_to_first_repetition"] = observed == first;
            persisted();
            require(r.at("check").at("passed") == true, std::string("Repaired output check failed in case ") + cases()[i].name + "; no further dispatch");
            require(!rep || observed == first, std::string("Repaired output differs between repetitions in case ") + cases()[i].name);
            if (!rep) first = std::move(observed);
        }
        c["sensitivity"] = sensitivityControls(index, g, oracle, first, &deadline);
        persisted();
        require(c.at("sensitivity").at("all_applicable_refused") == true,
                std::string("A sensitivity control was accepted in case ") + cases()[i].name + "; the oracle is not discriminating");
        repairedHashes.push_back(c.at("repetitions").front().at("observed_sha256"));
        c["passed"] = true; persisted();
    }
    report["phase"] = "original_control";
    uint64_t visible = 0;
    for (size_t i = 0; i < cases().size(); ++i) {
        deadline.check("control_case_admission");
        const uint32_t index = uint32_t(i); const Geometry g = geometry(cases()[i]);
        report["original_control"].push_back({{"index", i}, {"name", cases()[i].name}, {"gating", false}, {"repetitions", Json::array()}});
        Json& c = report["original_control"].back(); persisted();
        const std::vector<uint32_t> upload = uploadWords(index, g, &deadline);
        require(vc::sha256Words(upload) == report.at("cases").at(i).at("input_sha256"), "Control input differs from the repaired case input");
        const Oracle oracle = buildOracle(index, g, &deadline);
        std::vector<uint32_t> first; bool rejected = false, differs = false, nondeterministic = false;
        for (uint32_t rep = 0; rep < controlRepetitions; ++rep) {
            c["repetitions"].push_back(Json::object()); Json& r = c["repetitions"].back();
            std::vector<uint32_t> observed = dispatchOnce(device, g, upload, controlShader, report, r, deadline, persisted);
            r["check"] = checkOutput(index, g, oracle, observed, &deadline);
            r["equal_to_repaired_bits"] = r.at("observed_sha256") == repairedHashes[i];
            rejected = rejected || r.at("check").at("passed") == false;
            differs = differs || r.at("equal_to_repaired_bits") == false;
            if (rep) nondeterministic = nondeterministic || observed != first; else first = std::move(observed);
            persisted();
        }
        // Visible failure means an oracle rejection or unrepeatable bits. Bits that merely differ from the
        // repaired shader while passing every gate are reported separately (codegen may legitimately differ).
        c["visible_failure"] = rejected || nondeterministic;
        c["rejected_by_oracle"] = rejected; c["differs_from_repaired"] = differs; c["differs_between_repetitions"] = nondeterministic;
        visible += c.at("visible_failure").get<bool>() ? 1u : 0u;
        persisted();
    }
    report["original_control_summary"] = {
        {"cases_with_visible_failure", visible}, {"cases", cases().size()}, {"required_to_fail", false},
        {"interpretation", visible ? "The 2c44182 shader produced a visible failure on this device and schedule sample"
                                   : "No visible failure in this hardware schedule sample; the CPU schedules establish the hazard, "
                                     "and absence of a visible failure here is not evidence that the original is correct"}};
    deadline.check("after_last_dispatch");
    report["passed"] = true;
}

// ---------------------------------------------------------------------------------------------------
// Entry. The Windows wmain supplies Win32 services; CPU tests supply fakes. Nothing here creates a Job.
struct Host {
    std::function<bool(const std::filesystem::path&)> createFresh; // CREATE_NEW: false if it exists or cannot be created.
    std::function<void(const std::string&)> rewrite;               // Whole receipt rewrite, truncate and flush; throws.
    std::function<void()> close;
    std::function<JobLimits()> queryJob;
    std::function<Json()> producer;                                // Executable commitment and C++ compiler identity.
    std::function<Json(const std::string&)> driver;                // Adapter driver identity for the pinned LUID.
    std::function<Json()> shaderCompiler;                          // Loaded HLSL compiler module commitment.
    std::function<bool()> cancelled;
    std::function<Clock::time_point()> now = [] { return Clock::now(); };
};
inline Json initialReport() {
    return {{"schema", schema}, {"generator", generatorId}, {"state", "PREPARING"}, {"passed", false}, {"device_created", false},
            {"full_model_accepted", false}, {"trained_numerics_accepted", false}, {"ocr_accepted", false},
            {"page_throughput_accepted", false}, {"blank_page_cause_assigned", false}, {"automatic_rerun", false},
            {"original_control_required_to_fail", false}, {"external_job_created_by_this_process", false},
            {"required_external_owned_job", true}, {"required_external_kill_on_job_close", true},
            {"proposed_external_process_commit_cap_bytes", processCommitCap}, {"proposed_external_job_commit_cap_bytes", jobCommitCap},
            {"required_external_cpu", "One core, below normal priority"},
            {"timing_scope", "Per-dispatch GPU timestamps bound TDR risk only; no page latency or throughput is measured"},
            {"memory_scope", "Device memory JSON reports this process's DXGI local budget/usage on the pinned adapter"},
            {"dispatches_submitted", 0}, {"dispatches_completed_and_measured", 0}, {"plan_sha256", planSha256()}, {"plan", planJson()}};
}
inline Json inactiveSummary() {
    Json names = Json::array();
    for (const auto& s : cases()) names.push_back(s.name);
    return {{"state", "INACTIVE"}, {"schema", schema}, {"device_created", false}, {"model_loaded", false}, {"native_executed", false},
            {"execution_requires", "--execute, absolute --shader-root (ChandraNative\\shaders), lowercase --pci bb:dd.f, --luid hhhhhhhh:llllllll, "
                                   "absolute fresh --output outside the shader root, and the root-owned external kill-on-close Job"},
            {"cases", names}, {"dispatches", uint64_t(cases().size()) * (repairedRepetitions + controlRepetitions)},
            {"plan_sha256", planSha256()}};
}
enum ExitCode : int { exitPassOrInactive = 0, exitFailed = 1, exitRefused = 2, exitCancelled = 3 };
inline int entry(const std::vector<std::wstring>& argv, Host& host, std::ostream& out, std::ostream& err) {
    vc::Arguments args; std::filesystem::path root, output; std::string pci, luid; Json job;
    try {
        args = vc::parseArguments(argv);
        for (size_t i = 0; i < cases().size(); ++i) geometry(cases()[i]); // The whole plan is admitted before any action.
        if (!args.execute) { out << inactiveSummary().dump() << "\n"; return exitPassOrInactive; }
        root = args.values.at(L"--shader-root"); output = args.values.at(L"--output");
        pci = vc::pinnedPci(args.values.at(L"--pci")); luid = vc::pinnedLuid(args.values.at(L"--luid"));
        admitPaths(root, output);
        job = admitJob(host.queryJob());
        if (host.cancelled()) throw Cancelled("Cancellation requested before the receipt was created; nothing was started");
        require(host.createFresh(output), "Fresh task-owned receipt required (CREATE_NEW refused the path)");
    } catch (const Cancelled& error) {
        err << Json{{"state", "CANCELLED"}, {"schema", schema}, {"device_created", false}, {"error", vc::boundedError(error.what())}}.dump() << "\n";
        return exitCancelled;
    } catch (const std::exception& error) {
        err << Json{{"state", "REFUSED"}, {"schema", schema}, {"device_created", false}, {"error", vc::boundedError(error.what())}}.dump() << "\n";
        return exitRefused;
    }
    Json report = initialReport();
    const Deadline deadline(host.now, host.cancelled);
    const Persist persist = [&] {
        const BoundedReceipt bounded = boundedReceipt(report); host.rewrite(bounded.text);
        require(bounded.complete, "Receipt exceeded the 8 MiB cap; detail lost; no further dispatch");
    };
    auto finish = [&](const char* state, bool passed) {
        report["state"] = state; report["passed"] = passed; report["elapsed_seconds"] = deadline.seconds();
        try { host.rewrite(boundedReceipt(report).text); } catch (...) {}
        try { host.close(); } catch (...) {}
    };
    try {
        report["shader_root"] = root.u8string(); report["requested_pci"] = pci; report["requested_luid"] = luid;
        report["external_job"] = job;
        persist();
        report["producer"] = host.producer();
        report["shaders_before"] = verifyFrozenShaders(root);
        report["driver_identity"] = host.driver(luid);
        require(report.at("driver_identity").is_object() && report.at("driver_identity").contains("luid") &&
                report.at("driver_identity").at("luid") == luid && report.at("driver_identity").contains("umd_driver_version") &&
                report.at("driver_identity").at("umd_driver_version").is_string() &&
                !report.at("driver_identity").at("umd_driver_version").get<std::string>().empty(),
                "Driver identity for the pinned LUID is missing or malformed; refusing before Device creation");
        persist();
        deadline.check("before_device_creation");
        {
            Device device(root.wstring(), pci, luid);
            report["device_created"] = true; persist();
            try {
                report["device_identity"] = Json::parse(device.identityJson());
                authenticateIdentity(report.at("device_identity"), pci, luid);
                report["memory_before"] = Json::parse(device.memoryJson());
                report["state"] = "RUNNING"; persist();
                run(device, report, deadline, persist);
            } catch (const std::exception& error) {
                // Persist first; then exactly one bounded completion wait; nothing is resubmitted.
                const bool cancelled = dynamic_cast<const Cancelled*>(&error) != nullptr;
                try {
                    report["state"] = cancelled ? "CANCELLED" : "FAIL_OR_INCOMPLETE"; report["passed"] = false;
                    report["error"] = vc::boundedError(error.what()); report["retirement_required"] = true; persist();
                } catch (...) {}
                try {
                    device.drain();
                    report["failure_drain"] = {{"completed", true}, {"tracked_after", device.trackedBufferBytes()}};
                } catch (const std::exception& drainError) {
                    report["failure_drain"] = {{"completed", false}, {"error", vc::boundedError(drainError.what())}};
                }
                try { persist(); } catch (...) {}
                throw;
            }
            device.drain();
            require(device.trackedBufferBytes() == 0, "Completed regression left owned buffers live");
            report["memory_after"] = Json::parse(device.memoryJson());
            report["tracked_after_final_drain"] = device.trackedBufferBytes();
        }
        report["shaders_after"] = verifyFrozenShaders(root);
        require(report.at("shaders_after") == report.at("shaders_before"), "Shader files changed during the run");
        report["shader_compiler_module"] = host.shaderCompiler();
        require(report.at("shader_compiler_module").is_object() && report.at("shader_compiler_module").contains("sha256") &&
                report.at("shader_compiler_module").at("sha256").is_string(), "Loaded HLSL compiler identity is missing");
        deadline.check("final_receipt");
        report["retirement_required"] = false;
        report["state"] = "TEXT_SOFTMAX_REGRESSION_PASS"; persist();
        finish("TEXT_SOFTMAX_REGRESSION_PASS", true);
        out << Json{{"state", "TEXT_SOFTMAX_REGRESSION_PASS"}, {"passed", true}, {"full_model_accepted", false},
                    {"trained_numerics_accepted", false}, {"page_throughput_accepted", false}}.dump() << "\n";
        return exitPassOrInactive;
    } catch (const Cancelled& error) {
        report["error"] = vc::boundedError(error.what());
        report["retirement_required"] = report.at("device_created") == true;
        finish("CANCELLED", false);
        err << "Text softmax regression CANCELLED: " << vc::boundedError(error.what()) << "\n";
        return exitCancelled;
    } catch (const std::exception& error) {
        report["error"] = vc::boundedError(error.what());
        report["retirement_required"] = report.at("device_created") == true;
        finish("FAIL_OR_INCOMPLETE", false); // An oversized report is still written as its bounded RECEIPT_OVERSIZE summary.
        err << "Text softmax regression FAIL_OR_INCOMPLETE: " << vc::boundedError(error.what()) << "\n";
        return exitFailed;
    }
}
} // namespace chandra::text_softmax_regression
