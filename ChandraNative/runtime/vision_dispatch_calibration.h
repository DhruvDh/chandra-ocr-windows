// New ChandraNative code, MPL-2.0. Portable core of the public, model-free vision attention dispatch
// calibration. Windows entry: vision_dispatch_calibration.cpp; CPU harness:
// vision_dispatch_calibration_test.cpp. No model, weight, pixel or private asset is read. Root owns the
// external Job, CPU affinity/priority, process commit cap, device admission, build and native execution.
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::vision_calibration {
using Json = nlohmann::json;
using chandra::dc::Buffer;
using chandra::dc::Device;

constexpr const char* schema = "private.chandra.directcompute.vision-dispatch-calibration.v1";
constexpr const char* generatorId = "public_lowbias32_bf16_dyadic_attention_v1";
// Production constants from vision_model.cpp/vision_model.h (VisionLimits defaults) and the shaders.
constexpr uint32_t hidden = 1024, heads = 16, headDim = 64, queryTile = 32, keyTile = 128, reduceGroup = 256;
constexpr uint32_t maximumPatches = 32768;
constexpr double dispatchLimitMilliseconds = 100.0;
constexpr double extrapolationMargin = 2.0; // Refusal-only margin before a larger phase; see docs.
constexpr uint32_t deadlineSeconds = 120;
constexpr uint64_t deviceAllocationCap = 320ull << 20;    // Tracked buffers plus one readback staging copy.
constexpr uint64_t hostVectorCap = 192ull << 20;          // Upload, readback and oracle vectors.
constexpr uint64_t receiptCap = 8ull << 20;
constexpr uint64_t processCommitCap = 768ull << 20, jobCommitCap = 1024ull << 20; // Proposed external caps.
constexpr uint32_t sentinelBits = 0x7fbadbadu; // Output storage before dispatch; signalling-NaN bits no stage can write here.
constexpr uint32_t poisonBits = 0x7fc00000u;   // Q/K/V rows the production tile must never read.
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24,
              "IEEE binary32 required");
static_assert(std::numeric_limits<double>::digits >= 53, "Independent binary64 oracle required");
#if defined(FLT_EVAL_METHOD)
static_assert(FLT_EVAL_METHOD == 0, "Each binary32 CPU operation must round to binary32");
#endif

// Production cbuffer layouts, copied byte for byte from vision_model.cpp (file-local there, so not
// exported). The CPU harness proves the calibration's words equal the unchanged production transcript.
struct AttentionParams {
    uint32_t sequenceStart, sequenceLength, queryStart, queryCount;
    uint32_t keyStart, keyCount, tileIndex, tileCount;
    uint32_t width, heads, headDim, pad;
};
struct SoftmaxParams { uint32_t queryHeads, sequenceLength, pad0, pad1; };
struct ReduceParams { uint32_t count, rowFirst, tileCount, pad; };
static_assert(sizeof(AttentionParams) == 48 && sizeof(SoftmaxParams) == 16 && sizeof(ReduceParams) == 16, "Vision HLSL ABI");
constexpr std::array<const char*, 4> stageNames{{"scores", "softmax", "values", "reduce"}};
constexpr std::array<const char*, 4> stageShaders{{"runtime/vision_attention_scores.hlsl", "runtime/vision_attention_softmax.hlsl",
                                                   "runtime/vision_attention_values.hlsl", "runtime/vision_attention_reduce.hlsl"}};

// Frozen production shader bytes (LF). Execution refuses before Device creation if the shader root differs.
struct FrozenSource { const char* relative; uint64_t bytes; const char* sha256; };
constexpr std::array<FrozenSource, 5> frozenShaders{{
    {"runtime/vision_attention_scores.hlsl", 992, "a3fd9a786a69f12882ecebd528becb66fa4fb89d8deb9592f2eb0e3b0e519e30"},
    {"runtime/vision_attention_softmax.hlsl", 1420, "257c2c1105db27c85f9f7fc8238e66793b11ac37379d2f599819a53b0c95f2dc"},
    {"runtime/vision_attention_values.hlsl", 875, "3ae108c8bef2d17dcbb214ef1ef5dd684e791d5dc35a468dc47c666da434183c"},
    {"runtime/vision_attention_reduce.hlsl", 528, "368516015a95d2e6196d2672a5f9ed6510545aa7d5a0fde72df423524ef383e1"},
    {"runtime/vision_common.hlsl", 410, "442da3bfcebe567fdb1f5f97b392557ceba0b9f71505435012dc615e565988c5"}}};

inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
// One absolute source deadline from start; it is never reset, extended or restarted.
struct Deadline {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    bool expired() const { return std::chrono::steady_clock::now() - start >= std::chrono::seconds(deadlineSeconds); }
    void check() const { require(!expired(), "120-second source deadline exceeded; retire the externally owned Job"); }
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};
inline double millisecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
inline uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result; }
inline float fromBits(uint32_t value) { float result; std::memcpy(&result, &value, sizeof(result)); return result; }
// Independent C++ statement of vision_common.hlsl vision_bf16: NaN quieted/truncated, otherwise RNE.
inline uint32_t bf16Bits(uint32_t u) {
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) return (u & 0xffff0000u) | 0x00400000u;
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}
inline bool bf16Exact(float value) { return (bits(value) & 0xffffu) == 0; }

// Portable SHA-256 (FIPS 180-4) for bit commitments over little-endian 32-bit storage words.
class Sha256 {
    uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    unsigned char block[64] = {};
    uint64_t total = 0; size_t used = 0;
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }
    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i)
            w[i] = uint32_t(block[4 * i]) << 24 | uint32_t(block[4 * i + 1]) << 16 | uint32_t(block[4 * i + 2]) << 8 | uint32_t(block[4 * i + 3]);
        for (unsigned i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
        for (unsigned i = 0; i < 64; ++i) {
            uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
public:
    void bytes(const void* data, size_t size) {
        const auto* p = static_cast<const unsigned char*>(data);
        total += size;
        while (size) {
            size_t n = std::min(size, size_t(64) - used);
            std::memcpy(block + used, p, n); used += n; p += n; size -= n;
            if (used == 64) { compress(); used = 0; }
        }
    }
    void word(uint32_t value) {
        const unsigned char le[4] = {static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8),
                                     static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 24)};
        bytes(le, 4);
    }
    void words(const std::vector<uint32_t>& values) { for (uint32_t v : values) word(v); }
    std::string hex() const {
        Sha256 copy = *this;
        const uint64_t bitLength = copy.total * 8;
        const unsigned char one = 0x80, zero = 0;
        copy.bytes(&one, 1);
        while (copy.used != 56) copy.bytes(&zero, 1);
        for (int i = 7; i >= 0; --i) { unsigned char byte = static_cast<unsigned char>(bitLength >> (8 * i)); copy.bytes(&byte, 1); }
        const char* digits = "0123456789abcdef"; std::string text;
        for (uint32_t s : copy.state) for (int i = 28; i >= 0; i -= 4) text.push_back(digits[(s >> i) & 15u]);
        return text;
    }
};
inline std::string sha256Words(const std::vector<uint32_t>& values) { Sha256 hash; hash.words(values); return hash.hex(); }

// ---------------------------------------------------------------------------------------------------
// Phases: one bounded production query tile each, ordered by increasing work. Grids are production
// gridTHW values; the tile is the production tile at queryStart within frame `frame`.
enum class Fixture { nonuniform, uniform };
struct Grid { uint32_t temporal, height, width; };
struct PhaseSpec { const char* name; std::vector<Grid> grids; uint32_t frame, queryStart; Fixture fixture; };
inline std::vector<PhaseSpec> phases() {
    return {
        {"tail_two_frame_L200_q192_nonuniform", {{1, 4, 6}, {1, 10, 20}}, 1, 192, Fixture::nonuniform},
        {"tiny_L832_q416_nonuniform", {{1, 32, 26}}, 0, 416, Fixture::nonuniform},
        {"tiny_L832_q800_uniform", {{1, 32, 26}}, 0, 800, Fixture::uniform},
        {"cli_L12096_q12064_uniform", {{1, 126, 96}}, 0, 12064, Fixture::uniform}};
}
struct Tile {
    uint32_t patches = 0, frames = 0, sequenceStart = 0, sequenceLength = 0, queryStart = 0, queries = 0;
    uint32_t queryHeads = 0, keyTiles = 0, lastTileKeys = 0, frameQueryTiles = 0;
    uint32_t firstQueryRow() const { return sequenceStart + queryStart; }
};
// Same admission as VisionModel::forecast/geometry: positive even h/w, <=32768 patches per frame and in
// total, patch count divisible by four. The tile is exactly one iteration of VisionModel::attention.
inline Tile tile(const PhaseSpec& spec) {
    Tile t; uint64_t total = 0; uint32_t index = 0; bool found = false;
    for (const auto& g : spec.grids) {
        const uint64_t size = uint64_t(g.height) * g.width;
        require(g.temporal && g.height && g.width && !(g.height % 2) && !(g.width % 2) && size <= maximumPatches,
                "Phase grid fails production vision geometry");
        for (uint32_t frame = 0; frame < g.temporal; ++frame, ++index) {
            if (index == spec.frame) { t.sequenceStart = uint32_t(total); t.sequenceLength = uint32_t(size); found = true; }
            total += size;
            require(total <= maximumPatches, "Phase patch count exceeds production admission");
        }
    }
    require(found && total && total % 4 == 0, "Phase frame absent or merge geometry invalid");
    require(spec.queryStart % queryTile == 0 && spec.queryStart < t.sequenceLength, "Query start must be a production query-tile boundary");
    t.patches = uint32_t(total); t.frames = index; t.queryStart = spec.queryStart;
    t.queries = std::min(queryTile, t.sequenceLength - spec.queryStart); t.queryHeads = t.queries * heads;
    t.keyTiles = (t.sequenceLength + keyTile - 1) / keyTile; t.lastTileKeys = t.sequenceLength - (t.keyTiles - 1) * keyTile;
    t.frameQueryTiles = (t.sequenceLength + queryTile - 1) / queryTile;
    return t;
}
inline std::array<uint32_t, 3> stageGroups(const Tile& t, size_t stage) {
    if (stage == 0 || stage == 2) return {{t.keyTiles, t.queryHeads, 1}};
    if (stage == 1) return {{t.queryHeads, 1, 1}};
    return {{(t.queries * hidden + reduceGroup - 1) / reduceGroup, 1, 1}};
}

// Structural counts from the shader source; no cache or DRAM counters are implied. "Issued" counts every
// thread load/store; "dispatch-unique" counts each distinct address once per dispatch (perfect reuse
// inside one dispatch, none between dispatches). Real off-chip bytes are unmeasured.
struct StageCounts {
    uint64_t threads = 0, multiplyAdds = 0, exponentials = 0, divisions = 0, additions = 0;
    uint64_t issuedLoadBytes = 0, issuedStoreBytes = 0, dispatchUniqueBytes = 0;
    StageCounts& operator+=(const StageCounts& o) {
        threads += o.threads; multiplyAdds += o.multiplyAdds; exponentials += o.exponentials; divisions += o.divisions;
        additions += o.additions; issuedLoadBytes += o.issuedLoadBytes; issuedStoreBytes += o.issuedStoreBytes;
        dispatchUniqueBytes += o.dispatchUniqueBytes; return *this;
    }
};
inline StageCounts stageCounts(uint64_t queries, uint64_t length, size_t stage) {
    const uint64_t qh = queries * heads, tiles = (length + keyTile - 1) / keyTile; StageCounts c;
    if (stage == 0) {
        c.threads = tiles * qh * keyTile; c.multiplyAdds = qh * length * headDim;
        c.issuedLoadBytes = (tiles * qh * headDim + qh * length * headDim) * 4; c.issuedStoreBytes = qh * length * 4;
        c.dispatchUniqueBytes = queries * hidden * 4 + length * hidden * 4 + qh * length * 4;
    } else if (stage == 1) {
        c.threads = qh * keyTile; c.exponentials = qh * length; c.divisions = qh * length; c.additions = qh * length;
        c.issuedLoadBytes = 3 * qh * length * 4; c.issuedStoreBytes = 2 * qh * length * 4; c.dispatchUniqueBytes = 2 * qh * length * 4;
    } else if (stage == 2) {
        c.threads = tiles * qh * headDim; c.multiplyAdds = qh * length * headDim;
        c.issuedLoadBytes = 2 * qh * length * headDim * 4; c.issuedStoreBytes = tiles * queries * hidden * 4;
        c.dispatchUniqueBytes = qh * length * 4 + length * hidden * 4 + tiles * queries * hidden * 4;
    } else {
        c.threads = (queries * hidden + reduceGroup - 1) / reduceGroup * reduceGroup; c.additions = tiles * queries * hidden;
        c.issuedLoadBytes = tiles * queries * hidden * 4; c.issuedStoreBytes = queries * hidden * 4;
        c.dispatchUniqueBytes = c.issuedLoadBytes + c.issuedStoreBytes;
    }
    return c;
}
inline Json countsJson(const StageCounts& c) {
    return {{"threads", c.threads}, {"multiply_adds", c.multiplyAdds}, {"exponentials", c.exponentials},
            {"divisions", c.divisions}, {"additions", c.additions}, {"issued_load_bytes", c.issuedLoadBytes},
            {"issued_store_bytes", c.issuedStoreBytes}, {"dispatch_unique_bytes", c.dispatchUniqueBytes}};
}
// Totals for one complete vision block's attention over one frame: every production query tile.
inline Json blockCountsJson(uint32_t length) {
    Json stages = Json::object(); StageCounts all; uint64_t tiles = 0;
    std::array<StageCounts, 4> sums{};
    for (uint32_t first = 0; first < length; first += queryTile, ++tiles)
        for (size_t s = 0; s < 4; ++s) sums[s] += stageCounts(std::min(queryTile, length - first), length, s);
    for (size_t s = 0; s < 4; ++s) { stages[stageNames[s]] = countsJson(sums[s]); all += sums[s]; }
    return {{"sequence_length", length}, {"query_tiles", tiles}, {"dispatches", 4 * tiles},
            {"attention_multiply_adds", all.multiplyAdds}, {"stages", stages}, {"total", countsJson(all)}};
}

struct Footprint {
    uint64_t q = 0, k = 0, v = 0, result = 0, scores = 0, partials = 0, deviceTracked = 0, stagingMaximum = 0;
    uint64_t deviceWithStaging = 0, uploadVector = 0, readbackVectors = 0, oracleScratch = 0, hostPeak = 0;
};
inline Footprint footprint(const Tile& t) {
    Footprint f;
    f.q = f.k = f.v = f.result = uint64_t(t.patches) * hidden * 4;
    f.scores = uint64_t(t.queryHeads) * t.sequenceLength * 4;
    f.partials = uint64_t(t.queries) * hidden * t.keyTiles * 4;
    f.deviceTracked = f.q + f.k + f.v + f.result + f.scores + f.partials;
    f.stagingMaximum = std::max({f.scores, f.partials, f.result});
    f.deviceWithStaging = f.deviceTracked + f.stagingMaximum;
    f.uploadVector = std::max({f.q, f.scores, f.partials});
    // Observed scores, probabilities, partials and the complete result are held together at the final check.
    f.readbackVectors = 2 * f.scores + f.partials + f.result;
    // Key integer table, two Q x L double rows (probabilities/bounds) and small per-tile accumulators.
    f.oracleScratch = uint64_t(t.sequenceLength) * hidden + 2ull * t.queries * t.sequenceLength * 8 +
                      uint64_t(t.queries) * hidden * 4 + 3ull * t.queries * headDim * 8;
    f.hostPeak = std::max(f.uploadVector + uint64_t(t.sequenceLength) * hidden, f.readbackVectors + f.oracleScratch);
    return f;
}
inline Json footprintJson(const Footprint& f) {
    return {{"device_buffers", {{"q", f.q}, {"k", f.k}, {"v", f.v}, {"result", f.result}, {"scores", f.scores}, {"partials", f.partials}}},
            {"device_tracked_bytes", f.deviceTracked}, {"maximum_readback_staging_bytes", f.stagingMaximum},
            {"device_tracked_plus_staging_bytes", f.deviceWithStaging}, {"host_upload_vector_bytes", f.uploadVector},
            {"host_readback_vectors_bytes", f.readbackVectors}, {"host_oracle_scratch_bytes", f.oracleScratch},
            {"host_vector_peak_bytes", f.hostPeak}};
}

// ---------------------------------------------------------------------------------------------------
// Public generator v1. lowbias32 (C. Wellons) chained over (tag, row, column). All Q/K values are
// integer/4 with |integer| <= 8, so they are BF16-exact like production RoPE outputs; every product is
// integer/16 and every partial dot sum stays below 4096/16, so no FP32 operation in the 64-term score
// rounds in any order and x0.125 is exact. V values are +/-m*2^-s, m in 1..15, s in 3..8 (BF16-exact).
inline uint32_t lowbias32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
inline uint32_t mix(uint32_t tag, uint32_t a, uint32_t b) { return lowbias32(lowbias32(lowbias32(tag) ^ a) ^ b); }
constexpr uint32_t tagQ = 0x51755259u, tagK = 0x4b657953u, tagV = 0x56616c73u, tagG = 0x47616e67u, tagB = 0x42617365u, tagD = 0x44697666u;
inline int nonuniformQ(uint32_t row, uint32_t col) { return int(mix(tagQ, row, col) % 17u) - 8; }
inline int nonuniformK(uint32_t row, uint32_t col) { return int(mix(tagK, row, col) % 17u) - 8; }
// Uniform fixture: Q repeats g within each column pair; K pairs are (b+d, b-d) with b per head/pair and
// d varying per key row. Every key of a query row/head therefore scores sum(g*2b)/128, a constant that
// differs between rows; if the raw constant is zero, g of pair 0 is raised by one (b of pair 0 is 1 or 2).
inline int uniformPairQ(uint32_t row, uint32_t head, uint32_t pair) { return int(mix(tagG, row, head * 32 + pair) % 9u) - 4; }
inline int uniformPairBase(uint32_t head, uint32_t pair) {
    return pair == 0 ? 1 + int(mix(tagB, head, 0) % 2u) : int(mix(tagB, head, pair) % 5u) - 2;
}
inline int uniformPairSplit(uint32_t row, uint32_t head, uint32_t pair) { return int(mix(tagD, row, head * 32 + pair) % 9u) - 4; }
inline bool uniformRaised(uint32_t row, uint32_t head) {
    int sum = 0;
    for (uint32_t pair = 0; pair < 32; ++pair) sum += uniformPairQ(row, head, pair) * 2 * uniformPairBase(head, pair);
    return sum == 0;
}
inline int uniformScoreInteger(uint32_t row, uint32_t head) {
    int sum = 0;
    for (uint32_t pair = 0; pair < 32; ++pair) sum += uniformPairQ(row, head, pair) * 2 * uniformPairBase(head, pair);
    return sum == 0 ? 2 * uniformPairBase(head, 0) : sum;
}
inline int uniformQ(uint32_t row, uint32_t col) {
    const uint32_t head = col / headDim, pair = (col % headDim) / 2;
    return uniformPairQ(row, head, pair) + (pair == 0 && uniformRaised(row, head) ? 1 : 0);
}
inline int uniformK(uint32_t row, uint32_t col) {
    const uint32_t head = col / headDim, pair = (col % headDim) / 2;
    const int split = uniformPairSplit(row, head, pair);
    return uniformPairBase(head, pair) + (col % 2 ? -split : split);
}
inline int queryInteger(Fixture f, uint32_t row, uint32_t col) { return f == Fixture::uniform ? uniformQ(row, col) : nonuniformQ(row, col); }
inline int keyInteger(Fixture f, uint32_t row, uint32_t col) { return f == Fixture::uniform ? uniformK(row, col) : nonuniformK(row, col); }
inline float valueV(uint32_t row, uint32_t col) {
    const uint32_t r = mix(tagV, row, col);
    const float magnitude = std::ldexp(float(1 + r % 15u), -int(3 + (r >> 8) % 6u));
    return (r >> 16) & 1u ? -magnitude : magnitude;
}
inline bool queryRow(const Tile& t, uint32_t row) { return row >= t.firstQueryRow() && row < t.firstQueryRow() + t.queries; }
inline bool frameRow(const Tile& t, uint32_t row) { return row >= t.sequenceStart && row < t.sequenceStart + t.sequenceLength; }

// ---------------------------------------------------------------------------------------------------
// Numerical screens (derived in docs/directcompute-vision-calibration.md). u is the binary32 unit
// roundoff. Assumptions, each named: FP32 add/sub/mul round to nearest even with no contraction under
// `precise` (D3D11 rule; the GEMV calibration's order-sensitive phase passed natively under it); HLSL
// exp lowers to 2^(x*fl32(log2 e)); the exp2 instruction's relative error is at most exp2RelativeError;
// FP32 division is within 2.5 ULP (D3D11). Screens are a priori bounds, never fitted to observations.
constexpr double unitRoundoff = 0x1p-24;
constexpr double exp2RelativeError = 0x1p-21;
constexpr double divisionRelativeError = 2.5 * 0x1p-23;
constexpr double bf16RelativeHalfUlp = 0x1p-8;
constexpr double oracleRelativeMargin = 0x1p-36; // Binary64 oracle error allowance, far below every screen.
inline double gammaBound(uint64_t n) { const double nu = double(n) * unitRoundoff; return nu / (1.0 - nu); }
// Relative error of one computed exponential at exact argument x: rounding of x*c (c=fl32(log2 e)) shifts
// the 2^t argument by at most |x|log2(e)(2u+u^2); exp2 adds exp2RelativeError.
inline double exponentialRelativeBound(double absX) {
    const double tau = absX * 1.4426950408889634 * (2 * unitRoundoff + unitRoundoff * unitRoundoff);
    return std::exp2(tau) * (1 + exp2RelativeError) - 1;
}
// Per-term roundings in the softmax sum: lane-sequential sum from +0 then a seven-level tree.
inline uint64_t softmaxSumRoundings(uint32_t length) { return (length + keyTile - 1) / keyTile - 1 + 7; }
inline double probabilityRelativeBound(double termBound, double rowMaximumBound, uint32_t length) {
    const double g = gammaBound(softmaxSumRoundings(length)), phi = divisionRelativeError;
    const double upper = (1 + termBound) * (1 + phi) / ((1 - rowMaximumBound) * (1 - g)) - 1;
    const double lower = 1 - (1 - termBound) * (1 - phi) / ((1 + rowMaximumBound) * (1 + g));
    return std::max(upper, lower);
}
// Product rounding, <=127 in-tile additions after the exact first and <=T-1 tile additions.
inline uint64_t outputRoundings(uint32_t length) { return 128 + (length + keyTile - 1) / keyTile; }

// D3D11 binary32 arithmetic for conditional exact expectations: subnormal operands/results flush to
// sign-preserved zero. The fixtures never produce subnormals; flushes are counted and reported.
struct Fp32 {
    uint64_t flushes = 0;
    float ftz(float x) { if (x != 0.0f && std::fabs(x) < FLT_MIN) { ++flushes; return std::copysign(0.0f, x); } return x; }
    float mul(float a, float b) { const float p = ftz(a) * ftz(b); return ftz(p); }
    float add(float a, float b) { const float s = ftz(a) + ftz(b); return ftz(s); }
};

struct ExactCheck {
    Sha256 expectedHash, observedHash; uint64_t elements = 0, mismatches = 0; Json first = Json::array();
    void add(uint64_t index, uint32_t expected, uint32_t observed) {
        expectedHash.word(expected); observedHash.word(observed); ++elements;
        if (expected != observed && ++mismatches <= 16)
            first.push_back({{"index", index}, {"expected_bits", expected}, {"observed_bits", observed}});
    }
    Json json(const char* rule) const {
        return {{"kind", "exact"}, {"rule", rule}, {"elements", elements}, {"mismatches", mismatches}, {"first_mismatches", first},
                {"expected_sha256", expectedHash.hex()}, {"observed_sha256", observedHash.hex()}, {"passed", mismatches == 0}};
    }
};
struct ScreenCheck {
    Sha256 observedHash; uint64_t elements = 0, violations = 0, equalToRoundedOracle = 0;
    double maximumRelative = 0, maximumRatio = 0; Json first = Json::array();
    void add(uint64_t index, uint32_t observed, double oracle, double allowed) {
        observedHash.word(observed); ++elements;
        const double value = double(fromBits(observed)), error = std::fabs(value - oracle);
        equalToRoundedOracle += observed == bits(float(oracle)) ? 1u : 0u;
        const bool ok = std::isfinite(value) && error <= allowed;
        if (std::isfinite(value)) {
            if (oracle != 0) maximumRelative = std::max(maximumRelative, error / std::fabs(oracle));
            if (allowed > 0) maximumRatio = std::max(maximumRatio, error / allowed);
        }
        if (!ok && ++violations <= 16)
            first.push_back({{"index", index}, {"observed_bits", observed}, {"oracle", oracle}, {"allowed_absolute_error", allowed}});
    }
    Json json(const char* rule) const {
        return {{"kind", "screen"}, {"rule", rule}, {"elements", elements}, {"violations", violations}, {"first_violations", first},
                {"maximum_relative_error", maximumRelative}, {"maximum_error_to_screen_ratio", maximumRatio},
                {"observed_equal_to_rounded_binary64", equalToRoundedOracle},
                {"observed_sha256", observedHash.hex()}, {"passed", violations == 0}};
    }
};

// Key integers for the frame, [key][1024], built once per phase for the score oracle.
inline std::vector<int8_t> keyTable(const PhaseSpec& spec, const Tile& t, const Deadline& deadline) {
    std::vector<int8_t> table(size_t(t.sequenceLength) * hidden);
    for (uint32_t key = 0; key < t.sequenceLength; ++key) {
        if (!(key % 256)) deadline.check();
        for (uint32_t col = 0; col < hidden; ++col)
            table[size_t(key) * hidden + col] = static_cast<int8_t>(keyInteger(spec.fixture, t.sequenceStart + key, col));
    }
    return table;
}
// Exact: every complete selected-tile score is an integer dot/128 computed from generator integers.
inline Json checkScores(const PhaseSpec& spec, const Tile& t, const std::vector<int8_t>& keys,
                        const std::vector<uint32_t>& observed, const Deadline& deadline) {
    require(observed.size() == size_t(t.queryHeads) * t.sequenceLength, "Incomplete scores readback");
    ExactCheck check; int minimum = std::numeric_limits<int>::max(), maximum = std::numeric_limits<int>::min();
    for (uint32_t qh = 0; qh < t.queryHeads; ++qh) {
        deadline.check();
        const uint32_t qr = qh / heads, h = qh % heads, row = t.firstQueryRow() + qr;
        int query[headDim];
        for (uint32_t c = 0; c < headDim; ++c) query[c] = queryInteger(spec.fixture, row, h * headDim + c);
        const int uniformScore = spec.fixture == Fixture::uniform ? uniformScoreInteger(row, h) : 0;
        for (uint32_t key = 0; key < t.sequenceLength; ++key) {
            const int8_t* k = keys.data() + size_t(key) * hidden + h * headDim;
            int sum = 0;
            for (uint32_t c = 0; c < headDim; ++c) sum += query[c] * k[c];
            if (spec.fixture == Fixture::uniform && sum != uniformScore)
                throw std::logic_error("Uniform fixture identity failed in source generator");
            minimum = std::min(minimum, sum); maximum = std::max(maximum, sum);
            const float expected = float(sum) / 128.0f;
            require(double(expected) * 128.0 == double(sum), "Generated score is not exactly FP32");
            check.add(uint64_t(qh) * t.sequenceLength + key, bits(expected), observed[size_t(qh) * t.sequenceLength + key]);
        }
    }
    Json result = check.json("bit-exact: integer dot of BF16-exact dyadic Q/K over 64 columns, times 0.125");
    result["score_integer_range"] = {minimum, maximum};
    return result;
}
// Binary64 softmax for the Q rows of one head, plus each probability's a priori relative bound.
struct HeadOracle { std::vector<double> p, bound; };
inline void softmaxOracle(const Tile& t, uint32_t h, const std::vector<uint32_t>& scores, HeadOracle& out) {
    const size_t length = t.sequenceLength;
    out.p.assign(size_t(t.queries) * length, 0.0); out.bound.assign(out.p.size(), 0.0);
    for (uint32_t qr = 0; qr < t.queries; ++qr) {
        const uint32_t* row = scores.data() + size_t(qr * heads + h) * length;
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < length; ++k) maximum = std::max(maximum, double(fromBits(row[k])));
        double sum = 0, compensation = 0, rowBound = 0;
        double* p = out.p.data() + size_t(qr) * length; double* bound = out.bound.data() + size_t(qr) * length;
        for (size_t k = 0; k < length; ++k) {
            const double x = double(fromBits(row[k])) - maximum;
            require(double(float(x)) == x, "Score minus row maximum is not exactly FP32");
            p[k] = std::exp(x); bound[k] = exponentialRelativeBound(std::fabs(x)); rowBound = std::max(rowBound, bound[k]);
            const double next = sum + p[k]; // Neumaier compensated binary64 sum.
            compensation += std::fabs(sum) >= p[k] ? (sum - next) + p[k] : (p[k] - next) + sum; sum = next;
        }
        sum += compensation;
        for (size_t k = 0; k < length; ++k) { p[k] /= sum; bound[k] = probabilityRelativeBound(bound[k], rowBound, t.sequenceLength); }
    }
}
inline Json checkProbabilities(const PhaseSpec& spec, const Tile& t, const std::vector<uint32_t>& scores,
                               const std::vector<uint32_t>& observed, const Deadline& deadline) {
    require(observed.size() == scores.size(), "Incomplete probability readback");
    ScreenCheck check; HeadOracle oracle; double maximumBound = 0;
    for (uint32_t h = 0; h < heads; ++h) {
        deadline.check();
        softmaxOracle(t, h, scores, oracle);
        for (uint32_t qr = 0; qr < t.queries; ++qr)
            for (uint32_t k = 0; k < t.sequenceLength; ++k) {
                const size_t o = size_t(qr) * t.sequenceLength + k, index = size_t(qr * heads + h) * t.sequenceLength + k;
                maximumBound = std::max(maximumBound, oracle.bound[o]);
                check.add(index, observed[index], oracle.p[o], (oracle.bound[o] + oracleRelativeMargin) * oracle.p[o]);
            }
    }
    Json result = check.json("|p_observed - p_binary64| <= (B_k + 2^-36) p_binary64; B_k from exp/sum/division bounds");
    result["maximum_relative_screen"] = maximumBound;
    result["sum_roundings_per_term"] = softmaxSumRoundings(t.sequenceLength);
    if (spec.fixture == Fixture::uniform) {
        // Exact algebraic property: identical exponent arguments give identical probabilities everywhere.
        uint64_t differing = 0;
        for (uint32_t word : observed) differing += word != observed.front() ? 1u : 0u;
        const float reciprocal = 1.0f / float(t.sequenceLength);
        const int64_t distance = int64_t(observed.front()) - int64_t(bits(reciprocal));
        result["uniform"] = {{"rule", "bit-exact: every probability word identical"}, {"differing_words", differing},
                             {"passed", differing == 0}, {"probability_bits", observed.front()},
                             {"correctly_rounded_reciprocal_bits", bits(reciprocal)},
                             {"ulp_distance_from_correctly_rounded_reciprocal",
                              distance < 0 ? -distance : distance}};
        if (differing) result["passed"] = false;
    }
    return result;
}
// Exact, conditional on the observed probabilities: production values order (ascending key inside
// each 128-key tile, FP32 multiply then add from +0) under D3D11 binary32 rules.
inline Json checkPartials(const Tile& t, const std::vector<uint32_t>& probabilities, const std::vector<uint32_t>& observed,
                          const Deadline& deadline) {
    const size_t perTile = size_t(t.queries) * hidden;
    require(observed.size() == perTile * t.keyTiles, "Incomplete partial readback");
    ExactCheck check; Fp32 fp; std::vector<float> expected(perTile); float value[headDim];
    for (uint32_t tile = 0; tile < t.keyTiles; ++tile) {
        deadline.check();
        std::fill(expected.begin(), expected.end(), 0.0f);
        const uint32_t keys = std::min(keyTile, t.sequenceLength - tile * keyTile);
        for (uint32_t h = 0; h < heads; ++h)
            for (uint32_t j = 0; j < keys; ++j) {
                const uint32_t key = tile * keyTile + j;
                for (uint32_t c = 0; c < headDim; ++c) value[c] = valueV(t.sequenceStart + key, h * headDim + c);
                for (uint32_t qr = 0; qr < t.queries; ++qr) {
                    const float p = fromBits(probabilities[size_t(qr * heads + h) * t.sequenceLength + key]);
                    float* sum = expected.data() + size_t(qr) * hidden + h * headDim;
                    for (uint32_t c = 0; c < headDim; ++c) sum[c] = fp.add(sum[c], fp.mul(p, value[c]));
                }
            }
        for (size_t i = 0; i < perTile; ++i) check.add(tile * perTile + i, bits(expected[i]), observed[tile * perTile + i]);
    }
    Json result = check.json("bit-exact given observed probabilities: ascending FP32 p*v from +0 per 128-key tile");
    result["d3d11_subnormal_flushes"] = fp.flushes;
    return result;
}
// Exact, conditional on the observed partials: ascending FP32 tile sum from +0, then vision_bf16. Rows
// outside the tile must still hold the sentinel, so the complete result buffer is compared.
inline Json checkResult(const Tile& t, const std::vector<uint32_t>& partials, const std::vector<uint32_t>& observed,
                        const Deadline& deadline) {
    const size_t count = size_t(t.queries) * hidden;
    require(observed.size() == size_t(t.patches) * hidden && partials.size() == count * t.keyTiles, "Incomplete result readback");
    ExactCheck check; Fp32 fp; uint64_t untouched = 0;
    for (size_t index = 0; index < observed.size(); ++index) {
        if (!(index % (1u << 20))) deadline.check();
        const uint32_t row = uint32_t(index / hidden);
        uint32_t expected = sentinelBits;
        if (row >= t.firstQueryRow() && row < t.firstQueryRow() + t.queries) {
            const size_t x = index - size_t(t.firstQueryRow()) * hidden; float sum = 0.0f;
            for (uint32_t tile = 0; tile < t.keyTiles; ++tile) sum = fp.add(sum, fromBits(partials[tile * count + x]));
            expected = bf16Bits(bits(sum));
        } else {
            untouched += observed[index] == sentinelBits ? 1u : 0u;
        }
        check.add(index, expected, observed[index]);
    }
    Json result = check.json("bit-exact given observed partials: ascending FP32 tile sum from +0, vision_bf16; other rows keep sentinel");
    result["rows_outside_tile_with_sentinel_words"] = untouched;
    result["d3d11_subnormal_flushes"] = fp.flushes;
    return result;
}
// Independent end to end: binary64 attention from the exact scores, against the stored BF16 tile.
inline Json checkOutputScreen(const Tile& t, const std::vector<uint32_t>& scores, const std::vector<uint32_t>& observed,
                              const Deadline& deadline) {
    ScreenCheck check; HeadOracle oracle; const double g = gammaBound(outputRoundings(t.sequenceLength));
    std::vector<double> o(size_t(t.queries) * headDim), a(o.size()), e(o.size()); float value[headDim];
    uint64_t equalToBf16OfBinary64 = 0;
    for (uint32_t h = 0; h < heads; ++h) {
        deadline.check();
        softmaxOracle(t, h, scores, oracle);
        std::fill(o.begin(), o.end(), 0.0); std::fill(a.begin(), a.end(), 0.0); std::fill(e.begin(), e.end(), 0.0);
        for (uint32_t key = 0; key < t.sequenceLength; ++key) {
            for (uint32_t c = 0; c < headDim; ++c) value[c] = valueV(t.sequenceStart + key, h * headDim + c);
            for (uint32_t qr = 0; qr < t.queries; ++qr) {
                const double p = oracle.p[size_t(qr) * t.sequenceLength + key], b = oracle.bound[size_t(qr) * t.sequenceLength + key];
                for (uint32_t c = 0; c < headDim; ++c) {
                    const size_t i = size_t(qr) * headDim + c; const double v = double(value[c]);
                    o[i] += p * v; a[i] += p * std::fabs(v); e[i] += b * p * std::fabs(v);
                }
            }
        }
        for (uint32_t qr = 0; qr < t.queries; ++qr)
            for (uint32_t c = 0; c < headDim; ++c) {
                const size_t i = size_t(qr) * headDim + c, index = size_t(t.firstQueryRow() + qr) * hidden + h * headDim + c;
                // |Ohat - O| <= E = sum B p|v| + gamma (sum p|v| + sum B p|v|); BF16 RNE adds <= 2^-8 |Ohat|.
                const double bound = e[i] + g * (a[i] + e[i]);
                const double allowed = bf16RelativeHalfUlp * (std::fabs(o[i]) + bound) + bound + oracleRelativeMargin * a[i];
                check.add(index, observed[index], o[i], allowed);
                equalToBf16OfBinary64 += observed[index] == bf16Bits(bits(float(o[i]))) ? 1u : 0u;
            }
    }
    Json result = check.json("|bf16_observed - O_binary64| <= 2^-8(|O|+E) + E + 2^-36 sum p|v|; E from probability and FP32 accumulation bounds");
    result["observed_equal_to_bf16_of_rounded_binary64"] = equalToBf16OfBinary64;
    result["accumulation_roundings_per_term"] = outputRoundings(t.sequenceLength);
    return result;
}

// ---------------------------------------------------------------------------------------------------
// One production tile, split into four individually profiled stages. Every dispatch argument, binding
// and group count equals one iteration of VisionModel::attention (vision_model.cpp).
struct TileBuffers { Buffer q, k, v, result, scores, partials; };
inline void dispatchStage(Device& d, const Tile& t, TileBuffers& b, size_t stage) {
    AttentionParams p{t.sequenceStart, t.sequenceLength, t.queryStart, t.queries, 0, 0, 0, t.keyTiles, hidden, heads, headDim, 0};
    if (stage == 0) d.dispatch(stageShaders[0], {&b.q, &b.k}, {&b.scores}, &p, sizeof(p), t.keyTiles, t.queryHeads);
    else if (stage == 1) {
        SoftmaxParams softmax{t.queryHeads, t.sequenceLength, 0, 0};
        d.dispatch(stageShaders[1], {}, {&b.scores}, &softmax, sizeof(softmax), t.queryHeads);
    } else if (stage == 2) d.dispatch(stageShaders[2], {&b.scores, &b.v}, {&b.partials}, &p, sizeof(p), t.keyTiles, t.queryHeads);
    else {
        ReduceParams reduce{t.queries * hidden, t.sequenceStart + t.queryStart, t.keyTiles, 0};
        d.dispatch(stageShaders[3], {&b.partials}, {&b.result}, &reduce, sizeof(reduce), (reduce.count + reduceGroup - 1) / reduceGroup);
    }
}
inline double profileCheck(const Json& profile, size_t stage, const std::array<uint32_t, 3>& groups) {
    require(profile.is_object() && profile.contains("disjoint") && profile.at("disjoint") == false &&
            profile.contains("frequency") && profile.at("frequency").is_number_unsigned() &&
            profile.at("frequency").get<uint64_t>() > 0, "Invalid or disjoint GPU timing window; admission fails");
    const auto& dispatches = profile.at("dispatches");
    require(dispatches.is_array() && dispatches.size() == 1, "Exactly one measured dispatch required per stage window; admission fails");
    const auto& dispatch = dispatches.front();
    require(dispatch.at("shader") == stageShaders[stage] && dispatch.at("groups") == Json(groups),
            "Measured dispatch shader or geometry differs from the production stage");
    require(dispatch.at("gpu_milliseconds").is_number(), "Missing GPU duration; admission fails");
    const double ms = dispatch.at("gpu_milliseconds").get<double>();
    require(std::isfinite(ms) && ms >= 0.0, "Invalid completed dispatch duration; admission fails");
    return ms;
}
inline void memoryCheck(Device& device, const Deadline& deadline, const Footprint& f) {
    deadline.check();
    require(device.trackedBufferBytes() <= f.deviceTracked && f.deviceWithStaging <= deviceAllocationCap,
            "Calibration device allocation forecast exceeded");
}
inline Buffer upload(Device& device, std::vector<uint32_t>& host, size_t words, const std::function<uint32_t(size_t)>& word,
                     Json& commitments, const char* name, const Deadline& deadline) {
    require(words && uint64_t(words) * 4 <= hostVectorCap && words <= std::numeric_limits<uint32_t>::max(), "Host upload vector exceeds cap");
    host.resize(words); Sha256 hash;
    for (size_t i = 0; i < words; ++i) { if (!(i % (1u << 20))) deadline.check(); host[i] = word(i); hash.word(host[i]); }
    commitments[name] = {{"words", words}, {"sha256", hash.hex()}};
    deadline.check(); // Immediately before the Device allocation.
    return device.words(uint32_t(words), host.data());
}
inline TileBuffers allocate(Device& device, const PhaseSpec& spec, const Tile& t, Json& commitments, const Deadline& deadline) {
    std::vector<uint32_t> host; TileBuffers b; const size_t global = size_t(t.patches) * hidden;
    b.q = upload(device, host, global, [&](size_t i) {
        const uint32_t row = uint32_t(i / hidden), col = uint32_t(i % hidden);
        return queryRow(t, row) ? bits(float(queryInteger(spec.fixture, row, col)) / 4.0f) : poisonBits;
    }, commitments, "q", deadline);
    b.k = upload(device, host, global, [&](size_t i) {
        const uint32_t row = uint32_t(i / hidden), col = uint32_t(i % hidden);
        return frameRow(t, row) ? bits(float(keyInteger(spec.fixture, row, col)) / 4.0f) : poisonBits;
    }, commitments, "k", deadline);
    b.v = upload(device, host, global, [&](size_t i) {
        const uint32_t row = uint32_t(i / hidden), col = uint32_t(i % hidden);
        return frameRow(t, row) ? bits(valueV(row, col)) : poisonBits;
    }, commitments, "v", deadline);
    auto sentinel = [](size_t) { return sentinelBits; };
    b.result = upload(device, host, global, sentinel, commitments, "result_sentinel", deadline);
    b.scores = upload(device, host, size_t(t.queryHeads) * t.sequenceLength, sentinel, commitments, "scores_sentinel", deadline);
    b.partials = upload(device, host, size_t(t.queries) * hidden * t.keyTiles, sentinel, commitments, "partials_sentinel", deadline);
    return b; // The host vector is released here, before any readback.
}
inline void requireInputsExact(const PhaseSpec& spec, const Tile& t) {
    for (uint32_t col = 0; col < hidden; ++col) {
        const float q = float(queryInteger(spec.fixture, t.firstQueryRow(), col)) / 4.0f;
        const float k = float(keyInteger(spec.fixture, t.sequenceStart, col)) / 4.0f, v = valueV(t.sequenceStart, col);
        require(bf16Exact(q) && bf16Exact(k) && bf16Exact(v) && v != 0.0f, "Generator value is not a nonzero BF16-exact input");
    }
}
using Persist = std::function<void()>;
// Last host checks before a stage submission. Receipt persistence and beginProfile (whose drain can wait
// for the Device timeout) can block, so the same absolute deadline is enforced after each. A refused stage
// records where it stopped and is never dispatched; the caller's single bounded failure drain follows.
inline void admitDispatch(const Deadline& deadline, Json& stage, size_t s, const char* boundary) {
    if (!deadline.expired()) return;
    stage["dispatch_call_started"] = false; stage["refused_before_dispatch"] = std::string("deadline_expired_") + boundary;
    throw std::runtime_error(std::string("120-second source deadline exceeded ") + boundary + "; stage " + stageNames[s] +
                             " not dispatched; retire the externally owned Job");
}
// Refusal-only admission of a larger phase: extrapolate each stage linearly in its work units from the
// completed phase with the most units, times the declared margin, and refuse at or above the gate.
inline Json extrapolationAdmission(const Json& completed, const Tile& next) {
    Json stages = Json::array(); bool admitted = true;
    for (size_t s = 0; s < 4; ++s) {
        const StageCounts want = stageCounts(next.queries, next.sequenceLength, s);
        const double units = double(s == 1 ? want.exponentials : s == 3 ? want.additions : want.multiplyAdds);
        double bestUnits = 0, bestMs = 0; std::string from;
        for (const auto& phase : completed) {
            const auto& stage = phase.at("stages").at(s);
            const double have = double(s == 1 ? stage.at("counts").at("exponentials").get<uint64_t>() :
                                       s == 3 ? stage.at("counts").at("additions").get<uint64_t>() :
                                                stage.at("counts").at("multiply_adds").get<uint64_t>());
            // The latest of equal-work references wins.
            if (have >= bestUnits) { bestUnits = have; bestMs = stage.at("gpu_milliseconds").get<double>(); from = phase.at("name"); }
        }
        require(bestUnits > 0, "No completed reference measurement for extrapolation");
        const double predicted = bestMs * units / bestUnits * extrapolationMargin;
        admitted = admitted && predicted < dispatchLimitMilliseconds;
        stages.push_back({{"stage", stageNames[s]}, {"reference_phase", from}, {"reference_gpu_milliseconds", bestMs},
                          {"work_ratio", units / bestUnits}, {"margin", extrapolationMargin}, {"predicted_milliseconds", predicted}});
    }
    return {{"stages", stages}, {"admitted", admitted}};
}
inline void runPhase(Device& device, const PhaseSpec& spec, Json& report, Json& phase, const Deadline& deadline, const Persist& persist) {
    const Tile t = tile(spec); const Footprint f = footprint(t);
    require(f.deviceWithStaging <= deviceAllocationCap && f.hostPeak <= hostVectorCap, "Phase forecast exceeds declared caps");
    requireInputsExact(spec, t);
    const double phaseStart = deadline.seconds(); double oracleSeconds = 0;
    {
        const std::vector<int8_t> keys = keyTable(spec, t, deadline);
        phase["input_commitments"] = Json::object();
        TileBuffers b = allocate(device, spec, t, phase["input_commitments"], deadline);
        memoryCheck(device, deadline, f);
        phase["memory_before"] = Json::parse(device.memoryJson());
        phase["upload_seconds"] = deadline.seconds() - phaseStart;
        std::vector<uint32_t> scores, probabilities, partials, result;
        for (size_t s = 0; s < 4; ++s) {
            auto& stage = phase["stages"][s];
            stage["dispatch_call_started"] = true; persist();
            admitDispatch(deadline, stage, s, "after_receipt_persistence");
            device.beginProfile();
            admitDispatch(deadline, stage, s, "after_profile_preparation");
            const auto submit = std::chrono::steady_clock::now();
            dispatchStage(device, t, b, s);
            stage["submitted"] = true; report["dispatches_submitted"] = report["dispatches_submitted"].get<uint64_t>() + 1;
            stage["gpu_profile"] = Json::parse(device.finishProfile());
            stage["host_submit_to_drained_milliseconds"] = millisecondsSince(submit);
            const double ms = profileCheck(stage.at("gpu_profile"), s, stageGroups(t, s));
            stage["completed"] = true; stage["gpu_milliseconds"] = ms;
            report["dispatches_completed_and_measured"] = report["dispatches_completed_and_measured"].get<uint64_t>() + 1;
            const StageCounts c = stageCounts(t.queries, t.sequenceLength, s);
            if (ms > 0) stage["derived"] = {{"giga_multiply_adds_per_second", double(c.multiplyAdds) / ms * 1e-6},
                                            {"dispatch_unique_gigabytes_per_second", double(c.dispatchUniqueBytes) / ms * 1e-6},
                                            {"issued_gigabytes_per_second", double(c.issuedLoadBytes + c.issuedStoreBytes) / ms * 1e-6}};
            stage["strictly_below_gate"] = ms < dispatchLimitMilliseconds;
            memoryCheck(device, deadline, f);
            require(ms < dispatchLimitMilliseconds, std::string("Stage ") + stageNames[s] +
                    " completed at or above the 100 ms gate; no further dispatch");
            const auto readStart = std::chrono::steady_clock::now();
            const Buffer& source = s <= 1 ? b.scores : s == 2 ? b.partials : b.result;
            std::vector<uint32_t> observed = device.readWords(source);
            stage["host_readback_milliseconds"] = millisecondsSince(readStart);
            const auto checkStart = std::chrono::steady_clock::now();
            if (s == 0) { stage["check"] = checkScores(spec, t, keys, observed, deadline); scores = std::move(observed); }
            else if (s == 1) { stage["check"] = checkProbabilities(spec, t, scores, observed, deadline); probabilities = std::move(observed); }
            else if (s == 2) { stage["check"] = checkPartials(t, probabilities, observed, deadline); partials = std::move(observed); }
            else {
                stage["check"] = checkResult(t, partials, observed, deadline);
                if (stage["check"].at("passed") == true) phase["output_screen"] = checkOutputScreen(t, scores, observed, deadline);
                result = std::move(observed);
            }
            oracleSeconds += millisecondsSince(checkStart) / 1000.0;
            persist();
            require(stage["check"].at("passed") == true, std::string("Stage ") + stageNames[s] + " output check failed; stop before further dispatch");
        }
        require(phase.at("output_screen").at("passed") == true, "Independent binary64 output screen failed");
        double tileMs = 0;
        for (const auto& stage : phase.at("stages")) tileMs += stage.at("gpu_milliseconds").get<double>();
        phase["tile_gpu_milliseconds"] = tileMs;
        phase["oracle_seconds"] = oracleSeconds;
        phase["memory_after"] = Json::parse(device.memoryJson());
    }
    device.drain(); deadline.check();
    phase["tracked_after_phase_release"] = device.trackedBufferBytes();
    require(device.trackedBufferBytes() == 0, "Completed phase buffers were not released");
    phase["elapsed_seconds"] = deadline.seconds() - phaseStart;
    phase["passed"] = true;
}
inline Json phasePlan(const PhaseSpec& spec) {
    const Tile t = tile(spec); Json stages = Json::array();
    for (size_t s = 0; s < 4; ++s)
        stages.push_back({{"name", stageNames[s]}, {"shader", stageShaders[s]}, {"groups", stageGroups(t, s)},
                          {"counts", countsJson(stageCounts(t.queries, t.sequenceLength, s))}});
    Json grids = Json::array();
    for (const auto& g : spec.grids) grids.push_back({g.temporal, g.height, g.width});
    return {{"name", spec.name}, {"fixture", spec.fixture == Fixture::uniform ? "uniform_constant_nonzero_dyadic_scores" : "nonuniform_dyadic_scores"},
            {"grid_thw", grids}, {"patches", t.patches}, {"frame_index", spec.frame}, {"sequence_start", t.sequenceStart},
            {"sequence_length", t.sequenceLength}, {"query_start", t.queryStart}, {"queries", t.queries},
            {"query_heads", t.queryHeads}, {"key_tiles", t.keyTiles}, {"last_key_tile_keys", t.lastTileKeys},
            {"frame_query_tiles", t.frameQueryTiles}, {"stages", stages}, {"footprint", footprintJson(footprint(t))}};
}
inline Json planJson(const std::vector<PhaseSpec>& specs = phases()) {
    Json list = Json::array(); uint64_t deviceMaximum = 0, hostMaximum = 0;
    for (const auto& spec : specs) {
        list.push_back(phasePlan(spec)); const Footprint f = footprint(tile(spec));
        deviceMaximum = std::max(deviceMaximum, f.deviceWithStaging); hostMaximum = std::max(hostMaximum, f.hostPeak);
    }
    Json frozen = Json::array();
    for (const auto& s : frozenShaders) frozen.push_back({{"path", s.relative}, {"bytes", s.bytes}, {"sha256", s.sha256}});
    return {{"phases", list}, {"dispatches", 4 * specs.size()}, {"maximum_device_tracked_plus_staging_bytes", deviceMaximum},
            {"maximum_host_vector_bytes", hostMaximum}, {"device_allocation_cap_bytes", deviceAllocationCap},
            {"host_vector_cap_bytes", hostVectorCap}, {"receipt_cap_bytes", receiptCap},
            {"frozen_shaders", frozen},
            {"production_block_counts", {{"tiny_grid_1x32x26", blockCountsJson(832)}, {"cli_grid_1x126x96", blockCountsJson(12096)}}},
            {"screens", {{"unit_roundoff", unitRoundoff}, {"exp2_relative_error_assumed", exp2RelativeError},
                         {"division_relative_error_d3d11", divisionRelativeError}, {"bf16_relative_half_ulp", bf16RelativeHalfUlp},
                         {"oracle_relative_margin", oracleRelativeMargin}}},
            {"gates", {{"dispatch_limit_milliseconds", dispatchLimitMilliseconds}, {"extrapolation_margin", extrapolationMargin},
                       {"source_deadline_seconds", deadlineSeconds}}}};
}
inline Json initialReport() {
    return {{"schema", schema}, {"generator", generatorId}, {"state", "PREPARING"}, {"native_executed", false}, {"passed", false},
            {"full_model_accepted", false}, {"trained_numerics_accepted", false}, {"ocr_accepted", false},
            {"data_dependent_speed_accepted", false}, {"end_to_end_performance_accepted", false},
            {"sustained_throughput_accepted", false}, {"automatic_rerun", false}, {"retirement_required_on_failure", true},
            {"dispatch_limit_milliseconds", dispatchLimitMilliseconds}, {"source_deadline_seconds", deadlineSeconds},
            {"required_external_owned_job", true}, {"required_external_kill_on_job_close", true},
            {"required_external_process_commit_cap_bytes", processCommitCap},
            {"proposed_external_job_commit_cap_bytes", jobCommitCap},
            {"required_external_cpu", "One core, below normal priority"},
            {"memory_scope", "Device memory JSON reports this process's DXGI local budget/usage on the pinned adapter, not global free VRAM"},
            {"dispatches_submitted", 0}, {"dispatches_completed_and_measured", 0},
            {"plan", planJson()}, {"phases", Json::array()}};
}
inline void run(Device& device, Json& report, const Deadline& deadline, const Persist& persist,
                const std::vector<PhaseSpec>& specs = phases()) {
    require(device.trackedBufferBytes() == 0, "Calibration requires a device with no owned buffers");
    for (size_t i = 0; i < specs.size(); ++i) {
        deadline.check();
        Json phase = phasePlan(specs[i]);
        phase["passed"] = false;
        for (auto& stage : phase["stages"]) { stage["dispatch_call_started"] = false; stage["submitted"] = false; stage["completed"] = false; }
        if (i) phase["extrapolation_admission"] = extrapolationAdmission(report.at("phases"), tile(specs[i]));
        report["phases"].push_back(std::move(phase));
        Json& current = report["phases"].back();
        persist();
        if (i) require(current.at("extrapolation_admission").at("admitted") == true,
                       "Extrapolated stage time reaches the 100 ms gate; refuse the larger phase before allocation");
        runPhase(device, specs[i], report, current, deadline, persist);
        persist();
    }
    deadline.check();
    for (const auto& phase : report.at("phases")) {
        if (phase.at("sequence_length") != 12096 || phase.at("passed") != true) continue;
        double tileMs = 0;
        for (const auto& stage : phase.at("stages")) tileMs += stage.at("gpu_milliseconds").get<double>();
        const double tiles = double(phase.at("frame_query_tiles").get<uint32_t>());
        report["cli_attention_projection"] = {
            {"measured_tile_gpu_milliseconds", tileMs}, {"query_tiles_per_block", tiles},
            {"projected_block_gpu_milliseconds", tileMs * tiles}, {"projected_24_block_gpu_milliseconds", tileMs * tiles * 24},
            {"scope", "Linear projection of one measured tile; excludes per-tile allocation, submission, drain polling, "
                      "inter-dispatch gaps, other vision operators, text decode and concurrency. Not a pages/sec measurement"}};
    }
    report["passed"] = true;
}

// Bounded receipt text. Invalid UTF-8 in driver/compiler messages is replaced, not thrown. Serialization
// streams through a sink that keeps at most receiptCap bytes and only counts the rest, so no unbounded
// second copy is ever built. An oversized report (not expected; receipts are about 0.1 MiB) becomes a
// failing RECEIPT_OVERSIZE summary of bounded fields, never truncated JSON, and is itself checked.
constexpr size_t fallbackExcerptBytes = 4096; // JSON-escaped bytes kept from each original string.
constexpr size_t fallbackKeyBytes = 64, fallbackLargestFields = 8;
constexpr size_t storedErrorBytes = 64u << 10; // Exception text kept in a live report before truncation.
namespace receipt_detail {
class Sink : public nlohmann::detail::output_adapter_protocol<char> {
public:
    Sink(std::string* text, uint64_t keep) : kept(text), limit(keep) {}
    void write_character(char c) override { write_characters(&c, 1); }
    void write_characters(const char* s, std::size_t n) override {
        total += n;
        if (!kept || overflow) return;
        if (kept->size() + n <= limit) { kept->append(s, n); return; }
        overflow = true; std::string().swap(*kept);
    }
    uint64_t total = 0; bool overflow = false;
private:
    std::string* kept; uint64_t limit;
};
// Exactly the bytes of value.dump(2, ' ', false, replace); keeps them in *text only while they fit `keep`.
inline uint64_t serialize(const Json& value, std::string* text, uint64_t keep) {
    const auto sink = std::make_shared<Sink>(text, keep);
    nlohmann::detail::serializer<Json> serializer(sink, ' ', Json::error_handler_t::replace);
    serializer.dump(value, true, false, 2);
    return sink->total;
}
} // namespace receipt_detail

// Valid UTF-8 prefix whose JSON-escaped form is at most `budget` bytes. Invalid sequences become U+FFFD
// and are counted; code points are never split. escapedBytes equals the serializer's output length.
struct Utf8Excerpt { std::string text; uint64_t sourceBytes = 0, keptSourceBytes = 0, escapedBytes = 0, replaced = 0; bool truncated = false; };
inline Utf8Excerpt utf8Excerpt(const char* data, size_t size, size_t budget) {
    Utf8Excerpt e; e.sourceBytes = size; size_t i = 0;
    while (i < size) {
        const auto at = [&](size_t k) { return static_cast<unsigned char>(data[i + k]); };
        const unsigned char c = at(0);
        size_t length = c < 0x80 ? 1 : (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : (c & 0xf8) == 0xf0 ? 4 : 0;
        bool valid = length && length <= size - i;
        for (size_t k = 1; valid && k < length; ++k) valid = (at(k) & 0xc0) == 0x80;
        if (valid && length > 1) { // Reject overlong forms, surrogates and code points above U+10FFFF.
            uint32_t point = uint32_t(c) & uint32_t(0x7fu >> length);
            for (size_t k = 1; k < length; ++k) point = point << 6 | (uint32_t(at(k)) & 0x3fu);
            valid = (length == 2 && point >= 0x80) || (length == 3 && point >= 0x800 && (point < 0xd800 || point > 0xdfff)) ||
                    (length == 4 && point >= 0x10000 && point <= 0x10ffff);
        }
        const char* piece = data + i; size_t pieceBytes = length, cost = length;
        if (!valid) { piece = "\xef\xbf\xbd"; pieceBytes = 3; cost = 3; length = 1; }
        else if (length == 1 && (c == '"' || c == '\\' || c < 0x20))
            cost = c == '"' || c == '\\' || c == '\b' || c == '\f' || c == '\n' || c == '\r' || c == '\t' ? 2 : 6;
        if (e.escapedBytes + cost > budget) break;
        e.text.append(piece, pieceBytes); e.escapedBytes += cost; e.replaced += valid ? 0u : 1u; i += length;
    }
    e.keptSourceBytes = i; e.truncated = i < size;
    return e;
}
inline std::string sha256Bytes(const char* data, size_t size) { Sha256 hash; hash.bytes(data, size); return hash.hex(); }
// Exception text for a live report: unchanged up to storedErrorBytes, otherwise a bounded valid-UTF-8
// excerpt with an explicit lost-detail suffix carrying the full byte length and SHA-256.
inline std::string boundedError(const char* message) {
    if (!message) return "(null exception message)";
    const size_t size = std::strlen(message);
    if (size <= storedErrorBytes) return message;
    const Utf8Excerpt e = utf8Excerpt(message, size, storedErrorBytes);
    return e.text + " [ERROR TRUNCATED, detail lost: " + std::to_string(size) + " bytes, sha256 " + sha256Bytes(message, size) +
           ", first " + std::to_string(e.keptSourceBytes) + " bytes shown]";
}
// Bounded description of one original field: strings keep length, SHA-256 and an excerpt; any other
// value keeps only its type and serialized size.
inline Json fieldDigest(const Json& value) {
    if (!value.is_string()) return {{"type", value.type_name()}, {"serialized_bytes", receipt_detail::serialize(value, nullptr, 0)}};
    const auto& text = value.get_ref<const std::string&>();
    const Utf8Excerpt e = utf8Excerpt(text.data(), text.size(), fallbackExcerptBytes);
    return {{"type", "string"}, {"bytes", text.size()}, {"sha256", sha256Bytes(text.data(), text.size())}, {"excerpt", e.text},
            {"excerpt_source_bytes", e.keptSourceBytes}, {"excerpt_truncated", e.truncated}, {"excerpt_invalid_utf8_replaced", e.replaced}};
}
// Last-resort summary built from constants and integers only.
inline std::string minimalOversizeReceipt(uint64_t fullBytes) {
    return std::string("{\n  \"schema\": \"") + schema + "\",\n  \"state\": \"RECEIPT_OVERSIZE\",\n  \"passed\": false,\n"
           "  \"retirement_required\": true,\n  \"detail_lost\": true,\n  \"error\": \"Receipt exceeded the cap; bounded summary "
           "unavailable; retire the externally owned Job\",\n  \"full_receipt_bytes\": " + std::to_string(fullBytes) +
           ",\n  \"receipt_cap_bytes\": " + std::to_string(receiptCap) + "\n}\n";
}
inline std::string oversizeReceipt(const Json& report, uint64_t fullBytes) {
    const bool object = report.is_object();
    auto field = [&](const char* key) -> Json {
        if (!object || !report.contains(key)) return {{"present", false}};
        Json digest = fieldDigest(report.at(key)); digest["present"] = true; return digest;
    };
    auto scalarIn = [](const Json& owner, const char* key) -> Json { // Only booleans and integers are copied verbatim.
        if (!owner.is_object() || !owner.contains(key)) return nullptr;
        const Json& value = owner.at(key);
        return value.is_boolean() || value.is_number_integer() ? value : Json(nullptr);
    };
    auto scalar = [&](const char* key) { return scalarIn(report, key); };
    Json drain = nullptr;
    if (object && report.contains("failure_drain"))
        drain = {{"completed", scalarIn(report.at("failure_drain"), "completed")}, {"tracked_after", scalarIn(report.at("failure_drain"), "tracked_after")}};
    std::vector<std::pair<uint64_t, const std::string*>> sizes;
    if (object) for (auto it = report.begin(); it != report.end(); ++it) sizes.emplace_back(receipt_detail::serialize(it.value(), nullptr, 0), &it.key());
    std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    Json largest = Json::array();
    for (size_t i = 0; i < sizes.size() && i < fallbackLargestFields; ++i) {
        const Utf8Excerpt key = utf8Excerpt(sizes[i].second->data(), sizes[i].second->size(), fallbackKeyBytes);
        largest.push_back({{"key", key.text}, {"key_bytes", sizes[i].second->size()}, {"key_truncated", key.truncated},
                           {"standalone_serialized_bytes", sizes[i].first}});
    }
    const Json summary = {
        {"schema", schema}, {"state", "RECEIPT_OVERSIZE"}, {"passed", false}, {"retirement_required", true}, {"detail_lost", true},
        {"error", "Receipt of " + std::to_string(fullBytes) + " bytes exceeds the " + std::to_string(receiptCap) +
                  "-byte cap; all detail outside this bounded summary is lost; retire the externally owned Job"},
        {"lost_detail", "Phases, stage checks, profiles, identity, memory and every field not listed here are omitted. Retained "
                        "strings are bounded excerpts with full byte length and SHA-256; other values keep type and size only"},
        {"full_receipt_bytes", fullBytes}, {"receipt_cap_bytes", receiptCap}, {"original_type", report.type_name()},
        {"original_state", field("state")}, {"original_error", field("error")}, {"original_passed", scalar("passed")},
        {"native_executed", scalar("native_executed")}, {"device_created", scalar("device_created")},
        {"original_retirement_required", scalar("retirement_required")}, {"dispatches_submitted", scalar("dispatches_submitted")},
        {"dispatches_completed_and_measured", scalar("dispatches_completed_and_measured")}, {"failure_drain", drain},
        {"original_top_level_fields", sizes.size()}, {"largest_top_level_fields", largest}};
    std::string text;
    if (receipt_detail::serialize(summary, &text, receiptCap - 1) > receiptCap - 1) return minimalOversizeReceipt(fullBytes);
    return text + "\n";
}
struct BoundedReceipt { std::string text; bool complete = false; uint64_t fullBytes = 0; };
inline BoundedReceipt boundedReceipt(const Json& report) {
    BoundedReceipt r;
    r.fullBytes = receipt_detail::serialize(report, &r.text, receiptCap - 1) + 1; // + final newline
    if (r.fullBytes <= receiptCap) { r.text.push_back('\n'); r.complete = true; return r; }
    try { r.text = oversizeReceipt(report, r.fullBytes); } catch (const std::exception&) { r.text = minimalOversizeReceipt(r.fullBytes); }
    return r;
}
inline std::string receiptText(const Json& report) { return boundedReceipt(report).text; }

// ---------------------------------------------------------------------------------------------------
// Host admission helpers, portable for CPU tests. Nothing here creates a device.
struct Arguments { bool execute = false; std::map<std::wstring, std::wstring> values; };
inline std::string ascii(const std::wstring& text) {
    std::string result;
    for (wchar_t c : text) { require(c > 0 && c < 128, "ASCII PCI and LUID required"); result.push_back(char(c)); }
    return result;
}
// Exact adapter_identity.h spellings: PCI "bb:dd.f" and LUID "hhhhhhhh:llllllll", lowercase hex.
inline std::string pinnedPci(const std::wstring& text) {
    const std::string s = ascii(text); auto hex = [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); };
    require(s.size() == 7 && hex(s[0]) && hex(s[1]) && s[2] == ':' && hex(s[3]) && hex(s[4]) && s[5] == '.' && s[6] >= '0' && s[6] <= '7',
            "PCI pin must be lowercase bb:dd.f");
    return s;
}
inline std::string pinnedLuid(const std::wstring& text) {
    const std::string s = ascii(text); bool ok = s.size() == 17 && s[8] == ':';
    for (size_t i = 0; ok && i < s.size(); ++i) ok = i == 8 || (s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f');
    require(ok, "LUID pin must be lowercase hhhhhhhh:llllllll");
    return s;
}
// Strict: each known option at most once, followed by a nonempty value that is not option-shaped (no
// leading '-'; no valid pin or absolute path has one), so "--pci --execute" cannot swallow a flag. Pin and
// path syntax is checked in every mode, so the inactive plan is printed only for genuinely valid
// arguments without --execute. Existence and freshness remain execute-time checks in the entry.
inline Arguments parseArguments(const std::vector<std::wstring>& argv) {
    Arguments result;
    for (size_t i = 0; i < argv.size(); ++i) {
        const std::wstring& name = argv[i];
        if (name == L"--execute") { require(!result.execute, "Duplicate execute flag"); result.execute = true; continue; }
        require(name == L"--shader-root" || name == L"--pci" || name == L"--luid" || name == L"--output", "Unknown calibration option");
        const std::string option = ascii(name);
        require(i + 1 < argv.size() && !argv[i + 1].empty() && argv[i + 1].front() != L'-',
                "Option value absent or option-shaped after " + option + "; a nonempty non-option value is required");
        require(result.values.emplace(name, argv[++i]).second, "Duplicate calibration option " + option);
    }
    if (result.values.count(L"--pci")) pinnedPci(result.values.at(L"--pci"));
    if (result.values.count(L"--luid")) pinnedLuid(result.values.at(L"--luid"));
    for (const auto* name : {L"--shader-root", L"--output"})
        require(!result.values.count(name) || std::filesystem::path(result.values.at(name)).is_absolute(), "Absolute shader root and output paths required");
    if (result.execute)
        for (const auto* name : {L"--shader-root", L"--pci", L"--luid", L"--output"})
            require(result.values.count(name) != 0, "Explicit shader root, PCI, LUID and fresh output required");
    return result;
}
// Hashes the exact shader bytes the Device will compile and refuses any difference before Device creation.
inline Json verifyFrozenShaders(const std::filesystem::path& shaderRoot) {
    Json result = Json::array();
    for (const auto& frozen : frozenShaders) {
        const auto path = shaderRoot / std::filesystem::path(frozen.relative);
        require(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == frozen.bytes,
                std::string("Frozen production shader absent or resized: ") + frozen.relative);
        std::ifstream file(path, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        require(file.good() || file.eof(), std::string("Shader read failed: ") + frozen.relative);
        require(data.size() == frozen.bytes, std::string("Shader size changed while reading: ") + frozen.relative);
        Sha256 hash; hash.bytes(data.data(), data.size()); const std::string digest = hash.hex();
        result.push_back({{"path", frozen.relative}, {"bytes", data.size()}, {"sha256", digest}, {"frozen_sha256", frozen.sha256}});
        require(digest == frozen.sha256, std::string("Shader bytes differ from frozen production source: ") + frozen.relative);
    }
    return result;
}
} // namespace chandra::vision_calibration
