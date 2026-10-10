// New ChandraNative code, MPL-2.0. Portable core of the public, model-free B1 GEMV candidate
// calibration. Windows entry: gemv_candidate_calibration.cpp; CPU harness: gemv_candidate_test.cpp.
// Root owns the external Job, CPU affinity/priority, process commit cap, device admission and runs.
// One plan serves every ordered route (gemv_variants.h); only the expected shader and groups differ.
#pragma once
#include "api.h"
#include "gemv_variants.h"
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::dc::experimental {
const char* gemvB1Selection(); // operators.cpp; throws std::invalid_argument for an invalid selector.
}

namespace chandra::gemv_calibration {
using Json = nlohmann::json;
using chandra::dc::Buffer;
using chandra::dc::Device;
using chandra::dc::Weight;

using chandra::gemv_variants::Shape;
// v2: the selected ordered route (ordered, ordered32 or ordered64) sets the expected shader and groups;
// two phases were added to the six of v1, which keep their names, operands and order.
constexpr const char* schema = "private.chandra.directcompute.gemv-b1-candidate-calibration.v2";
constexpr uint32_t maximumDispatchOutputs = 1024, maximumInputWidth = 9216;
constexpr uint64_t allocationCap = 256ull * 1024 * 1024;
constexpr double dispatchLimitMilliseconds = 100.0;
constexpr uint32_t deadlineSeconds = 120;
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24,
              "IEEE binary32 required");
static_assert(std::numeric_limits<double>::digits >= 53, "Independent binary64 oracle required");
#if defined(FLT_EVAL_METHOD)
static_assert(FLT_EVAL_METHOD == 0, "Each binary32 CPU operation must round to binary32");
#endif

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
inline uint32_t roundedBF16Bits(uint32_t value) {
    if ((value & 0x7fffffffu) > 0x7f800000u) return (value & 0xffff0000u) | 0x00400000u;
    return (value + 0x7fffu + ((value >> 16) & 1u)) & 0xffff0000u;
}
inline uint64_t fnv1a(const std::vector<uint32_t>& words) {
    uint64_t hash = 14695981039346656037ull;
    for (uint32_t word : words) for (int i = 0; i < 4; ++i) { hash ^= (word >> (8 * i)) & 0xffu; hash *= 1099511628211ull; }
    return hash;
}
inline std::string hex64(uint64_t value) {
    const char* digits = "0123456789abcdef"; std::string text(16, '0');
    for (int i = 15; i >= 0; --i) { text[size_t(i)] = digits[value & 15u]; value >>= 4; }
    return text;
}

// Generators. All operands are public functions of their indices; no model asset is read.
// dyadic_v1 reuses dispatch_calibration.cpp's public_integer_mod33_dyadic_v1 row 0 and adds a bias
// integer. Every product is integer/1024 and every partial sum numerator stays below 9216*256+512 <
// 2^24, so no FP32 product or addition rounds in any order: these phases are order-insensitive.
inline int dyadicInput(uint32_t k) { return int((uint64_t(k) * 13 + uint64_t(k / 33) * 5) % 33) - 16; }
inline int dyadicWeight(uint32_t output, uint32_t k) {
    return int((uint64_t(output) * 17 + uint64_t(k) * 7 + uint64_t(k / 33) * 3) % 33) - 16;
}
inline int dyadicBias(uint32_t output) { return int((uint64_t(output) * 29 + 11) % 33) - 16; }
// order_v1: splitmix64 stream. FP32 inputs have exponents -4..3 and full 23-bit mantissas; BF16
// weights have exponents -8..0. Products are multiples of 2^-42 and below 2^5, so no subnormal or
// overflow can occur, while ordinary FP32 additions do round: this phase is order-sensitive.
inline uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31);
}
inline float orderInput(uint32_t k) {
    uint64_t r = splitmix(0x6f72646572000000ull + k);
    uint32_t exponent = uint32_t(127 - 4 + (r >> 40) % 8), mantissa = uint32_t(r & 0x7fffffu);
    return fromBits(uint32_t((r >> 63) << 31) | (exponent << 23) | mantissa);
}
inline uint16_t orderWeight(uint32_t output, uint32_t k) {
    uint64_t r = splitmix(0x7765696768740000ull ^ (uint64_t(output) << 20) ^ k);
    uint32_t exponent = uint32_t(127 - 8 + (r >> 40) % 9), mantissa = uint32_t(r & 0x7fu);
    return uint16_t(((r >> 63) << 15) | (exponent << 7) | mantissa);
}
// specials_v1: K=4, seven outputs, each with at most one nonzero product (or one +Inf/-Inf pair),
// so the FP32 result is order-independent. Covers +0 from negative-zero products, overflow to
// +/-Inf, Inf-Inf NaN, BF16 ties to even downward/upward and signed-zero/-Inf bias addition.
constexpr uint32_t specialInputs[4] = {0x3f808000u, 0x71800000u, 0xf1800000u, 0x80000000u}; // 1+2^-8, 2^100, -2^100, -0
constexpr uint16_t specialWeights[7][4] = {
    {0x0000, 0x0000, 0x0000, 0x3f80}, {0x0000, 0x7180, 0x0000, 0x0000}, {0x0000, 0x7180, 0x7180, 0x0000},
    {0x3f80, 0x0000, 0x0000, 0x0000}, {0xbf80, 0x0000, 0x0000, 0x0000}, {0x0000, 0xf180, 0x0000, 0x0000},
    {0x0000, 0x0000, 0x0000, 0x0000}};
constexpr uint16_t specialBias[7] = {0x8000, 0x3f80, 0x0000, 0x3c00, 0xbc00, 0x0000, 0xff80}; // -0,1,0,2^-7,-2^-7,0,-Inf

enum class Generator { dyadic, specials, order };
enum class BiasKind { none, bf16, fp32 };
struct PhaseSpec {
    const char* name; Generator generator; uint32_t inputWidth, outputs; std::vector<uint32_t> shardRows;
    bool bf16Weights; BiasKind bias;
};
inline std::vector<PhaseSpec> phases() {
    return {
        {"specials_signed_zero_inf_nan_ties", Generator::specials, 4, 7, {7}, true, BiasKind::bf16},
        {"ragged_odd_k_bf16_three_shards_bf16_bias", Generator::dyadic, 37, 21, {6, 9, 6}, true, BiasKind::bf16},
        {"odd_k_fp32_weights_two_shards_fp32_bias", Generator::dyadic, 67, 40, {17, 23}, false, BiasKind::fp32},
        // Order-sensitive odd K: odd-parity full tiles and K tails for every ordered tile, one-output
        // group tails for 8, 32 and 64 outputs per group, and a half-used final packed word per shard.
        {"odd_k1001_order_sensitive_two_shards_bf16_bias", Generator::order, 1001, 130, {65, 65}, true, BiasKind::bf16},
        {"multi_chunk_k1001_two_shards", Generator::dyadic, 1001, 2600, {1500, 1100}, true, BiasKind::none},
        // The most frequent decode input width (hidden size 2560), one complete 1024-output chunk.
        {"full_chunk_k2560_order_sensitive", Generator::order, 2560, 1024, {1024}, true, BiasKind::none},
        {"full_width_k9216_dyadic", Generator::dyadic, 9216, 1024, {1024}, true, BiasKind::none},
        {"full_width_k9216_order_sensitive_fp32", Generator::order, 9216, 1024, {1024}, true, BiasKind::none}};
}
struct Call { bool roundOutputBF16; bool bias; };
inline std::vector<Call> calls(const PhaseSpec& spec) {
    if (spec.generator == Generator::specials) return {{false, false}, {true, false}, {false, true}, {true, true}};
    return {{false, spec.bias != BiasKind::none}, {true, spec.bias != BiasKind::none}};
}
inline std::vector<uint32_t> expectedGroups(const PhaseSpec& spec, const Shape& shape) {
    std::vector<uint32_t> groups;
    for (uint32_t rows : spec.shardRows)
        for (uint32_t first = 0; first < rows; first += maximumDispatchOutputs)
            groups.push_back((std::min(rows - first, maximumDispatchOutputs) + shape.outputsPerGroup - 1) / shape.outputsPerGroup);
    return groups;
}
inline uint64_t shardWords(const PhaseSpec& spec, uint32_t rows) {
    uint64_t elements = uint64_t(rows) * spec.inputWidth;
    return spec.bf16Weights ? (elements + 1) / 2 : elements;
}
// Tracked device bytes while one phase output is live, plus its readback staging copy.
inline uint64_t phaseForecastBytes(const PhaseSpec& spec) {
    uint64_t total = uint64_t(spec.inputWidth) * 4 + 2ull * spec.outputs * 4;
    for (uint32_t rows : spec.shardRows) total += shardWords(spec, rows) * 4;
    if (spec.bias == BiasKind::bf16) total += (uint64_t(spec.outputs) + 1) / 2 * 4;
    if (spec.bias == BiasKind::fp32) total += uint64_t(spec.outputs) * 4;
    return total;
}
inline uint64_t maximumForecastBytes() {
    uint64_t maximum = 0;
    for (const auto& spec : phases()) maximum = std::max(maximum, phaseForecastBytes(spec));
    return maximum;
}

inline float inputValue(const PhaseSpec& spec, uint32_t k) {
    if (spec.generator == Generator::dyadic) return float(dyadicInput(k)) / 32.0f;
    if (spec.generator == Generator::specials) return fromBits(specialInputs[k]);
    return orderInput(k);
}
inline uint32_t weightBits(const PhaseSpec& spec, uint32_t output, uint32_t k) {
    if (spec.generator == Generator::dyadic) {
        uint32_t value = bits(float(dyadicWeight(output, k)) / 32.0f);
        require(roundedBF16Bits(value) == value, "Generated dyadic weight is not exactly BF16");
        return value;
    }
    if (spec.generator == Generator::specials) return uint32_t(specialWeights[output][k]) << 16;
    return uint32_t(orderWeight(output, k)) << 16;
}
inline uint32_t biasBits(const PhaseSpec& spec, uint32_t output) {
    if (spec.generator == Generator::specials) return uint32_t(specialBias[output]) << 16;
    return bits(float(dyadicBias(output)) / 32.0f);
}
inline Buffer upload(Device& device, const std::vector<uint32_t>& words, uint64_t logicalElements, bool packed) {
    Buffer buffer = device.words(uint32_t(words.size()), words.data());
    buffer.logicalElements = logicalElements; buffer.packedBF16 = packed;
    return buffer;
}
inline std::vector<uint32_t> pack(const std::vector<uint32_t>& fp32Bits, bool bf16) {
    if (!bf16) return fp32Bits;
    std::vector<uint32_t> words((fp32Bits.size() + 1) / 2, 0);
    for (size_t i = 0; i < fp32Bits.size(); ++i) {
        require((fp32Bits[i] & 0xffffu) == 0, "Packed operand is not exactly BF16");
        words[i / 2] |= (fp32Bits[i] >> 16) << (16 * (i % 2));
    }
    return words;
}

struct Expected { std::vector<uint32_t> fp32, fp32WithBias; Json observations; };
// Independent CPU expectations; they never read packed words or device buffers.
inline Expected expectedOutputs(const PhaseSpec& spec, const Deadline& deadline) {
    Expected result; result.fp32.resize(spec.outputs); result.fp32WithBias.resize(spec.outputs);
    double maximumAbsolute = 0, maximumUlps = 0; uint64_t differing = 0;
    for (uint32_t output = 0; output < spec.outputs; ++output) {
        if (!(output % 16)) deadline.check();
        if (spec.generator == Generator::dyadic) {
            double sum = 0.0; // Exact binary64 dot: every term and partial sum is integer/1024 below 2^24/1024.
            for (uint32_t k = 0; k < spec.inputWidth; ++k)
                sum += (double(dyadicInput(k)) / 32.0) * (double(dyadicWeight(output, k)) / 32.0);
            require(double(float(sum)) == sum, "Generated dyadic dot is not exactly FP32");
            double withBias = sum + double(dyadicBias(output)) / 32.0;
            require(double(float(withBias)) == withBias, "Generated dyadic biased dot is not exactly FP32");
            result.fp32[output] = bits(float(sum)); result.fp32WithBias[output] = bits(float(withBias));
            continue;
        }
        // Sequential ascending-K binary32 dot from +0: each product and addition rounds once.
        float sum = 0.0f; double higher = 0.0, compensation = 0.0;
        for (uint32_t k = 0; k < spec.inputWidth; ++k) {
            float a = inputValue(spec, k), w = fromBits(weightBits(spec, output, k));
            float product = a * w;
            sum = sum + product;
            if (spec.generator == Generator::order) {
                require(!(std::fabs(product) < FLT_MIN && product != 0.0f) && !(std::fabs(sum) < FLT_MIN && sum != 0.0f) &&
                        std::isfinite(sum), "Order generator produced a subnormal or nonfinite intermediate");
                double term = double(a) * double(w), next = higher + term; // Neumaier compensated binary64.
                compensation += std::fabs(higher) >= std::fabs(term) ? (higher - next) + term : (term - next) + higher;
                higher = next;
            }
        }
        result.fp32[output] = bits(sum);
        float biased = sum + fromBits(biasBits(spec, output));
        result.fp32WithBias[output] = bits(biased);
        if (spec.generator == Generator::order) {
            double reference = higher + compensation, error = std::fabs(double(sum) - reference);
            double ulp = std::ldexp(1.0, std::max(std::ilogb(reference), -126) - 23);
            maximumAbsolute = std::max(maximumAbsolute, error); maximumUlps = std::max(maximumUlps, error / ulp);
            differing += bits(sum) != bits(float(reference));
        }
    }
    if (spec.generator == Generator::order)
        result.observations = {{"ascending_fp32_vs_compensated_binary64_maximum_absolute", maximumAbsolute},
                               {"ascending_fp32_vs_compensated_binary64_maximum_fp32_ulps", maximumUlps},
                               {"outputs_differing_from_rounded_binary64", differing},
                               {"scope", "CPU observation of the expected sequential FP32 order; not an acceptance threshold"}};
    return result;
}
inline bool sameOutput(uint32_t observed, uint32_t expected, bool rounded) {
    bool expectedNaN = (expected & 0x7fffffffu) > 0x7f800000u;
    if (!expectedNaN) return observed == expected;
    // NaN payloads from arithmetic are hardware-defined; the BF16 cast must still quiet and truncate.
    bool observedNaN = (observed & 0x7fffffffu) > 0x7f800000u;
    return observedNaN && (!rounded || ((observed & 0xffffu) == 0 && (observed & 0x00400000u) != 0));
}
inline Json comparison(const std::vector<uint32_t>& observed, const std::vector<uint32_t>& expected, bool rounded) {
    require(observed.size() == expected.size(), "Incomplete output readback");
    Json failures = Json::array();
    for (size_t i = 0; i < expected.size(); ++i) if (!sameOutput(observed[i], expected[i], rounded)) failures.push_back(i);
    return {{"elements", expected.size()}, {"passed", failures.empty()}, {"mismatch_indices", std::move(failures)},
            {"nan_rule", rounded ? "expected NaN requires observed NaN with zero low half and quiet bit" : "expected NaN requires observed NaN"},
            {"expected_fp32_storage_bits", expected}, {"observed_fp32_storage_bits", observed}};
}
inline uint64_t bf16Ties(const std::vector<uint32_t>& fp32) {
    uint64_t ties = 0;
    for (uint32_t value : fp32) ties += (value & 0x7fffffffu) <= 0x7f800000u && (value & 0xffffu) == 0x8000u;
    return ties;
}

struct Operands { Weight weight, bias; Buffer input; Json hashes; };
inline Operands operands(Device& device, const PhaseSpec& spec, const Deadline& deadline) {
    Operands result;
    std::vector<uint32_t> input(spec.inputWidth);
    for (uint32_t k = 0; k < spec.inputWidth; ++k) input[k] = bits(inputValue(spec, k));
    result.hashes["input_fp32_words_fnv1a64"] = hex64(fnv1a(input));
    result.input = device.floats(spec.inputWidth, reinterpret_cast<const float*>(input.data()));
    result.weight.rows = spec.outputs; result.weight.cols = spec.inputWidth; result.weight.bf16 = spec.bf16Weights;
    Json shardHashes = Json::array(); uint32_t firstRow = 0;
    for (uint32_t rows : spec.shardRows) {
        deadline.check();
        std::vector<uint32_t> values(size_t(rows) * spec.inputWidth);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t k = 0; k < spec.inputWidth; ++k) values[size_t(r) * spec.inputWidth + k] = weightBits(spec, firstRow + r, k);
        auto words = pack(values, spec.bf16Weights);
        std::vector<uint32_t>().swap(values);
        require(words.size() == shardWords(spec, rows), "Shard word forecast differs");
        shardHashes.push_back({{"first_row", firstRow}, {"rows", rows}, {"words", words.size()}, {"words_fnv1a64", hex64(fnv1a(words))}});
        result.weight.shardFirstRows.push_back(firstRow);
        result.weight.shards.push_back(upload(device, words, uint64_t(rows) * spec.inputWidth, spec.bf16Weights));
        firstRow += rows;
    }
    require(firstRow == spec.outputs, "Phase shard rows do not cover outputs");
    result.hashes["weight_shards"] = std::move(shardHashes);
    if (spec.bias != BiasKind::none) {
        std::vector<uint32_t> values(spec.outputs);
        for (uint32_t output = 0; output < spec.outputs; ++output) values[output] = biasBits(spec, output);
        auto words = pack(values, spec.bias == BiasKind::bf16);
        result.hashes["bias_words_fnv1a64"] = hex64(fnv1a(words));
        result.bias.rows = spec.outputs; result.bias.cols = 1; result.bias.bf16 = spec.bias == BiasKind::bf16;
        result.bias.shardFirstRows = {0};
        result.bias.shards.push_back(upload(device, words, spec.outputs, spec.bias == BiasKind::bf16));
    }
    return result;
}
inline bool profileCheck(const Json& profile, const Shape& shape, const std::vector<uint32_t>& groups, size_t callCount, Json& summary) {
    require(profile.is_object() && profile.at("disjoint") == false && profile.at("frequency").is_number_unsigned() &&
            profile.at("frequency").get<uint64_t>() > 0, "Invalid or disjoint GPU timing window");
    const auto& dispatches = profile.at("dispatches");
    require(dispatches.is_array() && dispatches.size() == groups.size() * callCount,
            "Dispatch count differs from the candidate geometry forecast");
    bool below = true; double maximum = 0, total = 0;
    for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& dispatch = dispatches[i];
        require(dispatch.at("shader") == shape.shader && dispatch.at("groups") == Json::array({groups[i % groups.size()], 1, 1}),
                "Unexpected candidate dispatch shader or geometry; no silent fallback is admitted");
        double ms = dispatch.at("gpu_milliseconds").get<double>();
        require(std::isfinite(ms) && ms >= 0.0, "Invalid completed dispatch duration");
        below = below && ms < dispatchLimitMilliseconds; maximum = std::max(maximum, ms); total += ms;
    }
    summary = {{"dispatches", dispatches.size()}, {"maximum_gpu_milliseconds", maximum}, {"sum_gpu_milliseconds", total}};
    return below;
}
inline void memoryCheck(Device& device, const Deadline& deadline, uint64_t forecast) {
    deadline.check();
    require(device.trackedBufferBytes() <= forecast && forecast <= allocationCap, "Calibration tensor/readback forecast exceeded");
}

// Malformed B1 admissions must be refused before output allocation or any dispatch.
inline void invalidAdmission(Device& device, Json& report, const Deadline& deadline) {
    Json cases = Json::array();
    auto refused = [&](const char* name, const Buffer& input, const Weight& weight, const Weight* bias) {
        deadline.check();
        uint64_t before = device.trackedBufferBytes(); std::string message;
        try { (void)chandra::dc::linear(device, input, weight, 1, true, bias); }
        catch (const std::invalid_argument& error) { message = error.what(); }
        require(!message.empty(), std::string("Malformed admission was not refused: ") + name);
        require(device.trackedBufferBytes() == before, std::string("Refused admission allocated: ") + name);
        cases.push_back({{"case", name}, {"refused", true}, {"message", message}});
    };
    auto weightOf = [&](uint32_t rows, uint32_t cols, bool bf16) {
        Weight w; w.rows = rows; w.cols = cols; w.bf16 = bf16; w.shardFirstRows = {0};
        uint64_t elements = uint64_t(rows) * cols;
        w.shards.push_back(upload(device, std::vector<uint32_t>(size_t(bf16 ? (elements + 1) / 2 : elements), 0), elements, bf16));
        return w;
    };
    device.beginProfile();
    {
        auto wide = weightOf(1, maximumInputWidth + 1, true);
        Buffer wideInput = device.floats(maximumInputWidth + 1);
        refused("input_width_9217_exceeds_pinned_maximum", wideInput, wide, nullptr);
    }
    {
        auto w = weightOf(4, 8, true);
        Buffer input = device.floats(8), longer = device.floats(9);
        refused("activation_extent_differs", longer, w, nullptr);
        Buffer packedInput = input; packedInput.packedBF16 = true;
        refused("packed_activation", packedInput, w, nullptr);
        Weight gap = w; gap.shardFirstRows = {1};
        refused("weight_shard_offset_gap", input, gap, nullptr);
        Weight dtype = w; dtype.bf16 = false;
        refused("weight_dtype_flag_differs", input, dtype, nullptr);
        Weight partial = w; partial.rows = 5;
        refused("weight_shards_do_not_cover_rows", input, partial, nullptr);
        Weight words = w; words.shards[0].words = 15;
        refused("packed_word_count_differs", input, words, nullptr);
        Weight ragged = w; ragged.shards[0].logicalElements = 31;
        refused("weight_shard_partial_row", input, ragged, nullptr);
        auto narrowBias = weightOf(3, 1, true);
        refused("bias_width_differs", input, w, &narrowBias);
    }
    auto profile = Json::parse(device.finishProfile());
    require(profile.at("dispatches").empty(), "A refused admission submitted a dispatch");
    device.drain();
    report["invalid_admission"] = {{"cases", std::move(cases)}, {"dispatches", 0}, {"passed", true}};
}

// The ordered route named by the process selector; anything else is refused before any device exists.
inline const Shape& orderedRoute(const std::string& selection) {
    const Shape* shape = chandra::gemv_variants::shape(selection.c_str());
    require(shape != nullptr && shape->route != chandra::gemv_variants::Route::parallel32,
            "Sequential bit-equality calibration requires CHANDRA_EXPERIMENTAL_GEMV_B1=ordered, ordered32 or ordered64; parallel32 needs independent numerical admission");
    return *shape;
}
inline void describeRoute(Json& report, const Shape& shape) {
    report["candidate"] = std::string("CHANDRA_EXPERIMENTAL_GEMV_B1=") + shape.selector;
    report["shader"] = shape.shader;
    report["route_geometry"] = {{"outputs_per_group", shape.outputsPerGroup}, {"threads_per_group", shape.threads},
                                {"k_tile", shape.tile}, {"groupshared_row_stride_floats", shape.stride},
                                {"groupshared_bytes", shape.groupsharedBytes()}};
}

inline void run(Device& device, Json& report, const Deadline& deadline, const Shape& shape) {
    require(shape.route != chandra::gemv_variants::Route::parallel32,
            "Sequential bit-equality calibration refuses reassociated parallel32 before allocation or dispatch");
    invalidAdmission(device, report, deadline);
    require(device.trackedBufferBytes() == 0, "Invalid-admission buffers were not released");
    for (const auto& spec : phases()) {
        const auto groups = expectedGroups(spec, shape); const auto plannedCalls = calls(spec);
        const uint64_t forecast = phaseForecastBytes(spec);
        report["phases"].push_back({{"name", spec.name}, {"input_width", spec.inputWidth}, {"outputs", spec.outputs},
                                    {"shard_rows", spec.shardRows}, {"bf16_weights", spec.bf16Weights},
                                    {"bias", spec.bias == BiasKind::none ? "none" : spec.bias == BiasKind::bf16 ? "bf16" : "fp32"},
                                    {"generator", spec.generator == Generator::dyadic ? "dyadic_v1" : spec.generator == Generator::specials ? "specials_v1" : "order_v1"},
                                    {"expected_groups_per_call", groups}, {"calls", plannedCalls.size()},
                                    {"forecast_tracked_plus_readback_bytes", forecast},
                                    {"passed", false}, {"complete_reads", false}, {"dispatches_submitted", 0}});
        auto& phase = report["phases"].back();
        double phaseStart = deadline.seconds();
        {
            auto expected = expectedOutputs(spec, deadline);
            phase["expected_bf16_ties_in_fp32_dots"] = bf16Ties(expected.fp32);
            if (!expected.observations.is_null()) phase["cpu_observations"] = expected.observations;
            auto data = operands(device, spec, deadline);
            phase["operands"] = data.hashes;
            phase["oracle_and_upload_seconds"] = deadline.seconds() - phaseStart;
            memoryCheck(device, deadline, forecast);
            phase["memory_before"] = Json::parse(device.memoryJson());
            phase["results"] = Json::array();
            device.beginProfile();
            uint64_t submitted = 0;
            for (const auto& call : plannedCalls) {
                const auto& reference = call.bias ? expected.fp32WithBias : expected.fp32;
                std::vector<uint32_t> want(reference.size());
                for (size_t i = 0; i < want.size(); ++i) want[i] = call.roundOutputBF16 ? roundedBF16Bits(reference[i]) : reference[i];
                bool exact = false;
                {
                    auto output = chandra::dc::linear(device, data.input, data.weight, 1, call.roundOutputBF16, call.bias ? &data.bias : nullptr);
                    submitted += groups.size(); phase["dispatches_submitted"] = submitted;
                    memoryCheck(device, deadline, forecast);
                    auto observed = device.readWords(output);
                    Json result = comparison(observed, want, call.roundOutputBF16);
                    result["round_output_bf16"] = call.roundOutputBF16; result["bias"] = call.bias;
                    exact = result.at("passed") == true;
                    phase["results"].push_back(std::move(result));
                }
                deadline.check();
                if (!exact) {
                    phase["gpu_profile"] = Json::parse(device.finishProfile());
                    throw std::runtime_error("Candidate output mismatch; stop without rerun or larger phase");
                }
            }
            phase["complete_reads"] = true;
            phase["gpu_profile"] = Json::parse(device.finishProfile());
            Json timing; bool below = profileCheck(phase.at("gpu_profile"), shape, groups, plannedCalls.size(), timing);
            phase["timing_summary"] = timing;
            phase["all_dispatches_strictly_below_100_ms"] = below;
            phase["memory_after"] = Json::parse(device.memoryJson());
            memoryCheck(device, deadline, forecast);
            phase["elapsed_seconds"] = deadline.seconds() - phaseStart;
            phase["passed"] = below;
            require(below, "A dispatch reached or exceeded 100 ms; stop without entering a larger phase");
        }
        device.drain(); deadline.check();
        phase["tracked_after_phase_release"] = device.trackedBufferBytes();
        require(device.trackedBufferBytes() == 0, "Completed phase buffers were not released");
    }
    deadline.check();
    report["passed"] = true;
}
inline Json initialReport() {
    return {{"schema", schema}, {"state", "PREPARING"}, {"native_executed", false}, {"passed", false},
            {"candidate", nullptr}, {"shader", nullptr},
            {"full_model_accepted", false}, {"trained_numerics_accepted", false}, {"ocr_accepted", false},
            {"end_to_end_performance_accepted", false}, {"automatic_rerun", false},
            {"dispatch_limit_milliseconds", dispatchLimitMilliseconds}, {"source_deadline_seconds", deadlineSeconds},
            {"required_external_owned_job", true}, {"required_external_process_commit_cap_bytes", allocationCap},
            {"required_external_cpu", "One core, below normal priority"},
            {"maximum_phase_tracked_plus_readback_bytes", maximumForecastBytes()},
            {"phases", Json::array()}};
}
} // namespace chandra::gemv_calibration
