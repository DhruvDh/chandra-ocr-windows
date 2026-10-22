// New ChandraNative code, MPL-2.0. Public, finite GEMM safety calibration.
// No model/private assets. Root owns the external Job, CPU affinity/priority,
// process commit limit, physical device admission, build and native execution.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "api.h"
#include "../vendor/nlohmann/json.hpp"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;
using namespace chandra::dc;
namespace {
constexpr uint32_t inputWidth = 9216, outputWidth = 1024;
constexpr uint32_t phases[] = {1, 16, 32};
constexpr uint64_t allocationCap = 256ull * 1024 * 1024;
constexpr uint64_t weightElements = uint64_t(inputWidth) * outputWidth;
constexpr uint64_t weightBytes = weightElements * 2;
constexpr uint64_t maximumInputBytes = 32ull * inputWidth * 4;
constexpr uint64_t maximumOutputBytes = 32ull * outputWidth * 4;
constexpr uint64_t maximumTrackedBytes = weightBytes + maximumInputBytes + 2 * maximumOutputBytes;
constexpr uint64_t maximumWithReadback = maximumTrackedBytes + maximumOutputBytes;
constexpr double dispatchLimitMilliseconds = 100.0;
constexpr uint32_t deadlineSeconds = 120;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24, "IEEE binary32 required");
static_assert(std::numeric_limits<double>::digits >= 53, "Independent binary64 oracle required");
static_assert(uint64_t(inputWidth) * 16 * 16 < (1ull << 24), "Every integer partial sum is exactly FP32");
static_assert(maximumWithReadback < allocationCap, "Complete tensors and readback must fit the native cap");

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::string ascii(const std::wstring& text) {
    std::string result;
    for (wchar_t c : text) { require(c > 0 && c < 128, "ASCII PCI and LUID required"); result.push_back(char(c)); }
    return result;
}
struct Deadline {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    void check() const {
        require(std::chrono::steady_clock::now() - start < std::chrono::seconds(deadlineSeconds),
                "120-second source deadline exceeded; retire the externally owned Job");
    }
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};
uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result; }
float fromBits(uint32_t value) { float result; std::memcpy(&result, &value, sizeof(result)); return result; }
uint32_t roundedBF16Bits(uint32_t value) {
    if ((value & 0x7fffffffu) > 0x7f800000u) return (value & 0xffff0000u) | 0x00400000u;
    return (value + 0x7fffu + ((value >> 16) & 1u)) & 0xffff0000u;
}

// Public generator v1. All integers are in [-16,16]. Division by 32 is
// exact BF16; each product is integer/1024. Even the worst absolute partial
// integer sum is <= 9216*256 = 2359296 < 2^24, so neither sequential FP32
// products nor additions round. The separate oracle never reads packed words.
int inputInteger(uint32_t row, uint32_t k) {
    return int((uint64_t(row) * 19 + uint64_t(k) * 13 + uint64_t(k / 33) * 5) % 33) - 16;
}
int weightInteger(uint32_t output, uint32_t k) {
    return int((uint64_t(output) * 17 + uint64_t(k) * 7 + uint64_t(k / 33) * 3) % 33) - 16;
}
std::vector<uint32_t> packedWeights(const Deadline& deadline) {
    std::vector<uint32_t> words(size_t(weightElements / 2), 0);
    for (uint32_t output = 0; output < outputWidth; ++output) {
        if (!(output % 16)) deadline.check();
        for (uint32_t k = 0; k < inputWidth; ++k) {
            float value = float(weightInteger(output, k)) / 32.0f;
            uint32_t expanded = roundedBF16Bits(bits(value));
            require(fromBits(expanded) == value, "Generated weight is not exactly BF16");
            size_t index = size_t(output) * inputWidth + k;
            words[index / 2] |= (expanded >> 16) << (16 * (index % 2));
        }
    }
    return words;
}
std::vector<float> inputValues(uint32_t batch, const Deadline& deadline) {
    std::vector<float> input(size_t(batch) * inputWidth);
    for (uint32_t row = 0; row < batch; ++row) {
        deadline.check();
        for (uint32_t k = 0; k < inputWidth; ++k) {
            float value = float(inputInteger(row, k)) / 32.0f;
            require(roundedBF16Bits(bits(value)) == bits(value), "Generated activation is not exactly BF16");
            input[size_t(row) * inputWidth + k] = value;
        }
    }
    return input;
}
struct Expected { std::vector<uint32_t> fp32, bf16; };
Expected expectedOutputs(uint32_t batch, const Deadline& deadline) {
    Expected expected;
    expected.fp32.resize(size_t(batch) * outputWidth);
    expected.bf16.resize(expected.fp32.size());
    for (uint32_t row = 0; row < batch; ++row) {
        for (uint32_t output = 0; output < outputWidth; ++output) {
            if (!(output % 16)) deadline.check();
            double sum = 0.0;
            for (uint32_t k = 0; k < inputWidth; ++k) {
                // Independent exact double dot, not a float/device reduction.
                double a = double(inputInteger(row, k)) / 32.0;
                double w = double(weightInteger(output, k)) / 32.0;
                sum += a * w;
            }
            float value = float(sum);
            require(double(value) == sum, "Generated oracle dot is not exactly FP32");
            size_t index = size_t(row) * outputWidth + output;
            expected.fp32[index] = bits(value);
            expected.bf16[index] = roundedBF16Bits(bits(value));
        }
    }
    return expected;
}
void memoryCheck(Device& device, const Deadline& deadline) {
    deadline.check();
    require(device.trackedBufferBytes() <= maximumTrackedBytes && maximumWithReadback <= allocationCap,
            "Calibration tensor/readback forecast exceeded");
    // Device independently checks the real WDDM budget and 2 GiB local
    // headroom before each allocation and readback staging allocation.
}
Json comparison(const std::vector<uint32_t>& observed, const std::vector<uint32_t>& expected) {
    require(observed.size() == expected.size(), "Incomplete output readback");
    Json failures = Json::array();
    for (size_t i = 0; i < expected.size(); ++i) if (observed[i] != expected[i]) failures.push_back(i);
    return {{"elements", expected.size()}, {"passed", failures.empty()},
            {"mismatch_indices", std::move(failures)}, {"expected_fp32_storage_bits", expected},
            {"observed_fp32_storage_bits", observed}};
}
bool profileCheck(const Json& profile, uint32_t batch) {
    require(profile.is_object() && profile.at("disjoint") == false &&
            profile.at("frequency").is_number_unsigned() && profile.at("frequency").get<uint64_t>() > 0,
            "Invalid or disjoint GPU timing window");
    const auto& dispatches = profile.at("dispatches");
    require(dispatches.is_array() && dispatches.size() == 2, "Exactly two complete linear dispatches required");
    bool below = true;
    for (const auto& dispatch : dispatches) {
        require(dispatch.at("shader") == "runtime/linear.hlsl" &&
                dispatch.at("groups") == Json::array({64, (batch + 15) / 16, 1}),
                "Unexpected calibration dispatch shape or shader");
        double ms = dispatch.at("gpu_milliseconds").get<double>();
        require(std::isfinite(ms) && ms >= 0.0, "Invalid completed dispatch duration");
        below = below && ms < dispatchLimitMilliseconds;
    }
    return below;
}
void run(Device& device, Json& report, const Deadline& deadline) {
    auto packed = packedWeights(deadline);
    Weight weight;
    weight.rows = outputWidth; weight.cols = inputWidth; weight.bf16 = true;
    weight.shardFirstRows = {0};
    weight.shards.push_back(device.words(uint32_t(packed.size()), packed.data()));
    weight.shards.front().packedBF16 = true; weight.shards.front().logicalElements = weightElements;
    // Retire the CPU upload vector; later oracle values come from public integers.
    std::vector<uint32_t>().swap(packed);
    memoryCheck(device, deadline);
    for (uint32_t batch : phases) {
        report["phases"].push_back({{"batch_rows", batch}, {"input_width", inputWidth},
                                    {"output_width", outputWidth}, {"passed", false},
                                    {"complete_reads", false}, {"dispatches_submitted", 0}});
        auto& phase = report["phases"].back();
        double phaseStart = deadline.seconds();
        {
            auto input = inputValues(batch, deadline);
            // CPU expected work and upload precede the GPU timestamp window.
            auto expected = expectedOutputs(batch, deadline);
            phase["oracle_preparation_seconds"] = deadline.seconds() - phaseStart;
            auto activation = device.floats(uint32_t(input.size()), input.data());
            std::vector<float>().swap(input);
            memoryCheck(device, deadline);
            phase["memory_before"] = Json::parse(device.memoryJson());
            device.beginProfile();
            auto raw = linear(device, activation, weight, batch, false);
            phase["dispatches_submitted"] = 1;
            memoryCheck(device, deadline);
            auto rawBits = device.readWords(raw);
            phase["fp32"] = comparison(rawBits, expected.fp32);
            deadline.check();
            if (phase.at("fp32").at("passed") != true) {
                phase["gpu_profile"] = Json::parse(device.finishProfile());
                throw std::runtime_error("Exact unrounded dyadic mismatch; stop before rounded dispatch");
            }
            auto rounded = linear(device, activation, weight, batch, true);
            phase["dispatches_submitted"] = 2;
            memoryCheck(device, deadline);
            auto roundedBits = device.readWords(rounded);
            phase["bf16"] = comparison(roundedBits, expected.bf16);
            phase["complete_reads"] = true;
            deadline.check();
            phase["gpu_profile"] = Json::parse(device.finishProfile());
            // One disjoint window per phase, exactly one unrounded and one
            // rounded GEMM, no warmup/repeat. The window closes before any
            // larger phase is admitted. CPU work/readback is not dispatch time.
            bool below = profileCheck(phase.at("gpu_profile"), batch);
            phase["all_dispatches_strictly_below_100_ms"] = below;
            phase["memory_after"] = Json::parse(device.memoryJson());
            memoryCheck(device, deadline);
            bool exact = phase.at("fp32").at("passed") == true && phase.at("bf16").at("passed") == true;
            phase["passed"] = below && exact;
            phase["elapsed_seconds"] = deadline.seconds() - phaseStart;
            require(exact, "Exact dyadic output mismatch; stop without rerun");
            require(below, "A dispatch reached or exceeded 100 ms; stop without entering a larger phase");
        }
        device.drain(); memoryCheck(device, deadline);
        phase["tracked_after_phase_release"] = device.trackedBufferBytes();
        require(device.trackedBufferBytes() == weightBytes, "Completed phase buffers were not released");
    }
    deadline.check();
    report["passed"] = true;
}
void writeFresh(HANDLE file, const Json& report) {
    std::string text = report.dump(2) + "\n";
    require(text.size() <= 8 * 1024 * 1024, "Complete calibration receipt exceeded 8 MiB");
    LARGE_INTEGER first{};
    require(SetFilePointerEx(file, first, nullptr, FILE_BEGIN), "Receipt seek failed");
    DWORD written = 0;
    require(WriteFile(file, text.data(), DWORD(text.size()), &written, nullptr) && written == text.size(), "Receipt write failed");
    require(SetEndOfFile(file) && FlushFileBuffers(file), "Receipt truncate/flush failed");
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    Json report = {{"schema", "private.chandra.directcompute.dispatch-calibration.v1"},
                   {"generator", "public_integer_mod33_dyadic_v1"},
                   {"state", "PREPARING"}, {"native_executed", false}, {"passed", false},
                   {"full_model_accepted", false}, {"trained_numerics_accepted", false},
                   {"end_to_end_performance_accepted", false}, {"automatic_rerun", false},
                   {"dispatch_limit_milliseconds", dispatchLimitMilliseconds},
                   {"source_deadline_seconds", deadlineSeconds},
                   {"required_external_owned_job", true},
                   {"required_external_process_commit_cap_bytes", allocationCap},
                   {"required_external_cpu", "One core, below normal priority"},
                   {"maximum_owned_tensors_plus_readback_bytes", maximumWithReadback},
                   {"oracle_maximum_integer_partial_sum", uint64_t(inputWidth) * 256},
                   {"phases", Json::array()}};
    HANDLE receipt = INVALID_HANDLE_VALUE;
    try {
        bool execute = false;
        std::map<std::wstring, std::wstring> args;
        for (int i = 1; i < argc; ++i) {
            std::wstring name = argv[i];
            if (name == L"--execute") { require(!execute, "Duplicate execute flag"); execute = true; }
            else {
                require(name == L"--shader-root" || name == L"--pci" || name == L"--luid" || name == L"--output", "Unknown calibration option");
                require(i + 1 < argc, "Option value absent");
                require(args.emplace(name, argv[++i]).second, "Duplicate calibration option");
            }
        }
        if (!execute) { std::cout << "{\"state\":\"INACTIVE\",\"device_created\":false,\"model_loaded\":false}\n"; return 0; }
        for (const auto* name : {L"--shader-root", L"--pci", L"--luid", L"--output"})
            require(args.count(name) && !args.at(name).empty(), "Explicit shader root, PCI, LUID and fresh output required");
        std::filesystem::path root(args.at(L"--shader-root"));
        require(root.is_absolute() && std::filesystem::is_directory(root) &&
                std::filesystem::is_regular_file(root / L"runtime" / L"linear.hlsl"), "Absolute existing shader root and linear shader required");
        std::string pci = ascii(args.at(L"--pci")), luid = ascii(args.at(L"--luid"));
        std::filesystem::path output(args.at(L"--output"));
        require(output.is_absolute() && std::filesystem::is_directory(output.parent_path()), "Absolute fresh task-owned result path required");
        receipt = CreateFileW(output.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(receipt != INVALID_HANDLE_VALUE, "Fresh task-owned calibration receipt required");
        report["shader_root"] = root.u8string(); report["requested_pci"] = pci; report["requested_luid"] = luid;
        writeFresh(receipt, report);
        Deadline deadline;
        {
            Device device(root.wstring(), pci, luid);
            report["native_executed"] = true;
            report["device_identity"] = Json::parse(device.identityJson());
            report["memory_before"] = Json::parse(device.memoryJson());
            run(device, report, deadline);
            device.drain(); deadline.check();
            require(device.trackedBufferBytes() == 0, "Completed calibration left owned buffers live");
            report["memory_after"] = Json::parse(device.memoryJson());
        }
        report["elapsed_seconds"] = deadline.seconds();
        report["state"] = "THREE_PHASE_CALIBRATION_PASS";
        writeFresh(receipt, report); CloseHandle(receipt); receipt = INVALID_HANDLE_VALUE;
        std::cout << "{\"state\":\"THREE_PHASE_CALIBRATION_PASS\",\"passed\":true,\"full_model_accepted\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        report["passed"] = false; report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = error.what();
        if (receipt != INVALID_HANDLE_VALUE) { try { writeFresh(receipt, report); } catch (...) {} CloseHandle(receipt); }
        std::cerr << "Calibration FAIL_OR_INCOMPLETE: " << error.what() << "\n";
        return 1;
    }
}
