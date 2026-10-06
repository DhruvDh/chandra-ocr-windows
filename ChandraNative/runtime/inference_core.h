// New code, MPL-2.0. Shared Windows input authentication, reports and one-request generation for the
// connected CLI (inference.cpp) and the resident worker (worker.cpp). Moved from inference.cpp without
// semantic change: no tokenizer, processor, engine fallback, request replay or implicit adapter selection.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include "api.h"
#include "text_model.h"
#include "vision_model.h"
#include "../vendor/nlohmann/json.hpp"
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::dc::inference {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr uint32_t vocabulary = 248320, imageToken = 248056, textWidth = 2560;
constexpr uint32_t contextLimit = 16384, normalOutputLimit = 12384;
constexpr uint64_t inputByteLimit = 256ull * 1024 * 1024, manifestByteLimit = 1024 * 1024;
constexpr const char* modelRevision = "af93b47dba1b47b6640c86ccf487ed2260ab9a09";
constexpr const char* inputSchema = "chandra.directcompute.input.v1";
constexpr const char* chandraCallerSha = "185ef7cbff08dea1a196f7fd96993f596d5943b90c2edcdaba001f0487cd1921";
constexpr const char* generationConfigSha = "0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e";
constexpr const char* tokenizerConfigSha = "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8";
static_assert(sizeof(float) == 4 && sizeof(int64_t) == 8 && std::numeric_limits<float>::is_iec559, "Original IEEE FP32/I64 input ABI");

void require(bool value, const std::string& reason);
uint64_t product(uint64_t a, uint64_t b);
uint64_t sum(uint64_t a, uint64_t b);
uint64_t unsignedNumber(const Json& j);
std::string utf8(const std::wstring& value);
std::string digestText(const Json& value);
Json strictJson(const std::vector<char>& bytes);
struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    explicit Handle(HANDLE h) : value(h) { require(h != INVALID_HANDLE_VALUE, "Opening an input/output file failed"); }
    ~Handle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete; Handle& operator=(const Handle&) = delete;
};
struct Sha256 {
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr; std::vector<UCHAR> object;
    Sha256();
    ~Sha256();
    Sha256(const Sha256&) = delete; Sha256& operator=(const Sha256&) = delete;
    void update(const void* data, uint32_t bytes);
    std::string finish();
};
// Read once through a non-write-shared file handle; hash exactly the bytes used.
void authenticatedRead(const std::filesystem::path& path, void* destination, uint64_t bytes, const std::string& expected);
// A local relative name below root: no rooted/device path, traversal or reparse component.
std::filesystem::path localFile(const std::filesystem::path& root, const Json& name);
Json authenticatedMetadata(const std::filesystem::path& root, const Json& name, const std::string& sha);
void authenticateCaller(const Json& generation, const std::filesystem::path& root);
struct Descriptor { std::filesystem::path path; std::string dtype, sha; uint64_t bytes = 0, elements = 0; std::vector<uint32_t> shape; };
Descriptor descriptor(const Json& row, const std::filesystem::path& root);
template<class T> std::vector<T> tensor(const Descriptor& d, const char* dtype, const std::vector<uint32_t>& shape) {
    require(d.dtype == dtype && d.shape == shape && d.bytes == product(d.elements, sizeof(T)), "Exact tensor ABI required");
    std::vector<T> result(static_cast<size_t>(d.elements)); authenticatedRead(d.path, result.data(), d.bytes, d.sha); return result;
}
struct Input {
    Json manifest; std::map<std::string, Descriptor> descriptors;
    std::vector<uint32_t> ids, visionRowMap, stopIds; std::vector<float> pixels;
    TextPositions positions; std::vector<VisionGrid> grids; uint32_t maximumPosition = 0, patchRows = 0;
    uint64_t tensorBytes = 0; VisionForecast visionForecast;
};
// Authenticates typed raw tensors, geometry, caller stops, positions and finite pixels on the CPU.
// No Device exists or is touched; every refusal throws before any hardware work.
Input authenticateInput(const std::filesystem::path& manifest, const std::string& manifestSha);
Json hostMemory();
Json visionEstimate(const VisionForecast& f);
// The CLI forecast result without its optional diagnostic section.
Json forecastReport(const Input& in, const std::string& manifestSha, uint32_t diagnosticCap);
// Initial result fields of one executed request; passed stays false until the caller proves retirement.
Json executionReport(const Input& in, const std::string& manifestSha, uint32_t diagnosticCap);
// Appends the generated IDs, complete history, phases and timing scope.
void finishExecutionReport(Json& report, const Input& in, const std::vector<uint32_t>& generated, const Json& phases, const Json& memories, Clock::time_point start);
class NewOutput {
    Handle file;
public:
    explicit NewOutput(const std::filesystem::path& path);
    void line(const Json& value);
};
struct Greedy { uint32_t token = 0; float best = 0, second = 0; uint32_t ties = 0; };
Greedy greedy(Device& device, const TextResult& result);
Buffer merge(Device& device, const Input& in, const Buffer& text, const Buffer& vision);
// Optional synchronous Device-thread hooks. Every member is empty in the CLI's default flow, where
// generate() issues exactly the predecessor's allocations, dispatches, drains, readbacks and releases.
// checkpoint(name) runs only at host boundaries before completion; a throw there or from an observer
// poisons, retires and drains the request like any graph failure. token(index,row) runs after the
// row was durably written. Observers keep their borrowed, read-only contracts.
struct RequestHooks {
    std::function<void(const char*)> phase;
    std::function<void(const char*)> checkpoint;
    VisionObserver vision; TextObserver text;
    std::function<void(const Buffer&, uint32_t rows)> merged;
    std::function<void(uint32_t index, const Json& row)> token;
};
// One B1 page on an existing Device and authenticated complete weights: vision forward, ordered merge,
// a fresh TextRequest prefill and greedy cached decode up to the stop IDs or the allowance (an explicit
// nonzero diagnosticCap shortens it). Writes output/generated-tokens.jsonl and request fields into
// report. On any exception it marks the request failed, retires its cache, drains, records the drain
// outcome and rethrows. Device and weights stay owned by the caller.
void generate(Device& device, ModelWeights& weights, const Input& in, uint32_t diagnosticCap, const std::filesystem::path& output,
    Json& report, std::vector<uint32_t>& generated, const RequestHooks& hooks);
}
