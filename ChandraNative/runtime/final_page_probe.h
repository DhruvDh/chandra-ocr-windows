// New ChandraNative code, MPL-2.0. Portable core of the inactive-by-default, model-free final-page
// probe (docs/directcompute-final-page-probe.md). It mirrors only the geometry of the tied-head shard 0
// that failed the native-0 control (row 26213, packed words 768..831, shard byte 134213632), on one
// original-shape buffer (ByteWidth 134215680) and one padded-allocation contrast (134217728) with the same
// logical shape and SRV extent. Three observations of the selected final rows are compared against the
// CPU bytes: an independent CopySubresourceRegion-to-staging read, the existing weight_row_checksum.hlsl
// screen and the existing weight_row_words.hlsl SRV row copy. Every observation is a copied-byte
// observation; none proves physical storage, allocation overlap, driver cause or time of origin.
// Windows host: final_page_probe.cpp. CPU tests: final_page_probe_test.cpp. No model or listener.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../vendor/nlohmann/json.hpp"
#include "vision_dispatch_calibration.h" // Reused, unchanged: portable SHA-256 only.

namespace chandra::final_page_probe {
using Json = nlohmann::json;

constexpr uint32_t rowWords = 1280, rowBytes = rowWords * 4;              // 2560 BF16 columns.
constexpr uint32_t shardRows = 26214;                                     // 128 MiB / 5120, floor.
constexpr uint64_t logicalBytes = uint64_t(shardRows) * rowBytes;         // 134215680 = 128 MiB - 2048.
constexpr uint32_t logicalWords = uint32_t(logicalBytes / 4);
constexpr uint64_t paddedBytes = 128ull * 1024 * 1024;                     // 134217728.
constexpr uint32_t paddedWords = uint32_t(paddedBytes / 4);
constexpr uint32_t finalRow = shardRows - 1, neighbourRow = shardRows - 2; // 26213 and 26212.
constexpr uint32_t selectedRows = 2, selectedFirstRow = neighbourRow;
constexpr uint64_t selectedFirstByte = uint64_t(selectedFirstRow) * rowBytes; // 134205440.
constexpr uint32_t selectedBytes = selectedRows * rowBytes;                    // 10240 would exceed 8 KiB: see below.
// The direct copy reads only the final row (5120 bytes) plus, for the padded contrast, the 2048 physical
// padding bytes as a separate non-gating record: at most 7168 bytes of staging per read.
constexpr uint64_t directFirstByte = uint64_t(finalRow) * rowBytes;         // 134210560.
constexpr uint32_t directBytes = rowBytes;
constexpr uint32_t paddingBytes = uint32_t(paddedBytes - logicalBytes);    // 2048.
constexpr uint32_t stagingBytes = 8192;
constexpr uint64_t retainedBlockByte = directFirstByte + 768 * 4;           // 134213632 = 128 MiB - 4096.
constexpr uint64_t finalPageByte = (logicalBytes / 4096) * 4096;            // 134213632: first byte of the half-filled page.
constexpr uint32_t checksumThreads = 64, rowCopyThreads = 256;
constexpr uint32_t churnIterations = 8, churnWords = 256 * 1024;             // 1 MiB scratch, one live at a time.
constexpr uint32_t dispatchTimeoutMs = 2000, slowDispatchMs = 250;
constexpr const char* retainedSourceSha256 = "e696b6bda9b22dba3ad2a49132d6353302f5b20ad2627909ae5474586a46c07f";

static_assert(logicalBytes == 134215680ull && paddedBytes - logicalBytes == 2048, "Native-0 shard 0 geometry");
static_assert(retainedBlockByte == 134213632ull && retainedBlockByte == finalPageByte, "Retained block starts the final page");
static_assert(directBytes + paddingBytes <= stagingBytes, "Staging stays at or below 8 KiB");

enum class Contrast { original, padded };
inline const char* name(Contrast c) { return c == Contrast::original ? "original" : "padded"; }
inline uint64_t physicalBytes(Contrast c) { return c == Contrast::original ? logicalBytes : paddedBytes; }
// SRV visibility is the logical shape in both contrasts; only ByteWidth changes.
inline uint32_t srvWords(Contrast) { return logicalWords; }

// Deterministic finite BF16 words: splitmix64 per word; an all-ones exponent is cleared to stay finite.
inline uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31);
}
inline uint16_t finiteBf16(uint16_t h) { return ((h >> 7) & 0xffu) == 0xffu ? uint16_t(h & ~0x4000u) : h; }
inline uint32_t syntheticWord(uint32_t index) {
    const uint64_t r = splitmix(0x46494e414c504147ull ^ index); // "FINALPAG"
    return uint32_t(finiteBf16(uint16_t(r))) | (uint32_t(finiteBf16(uint16_t(r >> 16))) << 16);
}
inline uint32_t paddingWord(uint32_t index) { return 0xa5a50000u | (index & 0xffffu); }
inline bool finiteWord(uint32_t w) {
    return ((w >> 7) & 0xffu) != 0xffu && ((w >> 23) & 0xffu) != 0xffu;
}

// Fills words [0, physical/4). The optional retained source row (exactly 5120 admitted bytes) is
// spliced at the final row; padding words carry a sentinel so their copied bytes are defined.
inline void fill(std::vector<uint32_t>& out, Contrast c, const std::vector<uint8_t>* sourceRow) {
    const uint32_t words = uint32_t(physicalBytes(c) / 4);
    out.assign(words, 0);
    for (uint32_t i = 0; i < logicalWords; ++i) out[i] = syntheticWord(i);
    for (uint32_t i = logicalWords; i < words; ++i) out[i] = paddingWord(i);
    if (sourceRow) {
        if (sourceRow->size() != rowBytes) throw std::runtime_error("Retained source row must be exactly 5120 bytes");
        std::memcpy(out.data() + uint64_t(finalRow) * rowWords, sourceRow->data(), rowBytes);
    }
}

// The existing checksum screen (weight_row_checksum.hlsl): sum of (2i+1)*word with uint32 wrap.
inline uint32_t rowChecksum(const uint32_t* words, uint32_t count) {
    uint32_t sum = 0; for (uint32_t i = 0; i < count; ++i) sum += (2 * i + 1) * words[i]; return sum;
}

struct Range { uint64_t firstByte = 0, lastByte = 0; uint32_t words = 0; };
struct Difference { std::vector<uint32_t> words; std::vector<Range> ranges; };
// Compares observed words against expected words that both start at absolute shard byte base.
inline Difference compare(const uint32_t* expected, const uint32_t* observed, uint32_t count, uint64_t base) {
    Difference d;
    for (uint32_t i = 0; i < count; ++i) {
        if (expected[i] == observed[i]) continue;
        d.words.push_back(i); const uint64_t at = base + uint64_t(i) * 4;
        if (!d.ranges.empty() && d.ranges.back().lastByte + 1 == at) { d.ranges.back().lastByte = at + 3; ++d.ranges.back().words; }
        else d.ranges.push_back({at, at + 3, 1});
    }
    return d;
}
inline std::string hex(uint32_t w) { char t[11]; std::snprintf(t, sizeof(t), "0x%08x", w); return t; }
inline Json describe(const Difference& d, const uint32_t* expected, const uint32_t* observed, uint64_t base) {
    Json ranges = Json::array(), words = Json::array();
    for (const auto& r : d.ranges)
        ranges.push_back({{"first_shard_byte", r.firstByte}, {"last_shard_byte", r.lastByte}, {"words", r.words},
                          {"page_offset_of_first_byte", r.firstByte % 4096}, {"distance_to_128MiB", paddedBytes - r.firstByte}});
    for (size_t k = 0; k < d.words.size() && k < 256; ++k) {
        const uint32_t i = d.words[k];
        words.push_back({{"shard_byte", base + uint64_t(i) * 4}, {"expected", hex(expected[i])}, {"observed", hex(observed[i])}});
    }
    return {{"identical", d.words.empty()}, {"differing_words", d.words.size()}, {"ranges", ranges},
            {"first_256_differences", words}, {"scope", "Copied-byte comparison of the selected region only"}};
}
// Recognises the native-0 retained signature without using it as a gate.
inline bool retainedSignature(const uint32_t* observedRow) {
    uint32_t pairs = 0;
    for (uint32_t i = 768; i < 832; i += 2) pairs += observedRow[i] == 0x24000006u && observedRow[i + 1] == 8u;
    return pairs >= 16;
}

// ---- Host-boundary contracts (repair of the rejected source11 host) ----
// Constant upload ABI. Both shaders declare cbuffer Params of eight uint (32 bytes); the D3D11 constant buffer is
// created with ByteWidth 256, and UpdateSubresource with a null box reads ByteWidth bytes from the source pointer.
// The uploaded object is therefore a zero-initialised 256-byte Constants whose first eight words are Params.
constexpr uint32_t constantBufferBytes = 256, paramWords = 8;
struct Constants { uint32_t words[constantBufferBytes / 4] = {}; };
static_assert(sizeof(Constants) == constantBufferBytes && constantBufferBytes % 16 == 0, "Full constant upload object");
inline Constants params(std::initializer_list<uint32_t> values) {
    if (values.size() > paramWords) throw std::runtime_error("Params has eight words");
    Constants c; std::copy(values.begin(), values.end(), c.words); return c;
}
// Rejects any upload whose source object is not exactly the descriptor ByteWidth (the source11 call passed 32 bytes).
inline void requireFullConstantUpload(uint64_t descriptorByteWidth, uint64_t sourceObjectBytes) {
    if (descriptorByteWidth != constantBufferBytes || sourceObjectBytes != descriptorByteWidth)
        throw std::runtime_error("Constant upload must supply exactly the 256-byte descriptor width");
}

// Buffer/view/copy bounds checked at the Windows helper boundary for every call.
inline void requireViewWords(uint64_t byteWidth, uint64_t words) {
    if (byteWidth == 0 || byteWidth > paddedBytes || byteWidth % 4 || words > byteWidth / 4)
        throw std::runtime_error("View words exceed the buffer ByteWidth");
}
inline void requireCopyBox(uint64_t sourceByteWidth, uint64_t first, uint64_t bytes, uint64_t destinationOffset, uint64_t destinationByteWidth) {
    if (sourceByteWidth > 0xffffffffull || destinationByteWidth > 0xffffffffull || bytes == 0 || bytes % 4 || first % 4 || destinationOffset % 4 ||
        first > sourceByteWidth || bytes > sourceByteWidth - first || destinationOffset > destinationByteWidth || bytes > destinationByteWidth - destinationOffset)
        throw std::runtime_error("Copy box outside source or destination bounds");
}

// Bounded file admission: one open handle, at most maxBytes + 1 bytes read, exact size, no stream error, pinned hash.
// The admitted in-memory bytes are what the host hashes, compiles or splices; the path is never reopened.
constexpr uint32_t shaderSourceLimit = 4096, bytecodeLimit = 16384;
struct FilePolicy { const char* label; uint32_t bytes; const char* sha256; };
inline constexpr FilePolicy sourceRowPolicy{"source row", rowBytes, retainedSourceSha256};
inline constexpr FilePolicy checksumShaderPolicy{"runtime/weight_row_checksum.hlsl", 949, "ea462a1a71008cb56e86ce54f77fc7903ffa5a2e422ae722dfc1752a182e0c74"};
inline constexpr FilePolicy rowCopyShaderPolicy{"runtime/weight_row_words.hlsl", 683, "71fcda89e31d3c7a423bcb9ae73a9dc97cc89bd5463244dd9c965b5bbcd1fa97"};
static_assert(checksumShaderPolicy.bytes <= shaderSourceLimit && rowCopyShaderPolicy.bytes <= shaderSourceLimit, "Pinned shader sizes");
inline std::string sha256(const void* data, size_t bytes) { return vision_calibration::sha256Bytes(static_cast<const char*>(data), bytes); }
struct AdmittedFile { std::string path; std::vector<uint8_t> bytes; std::string sha256; };
inline AdmittedFile admitFile(const std::filesystem::path& path, const FilePolicy& policy) {
    if (policy.bytes == 0 || policy.bytes > rowBytes) throw std::runtime_error("Admission policy bound");
    std::ifstream in;
    try { in.open(path, std::ios::binary); } catch (const std::exception&) { throw std::runtime_error(std::string(policy.label) + ": cannot open"); }
    if (!in.is_open()) throw std::runtime_error(std::string(policy.label) + ": cannot open");
    std::vector<uint8_t> bytes(size_t(policy.bytes) + 1); // Bounded before reading; the extra byte detects oversize.
    std::streamsize got = 0;
    try { in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())); got = in.gcount(); }
    catch (const std::exception&) { throw std::runtime_error(std::string(policy.label) + ": read error"); }
    if (in.bad()) throw std::runtime_error(std::string(policy.label) + ": read error");
    if (got > std::streamsize(policy.bytes)) throw std::runtime_error(std::string(policy.label) + ": extra bytes beyond the pinned size");
    if (!in.eof()) throw std::runtime_error(std::string(policy.label) + ": read error");
    if (got < std::streamsize(policy.bytes)) throw std::runtime_error(std::string(policy.label) + ": short file");
    bytes.resize(policy.bytes);
    AdmittedFile f{path.u8string(), std::move(bytes), {}}; f.sha256 = sha256(f.bytes.data(), f.bytes.size());
    if (f.sha256 != policy.sha256) throw std::runtime_error(std::string(policy.label) + ": SHA-256 differs from the pinned identity");
    return f;
}
// The host compiles with a null include handler; this screen also refuses any preprocessor include text.
inline void requireNoInclude(const std::vector<uint8_t>& source) {
    const std::string text(source.begin(), source.end());
    for (size_t at = text.find('#'); at != std::string::npos; at = text.find('#', at + 1)) {
        size_t k = at + 1; while (k < text.size() && (text[k] == ' ' || text[k] == '\t')) ++k;
        if (text.compare(k, 7, "include") == 0) throw std::runtime_error("Shader include dependencies are refused");
    }
}

// Timing. Every operation starts before its first submission call (UpdateSubresource, Dispatch or Copy) and
// finishes after its completed event drain and Map/copy-out. Pending past the limit and late success are both
// rejected; a declared whole-run deadline covers every operation and the CPU work between them. This in-process
// deadline cannot preempt a blocked driver call: root's external wall timer and kill-on-close Job are required.
constexpr uint32_t slowOperationMs = 250, longOperationMs = 2000, runDeadlineMs = 120000;
constexpr uint32_t shaderOperationsPerContrast = 1 + 3 + churnIterations + 3 + 1; // upload, 3 reads, churn, 3 reads, release.
constexpr uint32_t directOperationsPerContrast = 1 + 1 + churnIterations + 1 + 1;
struct Clock {
    virtual ~Clock() = default;
    virtual uint64_t nowMs() = 0;
    virtual void pause() = 0;
};
struct SteadyClock final : Clock {
    uint64_t nowMs() override {
        return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    void pause() override { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
};
class Timer {
public:
    struct Op { std::string label; uint64_t start = 0, limit = 0; };
    Timer(Clock& clock, uint32_t maxOperations, uint64_t runLimitMs = runDeadlineMs)
        : clock_(clock), start_(clock.nowMs()), runLimit_(runLimitMs), maxOperations_(maxOperations) {}
    void requireRun(const std::string& at) {
        if (clock_.nowMs() - start_ > runLimit_) throw std::runtime_error("Whole-run deadline exceeded at " + at);
    }
    Op begin(const std::string& label, uint64_t limitMs) {
        requireRun(label);
        if (++operations_ > maxOperations_) throw std::runtime_error("Declared operation count exceeded at " + label);
        return {label, clock_.nowMs(), limitMs};
    }
    void check(const Op& op) {
        if (clock_.nowMs() - op.start > op.limit) throw std::runtime_error("Operation deadline exceeded while pending at " + op.label);
        requireRun(op.label);
    }
    void finish(const Op& op) {
        const uint64_t ms = clock_.nowMs() - op.start;
        records_.push_back({{"label", op.label}, {"ms", ms}, {"limit_ms", op.limit}});
        if (ms > op.limit) throw std::runtime_error("Late completion rejected at " + op.label);
        requireRun(op.label);
    }
    void pause() { clock_.pause(); }
    uint32_t operations() const { return operations_; }
    const Json& records() const { return records_; }
private:
    Clock& clock_; uint64_t start_, runLimit_; uint32_t maxOperations_, operations_ = 0; Json records_ = Json::array();
};
// One timed operation: submit (all API submission calls and the event End/Flush), poll the drain, then poll the
// Map/copy-out. The fake test delays each callback to prove the measured interval covers all three.
template <class Submit, class Drained, class Mapped>
void timedOperation(Timer& t, const std::string& label, uint64_t limitMs, Submit&& submit, Drained&& drained, Mapped&& mapped) {
    const Timer::Op op = t.begin(label, limitMs);
    submit();
    while (!drained()) { t.check(op); t.pause(); }
    while (!mapped()) { t.check(op); t.pause(); }
    t.finish(op);
}

// Allocation accounting. Requested GPU payload is the sum of requested ByteWidths that may be live together; it is
// not a measurement of driver or physical allocation. Host payload is an upper bound on probe-owned vectors
// live at once. Process, compiler-internal, runtime and driver allocations are unmeasured here and are bounded
// only by the external Job process-memory limit that root owns.
constexpr uint32_t reportLimitBytes = 1024 * 1024, jsonDomFactor = 8;
struct Forecast {
    uint64_t gpuLarge = 0, gpuScratch = 0, gpuOutput = 0, gpuStaging = 0, gpuConstants = 0, requestedGpuPayload = 0;
    uint64_t hostInput = 0, hostSourceRow = 0, hostShaderSource = 0, hostBytecode = 0, hostReadback = 0, hostComparison = 0, hostReport = 0, hostPayload = 0;
    uint64_t total = 0;
};
inline Forecast forecast() {
    Forecast f;
    f.gpuLarge = paddedBytes; f.gpuScratch = uint64_t(churnWords) * 4; f.gpuOutput = rowBytes; f.gpuStaging = stagingBytes; f.gpuConstants = constantBufferBytes;
    f.requestedGpuPayload = f.gpuLarge + f.gpuScratch + f.gpuOutput + f.gpuStaging + f.gpuConstants;
    f.hostInput = paddedBytes;                                  // one fill vector at a time.
    f.hostSourceRow = rowBytes + 1;                             // admission buffer includes the oversize sentinel.
    f.hostShaderSource = 2ull * (shaderSourceLimit + 1);
    f.hostBytecode = 2ull * bytecodeLimit;                      // copied-out bytecode cap; compiler working memory unmeasured.
    f.hostReadback = 4ull * stagingBytes;                       // combined read, direct/padding/rowCopy/checksum copies, one temporary.
    f.hostComparison = 3ull * (uint64_t(rowWords) * 4 + uint64_t(rowWords / 2 + 1) * sizeof(Range)); // three Differences.
    f.hostReport = uint64_t(reportLimitBytes) * (1 + jsonDomFactor); // serialized cap plus an assumed DOM overhead.
    f.hostPayload = f.hostInput + f.hostSourceRow + f.hostShaderSource + f.hostBytecode + f.hostReadback + f.hostComparison + f.hostReport;
    f.total = f.requestedGpuPayload + f.hostPayload;
    return f;
}
inline uint32_t groups(uint32_t threads, uint32_t perGroup) { return (threads + perGroup - 1) / perGroup; }
inline Json plan(bool directOnly = false) {
    const Forecast f = forecast();
    if (f.total >= 512ull * 1024 * 1024) throw std::runtime_error("Forecast exceeds 512 MiB");
    const uint32_t operations = 2 * (directOnly ? directOperationsPerContrast : shaderOperationsPerContrast);
    const uint64_t operationBudget = 2ull * 2 * longOperationMs + uint64_t(operations - 4) * slowOperationMs;
    Json contrasts = Json::array();
    for (Contrast c : {Contrast::original, Contrast::padded})
        contrasts.push_back({{"name", name(c)}, {"logical_bytes", logicalBytes}, {"physical_byte_width", physicalBytes(c)},
                             {"srv_words", srvWords(c)}, {"srv_bytes", uint64_t(srvWords(c)) * 4},
                             {"physical_padding_bytes", physicalBytes(c) - logicalBytes}});
    Json dispatches = nullptr;
    if (!directOnly)
        dispatches = {{"checksum", {{"shader", checksumShaderPolicy.label}, {"sha256", checksumShaderPolicy.sha256}, {"bytes", checksumShaderPolicy.bytes},
                                    {"rows", selectedRows}, {"first_row", selectedFirstRow}, {"groups", groups(selectedRows, checksumThreads)}}},
                      {"row_copy", {{"shader", rowCopyShaderPolicy.label}, {"sha256", rowCopyShaderPolicy.sha256}, {"bytes", rowCopyShaderPolicy.bytes},
                                    {"first_word", uint64_t(finalRow) * rowWords}, {"count", rowWords}, {"groups", groups(rowWords, rowCopyThreads)}}},
                      {"compile", "D3DCompile of the admitted in-memory bytes, null include handler, cs_5_0"},
                      {"constant_upload_bytes", constantBufferBytes}};
    return {{"schema", "chandra.directcompute.final-page-probe-plan.v2"}, {"model_free", true}, {"device_created", false},
            {"mode", directOnly ? "direct_only" : "direct_and_srv"},
            {"geometry", {{"row_words", rowWords}, {"row_bytes", rowBytes}, {"shard_rows", shardRows}, {"final_row", finalRow},
                          {"neighbour_row", neighbourRow}, {"final_page_first_byte", finalPageByte},
                          {"retained_block_first_byte", retainedBlockByte}, {"retained_block_words", "768..831"}}},
            {"contrasts", contrasts}, {"large_buffers_live_at_once", 1},
            {"direct_copy", {{"api", "ID3D11DeviceContext::CopySubresourceRegion to a staging buffer, no SRV or shader"}, {"first_byte", directFirstByte},
                             {"bytes", directBytes}, {"padded_padding_record_bytes", paddingBytes}, {"staging_bytes", stagingBytes}}},
            {"dispatches", dispatches},
            {"churn", {{"iterations", churnIterations}, {"scratch_bytes", uint64_t(churnWords) * 4},
                       {"operation", directOnly ? "CopySubresourceRegion of the final row into the scratch" : "row_copy dispatch of the final row into the scratch"}}},
            {"source_row_admission", {{"bytes", sourceRowPolicy.bytes}, {"sha256", sourceRowPolicy.sha256}}},
            {"timing", {{"operation_scope", "before the first UpdateSubresource/Dispatch/Copy through completed event drain and Map copy-out"},
                        {"slow_operation_ms", slowOperationMs}, {"upload_and_release_operation_ms", longOperationMs},
                        {"operations", operations}, {"declared_operation_budget_ms", operationBudget}, {"whole_run_deadline_ms", runDeadlineMs},
                        {"late_success", "rejected"},
                        {"external_requirement", "root supplies an independent finite wall timer and kill-on-close Job; a blocked driver call cannot be preempted in process"}}},
            {"forecast_bytes", {{"requested_gpu_payload", {{"large", f.gpuLarge}, {"scratch", f.gpuScratch}, {"output", f.gpuOutput},
                                                           {"staging", f.gpuStaging}, {"constants", f.gpuConstants}, {"total", f.requestedGpuPayload}}},
                                {"host_payload_bound", {{"input", f.hostInput}, {"source_row", f.hostSourceRow}, {"shader_source", f.hostShaderSource},
                                                        {"bytecode", f.hostBytecode}, {"readback", f.hostReadback}, {"comparison", f.hostComparison},
                                                        {"report", f.hostReport}, {"total", f.hostPayload}}},
                                {"total", f.total}, {"limit", 512ull * 1024 * 1024},
                                {"process_and_driver_allocation", "unmeasured; bounded only by the external Job process-memory limit"},
                                {"byte_width_meaning", "requested resource size, not physical or driver page allocation"}}},
            {"full_buffer_readback", false},
            {"claim_limit", "A synthetic pass cannot clear the trained native-0 failure; an SRV copy never proves physical storage"}};
}

struct Observation { std::vector<uint32_t> direct, padding, rowCopy, checksums; };
// Gates one probe phase. Each observation is compared independently against the CPU bytes. In direct-only mode
// the SRV row copy and checksum screen are absent by construction and the direct copy alone gates.
inline Json judge(const std::vector<uint32_t>& cpu, Contrast c, const Observation& o, bool directOnly = false) {
    const uint32_t* expectRow = cpu.data() + uint64_t(finalRow) * rowWords;
    if (o.direct.size() != rowWords) throw std::runtime_error("Observation geometry mismatch");
    if (directOnly ? (!o.rowCopy.empty() || !o.checksums.empty()) : (o.rowCopy.size() != rowWords || o.checksums.size() != selectedRows))
        throw std::runtime_error("Observation geometry mismatch");
    const Difference direct = compare(expectRow, o.direct.data(), rowWords, directFirstByte);
    Json sums = Json::array(); bool sumsMatch = true, copiedMatch = true; Json srvCopy = nullptr, screen = nullptr, srvSignature = nullptr, agree = nullptr;
    if (!directOnly) {
        const Difference copied = compare(expectRow, o.rowCopy.data(), rowWords, directFirstByte);
        copiedMatch = copied.words.empty();
        for (uint32_t r = 0; r < selectedRows; ++r) {
            const uint32_t want = rowChecksum(cpu.data() + uint64_t(selectedFirstRow + r) * rowWords, rowWords);
            sumsMatch = sumsMatch && want == o.checksums[r];
            sums.push_back({{"row", selectedFirstRow + r}, {"expected", hex(want)}, {"observed", hex(o.checksums[r])}, {"match", want == o.checksums[r]}});
        }
        srvCopy = describe(copied, expectRow, o.rowCopy.data(), directFirstByte);
        screen = {{"rows", sums}, {"all_match", sumsMatch}};
        srvSignature = retainedSignature(o.rowCopy.data()); agree = o.direct == o.rowCopy;
    }
    Json padding = nullptr;
    if (c == Contrast::padded) {
        if (o.padding.size() != paddingBytes / 4) throw std::runtime_error("Padding observation geometry mismatch");
        const Difference p = compare(cpu.data() + logicalWords, o.padding.data(), paddingBytes / 4, logicalBytes);
        padding = describe(p, cpu.data() + logicalWords, o.padding.data(), logicalBytes);
        padding["gating"] = false; padding["visible_through_srv"] = false;
    } else if (!o.padding.empty()) throw std::runtime_error("Original contrast has no physical padding to read");
    const bool pass = direct.words.empty() && copiedMatch && sumsMatch;
    return {{"contrast", name(c)}, {"pass", pass}, {"mode", directOnly ? "direct_only" : "direct_and_srv"},
            {"direct_copy", describe(direct, expectRow, o.direct.data(), directFirstByte)},
            {"srv_row_copy", srvCopy},
            {"checksum_screen", screen},
            {"padding_record", padding},
            {"retained_signature", {{"direct_copy", retainedSignature(o.direct.data())}, {"srv_row_copy", srvSignature}}},
            {"direct_and_srv_agree", agree},
            {"physical_storage_proven", false},
            {"interpretation", pass ? "Every copied observation equals the CPU bytes for this synthetic input only"
                                    : "At least one copied observation differs; cause, origin time and physical storage remain unknown"}};
}

struct Arguments { bool execute = false, directOnly = false; std::string pci, luid, shaderRoot, sourceRow, output; };
inline Arguments parse(const std::vector<std::string>& args) {
    Arguments a; std::map<std::string, std::string*> values{{"--pci", &a.pci}, {"--luid", &a.luid}, {"--shader-root", &a.shaderRoot},
                                                            {"--source-row", &a.sourceRow}, {"--output", &a.output}};
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--execute") { if (a.execute) throw std::runtime_error("Duplicate --execute"); a.execute = true; continue; }
        if (args[i] == "--direct-only") { if (a.directOnly) throw std::runtime_error("Duplicate --direct-only"); a.directOnly = true; continue; }
        auto it = values.find(args[i]);
        if (it == values.end() || i + 1 >= args.size()) throw std::runtime_error("Unknown or incomplete argument: " + args[i]);
        if (!it->second->empty()) throw std::runtime_error("Duplicate argument: " + args[i]);
        *it->second = args[++i]; if (it->second->empty()) throw std::runtime_error("Empty argument: " + args[i - 1]);
    }
    if (a.execute && (a.pci.empty() || a.luid.empty() || a.output.empty()))
        throw std::runtime_error("--execute requires explicit --pci, --luid and --output");
    if (a.execute && a.directOnly != a.shaderRoot.empty())
        throw std::runtime_error("--execute requires --shader-root, except --direct-only, which refuses it");
    if (!a.execute && (!a.pci.empty() || !a.luid.empty() || !a.sourceRow.empty() || !a.shaderRoot.empty() || !a.output.empty()))
        throw std::runtime_error("Adapter, shader, source-row and output arguments are accepted only with --execute");
    return a;
}

// Every file input is admitted from memory before the Job screen, adapter enumeration, Device or large allocation.
struct Admitted { bool sourceRow = false; AdmittedFile source, checksumShader, rowCopyShader; };
inline Admitted admitInputs(const Arguments& a) {
    if (!a.execute) throw std::runtime_error("Admission is only for --execute");
    Admitted m;
    if (!a.sourceRow.empty()) { m.source = admitFile(a.sourceRow, sourceRowPolicy); m.sourceRow = true; }
    if (!a.directOnly) {
        const std::filesystem::path root(a.shaderRoot);
        m.checksumShader = admitFile(root / checksumShaderPolicy.label, checksumShaderPolicy); requireNoInclude(m.checksumShader.bytes);
        m.rowCopyShader = admitFile(root / rowCopyShaderPolicy.label, rowCopyShaderPolicy); requireNoInclude(m.rowCopyShader.bytes);
    }
    return m;
}
// Host hooks in the order the Windows host calls them; the portable test counts them to prove that inactive mode and
// every admission rejection happen before any Job screen, adapter, Device or large allocation.
struct Host {
    virtual ~Host() = default;
    virtual Json execute(const Arguments&, const Admitted&) = 0; // Job screen, adapter, Device, probe.
};
inline Json run(const Arguments& a, Host& host) {
    if (!a.execute) return plan(a.directOnly);
    const Admitted m = admitInputs(a);
    return host.execute(a, m);
}
} // namespace chandra::final_page_probe
