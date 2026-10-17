// New code, MPL-2.0. Fixed CPU source observer; no alternate upload, GPU operation or repair.
#pragma once
#include "../vendor/nlohmann/json.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace chandra::dc::import_row {
using Json = nlohmann::json;
constexpr uint32_t row = 26213, rowBytes = 5120, rowByteOffset = 134210560, shardBytes = 134215680;
constexpr uint32_t maximumProgressRecords = 8, maximumProgressRecordBytes = 8192;
struct Context {
    uint64_t fileBytes = 0, payloadBegin = 0, tensorBegin = 0, shardFileOffset = 0;
    uint64_t mappedFileOffset = 0, mappedBytes = 0, skippedPrefix = 0;
    uint32_t allocationGranularity = 0;
};
inline void validate(const Context& c, uint32_t bytes) {
    // Subtractions avoid overflow before any source pointer arithmetic or read.
    if (bytes != shardBytes || c.fileBytes != 10591220088ull || !c.allocationGranularity ||
        c.payloadBegin > c.fileBytes || c.tensorBegin > c.fileBytes - c.payloadBegin ||
        c.shardFileOffset != c.payloadBegin + c.tensorBegin ||
        c.shardFileOffset > c.fileBytes || bytes > c.fileBytes - c.shardFileOffset ||
        c.mappedFileOffset != c.shardFileOffset - c.shardFileOffset % c.allocationGranularity ||
        c.skippedPrefix != c.shardFileOffset - c.mappedFileOffset ||
        c.mappedBytes != c.skippedPrefix + bytes ||
        c.mappedFileOffset > c.fileBytes || c.mappedBytes > c.fileBytes - c.mappedFileOffset ||
        rowByteOffset > bytes || rowBytes > bytes - rowByteOffset)
        throw std::runtime_error("Fixed import upload-source range differs");
}
using Sink = std::function<void(const std::string&, const void*, size_t)>;
using Digest = std::function<std::string(const void*, size_t)>;
using Progress = std::function<void(const Json&)>;
class Observer {
    Sink sink; Digest digest; Progress progress;
    Json state = {{"schema", "chandra.directcompute.import-row-source.v1"}, {"requested", true}, {"complete", false},
        {"target_tensor", "model.language_model.embed_tokens.weight"}, {"shared_head", "lm_head.weight"},
        {"shard_index", 0}, {"row", row}, {"row_bytes", rowBytes}, {"row_byte_offset", rowByteOffset},
        {"upload_api", "Device::upload / immediate-context UpdateSubresource"}, {"create_buffer_initial_data", false},
        {"allocation_selector", "exact"}, {"upload_source_bytes", shardBytes}, {"logical_buffer_bytes", shardBytes},
        {"physical_buffer_bytes", shardBytes}, {"upload_destination_box", nullptr},
        {"upload_call_admissions", 0}, {"upload_return_observed", false}, {"drain_call_admissions", 0}, {"drain_completed", false},
        {"progress_records_written", 0}, {"captures", Json::array()}, {"error", nullptr},
        {"additional_GPU_operations", 0}, {"source_pointer_redirected", false}, {"request_replayed", false},
        {"whole_file_hash_and_tie_equality_passed_before_capture", false},
        {"scope", "CPU memcpy from the original upload pointer; page/cache/timing observer effects possible; no driver/storage/numerical acceptance"}};
    const void* original = nullptr;
    uint32_t phase = 0, records = 0;
    void emit() {
        if (!progress || records >= maximumProgressRecords) throw std::runtime_error("Import source progress bound exceeded");
        Json record = state; record["progress_records_written"] = records + 1;
        if (record.dump().size() + 1 > maximumProgressRecordBytes) throw std::runtime_error("Import source progress record exceeds bound");
        progress(record); ++records; state["progress_records_written"] = records;
    }
    void capture(const char* checkpoint, const char* filename, const void* source, uint32_t bytes) {
        if (!source || source != original || bytes != shardBytes) throw std::runtime_error("Import source pointer/extent changed");
        state["captures"].push_back({{"checkpoint", checkpoint}, {"file", filename}, {"bytes", rowBytes},
            {"copy_completed", false}, {"sha256", nullptr}, {"raw_file_written_and_flushed", false}, {"operation_wall_ms", nullptr}});
        auto& entry = state["captures"].back(); const auto began = std::chrono::steady_clock::now();
        std::array<unsigned char, rowBytes> snapshot{};
        std::memcpy(snapshot.data(), static_cast<const unsigned char*>(source) + rowByteOffset, rowBytes);
        entry["copy_completed"] = true;
        entry["sha256"] = digest(snapshot.data(), snapshot.size());
        sink(filename, snapshot.data(), snapshot.size()); entry["raw_file_written_and_flushed"] = true;
        entry["operation_wall_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
    }
public:
    Observer(Sink s, Digest d, Progress p) : sink(std::move(s)), digest(std::move(d)), progress(std::move(p)) {
        if (!sink || !digest || !progress) throw std::invalid_argument("Import source observer callbacks required");
    }
    void beforeSubmit(const void* source, uint32_t bytes, const Context& c) {
        if (phase || original || !source) throw std::runtime_error("Import source target must be observed once");
        validate(c, bytes); original = source; phase = 1;
        state["whole_file_hash_and_tie_equality_passed_before_capture"] = true;
        state["source_range"] = {{"file_bytes", c.fileBytes}, {"payload_begin", c.payloadBegin}, {"tensor_begin", c.tensorBegin},
            {"shard_file_offset", c.shardFileOffset}, {"row_file_offset", c.shardFileOffset + rowByteOffset},
            {"mapped_file_offset", c.mappedFileOffset}, {"mapped_bytes", c.mappedBytes}, {"skipped_prefix", c.skippedPrefix},
            {"allocation_granularity", c.allocationGranularity}, {"source_pointer_mod_4096", reinterpret_cast<uintptr_t>(source) % 4096}};
        emit(); capture("before_upload", "before_upload.row26213.cpu.bf16", source, bytes); emit();
    }
    void submitting() {
        if (phase != 1 || state["upload_call_admissions"] != 0) throw std::runtime_error("Import source upload sequencing differs");
        // Admission precedes the call; a failed journal write can prevent that call.
        state["upload_call_admissions"] = 1; phase = 2; emit();
    }
    void afterReturn(const void* source, uint32_t bytes) {
        if (phase != 2) throw std::runtime_error("Import source return sequencing differs");
        state["upload_return_observed"] = true; phase = 3;
        capture("after_upload_return", "after_upload_return.row26213.cpu.bf16", source, bytes); emit();
    }
    void draining() {
        if (phase != 3 || state["drain_call_admissions"] != 0) throw std::runtime_error("Import source drain sequencing differs");
        state["drain_call_admissions"] = 1; phase = 4; emit();
    }
    void afterDrain(const void* source, uint32_t bytes) {
        if (phase != 4) throw std::runtime_error("Import source completion sequencing differs");
        state["drain_completed"] = true;
        capture("after_upload_drain", "after_upload_drain.row26213.cpu.bf16", source, bytes);
        // Completion also requires the terminal journal flush; failure leaves phase4.
        state["complete"] = true; emit(); phase = 5;
    }
    void failure(const std::string& reason) noexcept {
        try { state["complete"] = false; state["error"] = reason; emit(); }
        catch (...) { try { state["progress_write_failed"] = true; } catch (...) {} }
    }
    bool completed() const { return phase == 5 && state.at("complete").get<bool>(); }
    Json report() const { return state; }
};
} // namespace chandra::dc::import_row
