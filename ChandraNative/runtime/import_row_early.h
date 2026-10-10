// New code, MPL-2.0. Opt-in early read of the original weight resource, no upload alternative.
#pragma once
#include "api.h"
#include "import_row_source.h"
#include <memory>

namespace chandra::dc::import_row {
constexpr const char* earlyFile = "after_upload_drain.row26213.direct.bf16";
constexpr uint32_t earlyProgressRecords = 6, earlyProgressRecordBytes = 8192;
class EarlyObserver {
    Sink sink; Digest digest; Progress progress;
    std::weak_ptr<Storage> capturedStorage;
    direct_head_row::Receipt receipt;
    uint32_t records = 0; bool started = false, captured = false, finished = false;
    Json state = {{"schema", "chandra.directcompute.import-row-early-copy.v1"}, {"requested", true}, {"complete", false},
        {"capture_complete", false}, {"raw_file_written_and_flushed", false}, {"file", earlyFile}, {"bytes", rowBytes},
        {"artifact_relative_path", "upload-early-audit/after_upload_drain.row26213.direct.bf16"},
        {"journal_relative_path", "upload-early-audit/progress.jsonl"},
        {"target_tensor", "model.language_model.embed_tokens.weight"}, {"shared_head", "lm_head.weight"},
        {"row", row}, {"shard_index", 0}, {"first_word", direct_head_row::firstWord}, {"word_count", direct_head_row::rowWords},
        {"at", "after_existing_upload_drain_and_CPU_source_capture_before_next_import"},
        {"plan", {{"copies", 1}, {"bytes", rowBytes}, {"additional_staging_bytes", rowBytes}, {"drain_timeout_ms", 10000},
            {"post_copy_readiness_budget_ms", 10000}, {"maximum_progress_records", earlyProgressRecords},
            {"maximum_progress_record_bytes", earlyProgressRecordBytes}}},
        {"allocation_selector", "exact"}, {"later_import_started_before_capture", false},
        {"existing_upload_drain_completed", false}, {"existing_CPU_after_drain_capture_completed", false},
        {"direct_read_call_admissions", 0}, {"same_storage_after_import_checked", false}, {"same_storage_after_import", nullptr},
        {"progress_records_written", 0}, {"error", nullptr}, {"request_replayed", false},
        {"scope", "One early underlying-buffer copy; can change order/residency; no production behavior, driver/storage cause, numerical/OCR or retirement acceptance"}};
    void recordTransport() {
        const auto& r = receipt;
        state["transport"] = {{"api", "ID3D11DeviceContext::CopySubresourceRegion"},
            {"source_first_byte", r.extent.firstByte}, {"copy_bytes", r.extent.bytes},
            {"source_logical_bytes", r.extent.logicalBytes}, {"source_physical_bytes", r.extent.physicalBytes},
            {"source_descriptor", {{"usage", r.sourceUsage}, {"bind_flags", r.sourceBindFlags}, {"misc_flags", r.sourceMiscFlags},
                {"cpu_access_flags", r.sourceCPUAccess}, {"structure_byte_stride", r.sourceStride}}},
            {"copies_submitted", r.copiesSubmitted}, {"copy_resubmitted", false},
            {"before_copy_drain_completed", r.beforeCopyDrainCompleted}, {"after_copy_drain_completed", r.afterCopyDrainCompleted},
            {"staging_bytes", r.extent.bytes}, {"staging_created", r.stagingCreated}, {"staging_released", r.stagingReleased},
            {"map_attempted", r.mapAttempted}, {"map_status", r.mapAttempted ? Json(uint32_t(r.readiness.status)) : Json(nullptr)},
            {"map_attempts", r.readiness.attempts}, {"still_drawing_results", r.readiness.stillDrawing},
            {"map_wait_nanoseconds", r.readiness.waited.count()}, {"map_copied", r.mapAttempted && r.readiness.result == readback::Result::copied},
            {"unmapped", r.unmapped}, {"operation_nanoseconds", r.operationNanoseconds},
            {"device_removed_reason_queried", r.deviceRemovedReasonQueried},
            {"device_removed_reason", r.deviceRemovedReasonQueried ? Json(uint32_t(r.deviceRemovedReason)) : Json(nullptr)},
            {"retirement_scope", "Local staging reference release only; root owns worker/Job/channel retirement"}};
    }
    void emit() {
        if (records >= earlyProgressRecords) throw std::runtime_error("Early copy progress record count exceeded");
        Json out = state; out["progress_records_written"] = records + 1;
        if (out.dump().size() + 1 > earlyProgressRecordBytes) throw std::runtime_error("Early copy progress record extent exceeded");
        progress(out); ++records; state["progress_records_written"] = records;
    }
public:
    EarlyObserver(Sink s, Digest d, Progress p) : sink(std::move(s)), digest(std::move(d)), progress(std::move(p)) {
        if (!sink || !digest || !progress) throw std::invalid_argument("Early copy callbacks required");
    }
    void capture(Device& device, const Buffer& buffer, const Observer& source) {
        if (started) throw std::runtime_error("Early import row must be captured once; no replay");
        started = true;
        if (!source.completed()) throw std::runtime_error("Early copy requires completed current CPU upload-source audit");
        direct_head_row::requireKnown(248320,2560,0,buffer.logicalElements,buffer.words,buffer.packedBF16);
        if (!buffer.storage) throw std::runtime_error("Early copy source storage absent");
        const auto cpu = source.report(); const auto& last = cpu.at("captures").back();
        if (cpu.at("allocation_selector") != "exact" || last.at("checkpoint") != "after_upload_drain" ||
            last.at("bytes") != rowBytes || !last.at("raw_file_written_and_flushed").get<bool>())
            throw std::runtime_error("Early copy CPU checkpoint binding differs");
        capturedStorage = buffer.storage;
        state["existing_upload_drain_completed"] = cpu.at("drain_completed");
        state["existing_CPU_after_drain_capture_completed"] = true;
        state["source_range"] = cpu.at("source_range");
        state["CPU_after_drain_sha256"] = last.at("sha256");
        state["CPU_after_drain_file"] = "upload-source-audit/after_upload_drain.row26213.cpu.bf16";
        state["source_buffer"] = {{"words", buffer.words}, {"logical_elements", buffer.logicalElements}, {"packed_BF16", buffer.packedBF16},
            {"identity_scope", "Original shard0 Storage object; later equality checks the stored embedding and tied head shared_ptr"}};
        state["device_identity"] = Json::parse(device.identityJson()); emit();
        state["direct_read_call_admissions"] = 1;
        try {
            const auto gpu = device.readWordsRange(buffer, direct_head_row::firstWord, direct_head_row::rowWords, receipt);
            recordTransport();
            if (gpu.size() != direct_head_row::rowWords || receipt.extent.firstByte != rowByteOffset || receipt.extent.bytes != rowBytes ||
                receipt.extent.logicalBytes != shardBytes || receipt.extent.physicalBytes != shardBytes || receipt.copiesSubmitted != 1 ||
                !receipt.beforeCopyDrainCompleted || !receipt.afterCopyDrainCompleted || !receipt.stagingReleased || !receipt.unmapped ||
                !receipt.mapAttempted || receipt.readiness.result != readback::Result::copied)
                throw std::runtime_error("Early direct row transport incomplete or extent differs");
            state["gpu_sha256"] = digest(gpu.data(), rowBytes);
            sink(earlyFile, gpu.data(), rowBytes); state["raw_file_written_and_flushed"] = true;
            state["sha_matches_CPU_after_drain"] = state.at("gpu_sha256") == state.at("CPU_after_drain_sha256");
            state["capture_complete"] = true; emit(); captured = true;
        } catch (...) { recordTransport(); throw; }
    }
    void afterImport(const Buffer& embedding, const Buffer& head) {
        if (!captured || finished) throw std::runtime_error("Early copy import identity boundary sequencing differs");
        const auto original = capturedStorage.lock();
        state["same_storage_after_import_checked"] = true;
        const bool same = original && original == embedding.storage && original == head.storage;
        state["same_storage_after_import"] = same;
        if (!same) throw std::runtime_error("Early copy resource identity changed during import");
        state["complete"] = true; emit(); finished = true;
    }
    void failure(const std::string& reason) noexcept {
        try { state["complete"] = false; state["error"] = reason; recordTransport(); emit(); }
        catch (...) { try { state["progress_write_failed"] = true; } catch (...) {} }
    }
    bool completed() const { return finished && state.at("complete").get<bool>(); }
    Json report() const { return state; }
};
} // namespace chandra::dc::import_row
