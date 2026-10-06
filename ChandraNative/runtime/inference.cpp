// New code, MPL-2.0. Connected native page path; unqualified until actual graph/OCR checks.
// No tokenizer, processor, engine fallback, request replay, or implicit adapter selection.
// Input authentication and the one-request generation body live in inference_core.cpp, shared
// unchanged with the resident worker; this file keeps CLI options, diagnostics and process lifetime.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define PSAPI_VERSION 2
#include "inference_core.h"
#include "diagnostics.h"
#include "head_audit.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace chandra::dc;
using namespace chandra::dc::inference;
struct Options {
    bool execute = false, forecast = false, diagnosticDump = false, headRowAudit = false; uint32_t diagnosticCap = 0;
    std::filesystem::path model, shaders, manifest, output, diagnosticPlan; std::string pci, luid, manifestSha, diagnosticPlanSha;
};
Options options(int argc, wchar_t** argv) {
    Options o; std::set<std::wstring> seen;
    for (int i = 1; i < argc; ++i) {
        std::wstring name = argv[i]; require(seen.insert(name).second, "Duplicate CLI option");
        if (name == L"--execute") { o.execute = true; continue; }
        if (name == L"--forecast") { o.forecast = true; continue; }
        if (name == L"--diagnostic-dump") { o.diagnosticDump = true; continue; }
        if (name == L"--head-row-audit") { o.headRowAudit = true; continue; }
        require(i + 1 < argc, "CLI option value is missing"); std::wstring value = argv[++i];
        if (name == L"--model-dir") o.model = value;
        else if (name == L"--shader-root") o.shaders = value;
        else if (name == L"--input-manifest") o.manifest = value;
        else if (name == L"--input-sha256") o.manifestSha = digestText(utf8(value));
        else if (name == L"--output-dir") o.output = value;
        else if (name == L"--pci") o.pci = utf8(value);
        else if (name == L"--luid") o.luid = utf8(value);
        else if (name == L"--diagnostic-plan") o.diagnosticPlan = value;
        else if (name == L"--diagnostic-plan-sha256") o.diagnosticPlanSha = digestText(utf8(value));
        else if (name == L"--diagnostic-token-cap") {
            auto s = utf8(value); require(!s.empty() && s.size() <= 5 && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }), "Explicit positive diagnostic token cap required");
            auto n = std::stoul(s); require(n && n < normalOutputLimit, "Diagnostic cap must be shorter than the normal allowance"); o.diagnosticCap = uint32_t(n);
        } else throw std::runtime_error("Unknown CLI option: " + utf8(name));
    }
    require(o.execute != o.forecast, "Inactive by default: choose --execute or --forecast explicitly");
    for (const auto* p : {&o.model, &o.shaders, &o.manifest, &o.output}) require(!p->empty() && p->is_absolute(), "Explicit absolute model/shader/input/output paths required");
    require(!o.pci.empty() && !o.luid.empty() && !o.manifestSha.empty(), "Explicit PCI, LUID and input-manifest SHA-256 required");
    require(o.diagnosticPlan.empty() == o.diagnosticPlanSha.empty() && (o.diagnosticPlan.empty() || (o.diagnosticDump && o.diagnosticPlan.is_absolute())), "A diagnostic plan requires --diagnostic-dump, an absolute path and its SHA-256");
    // Bounded native-1 locator: the prefill prediction and at most one cached decode prediction.
    require(!o.headRowAudit || (o.execute && o.diagnosticCap >= 1 && o.diagnosticCap <= head_audit::maximumPredictions), "--head-row-audit requires --execute and --diagnostic-token-cap 1 or 2");
    require(std::filesystem::is_directory(o.model) && std::filesystem::is_directory(o.shaders), "Explicit model and shader directories must exist");
    require(!std::filesystem::exists(o.output), "Fresh output directory required; existing output is never overwritten"); return o;
}
// Optional diagnostics are resolved on the CPU after input authentication and before any output,
// device or model work, so an invalid or over-budget selection is refused before hardware execution.
struct DiagnosticRequest { diagnostics::Geometry geometry; diagnostics::Plan plan; Json planFile; };
std::unique_ptr<DiagnosticRequest> diagnosticRequest(const Input& in, const Options& o) {
    if (!o.diagnosticDump) return nullptr;
    auto d = std::make_unique<DiagnosticRequest>(); auto& g = d->geometry;
    g.patchRows = in.patchRows; g.mergedRows = in.visionForecast.mergedRows; g.generationLimit = o.diagnosticCap ? o.diagnosticCap : normalOutputLimit;
    g.grids = in.grids; g.ids = in.ids; g.visionRowMap = in.visionRowMap; g.positions = in.positions;
    Json requested; d->planFile = nullptr;
    if (!o.diagnosticPlan.empty()) {
        auto size = std::filesystem::file_size(o.diagnosticPlan); require(size && size <= manifestByteLimit, "Bounded diagnostic plan required");
        std::vector<char> bytes(static_cast<size_t>(size)); authenticatedRead(o.diagnosticPlan, bytes.data(), size, o.diagnosticPlanSha);
        requested = strictJson(bytes); d->planFile = {{"sha256", o.diagnosticPlanSha}, {"bytes", size}};
    }
    d->plan = diagnostics::resolve(o.diagnosticPlan.empty() ? nullptr : &requested, g); return d;
}
std::string fileSha256(const std::filesystem::path& path, uint64_t limit) {
    DWORD attributes = GetFileAttributesW(path.c_str());
    require(attributes != INVALID_FILE_ATTRIBUTES && !(attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)), "Regular non-reparse source file required");
    Handle file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
    LARGE_INTEGER length{}; require(GetFileSizeEx(file.value, &length) && length.QuadPart >= 0 && uint64_t(length.QuadPart) <= limit, "Bounded source file required");
    Sha256 sha; std::vector<unsigned char> block(1024 * 1024);
    for (uint64_t done = 0; done < uint64_t(length.QuadPart);) {
        DWORD n = DWORD(std::min<uint64_t>(uint64_t(length.QuadPart) - done, block.size())), got = 0;
        require(ReadFile(file.value, block.data(), n, &got, nullptr) && got == n, "Incomplete source read"); sha.update(block.data(), got); done += got;
    }
    return sha.finish();
}
// The executable image and the HLSL tree compiled at first dispatch identify this producer; neither is a source revision claim.
Json diagnosticProducer(const Options& o) {
    std::wstring executable(32768, L'\0'); DWORD n = GetModuleFileNameW(nullptr, executable.data(), DWORD(executable.size()));
    require(n && n < executable.size(), "Executable path unavailable"); executable.resize(n);
    std::vector<std::pair<std::string, std::string>> files; std::error_code error;
    std::filesystem::recursive_directory_iterator it(o.shaders, error), end; require(!error, "Shader tree enumeration failed");
    for (; it != end; it.increment(error)) {
        require(!error, "Shader tree enumeration failed"); DWORD attributes = GetFileAttributesW(it->path().c_str());
        require(attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT), "Shader tree reparse point refused");
        if (attributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        require(files.size() < 1024, "Shader tree exceeds 1024 files");
        files.emplace_back(std::filesystem::relative(it->path(), o.shaders).generic_u8string(), fileSha256(it->path(), 16ull * 1024 * 1024));
    }
    require(!error, "Shader tree enumeration failed"); std::sort(files.begin(), files.end());
    std::string canonical; for (const auto& [path, sha] : files) canonical += path + "\t" + sha + "\n";
    Sha256 tree; if (!canonical.empty()) tree.update(canonical.data(), uint32_t(canonical.size()));
    return {{"kind", "directcompute_native"}, {"executable_sha256", fileSha256(executable, 256ull * 1024 * 1024)},
        {"shader_root", {{"files", files.size()}, {"tree_sha256", tree.finish()}, {"canonical", "sorted generic relative path, tab, file SHA-256, newline"}}},
        {"scope", "Executable image and HLSL files present at diagnostic start; shaders compile lazily from this root"}};
}
Json diagnosticCommitments(const Input& in, const Options& o, const DiagnosticRequest& d) {
    return {{"model_revision", modelRevision}, {"input_manifest_sha256", o.manifestSha}, {"plan_file", d.planFile},
        {"input", {{"prompt_tokens", in.ids.size()}, {"patch_rows", in.patchRows}, {"merged_rows", in.visionForecast.mergedRows},
            {"image_grid_thw", in.manifest.at("image_grid_thw")}, {"next_decode_position", in.maximumPosition + 1}, {"stop_token_ids", in.stopIds},
            {"generation_limit", d.geometry.generationLimit}, {"diagnostic_token_cap", o.diagnosticCap ? Json(o.diagnosticCap) : Json(nullptr)}}}};
}
class DiagnosticFile final : public diagnostics::File {
    Handle file;
public:
    explicit DiagnosticFile(const std::filesystem::path& path) : file(CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr)) {}
    void write(const void* data, size_t bytes) override {
        for (auto* p = static_cast<const char*>(data); bytes;) {
            DWORD n = DWORD(std::min<size_t>(bytes, 16u * 1024 * 1024)), wrote = 0;
            require(WriteFile(file.value, p, n, &wrote, nullptr) && wrote == n, "Writing diagnostic output failed"); p += n; bytes -= n;
        }
    }
    void flush() override { require(FlushFileBuffers(file.value), "Flushing diagnostic output failed"); }
};
class DiagnosticDirectory final : public diagnostics::Directory {
    std::filesystem::path root;
public:
    explicit DiagnosticDirectory(std::filesystem::path path) : root(std::move(path)) {
        require(CreateDirectoryW(root.c_str(), nullptr), "Fresh diagnostic directory creation failed; existing output is never reused");
        DWORD attributes = GetFileAttributesW(root.c_str());
        require(attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT), "Diagnostic directory must be a plain directory");
    }
    std::unique_ptr<diagnostics::File> createNew(const std::string& name) override {
        require(diagnostics::safeName(name), "Unsafe diagnostic filename refused"); return std::make_unique<DiagnosticFile>(root / std::filesystem::u8path(name));
    }
};
// --head-row-audit (head_audit.h): authenticated rows come from the held model handle at payload offsets
// taken from the importer's own inventory; raw GPU row dumps go to a fresh directory under output.
std::unique_ptr<head_audit::Auditor> headAudit(Device& device, const ModelWeights& weights, const Json& provenance, const Handle& model, const std::filesystem::path& directory, uint32_t requestedPredictions) {
    LARGE_INTEGER size{}; require(GetFileSizeEx(model.value, &size) && uint64_t(size.QuadPart) == provenance.at("model_bytes").get<uint64_t>(), "Head audit model file length differs from the importer");
    auto readAt = [file = model.value](uint64_t offset, void* destination, uint32_t bytes) {
        OVERLAPPED at{}; at.Offset = DWORD(offset); at.OffsetHigh = DWORD(offset >> 32); DWORD got = 0;
        require(bytes <= head_audit::sourceChunkBytes && ReadFile(file, destination, bytes, &got, &at) && got == bytes, "Head audit source read incomplete");
    };
    uint8_t prefix[8]; readAt(0, prefix, 8); uint64_t header = 0; for (unsigned i = 0; i < 8; ++i) header |= uint64_t(prefix[i]) << (8 * i);
    require(header > 1 && header <= 1024 * 1024, "Head audit safetensors header length outside bound");
    const uint64_t payload = 8 + header; uint64_t headBegin = UINT64_MAX, normBegin = UINT64_MAX;
    for (const auto& row : provenance.at("inventory")) {
        if (row.at("name") == "model.language_model.embed_tokens.weight") headBegin = row.at("data_offsets").at(0).get<uint64_t>();
        if (row.at("name") == "model.language_model.norm.weight") normBegin = row.at("data_offsets").at(0).get<uint64_t>();
    }
    require(headBegin != UINT64_MAX && normBegin != UINT64_MAX, "Head audit needs the importer's embedding and final-norm offsets");
    require(CreateDirectoryW(directory.c_str(), nullptr), "Fresh head audit directory creation failed; existing output is never reused");
    return std::make_unique<head_audit::Auditor>(device, weights.at("model.language_model.embed_tokens.weight"), head_audit::Tensor{headBegin}, head_audit::Tensor{normBegin}, textWidth, requestedPredictions,
        [readAt, payload](uint64_t offset, void* destination, uint32_t bytes) { readAt(payload + offset, destination, bytes); },
        [directory](const std::string& name, const void* data, size_t bytes) {
            require(diagnostics::safeName(name), "Unsafe head audit filename refused");
            DiagnosticFile file(directory / std::filesystem::u8path(name)); file.write(data, bytes); file.flush();
        },
        [](const void* data, size_t bytes) { return diagnostics::sha256(data, bytes); },
        [observed = &device] { return Json::parse(observed->memoryJson()); });
}
Json forecast(const Input& in, const Options& o, const DiagnosticRequest* diagnostic) {
    Json result = forecastReport(in, o.manifestSha, o.diagnosticCap);
    if (diagnostic) result["diagnostics"] = {{"requested", true}, {"dump_written", false}, {"plan", diagnostic->plan.resolved}, {"plan_sha256", diagnostic->plan.sha256}, {"plan_file", diagnostic->planFile}};
    return result;
}
Json execute(const Input& in, const Options& o, const DiagnosticRequest* diagnostic) {
    auto start = Clock::now(); Json phases = Json::array(), memories = Json::array(); std::vector<uint32_t> generated;
    Json report = executionReport(in, o.manifestSha, o.diagnosticCap);
    if (diagnostic) report["diagnostic_dump_requested"] = true;
    // A requested head audit reads audit_incomplete until its own report replaces this entry, so a run
    // failing before or during the audit never reads as clean. Every requested prediction must be audited.
    if (o.headRowAudit) report["head_row_audit"] = head_audit::notStarted(o.diagnosticCap);
    std::unique_ptr<Device> device; std::unique_ptr<diagnostics::Recorder> recorder; auto stamp = Clock::now();
    auto phase = [&](const char* name) { auto now = Clock::now(); phases.push_back({{"name", name}, {"wall_seconds", std::chrono::duration<double>(now - stamp).count()}}); stamp = now; if (device) memories.push_back({{"phase", name}, {"observed", Json::parse(device->memoryJson())}, {"host_process", hostMemory()}}); if (recorder) recorder->boundary(name); };
    try {
        // Readbacks happen synchronously in observers on this Device thread, after the graph's own drains.
        if (diagnostic) recorder = std::make_unique<diagnostics::Recorder>(diagnostic->plan, diagnostic->geometry, std::make_unique<DiagnosticDirectory>(o.output / L"diagnostics"),
            [&device](const Buffer& b) { return device->readFloats(b); }, diagnosticCommitments(in, o, *diagnostic), diagnosticProducer(o));
        device = std::make_unique<Device>(o.shaders.wstring(), o.pci, o.luid); phase("device_creation"); report["device_identity"] = Json::parse(device->identityJson());
        {
            // Empty selection is the complete normal inference graph. No
            // diagnostic subset or CPU-resident tensor copy is substituted.
            // --head-row-audit holds this read-only, non-write-shared handle across the importer's whole-file
            // SHA-256 and every later audit read, so no writer can change the authenticated bytes in between.
            std::unique_ptr<Handle> auditSource;
            if (o.headRowAudit) auditSource = std::make_unique<Handle>(CreateFileW((o.model / L"model.safetensors").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
            ModelWeights weights(*device, o.model.wstring()); phase("authenticated_model_import_upload"); report["model_provenance"] = Json::parse(weights.provenanceJson());
            require(report["model_provenance"].at("full_graph_requested") == true && report["model_provenance"].at("omitted_graph_tensors").empty() && report["model_provenance"].at("tie_byte_equality") == true, "Full original graph and authenticated tied output head required");
            if (recorder) recorder->authenticatedModel(report["model_provenance"], report["device_identity"]);
            std::unique_ptr<head_audit::Auditor> audit;
            if (auditSource) audit = headAudit(*device, weights, report["model_provenance"], *auditSource, o.output / L"head-row-audit", o.diagnosticCap);
            if (audit) {
                try { audit->afterImport(); }
                catch (const std::exception& e) { report["head_row_audit"] = head_audit::finishAfterFailure(*audit, e.what()); throw; }
                catch (...) { report["head_row_audit"] = head_audit::finishAfterFailure(*audit, "non-standard exception"); throw; }
                phase("head_row_audit_after_import");
            }
            RequestHooks hooks; hooks.phase = phase; // Observers stay empty unless a dump or head audit was explicitly requested.
            if (recorder) {
                hooks.vision = [&](const std::string& name, const Buffer& b, uint32_t rows, uint32_t width) { recorder->vision(name, b, rows, width); };
                hooks.merged = [&](const Buffer& b, uint32_t rows) { recorder->merged(b, rows, textWidth); };
            }
            if (recorder || audit) hooks.text = [&](const TextObservation& observation) { if (recorder) recorder->text(observation); if (audit) audit->observe(observation); };
            try { generate(*device, weights, in, o.diagnosticCap, o.output, report, generated, hooks); }
            catch (const std::exception& e) { if (audit) report["head_row_audit"] = head_audit::finishAfterFailure(*audit, e.what()); throw; }
            catch (...) { if (audit) report["head_row_audit"] = head_audit::finishAfterFailure(*audit, "non-standard exception"); throw; }
            if (audit) report["head_row_audit"] = audit->finish(head_audit::RunEnd::completed);
        }
        device->drain(); phase("model_buffer_release_and_drain");
        require(device->trackedBufferBytes() == 0, "Owned graph buffers did not retire to zero"); report["owned_buffer_zero"] = true;
        report["passed"] = true; report["passed_scope"] = "Executable graph returned finite greedy tokens and retired owned buffers; numerical/full-OCR qualification is pending";
    } catch (const std::exception& e) {
        report["passed"] = false; report["error"] = e.what(); report["owned_worker_retirement_required"] = true;
        if (device) { try { device->drain(); report["final_memory"] = Json::parse(device->memoryJson()); report["final_drain_completed"] = true; } catch (const std::exception& drain) { report["final_drain_completed"] = false; report["final_drain_error"] = drain.what(); } }
    }
    if (recorder) {
        // The terminal line is appended after success, failure or retirement; dumps stay unqualified either way.
        try { report["diagnostics"] = recorder->finish(report.at("passed").get<bool>(), report.contains("error") ? report.at("error").get<std::string>() : std::string()); }
        catch (const std::exception& e) {
            report["diagnostics"] = {{"finish_error", e.what()}, {"dump_complete", false}, {"qualified", false}};
            if (report.at("passed").get<bool>()) { report["passed"] = false; report["error"] = std::string("Diagnostic dump finalization failed: ") + e.what(); }
        }
    }
    finishExecutionReport(report, in, generated, phases, memories, start);
    return report;
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    Options o; bool created = false; auto started = Clock::now();
    try {
        o = options(argc, argv); auto in = authenticateInput(o.manifest, o.manifestSha); // No Device before all input/hash/geometry/finite guards.
        double authenticationSeconds = std::chrono::duration<double>(Clock::now() - started).count();
        auto diagnostic = diagnosticRequest(in, o); // Invalid or over-budget selections refuse before any output or Device.
        require(std::filesystem::create_directory(o.output), "Fresh output directory creation failed"); created = true;
        auto report = o.forecast ? forecast(in, o, diagnostic.get()) : execute(in, o, diagnostic.get());
        report["input_authentication_wall_seconds"] = authenticationSeconds; report["final_host_process_memory"] = hostMemory();
        report["total_process_path_wall_seconds"] = std::chrono::duration<double>(Clock::now() - started).count();
        NewOutput receipt(o.output / L"result.json"); receipt.line(report); std::cout << report.dump() << '\n';
        return o.forecast ? 0 : (report.at("passed").get<bool>() ? 0 : 1);
    } catch (const std::exception& e) {
        Json error = {{"schema", "chandra.directcompute.result.v1"}, {"passed", false}, {"error", e.what()}, {"request_replayed", false}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
        if (created) { try { NewOutput receipt(o.output / L"failure.json"); receipt.line(error); } catch (...) {} }
        std::cerr << error.dump() << '\n'; return 1;
    }
}
