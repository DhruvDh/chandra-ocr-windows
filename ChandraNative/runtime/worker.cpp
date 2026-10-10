// New code, MPL-2.0. Resident native Chandra worker: one Device and one authenticated complete
// ModelWeights per process, B1 requests executed serially on the Device-owning thread through the
// same request path as chandra-inference.exe (inference_core.cpp), over a versioned, bounded
// newline-JSON stdin/stdout protocol. Inactive by default. The separately explicit static B2 mode
// batches decode only; one Device thread owns all dispatches. No model reload, engine/CPU fallback or request replay. A bounded stdin reader thread performs
// admission, cancellation notification and status only; it never touches the Device or GPU buffers.
// Input ends at EOF, at a read failure (reported, exit 3) or at the protocol-error ceiling (exit 3 whichever
// cause closed the worker first); at exit the worker cancels a still-pending read and joins the reader before
// its final event.
// Source cannot preempt an in-flight dispatch: the external kill-on-close Job and parent lease remain
// the owner's obligation. See docs/directcompute-resident-worker.md.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define PSAPI_VERSION 2
#include "inference_core.h"
#include "padded32_opt_in.h"
#include "final_import_arithmetic_gate.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace chandra::dc::experimental { const char* gemvB1Selection(); } // operators.cpp; not part of api.h.

namespace {
using namespace chandra::dc;
using namespace chandra::dc::inference;
constexpr const char* eventSchema = "chandra.directcompute.worker-event.v1";
constexpr const char* requestSchema = "chandra.directcompute.worker-request.v1";
constexpr const char* resultSchema = "chandra.directcompute.worker-result.v1";
constexpr size_t maxLineBytes = 4096, maxEventBytes = 65536, maxDetailBytes = 512, maxIdBytes = 64, maxOutputNameBytes = 64, maxComponentBytes = 128;
constexpr uint32_t maxProtocolErrors = 16, minLeaseMilliseconds = 1000, maxLeaseMilliseconds = 600000, readerStopAttempts = 50;
constexpr uint64_t maxLifetimeRequests = 65536;
constexpr std::chrono::milliseconds readerStopWait(20);
const std::set<uint32_t> callerStops = {248044, 248046};
constexpr const char* obligations = "Source guards bound input, admission, buffers and drains but cannot preempt an in-flight dispatch or prove arbitrary workloads safe. "
    "The owner must supply a finite kill-on-close Windows Job, a live parent holding the stdin lease, private input/output roots, adapter admission and closure evidence.";

struct Options {
    bool execute = false, plan = false, experimentalPadded32 = false, experimentalB2 = false; uint32_t leaseMilliseconds = 0;
    std::filesystem::path model, shaders, inputRoot, outputRoot; std::string pci, luid;
};
Options options(int argc, wchar_t** argv) {
    Options o; std::set<std::wstring> seen;
    for (int i = 1; i < argc; ++i) {
        std::wstring name = argv[i]; require(seen.insert(name).second, "Duplicate worker option");
        if (name == L"--execute") { o.execute = true; continue; }
        if (name == L"--plan") { o.plan = true; continue; }
        if (name == L"--experimental-padded32") { o.experimentalPadded32 = true; continue; }
        if (name == L"--experimental-b2-decode") { o.experimentalB2 = true; continue; }
        require(i + 1 < argc, "Worker option value is missing"); std::wstring value = argv[++i];
        if (name == L"--model-dir") o.model = value;
        else if (name == L"--shader-root") o.shaders = value;
        else if (name == L"--input-root") o.inputRoot = value;
        else if (name == L"--output-root") o.outputRoot = value;
        else if (name == L"--pci") o.pci = utf8(value);
        else if (name == L"--luid") o.luid = utf8(value);
        else if (name == L"--lease-ms") {
            auto s = utf8(value); require(!s.empty() && s.size() <= 6 && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }), "Explicit lease milliseconds required");
            auto n = std::stoul(s); require(n >= minLeaseMilliseconds && n <= maxLeaseMilliseconds, "Lease must be between 1000 and 600000 ms"); o.leaseMilliseconds = uint32_t(n);
        } else throw std::runtime_error("Unknown worker option: " + utf8(name));
    }
    require(!(o.execute && o.plan), "Choose exactly one of --execute or --plan");
    require(o.experimentalPadded32 == o.experimentalB2, "Dedicated cohort mode requires both explicit padded32 and B2 flags");
    return o;
}
// Every component of an absolute root, from the volume root down, must exist without reparse/symlink aliasing.
void plainDirectory(const std::filesystem::path& path, const char* what) {
    require(!path.empty() && path.is_absolute(), std::string(what) + " must be an explicit absolute path");
    auto prefix = path.root_path();
    for (const auto& component : path.relative_path()) {
        require(!component.empty() && component != L"." && component != L"..", std::string(what) + " must be normalized without trailing separators");
        prefix /= component; DWORD attributes = GetFileAttributesW(prefix.c_str());
        require(attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT), std::string(what) + " has a missing or reparse/symlink component");
    }
    DWORD attributes = GetFileAttributesW(path.c_str());
    require(attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT), std::string(what) + " must be an existing plain directory");
}
bool nested(const std::filesystem::path& a, const std::filesystem::path& b) {
    auto i = a.begin(), j = b.begin();
    for (; i != a.end() && j != b.end(); ++i, ++j) if (*i != *j) return false;
    return true; // One is a prefix of the other, or they are equal.
}
bool reservedDevice(std::string name) {
    name = name.substr(0, name.find('.'));
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    if (name == "CON" || name == "PRN" || name == "AUX" || name == "NUL" || name == "CONIN$" || name == "CONOUT$") return true;
    return name.size() == 4 && (name.rfind("COM", 0) == 0 || name.rfind("LPT", 0) == 0) && name[3] >= '0' && name[3] <= '9';
}
bool safeComponent(const std::string& s, size_t limit) {
    return !s.empty() && s.size() <= limit && s != "." && s != ".." && s.back() != '.' && s.front() != '.' && !reservedDevice(s) &&
        std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '_' || c == '-'; });
}
bool safeId(const std::string& s) {
    return !s.empty() && s.size() <= maxIdBytes && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '_' || c == '-' || c == ':'; });
}
// Untrusted bytes (parser messages, paths, exception text) become valid UTF-8 before any JSON dump:
// an invalid or truncated sequence is replaced by '?', so dump() cannot throw on event text.
std::string utf8Safe(const std::string& text) {
    std::string out; out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        bool valid = n != 0 && i + n <= text.size() && (n != 2 || c >= 0xc2);
        for (size_t k = 1; valid && k < n; ++k) valid = (static_cast<unsigned char>(text[i + k]) & 0xc0) == 0x80;
        if (valid && n == 3) { const unsigned v = ((c & 15u) << 12) | ((static_cast<unsigned char>(text[i + 1]) & 63u) << 6); valid = v >= 0x800 && (v < 0xd800 || v > 0xdfff); }
        if (valid && n == 4) { const unsigned v = ((c & 7u) << 18) | ((static_cast<unsigned char>(text[i + 1]) & 63u) << 12); valid = v >= 0x10000 && v <= 0x10ffff; }
        if (valid) { out.append(text, i, n); i += n; } else { out.push_back('?'); ++i; }
    }
    return out;
}
std::string bounded(std::string text) { if (text.size() > maxDetailBytes) { text.resize(maxDetailBytes - 3); text += "..."; } return utf8Safe(text); }
// Names of statuses a standard-input read is known to fail with; any other code is reported by number only.
Json readStatusName(DWORD error) {
    switch (error) {
    case ERROR_INVALID_FUNCTION: return "ERROR_INVALID_FUNCTION";
    case ERROR_ACCESS_DENIED: return "ERROR_ACCESS_DENIED";
    case ERROR_INVALID_HANDLE: return "ERROR_INVALID_HANDLE";
    case ERROR_NO_DATA: return "ERROR_NO_DATA";
    case ERROR_PIPE_NOT_CONNECTED: return "ERROR_PIPE_NOT_CONNECTED";
    case ERROR_OPERATION_ABORTED: return "ERROR_OPERATION_ABORTED";
    default: return nullptr;
    }
}
Json observeJob() {
    BOOL inJob = FALSE;
    if (!IsProcessInJob(GetCurrentProcess(), nullptr, &inJob)) return {{"observed", false}, {"win32_error", GetLastError()}};
    Json job = {{"observed", true}, {"in_job", inJob != FALSE}};
    if (!inJob) return job;
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{}; DWORD returned = 0;
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &info, DWORD(sizeof(info)), &returned)) {
        job["limits_observed"] = false; job["win32_error"] = GetLastError(); return job;
    }
    const DWORD flags = info.BasicLimitInformation.LimitFlags;
    job["limits_observed"] = true; job["limit_flags"] = flags; job["kill_on_job_close"] = (flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) != 0;
    job["process_memory_limit_bytes"] = (flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) ? Json(uint64_t(info.ProcessMemoryLimit)) : Json(nullptr);
    job["job_memory_limit_bytes"] = (flags & JOB_OBJECT_LIMIT_JOB_MEMORY) ? Json(uint64_t(info.JobMemoryLimit)) : Json(nullptr);
    job["active_process_limit"] = (flags & JOB_OBJECT_LIMIT_ACTIVE_PROCESS) ? Json(info.BasicLimitInformation.ActiveProcessLimit) : Json(nullptr);
    job["breakaway_allowed"] = (flags & (JOB_OBJECT_LIMIT_BREAKAWAY_OK | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)) != 0;
    job["scope"] = "Immediate Job of this process as observed at startup; enforcement and closure remain owner obligations";
    return job;
}
bool jobAdmitsExecution(const Json& job) {
    return job.value("in_job", false) && job.value("limits_observed", false) && job.value("kill_on_job_close", false) && !job.value("breakaway_allowed", true) &&
        (!job.at("process_memory_limit_bytes").is_null() || !job.at("job_memory_limit_bytes").is_null());
}
std::string sha256Text(const std::string& text) { Sha256 sha; if (!text.empty()) sha.update(text.data(), uint32_t(text.size())); return sha.finish(); }
// Writes one exclusive receipt and returns the SHA-256 of the exact bytes written.
std::string writeReceipt(const std::filesystem::path& directory, const Json& value) {
    NewOutput receipt(directory / L"result.json"); receipt.line(value); return sha256Text(value.dump() + "\n");
}


// One fresh, private directory per worker lifetime. The leading dot is already refused by
// safeComponent(), so no submit message can use this directory as request output.
std::filesystem::path residentImportDirectory(const std::filesystem::path& root) {
    const auto name = L".resident-import-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(Clock::now().time_since_epoch().count());
    const auto directory = root / name;
    require(CreateDirectoryW(directory.c_str(), nullptr), "Fresh private resident-import directory required");
    const DWORD a = GetFileAttributesW(directory.c_str());
    require(a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) && !(a & FILE_ATTRIBUTE_REPARSE_POINT), "Resident-import directory must be plain");
    return directory;
}
Json persistResidentMetadata(const std::filesystem::path& path, const Json& value) {
    const auto bytes = value.dump() + "\n";
    require(bytes.size() <= manifestByteLimit, "Resident import metadata exceeds unchanged 1 MiB bound");
    { NewOutput file(path); file.line(value); }
    return {{"file", utf8(path.filename().wstring())}, {"bytes", bytes.size()}, {"sha256", sha256Text(bytes)}};
}
std::unique_ptr<import_row::FinalTailObserver> residentFinalTailObserver(const std::filesystem::path& directory,
        std::unique_ptr<NewOutput>& journal, uint32_t& records, uint64_t& bytes) {
    journal = std::make_unique<NewOutput>(directory / L"progress.jsonl");
    return std::make_unique<import_row::FinalTailObserver>(WeightImportAPI::defaultInitial,
        [directory](const std::string& name, const void* data, size_t count) {
            require(((name == import_row::finalTailSourceFile || name == import_row::finalTailGoodFile) && count > 0 && count <= import_row::final_tails::rowArrayBytes && count % import_row::tails::rowBytes == 0) ||
                (name == import_row::finalTailBadFile && count == import_row::tails::rowBytes), "Fixed resident final-tail payload name/extent required");
            Handle file(CreateFileW((directory / std::filesystem::u8path(name)).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr));
            DWORD wrote = 0;
            require(WriteFile(file.value, data, DWORD(count), &wrote, nullptr) && wrote == count, "Writing resident final-tail payload failed");
            require(FlushFileBuffers(file.value), "Flushing resident final-tail payload failed");
        }, [](const void* data, size_t count) { Sha256 hash; hash.update(data, uint32_t(count)); return hash.finish(); },
        [&journal, &records, &bytes](const Json& record) {
            const auto count = record.dump().size() + 1;
            require(count <= 4096 && records < 32 && bytes + count <= 128 * 1024, "Resident final-tail journal exceeds unchanged full-page tail profile");
            journal->line(record); ++records; bytes += count;
        }, true);
}
void requireResidentImportProvenance(const Json& provenance) {
    auto number = [](const Json& v, uint64_t n) { return v.is_number_unsigned() && v.get<uint64_t>() == n; };
    require(provenance.at("schema") == "chandra.directcompute.model-weights.v2" && provenance.at("model") == "datalab-to/chandra-ocr-2" && provenance.at("revision") == modelRevision &&
        provenance.at("model_sha256") == "0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847" && number(provenance.at("model_bytes"), 10591220088ull) &&
        provenance.at("config_sha256") == "e26f17b70463de21fd68f1ef6d8f67f8e33e65d55a7a6895242130a18db60587" && number(provenance.at("config_bytes"), 2773) &&
        provenance.at("full_graph_requested") == true && provenance.at("omitted_graph_tensors").empty() && provenance.at("tie_byte_equality_performed") == true && provenance.at("tie_byte_equality") == true && provenance.at("implicit_cast_or_dequantization") == false &&
        provenance.at("weight_import_api") == "default-initial" && provenance.at("weight_srv_only") == import_row::FinalTailObserver::forecast(WeightImportAPI::defaultInitial, true).at("weight_srv_only") &&
        number(provenance.at("API_operations_completed"), 732) && number(provenance.at("UpdateSubresource_calls"), 0) && number(provenance.at("DEFAULT_initial_data_Device_creations"), 732) &&
        number(provenance.at("uploaded_storage_bytes"), 9078531072ull), "Resident import provenance differs from the fixed original DEFAULT-initial SRV-only graph");
}

bool cohortManifestAllowed(const std::string& sha) {
    return sha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae" ||
        sha=="aab97d615f7e3181b9a3ef814f8784030f17fb186a31964df0a2a6063512e8d1" ||
        sha=="62c362251b128392f3cb3fcff6e8e25531b585038d12391ecaa1a153d6979081" ||
        sha=="df67a54b1d43b83f178b8b7c6627480d114bf07e68ddee2ba08b137f34307c04";
}
void requireCohortInput(const Input& in,const std::string& manifestSha) {
    if(cohortManifestAllowed(manifestSha)) {
        require(in.ids.size()==3617 && in.manifest.at("image").at("dimensions")==Json({1624,2100}),"Exact full-size diagnostic corpus page and 3617-token prompt required");
        return;
    }
    // Only after original authenticateInput has checked the actual manifest and all raw tensor/caller bytes.
    require(in.manifest.at("source")=="chandra.native-endpoint.request.v1" &&
        in.manifest.at("processor_profile")=="northstone-serving" &&
        in.manifest.at("processor_kwargs")==Json({{"size",{{"shortest_edge",3136},{"longest_edge",3145728}}}}) &&
        in.manifest.at("processor_files")==Json::parse("{\"chat_template.jinja\":\"0d158f349ca965f7eea9db0eb45cd177b85bb0e4ae05dcdd0f060da8f7d41812\",\"config.json\":\"e26f17b70463de21fd68f1ef6d8f67f8e33e65d55a7a6895242130a18db60587\",\"generation_config.json\":\"0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e\",\"preprocessor_config.json\":\"957eb01d1ea45341a92d543daec95857a7cbeff5803834bc0603b27ba7b41b3f\",\"processor_config.json\":\"14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b\",\"tokenizer.json\":\"87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4\",\"tokenizer_config.json\":\"316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8\",\"video_preprocessor_config.json\":\"de7ba2c4528aa3c92754dc61ae83f1871369cbf7ab4298dcaa99c2f5a7c80848\"}"),
        "Exact production endpoint source and pinned processor/profile required");
    require(in.patchRows<=12288 && in.grids.size()==1 && in.grids[0].temporal==1,
        "Production cohort retains the existing serving patch ceiling and one still image");
    for(const char* name:{"attention_mask","mm_token_type_ids","text_position_ids","rope_deltas"})
        require(in.descriptors.count(name)!=0,"Complete production package tensor inventory required");
}
Json cohortSelection(const Options& o,const std::string& b1) {
    require(o.experimentalB2 && o.experimentalPadded32 && b1=="ordered", "Cohort mode requires both explicit flags and captured ordered B1");
    for(const auto& entry:std::array<std::pair<const char*,std::pair<uint32_t,const char*>>,2>{{
        {chandra::padded32_opt_in::shader,{3356,chandra::padded32_opt_in::shaderSha256}},
        {chandra::padded32_opt_in::fallbackShader,{2444,chandra::padded32_opt_in::fallbackSha256}}}}) {
        const auto path=localFile(o.shaders,entry.first);std::vector<char> bytes(entry.second.first);
        authenticatedRead(path,bytes.data(),bytes.size(),entry.second.second);
    }
    return {{"schema","chandra.directcompute.worker-ordered-padded32-b2-selection.v1"},{"explicit_B2",true},{"max_decode_slots",2},
        {"B1_selector","ordered"},{"multirow_shader",chandra::padded32_opt_in::shader},{"multirow_shader_sha256",chandra::padded32_opt_in::shaderSha256},
        {"fallback_shader",chandra::padded32_opt_in::fallbackShader},{"fallback_shader_sha256",chandra::padded32_opt_in::fallbackSha256},
        {"shader_bytes_authenticated_before_Device",true},{"unsupported_multirow_shapes_use_original_fallback",true},
        {"parallel32_excluded",true},{"trained_numerical_OCR_PPS_acceptance",false}};
}
struct Canceled : std::runtime_error {
    std::string boundary;
    explicit Canceled(std::string b) : std::runtime_error("Request canceled at safe boundary " + b), boundary(std::move(b)) {}
};
struct Request {
    std::string id; std::filesystem::path manifest, output; std::string manifestSha, outputName; uint32_t cap = 0; uint64_t index = 0;
    bool cohort = false; std::vector<std::shared_ptr<Request>> pages;
    std::atomic<bool> cancel{false};
    bool started = false, cancelRequested = false; std::string cancelOrigin, phase; // Guarded by Shared::lock.
};
struct Terminal {
    std::string status, failureClass, error, boundary, stopReason, resultSha, origin; Json stopToken = nullptr; // origin: cancel origin snapshot taken under the lock.
    bool dispatched = false, retired = false, drained = false, released = false, poisoned = false; uint64_t generated = 0;
    Json tracked = nullptr;
};

class Worker {
public:
    Worker(Options o, std::string selection, Json observedJob) : options(std::move(o)), gemvSelection(std::move(selection)), job(std::move(observedJob)) {
        output = GetStdHandle(STD_OUTPUT_HANDLE);
        if(options.experimentalB2){cohortArithmetic=cohortSelection(options,gemvSelection);chandra::dc::experimental::enableGemmPadded32();}
    }
    // Startup: Device and the complete authenticated model exist only in execute mode, created on this thread.
    bool start() {
        std::unique_ptr<import_row::FinalTailObserver> finalTails; std::unique_ptr<NewOutput> progress;
        uint32_t tailRecords = 0; uint64_t tailBytes = 0; std::filesystem::path auditDirectory;
        try {
            if (options.execute) {
                require(gemvSelection == "ordered", "Resident SRV-only execution requires explicit ordered B1 selection");
                auditDirectory = residentImportDirectory(options.outputRoot);
                finalTails = residentFinalTailObserver(auditDirectory, progress, tailRecords, tailBytes);
                heldModel = std::make_unique<Handle>(CreateFileW((options.model / L"model.safetensors").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
                heldConfig = std::make_unique<Handle>(CreateFileW((options.model / L"config.json").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
                device = std::make_unique<Device>(options.shaders.wstring(), options.pci, options.luid);
                identity = Json::parse(device->identityJson());
                weights = std::make_unique<ModelWeights>(*device, options.model.wstring(), 12ull * 1024 * 1024 * 1024, std::vector<std::string>{}, WeightStorageExperiment::exact,
                    nullptr, nullptr, nullptr, nullptr, WeightImportAPI::defaultInitial, finalTails.get(), true);
                provenance = Json::parse(weights->provenanceJson());
                finalTails->afterImport(*device, weights->at("model.language_model.embed_tokens.weight"), weights->at("lm_head.weight"));
                requireCleanFinalImportForArithmetic(*finalTails, WeightImportAPI::defaultInitial, false);
                finalTails->releaseTargets();
                requireCleanFinalImportForArithmetic(*finalTails, WeightImportAPI::defaultInitial, true);
                requireResidentImportProvenance(provenance);
                const auto provenanceFile = persistResidentMetadata(auditDirectory / L"model-provenance.json", provenance);
                const auto auditFile = persistResidentMetadata(auditDirectory / L"audit.json", finalTails->report());
                residentImport = {{"weight_import_api", "default-initial"}, {"weight_srv_only", true}, {"clean_final_import_gate_completed", true},
                    {"audit_directory", utf8(auditDirectory.filename().wstring())}, {"audit", auditFile}, {"full_model_provenance", provenanceFile}};
                residentBytes = device->trackedBufferBytes();
                require(residentBytes == provenance.at("tracked_buffer_bytes_after") && residentBytes == 9078531072ull, "Resident ownership baseline differs after observer reference release");
            }
        } catch (const std::exception& e) {
            if (finalTails) {
                if (!finalTails->completed()) finalTails->failure(e.what());
                finalTails->releaseTargets();
                try { persistResidentMetadata(auditDirectory / L"startup-failure.json", {{"error", bounded(e.what())}, {"final_import_tail_audit", finalTails->report()}, {"accepting", false}}); } catch (const std::exception&) {}
            }
            std::lock_guard<std::mutex> guard(lock); emitLocked({{"event", "fatal"}, {"stage", "startup"}, {"error", bounded(e.what())}, {"device_created", device != nullptr}, {"model_loaded", false}});
            poisoned = true; closing = true; closeReason = "startup_failed"; return false;
        }
        std::lock_guard<std::mutex> guard(lock);
        deviceLive = device != nullptr; modelLive = weights != nullptr;
        if (options.leaseMilliseconds) renewLeaseLocked();
        Json model = {{"alias", "chandra"}, {"revision", modelRevision}, {"context_limit", contextLimit}, {"normal_output_allowance", normalOutputLimit}, {"caller_stop_token_ids", Json(callerStops)}};
        Json resident = nullptr;
        if (weights) resident = {{"revision", provenance.value("revision", Json(nullptr))}, {"model_sha256", provenance.value("model_sha256", Json(nullptr))},
            {"config_sha256", provenance.value("config_sha256", Json(nullptr))}, {"uploaded_storage_bytes", provenance.value("uploaded_storage_bytes", Json(nullptr))},
            {"tie_byte_equality", true}, {"full_graph_requested", true}, {"provenance_sha256", sha256Text(provenance.dump())}, {"resident_tracked_bytes", residentBytes}, {"imports", 1}, {"import", residentImport}};
        Json ready = {{"event", "ready"}, {"mode", options.execute ? "execute" : "plan"}, {"accepting", true},
            {"protocol", {{"request_schema", requestSchema}, {"max_line_bytes", maxLineBytes}, {"max_event_bytes", maxEventBytes}, {"active_slots", 1}, {"waiting_slots", 1},
                {"max_lifetime_requests", maxLifetimeRequests}, {"max_protocol_errors", maxProtocolErrors},
                {"execution", "Serial B1 requests on the Device-owning thread; no batching or concurrent GPU execution"}}},
            {"model", model}, {"resident_model", resident}, {"device_identity", identity}, {"device_created", device != nullptr}, {"model_loaded", weights != nullptr},
            {"gemv_b1_selection", gemvSelection}, {"job", job}, {"lease_ms", options.leaseMilliseconds ? Json(options.leaseMilliseconds) : Json(nullptr)},
            {"obligations", obligations}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
        if(options.experimentalB2){ready["protocol"]["active_slots"]=2;ready["protocol"]["waiting_slots"]=0;ready["protocol"]["execution"]="Explicit static one/two-page cohort: independent serial vision/prefill; one page uses original advance, two pages use ordered B1 plus padded32 B2 decode";ready["cohort_arithmetic"]=cohortArithmetic;ready["cohort_cancel_scope"]="whole_cohort_only";}
        if(device)ready["drain_wait"]=Json::parse(device->memoryJson()).at("drain_wait");
        emitLocked(std::move(ready));
        return true;
    }
    // Starts the stdin reader. finish() cancels its pending read and waits for it, bounded, before the exit event.
    std::thread launchReader(const std::shared_ptr<Worker>& self) {
        { std::lock_guard<std::mutex> guard(lock); readerRunning = true; inputState = "open"; }
        std::thread reader([self] { self->read(); });
        std::lock_guard<std::mutex> guard(lock); readerThread = reader.native_handle(); return reader;
    }
    bool readerStopped() { std::lock_guard<std::mutex> guard(lock); return !readerRunning; }
    // Bounded line reader; runs on its own thread and never touches the Device.
    void read() {
        readInput();
        std::lock_guard<std::mutex> guard(lock); readerRunning = false; readerExited.notify_all();
    }
    // Device-owning thread: run admitted requests serially until closing, then release and drain.
    int serve() {
        for (;;) {
            std::shared_ptr<Request> next;
            {
                std::unique_lock<std::mutex> guard(lock);
                for (;;) {
                    if (active && !active->started) { next = active; next->started = true; break; }
                    if (closing && !active) break;
                    if (options.leaseMilliseconds) {
                        if (Clock::now() >= leaseDeadline()) { closeLocked("lease_expired"); continue; }
                        wake.wait_until(guard, leaseDeadline());
                    } else wake.wait(guard);
                }
            }
            if (!next) break;
            if(next->cohort)runCohort(*next);else run(*next);
        }
        return finish();
    }
private:
    Options options; std::string gemvSelection; Json job; HANDLE output = INVALID_HANDLE_VALUE;
    std::unique_ptr<Device> device; std::unique_ptr<ModelWeights> weights; std::unique_ptr<Handle> heldModel, heldConfig; Json identity = nullptr, provenance = nullptr, residentImport = nullptr, cohortArithmetic = nullptr; uint64_t residentBytes = 0;
    std::array<TextRequest,2> cohortRequests; bool cohortOwnersHeld = false; // Device thread; retained through failed drain until worker retirement.
    std::mutex lock; std::condition_variable wake;
    std::shared_ptr<Request> active, waiting; std::set<std::string> admitted;
    bool closing = false, poisoned = false, finished = false, channelBroken = false, deviceLive = false, modelLive = false; std::string closeReason; // Guarded by lock.
    // Input channel, guarded by lock. inputState: not_started, open, eof, failed, protocol_error_limit or canceled_at_exit.
    bool readerRunning = false, stopInput = false, readCancelRequested = false; std::string inputState = "not_started"; Json inputError = nullptr;
    std::thread::native_handle_type readerThread{}; std::condition_variable readerExited;
    uint64_t sequence = 0, rejected = 0; uint32_t protocolErrors = 0; Json terminals = {{"completed", 0}, {"planned", 0}, {"failed", 0}, {"canceled", 0}};
    std::atomic<int64_t> leaseTicks{0};
    // Owned by the worker (not a function-local static) so a reader left blocked at exit never sees it destroyed.
    const std::map<std::string, std::set<std::string>> fields = {
        {"submit", {"schema", "type", "id", "model", "input_manifest", "input_sha256", "output", "diagnostic_token_cap"}},
        {"submit_cohort", {"schema", "type", "id", "model", "requests"}},
        {"cancel", {"schema", "type", "id"}}, {"status", {"schema", "type"}}, {"lease", {"schema", "type"}}, {"shutdown", {"schema", "type"}}};

    Clock::time_point leaseDeadline() const { return Clock::time_point(Clock::duration(leaseTicks.load())); }
    void renewLeaseLocked() { leaseTicks.store((Clock::now() + std::chrono::milliseconds(options.leaseMilliseconds)).time_since_epoch().count()); }
    // Every stdout line is one schema-tagged event written under the lock; failure closes the worker.
    void emitLocked(Json event) {
        if (finished || channelBroken) return;
        event["schema"] = eventSchema; event["seq"] = sequence++;
        std::string text = event.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n";
        if (text.size() > maxEventBytes) text = Json{{"schema", eventSchema}, {"seq", event["seq"]}, {"event", "internal_error"}, {"detail", "Event exceeded its bound and was withheld"}}.dump() + "\n";
        DWORD wrote = 0;
        if (!WriteFile(output, text.data(), DWORD(text.size()), &wrote, nullptr) || wrote != text.size()) { channelBroken = true; closeLocked("event_channel_failed"); }
    }
    // The ceiling is absolute: the error that reaches it closes the worker and ends input at once, including
    // lines already buffered, so no further input is parsed and no later protocol_error event exists. It always
    // classifies input, even when an earlier close (such as shutdown) keeps its place as the close reason.
    void protocolErrorLocked(const char* reason, const std::string& detail) {
        if (stopInput || protocolErrors >= maxProtocolErrors) return;
        ++protocolErrors; emitLocked({{"event", "protocol_error"}, {"reason", reason}, {"detail", bounded(detail)}, {"count", protocolErrors}, {"limit", maxProtocolErrors}});
        if (protocolErrors < maxProtocolErrors) return;
        inputState = "protocol_error_limit";
        closeLocked("protocol_error_limit");
    }
    void readInput() {
        HANDLE input = GetStdHandle(STD_INPUT_HANDLE); std::vector<char> chunk(65536); std::string line; bool discarding = false;
        while (keepReading()) {
            DWORD got = 0; const BOOL ok = ReadFile(input, chunk.data(), DWORD(chunk.size()), &got, nullptr);
            const DWORD error = ok ? DWORD(0) : GetLastError(); // Retained before any other call can replace it.
            if (!ok || got == 0) { std::lock_guard<std::mutex> guard(lock); endInputLocked(ok != FALSE, error, line.size()); return; }
            if (!keepReading()) return; // Bytes that arrive after finish() stopped input are not parsed.
            for (DWORD i = 0; i < got; ++i) {
                const char c = chunk[i];
                if (c == '\n') {
                    if (!discarding) {
                        try { handle(line); }
                        catch (const std::exception& e) { std::lock_guard<std::mutex> guard(lock); protocolErrorLocked("unprocessable", e.what()); }
                    }
                    line.clear(); discarding = false;
                    if (!keepReading()) return;
                    continue;
                }
                if (discarding) continue;
                if (line.size() >= maxLineBytes) {
                    discarding = true; line.clear();
                    { std::lock_guard<std::mutex> guard(lock); protocolErrorLocked("line_too_long", "Line exceeds " + std::to_string(maxLineBytes) + " bytes; discarded through its newline"); }
                    if (!keepReading()) return;
                    continue;
                }
                line.push_back(c);
            }
        }
    }
    // Reader thread: false once input has ended, which the reader never reopens; records a stop requested by finish().
    bool keepReading() {
        std::lock_guard<std::mutex> guard(lock);
        if (stopInput && inputState == "open") inputState = "canceled_at_exit";
        return inputState == "open";
    }
    // Reader thread. A successful zero-byte read and ERROR_BROKEN_PIPE (the parent closed its end) are end of input.
    // An unterminated final line is counted while input is still open, so when it is the sixteenth error the
    // ceiling, not EOF, classifies input. ERROR_OPERATION_ABORTED is expected only after finish() canceled the read.
    // Any other status, including an abort the worker did not request, is a channel failure: it is reported with its
    // Win32 code, closes the worker like EOF (retiring the waiting request, stopping the active one at its next safe
    // boundary) and exits 3.
    void endInputLocked(bool ok, DWORD error, size_t unterminated) {
        if (!ok) inputError = error;
        if (ok || error == ERROR_BROKEN_PIPE) {
            if (unterminated) protocolErrorLocked("incomplete_final_line", "Input ended without a newline");
            if (inputState == "open") inputState = "eof";
            closeLocked("input_eof"); return;
        }
        if (stopInput && error == ERROR_OPERATION_ABORTED) { inputState = "canceled_at_exit"; return; }
        inputState = "failed";
        emitLocked({{"event", "input_channel_failed"}, {"win32_error", error}, {"win32_name", readStatusName(error)}, {"unterminated_bytes", unterminated},
            {"detail", "ReadFile on standard input failed with a status that is not end of input; no further input is read. Admission stops, the waiting request "
                "is retired, the active request stops at its next safe boundary and drains, the model is released and the worker exits 3. Nothing is replayed: "
                "give a new worker a readable pipe and resubmit."}});
        closeLocked("input_channel_failed");
    }
    // Device thread at exit. CancelSynchronousIo completes a pending ReadFile with ERROR_OPERATION_ABORTED, which the
    // reader attributes to this request. Between reads it finds nothing to cancel and the reader sees stopInput
    // instead, so it repeats, bounded; a reader still blocked after that is left to end with the process.
    void stopReaderLocked(std::unique_lock<std::mutex>& guard) {
        stopInput = true;
        for (uint32_t attempt = 0; readerRunning && attempt < readerStopAttempts; ++attempt) {
            if (CancelSynchronousIo(readerThread)) readCancelRequested = true;
            readerExited.wait_for(guard, readerStopWait, [&] { return !readerRunning; });
        }
    }
    void rejectLocked(const Json& submitted, const char* reason, const std::string& detail) {
        ++rejected; emitLocked({{"event", "rejected"}, {"submitted_id", submitted}, {"reason", reason}, {"detail", bounded(detail)}});
    }
    // Stops admission, retires the waiting request before any dispatch and asks the active one to stop at its next safe boundary.
    void closeLocked(const std::string& reason) {
        if (closing) return;
        closing = true; closeReason = reason; emitLocked({{"event", "closing"}, {"reason", reason}, {"accepting", false}});
        if (waiting) { auto r = waiting; waiting.reset(); r->cancelOrigin = reason; retireUndispatchedLocked(*r, "canceled", ""); }
        if (active) {
            active->cancel = true;
            if (!active->cancelRequested) { active->cancelRequested = true; active->cancelOrigin = reason; emitLocked({{"event", "cancel_requested"}, {"id", active->id}, {"state", active->started ? "active" : "admitted"}, {"origin", reason}, {"released", false}}); }
        }
        wake.notify_all();
    }
    Json terminalEvent(const Request& r, const Terminal& t) {
        return {{"event", "terminal"}, {"id", r.id}, {"status", t.status}, {"failure_class", t.failureClass.empty() ? Json(nullptr) : Json(t.failureClass)},
            {"error", t.error.empty() ? Json(nullptr) : Json(bounded(t.error))}, {"dispatched", t.dispatched}, {"cancel_origin", t.origin.empty() ? Json(nullptr) : Json(t.origin)},
            {"cancel_boundary", t.boundary.empty() ? Json(nullptr) : Json(t.boundary)}, {"generated_tokens", t.generated}, {"stop_reason", t.stopReason.empty() ? Json(nullptr) : Json(t.stopReason)},
            {"stop_token_id", t.stopToken}, {"request_retired", t.retired}, {"drained", t.drained}, {"tracked_buffer_bytes_after", t.tracked},
            {"resident_model_bytes", options.execute ? Json(residentBytes) : Json(nullptr)}, {"released", t.released}, {"worker_poisoned", t.poisoned}, {"request_replayed", false},
            {"result_file", t.resultSha.empty() ? Json(nullptr) : Json("result.json")}, {"result_sha256", t.resultSha.empty() ? Json(nullptr) : Json(t.resultSha)},
            {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
    }
    Json receipt(const Request& r, const Terminal& t) {
        return {{"schema", resultSchema}, {"request_id", r.id}, {"terminal_status", t.status}, {"failure_class", t.failureClass.empty() ? Json(nullptr) : Json(t.failureClass)},
            {"error", t.error.empty() ? Json(nullptr) : Json(utf8Safe(t.error))}, {"dispatched", t.dispatched}, {"cancel_origin", t.origin.empty() ? Json(nullptr) : Json(t.origin)},
            {"input_manifest_sha256", r.manifestSha}, {"request_replayed", false}, {"passed", false}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
    }
    // A request that never reached the Device: its slot is free once its terminal event is written.
    void retireUndispatchedLocked(Request& r, const char* status, const std::string& failureClass) {
        Terminal t; t.status = status; t.failureClass = failureClass; t.released = true; t.retired = true; t.origin = r.cancelOrigin;
        if (failureClass == "worker_poisoned") t.error = "Worker poisoned before this request was dispatched; it is not replayed";
        try { t.resultSha = writeReceipt(r.output, receipt(r, t)); } catch (const std::exception& e) { t.error = t.error.empty() ? std::string("Receipt write failed: ") + e.what() : t.error; }
        terminals[t.status] = terminals[t.status].get<uint64_t>() + 1; emitLocked(terminalEvent(r, t));
    }
    void handle(const std::string& line) {
        Json message;
        try { message = strictJson(std::vector<char>(line.begin(), line.end())); }
        catch (const std::exception& e) { std::lock_guard<std::mutex> guard(lock); protocolErrorLocked("malformed_json", e.what()); return; }
        std::lock_guard<std::mutex> guard(lock);
        if (stopInput) return;
        if (!message.contains("schema") || message.at("schema") != requestSchema) { protocolErrorLocked("schema", std::string("Every message requires schema ") + requestSchema); return; }
        const Json type = message.value("type", Json(nullptr));
        if (!type.is_string()) { protocolErrorLocked("unknown_type", "Message type must be a string"); return; }
        const std::string kind = type.get<std::string>();
        auto allowed = fields.find(kind);
        if (allowed == fields.end()) { protocolErrorLocked("unknown_type", "Unknown message type"); return; }
        if (options.leaseMilliseconds && !closing) renewLeaseLocked(); // Any well-formed versioned message renews the parent lease.
        Json submittedId = message.contains("id") && message.at("id").is_string() && safeId(message.at("id").get<std::string>()) ? message.at("id") : Json(nullptr);
        for (auto it = message.begin(); it != message.end(); ++it) if (!allowed->second.count(it.key())) {
            if (kind == "submit" || kind == "submit_cohort") rejectLocked(submittedId, "unknown_field", "Unknown field " + it.key());
            else if (kind == "cancel") emitLocked({{"event", "cancel_rejected"}, {"submitted_id", submittedId}, {"reason", "unknown_field"}, {"detail", bounded("Unknown field " + it.key())}});
            else protocolErrorLocked("unknown_field", "Unknown field " + it.key());
            return;
        }
        if (kind == "status") { emitLocked(status()); return; }
        if (kind == "lease") return;
        if (kind == "shutdown") { closeLocked("shutdown_requested"); return; }
        if (kind == "cancel") { cancelLocked(message, submittedId); return; }
        if(kind=="submit_cohort")submitCohortLocked(message,submittedId);else submitLocked(message, submittedId);
    }
    void cancelLocked(const Json& message, const Json& submittedId) {
        auto refuse = [&](const char* reason) { emitLocked({{"event", "cancel_rejected"}, {"submitted_id", submittedId}, {"reason", reason}}); };
        if (submittedId.is_null()) { refuse("invalid_id"); return; }
        const std::string id = message.at("id").get<std::string>();
        if(active && active->cohort)for(const auto& page:active->pages)if(page->id==id){refuse("cohort_member_cancel_not_supported");return;}
        if (waiting && waiting->id == id) {
            auto r = waiting; waiting.reset(); r->cancelRequested = true; r->cancelOrigin = "client";
            emitLocked({{"event", "cancel_requested"}, {"id", id}, {"state", "waiting"}, {"origin", "client"}, {"released", false}});
            retireUndispatchedLocked(*r, "canceled", ""); return;
        }
        if (active && active->id == id) {
            if (active->cancelRequested) { refuse("cancel_already_requested"); return; }
            active->cancelRequested = true; active->cancelOrigin = "client"; active->cancel = true;
            emitLocked({{"event", "cancel_requested"}, {"id", id}, {"state", active->started ? "active" : "admitted"}, {"origin", "client"}, {"released", false}});
            wake.notify_all(); return;
        }
        refuse(admitted.count(id) ? "already_terminal" : "unknown_id");
    }
    void submitLocked(const Json& m, const Json& submittedId) {
        if(options.experimentalB2){rejectLocked(submittedId,"cohort_mode_only","Dedicated B2 mode accepts submit_cohort only; default single-page worker is unchanged");return;}
        for (auto name : {"id", "model", "input_manifest", "input_sha256", "output"}) if (!m.contains(name)) { rejectLocked(submittedId, "missing_field", std::string("Missing field ") + name); return; }
        if (submittedId.is_null()) { rejectLocked(nullptr, "invalid_id", "Request id must be 1-64 characters of [A-Za-z0-9._:-]"); return; }
        const std::string id = m.at("id").get<std::string>();
        if (admitted.count(id)) { rejectLocked(submittedId, "duplicate_id", "Request id was already admitted by this worker"); return; }
        if (m.at("model") != "chandra") { rejectLocked(submittedId, "invalid_model", "Only model alias chandra is served"); return; }
        auto r = std::make_shared<Request>(); r->id = id;
        try { r->manifestSha = digestText(m.at("input_sha256")); } catch (const std::exception& e) { rejectLocked(submittedId, "invalid_sha256", e.what()); return; }
        if (m.contains("diagnostic_token_cap")) {
            const auto& cap = m.at("diagnostic_token_cap");
            if (!cap.is_number_unsigned() || cap.get<uint64_t>() == 0 || cap.get<uint64_t>() >= normalOutputLimit) { rejectLocked(submittedId, "invalid_limit", "An explicit diagnostic cap must be an integer from 1 to 12383; omit it for the normal allowance"); return; }
            r->cap = uint32_t(cap.get<uint64_t>());
        }
        try {
            const auto& name = m.at("input_manifest"); require(name.is_string(), "Input manifest must be a relative string path");
            const std::string text = name.get<std::string>(); size_t start = 0;
            require(!text.empty() && text.size() <= 512, "Input manifest path must be 1-512 bytes");
            while (start <= text.size()) {
                size_t end = text.find('/', start); if (end == std::string::npos) end = text.size();
                require(safeComponent(text.substr(start, end - start), maxComponentBytes), "Input manifest components must be safe names of [A-Za-z0-9._-]; no traversal or device names");
                start = end + 1;
            }
            r->manifest = localFile(options.inputRoot, name);
        } catch (const std::exception& e) { rejectLocked(submittedId, "invalid_input_path", e.what()); return; }
        const auto& outputName = m.at("output");
        if (!outputName.is_string() || !safeComponent(outputName.get<std::string>(), maxOutputNameBytes)) { rejectLocked(submittedId, "invalid_output_name", "Output must be one new directory name of [A-Za-z0-9._-]"); return; }
        r->outputName = outputName.get<std::string>(); r->output = options.outputRoot / std::filesystem::u8path(r->outputName);
        if (closing) { rejectLocked(submittedId, poisoned ? "worker_poisoned" : "closing", "Worker is not accepting requests: " + closeReason); return; }
        if (admitted.size() >= maxLifetimeRequests) { rejectLocked(submittedId, "lifetime_limit", "Worker request-ID history is exhausted; start a new worker"); return; }
        if (active && waiting) { rejectLocked(submittedId, "busy", "One active and one waiting request already hold capacity"); return; }
        if (GetFileAttributesW(r->output.c_str()) != INVALID_FILE_ATTRIBUTES || !CreateDirectoryW(r->output.c_str(), nullptr)) { rejectLocked(submittedId, "output_exists", "Output directory exists or could not be created exclusively"); return; }
        DWORD attributes = GetFileAttributesW(r->output.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY) || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) { rejectLocked(submittedId, "output_exists", "Created output is not a plain directory"); return; }
        admitted.insert(id); r->index = admitted.size();
        const char* slot = active ? "waiting" : "active"; (active ? waiting : active) = r;
        emitLocked({{"event", "admitted"}, {"id", id}, {"slot", slot}, {"request_index", r->index}, {"output", r->outputName}, {"input_sha256", r->manifestSha},
            {"diagnostic_token_cap", r->cap ? Json(r->cap) : Json(nullptr)}, {"output_allowance", r->cap ? r->cap : normalOutputLimit}});
        wake.notify_all();
    }
    void submitCohortLocked(const Json& m,const Json& submittedId) {
        if(!options.experimentalB2){rejectLocked(submittedId,"explicit_cohort_mode_required","Both explicit padded32 and B2 worker flags required");return;}
        if(submittedId.is_null()){rejectLocked(nullptr,"invalid_id","A bounded cohort id is required");return;}
        if(!m.contains("model") || m.at("model")!="chandra" || !m.contains("requests") || !m.at("requests").is_array() || (m.at("requests").empty() || m.at("requests").size()>2)){rejectLocked(submittedId,"invalid_cohort","One or two page descriptors and model chandra required");return;}
        if(closing){rejectLocked(submittedId,poisoned?"worker_poisoned":"closing","Worker is not accepting a cohort");return;}
        if(active || waiting){rejectLocked(submittedId,"busy","A static cohort requires both native capacity slots idle");return;}
        if(admitted.size()>maxLifetimeRequests-1-m.at("requests").size()){rejectLocked(submittedId,"lifetime_limit","Fresh group/page IDs must fit the unchanged history bound");return;}
        auto group=std::make_shared<Request>();group->id=submittedId.get<std::string>();group->cohort=true;group->pages.resize(m.at("requests").size());
        std::set<std::string> ids={group->id},owners;
        try {
            require(!admitted.count(group->id),"Cohort id was already admitted");
            for(uint32_t slot=0;slot<group->pages.size();++slot) {
                const auto& item=m.at("requests").at(slot);require(item.is_object() && item.size()==4,"Each page has exactly id,input_manifest,input_sha256,output; no diagnostic cap");
                for(const char* key:{"id","input_manifest","input_sha256","output"})require(item.contains(key),std::string("Missing cohort page field ")+key);
                require(item.at("id").is_string() && safeId(item.at("id").get<std::string>()),"Bounded distinct page id required");
                auto page=std::make_shared<Request>();page->id=item.at("id").get<std::string>();
                require(ids.insert(page->id).second && !admitted.count(page->id),"Group/page IDs collide or were already admitted");
                page->manifestSha=digestText(item.at("input_sha256")); // Original authenticateInput and requireCohortInput admit exact diagnostic or production packages on the Device thread.
                require(item.at("input_manifest").is_string(),"Relative page manifest string required");const auto path=item.at("input_manifest").get<std::string>();
                require(!path.empty() && path.size()<=512,"Bounded relative manifest required");size_t start=0;
                while(start<=path.size()){size_t end=path.find('/',start);if(end==std::string::npos)end=path.size();require(safeComponent(path.substr(start,end-start),maxComponentBytes),"Safe local manifest components required");start=end+1;}
                page->manifest=localFile(options.inputRoot,item.at("input_manifest"));
                require(item.at("output").is_string() && safeComponent(item.at("output").get<std::string>(),maxOutputNameBytes),"Safe fresh page output name required");
                page->outputName=item.at("output").get<std::string>();std::string folded=page->outputName;
                std::transform(folded.begin(),folded.end(),folded.begin(),[](unsigned char c){return char(std::tolower(c));});
                require(owners.insert(folded).second,"Page output owners alias after Windows case folding");
                page->output=options.outputRoot/std::filesystem::u8path(page->outputName);
                require(GetFileAttributesW(page->output.c_str())==INVALID_FILE_ATTRIBUTES,"Page output already exists");
                group->pages[slot]=std::move(page);
            }
            // Both descriptors and owner collisions are admitted before either fresh directory is reserved.
            for(const auto& page:group->pages){require(CreateDirectoryW(page->output.c_str(),nullptr),"Fresh cohort output reservation failed; no page model work submitted");
                DWORD a=GetFileAttributesW(page->output.c_str());require(a!=INVALID_FILE_ATTRIBUTES && (a&FILE_ATTRIBUTE_DIRECTORY) && !(a&FILE_ATTRIBUTE_REPARSE_POINT),"Created cohort output must be a plain directory");}
        } catch(const std::exception& e){rejectLocked(submittedId,"invalid_cohort",e.what());return;}
        admitted.insert(group->id);group->index=admitted.size();
        for(const auto& page:group->pages){admitted.insert(page->id);page->index=admitted.size();}
        active=group;
        for(uint32_t slot=0;slot<group->pages.size();++slot){const auto& page=*group->pages[slot];emitLocked({{"event","admitted"},{"id",page.id},{"cohort_id",group->id},{"cohort_slot",slot},{"slot","active_cohort"},
            {"request_index",page.index},{"output",page.outputName},{"input_sha256",page.manifestSha},{"diagnostic_token_cap",nullptr},{"output_allowance",normalOutputLimit}});}
        wake.notify_all();
    }
    Json status() {
        auto describe = [&](const std::shared_ptr<Request>& r) -> Json {
            if (!r) return nullptr;
            Json result={{"id", r->id}, {"started", r->started}, {"phase", r->phase.empty() ? Json(nullptr) : Json(r->phase)}, {"cancel_requested", r->cancelRequested}};
            if(r->cohort){result["cohort_pages"]=Json::array();for(uint32_t slot=0;slot<r->pages.size();++slot)result["cohort_pages"].push_back({{"id",r->pages[slot]->id},{"slot",slot},{"output",r->pages[slot]->outputName}});result["cancel_scope"]="whole_cohort_only";}
            return result;
        };
        Json lease = nullptr;
        if (options.leaseMilliseconds) lease = {{"ms", options.leaseMilliseconds}, {"remaining_ms", std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(leaseDeadline() - Clock::now()).count())}};
        return {{"event", "status"}, {"mode", options.execute ? "execute" : "plan"}, {"state", poisoned ? "poisoned" : closing ? "closing" : active ? "running" : "idle"},
            {"accepting", !closing}, {"active", describe(active)}, {"waiting", describe(waiting)}, {"admitted_total", admitted.size()}, {"terminal_counts", terminals},
            {"rejected_total", rejected}, {"protocol_errors", protocolErrors}, {"device_created", deviceLive}, {"model_loaded", modelLive},
            {"status_wakes_device", false}, {"lease", lease}, {"qualified_OCR", false}, {"performance_claim", false}};
    }
    // Device thread only. Lease expiry is noticed here and while idle; the flag itself is set by the reader.
    void checkpoint(Request& r, const std::string& boundary) {
        if (options.leaseMilliseconds && Clock::now() >= leaseDeadline()) { std::lock_guard<std::mutex> guard(lock); closeLocked("lease_expired"); }
        if (r.cancel.load()) throw Canceled(boundary);
    }
    static bool visionBoundary(const std::string& name) {
        auto ends = [&](const std::string& suffix) { return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0; };
        return name == "patch_embedding" || name == "position_added" || name == "pooler_output" || (name.rfind("blocks.", 0) == 0 && ends(".output"));
    }
    void run(Request& r) {
        { std::lock_guard<std::mutex> guard(lock); emitLocked({{"event", "started"}, {"id", r.id}, {"request_index", r.index}, {"mode", options.execute ? "execute" : "plan"}}); }
        Terminal t; Json report;
        try {
            checkpoint(r, "before_input_authentication");
            Input in;
            try {
                in = authenticateInput(r.manifest, r.manifestSha);
                require(std::set<uint32_t>(in.stopIds.begin(), in.stopIds.end()) == callerStops, "Authenticated caller stop IDs must be exactly 248044 and 248046");
            } catch (const Canceled&) { throw; }
            catch (const std::exception& e) { t.status = "failed"; t.failureClass = "input_refused"; t.error = utf8Safe(e.what()); }
            if (t.status.empty()) {
                checkpoint(r, "after_input_authentication");
                if (options.execute) execute(r, in, t, report);
                else { report = forecastReport(in, r.manifestSha, r.cap); t.status = "planned"; t.released = t.retired = true; }
            }
        } catch (const Canceled& c) { t.status = "canceled"; t.boundary = c.boundary; t.released = t.retired = true; }
        if (!t.dispatched && t.status != "completed") { t.released = t.retired = true; }
        { std::lock_guard<std::mutex> guard(lock); t.origin = r.cancelOrigin; }
        if (report.is_null()) report = receipt(r, t);
        else {
            report["worker"] = {{"schema", resultSchema}, {"request_id", r.id}, {"request_index", r.index}, {"terminal_status", t.status}, {"resident_model", options.execute},
                {"model_imports_this_process", options.execute ? 1 : 0}, {"gemv_b1_selection", gemvSelection}, {"cancel_origin", t.origin.empty() ? Json(nullptr) : Json(t.origin)},
                {"cancel_boundary", t.boundary.empty() ? Json(nullptr) : Json(t.boundary)}, {"released", t.released}, {"worker_poisoned", t.poisoned}, {"request_replayed", false}};
        }
        try { t.resultSha = writeReceipt(r.output, report); }
        catch (const std::exception& e) {
            if (t.status == "completed" || t.status == "planned") { t.status = "failed"; t.failureClass = "receipt_write_failed"; }
            t.error = (t.error.empty() ? std::string() : t.error + "; ") + "Receipt write failed: " + e.what();
        }
        std::lock_guard<std::mutex> guard(lock);
        terminals[t.status] = terminals[t.status].get<uint64_t>() + 1; emitLocked(terminalEvent(r, t));
        if (t.poisoned) {
            // The waiting request never reached the Device; it fails without dispatch or replay.
            auto queued = waiting; waiting.reset(); active.reset(); poisoned = true; closeLocked("worker_poisoned");
            if (queued) retireUndispatchedLocked(*queued, "failed", "worker_poisoned");
        } else { active = waiting; waiting.reset(); }
        wake.notify_all();
    }
    void runCohort(Request& group) {
        std::array<Terminal,2> terminal;
        std::array<Json,2> reports;
        std::array<Input,2> inputs;
        {std::lock_guard<std::mutex> guard(lock);for(const auto& page:group.pages)page->started=true;emitLocked({{"event","cohort_started"},{"id",group.id},{"mode",options.execute?"execute":"plan"},{"slots",group.pages.size()}});}
        uint32_t authenticating=0;bool executionEntered=false;
        try {
            checkpoint(group,"before_cohort_input_authentication");
            for(uint32_t slot=0;slot<group.pages.size();++slot){authenticating=slot;const auto& page=*group.pages[slot];inputs[slot]=authenticateInput(page.manifest,page.manifestSha);
                require(std::set<uint32_t>(inputs[slot].stopIds.begin(),inputs[slot].stopIds.end())==callerStops,"Exact authenticated caller stop IDs required");
                requireCohortInput(inputs[slot],page.manifestSha);}
            checkpoint(group,"after_cohort_input_authentication");
            if(options.execute){executionEntered=true;executeCohort(group,inputs,terminal,reports);}
            else for(uint32_t slot=0;slot<group.pages.size();++slot){reports[slot]=forecastReport(inputs[slot],group.pages[slot]->manifestSha,0);terminal[slot].status="planned";terminal[slot].retired=terminal[slot].released=true;}
        } catch(const Canceled& c){for(uint32_t slot=0;slot<group.pages.size();++slot){auto& t=terminal[slot];t.status="canceled";t.boundary=c.boundary;t.error=c.what();t.retired=t.drained=t.released=!executionEntered;t.poisoned=executionEntered;}if(executionEntered)cohortOwnersHeld=true;}
        catch(const std::exception& e){for(uint32_t slot=0;slot<group.pages.size();++slot){auto& t=terminal[slot];t.status="failed";t.failureClass=executionEntered?"cohort_execution_failure":slot==authenticating?"input_refused":"cohort_input_refused";t.error=utf8Safe(e.what());t.retired=t.drained=t.released=!executionEntered;t.poisoned=executionEntered;}if(executionEntered)cohortOwnersHeld=true;}
        for(uint32_t slot=0;slot<group.pages.size();++slot) {
            auto& t=terminal[slot];const auto& page=*group.pages[slot];
            {std::lock_guard<std::mutex> guard(lock);t.origin=group.cancelOrigin;}
            if(reports[slot].is_null())reports[slot]=receipt(page,t);
            auto& report=reports[slot];
            report["schema"]="chandra.directcompute.worker-b2-page-result.v1";
            report["mode"]=options.execute?"resident_ordered_B1_padded32_B2_page":"resident_B2_cohort_plan";
            report["cohort_page_count"]=group.pages.size();report["decode_arithmetic"]=group.pages.size()==1?"original_single_row_advance":"advanceCohort_1_or_2_active_rows";
            report["request_retired"]=t.retired;report["drained"]=t.drained;report["request_buffers_released"]=t.released;
            report["worker"]={{"schema","chandra.directcompute.worker-b2-page-owner.v1"},{"request_id",page.id},{"request_index",page.index},
                {"cohort_id",group.id},{"cohort_slot",slot},{"cohort_page_count",group.pages.size()},{"terminal_status",t.status},{"resident_model",options.execute},{"model_imports_this_process",options.execute?1:0},
                {"cohort_arithmetic",cohortArithmetic},{"cancel_scope","whole_cohort_only"},{"cancel_origin",t.origin.empty()?Json(nullptr):Json(t.origin)},
                {"cancel_boundary",t.boundary.empty()?Json(nullptr):Json(t.boundary)},{"released",t.released},{"worker_poisoned",t.poisoned},{"request_replayed",false}};
            try{t.resultSha=writeReceipt(page.output,report);}catch(const std::exception& e){if(t.status=="completed"||t.status=="planned"){t.status="failed";t.failureClass="receipt_write_failed";}t.error=(t.error.empty()?std::string():t.error+"; ")+"Receipt write failed: "+e.what();}
        }
        std::lock_guard<std::mutex> guard(lock);
        Json pages=Json::array();bool workerFailed=false,bothReleased=true;std::string status=options.execute?"completed":"planned";
        for(uint32_t slot=0;slot<group.pages.size();++slot){const auto& t=terminal[slot];const auto& page=*group.pages[slot];
            terminals[t.status]=terminals[t.status].get<uint64_t>()+1;auto event=terminalEvent(page,t);event["cohort_id"]=group.id;event["cohort_slot"]=slot;emitLocked(std::move(event));
            pages.push_back({{"id",page.id},{"slot",slot},{"status",t.status},{"released",t.released},{"result_sha256",t.resultSha.empty()?Json(nullptr):Json(t.resultSha)}});
            workerFailed=workerFailed||t.poisoned;bothReleased=bothReleased&&t.released;
            if(t.status=="failed")status="failed";else if(t.status=="canceled" && status!="failed")status="canceled";
        }
        emitLocked({{"event","cohort_terminal"},{"id",group.id},{"status",status},{"pages",pages},{"released",bothReleased},{"worker_poisoned",workerFailed},{"request_replayed",false}});
        active.reset();
        if(workerFailed){poisoned=true;closeLocked("worker_poisoned");}
        wake.notify_all();
    }
    void executeCohort(Request& group,const std::array<Input,2>& inputs,std::array<Terminal,2>& terminal,std::array<Json,2>& reports) {
        if(cohortOwnersHeld || device->trackedBufferBytes()!=residentBytes){for(uint32_t slot=0;slot<group.pages.size();++slot){auto& t=terminal[slot];t.status="failed";t.failureClass="ownership_accounting_failure";t.error="Cohort owners or tracked buffers differ from resident baseline before dispatch";t.poisoned=true;}return;}
        const auto begun=Clock::now();std::array<Clock::time_point,2> stamp={begun,begun};
        std::array<Json,2> phases={Json::array(),Json::array()},memories={Json::array(),Json::array()};
        std::array<std::vector<uint32_t>,2> generated;
        std::array<RequestHooks,2> hooks;
        std::array<const Input*,2> ownedInputs={&inputs[0],group.pages.size()==2?&inputs[1]:nullptr};
        std::array<std::filesystem::path,2> outputs={group.pages[0]->output,group.pages.size()==2?group.pages[1]->output:std::filesystem::path{}};
        for(uint32_t slot=0;slot<group.pages.size();++slot){const auto& page=*group.pages[slot];reports[slot]=executionReport(inputs[slot],page.manifestSha,0);
            reports[slot]["device_identity"]=identity;reports[slot]["model_provenance"]=provenance;reports[slot]["resident_import"]=residentImport;
            reports[slot]["cohort_arithmetic"]=cohortArithmetic;
            hooks[slot].phase=[&,slot](const char* name){auto now=Clock::now();double seconds=std::chrono::duration<double>(now-stamp[slot]).count();stamp[slot]=now;
                phases[slot].push_back({{"name",name},{"wall_seconds",seconds}});Json memory=Json::parse(device->memoryJson());memories[slot].push_back({{"phase",name},{"observed",memory},{"host_process",hostMemory()}});
                std::lock_guard<std::mutex> guard(lock);group.pages[slot]->phase=name;group.phase=name;emitLocked({{"event","phase"},{"id",group.pages[slot]->id},{"cohort_id",group.id},{"cohort_slot",slot},{"name",name},{"wall_seconds",seconds},{"device_memory",memory}});};
            hooks[slot].checkpoint=[&](const char* name){checkpoint(group,name);};
            hooks[slot].vision=[&](const std::string& name,const Buffer&,uint32_t,uint32_t){if(visionBoundary(name))checkpoint(group,"vision."+name);};
            hooks[slot].text=[&](const TextObservation& o){if(o.stage==TextStage::LayerOutput)checkpoint(group,"text.layer."+std::to_string(o.layer));else if(o.stage==TextStage::Logits)checkpoint(group,"text.logits");};
            hooks[slot].token=[&,slot](uint32_t index,const Json& row){std::lock_guard<std::mutex> guard(lock);emitLocked({{"event","token"},{"id",group.pages[slot]->id},{"cohort_id",group.id},{"cohort_slot",slot},
                {"index",index},{"token_id",row.at("token_id")},{"stop",row.at("stop")},{"best_logit",row.at("best_logit")},{"runner_up_logit",row.at("runner_up_logit")},{"top_margin",row.at("top_margin")},{"maximum_tie_count",row.at("maximum_tie_count")}});};
            terminal[slot].dispatched=true;
        }
        cohortOwnersHeld=true;bool canceled=false,failed=false;std::string error,boundary;
        try{generateCohort(*device,*weights,ownedInputs,outputs,reports,generated,cohortRequests,hooks,true,uint32_t(group.pages.size()));}
        catch(const Canceled& c){canceled=true;error=c.what();boundary=c.boundary;}
        catch(const std::exception& e){failed=true;error=utf8Safe(e.what());}
        bool released=true;for(uint32_t slot=0;slot<group.pages.size();++slot)released=released && reports[slot].value("request_retired",false) && reports[slot].value("drained",false);
        uint64_t after=0;try{after=device->trackedBufferBytes();released=released && after==residentBytes;}catch(...){released=false;}
        cohortOwnersHeld=!released;
        if(!released && error.empty())error="Cohort buffers did not retire to the resident baseline after drain";
        const bool workerFailed=failed || !released;
        for(uint32_t slot=0;slot<group.pages.size();++slot){auto& t=terminal[slot];auto& report=reports[slot];
            const bool ownComplete=report.value("generation_completed",false) && released;
            t.status=ownComplete?"completed":canceled?"canceled":"failed";
            t.failureClass=t.status=="failed"?(failed?"device_or_graph_failure":"ownership_accounting_failure"):"";
            t.error=ownComplete?std::string():error;t.boundary=t.status=="canceled"?boundary:std::string();t.poisoned=workerFailed;
            t.retired=report.value("request_retired",false);t.drained=report.value("drained",false);t.released=released;t.tracked=after;t.generated=generated[slot].size();
            t.stopReason=report.value("stop_reason",std::string());t.stopToken=report.value("stop_token_id",Json(nullptr));
            report["passed"]=t.status=="completed";report["request_buffers_released"]=released;report["tracked_buffer_bytes_after"]=after;report["resident_model_bytes"]=residentBytes;
            report["owned_worker_retirement_required"]=workerFailed;report["cohort_error"]=error.empty()?Json(nullptr):Json(error);
            if(t.status=="completed")report["passed_scope"]="This page journal completed with finite greedy tokens and proven cohort retirement; trained numerical/full-OCR qualification is pending";
            else report["error"]=t.error;
            finishExecutionReport(report,inputs[slot],generated[slot],phases[slot],memories[slot],begun);
            report["timing_scope"]="One static one/two-page cohort interval; one page uses original single-row advance, two use shared decode; phase intervals include cohort waiting and prove no sustained pages/s";
        }
    }
    void execute(Request& r, const Input& in, Terminal& t, Json& report) {
        uint64_t before = device->trackedBufferBytes();
        if (before != residentBytes) { t.status = "failed"; t.failureClass = "ownership_accounting_failure"; t.error = "Tracked buffers differ from the resident model before dispatch"; t.poisoned = true; return; }
        report = executionReport(in, r.manifestSha, r.cap); report["device_identity"] = identity; report["model_provenance"] = provenance;
        auto begun = Clock::now(), stamp = begun; Json phases = Json::array(), memories = Json::array(); std::vector<uint32_t> generated;
        RequestHooks hooks;
        hooks.phase = [&](const char* name) {
            auto now = Clock::now(); const double seconds = std::chrono::duration<double>(now - stamp).count(); stamp = now;
            phases.push_back({{"name", name}, {"wall_seconds", seconds}}); Json observed = Json::parse(device->memoryJson());
            memories.push_back({{"phase", name}, {"observed", observed}, {"host_process", hostMemory()}});
            std::lock_guard<std::mutex> guard(lock); r.phase = name; emitLocked({{"event", "phase"}, {"id", r.id}, {"name", name}, {"wall_seconds", seconds}, {"device_memory", observed}});
        };
        hooks.checkpoint = [&](const char* name) { checkpoint(r, name); };
        hooks.vision = [&](const std::string& name, const Buffer&, uint32_t, uint32_t) { if (visionBoundary(name)) checkpoint(r, "vision." + name); };
        hooks.text = [&](const TextObservation& o) {
            if (o.stage == TextStage::LayerOutput) checkpoint(r, "text.layer." + std::to_string(o.layer));
            else if (o.stage == TextStage::Logits) checkpoint(r, "text.logits");
        };
        hooks.token = [&](uint32_t index, const Json& row) {
            std::lock_guard<std::mutex> guard(lock);
            emitLocked({{"event", "token"}, {"id", r.id}, {"index", index}, {"token_id", row.at("token_id")}, {"stop", row.at("stop")}, {"best_logit", row.at("best_logit")},
                {"runner_up_logit", row.at("runner_up_logit")}, {"top_margin", row.at("top_margin")}, {"maximum_tie_count", row.at("maximum_tie_count")}});
        };
        t.dispatched = true;
        try { generate(*device, *weights, in, r.cap, r.output, report, generated, hooks); t.status = "completed"; }
        catch (const Canceled& c) { t.status = "canceled"; t.boundary = c.boundary; t.error = c.what(); }
        catch (const std::exception& e) { t.status = "failed"; t.failureClass = "device_or_graph_failure"; t.error = utf8Safe(e.what()); }
        t.retired = report.value("request_retired", false); t.drained = report.value("drained", false); t.generated = generated.size();
        t.stopReason = report.value("stop_reason", std::string()); t.stopToken = report.value("stop_token_id", Json(nullptr));
        try { uint64_t after = device->trackedBufferBytes(); t.tracked = after; t.released = t.retired && t.drained && after == residentBytes; }
        catch (const std::exception&) { t.released = false; }
        // Only completion or a cancellation at a safe boundary followed by a proven drain back to the
        // resident model leaves a healthy Device. Every other failure poisons the worker; nothing replays.
        if (t.status == "failed") t.poisoned = true;
        else if (!t.released) { t.poisoned = true; t.error = (t.error.empty() ? std::string() : t.error + "; ") + "Request buffers did not retire to the resident model after drain"; if (t.status == "completed") { t.status = "failed"; t.failureClass = "ownership_accounting_failure"; } }
        report["passed"] = t.status == "completed";
        if (t.status == "completed") report["passed_scope"] = "Executable graph returned finite greedy tokens, retired the request and drained to the resident model; numerical/full-OCR qualification is pending";
        else { report["error"] = t.error; report["owned_worker_retirement_required"] = t.poisoned; }
        report["request_buffers_released"] = t.released; report["tracked_buffer_bytes_after"] = t.tracked; report["resident_model_bytes"] = residentBytes;
        finishExecutionReport(report, in, generated, phases, memories, begun);
    }
    int finish() {
        Json release = {{"event", "exit"}, {"device_created", device != nullptr}, {"model_released", false}, {"drained", nullptr}, {"tracked_buffer_bytes_after_release", nullptr}, {"device_destroyed", false}};
        bool clean = true;
        if (device) {
            if(cohortOwnersHeld){
                try{device->drain();for(auto& request:cohortRequests){request.failed=true;request.retired=true;request.layers={};}cohortOwnersHeld=false;release["cohort_cache_retirement_drain_completed"]=true;}
                catch(const std::exception& e){clean=false;release["cohort_cache_retirement_drain_completed"]=false;release["cohort_cache_owners_retained_until_worker_retirement"]=true;release["cohort_cache_drain_error"]=bounded(e.what());}
            }
            try { weights.reset(); { std::lock_guard<std::mutex> guard(lock); modelLive = false; } release["model_released"] = true; device->drain(); release["drained"] = true; }
            catch (const std::exception& e) { clean = false; release["drained"] = false; release["drain_error"] = bounded(e.what()); }
            try { uint64_t tracked = device->trackedBufferBytes(); release["tracked_buffer_bytes_after_release"] = tracked; clean = clean && tracked == 0; }
            catch (const std::exception& e) { clean = false; release["tracking_error"] = bounded(e.what()); }
            heldConfig.reset(); heldModel.reset();
            device.reset(); release["device_destroyed"] = true; std::lock_guard<std::mutex> guard(lock); deviceLive = false;
        }
        heldConfig.reset(); heldModel.reset(); // Also closes partial startup handles if Device creation failed.
        std::unique_lock<std::mutex> guard(lock);
        stopReaderLocked(guard);
        // The first close reason is kept, but EOF or shutdown exits 0 only if no channel fact was recorded meanwhile:
        // a read failure, the protocol-error ceiling (also checked by count, so no ordering of EOF, shutdown and the
        // last error can hide it) or a broken event channel, whose lost events make the exit 3.
        const bool channelsHealthy = inputState != "failed" && inputState != "protocol_error_limit" && protocolErrors < maxProtocolErrors && !channelBroken;
        int code = poisoned || !clean ? 2 : (closeReason == "input_eof" || closeReason == "shutdown_requested") && channelsHealthy ? 0 : 3;
        if (closeReason == "startup_failed") code = 1;
        release["input"] = {{"state", readerRunning ? "open" : inputState}, {"win32_error", inputError}, {"read_cancel_requested", readCancelRequested}, {"reader_stopped", !readerRunning}};
        release["reason"] = closeReason; release["clean"] = clean && !poisoned; release["exit_code"] = code; release["admitted_total"] = admitted.size();
        release["terminal_counts"] = terminals; release["rejected_total"] = rejected; release["protocol_errors"] = protocolErrors; release["request_replayed"] = false;
        release["obligations"] = obligations; emitLocked(release); finished = true;
        // The exit event's own write is the last channel fact. If it is the first write to fail, the event and the
        // exit_code it carries were not delivered, so a clean EOF or shutdown becomes 3; 1 and 2 keep their precedence.
        if (channelBroken && code == 0) code = 3;
        return code;
    }
};
void standardError(const std::string& text) {
    std::string line = "chandra-worker: " + bounded(text) + "\n"; DWORD wrote = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), line.data(), DWORD(line.size()), &wrote, nullptr);
}
void fatalEvent(const std::string& stage, const std::string& error, const std::string& selection) {
    std::string text = Json{{"schema", eventSchema}, {"seq", 0}, {"event", "fatal"}, {"stage", stage}, {"error", bounded(error)}, {"device_created", false}, {"model_loaded", false},
        {"gemv_b1_selection", selection.empty() ? Json(nullptr) : Json(selection)}}.dump() + "\n";
    DWORD wrote = 0; WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text.data(), DWORD(text.size()), &wrote, nullptr); standardError(stage + ": " + error);
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    std::string selection;
    try {
        // The process-scoped selector was captured before main; an invalid value refuses here, before any Device or model.
        selection = chandra::dc::experimental::gemvB1Selection();
        Options o = options(argc, argv);
        if (!o.execute && !o.plan) {
            std::string text = Json{{"schema", eventSchema}, {"seq", 0}, {"event", "inactive"}, {"device_created", false}, {"model_loaded", false}, {"gemv_b1_selection", selection},
                {"detail", "Inactive by default: --plan authenticates and forecasts requests on the CPU; --execute also creates the Device and imports the model once"}}.dump() + "\n";
            DWORD wrote = 0; WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text.data(), DWORD(text.size()), &wrote, nullptr); return 0;
        }
        for (const auto* p : {&o.model, &o.shaders}) require(!p->empty() && p->is_absolute() && std::filesystem::is_directory(*p), "Explicit absolute existing model and shader directories required");
        plainDirectory(o.inputRoot, "Input root"); plainDirectory(o.outputRoot, "Output root");
        // Both roots are plain, so canonical forms only normalize case/short names for this comparison.
        require(!nested(std::filesystem::canonical(o.inputRoot), std::filesystem::canonical(o.outputRoot)), "Input and output roots must be distinct and not nested");
        require(!o.pci.empty() && !o.luid.empty(), "Explicit PCI and LUID required; the Device verifies both before creation");
        Json job = observeJob();
        if (o.execute) {
            require(o.leaseMilliseconds != 0, "Execution requires an explicit --lease-ms parent lease");
            require(jobAdmitsExecution(job), "Execution requires an external kill-on-close Job with a finite process or job memory limit and no breakaway");
        }
        auto worker = std::make_shared<Worker>(o, selection, job);
        if (!worker->start()) return worker->serve(); // Releases whatever startup created, then reports exit.
        std::thread reader = worker->launchReader(worker);
        int code = 1;
        try { code = worker->serve(); } catch (...) { reader.detach(); throw; }
        if (worker->readerStopped()) reader.join(); else reader.detach(); // A read that could not be canceled ends with the process.
        return code;
    } catch (const std::exception& e) {
        fatalEvent("configuration", e.what(), selection); return 1;
    }
}
