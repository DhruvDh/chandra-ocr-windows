// New code, MPL-2.0. Resident native Chandra worker: one Device and one authenticated complete
// ModelWeights per process, B1 requests executed serially on the Device-owning thread through the
// same request path as chandra-inference.exe (inference_core.cpp), over a versioned, bounded
// newline-JSON stdin/stdout protocol. Inactive by default. No batching, concurrent GPU execution,
// model reload, engine/CPU fallback or request replay. A bounded stdin reader thread performs
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
    bool execute = false, plan = false; uint32_t leaseMilliseconds = 0;
    std::filesystem::path model, shaders, inputRoot, outputRoot; std::string pci, luid;
};
Options options(int argc, wchar_t** argv) {
    Options o; std::set<std::wstring> seen;
    for (int i = 1; i < argc; ++i) {
        std::wstring name = argv[i]; require(seen.insert(name).second, "Duplicate worker option");
        if (name == L"--execute") { o.execute = true; continue; }
        if (name == L"--plan") { o.plan = true; continue; }
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

struct Canceled : std::runtime_error {
    std::string boundary;
    explicit Canceled(std::string b) : std::runtime_error("Request canceled at safe boundary " + b), boundary(std::move(b)) {}
};
struct Request {
    std::string id; std::filesystem::path manifest, output; std::string manifestSha, outputName; uint32_t cap = 0; uint64_t index = 0;
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
    }
    // Startup: Device and the complete authenticated model exist only in execute mode, created on this thread.
    bool start() {
        try {
            if (options.execute) {
                device = std::make_unique<Device>(options.shaders.wstring(), options.pci, options.luid);
                identity = Json::parse(device->identityJson());
                weights = std::make_unique<ModelWeights>(*device, options.model.wstring());
                provenance = Json::parse(weights->provenanceJson());
                require(provenance.at("full_graph_requested") == true && provenance.at("omitted_graph_tensors").empty() && provenance.at("tie_byte_equality") == true, "Full original graph and authenticated tied output head required");
                residentBytes = device->trackedBufferBytes();
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> guard(lock); emitLocked({{"event", "fatal"}, {"stage", "startup"}, {"error", bounded(e.what())}, {"device_created", device != nullptr}, {"model_loaded", weights != nullptr}});
            poisoned = true; closing = true; closeReason = "startup_failed"; return false;
        }
        std::lock_guard<std::mutex> guard(lock);
        deviceLive = device != nullptr; modelLive = weights != nullptr;
        if (options.leaseMilliseconds) renewLeaseLocked();
        Json model = {{"alias", "chandra"}, {"revision", modelRevision}, {"context_limit", contextLimit}, {"normal_output_allowance", normalOutputLimit}, {"caller_stop_token_ids", Json(callerStops)}};
        Json resident = nullptr;
        if (weights) resident = {{"revision", provenance.value("revision", Json(nullptr))}, {"model_sha256", provenance.value("model_sha256", Json(nullptr))},
            {"config_sha256", provenance.value("config_sha256", Json(nullptr))}, {"uploaded_storage_bytes", provenance.value("uploaded_storage_bytes", Json(nullptr))},
            {"tie_byte_equality", true}, {"full_graph_requested", true}, {"provenance_sha256", sha256Text(provenance.dump())}, {"resident_tracked_bytes", residentBytes}, {"imports", 1}};
        emitLocked({{"event", "ready"}, {"mode", options.execute ? "execute" : "plan"}, {"accepting", true},
            {"protocol", {{"request_schema", requestSchema}, {"max_line_bytes", maxLineBytes}, {"max_event_bytes", maxEventBytes}, {"active_slots", 1}, {"waiting_slots", 1},
                {"max_lifetime_requests", maxLifetimeRequests}, {"max_protocol_errors", maxProtocolErrors},
                {"execution", "Serial B1 requests on the Device-owning thread; no batching or concurrent GPU execution"}}},
            {"model", model}, {"resident_model", resident}, {"device_identity", identity}, {"device_created", device != nullptr}, {"model_loaded", weights != nullptr},
            {"gemv_b1_selection", gemvSelection}, {"job", job}, {"lease_ms", options.leaseMilliseconds ? Json(options.leaseMilliseconds) : Json(nullptr)},
            {"obligations", obligations}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}});
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
            run(*next);
        }
        return finish();
    }
private:
    Options options; std::string gemvSelection; Json job; HANDLE output = INVALID_HANDLE_VALUE;
    std::unique_ptr<Device> device; std::unique_ptr<ModelWeights> weights; Json identity = nullptr, provenance = nullptr; uint64_t residentBytes = 0;
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
            if (kind == "submit") rejectLocked(submittedId, "unknown_field", "Unknown field " + it.key());
            else if (kind == "cancel") emitLocked({{"event", "cancel_rejected"}, {"submitted_id", submittedId}, {"reason", "unknown_field"}, {"detail", bounded("Unknown field " + it.key())}});
            else protocolErrorLocked("unknown_field", "Unknown field " + it.key());
            return;
        }
        if (kind == "status") { emitLocked(status()); return; }
        if (kind == "lease") return;
        if (kind == "shutdown") { closeLocked("shutdown_requested"); return; }
        if (kind == "cancel") { cancelLocked(message, submittedId); return; }
        submitLocked(message, submittedId);
    }
    void cancelLocked(const Json& message, const Json& submittedId) {
        auto refuse = [&](const char* reason) { emitLocked({{"event", "cancel_rejected"}, {"submitted_id", submittedId}, {"reason", reason}}); };
        if (submittedId.is_null()) { refuse("invalid_id"); return; }
        const std::string id = message.at("id").get<std::string>();
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
    Json status() {
        auto describe = [&](const std::shared_ptr<Request>& r) -> Json {
            if (!r) return nullptr;
            return {{"id", r->id}, {"started", r->started}, {"phase", r->phase.empty() ? Json(nullptr) : Json(r->phase)}, {"cancel_requested", r->cancelRequested}};
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
            try { weights.reset(); { std::lock_guard<std::mutex> guard(lock); modelLive = false; } release["model_released"] = true; device->drain(); release["drained"] = true; }
            catch (const std::exception& e) { clean = false; release["drained"] = false; release["drain_error"] = bounded(e.what()); }
            try { uint64_t tracked = device->trackedBufferBytes(); release["tracked_buffer_bytes_after_release"] = tracked; clean = clean && tracked == 0; }
            catch (const std::exception& e) { clean = false; release["tracking_error"] = bounded(e.what()); }
            device.reset(); release["device_destroyed"] = true; std::lock_guard<std::mutex> guard(lock); deviceLive = false;
        }
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
