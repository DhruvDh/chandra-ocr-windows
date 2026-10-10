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
#include "import_row_source.h"
#include "import_row_early.h"
#include "import_row_watch.h"
#include "shard1_boundary_watch.h"
#include "final_import_tail_audit.h"
#include "final_import_arithmetic_gate.h"
#include "parallel32_opt_in.h"
#include "padded32_opt_in.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::dc::experimental { const char* gemvB1Selection(); }

namespace {
using namespace chandra::dc;
using namespace chandra::dc::inference;
struct Options {
    bool execute = false, forecast = false, diagnosticDump = false, headRowAudit = false, headRowDirectCopy = false, importRowSource = false, importRowEarly = false, importRowWatch = false, importRowWatchArithmetic = false, headShardTailAudit = false, shard1BoundaryWatch = false; uint32_t diagnosticCap = 0;
    bool finalImportTailAudit=false,weightSrvOnly=false,finalImportTailArithmetic=false;WeightImportAPI weightImportAPI=WeightImportAPI::updateSubresource;
    bool finalImportFullPage=false,experimentalParallel32=false,experimentalPadded32=false;
    WeightStorageExperiment weightStorageExperiment = WeightStorageExperiment::exact;
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
        if (name == L"--head-row-direct-copy") { o.headRowDirectCopy = true; continue; }
        if (name == L"--import-row-source-audit") { o.importRowSource = true; continue; }
        if (name == L"--import-row-early-copy") { o.importRowEarly = true; continue; }
        if (name == L"--import-final-head-tail-before-arithmetic") { o.finalImportTailArithmetic=true;continue; }
        if (name == L"--import-final-head-tail-full-page") { o.finalImportFullPage=true;continue; }
        if (name == L"--experimental-parallel32") { o.experimentalParallel32=true;continue; }
        if (name == L"--experimental-padded32") { o.experimentalPadded32=true;continue; }
        if (name == L"--weight-srv-only") { o.weightSrvOnly=true;continue; }
        if (name == L"--import-final-head-tail-audit") { o.finalImportTailAudit=true;continue; }
        if (name == L"--import-shard1-boundary-watch") { o.shard1BoundaryWatch = true; continue; }
        if (name == L"--import-row-watch") { o.importRowWatch = true; continue; }
        if (name == L"--import-head-shard-tail-audit") { o.headShardTailAudit = true; continue; }
        if (name == L"--import-row-watch-before-arithmetic") { o.importRowWatchArithmetic = true; continue; }
        require(i + 1 < argc, "CLI option value is missing"); std::wstring value = argv[++i];
        if (name == L"--model-dir") o.model = value;
        else if (name == L"--shader-root") o.shaders = value;
        else if (name == L"--input-manifest") o.manifest = value;
        else if (name == L"--input-sha256") o.manifestSha = digestText(utf8(value));
        else if (name == L"--output-dir") o.output = value;
        else if (name == L"--pci") o.pci = utf8(value);
        else if (name == L"--luid") o.luid = utf8(value);
        else if (name == L"--weight-import-api") o.weightImportAPI=weight_import::parse(utf8(value));
        else if (name == L"--weight-storage-experiment") o.weightStorageExperiment = weight_storage::parse(utf8(value));
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
    require(!o.headRowDirectCopy || o.headRowAudit, "--head-row-direct-copy requires the bounded --head-row-audit diagnostic");
    require(!o.importRowSource || ((o.headRowDirectCopy || o.importRowWatch) && o.weightStorageExperiment == WeightStorageExperiment::exact), "--import-row-source-audit requires --head-row-direct-copy or --import-row-watch and exact weight allocation");
    require(!o.importRowEarly || o.importRowSource, "--import-row-early-copy requires --import-row-source-audit");
    require(!o.importRowWatch || ((o.execute || (o.forecast && o.headShardTailAudit)) && o.importRowSource && !o.importRowEarly && !o.headRowAudit && !o.headRowDirectCopy && !o.diagnosticDump && !o.diagnosticCap && o.diagnosticPlan.empty() && o.weightStorageExperiment == WeightStorageExperiment::exact), "--import-row-watch is import-only: execute/source audit/exact required; early/head/operand/generation diagnostics excluded");
    require(!o.headShardTailAudit || (o.importRowWatch && o.importRowSource && !o.importRowWatchArithmetic && o.manifestSha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae"), "Head-shard tail audit requires exact original native0 and import-only watcher/source flags; arithmetic excluded");
    require(!o.importRowWatchArithmetic || (o.execute && !o.importRowWatch && !o.importRowEarly && o.importRowSource && o.headRowAudit && o.headRowDirectCopy && o.diagnosticCap==2 && o.diagnosticDump && !o.diagnosticPlan.empty() &&
        o.weightStorageExperiment==WeightStorageExperiment::exact && o.manifestSha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae" && o.diagnosticPlanSha=="b164ae5bac36d0892e96cd68092a5139bacc5fe8cf39bc77e28126cd7a614f82"),
        "Watched arithmetic requires exact original native0/cap2/source/head/direct/fixed operand plan; separate early/import-only flags excluded");
    require(!o.shard1BoundaryWatch || (!o.importRowWatch && !o.importRowWatchArithmetic && !o.headShardTailAudit && !o.importRowSource && !o.importRowEarly && !o.headRowAudit && !o.headRowDirectCopy && !o.diagnosticDump && !o.diagnosticCap && o.diagnosticPlan.empty() &&
        o.weightStorageExperiment==WeightStorageExperiment::exact && o.manifestSha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae"), "Shard1 boundary watcher is separate exact original native0 import-only; other observers/arithmetic excluded");
    require(o.weightImportAPI==WeightImportAPI::updateSubresource || o.finalImportTailAudit,"DEFAULT initial data requires explicit final-head-tail experiment");
    require(!o.finalImportTailAudit || o.finalImportTailArithmetic || o.finalImportFullPage || (!o.shard1BoundaryWatch&&!o.importRowWatch&&!o.importRowWatchArithmetic&&!o.headShardTailAudit&&!o.importRowSource&&!o.importRowEarly&&!o.headRowAudit&&!o.headRowDirectCopy&&!o.diagnosticDump&&!o.diagnosticCap&&o.diagnosticPlan.empty()&&o.weightStorageExperiment==WeightStorageExperiment::exact&&o.manifestSha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae"),"Final-head-tail API experiment is exact original native0 import-only; old observers/arithmetic excluded");
    require(weight_srv_only::admitted(o.weightSrvOnly,o.finalImportTailAudit,o.weightStorageExperiment==WeightStorageExperiment::exact,o.weightImportAPI==WeightImportAPI::defaultInitial,seen.count(L"--weight-import-api")!=0),"--weight-srv-only requires final-head-tail audit, exact allocation and explicit --weight-import-api default-initial");
    require(!o.finalImportTailArithmetic || (seen.count(L"--weight-import-api")==1&&o.execute&&o.finalImportTailAudit&&o.weightSrvOnly&&o.weightImportAPI==WeightImportAPI::defaultInitial&&!o.shard1BoundaryWatch&&!o.importRowWatch&&!o.importRowWatchArithmetic&&!o.headShardTailAudit&&!o.importRowSource&&!o.importRowEarly&&o.headRowAudit&&o.headRowDirectCopy&&o.diagnosticCap==2&&o.diagnosticDump&&!o.diagnosticPlan.empty()&&o.weightStorageExperiment==WeightStorageExperiment::exact&&o.manifestSha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae"&&o.diagnosticPlanSha=="b164ae5bac36d0892e96cd68092a5139bacc5fe8cf39bc77e28126cd7a614f82"),"SRV-only final-tail arithmetic requires explicit DEFAULT initial/exact/native0/cap2/head/direct/fixed operand plan; historical upload-source/early/watch flags excluded");
    require(!o.finalImportFullPage || (seen.count(L"--weight-import-api")==1&&o.execute&&!o.forecast&&o.finalImportTailAudit&&o.weightSrvOnly&&o.weightImportAPI==WeightImportAPI::defaultInitial&&o.weightStorageExperiment==WeightStorageExperiment::exact&&!o.finalImportTailArithmetic&&!o.shard1BoundaryWatch&&!o.importRowWatch&&!o.importRowWatchArithmetic&&!o.headShardTailAudit&&!o.importRowSource&&!o.importRowEarly&&!o.headRowAudit&&!o.headRowDirectCopy&&!o.diagnosticDump&&!o.diagnosticCap&&o.diagnosticPlan.empty()&&o.diagnosticPlanSha.empty()&&(o.manifestSha=="0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae"||o.manifestSha=="aab97d615f7e3181b9a3ef814f8784030f17fb186a31964df0a2a6063512e8d1"||o.manifestSha=="62c362251b128392f3cb3fcff6e8e25531b585038d12391ecaa1a153d6979081"||o.manifestSha=="df67a54b1d43b83f178b8b7c6627480d114bf07e68ddee2ba08b137f34307c04")),"SRV-only full page requires explicit DEFAULT initial/exact/original-corpus/execute/final-tail/SRV-only; cap2, forecast and diagnostic/observer flags excluded");
    require(!o.experimentalParallel32||(o.finalImportFullPage!=o.finalImportTailArithmetic),"--experimental-parallel32 requires exactly one unchanged SRV-only DEFAULT-initial full-page or fixed cap2 route; other paths excluded");
    require(!o.experimentalPadded32||(o.finalImportFullPage!=o.finalImportTailArithmetic),"--experimental-padded32 requires exactly one unchanged SRV-only DEFAULT-initial full-page or fixed cap2 route; other paths excluded");
    if(o.importRowWatchArithmetic||o.finalImportTailArithmetic||o.finalImportFullPage){
        const char* mode=nullptr;
#ifdef _MSC_VER
        struct Free {void operator()(char* p) const noexcept {std::free(p);}};
        char* value=nullptr;size_t length=0;
        const auto status=_dupenv_s(&value,&length,"CHANDRA_EXPERIMENTAL_GEMV_B1");
        const std::unique_ptr<char,Free> owned(value);
        require(status==0,"Watched arithmetic ordered B1 environment could not be read");
        mode=owned.get();
#else
        mode=std::getenv("CHANDRA_EXPERIMENTAL_GEMV_B1");
#endif
        if(o.experimentalParallel32)
            require(chandra::parallel32_opt_in::admitted(true,o.finalImportFullPage,o.finalImportTailArithmetic,mode,experimental::gemvB1Selection()),
                    "Experimental arithmetic requires explicit CLI opt-in and matching process-start/live parallel32 selector; no ordered fallback");
        else require(mode && std::strncmp(mode,"ordered",8)==0,"Watched arithmetic requires the existing explicit ordered B1 environment");
    }
    require(std::filesystem::is_directory(o.model) && std::filesystem::is_directory(o.shaders), "Explicit model and shader directories must exist");
    require(!std::filesystem::exists(o.output), "Fresh output directory required; existing output is never overwritten");
    if(o.experimentalPadded32){
        require(std::strcmp(experimental::gemvB1Selection(),o.experimentalParallel32?"parallel32":"ordered")==0,
                "Padded32 requires the separately admitted unchanged captured B1 selection");
        experimental::enableGemmPadded32();
    }
    return o;
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
// Called only after the explicit selector admission, before Device/model work. Hashes the selected
// shader source snapshot; root separately binds the compiled producer and its immutable custody.
Json parallel32Selection(const Options& o) {
    const char* captured=experimental::gemvB1Selection();
    require(o.experimentalParallel32&&std::strcmp(captured,"parallel32")==0,"Experimental arithmetic selection changed before metadata");
    const auto digest=fileSha256(o.shaders/L"runtime"/L"linear_gemv_parallel32.hlsl",64*1024);
    require(digest==chandra::parallel32_opt_in::shaderSha256,"Experimental arithmetic requires the byte-identical accepted parallel32 shader");
    return {{"selector",captured},{"route","parallel32"},{"explicit_cli_opt_in",o.experimentalParallel32},{"cli_flag","--experimental-parallel32"},
        {"selector_environment_variable","CHANDRA_EXPERIMENTAL_GEMV_B1"},{"admitted_environment_value","parallel32"},
        {"shader","runtime/linear_gemv_parallel32.hlsl"},{"shader_source_sha256",digest},
        {"arithmetic_change",true},{"lane_terms","ascending k mod 32 across TILE512"},{"tree_offsets",{16,8,4,2,1}},
        {"bias","one original FP32 addition after tree"},{"output_cast","original optional BF16 RNE; NaNs quieted/truncated"},
        {"scope","process-start operator selector and shader-source snapshot; root separately binds compilation and custody"},
        {"trained_numerics_accepted",false},{"complete_OCR_or_PPS_accepted",false}};
}
// Selected and source-pinned before Device/model work. Unsupported shapes retain the original shader.
Json padded32Selection(const Options& o) {
    require(o.experimentalPadded32&&experimental::gemmPadded32Enabled(),"Explicit padded32 CLI latch required before metadata");
    const char* b1=experimental::gemvB1Selection();
    require(std::strcmp(b1,o.experimentalParallel32?"parallel32":"ordered")==0,"Padded32 B1 selection differs from its separate admission");
    const auto digest=fileSha256(o.shaders/L"runtime"/L"linear_gemm_padded32.hlsl",64*1024);
    const auto fallback=fileSha256(o.shaders/L"runtime"/L"linear.hlsl",64*1024);
    require(digest==chandra::padded32_opt_in::shaderSha256&&fallback==chandra::padded32_opt_in::fallbackSha256,
            "Padded32 and fallback require the exact accepted and baseline shader sources");
    return {{"selector","padded32"},{"explicit_cli_opt_in",true},{"cli_flag","--experimental-padded32"},
        {"shader",chandra::padded32_opt_in::shader},{"shader_source_sha256",digest},
        {"supported_dispatch_shape","BF16 weights; 2<=chunk_rows<=32; 32<=K<=9216; K divisible by 32"},
        {"fallback_shader",chandra::padded32_opt_in::fallbackShader},{"fallback_source_sha256",fallback},
        {"fallback_policy","Other multirow dispatch shapes, including one-row tails, retain linear.hlsl"},
        {"B1_selector",b1},{"B1_selection_changed",false},{"scalar_operation_order_changed",false},
        {"arithmetic","Original ascending-k precise FP32 product then addition, original bias and optional BF16 RNE"},
        {"scope","CLI-latched operator selection and shader-source snapshot before model work; dispatch counts are reported separately"},
        {"trained_numerics_accepted",false},{"complete_OCR_or_PPS_accepted",false}};
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
    Json producer = {{"kind", "directcompute_native"}, {"executable_sha256", fileSha256(executable, 256ull * 1024 * 1024)},
        {"shader_root", {{"files", files.size()}, {"tree_sha256", tree.finish()}, {"canonical", "sorted generic relative path, tab, file SHA-256, newline"}}},
        {"scope", "Executable image and HLSL files present at diagnostic start; shaders compile lazily from this root"}};
    if(o.experimentalParallel32)producer["experimental_gemv_B1"]=parallel32Selection(o);
    if(o.experimentalPadded32)producer["experimental_gemm_multirow"]=padded32Selection(o);
    return producer;
}
Json diagnosticCommitments(const Input& in, const Options& o, const DiagnosticRequest& d) {
    Json commitments = {{"model_revision", modelRevision}, {"input_manifest_sha256", o.manifestSha}, {"plan_file", d.planFile},
        {"input", {{"prompt_tokens", in.ids.size()}, {"patch_rows", in.patchRows}, {"merged_rows", in.visionForecast.mergedRows},
            {"image_grid_thw", in.manifest.at("image_grid_thw")}, {"next_decode_position", in.maximumPosition + 1}, {"stop_token_ids", in.stopIds},
            {"generation_limit", d.geometry.generationLimit}, {"diagnostic_token_cap", o.diagnosticCap ? Json(o.diagnosticCap) : Json(nullptr)}}}};
    if(o.experimentalParallel32)commitments["experimental_gemv_B1"]=parallel32Selection(o);
    if(o.experimentalPadded32)commitments["experimental_gemm_multirow"]=padded32Selection(o);
    return commitments;
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
    if (o.weightStorageExperiment != WeightStorageExperiment::exact) result["weight_storage_experiment"] = Json::parse(weightStorageForecastJson(o.weightStorageExperiment));
    if (diagnostic) result["diagnostics"] = {{"requested", true}, {"dump_written", false}, {"plan", diagnostic->plan.resolved}, {"plan_sha256", diagnostic->plan.sha256}, {"plan_file", diagnostic->planFile}};
    if(o.headShardTailAudit)result["head_shard_tail_audit"]=import_row::HeadTailObserver::forecast();
    if(o.shard1BoundaryWatch)result["shard1_boundary_watch"]=import_row::Shard1Observer::forecast();
    if(o.finalImportTailAudit)result["final_import_tail_audit"]=import_row::FinalTailObserver::forecast(o.weightImportAPI,o.weightSrvOnly);
    return result;
}
// Optional bridge only: same watcher and fixed leaves; the original graph remains live after the import gate.
std::unique_ptr<import_row::WatchObserver> arithmeticImportWatch(const Options& o,std::unique_ptr<NewOutput>& journal){
    const auto directory=o.output/L"import-row-watch";require(CreateDirectoryW(directory.c_str(),nullptr),"Fresh arithmetic import watch directory required");
    journal=std::make_unique<NewOutput>(directory/L"progress.jsonl");
    return std::make_unique<import_row::WatchObserver>([directory](const std::string& name,const void* data,size_t bytes){
        require((name=="plan.json"&&bytes>0&&bytes<=import_row::watch::maximumPlanBytes)||((name==import_row::watchInitialFile||name==import_row::watchGoodFile||name==import_row::watchBadFile)&&bytes==import_row::rowBytes),"Fixed bounded arithmetic watch filename/extent required");
        DiagnosticFile file(directory/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
    },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&journal](const Json& record){journal->line(record);});
}
void requireCleanImportForArithmetic(const import_row::WatchObserver& watch){
    const auto state=watch.report();const auto& c=state.at("counts");
    require(watch.completed() && state.at("outcome")=="no_fault_completed_import" && state.at("same_storage_after_import")==true &&
        c.at("operations_admitted")==732 && c.at("original_allocations_completed")==732 && c.at("original_allocation_bytes")==9078531072ull &&
        c.at("original_upload_calls_admitted")==732 && c.at("original_upload_returns_observed")==732 && c.at("original_upload_drains_completed")==732 && c.at("completed_original_upload_bytes")==9078531072ull &&
        c.at("completed_samples")==733 && c.at("direct_read_call_admissions")==733 && c.at("current_operation_phase")==0,
        "Arithmetic blocked: complete 733-sample no-fault import of the same original Storage is required");
}
Json execute(const Input& in, const Options& o, const DiagnosticRequest* diagnostic) {
    auto start = Clock::now(); Json phases = Json::array(), memories = Json::array(); std::vector<uint32_t> generated;
    Json report = executionReport(in, o.manifestSha, o.diagnosticCap);
    if(o.importRowWatchArithmetic){report["schema"]="chandra.directcompute.watched-import-cap2-result.v1";report["mode"]="watched_import_then_original_cap2_arithmetic";report["import_watch_before_arithmetic_requested"]=true;report["clean_import_gate_completed"]=false;report["arithmetic_call_admitted"]=false;report["arithmetic_return_observed"]=false;report["watched_model_reused_for_arithmetic"]=false;report["ordered_B1_environment"]="ordered";}
    if (diagnostic) report["diagnostic_dump_requested"] = true;
    // A requested head audit reads audit_incomplete until its own report replaces this entry, so a run
    // failing before or during the audit never reads as clean. Every requested prediction must be audited.
    if (o.headRowAudit) report["head_row_audit"] = head_audit::notStarted(o.diagnosticCap);
    if (o.headRowDirectCopy) report["head_row_direct_copy_requested"] = true;
    if (o.importRowSource) report["import_row_source_audit"] = {{"schema", "chandra.directcompute.import-row-source.v1"}, {"requested", true}, {"complete", false}};
    if (o.importRowEarly) report["import_row_early_copy"] = {{"schema", "chandra.directcompute.import-row-early-copy.v1"}, {"requested", true}, {"complete", false}};
    std::unique_ptr<import_row::Observer> sourceObserver;
    std::unique_ptr<NewOutput> sourceProgress;
    std::unique_ptr<import_row::EarlyObserver> earlyObserver;
    std::unique_ptr<NewOutput> earlyProgress;
    std::unique_ptr<import_row::WatchObserver> arithmeticWatch;std::unique_ptr<NewOutput> arithmeticWatchProgress;
    std::unique_ptr<Device> device; std::unique_ptr<diagnostics::Recorder> recorder; auto stamp = Clock::now();
    auto phase = [&](const char* name) { auto now = Clock::now(); phases.push_back({{"name", name}, {"wall_seconds", std::chrono::duration<double>(now - stamp).count()}}); stamp = now; if (device) memories.push_back({{"phase", name}, {"observed", Json::parse(device->memoryJson())}, {"host_process", hostMemory()}}); if (recorder) recorder->boundary(name); };
    try {
        if (o.importRowSource) {
            const auto directory = o.output / L"upload-source-audit";
            require(CreateDirectoryW(directory.c_str(), nullptr), "Fresh upload source directory creation failed");
            sourceProgress = std::make_unique<NewOutput>(directory / L"progress.jsonl");
            sourceObserver = std::make_unique<import_row::Observer>(
                [directory](const std::string& name, const void* data, size_t bytes) {
                    require(diagnostics::safeName(name) && bytes == import_row::rowBytes, "Fixed bounded upload source filename/extent required");
                    DiagnosticFile file(directory / std::filesystem::u8path(name)); file.write(data, bytes); file.flush();
                }, [](const void* data, size_t bytes) { return diagnostics::sha256(data, bytes); },
                [&sourceProgress](const Json& record) { sourceProgress->line(record); });
        }
        if (o.importRowEarly) {
            const auto directory = o.output / L"upload-early-audit";
            require(CreateDirectoryW(directory.c_str(), nullptr), "Fresh early upload audit directory creation failed");
            earlyProgress = std::make_unique<NewOutput>(directory / L"progress.jsonl");
            earlyObserver = std::make_unique<import_row::EarlyObserver>(
                [directory](const std::string& name, const void* data, size_t bytes) {
                    require(name == import_row::earlyFile && bytes == import_row::rowBytes, "Fixed early upload audit filename/extent required");
                    DiagnosticFile file(directory / std::filesystem::u8path(name)); file.write(data, bytes); file.flush();
                }, [](const void* data, size_t bytes) { return diagnostics::sha256(data, bytes); },
                [&earlyProgress](const Json& record) { earlyProgress->line(record); });
        }
        if(o.importRowWatchArithmetic)arithmeticWatch=arithmeticImportWatch(o,arithmeticWatchProgress);
        // Readbacks happen synchronously in observers on this Device thread, after the graph's own drains.
        if (diagnostic) recorder = std::make_unique<diagnostics::Recorder>(diagnostic->plan, diagnostic->geometry, std::make_unique<DiagnosticDirectory>(o.output / L"diagnostics"),
            [&device](const Buffer& b) { return device->readFloats(b); }, diagnosticCommitments(in, o, *diagnostic), diagnosticProducer(o));
        device = std::make_unique<Device>(o.shaders.wstring(), o.pci, o.luid); phase("device_creation"); report["device_identity"] = Json::parse(device->identityJson());
        {
            // Empty selection is the complete normal inference graph. No
            // diagnostic subset or CPU-resident tensor copy is substituted.
            // --head-row-audit holds this read-only, non-write-shared handle across the importer's whole-file
            // SHA-256 and every later audit read, so no writer can change the authenticated bytes in between.
            std::unique_ptr<Handle> auditSource,auditConfig;
            if (o.headRowAudit) auditSource = std::make_unique<Handle>(CreateFileW((o.model / L"model.safetensors").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
            if(o.importRowWatchArithmetic)auditConfig=std::make_unique<Handle>(CreateFileW((o.model/L"config.json").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
            ModelWeights weights(*device, o.model.wstring(), 12ull * 1024 * 1024 * 1024, {}, o.weightStorageExperiment, sourceObserver.get(), earlyObserver.get(), arithmeticWatch.get());
            if(arithmeticWatch){
                arithmeticWatch->afterImport(*device,weights.at("model.language_model.embed_tokens.weight").shards.at(0),weights.at("lm_head.weight").shards.at(0));
                requireCleanImportForArithmetic(*arithmeticWatch);report["clean_import_gate_completed"]=true;
                // Only the observer's extra reference is released: weights and Device still own the same imported graph.
                arithmeticWatch->releaseTarget();report["import_row_watch"]=arithmeticWatch->report();report["model_buffers_retained_after_local_watch_reference_release"]=true;
            }
            phase("authenticated_model_import_upload"); report["model_provenance"] = Json::parse(weights.provenanceJson());
            require(report["model_provenance"].at("full_graph_requested") == true && report["model_provenance"].at("omitted_graph_tensors").empty() && report["model_provenance"].at("tie_byte_equality") == true, "Full original graph and authenticated tied output head required");
            if (recorder) recorder->authenticatedModel(report["model_provenance"], report["device_identity"]);
            std::unique_ptr<head_audit::Auditor> audit;
            if (auditSource) audit = headAudit(*device, weights, report["model_provenance"], *auditSource, o.output / L"head-row-audit", o.diagnosticCap);
            if (audit) {
                try { audit->afterImport(o.headRowDirectCopy); }
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
            if(arithmeticWatch){requireCleanImportForArithmetic(*arithmeticWatch);report["arithmetic_call_admitted"]=true;report["watched_model_reused_for_arithmetic"]=true;}
            try { generate(*device, weights, in, o.diagnosticCap, o.output, report, generated, hooks);if(arithmeticWatch)report["arithmetic_return_observed"]=true; }
            catch (const std::exception& e) { if (audit) report["head_row_audit"] = head_audit::finishAfterFailure(*audit, e.what()); throw; }
            catch (...) { if (audit) report["head_row_audit"] = head_audit::finishAfterFailure(*audit, "non-standard exception"); throw; }
            if (audit) report["head_row_audit"] = audit->finish(head_audit::RunEnd::completed);
        }
        device->drain(); phase("model_buffer_release_and_drain");
        require(device->trackedBufferBytes() == 0, "Owned graph buffers did not retire to zero"); report["owned_buffer_zero"] = true;
        report["passed"] = true; report["passed_scope"] = "Executable graph returned finite greedy tokens and retired owned buffers; numerical/full-OCR qualification is pending";
    } catch (const std::exception& e) {
        report["passed"] = false; report["error"] = e.what(); report["owned_worker_retirement_required"] = true;
        if (sourceObserver && !sourceObserver->completed()) sourceObserver->failure(e.what());
        if (earlyObserver && !earlyObserver->completed()) earlyObserver->failure(e.what());
        if(arithmeticWatch){if(!arithmeticWatch->completed())arithmeticWatch->failure(e.what());arithmeticWatch->releaseTarget();}
        if (device) { try { device->drain(); report["final_memory"] = Json::parse(device->memoryJson()); report["final_drain_completed"] = true; } catch (const std::exception& drain) { report["final_drain_completed"] = false; report["final_drain_error"] = drain.what(); } }
    }
    if (sourceObserver) report["import_row_source_audit"] = sourceObserver->report();
    if (earlyObserver) report["import_row_early_copy"] = earlyObserver->report();
    if(arithmeticWatch){report["import_row_watch"]=arithmeticWatch->report();report["import_watch_completion_scope"]="One selected original row at 733 import cutoffs; local watcher release/source-map scope return are not model/GPU/Job retirement";}
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
// Separate typed diagnostic: no generate(), head shader audit, vision/text arithmetic or operands.
Json executeImportWatch(const Input& in,const Options& o){
    Json report={{"schema","chandra.directcompute.import-row-watch-result.v1"},{"diagnostic_type","import_only_original_weight_row_watch"},{"passed",false},
        {"diagnostic_completed",false},{"outcome","partial_or_refused"},{"actual_inference",false},{"complete_inference",false},{"qualified_OCR",false},{"qualified_full_graph",false},{"performance_claim",false},
        {"vision_execution_started",false},{"text_execution_started",false},{"generated_token_count",0},{"request_replayed",false},{"input_manifest_sha256",o.manifestSha},{"context_limit",contextLimit},{"normal_output_allowance",normalOutputLimit},
        {"prompt_tokens",in.ids.size()},{"model_import_completed",false},{"owned_buffer_zero",false},{"owned_worker_retirement_required",true}};
    if(o.headShardTailAudit){report["schema"]="chandra.directcompute.head-shard-tail-watch-result.v1";report["diagnostic_type"]="import_only_original_head_shard_tails";report["head_shard_tail_audit_requested"]=true;}
    std::unique_ptr<Device> device;std::unique_ptr<import_row::Observer> source;std::unique_ptr<import_row::WatchObserver> watch;std::unique_ptr<import_row::HeadTailObserver> tails;
    std::unique_ptr<NewOutput> sourceProgress,watchProgress,tailProgress;
    std::unique_ptr<Handle> heldModel,heldConfig;
    try{
        heldModel=std::make_unique<Handle>(CreateFileW((o.model/L"model.safetensors").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
        heldConfig=std::make_unique<Handle>(CreateFileW((o.model/L"config.json").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
        report["non_write_shared_model_and_config_handles_held"]=true;
        const auto sourceDir=o.output/L"upload-source-audit",watchDir=o.output/L"import-row-watch";
        require(CreateDirectoryW(sourceDir.c_str(),nullptr)&&CreateDirectoryW(watchDir.c_str(),nullptr),"Fresh import watch directories required");
        sourceProgress=std::make_unique<NewOutput>(sourceDir/L"progress.jsonl");watchProgress=std::make_unique<NewOutput>(watchDir/L"progress.jsonl");
        source=std::make_unique<import_row::Observer>([sourceDir](const std::string& name,const void* data,size_t bytes){
            require(diagnostics::safeName(name)&&bytes==import_row::rowBytes,"Fixed bounded CPU source output required");DiagnosticFile file(sourceDir/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
        },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&sourceProgress](const Json& record){sourceProgress->line(record);});
        if(o.headShardTailAudit){const auto tailDir=o.output/L"head-tail-audit";require(CreateDirectoryW(tailDir.c_str(),nullptr),"Fresh head-tail directory required");tailProgress=std::make_unique<NewOutput>(tailDir/L"progress.jsonl");
            tails=std::make_unique<import_row::HeadTailObserver>([tailDir](const std::string& name,const void* data,size_t bytes){
                require((name==import_row::tailSourceFile||name==import_row::tailGoodFile||name==import_row::tailBadFile)&&bytes==import_row::tails::rowBytes,"Fixed bounded tail filename/extent required");DiagnosticFile file(tailDir/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
            },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&tailProgress](const Json& record){tailProgress->line(record);});
        }
        watch=std::make_unique<import_row::WatchObserver>([watchDir](const std::string& name,const void* data,size_t bytes){
            require((name=="plan.json"&&bytes>0&&bytes<=import_row::watch::maximumPlanBytes)||((name==import_row::watchInitialFile||name==import_row::watchGoodFile||name==import_row::watchBadFile)&&bytes==import_row::rowBytes),"Fixed bounded watch filename/extent required");
            DiagnosticFile file(watchDir/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
        },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&watchProgress](const Json& record){watchProgress->line(record);},tails.get());
        device=std::make_unique<Device>(o.shaders.wstring(),o.pci,o.luid);report["device_identity"]=Json::parse(device->identityJson());
        {
            ModelWeights weights(*device,o.model.wstring(),12ull*1024*1024*1024,{},WeightStorageExperiment::exact,source.get(),nullptr,watch.get());
            report["model_import_completed"]=true;
            const auto provenance=Json::parse(weights.provenanceJson());Json summary={{"schema","chandra.directcompute.import-row-watch-model-provenance.v1"},{"full_ordered_upload_plan","import-row-watch/plan.json"}};
            for(const char* key:{"model","revision","model_bytes","model_sha256","config_bytes","config_sha256","tensor_count","full_graph_requested","uploaded_tensor_names","maximum_shard_bytes","maximum_tracked_weight_bytes","tracked_buffer_bytes_before","uploaded_storage_bytes","tracked_buffer_bytes_after","tie_byte_equality_performed","tie_byte_equality","mapping_lifetime","implicit_cast_or_dequantization","numerical_acceptance","full_model_qualified"})summary[key]=provenance.at(key);
            report["model_provenance"]=std::move(summary);
            watch->afterImport(*device,weights.at("model.language_model.embed_tokens.weight").shards.at(0),weights.at("lm_head.weight").shards.at(0));
            if(tails)tails->afterImport(*device,weights.at("model.language_model.embed_tokens.weight"),weights.at("lm_head.weight"),watch->report().at("counts"));
        }
    }catch(const import_row::HeadTailStop&){report["first_fault_stop_observed"]=true;
    }catch(const import_row::WatchStop&){
        // Typed first-fault completion; constructor unwinds without another import or arithmetic.
        report["first_fault_stop_observed"]=true;
    }catch(const std::exception& e){
        report["error"]=e.what();if(source&&!source->completed())source->failure(e.what());if(watch)watch->failure(e.what());if(tails)tails->failure(e.what());
    }catch(...){if(!tails)throw;report["error"]="Non-standard head-tail import exception";if(watch)watch->failure("Non-standard head-tail import exception");tails->failure("Non-standard head-tail import exception");}
    if(watch){watch->releaseTarget();report["import_row_watch"]=watch->report();report["diagnostic_completed"]=watch->completed();report["outcome"]=watch->report().at("outcome");}
    if(tails){tails->importStackClosed();tails->releaseTargets();report["head_shard_tail_audit"]=tails->report();report["diagnostic_completed"]=tails->completed();report["outcome"]=tails->report().at("outcome");}
    if(source)report["import_row_source_audit"]=source->report();
    // Independent disposal attempt follows every complete/partial observation; collection never substitutes for retirement.
    if(device){try{device->drain();report["final_drain_completed"]=true;report["final_memory"]=Json::parse(device->memoryJson());report["owned_buffer_zero"]=device->trackedBufferBytes()==0;}
        catch(const std::exception& e){report["final_drain_completed"]=false;report["final_drain_error"]=e.what();}}
    report["passed"]=report.at("diagnostic_completed")==true&&report.at("owned_buffer_zero")==true&&report.value("final_drain_completed",false)&&!report.contains("error");
    report["passed_scope"]="Typed diagnostic evidence completed and local buffer references drained; first_fault is an observed mismatch, no_fault is only observed cutoff equality; root retirement and numerical/OCR acceptance remain separate";
    heldConfig.reset();heldModel.reset();report["held_model_config_handles_released"]=true;
    return report;
}
// Separate explicit bridge: original final-tail API observer; no historical upload-source callbacks.
std::unique_ptr<import_row::FinalTailObserver> arithmeticFinalTailObserver(const Options& o,std::unique_ptr<NewOutput>& journal){
 const auto directory=o.output/L"final-import-tail-audit";require(CreateDirectoryW(directory.c_str(),nullptr),"Fresh bridge final-import-tail directory required");journal=std::make_unique<NewOutput>(directory/L"progress.jsonl");
 return std::make_unique<import_row::FinalTailObserver>(o.weightImportAPI,[directory](const std::string& name,const void* data,size_t bytes){
  require(((name==import_row::finalTailSourceFile||name==import_row::finalTailGoodFile)&&bytes>0&&bytes<=import_row::final_tails::rowArrayBytes&&bytes%import_row::tails::rowBytes==0)||(name==import_row::finalTailBadFile&&bytes==import_row::tails::rowBytes),"Fixed bridge final-tail payload name/extent required");
  DiagnosticFile file(directory/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
 },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&journal](const Json& record){journal->line(record);},o.weightSrvOnly);
}
// The unchanged final-tail gate is shared with resident worker startup.
using chandra::dc::inference::requireCleanFinalImportForArithmetic;

Json executeFinalTailArithmetic(const Input& in, const Options& o, const DiagnosticRequest* diagnostic) {
    auto start = Clock::now(); Json phases = Json::array(), memories = Json::array(); std::vector<uint32_t> generated;
    Json report = executionReport(in, o.manifestSha, o.diagnosticCap);
    require(diagnostic!=nullptr,"Final-tail arithmetic requires its authenticated fixed operand request");
    report["schema"]="chandra.directcompute.final-import-cap2-result.v1";report["mode"]="final_import_then_original_cap2_arithmetic";report["import_only"]=false;
    report["weight_srv_only_requested"]=true;report["plan"]=import_row::FinalTailObserver::forecast(o.weightImportAPI,true);report["final_import_before_arithmetic_requested"]=true;report["weight_import_api"]=weight_import::name(o.weightImportAPI);report["D3D11_usage"]="DEFAULT";report["ordered_B1_environment"]="ordered";
    if(o.experimentalParallel32){report["schema"]=chandra::parallel32_opt_in::cap2Schema;report["mode"]=chandra::parallel32_opt_in::cap2Mode;report["ordered_B1_environment"]="parallel32";report["experimental_gemv_B1"]=parallel32Selection(o);}
    if(o.experimentalPadded32){report["schema"]=chandra::padded32_opt_in::schema(false,o.experimentalParallel32);report["mode"]=chandra::padded32_opt_in::mode(false,o.experimentalParallel32);report["experimental_gemm_multirow"]=padded32Selection(o);}
    report["model_import_completed"]=false;report["clean_final_import_gate_completed"]=false;report["arithmetic_call_admitted"]=false;report["arithmetic_return_observed"]=false;report["initialized_model_reused_for_arithmetic"]=false;report["owned_buffer_zero"]=false;
    report["historical_import_row_source_audit_requested"]=false;report["historical_CPU_source_captures_created"]=false;report["numerical_qualification_accepted"]=false;
    report["import_source_evidence_scope"]="Unchanged ten final-tail snapshots from actual API submission pointers; not the three historical UpdateSubresource source captures";
    report["source_maps_released_scope"]="Constructor-return RAII view/handle scope, not separately checked Win32 unmap/CloseHandle success";
    report["full_model_provenance_file"]={{"schema","chandra.directcompute.final-import-cap2-provenance-binding.v1"},{"file","head-row-audit/model-provenance.json"},{"write_attempted",false},{"written_and_flushed",false},{"committed_bytes",0},{"sha256",nullptr}};
    if (diagnostic) report["diagnostic_dump_requested"] = true;
    // A requested head audit reads audit_incomplete until its own report replaces this entry, so a run
    // failing before or during the audit never reads as clean. Every requested prediction must be audited.
    if (o.headRowAudit) report["head_row_audit"] = head_audit::notStarted(o.diagnosticCap);
    if (o.headRowDirectCopy) report["head_row_direct_copy_requested"] = true;
    std::unique_ptr<import_row::FinalTailObserver> finalTails;std::unique_ptr<NewOutput> finalTailProgress;
    std::unique_ptr<Device> device; std::unique_ptr<diagnostics::Recorder> recorder; auto stamp = Clock::now();
    auto phase = [&](const char* name) { auto now = Clock::now(); phases.push_back({{"name", name}, {"wall_seconds", std::chrono::duration<double>(now - stamp).count()}}); stamp = now; if (device) memories.push_back({{"phase", name}, {"observed", Json::parse(device->memoryJson())}, {"host_process", hostMemory()}}); if (recorder) recorder->boundary(name); };
    try {
        finalTails=arithmeticFinalTailObserver(o,finalTailProgress);
        // Readbacks happen synchronously in observers on this Device thread, after the graph's own drains.
        if (diagnostic) recorder = std::make_unique<diagnostics::Recorder>(diagnostic->plan, diagnostic->geometry, std::make_unique<DiagnosticDirectory>(o.output / L"diagnostics"),
            [&device](const Buffer& b) { return device->readFloats(b); }, diagnosticCommitments(in, o, *diagnostic), diagnosticProducer(o));
        device = std::make_unique<Device>(o.shaders.wstring(), o.pci, o.luid); phase("device_creation"); report["device_identity"] = Json::parse(device->identityJson());
        {
            // Empty selection is the complete normal inference graph. No
            // diagnostic subset or CPU-resident tensor copy is substituted.
            // --head-row-audit holds this read-only, non-write-shared handle across the importer's whole-file
            // SHA-256 and every later audit read, so no writer can change the authenticated bytes in between.
            std::unique_ptr<Handle> auditSource,auditConfig;
            if (o.headRowAudit) auditSource = std::make_unique<Handle>(CreateFileW((o.model / L"model.safetensors").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
            auditConfig=std::make_unique<Handle>(CreateFileW((o.model/L"config.json").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
            ModelWeights weights(*device,o.model.wstring(),12ull*1024*1024*1024,{},WeightStorageExperiment::exact,nullptr,nullptr,nullptr,nullptr,o.weightImportAPI,finalTails.get(),o.weightSrvOnly);
            report["model_import_completed"]=true;auto provenance=Json::parse(weights.provenanceJson());if(o.experimentalParallel32)provenance["experimental_gemv_B1"]=report.at("experimental_gemv_B1");if(o.experimentalPadded32)provenance["experimental_gemm_multirow"]=report.at("experimental_gemm_multirow");
            finalTails->afterImport(*device,weights.at("model.language_model.embed_tokens.weight"),weights.at("lm_head.weight"));
            requireCleanFinalImportForArithmetic(*finalTails,o.weightImportAPI,false);report["clean_final_import_gate_completed"]=true;
            finalTails->releaseTargets();report["final_import_tail_audit"]=finalTails->report();report["model_buffers_retained_after_local_final_tail_reference_release"]=true;
            phase("authenticated_model_import_upload");
            require(provenance.at("full_graph_requested")==true&&provenance.at("omitted_graph_tensors").empty()&&provenance.at("tie_byte_equality")==true,"Full original graph and authenticated tied output head required");
            Json summary={{"schema","chandra.directcompute.final-import-arithmetic-model-summary.v1"},{"full_provenance_schema",provenance.at("schema")}};
            for(const char* key:{"model","revision","model_bytes","model_sha256","config_bytes","config_sha256","tensor_count","full_graph_requested","uploaded_tensor_names","maximum_shard_bytes","maximum_tracked_weight_bytes","tracked_buffer_bytes_before","uploaded_storage_bytes","tracked_buffer_bytes_after","tie_byte_equality_performed","tie_byte_equality","mapping_lifetime","implicit_cast_or_dequantization","numerical_acceptance","full_model_qualified","weight_import_api","API_operations_completed","UpdateSubresource_calls","DEFAULT_initial_data_Device_creations"})summary[key]=provenance.at(key);
            summary["weight_srv_only"]=provenance.at("weight_srv_only");if(o.experimentalParallel32)summary["experimental_gemv_B1"]=provenance.at("experimental_gemv_B1");if(o.experimentalPadded32)summary["experimental_gemm_multirow"]=provenance.at("experimental_gemm_multirow");report["model_provenance_summary"]=std::move(summary);
            const auto fullProvenance=provenance.dump()+"\n";require(fullProvenance.size()<=manifestByteLimit,"Complete bridge model provenance exceeds existing 1 MiB metadata bound");
            const auto fullProvenanceSha=diagnostics::sha256(fullProvenance.data(),fullProvenance.size());require(fullProvenanceSha.size()==64&&std::all_of(fullProvenanceSha.begin(),fullProvenanceSha.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}),"Complete bridge model provenance hash unavailable");
            if (recorder) recorder->authenticatedModel(provenance, report["device_identity"]);
            std::unique_ptr<head_audit::Auditor> audit;
            if (auditSource) audit = headAudit(*device, weights, provenance, *auditSource, o.output / L"head-row-audit", o.diagnosticCap);
            report["full_model_provenance_file"]["write_attempted"]=true;
            {DiagnosticFile file(o.output/L"head-row-audit"/L"model-provenance.json");file.write(fullProvenance.data(),fullProvenance.size());file.flush();}
            report["full_model_provenance_file"]["committed_bytes"]=fullProvenance.size();report["full_model_provenance_file"]["sha256"]=fullProvenanceSha;report["full_model_provenance_file"]["written_and_flushed"]=true;
            if (audit) {
                try { audit->afterImport(o.headRowDirectCopy); }
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
            requireCleanFinalImportForArithmetic(*finalTails,o.weightImportAPI,true);require(report.at("full_model_provenance_file").at("written_and_flushed")==true,"Complete provenance persistence required before arithmetic");require(report.dump().size()+1<=128*1024,"Bridge report exceeds existing 128 KiB before arithmetic");report["arithmetic_call_admitted"]=true;report["initialized_model_reused_for_arithmetic"]=true;
            try { generate(*device, weights, in, o.diagnosticCap, o.output, report, generated, hooks);report["arithmetic_return_observed"]=true; }
            catch (const std::exception& e) { if (audit) report["head_row_audit"] = head_audit::finishAfterFailure(*audit, e.what()); throw; }
            catch (...) { if (audit) report["head_row_audit"] = head_audit::finishAfterFailure(*audit, "non-standard exception"); throw; }
            if (audit) report["head_row_audit"] = audit->finish(head_audit::RunEnd::completed);
        }
        device->drain(); phase("model_buffer_release_and_drain");
        require(device->trackedBufferBytes() == 0, "Owned graph buffers did not retire to zero"); report["owned_buffer_zero"] = true;
        report["passed"] = true; report["passed_scope"] = "Executable graph returned finite greedy tokens and retired owned buffers; numerical/full-OCR qualification is pending";
    } catch (const std::exception& e) {
        report["passed"] = false; report["error"] = e.what(); report["owned_worker_retirement_required"] = true;
        if(finalTails){if(!finalTails->completed())finalTails->failure(e.what());finalTails->releaseTargets();}
        if (device) { try { device->drain(); report["final_memory"] = Json::parse(device->memoryJson()); report["final_drain_completed"] = true; } catch (const std::exception& drain) { report["final_drain_completed"] = false; report["final_drain_error"] = drain.what(); } }
    } catch (...) {
        report["passed"]=false;report["error"]="Non-standard final-import arithmetic exception";report["owned_worker_retirement_required"]=true;
        if(finalTails){if(!finalTails->completed())finalTails->failure("Non-standard final-import arithmetic exception");finalTails->releaseTargets();}
        if(device){try{device->drain();report["final_memory"]=Json::parse(device->memoryJson());report["final_drain_completed"]=true;}catch(const std::exception& drain){report["final_drain_completed"]=false;report["final_drain_error"]=drain.what();}}
    }
    if(finalTails){finalTails->releaseTargets();report["final_import_tail_audit"]=finalTails->report();report["import_outcome"]=finalTails->report().at("outcome");}
    if (recorder) {
        // The terminal line is appended after success, failure or retirement; dumps stay unqualified either way.
        try { report["diagnostics"] = recorder->finish(report.at("passed").get<bool>(), report.contains("error") ? report.at("error").get<std::string>() : std::string()); }
        catch (const std::exception& e) {
            report["diagnostics"] = {{"finish_error", e.what()}, {"dump_complete", false}, {"qualified", false}};
            if (report.at("passed").get<bool>()) { report["passed"] = false; report["error"] = std::string("Diagnostic dump finalization failed: ") + e.what(); }
        }
    }
    report["outcome"]=report.at("passed").get<bool>()?"clean_tails_then_original_cap2_returned":"failed_or_refused";
    finishExecutionReport(report, in, generated, phases, memories, start);
    return report;
}

// Separate full-page profile. The original cap2/observer/output contracts above remain unchanged.
constexpr uint64_t fullPageTerminalBytes=manifestByteLimit;
constexpr uint64_t fullPageTokenRecordBytes=1024;
constexpr uint64_t fullPageTokenJournalSuccessBytes=uint64_t(normalOutputLimit)*fullPageTokenRecordBytes;
// generate() flushes a token row before its token hook. Preserve a rejected/partial final row too;
// NewOutput's unchanged 4 MiB per-line guard is the worst-case last-write bound, not a success claim.
constexpr uint64_t fullPageTokenJournalFailureBytes=uint64_t(normalOutputLimit-1)*fullPageTokenRecordBytes+manifestByteLimit*4;
constexpr uint64_t fullPageAggregateBytes=20ull*1024*1024;
static_assert(fullPageTokenJournalFailureBytes+3*fullPageTerminalBytes+128*1024+2*import_row::final_tails::rowArrayBytes+import_row::tails::rowBytes<=fullPageAggregateBytes,"Full-page profile must retain terminal/failure/provenance and failed token evidence");
Json fullPageOutputProfile(){
 return {{"schema","chandra.directcompute.final-import-full-page-output-profile.v1"},{"terminal_result_or_failure_bytes",fullPageTerminalBytes},{"full_model_provenance_bytes",manifestByteLimit},{"generated_token_records",normalOutputLimit},{"successful_token_record_bytes",fullPageTokenRecordBytes},{"successful_token_journal_bytes",fullPageTokenJournalSuccessBytes},{"failed_token_journal_bytes",fullPageTokenJournalFailureBytes},{"token_bound_checked_after_flush",true},{"final_tail_journal_records",32},{"final_tail_journal_record_bytes",4096},{"final_tail_journal_bytes",128*1024},{"source_tail_bytes",import_row::final_tails::rowArrayBytes},{"matched_tail_bytes",import_row::final_tails::rowArrayBytes},{"first_bad_tail_bytes",import_row::tails::rowBytes},{"maximum_files",8},{"aggregate_artifact_bytes",fullPageAggregateBytes},{"historical_profiles_changed",false}};
}
std::unique_ptr<import_row::FinalTailObserver> fullPageFinalTailObserver(const Options& o,std::unique_ptr<NewOutput>& journal,uint32_t& records,uint64_t& bytes){
 const auto directory=o.output/L"final-import-tail-audit";require(CreateDirectoryW(directory.c_str(),nullptr),"Fresh full-page final-import-tail directory required");journal=std::make_unique<NewOutput>(directory/L"progress.jsonl");
 return std::make_unique<import_row::FinalTailObserver>(o.weightImportAPI,[directory](const std::string& name,const void* data,size_t count){
  require(((name==import_row::finalTailSourceFile||name==import_row::finalTailGoodFile)&&count>0&&count<=import_row::final_tails::rowArrayBytes&&count%import_row::tails::rowBytes==0)||(name==import_row::finalTailBadFile&&count==import_row::tails::rowBytes),"Fixed full-page final-tail payload name/extent required");
  DiagnosticFile file(directory/std::filesystem::u8path(name));file.write(data,count);file.flush();
 },[](const void* data,size_t count){return diagnostics::sha256(data,count);},[&journal,&records,&bytes](const Json& record){
  const auto count=record.dump().size()+1;require(count<=4096&&records<32&&bytes+count<=128*1024,"Full-page final-tail journal exceeds unchanged tail profile");journal->line(record);++records;bytes+=count;
 },true);
}
Json executeFinalTailFullPage(const Input& in,const Options& o){
 auto start=Clock::now();Json phases=Json::array(),memories=Json::array();std::vector<uint32_t> generated;
 Json report=executionReport(in,o.manifestSha,0);report["schema"]="chandra.directcompute.final-import-full-page-result.v1";report["mode"]="final_import_then_original_full_page_generation";report["import_only"]=false;
 report["output_profile"]=fullPageOutputProfile();report["weight_srv_only_requested"]=true;report["plan"]=import_row::FinalTailObserver::forecast(o.weightImportAPI,true);report["weight_import_api"]=weight_import::name(o.weightImportAPI);report["D3D11_usage"]="DEFAULT";report["ordered_B1_environment"]="ordered";
 if(o.experimentalParallel32){report["schema"]=chandra::parallel32_opt_in::fullPageSchema;report["mode"]=chandra::parallel32_opt_in::fullPageMode;report["ordered_B1_environment"]="parallel32";report["experimental_gemv_B1"]=parallel32Selection(o);}
    if(o.experimentalPadded32){report["schema"]=chandra::padded32_opt_in::schema(true,o.experimentalParallel32);report["mode"]=chandra::padded32_opt_in::mode(true,o.experimentalParallel32);report["experimental_gemm_multirow"]=padded32Selection(o);}
 report["generated_token_file"]="generated-tokens.jsonl";
 report["model_import_completed"]=false;report["clean_final_import_gate_completed"]=false;report["arithmetic_call_admitted"]=false;report["arithmetic_return_observed"]=false;report["initialized_model_reused_for_arithmetic"]=false;report["owned_buffer_zero"]=false;report["complete_page_eligible"]=false;report["numerical_qualification_accepted"]=false;report["owned_worker_retirement_required"]=true;
 report["native_cancellation_path_used"]=false;report["complete_page_eligibility_scope"]="Natural caller stop, complete ordinary generation and local buffer retirement only; actual root numerical/OCR acceptance and physical worker closure remain separate";
 report["source_maps_released_scope"]="Constructor-return RAII view/handle scope, not separately checked Win32 unmap/CloseHandle success";
 report["full_model_provenance_file"]={{"schema","chandra.directcompute.final-import-full-page-provenance-binding.v1"},{"file","model-provenance.json"},{"write_attempted",false},{"written_and_flushed",false},{"committed_bytes",0},{"sha256",nullptr}};
 std::unique_ptr<import_row::FinalTailObserver> finalTails;std::unique_ptr<NewOutput> progress;std::unique_ptr<Device> device;std::unique_ptr<Handle> heldModel,heldConfig;
 uint32_t tailRecords=0,tokenRecords=0;uint64_t tailBytes=0,tokenBytes=0;auto stamp=Clock::now();
 auto phase=[&](const char* name){auto now=Clock::now();phases.push_back({{"name",name},{"wall_seconds",std::chrono::duration<double>(now-stamp).count()}});stamp=now;if(device)memories.push_back({{"phase",name},{"observed",Json::parse(device->memoryJson())},{"host_process",hostMemory()}});};
 try{
  finalTails=fullPageFinalTailObserver(o,progress,tailRecords,tailBytes);
  heldModel=std::make_unique<Handle>(CreateFileW((o.model/L"model.safetensors").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
  heldConfig=std::make_unique<Handle>(CreateFileW((o.model/L"config.json").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));report["non_write_shared_model_and_config_handles_held"]=true;
  device=std::make_unique<Device>(o.shaders.wstring(),o.pci,o.luid);phase("device_creation");report["device_identity"]=Json::parse(device->identityJson());
  {
   ModelWeights weights(*device,o.model.wstring(),12ull*1024*1024*1024,{},WeightStorageExperiment::exact,nullptr,nullptr,nullptr,nullptr,o.weightImportAPI,finalTails.get(),true);
   report["model_import_completed"]=true;auto provenance=Json::parse(weights.provenanceJson());if(o.experimentalParallel32)provenance["experimental_gemv_B1"]=report.at("experimental_gemv_B1");if(o.experimentalPadded32)provenance["experimental_gemm_multirow"]=report.at("experimental_gemm_multirow");
   finalTails->afterImport(*device,weights.at("model.language_model.embed_tokens.weight"),weights.at("lm_head.weight"));
   requireCleanFinalImportForArithmetic(*finalTails,o.weightImportAPI,false);report["clean_final_import_gate_completed"]=true;
   finalTails->releaseTargets();report["final_import_tail_audit"]=finalTails->report();report["model_buffers_retained_after_local_final_tail_reference_release"]=true;phase("authenticated_model_import_upload");
   require(provenance.at("full_graph_requested")==true&&provenance.at("omitted_graph_tensors").empty()&&provenance.at("tie_byte_equality")==true,"Full original graph and authenticated tied output head required");
   Json summary={{"schema","chandra.directcompute.final-import-full-page-model-summary.v1"},{"full_provenance_schema",provenance.at("schema")}};
   for(const char* key:{"model","revision","model_bytes","model_sha256","config_bytes","config_sha256","tensor_count","full_graph_requested","uploaded_tensor_names","maximum_shard_bytes","maximum_tracked_weight_bytes","tracked_buffer_bytes_before","uploaded_storage_bytes","tracked_buffer_bytes_after","tie_byte_equality_performed","tie_byte_equality","mapping_lifetime","implicit_cast_or_dequantization","numerical_acceptance","full_model_qualified","weight_import_api","API_operations_completed","UpdateSubresource_calls","DEFAULT_initial_data_Device_creations"})summary[key]=provenance.at(key);
   summary["weight_srv_only"]=provenance.at("weight_srv_only");if(o.experimentalParallel32)summary["experimental_gemv_B1"]=provenance.at("experimental_gemv_B1");if(o.experimentalPadded32)summary["experimental_gemm_multirow"]=provenance.at("experimental_gemm_multirow");report["model_provenance_summary"]=std::move(summary);
   const auto fullProvenance=provenance.dump()+"\n";require(fullProvenance.size()<=manifestByteLimit,"Complete full-page model provenance exceeds unchanged 1 MiB metadata bound");
   const auto digest=diagnostics::sha256(fullProvenance.data(),fullProvenance.size());require(digest.size()==64&&std::all_of(digest.begin(),digest.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}),"Complete full-page model provenance hash unavailable");
   report["full_model_provenance_file"]["write_attempted"]=true;{DiagnosticFile file(o.output/L"model-provenance.json");file.write(fullProvenance.data(),fullProvenance.size());file.flush();}
   report["full_model_provenance_file"]["committed_bytes"]=fullProvenance.size();report["full_model_provenance_file"]["sha256"]=digest;report["full_model_provenance_file"]["written_and_flushed"]=true;
   RequestHooks hooks;hooks.phase=phase;hooks.token=[&](uint32_t index,const Json& row){
    const auto count=row.dump().size()+1;require(index==tokenRecords&&index<normalOutputLimit&&count<=fullPageTokenRecordBytes&&tokenBytes+count<=fullPageTokenJournalSuccessBytes,"Full-page committed token journal exceeds declared successful profile");++tokenRecords;tokenBytes+=count;
   };
   requireCleanFinalImportForArithmetic(*finalTails,o.weightImportAPI,true);require(report.at("full_model_provenance_file").at("written_and_flushed")==true,"Complete provenance persistence required before full-page arithmetic");require(report.dump().size()+1<=fullPageTerminalBytes,"Full-page report exceeds separate 1 MiB profile before arithmetic");report["arithmetic_call_admitted"]=true;report["initialized_model_reused_for_arithmetic"]=true;
   generate(*device,weights,in,0,o.output,report,generated,hooks);report["arithmetic_return_observed"]=true;
   require(tokenRecords==generated.size()&&tokenRecords<=normalOutputLimit,"Every returned full-page token requires its committed journal record");
  }
  device->drain();phase("model_buffer_release_and_drain");require(device->trackedBufferBytes()==0,"Owned full-page graph buffers did not retire to zero");report["owned_buffer_zero"]=true;
  report["complete_page_eligible"]=report.at("generation_completed")==true&&report.at("stop_reason")=="stop_token"&&report.at("producer_stop_token_observed")==true&&report.at("request_failed")==false&&report.at("request_retired")==true&&report.at("drained")==true&&report.at("request_replayed")==false&&report.at("normal_output_allowance_preserved")==true;
  report["passed"]=report.at("complete_page_eligible");report["passed_scope"]=report.at("complete_page_eligible")==true?"Original generation returned finite tokens, natural caller stop and locally retired buffers; root numerical/OCR qualification remains pending":"Original generation exhausted the unchanged normal allowance; complete partial token history is retained but complete-page eligibility is false";
 }catch(const std::exception& e){report["passed"]=false;report["complete_page_eligible"]=false;report["error"]=e.what();if(finalTails){if(!finalTails->completed())finalTails->failure(e.what());finalTails->releaseTargets();}
  if(device){try{device->drain();report["final_memory"]=Json::parse(device->memoryJson());report["final_drain_completed"]=true;report["owned_buffer_zero"]=device->trackedBufferBytes()==0;}catch(const std::exception& drain){report["final_drain_completed"]=false;report["final_drain_error"]=drain.what();}}
 }catch(...){report["passed"]=false;report["complete_page_eligible"]=false;report["error"]="Non-standard full-page generation exception";if(finalTails){if(!finalTails->completed())finalTails->failure("Non-standard full-page generation exception");finalTails->releaseTargets();}
  if(device){try{device->drain();report["final_memory"]=Json::parse(device->memoryJson());report["final_drain_completed"]=true;report["owned_buffer_zero"]=device->trackedBufferBytes()==0;}catch(const std::exception& drain){report["final_drain_completed"]=false;report["final_drain_error"]=drain.what();}}
 }
 if(finalTails){finalTails->releaseTargets();report["final_import_tail_audit"]=finalTails->report();report["import_outcome"]=finalTails->report().at("outcome");}
 finalTails.reset();progress.reset();heldConfig.reset();heldModel.reset();report["held_model_config_handles_released"]=true;report["local_final_tail_observer_released"]=true;report["local_final_tail_journal_handle_released"]=true;
 report["final_tail_journal_committed_records"]=tailRecords;report["final_tail_journal_committed_bytes"]=tailBytes;report["generated_token_journal_verified_records"]=tokenRecords;report["generated_token_journal_verified_bytes"]=tokenBytes;
 report["outcome"]=report.at("passed").get<bool>()?"clean_tails_then_natural_full_page_returned":(report.value("stop_reason",std::string())=="normal_output_length"?"normal_output_length_ineligible":"failed_or_refused");
 finishExecutionReport(report,in,generated,phases,memories,start);return report;
}

// Separate API-comparison import-only point; no generation or historical watcher.
Json executeFinalTailImport(const Options& o){
 const auto directory=o.output/L"final-import-tail-audit";require(CreateDirectoryW(directory.c_str(),nullptr),"Fresh final-import-tail directory required");NewOutput progress(directory/L"progress.jsonl");
 import_row::FinalTailObserver observer(o.weightImportAPI,[directory](const std::string& name,const void* data,size_t bytes){
  require(((name==import_row::finalTailSourceFile||name==import_row::finalTailGoodFile)&&bytes>0&&bytes<=import_row::final_tails::rowArrayBytes&&bytes%import_row::tails::rowBytes==0)||(name==import_row::finalTailBadFile&&bytes==import_row::tails::rowBytes),"Fixed final-tail payload name/extent required");
  DiagnosticFile file(directory/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
 },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&progress](const Json& record){progress.line(record);},o.weightSrvOnly);
 Handle heldModel(CreateFileW((o.model/L"model.safetensors").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
 Handle heldConfig(CreateFileW((o.model/L"config.json").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
 Device device(o.shaders.wstring(),o.pci,o.luid);
 auto result=import_row::runFinalTailImport(device,o.model.wstring(),o.weightImportAPI,observer,o.weightSrvOnly);result["device_identity"]=Json::parse(device.identityJson());result["input_manifest_sha256"]=o.manifestSha;result["plan"]=import_row::FinalTailObserver::forecast(o.weightImportAPI,o.weightSrvOnly);return result;
}

// Separate import-only CLI; no arithmetic calls or original watcher mode changes.
Json executeShard1Watch(const Input& in,const Options& o){
    Json report={{"schema","chandra.directcompute.shard1-boundary-watch-result.v1"},{"diagnostic_type","import_only_original_shard1_boundary_watch"},{"passed",false},
        {"diagnostic_completed",false},{"outcome","partial_or_refused"},{"actual_inference",false},{"complete_inference",false},{"qualified_OCR",false},{"qualified_full_graph",false},{"performance_claim",false},
        {"vision_execution_started",false},{"text_execution_started",false},{"generated_token_count",0},{"request_replayed",false},{"input_manifest_sha256",o.manifestSha},{"context_limit",contextLimit},{"normal_output_allowance",normalOutputLimit},
        {"prompt_tokens",in.ids.size()},{"model_import_completed",false},{"owned_buffer_zero",false},{"owned_worker_retirement_required",true}};
    std::unique_ptr<Device> device;std::unique_ptr<import_row::Shard1Observer> observer;std::unique_ptr<NewOutput> progress;
    std::unique_ptr<Handle> heldModel,heldConfig;
    try{
        heldModel=std::make_unique<Handle>(CreateFileW((o.model/L"model.safetensors").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
        heldConfig=std::make_unique<Handle>(CreateFileW((o.model/L"config.json").c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr));
        report["non_write_shared_model_and_config_handles_held"]=true;
        const auto directory=o.output/L"shard1-boundary-watch";require(CreateDirectoryW(directory.c_str(),nullptr),"Fresh shard1 observer directory required");progress=std::make_unique<NewOutput>(directory/L"progress.jsonl");
        observer=std::make_unique<import_row::Shard1Observer>([directory](const std::string& name,const void* data,size_t bytes){
            require((name=="plan.json"&&bytes>0&&bytes<=import_row::shard1::maximumPlanBytes)||((name==import_row::shard1SourceFile||name==import_row::shard1GoodFile||name==import_row::shard1BadFile)&&bytes==import_row::shard1::rowBytes),"Fixed bounded shard1 output required");
            DiagnosticFile file(directory/std::filesystem::u8path(name));file.write(data,bytes);file.flush();
        },[](const void* data,size_t bytes){return diagnostics::sha256(data,bytes);},[&progress](const Json& record){progress->line(record);});
        device=std::make_unique<Device>(o.shaders.wstring(),o.pci,o.luid);report["device_identity"]=Json::parse(device->identityJson());
        {
            ModelWeights weights(*device,o.model.wstring(),12ull*1024*1024*1024,{},WeightStorageExperiment::exact,nullptr,nullptr,nullptr,observer.get());
            report["model_import_completed"]=true;const auto provenance=Json::parse(weights.provenanceJson());
            Json summary={{"schema","chandra.directcompute.shard1-boundary-model-provenance.v1"},{"full_ordered_upload_plan","shard1-boundary-watch/plan.json"}};
            for(const char* key:{"model","revision","model_bytes","model_sha256","config_bytes","config_sha256","tensor_count","full_graph_requested","uploaded_tensor_names","maximum_shard_bytes","maximum_tracked_weight_bytes","tracked_buffer_bytes_before","uploaded_storage_bytes","tracked_buffer_bytes_after","tie_byte_equality_performed","tie_byte_equality","mapping_lifetime","implicit_cast_or_dequantization","numerical_acceptance","full_model_qualified"})summary[key]=provenance.at(key);
            report["model_provenance"]=std::move(summary);
            observer->afterImport(*device,weights.at("model.language_model.embed_tokens.weight"),weights.at("lm_head.weight"));
        }
    }catch(const import_row::Shard1Stop&){report["first_fault_stop_observed"]=true;
    }catch(const std::exception& e){report["error"]=e.what();if(observer)observer->failure(e.what());
    }catch(...){report["error"]="Non-standard shard1 import exception";if(observer)observer->failure("Non-standard shard1 import exception");}
    if(observer){observer->stackClosedAndReleaseTarget();report["shard1_boundary_watch"]=observer->report();report["diagnostic_completed"]=observer->completed();report["outcome"]=observer->report().at("outcome");}
    if(device){try{device->drain();report["final_drain_completed"]=true;report["final_memory"]=Json::parse(device->memoryJson());report["owned_buffer_zero"]=device->trackedBufferBytes()==0;}
        catch(const std::exception& e){report["final_drain_completed"]=false;report["final_drain_error"]=e.what();}}
    report["passed"]=report.at("diagnostic_completed")==true&&report.at("owned_buffer_zero")==true&&report.value("final_drain_completed",false)&&!report.contains("error");
    report["passed_scope"]="Typed observation completed and local buffers drained; first_fault is observed corruption, not numerical success; root physical retirement remains separate";
    heldConfig.reset();heldModel.reset();report["held_model_config_handles_released"]=true;return report;
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    Options o; bool created = false; auto started = Clock::now();
    bool finalFullPageRequested=false,parallel32Requested=false,padded32Requested=false;for(int index=1;index<argc;++index){if(std::wstring(argv[index])==L"--import-final-head-tail-full-page")finalFullPageRequested=true;if(std::wstring(argv[index])==L"--experimental-parallel32")parallel32Requested=true;if(std::wstring(argv[index])==L"--experimental-padded32")padded32Requested=true;}
    bool finalTailArithmeticRequested=false;for(int index=1;index<argc;++index)if(std::wstring(argv[index])==L"--import-final-head-tail-before-arithmetic")finalTailArithmeticRequested=true;
    bool weightSrvOnlyRequested=false;for(int index=1;index<argc;++index)if(std::wstring(argv[index])==L"--weight-srv-only")weightSrvOnlyRequested=true;
    bool finalTailRequested=false;for(int index=1;index<argc;++index)if(std::wstring(argv[index])==L"--import-final-head-tail-audit")finalTailRequested=true;
    bool shard1Requested=false;for(int index=1;index<argc;++index)if(std::wstring(argv[index])==L"--import-shard1-boundary-watch")shard1Requested=true;
    bool watchRequested=false,arithmeticWatchRequested=false,tailRequested=false;for(int index=1;index<argc;++index){if(std::wstring(argv[index])==L"--import-row-watch")watchRequested=true;if(std::wstring(argv[index])==L"--import-row-watch-before-arithmetic")arithmeticWatchRequested=true;if(std::wstring(argv[index])==L"--import-head-shard-tail-audit")tailRequested=true;} // Classification only; options() still performs every refusal before execution.
    try {
        o = options(argc, argv); auto in = authenticateInput(o.manifest, o.manifestSha); // No Device before all input/hash/geometry/finite guards.
        double authenticationSeconds = std::chrono::duration<double>(Clock::now() - started).count();
        auto diagnostic = diagnosticRequest(in, o); // Invalid or over-budget selections refuse before any output or Device.
        require(std::filesystem::create_directory(o.output), "Fresh output directory creation failed"); created = true;
        auto report = o.finalImportFullPage ? executeFinalTailFullPage(in,o) : o.forecast ? forecast(in, o, diagnostic.get()) : (o.finalImportTailAudit ? (o.finalImportTailArithmetic ? executeFinalTailArithmetic(in,o,diagnostic.get()) : executeFinalTailImport(o)) : (o.shard1BoundaryWatch ? executeShard1Watch(in,o) : (o.importRowWatch ? executeImportWatch(in, o) : execute(in, o, diagnostic.get()))));
        if(o.experimentalPadded32){const auto counts=experimental::gemmPadded32Counts();report["experimental_gemm_multirow_dispatches"]={{"padded32",counts.padded32},{"fallback",counts.fallback},{"scope","Successful Device::dispatch returns for original rows>1 calls; includes ragged one-row fallback tails; excludes B1; not GPU completion proof"}};}
        report["input_authentication_wall_seconds"] = authenticationSeconds; report["final_host_process_memory"] = hostMemory();
        report["total_process_path_wall_seconds"] = std::chrono::duration<double>(Clock::now() - started).count();
        if(o.finalImportFullPage)require(report.dump().size()+1<=fullPageTerminalBytes,"Full-page terminal exceeds its separate 1 MiB profile");
        else if(o.importRowWatch || o.shard1BoundaryWatch || o.finalImportTailAudit)require(report.dump().size()+1<=128*1024,"Bounded import-only result exceeds 128 KiB");
        NewOutput receipt(o.output / L"result.json"); receipt.line(report); std::cout << report.dump() << '\n';
        return o.forecast ? 0 : (report.at("passed").get<bool>() ? 0 : 1);
    } catch (const std::exception& e) {
        Json error = {{"schema", "chandra.directcompute.result.v1"}, {"passed", false}, {"error", e.what()}, {"request_replayed", false}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
        if(weightSrvOnlyRequested)error["weight_srv_only_requested"]=true;
        if(finalTailRequested){error["schema"]="chandra.directcompute.final-import-tail-result.v1";error["import_only"]=true;error["diagnostic_completed"]=false;error["outcome"]="partial_or_refused";error["actual_inference"]=false;error["complete_inference"]=false;error["arithmetic_gate_eligible"]=false;}
        if(shard1Requested){error["schema"]="chandra.directcompute.shard1-boundary-watch-result.v1";error["diagnostic_type"]="import_only_original_shard1_boundary_watch";error["diagnostic_completed"]=false;error["outcome"]="partial_or_refused";error["actual_inference"]=false;error["complete_inference"]=false;}
        if(watchRequested){error["schema"]="chandra.directcompute.import-row-watch-result.v1";error["diagnostic_type"]="import_only_original_weight_row_watch";error["diagnostic_completed"]=false;error["outcome"]="partial_or_refused";error["actual_inference"]=false;error["complete_inference"]=false;}
        if(tailRequested){error["schema"]="chandra.directcompute.head-shard-tail-watch-result.v1";error["diagnostic_type"]="import_only_original_head_shard_tails";error["head_shard_tail_audit_requested"]=true;error["diagnostic_completed"]=false;error["outcome"]="partial_or_refused";error["actual_inference"]=false;error["complete_inference"]=false;}
        if(arithmeticWatchRequested&&!tailRequested){error["schema"]="chandra.directcompute.watched-import-cap2-result.v1";error["mode"]="watched_import_then_original_cap2_arithmetic";error["import_watch_before_arithmetic_requested"]=true;error["clean_import_gate_completed"]=created?Json(nullptr):Json(false);error["arithmetic_call_admitted"]=created?Json(nullptr):Json(false);error["arithmetic_return_observed"]=created?Json(nullptr):Json(false);error["failure_scope"]="CLI/options/input/output or terminal report persistence; after output creation arithmetic status is unknown here";}
        if(finalTailArithmeticRequested){error["schema"]="chandra.directcompute.final-import-cap2-result.v1";error["mode"]="final_import_then_original_cap2_arithmetic";error["import_only"]=false;error["final_import_before_arithmetic_requested"]=true;error["model_import_completed"]=created?Json(nullptr):Json(false);error["clean_final_import_gate_completed"]=created?Json(nullptr):Json(false);error["arithmetic_call_admitted"]=created?Json(nullptr):Json(false);error["arithmetic_return_observed"]=created?Json(nullptr):Json(false);error["initialized_model_reused_for_arithmetic"]=created?Json(nullptr):Json(false);error["actual_inference"]=created?Json(nullptr):Json(false);error["failure_scope"]="CLI/options/input/output or terminal report persistence; after output creation import/arithmetic status is unknown here";error["owned_worker_retirement_required"]=true;}
        if(finalFullPageRequested){error["schema"]="chandra.directcompute.final-import-full-page-result.v1";error["mode"]="final_import_then_original_full_page_generation";error["import_only"]=false;error["complete_page_eligible"]=false;error["output_profile"]=fullPageOutputProfile();error["model_import_completed"]=created?Json(nullptr):Json(false);error["clean_final_import_gate_completed"]=created?Json(nullptr):Json(false);error["arithmetic_call_admitted"]=created?Json(nullptr):Json(false);error["arithmetic_return_observed"]=created?Json(nullptr):Json(false);error["initialized_model_reused_for_arithmetic"]=created?Json(nullptr):Json(false);error["actual_inference"]=created?Json(nullptr):Json(false);error["complete_inference"]=created?Json(nullptr):Json(false);error["generation_completed"]=created?Json(nullptr):Json(false);error["diagnostic_completed"]=created?Json(nullptr):Json(false);error["failure_scope"]="CLI/options/input/output or terminal report persistence; after output creation import/arithmetic status is unknown here; generated-token leaves are preserved";error["owned_worker_retirement_required"]=true;}
        if(parallel32Requested){error["schema"]=finalFullPageRequested?chandra::parallel32_opt_in::fullPageSchema:(finalTailArithmeticRequested?chandra::parallel32_opt_in::cap2Schema:chandra::parallel32_opt_in::refusedSchema);error["mode"]=finalFullPageRequested?chandra::parallel32_opt_in::fullPageMode:(finalTailArithmeticRequested?chandra::parallel32_opt_in::cap2Mode:"refused_parallel32_before_route_admission");error["experimental_parallel32_requested"]=true;error["expected_gemv_B1_selector"]="parallel32";error["gemv_B1_selection_verified"]=false;error["numerical_qualification_accepted"]=false;}
        if(padded32Requested){error["schema"]=(finalFullPageRequested||finalTailArithmeticRequested)?chandra::padded32_opt_in::schema(finalFullPageRequested,parallel32Requested):"chandra.directcompute.padded32-refused-result.v1";error["mode"]=(finalFullPageRequested||finalTailArithmeticRequested)?chandra::padded32_opt_in::mode(finalFullPageRequested,parallel32Requested):"refused_padded32_before_route_admission";error["experimental_padded32_requested"]=true;error["expected_gemm_multirow_selector"]="padded32";error["gemm_multirow_selection_verified"]=false;error["numerical_qualification_accepted"]=false;}
        if (created) { try { if(finalFullPageRequested)require(error.dump().size()+1<=fullPageTerminalBytes,"Full-page failure exceeds its separate 1 MiB profile");NewOutput receipt(o.output / L"failure.json"); receipt.line(error); } catch (...) {} }
        std::cerr << error.dump() << '\n'; return 1;
    }
}
