// New code, MPL-2.0. Connected native page path; unqualified until actual graph/OCR checks.
// No tokenizer, processor, engine fallback, request replay, or implicit adapter selection.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define PSAPI_VERSION 2
#include "api.h"
#include "text_model.h"
#include "vision_model.h"
#include "diagnostics.h"
#include "../vendor/nlohmann/json.hpp"
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using namespace chandra::dc;
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

void require(bool value, const std::string& reason) { if (!value) throw std::runtime_error(reason); }
uint64_t product(uint64_t a, uint64_t b) {
    require(!b || a <= UINT64_MAX / b, "Input extent multiplication overflow"); return a * b;
}
uint64_t sum(uint64_t a, uint64_t b) {
    require(b <= UINT64_MAX - a, "Input extent addition overflow"); return a + b;
}
uint64_t unsignedNumber(const Json& j) {
    require(j.is_number_unsigned() || (j.is_number_integer() && j.get<int64_t>() >= 0), "Nonnegative integer required");
    return j.get<uint64_t>();
}
std::string utf8(const std::wstring& value) {
    require(value.size() <= INT_MAX, "CLI value is too long");
    if (value.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), int(value.size()), nullptr, 0, nullptr, nullptr);
    require(n > 0, "Invalid Unicode CLI value"); std::string result(size_t(n), '\0');
    require(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), int(value.size()), result.data(), n, nullptr, nullptr) == n, "CLI Unicode conversion failed");
    return result;
}
std::string digestText(const Json& value) {
    require(value.is_string(), "SHA-256 must be a string"); auto s = value.get<std::string>();
    require(s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }), "Lowercase SHA-256 required");
    return s;
}
Json strictJson(const std::vector<char>& bytes) {
    std::vector<std::set<std::string>> objects;
    auto check = [&](int, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start) objects.emplace_back();
        else if (event == Json::parse_event_t::key) require(!objects.empty() && objects.back().insert(value.get<std::string>()).second, "Duplicate input JSON key");
        else if (event == Json::parse_event_t::object_end) { require(!objects.empty(), "Unbalanced input JSON object"); objects.pop_back(); }
        return true;
    };
    auto result = Json::parse(bytes.begin(), bytes.end(), check, true, false);
    require(result.is_object(), "Input manifest must be an object"); return result;
}
struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    explicit Handle(HANDLE h) : value(h) { require(h != INVALID_HANDLE_VALUE, "Opening an input/output file failed"); }
    ~Handle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete; Handle& operator=(const Handle&) = delete;
};
struct Sha256 {
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr; std::vector<UCHAR> object;
    Sha256() {
        require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0, "Opening SHA-256 provider failed");
        ULONG bytes = 0, used = 0;
        if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&bytes), sizeof(bytes), &used, 0) < 0 || used != sizeof(bytes)) {
            BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr; throw std::runtime_error("SHA-256 object size failed");
        }
        try { object.resize(bytes); }
        catch (...) { BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr; throw; }
        if (BCryptCreateHash(algorithm, &hash, object.data(), bytes, nullptr, 0, 0) < 0) {
            BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr; throw std::runtime_error("Creating SHA-256 failed");
        }
    }
    ~Sha256() { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
    void update(const void* data, uint32_t bytes) { require(BCryptHashData(hash, static_cast<PUCHAR>(const_cast<void*>(data)), bytes, 0) >= 0, "Streaming SHA-256 failed"); }
    std::string finish() {
        std::array<UCHAR, 32> result{}; require(BCryptFinishHash(hash, result.data(), ULONG(result.size()), 0) >= 0, "Finishing SHA-256 failed");
        static const char* hex = "0123456789abcdef"; std::string text; text.reserve(64);
        for (auto c : result) { text.push_back(hex[c >> 4]); text.push_back(hex[c & 15]); } return text;
    }
};
// Read once through a non-write-shared file handle; hash exactly the bytes used.
// Large tensors go directly into their typed vector, with no full payload copy.
void authenticatedRead(const std::filesystem::path& path, void* destination, uint64_t bytes, const std::string& expected) {
    require(bytes && bytes <= inputByteLimit && destination, "Bounded nonempty input required");
    DWORD attributes = GetFileAttributesW(path.c_str());
    require(attributes != INVALID_FILE_ATTRIBUTES && !(attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)), "Regular non-reparse input file required");
    Handle file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
    LARGE_INTEGER length{}; require(GetFileSizeEx(file.value, &length) && length.QuadPart >= 0 && uint64_t(length.QuadPart) == bytes, "Input byte length differs");
    Sha256 sha; auto* out = static_cast<unsigned char*>(destination);
    for (uint64_t done = 0; done < bytes;) {
        DWORD n = DWORD(std::min<uint64_t>(bytes - done, 1024 * 1024)), got = 0;
        require(ReadFile(file.value, out + size_t(done), n, &got, nullptr) && got == n, "Incomplete input read");
        sha.update(out + size_t(done), got); done += got;
    }
    char extra; DWORD got = 0; require(ReadFile(file.value, &extra, 1, &got, nullptr) && got == 0, "Input changed length during read");
    require(sha.finish() == expected, "Input SHA-256 differs");
}
std::filesystem::path localFile(const std::filesystem::path& root, const Json& name) {
    require(name.is_string(), "Relative input filename required"); auto s = name.get<std::string>();
    require(!s.empty() && s.size() <= 512 && s.front() != '/' && s.back() != '/' && s.find('\\') == std::string::npos && s.find(':') == std::string::npos && s.find('\0') == std::string::npos, "Local relative filename required; no rooted/device path");
    auto relative = std::filesystem::u8path(s);
    require(!relative.is_absolute() && !relative.has_root_path(), "Input traversal refused");
    auto candidate = root;
    for (const auto& component : relative) {
        require(component != L"." && component != L".." && !component.empty(), "Input traversal refused"); candidate /= component;
        DWORD attributes = GetFileAttributesW(candidate.c_str());
        require(attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT), "Input reparse point refused");
    }
    require(std::filesystem::is_regular_file(candidate), "Input tensor is not a regular file");
    return candidate;
}
Json authenticatedMetadata(const std::filesystem::path& root, const Json& name, const std::string& sha) {
    auto path = localFile(root, name); uint64_t size = std::filesystem::file_size(path);
    require(size && size <= manifestByteLimit, "Bounded provenance metadata required");
    std::vector<char> bytes(static_cast<size_t>(size)); authenticatedRead(path, bytes.data(), size, sha); return strictJson(bytes);
}
void authenticateCaller(const Json& generation, const std::filesystem::path& root) {
    const auto& provenance = generation.at("stop_token_provenance");
    require(digestText(provenance.at("caller_source_sha256")) == chandraCallerSha &&
        digestText(provenance.at("generation_config_sha256")) == generationConfigSha &&
        digestText(provenance.at("tokenizer_config_sha256")) == tokenizerConfigSha, "Pinned actual Chandra caller/config/tokenizer provenance required");
    auto callerPath = localFile(root, provenance.at("caller_source_file"));
    auto size = std::filesystem::file_size(callerPath); require(size == 3178, "Pinned actual Chandra caller byte length differs");
    std::vector<char> source(static_cast<size_t>(size)); authenticatedRead(callerPath, source.data(), size, chandraCallerSha);
    auto config = authenticatedMetadata(root, "generation_config.json", generationConfigSha);
    auto tokenizer = authenticatedMetadata(root, "tokenizer_config.json", tokenizerConfigSha);
    Json stops = Json::array(); const auto& eos = config.at("eos_token_id");
    if (eos.is_array()) { require(!eos.empty() && eos.size() <= 16, "Pinned EOS list is invalid"); for (const auto& id : eos) stops.push_back(unsignedNumber(id)); }
    else stops.push_back(unsignedNumber(eos));
    std::vector<uint32_t> turnEnd;
    for (auto it = tokenizer.at("added_tokens_decoder").begin(); it != tokenizer.at("added_tokens_decoder").end(); ++it) {
        if (it.value().at("content") != "<|im_end|>") continue;
        require(!it.key().empty() && it.key().size() <= 6 && std::all_of(it.key().begin(), it.key().end(), [](char c) { return c >= '0' && c <= '9'; }), "Invalid tokenizer turn-end ID");
        auto id = std::stoul(it.key()); require(id < vocabulary, "Turn-end outside vocabulary"); turnEnd.push_back(uint32_t(id));
    }
    require(turnEnd.size() == 1, "One exact tokenizer turn-end ID required");
    if (std::find(stops.begin(), stops.end(), Json(turnEnd[0])) == stops.end()) stops.push_back(turnEnd[0]);
    require(stops == generation.at("stop_token_ids") && stops == provenance.at("caller_stop_token_ids"), "Authenticated actual caller stop extraction differs");
}
struct Descriptor { std::filesystem::path path; std::string dtype, sha; uint64_t bytes = 0, elements = 0; std::vector<uint32_t> shape; };
Descriptor descriptor(const Json& row, const std::filesystem::path& root) {
    require(row.is_object(), "Tensor descriptor required"); Descriptor d;
    d.path = localFile(root, row.at("file")); d.sha = digestText(row.at("sha256"));
    require(row.at("dtype").is_string(), "Tensor dtype is not a string"); d.dtype = row.at("dtype").get<std::string>();
    require(d.dtype == "I64" || d.dtype == "F32", "Unknown input dtype; no implicit conversion");
    require(row.at("byte_order") == "little", "Original little-endian raw tensor required");
    const auto& shape = row.at("shape"); require(shape.is_array() && !shape.empty() && shape.size() <= 3, "Invalid input tensor rank");
    d.elements = 1;
    for (const auto& v : shape) { uint64_t n = unsignedNumber(v); require(n && n <= UINT32_MAX, "Invalid input tensor dimension"); d.shape.push_back(uint32_t(n)); d.elements = product(d.elements, n); }
    d.bytes = unsignedNumber(row.at("bytes")); require(d.bytes == product(d.elements, d.dtype == "I64" ? 8 : 4) && d.bytes <= inputByteLimit, "Tensor shape/type/byte extent differs");
    const auto& strides = row.at("strides"); require(strides.is_array() && strides.size() == d.shape.size(), "Original contiguous element strides required");
    uint64_t stride = 1;
    for (size_t i = d.shape.size(); i-- > 0;) { require(unsignedNumber(strides[i]) == stride, "Noncontiguous input tensor refused"); stride = product(stride, d.shape[i]); }
    return d;
}
template<class T> std::vector<T> tensor(const Descriptor& d, const char* dtype, const std::vector<uint32_t>& shape) {
    require(d.dtype == dtype && d.shape == shape && d.bytes == product(d.elements, sizeof(T)), "Exact tensor ABI required");
    std::vector<T> result(static_cast<size_t>(d.elements)); authenticatedRead(d.path, result.data(), d.bytes, d.sha); return result;
}
struct Options {
    bool execute = false, forecast = false, diagnosticDump = false; uint32_t diagnosticCap = 0;
    std::filesystem::path model, shaders, manifest, output, diagnosticPlan; std::string pci, luid, manifestSha, diagnosticPlanSha;
};
Options options(int argc, wchar_t** argv) {
    Options o; std::set<std::wstring> seen;
    for (int i = 1; i < argc; ++i) {
        std::wstring name = argv[i]; require(seen.insert(name).second, "Duplicate CLI option");
        if (name == L"--execute") { o.execute = true; continue; }
        if (name == L"--forecast") { o.forecast = true; continue; }
        if (name == L"--diagnostic-dump") { o.diagnosticDump = true; continue; }
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
    require(std::filesystem::is_directory(o.model) && std::filesystem::is_directory(o.shaders), "Explicit model and shader directories must exist");
    require(!std::filesystem::exists(o.output), "Fresh output directory required; existing output is never overwritten"); return o;
}
struct Input {
    Json manifest; std::map<std::string, Descriptor> descriptors;
    std::vector<uint32_t> ids, visionRowMap, stopIds; std::vector<float> pixels;
    TextPositions positions; std::vector<VisionGrid> grids; uint32_t maximumPosition = 0, patchRows = 0;
    uint64_t tensorBytes = 0; VisionForecast visionForecast;
};
Json hostMemory() {
    PROCESS_MEMORY_COUNTERS_EX memory{}; memory.cb = sizeof(memory);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)))
        return {{"available", false}, {"win32_error", GetLastError()}};
    return {{"available", true}, {"working_set_bytes", memory.WorkingSetSize}, {"peak_working_set_bytes", memory.PeakWorkingSetSize},
        {"private_usage_bytes", memory.PrivateUsage}, {"scope", "Current native process observation; external Job/worker closure remains root-owned"}};
}
Input input(const Options& o) {
    const uint16_t endian = 1; require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "Native input ABI requires little-endian host storage");
    Input in; auto n = std::filesystem::file_size(o.manifest); require(n && n <= manifestByteLimit, "Bounded manifest required");
    std::vector<char> bytes(static_cast<size_t>(n)); authenticatedRead(o.manifest, bytes.data(), n, o.manifestSha); in.manifest = strictJson(bytes);
    const auto& m = in.manifest;
    require(m.at("schema") == inputSchema && m.at("model") == "datalab-to/chandra-ocr-2" && m.at("revision") == modelRevision, "Pinned model/input schema required");
    require(unsignedNumber(m.at("image_token_id")) == imageToken, "Exact image token ID required");
    require(m.contains("image") && m.at("image").is_object() && m.contains("prompt") && m.at("prompt").is_object() && m.at("processor_profile").is_string() && m.at("processor_kwargs").is_object(), "Source image/prompt/processor metadata required");
    digestText(m.at("image").at("sha256")); digestText(m.at("image").at("pixels_sha256")); digestText(m.at("prompt").at("sha256"));
    require(m.at("image").at("dimensions").is_array() && m.at("image").at("dimensions").size() == 2 && unsignedNumber(m.at("image").at("dimensions")[0]) && unsignedNumber(m.at("image").at("dimensions")[1]), "Exact source pixel dimensions required");
    const auto& generation = m.at("generation"); require(generation.is_object() && unsignedNumber(generation.at("context_limit")) == contextLimit && unsignedNumber(generation.at("max_output_tokens")) == normalOutputLimit, "Full normal context/output allowance required");
    const auto& stop = generation.at("stop_token_ids"); require(stop.is_array() && !stop.empty() && stop.size() <= 16, "Explicit caller-derived stop tokens required");
    std::set<uint32_t> stops;
    for (const auto& id : stop) { auto v = unsignedNumber(id); require(v < vocabulary && stops.insert(uint32_t(v)).second, "Invalid/duplicate stop token"); in.stopIds.push_back(uint32_t(v)); }
    const auto& provenance = generation.at("stop_token_provenance");
    require(provenance.is_object() && provenance.at("caller_source_file").is_string() && !provenance.at("caller_source_file").get<std::string>().empty() && provenance.at("caller_stop_token_ids") == stop, "Stop tokens must be bound to the actual Chandra caller");
    digestText(provenance.at("caller_source_sha256"));
    auto root = std::filesystem::canonical(o.manifest.parent_path()); authenticateCaller(generation, root);
    const auto& tensors = m.at("tensors"); require(tensors.is_object(), "Named raw tensors required");
    const std::set<std::string> allowed = {"input_ids", "pixel_values", "image_grid_thw", "position_ids", "attention_mask", "mm_token_type_ids", "text_position_ids", "rope_deltas"};
    std::set<std::filesystem::path> filenames;
    for (auto it = tensors.begin(); it != tensors.end(); ++it) {
        require(allowed.count(it.key()) != 0, "Unknown input tensor name"); auto d = descriptor(it.value(), root);
        require(filenames.insert(d.path).second, "Tensor filenames must be distinct"); in.tensorBytes = sum(in.tensorBytes, d.bytes);
        require(in.tensorBytes <= inputByteLimit, "Complete raw input payload exceeds 256 MiB"); in.descriptors.emplace(it.key(), std::move(d));
    }
    for (auto name : {"input_ids", "pixel_values", "image_grid_thw", "position_ids"}) require(in.descriptors.count(name) != 0, "Required input tensor missing");
    uint64_t promptTokens = unsignedNumber(m.at("prompt_tokens")); require(promptTokens && promptTokens <= contextLimit && promptTokens + normalOutputLimit <= contextLimit, "Prompt must retain the entire normal output allowance; no implicit truncation");
    uint32_t count = uint32_t(promptTokens); auto ids = tensor<int64_t>(in.descriptors.at("input_ids"), "I64", {1, count});
    in.ids.reserve(count); in.visionRowMap.reserve(count); Json imagePositions = Json::array(); uint32_t visionRow = 0;
    for (uint32_t i = 0; i < count; ++i) {
        require(ids[i] >= 0 && uint64_t(ids[i]) < vocabulary, "Prompt token outside vocabulary"); in.ids.push_back(uint32_t(ids[i]));
        if (in.ids.back() == imageToken) { imagePositions.push_back(i); in.visionRowMap.push_back(visionRow++); }
        else in.visionRowMap.push_back(UINT32_MAX);
    }
    require(m.at("image_token_positions") == imagePositions && visionRow != 0, "Exact ordered image-token locations required");
    const auto& pd = in.descriptors.at("pixel_values"); require(pd.dtype == "F32" && pd.shape.size() == 2 && pd.shape[1] == 1536, "Original pixel_values [patchRows,1536] F32 required");
    in.patchRows = pd.shape[0]; require(in.patchRows <= 32768, "Original patch rows exceed the native source limit");
    const auto& gd = in.descriptors.at("image_grid_thw"); require(gd.shape.size() == 2 && gd.shape[1] == 3 && gd.shape[0] <= count, "Image grid shape differs");
    auto grid = tensor<int64_t>(gd, "I64", {gd.shape[0], 3}); Json gridJson = Json::array();
    for (uint32_t i = 0; i < gd.shape[0]; ++i) {
        std::array<uint32_t, 3> v{}; for (uint32_t j = 0; j < 3; ++j) { require(grid[i * 3 + j] > 0 && uint64_t(grid[i * 3 + j]) <= 32768, "Invalid grid THW coordinate"); v[j] = uint32_t(grid[i * 3 + j]); }
        in.grids.push_back({v[0], v[1], v[2]}); gridJson.push_back({v[0], v[1], v[2]});
    }
    require(m.at("image_grid_thw") == gridJson, "Grid metadata/raw tensor differ");
    in.visionForecast = VisionModel::forecast(in.patchRows, in.grids);
    require(in.visionForecast.mergedRows == visionRow, "Vision pooler rows must exactly match ordered image-token rows");
    auto positions = tensor<int64_t>(in.descriptors.at("position_ids"), "I64", {3, 1, count});
    std::array<std::vector<uint32_t>*, 3> axes = {&in.positions.temporal, &in.positions.height, &in.positions.width};
    for (uint32_t a = 0; a < 3; ++a) for (uint32_t i = 0; i < count; ++i) {
        auto v = positions[size_t(a) * count + i]; require(v >= 0 && uint64_t(v) < contextLimit, "Invalid three-axis prompt position"); axes[a]->push_back(uint32_t(v)); in.maximumPosition = std::max(in.maximumPosition, uint32_t(v));
    }
    require(uint64_t(in.maximumPosition) + normalOutputLimit < contextLimit, "Three-axis positions must preserve normal decode capacity");
    if (m.contains("positions")) require(unsignedNumber(m.at("positions").at("next_decode_position")) == uint64_t(in.maximumPosition) + 1, "Position metadata/decode coordinate differ");
    for (auto name : {"attention_mask", "mm_token_type_ids", "text_position_ids"}) if (in.descriptors.count(name)) {
        auto values = tensor<int64_t>(in.descriptors.at(name), "I64", {1, count});
        for (uint32_t i = 0; i < count; ++i) {
            if (std::string(name) == "attention_mask") require(values[i] == 1, "B1 prompt requires the full unpadded attention mask");
            else if (std::string(name) == "mm_token_type_ids") require(values[i] == (in.ids[i] == imageToken ? 1 : 0), "Original image modality token map differs");
            else require(values[i] == int64_t(i), "Source B1 text_position_ids must preserve exact causal row order");
        }
    }
    if (in.descriptors.count("rope_deltas")) {
        auto values = tensor<int64_t>(in.descriptors.at("rope_deltas"), "I64", {1, 1});
        require(values[0] >= -int64_t(count) && values[0] < int64_t(contextLimit) && int64_t(count) + values[0] == int64_t(in.maximumPosition) + 1, "Source rope_delta must agree with max prompt axes plus one");
    }
    in.pixels = tensor<float>(pd, "F32", {in.patchRows, 1536});
    require(std::all_of(in.pixels.begin(), in.pixels.end(), [](float v) { return std::isfinite(v); }), "Nonfinite pixel tensor refused before Device creation");
    return in;
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
Json visionEstimate(const VisionForecast& f) {
    return {{"patch_rows", f.patchRows}, {"merged_rows", f.mergedRows}, {"frames", f.frames}, {"maximum_frame_rows", f.maximumFrameRows},
        {"hidden_buffer_bytes", f.hiddenBufferBytes}, {"rotary_buffer_bytes", f.rotaryBufferBytes}, {"attention_score_bytes", f.attentionScoreBytes},
        {"attention_partial_bytes", f.attentionPartialBytes}, {"row_intermediate_bytes", f.rowIntermediateBytes}, {"maximum_single_buffer_bytes", f.maximumSingleBufferBytes},
        {"workspace_upper_bound_bytes", f.workspaceUpperBoundBytes}, {"vision_weight_bytes", f.visionWeightBytes},
        {"attention_dispatches_per_block", f.attentionDispatchesPerBlock}, {"attention_multiply_adds_per_block", f.attentionMultiplyAddsPerBlock}};
}
Json forecast(const Input& in, const Options& o, const DiagnosticRequest* diagnostic) {
    uint64_t caches = 8ull * 2 * contextLimit * 1024 * 4 + 24ull * (8192 * 4 + 32 * 128 * 128) * 4;
    Json result = {{"schema", "chandra.directcompute.forecast.v1"}, {"mode", "forecast"}, {"passed", false}, {"source_forecast_completed", true},
        {"device_created", false}, {"model_uploaded", false}, {"actual_inference", false}, {"input_manifest_sha256", o.manifestSha},
        {"input_tensor_bytes", in.tensorBytes}, {"prompt_tokens", in.ids.size()}, {"context_limit", contextLimit}, {"normal_output_allowance", normalOutputLimit},
        {"diagnostic_token_cap", o.diagnosticCap ? Json(o.diagnosticCap) : Json(nullptr)}, {"image_token_rows", in.visionForecast.mergedRows},
        {"vision", visionEstimate(in.visionForecast)}, {"text_request_cache_bytes", caches}, {"one_prompt_embedding_buffer_bytes", in.ids.size() * textWidth * 4ull},
        {"model_weight_upload_budget_bytes", 12ull * 1024 * 1024 * 1024}, {"pinned_model_file_bytes", 10591220088ull},
        {"limits", "Geometry/source estimates only; source upload ceiling is not observed native live memory or driver fit"},
        {"input_metadata", in.manifest}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
    if (diagnostic) result["diagnostics"] = {{"requested", true}, {"dump_written", false}, {"plan", diagnostic->plan.resolved}, {"plan_sha256", diagnostic->plan.sha256}, {"plan_file", diagnostic->planFile}};
    return result;
}
class NewOutput {
    Handle file;
public:
    explicit NewOutput(const std::filesystem::path& path) : file(CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr)) {}
    void line(const Json& value) {
        auto text = value.dump() + "\n"; require(text.size() <= manifestByteLimit * 4, "Bounded output row required");
        DWORD wrote = 0; require(WriteFile(file.value, text.data(), DWORD(text.size()), &wrote, nullptr) && wrote == text.size(), "Writing complete output failed");
        require(FlushFileBuffers(file.value), "Flushing native output failed");
    }
};
struct Greedy { uint32_t token = 0; float best = 0, second = 0; uint32_t ties = 0; };
Greedy greedy(Device& device, const TextResult& result) {
    require(result.rows == 1 && !result.logits.packedBF16 && result.logits.logicalElements == vocabulary && result.logits.words == vocabulary, "Complete last-position vocabulary logits required");
    auto logits = device.readFloats(result.logits); require(logits.size() == vocabulary, "Full vocabulary readback required");
    // Refuse every nonfinite component before emitting any token. Strict greater
    // preserves the first index when equal maximum values occur, including +/-0.
    require(std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); }), "Nonfinite all-vocabulary logits; failed request must retire");
    Greedy g; g.best = logits[0]; g.second = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 1; i < vocabulary; ++i) {
        if (logits[i] > g.best) { g.second = g.best; g.best = logits[i]; g.token = i; }
        else g.second = std::max(g.second, logits[i]);
    }
    for (float value : logits) if (value == g.best) ++g.ties; return g;
}
Buffer merge(Device& device, const Input& in, const Buffer& text, const Buffer& vision) {
    uint32_t rows = uint32_t(in.ids.size()), visionRows = in.visionForecast.mergedRows;
    require(!text.packedBF16 && !vision.packedBF16 && text.logicalElements == uint64_t(rows) * textWidth && text.words == rows * textWidth && vision.logicalElements == uint64_t(visionRows) * textWidth && vision.words == visionRows * textWidth, "Exact embedding and full vision pooler ABI required");
    auto map = device.words(rows, in.visionRowMap.data()); auto output = device.floats(rows * textWidth);
    struct Params { uint32_t tokens, width, visionRows, firstElement, count, reserved0, reserved1, reserved2; };
    static_assert(sizeof(Params) == 32, "Multimodal merge HLSL constant ABI");
    for (uint32_t first = 0; first < rows * textWidth;) {
        uint32_t count = std::min(rows * textWidth - first, 65535u * 256u);
        Params p{rows, textWidth, visionRows, first, count, 0, 0, 0};
        device.dispatch("runtime/multimodal_merge.hlsl", {&text, &vision, &map}, {&output}, &p, sizeof(p), (count + 255) / 256); first += count;
    }
    device.drain(); return output;
}
Json execute(const Input& in, const Options& o, const DiagnosticRequest* diagnostic) {
    auto start = Clock::now(); Json phases = Json::array(), memories = Json::array(); std::vector<uint32_t> generated;
    Json report = {{"schema", "chandra.directcompute.result.v1"}, {"mode", o.diagnosticCap ? "full_graph_token_diagnostic" : "full_page_generation"}, {"passed", false},
        {"input_manifest_sha256", o.manifestSha}, {"input_metadata", in.manifest}, {"model_revision", modelRevision}, {"prompt_tokens", in.ids.size()}, {"prompt_token_ids", in.ids},
        {"context_limit", contextLimit}, {"normal_output_allowance", normalOutputLimit}, {"diagnostic_token_cap", o.diagnosticCap ? Json(o.diagnosticCap) : Json(nullptr)},
        {"stop_token_ids", in.stopIds}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}, {"request_replayed", false}};
    if (diagnostic) report["diagnostic_dump_requested"] = true;
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
            ModelWeights weights(*device, o.model.wstring()); phase("authenticated_model_import_upload"); report["model_provenance"] = Json::parse(weights.provenanceJson());
            require(report["model_provenance"].at("full_graph_requested") == true && report["model_provenance"].at("omitted_graph_tensors").empty() && report["model_provenance"].at("tie_byte_equality") == true, "Full original graph and authenticated tied output head required");
            if (recorder) recorder->authenticatedModel(report["model_provenance"], report["device_identity"]);
            TextModel text(*device, weights); VisionModel vision(*device, weights); TextRequest request;
            VisionObserver visionObserver; TextObserver textObserver; // Empty unless a dump was explicitly requested.
            if (recorder) {
                visionObserver = [&](const std::string& name, const Buffer& b, uint32_t rows, uint32_t width) { recorder->vision(name, b, rows, width); };
                textObserver = [&](const TextObservation& observation) { recorder->text(observation); };
            }
            Buffer embeddings; TextResult logits;
            try {
                auto visual = vision.forward(in.pixels.data(), in.patchRows, in.grids, visionObserver); phase("complete_vision_forward");
                auto tokens = embedding(*device, weights.at("model.language_model.embed_tokens.weight"), in.ids); phase("text_embedding");
                embeddings = merge(*device, in, tokens, visual.poolerOutput); if (recorder) recorder->merged(embeddings, uint32_t(in.ids.size()), textWidth); phase("ordered_multimodal_merge");
                tokens = {}; visual = {}; device->drain();
                request = text.newRequest(); phase("text_request_cache_creation");
                logits = text.prefill(request, embeddings, in.positions, true, textObserver); phase("text_prefill"); embeddings = {}; device->drain();
                NewOutput tokenRows(o.output / L"generated-tokens.jsonl"); uint32_t limit = o.diagnosticCap ? o.diagnosticCap : normalOutputLimit;
                generated.reserve(limit); bool stopped = false;
                for (uint32_t index = 0; index < limit; ++index) {
                    auto next = greedy(*device, logits); bool stop = std::find(in.stopIds.begin(), in.stopIds.end(), next.token) != in.stopIds.end();
                    generated.push_back(next.token);
                    tokenRows.line({{"generated_index", index}, {"token_id", next.token}, {"stop", stop}, {"best_logit", next.best}, {"runner_up_logit", next.second},
                        {"top_margin", double(next.best) - double(next.second)}, {"maximum_tie_count", next.ties}, {"argmax_tie_policy", "first vocabulary index"},
                        {"logit_storage", "full 248320-value FP32 readback; graph cast boundaries remain explicit"}});
                    if (stop) { stopped = true; break; }
                    if (index + 1 == limit) break;
                    uint32_t coordinate = in.maximumPosition + index + 1;
                    TextPositions position{{coordinate}, {coordinate}, {coordinate}};
                    logits = text.advance(request, next.token, position, textObserver);
                }
                phase("greedy_cached_decode_and_token_readback");
                report["stop_reason"] = stopped ? "stop_token" : (o.diagnosticCap ? "explicit_diagnostic_token_cap" : "normal_output_length");
                report["stop_token_id"] = stopped ? Json(generated.back()) : Json(nullptr);
                report["producer_stop_token_observed"] = stopped; report["generation_completed"] = true;
                report["complete_page_claim"] = false; report["normal_output_allowance_preserved"] = true;
                report["request_cache_tokens_before_retirement"] = request.tokens; report["cached_advances_before_retirement"] = request.generated;
                report["all_vocabulary_finite_observations"] = generated.size(); report["argmax_tie_policy"] = "first vocabulary index";
                logits = {}; text.retire(request); device->drain(); phase("request_retirement_and_drain");
                report["request_failed"] = false; report["request_retired"] = request.retired; report["drained"] = true;
            } catch (...) {
                request.failed = true; text.retire(request); report["request_failed"] = true; report["request_retired"] = request.retired;
                embeddings = {}; logits = {};
                try { device->drain(); report["drained"] = true; } catch (const std::exception& e) { report["drained"] = false; report["drain_error"] = e.what(); report["owned_worker_retirement_required"] = true; }
                throw;
            }
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
    report["generated_token_ids"] = generated; report["generated_tokens_including_stop"] = generated.size();
    std::vector<uint32_t> history = in.ids; history.insert(history.end(), generated.begin(), generated.end()); report["complete_token_history"] = history;
    report["phases"] = phases; report["memory_observations"] = memories; report["duration_wall_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
    report["timing_scope"] = "One native request with authenticated preprepared inputs; excludes original image decoding/processor work and proves no page throughput";
    return report;
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    Options o; bool created = false; auto started = Clock::now();
    try {
        o = options(argc, argv); auto in = input(o); // No Device before all input/hash/geometry/finite guards.
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
