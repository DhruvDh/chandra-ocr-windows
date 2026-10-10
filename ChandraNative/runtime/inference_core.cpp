// New code, MPL-2.0. Connected native page path; unqualified until actual graph/OCR checks.
// Moved from inference.cpp so the CLI and the resident worker share one authenticated request path.
#include "inference_core.h"
#include "padded32_opt_in.h"
#include <psapi.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <set>

namespace chandra::dc::experimental { const char* gemvB1Selection(); }

namespace chandra::dc::inference {
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
Sha256::Sha256() {
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
Sha256::~Sha256() { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
void Sha256::update(const void* data, uint32_t bytes) { require(BCryptHashData(hash, static_cast<PUCHAR>(const_cast<void*>(data)), bytes, 0) >= 0, "Streaming SHA-256 failed"); }
std::string Sha256::finish() {
    std::array<UCHAR, 32> result{}; require(BCryptFinishHash(hash, result.data(), ULONG(result.size()), 0) >= 0, "Finishing SHA-256 failed");
    static const char* hex = "0123456789abcdef"; std::string text; text.reserve(64);
    for (auto c : result) { text.push_back(hex[c >> 4]); text.push_back(hex[c & 15]); } return text;
}
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
Json hostMemory() {
    PROCESS_MEMORY_COUNTERS_EX memory{}; memory.cb = sizeof(memory);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)))
        return {{"available", false}, {"win32_error", GetLastError()}};
    return {{"available", true}, {"working_set_bytes", memory.WorkingSetSize}, {"peak_working_set_bytes", memory.PeakWorkingSetSize},
        {"private_usage_bytes", memory.PrivateUsage}, {"scope", "Current native process observation; external Job/worker closure remains root-owned"}};
}
Input authenticateInput(const std::filesystem::path& manifest, const std::string& manifestSha) {
    const uint16_t endian = 1; require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "Native input ABI requires little-endian host storage");
    Input in; auto n = std::filesystem::file_size(manifest); require(n && n <= manifestByteLimit, "Bounded manifest required");
    std::vector<char> bytes(static_cast<size_t>(n)); authenticatedRead(manifest, bytes.data(), n, manifestSha); in.manifest = strictJson(bytes);
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
    auto root = std::filesystem::canonical(manifest.parent_path()); authenticateCaller(generation, root);
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
Json visionEstimate(const VisionForecast& f) {
    return {{"patch_rows", f.patchRows}, {"merged_rows", f.mergedRows}, {"frames", f.frames}, {"maximum_frame_rows", f.maximumFrameRows},
        {"hidden_buffer_bytes", f.hiddenBufferBytes}, {"rotary_buffer_bytes", f.rotaryBufferBytes}, {"attention_score_bytes", f.attentionScoreBytes},
        {"attention_partial_bytes", f.attentionPartialBytes}, {"row_intermediate_bytes", f.rowIntermediateBytes}, {"maximum_single_buffer_bytes", f.maximumSingleBufferBytes},
        {"workspace_upper_bound_bytes", f.workspaceUpperBoundBytes}, {"vision_weight_bytes", f.visionWeightBytes},
        {"attention_dispatches_per_block", f.attentionDispatchesPerBlock}, {"attention_multiply_adds_per_block", f.attentionMultiplyAddsPerBlock}};
}
Json forecastReport(const Input& in, const std::string& manifestSha, uint32_t diagnosticCap) {
    uint64_t caches = 8ull * 2 * contextLimit * 1024 * 4 + 24ull * (8192 * 4 + 32 * 128 * 128) * 4;
    return {{"schema", "chandra.directcompute.forecast.v1"}, {"mode", "forecast"}, {"passed", false}, {"source_forecast_completed", true},
        {"device_created", false}, {"model_uploaded", false}, {"actual_inference", false}, {"input_manifest_sha256", manifestSha},
        {"input_tensor_bytes", in.tensorBytes}, {"prompt_tokens", in.ids.size()}, {"context_limit", contextLimit}, {"normal_output_allowance", normalOutputLimit},
        {"diagnostic_token_cap", diagnosticCap ? Json(diagnosticCap) : Json(nullptr)}, {"image_token_rows", in.visionForecast.mergedRows},
        {"vision", visionEstimate(in.visionForecast)}, {"text_request_cache_bytes", caches}, {"one_prompt_embedding_buffer_bytes", in.ids.size() * textWidth * 4ull},
        {"model_weight_upload_budget_bytes", 12ull * 1024 * 1024 * 1024}, {"pinned_model_file_bytes", 10591220088ull},
        {"limits", "Geometry/source estimates only; source upload ceiling is not observed native live memory or driver fit"},
        {"input_metadata", in.manifest}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}};
}
Json executionReport(const Input& in, const std::string& manifestSha, uint32_t diagnosticCap) {
    return {{"schema", "chandra.directcompute.result.v1"}, {"mode", diagnosticCap ? "full_graph_token_diagnostic" : "full_page_generation"}, {"passed", false},
        {"input_manifest_sha256", manifestSha}, {"input_metadata", in.manifest}, {"model_revision", modelRevision}, {"prompt_tokens", in.ids.size()}, {"prompt_token_ids", in.ids},
        {"context_limit", contextLimit}, {"normal_output_allowance", normalOutputLimit}, {"diagnostic_token_cap", diagnosticCap ? Json(diagnosticCap) : Json(nullptr)},
        {"stop_token_ids", in.stopIds}, {"qualified_full_graph", false}, {"qualified_OCR", false}, {"performance_claim", false}, {"request_replayed", false}};
}
void finishExecutionReport(Json& report, const Input& in, const std::vector<uint32_t>& generated, const Json& phases, const Json& memories, Clock::time_point start) {
    report["generated_token_ids"] = generated; report["generated_tokens_including_stop"] = generated.size();
    std::vector<uint32_t> history = in.ids; history.insert(history.end(), generated.begin(), generated.end()); report["complete_token_history"] = history;
    report["phases"] = phases; report["memory_observations"] = memories; report["duration_wall_seconds"] = std::chrono::duration<double>(Clock::now() - start).count();
    report["timing_scope"] = "One native request with authenticated preprepared inputs; excludes original image decoding/processor work and proves no page throughput";
}
NewOutput::NewOutput(const std::filesystem::path& path) : file(CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr)) {}
void NewOutput::line(const Json& value) {
    auto text = value.dump() + "\n"; require(text.size() <= manifestByteLimit * 4, "Bounded output row required");
    DWORD wrote = 0; require(WriteFile(file.value, text.data(), DWORD(text.size()), &wrote, nullptr) && wrote == text.size(), "Writing complete output failed");
    require(FlushFileBuffers(file.value), "Flushing native output failed");
}
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
    for (float value : logits) if (value == g.best) ++g.ties;
    return g;
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
void generate(Device& device, ModelWeights& weights, const Input& in, uint32_t diagnosticCap, const std::filesystem::path& output,
    Json& report, std::vector<uint32_t>& generated, const RequestHooks& hooks) {
    auto phase = [&](const char* name) { if (hooks.phase) hooks.phase(name); };
    auto checkpoint = [&](const char* name) { if (hooks.checkpoint) hooks.checkpoint(name); };
    TextModel text(device, weights); VisionModel vision(device, weights); TextRequest request;
    Buffer embeddings; TextResult logits;
    try {
        auto visual = vision.forward(in.pixels.data(), in.patchRows, in.grids, hooks.vision); phase("complete_vision_forward"); checkpoint("complete_vision_forward");
        auto tokens = embedding(device, weights.at("model.language_model.embed_tokens.weight"), in.ids); phase("text_embedding"); checkpoint("text_embedding");
        embeddings = merge(device, in, tokens, visual.poolerOutput); if (hooks.merged) hooks.merged(embeddings, uint32_t(in.ids.size())); phase("ordered_multimodal_merge");
        tokens = {}; visual = {}; device.drain(); checkpoint("ordered_multimodal_merge");
        request = text.newRequest(); phase("text_request_cache_creation"); checkpoint("text_request_cache_creation");
        logits = text.prefill(request, embeddings, in.positions, true, hooks.text); phase("text_prefill"); embeddings = {}; device.drain(); checkpoint("text_prefill");
        NewOutput tokenRows(output / L"generated-tokens.jsonl"); uint32_t limit = diagnosticCap ? diagnosticCap : normalOutputLimit;
        generated.reserve(limit); bool stopped = false;
        for (uint32_t index = 0; index < limit; ++index) {
            auto next = greedy(device, logits); bool stop = std::find(in.stopIds.begin(), in.stopIds.end(), next.token) != in.stopIds.end();
            generated.push_back(next.token);
            Json row = {{"generated_index", index}, {"token_id", next.token}, {"stop", stop}, {"best_logit", next.best}, {"runner_up_logit", next.second},
                {"top_margin", double(next.best) - double(next.second)}, {"maximum_tie_count", next.ties}, {"argmax_tie_policy", "first vocabulary index"},
                {"logit_storage", "full 248320-value FP32 readback; graph cast boundaries remain explicit"}};
            tokenRows.line(row); if (hooks.token) hooks.token(index, row);
            if (stop) { stopped = true; break; }
            if (index + 1 == limit) break;
            checkpoint("decode_step");
            uint32_t coordinate = in.maximumPosition + index + 1;
            TextPositions position{{coordinate}, {coordinate}, {coordinate}};
            logits = text.advance(request, next.token, position, hooks.text);
        }
        phase("greedy_cached_decode_and_token_readback");
        report["stop_reason"] = stopped ? "stop_token" : (diagnosticCap ? "explicit_diagnostic_token_cap" : "normal_output_length");
        report["stop_token_id"] = stopped ? Json(generated.back()) : Json(nullptr);
        report["producer_stop_token_observed"] = stopped; report["generation_completed"] = true;
        report["complete_page_claim"] = false; report["normal_output_allowance_preserved"] = true;
        report["request_cache_tokens_before_retirement"] = request.tokens; report["cached_advances_before_retirement"] = request.generated;
        report["all_vocabulary_finite_observations"] = generated.size(); report["argmax_tie_policy"] = "first vocabulary index";
        logits = {}; text.retire(request); device.drain(); phase("request_retirement_and_drain");
        report["request_failed"] = false; report["request_retired"] = request.retired; report["drained"] = true;
    } catch (...) {
        request.failed = true; text.retire(request); report["request_failed"] = true; report["request_retired"] = request.retired;
        embeddings = {}; logits = {};
        try { device.drain(); report["drained"] = true; } catch (const std::exception& e) { report["drained"] = false; report["drain_error"] = e.what(); report["owned_worker_retirement_required"] = true; }
        throw;
    }
}
void generateCohort(Device& device, ModelWeights& weights, const std::array<const Input*,2>& inputs,
    const std::array<std::filesystem::path,2>& outputs, std::array<Json,2>& reports,
    std::array<std::vector<uint32_t>,2>& generated, std::array<TextRequest,2>& requests,
    const std::array<RequestHooks,2>& hooks, bool explicitOrderedB2, uint32_t pageCount) {
    require(explicitOrderedB2 && experimental::gemmPadded32Enabled() &&
        std::strcmp(experimental::gemvB1Selection(), "ordered") == 0,
        "Cohort requires explicit B2, padded32 and captured ordered B1 before model work");
    require(pageCount>=1 && pageCount<=2 && inputs[0] && (pageCount==1 || (inputs[1] && outputs[0]!=outputs[1])), "One or two input owners and distinct paired outputs required");
    for (uint32_t slot=0; slot<pageCount; ++slot) {
        const auto& in=*inputs[slot];
        require((in.manifest.value("source",Json(nullptr))=="chandra.native-endpoint.request.v1" ||
            (in.ids.size()==3617 && in.manifest.at("image").at("dimensions")==Json({1624,2100}))) &&
            in.manifest.at("generation").at("context_limit")==contextLimit &&
            in.manifest.at("generation").at("max_output_tokens")==normalOutputLimit,
            "Cohort retains original diagnostic inputs or authenticated production input and normal capacities");
        for (const auto& c:requests[slot].layers)
            require(!c.keys.storage && !c.values.storage && !c.conv.storage && !c.recurrent.storage,
                "Cohort owner still holds a prior cache; no reuse after failure");
        require(generated[slot].empty(), "Fresh generated history owner required");
    }
    TextModel text(device, weights); VisionModel vision(device, weights);
    std::array<Buffer,2> embeddings;
    std::array<TextResult,2> logits;
    std::array<std::unique_ptr<NewOutput>,2> journals;
    auto phase=[&](uint32_t slot,const char* name){if(hooks[slot].phase)hooks[slot].phase(name);};
    auto checkpoint=[&](uint32_t slot,const char* name){if(hooks[slot].checkpoint)hooks[slot].checkpoint(name);};
    try {
        // Both CREATE_NEW journal owners are opened before the first per-page model operation.
        for (uint32_t slot=0; slot<pageCount; ++slot) {
            journals[slot]=std::make_unique<NewOutput>(outputs[slot]/L"generated-tokens.jsonl");
            generated[slot].reserve(normalOutputLimit);
        }
        for (uint32_t slot=0; slot<pageCount; ++slot) {
            const auto& in=*inputs[slot];
            checkpoint(slot,"before_vision_forward");
            auto visual=vision.forward(in.pixels.data(),in.patchRows,in.grids,hooks[slot].vision);
            phase(slot,"complete_vision_forward");checkpoint(slot,"complete_vision_forward");
            auto tokens=embedding(device,weights.at("model.language_model.embed_tokens.weight"),in.ids);
            phase(slot,"text_embedding");checkpoint(slot,"text_embedding");
            embeddings[slot]=merge(device,in,tokens,visual.poolerOutput);
            if(hooks[slot].merged)hooks[slot].merged(embeddings[slot],uint32_t(in.ids.size()));
            phase(slot,"ordered_multimodal_merge");tokens={};visual={};device.drain();checkpoint(slot,"ordered_multimodal_merge");
            requests[slot]=text.newRequest();phase(slot,"text_request_cache_creation");checkpoint(slot,"text_request_cache_creation");
            logits[slot]=text.prefill(requests[slot],embeddings[slot],in.positions,true,hooks[slot].text);
            phase(slot,"text_prefill");embeddings[slot]={};device.drain();checkpoint(slot,"text_prefill");
        }
        std::array<bool,2> active={true,pageCount==2};
        while(active[0] || active[1]) {
            std::array<TextDecodeSlot,2> call;
            std::array<TextObserver,2> observers;
            for(uint32_t slot=0;slot<pageCount;++slot) {
                call[slot].request=&requests[slot];
                if(!active[slot])continue;
                const auto& in=*inputs[slot];auto next=greedy(device,logits[slot]);
                const uint32_t index=uint32_t(generated[slot].size());
                const bool stop=std::find(in.stopIds.begin(),in.stopIds.end(),next.token)!=in.stopIds.end();
                generated[slot].push_back(next.token);
                Json row={{"generated_index",index},{"token_id",next.token},{"stop",stop},{"best_logit",next.best},{"runner_up_logit",next.second},
                    {"top_margin",double(next.best)-double(next.second)},{"maximum_tie_count",next.ties},{"argmax_tie_policy","first vocabulary index"},
                    {"logit_storage","full 248320-value FP32 readback; graph cast boundaries remain explicit"}};
                journals[slot]->line(row);if(hooks[slot].token)hooks[slot].token(index,row);
                if(stop || generated[slot].size()==normalOutputLimit) {
                    active[slot]=false;
                    reports[slot]["stop_reason"]=stop?"stop_token":"normal_output_length";
                    reports[slot]["stop_token_id"]=stop?Json(next.token):Json(nullptr);
                    reports[slot]["producer_stop_token_observed"]=stop;
                    reports[slot]["generation_completed"]=true;
                    continue;
                }
                uint32_t coordinate=in.maximumPosition+index+1;
                call[slot].tokenId=next.token;call[slot].position={{coordinate},{coordinate},{coordinate}};
                call[slot].active=true;observers[slot]=hooks[slot].text;
            }
            if(!active[0] && !active[1])break;
            // Cancellation is group-scoped and observed only at these host/observer boundaries.
            for(uint32_t slot=0;slot<pageCount;++slot)checkpoint(slot,"decode_step");
            if(pageCount==1)logits[0]=text.advance(requests[0],call[0].tokenId,call[0].position,observers[0]);
            else {auto next=text.advanceCohort(call,true,observers);
                for(uint32_t slot=0;slot<pageCount;++slot)if(next.active[slot])logits[slot]=std::move(next.slots[slot]);}
        }
        for(uint32_t slot=0;slot<pageCount;++slot) {
            phase(slot,"greedy_cached_decode_and_token_readback");
            auto& report=reports[slot];report["generation_completed"]=true;report["complete_page_claim"]=false;
            report["normal_output_allowance_preserved"]=true;report["request_cache_tokens_before_retirement"]=requests[slot].tokens;
            report["cached_advances_before_retirement"]=requests[slot].generated;
            report["all_vocabulary_finite_observations"]=generated[slot].size();report["argmax_tie_policy"]="first vocabulary index";
        }
        device.drain(); // Prove completion before any cache/result owner is released.
        embeddings={};logits={};
        for(uint32_t slot=0;slot<pageCount;++slot)text.retire(requests[slot]);
        device.drain();
        for(uint32_t slot=0;slot<pageCount;++slot) {
            reports[slot]["request_failed"]=false;reports[slot]["request_retired"]=true;reports[slot]["drained"]=true;
            phase(slot,"request_retirement_and_drain");
        }
    } catch(...) {
        const auto original=std::current_exception();
        for(uint32_t slot=0;slot<pageCount;++slot){auto& r=requests[slot];r.failed=true;for(auto& c:r.layers)c.failed=true;}
        bool drained=false;std::exception_ptr drainFailure;
        try {device.drain();drained=true;}catch(...){drainFailure=std::current_exception();}
        if(drained){embeddings={};logits={};for(uint32_t slot=0;slot<pageCount;++slot)text.retire(requests[slot]);}
        // Reporting cannot replace the original graph/journal/cancellation exception.
        try {for(uint32_t slot=0;slot<pageCount;++slot){reports[slot]["request_failed"]=true;reports[slot]["request_retired"]=requests[slot].retired;
            reports[slot]["drained"]=drained;reports[slot]["owned_worker_retirement_required"]=!drained;
            if(!drained){try{std::rethrow_exception(drainFailure);}catch(const std::exception& e){reports[slot]["drain_error"]=e.what();}catch(...){reports[slot]["drain_error"]="Non-standard drain failure";}}}}
        catch(...) {}
        std::rethrow_exception(original);
    }
}
}
