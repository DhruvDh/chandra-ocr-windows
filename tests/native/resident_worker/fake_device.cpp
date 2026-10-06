// Test-only, MPL-2.0. Recording Device/ModelWeights for CPU lifecycle tests of the CLI and worker.
// No GPU, arithmetic or weight bytes: dispatches are traced, never executed. Allocation, dispatch,
// drain, readback and release lines use the diagnostics_fake_device.h format, plus lifecycle markers.
// A full-vocabulary readback returns logits whose argmax is the next CHANDRA_FAKE_TOKENS entry (or a
// NaN for "nan"); this is a control-flow script, not a numerical result. Other knobs:
// CHANDRA_FAKE_FAIL=kind:N (kind dispatch|drain|read|alloc, Nth call throws), CHANDRA_FAKE_GATE=path
// with CHANDRA_FAKE_GATE_READ=N (the Nth vocabulary readback waits for that file), CHANDRA_FAKE_GATE_DISPATCH=N
// (the Nth dispatch waits) or CHANDRA_FAKE_GATE_DRAIN=N (the Nth drain waits), CHANDRA_FAKE_IMPORT_FAIL=1,
// CHANDRA_FAKE_DRAIN_FAILS_AFTER_GATE=1 (every drain after a gate opens throws, like a device that stopped
// completing work) and CHANDRA_FAKE_TRACE=path (trace written at process exit). Every member call checks the owner thread.
#include "../../../ChandraNative/runtime/api.h"
#include "../../../ChandraNative/vendor/nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <vector>

namespace chandra::dc {
namespace {
struct Trace {
    std::mutex lock; std::vector<std::string> lines; uint64_t serial = 0, live = 0;
    std::map<std::string, uint64_t> counts; bool gateOpened = false;
    ~Trace() {
        const char* path = std::getenv("CHANDRA_FAKE_TRACE"); if (!path) return;
        std::ofstream out(path, std::ios::binary); for (const auto& line : lines) out << line << '\n';
    }
};
Trace& trace() { static Trace t; return t; }
void line(const std::string& text) { std::lock_guard<std::mutex> guard(trace().lock); trace().lines.push_back(text); }
std::string digest(const void* data, size_t bytes) {
    uint64_t h = 1469598103934665603ull; auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; i++) { h ^= p[i]; h *= 1099511628211ull; }
    char text[17]; std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h)); return text;
}
float pattern(uint64_t serial, uint64_t index) { return float(int((serial * 131 + index * 7) % 251) - 125); }
// Waits until the test opens the gate file; bounded so an abandoned test cannot hang forever.
void gate(const char* path, const std::string& label) {
    line("gate_wait " + label); struct stat s{};
    { std::ofstream marker(std::string(path) + ".waiting"); marker << label << '\n'; } // Tells the test the Device thread is parked here.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (stat(path, &s) != 0) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("Fake gate was never opened");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    line("gate_open " + label); trace().gateOpened = true;
}
void maybeFail(const std::string& kind) {
    uint64_t n = ++trace().counts[kind]; const char* spec = std::getenv("CHANDRA_FAKE_FAIL"); if (!spec) return;
    std::string s(spec); auto colon = s.find(':');
    if (colon != std::string::npos && s.substr(0, colon) == kind && std::strtoull(s.c_str() + colon + 1, nullptr, 10) == n) {
        line("injected_failure " + kind + " " + std::to_string(n)); throw std::runtime_error("Injected fake device " + kind + " failure");
    }
}
std::vector<std::string> script() {
    std::vector<std::string> result; const char* text = std::getenv("CHANDRA_FAKE_TOKENS"); if (!text) return result;
    std::string s(text); size_t start = 0;
    while (start < s.size()) { size_t end = s.find(',', start); if (end == std::string::npos) end = s.size(); result.push_back(s.substr(start, end - start)); start = end + 1; }
    return result;
}
}
struct Storage {
    uint64_t serial = 0; uint32_t bytes = 0;
    ~Storage() { trace().live -= bytes; line("release " + std::to_string(serial)); }
};
struct Device::Impl {
    std::vector<std::shared_ptr<Storage>> pending; std::thread::id owner = std::this_thread::get_id();
    std::vector<std::string> tokens = script(); uint64_t vocabularyReads = 0;
    void thread() const {
        if (std::this_thread::get_id() != owner) { line("foreign_thread"); throw std::runtime_error("Fake D3D11 actor used from another thread"); }
    }
    void retain(const Buffer& b) {
        if (std::none_of(pending.begin(), pending.end(), [&](const auto& p) { return p == b.storage; })) pending.push_back(b.storage);
    }
    Buffer make(uint32_t count, const void* initial) {
        thread(); maybeFail("alloc");
        if (!count || uint64_t(count) * 4 > 128ull * 1024 * 1024) throw std::runtime_error("Fake buffer must be nonempty and at most 128 MiB");
        auto s = std::make_shared<Storage>(); auto& t = trace(); s->serial = ++t.serial; s->bytes = count * 4; t.live += s->bytes;
        line("alloc " + std::to_string(s->serial) + " words " + std::to_string(count) + " init " + (initial ? digest(initial, size_t(count) * 4) : "none"));
        return {s, count, count, false};
    }
};
Device::Device(const std::wstring& shaders, const std::string& pci, const std::string& luid) : impl(std::make_unique<Impl>()) {
    if (pci.empty()) throw std::runtime_error("Explicit physical PCI identity is required");
    line("device_create " + pci + " " + luid + " shaders " + std::to_string(shaders.size()));
}
Device::~Device() { line("device_destroy"); }
Buffer Device::floats(uint32_t count, const float* initial) { return impl->make(count, initial); }
Buffer Device::words(uint32_t count, const uint32_t* initial) { return impl->make(count, initial); }
void Device::upload(Buffer& b, const void* values, uint32_t bytes) {
    impl->thread(); impl->retain(b); line("upload " + std::to_string(b.storage->serial) + " " + digest(values, bytes));
}
void Device::dispatch(const std::string& shader, const std::vector<const Buffer*>& inputs, const std::vector<Buffer*>& outputs,
                      const void* parameters, uint32_t parameterBytes, uint32_t x, uint32_t y, uint32_t z) {
    impl->thread();
    const char* path = std::getenv("CHANDRA_FAKE_GATE"); const char* at = std::getenv("CHANDRA_FAKE_GATE_DISPATCH");
    if (path && at && std::strtoull(at, nullptr, 10) == trace().counts["dispatch"] + 1) gate(path, "dispatch " + std::string(at));
    maybeFail("dispatch");
    std::string text = "dispatch " + shader + " in";
    for (auto* b : inputs) { if (!b || !b->storage) throw std::runtime_error("Fake null SRV"); text += " " + std::to_string(b->storage->serial); impl->retain(*b); }
    text += " out";
    for (auto* b : outputs) { if (!b || !b->storage) throw std::runtime_error("Fake null UAV"); text += " " + std::to_string(b->storage->serial); impl->retain(*b); }
    text += " params " + (parameterBytes ? digest(parameters, parameterBytes) : "none") + " groups " + std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(z);
    line(text);
}
void Device::drain(uint32_t timeout) {
    impl->thread(); if (!timeout || timeout > 30000) throw std::runtime_error("Bounded drain timeout required");
    if (trace().gateOpened && std::getenv("CHANDRA_FAKE_DRAIN_FAILS_AFTER_GATE")) { line("injected_failure drain after gate"); throw std::runtime_error("Injected fake drain deadline after gate"); }
    const char* path = std::getenv("CHANDRA_FAKE_GATE"); const char* at = std::getenv("CHANDRA_FAKE_GATE_DRAIN");
    if (path && at && std::strtoull(at, nullptr, 10) == trace().counts["drain"] + 1) gate(path, "drain " + std::string(at));
    maybeFail("drain"); line("drain"); impl->pending.clear();
}
std::vector<uint32_t> Device::readWords(const Buffer& b) {
    impl->thread(); maybeFail("read");
    if (!b.storage) throw std::runtime_error("Fake readback of empty buffer");
    line("read " + std::to_string(b.storage->serial)); impl->pending.clear();
    std::vector<uint32_t> result(b.words);
    if (b.words == 248320 && b.logicalElements == 248320 && !b.packedBF16) {
        uint64_t index = impl->vocabularyReads++;
        const char* path = std::getenv("CHANDRA_FAKE_GATE"); const char* at = std::getenv("CHANDRA_FAKE_GATE_READ");
        if (path && at && std::strtoull(at, nullptr, 10) == index + 1) gate(path, "read " + std::to_string(index + 1));
        if (index >= impl->tokens.size()) throw std::runtime_error("Fake token script exhausted");
        const std::string& next = impl->tokens[size_t(index)];
        for (uint32_t i = 0; i < b.words; i++) { float v = -1.0f - float(i % 13) * 0.5f; std::memcpy(&result[i], &v, 4); }
        if (next == "nan") { float v = std::numeric_limits<float>::quiet_NaN(); std::memcpy(&result[17], &v, 4); }
        else { uint32_t token = uint32_t(std::stoul(next)); float v = 4.0f; std::memcpy(&result[token], &v, 4); }
        line("logits " + std::to_string(index + 1) + " " + next);
        return result;
    }
    for (uint32_t i = 0; i < b.words; i++) { float v = pattern(b.storage->serial, i); std::memcpy(&result[i], &v, 4); }
    return result;
}
std::vector<float> Device::readFloats(const Buffer& b) {
    auto words = readWords(b); std::vector<float> result(words.size()); std::memcpy(result.data(), words.data(), words.size() * 4); return result;
}
void Device::zero(Buffer& b) { impl->thread(); impl->retain(b); line("zero " + std::to_string(b.storage->serial)); }
uint64_t Device::trackedBufferBytes() const { impl->thread(); line("tracked " + std::to_string(trace().live)); return trace().live; }
std::string Device::identityJson() const { impl->thread(); return "{\"fake_test_device\":true,\"pci_bdf\":\"fake\"}"; }
std::string Device::memoryJson() const { impl->thread(); line("memory"); return "{\"fake_test_device\":true}"; }
void Device::beginProfile() { impl->thread(); }
std::string Device::finishProfile() { impl->thread(); return "{}"; }

struct ModelWeights::Impl { std::map<std::string, Weight> weights; uint64_t bytes = 0; };
ModelWeights::ModelWeights(Device& d, const std::wstring&, uint64_t, const std::vector<std::string>& selected) : impl(std::make_unique<Impl>()) {
    line("model_import_begin selected " + std::to_string(selected.size()));
    if (std::getenv("CHANDRA_FAKE_IMPORT_FAIL")) { line("model_import_failed"); throw std::runtime_error("Injected fake model import failure"); }
    auto add = [&](const std::string& name, uint32_t rows, uint32_t cols) {
        Weight w; w.rows = rows; w.cols = cols; w.bf16 = true; const uint32_t perShard = uint32_t((64ull << 20) / cols);
        for (uint32_t first = 0; first < rows;) {
            const uint32_t n = std::min(rows - first, perShard); const uint64_t elements = uint64_t(n) * cols;
            Buffer b = d.words(uint32_t((elements + 1) / 2)); b.logicalElements = elements; b.packedBF16 = true; impl->bytes += uint64_t(b.words) * 4;
            w.shards.push_back(b); w.shardFirstRows.push_back(first); first += n;
        }
        impl->weights.emplace(name, std::move(w));
    };
    const std::string t = "model.language_model.";
    for (uint32_t l = 0; l < 32; l++) {
        const auto b = t + "layers." + std::to_string(l) + ".";
        add(b + "input_layernorm.weight", 1, 2560); add(b + "post_attention_layernorm.weight", 1, 2560);
        add(b + "mlp.gate_proj.weight", 9216, 2560); add(b + "mlp.up_proj.weight", 9216, 2560); add(b + "mlp.down_proj.weight", 2560, 9216);
        if (l % 4 == 3) {
            add(b + "self_attn.q_proj.weight", 8192, 2560); add(b + "self_attn.k_proj.weight", 1024, 2560); add(b + "self_attn.v_proj.weight", 1024, 2560);
            add(b + "self_attn.q_norm.weight", 1, 256); add(b + "self_attn.k_norm.weight", 1, 256); add(b + "self_attn.o_proj.weight", 2560, 4096);
        } else {
            add(b + "linear_attn.in_proj_qkv.weight", 8192, 2560); add(b + "linear_attn.in_proj_z.weight", 4096, 2560);
            add(b + "linear_attn.in_proj_a.weight", 32, 2560); add(b + "linear_attn.in_proj_b.weight", 32, 2560);
            add(b + "linear_attn.conv1d.weight", 8192, 4); add(b + "linear_attn.A_log", 1, 32); add(b + "linear_attn.dt_bias", 1, 32);
            add(b + "linear_attn.norm.weight", 1, 128); add(b + "linear_attn.out_proj.weight", 2560, 4096);
        }
    }
    add(t + "norm.weight", 1, 2560); add(t + "embed_tokens.weight", 248320, 2560);
    const std::string v = "model.visual.";
    add(v + "patch_embed.proj.weight", 1024, 1536); add(v + "patch_embed.proj.bias", 1, 1024); add(v + "pos_embed.weight", 2304, 1024);
    for (uint32_t i = 0; i < 24; i++) {
        const auto b = v + "blocks." + std::to_string(i) + ".";
        for (auto n : {"norm1.", "norm2."}) { add(b + n + "weight", 1, 1024); add(b + n + "bias", 1, 1024); }
        add(b + "attn.qkv.weight", 3072, 1024); add(b + "attn.qkv.bias", 1, 3072); add(b + "attn.proj.weight", 1024, 1024); add(b + "attn.proj.bias", 1, 1024);
        add(b + "mlp.linear_fc1.weight", 4096, 1024); add(b + "mlp.linear_fc1.bias", 1, 4096); add(b + "mlp.linear_fc2.weight", 1024, 4096); add(b + "mlp.linear_fc2.bias", 1, 1024);
    }
    add(v + "merger.norm.weight", 1, 1024); add(v + "merger.norm.bias", 1, 1024);
    add(v + "merger.linear_fc1.weight", 4096, 4096); add(v + "merger.linear_fc1.bias", 1, 4096);
    add(v + "merger.linear_fc2.weight", 2560, 4096); add(v + "merger.linear_fc2.bias", 1, 2560);
    line("model_import_end tensors " + std::to_string(impl->weights.size()) + " last_serial " + std::to_string(trace().serial) + " bytes " + std::to_string(impl->bytes));
}
ModelWeights::~ModelWeights() { line("model_release_begin"); impl.reset(); line("model_release_end"); }
const Weight& ModelWeights::at(const std::string& name) const {
    auto it = impl->weights.find(name); if (it == impl->weights.end()) throw std::out_of_range("Fake weight absent: " + name); return it->second;
}
bool ModelWeights::contains(const std::string& name) const { return impl->weights.count(name) != 0; }
std::string ModelWeights::provenanceJson() const {
    return nlohmann::json{{"fake_test_device", true}, {"schema", "fake.model-weights"}, {"revision", "af93b47dba1b47b6640c86ccf487ed2260ab9a09"},
        {"model_bytes", 10591220088ull}, {"model_sha256", std::string(64, '0')}, {"config_sha256", std::string(64, '1')}, {"full_graph_requested", true}, {"omitted_graph_tensors", nlohmann::json::array()},
        {"tie_byte_equality", true}, {"uploaded_storage_bytes", impl->bytes}, {"uploaded_tensor_names", impl->weights.size()}}.dump();
}
}
