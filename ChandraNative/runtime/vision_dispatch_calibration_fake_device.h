// New ChandraNative code, MPL-2.0. Test-only CPU Device for the vision dispatch calibration; include in
// exactly one TU. Execute mode runs C++ transliterations of the four production vision_attention shaders
// with D3D11 binary32 rules (sign-preserving subnormal flush) and bounds-checks every raw load/store;
// record mode only traces dispatches, for the unchanged production graph. The transliterations are
// evidence about algorithm, addressing and host control flow, not about fxc, the Intel driver or A770.
#pragma once
#include "api.h"
#include "vision_dispatch_calibration.h"
#include <array>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace chandra::dc {
struct Accounting { uint64_t bytes = 0, peak = 0; };
struct Storage {
    std::vector<uint32_t> data; std::shared_ptr<Accounting> account; uint64_t id = 0; uint32_t bytes = 0;
    ~Storage() { if (account) account->bytes -= bytes; }
};
}

namespace vdc_fake {
using chandra::vision_calibration::Json;
using chandra::vision_calibration::bits;
using chandra::vision_calibration::fromBits;
enum class Fault { none, scoresIgnoreSequenceStart, softmaxDropLastLane, softmaxPerturbEvenKeys, valuesSkipLastKey,
                   reduceSkipLastTile, reduceIgnoreRowFirst };
// giantError throws a receiptCap+1-byte message; giantMetadata adds a receiptCap-byte nested profile field.
enum class ProfileFault { none, disjointThrows, disjointReported, missing, wrongGroups, negative, giantError, giantMetadata };
struct Event {
    std::string kind, shader; std::vector<uint32_t> params; std::array<uint32_t, 3> groups{{0, 0, 0}};
    std::vector<uint64_t> inputs, outputs; std::vector<uint32_t> inputWords, outputWords; uint64_t id = 0, words = 0;
};
struct Config {
    bool execute = true; Fault fault = Fault::none; double expPerturbation = 0;
    ProfileFault profileFault = ProfileFault::none; uint64_t profileFaultDispatch = 0; // 1-based dispatch serial
    std::function<double(const std::string&, uint64_t)> gpuMilliseconds = [](const std::string&, uint64_t) { return 0.25; };
    // Runs inside beginProfile after its drain (profile preparation), with the 1-based window number.
    std::function<void(uint64_t)> beginProfileHook;
};
inline Config& config() { static Config value; return value; }
inline std::vector<Event>& events() { static std::vector<Event> value; return value; }
inline uint64_t& dispatchSerial() { static uint64_t value = 0; return value; }
inline uint64_t& flushes() { static uint64_t value = 0; return value; }
inline uint64_t& profileBegins() { static uint64_t value = 0; return value; }
inline void check(bool ok, const std::string& message) { if (!ok) throw std::logic_error("FAKE DEVICE CHECK FAILED: " + message); }

// D3D11 binary32: subnormal operands and results flush to sign-preserved zero.
inline float ftz(float x) { if (x != 0.0f && std::fabs(x) < FLT_MIN) { ++flushes(); return std::copysign(0.0f, x); } return x; }
inline float fmul(float a, float b) { const float r = ftz(a) * ftz(b); return ftz(r); }
inline float fadd(float a, float b) { const float r = ftz(a) + ftz(b); return ftz(r); }
inline float fsub(float a, float b) { const float r = ftz(a) - ftz(b); return ftz(r); }
inline float fdiv(float a, float b) { const float r = ftz(a) / ftz(b); return ftz(r); }
inline float fmax32(float a, float b) { return std::fmax(ftz(a), ftz(b)); }
// Model of fxc's lowering: mul by fl32(log2 e) = 1.44269502f, then the base-2 exp instruction.
inline float fexp(float x) { return ftz(std::exp2(fmul(x, 1.44269502f))); }

struct Raw {
    std::vector<uint32_t>& data; const char* name; std::vector<uint8_t>* written = nullptr;
    uint32_t load(uint64_t address) const {
        check(address % 4 == 0 && address / 4 < data.size(), std::string("out-of-bounds raw load from ") + name);
        return data[size_t(address / 4)];
    }
    void store(uint64_t address, uint32_t value) {
        check(address % 4 == 0 && address / 4 < data.size(), std::string("out-of-bounds raw store to ") + name);
        if (written) { check(!(*written)[size_t(address / 4)], std::string("duplicate store in one dispatch to ") + name); (*written)[size_t(address / 4)] = 1; }
        data[size_t(address / 4)] = value;
    }
};
using chandra::vision_calibration::AttentionParams;
using chandra::vision_calibration::SoftmaxParams;
using chandra::vision_calibration::ReduceParams;

// vision_attention_scores.hlsl: groups (key tiles, query heads), 128 lanes.
inline void scores(const AttentionParams& p, uint32_t gx, uint32_t gy, const Raw& q, const Raw& k, Raw& out) {
    for (uint32_t y = 0; y < gy; ++y)
        for (uint32_t x = 0; x < gx; ++x) {
            const uint32_t qh = y, qr = qh / 16, h = qh % 16;
            float queryVector[64];
            for (uint32_t lane = 0; lane < 64; ++lane)
                queryVector[lane] = fromBits(q.load((uint64_t(p.sequenceStart + p.queryStart + qr) * 1024 + h * 64 + lane) * 4));
            for (uint32_t lane = 0; lane < 128; ++lane) { // After GroupMemoryBarrierWithGroupSync.
                const uint32_t key = x * 128 + lane; float sum = 0.0f;
                if (key < p.sequenceLength) {
                    const uint32_t kr = config().fault == Fault::scoresIgnoreSequenceStart ? key : p.sequenceStart + key;
                    for (uint32_t c = 0; c < 64; ++c) sum = fadd(sum, fmul(queryVector[c], fromBits(k.load((uint64_t(kr) * 1024 + h * 64 + c) * 4))));
                }
                const float value = fmul(sum, 0.125f);
                if (key < p.sequenceLength) out.store((uint64_t(qh) * p.sequenceLength + key) * 4, bits(value));
            }
        }
}
// vision_attention_softmax.hlsl: one group per query head row, 128 lanes, groupshared r[128].
inline void softmax(const SoftmaxParams& p, uint32_t gx, Raw& s) {
    check(gx == p.queryHeads, "softmax group count differs from queryHeads");
    for (uint32_t g = 0; g < gx; ++g) {
        float r[128];
        for (uint32_t lane = 0; lane < 128; ++lane) {
            float maximum = -3.402823466e38f;
            for (uint32_t key = lane; key < p.sequenceLength; key += 128)
                maximum = fmax32(maximum, fromBits(s.load((uint64_t(g) * p.sequenceLength + key) * 4)));
            r[lane] = maximum;
        }
        for (uint32_t step = 64; step; step >>= 1) for (uint32_t lane = 0; lane < step; ++lane) r[lane] = fmax32(r[lane], r[lane + step]);
        const float maximum = r[0];
        for (uint32_t lane = 0; lane < 128; ++lane) {
            float sum = 0.0f;
            for (uint32_t key = lane; key < p.sequenceLength; key += 128) {
                const uint64_t index = (uint64_t(g) * p.sequenceLength + key) * 4;
                float e = fexp(fsub(fromBits(s.load(index)), maximum));
                if (config().fault == Fault::softmaxPerturbEvenKeys && key % 2 == 0) e = float(double(e) * (1 + config().expPerturbation));
                s.store(index, bits(e)); sum = fadd(sum, e);
            }
            r[lane] = config().fault == Fault::softmaxDropLastLane && lane == 127 ? 0.0f : sum;
        }
        for (uint32_t step = 64; step; step >>= 1) for (uint32_t lane = 0; lane < step; ++lane) r[lane] = fadd(r[lane], r[lane + step]);
        const float denominator = r[0];
        for (uint32_t lane = 0; lane < 128; ++lane)
            for (uint32_t key = lane; key < p.sequenceLength; key += 128) {
                const uint64_t index = (uint64_t(g) * p.sequenceLength + key) * 4;
                s.store(index, bits(fdiv(fromBits(s.load(index)), denominator)));
            }
    }
}
// vision_attention_values.hlsl: groups (key tiles, query heads), 64 lanes (one per head column).
inline void values(const AttentionParams& p, uint32_t gx, uint32_t gy, const Raw& s, const Raw& v, Raw& out) {
    for (uint32_t y = 0; y < gy; ++y)
        for (uint32_t tile = 0; tile < gx; ++tile)
            for (uint32_t c = 0; c < 64; ++c) {
                const uint32_t qh = y, h = qh % 16, qr = qh / 16;
                uint32_t keys = std::min(128u, p.sequenceLength - tile * 128);
                if (config().fault == Fault::valuesSkipLastKey && tile + 1 == gx) --keys;
                float sum = 0.0f;
                for (uint32_t j = 0; j < keys; ++j) {
                    const uint32_t kr = tile * 128 + j;
                    const float probability = fromBits(s.load((uint64_t(qh) * p.sequenceLength + kr) * 4));
                    sum = fadd(sum, fmul(probability, fromBits(v.load((uint64_t(p.sequenceStart + kr) * 1024 + h * 64 + c) * 4))));
                }
                out.store((uint64_t(tile * p.queryCount + qr) * 1024 + h * 64 + c) * 4, bits(sum));
            }
}
// vision_attention_reduce.hlsl: 256-thread groups over count = queries*1024 outputs.
inline void reduce(const ReduceParams& p, uint32_t gx, const Raw& partials, Raw& out) {
    const uint32_t tiles = config().fault == Fault::reduceSkipLastTile ? p.tileCount - 1 : p.tileCount;
    const uint32_t rowFirst = config().fault == Fault::reduceIgnoreRowFirst ? 0 : p.rowFirst;
    for (uint32_t x = 0; x < gx * 256; ++x) {
        if (x >= p.count) continue;
        float sum = 0.0f;
        for (uint32_t tile = 0; tile < tiles; ++tile) sum = fadd(sum, fromBits(partials.load((uint64_t(tile) * p.count + x) * 4)));
        out.store((uint64_t(rowFirst) * 1024 + x) * 4, chandra::vision_calibration::bf16Bits(bits(sum)));
    }
}
} // namespace vdc_fake

namespace chandra::dc {
struct Device::Impl {
    std::shared_ptr<Accounting> account = std::make_shared<Accounting>();
    std::vector<std::shared_ptr<Storage>> pending;
    std::vector<vdc_fake::Event> timings;
    bool profiling = false;
    void valid(const Buffer& b) const {
        if (!b.storage || b.storage->account != account || !b.words || uint64_t(b.words) * 4 != b.storage->bytes)
            throw std::runtime_error("Invalid buffer or foreign D3D11 device");
        const uint64_t capacity = uint64_t(b.words) * (b.packedBF16 ? 2 : 1);
        if (!b.logicalElements || b.logicalElements > capacity) throw std::runtime_error("Invalid logical buffer extent");
    }
    Buffer make(uint32_t count, const void* initial) {
        const uint64_t bytes = uint64_t(count) * 4;
        if (!count || bytes > 128ull * 1024 * 1024) throw std::runtime_error("D3D11 buffer must be nonempty and at most 128 MiB");
        static uint64_t ids = 0;
        auto s = std::make_shared<Storage>();
        if (vdc_fake::config().execute) {
            s->data.assign(count, 0xdeadbeefu); // Uninitialized D3D memory is not assumed to be zero.
            if (initial) std::memcpy(s->data.data(), initial, size_t(bytes));
        }
        s->account = account; s->bytes = uint32_t(bytes); s->id = ++ids;
        account->bytes += bytes; account->peak = std::max(account->peak, account->bytes);
        vdc_fake::Event e; e.kind = "alloc"; e.id = s->id; e.words = count; vdc_fake::events().push_back(e);
        return {s, count, count, false};
    }
};
Device::Device(const std::wstring&, const std::string& pci, const std::string&) : impl(std::make_unique<Impl>()) {
    if (pci.empty()) throw std::runtime_error("Explicit physical PCI identity is required");
}
Device::~Device() = default;
Buffer Device::floats(uint32_t count, const float* values) { return impl->make(count, values); }
Buffer Device::words(uint32_t count, const uint32_t* values) { return impl->make(count, values); }
void Device::upload(Buffer& b, const void* values, uint32_t bytes) {
    impl->valid(b); if (!values || bytes != b.storage->bytes) throw std::runtime_error("Complete bounded buffer upload required");
    if (vdc_fake::config().execute) std::memcpy(b.storage->data.data(), values, bytes);
    impl->pending.push_back(b.storage);
}
void Device::dispatch(const std::string& name, const std::vector<const Buffer*>& inputs, const std::vector<Buffer*>& outputs,
                      const void* params, uint32_t paramBytes, uint32_t x, uint32_t y, uint32_t z) {
    if (inputs.size() > 16 || outputs.empty() || outputs.size() > 8 || paramBytes > 256 || (paramBytes && !params) || !x || !y || !z ||
        x > 65535 || y > 65535 || z > 65535) throw std::runtime_error("Invalid bounded compute dispatch");
    std::unordered_set<const Storage*> writable;
    for (auto p : outputs) { if (!p) throw std::runtime_error("Null UAV"); impl->valid(*p); if (!writable.insert(p->storage.get()).second) throw std::runtime_error("Aliased UAVs"); }
    for (auto p : inputs) { if (!p) throw std::runtime_error("Null SRV"); impl->valid(*p); if (writable.count(p->storage.get())) throw std::runtime_error("SRV/UAV alias forbidden"); }
    vdc_fake::Event e; e.kind = "dispatch"; e.shader = name; e.groups = {{x, y, z}}; e.params.resize(paramBytes / 4);
    if (paramBytes) std::memcpy(e.params.data(), params, paramBytes);
    for (auto p : inputs) { e.inputs.push_back(p->storage->id); e.inputWords.push_back(p->words); impl->pending.push_back(p->storage); }
    for (auto p : outputs) { e.outputs.push_back(p->storage->id); e.outputWords.push_back(p->words); impl->pending.push_back(p->storage); }
    e.id = ++vdc_fake::dispatchSerial();
    if (vdc_fake::config().execute) {
        using namespace vdc_fake;
        auto raw = [](const Buffer* b, const char* label, std::vector<uint8_t>* written = nullptr) { return Raw{b->storage->data, label, written}; };
        std::vector<uint8_t> written(outputs[0]->storage->data.size(), 0);
        if (name == chandra::vision_calibration::stageShaders[0]) {
            check(paramBytes == sizeof(AttentionParams) && inputs.size() == 2 && outputs.size() == 1 && z == 1, "scores ABI: 48-byte cbuffer, t0-t1, u0");
            AttentionParams p; std::memcpy(&p, params, sizeof(p)); Raw out = raw(outputs[0], "scores", &written);
            vdc_fake::scores(p, x, y, raw(inputs[0], "q"), raw(inputs[1], "k"), out);
        } else if (name == chandra::vision_calibration::stageShaders[1]) {
            check(paramBytes == sizeof(SoftmaxParams) && inputs.empty() && outputs.size() == 1 && y == 1 && z == 1, "softmax ABI: 16-byte cbuffer, u0");
            SoftmaxParams p; std::memcpy(&p, params, sizeof(p)); Raw s = raw(outputs[0], "scores");
            vdc_fake::softmax(p, x, s);
        } else if (name == chandra::vision_calibration::stageShaders[2]) {
            check(paramBytes == sizeof(AttentionParams) && inputs.size() == 2 && outputs.size() == 1 && z == 1, "values ABI: 48-byte cbuffer, t0-t1, u0");
            AttentionParams p; std::memcpy(&p, params, sizeof(p)); Raw out = raw(outputs[0], "partials", &written);
            vdc_fake::values(p, x, y, raw(inputs[0], "probabilities"), raw(inputs[1], "v"), out);
        } else if (name == chandra::vision_calibration::stageShaders[3]) {
            check(paramBytes == sizeof(ReduceParams) && inputs.size() == 1 && outputs.size() == 1 && y == 1 && z == 1, "reduce ABI: 16-byte cbuffer, t0, u0");
            ReduceParams p; std::memcpy(&p, params, sizeof(p)); Raw out = raw(outputs[0], "result", &written);
            vdc_fake::reduce(p, x, raw(inputs[0], "partials"), out);
        } else {
            throw std::runtime_error("Unemulated shader " + name);
        }
    }
    vdc_fake::events().push_back(e);
    if (impl->profiling) impl->timings.push_back(e);
}
std::vector<uint32_t> Device::readWords(const Buffer& b) {
    impl->valid(b); drain();
    if (!vdc_fake::config().execute) throw std::runtime_error("Record-only fake device has no data to read");
    vdc_fake::Event e; e.kind = "read"; e.id = b.storage->id; vdc_fake::events().push_back(e);
    return b.storage->data;
}
std::vector<float> Device::readFloats(const Buffer& b) {
    auto words = readWords(b); std::vector<float> result(words.size()); std::memcpy(result.data(), words.data(), words.size() * 4); return result;
}
void Device::zero(Buffer& b) { impl->valid(b); std::fill(b.storage->data.begin(), b.storage->data.end(), 0u); impl->pending.push_back(b.storage); }
void Device::drain(uint32_t timeout) {
    if (!timeout || timeout > 30000) throw std::runtime_error("Bounded drain timeout required");
    impl->pending.clear(); vdc_fake::Event e; e.kind = "drain"; vdc_fake::events().push_back(e);
}
uint64_t Device::trackedBufferBytes() const { return impl->account->bytes; }
std::string Device::identityJson() const { return "{\"fake_cpu_device\":true}"; }
std::string Device::memoryJson() const {
    return vdc_fake::Json{{"tracked_live", impl->account->bytes}, {"tracked_peak", impl->account->peak}, {"fake_cpu_device", true}}.dump();
}
void Device::beginProfile() {
    if (impl->profiling || !impl->timings.empty()) throw std::runtime_error("Profile window already open or uncollected");
    drain(); ++vdc_fake::profileBegins();
    if (vdc_fake::config().beginProfileHook) vdc_fake::config().beginProfileHook(vdc_fake::profileBegins());
    impl->profiling = true;
}
std::string Device::finishProfile() {
    if (!impl->profiling) throw std::runtime_error("No profile window");
    impl->profiling = false; drain();
    const auto& cfg = vdc_fake::config();
    bool faulted = false;
    for (const auto& r : impl->timings) faulted = faulted || r.id == cfg.profileFaultDispatch;
    const auto fault = faulted ? cfg.profileFault : vdc_fake::ProfileFault::none;
    // Mirrors device.cpp: a disjoint window throws and leaves its records uncollected.
    if (fault == vdc_fake::ProfileFault::disjointThrows) throw std::runtime_error("GPU timestamps disjoint or unavailable; timing rejected");
    if (fault == vdc_fake::ProfileFault::giantError)
        throw std::runtime_error("Synthetic driver message " + std::string(size_t(chandra::vision_calibration::receiptCap) + 1, 'x') + " \xc3\xa9");
    vdc_fake::Json rows = vdc_fake::Json::array();
    if (fault != vdc_fake::ProfileFault::missing)
        for (const auto& r : impl->timings) {
            const auto groups = fault == vdc_fake::ProfileFault::wrongGroups ? std::array<uint32_t, 3>{{1, 1, 1}} : r.groups;
            const double ms = fault == vdc_fake::ProfileFault::negative ? -1.0 : cfg.gpuMilliseconds(r.shader, r.id);
            rows.push_back({{"shader", r.shader}, {"groups", groups}, {"gpu_milliseconds", ms}});
        }
    impl->timings.clear();
    vdc_fake::Json profile = {{"frequency", 1000000u}, {"disjoint", fault == vdc_fake::ProfileFault::disjointReported},
                              {"dispatches", rows}, {"synthetic_cpu_timing", true}};
    if (fault == vdc_fake::ProfileFault::giantMetadata)
        profile["driver_annotations"] = {{"note", std::string(size_t(chandra::vision_calibration::receiptCap), 'm')}};
    return profile.dump();
}
} // namespace chandra::dc
