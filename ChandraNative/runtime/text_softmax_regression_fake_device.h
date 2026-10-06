// New ChandraNative code, MPL-2.0. Test-only CPU Device for the text softmax regression; include in exactly
// one TU. It keeps device.cpp's buffer/dispatch validation and executes C++ transliterations of the repaired
// and the 2c44182 text_attention_softmax.hlsl under D3D11 binary32 rules (sign-preserving subnormal flush)
// with bounds-checked raw loads/stores. Lanes run in hardware-like waves: every lane of a wave executes each
// statement before any lane of that wave executes the next, and waves interleave between barriers in the
// configured order. That reproduces the original shader's shared-maximum hazard only when a later wave owns
// keys. It is evidence about algorithm, indexing and host lifecycle, not about fxc, the Intel driver or A770.
#pragma once
#include "api.h"
#include "text_softmax_regression.h"
#include <cfloat>
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

namespace tsr_fake {
namespace tsr = chandra::text_softmax_regression;
using tsr::Json;
// Numerical faults change what the "repaired" shader name computes; the control name always runs the original.
// alternateAdmissibleWord swaps one word to the other value of its two-value gate: only the repetition check sees it.
enum class Fault { none, repairedWithoutBarrier, dropLastLanePartial, guardStore, maskedNonzero, alternateAdmissibleWord, nanWord };
enum class ProfileFault { none, disjointThrows, disjointReported, missing, wrongGroups, wrongShader, negative, giantError };
enum class WaveOrder { ascending, descending };
struct Config {
    uint32_t waveWidth = 32; WaveOrder order = WaveOrder::ascending;
    Fault fault = Fault::none; uint64_t faultDispatch = 0;          // 0: every dispatch of the faulted shader name.
    ProfileFault profileFault = ProfileFault::none; uint64_t profileFaultDispatch = 0;
    std::function<double(const std::string&, uint64_t)> gpuMilliseconds = [](const std::string&, uint64_t) { return 0.25; };
    std::function<void(uint64_t)> dispatchHook, beginProfileHook;   // Serial / window number, before execution.
    bool constructorThrows = false, drainThrowsAfterFailure = false; std::string identityOverride;
};
inline Config& config() { static Config value; return value; }
struct Counters { uint64_t dispatches = 0, profiles = 0, drains = 0, reads = 0, allocations = 0, flushes = 0; std::vector<std::string> shaders; };
inline Counters& counters() { static Counters value; return value; }
inline void reset(Config c = {}) { config() = std::move(c); counters() = {}; }
inline void check(bool ok, const std::string& message) { if (!ok) throw std::logic_error("FAKE DEVICE CHECK FAILED: " + message); }

inline float ftz(float x) { if (x != 0.0f && std::fabs(x) < FLT_MIN) { ++counters().flushes; return std::copysign(0.0f, x); } return x; }
inline float fadd(float a, float b) { const float r = ftz(a) + ftz(b); return ftz(r); }
inline float fsub(float a, float b) { const float r = ftz(a) - ftz(b); return ftz(r); }
inline float fdiv(float a, float b) { const float r = ftz(a) / ftz(b); return ftz(r); }
inline float fmax32(float a, float b) { return std::fmax(ftz(a), ftz(b)); }
// fxc's expected lowering: multiply by fl32(log2 e), then the base-2 exponential.
inline float fexp(float x) { const float t = ftz(ftz(x) * 1.44269502f); return ftz(std::exp2(t)); }
inline uint32_t bf(float x) { // The shader's bf(): quiet NaN, otherwise round-to-nearest-even to BF16.
    uint32_t u = tsr::vc::bits(x);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) return u | 0x00400000u;
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}
struct Raw {
    std::vector<uint32_t>& data;
    uint32_t load(uint64_t address) const { check(address % 4 == 0 && address / 4 < data.size(), "out-of-bounds raw load"); return data[size_t(address / 4)]; }
    void store(uint64_t address, uint32_t value) { check(address % 4 == 0 && address / 4 < data.size(), "out-of-bounds raw store"); data[size_t(address / 4)] = value; }
};
// text_attention_softmax.hlsl, one group per row. `barrier` selects the repaired shader (barrier after
// m=tmp[0]); without it, waves run that barrier interval one after another in the configured order.
inline void softmax(const tsr::TextParams& p, uint32_t groups, Raw& s, bool barrier, bool dropLastLane, const Config& c) {
    const uint32_t lanes = tsr::lanes, count = p.count, width = c.waveWidth, waves = lanes / width;
    check(width >= 1 && lanes % width == 0, "wave width must divide 128");
    for (uint32_t g = 0; g < groups; ++g) {
        float tmp[tsr::lanes], m[tsr::lanes], total[tsr::lanes];
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            float maximum = -3.402823466e38f;
            for (uint32_t k = lane; k < count; k += lanes) maximum = fmax32(maximum, tsr::vc::fromBits(s.load((uint64_t(g) * count + k) * 4)));
            tmp[lane] = maximum;
        }
        for (uint32_t step = 64; step; step >>= 1) for (uint32_t lane = 0; lane < step; ++lane) tmp[lane] = fmax32(tmp[lane], tmp[lane + step]);
        auto partial = [&](uint32_t lane) {
            float sum = 0;
            for (uint32_t k = lane; k < count; k += lanes) sum = fadd(sum, fexp(fsub(tsr::vc::fromBits(s.load((uint64_t(g) * count + k) * 4)), m[lane])));
            total[lane] = sum;
        };
        if (barrier) {
            for (uint32_t lane = 0; lane < lanes; ++lane) m[lane] = tmp[0];
            for (uint32_t lane = 0; lane < lanes; ++lane) { partial(lane); tmp[lane] = total[lane]; }
        } else {
            for (uint32_t i = 0; i < waves; ++i) {
                const uint32_t wave = c.order == WaveOrder::ascending ? i : waves - 1 - i, first = wave * width;
                for (uint32_t lane = first; lane < first + width; ++lane) m[lane] = tmp[0];
                for (uint32_t lane = first; lane < first + width; ++lane) partial(lane);
                for (uint32_t lane = first; lane < first + width; ++lane) tmp[lane] = total[lane];
            }
        }
        if (dropLastLane) tmp[lanes - 1] = 0.0f;
        for (uint32_t step = 64; step; step >>= 1) for (uint32_t lane = 0; lane < step; ++lane) tmp[lane] = fadd(tmp[lane], tmp[lane + step]);
        const float denominator = tmp[0];
        for (uint32_t lane = 0; lane < lanes; ++lane)
            for (uint32_t k = lane; k < count; k += lanes) {
                const uint64_t address = (uint64_t(g) * count + k) * 4;
                s.store(address, bf(fdiv(fexp(fsub(tsr::vc::fromBits(s.load(address)), m[lane])), denominator)));
            }
    }
}
} // namespace tsr_fake

namespace chandra::dc {
struct Device::Impl {
    std::shared_ptr<Accounting> account = std::make_shared<Accounting>();
    std::vector<std::shared_ptr<Storage>> pending;
    std::vector<std::pair<std::string, std::array<uint32_t, 3>>> timings; std::vector<uint64_t> timingSerials;
    bool profiling = false, failed = false; std::string pci, luid;
    void valid(const Buffer& b) const {
        if (!b.storage || b.storage->account != account || !b.words || uint64_t(b.words) * 4 != b.storage->bytes)
            throw std::runtime_error("Invalid buffer or foreign D3D11 device");
        if (!b.logicalElements || b.logicalElements > uint64_t(b.words) * (b.packedBF16 ? 2 : 1)) throw std::runtime_error("Invalid logical buffer extent");
    }
    Buffer make(uint32_t count, const void* initial) {
        const uint64_t bytes = uint64_t(count) * 4;
        if (!count || bytes > 128ull * 1024 * 1024) throw std::runtime_error("D3D11 buffer must be nonempty and at most 128 MiB");
        static uint64_t ids = 0;
        auto s = std::make_shared<Storage>();
        s->data.assign(count, 0xdeadbeefu); // Uninitialized D3D memory is not assumed to be zero.
        if (initial) std::memcpy(s->data.data(), initial, size_t(bytes));
        s->account = account; s->bytes = uint32_t(bytes); s->id = ++ids;
        account->bytes += bytes; account->peak = std::max(account->peak, account->bytes);
        ++tsr_fake::counters().allocations;
        return {s, count, count, false};
    }
};
Device::Device(const std::wstring&, const std::string& pci, const std::string& luid) : impl(std::make_unique<Impl>()) {
    if (pci.empty()) throw std::runtime_error("Explicit physical PCI identity is required");
    if (tsr_fake::config().constructorThrows) throw std::runtime_error("requested Intel adapter identity not found");
    impl->pci = pci; impl->luid = luid;
}
Device::~Device() = default;
Buffer Device::floats(uint32_t count, const float* values) { return impl->make(count, values); }
Buffer Device::words(uint32_t count, const uint32_t* values) { return impl->make(count, values); }
void Device::upload(Buffer& b, const void* values, uint32_t bytes) {
    impl->valid(b); if (!values || bytes != b.storage->bytes) throw std::runtime_error("Complete bounded buffer upload required");
    std::memcpy(b.storage->data.data(), values, bytes); impl->pending.push_back(b.storage);
}
void Device::dispatch(const std::string& name, const std::vector<const Buffer*>& inputs, const std::vector<Buffer*>& outputs,
                      const void* params, uint32_t paramBytes, uint32_t x, uint32_t y, uint32_t z) {
    namespace tsr = chandra::text_softmax_regression;
    if (inputs.size() > 16 || outputs.empty() || outputs.size() > 8 || paramBytes > 256 || (paramBytes && !params) || !x || !y || !z ||
        x > 65535 || y > 65535 || z > 65535) throw std::runtime_error("Invalid bounded compute dispatch");
    std::unordered_set<const Storage*> writable;
    for (auto p : outputs) { if (!p) throw std::runtime_error("Null UAV"); impl->valid(*p); if (!writable.insert(p->storage.get()).second) throw std::runtime_error("Aliased UAVs"); }
    for (auto p : inputs) { if (!p) throw std::runtime_error("Null SRV"); impl->valid(*p); if (writable.count(p->storage.get())) throw std::runtime_error("SRV/UAV alias forbidden"); }
    auto& c = tsr_fake::config(); auto& n = tsr_fake::counters();
    const uint64_t serial = ++n.dispatches; n.shaders.push_back(name);
    if (c.dispatchHook) c.dispatchHook(serial);
    const bool repaired = name == tsr::repairedShader, control = name == tsr::controlShader;
    if (!repaired && !control) throw std::runtime_error("Shader compile " + name + ": not a text softmax regression shader");
    tsr_fake::check(paramBytes == sizeof(tsr::TextParams) && inputs.empty() && outputs.size() == 1 && y == 1 && z == 1,
                    "text softmax ABI: 32-byte cbuffer, u0 only, groups (x,1,1)");
    tsr::TextParams p; std::memcpy(&p, params, sizeof(p));
    tsr_fake::check(x == p.rows * tsr::heads && p.width == tsr::heads && p.count == p.base + p.rows, "groups/cbuffer differ from production");
    for (auto out : outputs) impl->pending.push_back(out->storage);
    const bool faulted = repaired && c.fault != tsr_fake::Fault::none && (!c.faultDispatch || c.faultDispatch == serial);
    tsr_fake::Raw scores{outputs[0]->storage->data};
    tsr_fake::softmax(p, x, scores, repaired && !(faulted && c.fault == tsr_fake::Fault::repairedWithoutBarrier),
                      faulted && c.fault == tsr_fake::Fault::dropLastLanePartial, c);
    auto& data = outputs[0]->storage->data; const size_t plane = size_t(x) * p.count;
    if (faulted && c.fault == tsr_fake::Fault::guardStore && plane < data.size()) data[plane] = 0;
    if (faulted && c.fault == tsr_fake::Fault::maskedNonzero && p.rows > 1) data[p.base + 1] = 0x00800000u; // Row 0 admits base + 1 keys.
    if (faulted && c.fault == tsr_fake::Fault::alternateAdmissibleWord) {
        uint32_t index = UINT32_MAX;
        for (size_t i = 0; i < tsr::cases().size(); ++i)
            if (tsr::cases()[i].base == p.base && tsr::cases()[i].rows == p.rows) index = uint32_t(i);
        tsr_fake::check(index != UINT32_MAX, "alternateAdmissibleWord needs a planned case");
        const auto g = tsr::geometry(tsr::cases()[index]); const auto o = tsr::buildOracle(index, g); bool swapped = false;
        for (size_t i = 0; i < g.planeWords && !swapped; ++i)
            if (o.low[i] != o.high[i] && tsr::rowKind(index, uint32_t(i / g.count)) != tsr::RowKind::uniform) {
                data[i] = uint32_t(data[i] >> 16 == o.low[i] ? o.high[i] : o.low[i]) << 16; swapped = true;
            }
        tsr_fake::check(swapped, "no two-value gate in this case");
    }
    if (faulted && c.fault == tsr_fake::Fault::nanWord) data[0] = 0x7fc00000u;
    if (impl->profiling) { impl->timings.push_back({name, {{x, y, z}}}); impl->timingSerials.push_back(serial); }
}
std::vector<uint32_t> Device::readWords(const Buffer& b) { impl->valid(b); drain(); ++tsr_fake::counters().reads; return b.storage->data; }
std::vector<float> Device::readFloats(const Buffer& b) {
    auto w = readWords(b); std::vector<float> r(w.size()); std::memcpy(r.data(), w.data(), w.size() * 4); return r;
}
void Device::zero(Buffer& b) { impl->valid(b); std::fill(b.storage->data.begin(), b.storage->data.end(), 0u); impl->pending.push_back(b.storage); }
void Device::drain(uint32_t timeout) {
    if (!timeout || timeout > 30000) throw std::runtime_error("Bounded drain timeout required");
    ++tsr_fake::counters().drains;
    if (impl->failed && tsr_fake::config().drainThrowsAfterFailure) throw std::runtime_error("D3D11 drain deadline exceeded; retire owned worker");
    impl->pending.clear();
}
uint64_t Device::trackedBufferBytes() const { return impl->account->bytes; }
std::string Device::identityJson() const {
    if (!tsr_fake::config().identityOverride.empty()) return tsr_fake::config().identityOverride;
    return tsr_fake::Json{{"pci_bdf", impl->pci}, {"luid", impl->luid}, {"vendor_id", 0x8086u}, {"device_id", 0x56a0u},
                          {"description", "Fake CPU stand-in for Intel(R) Arc(TM) A770 Graphics"}, {"fake_cpu_device", true}}.dump();
}
std::string Device::memoryJson() const {
    return tsr_fake::Json{{"tracked_live", impl->account->bytes}, {"tracked_peak", impl->account->peak}, {"fake_cpu_device", true}}.dump();
}
void Device::beginProfile() {
    if (impl->profiling || !impl->timings.empty()) throw std::runtime_error("Profile window already open or uncollected");
    drain(); const uint64_t window = ++tsr_fake::counters().profiles;
    if (tsr_fake::config().beginProfileHook) tsr_fake::config().beginProfileHook(window);
    impl->profiling = true;
}
std::string Device::finishProfile() {
    if (!impl->profiling) throw std::runtime_error("No profile window");
    impl->profiling = false; drain();
    const auto& c = tsr_fake::config(); bool faulted = false;
    for (uint64_t serial : impl->timingSerials) faulted = faulted || serial == c.profileFaultDispatch;
    const auto fault = faulted ? c.profileFault : tsr_fake::ProfileFault::none;
    if (fault == tsr_fake::ProfileFault::disjointThrows) { impl->failed = true; throw std::runtime_error("GPU timestamps disjoint or unavailable; timing rejected"); }
    if (fault == tsr_fake::ProfileFault::giantError)
        throw std::runtime_error("Synthetic driver message " + std::string(size_t(chandra::text_softmax_regression::receiptCap) + 1, 'x'));
    tsr_fake::Json rows = tsr_fake::Json::array();
    if (fault != tsr_fake::ProfileFault::missing)
        for (size_t i = 0; i < impl->timings.size(); ++i) {
            const auto& t = impl->timings[i];
            const auto groups = fault == tsr_fake::ProfileFault::wrongGroups ? std::array<uint32_t, 3>{{1, 1, 1}} : t.second;
            const std::string shader = fault == tsr_fake::ProfileFault::wrongShader ? std::string("runtime/text_attention_values.hlsl") : t.first;
            const double ms = fault == tsr_fake::ProfileFault::negative ? -1.0 : c.gpuMilliseconds(t.first, impl->timingSerials[i]);
            rows.push_back({{"shader", shader}, {"groups", groups}, {"gpu_milliseconds", ms}});
        }
    impl->timings.clear(); impl->timingSerials.clear();
    return tsr_fake::Json{{"frequency", 1000000u}, {"disjoint", fault == tsr_fake::ProfileFault::disjointReported}, {"dispatches", rows},
                          {"synthetic_cpu_timing", true}}.dump();
}
} // namespace chandra::dc
