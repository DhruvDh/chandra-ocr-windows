// New ChandraNative code, MPL-2.0. CPU-only B1 GEMV candidate harness; no GPU, model or network.
// A fake Device executes C++ transliterations of linear.hlsl and linear_gemv.hlsl with D3D11 float32
// rules (sign-preserving flush of subnormal operands/results) and checks every buffer and
// groupshared access. Transliterations are evidence about the algorithm, addressing and host
// integration, not about fxc, the Intel driver or A770 execution. Run gemv_candidate_test.sh.
#include "api.h"
#include "gemv_candidate_calibration.h"
#include <cstdio>
#include <iostream>
#include <memory>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>

namespace chandra::dc {
struct Accounting { uint64_t bytes = 0, peak = 0, allocations = 0; };
struct Storage {
    std::vector<uint32_t> data; std::shared_ptr<Accounting> account; uint64_t id = 0; uint32_t bytes = 0;
    ~Storage() { if (account) account->bytes -= bytes; }
};
struct DispatchRecord { std::string shader; std::vector<uint32_t> params; uint32_t x, y, z; std::vector<uint64_t> inputs, outputs; };
std::vector<DispatchRecord> dispatchLog; // Process-wide transcript of every fake dispatch (Impl is private).
struct Device::Impl {
    std::shared_ptr<Accounting> account = std::make_shared<Accounting>();
    std::vector<std::shared_ptr<Storage>> pending;
    std::vector<DispatchRecord> log, timings;
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
        auto s = std::make_shared<Storage>();
        s->data.assign(count, 0xdeadbeefu); // Uninitialized D3D memory is not assumed to be zero.
        if (initial) std::memcpy(s->data.data(), initial, size_t(bytes));
        static uint64_t processAllocations = 0; // Ids stay unique across fake devices.
        s->account = account; s->bytes = uint32_t(bytes); s->id = ++processAllocations; ++account->allocations;
        account->bytes += bytes; account->peak = std::max(account->peak, account->bytes);
        return {s, count, count, false};
    }
};
}

namespace {
using namespace chandra::dc;
using chandra::gemv_calibration::Json;
using chandra::gemv_calibration::bits;
using chandra::gemv_calibration::fromBits;
using chandra::gemv_calibration::roundedBF16Bits;
using chandra::gemv_calibration::fnv1a;
using chandra::gemv_calibration::hex64;
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error("CHECK FAILED: " + message); }

// D3D11 float32 arithmetic: subnormal operands and results flush to zero preserving sign.
uint64_t flushed = 0;
float ftz(float x) { if (x != 0.0f && std::fabs(x) < FLT_MIN) { ++flushed; return std::copysign(0.0f, x); } return x; }
float mul32(float a, float b) { return ftz(ftz(a) * ftz(b)); }
float add32(float a, float b) { return ftz(ftz(a) + ftz(b)); }
float bf16Rne(float value) { return fromBits(roundedBF16Bits(bits(value))); }

struct Raw { const std::vector<uint32_t>& data; const char* name;
    uint32_t load(uint32_t byteAddress) const {
        check(byteAddress % 4 == 0 && byteAddress / 4 < data.size(), std::string("out-of-bounds raw load from ") + name);
        return data[byteAddress / 4];
    }
};
float packedValue(const Raw& source, uint32_t index, bool bf16) {
    if (!bf16) return fromBits(source.load(index * 4));
    uint32_t word = source.load((index >> 1) * 4);
    return fromBits(((word >> ((index & 1) * 16)) & 0xffffu) << 16);
}
struct Params { uint32_t inputWidth, outputWidth, batchRows, firstOutput, shardRows, rowFirst, flags, weightRowFirst; };

// Transliteration of linear.hlsl. Its tile staging copies exact values, so each active thread is
// its ascending-k dot; inactive threads have no stores and are skipped.
void emulateLinear(const Params& p, uint32_t gx, uint32_t gy, const Raw& input, const Raw& weight, const Raw& bias, std::vector<uint32_t>& output) {
    for (uint32_t groupY = 0; groupY < gy; ++groupY) for (uint32_t groupX = 0; groupX < gx; ++groupX)
        for (uint32_t ly = 0; ly < 16; ++ly) for (uint32_t lx = 0; lx < 16; ++lx) {
            uint32_t localOut = groupX * 16 + lx, localRow = groupY * 16 + ly, row = p.rowFirst + localRow;
            if (!(localRow < p.batchRows && localOut < p.shardRows)) continue;
            float sum = 0.0f;
            for (uint32_t k = 0; k < p.inputWidth; ++k)
                sum = add32(sum, mul32(fromBits(input.load((row * p.inputWidth + k) * 4)),
                                       packedValue(weight, (p.weightRowFirst + localOut) * p.inputWidth + k, (p.flags & 1) != 0)));
            uint32_t column = p.firstOutput + localOut;
            if (p.flags & 4) sum = add32(sum, packedValue(bias, column, (p.flags & 8) != 0));
            if (p.flags & 2) sum = bf16Rne(sum);
            uint32_t address = (row * p.outputWidth + column) * 4;
            check(address / 4 < output.size(), "linear.hlsl out-of-bounds store");
            output[address / 4] = bits(sum);
        }
}

// Transliteration of linear_gemv.hlsl, phase by phase between its two barriers. Groupshared
// slots carry a tile epoch: a read of a slot not staged in the current tile, a double store in
// one phase or an out-of-row position fails the test.
constexpr uint32_t ROWS = 8, LANES = 32, THREADS = 256, TILE = 512, STRIDE = 513, STAGED = 16;
uint64_t gemvGroupsEmulated = 0;
void emulateGemv(const Params& p, uint32_t gx, uint32_t gy, uint32_t gz, const Raw& input, const Raw& weight, const Raw& bias,
                 std::vector<uint32_t>& output, std::set<uint32_t>& stored) {
    check(gy == 1 && gz == 1 && p.batchRows == 1 && p.rowFirst == 0, "GEMV host contract: one group row, batchRows 1, rowFirst 0");
    std::vector<float> tileWeight(ROWS * STRIDE), tileInput(TILE);
    std::vector<uint64_t> weightEpoch(ROWS * STRIDE, 0), inputEpoch(TILE, 0);
    uint64_t epoch = 0;
    const bool bf16 = (p.flags & 1) != 0;
    for (uint32_t gid = 0; gid < gx; ++gid) {
        ++gemvGroupsEmulated;
        float sums[ROWS] = {};
        for (uint32_t begin = 0; begin < p.inputWidth; begin += TILE) {
            ++epoch;
            uint32_t n = std::min(TILE, p.inputWidth - begin);
            // Phase A: every thread stages, then barrier.
            for (uint32_t thread = 0; thread < THREADS; ++thread) {
                uint32_t stageRow = thread / LANES, stageLane = thread % LANES, stageOut = gid * ROWS + stageRow;
                uint32_t first = (p.weightRowFirst + stageOut) * p.inputWidth + begin; // uint32 arithmetic, as HLSL
                uint32_t parity = bf16 ? (first & 1) : 0, base = bf16 ? (first >> 1) : first;
                uint32_t count = stageOut < p.shardRows ? (bf16 ? ((parity + n + 1) >> 1) : n) : 0;
                uint32_t staged[STAGED];
                for (uint32_t i = 0; i < STAGED; ++i) {
                    uint32_t j = stageLane + i * LANES; staged[i] = 0;
                    if (j < count) staged[i] = weight.load((base + j) * 4);
                }
                for (uint32_t m = 0; m < TILE / THREADS; ++m) {
                    uint32_t index = thread + m * THREADS;
                    if (index < n) {
                        check(inputEpoch[index] != epoch, "groupshared input store race");
                        tileInput[index] = fromBits(input.load((begin + index) * 4)); inputEpoch[index] = epoch;
                    }
                }
                auto store = [&](uint32_t position, uint32_t value) {
                    check(position < n && position < STRIDE, "staged weight position outside the tile row");
                    uint32_t slot = stageRow * STRIDE + position;
                    check(weightEpoch[slot] != epoch, "groupshared weight store race");
                    tileWeight[slot] = fromBits(value); weightEpoch[slot] = epoch;
                };
                for (uint32_t s = 0; s < STAGED; ++s) {
                    uint32_t j = stageLane + s * LANES;
                    if (j < count) {
                        if (!bf16) store(j, staged[s]);
                        else {
                            if (2 * j >= parity) store(2 * j - parity, staged[s] << 16);
                            if (2 * j + 1 - parity < n) store(2 * j + 1 - parity, staged[s] & 0xffff0000u);
                        }
                    }
                }
            }
            // Phase B: owners continue their sequential sums, then barrier.
            for (uint32_t thread = 0; thread < ROWS; ++thread) {
                if (!(gid * ROWS + thread < p.shardRows)) continue;
                uint32_t row = thread * STRIDE;
                for (uint32_t k = 0; k < n; ++k) {
                    check(inputEpoch[k] == epoch && weightEpoch[row + k] == epoch, "owner read an unstaged or stale groupshared slot");
                    sums[thread] = add32(sums[thread], mul32(tileInput[k], tileWeight[row + k]));
                }
            }
        }
        for (uint32_t thread = 0; thread < ROWS; ++thread) {
            uint32_t localOut = gid * ROWS + thread;
            if (!(localOut < p.shardRows)) continue;
            uint32_t column = p.firstOutput + localOut;
            float sum = sums[thread];
            if (p.flags & 4) sum = add32(sum, packedValue(bias, column, (p.flags & 8) != 0));
            if (p.flags & 2) sum = bf16Rne(sum);
            check(column < output.size() && column < p.outputWidth, "linear_gemv.hlsl out-of-bounds store");
            check(stored.insert(column).second, "two GEMV threads stored the same output column");
            output[column] = bits(sum);
        }
    }
}
// Output columns stored by GEMV threads, per output buffer id: detects duplicate or missing stores.
std::map<uint64_t, std::set<uint32_t>> gemvStored;
}

namespace chandra::dc {
Device::Device(const std::wstring&, const std::string& pci, const std::string&) : impl(std::make_unique<Impl>()) {
    if (pci.empty()) throw std::runtime_error("Explicit physical PCI identity is required");
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
    if (inputs.size() > 16 || outputs.empty() || outputs.size() > 8 || paramBytes > 256 || (paramBytes && !params) || !x || !y || !z ||
        x > 65535 || y > 65535 || z > 65535) throw std::runtime_error("Invalid bounded compute dispatch");
    std::unordered_set<const Storage*> writable;
    for (auto p : outputs) { if (!p) throw std::runtime_error("Null UAV"); impl->valid(*p); if (!writable.insert(p->storage.get()).second) throw std::runtime_error("Aliased UAVs"); }
    for (auto p : inputs) { if (!p) throw std::runtime_error("Null SRV"); impl->valid(*p); if (writable.count(p->storage.get())) throw std::runtime_error("SRV/UAV alias forbidden"); }
    DispatchRecord record{name, std::vector<uint32_t>(paramBytes / 4), x, y, z, {}, {}};
    if (paramBytes) std::memcpy(record.params.data(), params, paramBytes);
    for (auto p : inputs) { record.inputs.push_back(p->storage->id); impl->pending.push_back(p->storage); }
    for (auto p : outputs) { record.outputs.push_back(p->storage->id); impl->pending.push_back(p->storage); }
    check(paramBytes == sizeof(Params) && inputs.size() == 3 && outputs.size() == 1, "linear ABI: 32-byte cbuffer, t0-t2, u0");
    Params p; std::memcpy(&p, params, sizeof(p));
    Raw in{inputs[0]->storage->data, "input"}, w{inputs[1]->storage->data, "weight"}, b{inputs[2]->storage->data, "bias"};
    if (name == "runtime/linear.hlsl") emulateLinear(p, x, y, in, w, b, outputs[0]->storage->data);
    else if (name == "runtime/linear_gemv.hlsl") {
        emulateGemv(p, x, y, z, in, w, b, outputs[0]->storage->data, gemvStored[outputs[0]->storage->id]);
    } else throw std::runtime_error("Unemulated shader " + name);
    impl->log.push_back(record); dispatchLog.push_back(record);
    if (impl->profiling) impl->timings.push_back(std::move(record));
}
std::vector<uint32_t> Device::readWords(const Buffer& b) { impl->valid(b); drain(); return b.storage->data; }
std::vector<float> Device::readFloats(const Buffer& b) {
    if (b.packedBF16) throw std::runtime_error("Packed BF16 is not float storage");
    auto words = readWords(b); std::vector<float> result(words.size()); std::memcpy(result.data(), words.data(), words.size() * 4); return result;
}
void Device::zero(Buffer& b) { impl->valid(b); std::fill(b.storage->data.begin(), b.storage->data.end(), 0u); }
void Device::drain(uint32_t timeout) { if (!timeout || timeout > 30000) throw std::runtime_error("Bounded drain timeout required"); impl->pending.clear(); }
uint64_t Device::trackedBufferBytes() const { return impl->account->bytes; }
std::string Device::identityJson() const { return "{\"fake_cpu_device\":true}"; }
std::string Device::memoryJson() const {
    return Json{{"tracked_live", impl->account->bytes}, {"tracked_peak", impl->account->peak}, {"fake_cpu_device", true}}.dump();
}
void Device::beginProfile() {
    if (impl->profiling || !impl->timings.empty()) throw std::runtime_error("Profile window already open or uncollected");
    drain(); impl->profiling = true;
}
std::string Device::finishProfile() {
    if (!impl->profiling) throw std::runtime_error("No profile window");
    impl->profiling = false; drain();
    Json rows = Json::array();
    for (const auto& r : impl->timings) rows.push_back({{"shader", r.shader}, {"groups", {r.x, r.y, r.z}}, {"gpu_milliseconds", 0.0}});
    impl->timings.clear();
    return Json{{"frequency", 1u}, {"disjoint", false}, {"dispatches", rows}, {"synthetic_cpu_timing", true}}.dump();
}
}

namespace {
#ifndef GEMV_TEST_PREDECESSOR
const char* selection() { return chandra::dc::experimental::gemvB1Selection(); }
#endif
struct Case {
    std::string name; uint32_t rows, cols, batch; std::vector<uint32_t> shards; bool bf16; int bias; // 0 none, 1 bf16, 2 fp32
    int values; // 0 order-sensitive finite, 1 dyadic, 2 specials-rich
};
uint64_t mix(uint64_t x) { return chandra::gemv_calibration::splitmix(x); }
float inputFor(const Case& c, uint32_t row, uint32_t k) {
    uint64_t r = mix(0x1000000000ull * (row + 1) + k + c.cols * 7919ull);
    if (c.values == 1) return float(int(r % 33) - 16) / 32.0f;
    if (c.values == 2) {
        switch (r % 23) {
        case 0: return -0.0f; case 1: return 0.0f; case 2: return INFINITY; case 3: return -INFINITY;
        case 4: return fromBits(0x7fc00000u); case 5: return fromBits(0x00000001u); case 6: return fromBits(0x3f808000u);
        default: break;
        }
    }
    uint32_t exponent = uint32_t(127 - 6 + (r >> 40) % 12);
    return fromBits(uint32_t((r >> 63) << 31) | (exponent << 23) | uint32_t(r & 0x7fffffu));
}
uint32_t weightFor(const Case& c, uint32_t row, uint32_t k) { // FP32 bits; BF16-exact when c.bf16
    uint64_t r = mix(0x2000000000ull * (row + 3) + k * 31ull + c.cols);
    if (c.values == 1) return bits(float(int(r % 33) - 16) / 32.0f);
    if (c.values == 2 && r % 29 == 0) return 0x80000000u;
    uint32_t exponent = uint32_t(127 - 9 + (r >> 40) % 10);
    uint32_t value = uint32_t((r >> 63) << 31) | (exponent << 23) | uint32_t(r & 0x7fffffu);
    return c.bf16 ? (value & 0xffff0000u) : value;
}
uint32_t biasFor(const Case& c, uint32_t row) {
    uint32_t value = bits(float(int(mix(0x3000000000ull + row) % 65) - 32) / 64.0f);
    if (c.values == 2 && row % 5 == 1) value = 0x80000000u;
    return c.bias == 1 ? (value & 0xffff0000u) : value;
}
struct Built { Buffer input; Weight weight, bias; };
Built build(Device& d, const Case& c) {
    Built b;
    std::vector<float> input(size_t(c.batch) * c.cols);
    for (uint32_t r = 0; r < c.batch; ++r) for (uint32_t k = 0; k < c.cols; ++k) input[size_t(r) * c.cols + k] = inputFor(c, r, k);
    b.input = d.floats(uint32_t(input.size()), input.data());
    b.weight.rows = c.rows; b.weight.cols = c.cols; b.weight.bf16 = c.bf16;
    uint32_t first = 0;
    for (uint32_t rows : c.shards) {
        std::vector<uint32_t> values(size_t(rows) * c.cols);
        for (uint32_t r = 0; r < rows; ++r) for (uint32_t k = 0; k < c.cols; ++k) values[size_t(r) * c.cols + k] = weightFor(c, first + r, k);
        auto words = chandra::gemv_calibration::pack(values, c.bf16);
        b.weight.shardFirstRows.push_back(first);
        b.weight.shards.push_back(chandra::gemv_calibration::upload(d, words, uint64_t(rows) * c.cols, c.bf16));
        first += rows;
    }
    check(first == c.rows, "case shards cover rows");
    if (c.bias) {
        std::vector<uint32_t> values(c.rows);
        for (uint32_t r = 0; r < c.rows; ++r) values[r] = biasFor(c, r);
        b.bias.rows = c.rows; b.bias.cols = 1; b.bias.bf16 = c.bias == 1; b.bias.shardFirstRows = {0};
        b.bias.shards.push_back(chandra::gemv_calibration::upload(d, chandra::gemv_calibration::pack(values, c.bias == 1), c.rows, c.bias == 1));
    }
    return b;
}
// Independent reference from generator values (never packed words): ascending FP32 from +0.
std::vector<uint32_t> reference(const Case& c, bool rounded, bool withBias) {
    std::vector<uint32_t> out(size_t(c.batch) * c.rows);
    for (uint32_t b = 0; b < c.batch; ++b) for (uint32_t o = 0; o < c.rows; ++o) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < c.cols; ++k) sum = add32(sum, mul32(inputFor(c, b, k), fromBits(weightFor(c, o, k))));
        if (withBias) sum = add32(sum, fromBits(biasFor(c, o)));
        out[size_t(b) * c.rows + o] = bits(rounded ? bf16Rne(sum) : sum);
    }
    return out;
}
bool same(uint32_t observed, uint32_t expected) {
    bool en = (expected & 0x7fffffffu) > 0x7f800000u, on = (observed & 0x7fffffffu) > 0x7f800000u;
    return en ? on : observed == expected;
}
std::vector<Case> cases() {
    return {
        {"k7_bf16_shards_3_2_bf16_bias", 5, 7, 1, {3, 2}, true, 1, 0},
        {"k7_fp32_weights_fp32_bias", 5, 7, 1, {5}, false, 2, 0},
        {"k1_minimum", 3, 1, 1, {3}, true, 0, 0},
        {"k513_rows9_odd_shards_4_5", 9, 513, 1, {4, 5}, true, 0, 0},
        {"k1027_fp32_ragged_tail_shards_6_7", 13, 1027, 1, {6, 7}, false, 1, 0},
        {"k2051_rows2600_shards_1500_1100_bf16_bias", 2600, 2051, 1, {1500, 1100}, true, 1, 0},
        {"k9216_rows1030_two_chunks", 1030, 9216, 1, {1030}, true, 0, 0},
        {"k9216_fp32_weights_shards_9_8_fp32_bias", 17, 9216, 1, {9, 8}, false, 2, 0},
        {"k33_specials_signed_zero_inf_nan", 40, 33, 1, {40}, true, 1, 2},
        {"k9216_dyadic", 64, 9216, 1, {64}, true, 1, 1},
        {"batch2_predecessor_route", 37, 515, 2, {20, 17}, true, 1, 0},
        {"batch33_predecessor_route", 21, 129, 33, {21}, false, 2, 0}};
}

// Runs every case through the real linear() and checks outputs against the independent reference.
// Returns a deterministic dispatch/output transcript for comparison with the predecessor build.
std::string callable(bool expectGemv, bool verbose) {
    std::ostringstream transcript; uint64_t calls = 0, dispatches = 0;
    Device d(L"unused", "fake:00.0", "");
    for (const auto& c : cases()) {
        auto b = build(d, c);
        for (bool rounded : {false, true}) {
            const size_t before = dispatchLog.size();
            auto out = linear(d, b.input, b.weight, c.batch, rounded, c.bias ? &b.bias : nullptr);
            auto observed = d.readWords(out);
            auto expected = reference(c, rounded, c.bias != 0);
            check(observed.size() == expected.size(), c.name + " output extent");
            for (size_t i = 0; i < expected.size(); ++i)
                check(same(observed[i], expected[i]), c.name + " output " + std::to_string(i) + " differs from ascending FP32 reference");
            const std::vector<DispatchRecord> log(dispatchLog.begin() + std::ptrdiff_t(before), dispatchLog.end());
            bool gemvCall = expectGemv && c.batch == 1;
            if (gemvCall) check(gemvStored[out.storage->id].size() == c.rows, c.name + " GEMV did not store every output exactly once");
            // Independent geometry forecast for the candidate route.
            size_t index = 0; uint32_t first = 0;
            for (uint32_t rows : c.shards) {
                for (uint32_t chunk = 0; chunk < rows; chunk += 1024) {
                    uint32_t current = std::min(rows - chunk, 1024u);
                    if (gemvCall) {
                        check(index < log.size(), c.name + " missing GEMV dispatch");
                        const auto& r = log[index++];
                        uint32_t flags = uint32_t(c.bf16) | (uint32_t(rounded) << 1) | (uint32_t(c.bias != 0) << 2) | (uint32_t(c.bias == 1) << 3);
                        std::vector<uint32_t> want{c.cols, c.rows, 1, first + chunk, current, 0, flags, chunk};
                        check(r.shader == "runtime/linear_gemv.hlsl" && r.params == want && r.x == (current + 7) / 8 && r.y == 1 && r.z == 1,
                              c.name + " GEMV dispatch shader/params/geometry");
                    } else {
                        for (uint32_t row = 0; row < c.batch; row += 32) {
                            check(index < log.size() && log[index].shader == "runtime/linear.hlsl", c.name + " predecessor route");
                            ++index;
                        }
                    }
                }
                first += rows;
            }
            check(index == log.size(), c.name + " dispatch count differs from forecast");
            for (const auto& r : log) {
                transcript << c.name << " rounded=" << rounded << " " << r.shader << " groups=" << r.x << "," << r.y << "," << r.z << " params=";
                for (auto v : r.params) transcript << v << ",";
                transcript << " srv=";
                for (auto v : r.inputs) transcript << v << ",";
                transcript << " uav=" << r.outputs.at(0) << "\n";
            }
            transcript << c.name << " rounded=" << rounded << " output_fnv1a64=" << hex64(fnv1a(observed)) << "\n";
            ++calls; dispatches += log.size();
            d.drain();
        }
    }
    check(d.trackedBufferBytes() == 0, "callable cases released all buffers");
    if (verbose) std::cout << "callable cases: " << cases().size() << " shapes, " << calls << " linear calls, " << dispatches
                           << " dispatches, all outputs bit-equal to the independent ascending FP32 reference\n";
    return transcript.str();
}
}

#ifndef GEMV_TEST_PREDECESSOR // Candidate-only modes; the predecessor build provides --transcript.
namespace {
// ---- Numerical observations: ordered GEMV, ascending FP32, a split-K tree alternative, binary64. ----
// split32 models the rejected alternative: 32 lanes per output, lane l owns element pairs (2m, 2m+1)
// with m % 32 == l (one packed BF16 word per lane per iteration), each lane sums its terms in
// ascending order from +0 with FP32 rounding, then a fixed groupshared tree adds lane l+s into
// lane l for s = 16, 8, 4, 2, 1. Every operation is one rounded FP32 operation with D3D11 flushing.
float ascending(const std::vector<float>& a, const std::vector<float>& w) {
    float sum = 0.0f;
    for (size_t k = 0; k < a.size(); ++k) sum = add32(sum, mul32(a[k], w[k]));
    return sum;
}
float split32(const std::vector<float>& a, const std::vector<float>& w) {
    float lanes[32] = {};
    for (size_t k = 0; k < a.size(); ++k) lanes[(k / 2) % 32] = add32(lanes[(k / 2) % 32], mul32(a[k], w[k]));
    for (uint32_t s = 16; s >= 1; s /= 2) for (uint32_t l = 0; l < s; ++l) lanes[l] = add32(lanes[l], lanes[l + s]);
    return lanes[0];
}
// Higher precision reference: exact binary64 products (24+8 significant bits) with Neumaier
// compensated binary64 summation; D3D11 flushing is applied to operands only.
struct Higher { double value, magnitude; };
Higher higher(const std::vector<float>& a, const std::vector<float>& w) {
    double sum = 0, compensation = 0, magnitude = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        double term = double(ftz(a[k])) * double(ftz(w[k])), next = sum + term;
        compensation += std::fabs(sum) >= std::fabs(term) ? (sum - next) + term : (term - next) + sum;
        sum = next; magnitude += std::fabs(term);
    }
    return {sum + compensation, magnitude};
}
// Ordered GEMV through the kernel transliteration itself (K tiles, staging, BF16 packing).
float orderedGemv(const std::vector<float>& a, const std::vector<float>& w) {
    std::vector<uint32_t> in(a.size()), wb(w.size());
    for (size_t k = 0; k < a.size(); ++k) { in[k] = bits(a[k]); wb[k] = bits(w[k]); check((wb[k] & 0xffffu) == 0, "BF16 weight"); }
    auto packed = chandra::gemv_calibration::pack(wb, true);
    std::vector<uint32_t> biasWords(1, 0), out(1, 0); std::set<uint32_t> stored;
    Params p{uint32_t(a.size()), 1, 1, 0, 1, 0, 1, 0};
    emulateGemv(p, 1, 1, 1, Raw{in, "input"}, Raw{packed, "weight"}, Raw{biasWords, "bias"}, out, stored);
    return fromBits(out[0]);
}
float bf16Value(uint64_t r, int lowExponent, int span) {
    uint32_t exponent = uint32_t(127 + lowExponent + int((r >> 40) % uint64_t(span)));
    return fromBits(uint32_t((r >> 63) << 31) | (exponent << 23) | (uint32_t(r) & 0x7f0000u));
}
float normalish(uint64_t r) { // Irwin-Hall(4) approximation of N(0,1), then BF16 RNE like graph activations.
    double s = 0; for (int i = 0; i < 4; ++i) s += double((r >> (16 * i)) & 0xffffu) / 65535.0;
    return bf16Rne(float((s - 2.0) * std::sqrt(3.0)));
}
struct Vectors { std::vector<float> a, w; };
using Maker = Vectors (*)(uint32_t output);
Vectors randomDot(uint32_t output, uint32_t k, double scale) {
    Vectors v{std::vector<float>(k), std::vector<float>(k)};
    for (uint32_t i = 0; i < k; ++i) {
        v.a[i] = normalish(mix(0x4000000000ull * (output + 1) + i));
        v.w[i] = bf16Rne(float(double(normalish(mix(0x5000000000ull * (output + 1) + i))) * scale));
    }
    return v;
}
Json observe(const char* name, uint32_t outputs, const std::function<Vectors(uint32_t)>& make) {
    double ordAbs = 0, splitAbs = 0, ordUlp = 0, splitUlp = 0, ordRel = 0, splitRel = 0;
    uint64_t orderedMismatch = 0, splitDiffers = 0, bf16Differs = 0, splitCloser = 0, ascendingCloser = 0, nonfinite = 0;
    Json nonfiniteExamples = Json::array();
    const uint64_t flushedBefore = flushed;
    for (uint32_t o = 0; o < outputs; ++o) {
        auto v = make(o);
        float asc = ascending(v.a, v.w), ord = orderedGemv(v.a, v.w), spl = split32(v.a, v.w);
        orderedMismatch += !same(bits(ord), bits(asc)) || ((bits(asc) & 0x7fffffffu) <= 0x7f800000u && bits(ord) != bits(asc));
        splitDiffers += bits(spl) != bits(asc);
        bf16Differs += roundedBF16Bits(bits(spl)) != roundedBF16Bits(bits(asc));
        if (!std::isfinite(asc) || !std::isfinite(spl)) {
            ++nonfinite;
            if (nonfiniteExamples.size() < 4) nonfiniteExamples.push_back({{"ascending_bits", hex64(bits(asc)).substr(8)}, {"split32_bits", hex64(bits(spl)).substr(8)}});
            continue;
        }
        auto h = higher(v.a, v.w);
        double ulp = std::ldexp(1.0, std::max(std::ilogb(h.value == 0 ? FLT_MIN : h.value), -126) - 23);
        double ea = std::fabs(double(asc) - h.value), es = std::fabs(double(spl) - h.value);
        ordAbs = std::max(ordAbs, ea); splitAbs = std::max(splitAbs, es);
        ordUlp = std::max(ordUlp, ea / ulp); splitUlp = std::max(splitUlp, es / ulp);
        if (h.magnitude > 0) { ordRel = std::max(ordRel, ea / h.magnitude); splitRel = std::max(splitRel, es / h.magnitude); }
        splitCloser += es < ea; ascendingCloser += ea < es;
    }
    check(orderedMismatch == 0, std::string(name) + ": ordered GEMV transliteration differs from ascending FP32");
    return {{"scenario", name}, {"outputs", outputs}, {"ordered_gemv_bit_equal_to_ascending_fp32", orderedMismatch == 0},
            {"split32_fp32_bits_differ_from_ascending", splitDiffers}, {"split32_bf16_rounded_differs_from_ascending", bf16Differs},
            {"nonfinite_outputs", nonfinite}, {"nonfinite_examples", nonfiniteExamples},
            {"ascending_max_abs_error_vs_higher", ordAbs}, {"split32_max_abs_error_vs_higher", splitAbs},
            {"ascending_max_error_fp32_ulps_of_reference", ordUlp}, {"split32_max_error_fp32_ulps_of_reference", splitUlp},
            {"ascending_max_error_over_sum_abs_terms", ordRel}, {"split32_max_error_over_sum_abs_terms", splitRel},
            {"outputs_split32_closer", splitCloser}, {"outputs_ascending_closer", ascendingCloser},
            {"d3d11_flushed_subnormal_operations", flushed - flushedBefore}};
}
Json numerics() {
    Json rows = Json::array();
    for (uint32_t k : {2560u, 4096u, 9216u})
        rows.push_back(observe(k == 2560 ? "random_k2560_bf16_activation_weight_scale_0.02" : k == 4096 ? "random_k4096_bf16_activation_weight_scale_0.02"
                                                                                                        : "random_k9216_bf16_activation_weight_scale_0.02",
                               64, [k](uint32_t o) { return randomDot(o, k, 0.02); }));
    rows.push_back(observe("cancellation_large_pair_absorbs_small_terms", 64, [](uint32_t o) {
        auto v = randomDot(o, 4096, 0.02);
        v.a[0] = 65536.0f; v.w[0] = 16.0f; v.a[4095] = -65536.0f; v.w[4095] = 16.0f; // +2^20 first, -2^20 last
        return v; }));
    rows.push_back(observe("cancellation_alternating_magnitude_2^12", 64, [](uint32_t o) {
        auto v = randomDot(o, 2560, 0.02);
        for (uint32_t i = 0; i < 2560; i += 64) { v.a[i] = 4096.0f; v.w[i] = (i / 64) % 2 ? -1.0f : 1.0f; }
        return v; }));
    rows.push_back(observe("signed_zero_products", 16, [](uint32_t o) {
        Vectors v{std::vector<float>(512), std::vector<float>(512)};
        for (uint32_t i = 0; i < 512; ++i) {
            uint64_t r = mix(0x6000000000ull * (o + 1) + i);
            v.a[i] = (r & 1) ? -0.0f : bf16Value(r, -3, 6);
            v.w[i] = (r & 2) ? -0.0f : ((r & 1) ? bf16Value(r >> 3, -3, 6) : 0.0f);
            if (o % 2) v.a[i] = -std::fabs(v.a[i]);
        }
        return v; }));
    rows.push_back(observe("wide_magnitudes_exponents_-60_to_60", 64, [](uint32_t o) {
        Vectors v{std::vector<float>(4096), std::vector<float>(4096)};
        for (uint32_t i = 0; i < 4096; ++i) { v.a[i] = bf16Value(mix(0x7000000000ull * (o + 1) + i), -60, 61); v.w[i] = bf16Value(mix(0x8000000000ull * (o + 1) + i), -1, 61); }
        return v; }));
    rows.push_back(observe("bf16_boundary_ties_near_1+2^-8", 256, [](uint32_t o) {
        // Exact value 1 + 2^-8 (a BF16 tie) plus +/- tiny terms whose FP32 rounding depends on order.
        Vectors v{std::vector<float>(2048), std::vector<float>(2048)};
        for (uint32_t i = 0; i < 2048; ++i) {
            uint64_t r = mix(0x9000000000ull * (o + 1) + i);
            v.a[i] = bf16Value(r, -30, 8); v.w[i] = (r & 4) ? 1.0f : -1.0f;
        }
        v.a[0] = 1.0f; v.w[0] = 1.0f; v.a[1] = 0.00390625f; v.w[1] = 1.0f;
        return v; }));
    rows.push_back(observe("overflow_order_MAX_MAX_-MAX_-MAX", 1, [](uint32_t) {
        float big = fromBits(0x7f7f0000u); // Largest finite BF16.
        return Vectors{{big, big, big, big}, {1.0f, 1.0f, -1.0f, -1.0f}}; }));
    return rows;
}

// ---- Process-level modes, selected by gemv_candidate_test.sh. ----
int runCandidate() {
    check(std::string(selection()) == "ordered", "candidate mode requires CHANDRA_EXPERIMENTAL_GEMV_B1=ordered");
    callable(true, true);
    Device d(L"unused", "fake:00.0", "");
    Json report = chandra::gemv_calibration::initialReport();
    chandra::gemv_calibration::Deadline deadline;
    const size_t before = dispatchLog.size();
    chandra::gemv_calibration::run(d, report, deadline);
    const uint64_t planDispatches = dispatchLog.size() - before;
    check(report.at("passed") == true && report.at("invalid_admission").at("cases").size() == 9, "calibration plan on emulator");
    uint64_t phaseDispatches = 0;
    for (const auto& phase : report.at("phases")) {
        check(phase.at("passed") == true && phase.at("complete_reads") == true, "plan phase passed");
        phaseDispatches += phase.at("timing_summary").at("dispatches").get<uint64_t>();
    }
    check(phaseDispatches == planDispatches, "plan dispatch count equals profiled dispatch count");
    check(report.dump().size() < 8 * 1024 * 1024, "plan receipt bounded");
    Json summary = Json::array();
    for (const auto& phase : report.at("phases"))
        summary.push_back({{"name", phase.at("name")}, {"dispatches", phase.at("timing_summary").at("dispatches")},
                           {"groups_per_call", phase.at("expected_groups_per_call")},
                           {"forecast_bytes", phase.at("forecast_tracked_plus_readback_bytes")},
                           {"expected_bf16_ties", phase.at("expected_bf16_ties_in_fp32_dots")},
                           {"cpu_observations", phase.contains("cpu_observations") ? phase.at("cpu_observations") : Json()}});
    std::cout << "calibration plan on emulated device: PASS, receipt bytes " << report.dump(2).size()
              << ", maximum forecast bytes " << report.at("maximum_phase_tracked_plus_readback_bytes") << "\n"
              << summary.dump(1) << "\n";
    std::cout << "GEMV workgroups emulated: " << gemvGroupsEmulated << "\n";
    return 0;
}
int runPredecessor() {
    check(std::string(selection()) == "predecessor", "default mode requires CHANDRA_EXPERIMENTAL_GEMV_B1 unset or 0");
    callable(false, true);
    // The candidate calibration must detect that no candidate dispatch occurred.
    Device d(L"unused", "fake:00.0", "");
    Json report = chandra::gemv_calibration::initialReport();
    chandra::gemv_calibration::Deadline deadline;
    std::string message;
    try { chandra::gemv_calibration::run(d, report, deadline); } catch (const std::exception& error) { message = error.what(); }
    check(message.find("no silent fallback") != std::string::npos, "calibration refuses predecessor dispatches: " + message);
    std::cout << "calibration plan refuses predecessor route: " << message << "\n";
    return 0;
}
int runInvalid() {
    std::string message;
    try { (void)selection(); } catch (const std::invalid_argument& error) { message = error.what(); }
    check(!message.empty(), "invalid selector must be refused by the accessor");
    Device d(L"unused", "fake:00.0", "");
    Case c{"invalid", 4, 8, 1, {4}, true, 0, 0};
    auto b = build(d, c);
    for (uint32_t batch : {1u, 2u}) {
        const uint64_t tracked = d.trackedBufferBytes(); const size_t logged = dispatchLog.size(); std::string refused;
        Buffer input = batch == 1 ? b.input : d.floats(16);
        const uint64_t trackedWithInput = d.trackedBufferBytes();
        try { (void)linear(d, input, b.weight, batch, true, nullptr); } catch (const std::invalid_argument& error) { refused = error.what(); }
        check(refused == message, "linear refuses invalid selector for batch " + std::to_string(batch));
        check(d.trackedBufferBytes() == trackedWithInput && dispatchLog.size() == logged, "refusal before allocation/dispatch");
        (void)tracked;
    }
    std::cout << "invalid selector refused before allocation: " << message << "\n";
    return 0;
}
}
#endif

int main(int argc, char** argv) {
    try {
        std::string mode = argc == 2 ? argv[1] : "";
        if (mode == "--transcript") { std::cout << callable(false, false); return 0; }
#ifndef GEMV_TEST_PREDECESSOR
        if (mode == "--candidate") return runCandidate();
        if (mode == "--candidate-transcript") {
            check(std::string(selection()) == "ordered", "candidate transcript requires the ordered selector");
            std::cout << callable(true, false); return 0;
        }
        if (mode == "--predecessor") return runPredecessor();
        if (mode == "--invalid") return runInvalid();
        if (mode == "--numerics") { std::cout << numerics().dump(1) << "\n"; return 0; }
#endif
        std::cerr << "usage: gemv-candidate-test --transcript|--candidate|--predecessor|--invalid|--numerics\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
