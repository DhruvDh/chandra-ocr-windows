// New code, MPL-2.0. CPU-only tests for head_audit.h and the native-1 failure mechanism. No GPU, D3D11,
// Windows or model weights: a test-only Device keeps buffers as host words and executes the two audit
// shaders and embedding.hlsl on the CPU (out-of-range access throws instead of D3D11's zero/discard).
// The real portable importer shard planner and operators.cpp embedding() are linked unchanged.
#include "model_weights.cpp"
#include "head_audit.h"
#include "diagnostics.h"
#include "gemv_variants.h"
#include <array>
#include <cstdio>
#include <iostream>

namespace chandra::dc {
struct Storage { std::vector<uint32_t> data; };
struct Device::Impl {};
Device::Device(const std::wstring&, const std::string&, const std::string&) : impl(std::make_unique<Impl>()) {}
Device::~Device() = default;
namespace fake {
struct Dispatch { std::string shader; std::array<uint32_t, 8> p{}; uint32_t x = 0; };
std::vector<Dispatch> log; uint64_t readbackBytes = 0, largestReadback = 0;
// Fault injection for embedding.hlsl: XOR into the stored bits of one output element for one token.
uint32_t embeddingFaultToken = UINT32_MAX, embeddingFaultXor = 0;
// weight_row_checksum.hlsl writes a wrong checksum for this global row (a read or dispatch fault, not storage).
uint32_t checksumFaultRow = UINT32_MAX;
// >= 0: that many readbacks succeed, then the next one throws once (device removal, failed map).
int readbacksUntilFailure = -1;
void reset() { log.clear(); readbackBytes = largestReadback = 0; embeddingFaultToken = checksumFaultRow = UINT32_MAX; embeddingFaultXor = 0; readbacksUntilFailure = -1; }
Buffer make(uint32_t count, const void* initial) {
    if (!count || uint64_t(count) * 4 > 128ull << 20) throw std::runtime_error("Fake buffer must be nonempty and at most 128 MiB");
    auto s = std::make_shared<Storage>(); s->data.assign(count, 0u);
    if (initial) std::memcpy(s->data.data(), initial, size_t(count) * 4);
    return {s, count, count, false};
}
uint32_t load(const Buffer& b, uint64_t byte) {
    if (byte % 4 || byte / 4 >= b.storage->data.size()) throw std::runtime_error("Fake shader load outside raw buffer");
    return b.storage->data[size_t(byte / 4)];
}
void store(Buffer& b, uint64_t byte, uint32_t v) {
    if (byte % 4 || byte / 4 >= b.storage->data.size()) throw std::runtime_error("Fake shader store outside raw buffer");
    b.storage->data[size_t(byte / 4)] = v;
}
}
Buffer Device::floats(uint32_t count, const float* initial) { return fake::make(count, initial); }
Buffer Device::words(uint32_t count, const uint32_t* initial) { return fake::make(count, initial); }
void Device::upload(Buffer& b, const void* values, uint32_t bytes) {
    if (!values || bytes != b.storage->data.size() * 4) throw std::runtime_error("Complete fake upload required");
    std::memcpy(b.storage->data.data(), values, bytes);
}
void Device::drain(uint32_t) {}
std::vector<uint32_t> Device::readWords(const Buffer& b) {
    if (fake::readbacksUntilFailure >= 0 && fake::readbacksUntilFailure-- == 0) throw std::runtime_error("Injected fake readback failure");
    fake::readbackBytes += b.storage->data.size() * 4; fake::largestReadback = std::max<uint64_t>(fake::largestReadback, b.storage->data.size() * 4);
    return b.storage->data;
}
std::vector<float> Device::readFloats(const Buffer& b) {
    auto w = readWords(b); std::vector<float> f(w.size()); std::memcpy(f.data(), w.data(), w.size() * 4); return f;
}
void Device::dispatch(const std::string& shader, const std::vector<const Buffer*>& in, const std::vector<Buffer*>& out, const void* params, uint32_t bytes, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z || x > 65535 || y > 65535 || bytes > 32 || out.size() != 1) throw std::runtime_error("Fake dispatch shape invalid");
    fake::Dispatch d; d.shader = shader; d.x = x; std::memcpy(d.p.data(), params, bytes); fake::log.push_back(d);
    const auto& p = d.p; Buffer& o = *out[0];
    if (shader == head_audit::checksumShader) {
        for (uint32_t row = 0; row < x * 64; ++row) {
            if (row >= p[1]) continue;
            uint32_t base = (p[2] + row) * p[0], sum = 0;
            for (uint32_t i = 0; i < p[0]; ++i) sum += (2 * i + 1) * fake::load(*in[0], uint64_t(base + i) * 4);
            fake::store(o, uint64_t(p[3] + row) * 4, p[3] + row == fake::checksumFaultRow ? sum ^ 1u : sum);
        }
    } else if (shader == head_audit::copyShader) {
        for (uint32_t i = 0; i < x * 256; ++i) if (i < p[1]) fake::store(o, uint64_t(p[2] + i) * 4, fake::load(*in[0], uint64_t(p[0] + i) * 4));
    } else if (shader == "runtime/embedding.hlsl") { // width, rows, firstWeightRow, shardRows, tokenFirst, flags
        for (uint32_t local = 0; local < y; ++local) for (uint32_t column = 0; column < x * 256; ++column) {
            if (column >= p[0] || local >= p[1]) continue;
            uint32_t row = p[4] + local, token = fake::load(*in[0], uint64_t(row) * 4);
            if (token < p[2] || token - p[2] >= p[3]) continue;
            uint32_t index = (token - p[2]) * p[0] + column, word = fake::load(*in[1], uint64_t(index >> 1) * 4);
            uint32_t bits = ((word >> ((index & 1) * 16)) & 0xffff) << 16;
            if (token == fake::embeddingFaultToken && column == 7) bits ^= fake::embeddingFaultXor;
            fake::store(o, uint64_t(row * p[0] + column) * 4, bits);
        }
    } else throw std::runtime_error("Fake device has no shader " + shader);
}
}

namespace {
using namespace chandra::dc;
using namespace chandra::dc::head_audit;
unsigned passed = 0;
void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error("FAILED: " + what); ++passed; }
template<class F> void rejects(F f, const std::string& what) { bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } check(caught, "refuses " + what); }
struct Lcg { uint64_t s; explicit Lcg(uint64_t seed) : s(seed) {} uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(s >> 33); }
    double uniform() { return (next() + 0.5) / 2147483648.0; } double normal() { return std::sqrt(-2 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()); } };
uint16_t bf16Bits(float v) { return uint16_t(asBits(bf16Rne(v)) >> 16); }
// Trained-like packed BF16 row: N(0, 0.02) rounded to BF16, as in the pinned embedding statistics' order of magnitude.
std::vector<uint32_t> trainedRow(Lcg& g, uint32_t cols) {
    std::vector<uint32_t> w(cols / 2);
    for (uint32_t k = 0; k < cols; ++k) w[k / 2] |= uint32_t(bf16Bits(float(0.02 * g.normal()))) << ((k & 1) * 16);
    return w;
}
// Hidden row resembling the native-0 final-norm dumps (||y||2 ~ 150, BF16 values).
std::vector<float> hiddenRow(Lcg& g, uint32_t cols, double scale = 3.0) { std::vector<float> y(cols); for (auto& v : y) v = bf16Rne(float(scale * g.normal())); return y; }
// norm.hlsl RMS path, onePlusWeight, BF16 output: 256 lanes ascending, pairwise tree, rsqrt(mean+eps).
std::vector<float> rmsNormEmulated(const std::vector<float>& x, const std::vector<float>& gain, float eps) {
    const uint32_t n = uint32_t(x.size()); float partial[256];
    for (uint32_t lane = 0; lane < 256; ++lane) { float sum = 0.0f; for (uint32_t j = lane; j < n; j += 256) sum = flush(sum + flush(x[j] * x[j])); partial[lane] = sum; }
    for (uint32_t stride = 128; stride; stride >>= 1) for (uint32_t lane = 0; lane < stride; ++lane) partial[lane] = flush(partial[lane] + partial[lane + stride]);
    const float variance = partial[0] / float(n), scale = 1.0f / std::sqrt(variance + eps);
    std::vector<float> y(n);
    for (uint32_t j = 0; j < n; ++j) y[j] = bf16Rne(flush(flush(x[j] * scale) * flush(1.0f + gain[j])));
    return y;
}

// A synthetic tied head in packed BF16 shards plus a payload holding the head then the final-norm gain.
struct Head {
    uint32_t rows = 0, cols = 0; std::vector<uint32_t> payload; Weight weight; std::vector<uint32_t> shardRows;
    uint64_t normOffsetBytes() const { return uint64_t(rows) * cols * 2; }
    const uint32_t* sourceRow(uint32_t r) const { return payload.data() + size_t(r) * cols / 2; }
};
Head makeHead(Device& d, std::vector<uint32_t> shardRows, uint32_t cols, uint64_t seed) {
    Head h; h.cols = cols; h.shardRows = shardRows; Lcg g(seed);
    for (auto n : shardRows) h.rows += n;
    for (uint32_t r = 0; r < h.rows; ++r) { auto w = trainedRow(g, cols); h.payload.insert(h.payload.end(), w.begin(), w.end()); }
    // Final-norm gain w with 1+w in [1,5]: native-0 final-norm rows (||y||2 ~ 106..163) imply max|1+w| >= 3.
    for (uint32_t k = 0; k < cols; k += 2) h.payload.push_back(uint32_t(bf16Bits(float(4.0 * g.uniform()))) | uint32_t(bf16Bits(float(4.0 * g.uniform()))) << 16);
    h.weight.rows = h.rows; h.weight.cols = cols; h.weight.bf16 = true; uint32_t first = 0;
    for (auto n : shardRows) {
        Buffer b = d.words(n * cols / 2); b.packedBF16 = true; b.logicalElements = uint64_t(n) * cols;
        d.upload(b, h.payload.data() + size_t(first) * cols / 2, n * cols * 2);
        h.weight.shards.push_back(b); h.weight.shardFirstRows.push_back(first); first += n;
    }
    return h;
}
uint32_t* stored(Head& h, uint32_t row) {
    for (size_t s = h.weight.shards.size(); s-- > 0;) if (row >= h.weight.shardFirstRows[s]) return h.weight.shards[s].storage->data.data() + size_t(row - h.weight.shardFirstRows[s]) * h.cols / 2;
    throw std::out_of_range("row");
}
// Finite but enormous BF16 row aligned with y: every element is +/-2^90 (~1.2e27) with sign(y_k).
void corruptAligned(uint32_t* words, const std::vector<float>& y) {
    for (uint32_t k = 0; k < y.size(); ++k) {
        uint32_t bits = uint32_t(bf16Bits(y[k] < 0 ? -std::ldexp(1.0f, 90) : std::ldexp(1.0f, 90)));
        words[k / 2] = (words[k / 2] & ~(0xffffu << ((k & 1) * 16))) | (bits << ((k & 1) * 16));
    }
}
std::vector<float> gpuLogits(Head& h, const std::vector<float>& y) {
    std::vector<float> l(h.rows); for (uint32_t r = 0; r < h.rows; ++r) l[r] = headLogit(y.data(), stored(h, r), h.cols); return l;
}
std::vector<float> gain(const Head& h) { std::vector<float> w(h.cols); for (uint32_t k = 0; k < h.cols; ++k) w[k] = bf16At(h.payload.data() + h.rows * h.cols / 2, k); return w; }

struct Run {
    Device device{L"", "", ""}; Head head; std::map<std::string, std::vector<uint8_t>> files; std::unique_ptr<Auditor> audit;
    TextPositions positions{{0}, {0}, {0}}; bool failSource = false; uint32_t calls = 0;
    Run(std::vector<uint32_t> shards, uint32_t cols, uint64_t seed, uint32_t requested = 2) : head(makeHead(device, shards, cols, seed)) {
        fake::reset();
        audit = std::make_unique<Auditor>(device, head.weight, Tensor{0}, Tensor{head.normOffsetBytes()}, cols, requested,
            [this](uint64_t offset, void* dst, uint32_t bytes) {
                if (failSource) throw std::runtime_error("Injected source read failure");
                if (offset % 4 || offset + bytes > head.payload.size() * 4) throw std::runtime_error("source read outside payload");
                std::memcpy(dst, reinterpret_cast<const uint8_t*>(head.payload.data()) + offset, bytes); },
            [this](const std::string& name, const void* p, size_t n) { check(diagnostics::safeName(name), "row dump name passes the CLI safeName rule"); check(files.emplace(name, std::vector<uint8_t>(static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n)).second, "row dump names are unique"); },
            [](const void* p, size_t n) { return diagnostics::sha256(p, n); }, [] { return Json{{"fake", true}}; });
    }
    // One graph call as TextModel::forward reports it: final norm of the call's last row, then logits.
    // Each call has its own tokens_before, as prefill and every cached advance do.
    void predict(const std::vector<float>& y, const std::vector<float>& logits) {
        Buffer hidden = device.floats(uint32_t(y.size()), y.data()), values = device.floats(uint32_t(logits.size()), logits.data()); const uint32_t before = calls++;
        audit->observe({TextStage::FinalNorm, UINT32_MAX, hidden, 1, head.cols, before, 0, 1, 0, 1, 1, positions, nullptr});
        audit->observe({TextStage::Logits, UINT32_MAX, values, 1, head.rows, before, 0, 1, 0, 1, 1, positions, nullptr});
    }
    Json checkpoint(size_t i) { return audit->finish()["checkpoints"].at(i); }
    std::string stage() { return audit->finish()["first_invalid_stage"]; }
};
const std::vector<uint32_t> smallShards{40, 40, 17};
// Inverse of an odd number modulo 2^32 (Newton iteration).
uint32_t inverseOdd(uint32_t a) { uint32_t x = a; for (int i = 0; i < 5; ++i) x *= 2u - a * x; return x; }
// Adds delta to word i and the compensating multiple to word j, so the odd-multiplier checksum is unchanged.
void collide(uint32_t* row, uint32_t i, uint32_t j, uint32_t delta) { row[i] += delta; row[j] -= (2u * i + 1u) * delta * inverseOdd(2u * j + 1u); }
bool contains(const Json& list, const Json& value) { return std::find(list.begin(), list.end(), value) != list.end(); }

void pinnedShardArithmetic() {
    weight_detail::Tensor t; t.name = "model.language_model.embed_tokens.weight"; t.rows = 248320; t.cols = 2560; t.elements = 248320ull * 2560; t.bf16 = true;
    auto plan = weight_detail::shardPlan(t);
    check(plan.size() == 10 && plan[0].rows == 26214 && plan[9].rows == 12394 && plan[1].storageBytes == 134215680, "pinned embed plan is 9x26214 rows + 12394 at 128 MiB");
    check(plan[1].firstRow + plan[1].rows - 1 == 52427, "token 52427 is the final row of embed shard 1");
    check(uint64_t(plan[1].rows - 1) * 5120 + 5120 == plan[1].storageBytes, "row 52427 ends exactly at shard 1's last stored byte");
    const auto gemv = chandra::gemv_variants::shape(chandra::gemv_variants::Route::ordered);
    check(26214 % 1024 == 614 && gemv->groups(614) == 77 && 25600 + 613 == 26213, "ordered GEMV covers local row 26213 in the last 614-output dispatch, last group");
}
void failureSignature() {
    Lcg g(7); const uint32_t cols = 2560, bad = 63; Device d(L"", "", ""); Head h = makeHead(d, {32, 32}, cols, 11); // bad: final row of shard 1.
    auto y = hiddenRow(g, cols); auto w = gain(h); double maxGain = 0; for (float v : w) maxGain = std::max(maxGain, std::fabs(double(1.0f + v)));
    auto healthy = gpuLogits(h, y); auto before = rank(healthy);
    check(std::fabs(before.bestValue) < 100 && before.ties == 1, "trained-like head gives ordinary logits");
    corruptAligned(stored(h, bad), y);
    auto broken = gpuLogits(h, y); auto even = rank(broken);
    check(even.best == bad && std::isfinite(even.bestValue) && even.bestValue > 1e29f && even.bestValue < 1e32f, "a finite 2^90-scale stored row becomes the finite ~1e30 argmax");
    check(even.runnerUp == before.best && even.runnerUpValue == before.bestValue, "every other logit and the runner-up are unchanged");
    std::vector<float> x(cols); for (uint32_t k = 0; k < cols; ++k) x[k] = bf16At(stored(h, bad), k);
    auto zero = rmsNormEmulated(x, w, 1e-6f); bool allZero = std::all_of(zero.begin(), zero.end(), [](float v) { return v == 0.0f; });
    check(allZero, "RMSNorm of that embedding overflows sum(x^2) to +inf and yields exact zeros, not NaN");
    for (auto& v : x) v = bf16Rne(v + 5.0f);
    auto perturbed = rmsNormEmulated(x, w, 1e-6f);
    check(std::all_of(perturbed.begin(), perturbed.end(), [](float v) { return v == 0.0f; }), "finite sublayer additions to the huge residual keep the final norm at zero");
    auto odd = rank(gpuLogits(h, zero));
    check(odd.best == 0 && odd.bestValue == 0.0f && odd.ties == h.rows && odd.runnerUpValue == 0.0f, "zero hidden ties every logit at 0 so greedy picks ID 0 (odd-step signature)");
    std::vector<float> normal(cols); for (uint32_t k = 0; k < cols; ++k) normal[k] = bf16At(stored(h, 0), k);
    auto ordinary = rmsNormEmulated(normal, w, 1e-6f); double n2 = 0; for (float v : ordinary) n2 += double(v) * v;
    check(n2 > 1000 && std::sqrt(n2) <= normContract(cols, maxGain), "an ordinary embedding row normalizes to a nonzero row within the RMSNorm contract");
    std::vector<float> nothing(cols, 0.0f); auto legitimate = rmsNormEmulated(nothing, w, 1e-6f);
    check(std::all_of(legitimate.begin(), legitimate.end(), [](float v) { return v == 0.0f; }), "a zero pre-norm row gives the same all-zero final norm with sum(x^2)=0, no overflow");
}
void contractsAreSound() {
    Lcg g(19); const uint32_t cols = 2560; uint32_t samples = 0;
    for (int t = 0; t < 64; ++t) {
        auto y = hiddenRow(g, cols, 0.5 + 6 * g.uniform()); auto w = trainedRow(g, cols);
        if (t % 2) for (uint32_t k = 0; k < cols; ++k) { // Adversarially aligned signs maximize |dot|.
            uint32_t half = (w[k / 2] >> ((k & 1) * 16)) & 0x7fffu; if (y[k] < 0) half |= 0x8000u;
            w[k / 2] = (w[k / 2] & ~(0xffffu << ((k & 1) * 16))) | (half << ((k & 1) * 16)); }
        double yn = 0, yl = 0, dot = 0, absolute = 0; for (uint32_t k = 0; k < cols; ++k) { yn += double(y[k]) * y[k]; yl += std::fabs(y[k]); dot += double(y[k]) * bf16At(w.data(), k); absolute += std::fabs(double(y[k]) * bf16At(w.data(), k)); }
        auto s = rowStats(w.data(), cols); float logit = headLogit(y.data(), w.data(), cols);
        check(std::fabs(double(logit)) <= logitContract(std::sqrt(yn), s.l2, yl + s.l1, cols), "Cauchy-Schwarz logit contract holds for the emulated ordered head");
        check(std::fabs(double(logit) - dot) <= headErrorBound(absolute, yl + s.l1, cols), "a-priori FP32+BF16 error bound holds");
        std::vector<float> x(cols), gainValues(cols); const double scale = std::ldexp(1.0, int(g.next() % 80) - 40);
        for (uint32_t k = 0; k < cols; ++k) { x[k] = float(scale * g.normal()); gainValues[k] = bf16Rne(float(0.5 * g.normal())); }
        double maxGain = 0; for (float v : gainValues) maxGain = std::max(maxGain, std::fabs(double(1.0f + v)));
        auto yy = rmsNormEmulated(x, gainValues, 1e-6f); double n2 = 0; for (float v : yy) n2 += double(v) * v;
        check(std::sqrt(n2) <= normContract(cols, maxGain), "RMSNorm hidden-row contract holds across 2^-40..2^40 input scales"); ++samples;
    }
    check(samples == 64, "contract sample count");
    { // Cauchy-Schwarz equality with upward BF16 rounding: y = w = 1+2^-7, exact dot 2600.15625 rounds to 2608.
        std::vector<float> y(2560, 1.0078125f); std::vector<uint32_t> w(1280, 0x3f813f81u); auto s = rowStats(w.data(), 2560);
        const float logit = headLogit(y.data(), w.data(), 2560); const double plain = std::sqrt(2560.0) * 1.0078125 * s.l2;
        check(logit == 2608.0f && double(logit) > plain && double(logit) <= logitContract(std::sqrt(2560.0) * 1.0078125, s.l2, 2.0 * 2560 * 1.0078125, 2560),
              "tight Cauchy-Schwarz case needs and satisfies the rounding term");
    }
    for (float gainValue : {0.0f, 0.0078125f, 2.5f}) { // Constant gain makes the RMSNorm contract tight.
        std::vector<float> x(2560), gains(2560, gainValue); for (auto& v : x) v = float(g.normal());
        auto yy = rmsNormEmulated(x, gains, 1e-6f); double n2 = 0; for (float v : yy) n2 += double(v) * v;
        const double tight = std::sqrt(2560.0) * std::fabs(double(1.0f + gainValue));
        check(std::sqrt(n2) >= 0.995 * tight && std::sqrt(n2) <= normContract(2560, std::fabs(double(1.0f + gainValue))), "RMSNorm contract is tight and holds for a constant gain");
    }
    Lcg h(23); std::vector<uint32_t> row = trainedRow(h, 2560); const uint32_t base = rowChecksum(row.data(), 1280);
    for (uint32_t i = 0; i < 1280; ++i) for (uint32_t delta : {1u, 0x8000u, 0x80000000u, h.next() | 1u, h.next() << 7 | 0x80u}) {
        auto changed = row; changed[i] += delta; if (changed[i] == row[i]) continue;
        if (rowChecksum(changed.data(), 1280) == base) throw std::runtime_error("FAILED: single-word change escaped the odd-multiplier checksum");
    }
    ++passed;
}
void healthyAndBounded() {
    Run run({1100, 1100, 300}, 2560, 31, 2); auto c = run.audit->afterImport();
    check(c["checksum_mismatch_rows"] == 0 && c["row_dumps"].empty() && c["exact_comparison"]["rows_compared"] == 0 && c["complete"] == true, "healthy import: no mismatch and nothing copied, so no identity is claimed");
    Lcg g(5); auto y = hiddenRow(g, 2560); run.predict(y, gpuLogits(run.head, y));
    auto p = run.checkpoint(1);
    check(p["checksum_mismatch_rows"] == 0 && p["logit_contract_violations"] == 0 && p["embedding_lookup"]["elements_differing_from_source"] == 0 && p["findings"].empty(), "healthy prediction is clean");
    check(p["exact_comparison"]["rows_compared"] == 2 && p["exact_comparison"]["rows_identical"].size() == 2 && p["row_dumps"][0]["reasons"] == Json::array({"argmax"}) && p["row_dumps"][1]["reasons"] == Json::array({"runner_up"}), "argmax and runner-up are copied and proven identical");
    for (const auto& r : p["head_recompute"]) check(r["bitwise_equal_gpu_row"] == true && r["bitwise_equal_source_row"] == true && r["within_bound"] == true, "head recompute is bitwise");
    auto partial = run.audit->finish();
    check(partial["first_invalid_stage"] == "audit_incomplete" && partial["audit_complete"] == false && partial["predictions_audited"] == 1 && partial["predictions_requested"] == 2, "cap2 run with one audited prediction is incomplete, not clean");
    uint32_t checksumDispatches = 0; uint64_t covered = 0;
    for (const auto& d : fake::log) if (d.shader == checksumShader) { ++checksumDispatches; covered += d.p[1]; check(d.p[1] <= 1024 && d.p[0] == 1280 && d.x * 64 >= d.p[1], "checksum dispatch rows <= 1024, K words 1280, full coverage"); }
    check(checksumDispatches == 2 * (2 + 2 + 1) && covered == 2 * 2500, "1100-row shards split into 1024+76; each checkpoint covers every row once");
    check(fake::largestReadback == 2560 * 4, "largest readback is one 2560-float row (checksums 10,000 B, logits 10,000 B, 2 row dumps 10,240 B)");
    run.predict(y, gpuLogits(run.head, y)); const uint64_t readbacks = fake::readbackBytes; run.predict(y, gpuLogits(run.head, y));
    check(fake::readbackBytes == readbacks, "observations past the requested predictions read nothing back");
    auto done = run.audit->finish(RunEnd::completed);
    check(done["checkpoints"].size() == 3 && done["predictions_audited"] == 2 && done["audit_complete"] == true && done["first_invalid_stage"] == "no_invalid_value_in_audited_head_path", "complete cap2 control is clean");
}
void storageCorruption() {
    for (bool atUpload : {true, false}) {
        Run run(smallShards, 2560, 41); const uint32_t bad = 79; // final row of shard 1, as 52427 is.
        Lcg g(9); auto y = hiddenRow(g, 2560);
        if (atUpload) corruptAligned(stored(run.head, bad), y);
        auto c = run.audit->afterImport();
        if (!atUpload) { check(c["checksum_mismatch_rows"] == 0, "storage intact after import"); corruptAligned(stored(run.head, bad), y); }
        else check(c["checksum_mismatch_rows"] == 1 && c["first_mismatch_rows"] == Json::array({bad}) && c["row_dumps"][0]["differing_words"] == 1280 && c["row_dumps"][0]["reasons"] == Json::array({"checksum_mismatch"}), "upload corruption localized to the shard-final row");
        run.predict(y, gpuLogits(run.head, y)); auto p = run.checkpoint(1);
        check(p["argmax"] == bad && p["argmax_logit"].get<double>() > 1e29 && p["checksum_mismatch_rows"] == 1, "prediction reproduces the huge shard-final argmax");
        check(p["logit_contract_violations"] == 1 && p["first_logit_contract_violations"][0]["row"] == bad && p["first_logit_contract_violations"][0]["exact_match"] == false, "only the corrupted row violates its authenticated Cauchy-Schwarz contract");
        check(p["row_dumps"][0]["reasons"] == Json::array({"argmax", "checksum_mismatch", "logit_contract_violation"}), "one copy serves every reason a row was selected");
        check(p["embedding_lookup"]["fp64_sum_of_squares_exceeds_fp32_max"] == true && p["embedding_lookup"]["elements_differing_from_source"] == 2560, "embedding of that token is the huge stored row");
        const auto& dump = p["row_dumps"][0]; const auto& bytes = run.files.at(dump["file"].get<std::string>());
        check(dump["row"] == bad && bytes.size() == 5120 && std::memcmp(bytes.data(), stored(run.head, bad), 5120) == 0 && dump["gpu_sha256"] == diagnostics::sha256(bytes.data(), bytes.size()), "raw GPU row dump is exact and hashed");
        check(run.stage() == (atUpload ? "tied_head_storage_differs_after_import" : "tied_head_storage_changed_after_import@prediction0"), "storage hypothesis discriminated by checkpoint");
    }
    Run many(smallShards, 2560, 43); for (uint32_t r = 0; r < 40; ++r) stored(many.head, r * 2)[r % 1280] ^= 0x00010000u;
    auto c = many.audit->afterImport();
    check(c["checksum_mismatch_rows"] == 40 && c["first_mismatch_rows"].size() == 40 && c["row_dumps"].size() == 32 && many.files.size() == 32 && c["exact_comparison"]["checksum_mismatch_rows_not_compared"] == 8, "more than 32 bad rows: all counted, 32 copied, 8 screened only");
    check(many.stage() == "tied_head_storage_differs_after_import", "copied differences classify the import");
}
// Defect 1 (closed counterexample): multi-word changes that keep the 32-bit checksum.
void checksumCollisions() {
    { Lcg g(3); for (int t = 0; t < 256; ++t) { auto row = trainedRow(g, 2560); const auto before = rowChecksum(row.data(), 1280); auto changed = row;
        const uint32_t i = g.next() % 1280, j = (i + 1 + g.next() % 1279) % 1280; collide(changed.data(), i, j, g.next() | 1u);
        if (t % 2) { const uint32_t k = (j + 1 + g.next() % 1279) % 1280; if (k != i) collide(changed.data(), k, i, g.next()); }
        if (rowChecksum(changed.data(), 1280) != before || changed == row) throw std::runtime_error("FAILED: collision construction"); }
      ++passed; }
    { // The supplied counterexample: after a healthy import, words 0 and 1 of the argmax change by +3 and -1.
        Run run(smallShards, 2560, 103); run.audit->afterImport(); Lcg random(109); auto hidden = hiddenRow(random, 2560);
        auto original = gpuLogits(run.head, hidden); const uint32_t selected = rank(original).best; auto* row = stored(run.head, selected);
        const uint32_t before = rowChecksum(row, 1280); row[0] += 3u; row[1] -= 1u;
        check(selected == 88 && rowChecksum(row, 1280) == before, "counterexample collision keeps the checksum of row 88");
        run.predict(hidden, original); auto p = run.checkpoint(1);
        check(p["checksum_mismatch_rows"] == 0 && p["row_dumps"][0]["differing_words"] == 2 && p["row_dumps"][0]["exact_match"] == false && p["row_dumps"][0]["checksum_matches_source"] == true, "screen matches, exact copy differs in two words");
        check(p["exact_comparison"]["differing_rows_with_matching_checksum"] == Json::array({88}) && p["exact_comparison"]["rows_differing"] == Json::array({88}), "collision row reported as exact difference");
        auto f = run.audit->finish();
        check(f["first_invalid_stage"] == "tied_head_row_bytes_differ_with_matching_checksum@prediction0", "exact difference outranks the embedding symptom");
        check(contains(f["findings"], "embedding_lookup_differs_from_authenticated_row@prediction0") && f["findings"][0] == "tied_head_row_bytes_differ_with_matching_checksum@prediction0", "embedding symptom kept as a later finding");
    }
    { // Three-word collision before import: import screens clean but copies nothing, so no change after import is claimed.
        Run run(smallShards, 2560, 105); Lcg g(11); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const uint32_t best = rank(l).best;
        collide(stored(run.head, best), 7, 900, 0x00010001u); collide(stored(run.head, best), 1279, 7, 0x80000000u);
        auto c = run.audit->afterImport();
        check(c["checksum_mismatch_rows"] == 0 && c["exact_comparison"]["rows_compared"] == 0, "import screen cannot see the collision and compares no row");
        run.predict(y, l);
        check(run.stage() == "tied_head_row_bytes_differ_with_matching_checksum@prediction0", "collision present at import is not called a change after import");
    }
    { // Runner-up collision after import, then a collision on a contract-violating, unselected row.
        Run run(smallShards, 2560, 107); run.audit->afterImport(); Lcg g(12); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const auto r = rank(l);
        collide(stored(run.head, r.runnerUp), 100, 200, 5u); run.predict(y, l);
        check(run.stage() == "tied_head_row_bytes_differ_with_matching_checksum@prediction0" && run.checkpoint(1)["row_dumps"][1]["exact_match"] == false, "runner-up collision detected by its exact copy");
    }
    for (bool collided : {false, true}) {
        Run run(smallShards, 2560, 109, 1); run.audit->afterImport(); Lcg g(13); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const auto r = rank(l);
        uint32_t victim = 0; while (victim == r.best || victim == r.runnerUp) ++victim;
        l[victim] = -1e30f; if (collided) collide(stored(run.head, victim), 3, 4, 9u);
        run.predict(y, l); auto p = run.checkpoint(1);
        check(p["row_dumps"].size() == 3 && p["row_dumps"][2]["row"] == victim && p["row_dumps"][2]["reasons"] == Json::array({"logit_contract_violation"}) && p["first_logit_contract_violations"][0]["exact_match"] == !collided, "contract-violating row is copied and compared");
        check(run.stage() == (collided ? "tied_head_row_bytes_differ_with_matching_checksum@prediction0" : "head_projection_disagrees_with_authenticated_rows@prediction0"), "head fault is named only over exactly identical violating rows");
    }
    { // Second prediction: the collision appears after a clean first prediction.
        Run run(smallShards, 2560, 111); run.audit->afterImport(); Lcg g(14); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y);
        run.predict(y, l); collide(stored(run.head, rank(l).best), 9, 10, 77u); run.predict(y, l);
        check(run.stage() == "tied_head_row_bytes_differ_with_matching_checksum@prediction1" && run.audit->finish()["audit_complete"] == true, "collision at prediction1 after a clean prediction0");
    }
    { // Unselected collision: invisible by design; the report claims identity only for compared rows.
        Run run(smallShards, 2560, 113, 1); run.audit->afterImport(); Lcg g(15); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const auto r = rank(l);
        uint32_t hidden = 0; while (hidden == r.best || hidden == r.runnerUp) ++hidden;
        collide(stored(run.head, hidden), 0, 1, 3u); run.predict(y, l); auto f = run.audit->finish(RunEnd::completed); auto p = f["checkpoints"][1];
        check(f["first_invalid_stage"] == "no_invalid_value_in_audited_head_path" && !contains(p["exact_comparison"]["rows_identical"], hidden) && p["exact_comparison"]["rows_compared"] == 2, "uncopied collided row stays outside the identity claim");
        check(f["evidence"].get<std::string>().find("32-bit checksum screen") != std::string::npos && f["hypotheses"]["no_invalid_value_in_audited_head_path"].get<std::string>().find("only for copied rows") != std::string::npos, "report states the screen/identity limit");
    }
    for (bool atImport : {true, false}) { // A wrong checksum read with identical copied bytes is a read disagreement, not storage.
        Run run(smallShards, 2560, 115, 1); Lcg g(16); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const uint32_t best = rank(l).best;
        if (atImport) fake::checksumFaultRow = 50;
        run.audit->afterImport();
        if (!atImport) fake::checksumFaultRow = best;
        run.predict(y, l); auto p = run.checkpoint(atImport ? 0 : 1);
        check(p["exact_comparison"]["identical_rows_with_mismatching_checksum"] == Json::array({atImport ? 50u : best}), "disagreeing reads listed");
        check(run.stage() == (atImport ? "checksum_screen_disagrees_with_exact_row_copy@after_import" : "checksum_screen_disagrees_with_exact_row_copy@prediction0"), "screen/copy disagreement is not a storage difference");
    }
}
// Defect 2: an all-zero final-norm row is reported with its cause unknown.
void zeroFinalNorm() {
    Lcg g(17); std::vector<float> zeros(2560, 0.0f), ones(2560, 1.0f); auto legitimate = rmsNormEmulated(zeros, ones, 1e-6f);
    { Run run(smallShards, 2560, 127, 1); run.audit->afterImport(); run.predict(legitimate, gpuLogits(run.head, legitimate)); auto f = run.audit->finish(RunEnd::completed);
      check(f["first_invalid_stage"] == "final_norm_hidden_all_zero_cause_unknown@prediction0" && f["checkpoints"][1]["final_norm_hidden"]["pre_norm_row_recorded"] == false, "legitimate zero row reported, cause unknown");
      bool claims = false; for (const auto& s : f["findings"]) claims = claims || s.get<std::string>().find("overflow") != std::string::npos;
      for (auto it = f["hypotheses"].begin(); it != f["hypotheses"].end(); ++it) claims = claims || it.key().find("overflow") != std::string::npos;
      check(!claims && f["hypotheses"]["final_norm_hidden_all_zero_cause_unknown"].get<std::string>().find("zero pre-norm row gives this legitimately") != std::string::npos, "no stage asserts overflow; prose names both causes"); }
    { Run run(smallShards, 2560, 129, 1); run.audit->afterImport(); std::vector<float> huge(2560); for (uint32_t k = 0; k < 2560; ++k) huge[k] = std::ldexp(k % 2 ? -1.0f : 1.0f, 90);
      auto overflowed = rmsNormEmulated(huge, ones, 1e-6f); run.predict(overflowed, gpuLogits(run.head, overflowed));
      check(run.stage() == "final_norm_hidden_all_zero_cause_unknown@prediction0", "overflow-produced zero row gets the same cause-unknown stage"); }
    { Run run(smallShards, 2560, 131, 1); run.audit->afterImport(); auto l = gpuLogits(run.head, zeros); l[5] = 7.0f; run.predict(zeros, l); auto f = run.audit->finish();
      check(f["first_invalid_stage"] == "head_projection_disagrees_with_authenticated_rows@prediction0" && contains(f["findings"], "final_norm_hidden_all_zero_cause_unknown@prediction0"), "a provably wrong logit outranks the zero row within its checkpoint"); }
    { Run run(smallShards, 2560, 133); run.audit->afterImport(); auto y = hiddenRow(g, 2560); run.predict(y, gpuLogits(run.head, y)); run.predict(zeros, gpuLogits(run.head, zeros));
      check(run.stage() == "final_norm_hidden_all_zero_cause_unknown@prediction1", "second prediction is audited in order"); }
    { Run run(smallShards, 2560, 135); run.audit->afterImport(); auto l = gpuLogits(run.head, zeros); stored(run.head, 0)[3] ^= 1u; run.predict(zeros, l);
      check(run.stage() == "tied_head_storage_changed_after_import@prediction0", "storage change outranks the zero row"); }
}
// Defect 3: requested, started and audited predictions; failure, cancellation and terminal boundaries.
void completion() {
    for (uint32_t requested : {1u, 2u}) {
        Run run(smallShards, 2560, 141, requested); auto none = run.audit->finish();
        check(none["first_invalid_stage"] == "audit_incomplete" && none["incomplete_reason"] == "import audit not completed", "no import: incomplete");
        run.audit->afterImport(); auto f = run.audit->finish(); auto again = run.audit->finish();
        check(f["first_invalid_stage"] == "audit_incomplete" && f["predictions_audited"] == 0 && f["predictions_requested"] == requested && f["audit_complete"] == false && f == again, "import then finish with zero predictions is incomplete");
        auto ended = run.audit->finish(RunEnd::completed);
        check(ended["first_invalid_stage"] == "audit_incomplete" && ended["incomplete_reason"] == "0 of " + std::to_string(requested) + " requested predictions audited", "a completed run end never completes the audit");
    }
    { Run run(smallShards, 2560, 143, 1); run.audit->afterImport(); Lcg g(1); auto y = hiddenRow(g, 2560); run.predict(y, gpuLogits(run.head, y));
      auto f = run.audit->finish(RunEnd::completed); check(f["audit_complete"] == true && f["first_invalid_stage"] == "no_invalid_value_in_audited_head_path" && f["run_end"]["state"] == "completed", "complete cap1 control is clean"); }
    { // Early stop (as at a caller EOS) after one of two requested predictions.
      Run run(smallShards, 2560, 145, 2); run.audit->afterImport(); Lcg g(2); auto y = hiddenRow(g, 2560); run.predict(y, gpuLogits(run.head, y));
      check(run.audit->finish(RunEnd::completed)["first_invalid_stage"] == "audit_incomplete", "run ending after one of two requested predictions is incomplete"); }
    { // Finding at prediction0, then the run fails: the finding is kept, completion is not implied.
      Run run(smallShards, 2560, 147, 2); run.audit->afterImport(); Lcg g(3); auto y = hiddenRow(g, 2560); corruptAligned(stored(run.head, 79), y); run.predict(y, gpuLogits(run.head, y));
      auto f = finishAfterFailure(*run.audit, "Request canceled at safe boundary decode_step");
      check(f["first_invalid_stage"] == "tied_head_storage_changed_after_import@prediction0" && f["audit_complete"] == false && f["run_end"]["state"] == "failed" && f["run_end"]["error"] == "Request canceled at safe boundary decode_step", "observed first finding preserved without completion");
      check(run.audit->finish() == f && finishAfterFailure(*run.audit, "Request canceled at safe boundary decode_step") == f, "repeated finish returns the same report");
      rejects([&] { run.audit->finish(RunEnd::completed); }, "a conflicting run end after a terminal finish");
      check(finishAfterFailure(*run.audit, "other")["finish_error"].is_string(), "the failure-path finish reports instead of throwing");
      rejects([&] { run.predict(y, gpuLogits(run.head, y)); }, "observation after a terminal finish");
      rejects([&] { run.audit->finish(RunEnd::completed, "error"); }, "a completed run end with an error"); }
    { // Cancellation between predictions with nothing found yet.
      Run run(smallShards, 2560, 149, 2); run.audit->afterImport(); Lcg g(4); auto y = hiddenRow(g, 2560); run.predict(y, gpuLogits(run.head, y));
      auto f = finishAfterFailure(*run.audit, "Request canceled at safe boundary decode_step");
      check(f["first_invalid_stage"] == "audit_incomplete" && f["predictions_audited"] == 1 && f["findings"].empty(), "canceled run is incomplete, never a non-reproduction"); }
    { // Readback failure inside the storage screen of prediction0.
      Run run(smallShards, 2560, 151, 1); run.audit->afterImport(); Lcg g(5); auto y = hiddenRow(g, 2560); fake::readbacksUntilFailure = 2;
      rejects([&] { run.predict(y, gpuLogits(run.head, y)); }, "a failed checksum readback");
      auto f = finishAfterFailure(*run.audit, "Injected fake readback failure"); auto p = f["checkpoints"][1];
      check(f["first_invalid_stage"] == "audit_incomplete" && f["predictions_started"] == 1 && f["predictions_audited"] == 0 && f["audit_errors"][0]["at"] == "prediction0", "failed prediction is incomplete and attributed");
      check(p["complete"] == false && p["error"] == "Injected fake readback failure" && p["final_norm_hidden"]["finite"] == true && !p.contains("checksum_mismatch_rows"), "partial checkpoint keeps only what was observed"); }
    { // Failure at the embedding readback after a provably wrong selected logit was observed.
      Run run(smallShards, 2560, 153, 1); run.audit->afterImport(); Lcg g(6); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const uint32_t best = rank(l).best;
      l[best] = bf16Rne(l[best] + 4.0f); fake::readbacksUntilFailure = 4; rejects([&] { run.predict(y, l); }, "a failed embedding readback");
      auto f = finishAfterFailure(*run.audit, "Injected fake readback failure");
      check(f["first_invalid_stage"] == "head_projection_disagrees_with_authenticated_rows@prediction0" && f["audit_complete"] == false && f["checkpoints"][1]["findings"] == Json::array({"head_projection_disagrees_with_authenticated_rows@prediction0"}), "finding before the failure is kept, completion is not"); }
    { // Failure while the final-norm row is read: no prediction starts.
      Run run(smallShards, 2560, 155, 1); run.audit->afterImport(); Lcg g(7); auto y = hiddenRow(g, 2560); fake::readbacksUntilFailure = 0;
      rejects([&] { run.predict(y, gpuLogits(run.head, y)); }, "a failed final-norm readback");
      auto f = run.audit->finish(); check(f["first_invalid_stage"] == "audit_incomplete" && f["predictions_started"] == 0 && f["audit_errors"][0]["at"] == "prediction0", "final-norm failure leaves the requested prediction unaudited");
      rejects([&] { run.predict(y, gpuLogits(run.head, y)); }, "observation after an audit failure"); }
    { // Import failure: source read and checksum readback.
      Run run(smallShards, 2560, 157, 1); run.failSource = true; rejects([&] { run.audit->afterImport(); }, "a failed source read");
      auto f = finishAfterFailure(*run.audit, "Injected source read failure");
      check(f["first_invalid_stage"] == "audit_incomplete" && f["import_audited"] == false && f["checkpoints"][0]["complete"] == false, "failed import is incomplete with its partial checkpoint");
      Run late(smallShards, 2560, 159, 1); stored(late.head, 3)[0] ^= 1u; fake::readbacksUntilFailure = 1; rejects([&] { late.audit->afterImport(); }, "a failed import row-copy readback");
      check(late.stage() == "audit_incomplete" && late.audit->finish()["checkpoints"][0]["checksum_mismatch_rows"] == 1, "unconfirmed screen mismatch alone is not a storage finding"); }
    { Json pending = notStarted(2); check(pending["first_invalid_stage"] == "audit_incomplete" && pending["audit_complete"] == false && pending["predictions_requested"] == 2, "CLI placeholder before the auditor exists is incomplete"); }
}
void observerOrdering() {
    TextPositions pos{{0}, {0}, {0}};
    { Run run(smallShards, 2560, 161, 1); Buffer logits = run.device.floats(97);
      rejects([&] { run.audit->observe({TextStage::Logits, UINT32_MAX, logits, 1, 97, 0, 0, 1, 0, 1, 1, pos, nullptr}); }, "an observation before the import audit");
      rejects([&] { run.audit->afterImport(); }, "an import audit after a refused observation");
      check(run.stage() == "audit_incomplete" && run.audit->finish()["audit_errors"].size() == 1, "refused early observation recorded"); }
    { Run run(smallShards, 2560, 163, 1); run.audit->afterImport(); Buffer logits = run.device.floats(97);
      rejects([&] { run.audit->observe({TextStage::Logits, UINT32_MAX, logits, 1, 97, 0, 0, 1, 0, 1, 1, pos, nullptr}); }, "logits without a final-norm row");
      check(run.stage() == "audit_incomplete", "logits-first ordering leaves the audit incomplete"); }
    { // Prefill-like call of 3 tokens in chunks 2+1: only the last chunk's last row is the prediction's hidden row.
      Run run(smallShards, 2560, 165, 1); run.audit->afterImport(); Lcg g(8); auto y = hiddenRow(g, 2560); std::vector<float> first(2 * 2560, 0.0f);
      Buffer chunk0 = run.device.floats(2 * 2560, first.data()), chunk1 = run.device.floats(2560, y.data()), layer = run.device.floats(2560, first.data());
      auto l = gpuLogits(run.head, y); Buffer values = run.device.floats(97, l.data()); const uint64_t before = fake::readbackBytes;
      run.audit->observe({TextStage::LayerOutput, 31, layer, 1, 2560, 0, 0, 3, 2, 1, 3, pos, nullptr});
      run.audit->observe({TextStage::FinalNorm, UINT32_MAX, chunk0, 2, 2560, 0, 0, 3, 0, 2, 2, pos, nullptr});
      check(fake::readbackBytes == before, "layer outputs and non-final chunks are not read back");
      run.audit->observe({TextStage::FinalNorm, UINT32_MAX, chunk1, 1, 2560, 0, 0, 3, 2, 1, 3, pos, nullptr});
      run.audit->observe({TextStage::Logits, UINT32_MAX, values, 1, 97, 0, 0, 3, 2, 1, 3, pos, nullptr});
      auto f = run.audit->finish(); check(f["first_invalid_stage"] == "no_invalid_value_in_audited_head_path" && f["checkpoints"][1]["call"]["call_tokens"] == 3, "last chunk's last row audited"); }
    { Run run(smallShards, 2560, 167, 2); run.audit->afterImport(); Lcg g(9); auto y = hiddenRow(g, 2560); Buffer hidden = run.device.floats(2560, y.data()), values = run.device.floats(97);
      run.audit->observe({TextStage::FinalNorm, UINT32_MAX, hidden, 1, 2560, 0, 0, 1, 0, 1, 1, pos, nullptr});
      rejects([&] { run.audit->observe({TextStage::Logits, UINT32_MAX, values, 1, 97, 7, 0, 1, 0, 1, 8, pos, nullptr}); }, "logits of a different call than the kept final-norm row"); }
    { Run fresh(smallShards, 2560, 169); fresh.audit->afterImport(); Lcg g(10); auto y = hiddenRow(g, 2560); fresh.predict(y, gpuLogits(fresh.head, y));
      Buffer again = fresh.device.floats(97);
      rejects([&] { fresh.audit->observe({TextStage::Logits, UINT32_MAX, again, 1, 97, 0, 0, 1, 0, 1, 1, pos, nullptr}); }, "a second prediction reusing the previous call's final-norm row");
      check(fresh.stage() == "audit_incomplete" && fresh.audit->finish()["predictions_audited"] == 1, "refused reuse leaves cap2 incomplete"); }
}
void readPathAndUpstream() {
    { Run run(smallShards, 2560, 51); run.audit->afterImport(); Lcg g(3); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); l[79] = 9.953037915854457e29f;
      run.predict(y, l); check(run.stage() == "head_projection_disagrees_with_authenticated_rows@prediction0", "same symptom with intact storage is a head projection fault"); }
    { Run run(smallShards, 2560, 53); run.audit->afterImport(); Lcg g(4); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y); const uint32_t best = rank(l).best;
      l[best] = bf16Rne(l[best] + 4.0f); run.predict(y, l); auto p = run.checkpoint(1);
      check(p["head_recompute"][0]["within_bound"] == false && p["logit_contract_violations"] == 0, "a plausible-sized wrong selected logit is caught by recompute, not the contract");
      check(run.stage() == "head_projection_disagrees_with_authenticated_rows@prediction0", "small selected-logit fault classified"); }
    { Run run(smallShards, 2560, 55); run.audit->afterImport(); Lcg g(5); auto y = hiddenRow(g, 2560); auto l = gpuLogits(run.head, y);
      fake::embeddingFaultToken = rank(l).best; fake::embeddingFaultXor = 0x00010000u; run.predict(y, l); fake::embeddingFaultToken = UINT32_MAX;
      check(run.checkpoint(1)["embedding_lookup"]["elements_differing_from_source"] == 1, "one-element embedding read fault counted");
      check(run.stage() == "embedding_lookup_differs_from_authenticated_row@prediction0", "embedding read-path fault classified"); }
    { Run run(smallShards, 2560, 59); run.audit->afterImport(); Lcg g(6); auto y = hiddenRow(g, 2560, 400.0); run.predict(y, gpuLogits(run.head, y));
      check(run.stage() == "final_norm_hidden_outside_rmsnorm_contract@prediction0", "impossible final-norm row classified"); }
}
void refusals() {
    Device d(L"", "", ""); Head h = makeHead(d, smallShards, 2560, 71); auto none = [](uint64_t, void*, uint32_t) {}; auto sink = [](const std::string&, const void*, size_t) {};
    auto digest = [](const void* p, size_t n) { return diagnostics::sha256(p, n); };
    rejects([&] { Auditor a(d, h.weight, {}, {}, 2048, 2, none, sink, digest, {}); }, "norm width differing from head K");
    rejects([&] { Weight w = h.weight; w.shardFirstRows[1] += 1; Auditor a(d, w, {}, {}, 2560, 2, none, sink, digest, {}); }, "shard gap");
    rejects([&] { Weight w = h.weight; w.rows += 1; Auditor a(d, w, {}, {}, 2560, 2, none, sink, digest, {}); }, "incomplete coverage");
    rejects([&] { Weight w = h.weight; w.cols = 9218; Auditor a(d, w, {}, {}, 9218, 2, none, sink, digest, {}); }, "K above 9216");
    rejects([&] { Auditor a(d, h.weight, {}, {}, 2560, 2, {}, sink, digest, {}); }, "missing source");
    rejects([&] { Auditor a(d, h.weight, {}, {}, 2560, 0, none, sink, digest, {}); }, "zero requested predictions");
    rejects([&] { Auditor a(d, h.weight, {}, {}, 2560, 3, none, sink, digest, {}); }, "three requested predictions");
    Run run(smallShards, 2560, 73); run.audit->afterImport(); rejects([&] { run.audit->afterImport(); }, "a second import audit");
    Facts f; f.requestedPredictions = 2; f.completedPredictions = 2; f.importAudited = true; f.import.screened = f.import.compared = true; f.predictions.resize(2);
    for (auto& p : f.predictions) { p.storage.screened = p.storage.compared = true; p.hiddenObserved = p.headObserved = p.embeddingObserved = true; }
    check(firstInvalidStage(f) == "no_invalid_value_in_audited_head_path", "fully observed clean facts");
    f.predictions[1].storage.differingScreenMismatched = 3; f.predictions[0].embeddingMatchesSource = false;
    check(firstInvalidStage(f) == "embedding_lookup_differs_from_authenticated_row@prediction0", "earlier prediction wins over a later storage change");
    f.predictions[0].embeddingMatchesSource = true; f.predictions[0].headObserved = false;
    check(firstInvalidStage(f) == "audit_incomplete" && findings(f) == std::vector<std::string>{"tied_head_storage_changed_after_import@prediction1"}, "an unobserved earlier stage blocks a later first-stage claim but stays a finding");
    Facts screenOnly = f; screenOnly.predictions.clear(); screenOnly.import.screenMismatchRows = 1;
    check(firstInvalidStage(screenOnly) == "tied_head_storage_differs_after_import", "screen-only evidence (no row copied) still names the import");
}
}

int main() {
    try {
        pinnedShardArithmetic(); failureSignature(); contractsAreSound(); healthyAndBounded(); storageCorruption(); checksumCollisions(); zeroFinalNorm();
        completion(); observerOrdering(); readPathAndUpstream(); refusals();
        std::cout << Json{{"test", "directcompute_head_row_audit"}, {"checks_passed", passed}, {"gpu", false}, {"model_weights_loaded", false}}.dump() << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
