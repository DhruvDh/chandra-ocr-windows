// New ChandraNative code, MPL-2.0. CPU checks of the text softmax regression core (text_softmax_regression.h)
// on the test-only fake Device and a fake host: plan geometry and refusals, the generator, the independent
// oracle and its gates, sensitivity controls, the wave-model hazard, strict arguments, external Job and
// identity admission, deadline, cancellation, drain, tracked-buffer retirement and bounded receipts. No GPU,
// D3D11, fxc, model or network. Usage: text-softmax-regression-test SHADER_ROOT FRESH_SCRATCH [--quick]
#include "text_softmax_regression_fake_device.h"
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace tsr = chandra::text_softmax_regression;
namespace vc = chandra::vision_calibration;
using tsr::Json;
namespace fs = std::filesystem;

namespace {
uint64_t checks = 0, failures = 0;
#define EXPECT(condition) do { ++checks; if (!(condition)) { std::cerr << __FILE__ << ":" << __LINE__ << ": expected " #condition "\n"; ++failures; } } while (0)
template <class F> std::string thrown(F f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return "";
}
bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
std::wstring wide(const std::string& s) { return std::wstring(s.begin(), s.end()); }
std::vector<uint32_t> nearestOutput(const tsr::Geometry& g, const tsr::Oracle& o, int which = 0) {
    std::vector<uint32_t> words(g.bufferWords, tsr::sentinelBits);
    for (uint32_t group = 0; group < g.groups; ++group)
        for (uint32_t key = 0; key < g.count; ++key) {
            const size_t i = size_t(group) * g.count + key;
            words[i] = key < tsr::validKeys(g, group) ? uint32_t(which < 0 ? o.low[i] : which > 0 ? o.high[i] : o.nearest[i]) << 16 : 0u;
        }
    return words;
}

void sha256Vectors() {
    EXPECT(vc::sha256Bytes("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT(vc::sha256Bytes("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

void frozenShaders(const fs::path& shaders, const fs::path& scratch) {
    const Json verified = tsr::verifyFrozenShaders(shaders);
    EXPECT(verified.size() == 2 && verified.at(0).at("sha256") == tsr::frozenShaders[0].sha256 && verified.at(1).at("sha256") == tsr::frozenShaders[1].sha256);
    // The repaired shader is the original plus exactly the barrier and its comment.
    std::ifstream a(shaders / "runtime/text_attention_softmax.hlsl"), b(shaders / "control/text_attention_softmax_2c44182.hlsl");
    std::string repaired((std::istreambuf_iterator<char>(a)), {}), original((std::istreambuf_iterator<char>(b)), {});
    EXPECT(contains(repaired, "m=tmp[0];GroupMemoryBarrierWithGroupSync();float total=0;") && contains(original, "m=tmp[0];float total=0;"));
    const std::string comment = "// The barrier after m=tmp[0] makes every lane read the broadcast maximum before lane 0 overwrites tmp[0] with its exp partial.\n";
    std::string reconstructed = repaired.substr(comment.size());
    const size_t at = reconstructed.find("GroupMemoryBarrierWithGroupSync();float total=0;");
    reconstructed.erase(at, std::string("GroupMemoryBarrierWithGroupSync();").size());
    EXPECT(repaired.rfind(comment, 0) == 0 && reconstructed == original);
    for (const char* mutated : {"runtime/text_attention_softmax.hlsl", "control/text_attention_softmax_2c44182.hlsl"}) {
        const fs::path root = scratch / (std::string("frozen-") + (mutated[0] == 'r' ? "repaired" : "control"));
        fs::create_directories(root / "runtime"); fs::create_directories(root / "control");
        for (const auto& f : tsr::frozenShaders) fs::copy_file(shaders / f.relative, root / f.relative);
        std::fstream file(root / mutated, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(100); file.put('~'); file.close();
        EXPECT(contains(thrown([&] { tsr::verifyFrozenShaders(root); }), "Shader bytes differ from the frozen source"));
        fs::remove(root / mutated);
        EXPECT(contains(thrown([&] { tsr::verifyFrozenShaders(root); }), "Frozen shader absent or resized"));
    }
}

void plan() {
    const auto& list = tsr::cases();
    std::set<std::string> names; std::set<uint32_t> counts; uint64_t plane = 0;
    for (const auto& c : list) {
        names.insert(c.name); const auto g = tsr::geometry(c); counts.insert(g.count); plane += g.planeWords;
        EXPECT(g.groups == 16 * c.rows && g.count == c.base + c.rows && g.planeWords == uint64_t(g.groups) * g.count);
        EXPECT(g.bufferWords == g.planeWords + 128 && g.deviceBytes <= tsr::deviceAllocationCap && g.hostBytes <= tsr::hostVectorCap);
        EXPECT(g.keysPerLane <= 128 && g.groups <= 1024 && g.count <= 16384);
        const auto p = tsr::params(g);
        EXPECT(p.rows == c.rows && p.width == 16 && p.offset == 0 && p.stride == 0 && p.base == c.base && p.count == c.base + c.rows && p.mode == 0 && p.pad == 0);
    }
    EXPECT(names.size() == list.size() && list.size() == 17);
    for (uint32_t want : {1u, 5u, 64u, 127u, 128u, 129u, 801u, 3584u, 3617u, 3618u, 3622u, 16000u, 16384u}) EXPECT(counts.count(want) == 1);
    const auto tail = tsr::geometry(list.at(10));
    EXPECT(std::string(list.at(10).name) == "recorded_tail_c3617" && tail.base == 3584 && tail.rows == 33 && tail.groups == 528 &&
           tail.minimumValid == 3585 && tail.maximumValid == 3617 && tail.keysPerLane == 29);
    EXPECT(tsr::geometry(list.at(9)).deviceBytes == 29361152 && plane == 7489984);
    const Json p = tsr::planJson();
    EXPECT(p.at("dispatches") == 102 && p.at("maximum_device_plus_staging_bytes") == 29361152 && p.at("original_control_gating") == false);
    EXPECT(tsr::planSha256() == tsr::planSha256() && tsr::planSha256().size() == 64);
    // Five keys stay in one wave of any width >= 8; 129 keys occupy every wave.
    EXPECT(tsr::wavesWithKeys(5, 8) == 1 && tsr::wavesWithKeys(64, 32) == 2 && tsr::wavesWithKeys(129, 32) == 4 && tsr::wavesWithKeys(16384, 8) == 16);
    // Refusals before allocation, including overflow helpers.
    EXPECT(contains(thrown([] { tsr::geometry({"x", 0, 0, ""}); }), "rows must be 1..64"));
    EXPECT(contains(thrown([] { tsr::geometry({"x", 0, 65, ""}); }), "rows must be 1..64"));
    EXPECT(contains(thrown([] { tsr::geometry({"x", 16384, 1, ""}); }), "16,384-token context"));
    EXPECT(contains(thrown([] { tsr::geometry({"x", 0xffffffffu, 2, ""}); }), "16,384-token context"));
    EXPECT(contains(thrown([] { tsr::geometry({"x", 16320, 64, ""}); }), "per-dispatch word cap"));
    EXPECT(contains(thrown([] { tsr::checkedAdd(UINT64_MAX, 1, "t"); }), "Integer overflow"));
    EXPECT(contains(thrown([] { tsr::checkedMul(1ull << 33, 1ull << 31, "t"); }), "Integer overflow"));
    EXPECT(contains(thrown([] { tsr::checked32(1ull << 32, "t"); }), "exceeds 32 bits"));
    EXPECT(tsr::checkedMul(0, UINT64_MAX, "t") == 0 && tsr::checkedAdd(UINT64_MAX - 1, 1, "t") == UINT64_MAX);
}

void productionSource(const fs::path& shaders) {
    std::ifstream f(shaders.parent_path() / "runtime/text_model.cpp");
    const std::string text((std::istreambuf_iterator<char>(f)), {});
    EXPECT(contains(text, "p={n,16,0,0,c.length,c.length+n,0,0};"));
    EXPECT(contains(text, "device.dispatch(\"text_attention_softmax\",{},{&scores},&p,sizeof(p),n*16);"));
    EXPECT(contains(text, "Buffer scores=device.floats(n*16*(c.length+n))"));
}

void generator() {
    std::map<std::string, uint64_t> kinds;
    for (uint32_t i = 0; i < tsr::cases().size(); ++i) {
        const auto g = tsr::geometry(tsr::cases()[i]); const auto words = tsr::uploadWords(i, g);
        std::set<int> rowKinds;
        for (uint32_t group = 0; group < g.groups; ++group) {
            const uint32_t valid = tsr::validKeys(g, group); rowKinds.insert(int(tsr::rowKind(i, group)));
            EXPECT(valid == g.base + group / 16 + 1 && valid <= g.count);
            uint32_t spikes = 0; const uint32_t first = words[size_t(group) * g.count];
            for (uint32_t key = 0; key < g.count; key += (g.count > 4096 ? 7 : 1)) {
                const uint32_t w = words[size_t(group) * g.count + key];
                if (key >= valid) { EXPECT(w == tsr::maskedScoreBits); continue; }
                const float v = vc::fromBits(w);
                EXPECT((w & 0xffffu) == 0 && v * 16.0f == std::nearbyint(v * 16.0f) && std::fabs(v) <= 255.0f / 16.0f);
                if (tsr::rowKind(i, group) == tsr::RowKind::uniform) EXPECT(w == first);
                if (tsr::rowKind(i, group) == tsr::RowKind::spike && v >= 8.0f) ++spikes;
            }
            if (tsr::rowKind(i, group) == tsr::RowKind::spike && g.count <= 4096) EXPECT(spikes == 1);
            ++kinds[tsr::kindName(tsr::rowKind(i, group))];
        }
        EXPECT(rowKinds.size() == (g.groups >= 3 ? 3u : g.groups));
        for (uint32_t k = g.planeWords; k < g.bufferWords; ++k) EXPECT(words[k] == tsr::sentinelBits);
        EXPECT(tsr::validKeys(g, g.groups - 1) == g.count && tsr::validKeys(g, 0) == g.base + 1);
    }
    EXPECT(kinds.size() == 3);
}

// Exact BF16 rounding of binary64 against a direct nearest/even comparison, and against the shader's bf().
void bf16Rounding() {
    EXPECT(tsr::bf16NearestBits(1.0 + 0x1p-8) == 0x3f800000u && tsr::bf16NearestBits(1.0 + 3 * 0x1p-8) == 0x3f820000u);
    EXPECT(tsr::bf16NearestBits(1.0 + 0x1p-8 + 0x1p-40) == 0x3f810000u && tsr::bf16NearestBits(1.9999999) == 0x40000000u);
    EXPECT(contains(thrown([] { tsr::bf16NearestBits(0x1p-127); }), "normal BF16 range"));
    EXPECT(tsr::bf16ModelBits(0x1p130) == 0x7f800000u && tsr::bf16ModelBits(0x1p-130) == 0 && tsr::bf16ModelBits(std::nan("")) == 0x7fc00000u);
    uint64_t state = 0x1234567u;
    for (int i = 0; i < 200000; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const double v = std::ldexp(1.0 + double(state >> 11) * 0x1p-53, int((state >> 3) % 60) - 50);
        const uint32_t got = tsr::bf16NearestBits(v);
        uint32_t below = vc::bits(float(v)) & 0xffff0000u; if (double(vc::fromBits(below)) > v) below -= 0x10000u;
        const uint32_t above = below + 0x10000u;
        const double d1 = v - double(vc::fromBits(below)), d2 = double(vc::fromBits(above)) - v;
        const uint32_t want = d1 < d2 ? below : d2 < d1 ? above : ((below >> 16) & 1u ? above : below);
        EXPECT(got == want);
        const float f = float(v);
        if (std::isnormal(f)) EXPECT(tsr::bf16NearestBits(double(f)) == tsr_fake::bf(f));
        if (failures > 20) return;
    }
}

// Control-model BF16 words at the edges, against hand-derived bit patterns and against the shader's integer
// carry rounding (tsr_fake::bf) of FP32 inputs, with D3D11 flushing FP32 subnormals to a signed zero.
void bf16ModelEdges() {
    const struct { double v; uint32_t want; } fixed[] = {
        {0x1p127, 0x7f000000u}, {-0x1p127, 0xff000000u},                          // Finite 2^127.
        {0x1p127 + 0x1p119, 0x7f000000u}, {-0x1p127 - 0x1p119, 0xff000000u},      // Tie to even (down).
        {0x1p127 + 3 * 0x1p119, 0x7f020000u}, {-0x1p127 - 3 * 0x1p119, 0xff020000u}, // Tie to even (up).
        {0x1p127 + 0x1p119 + 0x1p75, 0x7f010000u},                               // One binary64 ulp above a tie.
        {0x1p128 - 0x1p120, 0x7f7f0000u}, {-0x1p128 + 0x1p120, 0xff7f0000u},      // Largest finite BF16.
        {0x1p128 - 0x1p119 - 0x1p75, 0x7f7f0000u}, {-0x1p128 + 0x1p119 + 0x1p75, 0xff7f0000u},
        {0x1p128 - 0x1p119, 0x7f800000u}, {-0x1p128 + 0x1p119, 0xff800000u},      // Overflow tie goes to even: infinity.
        {0x1.fffffep127, 0x7f800000u}, {-0x1.fffffep127, 0xff800000u},            // Largest finite FP32.
        {0x1p130, 0x7f800000u}, {-0x1p1000, 0xff800000u},
        {std::numeric_limits<double>::max(), 0x7f800000u}, {-std::numeric_limits<double>::infinity(), 0xff800000u},
        {std::numeric_limits<double>::infinity(), 0x7f800000u},
        {0x1p127 - 0x1p118, 0x7f000000u}, {0x1p127 - 0x1p119, 0x7eff0000u},        // Rounding up into the top binade.
        {1.0 + 0x1p-8, 0x3f800000u}, {-1.0 - 3 * 0x1p-8, 0xbf820000u},
        {0x1p-126, 0x00800000u}, {-0x1p-126, 0x80800000u},                        // Smallest normal.
        {0x1.fffffffffffffp-127, 0u}, {-0x1.fffffffffffffp-127, 0x80000000u},     // Flushed, sign kept.
        {0x1p-130, 0u}, {-0x1p-149, 0x80000000u}, {0.0, 0u}, {-0.0, 0x80000000u},
        {std::nan(""), 0x7fc00000u}, {-std::nan(""), 0x7fc00000u}};
    for (const auto& c : fixed) {
        if (tsr::bf16ModelBits(c.v) != c.want)
            std::cerr << "bf16ModelBits(" << std::hexfloat << c.v << std::defaultfloat << ") = 0x" << std::hex
                      << tsr::bf16ModelBits(c.v) << ", want 0x" << c.want << std::dec << "\n";
        EXPECT(tsr::bf16ModelBits(c.v) == c.want);
    }
    const auto carry = [](uint32_t u) {
        const float f = vc::fromBits(u);
        return std::fpclassify(f) == FP_SUBNORMAL ? u & 0x80000000u : tsr_fake::bf(f);
    };
    uint64_t mismatches = 0;
    // Every BF16 high half, each with the low halves that decide rounding (exact, below/at/above a tie, top).
    for (uint32_t high = 0; high < 0x10000u; ++high)
        for (const uint32_t low : {0x0000u, 0x0001u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
            const uint32_t u = high << 16 | low;
            if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) continue; // NaN: canonical above.
            mismatches += tsr::bf16ModelBits(double(vc::fromBits(u))) != carry(u);
        }
    // Every FP32 value of both signs in the top binade [2^127, 2^128), where the old helper overflowed.
    for (uint32_t m = 0; m < 0x800000u; ++m)
        for (const uint32_t s : {0u, 0x80000000u}) {
            const uint32_t u = s | 0x7f000000u | m;
            mismatches += tsr::bf16ModelBits(double(vc::fromBits(u))) != carry(u);
        }
    EXPECT(mismatches == 0);
    // The oracle rounding still refuses everything outside the normal BF16 range.
    EXPECT(contains(thrown([] { tsr::bf16NearestBits(0x1p127); }), "normal BF16 range"));
    EXPECT(contains(thrown([] { tsr::bf16NearestBits(-1.0); }), "normal BF16 range"));
}

void oracle(bool quick) {
    std::vector<int> j; std::vector<double> e, p;
    for (uint32_t i = 0; i < tsr::cases().size(); ++i) {
        const auto g = tsr::geometry(tsr::cases()[i]);
        if (quick && g.planeWords > 300000) continue;
        const auto o = tsr::buildOracle(i, g);
        EXPECT(o.validWords + g.maskedWords == g.planeWords && o.wideWords == 0);
        EXPECT(o.maximumBound > 0 && o.maximumBound < 1e-4 && o.minimumProbability > 1e-13 && o.twoValueWords * 20 < o.validWords + 20);
        // Every admissible endpoint and the correctly rounded value pass; the gates are internally consistent.
        for (int which : {-1, 0, 1}) EXPECT(tsr::checkOutput(i, g, o, nearestOutput(g, o, which)).at("passed") == true);
        const Json controls = tsr::sensitivityControls(i, g, o, nearestOutput(g, o));
        EXPECT(controls.at("all_applicable_refused") == true);
        std::map<std::string, bool> applicable;
        for (const auto& c : controls.at("controls")) applicable[c.at("name")] = c.at("applicable");
        EXPECT(applicable.at("next_bf16_above_admissible_interval") && applicable.at("nan_word") && applicable.at("guard_word_overwritten"));
        EXPECT(applicable.at("masked_word_smallest_normal") == (g.maskedWords > 0));
        EXPECT(applicable.at("original_hazard_model_wave32_ascending") == (g.count > 32));
        EXPECT(applicable.at("original_hazard_model_wave16_ascending") == (g.count > 16));
        EXPECT(applicable.at("lane127_partial_dropped") == (g.count >= 128));
        // Independent cross-check of two rows in long double.
        for (uint32_t group : {0u, g.groups - 1}) {
            tsr::rowProbabilities(i, g, group, j, e, p);
            long double sum = 0; int maximum = *std::max_element(j.begin(), j.end());
            for (int v : j) sum += std::exp((long double)(v - maximum) / 16.0L);
            for (size_t k = 0; k < j.size(); k += 1 + j.size() / 64)
                EXPECT(std::fabs((long double)p[k] - std::exp((long double)(j[k] - maximum) / 16.0L) / sum) <= 1e-14L * (long double)p[k]);
        }
    }
    // The gate edge: one BF16 step outside either endpoint is refused, the endpoint itself admitted.
    const auto g = tsr::geometry(tsr::cases().at(6)); const auto o = tsr::buildOracle(6, g);
    auto words = nearestOutput(g, o); const size_t i = size_t(g.groups - 1) * g.count + 5;
    words[i] = uint32_t(o.high[i]) << 16; EXPECT(tsr::checkOutput(6, g, o, words).at("passed") == true);
    words[i] = (uint32_t(o.high[i]) << 16) + 0x10000u;
    const Json refused = tsr::checkOutput(6, g, o, words);
    EXPECT(refused.at("passed") == false && refused.at("outside_admissible_interval") == 1 && refused.at("first_violations").at(0).at("key") == 5);
}

// Full plan through run() on wave models. The control is never gating; it is visibly wrong exactly when a
// key-owning lane lies outside lane 0's wave and lane 0's wave runs first.
Json fakePlan(uint32_t width, tsr_fake::WaveOrder order) {
    tsr_fake::Config c; c.waveWidth = width; c.order = order; tsr_fake::reset(c);
    chandra::dc::Device device(L"", "03:00.0", "00000000:0000abcd");
    Json report = tsr::initialReport(); const tsr::Deadline deadline; uint64_t persists = 0;
    tsr::run(device, report, deadline, [&] { ++persists; });
    EXPECT(report.at("passed") == true && report.at("dispatches_submitted") == 102 && report.at("dispatches_completed_and_measured") == 102);
    EXPECT(device.trackedBufferBytes() == 0 && persists >= 2 * 102 && tsr_fake::counters().allocations == 102);
    std::vector<std::string> visible;
    for (const auto& control : report.at("original_control")) {
        const auto g = tsr::geometry(tsr::cases().at(control.at("index").get<size_t>()));
        const bool expected = order == tsr_fake::WaveOrder::ascending && width < 128 && g.count > width;
        EXPECT(control.at("visible_failure") == expected && control.at("rejected_by_oracle") == expected);
        EXPECT(control.at("differs_from_repaired") == expected && control.at("differs_between_repetitions") == false);
        if (expected) visible.push_back(control.at("name"));
    }
    for (const auto& item : report.at("cases")) {
        EXPECT(item.at("passed") == true && item.at("sensitivity").at("all_applicable_refused") == true);
        for (const auto& r : item.at("repetitions")) EXPECT(r.at("tracked_after_release") == 0 && r.at("check").at("passed") == true);
    }
    return {{"wave_width", width}, {"order", order == tsr_fake::WaveOrder::ascending ? "ascending" : "descending"},
            {"control_cases_visibly_wrong", visible.size()}, {"first_repaired_output_sha256", report.at("cases").at(10).at("repetitions").at(0).at("observed_sha256")}};
}

// ---- Entry with a fake host --------------------------------------------------------------------------
struct FakeHost {
    std::map<fs::path, std::string> files; fs::path open; bool closed = false; uint64_t rewrites = 0;
    tsr::JobLimits job{true, tsr::jobLimitKillOnJobClose | tsr::jobLimitProcessMemory, 768ull << 20, 0};
    bool cancel = false, driverMissing = false, compilerMissing = false, producerThrows = false, createFails = false;
    std::chrono::seconds offset{0}; tsr::Clock::time_point base = tsr::Clock::now();
    std::function<void(uint64_t)> rewriteHook;
    tsr::Host host() {
        tsr::Host h;
        h.createFresh = [this](const fs::path& p) { if (createFails || files.count(p) || fs::exists(p)) return false; files[p] = ""; open = p; return true; };
        h.rewrite = [this](const std::string& text) { ++rewrites; if (rewriteHook) rewriteHook(rewrites); files.at(open) = text; };
        h.close = [this] { closed = true; };
        h.queryJob = [this] { return job; };
        h.producer = [this]() -> Json { if (producerThrows) throw std::runtime_error("Executable path unavailable"); return {{"executable", {{"sha256", std::string(64, 'a')}}}, {"cxx", "fake"}}; };
        h.driver = [this](const std::string& luid) -> Json {
            if (driverMissing) return Json::object();
            return {{"luid", luid}, {"umd_driver_version", "32.0.101.6979"}, {"source", "fake"}};
        };
        h.shaderCompiler = [this]() -> Json { if (compilerMissing) return Json::object(); return {{"path", "fake"}, {"bytes", 1}, {"sha256", std::string(64, 'c')}}; };
        h.cancelled = [this] { return cancel; };
        h.now = [this] { return base + offset; };
        return h;
    }
    Json receipt() const { return files.empty() ? Json() : Json::parse(files.begin()->second); }
};
struct Outcome { int code = -1; std::string out, err; Json receipt; FakeHost host; };
Outcome invoke(const std::vector<std::string>& args, const std::function<void(FakeHost&)>& setup = {}, tsr_fake::Config config = {}) {
    Outcome o; tsr_fake::reset(std::move(config)); if (setup) setup(o.host);
    std::vector<std::wstring> argv; for (const auto& a : args) argv.push_back(wide(a));
    std::ostringstream out, err; tsr::Host h = o.host.host();
    o.code = tsr::entry(argv, h, out, err); o.out = out.str(); o.err = err.str();
    if (!o.host.files.empty() && !o.host.files.begin()->second.empty()) o.receipt = Json::parse(o.host.files.begin()->second);
    return o;
}
std::vector<std::string> execute(const fs::path& shaders, const fs::path& output) {
    return {"--execute", "--shader-root", shaders.string(), "--pci", "03:00.0", "--luid", "00000000:0000abcd", "--output", output.string()};
}

void arguments(const fs::path& shaders, const fs::path& scratch) {
    const std::string pci = "03:00.0", luid = "00000000:0000abcd", out = (scratch / "args.json").string(), root = shaders.string();
    for (const auto& args : {std::vector<std::string>{}, std::vector<std::string>{"--pci", pci},
                             std::vector<std::string>{"--shader-root", root, "--pci", pci, "--luid", luid, "--output", out}}) {
        const auto o = invoke(args);
        EXPECT(o.code == 0 && o.host.files.empty() && tsr_fake::counters().dispatches == 0 && o.err.empty());
        const Json j = Json::parse(o.out);
        EXPECT(j.at("state") == "INACTIVE" && j.at("device_created") == false && j.at("dispatches") == 102 && j.at("plan_sha256") == tsr::planSha256());
    }
    const std::vector<std::pair<std::vector<std::string>, std::string>> refusals = {
        {{"--model", "x"}, "Unknown calibration option"},
        {{"--pci", pci, "--pci", pci}, "Duplicate calibration option --pci"},
        {{"--output", out, "--output", (scratch / "other.json").string()}, "Duplicate calibration option --output"},
        {{"--execute", "--execute"}, "Duplicate execute flag"},
        {{"--pci", "--execute"}, "Option value absent or option-shaped after --pci"},
        {{"--pci"}, "Option value absent or option-shaped after --pci"},
        {{"--luid", ""}, "after --luid"},
        {{"--pci=" + pci}, "Unknown calibration option"},
        {{"--pci", pci, "extra"}, "Unknown calibration option"},
        {{"--pci", "3:00.0"}, "PCI pin must be lowercase"},
        {{"--luid", "00000000:0000ABCD"}, "LUID pin must be lowercase"},
        {{"--shader-root", "ChandraNative/shaders"}, "Absolute shader root and output paths required"},
        {{"--output", "relative.json"}, "Absolute shader root and output paths required"},
        {{"--execute", "--shader-root", root, "--pci", pci, "--output", out}, "Explicit shader root, PCI, LUID and fresh output required"},
        {{"--execute", "--shader-root", root, "--pci", pci, "--luid", luid, "--output", (shaders / "runtime" / "receipt.json").string()},
         "outside the authenticated shader root"},
        {{"--execute", "--shader-root", root, "--pci", pci, "--luid", luid, "--output", (scratch / "missing-dir" / "r.json").string()},
         "in an existing directory"},
        {{"--execute", "--shader-root", (scratch / "absent-root").string(), "--pci", pci, "--luid", luid, "--output", out}, "Absolute existing shader root"}};
    for (const auto& [args, want] : refusals) {
        const auto o = invoke(args);
        EXPECT(o.code == 2 && o.out.empty() && o.host.files.empty() && tsr_fake::counters().dispatches == 0 && contains(o.err, want));
        if (!contains(o.err, want)) std::cerr << "  refusal stderr: " << o.err << "\n";
    }
    { // A fresh path is required: an existing file is never touched.
        const fs::path existing = scratch / "existing.json"; { std::ofstream f(existing); f << "precious"; }
        const auto o = invoke(execute(shaders, existing));
        std::ifstream f(existing); std::string text((std::istreambuf_iterator<char>(f)), {});
        EXPECT(o.code == 2 && contains(o.err, "already exists") && text == "precious" && o.host.files.empty());
    }
    { // CREATE_NEW is authoritative even when the path appeared after the existence check.
      const auto o = invoke(execute(shaders, scratch / "raced.json"), [](FakeHost& h) { h.createFails = true; });
      EXPECT(o.code == 2 && contains(o.err, "CREATE_NEW") && o.host.files.empty() && tsr_fake::counters().dispatches == 0); }
    const std::vector<std::pair<tsr::JobLimits, std::string>> jobs = {
        {{false, 0, 0, 0}, "externally owned Job"}, {{true, tsr::jobLimitProcessMemory, 1 << 20, 0}, "kill on close"},
        {{true, tsr::jobLimitKillOnJobClose, 0, 0}, "bound memory"}, {{true, tsr::jobLimitKillOnJobClose | tsr::jobLimitJobMemory, 0, 2ull << 30}, "exceeds 1 GiB"}};
    for (const auto& [job, want] : jobs) {
        const auto o = invoke(execute(shaders, scratch / "job.json"), [&job = job](FakeHost& h) { h.job = job; });
        EXPECT(o.code == 2 && contains(o.err, want) && o.host.files.empty() && tsr_fake::counters().dispatches == 0);
    }
    { const auto o = invoke(execute(shaders, scratch / "early-cancel.json"), [](FakeHost& h) { h.cancel = true; });
      EXPECT(o.code == 3 && contains(o.err, "CANCELLED") && o.host.files.empty() && tsr_fake::counters().dispatches == 0); }
}

void lifecycle(const fs::path& shaders, const fs::path& scratch) {
    int serial = 0;
    auto path = [&] { return scratch / ("receipt-" + std::to_string(++serial) + ".json"); };
    auto stopped = [](const Outcome& o, int code, const char* state, uint64_t submitted, const std::string& error) {
        const Json& r = o.receipt;
        const bool ok = o.code == code && r.at("state") == state && r.at("passed") == false && r.at("dispatches_submitted") == submitted &&
                        contains(r.at("error").get<std::string>(), error) && o.host.closed && tsr_fake::counters().dispatches == submitted;
        if (!ok) std::cerr << "  stop mismatch: code " << o.code << " state " << r.value("state", "?") << " submitted " << r.value("dispatches_submitted", 0)
                           << " error " << r.value("error", "?").substr(0, 300) << "\n";
        return ok;
    };
    auto drained = [](const Outcome& o) {
        return o.receipt.at("retirement_required") == true && o.receipt.at("failure_drain").at("completed") == true &&
               o.receipt.at("failure_drain").at("tracked_after") == 0;
    };
    { // Complete pass.
        const auto o = invoke(execute(shaders, path()));
        const Json& r = o.receipt;
        EXPECT(o.code == 0 && r.at("state") == "TEXT_SOFTMAX_REGRESSION_PASS" && r.at("passed") == true && r.at("dispatches_submitted") == 102);
        EXPECT(r.at("tracked_after_final_drain") == 0 && r.at("shaders_after") == r.at("shaders_before") && r.at("retirement_required") == false);
        EXPECT(r.at("device_identity").at("pci_bdf") == "03:00.0" && r.at("external_job").at("created_by_this_process") == false);
        EXPECT(r.at("driver_identity").at("umd_driver_version") == "32.0.101.6979" && r.at("plan_sha256") == tsr::planSha256());
        EXPECT(r.at("original_control_summary").at("required_to_fail") == false && r.at("cases").size() == 17 && r.at("original_control").size() == 17);
        EXPECT(o.host.files.begin()->second.size() < tsr::receiptCap && o.host.closed && contains(o.out, "\"page_throughput_accepted\":false"));
        std::cout << "  pass receipt bytes " << o.host.files.begin()->second.size() << ", rewrites " << o.host.rewrites << "\n";
    }
    tsr_fake::Config c;
    c.fault = tsr_fake::Fault::repairedWithoutBarrier;
    { const auto o = invoke(execute(shaders, path()), {}, c); // First case with a key-owning lane outside wave 0 (W=32): case 2.
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 7, "Repaired output check failed in case first_prefill_tile_c64") && drained(o));
      EXPECT(o.receipt.at("cases").at(2).at("repetitions").at(0).at("check").at("outside_admissible_interval").get<uint64_t>() > 0); }
    c.fault = tsr_fake::Fault::dropLastLanePartial;
    { const auto o = invoke(execute(shaders, path()), {}, c); EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 13, "case decode_c128") && drained(o)); }
    c.fault = tsr_fake::Fault::guardStore;
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 1, "single_key_decode_c1") && o.receipt.at("cases").at(0).at("repetitions").at(0).at("check").at("guard_changed") == 1); }
    c.fault = tsr_fake::Fault::maskedNonzero;
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 4, "operator_fixture_like_c5") && o.receipt.at("cases").at(1).at("repetitions").at(0).at("check").at("masked_not_positive_zero") == 1); }
    c.fault = tsr_fake::Fault::nanWord;
    { const auto o = invoke(execute(shaders, path()), {}, c); EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 1, "single_key_decode_c1")); }
    c.fault = tsr_fake::Fault::alternateAdmissibleWord; c.faultDispatch = 8; // Case 2, second repetition: still admissible.
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 8, "differs between repetitions in case first_prefill_tile_c64"));
      EXPECT(o.receipt.at("cases").at(2).at("repetitions").at(1).at("check").at("passed") == true &&
             o.receipt.at("cases").at(2).at("repetitions").at(1).at("bit_identical_to_first_repetition") == false); }
    c = {};
    for (auto fault : {tsr_fake::ProfileFault::disjointThrows, tsr_fake::ProfileFault::disjointReported, tsr_fake::ProfileFault::missing,
                       tsr_fake::ProfileFault::wrongGroups, tsr_fake::ProfileFault::wrongShader, tsr_fake::ProfileFault::negative}) {
        c.profileFault = fault; c.profileFaultDispatch = 1;
        const auto o = invoke(execute(shaders, path()), {}, c);
        EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 1, "") && o.receipt.at("dispatches_completed_and_measured") == 0 && drained(o));
        EXPECT(o.receipt.at("cases").at(0).at("repetitions").at(0).at("submitted") == true && o.receipt.at("cases").at(0).at("repetitions").at(0).at("completed") == false);
    }
    c = {}; c.gpuMilliseconds = [](const std::string&, uint64_t id) { return id == 5 ? 100.0 : 0.25; };
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 5, "at or above the 100 ms gate") && o.receipt.at("dispatches_completed_and_measured") == 5);
      EXPECT(o.receipt.at("cases").at(1).at("repetitions").at(1).at("strictly_below_gate") == false); }
    c.gpuMilliseconds = [](const std::string&, uint64_t id) { return id == 52 ? 250.0 : 0.25; }; // First original-control dispatch.
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 52, "at or above the 100 ms gate") && o.receipt.at("phase") == "original_control" && drained(o)); }
    c = {}; c.profileFault = tsr_fake::ProfileFault::disjointThrows; c.profileFaultDispatch = 2; c.drainThrowsAfterFailure = true;
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 2, "disjoint") && o.receipt.at("failure_drain").at("completed") == false); }
    c = {}; c.profileFault = tsr_fake::ProfileFault::giantError; c.profileFaultDispatch = 2;
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 2, "[ERROR TRUNCATED, detail lost: 8388634 bytes, sha256 ") && o.host.files.begin()->second.size() < (1u << 20)); }
    // Cancellation: inside a dispatch (that dispatch completes; the next host check stops), inside profile preparation.
    { FakeHost* hp = nullptr; c = {}; c.dispatchHook = [&hp](uint64_t s) { if (s == 3) hp->cancel = true; };
      const auto o = invoke(execute(shaders, path()), [&hp](FakeHost& h) { hp = &h; }, c);
      EXPECT(stopped(o, 3, "CANCELLED", 3, "Cancellation requested") && drained(o)); }
    { FakeHost* hp = nullptr; c = {}; c.beginProfileHook = [&hp](uint64_t w) { if (w == 4) hp->cancel = true; };
      const auto o = invoke(execute(shaders, path()), [&hp](FakeHost& h) { hp = &h; }, c);
      EXPECT(stopped(o, 3, "CANCELLED", 3, "after_profile_preparation") && drained(o));
      EXPECT(o.receipt.at("cases").at(1).at("repetitions").at(0).at("dispatch_call_started") == true &&
             o.receipt.at("cases").at(1).at("repetitions").at(0).at("submitted") == false); }
    { FakeHost* hp = nullptr; // Cancellation before Device creation leaves no device and no retirement.
      const auto o = invoke(execute(shaders, path()), [&hp](FakeHost& h) { hp = &h; h.rewriteHook = [&hp](uint64_t n) { if (n == 2) hp->cancel = true; }; });
      EXPECT(stopped(o, 3, "CANCELLED", 0, "before_device_creation") && o.receipt.at("device_created") == false && o.receipt.at("retirement_required") == false); }
    // Deadline: inside profile preparation, during the receipt persistence after a completed dispatch, after the last dispatch.
    { FakeHost* hp = nullptr; c = {}; c.beginProfileHook = [&hp](uint64_t w) { if (w == 2) hp->offset = std::chrono::seconds(121); };
      const auto o = invoke(execute(shaders, path()), [&hp](FakeHost& h) { hp = &h; }, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 1, "120-second source deadline exceeded at after_profile_preparation") && drained(o)); }
    { FakeHost* hp = nullptr; c = {}; c.dispatchHook = [&hp](uint64_t s) { if (s == 1) hp->rewriteHook = [&hp](uint64_t) { hp->offset = std::chrono::seconds(120); }; };
      const auto o = invoke(execute(shaders, path()), [&hp](FakeHost& h) { hp = &h; }, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 1, "deadline exceeded at before_allocation") && drained(o)); }
    { FakeHost* hp = nullptr; c = {}; c.dispatchHook = [&hp](uint64_t s) { if (s == 102) hp->offset = std::chrono::seconds(130); };
      const auto o = invoke(execute(shaders, path()), [&hp](FakeHost& h) { hp = &h; }, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 102, "deadline exceeded") && drained(o)); }
    // Identities: driver identity missing, producer unavailable, Device creation refused, created device mismatch.
    { const auto o = invoke(execute(shaders, path()), [](FakeHost& h) { h.driverMissing = true; });
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 0, "Driver identity for the pinned LUID is missing") && o.receipt.at("device_created") == false &&
             o.receipt.at("retirement_required") == false); }
    { const auto o = invoke(execute(shaders, path()), [](FakeHost& h) { h.producerThrows = true; });
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 0, "Executable path unavailable") && o.receipt.at("device_created") == false); }
    c = {}; c.constructorThrows = true;
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 0, "adapter identity not found") && o.receipt.at("device_created") == false); }
    c = {}; c.identityOverride = Json{{"pci_bdf", "04:00.0"}, {"luid", "00000000:0000abcd"}, {"vendor_id", 0x8086u}, {"description", "A770"}}.dump();
    { const auto o = invoke(execute(shaders, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 0, "PCI/LUID differs") && o.receipt.at("device_created") == true && drained(o)); }
    c = {}; c.identityOverride = Json{{"pci_bdf", "03:00.0"}, {"luid", "00000000:0000abcd"}, {"vendor_id", 0x1002u}, {"description", "A770"}}.dump();
    { const auto o = invoke(execute(shaders, path()), {}, c); EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 0, "not an Intel adapter")); }
    // Shader bytes: refused before the Device when changed beforehand; failed after the run when changed during it.
    const fs::path copy = scratch / "shader-copy";
    fs::create_directories(copy / "runtime"); fs::create_directories(copy / "control");
    for (const auto& f : tsr::frozenShaders) fs::copy_file(shaders / f.relative, copy / f.relative);
    c = {}; c.dispatchHook = [&copy](uint64_t s) {
        if (s == 10) { std::fstream f(copy / "control/text_attention_softmax_2c44182.hlsl", std::ios::in | std::ios::out | std::ios::binary); f.seekp(3); f.put('X'); }
    };
    { const auto o = invoke(execute(copy, path()), {}, c);
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 102, "Shader bytes differ from the frozen source") && o.receipt.at("tracked_after_final_drain") == 0); }
    { const auto o = invoke(execute(copy, path()));
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 0, "Shader bytes differ from the frozen source") && o.receipt.at("device_created") == false); }
    { const auto o = invoke(execute(shaders, path()), [](FakeHost& h) { h.compilerMissing = true; });
      EXPECT(stopped(o, 1, "FAIL_OR_INCOMPLETE", 102, "HLSL compiler identity is missing")); }
}

void receipts() {
    Json big = tsr::initialReport(); big["state"] = "RUNNING"; big["blob"] = std::string(size_t(tsr::receiptCap), 'b');
    big["failure_drain"] = {{"completed", true}, {"tracked_after", 0}};
    const auto bounded = tsr::boundedReceipt(big); const Json summary = Json::parse(bounded.text);
    EXPECT(!bounded.complete && summary.at("state") == "RECEIPT_OVERSIZE" && summary.at("passed") == false && bounded.text.size() < 65536);
    EXPECT(summary.at("original_state").at("excerpt") == "RUNNING" && summary.at("failure_drain").at("tracked_after") == 0);
    const auto normal = tsr::boundedReceipt(tsr::initialReport());
    EXPECT(normal.complete && normal.text == tsr::initialReport().dump(2) + "\n");
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--quick")) {
        std::cerr << "usage: text-softmax-regression-test SHADER_ROOT FRESH_SCRATCH [--quick]\n"; return 2;
    }
    const fs::path shaders = fs::absolute(argv[1]), scratch = fs::absolute(argv[2]); const bool quick = argc == 4;
    try {
        sha256Vectors(); frozenShaders(shaders, scratch); plan(); productionSource(shaders); generator(); bf16Rounding(); bf16ModelEdges(); oracle(quick);
        Json waves = Json::array();
        const std::vector<std::pair<uint32_t, tsr_fake::WaveOrder>> models = quick
            ? std::vector<std::pair<uint32_t, tsr_fake::WaveOrder>>{{32, tsr_fake::WaveOrder::ascending}}
            : std::vector<std::pair<uint32_t, tsr_fake::WaveOrder>>{{8, tsr_fake::WaveOrder::ascending}, {16, tsr_fake::WaveOrder::ascending},
                                                                    {32, tsr_fake::WaveOrder::ascending}, {64, tsr_fake::WaveOrder::ascending},
                                                                    {128, tsr_fake::WaveOrder::ascending}, {32, tsr_fake::WaveOrder::descending}};
        for (const auto& [width, order] : models) waves.push_back(fakePlan(width, order));
        for (const auto& w : waves) EXPECT(w.at("first_repaired_output_sha256") == waves.front().at("first_repaired_output_sha256"));
        arguments(shaders, scratch);
        if (!quick) lifecycle(shaders, scratch);
        receipts();
        std::cout << Json{{"checks", checks}, {"failures", failures}, {"quick", quick}, {"plan_sha256", tsr::planSha256()}, {"wave_models", waves}}.dump()
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "UNEXPECTED EXCEPTION: " << e.what() << "\n"; return 1;
    }
    return failures ? 1 : 0;
}
