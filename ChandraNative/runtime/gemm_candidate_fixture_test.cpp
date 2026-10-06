// New ChandraNative code, MPL-2.0. CPU-only checks of gemm_candidate_fixture.h: plan coverage
// and admission, exact operand generation, the integer oracle against independent derivations,
// sentinel/comparison/profile/forecast/schedule/argument/Job/identity rules, the run loop's stop
// behaviour and the candidate shader source structure. It never compiles or executes HLSL.
// abiModel() is a plain C++ statement of linear.hlsl's per-output ABI semantics; FakeDevice
// (a chandra::dc::Device implementation for this test only) maps all three shader names to it,
// so it checks fixture packing, addressing and harness control flow, never the kernels themselves.
// Usage: gemm-candidate-fixture-test SHADER_RUNTIME_DIRECTORY SUMMARY_JSON [--skip-large]
#include "gemm_candidate_fixture.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>

using namespace chandra::gemm_candidates;
namespace {
uint64_t checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) throw std::runtime_error("CHECK FAILED: " + what);
}
void bounded(bool ok) { if (!ok) throw std::runtime_error("CHECK FAILED: ABI model access outside its buffer"); }
template <class F> void refuses(F&& f, const std::string& what, const std::string& fragment = "") {
    std::string message;
    try { f(); } catch (const std::exception& e) { message = e.what(); }
    check(!message.empty(), "expected refusal: " + what);
    check(fragment.empty() || message.find(fragment) != std::string::npos, "refusal message for " + what + ": " + message);
}
bool finite(uint32_t word) { return (word & 0x7f800000u) != 0x7f800000u; }
bool subnormal(uint32_t word) { return (word & 0x7f800000u) == 0 && (word & 0x7fffffu) != 0; }

// linear.hlsl semantics per output, with HLSL uint32 index arithmetic and checked raw loads.
float load(const std::vector<uint32_t>& buffer, uint32_t byteAddress) {
    bounded(byteAddress % 4 == 0 && byteAddress / 4 < buffer.size());
    return fromBits(buffer[byteAddress / 4]);
}
float packedModel(const std::vector<uint32_t>& source, uint32_t index, bool bf16) {
    if (!bf16) return load(source, index * 4u);
    const uint32_t word = bits(load(source, (index >> 1) * 4u));
    return fromBits(((word >> ((index & 1u) * 16u)) & 0xffffu) << 16);
}
void abiModel(const LinearParams& p, const std::vector<uint32_t>& input, const std::vector<uint32_t>& weight,
              const std::vector<uint32_t>& bias, std::vector<uint32_t>& output) {
    for (uint32_t localRow = 0; localRow < p.batchRows; ++localRow)
        for (uint32_t localOut = 0; localOut < p.shardRows; ++localOut) {
            const uint32_t row = p.rowFirst + localRow;
            float sum = 0.0f;
            for (uint32_t k = 0; k < p.inputWidth; ++k) {
                const float product = load(input, (row * p.inputWidth + k) * 4u) *
                                      packedModel(weight, (p.weightRowFirst + localOut) * p.inputWidth + k, (p.flags & 1u) != 0);
                sum = sum + product;
            }
            const uint32_t column = p.firstOutput + localOut;
            if (p.flags & 4u) sum = sum + packedModel(bias, column, (p.flags & 8u) != 0);
            uint32_t stored = bits(sum);
            if (p.flags & 2u) stored = roundedBF16Bits(stored);
            const uint32_t address = (row * p.outputWidth + column) * 4u;
            bounded(address / 4 < output.size());
            output[address / 4] = stored;
        }
}

void arithmetic() {
    const int64_t one = int64_t(1) << 42;
    check(fp32BitsOfUnits(one) == 0x3f800000u && fp32BitsOfUnits(-one / 2) == 0xbf000000u, "binary32 bits of +1 and -0.5");
    check(fp32BitsOfUnits(1) == 0x2a800000u && fp32BitsOfUnits(0) == 0, "binary32 bits of 2^-42 and +0");
    check(roundSignificant(0x1000001, 24) == 0x1000000 && roundSignificant(0x1000003, 24) == 0x1000004, "rn24 ties to even");
    check(roundSignificant(-0x1000003, 24) == -0x1000004 && roundSignificant(0x1ffffff, 24) == 0x2000000, "rn24 sign and carry");
    check(roundSignificant(0x180, 8) == 0x180 && roundSignificant(0x181, 8) == 0x180 && roundSignificant(0x183, 8) == 0x184, "rn8");
    refuses([] { (void)fp32BitsOfUnits(0x1000001); }, "25 significant bits are not binary32");
    refuses([] { (void)toUnits(int64_t(1) << 40, -20); }, "operand outside the oracle grid");
    // Sweep: integer derivations against the CPU's own binary32 conversion and the bit formula.
    uint64_t state = 1;
    for (int i = 0; i < 200000; ++i) {
        state = splitmix(state);
        const int length = 1 + int(state % 60);
        int64_t value = int64_t((state >> 4) & ((1ull << length) - 1)) | (int64_t(1) << (length - 1));
        if (state >> 63) value = -value;
        const int64_t rounded = roundSignificant(value, 24);
        const float reference = float(std::ldexp(double(rounded), unitExponent));
        check(fp32BitsOfUnits(rounded) == bits(reference), "binary32 bits match ldexp conversion");
        check(bf16BitsOfUnits(rounded) == roundedBF16Bits(bits(reference)), "integer BF16 RNE matches the bit formula");
        const float direct = float(std::ldexp(double(value), unitExponent));
        if (length <= 53) check(bits(direct) == fp32BitsOfUnits(rounded), "rn24 matches the CPU's binary32 rounding");
    }
}

void coverage(const std::vector<CaseSpec>& cases) {
    auto any = [&](auto predicate) { return std::any_of(cases.begin(), cases.end(), predicate); };
    check(any([](const CaseSpec& s) { return s.inputWidth == 1; }), "K = 1");
    check(any([](const CaseSpec& s) { return s.inputWidth % 2 && s.bf16Weights && s.weightRowFirst % 2; }), "odd K BF16 with odd shard row start");
    check(any([](const CaseSpec& s) { return s.inputWidth % 32 && s.inputWidth > 64; }), "K tail inside both tile sizes");
    check(any([](const CaseSpec& s) { return s.inputWidth % 64 == 32; }), "K64 tail of exactly 32");
    check(any([](const CaseSpec& s) { return s.batchRows % 16 && s.batchRows > 16; }), "ragged rows across two row groups");
    check(any([](const CaseSpec& s) { return s.shardRows % 16 && s.shardRows > 16; }), "ragged output columns");
    check(any([](const CaseSpec& s) { return s.rowFirst && s.firstOutput() && s.weightRowFirst && s.shardFirstRow; }), "all offsets nonzero");
    check(any([](const CaseSpec& s) { return s.rowFirst + s.batchRows < s.totalRows; }), "rows after the dispatched rows");
    check(any([](const CaseSpec& s) { return s.firstOutput() + s.shardRows < s.outputWidth && s.firstOutput(); }), "columns on both sides");
    check(any([](const CaseSpec& s) { return s.bias == BiasKind::bf16 && s.firstOutput() % 2; }), "BF16 bias at an odd column");
    check(any([](const CaseSpec& s) { return s.bias == BiasKind::fp32; }) && any([](const CaseSpec& s) { return s.bias == BiasKind::none; }), "bias kinds");
    check(any([](const CaseSpec& s) { return !s.bf16Weights && s.generator == Generator::dyadic; }), "FP32 weights, exact");
    check(any([](const CaseSpec& s) { return !s.bf16Weights && s.generator == Generator::orderBf16Input; }), "FP32 weights, order-sensitive");
    check(any([](const CaseSpec& s) { return s.batchRows == 1 && s.inputWidth == 9216 && s.shardRows == 1024; }), "B1 K9216 fallback geometry");
    for (uint32_t rows : {16u, 32u})
        check(any([&](const CaseSpec& s) { return s.timingShape && s.batchRows == rows && s.inputWidth == 9216 && s.shardRows == 1024; }), "K9216/N1024 timing shape");
    check(any([](const CaseSpec& s) { return s.generator == Generator::orderFp32Input && s.inputWidth == 9216 && s.batchRows == 32; }), "full-size order-sensitive case");
    check(std::all_of(cases.begin(), cases.end(), [](const CaseSpec& s) { return s.guardWords > 0; }), "guard words in every case");
}

void admission(const std::vector<CaseSpec>& cases) {
    uint64_t maximumTracked = 0;
    for (const auto& s : cases) {
        Json a = admit(s);
        check(a.at("exact_for_every_summation_order") == (s.generator == Generator::dyadic), std::string("order sensitivity flag ") + s.name);
        check(a.at("oracle_partial_sum_bound_log2").get<double>() < 61.0, "partial sum bound");
        maximumTracked = std::max(maximumTracked, trackedBytes(s) + 4 * words(s).output);
    }
    check(maximumTracked <= deviceCaseCapBytes, "largest case fits the device cap");
    std::cout << "  largest case tracked bytes plus readback staging: " << maximumTracked << "\n";
    const CaseSpec base = cases[1];
    auto bad = [&](auto edit, const char* what, const char* fragment) { CaseSpec s = base; edit(s); refuses([&] { (void)admit(s); }, what, fragment); };
    bad([](CaseSpec& s) { s.inputWidth = 0; }, "K = 0", "Input width");
    bad([](CaseSpec& s) { s.inputWidth = 9217; }, "K = 9217", "Input width");
    bad([](CaseSpec& s) { s.batchRows = 0; }, "zero rows", "Batch rows");
    bad([](CaseSpec& s) { s.batchRows = 33; s.totalRows = 40; }, "33 rows", "Batch rows");
    bad([](CaseSpec& s) { s.shardRows = 0; }, "zero outputs", "Dispatch outputs");
    bad([](CaseSpec& s) { s.shardRows = 1025; }, "1025 outputs", "Dispatch outputs");
    bad([](CaseSpec& s) { s.rowFirst = s.totalRows - s.batchRows + 1; }, "rows past the activation", "activation rows");
    bad([](CaseSpec& s) { s.weightRowFirst = s.shardTotalRows - s.shardRows + 1; }, "outputs past the shard", "weight shard");
    bad([](CaseSpec& s) { s.shardTotalRows = s.outputWidth; }, "shard past the output width", "output width");
    bad([](CaseSpec& s) { s.generator = Generator::orderBf16Input; }, "24-bit weights stored as BF16", "significand");
    bad([](CaseSpec& s) { s.inputWidth = 9216; s.totalRows = 4000; }, "buffer above 128 MiB", "bounded raw");
    bad([](CaseSpec& s) { s.inputWidth = 9216; s.bf16Weights = false; s.totalRows = 32; s.shardTotalRows = 3600; s.outputWidth = 3700; }, "device cap", "device cap");
}

struct Totals { uint64_t tiesDown = 0, tiesUp = 0, changed = 0, negative = 0, zeros = 0; };
void fixtureAndOracle(const std::vector<CaseSpec>& cases, bool skipLarge, Json& summary, Totals& totals) {
    Deadline unlimited;
    for (const auto& s : cases) {
        if (skipLarge && uint64_t(s.batchRows) * s.shardRows * s.inputWidth > 50'000'000ull) continue;
        const auto started = std::chrono::steady_clock::now();
        unlimited.start = started;
        const auto in = inputWords(s), wt = weightWords(s, unlimited), bs = biasWords(s);
        bool signs[2] = {}; uint64_t zeroOperands = 0;
        for (uint32_t word : in) { check(finite(word) && !subnormal(word), "input words finite and normal or zero"); signs[word >> 31] = true; zeroOperands += !(word & 0x7fffffffu); }
        for (uint32_t word : wt) {
            if (s.bf16Weights) for (uint32_t half : {word << 16, word & 0xffff0000u}) check(finite(half) && !subnormal(half), "BF16 weights finite");
            else check(finite(word) && !subnormal(word), "FP32 weights finite");
        }
        if (s.bias == BiasKind::fp32) for (uint32_t word : bs) check(finite(word) && !subnormal(word), "FP32 bias finite");
        check(signs[0] && signs[1], std::string("signed inputs in ") + s.name);
        const auto e = oracle(s, unlimited, s.batchRows);
        const double oracleSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const Json& o = e.analysis.at("alternative_order_observation");
        if (s.generator == Generator::dyadic) {
            check(e.analysis.at("product_rounding_events") == 0 && e.analysis.at("sum_rounding_events") == 0, std::string("dyadic exactness ") + s.name);
            for (const char* key : {"fused_products_differ", "descending_k_differ", "split_k32_blocks_differ", "single_rounding_of_exact_dot_differ"})
                check(o.at(key) == 0, std::string("dyadic order insensitivity ") + key);
        } else {
            check(e.analysis.at("product_rounding_events").get<uint64_t>() > 0 && e.analysis.at("sum_rounding_events").get<uint64_t>() > 0,
                  std::string("order case rounds products and sums ") + s.name);
            for (const char* key : {"fused_products_differ", "descending_k_differ", "split_k32_blocks_differ", "single_rounding_of_exact_dot_differ"})
                check(o.at(key).get<uint64_t>() > 0, std::string("order case detects ") + key + " in " + s.name);
        }
        for (uint32_t word : e.raw) check(finite(word) && !subnormal(word), "expected outputs finite and normal or zero");
        totals.tiesDown += e.analysis.at("bf16_ties_round_down_to_even").get<uint64_t>();
        totals.tiesUp += e.analysis.at("bf16_ties_round_up_to_even").get<uint64_t>();
        totals.changed += e.analysis.at("bf16_changed_outputs").get<uint64_t>();
        totals.negative += e.analysis.at("negative_outputs").get<uint64_t>();
        totals.zeros += e.analysis.at("zero_outputs").get<uint64_t>();
        // ABI model over the uploaded words must reproduce the oracle image and leave every sentinel.
        for (bool round : {false, true}) {
            const uint32_t salt = 7 + uint32_t(round);
            auto image = sentinelImage(s, salt);
            abiModel(params(s, round), in, wt, s.bias == BiasKind::none ? wt : bs, image);
            const auto expected = expectedImage(s, round ? e.rounded : e.raw, salt);
            const Json c = compareImage(s, image, expected);
            check(c.at("passed") == true, std::string("ABI model equals oracle image and sentinels ") + s.name);
            check(regionOf(s, image) == (round ? e.rounded : e.raw), "region extraction");
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        // Deterministic summary (no timings), so separately compiled binaries can be byte-compared.
        summary.push_back({{"case", s.name}, {"generator", generatorName(s.generator)}, {"admission", admit(s)},
                           {"oracle", e.analysis}, {"zero_inputs", zeroOperands}});
        std::cout << "  " << s.name << ": operands+oracle " << oracleSeconds << " s, with ABI model " << seconds << " s\n";
    }
}

void imageChecks(const CaseSpec& s) {
    Deadline unlimited;
    const auto e = oracle(s, unlimited, 0);
    const auto expected = expectedImage(s, e.raw, 3);
    for (size_t i = 0; i < expected.size(); ++i)
        check(inRegion(s, i) ? finite(expected[i]) : !finite(expected[i]) && (expected[i] & 0x7fffffu), "sentinels are NaN, region finite");
    check(sentinelImage(s, 3) != sentinelImage(s, 4), "salts change every sentinel image");
    auto flipped = expected; flipped[size_t(s.rowFirst) * s.outputWidth + s.firstOutput()] ^= 1u;
    Json c = compareImage(s, flipped, expected);
    check(c.at("region_mismatches") == 1 && c.at("sentinel_mismatches") == 0 && c.at("passed") == false, "one region bit flip detected");
    flipped = expected; flipped.back() ^= 0x10u;
    c = compareImage(s, flipped, expected);
    check(c.at("region_mismatches") == 0 && c.at("sentinel_mismatches") == 1, "one guard-word change detected");
    c = compareImage(s, sentinelImage(s, 3), expected);
    check(c.at("region_mismatches") == uint64_t(s.batchRows) * s.shardRows, "a dispatch that stores nothing fails every region word");
    c = compareImage(s, expectedImage(s, e.raw, 2), expected);
    check(c.at("region_mismatches") == 0 && c.at("sentinel_mismatches") == words(s).output - uint64_t(s.batchRows) * s.shardRows,
          "a stale image from another salt fails every sentinel");
    check(c.at("first_mismatches").size() == reportedMismatches, "mismatch detail is bounded");
    refuses([&] { (void)compareImage(s, std::vector<uint32_t>(expected.begin(), expected.end() - 1), expected); }, "short readback", "Incomplete");
}

void profileChecks() {
    const std::array<uint32_t, 3> g{64, 2, 1};
    auto good = [&] { return Json{{"frequency", 12000000u}, {"disjoint", false},
                                  {"dispatches", Json::array({Json{{"shader", kernels[1].shader}, {"groups", g}, {"gpu_milliseconds", 6.4}}})}}; };
    check(checkProfile(good(), kernels[1].shader, g) == 6.4, "valid single-dispatch profile");
    Json p = good(); p["dispatches"][0]["gpu_milliseconds"] = 99.999;
    check(checkProfile(p, kernels[1].shader, g) == 99.999, "just below 100 ms admitted");
    auto bad = [&](auto edit, const char* what) { Json q = good(); edit(q); refuses([&] { (void)checkProfile(q, kernels[1].shader, g); }, what); };
    bad([](Json& q) { q["disjoint"] = true; }, "disjoint window");
    bad([](Json& q) { q.erase("disjoint"); }, "missing disjoint flag");
    bad([](Json& q) { q["disjoint"] = 0; }, "non-boolean disjoint flag");
    bad([](Json& q) { q["frequency"] = 0u; }, "zero frequency");
    bad([](Json& q) { q["frequency"] = -5; }, "negative frequency");
    bad([](Json& q) { q["dispatches"] = Json::array(); }, "missing dispatch");
    bad([](Json& q) { q["dispatches"].push_back(q["dispatches"][0]); }, "extra dispatch");
    bad([](Json& q) { q["dispatches"][0]["shader"] = kernels[0].shader; }, "different shader (no fallback)");
    bad([](Json& q) { q["dispatches"][0]["groups"] = Json::array({64, 1, 1}); }, "different groups");
    bad([](Json& q) { q["dispatches"][0].erase("gpu_milliseconds"); }, "missing duration");
    bad([](Json& q) { q["dispatches"][0]["gpu_milliseconds"] = std::nan(""); }, "NaN duration");
    bad([](Json& q) { q["dispatches"][0]["gpu_milliseconds"] = INFINITY; }, "infinite duration");
    bad([](Json& q) { q["dispatches"][0]["gpu_milliseconds"] = -0.001; }, "negative duration");
    bad([](Json& q) { q["dispatches"][0]["gpu_milliseconds"] = 100.0; }, "exactly 100 ms");
    bad([](Json& q) { q["dispatches"][0]["gpu_milliseconds"] = "6.4"; }, "string duration");
}

void forecastChecks(const std::vector<CaseSpec>& cases) {
    // Plausible model: 20 us launch cost plus the measured predecessor rate (3.05 ms for 64 groups
    // at K=9216), with candidates up to 3x slower. Every step of the plan must be admitted.
    for (double slowdown : {1.0, 3.0}) {
        for (size_t i = 0; i + 1 < cases.size(); ++i) {
            const double rate = 3.05 / (64.0 * 9216.0) * slowdown;
            const double t = 0.02 + rate * double(work(cases[i]));
            (void)admitNext({t, t, t}, work(cases[i]), work(cases[i + 1]));
        }
    }
    refuses([] { (void)admitNext({1.0, 30.0, 1.0}, 100, 200); }, "a slow candidate is stopped before a larger case", "linear_gemm_padded32");
    refuses([] { (void)admitNext({1.0, 1.0, std::nan("")}, 100, 200); }, "missing observation", "completed observation");
    const Json f = admitNext({1.0, 2.0, 3.0}, 100, 300);
    check(f.at(2).at("forecast_ms").get<double>() == 18.0, "forecast = 2 * slowest * work ratio");
    uint64_t largest = 0; for (const auto& s : cases) largest = std::max(largest, work(s));
    check(largest == 2ull * 64 * 9216, "largest work is the measured 32-row production dispatch");
}

void scheduleChecks() {
    const auto slots = timingSchedule(timingRounds, 2);
    check(slots.size() == timingRounds * 2 * 18, "timing slot count");
    for (uint32_t round = 0; round < timingRounds; ++round)
        for (uint32_t shape = 0; shape < 2; ++shape)
            for (uint32_t position = 0; position < 3; ++position)
                for (uint32_t kernel = 0; kernel < 3; ++kernel)
                    check(std::count_if(slots.begin(), slots.end(), [&](const TimingSlot& t) {
                              return t.round == round && t.shape == shape && t.position == position && t.kernel == kernel; }) == 2,
                          "each kernel holds each position twice per round and shape");
    check(slots.front().shape == 0 && slots[18 * 2].shape == 1, "shape order alternates by round");
    const Json t = timingSummary({3.0, 1.0, 2.0, 10.0});
    check(t.at("median_ms") == 2.5 && t.at("mean_ms") == 4.0 && t.at("minimum_ms") == 1.0 && t.at("maximum_ms") == 10.0, "timing summary");
}

void argumentChecks() {
    using V = std::vector<std::wstring>;
    check(!parseArguments({}).execute, "no arguments is inactive");
    const V full = {L"--execute", L"--plan", L"correctness-then-timing", L"--shader-root", L"C:\\s", L"--pci", L"03:00.0",
                    L"--luid", L"00000000:0001a2b3", L"--output", L"C:\\r\\receipt.json"};
    const Arguments a = parseArguments(full);
    check(a.execute && a.plan == PlanSelector::correctnessThenTiming && a.pci == L"03:00.0" && a.luid == L"00000000:0001a2b3", "valid execution arguments");
    check(!parseArguments({L"--plan", L"correctness"}).execute, "options without --execute stay inactive");
    auto bad = [&](V v, const char* what) { refuses([&] { (void)parseArguments(v); }, what); };
    bad({L"--execute", L"--execute"}, "duplicate execute");
    bad({L"--pci", L"03:00.0", L"--pci", L"03:00.0"}, "duplicate option");
    bad({L"--verbose"}, "unknown option");
    bad({L"--pci"}, "missing value");
    bad({L"--pci", L""}, "empty value");
    bad({L"--pci", L"--luid"}, "option as value");
    for (const wchar_t* plan : {L"", L"Correctness", L"timing", L"correctness ", L"correctness+timing"}) bad({L"--plan", plan}, "plan selector");
    for (const wchar_t* pci : {L"3:00.0", L"03:00.8", L"03:20.0", L"03:00:0", L"03:0A.0", L"003:00.0"}) bad({L"--pci", pci}, "PCI format");
    for (const wchar_t* luid : {L"0:1", L"00000000:0001A2B3", L"000000000001a2b3", L"00000000-0001a2b3"}) bad({L"--luid", luid}, "LUID format");
    for (size_t drop = 1; drop < full.size(); drop += 2) {
        V partial = full; partial.erase(partial.begin() + long(drop), partial.begin() + long(drop) + 2);
        bad(partial, "execution without every required option");
    }
}

void identityAndJob() {
    const Json identity = {{"pci_bdf", "03:00.0"}, {"luid", "00000000:0001a2b3"}, {"vendor_id", 32902}, {"description", "Intel(R) Arc(TM) A770 Graphics"}};
    authenticateIdentity(identity, "03:00.0", "00000000:0001a2b3");
    auto bad = [&](const char* key, Json value) { Json i = identity; i[key] = value; refuses([&] { authenticateIdentity(i, "03:00.0", "00000000:0001a2b3"); }, key); };
    bad("pci_bdf", "04:00.0"); bad("luid", "00000000:0001a2b4"); bad("vendor_id", 4318); bad("description", "Intel(R) Arc(TM) A750 Graphics"); bad("pci_bdf", nullptr);
    JobLimits j; j.inJob = true; j.flags = jobLimitKillOnJobClose | jobLimitProcessMemory; j.processMemory = jobMemoryCapBytes;
    check(admitJob(j).at("kill_on_close") == true, "kill-on-close process-memory Job");
    j.flags = jobLimitKillOnJobClose | jobLimitJobMemory; j.jobMemory = 512ull << 20;
    check(admitJob(j).at("job_memory_limit") == 512ull << 20, "job-memory Job");
    auto refused = [&](JobLimits l, const char* what) { refuses([&] { (void)admitJob(l); }, what); };
    JobLimits l = j; l.inJob = false; refused(l, "not in a Job");
    l = j; l.flags = jobLimitJobMemory; refused(l, "no kill on close");
    l = j; l.flags = jobLimitKillOnJobClose; refused(l, "no memory limit");
    l = j; l.jobMemory = 0; refused(l, "zero memory limit");
    l = j; l.jobMemory = jobMemoryCapBytes + 1; refused(l, "memory limit above 1 GiB");
    l = j; l.flags |= jobLimitProcessMemory; l.processMemory = 4ull << 30; refused(l, "either memory limit above 1 GiB");
}

std::string read(const std::string& path) {
    std::ifstream file(path, std::ios::binary); check(bool(file), "read " + path);
    std::stringstream text; text << file.rdbuf(); return text.str();
}
size_t count(const std::string& text, const std::string& token) {
    size_t n = 0; for (size_t at = text.find(token); at != std::string::npos; at = text.find(token, at + 1)) ++n; return n;
}
std::string between(const std::string& text, const std::string& first, const std::string& last) {
    const size_t a = text.find(first), b = text.find(last, a == std::string::npos ? 0 : a);
    check(a != std::string::npos && b != std::string::npos, "source anchors " + first);
    return text.substr(a, b + last.size() - a);
}
void sourceStructure(const std::string& directory) {
    const std::string production = read(directory + "/" + kernels[0].file);
    const std::string abi = between(production, "ByteAddressBuffer input", "& 0xffff0000);\n}\n");
    const std::string epilogue = between(production, "    if (localRow < batchRows && localOut < shardRows) {", "    }\n}\n");
    const std::string chain = "            precise float product = tileInput[lane.y][k] * tileWeight[lane.x][k];\n            sum = sum + product;\n";
    check(production.find(chain) != std::string::npos && count(production, "GroupMemoryBarrierWithGroupSync") == 2, "production anchors");
    for (size_t i = 1; i < kernels.size(); ++i) {
        const std::string text = read(directory + "/" + kernels[i].file);
        const std::string name = kernels[i].file;
        check(between(text, "ByteAddressBuffer input", "& 0xffff0000);\n}\n") == abi, name + ": bindings, cbuffer, packed() and bf16_rne() identical");
        check(between(text, "    if (localRow < batchRows && localOut < shardRows) {", "    }\n}\n") == epilogue, name + ": bias, rounding and store identical");
        check(text.find(chain) != std::string::npos, name + ": identical precise product-then-sum statement");
        check(text.find("#define TILE_K " + std::to_string(kernels[i].tileK) + "\n") != std::string::npos &&
              text.find("#define PARTS " + std::to_string(kernels[i].tileK / 16) + "\n") != std::string::npos, name + ": tile depth");
        check(count(text, "groupshared float tile") == 2 && count(text, "[16][TILE_K + 1];") == 2, name + ": odd padded stride");
        check(text.find("[numthreads(16,16,1)]") != std::string::npos && text.find("precise float sum = 0.0;") != std::string::npos, name + ": group and +0 start");
        check(text.find("for (uint begin = 0; begin < inputWidth; begin += TILE_K) {") != std::string::npos &&
              text.find("uint n = min(TILE_K, inputWidth - begin);\n        for (uint k = 0; k < n; ++k) {") != std::string::npos, name + ": ascending tiles and k");
        check(count(text, "GroupMemoryBarrierWithGroupSync();") == 2, name + ": two barriers per tile");
        const std::string body = text.substr(text.find("void main("));
        check(count(body, "return") == 0, name + ": no early exit from main");
        check(between(body, "for (uint begin", "GroupMemoryBarrierWithGroupSync();").find("if (") == std::string::npos, name + ": staging has no divergent branch");
        for (const char* forbidden : {"half", "min16", "mad(", "fma(", "dot(", "Wave", "Interlocked", "discard", "#include", "[branch]", "double"})
            check(text.find(forbidden) == std::string::npos, name + ": no " + forbidden);
        check(count(text, "precise float") == 2 && count(text, "sum = sum +") == 2, name + ": precise sum and product only");
    }
}
} // namespace

// ---- FakeDevice: chandra::dc::Device for this test binary only. Every shader name runs abiModel(),
// timings come from a linear work model, and faults are injected at a chosen dispatch ordinal. ----
namespace {
struct Faults {
    enum Kind { none, flipRegion, writeSentinel, skipStore, slow, disjoint, wrongShader } kind = none;
    uint64_t at = 0;          // 1-based dispatch ordinal
    double milliseconds = 0;  // reported duration for `slow`
};
Faults faults;
uint64_t fakeDispatches = 0;
bool fires(Faults::Kind kind) { return faults.kind == kind && faults.at == fakeDispatches; }
double modelMilliseconds(const LinearParams& p) {
    return 0.01 + 3.05 / (64.0 * 9216.0) * double((p.shardRows + 15) / 16) * double((p.batchRows + 15) / 16) * p.inputWidth;
}
} // namespace
namespace chandra::dc {
struct Storage {
    std::vector<uint32_t> data; std::shared_ptr<uint64_t> account;
    ~Storage() { if (account) *account -= data.size() * 4; }
};
struct Device::Impl { std::shared_ptr<uint64_t> account = std::make_shared<uint64_t>(0); bool profiling = false, disjoint = false; Json rows = Json::array(); };
Device::Device(const std::wstring&, const std::string&, const std::string&) : impl(std::make_unique<Impl>()) {}
Device::~Device() = default;
Buffer Device::words(uint32_t count, const uint32_t* initial) {
    if (!count || uint64_t(count) * 4 > maximumBufferBytes) throw std::runtime_error("FakeDevice: nonempty buffer of at most 128 MiB required");
    auto storage = std::make_shared<Storage>();
    storage->data.assign(count, 0u);
    if (initial) std::copy(initial, initial + count, storage->data.begin());
    storage->account = impl->account; *impl->account += uint64_t(count) * 4;
    return {storage, count, count, false};
}
void Device::upload(Buffer& b, const void* values, uint32_t bytes) {
    if (!b.storage || !values || bytes != b.storage->data.size() * 4) throw std::runtime_error("FakeDevice: complete upload required");
    std::memcpy(b.storage->data.data(), values, bytes);
}
void Device::dispatch(const std::string& name, const std::vector<const Buffer*>& inputs, const std::vector<Buffer*>& outputs,
                      const void* parameters, uint32_t parameterBytes, uint32_t x, uint32_t y, uint32_t z) {
    auto valid = [&](const Buffer* b) {
        return b && b->storage && b->storage->account == impl->account && uint64_t(b->words) * 4 == b->storage->data.size() * 4 &&
               b->logicalElements >= 1 && b->logicalElements <= uint64_t(b->words) * (b->packedBF16 ? 2 : 1);
    };
    bool known = false; for (const auto& k : kernels) known = known || name == k.shader;
    if (!known || inputs.size() != 3 || outputs.size() != 1 || parameterBytes != sizeof(LinearParams) || !parameters)
        throw std::runtime_error("FakeDevice: unexpected dispatch signature");
    for (const auto* b : inputs) if (!valid(b) || b->storage == outputs[0]->storage) throw std::runtime_error("FakeDevice: invalid or aliased SRV");
    if (!valid(outputs[0])) throw std::runtime_error("FakeDevice: invalid UAV");
    LinearParams p; std::memcpy(&p, parameters, sizeof(p));
    if (x != (p.shardRows + 15) / 16 || y != (p.batchRows + 15) / 16 || z != 1) throw std::runtime_error("FakeDevice: groups differ from the ABI");
    ++fakeDispatches;
    auto& out = outputs[0]->storage->data;
    if (!fires(Faults::skipStore)) abiModel(p, inputs[0]->storage->data, inputs[1]->storage->data, inputs[2]->storage->data, out);
    if (fires(Faults::flipRegion)) out[size_t(p.rowFirst) * p.outputWidth + p.firstOutput] ^= 1u;
    if (fires(Faults::writeSentinel)) out.back() ^= 0x10u;
    if (impl->profiling) {
        impl->rows.push_back({{"shader", fires(Faults::wrongShader) ? kernels[0].shader : name.c_str()}, {"groups", {x, y, z}},
                              {"gpu_milliseconds", fires(Faults::slow) ? faults.milliseconds : modelMilliseconds(p)}});
        impl->disjoint = impl->disjoint || fires(Faults::disjoint);
    }
}
std::vector<uint32_t> Device::readWords(const Buffer& b) { return b.storage->data; }
void Device::drain(uint32_t) {}
uint64_t Device::trackedBufferBytes() const { return *impl->account; }
std::string Device::identityJson() const {
    return Json{{"pci_bdf", "03:00.0"}, {"luid", "00000000:0001a2b3"}, {"vendor_id", 32902}, {"description", "FakeDevice A770 stand-in"}}.dump();
}
std::string Device::memoryJson() const { return Json{{"tracked_live", *impl->account}, {"fake_device", true}}.dump(); }
void Device::beginProfile() {
    if (impl->profiling || !impl->rows.empty()) throw std::runtime_error("FakeDevice: profile window already open");
    impl->profiling = true; impl->disjoint = false;
}
std::string Device::finishProfile() {
    if (!impl->profiling) throw std::runtime_error("FakeDevice: no profile window");
    impl->profiling = false;
    Json rows = std::move(impl->rows); impl->rows = Json::array();
    if (impl->disjoint) throw std::runtime_error("GPU timestamps disjoint or unavailable; timing rejected");
    return Json{{"frequency", 12000000u}, {"disjoint", false}, {"dispatches", rows}}.dump();
}
} // namespace chandra::dc

namespace {
// Test-only digest hook (FNV-1a 64); the Windows entry hashes with BCrypt SHA-256.
std::string testDigest(const void* data, size_t bytes) {
    uint64_t h = 14695981039346656037ull; const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    char text[17]; std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h)); return text;
}
struct RunResult { Json report; std::string error; uint64_t checkpoints = 0; size_t largestReceipt = 0; };
RunResult runWith(const std::vector<CaseSpec>& cases, PlanSelector selector, Faults f) {
    faults = f; fakeDispatches = 0;
    RunResult r; r.report = initialReport(cases);
    const Hooks hooks{testDigest, [&r](const Json& j) { ++r.checkpoints; r.largestReceipt = std::max(r.largestReceipt, j.dump(2).size()); }};
    chandra::dc::Device device(L"unused", "03:00.0", "00000000:0001a2b3");
    authenticateIdentity(Json::parse(device.identityJson()), "03:00.0", "00000000:0001a2b3");
    try { run(device, r.report, hooks, selector, Deadline{}, cases); } catch (const std::exception& e) { r.error = e.what(); }
    r.largestReceipt = std::max(r.largestReceipt, r.report.dump(2).size());
    check(device.trackedBufferBytes() == 0, "every buffer released after a pass or a stop");
    check(r.report.at("dispatches_submitted") == fakeDispatches, "receipt dispatch count equals dispatches the device saw");
    check(r.largestReceipt <= receiptCapBytes, "receipt within 8 MiB");
    return r;
}
const Json& lastRecord(const RunResult& r) {
    const Json& c = r.report.at("cases");
    if (r.report.contains("timing") && r.report.at("timing").contains("samples") && !r.report.at("timing").at("samples").empty())
        return r.report.at("timing").at("samples").back();
    return c.back().at("dispatches").back();
}
void stops(const std::vector<CaseSpec>& cases, PlanSelector selector, Faults f, uint64_t dispatches, const std::string& fragment, const char* what) {
    const RunResult r = runWith(cases, selector, f);
    check(r.error.find(fragment) != std::string::npos, std::string(what) + ": stop message (" + r.error + ")");
    check(fakeDispatches == dispatches, std::string(what) + ": no dispatch after the fault");
    check(r.report.value("correctness_passed", false) == (dispatches > cases.size() * 6), std::string(what) + ": correctness flag");
    if (f.kind == Faults::slow && f.milliseconds >= dispatchLimitMilliseconds) {
        const Json& last = lastRecord(r);
        check(last.at("submitted") == true && last.contains("gpu_profile") && !last.contains("comparison") && last.at("passed") == false,
              std::string(what) + ": profile retained, no readback admitted");
    }
    if (f.kind == Faults::flipRegion || f.kind == Faults::writeSentinel || f.kind == Faults::skipStore) {
        const Json& last = lastRecord(r);
        check(last.at("passed") == false && last.at("comparison").at("passed") == false, std::string(what) + ": failing record retained");
    }
}
void runLoop(const std::vector<CaseSpec>& all, bool skipLarge) {
    std::vector<CaseSpec> small(all.begin(), all.begin() + 5);
    small[1].timingShape = small[3].timingShape = true;  // reduced plan: two small timing shapes
    {
        const RunResult r = runWith(small, PlanSelector::correctness, {});
        check(r.error.empty() && fakeDispatches == 30 && r.checkpoints == 5 && r.report.at("correctness_passed") == true, "reduced correctness plan passes: " + r.error);
        for (size_t i = 0; i < small.size(); ++i) {
            const Json& c = r.report.at("cases").at(i);
            check(c.at("passed") == true && c.contains("growth_admission") == (i > 0) && c.at("dispatches").size() == 6, "case record complete");
            for (size_t d = 0; d < 6; ++d) {
                const Json& rec = c.at("dispatches").at(d);
                check(rec.at("kernel") == kernels[d % 3].shader && rec.at("params") == Json(paramWords(params(small[i], d >= 3))) &&
                      rec.at("groups") == Json(groups(small[i])) && rec.at("passed") == true &&
                      rec.contains("matches_predecessor_region") == (d % 3 != 0), "dispatch record order, constants and geometry");
            }
        }
    }
    {
        const RunResult r = runWith(small, PlanSelector::correctnessThenTiming, {});
        const Json& t = r.report.at("timing");
        check(r.error.empty() && fakeDispatches == 30 + timingRounds * 2 * 18 && t.at("samples").size() == timingRounds * 2 * 18, "timing schedule runs: " + r.error);
        check(t.at("summary").size() == 2 && t.at("summary").at(0).at("kernels").size() == 3 && t.at("admission").size() == 2, "timing summary");
        for (const auto& sample : t.at("samples")) check(sample.at("passed") == true && sample.at("call") == "bf16_rounded", "every timed dispatch checked");
        std::cout << "  reduced plan with timing: " << fakeDispatches << " dispatches, largest receipt " << r.largestReceipt << " bytes\n";
    }
    using F = Faults;
    stops(small, PlanSelector::correctness, {F::flipRegion, 8, 0}, 8, "linear_gemm_padded32.hlsl output differs", "candidate region bit flip");
    stops(small, PlanSelector::correctness, {F::writeSentinel, 9, 0}, 9, "linear_gemm_padded64.hlsl output differs", "candidate sentinel store");
    stops(small, PlanSelector::correctness, {F::skipStore, 11, 0}, 11, "linear_gemm_padded32.hlsl output differs", "candidate stores nothing");
    stops(small, PlanSelector::correctness, {F::flipRegion, 7, 0}, 7, "runtime/linear.hlsl output differs", "predecessor/oracle disagreement");
    stops(small, PlanSelector::correctness, {F::slow, 12, 100.0}, 12, "reached 100 ms", "100 ms dispatch");
    stops(small, PlanSelector::correctness, {F::disjoint, 3, 0}, 3, "disjoint", "disjoint window");
    stops(small, PlanSelector::correctness, {F::wrongShader, 2, 0}, 2, "Unexpected profiled shader", "fallback shader");
    stops(small, PlanSelector::correctness, {F::slow, 5, 30.0}, 6, "Forecast for runtime/linear_gemm_padded32.hlsl", "growth gate before a larger case");
    stops(small, PlanSelector::correctnessThenTiming, {F::flipRegion, 35, 0}, 35, "Timing dispatch output differs", "timing-phase mismatch");
    stops(small, PlanSelector::correctnessThenTiming, {F::slow, 40, 100.0}, 40, "reached 100 ms", "timing-phase 100 ms dispatch");
    if (!skipLarge) {
        const RunResult r = runWith(all, PlanSelector::correctness, {});
        check(r.error.empty() && fakeDispatches == all.size() * 6 && r.checkpoints == all.size(), "full correctness plan passes on the stand-in: " + r.error);
        std::cout << "  full correctness plan: " << fakeDispatches << " dispatches, largest receipt " << r.largestReceipt << " bytes\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--skip-large")) {
            std::cerr << "usage: gemm-candidate-fixture-test SHADER_RUNTIME_DIRECTORY SUMMARY_JSON [--skip-large]\n"; return 2;
        }
        const bool skipLarge = argc == 4;
        const auto cases = plan();
        std::cout << "== arithmetic\n"; arithmetic();
        std::cout << "== plan coverage and admission\n"; coverage(cases); admission(cases);
        std::cout << "== fixtures, oracle, independent derivations and ABI model" << (skipLarge ? " (cases <= 50M terms)" : "") << "\n";
        Json summary = Json::array(); Totals totals;
        fixtureAndOracle(cases, skipLarge, summary, totals);
        check(totals.tiesDown > 0 && totals.tiesUp > 0, "BF16 ties round both down and up to even");
        check(totals.changed > 0 && totals.negative > 0, "BF16 rounding changes outputs; negative outputs present");
        std::cout << "== images, profiles, forecasts, schedule, arguments, identity, Job\n";
        imageChecks(cases[1]); imageChecks(cases[4]);
        profileChecks(); forecastChecks(cases); scheduleChecks(); argumentChecks(); identityAndJob();
        std::cout << "== run loop on FakeDevice (control flow and stops only)\n"; runLoop(cases, skipLarge);
        std::cout << "== shader source structure\n"; sourceStructure(argv[1]);
        Json result = {{"checks", checks}, {"bf16_ties_round_down_to_even", totals.tiesDown}, {"bf16_ties_round_up_to_even", totals.tiesUp},
                       {"bf16_changed_outputs", totals.changed}, {"negative_outputs", totals.negative}, {"zero_outputs", totals.zeros},
                       {"cases", summary}};
        std::ofstream summaryFile(argv[2], std::ios::binary | std::ios::trunc);
        summaryFile << result.dump(1) << "\n";
        check(bool(summaryFile.flush()), "write summary JSON");
        std::cout << "ALL GEMM CANDIDATE FIXTURE CHECKS PASSED: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
