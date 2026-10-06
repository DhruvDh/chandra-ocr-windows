// New ChandraNative code, MPL-2.0. CPU-only checks for the vision dispatch calibration: no GPU, model,
// weights, network or install. Links the unchanged vision_model.cpp/operators.cpp against a record-only
// fake Device to prove the calibration issues production tiles, and runs the real calibration plan on an
// executing fake. Usage: vision-dispatch-calibration-test SHADER_ROOT FRESH_SCRATCH_DIRECTORY [--quick]
#include "vision_dispatch_calibration_fake_device.h"
#include "vision_model.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

namespace chandra::dc {
// Record-only vision weights with the pinned shapes validated by vision_model.cpp (block 0 only).
struct ModelWeights::Impl { std::map<std::string, Weight> weights; };
ModelWeights::ModelWeights(Device& d, const std::wstring&, uint64_t, const std::vector<std::string>&) : impl(std::make_unique<Impl>()) {
    auto add = [&](const std::string& name, uint32_t rows, uint32_t cols) {
        Weight w; w.rows = rows; w.cols = cols; w.bf16 = true;
        const uint64_t elements = uint64_t(rows) * cols;
        Buffer b = d.words(uint32_t((elements + 1) / 2)); b.logicalElements = elements; b.packedBF16 = true;
        w.shards.push_back(b); w.shardFirstRows.push_back(0);
        impl->weights.emplace("model.visual." + name, std::move(w));
    };
    add("patch_embed.proj.weight", 1024, 1536); add("patch_embed.proj.bias", 1, 1024); add("pos_embed.weight", 2304, 1024);
    const std::string b = "blocks.0.";
    for (auto norm : {"norm1.", "norm2."}) { add(b + norm + "weight", 1, 1024); add(b + norm + "bias", 1, 1024); }
    add(b + "attn.qkv.weight", 3072, 1024); add(b + "attn.qkv.bias", 1, 3072);
    add(b + "attn.proj.weight", 1024, 1024); add(b + "attn.proj.bias", 1, 1024);
    add(b + "mlp.linear_fc1.weight", 4096, 1024); add(b + "mlp.linear_fc1.bias", 1, 4096);
    add(b + "mlp.linear_fc2.weight", 1024, 4096); add(b + "mlp.linear_fc2.bias", 1, 1024);
}
ModelWeights::~ModelWeights() = default;
const Weight& ModelWeights::at(const std::string& name) const {
    auto it = impl->weights.find(name); if (it == impl->weights.end()) throw std::runtime_error("Fake weight absent: " + name); return it->second;
}
bool ModelWeights::contains(const std::string& name) const { return impl->weights.count(name) != 0; }
std::string ModelWeights::provenanceJson() const { return "{\"fake\":true}"; }
}

namespace {
namespace cal = chandra::vision_calibration;
using cal::Json;
using chandra::dc::Buffer;
using chandra::dc::Device;
using vdc_fake::Event;
std::filesystem::path shaderRoot, scratch;
uint64_t checks = 0;
void check(bool ok, const std::string& message) { ++checks; if (!ok) throw std::runtime_error("CHECK FAILED: " + message); }
void section(const std::string& name) { std::cout << "== " << name << std::endl; }
void resetFake(bool execute = true) {
    vdc_fake::config() = vdc_fake::Config{}; vdc_fake::config().execute = execute;
    vdc_fake::events().clear(); vdc_fake::dispatchSerial() = 0; vdc_fake::flushes() = 0; vdc_fake::profileBegins() = 0;
}
std::vector<Event> dispatches() {
    std::vector<Event> result;
    for (const auto& e : vdc_fake::events()) if (e.kind == "dispatch") result.push_back(e);
    return result;
}
template <class F> std::string thrown(F f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return {};
}
bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
// The pre-repair receipt serialization, which every receipt within the cap must still equal byte for byte.
std::string referenceReceipt(const Json& report) { return report.dump(2, ' ', false, Json::error_handler_t::replace) + "\n"; }
bool strictUtf8(const std::string& text) { return thrown([&] { (void)Json(text).dump(); }).empty(); } // nlohmann strict validator

void testSha256() {
    section("SHA-256 known answers (FIPS 180-4 examples)");
    auto digest = [](const std::string& text) { cal::Sha256 h; h.bytes(text.data(), text.size()); return h.hex(); };
    check(digest("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    check(digest("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    check(digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "two-block message");
    check(digest(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "million a");
    check(cal::sha256Words({0x64636261u}) == digest("abcd"), "little-endian word commitment");
}

void testFrozenShaders() {
    section("frozen production shader bytes");
    Json verified = cal::verifyFrozenShaders(shaderRoot);
    check(verified.size() == 5, "five frozen shader files");
    for (const auto& f : verified) std::cout << "  " << f.at("path").get<std::string>() << " " << f.at("sha256").get<std::string>() << "\n";
    const auto copy = scratch / "shader-copy";
    std::filesystem::create_directories(copy / "runtime");
    for (const auto& s : cal::frozenShaders) std::filesystem::copy_file(shaderRoot / s.relative, copy / s.relative);
    check(cal::verifyFrozenShaders(copy).size() == 5, "byte-identical copy accepted");
    {
        std::fstream f(copy / "runtime/vision_attention_softmax.hlsl", std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(100); f.put('#');
    }
    check(contains(thrown([&] { cal::verifyFrozenShaders(copy); }), "differ from frozen"), "single changed byte refused");
    std::filesystem::remove(copy / "runtime/vision_common.hlsl");
    std::filesystem::copy_file(shaderRoot / "runtime/vision_attention_softmax.hlsl", copy / "runtime/vision_attention_softmax.hlsl",
                               std::filesystem::copy_options::overwrite_existing);
    check(contains(thrown([&] { cal::verifyFrozenShaders(copy); }), "absent or resized"), "missing include refused");
}

void testGeometryAndForecast() {
    section("phase geometry, production admission and recorded CLI forecast");
    using chandra::dc::VisionModel;
    const auto cli = VisionModel::forecast(12096, {{1, 126, 96}});
    check(cli.patchRows == 12096 && cli.mergedRows == 3024 && cli.frames == 1 && cli.maximumFrameRows == 12096, "CLI rows");
    check(cli.maximumSingleBufferBytes == 49545216 && cli.workspaceUpperBoundBytes == 291337280, "CLI buffer/workspace forecast");
    check(cli.attentionDispatchesPerBlock == 1512 && cli.attentionMultiplyAddsPerBlock == 299649466368ull, "CLI attention forecast");
    check(cli.attentionScoreBytes == 24772608 && cli.attentionPartialBytes == 12451840, "CLI per-tile scratch forecast");
    const auto tiny = VisionModel::forecast(832, {{1, 32, 26}});
    const Json plan = cal::planJson();
    const auto& blocks = plan.at("production_block_counts");
    check(blocks.at("cli_grid_1x126x96").at("attention_multiply_adds") == cli.attentionMultiplyAddsPerBlock &&
          blocks.at("cli_grid_1x126x96").at("dispatches") == cli.attentionDispatchesPerBlock, "plan block counts equal production CLI forecast");
    check(blocks.at("tiny_grid_1x32x26").at("attention_multiply_adds") == tiny.attentionMultiplyAddsPerBlock &&
          blocks.at("tiny_grid_1x32x26").at("dispatches") == tiny.attentionDispatchesPerBlock, "plan block counts equal production tiny forecast");
    struct Want { uint32_t n, start, length, query, queries, tiles, last; };
    const Want wants[] = {{224, 24, 200, 192, 8, 2, 72}, {832, 0, 832, 416, 32, 7, 64}, {832, 0, 832, 800, 32, 7, 64}, {12096, 0, 12096, 12064, 32, 95, 64}};
    const auto specs = cal::phases();
    check(specs.size() == 4, "four phases");
    for (size_t i = 0; i < specs.size(); ++i) {
        const cal::Tile t = cal::tile(specs[i]); const Want& w = wants[i];
        check(t.patches == w.n && t.sequenceStart == w.start && t.sequenceLength == w.length && t.queryStart == w.query &&
              t.queries == w.queries && t.keyTiles == w.tiles && t.lastTileKeys == w.last, std::string("tile geometry ") + specs[i].name);
        std::vector<chandra::dc::VisionGrid> grids;
        for (const auto& g : specs[i].grids) grids.push_back({g.temporal, g.height, g.width});
        const auto f = VisionModel::forecast(t.patches, grids);
        const cal::Footprint fp = cal::footprint(t);
        check(f.attentionScoreBytes >= fp.scores && f.attentionPartialBytes >= fp.partials && f.hiddenBufferBytes == fp.q,
              "calibration scratch within the production forecast");
        check(fp.deviceWithStaging <= cal::deviceAllocationCap && fp.hostPeak <= cal::hostVectorCap, "phase within caps");
        std::cout << "  " << specs[i].name << ": device+staging " << fp.deviceWithStaging << " B, host peak " << fp.hostPeak << " B\n";
    }
    check(cal::tile(specs[1]).sequenceLength == tiny.maximumFrameRows, "tiny frame length");
    auto refused = [](cal::PhaseSpec spec) { return !thrown([&] { cal::tile(spec); }).empty(); };
    check(refused({"odd", {{1, 5, 4}}, 0, 0, cal::Fixture::uniform}), "odd grid refused");
    check(refused({"offset", {{1, 32, 26}}, 0, 16, cal::Fixture::uniform}), "non-tile query start refused");
    check(refused({"beyond", {{1, 32, 26}}, 0, 832, cal::Fixture::uniform}), "query start beyond frame refused");
    check(refused({"frame", {{1, 32, 26}}, 1, 0, cal::Fixture::uniform}), "absent frame refused");
    check(refused({"large", {{1, 128, 128}, {1, 128, 128}, {1, 2, 2}}, 0, 0, cal::Fixture::uniform}), "over 32768 patches refused");
    check(refused({"zero", {{0, 2, 2}}, 0, 0, cal::Fixture::uniform}), "zero temporal refused");
    check(!refused({"video", {{2, 2, 4}, {1, 10, 20}}, 2, 192, cal::Fixture::nonuniform}), "temporal frames expand like production");
    check(cal::tile({"video", {{2, 2, 4}, {1, 10, 20}}, 2, 192, cal::Fixture::nonuniform}).sequenceStart == 16, "expanded frame offset");
}

void testGenerators() {
    section("public generators: BF16-exact inputs and the uniform-score identity");
    for (const auto& spec : cal::phases()) {
        const cal::Tile t = cal::tile(spec); int minimum = 1 << 30, maximum = -(1 << 30); bool keysOk = true, queriesOk = true, identity = true;
        for (uint32_t key = 0; key < t.sequenceLength; ++key)
            for (uint32_t col = 0; col < cal::hidden; ++col) {
                const int k = cal::keyInteger(spec.fixture, t.sequenceStart + key, col);
                const float v = cal::valueV(t.sequenceStart + key, col);
                keysOk = keysOk && k >= -8 && k <= 8 && cal::bf16Exact(float(k) / 4.0f) &&
                         v != 0.0f && cal::bf16Exact(v) && std::fabs(v) >= 0x1p-8f && std::fabs(v) <= 1.875f;
            }
        for (uint32_t qr = 0; qr < t.queries; ++qr)
            for (uint32_t h = 0; h < cal::heads; ++h) {
                const uint32_t row = t.firstQueryRow() + qr;
                for (uint32_t c = 0; c < cal::headDim; ++c) {
                    const int q = cal::queryInteger(spec.fixture, row, h * cal::headDim + c);
                    queriesOk = queriesOk && q >= -8 && q <= 8 && cal::bf16Exact(float(q) / 4.0f);
                }
                if (spec.fixture != cal::Fixture::uniform) continue;
                const int score = cal::uniformScoreInteger(row, h);
                identity = identity && score != 0;
                minimum = std::min(minimum, score); maximum = std::max(maximum, score);
                for (uint32_t key = 0; key < t.sequenceLength; key += 61) { // checkScores re-proves every key in each run.
                    int dot = 0;
                    for (uint32_t c = 0; c < cal::headDim; ++c)
                        dot += cal::queryInteger(spec.fixture, row, h * cal::headDim + c) * cal::keyInteger(spec.fixture, t.sequenceStart + key, h * cal::headDim + c);
                    identity = identity && dot == score;
                }
            }
        check(keysOk, std::string("every frame K/V value in range, nonzero V and BF16-exact for ") + spec.name);
        check(queriesOk, std::string("every tile Q value in range and BF16-exact for ") + spec.name);
        check(identity, std::string("uniform scores nonzero and constant over sampled keys for ") + spec.name);
        if (spec.fixture == cal::Fixture::uniform) {
            std::cout << "  " << spec.name << ": uniform score integers in [" << minimum << ", " << maximum << "] / 128\n";
            check(minimum != maximum, "uniform constants differ between query rows/heads");
        }
    }
    uint64_t raised = 0;
    for (uint32_t row = 0; row < 12096; ++row) for (uint32_t h = 0; h < 16; ++h) raised += cal::uniformRaised(row, h) ? 1u : 0u;
    std::cout << "  uniform rows whose raw pair sum was zero and were raised: " << raised << " of " << 12096 * 16 << "\n";
}

void testScreens() {
    section("derived softmax/output screens: magnitudes and sensitivity");
    for (uint32_t length : {200u, 832u, 12096u}) {
        const double d0 = cal::exponentialRelativeBound(0), b0 = cal::probabilityRelativeBound(d0, d0, length);
        const double missingKey = 1.0 / double(length - 1) * double(length) - 1; // One key dropped from the denominator.
        std::cout << "  L=" << length << ": uniform probability screen " << b0 << " (" << b0 / cal::unitRoundoff
                  << " u); one dropped key shifts probability by " << missingKey << "\n";
        check(b0 > 0 && b0 < 1e-5 && missingKey > 8 * b0, "uniform screen separates a one-key normalization error");
    }
    check(cal::exponentialRelativeBound(0) == cal::exp2RelativeError, "zero argument costs only exp2 error");
    const double d12 = cal::exponentialRelativeBound(12.0);
    check(d12 > cal::exp2RelativeError && d12 < cal::exp2RelativeError + 30 * cal::unitRoundoff, "argument-rounding term at |x|=12");
    check(cal::softmaxSumRoundings(12096) == 101 && cal::softmaxSumRoundings(832) == 13 && cal::softmaxSumRoundings(200) == 8, "sum rounding counts");
    check(cal::outputRoundings(12096) == 223, "output accumulation rounding count");
    check(cal::bf16Bits(0x3f808000u) == 0x3f800000u && cal::bf16Bits(0x3f818000u) == 0x3f820000u && cal::bf16Bits(0x7f800001u) == 0x7fc00000u &&
          cal::bf16Bits(0x80000000u) == 0x80000000u && cal::bf16Bits(0x7f7fffffu) == 0x7f800000u, "vision_bf16 ties, NaN, signed zero, overflow");
}

void testArguments() {
    section("argument and pin admission (no device)");
    auto parse = [](std::vector<std::wstring> args) { return cal::parseArguments(args); };
    check(!parse({}).execute, "no arguments is inactive");
    check(!parse({L"--pci", L"03:00.0"}).execute, "options without --execute stay inactive");
    const std::vector<std::wstring> full = {L"--execute", L"--shader-root", L"/s", L"--pci", L"03:00.0", L"--luid", L"00000000:0000abcd", L"--output", L"/o"};
    check(parse(full).execute && parse(full).values.size() == 4, "complete execute arguments");
    check(contains(thrown([&] { parse({L"--execute", L"--execute"}); }), "Duplicate execute"), "duplicate execute refused");
    check(contains(thrown([&] { parse({L"--pci", L"a", L"--pci", L"b"}); }), "Duplicate calibration option"), "duplicate option refused");
    check(contains(thrown([&] { parse({L"--model", L"x"}); }), "Unknown calibration option"), "unknown option refused");
    check(contains(thrown([&] { parse({L"-execute"}); }), "Unknown calibration option"), "near-miss flag refused");
    check(contains(thrown([&] { parse({L"--output"}); }), "Option value absent"), "missing value refused");
    check(contains(thrown([&] { parse({L"--execute", L"--shader-root", L"/s"}); }), "Explicit shader root"), "incomplete execute refused");
    auto blank = full; blank[6] = L"";
    check(contains(thrown([&] { parse(blank); }), "Option value absent"), "empty LUID refused");
    // Review VDC-03: a value position never consumes an option token, an empty value or nothing.
    const std::vector<std::vector<std::wstring>> swallowed = {
        {L"--pci", L"--execute"}, {L"--pci", L"--unknown"}, {L"--output", L"--pci", L"03:00.0"}, {L"--luid", L"-"}, {L"--pci", L""},
        {L"--shader-root", L"-s"}, {L"--pci", L"--pci", L"03:00.0"}, {L"--pci", L"03:00.0", L"--luid"},
        {L"--execute", L"--shader-root", L"--pci", L"03:00.0", L"--luid", L"00000000:0000abcd", L"--output", L"/o"},
        {L"--execute", L"--shader-root", L"/s", L"--pci", L"03:00.0", L"--luid", L"00000000:0000abcd", L"--output", L"--execute"}};
    for (const auto& args : swallowed)
        check(contains(thrown([&] { parse(args); }), "Option value absent or option-shaped"), "option token, empty or missing value refused");
    check(contains(thrown([&] { parse({L"--pci", L"--execute"}); }), "after --pci"), "refusal names the option");
    for (const auto& args : std::vector<std::vector<std::wstring>>{{L"--unknown"}, {L"--execute", L"--unknown"}, {L"--pci=03:00.0"}, {L"--EXECUTE"},
                                                                   {L"--execute=1"}, {L"--"}, {L"03:00.0"}, {L"--pci", L"03:00.0", L"extra"}})
        check(contains(thrown([&] { parse(args); }), "Unknown calibration option"), "unknown option or stray token refused");
    check(contains(thrown([&] { parse({L"--pci", L"03:00.0", L"--execute", L"--execute"}); }), "Duplicate execute"), "duplicate execute after options refused");
    check(contains(thrown([&] { parse({L"--luid", L"00000000:0000abcd", L"--luid", L"00000000:0000abcd"}); }), "Duplicate calibration option --luid"),
          "duplicate identical option refused");
    // Syntax is checked without --execute too, so the inactive plan needs genuinely valid arguments.
    check(contains(thrown([&] { parse({L"--pci", L"3:00.0"}); }), "PCI pin"), "malformed PCI refused without execute");
    check(contains(thrown([&] { parse({L"--luid", L"00000000:0000ABCD"}); }), "LUID pin"), "malformed LUID refused without execute");
    check(contains(thrown([&] { parse({L"--shader-root", L"relative"}); }), "Absolute shader root and output"), "relative root refused without execute");
    check(contains(thrown([&] { parse({L"--output", L"relative.json"}); }), "Absolute shader root and output"), "relative output refused without execute");
    const auto inactive = parse({L"--shader-root", L"/s", L"--pci", L"03:00.0", L"--luid", L"00000000:0000abcd", L"--output", L"/o"});
    check(!inactive.execute && inactive.values.size() == 4, "complete valid options without --execute stay inactive");
    const auto reordered = parse({L"--output", L"/o", L"--luid", L"00000000:0000abcd", L"--execute", L"--pci", L"03:00.0", L"--shader-root", L"/s"});
    check(reordered.execute && reordered.values.at(L"--pci") == L"03:00.0" && reordered.values.at(L"--output") == L"/o", "execute admitted in any position");
    check(contains(thrown([&] { parse({L"--execute"}); }), "Explicit shader root"), "bare execute refused");
    check(cal::pinnedPci(L"03:00.0") == "03:00.0" && cal::pinnedLuid(L"00000000:0000abcd") == "00000000:0000abcd", "canonical pins");
    for (const wchar_t* bad : {L"3:00.0", L"03:00.8", L"0A:00.0", L"03:00.0 ", L"03-00.0", L"", L"03:00.é"})
        check(!thrown([&] { cal::pinnedPci(bad); }).empty(), "malformed PCI refused");
    for (const wchar_t* bad : {L"00000000:0000ABCD", L"0000000:0000abcd", L"00000000-0000abcd", L"00000000:0000abcd0"})
        check(!thrown([&] { cal::pinnedLuid(bad); }).empty(), "malformed LUID refused");
}

// Production tile transcripts: role-normalized shader, cbuffer words, groups, binding roles and sizes.
std::string normalizedQuad(const std::vector<Event>& quad) {
    std::map<uint64_t, std::string> roles = {{quad[0].inputs.at(0), "q"}, {quad[0].inputs.at(1), "k"}, {quad[0].outputs.at(0), "scores"},
                                             {quad[2].inputs.at(1), "v"}, {quad[2].outputs.at(0), "partials"}, {quad[3].outputs.at(0), "result"}};
    check(roles.size() == 6, "six distinct attention buffers");
    std::ostringstream text;
    for (const auto& d : quad) {
        text << d.shader << " params";
        for (uint32_t w : d.params) text << " " << w;
        text << " groups " << d.groups[0] << " " << d.groups[1] << " " << d.groups[2] << " in";
        for (size_t i = 0; i < d.inputs.size(); ++i) text << " " << roles.at(d.inputs[i]) << ":" << d.inputWords[i];
        text << " out";
        for (size_t i = 0; i < d.outputs.size(); ++i) text << " " << roles.at(d.outputs[i]) << ":" << d.outputWords[i];
        text << "\n";
    }
    return text.str();
}
struct ProductionTiles { std::map<std::string, std::string> byTile; std::map<uint32_t, uint32_t> tilesPerFrame; uint64_t attentionDispatches = 0; };
std::string tileKey(uint32_t start, uint32_t length, uint32_t query) { return std::to_string(start) + "/" + std::to_string(length) + "/" + std::to_string(query); }
ProductionTiles productionTranscript(const std::vector<cal::Grid>& grids) {
    resetFake(false);
    uint32_t n = 0; std::vector<chandra::dc::VisionGrid> visionGrids;
    for (const auto& g : grids) { n += g.temporal * g.height * g.width; visionGrids.push_back({g.temporal, g.height, g.width}); }
    {
        Device device(L"/fake", "00:00.0", ""); chandra::dc::ModelWeights weights(device, L"/fake");
        std::vector<float> patches(size_t(n) * 1536, 0.25f);
        chandra::dc::VisionModel model(device, weights);
        (void)model.firstBlock(patches.data(), n, visionGrids); // Unchanged production graph, block 0.
    }
    const auto& events = vdc_fake::events(); ProductionTiles result; const Event* rope = nullptr;
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind != "dispatch") continue;
        if (events[i].shader == "runtime/vision_rope.hlsl") rope = &events[i];
        if (events[i].shader != cal::stageShaders[0]) continue;
        check(rope != nullptr && i >= 2 && i + 4 < events.size(), "attention follows RoPE and has a complete tile");
        std::vector<Event> quad(events.begin() + long(i), events.begin() + long(i) + 4);
        for (size_t s = 0; s < 4; ++s) check(quad[s].kind == "dispatch" && quad[s].shader == cal::stageShaders[s], "production stage order");
        check(events[i + 4].kind == "drain", "production drains after each tile");
        check(events[i - 2].kind == "alloc" && events[i - 2].id == quad[0].outputs[0] && events[i - 1].kind == "alloc" &&
              events[i - 1].id == quad[2].outputs[0], "scores and partials are fresh per-tile allocations");
        check(quad[0].inputs[0] == rope->outputs.at(0) && quad[0].inputs[1] == rope->outputs.at(1) && quad[2].inputs[1] == rope->outputs.at(2),
              "attention reads RoPE Q, K and V");
        const uint32_t start = quad[0].params.at(0), length = quad[0].params.at(1), query = quad[0].params.at(2);
        result.byTile[tileKey(start, length, query)] = normalizedQuad(quad);
        ++result.tilesPerFrame[start]; result.attentionDispatches += 4; i += 3;
    }
    return result;
}
std::vector<Event> calibrationQuad(const cal::PhaseSpec& spec) {
    resetFake(false);
    const cal::Tile t = cal::tile(spec);
    {
        Device device(L"/fake", "00:00.0", ""); const uint32_t global = t.patches * cal::hidden;
        cal::TileBuffers b{device.floats(global), device.floats(global), device.floats(global), device.floats(global),
                           device.floats(t.queryHeads * t.sequenceLength), device.floats(t.queries * cal::hidden * t.keyTiles)};
        for (size_t s = 0; s < 4; ++s) cal::dispatchStage(device, t, b, s);
    }
    return dispatches();
}
void testProductionEquivalence() {
    section("calibration tiles equal unchanged VisionModel::attention transcripts");
    std::map<std::string, ProductionTiles> cache;
    for (const auto& spec : cal::phases()) {
        std::string gridKey;
        for (const auto& g : spec.grids) gridKey += std::to_string(g.temporal) + "x" + std::to_string(g.height) + "x" + std::to_string(g.width) + ";";
        if (!cache.count(gridKey)) cache[gridKey] = productionTranscript(spec.grids);
        const auto& production = cache.at(gridKey);
        const cal::Tile t = cal::tile(spec);
        const auto found = production.byTile.find(tileKey(t.sequenceStart, t.sequenceLength, t.queryStart));
        check(found != production.byTile.end(), std::string("production issues the selected tile for ") + spec.name);
        const std::string mine = normalizedQuad(calibrationQuad(spec));
        check(mine == found->second, std::string("byte-equal dispatch arguments, groups, bindings and sizes for ") + spec.name);
        check(production.tilesPerFrame.at(t.sequenceStart) == t.frameQueryTiles, "production tile count per frame");
        std::cout << "  " << spec.name << ": equal to production tile " << found->first << " (" << production.byTile.size()
                  << " production tiles, " << production.attentionDispatches << " attention dispatches in block 0)\n";
        if (t.sequenceLength == 12096) check(production.attentionDispatches == 1512, "CLI block issues 1512 attention dispatches");
    }
    std::cout << "  production CLI tile transcript:\n" << cache.at("1x126x96;").byTile.at(tileKey(0, 12096, 12064));
}

// Direct checks of the reusable oracle helpers on the tail tile, including corrupted observations.
void testHelpersDirectly() {
    section("oracle helpers on real fake readbacks and corrupted copies");
    resetFake(true);
    const cal::PhaseSpec spec = cal::phases().front(); const cal::Tile t = cal::tile(spec); cal::Deadline deadline;
    Device device(L"/fake", "00:00.0", ""); Json commitments = Json::object();
    cal::TileBuffers b = cal::allocate(device, spec, t, commitments, deadline);
    const auto keys = cal::keyTable(spec, t, deadline);
    cal::dispatchStage(device, t, b, 0); auto scores = device.readWords(b.scores);
    cal::dispatchStage(device, t, b, 1); auto probabilities = device.readWords(b.scores);
    cal::dispatchStage(device, t, b, 2); auto partials = device.readWords(b.partials);
    cal::dispatchStage(device, t, b, 3); auto result = device.readWords(b.result);
    check(cal::checkScores(spec, t, keys, scores, deadline).at("passed") == true, "scores pass");
    check(cal::checkProbabilities(spec, t, scores, probabilities, deadline).at("passed") == true, "probabilities pass");
    check(cal::checkPartials(t, probabilities, partials, deadline).at("passed") == true, "partials pass");
    check(cal::checkResult(t, partials, result, deadline).at("passed") == true, "result pass");
    Json screen = cal::checkOutputScreen(t, scores, result, deadline);
    check(screen.at("passed") == true, "output screen pass");
    auto flipped = scores; flipped[1234] ^= 1u;
    Json bad = cal::checkScores(spec, t, keys, flipped, deadline);
    check(bad.at("mismatches") == 1 && bad.at("first_mismatches").at(0).at("index") == 1234, "one-bit score flip located");
    check(bad.at("expected_sha256") != bad.at("observed_sha256"), "commitments differ on mismatch");
    flipped = partials; flipped.back() ^= 0x80000000u;
    check(cal::checkPartials(t, probabilities, flipped, deadline).at("mismatches") == 1, "partial sign flip located");
    flipped = result; flipped.front() = 0;
    check(cal::checkResult(t, partials, flipped, deadline).at("mismatches") == 1, "write outside the tile located");
    // Probability screen sensitivity: move one probability by 0.5x and 2x its own derived screen.
    cal::HeadOracle oracle; cal::softmaxOracle(t, 3, scores, oracle);
    const size_t key = 77, index = size_t(5 * cal::heads + 3) * t.sequenceLength + key, o = size_t(5) * t.sequenceLength + key;
    for (double factor : {0.5, 2.0}) {
        auto moved = probabilities;
        moved[index] = cal::bits(float(oracle.p[o] * (1 + factor * oracle.bound[o])));
        Json r = cal::checkProbabilities(spec, t, scores, moved, deadline);
        check((r.at("violations") == 0) == (factor < 1), "screen admits half and refuses twice its bound");
    }
    std::cout << "  tail screens: probability max ratio " << cal::checkProbabilities(spec, t, scores, probabilities, deadline).at("maximum_error_to_screen_ratio")
              << ", output max ratio " << screen.at("maximum_error_to_screen_ratio") << "\n";
    b = {}; device.drain(); check(device.trackedBufferBytes() == 0, "helper buffers released");
}

Json runPlan(const std::function<void(vdc_fake::Config&)>& setup, std::string& error, const std::vector<cal::PhaseSpec>& specs,
             uint64_t& tracked, cal::Deadline deadline = {}) {
    resetFake(true); setup(vdc_fake::config());
    Json report = cal::initialReport(); error.clear();
    try {
        Device device(L"/fake", "03:00.0", "00000000:0000abcd");
        try { cal::run(device, report, deadline, [&] { check(cal::receiptText(report).size() <= cal::receiptCap, "bounded progress receipt"); }, specs); }
        catch (...) { device.drain(); tracked = device.trackedBufferBytes(); throw; }
        tracked = device.trackedBufferBytes();
    } catch (const std::exception& e) { error = e.what(); }
    return report;
}

void testFullRun(bool quick) {
    section(quick ? "calibration plan on the executing fake (quick: phases 1-3)" : "complete calibration plan on the executing fake");
    auto specs = cal::phases(); if (quick) specs.pop_back();
    std::string error; uint64_t tracked = 1; const auto start = std::chrono::steady_clock::now();
    Json report = runPlan([](vdc_fake::Config&) {}, error, specs, tracked);
    check(error.empty() && report.at("passed") == true, "plan passes: " + error);
    check(tracked == 0, "all buffers released");
    check(report.at("dispatches_submitted") == 4 * specs.size() && report.at("dispatches_completed_and_measured") == 4 * specs.size(), "dispatch counters");
    check(vdc_fake::flushes() == 0, "no subnormal arithmetic in the transliterated shaders");
    // Each stage is alone in its profile window: drain, dispatch, drain.
    const auto& events = vdc_fake::events(); size_t seen = 0;
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind != "dispatch") continue;
        ++seen; check(i > 0 && i + 1 < events.size() && events[i - 1].kind == "drain" && events[i + 1].kind == "drain", "individual stage window");
    }
    check(seen == 4 * specs.size(), "no extra dispatch");
    const auto all = dispatches();
    for (size_t p = 0; p < specs.size(); ++p) {
        const std::vector<Event> quad(all.begin() + long(4 * p), all.begin() + long(4 * p) + 4);
        check(normalizedQuad(quad) == normalizedQuad(calibrationQuad(specs[p])), "executed tile equals the production-equal transcript");
        const auto& phase = report.at("phases").at(p);
        check(phase.at("passed") == true && phase.at("tracked_after_phase_release") == 0, "phase passed and released");
        const auto& stages = phase.at("stages");
        for (size_t s = 0; s < 4; ++s) check(stages.at(s).at("check").at("passed") == true && stages.at(s).at("completed") == true, "stage checks");
        const auto& probability = stages.at(1).at("check");
        std::cout << "  " << phase.at("name").get<std::string>() << ": probability screen max ratio "
                  << probability.at("maximum_error_to_screen_ratio") << " (screen " << probability.at("maximum_relative_screen") << ")"
                  << ", output screen max ratio " << phase.at("output_screen").at("maximum_error_to_screen_ratio")
                  << ", oracle " << phase.at("oracle_seconds") << " s";
        if (probability.contains("uniform")) std::cout << ", uniform ulp distance " << probability.at("uniform").at("ulp_distance_from_correctly_rounded_reciprocal");
        std::cout << "\n";
    }
    check(cal::boundedReceipt(report).complete && cal::receiptText(report) == referenceReceipt(report), "complete passing receipt unchanged");
    if (!quick) {
        check(report.contains("cli_attention_projection") && report.at("cli_attention_projection").at("query_tiles_per_block") == 378, "CLI projection");
        const auto path = scratch / "fake-full-report.json";
        std::ofstream(path, std::ios::binary) << cal::receiptText(report);
        std::cout << "  report " << std::filesystem::file_size(path) << " bytes written to " << path.string() << "\n";
    }
    std::cout << "  elapsed " << cal::millisecondsSince(start) / 1000 << " s\n";
}

void testFailures() {
    section("faults, gates and refusals stop before further dispatch");
    const auto all = cal::phases(); const std::vector<cal::PhaseSpec> first(all.begin(), all.begin() + 1), two(all.begin(), all.begin() + 2),
        three(all.begin(), all.begin() + 3);
    struct Case { const char* name; std::function<void(vdc_fake::Config&)> setup; std::vector<cal::PhaseSpec> specs; const char* error; uint64_t dispatches, measured; };
    const std::vector<Case> cases = {
        {"K row ignores frame start", [](auto& c) { c.fault = vdc_fake::Fault::scoresIgnoreSequenceStart; }, first, "Stage scores output check failed", 1, 1},
        {"softmax drops one lane from the denominator", [](auto& c) { c.fault = vdc_fake::Fault::softmaxDropLastLane; }, first, "Stage softmax output check failed", 2, 2},
        {"exp error 2^-14 on even keys", [](auto& c) { c.fault = vdc_fake::Fault::softmaxPerturbEvenKeys; c.expPerturbation = 0x1p-14; }, first, "Stage softmax output check failed", 2, 2},
        {"values skips the last key", [](auto& c) { c.fault = vdc_fake::Fault::valuesSkipLastKey; }, first, "Stage values output check failed", 3, 3},
        {"reduce skips the last tile", [](auto& c) { c.fault = vdc_fake::Fault::reduceSkipLastTile; }, first, "Stage reduce output check failed", 4, 4},
        {"reduce ignores rowFirst", [](auto& c) { c.fault = vdc_fake::Fault::reduceIgnoreRowFirst; }, first, "Stage reduce output check failed", 4, 4},
        {"100.0 ms softmax in phase 2", [](auto& c) { c.gpuMilliseconds = [](const std::string&, uint64_t id) { return id == 6 ? 100.0 : 0.25; }; }, two, "at or above the 100 ms gate", 6, 6},
        {"99.9 ms tail scores extrapolates past the gate", [](auto& c) { c.gpuMilliseconds = [](const std::string&, uint64_t id) { return id == 1 ? 99.9 : 0.25; }; }, two, "Extrapolated stage time", 4, 4},
        {"4 ms tiny scores refuses the CLI phase", [](auto& c) { c.gpuMilliseconds = [](const std::string&, uint64_t id) { return id == 9 ? 4.0 : 0.25; }; }, all, "Extrapolated stage time", 12, 12},
        {"disjoint window throws", [](auto& c) { c.profileFault = vdc_fake::ProfileFault::disjointThrows; c.profileFaultDispatch = 2; }, first, "disjoint", 2, 1},
        {"disjoint window reported", [](auto& c) { c.profileFault = vdc_fake::ProfileFault::disjointReported; c.profileFaultDispatch = 3; }, first, "Invalid or disjoint", 3, 2},
        {"missing measurement", [](auto& c) { c.profileFault = vdc_fake::ProfileFault::missing; c.profileFaultDispatch = 1; }, first, "Exactly one measured dispatch", 1, 0},
        {"measurement for another geometry", [](auto& c) { c.profileFault = vdc_fake::ProfileFault::wrongGroups; c.profileFaultDispatch = 4; }, first, "differs from the production stage", 4, 3},
        {"negative duration", [](auto& c) { c.profileFault = vdc_fake::ProfileFault::negative; c.profileFaultDispatch = 1; }, first, "Invalid completed dispatch duration", 1, 0}};
    for (const auto& c : cases) {
        std::string error; uint64_t tracked = 1;
        Json report = runPlan(c.setup, error, c.specs, tracked);
        const uint64_t issued = dispatches().size();
        check(contains(error, c.error), std::string(c.name) + ": expected error, got: " + error);
        check(issued == c.dispatches && report.at("dispatches_submitted") == c.dispatches, std::string(c.name) + ": no further dispatch");
        check(report.at("dispatches_completed_and_measured") == c.measured, std::string(c.name) + ": measured count");
        check(report.at("passed") == false && tracked == 0, std::string(c.name) + ": failed closed and released");
        check(cal::receiptText(report).size() < 1024 * 1024, std::string(c.name) + ": bounded receipt");
        check(cal::boundedReceipt(report).complete && cal::receiptText(report) == referenceReceipt(report), std::string(c.name) + ": receipt unchanged");
        std::cout << "  " << c.name << ": stopped after " << issued << " dispatches: " << error << "\n";
    }
    {
        std::string error; uint64_t tracked = 1;
        Json report = runPlan([](auto& c) { c.fault = vdc_fake::Fault::softmaxPerturbEvenKeys; c.expPerturbation = 0x1p-21; }, error, two, tracked);
        check(error.empty() && report.at("passed") == true, "exp error 2^-21 (within the assumed exp2 accuracy) passes nonuniform phases: " + error);
        std::cout << "  exp error 2^-21 on even keys: passes; probability max ratio "
                  << report.at("phases").at(1).at("stages").at(1).at("check").at("maximum_error_to_screen_ratio") << "\n";
    }
    {
        std::string error; uint64_t tracked = 1; cal::Deadline expired; expired.start -= std::chrono::seconds(cal::deadlineSeconds + 1);
        Json report = runPlan([](auto&) {}, error, all, tracked, expired);
        uint64_t allocations = 0;
        for (const auto& e : vdc_fake::events()) allocations += e.kind == "alloc" ? 1u : 0u;
        check(contains(error, "deadline exceeded") && dispatches().empty() && allocations == 0 && report.at("phases").empty(),
              "expired deadline refuses before allocation");
    }
    {
        Json big = {{"state", "X"}, {"error", std::string("\xff\xfe invalid UTF-8")}, {"blob", std::string(cal::receiptCap, 'x')}};
        const std::string text = cal::receiptText(big);
        check(text.size() < 4096 && contains(text, "RECEIPT_OVERSIZE") && contains(text, "\"passed\": false"), "oversized receipt becomes a failing summary");
    }
}
// Review VDC-01: the actual Deadline expires during receipt persistence or inside beginProfile.
struct HookedRun {
    Json report; std::string error;
    uint64_t dispatches = 0, profileWindows = 0, trackedAfterFailureDrain = 1;
    size_t eventsAtExpiry = 0, eventsAtRefusal = 0, eventsInFailureDrain = 0; bool injected = false;
};
void expire(cal::Deadline& deadline) { deadline.start = std::chrono::steady_clock::now() - std::chrono::seconds(cal::deadlineSeconds + 1); }
HookedRun runHooked(const std::vector<cal::PhaseSpec>& specs, const std::function<bool(const Json&, cal::Deadline&)>& onPersist,
                    const std::function<bool(uint64_t, cal::Deadline&)>& onProfile) {
    resetFake(true); HookedRun r; r.report = cal::initialReport(); cal::Deadline deadline;
    auto mark = [&](bool injected) { if (injected && !r.injected) { r.injected = true; r.eventsAtExpiry = vdc_fake::events().size(); } };
    if (onProfile) vdc_fake::config().beginProfileHook = [&](uint64_t window) { if (!r.injected) mark(onProfile(window, deadline)); };
    {
        Device device(L"/fake", "03:00.0", "00000000:0000abcd");
        try {
            cal::run(device, r.report, deadline, [&] {
                check(cal::receiptText(r.report).size() <= cal::receiptCap, "bounded progress receipt");
                if (onPersist && !r.injected) mark(onPersist(r.report, deadline));
            }, specs);
        } catch (const std::exception& e) { r.error = e.what(); }
        r.eventsAtRefusal = vdc_fake::events().size();
        device.drain(); // The caller's single bounded failure drain.
        r.eventsInFailureDrain = vdc_fake::events().size() - r.eventsAtRefusal;
        r.trackedAfterFailureDrain = device.trackedBufferBytes();
    }
    vdc_fake::config().beginProfileHook = nullptr;
    r.dispatches = dispatches().size(); r.profileWindows = vdc_fake::profileBegins();
    return r;
}
// Common refusal properties: the refused stage and everything after it never started, nothing reached
// the Device after the expiry, and one bounded drain retired every buffer.
void checkRefusal(const HookedRun& r, const std::string& name, size_t phase, size_t stage, const char* boundary, uint64_t dispatched) {
    check(r.injected && contains(r.error, std::string("120-second source deadline exceeded ") + boundary + "; stage " + cal::stageNames[stage] +
                                              " not dispatched"), name + ": deadline refusal, got: " + r.error);
    check(r.dispatches == dispatched && r.report.at("dispatches_submitted") == dispatched &&
          r.report.at("dispatches_completed_and_measured") == dispatched, name + ": no extra dispatch");
    check(r.eventsAtRefusal == r.eventsAtExpiry, name + ": no drain, allocation, dispatch or readback after the expiry");
    check(r.report.at("phases").size() == phase + 1 && r.report.at("passed") == false, name + ": no further phase");
    const auto& stages = r.report.at("phases").at(phase).at("stages");
    const auto& refused = stages.at(stage);
    check(refused.at("dispatch_call_started") == false && refused.at("submitted") == false && refused.at("completed") == false &&
          refused.at("refused_before_dispatch") == std::string("deadline_expired_") + boundary, name + ": refused stage recorded");
    for (size_t s = stage + 1; s < 4; ++s)
        check(stages.at(s).at("dispatch_call_started") == false && !stages.at(s).contains("refused_before_dispatch"), name + ": no further stage");
    check(r.eventsInFailureDrain == 1 && r.trackedAfterFailureDrain == 0, name + ": one bounded failure drain releases every buffer");
    check(cal::boundedReceipt(r.report).complete && cal::receiptText(r.report) == referenceReceipt(r.report), name + ": complete unchanged receipt");
    std::cout << "  " << name << ": " << r.dispatches << " dispatches, " << r.profileWindows << " profile windows: " << r.error << "\n";
}
void testDeadlineBoundaries() {
    section("source deadline enforced after receipt persistence and profile preparation, before every dispatch");
    const auto all = cal::phases(); const std::vector<cal::PhaseSpec> first(all.begin(), all.begin() + 1), two(all.begin(), all.begin() + 2);
    {   // Review counterexample 1: expiry in the receipt persisted immediately before the first dispatch.
        const HookedRun r = runHooked(first, [](const Json& report, cal::Deadline& d) {
            if (report.at("phases").empty() || report.at("phases").back().at("stages").at(0).at("dispatch_call_started") != true) return false;
            expire(d); return true;
        }, nullptr);
        checkRefusal(r, "expiry in pre-dispatch receipt persistence", 0, 0, "after_receipt_persistence", 0);
        check(r.profileWindows == 0, "no profile window opened after expiry");
    }
    {   // Review counterexample 2: expiry in the receipt persisted after the completed scores check.
        const HookedRun r = runHooked(first, [](const Json& report, cal::Deadline& d) {
            if (report.at("phases").empty() || !report.at("phases").back().at("stages").at(0).contains("check")) return false;
            expire(d); return true;
        }, nullptr);
        checkRefusal(r, "expiry after the scores check", 0, 1, "after_receipt_persistence", 1);
        const auto& scores = r.report.at("phases").at(0).at("stages").at(0);
        check(r.profileWindows == 1 && scores.at("completed") == true && scores.at("check").at("passed") == true, "scores completed; softmax never submitted");
    }
    // Actual profile-preparation expiry: inside beginProfile (after its drain) 500 ms of the real budget
    // remain and the deadline is still open; 600 ms of real elapsed time then cross it before the dispatch.
    for (uint64_t window : {uint64_t(1), uint64_t(3), uint64_t(6)}) {
        bool openBefore = false, expiredAfter = false;
        const HookedRun r = runHooked(two, nullptr, [&](uint64_t w, cal::Deadline& d) {
            if (w != window) return false;
            d.start = std::chrono::steady_clock::now() - std::chrono::seconds(cal::deadlineSeconds) + std::chrono::milliseconds(500);
            openBefore = !d.expired();
            std::this_thread::sleep_for(std::chrono::milliseconds(600));
            expiredAfter = d.expired(); return true;
        });
        check(openBefore && expiredAfter, "deadline open on entry to profile preparation and expired by real time inside it");
        checkRefusal(r, "real-time expiry inside profile window " + std::to_string(window), size_t(window - 1) / 4, size_t(window - 1) % 4,
                     "after_profile_preparation", window - 1);
        check(r.profileWindows == window, "the refused stage's profile window was the last");
    }
    {   // Expiry in the receipt persisted after a completed phase: the next phase is refused before allocation.
        const HookedRun r = runHooked(two, [](const Json& report, cal::Deadline& d) {
            if (report.at("phases").size() != 1 || report.at("phases").back().at("passed") != true) return false;
            expire(d); return true;
        }, nullptr);
        check(r.injected && contains(r.error, "120-second source deadline exceeded") && r.dispatches == 4 && r.report.at("phases").size() == 1 &&
              r.eventsAtRefusal == r.eventsAtExpiry && r.trackedAfterFailureDrain == 0, "expiry between phases refuses the next phase before allocation");
        std::cout << "  expiry after a completed phase: " << r.dispatches << " dispatches, next phase not allocated\n";
    }
}

// Review VDC-02: every receipt is bounded, including giant errors, multibyte text and nested metadata.
Json requireOversize(const Json& report, const std::string& name) {
    const cal::BoundedReceipt b = cal::boundedReceipt(report);
    check(!b.complete && b.text.size() <= cal::receiptCap && b.text.size() < 64 * 1024, name + ": bounded fallback");
    check(strictUtf8(b.text), name + ": fallback is valid UTF-8");
    const Json summary = Json::parse(b.text);
    check(summary.at("state") == "RECEIPT_OVERSIZE" && summary.at("passed") == false && summary.at("retirement_required") == true &&
          summary.at("detail_lost") == true && summary.at("full_receipt_bytes") == b.fullBytes && b.fullBytes > cal::receiptCap &&
          contains(summary.at("error").get<std::string>(), "retire the externally owned Job"), name + ": failing summary with lost-detail flag");
    std::cout << "  " << name << ": " << b.fullBytes << " bytes -> " << b.text.size() << "-byte summary\n";
    return summary;
}
void testReceiptBounds() {
    section("bounded receipts: giant error, multibyte and escaping extremes, nested metadata, unchanged normal receipts");
    const std::string giant(size_t(cal::receiptCap) + 1, 'x');
    {   // The exact review reproduction: initialReport with a receiptCap+1-character error.
        Json report = cal::initialReport(); report["error"] = giant;
        const uint64_t full = referenceReceipt(report).size();
        const Json s = requireOversize(report, "review VDC-02 receiptCap+1 error");
        const auto& e = s.at("original_error");
        check(s.at("full_receipt_bytes") == full, "exact full size reported");
        check(e.at("bytes") == cal::receiptCap + 1 && e.at("sha256") == cal::sha256Bytes(giant.data(), giant.size()) &&
              e.at("excerpt") == std::string(cal::fallbackExcerptBytes, 'x') && e.at("excerpt_truncated") == true, "original error length, hash and bounded excerpt");
        check(s.at("original_state").at("excerpt") == "PREPARING" && s.at("dispatches_submitted") == 0 && s.at("native_executed") == false,
              "original state and counters retained");
        check(s.at("largest_top_level_fields").at(0).at("key") == "error", "oversized field identified");
    }
    {   // Exactly at the cap the receipt is kept byte for byte; one byte more becomes the summary.
        Json report = {{"state", "PREPARING"}, {"blob", ""}};
        const size_t base = referenceReceipt(report).size();
        report["blob"] = std::string(size_t(cal::receiptCap) - base, 'b');
        const cal::BoundedReceipt at = cal::boundedReceipt(report);
        check(at.complete && at.text.size() == cal::receiptCap && at.text == referenceReceipt(report), "receipt of exactly receiptCap bytes unchanged");
        report["blob"] = std::string(size_t(cal::receiptCap) - base + 1, 'b');
        check(requireOversize(report, "receiptCap+1-byte receipt").at("full_receipt_bytes") == cal::receiptCap + 1, "one byte over becomes the summary");
    }
    {   // Escape-aware, code-point-safe excerpts measured against the serializer itself.
        struct Unit { const char* name; std::string text; size_t bytes; bool valid; };
        const std::vector<Unit> units = {{"two-byte U+00E9", "\xc3\xa9", 2, true}, {"three-byte U+20AC", "\xe2\x82\xac", 3, true},
                                         {"four-byte U+1F600", "\xf0\x9f\x98\x80", 4, true}, {"control U+0001", "\x01", 1, true},
                                         {"quote", "\"", 1, true}, {"backslash", "\\", 1, true}, {"newline", "\n", 1, true},
                                         {"invalid 0xff", "\xff", 1, false}, {"lone continuation", "\x80", 1, false},
                                         {"overlong slash", "\xc0\xaf", 2, false}, {"surrogate", "\xed\xa0\x80", 3, false}};
        for (const auto& u : units) {
            std::string text;
            while (text.size() <= cal::receiptCap) text += u.text;
            const cal::Utf8Excerpt x = cal::utf8Excerpt(text.data(), text.size(), cal::fallbackExcerptBytes);
            check(x.escapedBytes == Json(x.text).dump().size() - 2 && x.escapedBytes <= cal::fallbackExcerptBytes &&
                  x.escapedBytes + 6 > cal::fallbackExcerptBytes && x.truncated && strictUtf8(x.text), std::string(u.name) + ": excerpt fills the escaped budget");
            if (u.valid) check(x.replaced == 0 && x.text == text.substr(0, x.text.size()) && x.text.size() % u.bytes == 0, std::string(u.name) + ": whole code points");
            else check(x.replaced > 0 && x.text.size() == 3 * x.replaced, std::string(u.name) + ": replaced by U+FFFD");
            Json report = cal::initialReport(); report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = text;
            const Json s = requireOversize(report, std::string(u.name) + " error of " + std::to_string(text.size()) + " bytes");
            check(s.at("original_error").at("excerpt") == x.text && s.at("original_error").at("bytes") == text.size() &&
                  s.at("original_state").at("excerpt") == "FAIL_OR_INCOMPLETE", std::string(u.name) + ": summary keeps the bounded excerpt");
        }
        const std::string mixed = "abc\"\\\n\x01\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80\xff\xc0\xaf\xed\xa0\x80\xf4\x90\x80\x80z";
        const cal::Utf8Excerpt m = cal::utf8Excerpt(mixed.data(), mixed.size(), 1 << 20);
        check(!m.truncated && m.replaced == 10 && strictUtf8(m.text) && m.text.compare(0, 16, mixed, 0, 16) == 0 && m.text.back() == 'z' &&
              m.escapedBytes == Json(m.text).dump().size() - 2, "mixed text: overlong, surrogate, >U+10FFFF and stray bytes replaced");
        const std::string emoji = "a\xf0\x9f\x98\x80";
        const cal::Utf8Excerpt split = cal::utf8Excerpt(emoji.data(), emoji.size(), 4);
        check(split.text == "a" && split.truncated && split.keptSourceBytes == 1, "a code point that does not fit is never split");
        check(cal::utf8Excerpt("\x01", 1, 5).text.empty(), "a six-byte escape that does not fit is omitted");
    }
    {   // Giant nested metadata with a small actionable error, and many small nested values.
        Json report = cal::initialReport(); report["state"] = "FAIL_OR_INCOMPLETE"; report["error"] = "small actionable error";
        Json stage = Json::object(), phase = Json::object();
        stage["gpu_profile"]["driver_annotations"]["note"] = std::string(cal::receiptCap, 'm');
        phase["stages"] = Json::array(); phase["stages"].push_back(stage);
        report["phases"] = Json::array(); report["phases"].push_back(phase);
        report["failure_drain"] = {{"completed", false}, {"error", std::string(cal::receiptCap / 2, 'd')}};
        const Json s = requireOversize(report, "giant nested profile and drain metadata");
        check(s.at("original_error").at("excerpt") == "small actionable error" && s.at("original_error").at("excerpt_truncated") == false,
              "small original error kept whole");
        check(s.at("largest_top_level_fields").at(0).at("key") == "phases" && s.at("largest_top_level_fields").at(1).at("key") == "failure_drain",
              "nested oversize located");
        check(s.at("failure_drain").at("completed") == false && s.at("failure_drain").at("tracked_after").is_null(), "failure drain scalars kept, strings dropped");
        Json many = {{"state", "FAIL_OR_INCOMPLETE"}, {"many", Json::array()}};
        for (uint32_t i = 0; i < 1100000; ++i) many["many"].push_back(i);
        requireOversize(many, "1.1 million nested integers");
    }
    {   // Non-string state and error, wrong-type counter, oversized passing report, giant key, non-object report.
        Json report = {{"state", {{"nested", std::string(cal::receiptCap, 's')}}}, {"error", Json::array({1, 2, 3})},
                       {"dispatches_submitted", "not a number"}, {"passed", true}, {std::string(cal::receiptCap, 'k'), 1}};
        const Json s = requireOversize(report, "non-string state/error, wrong-type counter, giant key");
        check(s.at("original_state").at("type") == "object" && !s.at("original_state").contains("excerpt") &&
              s.at("original_error").at("type") == "array" && s.at("dispatches_submitted").is_null(), "non-string fields keep type and size only");
        check(s.at("original_passed") == true && s.at("passed") == false, "an oversized passing report is never reported as passed");
        bool giantKey = false;
        for (const auto& field : s.at("largest_top_level_fields"))
            if (field.at("key_bytes") == cal::receiptCap)
                giantKey = field.at("key_truncated") == true && field.at("key").get<std::string>().size() <= cal::fallbackKeyBytes;
        check(giantKey, "giant key reduced to a bounded excerpt");
        Json pass = cal::initialReport(); pass["state"] = "VISION_DISPATCH_CALIBRATION_PASS"; pass["passed"] = true; pass["blob"] = std::string(cal::receiptCap, 'p');
        const Json p = requireOversize(pass, "oversized passing report");
        check(p.at("original_state").at("excerpt") == "VISION_DISPATCH_CALIBRATION_PASS" && p.at("passed") == false, "oversized pass is not admitted");
        const Json top = requireOversize(Json(giant), "top-level string report");
        check(top.at("original_type") == "string" && top.at("original_state").at("present") == false, "non-object report summarized by type");
        const std::string minimal = cal::minimalOversizeReceipt(std::numeric_limits<uint64_t>::max());
        const Json m = Json::parse(minimal);
        check(minimal.size() < 1024 && m.at("state") == "RECEIPT_OVERSIZE" && m.at("passed") == false && m.at("detail_lost") == true &&
              m.at("full_receipt_bytes") == std::numeric_limits<uint64_t>::max(), "last-resort summary is bounded and failing");
    }
    {   // Exception text stored in a live report.
        check(cal::boundedError("short \xff error") == "short \xff error" && cal::boundedError(nullptr) == "(null exception message)", "short error unchanged");
        const std::string exact(cal::storedErrorBytes, 'e');
        check(cal::boundedError(exact.c_str()) == exact, "error of exactly storedErrorBytes unchanged");
        const std::string stored = cal::boundedError(giant.c_str());
        check(stored.size() <= cal::storedErrorBytes + 256 && stored.compare(0, cal::storedErrorBytes, giant, 0, cal::storedErrorBytes) == 0 &&
              contains(stored, "[ERROR TRUNCATED, detail lost: 8388609 bytes, sha256 " + cal::sha256Bytes(giant.data(), giant.size())) && strictUtf8(stored),
              "giant error reduced to a bounded excerpt with length and hash");
        std::string emoji;
        while (emoji.size() <= cal::storedErrorBytes) emoji += "\xf0\x9f\x98\x80";
        const std::string storedEmoji = cal::boundedError(emoji.c_str());
        check(strictUtf8(storedEmoji) && storedEmoji.compare(0, cal::storedErrorBytes, emoji, 0, cal::storedErrorBytes) == 0, "multibyte error truncated on a code point");
    }
    {   // Normal receipts are byte-identical to the pre-repair serialization.
        for (const Json& r : {cal::initialReport(), Json{{"state", "X"}, {"error", "\xff\xfe invalid UTF-8"}}})
            check(cal::boundedReceipt(r).complete && cal::receiptText(r) == referenceReceipt(r), "normal receipt unchanged");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--quick")) {
            std::cerr << "Usage: vision-dispatch-calibration-test SHADER_ROOT FRESH_SCRATCH_DIRECTORY [--quick]\n"; return 2;
        }
        shaderRoot = std::filesystem::absolute(argv[1]); scratch = std::filesystem::absolute(argv[2]);
        check(std::filesystem::is_directory(scratch) && std::filesystem::is_empty(scratch), "fresh empty scratch directory");
        const bool quick = argc == 4;
        testSha256(); testFrozenShaders(); testGeometryAndForecast(); testGenerators(); testScreens(); testArguments();
        testProductionEquivalence(); testHelpersDirectly(); testFailures(); testDeadlineBoundaries(); testReceiptBounds();
        testFullRun(quick);
        std::cout << "ALL " << checks << " VISION DISPATCH CALIBRATION CPU CHECKS PASSED" << (quick ? " (quick)" : "") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"; return 1;
    }
}
