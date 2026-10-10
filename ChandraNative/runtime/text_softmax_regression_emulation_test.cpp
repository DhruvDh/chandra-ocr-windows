// New code, MPL-2.0. Linux/POSIX CPU seam for the text softmax regression (tests/native/hlsl_cpu.py translation
// and the hlsl_cpu.h fiber emulator; not an MSVC or GPU test). It
//   * runs the unchanged TextModel attention layer record-only (Stub) at every planned (base, rows) and checks
//     that the regression's shader name, eight cbuffer words, groups and score-plane size equal production's;
//   * executes the translated unmodified repaired shader and the byte-identical 2c44182 control on every
//     planned input under ascending, descending and seeded-shuffle lane schedules, and judges each output with
//     the regression's own independent oracle and gates.
// Requires a generated header containing "runtime/text_attention_softmax.hlsl" and
// "control/text_attention_softmax_2c44182.hlsl". Prints one JSON object; exit 0 only if every expectation holds.
#include "shaders.generated.h"
#include "emulated_device.h"
#include "text_model.h"
#include "text_softmax_regression.h"
#include <iostream>

using namespace chandra::dc;
namespace tsr = chandra::text_softmax_regression;
namespace {
int failures = 0;
#define EXPECT(condition) do { if (!(condition)) { std::cerr << __FILE__ << ":" << __LINE__ << ": expected " #condition "\n"; failures++; } } while (0)
const char* orderName(hlsl_cpu::Order o) { return o == hlsl_cpu::Order::Ascending ? "ascending" : o == hlsl_cpu::Order::Descending ? "descending" : "shuffled"; }

tsr::Json transcript() {
    emulation::Config c; c.defaultMode = emulation::Mode::Stub; c.keepTrace = true; emulation::state().reset(c);
    Device d(L"", "emulated", "emulated"); ModelWeights w(d, L""); TextModel text(d, w);
    tsr::Json rows = tsr::Json::array();
    for (size_t i = 0; i < tsr::cases().size(); ++i) {
        const auto g = tsr::geometry(tsr::cases()[i]); const auto p = tsr::params(g);
        TextLayerCache cache; cache.keys = d.floats(16384 * 1024); cache.values = d.floats(16384 * 1024);
        d.zero(cache.keys); d.zero(cache.values); cache.length = g.base;
        Buffer h = d.floats(g.rows * 2560); TextPositions positions;
        for (uint32_t r = 0; r < g.rows; ++r) for (auto* axis : {&positions.temporal, &positions.height, &positions.width}) axis->push_back(g.base + r);
        const size_t before = emulation::state().dispatches.size();
        text.diagnosticLayer(3, h, g.rows, positions, cache);
        std::vector<const emulation::DispatchRecord*> softmax;
        for (size_t k = before; k < emulation::state().dispatches.size(); ++k)
            if (emulation::state().dispatches[k].shader == tsr::repairedShader) softmax.push_back(&emulation::state().dispatches[k]);
        EXPECT(softmax.size() == 1);
        if (softmax.size() != 1) continue;
        const auto& record = *softmax.front();
        const std::vector<uint32_t> words = {p.rows, p.width, p.offset, p.stride, p.base, p.count, p.mode, p.pad};
        bool tailZero = true; for (size_t k = 8; k < record.params.size(); ++k) tailZero = tailZero && record.params[k] == 0;
        const bool equal = std::equal(words.begin(), words.end(), record.params.begin()) && tailZero && record.x == g.groups && record.y == 1 &&
                           record.z == 1 && record.inputs.empty() && record.outputWords.size() == 1 && record.outputWords[0] == g.planeWords;
        EXPECT(equal && cache.length == g.base + g.rows);
        rows.push_back({{"case", tsr::cases()[i].name}, {"production_cbuffer", std::vector<uint32_t>(record.params.begin(), record.params.begin() + 8)},
                        {"production_groups", {record.x, record.y, record.z}}, {"production_score_words", record.outputWords[0]},
                        {"equal_to_regression", equal}});
    }
    return rows;
}

tsr::Json numerics(bool quick) {
    tsr::Json rows = tsr::Json::array();
    for (uint32_t i = 0; i < tsr::cases().size(); ++i) {
        const auto g = tsr::geometry(tsr::cases()[i]);
        if (quick && g.planeWords > 600000) continue;
        const auto upload = tsr::uploadWords(i, g); const auto oracle = tsr::buildOracle(i, g); const auto p = tsr::params(g);
        std::vector<uint32_t> repaired;
        for (const char* shader : {tsr::repairedShader, tsr::controlShader})
            for (auto order : {hlsl_cpu::Order::Ascending, hlsl_cpu::Order::Descending, hlsl_cpu::Order::Shuffled}) {
                emulation::Config c; c.order = order; c.seed = 0x5eed + i; c.keepTrace = false; emulation::state().reset(c);
                std::vector<uint32_t> observed; uint64_t races = 0, findings = 0;
                {
                    Device d(L"", "emulated", "emulated");
                    Buffer scores = d.words(g.bufferWords, upload.data());
                    d.dispatch(shader, {}, {&scores}, &p, sizeof(p), g.groups);
                    observed = d.readWords(scores);
                    const auto& f = emulation::state().findings;
                    races = f.count("groupshared_race_write_read") + f.count("groupshared_race_read_write"); findings = f.total;
                }
                const tsr::Json check = tsr::checkOutput(i, g, oracle, observed);
                const bool isRepaired = shader == std::string(tsr::repairedShader), passed = check.at("passed") == true;
                if (isRepaired && repaired.empty()) repaired = observed;
                const bool equal = observed == repaired;
                if (isRepaired) EXPECT(passed && findings == 0 && equal);
                else {
                    EXPECT(races > 0 && findings == races);
                    // Lane-granular ascending: lane 0 publishes its exp partial before lane 1 reads the maximum.
                    if (order == hlsl_cpu::Order::Ascending) EXPECT(passed == (g.count < 2) && equal == (g.count < 2));
                    if (order == hlsl_cpu::Order::Descending) EXPECT(passed && equal);
                }
                rows.push_back({{"case", tsr::cases()[i].name}, {"count", g.count}, {"groups", g.groups}, {"shader", shader},
                                {"order", orderName(order)}, {"oracle_passed", passed}, {"race_findings", races}, {"findings", findings},
                                {"equal_to_repaired_ascending", equal}, {"outside_admissible_interval", check.at("outside_admissible_interval")},
                                {"nonfinite", check.at("nonfinite")}, {"rows_failing_normalization", check.at("rows_failing_normalization")},
                                {"equal_to_correctly_rounded_binary64", check.at("equal_to_correctly_rounded_binary64")},
                                {"admitted_adjacent_to_correctly_rounded_binary64", check.at("admitted_adjacent_to_correctly_rounded_binary64")}});
            }
    }
    return rows;
}
} // namespace

int main(int argc, char** argv) {
    const bool quick = argc == 2 && std::string(argv[1]) == "--quick";
    if (argc > 2 || (argc == 2 && !quick)) { std::cerr << "usage: regression-emulation-test [--quick]\n"; return 2; }
    bool translated = false;
    for (const auto& e : hlsl_cpu::entries()) translated = translated || std::string(e.name) == tsr::controlShader;
    EXPECT(translated);
    const tsr::Json report = {{"transcript", transcript()}, {"numerics", numerics(quick)}, {"plan_sha256", tsr::planSha256()}};
    std::cout << tsr::Json{{"failures", failures}, {"quick", quick}, {"report", report}}.dump() << "\n";
    return failures ? 1 : 0;
}
