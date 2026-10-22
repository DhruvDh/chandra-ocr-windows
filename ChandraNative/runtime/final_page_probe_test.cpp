// New code, MPL-2.0. CPU-only tests for final_page_probe.h. No D3D11, GPU, model or listener.
// Usage: final-page-probe-test SCRATCH_DIR SHADER_ROOT [source-row26213.bf16]
//        final-page-probe-test --old-upload-overread   (negative control: must abort under ASan)
#include "final_page_probe.h"
#include <cstdio>
#include <fstream>
#include <iterator>
namespace p = chandra::final_page_probe;
namespace fs = std::filesystem;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #x); ++failures; } } while (0)
template <class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static p::Observation faithful(const std::vector<uint32_t>& cpu, p::Contrast c) {
    p::Observation o; const uint32_t* row = cpu.data() + uint64_t(p::finalRow) * p::rowWords;
    o.direct.assign(row, row + p::rowWords); o.rowCopy = o.direct;
    for (uint32_t r = 0; r < p::selectedRows; ++r) o.checksums.push_back(p::rowChecksum(cpu.data() + uint64_t(p::selectedFirstRow + r) * p::rowWords, p::rowWords));
    if (c == p::Contrast::padded) o.padding.assign(cpu.begin() + p::logicalWords, cpu.end());
    return o;
}

// Fake D3D11 constant buffer: UpdateSubresource with a null box reads exactly ByteWidth bytes from the source.
struct FakeConstantBuffer {
    uint64_t byteWidth = p::constantBufferBytes; std::vector<uint8_t> storage = std::vector<uint8_t>(p::constantBufferBytes, 0xcc);
    uint64_t uploads = 0, lastBytes = 0;
    void updateSubresourceNullBox(const void* source) { std::memcpy(storage.data(), source, byteWidth); ++uploads; lastBytes = byteWidth; }
};
// Mirrors the repaired host call: check the descriptor against the object, then upload the full object.
template <class T> static void hostUpload(FakeConstantBuffer& cb, const T& object) {
    p::requireFullConstantUpload(cb.byteWidth, sizeof(object)); cb.updateSubresourceNullBox(&object);
}
// The source11 call shape: a 32-byte params[8] passed to a 256-byte constant buffer without a size check.
static __attribute__((noinline)) void source11Upload(FakeConstantBuffer& cb, const uint32_t (&params)[8]) { cb.updateSubresourceNullBox(params); }

struct FakeClock final : p::Clock {
    uint64_t now = 1000, pauses = 0;
    uint64_t nowMs() override { return now; }
    void pause() override { ++pauses; now += 1; }
};
struct CountingHost final : p::Host {
    int calls = 0;
    p::Json execute(const p::Arguments&, const p::Admitted&) override { ++calls; return {{"device_created", true}}; }
};
static void writeBytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}
static std::vector<uint8_t> readBytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), {});
}
static std::string message(const std::function<void()>& f) {
    try { f(); } catch (const std::exception& e) { return e.what(); } return "";
}
static bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--old-upload-overread") {
        FakeConstantBuffer cb; uint32_t params[8]; for (uint32_t i = 0; i < 8; ++i) params[i] = uint32_t(argc) + i; // stack object.
        source11Upload(cb, params);
        std::printf("source11 upload completed without detection (%u)\n", unsigned(cb.storage[200]));
        return 0;
    }
    if (argc < 3) { std::fprintf(stderr, "usage: final-page-probe-test SCRATCH_DIR SHADER_ROOT [source-row]\n"); return 2; }
    const fs::path scratch(argv[1]), shaderRoot(argv[2]);
    const char* sourceRowPath = argc > 3 ? argv[3] : nullptr;
    // Shape and final-page offsets match the native-0 capture.
    CHECK(p::logicalBytes == 134215680ull && p::paddedBytes == 134217728ull);
    CHECK(uint64_t(p::finalRow) * p::rowBytes == 134210560ull);         // receipt tensor_relative_byte_offset of row 26213.
    CHECK(p::retainedBlockByte == 134213632ull && p::paddedBytes - p::retainedBlockByte == 4096);
    CHECK(p::logicalBytes % 4096 == 2048 && p::finalPageByte == p::retainedBlockByte);
    CHECK((p::retainedBlockByte - p::directFirstByte) / 4 == 768);
    CHECK(p::directFirstByte + p::directBytes == p::logicalBytes);       // direct copy ends at the logical end.
    CHECK(p::directBytes + p::paddingBytes <= 8192);
    // Partial-page versus padded visibility: SRV extent is the logical shape in both contrasts.
    CHECK(p::srvWords(p::Contrast::original) == p::srvWords(p::Contrast::padded));
    CHECK(p::physicalBytes(p::Contrast::original) == p::logicalBytes && p::physicalBytes(p::Contrast::padded) == p::paddedBytes);
    // Dispatch bounds cover only the selected rows / words.
    CHECK(p::groups(p::selectedRows, p::checksumThreads) == 1 && p::groups(p::rowWords, p::rowCopyThreads) == 5);
    CHECK(uint64_t(p::selectedFirstRow + p::selectedRows) * p::rowWords == p::logicalWords);
    // Forecast and inactive plan.
    const auto f = p::forecast(); CHECK(f.total < 512ull * 1024 * 1024);
    const auto plan = p::plan();
    CHECK(plan["device_created"] == false && plan["full_buffer_readback"] == false && plan["large_buffers_live_at_once"] == 1);
    CHECK(plan["contrasts"][0]["srv_bytes"] == plan["contrasts"][1]["srv_bytes"]);
    CHECK(plan["contrasts"][1]["physical_padding_bytes"] == 2048 && plan["contrasts"][0]["physical_padding_bytes"] == 0);
    // Inactive by default; strict adapter arguments only with --execute.
    CHECK(!p::parse({}).execute);
    CHECK(throws([] { p::parse({"--execute"}); }));
    CHECK(throws([] { p::parse({"--execute", "--pci", "x", "--shader-root", "s", "--output", "o"}); })); // LUID required.
    CHECK(throws([] { p::parse({"--pci", "x"}); }));
    CHECK(throws([] { p::parse({"--execute", "--execute"}); }));
    CHECK(throws([] { p::parse({"--bogus"}); }));
    CHECK(p::parse({"--execute", "--pci", "a", "--luid", "b", "--shader-root", "s", "--output", "o"}).execute);
    // Deterministic finite input and commitments.
    for (uint32_t i : {0u, 1u, 33553407u, 33553919u}) CHECK(p::syntheticWord(i) == p::syntheticWord(i) && p::finiteWord(p::syntheticWord(i)));
    uint32_t nonFinite = 0; for (uint32_t i = 0; i < 1u << 20; ++i) nonFinite += !p::finiteWord(p::syntheticWord(i));
    CHECK(nonFinite == 0 && p::finiteBf16(0x7f80) == 0x3f80 && p::finiteBf16(0xffc1) == 0xbfc1);
    std::vector<uint32_t> original, padded; p::fill(original, p::Contrast::original, nullptr); p::fill(padded, p::Contrast::padded, nullptr);
    CHECK(original.size() * 4 == p::logicalBytes && padded.size() * 4 == p::paddedBytes);
    CHECK(std::equal(original.begin(), original.end(), padded.begin()));   // identical logical bytes across contrasts.
    CHECK(padded[p::logicalWords] == p::paddingWord(p::logicalWords));
    // Retained source row splice, when the receipt file is supplied; it is admitted through the bounded host path.
    if (sourceRowPath) {
        const auto admitted = p::admitFile(sourceRowPath, p::sourceRowPolicy); const auto& row = admitted.bytes;
        CHECK(row.size() == p::rowBytes && admitted.sha256 == p::retainedSourceSha256);
        std::vector<uint32_t> spliced; p::fill(spliced, p::Contrast::original, &row);
        CHECK(std::memcmp(spliced.data() + uint64_t(p::finalRow) * p::rowWords, row.data(), p::rowBytes) == 0);
        CHECK(spliced[uint64_t(p::finalRow) * p::rowWords + 768] == 0x3bcbbc27u); // word-difference-observation source_words[0].
        CHECK(spliced[uint64_t(p::finalRow) * p::rowWords - 1] == original[uint64_t(p::finalRow) * p::rowWords - 1]);
    }
    std::vector<uint8_t> shortRow(100); CHECK(throws([&] { std::vector<uint32_t> v; p::fill(v, p::Contrast::original, &shortRow); }));
    // Faithful observations pass and make no physical claim.
    for (auto c : {p::Contrast::original, p::Contrast::padded}) {
        const auto& cpu = c == p::Contrast::original ? original : padded;
        const auto j = p::judge(cpu, c, faithful(cpu, c));
        CHECK(j["pass"] == true && j["physical_storage_proven"] == false && j["direct_and_srv_agree"] == true);
        CHECK(j["interpretation"].get<std::string>().find("synthetic input only") != std::string::npos);
    }
    // Deliberately corrupted block, reproducing the native-0 signature on the SRV path only.
    auto bad = faithful(original, p::Contrast::original);
    for (uint32_t i = 768; i < 832; i += 2) { bad.rowCopy[i] = 0x24000006u; bad.rowCopy[i + 1] = 8u; }
    bad.rowCopy[800] = 0x145a080fu; bad.rowCopy[801] = 0;
    auto j = p::judge(original, p::Contrast::original, bad);
    CHECK(j["pass"] == false && j["direct_copy"]["identical"] == true && j["srv_row_copy"]["differing_words"] == 64);
    CHECK(j["srv_row_copy"]["ranges"].size() == 1 && j["srv_row_copy"]["ranges"][0]["first_shard_byte"] == 134213632ull);
    CHECK(j["srv_row_copy"]["ranges"][0]["last_shard_byte"] == 134213887ull && j["srv_row_copy"]["ranges"][0]["distance_to_128MiB"] == 4096);
    CHECK(j["retained_signature"]["srv_row_copy"] == true && j["retained_signature"]["direct_copy"] == false);
    CHECK(j["direct_and_srv_agree"] == false && j["physical_storage_proven"] == false);
    // Checksum-only discrepancy and direct-copy-only discrepancy are each reported independently.
    auto sumBad = faithful(original, p::Contrast::original); sumBad.checksums[1] ^= 1;
    j = p::judge(original, p::Contrast::original, sumBad);
    CHECK(j["pass"] == false && j["checksum_screen"]["all_match"] == false && j["direct_copy"]["identical"] == true);
    auto directBad = faithful(padded, p::Contrast::padded); directBad.direct[1279] ^= 0x80000000u;
    j = p::judge(padded, p::Contrast::padded, directBad);
    CHECK(j["pass"] == false && j["direct_copy"]["ranges"][0]["first_shard_byte"] == 134215676ull && j["srv_row_copy"]["identical"] == true);
    // Padding differences are recorded but never gate and are never SRV-visible.
    auto padBad = faithful(padded, p::Contrast::padded); padBad.padding[0] = 0;
    j = p::judge(padded, p::Contrast::padded, padBad);
    CHECK(j["pass"] == true && j["padding_record"]["differing_words"] == 1 && j["padding_record"]["gating"] == false);
    CHECK(j["padding_record"]["visible_through_srv"] == false);
    // Geometry violations throw.
    auto wrong = faithful(original, p::Contrast::original); wrong.direct.pop_back();
    CHECK(throws([&] { p::judge(original, p::Contrast::original, wrong); }));
    auto orphan = faithful(original, p::Contrast::original); orphan.padding.assign(512, 0);
    CHECK(throws([&] { p::judge(original, p::Contrast::original, orphan); }));

    // ---- Constant upload host boundary ----
    {
        const uint32_t oldParams[8] = {p::rowWords, p::selectedRows, p::selectedFirstRow, 0, 0, 0, 0, 0};
        FakeConstantBuffer cb;
        const std::string old = message([&] { hostUpload(cb, oldParams); });   // source11 object shape is refused before any read.
        CHECK(has(old, "exactly the 256-byte") && cb.uploads == 0);
        CHECK(throws([] { p::requireFullConstantUpload(256, 32); }) && throws([] { p::requireFullConstantUpload(32, 32); }));
        CHECK(throws([] { p::requireFullConstantUpload(512, 512); }) && throws([] { p::requireFullConstantUpload(256, 257); }));
        const p::Constants c = p::params({p::rowWords, p::selectedRows, p::selectedFirstRow, 0});
        hostUpload(cb, c);
        CHECK(cb.uploads == 1 && cb.lastBytes == 256 && sizeof(c) == cb.byteWidth);
        uint32_t uploaded[64]; std::memcpy(uploaded, cb.storage.data(), 256);
        CHECK(uploaded[0] == p::rowWords && uploaded[1] == p::selectedRows && uploaded[2] == p::selectedFirstRow);
        bool tailZero = true; for (uint32_t i = 3; i < 64; ++i) tailZero = tailZero && uploaded[i] == 0;
        CHECK(tailZero);                                                        // fully initialised, no host bytes disclosed.
        CHECK(throws([] { p::params({1, 2, 3, 4, 5, 6, 7, 8, 9}); }));
        const p::Constants copy = p::params({p::finalRow * p::rowWords, p::rowWords, 0});
        CHECK(copy.words[0] == 33553920u - p::rowWords && copy.words[1] == p::rowWords && copy.words[7] == 0 && copy.words[63] == 0);
    }
    // ---- View and copy bounds at the helper boundary ----
    CHECK(!throws([] { p::requireViewWords(p::logicalBytes, p::logicalWords); }) && !throws([] { p::requireViewWords(p::paddedBytes, p::logicalWords); }));
    CHECK(throws([] { p::requireViewWords(p::logicalBytes, p::logicalWords + 1); }));
    CHECK(!throws([] { p::requireViewWords(p::rowBytes, p::rowWords); }) && throws([] { p::requireViewWords(p::rowBytes, p::rowWords + 1); }));
    CHECK(!throws([] { p::requireViewWords(uint64_t(p::churnWords) * 4, p::churnWords); }) && throws([] { p::requireViewWords(p::paddedBytes + 4, 1); }));
    CHECK(throws([] { p::requireViewWords(0, 0); }) && throws([] { p::requireViewWords(6, 1); }));
    CHECK(!throws([] { p::requireCopyBox(p::logicalBytes, p::directFirstByte, p::directBytes, 0, p::stagingBytes); }));
    CHECK(!throws([] { p::requireCopyBox(p::paddedBytes, p::logicalBytes, p::paddingBytes, p::directBytes, p::stagingBytes); }));
    CHECK(throws([] { p::requireCopyBox(p::logicalBytes, p::logicalBytes, p::paddingBytes, p::directBytes, p::stagingBytes); })); // no padding in original.
    CHECK(throws([] { p::requireCopyBox(p::logicalBytes, p::directFirstByte + 4, p::directBytes, 0, p::stagingBytes); }));       // source end + 4.
    CHECK(throws([] { p::requireCopyBox(p::paddedBytes, 0, p::stagingBytes, 4, p::stagingBytes); }));                         // destination end + 4.
    CHECK(throws([] { p::requireCopyBox(p::paddedBytes, 2, 4, 0, p::stagingBytes); }) && throws([] { p::requireCopyBox(p::paddedBytes, 0, 0, 0, 8); }));
    CHECK(throws([] { p::requireCopyBox(1ull << 33, 0, 4, 0, 8); }) && throws([] { p::requireCopyBox(16, ~0ull - 3, 8, 0, 8); }));
    CHECK(!throws([] { p::requireCopyBox(p::rowBytes, 0, p::selectedRows * 4, 0, p::stagingBytes); }));
    CHECK(!throws([] { p::requireCopyBox(p::logicalBytes, p::directFirstByte, p::rowBytes, 0, uint64_t(p::churnWords) * 4); }));

    // ---- Bounded admission of shaders and source rows, before any host call ----
    {
        const auto checksum = p::admitFile(shaderRoot / p::checksumShaderPolicy.label, p::checksumShaderPolicy);
        const auto rowCopy = p::admitFile(shaderRoot / p::rowCopyShaderPolicy.label, p::rowCopyShaderPolicy);
        CHECK(checksum.bytes.size() == 949 && rowCopy.bytes.size() == 683);
        CHECK(!throws([&] { p::requireNoInclude(checksum.bytes); p::requireNoInclude(rowCopy.bytes); }));
        const fs::path root = scratch / "shaders"; fs::create_directories(root / "runtime");
        const fs::path cs = root / p::checksumShaderPolicy.label, rc = root / p::rowCopyShaderPolicy.label;
        auto reset = [&] { writeBytes(cs, checksum.bytes); writeBytes(rc, rowCopy.bytes); };
        reset();
        p::Arguments a = p::parse({"--execute", "--pci", "a", "--luid", "b", "--shader-root", root.string(), "--output", "o"});
        CountingHost host;
        CHECK(p::run(a, host)["device_created"] == true && host.calls == 1);   // pinned bytes admitted, then one host call.
        const auto admitted = p::admitInputs(a);
        CHECK(admitted.checksumShader.bytes == checksum.bytes && admitted.rowCopyShader.sha256 == p::rowCopyShaderPolicy.sha256 && !admitted.sourceRow);
        auto rejected = [&](const char* part) {
            const int before = host.calls; const std::string m = message([&] { p::run(a, host); });
            const bool ok = has(m, part) && host.calls == before;
            if (!ok) std::fprintf(stderr, "admission message '%s' (wanted '%s')\n", m.c_str(), part);
            return ok;
        };
        auto changed = checksum.bytes; changed[100] ^= 1; writeBytes(cs, changed);
        CHECK(rejected("SHA-256 differs")); reset();
        auto included = rowCopy.bytes; std::memcpy(included.data(), "#include \"x.h\"\n", 15); writeBytes(rc, included);
        CHECK(rejected("SHA-256 differs") && throws([&] { p::requireNoInclude(included); })); reset();
        CHECK(throws([] { p::requireNoInclude(std::vector<uint8_t>{'#', ' ', '\t', 'i', 'n', 'c', 'l', 'u', 'd', 'e', '<'}); }));
        auto longer = checksum.bytes; longer.push_back('\n'); writeBytes(cs, longer);
        CHECK(rejected("extra bytes")); reset();
        writeBytes(cs, std::vector<uint8_t>(1u << 20, ' '));               // oversized: read stops at 950 bytes.
        CHECK(rejected("extra bytes")); reset();
        auto shorter = rowCopy.bytes; shorter.pop_back(); writeBytes(rc, shorter);
        CHECK(rejected("short file")); reset();
        fs::remove(rc); CHECK(rejected("cannot open"));
        fs::create_directory(rc); CHECK(rejected("runtime/weight_row_words.hlsl:")); fs::remove(rc); reset();   // unreadable: a directory.
        // Source rows. A deterministic wrong-content row has the right size but the wrong hash.
        const fs::path row = scratch / "row.bf16"; a.sourceRow = row.string();
        writeBytes(row, std::vector<uint8_t>(p::rowBytes, 7)); CHECK(rejected("source row: SHA-256 differs"));
        writeBytes(row, std::vector<uint8_t>(p::rowBytes - 1, 7)); CHECK(rejected("source row: short file"));
        writeBytes(row, std::vector<uint8_t>(p::rowBytes + 1, 7)); CHECK(rejected("source row: extra bytes"));
        writeBytes(row, std::vector<uint8_t>(64u << 20, 7)); CHECK(rejected("source row: extra bytes"));      // 64 MiB file, 5121-byte read.
        writeBytes(row, {}); CHECK(rejected("source row: short file"));
        fs::remove(row); CHECK(rejected("source row: cannot open"));
        fs::create_directory(row); CHECK(rejected("source row:")); fs::remove(row);
        if (sourceRowPath) {
            const auto good = readBytes(sourceRowPath); writeBytes(row, good);
            CHECK(p::run(a, host)["device_created"] == true);
            auto extra = good; extra.push_back(0); writeBytes(row, extra); CHECK(rejected("source row: extra bytes"));
            auto flipped = good; flipped[5119] ^= 0x80; writeBytes(row, flipped); CHECK(rejected("source row: SHA-256 differs"));
        }
        // Direct-only mode admits no shader: the direct D3D11 copy does not depend on shader files.
        p::Arguments d = p::parse({"--execute", "--direct-only", "--pci", "a", "--luid", "b", "--output", "o"});
        fs::remove(cs); const auto direct = p::admitInputs(d); CHECK(direct.checksumShader.bytes.empty() && direct.rowCopyShader.bytes.empty());
        const int before = host.calls; p::run(d, host); CHECK(host.calls == before + 1);
        // Inactive mode returns the plan without any host (Job, adapter, Device) call.
        CountingHost idle; const auto inactive = p::run(p::parse({}), idle);
        CHECK(idle.calls == 0 && inactive["device_created"] == false && inactive["mode"] == "direct_and_srv");
        CHECK(p::run(p::parse({"--direct-only"}), idle)["mode"] == "direct_only" && idle.calls == 0);
        CHECK(throws([] { p::admitInputs(p::parse({})); }));
        fs::remove_all(root);
    }
    // ---- Arguments for the shader-free mode ----
    CHECK(throws([] { p::parse({"--execute", "--direct-only", "--pci", "a", "--luid", "b", "--shader-root", "s", "--output", "o"}); }));
    CHECK(throws([] { p::parse({"--execute", "--pci", "a", "--luid", "b", "--output", "o"}); }));
    CHECK(throws([] { p::parse({"--output", "o"}); }) && throws([] { p::parse({"--shader-root", "s"}); }) && throws([] { p::parse({"--direct-only", "--direct-only"}); }));
    CHECK(throws([] { p::parse({"--execute", "--pci", "a", "--luid", "b", "--shader-root", "s"}); }));

    // ---- Timed operations with a fake clock and delayed fake calls ----
    {
        FakeClock clock; p::Timer t(clock, 100);
        int polls = 0;
        p::timedOperation(t, "normal", p::slowOperationMs, [&] { clock.now += 10; }, [&] { return ++polls >= 3; }, [&] { clock.now += 5; return true; });
        CHECK(t.records().size() == 1 && t.records()[0]["ms"] == 17 && t.records()[0]["limit_ms"] == 250);
        // Delayed submission (UpdateSubresource/Dispatch/Copy returns late): the event is then immediately complete.
        CHECK(has(message([&] { p::timedOperation(t, "slow_submit", p::slowOperationMs, [&] { clock.now += 300; }, [] { return true; }, [] { return true; }); }), "Late completion"));
        // Delayed Map returning success after the limit.
        CHECK(has(message([&] { p::timedOperation(t, "slow_map", p::slowOperationMs, [] {}, [] { return true; }, [&] { clock.now += 251; return true; }); }), "Late completion"));
        // Late successful drain: the event reports done on a poll that itself returned after the limit.
        CHECK(has(message([&] { p::timedOperation(t, "late_drain", p::slowOperationMs, [] {}, [&] { clock.now += 260; return true; }, [] { return true; }); }), "Late completion"));
        // Exactly at the limit is accepted; one millisecond beyond is not.
        CHECK(!throws([&] { p::timedOperation(t, "edge", p::slowOperationMs, [&] { clock.now += 250; }, [] { return true; }, [] { return true; }); }));
        // Never-completing drain and never-completing Map stop at the operation deadline with a finite number of polls.
        const uint64_t pauses = clock.pauses;
        CHECK(has(message([&] { p::timedOperation(t, "stuck_drain", p::slowOperationMs, [] {}, [] { return false; }, [] { return true; }); }), "pending"));
        CHECK(has(message([&] { p::timedOperation(t, "stuck_map", p::longOperationMs, [] {}, [] { return true; }, [] { return false; }); }), "pending"));
        CHECK(clock.pauses - pauses <= 251 + 2001);
        // Whole-run deadline: every operation is individually in time, but the run is not.
        FakeClock slow; p::Timer run(slow, 100, 1000); int done = 0;
        const std::string m = message([&] { for (;;) { p::timedOperation(run, "op", p::slowOperationMs, [&] { slow.now += 200; }, [] { return true; }, [] { return true; }); ++done; } });
        CHECK(has(m, "Whole-run deadline") && done == 5);                 // 1000 ms inclusive, sixth finishes at 1200.
        FakeClock idleClock; p::Timer cpuWork(idleClock, 4, 1000); idleClock.now += 1001; CHECK(throws([&] { cpuWork.requireRun("fill"); }));
        // Declared operation count.
        FakeClock countClock; p::Timer counted(countClock, 2);
        for (int i = 0; i < 2; ++i) p::timedOperation(counted, "x", 1, [] {}, [] { return true; }, [] { return true; });
        CHECK(has(message([&] { p::timedOperation(counted, "x", 1, [] {}, [] { return true; }, [] { return true; }); }), "operation count"));
        // Declared plan: 32 shader-mode operations (24 direct-only), budgets inside the whole-run deadline.
        CHECK(p::shaderOperationsPerContrast == 16 && p::directOperationsPerContrast == 12);
        const auto timing = p::plan()["timing"];
        CHECK(timing["operations"] == 32 && timing["declared_operation_budget_ms"] == 4 * 2000 + 28 * 250 && timing["whole_run_deadline_ms"] == 120000);
        CHECK(timing["declared_operation_budget_ms"].get<uint64_t>() < p::runDeadlineMs && p::plan(true)["timing"]["operations"] == 24);
    }
    // ---- Accounting: requested GPU payload versus bounded host payload ----
    {
        const auto fc = p::forecast();
        CHECK(fc.requestedGpuPayload == p::paddedBytes + 1048576 + 5120 + 8192 + 256);
        CHECK(fc.hostPayload >= 134242312ull);                                  // reviewer's simultaneous vector payload.
        CHECK(fc.hostReadback >= 7168 + 5120 + 2048 + 8 + 5120 + 8192 && fc.hostSourceRow == 5121);
        CHECK(fc.total == fc.requestedGpuPayload + fc.hostPayload && fc.total < 512ull * 1024 * 1024);
        const auto fb = p::plan()["forecast_bytes"];
        CHECK(fb["requested_gpu_payload"]["total"] == fc.requestedGpuPayload && fb["host_payload_bound"]["total"] == fc.hostPayload);
        CHECK(has(fb["process_and_driver_allocation"].get<std::string>(), "unmeasured"));
    }
    // ---- Direct-only judging ----
    {
        auto o = faithful(padded, p::Contrast::padded); o.rowCopy.clear(); o.checksums.clear();
        auto dj = p::judge(padded, p::Contrast::padded, o, true);
        CHECK(dj["pass"] == true && dj["mode"] == "direct_only" && dj["srv_row_copy"].is_null() && dj["direct_and_srv_agree"].is_null());
        o.direct[768] ^= 1; dj = p::judge(padded, p::Contrast::padded, o, true);
        CHECK(dj["pass"] == false && dj["direct_copy"]["ranges"][0]["first_shard_byte"] == 134213632ull);
        CHECK(throws([&] { p::judge(original, p::Contrast::original, faithful(original, p::Contrast::original), true); }));
        auto missing = faithful(original, p::Contrast::original); missing.checksums.clear();
        CHECK(throws([&] { p::judge(original, p::Contrast::original, missing); }));
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("final_page_probe_test: all checks passed");
    return 0;
}
