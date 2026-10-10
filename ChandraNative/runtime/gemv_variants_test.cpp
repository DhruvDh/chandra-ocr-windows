// New ChandraNative code, MPL-2.0. CPU-only checks of gemv_variants.h and the variant HLSL sources;
// no GPU, model, network or HLSL execution. Usage: gemv-variants-test SHADER_RUNTIME_DIRECTORY [--forecast]
// 1. Selector parsing and the shape table. 2. Exhaustive staging/owner index algebra of the row-major
// variants over every K tail, start parity, shard offset and output tail, including the uint32
// addressing bound. 3. Structural source checks: barriers only at the top of the uniform K loop, no
// early exit in main, no FP16/FMA/wave/atomic tokens, defines equal to the helper, and arithmetic text
// identical to linear_gemv.hlsl; each rule is also shown to reject a mutated source. 4. Per-token
// structural forecast for the pinned decode shapes.
#include "gemv_variants.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gv = chandra::gemv_variants;
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error("CHECK FAILED: " + message); }

// ---- 1. Selector and shape table ----
void selectorTable() {
    check(gv::parse(nullptr) == gv::Route::predecessor && gv::parse("0") == gv::Route::predecessor, "unset and 0 are the predecessor");
    check(gv::parse("ordered") == gv::Route::ordered && gv::parse("ordered32") == gv::Route::ordered32 &&
          gv::parse("ordered64") == gv::Route::ordered64 && gv::parse("parallel32") == gv::Route::parallel32, "exact experimental selectors");
    for (const char* bad : {"", "1", "00", "ORDERED", "Ordered32", " ordered32", "ordered32 ", "ordered 32", "ordered16",
                            "ordered8", "ordered128", "ordered320", "ordered6", "predecessor", "true", "ordered\t"})
        check(gv::parse(bad) == gv::Route::invalid, std::string("invalid selector accepted: \"") + bad + "\"");
    check(gv::shape(gv::Route::predecessor) == nullptr && gv::shape(gv::Route::invalid) == nullptr &&
          gv::shape(static_cast<const char*>(nullptr)) == nullptr && gv::shape("0") == nullptr && gv::shape("bogus") == nullptr,
          "no shape for predecessor or invalid routes");
    // Independent statement of the expected table.
    struct Want { const char* selector; const char* shader; uint32_t rows, tile, stride, bytes; };
    const Want want[] = {{"ordered", "runtime/linear_gemv.hlsl", 8, 512, 513, 18464},
                         {"ordered32", "runtime/linear_gemv_ordered32.hlsl", 32, 128, 129, 17024},
                         {"ordered64", "runtime/linear_gemv_ordered64.hlsl", 64, 64, 65, 16896},
                         {"parallel32", "runtime/linear_gemv_parallel32.hlsl", 8, 512, 513, 18464}};
    check(gv::shapeCount == 4, "three ordered routes and explicit reassociated parallel32");
    for (const auto& w : want) {
        const gv::Shape* s = gv::shape(w.selector);
        check(s && std::string(s->selector) == w.selector && std::string(s->shader) == w.shader && s->outputsPerGroup == w.rows &&
              s->threads == 256 && s->tile == w.tile && s->stride == w.stride && s->groupsharedBytes() == w.bytes &&
              s->groupsharedBytes() < 32768, std::string("shape table entry ") + w.selector);
        for (uint32_t outputs : {1u, 7u, 8u, 9u, 31u, 32u, 33u, 63u, 64u, 65u, 512u, 1023u, 1024u})
            check(s->groups(outputs) == (outputs + w.rows - 1) / w.rows && uint64_t(s->groups(outputs)) * w.rows >= outputs &&
                  uint64_t(s->groups(outputs) - 1) * w.rows < outputs, "groups cover outputs with one partial group at most");
        for (uint32_t k : {1u, 63u, 64u, 65u, 127u, 128u, 129u, 511u, 512u, 513u, 2560u, 4096u, 9216u})
            check(s->tiles(k) == (k + w.tile - 1) / w.tile, "tile count");
        check(s->groups(1024) <= 65535, "dispatch X within the SM5 limit");
    }
    check(!gv::staging::rowMajor(*gv::shape("parallel32")), "parallel32 uses ordered8 staging, not the ordered32/64 layout");
    std::cout << "selector table: exact selectors, " << 16 << " invalid spellings refused, 4 shapes\n";
}

// ---- 2. Staging and owner index algebra ----
struct Tally { uint64_t tiles = 0, stores = 0, loads = 0, extraWords = 0; };
// One group, one K tile: every thread's slots (as stage() defines them) must load each word of each
// live row's tile exactly once and store each position [0, n) of each live row exactly once, with the
// element the position denotes; nothing else is loaded or stored.
void tileInvariants(const gv::Shape& s, bool bf16, uint32_t inputWidth, uint32_t shardRows, uint32_t weightRowFirst,
                    uint32_t groupFirst, uint32_t begin, Tally& tally) {
    constexpr uint64_t none = ~uint64_t(0);
    const uint32_t n = std::min(s.tile, inputWidth - begin), wordsPerRow = s.tile + 1;
    const uint64_t shardElements = uint64_t(weightRowFirst + shardRows) * inputWidth;
    const uint64_t shardWords = bf16 ? (shardElements + 1) / 2 : shardElements;
    std::vector<uint64_t> stored(size_t(s.outputsPerGroup) * s.tile, none); // (row, position) -> element
    std::vector<uint8_t> loaded(size_t(s.outputsPerGroup) * wordsPerRow, 0); // (row, word within the row tile)
    for (uint32_t thread = 0; thread < s.threads; ++thread)
        for (uint32_t i = 0; i <= gv::staging::slots; ++i) {
            const auto slot = gv::staging::stage(s, inputWidth, shardRows, weightRowFirst, groupFirst, thread, i, begin, n, bf16);
            if (!(slot.word < slot.count)) continue;
            check(slot.row < s.outputsPerGroup && groupFirst + slot.row < shardRows && slot.word < wordsPerRow, "only live rows load");
            const uint64_t first = uint64_t(weightRowFirst + groupFirst + slot.row) * inputWidth + begin; // exact
            check(first < (uint64_t(1) << 32) && slot.base == (bf16 ? first >> 1 : first) && slot.parity == (bf16 ? first & 1 : 0),
                  "uint32 base and parity equal exact arithmetic");
            const uint64_t word = uint64_t(slot.base) + slot.word;
            check(word < shardWords && word * 4 + 3 <= 0xffffffffull, "load inside the shard and the uint32 byte range");
            check(loaded[size_t(slot.row) * wordsPerRow + slot.word]++ == 0, "a word of one row tile is loaded twice");
            ++tally.loads; tally.extraWords += i == gv::staging::slots;
            auto store = [&](uint32_t position, uint64_t element) {
                check(position < n && position < s.stride, "store outside the tile row");
                uint64_t& slotElement = stored[size_t(slot.row) * s.tile + position];
                check(slotElement == none, "groupshared position stored twice");
                slotElement = element; ++tally.stores;
            };
            if (!bf16) store(slot.word, word);
            else {
                // Same guards as the HLSL: low half -> 2j - parity, high half -> 2j + 1 - parity.
                if (2 * slot.word >= slot.parity) store(2 * slot.word - slot.parity, word * 2);
                if (2 * slot.word + 1 - slot.parity < n) store(2 * slot.word + 1 - slot.parity, word * 2 + 1);
            }
        }
    for (uint32_t row = 0; row < s.outputsPerGroup; ++row) {
        const bool live = groupFirst + row < shardRows;
        const uint64_t first = uint64_t(weightRowFirst + groupFirst + row) * inputWidth + begin;
        for (uint32_t position = 0; position < s.tile; ++position) {
            const uint64_t element = stored[size_t(row) * s.tile + position];
            check((live && position < n) == (element != none), "live rows are completely staged; tail rows and positions are untouched");
            if (element != none) check(element == first + position, "staged position holds the element it denotes");
        }
        // Exactly the words that hold elements [first, first + n), each once.
        const uint64_t words = live ? (bf16 ? (first + n - 1) / 2 - first / 2 + 1 : n) : 0;
        for (uint32_t w = 0; w < wordsPerRow; ++w)
            check(loaded[size_t(row) * wordsPerRow + w] == (w < words ? 1 : 0), "row tile words loaded exactly once");
    }
    ++tally.tiles;
}
// Every output of one <=1024-output dispatch is owned by exactly one (group, thread < outputsPerGroup).
void ownerCoverage(const gv::Shape& s, uint32_t outputs) {
    std::vector<uint32_t> owners(outputs, 0);
    for (uint32_t group = 0; group < s.groups(outputs); ++group)
        for (uint32_t thread = 0; thread < s.threads; ++thread) {
            const uint32_t localOut = group * s.outputsPerGroup + thread;
            if (thread < s.outputsPerGroup && localOut < outputs) ++owners[localOut];
        }
    for (uint32_t count : owners) check(count == 1, "each output has exactly one owner");
}
void stagingInvariants() {
    for (const gv::Shape& s : gv::shapes) {
        if (!gv::staging::rowMajor(s)) continue;
        Tally tally;
        // Every K tail n in [1, tile] with both parities (odd and even K), full tiles, pinned widths.
        std::vector<uint32_t> widths;
        for (uint32_t k = 1; k <= 2 * s.tile + 1; ++k) widths.push_back(k);
        for (uint32_t k : {1001u, 2560u, 4096u, 9216u}) widths.push_back(k);
        for (bool bf16 : {true, false})
            for (uint32_t k : widths)
                for (uint32_t weightRowFirst : {0u, 1u, 1024u, 1025u})
                    for (uint32_t live : {1u, 2u, s.outputsPerGroup - 1, s.outputsPerGroup, s.outputsPerGroup + 3, 2 * s.outputsPerGroup}) {
                        const uint32_t groups = s.groups(live);
                        // Large widths: first, last and one middle tile only.
                        const uint32_t tiles = s.tiles(k);
                        for (uint32_t t = 0; t < tiles; ++t) {
                            if (k > 2 * s.tile + 1 && t != 0 && t != tiles / 2 && t + 1 != tiles) continue;
                            for (uint32_t g = 0; g < groups; ++g)
                                tileInvariants(s, bf16, k, live, weightRowFirst, g * s.outputsPerGroup, t * s.tile, tally);
                        }
                    }
        // uint32 bound: the largest shard linear() admits at K = 9216 (and an odd K) for each weight type.
        for (bool bf16 : {true, false})
            for (uint32_t k : {9216u, 9215u}) {
                const uint64_t maximumElements = 0xffffffffull / (bf16 ? 2 : 4);
                const uint32_t rows = uint32_t(maximumElements / k), lastChunk = (rows - 1) / 1024 * 1024;
                const uint32_t live = rows - lastChunk;
                for (uint32_t t : {0u, s.tiles(k) - 1})
                    for (uint32_t g = 0; g < s.groups(live); ++g)
                        tileInvariants(s, bf16, k, live, lastChunk, g * s.outputsPerGroup, t * s.tile, tally);
            }
        for (uint32_t outputs = 1; outputs <= 1024; ++outputs) ownerCoverage(s, outputs);
        check(tally.extraWords > 0, "odd-parity full tiles exercised the extra packed word");
        std::cout << "staging invariants " << s.selector << ": " << tally.tiles << " group tiles, " << tally.loads << " loads ("
                  << tally.extraWords << " odd-parity extra words), " << tally.stores
                  << " groupshared stores, owners cover outputs 1..1024 exactly once\n";
    }
}

// ---- 3. HLSL source structure ----
std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary); check(bool(in), "cannot read " + path);
    std::ostringstream text; text << in.rdbuf(); return text.str();
}
std::string stripComments(const std::string& source) {
    std::string out; bool line = false;
    for (size_t i = 0; i < source.size(); ++i) {
        if (!line && source.compare(i, 2, "//") == 0) line = true;
        if (line && source[i] == '\n') line = false;
        if (!line) out.push_back(source[i]);
    }
    check(out.find("/*") == std::string::npos, "block comments are not used");
    return out;
}
// Text of the brace-delimited block whose header starts at `header` (header included).
std::string block(const std::string& code, const std::string& header, size_t from = 0) {
    size_t start = code.find(header, from); check(start != std::string::npos, "missing " + header);
    size_t open = code.find('{', start); int depth = 0;
    for (size_t i = open; i < code.size(); ++i) {
        if (code[i] == '{') ++depth;
        if (code[i] == '}' && --depth == 0) return code.substr(start, i + 1 - start);
    }
    throw std::runtime_error("unbalanced block " + header);
}
std::vector<std::string> tokens(const std::string& code) {
    std::vector<std::string> result; std::string current;
    for (char c : code) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') current.push_back(c);
        else if (!current.empty()) { result.push_back(current); current.clear(); }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}
std::string trim(const std::string& text) {
    size_t a = text.find_first_not_of(" \t\r\n"), b = text.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : text.substr(a, b + 1 - a);
}
std::map<std::string, std::string> defines(const std::string& code) {
    std::map<std::string, std::string> result; std::istringstream lines(code); std::string line;
    while (std::getline(lines, line)) {
        std::istringstream words(line); std::string directive, name, value;
        if (words >> directive && directive == "#define" && words >> name && words >> value) result[name] = value;
    }
    return result;
}
// Barrier rule: main contains no return/break/continue/discard, and each barrier starts its own
// statement whose enclosing blocks are exactly main and the uniform K loop.
void barrierStructure(const std::string& code) {
    const std::string mainBody = block(code, "void main(");
    for (const auto& t : tokens(mainBody))
        check(t != "return" && t != "break" && t != "continue" && t != "discard", "early exit '" + t + "' in main");
    const std::vector<std::string> expected = {"void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)",
                                               "for (uint begin = 0; begin < inputWidth; begin += TILE)"};
    std::vector<std::string> stack; std::string statement; int parens = 0, barriers = 0;
    const std::string barrier = "GroupMemoryBarrierWithGroupSync";
    for (size_t i = 0; i < mainBody.size(); ++i) {
        char c = mainBody[i];
        if (mainBody.compare(i, barrier.size(), barrier) == 0) {
            check(trim(statement).empty(), "barrier governed by a braceless statement: " + trim(statement));
            check(stack == expected, "barrier is not directly inside the uniform K loop");
            ++barriers;
        }
        if (c == '(') ++parens;
        if (c == ')') --parens;
        if (c == '{') { stack.push_back(trim(statement)); statement.clear(); continue; }
        if (c == '}') { check(!stack.empty(), "unbalanced main"); stack.pop_back(); statement.clear(); continue; }
        if (c == ';' && parens == 0) { statement.clear(); continue; }
        statement.push_back(c);
    }
    check(barriers == 2, "exactly two group barriers");
}
void forbiddenTokens(const std::string& code) {
    for (const auto& t : tokens(code)) {
        static const std::set<std::string> banned = {"half", "min16float", "min10float", "min16int", "min16uint", "double", "mad",
                                                     "fma", "dot", "include", "AllMemoryBarrierWithGroupSync", "DeviceMemoryBarrier"};
        check(!banned.count(t) && t.rfind("Wave", 0) != 0 && t.rfind("Interlocked", 0) != 0 && t.rfind("f16", 0) != 0,
              "forbidden token " + t);
    }
}
void sourceRules(const std::string& code) { barrierStructure(code); forbiddenTokens(code); }
// Each mutation of a passing source must be rejected, so the rules are not vacuous.
void rulesRejectMutations(const std::string& good) {
    auto rejects = [&](const char* name, const std::string& from, const std::string& to) {
        size_t at = good.find(from); check(at != std::string::npos, std::string("mutation anchor missing: ") + name);
        std::string bad = good; bad.replace(at, from.size(), to);
        bool rejected = false;
        try { sourceRules(bad); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, std::string("source rule accepted mutation: ") + name);
    };
    const std::string ownerBlock = "        if (owner) {\n            uint row = thread * STRIDE;";
    rejects("barrier inside owner branch", ownerBlock, "        if (owner) {\n            GroupMemoryBarrierWithGroupSync();\n            uint row = thread * STRIDE;");
    rejects("braceless conditional barrier", "        GroupMemoryBarrierWithGroupSync();\n        if (owner)",
            "        if (owner) GroupMemoryBarrierWithGroupSync();\n        if (owner)");
    rejects("early return", "    bool bf16 = (flags & 1) != 0;", "    bool bf16 = (flags & 1) != 0;\n    if (thread >= ROWS) return;");
    rejects("loop break", ownerBlock, "        if (owner) {\n            if (thread == 0) break;\n            uint row = thread * STRIDE;");
    rejects("non-uniform K loop", "begin < inputWidth; begin += TILE", "begin < inputWidth + thread; begin += TILE");
    rejects("FMA contraction", "precise float p0 = a0 * w0;", "precise float p0 = mad(a0, w0, 0.0);");
    rejects("FP16 arithmetic", "float a0 = tileInput[k]", "min16float a0 = tileInput[k]");
    rejects("third barrier", "    if (owner) {\n        uint column", "    GroupMemoryBarrierWithGroupSync();\n    if (owner) {\n        uint column");
}
void hlslSources(const std::string& directory) {
    const std::string original = stripComments(readFile(directory + "/linear_gemv.hlsl"));
    sourceRules(original); // The rules accept the immutable ordered8 kernel...
    rulesRejectMutations(original); // ...and reject its unsafe mutations.
    const auto originalDefines = defines(original);
    const gv::Shape* ordered = gv::shape("ordered");
    check(originalDefines.at("ROWS") == std::to_string(ordered->outputsPerGroup) && originalDefines.at("TILE") == std::to_string(ordered->tile) &&
          originalDefines.at("STRIDE") == std::to_string(ordered->stride) && originalDefines.at("THREADS") == std::to_string(ordered->threads),
          "ordered helper entry equals linear_gemv.hlsl");
    std::string normalized[2]; int index = 0;
    for (const char* selector : {"ordered32", "ordered64"}) {
        const gv::Shape& s = *gv::shape(selector);
        const std::string raw = readFile(directory + "/" + std::string(s.shader).substr(std::string("runtime/").size()));
        check(raw.find(std::string("(CHANDRA_EXPERIMENTAL_GEMV_B1=") + selector + ")") != std::string::npos, "header names its selector");
        const std::string code = stripComments(raw);
        sourceRules(code);
        rulesRejectMutations(code);
        const auto d = defines(code);
        check(d.size() == 6 && d.at("ROWS") == std::to_string(s.outputsPerGroup) && d.at("THREADS") == std::to_string(s.threads) &&
              d.at("TILE") == std::to_string(s.tile) && d.at("TILE_SHIFT") == std::to_string(gv::staging::tileShift(s.tile)) &&
              d.at("STRIDE") == std::to_string(s.stride) && d.at("SLOTS") == std::to_string(gv::staging::slots),
              std::string("defines equal the helper shape: ") + selector);
        check(code.find("[numthreads(THREADS,1,1)]") != std::string::npos, "numthreads uses THREADS");
        // Arithmetic text identical to linear_gemv.hlsl: conversions, the owner chain, the epilogue.
        for (const char* header : {"float packed(", "float bf16_rne(", "        if (owner) {\n            uint row = thread * STRIDE;"})
            check(block(code, header) == block(original, header), std::string("arithmetic text differs from linear_gemv.hlsl: ") + header);
        check(block(code, "    if (owner) {\n        uint column") == block(original, "    if (owner) {\n        uint column"), "epilogue text");
        for (const char* line : {"precise float sum = 0.0;", "uint n = min(TILE, inputWidth - begin);", "bool owner = thread < ROWS && localOut < shardRows;"})
            check(code.find(line) != std::string::npos && original.find(line) != std::string::npos, std::string("shared line: ") + line);
        // The two variants are the same text apart from their selector and geometry defines.
        std::string n = raw;
        for (const char* name : {"ROWS", "TILE", "TILE_SHIFT", "STRIDE"}) {
            const std::string line = std::string("#define ") + name + " " + d.at(name) + "\n";
            size_t at = n.find(line); check(at != std::string::npos, "define line"); n.replace(at, line.size(), std::string("#define ") + name + " X\n");
        }
        size_t at = n.find(selector); n.replace(at, std::string(selector).size(), "orderedX");
        check(n.find(selector) == std::string::npos, "selector named once");
        normalized[index++] = n;
    }
    check(normalized[0] == normalized[1], "ordered32 and ordered64 differ beyond selector and geometry defines");
    std::cout << "HLSL sources: barrier/early-exit/token rules pass for linear_gemv.hlsl and both variants and reject 8 mutations each; "
                 "defines equal the helper; packed, bf16_rne, owner chain and epilogue text equal linear_gemv.hlsl; variants differ only in geometry\n";
}

// ---- 4. Per-token structural forecast (pinned decode shapes; not a measurement) ----
struct Chunk { uint32_t inputWidth, outputs; };
std::vector<Chunk> decodeChunks() {
    std::vector<std::pair<uint32_t, uint32_t>> matrices; // (K, N), text decoder order
    for (uint32_t layer = 0; layer < 32; ++layer) {
        if (layer % 4 == 3) matrices.insert(matrices.end(), {{2560, 8192}, {2560, 1024}, {2560, 1024}, {4096, 2560}});
        else matrices.insert(matrices.end(), {{2560, 8192}, {2560, 4096}, {2560, 32}, {2560, 32}, {4096, 2560}});
        matrices.insert(matrices.end(), {{2560, 9216}, {2560, 9216}, {9216, 2560}});
    }
    matrices.push_back({2560, 248320}); // Tied LM head.
    std::vector<Chunk> chunks;
    for (auto [k, n] : matrices) {
        const uint32_t rowsPerShard = uint32_t((128ull * 1024 * 1024) / (uint64_t(k) * 2)); // model_weights.cpp, BF16
        for (uint32_t row = 0; row < n; row += rowsPerShard) {
            const uint32_t shard = std::min(rowsPerShard, n - row);
            for (uint32_t first = 0; first < shard; first += 1024) chunks.push_back({k, std::min(1024u, shard - first)});
        }
    }
    return chunks;
}
void forecast(bool print) {
    const auto chunks = decodeChunks();
    uint64_t macs = 0, weightBytes = 0;
    for (const auto& c : chunks) { macs += uint64_t(c.inputWidth) * c.outputs; weightBytes += uint64_t(c.inputWidth) * c.outputs * 2; }
    struct Row { std::string name; uint64_t groups = 0, barriers = 0, threadMacs = 0, inputBytes = 0, sharedWrite = 0, sharedRead = 0; uint32_t owners = 0, threads = 0; };
    std::vector<Row> rows;
    { // Predecessor linear.hlsl at batch 1: 16x16 groups, two barriers per 32-element K step, every thread runs its loop.
        Row r{"predecessor", 0, 0, 0, 0, 0, 0, 16, 256};
        for (const auto& c : chunks) {
            uint64_t g = (c.outputs + 15) / 16, steps = (c.inputWidth + 31) / 32;
            r.groups += g; r.barriers += g * 2 * steps; r.threadMacs += g * 256 * c.inputWidth; r.inputBytes += g * c.inputWidth * 4;
            r.sharedWrite += g * 2 * 16 * 32 * 4 * steps; r.sharedRead += g * 256 * 2 * 4ull * c.inputWidth;
        }
        rows.push_back(r);
    }
    for (const gv::Shape& s : gv::shapes) {
        // This legacy forecast models the ordered chains only. The parallel32 tree, extra barriers
        // and shared reduction traffic are modelled in the candidate's separate source report.
        if (s.route == gv::Route::parallel32) continue;
        Row r{s.selector, 0, 0, 0, 0, 0, 0, s.outputsPerGroup, s.threads};
        for (const auto& c : chunks) {
            uint64_t g = s.groups(c.outputs);
            r.groups += g; r.barriers += g * 2 * s.tiles(c.inputWidth); r.threadMacs += uint64_t(c.inputWidth) * c.outputs;
            r.inputBytes += g * c.inputWidth * 4; r.sharedWrite += uint64_t(c.inputWidth) * c.outputs * 4 + g * c.inputWidth * 4;
            r.sharedRead += uint64_t(c.inputWidth) * c.outputs * 8;
        }
        rows.push_back(r);
    }
    // The published ordered8/predecessor forecast (docs/directcompute-gemv-candidate.md) must reproduce.
    check(chunks.size() == 1431 && weightBytes == 8409579520ull && macs == 4204789760ull, "pinned decode shapes");
    check(rows[0].groups == 86278 && rows[0].barriers == 16425920 && rows[0].threadMacs == 67280568320ull && rows[0].inputBytes == 1051258880ull,
          "published predecessor forecast");
    check(rows[1].groups == 172547 && rows[1].barriers == 2053150 && rows[1].threadMacs == 4204789760ull && rows[1].inputBytes == 2102425600ull,
          "published ordered8 forecast");
    if (!print) { std::cout << "per-token forecast reproduces the published predecessor and ordered8 counts\n"; return; }
    std::cout << "B1 decode token: " << chunks.size() << " linear dispatches, " << weightBytes << " unique BF16 weight bytes, " << 2 * macs
              << " FP32 FLOPs (" << macs << " multiply-adds)\n";
    std::cout << "route,groups,owner_lanes/threads,barriers,thread_multiply_adds,input_staging_bytes,groupshared_bytes_written,groupshared_bytes_read\n";
    for (const auto& r : rows)
        std::cout << r.name << "," << r.groups << "," << r.owners << "/" << r.threads << "," << r.barriers << "," << r.threadMacs << ","
                  << r.inputBytes << "," << r.sharedWrite << "," << r.sharedRead << "\n";
    std::map<uint32_t, uint32_t> byOutputs; std::map<uint32_t, uint64_t> bytesByWidth;
    for (const auto& c : chunks) { ++byOutputs[c.outputs]; bytesByWidth[c.inputWidth] += uint64_t(c.inputWidth) * c.outputs * 2; }
    std::cout << "dispatches by outputs:"; for (auto [o, count] : byOutputs) std::cout << " " << o << "x" << count; std::cout << "\n";
    std::cout << "weight bytes by K:"; for (auto [k, b] : bytesByWidth) std::cout << " K" << k << "=" << b; std::cout << "\n";
    for (const gv::Shape& s : gv::shapes) {
        if (s.route == gv::Route::parallel32) continue;
        std::cout << s.selector << " at K9216xN1024: groups " << s.groups(1024) << ", tiles " << s.tiles(9216) << ", barriers per group "
                  << 2 * s.tiles(9216) << ", owner steps per barrier pair " << s.tile << ", input bytes " << uint64_t(s.groups(1024)) * 9216 * 4 << "\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--forecast")) {
            std::cerr << "usage: gemv-variants-test SHADER_RUNTIME_DIRECTORY [--forecast]\n"; return 2;
        }
        if (argc == 3) { forecast(true); return 0; }
        selectorTable();
        hlslSources(argv[1]);
        stagingInvariants();
        forecast(false);
        std::cout << "ALL GEMV VARIANT HELPER CHECKS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
