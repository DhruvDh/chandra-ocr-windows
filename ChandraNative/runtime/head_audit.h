// New code, MPL-2.0. Opt-in tied output-head storage audit (inference CLI --head-row-audit) for the
// native-1 failure: generated IDs alternated 52427/0 from the last-prefill prediction, 52427 is the
// final row of embed_tokens shard 1 (26214 rows of 5120 bytes per 128 MiB shard), and both the tied
// head projection and the embedding lookup read that row. The audit locates the first invalid stage;
// it repairs nothing, changes no default path and is never numerical or OCR acceptance.
//
// Bounds: checksum dispatches cover at most 1024 rows of at most 4608 packed words (K <= 9216 BF16);
// one checksum word per vocabulary row is read back (993,280 bytes for the pinned head); at most 32
// full rows (5120 bytes each) are copied per checkpoint; at most 2 predictions are audited. Source
// bytes are streamed in 4 MiB chunks. Portable: no D3D or Windows code; the CLI supplies the source.
// Evidence: the whole-row checksum is a 32-bit screen (a mismatch shows the checksum read differs, a
// match proves nothing); byte identity or difference is proven only for copied rows, word by word.
#pragma once
#include "api.h"
#include "text_model.h"
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::dc::head_audit {
using Json = nlohmann::json;
constexpr const char* schema = "chandra.directcompute.head-row-audit.v1";
constexpr const char* checksumShader = "runtime/weight_row_checksum.hlsl";
constexpr const char* copyShader = "runtime/weight_row_words.hlsl";
constexpr uint32_t maximumDispatchRows = 1024, maximumRowWords = 4608, maximumRowDumps = 32;
constexpr uint32_t maximumPredictions = 2, maximumListedRows = 64, checksumThreads = 64, copyThreads = 256;
constexpr uint64_t sourceChunkBytes = 4ull << 20;

// --- Arithmetic contracts (host, CPU tests and HLSL state the same operations) ---

// weight_row_checksum.hlsl: sum over packed words i of (2i+1)*word, uint32 wrap, ascending i.
// Every multiplier is odd, hence invertible modulo 2^32, so any change confined to one word of a
// row always changes that row's checksum. Multi-word changes can cancel, so a match is never taken as
// identity: only the exact word comparison of copied rows proves it, for those rows alone.
inline uint32_t rowChecksum(const uint32_t* words, uint32_t count) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) sum += (2u * i + 1u) * words[i];
    return sum;
}
inline float asFloat(uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return f; }
inline uint32_t asBits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
// Element k of a packed BF16 row: low half of word k/2 for even k, as in linear*.hlsl/embedding.hlsl.
inline float bf16At(const uint32_t* words, uint32_t k) { return asFloat(((words[k >> 1] >> ((k & 1) * 16)) & 0xffffu) << 16); }
// SM5 flushes FP32 denormals to sign-preserved zero on arithmetic inputs and results.
inline float flush(float v) { return std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; }
// bf16_rne() of linear.hlsl/linear_gemv.hlsl/norm.hlsl.
inline float bf16Rne(float value) {
    uint32_t bits = asBits(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return asFloat((bits & 0xffff0000u) | 0x00400000u);
    return asFloat((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}
// One head output exactly as linear.hlsl and linear_gemv.hlsl ("ordered") compute it with the
// graph's roundOutputBF16=true and no bias: sum=+0, sum=sum+(y[k]*w[k]) for ascending k, then RNE.
inline float headLogit(const float* y, const uint32_t* rowWords, uint32_t cols) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < cols; ++k) {
        const float product = flush(flush(y[k]) * flush(bf16At(rowWords, k)));
        sum = flush(sum + product);
    }
    return bf16Rne(sum);
}
// Higham (2002) Theorem 3.1: recursive FP32 dot of n terms has |error| <= gamma_n * sum|y_k w_k|.
inline double dotGamma(uint32_t n) { const double nu = double(n) * std::ldexp(1.0, -24); return nu / (1.0 - nu); }
// Absolute bound on |headLogit - exact dot| for given sum|y_k w_k| and sum(|y_k|+|w_k|): gamma_n,
// then BF16 RNE (relative 2^-9, taken as 2^-8 to absorb (1+gamma_n)), plus denormal flushes of at
// most 2^-126 per flushed operand times the other factor's magnitude and per flushed result.
inline double headErrorBound(double absoluteDot, double operandL1, uint32_t n) {
    return absoluteDot * (dotGamma(n) + std::ldexp(1.0, -8)) + (operandL1 + 2.0 * n) * std::ldexp(1.0, -126);
}
// Cauchy-Schwarz: sum|y_k w_k| <= ||y||2 ||w||2, so any correct head logit for authenticated row w obeys
// |logit| <= ||y||2 ||w||2 (1 + gamma_n + 2^-8) + flush term. Shape/dtype derived, not a logit cutoff.
inline double logitContract(double yNorm, double rowNorm, double operandL1, uint32_t n) {
    const double dot = yNorm * rowNorm;
    return dot + headErrorBound(dot, operandL1, n);
}
// RMSNorm (norm.hlsl, onePlusWeight): y_d = bf16(x_d * rsqrt(sum(x^2)/N + eps) * (1+w_d)). Exactly,
// sum (x_d s)^2 = N sum x^2 / (sum x^2 + N eps) <= N, so ||y||2 <= sqrt(N) max|1+w_d|. The factor
// 1+2^-6 envelopes gamma_N for the sum (1.6e-4 at N=2560), division/addition/product roundings, SM5
// rsqrt implementation error and BF16 RNE (2^-9). Overflowed sum(x^2) gives scale 0 and y == 0.
inline double normContract(uint32_t width, double maximumGain) { return std::sqrt(double(width)) * maximumGain * (1.0 + std::ldexp(1.0, -6)); }

struct RowStats { double l2 = 0, l1 = 0, maximumAbsolute = 0; uint32_t nonfinite = 0; };
inline RowStats rowStats(const uint32_t* words, uint32_t cols) {
    RowStats s; double squares = 0;
    for (uint32_t k = 0; k < cols; ++k) {
        const double v = bf16At(words, k);
        if (!std::isfinite(v)) { ++s.nonfinite; continue; }
        squares += v * v; s.l1 += std::fabs(v); s.maximumAbsolute = std::max(s.maximumAbsolute, std::fabs(v));
    }
    s.l2 = std::sqrt(squares); return s;
}
// greedy() of inference_core.cpp: strict greater keeps the first index of the maximum.
struct Ranked { uint32_t best = 0, runnerUp = 0; float bestValue = 0, runnerUpValue = 0; uint32_t ties = 0; };
inline Ranked rank(const std::vector<float>& logits) {
    if (logits.size() < 2) throw std::invalid_argument("Head audit ranking needs at least two logits");
    Ranked r; r.bestValue = logits[0];
    for (uint32_t i = 1; i < logits.size(); ++i) if (logits[i] > r.bestValue) { r.bestValue = logits[i]; r.best = i; }
    bool found = false;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == r.bestValue) ++r.ties;
        if (i != r.best && (!found || logits[i] > r.runnerUpValue)) { r.runnerUp = i; r.runnerUpValue = logits[i]; found = true; }
    }
    return r;
}

// --- First-invalid-stage classification over recorded facts (pure; tested directly) ---
// Storage evidence at one checkpoint. Copied rows are compared word by word with the authenticated
// bytes; that comparison, not the 32-bit screen, decides every storage stage the Auditor can reach.
struct StorageFacts {
    bool screened = false, compared = false;  // Every row's checksum read back; every selected row copied and compared.
    uint64_t screenMismatchRows = 0;          // GPU checksum differs from the authenticated row's.
    uint32_t comparedRows = 0;                // Rows copied and compared word by word (<= 32).
    uint32_t differingScreenMismatched = 0;   // Copied bytes differ and the checksum mismatched too.
    uint32_t differingScreenMatched = 0;      // Copied bytes differ although the checksum matched (collision or disagreeing reads).
    uint32_t identicalScreenMismatched = 0;   // Copied bytes identical although the checksum mismatched (disagreeing reads).
};
struct PredictionFacts {
    StorageFacts storage;
    bool hiddenObserved = false, hiddenFinite = true, hiddenWithinNormContract = true, hiddenZero = false;
    bool headObserved = false, headMatchesRecompute = true; // Recompute: argmax and runner-up within headErrorBound of the exact dot.
    uint64_t logitContractViolations = 0;     // Rows whose GPU logit exceeds the Cauchy-Schwarz contract.
    bool embeddingObserved = false, embeddingMatchesSource = true; // embedding() of the argmax equals its authenticated BF16 row.
};
// Predictions holds every started prediction in order; a failed one keeps the stages it observed.
struct Facts {
    uint32_t requestedPredictions = 0, completedPredictions = 0; bool importAudited = false, failed = false;
    StorageFacts import; std::vector<PredictionFacts> predictions;
};
struct StageResult { std::string name; bool observed = false, flagged = false; };
// checkpoint < 0 is the import checkpoint. Exact differences outrank screen disagreements; a screen
// mismatch with no copied row (unreachable in Auditor, which copies mismatching rows) counts alone.
inline StageResult storageStage(const StorageFacts& s, int checkpoint) {
    const std::string at = checkpoint < 0 ? "@after_import" : "@prediction" + std::to_string(checkpoint);
    if (s.differingScreenMismatched || s.differingScreenMatched) {
        if (checkpoint < 0) return {"tied_head_storage_differs_after_import", true, true};
        // A clean import screen plus a mismatch now means the checksum read changed since import; a
        // matching checksum cannot tell whether the copied difference already existed at import.
        return {(s.differingScreenMismatched ? "tied_head_storage_changed_after_import" : "tied_head_row_bytes_differ_with_matching_checksum") + at, true, true};
    }
    if (s.identicalScreenMismatched) return {"checksum_screen_disagrees_with_exact_row_copy" + at, true, true};
    if (!s.screened || !s.compared) return {"", false, false};
    if (s.screenMismatchRows) return {checkpoint < 0 ? "tied_head_storage_differs_after_import" : "tied_head_storage_changed_after_import" + at, true, true};
    return {"", true, false};
}
// Dataflow order: storage, final-norm row, head projection, embedding of the argmax (next input).
// An all-zero final-norm row is not provably invalid (a zero pre-norm row gives it legitimately and
// no pre-norm row is recorded), so it is named last, only when its checkpoint is otherwise clean.
inline std::vector<StageResult> predictionStages(const PredictionFacts& p, size_t index) {
    const std::string at = "@prediction" + std::to_string(index);
    return {storageStage(p.storage, int(index)),
        {"final_norm_hidden_outside_rmsnorm_contract" + at, p.hiddenObserved, p.hiddenObserved && (!p.hiddenFinite || !p.hiddenWithinNormContract)},
        {"head_projection_disagrees_with_authenticated_rows" + at, p.headObserved, p.headObserved && (!p.headMatchesRecompute || p.logitContractViolations)},
        {"embedding_lookup_differs_from_authenticated_row" + at, p.embeddingObserved, p.embeddingObserved && !p.embeddingMatchesSource},
        {"final_norm_hidden_all_zero_cause_unknown" + at, p.hiddenObserved, p.hiddenObserved && p.hiddenZero}};
}
inline std::vector<StageResult> stageWalk(const Facts& f) {
    std::vector<StageResult> walk{storageStage(f.import, -1)};
    for (size_t i = 0; i < f.predictions.size(); ++i) for (auto& s : predictionStages(f.predictions[i], i)) walk.push_back(std::move(s));
    return walk;
}
inline bool auditComplete(const Facts& f) {
    return f.importAudited && !f.failed && f.requestedPredictions && f.completedPredictions == f.requestedPredictions && f.predictions.size() == f.requestedPredictions;
}
// Every observed flagged stage, including any past an unobserved one; never a first-stage claim.
inline std::vector<std::string> findings(const Facts& f) {
    std::vector<std::string> out; for (const auto& s : stageWalk(f)) if (s.observed && s.flagged) out.push_back(s.name); return out;
}
// The earliest flagged stage with every earlier stage observed clean; an unobserved stage or an
// unaudited requested prediction ahead of any finding is audit_incomplete, never a clean result.
inline std::string firstInvalidStage(const Facts& f) {
    for (const auto& s : stageWalk(f)) { if (!s.observed) return "audit_incomplete"; if (s.flagged) return s.name; }
    return auditComplete(f) ? "no_invalid_value_in_audited_head_path" : "audit_incomplete";
}
inline Json hypotheses() {
    return {
        {"tied_head_storage_differs_after_import", "A copied row differs from its authenticated bytes once upload/import drained (or, with no row copied, its checksum does): upload or initial residency, or a read fault on the SRV path the graph shares"},
        {"tied_head_storage_changed_after_import", "Every checksum matched after import; at this prediction a row's checksum mismatches and its copy differs from the authenticated bytes: later change of resident bytes, residency migration, an unexpected writer, or a read fault on the shared SRV path"},
        {"tied_head_row_bytes_differ_with_matching_checksum", "A copied row differs from its authenticated bytes although its 32-bit checksum matched here and at import: a checksum collision or disagreeing reads. Whether the difference existed at import is unknown; nothing was copied there"},
        {"checksum_screen_disagrees_with_exact_row_copy", "A row's checksum mismatched but its copy equals the authenticated bytes: the two GPU reads disagree (checksum dispatch fault, transient read fault or bytes changing between reads); no storage difference is established"},
        {"final_norm_hidden_outside_rmsnorm_contract", "Final-norm row nonfinite or beyond sqrt(N)*max|1+w|: norm or upstream hidden-state fault"},
        {"head_projection_disagrees_with_authenticated_rows", "A GPU logit is not the dot of the observed final-norm row with the authenticated row. As the first stage, every row copied at this checkpoint (argmax, runner-up, contract-violating rows) was identical and no checksum mismatched: projection read or arithmetic fault"},
        {"embedding_lookup_differs_from_authenticated_row", "embedding() of the argmax differs from its authenticated row. As the first stage, that row's copy was identical: embedding read-path fault"},
        {"final_norm_hidden_all_zero_cause_unknown", "Every final-norm value is exactly zero. A zero pre-norm row gives this legitimately; FP32 overflow of sum(x^2) gives exact zeros too, for example from a finite huge stored embedding row. No pre-norm row is recorded, so no cause is assigned"},
        {"audit_incomplete", "A requested checkpoint or one of its stages was not audited before the first finding: nothing is claimed about unaudited stages and this is not a clean result"},
        {"no_invalid_value_in_audited_head_path", "Every requested checkpoint completed with no provably invalid value: not a reproduction or acceptance; identity is proven only for copied rows, the rest passed a 32-bit screen"}};
}

// Source of authenticated bytes: offsets are relative to the safetensors payload. The CLI holds a
// read-only, non-write-shared handle opened before the importer's whole-file SHA-256.
using SourceRead = std::function<void(uint64_t payloadOffset, void* destination, uint32_t bytes)>;
using RowSink = std::function<void(const std::string& name, const void* data, size_t bytes)>;
using Digest = std::function<std::string(const void*, size_t)>;
struct Tensor { uint64_t payloadBegin = 0; }; // data_offsets[0] of the tensor in the payload.

// --- Device-level audit (Device/Weight from api.h; Windows D3D11 in the CLI, a fake in CPU tests) ---
// How the observed run ended, as its caller saw it. Recorded only; completeness comes from the audit's own captures.
enum class RunEnd { unreported, completed, failed };
class Auditor {
public:
    Auditor(Device& d, const Weight& head, Tensor headSource, Tensor normSource, uint32_t normWidth, uint32_t requestedPredictions,
            SourceRead source, RowSink sink, Digest digest, std::function<Json()> memory)
        : device(d), weight(head), headTensor(headSource), normTensor(normSource), width(normWidth), read(std::move(source)),
          write(std::move(sink)), sha(std::move(digest)), observeMemory(std::move(memory)) {
        if (!weight.bf16 || !weight.rows || !weight.cols || weight.cols % 2 || weight.cols / 2 > maximumRowWords || weight.cols != width ||
            weight.shards.empty() || weight.shards.size() != weight.shardFirstRows.size() || !read || !write || !sha)
            throw std::invalid_argument("Head audit needs a packed BF16 head with even K <= 9216 equal to the norm width, and source/sink/digest");
        if (!requestedPredictions || requestedPredictions > maximumPredictions) throw std::invalid_argument("Head audit requests 1 or 2 predictions");
        facts.requestedPredictions = requestedPredictions; rowWords = weight.cols / 2; uint64_t next = 0;
        for (size_t s = 0; s < weight.shards.size(); ++s) {
            const auto& b = weight.shards[s];
            if (!b.storage || !b.packedBF16 || weight.shardFirstRows[s] != next || b.logicalElements % weight.cols ||
                b.words != b.logicalElements / 2 || uint64_t(b.words) * 4 > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("Head audit shard metadata differs from the importer contract");
            next += b.logicalElements / weight.cols;
        }
        if (next != weight.rows) throw std::invalid_argument("Head audit shards do not cover the vocabulary");
        report = {{"schema", schema}, {"rows", weight.rows}, {"cols", weight.cols}, {"shards", Json::array()},
            {"scope", "Opt-in bounded diagnostic of the tied embed_tokens/lm_head storage and its two readers; not a repair, numerical acceptance, OCR or throughput evidence"},
            {"evidence", "Every row: 32-bit checksum screen (a mismatch shows the checksum read differs; a match is not identity). Copied rows only: exact word comparison with the authenticated bytes. Both reads use the SRV path the graph uses, so neither separates VRAM contents from a read fault, and the audit's own reads may change residency"},
            {"bounds", {{"rows_per_checksum_dispatch", maximumDispatchRows}, {"words_per_row_max", maximumRowWords}, {"row_dumps_per_checkpoint", maximumRowDumps},
                {"audited_predictions", maximumPredictions}, {"checksum_readback_bytes", uint64_t(weight.rows) * 4}, {"source_chunk_bytes", sourceChunkBytes}}},
            {"checkpoints", Json::array()}};
        for (size_t s = 0; s < weight.shards.size(); ++s)
            report["shards"].push_back({{"first_row", weight.shardFirstRows[s]}, {"rows", weight.shards[s].logicalElements / weight.cols}});
    }
    // Streams every authenticated head row once (checksums and FP64 norms), reads the final-norm gain,
    // then screens all GPU row checksums. Call after ModelWeights construction, before any request.
    Json afterImport(bool directCopy = false) {
        admit("import audit");
        if (importStarted) throw std::logic_error("Import audit already started");
        importStarted = true; Json checkpoint = {{"name", "after_import"}, {"complete", false}};
        try {
            if (directCopy) directKnownRow(checkpoint); // Before every checksum dispatch and source-table stream.
            sourceChecksums.assign(weight.rows, 0); sourceNorms.assign(weight.rows, 0); sourceL1.assign(weight.rows, 0);
            const uint32_t rowBytes = rowWords * 4, chunkRows = uint32_t(std::max<uint64_t>(1, sourceChunkBytes / rowBytes));
            std::vector<uint32_t> chunk(size_t(chunkRows) * rowWords);
            for (uint32_t first = 0; first < weight.rows;) {
                const uint32_t count = std::min(chunkRows, weight.rows - first);
                read(headTensor.payloadBegin + uint64_t(first) * rowBytes, chunk.data(), count * rowBytes);
                for (uint32_t r = 0; r < count; ++r) {
                    const uint32_t* row = chunk.data() + size_t(r) * rowWords; const auto stats = rowStats(row, weight.cols);
                    if (stats.nonfinite) throw std::runtime_error("Authenticated head row contains a nonfinite BF16 value");
                    sourceChecksums[first + r] = rowChecksum(row, rowWords); sourceNorms[first + r] = stats.l2; sourceL1[first + r] = stats.l1;
                }
                first += count;
            }
            std::vector<uint32_t> gain(width / 2); read(normTensor.payloadBegin, gain.data(), width * 2);
            for (uint32_t k = 0; k < width; ++k) maximumGain = std::max(maximumGain, std::fabs(double(1.0f + bf16At(gain.data(), k))));
            report["source"] = {{"rows_streamed", weight.rows}, {"maximum_row_l2", *std::max_element(sourceNorms.begin(), sourceNorms.end())},
                {"final_norm_maximum_one_plus_gain", maximumGain}, {"rmsnorm_hidden_l2_contract", normContract(width, maximumGain)}};
            importScreen = storage(checkpoint, "after_import", {}, {}, facts.import);
            facts.importAudited = true; checkpoint["complete"] = true; return push(std::move(checkpoint));
        } catch (const std::exception& e) { failed("after_import", e.what(), &checkpoint); throw; }
    }
    // TextObserver tail: keeps the final-norm row of each call and audits the first requested predictions.
    // Refuses observations before a completed import, after a failure or after a terminal finish.
    void observe(const TextObservation& o) {
        admit("observation");
        if (!facts.importAudited) { failed("observation", "Head audit observation before a completed import audit", nullptr); throw std::logic_error("Head audit observation before a completed import audit"); }
        if (facts.completedPredictions >= facts.requestedPredictions) return;
        try {
            if (o.stage == TextStage::FinalNorm && o.first + o.count == o.callTokens) {
                if (o.width != width || !o.count) throw std::runtime_error("Head audit final-norm shape differs");
                auto h = device.readFloats(o.value);
                if (h.size() < size_t(o.count) * width) throw std::runtime_error("Head audit final-norm readback incomplete");
                hidden.assign(h.end() - width, h.end()); hiddenCall = {o.tokensBefore, o.generatedBefore, o.callTokens};
            } else if (o.stage == TextStage::Logits) {
                if (o.rows != 1 || o.width != weight.rows || hidden.size() != width) throw std::runtime_error("Head audit needs one last-row prediction after its final norm");
                if (hiddenCall != std::array<uint32_t, 3>{o.tokensBefore, o.generatedBefore, o.callTokens}) throw std::runtime_error("Head audit final-norm row belongs to a different call than the logits");
                auto row = std::move(hidden); hidden.clear(); // A failed prediction never leaves a reusable row.
                prediction(device.readFloats(o.value), row, o);
            }
        } catch (const std::exception& e) { failed("prediction" + std::to_string(facts.predictions.size() - (predictionOpen ? 1 : 0)), e.what(), nullptr); predictionOpen = false; throw; }
    }
    // Report over recorded facts. A completed or failed run end is recorded once and closes the audit;
    // repeating it returns the same report, a conflicting one is refused. unreported never closes.
    Json finish(RunEnd end = RunEnd::unreported, const std::string& error = {}) {
        if (end == RunEnd::completed && !error.empty()) throw std::logic_error("A completed run end carries no error");
        if (end != RunEnd::unreported) {
            const std::string reason = end == RunEnd::failed && error.empty() ? "run failed without an error message" : error;
            if (closed && (end != runEnd || reason != runError)) throw std::logic_error("Head audit run end already recorded differently");
            closed = true; runEnd = end; runError = reason;
        }
        const bool complete = auditComplete(facts);
        report["predictions_requested"] = facts.requestedPredictions; report["predictions_started"] = facts.predictions.size();
        report["predictions_audited"] = facts.completedPredictions; report["import_audited"] = facts.importAudited;
        report["audit_complete"] = complete; report["audit_errors"] = errors;
        report["run_end"] = {{"state", runEnd == RunEnd::completed ? "completed" : runEnd == RunEnd::failed ? "failed" : "unreported"},
            {"error", runError.empty() ? Json(nullptr) : Json(runError)}};
        report["incomplete_reason"] = complete ? Json(nullptr) : !facts.importAudited ? Json("import audit not completed")
            : facts.failed ? Json("audit error: " + errors.at(0).at("error").get<std::string>())
            : Json(std::to_string(facts.completedPredictions) + " of " + std::to_string(facts.requestedPredictions) + " requested predictions audited");
        report["findings"] = findings(facts); report["first_invalid_stage"] = firstInvalidStage(facts);
        report["classification"] = "first_invalid_stage is the earliest flagged stage (import, then each prediction: storage, final-norm contract, head, embedding, all-zero final norm) with every earlier stage observed clean; findings lists every observed flagged stage";
        report["hypotheses"] = hypotheses();
        return report;
    }
    const Facts& recorded() const { return facts; }

private:
    Device& device; const Weight& weight; Tensor headTensor, normTensor; uint32_t width = 0, rowWords = 0;
    SourceRead read; RowSink write; Digest sha; std::function<Json()> observeMemory;
    std::vector<uint32_t> sourceChecksums, importScreen; std::vector<double> sourceNorms, sourceL1; double maximumGain = 0;
    std::vector<float> hidden; std::array<uint32_t, 3> hiddenCall{}; Facts facts; Json report, errors = Json::array();
    bool importStarted = false, predictionOpen = false, closed = false; RunEnd runEnd = RunEnd::unreported; std::string runError;
    std::map<uint32_t, std::vector<uint32_t>> dumped;

    void admit(const char* what) const {
        if (closed) throw std::logic_error(std::string("Head audit ") + what + " after its terminal finish");
        if (facts.failed) throw std::logic_error(std::string("Head audit ") + what + " after an audit failure");
    }
    // Records the first and later errors; a partially built checkpoint is kept, marked incomplete.
    void failed(const std::string& at, const std::string& what, Json* partial) {
        facts.failed = true; errors.push_back({{"at", at}, {"error", what}});
        if (partial) { (*partial)["error"] = what; push(std::move(*partial)); }
    }
    // Device memory is context, not an audit fact: a failing query is recorded, never thrown.
    Json push(Json checkpoint) {
        if (observeMemory) { try { checkpoint["device_memory"] = observeMemory(); } catch (const std::exception& e) { checkpoint["device_memory_error"] = e.what(); } }
        report["checkpoints"].push_back(checkpoint); return checkpoint;
    }
    std::pair<size_t, uint32_t> locate(uint32_t row) const {
        for (size_t s = weight.shards.size(); s-- > 0;) if (row >= weight.shardFirstRows[s]) return {s, row - weight.shardFirstRows[s]};
        throw std::out_of_range("Head audit row outside the vocabulary");
    }
    void directKnownRow(Json& checkpoint) {
        using namespace direct_head_row;
        Json& out = checkpoint["direct_copy"];
        out = {{"schema", "chandra.directcompute.head-row-direct-copy.v1"}, {"requested", true}, {"complete", false},
            {"at", "after_import_before_checksum"}, {"row", row}, {"shard", 0}, {"local_row", row},
            {"file", file}, {"bytes", rowBytes}, {"raw_file_written", false}, {"affects_existing_classification", false},
            {"scope", "One CopySubresourceRegion from the underlying imported buffer to staging; no SRV, shader, cast or arithmetic; may change residency; no cause or numerical/OCR acceptance"}};
        Observation observed;
        auto record = [&] {
            const auto& r = observed.transport;
            out["raw_file_written"] = observed.rawFileWritten; out["source_read_completed"] = observed.sourceReadCompleted;
            out["complete"] = observed.complete;
            if (!observed.gpuSha256.empty()) out["gpu_sha256"] = observed.gpuSha256;
            if (!observed.sourceSha256.empty()) out["source_sha256"] = observed.sourceSha256;
            if (observed.complete) {
                out["differing_words"] = observed.differingWords; out["exact_match"] = observed.differingWords == 0;
                out["first_differing_word"] = observed.firstDifferingWord == UINT32_MAX ? Json(nullptr) : Json(observed.firstDifferingWord);
            }
            out["transport"] = {{"api", "ID3D11DeviceContext::CopySubresourceRegion"}, {"source_first_byte", r.extent.firstByte}, {"copy_bytes", r.extent.bytes},
                {"source_logical_bytes", r.extent.logicalBytes}, {"source_physical_bytes", r.extent.physicalBytes}, {"staging_bytes", r.extent.bytes},
                {"source_descriptor", {{"usage", r.sourceUsage}, {"bind_flags", r.sourceBindFlags}, {"misc_flags", r.sourceMiscFlags}, {"cpu_access_flags", r.sourceCPUAccess}, {"structure_byte_stride", r.sourceStride}}},
                {"copies_submitted", r.copiesSubmitted}, {"copy_resubmitted", false}, {"before_copy_drain_completed", r.beforeCopyDrainCompleted}, {"after_copy_drain_completed", r.afterCopyDrainCompleted},
                {"drain_timeout_ms", 10000}, {"post_copy_readiness_budget_ms", 10000}, {"map_attempted", r.mapAttempted},
                {"map_status", r.mapAttempted ? Json(uint32_t(r.readiness.status)) : Json(nullptr)}, {"map_attempts", r.readiness.attempts}, {"still_drawing_results", r.readiness.stillDrawing},
                {"map_wait_nanoseconds", r.readiness.waited.count()}, {"map_copied", r.mapAttempted && r.readiness.result == readback::Result::copied},
                {"unmapped", r.unmapped}, {"staging_created", r.stagingCreated}, {"staging_released", r.stagingReleased},
                {"device_removed_reason_queried", r.deviceRemovedReasonQueried}, {"device_removed_reason", r.deviceRemovedReasonQueried ? Json(uint32_t(r.deviceRemovedReason)) : Json(nullptr)},
                {"operation_nanoseconds", r.operationNanoseconds},
                {"retirement_scope", "Local observer staging reference release only; request/worker/Job retirement requires external ownership receipts"}};
        };
        try {
            const auto& b = weight.shards.at(0);
            requireKnown(weight.rows, weight.cols, weight.shardFirstRows.at(0), b.logicalElements, b.words, b.packedBF16);
            capture([&](Receipt& r) { return device.readWordsRange(b, firstWord, rowWords, r); }, write, read, sha,
                headTensor.payloadBegin + uint64_t(row) * rowBytes, observed);
            record();
        } catch (const std::exception& e) { record(); out["error"] = e.what(); throw; }
        catch (...) { record(); out["error"] = "non-standard exception"; throw; }
    }
    std::vector<uint32_t> gpuChecksums() {
        struct Params { uint32_t wordsPerRow, rows, firstRow, outputFirst, reserved0, reserved1, reserved2, reserved3; };
        static_assert(sizeof(Params) == 32, "HLSL cbuffer ABI");
        Buffer out = device.words(weight.rows);
        for (size_t s = 0; s < weight.shards.size(); ++s) {
            const uint32_t shardRows = uint32_t(weight.shards[s].logicalElements / weight.cols);
            for (uint32_t first = 0; first < shardRows;) {
                const uint32_t count = std::min(shardRows - first, maximumDispatchRows);
                Params p{rowWords, count, first, weight.shardFirstRows[s] + first, 0, 0, 0, 0};
                device.dispatch(checksumShader, {&weight.shards[s]}, {&out}, &p, sizeof(p), (count + checksumThreads - 1) / checksumThreads);
                first += count;
            }
        }
        auto words = device.readWords(out);
        if (words.size() != weight.rows) throw std::runtime_error("Head audit checksum readback incomplete");
        return words;
    }
    std::vector<uint32_t> gpuRows(const std::vector<uint32_t>& rows) {
        struct Params { uint32_t firstWord, count, outputFirst, reserved0, reserved1, reserved2, reserved3, reserved4; };
        static_assert(sizeof(Params) == 32, "HLSL cbuffer ABI");
        if (rows.empty() || rows.size() > maximumRowDumps) throw std::logic_error("Head audit row dump count outside 1..32");
        Buffer out = device.words(uint32_t(rows.size()) * rowWords);
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto [s, local] = locate(rows[i]);
            Params p{local * rowWords, rowWords, uint32_t(i) * rowWords, 0, 0, 0, 0, 0};
            device.dispatch(copyShader, {&weight.shards[s]}, {&out}, &p, sizeof(p), (rowWords + copyThreads - 1) / copyThreads);
        }
        auto words = device.readWords(out);
        if (words.size() != rows.size() * rowWords) throw std::runtime_error("Head audit row readback incomplete");
        return words;
    }
    // Every authenticated row read after streaming is checked against its streamed checksum.
    std::vector<uint32_t> sourceRow(uint32_t row) {
        std::vector<uint32_t> words(rowWords); read(headTensor.payloadBegin + uint64_t(row) * rowWords * 4, words.data(), rowWords * 4);
        if (rowChecksum(words.data(), rowWords) != sourceChecksums[row]) throw std::runtime_error("Head audit source row changed after streaming");
        return words;
    }
    using Selected = std::vector<std::pair<uint32_t, const char*>>;
    // Screens every row checksum, then copies (<= 32) the leading rows, the checksum-mismatching rows and
    // the trailing rows, in that order, and compares each copy word by word with the authenticated bytes.
    // Facts are recorded as observed, so a failure keeps what was seen. Returns the screen.
    std::vector<uint32_t> storage(Json& checkpoint, const std::string& name, const Selected& leading, const Selected& trailing, StorageFacts& s) {
        const auto gpu = gpuChecksums(); std::vector<uint32_t> mismatched;
        for (uint32_t r = 0; r < weight.rows; ++r) if (gpu[r] != sourceChecksums[r]) mismatched.push_back(r);
        Json listed = Json::array();
        for (size_t i = 0; i < mismatched.size() && i < maximumListedRows; ++i) listed.push_back(mismatched[i]);
        checkpoint["checksum_mismatch_rows"] = mismatched.size(); checkpoint["first_mismatch_rows"] = listed;
        if (!importScreen.empty()) {
            uint64_t changed = 0; for (uint32_t r = 0; r < weight.rows; ++r) changed += gpu[r] != importScreen[r];
            checkpoint["checksum_changed_since_import_rows"] = changed;
        }
        s.screenMismatchRows = mismatched.size(); s.screened = true;
        std::vector<uint32_t> dumps; std::map<uint32_t, Json> reasons;
        auto select = [&](uint32_t r, const char* why) {
            if (auto it = reasons.find(r); it != reasons.end()) it->second.push_back(why);
            else if (dumps.size() < maximumRowDumps) { dumps.push_back(r); reasons[r] = Json::array({why}); }
        };
        for (const auto& [r, why] : leading) select(r, why);
        for (auto r : mismatched) select(r, "checksum_mismatch");
        for (const auto& [r, why] : trailing) select(r, why);
        Json identical = Json::array(), differing = Json::array(), collided = Json::array(), disagreeing = Json::array(); uint64_t mismatchedCompared = 0;
        auto summarize = [&] {
            checkpoint["exact_comparison"] = {{"rows_selected", dumps.size()}, {"rows_compared", s.comparedRows}, {"rows_identical", identical}, {"rows_differing", differing},
                {"differing_rows_with_matching_checksum", collided}, {"identical_rows_with_mismatching_checksum", disagreeing},
                {"checksum_mismatch_rows_not_compared", mismatched.size() - mismatchedCompared},
                {"scope", "Byte identity or difference is proven for compared rows only; every other row passed or failed a 32-bit checksum screen"}};
        };
        checkpoint["row_dumps"] = Json::array(); dumped.clear(); summarize();
        if (!dumps.empty()) {
            const auto words = gpuRows(dumps);
            for (size_t i = 0; i < dumps.size(); ++i) {
                const uint32_t row = dumps[i], *g = words.data() + i * rowWords; const auto src = sourceRow(row);
                uint32_t differingWords = 0, firstDiffering = UINT32_MAX;
                for (uint32_t w = 0; w < rowWords; ++w) if (g[w] != src[w]) { ++differingWords; if (firstDiffering == UINT32_MAX) firstDiffering = w; }
                const auto gs = rowStats(g, weight.cols); const auto [shard, local] = locate(row);
                const std::string file = name + ".row" + std::to_string(row) + ".gpu.bf16"; write(file, g, size_t(rowWords) * 4);
                const bool screenMatches = gpu[row] == sourceChecksums[row], exact = differingWords == 0;
                checkpoint["row_dumps"].push_back({{"row", row}, {"reasons", reasons.at(row)}, {"shard", shard}, {"local_row", local}, {"file", file},
                    {"gpu_sha256", sha(g, size_t(rowWords) * 4)}, {"source_sha256", sha(src.data(), src.size() * 4)},
                    {"gpu_checksum", gpu[row]}, {"copied_checksum", rowChecksum(g, rowWords)}, {"source_checksum", sourceChecksums[row]},
                    {"checksum_matches_source", screenMatches}, {"exact_match", exact},
                    {"differing_words", differingWords}, {"first_differing_word", firstDiffering == UINT32_MAX ? Json(nullptr) : Json(firstDiffering)},
                    {"gpu_l2", gs.nonfinite ? Json(nullptr) : Json(gs.l2)}, {"gpu_max_abs", gs.maximumAbsolute}, {"gpu_nonfinite", gs.nonfinite}, {"source_l2", sourceNorms[row]}});
                ++s.comparedRows; mismatchedCompared += !screenMatches;
                if (exact) identical.push_back(row); else differing.push_back(row);
                if (!exact && screenMatches) { ++s.differingScreenMatched; collided.push_back(row); }
                if (!exact && !screenMatches) ++s.differingScreenMismatched;
                if (exact && !screenMatches) { ++s.identicalScreenMismatched; disagreeing.push_back(row); }
                dumped[row] = std::vector<uint32_t>(g, g + rowWords); summarize();
            }
        }
        s.compared = true; return gpu;
    }
    void prediction(const std::vector<float>& logits, const std::vector<float>& y, const TextObservation& o) {
        const uint32_t index = uint32_t(facts.predictions.size()); const std::string name = "prediction" + std::to_string(index);
        facts.predictions.emplace_back(); predictionOpen = true; PredictionFacts& p = facts.predictions.back();
        Json c = {{"name", name}, {"complete", false}, {"call", {{"tokens_before", o.tokensBefore}, {"generated_before", o.generatedBefore}, {"call_tokens", o.callTokens}}}};
        try {
            if (logits.size() != weight.rows) throw std::runtime_error("Head audit logits readback incomplete");
            const auto ranked = rank(logits);
            c["argmax"] = ranked.best; c["argmax_logit"] = ranked.bestValue; c["runner_up"] = ranked.runnerUp; c["runner_up_logit"] = ranked.runnerUpValue; c["maximum_tie_count"] = ranked.ties;
            double yNorm = 0, yL1 = 0; bool zero = true;
            for (float v : y) { if (!std::isfinite(v)) p.hiddenFinite = false; yNorm += double(v) * v; yL1 += std::fabs(double(v)); zero = zero && v == 0.0f; }
            yNorm = std::sqrt(yNorm); p.hiddenZero = zero;
            p.hiddenWithinNormContract = p.hiddenFinite && yNorm <= normContract(width, maximumGain); p.hiddenObserved = true;
            c["final_norm_hidden"] = {{"finite", p.hiddenFinite}, {"l2", p.hiddenFinite ? Json(yNorm) : Json(nullptr)}, {"rmsnorm_contract", normContract(width, maximumGain)}, {"all_zero", zero},
                {"pre_norm_row_recorded", false}};
            Json violations = Json::array(); Selected violating;
            for (uint32_t r = 0; r < weight.rows; ++r) {
                const double limit = logitContract(yNorm, sourceNorms[r], yL1 + sourceL1[r], weight.cols);
                if (!(std::fabs(double(logits[r])) <= limit)) {
                    if (violations.size() < maximumListedRows) violations.push_back({{"row", r}, {"logit", logits[r]}, {"contract", limit}});
                    if (violating.size() < maximumRowDumps) violating.push_back({r, "logit_contract_violation"});
                    ++p.logitContractViolations;
                }
            }
            c["logit_contract_violations"] = p.logitContractViolations; c["first_logit_contract_violations"] = violations;
            storage(c, name, {{ranked.best, "argmax"}, {ranked.runnerUp, "runner_up"}}, violating, p.storage);
            for (auto& v : c["first_logit_contract_violations"]) { auto it = dumped.find(v["row"].get<uint32_t>()); v["exact_match"] = it == dumped.end() ? Json(nullptr) : Json(it->second == sourceRow(it->first)); }
            Json recompute = Json::array(); bool matches = true;
            for (uint32_t r : {ranked.best, ranked.runnerUp}) {
                const auto& g = dumped.at(r); const auto src = sourceRow(r); double dot = 0, absolute = 0, operands = yL1;
                for (uint32_t k = 0; k < weight.cols; ++k) { const double w = bf16At(src.data(), k); dot += double(y[k]) * w; absolute += std::fabs(double(y[k]) * w); operands += std::fabs(w); }
                const float fromGpu = p.hiddenFinite ? headLogit(y.data(), g.data(), weight.cols) : std::numeric_limits<float>::quiet_NaN();
                const float fromSource = p.hiddenFinite ? headLogit(y.data(), src.data(), weight.cols) : std::numeric_limits<float>::quiet_NaN();
                const double bound = headErrorBound(absolute, operands, weight.cols);
                const bool within = std::fabs(double(logits[r]) - dot) <= bound; matches = matches && within;
                recompute.push_back({{"row", r}, {"gpu_logit", logits[r]}, {"cpu_ordered_from_gpu_row", fromGpu}, {"cpu_ordered_from_source_row", fromSource},
                    {"bitwise_equal_gpu_row", asBits(fromGpu) == asBits(logits[r])}, {"bitwise_equal_source_row", asBits(fromSource) == asBits(logits[r])},
                    {"exact_dot_source_row", dot}, {"a_priori_error_bound", bound}, {"within_bound", within}});
            }
            c["head_recompute"] = recompute; p.headMatchesRecompute = matches; p.headObserved = true;
            auto looked = device.readFloats(embedding(device, weight, {ranked.best}));
            const auto src = sourceRow(ranked.best); uint32_t differing = 0;
            if (looked.size() != weight.cols) throw std::runtime_error("Head audit embedding readback incomplete");
            for (uint32_t k = 0; k < weight.cols; ++k) if (asBits(looked[k]) != asBits(bf16At(src.data(), k))) ++differing;
            p.embeddingMatchesSource = differing == 0; p.embeddingObserved = true;
            double squares = 0, largest = 0; for (float v : looked) { squares += double(v) * v; largest = std::max(largest, std::fabs(double(v))); }
            c["embedding_lookup"] = {{"token", ranked.best}, {"elements_differing_from_source", differing}, {"l2", std::sqrt(squares)}, {"max_abs", largest},
                {"fp64_sum_of_squares", squares}, {"fp64_sum_of_squares_exceeds_fp32_max", squares > double(std::numeric_limits<float>::max())}};
            c["complete"] = true; c["findings"] = stageFindings(p, index);
            ++facts.completedPredictions; predictionOpen = false; push(std::move(c));
        } catch (const std::exception& e) { c["error"] = e.what(); c["findings"] = stageFindings(p, index); push(std::move(c)); throw; }
    }
    static Json stageFindings(const PredictionFacts& p, uint32_t index) {
        Json out = Json::array(); for (const auto& stage : predictionStages(p, index)) if (stage.observed && stage.flagged) out.push_back(stage.name); return out;
    }
};

// The CLI's head_row_audit entry until an Auditor exists, so a run that fails before the audit starts never reads as clean.
inline Json notStarted(uint32_t requestedPredictions) {
    return {{"schema", schema}, {"predictions_requested", requestedPredictions}, {"predictions_started", 0}, {"predictions_audited", 0}, {"import_audited", false},
        {"audit_complete", false}, {"incomplete_reason", "auditor not started"}, {"findings", Json::array()}, {"first_invalid_stage", "audit_incomplete"}, {"hypotheses", hypotheses()}};
}
// Terminal report for the CLI's failure paths. A failing finish is reported, never thrown over the run's own error.
inline Json finishAfterFailure(Auditor& audit, const std::string& error) {
    try { return audit.finish(RunEnd::failed, error); }
    catch (const std::exception& e) { return {{"schema", schema}, {"audit_complete", false}, {"first_invalid_stage", "audit_incomplete"}, {"finish_error", e.what()}, {"run_error", error}}; }
}
} // namespace chandra::dc::head_audit
