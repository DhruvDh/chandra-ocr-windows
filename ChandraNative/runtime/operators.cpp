// New code, MPL-2.0. See api.h: R32_FLOAT activations, explicit BF16 boundaries.
#include "api.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace chandra::dc {
namespace {
constexpr uint32_t maxGroups = 65535;
uint32_t elements(uint64_t count) {
    if (!count || count > std::numeric_limits<uint32_t>::max() / 4u)
        throw std::invalid_argument("Activation exceeds one raw-buffer address range");
    return static_cast<uint32_t>(count);
}
void activation(const Buffer& b, uint64_t count) {
    if (!b.storage || b.packedBF16 || b.logicalElements != count || uint64_t(b.words) < count)
        throw std::invalid_argument("Exact R32_FLOAT activation shape/storage required");
    elements(count);
}
void weight(const Weight& w) {
    if (!w.rows || !w.cols || w.shards.empty() || w.shards.size() != w.shardFirstRows.size())
        throw std::invalid_argument("Complete row-sharded weight required");
    uint64_t next = 0;
    for (size_t s = 0; s < w.shards.size(); ++s) {
        const auto& b = w.shards[s];
        if (!b.storage || b.packedBF16 != w.bf16 || w.shardFirstRows[s] != next ||
            !b.logicalElements || b.logicalElements % w.cols)
            throw std::invalid_argument("Weight shard gaps, overlap, dtype or partial row");
        if (b.logicalElements > std::numeric_limits<uint32_t>::max() / (w.bf16 ? 2ull : 4ull))
            throw std::invalid_argument("Weight shard byte addressing overflows uint32");
        const uint64_t bytes = b.logicalElements * (w.bf16 ? 2ull : 4ull);
        const uint64_t words = w.bf16 ? (b.logicalElements + 1) / 2 : b.logicalElements;
        if (bytes > std::numeric_limits<uint32_t>::max() || words != b.words || words > std::numeric_limits<uint32_t>::max() / 4u)
            throw std::invalid_argument("Weight shard raw byte range/packed count differs");
        next += b.logicalElements / w.cols;
    }
    if (next != w.rows) throw std::invalid_argument("Weight shards do not cover all output rows");
}
const Buffer& vectorWeight(const Weight& w, uint32_t width) {
    weight(w);
    if (uint64_t(w.rows) * w.cols != width || w.shards.size() != 1)
        throw std::invalid_argument("Norm/bias vector needs one complete width-element shard");
    return w.shards.front();
}
struct LinearParams {
    uint32_t inputWidth, outputWidth, batchRows, firstOutput;
    uint32_t shardRows, rowFirst, flags, weightRowFirst;
};
struct EmbeddingParams {
    uint32_t width, rows, firstWeightRow, shardRows;
    uint32_t tokenFirst, flags, reserved0, reserved1;
};
struct NormParams {
    uint32_t width, rows, rowFirst, flags;
    float epsilon; uint32_t layerNorm, reserved0, reserved1;
};
struct ElementParams { uint32_t count, first, operation, flags; uint32_t secondCount, reserved0, reserved1, reserved2; };
static_assert(sizeof(LinearParams) == 32 && sizeof(EmbeddingParams) == 32 &&
              sizeof(NormParams) == 32 && sizeof(ElementParams) == 32, "HLSL cbuffer ABI");
// Experimental B1 projection selector, captured once during this translation unit's static
// initialization (before main) and never re-read. Unset or "0" keeps the predecessor route;
// "ordered" admits linear_gemv.hlsl for rows == 1. Any other value, including an empty value,
// is refused by every linear() call before validation or allocation; there is no fallback.
enum class GemvB1 { predecessor, ordered, invalid };
struct GemvSelection { GemvB1 mode; char observed[33]; };
GemvSelection readGemvSelection() noexcept {
    GemvSelection result{GemvB1::predecessor, {}};
    const char* value = nullptr;
#ifdef _MSC_VER
    char* owned = nullptr; size_t length = 0;
    if (_dupenv_s(&owned, &length, "CHANDRA_EXPERIMENTAL_GEMV_B1") != 0) { result.mode = GemvB1::invalid; return result; }
    value = owned;
#else
    value = std::getenv("CHANDRA_EXPERIMENTAL_GEMV_B1");
#endif
    if (value) {
        result.mode = std::strcmp(value, "0") == 0 ? GemvB1::predecessor
                    : std::strcmp(value, "ordered") == 0 ? GemvB1::ordered : GemvB1::invalid;
        for (size_t i = 0; i + 1 < sizeof(result.observed) && value[i]; ++i)
            result.observed[i] = value[i] > 31 && value[i] < 127 ? value[i] : '?';
    }
#ifdef _MSC_VER
    std::free(owned);
#endif
    return result;
}
const GemvSelection gemvSelection = readGemvSelection();
bool orderedGemvB1() {
    if (gemvSelection.mode == GemvB1::invalid)
        throw std::invalid_argument(std::string("CHANDRA_EXPERIMENTAL_GEMV_B1 must be unset, \"0\" or \"ordered\"; observed \"") +
                                    gemvSelection.observed + "\"");
    return gemvSelection.mode == GemvB1::ordered;
}
constexpr uint32_t gemvOutputsPerGroup = 8; // linear_gemv.hlsl ROWS
}

namespace experimental {
// Not part of api.h: calibration and candidate tests declare this accessor themselves.
const char* gemvB1Selection() { return orderedGemvB1() ? "ordered" : "predecessor"; }
}

Buffer linear(Device& d, const Buffer& input, const Weight& w, uint32_t rows,
              bool roundOutputBF16, const Weight* bias) {
    const bool gemv = orderedGemvB1() && rows == 1;
    weight(w);
    if (w.cols > 9216) throw std::invalid_argument("Linear input width exceeds pinned graph maximum 9216");
    activation(input, uint64_t(rows) * w.cols);
    const Buffer* biasBuffer = bias ? &vectorWeight(*bias, w.rows) : &w.shards.front();
    Buffer output = d.floats(elements(uint64_t(rows) * w.rows));
    // Conservative TDR baseline: no split-K, so each dot keeps its complete order.
    // At K=9216 this bounds one call to 1024*32*9216 = 301,989,888 terms.
    for (size_t s = 0; s < w.shards.size(); ++s) {
        uint32_t totalShardRows = static_cast<uint32_t>(w.shards[s].logicalElements / w.cols);
        for (uint32_t weightFirst = 0; weightFirst < totalShardRows;) {
            uint32_t currentOutputs = std::min<uint32_t>(totalShardRows-weightFirst,1024);
            if (gemv) {
                // Same <=1024-output, complete-K bound as the predecessor; 8 outputs per group.
                LinearParams p{w.cols,w.rows,1,w.shardFirstRows[s]+weightFirst,currentOutputs,0,
                    uint32_t(w.bf16) | (uint32_t(roundOutputBF16)<<1) | (uint32_t(bias!=nullptr)<<2) |
                    (uint32_t(bias && bias->bf16)<<3),weightFirst};
                d.dispatch("runtime/linear_gemv.hlsl",{&input,&w.shards[s],biasBuffer},{&output},
                           &p,sizeof(p),(currentOutputs+gemvOutputsPerGroup-1)/gemvOutputsPerGroup);
                weightFirst+=currentOutputs;
                continue;
            }
            for (uint32_t first = 0; first < rows;) {
                uint32_t current = std::min<uint32_t>(rows-first,32);
                LinearParams p{w.cols,w.rows,current,w.shardFirstRows[s]+weightFirst,currentOutputs,first,
                    uint32_t(w.bf16) | (uint32_t(roundOutputBF16)<<1) | (uint32_t(bias!=nullptr)<<2) |
                    (uint32_t(bias && bias->bf16)<<3),weightFirst};
                d.dispatch("runtime/linear.hlsl",{&input,&w.shards[s],biasBuffer},{&output},
                           &p,sizeof(p),(currentOutputs+15)/16,(current+15)/16);
                first+=current;
            }
            weightFirst+=currentOutputs;
        }
    }
    return output;
}

Buffer embedding(Device& d, const Weight& w, const std::vector<uint32_t>& ids) {
    weight(w);
    if (ids.empty()) throw std::invalid_argument("Embedding token list is empty");
    for (uint32_t id : ids) if (id >= w.rows) throw std::out_of_range("Embedding token outside complete vocabulary");
    uint32_t count = elements(uint64_t(ids.size()) * w.cols);
    if ((uint64_t(w.cols)+255)/256 > maxGroups)
        throw std::invalid_argument("Embedding dispatch dimensions exceed SM5 limit");
    Buffer tokens = d.words(static_cast<uint32_t>(ids.size()),ids.data());
    Buffer output = d.floats(count);
    for (size_t s = 0; s < w.shards.size(); ++s) {
        for (uint32_t first=0;first<ids.size();) {
            uint32_t current=std::min<uint32_t>(static_cast<uint32_t>(ids.size())-first,maxGroups);
            EmbeddingParams p{w.cols,current,w.shardFirstRows[s],
                static_cast<uint32_t>(w.shards[s].logicalElements/w.cols),first,uint32_t(w.bf16),0,0};
            d.dispatch("runtime/embedding.hlsl",{&tokens,&w.shards[s]},{&output},&p,sizeof(p),(w.cols+255)/256,current);
            first+=current;
        }
    }
    return output;
}

Buffer rmsNorm(Device& d, const Buffer& input, const Weight& scale, uint32_t rows,
               uint32_t width, float epsilon, bool onePlusWeight, bool roundOutputBF16) {
    activation(input,uint64_t(rows)*width); const Buffer& w=vectorWeight(scale,width);
    if (!(epsilon>0) || !std::isfinite(epsilon)) throw std::invalid_argument("Finite positive norm epsilon required");
    Buffer output=d.floats(elements(uint64_t(rows)*width));
    for(uint32_t first=0;first<rows;) {
        uint32_t count=std::min(rows-first,maxGroups);
        NormParams p{width,count,first,uint32_t(scale.bf16)|(uint32_t(roundOutputBF16)<<1)|(uint32_t(onePlusWeight)<<2),epsilon,0,0,0};
        d.dispatch("runtime/norm.hlsl",{&input,&w,&w},{&output},&p,sizeof(p),count); first+=count;
    }
    return output;
}

Buffer layerNorm(Device& d, const Buffer& input, const Weight& scale, const Weight& bias,
                 uint32_t rows, uint32_t width, float epsilon, bool roundOutputBF16) {
    activation(input,uint64_t(rows)*width); const Buffer& w=vectorWeight(scale,width);const Buffer& b=vectorWeight(bias,width);
    if (!(epsilon>0) || !std::isfinite(epsilon)) throw std::invalid_argument("Finite positive norm epsilon required");
    Buffer output=d.floats(elements(uint64_t(rows)*width));
    for(uint32_t first=0;first<rows;) {
        uint32_t count=std::min(rows-first,maxGroups);
        NormParams p{width,count,first,uint32_t(scale.bf16)|(uint32_t(roundOutputBF16)<<1)|(uint32_t(bias.bf16)<<3),epsilon,1,0,0};
        d.dispatch("runtime/norm.hlsl",{&input,&w,&b},{&output},&p,sizeof(p),count);first+=count;
    }
    return output;
}

Buffer binary(Device& d,const Buffer& a,const Buffer& b,uint32_t count,uint32_t operation,bool roundOutputBF16) {
    activation(a,count);activation(b,b.logicalElements);
    if(operation>4 || b.logicalElements>count || count%b.logicalElements)
        throw std::invalid_argument("Binary operation or explicit repeating right-operand shape invalid");
    Buffer output=d.floats(elements(count));
    for(uint32_t first=0;first<count;) {
        uint32_t current=std::min(count-first,maxGroups*256);
        ElementParams p{current,first,operation,uint32_t(roundOutputBF16),static_cast<uint32_t>(b.logicalElements),0,0,0};
        d.dispatch("runtime/elementwise.hlsl",{&a,&b},{&output},&p,sizeof(p),(current+255)/256);first+=current;
    }
    return output;
}
Buffer unary(Device& d,const Buffer& a,uint32_t count,uint32_t operation,bool roundOutputBF16) {
    activation(a,count);if(operation>4)throw std::invalid_argument("Unknown unary operation");
    Buffer output=d.floats(elements(count));
    for(uint32_t first=0;first<count;) {
        uint32_t current=std::min(count-first,maxGroups*256);
        ElementParams p{current,first,operation,uint32_t(roundOutputBF16)|2u,0,0,0,0};
        d.dispatch("runtime/elementwise.hlsl",{&a,&a},{&output},&p,sizeof(p),(current+255)/256);first+=current;
    }
    return output;
}
}
