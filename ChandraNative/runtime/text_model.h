// New code, MPL-2.0. Pinned Qwen3.5 B1 graph; unvalidated until numerical qualification.
#pragma once
#include "api.h"
#include <array>
#include <functional>
namespace chandra::dc {
struct TextPositions { std::vector<uint32_t> temporal,height,width; };
struct TextLayerCache { Buffer keys,values,conv,recurrent; uint32_t length=0; bool failed=false; };
struct TextRequest {
 std::array<TextLayerCache,32> layers; uint32_t tokens=0,generated=0;
 uint32_t contextLimit=16384,outputLimit=12384; bool failed=false,retired=false;
};
struct TextResult { Buffer hidden,logits; uint32_t rows=0; };
// Explicit fixed two-slot decode call. An inactive slot is not submitted and stays untouched.
// Pointers/positions are copied at call entry; requests must remain alive on this Device thread.
struct TextDecodeSlot { TextRequest* request=nullptr; uint32_t tokenId=0; TextPositions position; bool active=false; };
struct TextDecodeCohortResult { std::array<TextResult,2> slots; std::array<bool,2> active{}; };
// Diagnostic view of an existing value after the graph's own drain. value is [rows,width].
// LayerOutput/FinalNorm/Logits row i is call row first+i, absolute causal row tokensBefore+first+i;
// positions/tokenIds are indexed by call row (tokenIds is null for embedding prefill).
// ConvState [8192,4] and RecurrentState [32 heads,128 key*128 value] follow call rows [first,first+count).
// cacheLength is the observed length after those rows: the layer's own cache for layer/state stages,
// the last layer's cache for FinalNorm and TextRequest::tokens for Logits.
enum class TextStage { LayerOutput, FinalNorm, Logits, ConvState, RecurrentState };
struct TextObservation {
 TextStage stage; uint32_t layer; const Buffer& value; uint32_t rows,width;
 uint32_t tokensBefore,generatedBefore,callTokens,first,count,cacheLength;
 const TextPositions& positions; const std::vector<uint32_t>* tokenIds;
};
// Optional and synchronous on the Device thread. An observer may read value but must not write,
// retain or dispatch with it. An observer exception poisons the request like any graph failure.
using TextObserver=std::function<void(const TextObservation&)>;
class TextModel {
 Device& device; ModelWeights& weights;
 Buffer mix(uint32_t,const Buffer&,uint32_t,const TextPositions&,TextLayerCache&);
 Buffer layer(uint32_t,const Buffer&,uint32_t,const TextPositions&,TextLayerCache&);
 TextResult forward(TextRequest&,const Buffer&,const TextPositions&,bool,const TextObserver&,const std::vector<uint32_t>*);
public:
 TextModel(Device&,ModelWeights&);
 TextRequest newRequest();
 // Drops owned cache buffers. Failed or retired requests cannot be resumed.
 void retire(TextRequest&) noexcept;
 // Embeddings [tokens,2560] may be freshly merged GPU vision/text embeddings.
 TextResult prefill(TextRequest&,const Buffer& embeddings,const TextPositions&,bool logitsLastOnly=true,const TextObserver& = {});
 TextResult prefill(TextRequest&,const std::vector<uint32_t>& ids,const TextPositions&,bool logitsLastOnly=true,const TextObserver& = {});
 TextResult advance(TextRequest&,uint32_t tokenId,const TextPositions&,const TextObserver& = {});
 // Experimental ordered B1 + separately enabled padded32 only; no endpoint or cancellation policy.
 // Requires 1–2 active, disjoint requests/caches. Results preserve slot identity and are drained.
 // Observers see each request's own one-row tensors/positions and retain the original no-write contract.
 // Graph/observer failure poisons ALL active caches, preserves the original exception and keeps
 // cache ownership with the caller. Before retire()/dropping those caches, the owner MUST prove
 // Device::drain() completion; if it cannot, retire the owning worker. No replay/resume is permitted.
 TextDecodeCohortResult advanceCohort(const std::array<TextDecodeSlot,2>&,
     bool explicitOrderedB2=false,const std::array<TextObserver,2>& = {});
 // Own cache supplied explicitly: this operation makes no whole-graph claim.
 Buffer diagnosticLayer(uint32_t,const Buffer&,uint32_t,const TextPositions&,TextLayerCache&);
};
}
