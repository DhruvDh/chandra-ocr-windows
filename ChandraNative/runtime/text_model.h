// New code, MPL-2.0. Pinned Qwen3.5 B1 graph; unvalidated until numerical qualification.
#pragma once
#include "api.h"
#include <array>
namespace chandra::dc {
struct TextPositions { std::vector<uint32_t> temporal,height,width; };
struct TextLayerCache { Buffer keys,values,conv,recurrent; uint32_t length=0; bool failed=false; };
struct TextRequest {
 std::array<TextLayerCache,32> layers; uint32_t tokens=0,generated=0;
 uint32_t contextLimit=16384,outputLimit=12384; bool failed=false,retired=false;
};
struct TextResult { Buffer hidden,logits; uint32_t rows=0; };
class TextModel {
 Device& device; ModelWeights& weights;
 Buffer mix(uint32_t,const Buffer&,uint32_t,const TextPositions&,TextLayerCache&);
 Buffer layer(uint32_t,const Buffer&,uint32_t,const TextPositions&,TextLayerCache&);
public:
 TextModel(Device&,ModelWeights&);
 TextRequest newRequest();
 // Drops owned cache buffers. Failed or retired requests cannot be resumed.
 void retire(TextRequest&) noexcept;
 // Embeddings [tokens,2560] may be freshly merged GPU vision/text embeddings.
 TextResult prefill(TextRequest&,const Buffer& embeddings,const TextPositions&,bool logitsLastOnly=true);
 TextResult prefill(TextRequest&,const std::vector<uint32_t>& ids,const TextPositions&,bool logitsLastOnly=true);
 TextResult advance(TextRequest&,uint32_t tokenId,const TextPositions&);
 // Own cache supplied explicitly: this operation makes no whole-graph claim.
 Buffer diagnosticLayer(uint32_t,const Buffer&,uint32_t,const TextPositions&,TextLayerCache&);
};
}
