// New code, MPL-2.0. Pinned Chandra/Qwen3.5 vision graph; DirectCompute only.
#pragma once
#include "api.h"
#include <functional>

namespace chandra::dc {
struct VisionGrid { uint32_t temporal, height, width; };
struct VisionLimits {
    uint32_t maximumPatches=32768, maximumFramePatches=32768;
    uint32_t rowTile=512, queryTile=32, keyTile=128;
    uint64_t maximumWorkspaceBytes=2ull*1024*1024*1024;
};
struct VisionForecast {
    uint32_t patchRows=0, mergedRows=0, frames=0, maximumFrameRows=0;
    uint64_t hiddenBufferBytes=0, rotaryBufferBytes=0, attentionScoreBytes=0, attentionPartialBytes=0, rowIntermediateBytes=0;
    uint64_t maximumSingleBufferBytes=0, workspaceUpperBoundBytes=0, visionWeightBytes=0;
    uint64_t attentionDispatchesPerBlock=0, attentionScoreMultiplyAddsPerDispatch=0;
    uint64_t attentionValueMultiplyAddsPerDispatch=0;
    uint64_t attentionMultiplyAddsPerBlock=0;
    // Shared linear/norm/elementwise dispatches are additional and depend on weight shards.
};
struct VisionOutput {
    Buffer lastHiddenState; // [patchRows,1024], expanded BF16 values in R32_FLOAT storage.
    Buffer poolerOutput;    // Full forward: [patchRows/4,2560]. First-block diagnostic: empty.
    VisionForecast forecast;
};
// Callback buffers are borrowed observations. Retaining them increases the forecasted live set.
// The graph drains bounded completed chunks/tiles/blocks; the root diagnostic owns readback.
using VisionObserver=std::function<void(const std::string&,const Buffer&,uint32_t,uint32_t)>;

class VisionModel {
    Device& device; const ModelWeights& weights; VisionLimits limits;
    VisionOutput run(const float*,uint32_t,const std::vector<VisionGrid>&,bool,const VisionObserver&);
public:
    VisionModel(Device&,const ModelWeights&,VisionLimits={});
    // Pure source-side geometry/resource forecast; call before admitting any native execution.
    static VisionForecast forecast(uint32_t patchRows,const std::vector<VisionGrid>&,VisionLimits={});
    // Input is the original processor tensor in [patchRows,3*2*16*16] contiguous order.
    // No pixel resize, regrouping, CPU numerical fallback, or cross-frame attention occurs.
    // Any dispatch/drain/observer exception requires root-owned worker retirement, not retry on that device.
    VisionOutput firstBlock(const float*,uint32_t patchRows,const std::vector<VisionGrid>&,const VisionObserver& = {});
    VisionOutput forward(const float*,uint32_t patchRows,const std::vector<VisionGrid>&,const VisionObserver& = {});
};
}
