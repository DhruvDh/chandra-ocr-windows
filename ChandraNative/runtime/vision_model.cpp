// New code, MPL-2.0. Source reference: pinned Chandra revision af93b47dba1b47b6640c86ccf487ed2260ab9a09.
// Transformers 5.18 modeling_qwen3_5.py SHA256 d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939.
// Layout/casts follow that graph. HLSL trig, exp, erf, dot/softmax/norm reductions remain numerical qualification obligations.
#include "vision_model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace chandra::dc {
namespace {
constexpr uint32_t hidden=1024,patchWidth=1536,heads=16,headDim=64;
constexpr uint64_t singleCap=128ull*1024*1024;
constexpr uint32_t dispatchElements=65535u*256u;
const std::string prefix="model.visual.";
struct Frame { uint32_t first,count; };
struct Geometry { std::vector<uint32_t> positions; std::vector<Frame> frames; };
struct CopyParams { uint32_t count,inputOffset,outputOffset,pad; };
struct PositionParams { uint32_t count,first,width,pad; };
struct RopeParams { uint32_t rows,rowFirst,count,pad; };
struct AttentionParams {
    uint32_t sequenceStart,sequenceLength,queryStart,queryCount;
    uint32_t keyStart,keyCount,tileIndex,tileCount;
    uint32_t width,heads,headDim,pad;
};
struct SoftmaxParams { uint32_t queryHeads,sequenceLength,pad0,pad1; };
struct ReduceParams { uint32_t count,rowFirst,tileCount,pad; };
static_assert(sizeof(AttentionParams)==48 && sizeof(CopyParams)==16 && sizeof(RopeParams)==16,"Vision HLSL ABI");

uint32_t count32(uint64_t n) {
    if(!n || n*4>singleCap || n>std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Vision raw buffer exceeds nonempty 128 MiB limit");
    return uint32_t(n);
}
void checkLimits(const VisionLimits& l) {
    if(!l.maximumPatches || l.maximumPatches>32768 || !l.maximumFramePatches || l.maximumFramePatches>32768 ||
       !l.rowTile || l.rowTile>512 || !l.queryTile || l.queryTile>128 || l.keyTile!=128 || !l.maximumWorkspaceBytes)
        throw std::invalid_argument("Vision requires <=32768 patches/frame, <=512 row tile, <=128 query tile and 128 key tile");
}
uint64_t weightElements() {
    const uint64_t block=3ull*hidden*hidden+3*hidden + uint64_t(hidden)*hidden+hidden + 4*hidden +
        2ull*hidden*4096+4096+hidden;
    return uint64_t(hidden)*patchWidth+hidden + 2304ull*hidden + 24*block +
        2*hidden+4096ull*4096+4096+2560ull*4096+2560;
}
Geometry geometry(uint32_t n,const std::vector<VisionGrid>& grids) {
    Geometry g;g.positions.reserve(size_t(n)*4);g.frames.reserve(n/4);
    uint64_t total=0;
    for(const auto& grid:grids) {
        const uint64_t frame=uint64_t(grid.height)*grid.width;
        if(!grid.temporal || !grid.height || !grid.width || grid.height%2 || grid.width%2 ||
           !frame || frame>32768 || uint64_t(grid.temporal)*frame>n-total)
            throw std::invalid_argument("Exact positive even spatial grid and matching processor row count required");
        for(uint32_t t=0;t<grid.temporal;t++) {
            g.frames.push_back({uint32_t(total),uint32_t(frame)});
            for(uint32_t within=0;within<frame;within++) {
                const uint32_t blocksW=grid.width/2;
                const uint32_t row=(within/(4*blocksW))*2+(within/2)%2;
                const uint32_t col=((within/4)%blocksW)*2+within%2;
                g.positions.insert(g.positions.end(),{row,col,grid.height,grid.width});
            }
            total+=frame;
        }
    }
    if(total!=n || grids.empty())throw std::invalid_argument("gridTHW sum differs from original processor patch tensor");
    return g;
}
void observe(const VisionObserver& observer,const std::string& name,const Buffer& b,uint32_t rows,uint32_t width) {
    if(observer)observer(name,b,rows,width);
}
const Weight& matrix(const ModelWeights& ws,std::string name,uint32_t rows,uint32_t cols) {
    const auto& w=ws.at(prefix+name);
    if(!w.bf16 || w.rows!=rows || w.cols!=cols || w.shards.empty() || w.shards.size()!=w.shardFirstRows.size())
        throw std::invalid_argument("Pinned BF16 vision matrix geometry differs: "+name);
    uint64_t next=0;
    for(size_t i=0;i<w.shards.size();i++) {
        const auto& s=w.shards[i];
        if(!s.storage || !s.packedBF16 || w.shardFirstRows[i]!=next || !s.logicalElements || s.logicalElements%cols ||
           s.words!=(s.logicalElements+1)/2 || uint64_t(s.words)*4>singleCap)
            throw std::invalid_argument("Vision weight shard coverage/storage differs: "+name);
        next+=s.logicalElements/cols;
    }
    if(next!=rows)throw std::invalid_argument("Vision weight output rows differ: "+name);
    return w;
}
const Weight& vector(const ModelWeights& ws,std::string name,uint32_t width) {
    const auto& w=ws.at(prefix+name);
    if(uint64_t(w.rows)*w.cols!=width || w.shards.size()!=1)
        throw std::invalid_argument("Vision vector shape/sharding differs: "+name);
    return matrix(ws,name,w.rows,w.cols);
}
void validateWeights(const ModelWeights& ws,bool full) {
    matrix(ws,"patch_embed.proj.weight",hidden,patchWidth);vector(ws,"patch_embed.proj.bias",hidden);
    const auto& pos=matrix(ws,"pos_embed.weight",2304,hidden);
    if(pos.shards.size()!=1)throw std::invalid_argument("Small vision learned position table needs one complete shard");
    for(uint32_t i=0;i<(full?24u:1u);i++) {
        const auto b="blocks."+std::to_string(i)+".";
        for(auto norm:{"norm1.","norm2."}) { vector(ws,b+norm+"weight",hidden);vector(ws,b+norm+"bias",hidden); }
        matrix(ws,b+"attn.qkv.weight",3072,hidden);vector(ws,b+"attn.qkv.bias",3072);
        matrix(ws,b+"attn.proj.weight",hidden,hidden);vector(ws,b+"attn.proj.bias",hidden);
        matrix(ws,b+"mlp.linear_fc1.weight",4096,hidden);vector(ws,b+"mlp.linear_fc1.bias",4096);
        matrix(ws,b+"mlp.linear_fc2.weight",hidden,4096);vector(ws,b+"mlp.linear_fc2.bias",hidden);
    }
    if(full) {
        vector(ws,"merger.norm.weight",hidden);vector(ws,"merger.norm.bias",hidden);
        matrix(ws,"merger.linear_fc1.weight",4096,4096);vector(ws,"merger.linear_fc1.bias",4096);
        matrix(ws,"merger.linear_fc2.weight",2560,4096);vector(ws,"merger.linear_fc2.bias",2560);
    }
}
void copy(Device& d,const Buffer& source,Buffer& dest,uint32_t count,uint32_t inputOffset=0,uint32_t outputOffset=0) {
    if(source.packedBF16 || dest.packedBF16 || uint64_t(inputOffset)+count>source.logicalElements ||
       uint64_t(outputOffset)+count>dest.logicalElements || source.storage==dest.storage)
        throw std::invalid_argument("Exact nonaliasing vision copy extent required");
    for(uint32_t first=0;first<count;) {
        uint32_t current=std::min(count-first,dispatchElements);
        CopyParams p{current,inputOffset+first,outputOffset+first,0};
        d.dispatch("runtime/vision_rows.hlsl",{&source},{&dest},&p,sizeof(p),(current+255)/256);first+=current;
    }
}
Buffer gather(Device& d,const Buffer& source,uint32_t rowFirst,uint32_t rows,uint32_t width) {
    Buffer result=d.floats(count32(uint64_t(rows)*width));copy(d,source,result,rows*width,rowFirst*width);return result;
}
Buffer projection(Device& d,const ModelWeights& ws,const Buffer& source,uint32_t rows,uint32_t inputWidth,
                  uint32_t outputWidth,const std::string& name,uint32_t rowTile) {
    Buffer result=d.floats(count32(uint64_t(rows)*outputWidth));
    const auto& w=matrix(ws,name+".weight",outputWidth,inputWidth);const auto& b=vector(ws,name+".bias",outputWidth);
    for(uint32_t first=0;first<rows;first+=rowTile) {
        uint32_t count=std::min(rowTile,rows-first);
        auto input=gather(d,source,first,count,inputWidth);
        auto output=linear(d,input,w,count,true,&b);copy(d,output,result,count*outputWidth,0,first*outputWidth);
        d.drain(); // Pending Storage ownership retires only after proven completion.
    }
    return result;
}
struct QKV { Buffer q,k,v; };
QKV qkv(Device& d,const ModelWeights& ws,const Buffer& normalized,const Buffer& rotary,uint32_t n,
        const std::string& block,uint32_t rowTile) {
    QKV result{d.floats(count32(uint64_t(n)*hidden)),d.floats(count32(uint64_t(n)*hidden)),d.floats(count32(uint64_t(n)*hidden))};
    const auto& w=matrix(ws,block+"attn.qkv.weight",3072,hidden);const auto& b=vector(ws,block+"attn.qkv.bias",3072);
    for(uint32_t first=0;first<n;first+=rowTile) {
        uint32_t count=std::min(rowTile,n-first);
        auto input=gather(d,normalized,first,count,hidden);auto combined=linear(d,input,w,count,true,&b);
        RopeParams p{count,first,count*hidden,0};
        d.dispatch("runtime/vision_rope.hlsl",{&combined,&rotary},{&result.q,&result.k,&result.v},&p,sizeof(p),(p.count+255)/256);
        d.drain();
    }
    return result;
}
Buffer attention(Device& d,const QKV& x,uint32_t n,const std::vector<Frame>& frames,const VisionLimits& l) {
    Buffer result=d.floats(count32(uint64_t(n)*hidden));
    for(const auto& frame:frames) {
        uint32_t tiles=(frame.count+l.keyTile-1)/l.keyTile;
        for(uint32_t first=0;first<frame.count;first+=l.queryTile) {
            uint32_t queries=std::min(l.queryTile,frame.count-first),qh=queries*heads;
            auto scores=d.floats(count32(uint64_t(qh)*frame.count));
            auto partials=d.floats(count32(uint64_t(queries)*hidden*tiles));
            AttentionParams p{frame.first,frame.count,first,queries,0,0,0,tiles,hidden,heads,headDim,0};
            d.dispatch("runtime/vision_attention_scores.hlsl",{&x.q,&x.k},{&scores},&p,sizeof(p),tiles,qh);
            SoftmaxParams softmax{qh,frame.count,0,0};
            d.dispatch("runtime/vision_attention_softmax.hlsl",{},{&scores},&softmax,sizeof(softmax),qh);
            d.dispatch("runtime/vision_attention_values.hlsl",{&scores,&x.v},{&partials},&p,sizeof(p),tiles,qh);
            ReduceParams reduce{queries*hidden,frame.first+first,tiles,0};
            d.dispatch("runtime/vision_attention_reduce.hlsl",{&partials},{&result},&reduce,sizeof(reduce),(reduce.count+255)/256);
            d.drain(); // Scores/partials are owned through this tile's EVENT completion, then released.
        }
    }
    return result;
}
Buffer mlp(Device& d,const ModelWeights& ws,const Buffer& normalized,uint32_t n,const std::string& block,
           uint32_t rowTile,const VisionObserver& observer) {
    Buffer result=d.floats(count32(uint64_t(n)*hidden));
    const auto& up=matrix(ws,block+"mlp.linear_fc1.weight",4096,hidden);const auto& ub=vector(ws,block+"mlp.linear_fc1.bias",4096);
    const auto& down=matrix(ws,block+"mlp.linear_fc2.weight",hidden,4096);const auto& db=vector(ws,block+"mlp.linear_fc2.bias",hidden);
    for(uint32_t first=0;first<n;first+=rowTile) {
        uint32_t count=std::min(rowTile,n-first);auto input=gather(d,normalized,first,count,hidden);
        auto upValue=linear(d,input,up,count,true,&ub);auto activated=unary(d,upValue,count*4096,1,true);
        auto output=linear(d,activated,down,count,true,&db);copy(d,output,result,count*hidden,0,first*hidden);
        d.drain();
        observe(observer,block+"mlp.fc1.tile."+std::to_string(first),upValue,count,4096);
        observe(observer,block+"mlp.gelu_tanh.tile."+std::to_string(first),activated,count,4096);
    }
    return result;
}
}

VisionModel::VisionModel(Device& d,const ModelWeights& w,VisionLimits l):device(d),weights(w),limits(l) {checkLimits(l);}
VisionForecast VisionModel::forecast(uint32_t n,const std::vector<VisionGrid>& grids,VisionLimits l) {
    checkLimits(l);
    if(!n || n>l.maximumPatches || n%4)throw std::invalid_argument("Original patch rows exceed vision admission or merge geometry");
    VisionForecast f;f.patchRows=n;f.mergedRows=n/4;
    uint64_t total=0;
    for(const auto& grid:grids) {
        uint64_t size=uint64_t(grid.height)*grid.width;
        if(!grid.temporal || !grid.height || !grid.width || grid.height%2 || grid.width%2 ||
           !size || size>l.maximumFramePatches || uint64_t(grid.temporal)*size>n-total)
            throw std::invalid_argument("Unmodified gridTHW fails bounded exact geometry");
        total+=uint64_t(grid.temporal)*size;f.frames+=grid.temporal;
        f.maximumFrameRows=std::max(f.maximumFrameRows,uint32_t(size));
        const uint64_t qt=(size+l.queryTile-1)/l.queryTile;
        f.attentionDispatchesPerBlock+=grid.temporal*qt*4;
        f.attentionMultiplyAddsPerBlock+=uint64_t(grid.temporal)*size*size*hidden*2;
    }
    if(grids.empty() || total!=n)throw std::invalid_argument("Original grid sum differs from original patches");
    f.hiddenBufferBytes=uint64_t(n)*hidden*4;
    f.rotaryBufferBytes=uint64_t(n)*128*4;
    f.attentionScoreBytes=uint64_t(l.queryTile)*heads*f.maximumFrameRows*4;
    f.attentionPartialBytes=uint64_t(l.queryTile)*hidden*((f.maximumFrameRows+127)/128)*4;
    f.rowIntermediateBytes=uint64_t(l.rowTile)*4096*4;
    f.maximumSingleBufferBytes=std::max({f.hiddenBufferBytes,f.attentionScoreBytes,f.attentionPartialBytes,f.rowIntermediateBytes});
    // Five global hidden-width buffers during SDPA, bounded score/AV partial scratch;
    // three globals during MLP, two 4096-wide chunks plus two 1024-wide chunks. Integer metadata included.
    uint64_t attentionPeak=5*f.hiddenBufferBytes+f.attentionScoreBytes+f.attentionPartialBytes;
    uint64_t mlpPeak=3*f.hiddenBufferBytes+2*f.rowIntermediateBytes+2*uint64_t(l.rowTile)*hidden*4;
    // A merger chunk keeps gather/fc1/GELU/fc2 plus retained patch hidden and normalized hidden.
    uint64_t mergerPeak=2*f.hiddenBufferBytes+uint64_t(n/4)*2560*4+
        3*f.rowIntermediateBytes+uint64_t(l.rowTile)*2560*4;
    f.workspaceUpperBoundBytes=std::max({attentionPeak,mlpPeak,mergerPeak})+uint64_t(n)*16+f.rotaryBufferBytes+64;
    f.visionWeightBytes=weightElements()*2;
    f.attentionScoreMultiplyAddsPerDispatch=uint64_t(l.queryTile)*heads*f.maximumFrameRows*headDim;
    f.attentionValueMultiplyAddsPerDispatch=f.attentionScoreMultiplyAddsPerDispatch;
    if(f.maximumSingleBufferBytes>singleCap || f.workspaceUpperBoundBytes>l.maximumWorkspaceBytes)
        throw std::invalid_argument("Vision forecast exceeds declared buffer/workspace limits");
    return f;
}
VisionOutput VisionModel::run(const float* patches,uint32_t n,const std::vector<VisionGrid>& grids,
                             bool full,const VisionObserver& observer) {
    const auto plan=forecast(n,grids,limits);validateWeights(weights,full);
    if(!patches)throw std::invalid_argument("Original processor tensor pointer required");
    for(uint64_t i=0;i<uint64_t(n)*patchWidth;i++)if(!std::isfinite(patches[i]))
        throw std::invalid_argument("Original processor tensor contains nonfinite values");
    const auto geo=geometry(n,grids);auto positions=device.words(n*4,geo.positions.data());
    auto inverse=device.floats(16);auto rotary=device.floats(count32(uint64_t(n)*128));
    device.dispatch("runtime/vision_rope_frequency.hlsl",{},{&inverse},nullptr,0,1);
    CopyParams rotaryParams{n*32,0,0,0};
    device.dispatch("runtime/vision_rope_positions.hlsl",{&positions,&inverse},{&rotary},&rotaryParams,sizeof(rotaryParams),(n*32+255)/256);
    device.drain();observe(observer,"rotary_cos_sin",rotary,n,128);inverse={};
    Buffer h=device.floats(count32(uint64_t(n)*hidden));
    const auto& patchWeight=matrix(weights,"patch_embed.proj.weight",hidden,patchWidth);
    const auto& patchBias=vector(weights,"patch_embed.proj.bias",hidden);
    for(uint32_t first=0;first<n;first+=limits.rowTile) {
        uint32_t rows=std::min(limits.rowTile,n-first);
        auto input=device.floats(rows*patchWidth,patches+uint64_t(first)*patchWidth);
        auto rounded=unary(device,input,rows*patchWidth,3,true);
        auto embedded=linear(device,rounded,patchWeight,rows,true,&patchBias);
        copy(device,embedded,h,rows*hidden,0,first*hidden);device.drain();
    }
    observe(observer,"patch_embedding",h,n,hidden);
    {
        auto pos=device.floats(count32(uint64_t(n)*hidden));
        const auto& table=matrix(weights,"pos_embed.weight",2304,hidden).shards.front();
        for(uint32_t first=0;first<n*hidden;) {
            uint32_t count=std::min(n*hidden-first,dispatchElements);PositionParams p{count,first,hidden,0};
            device.dispatch("runtime/vision_position.hlsl",{&positions,&table},{&pos},&p,sizeof(p),(count+255)/256);first+=count;
        }
        observe(observer,"position_embedding",pos,n,hidden);
        h=binary(device,h,pos,n*hidden,0,true);device.drain();
    }
    observe(observer,"position_added",h,n,hidden);
    for(uint32_t i=0;i<(full?24u:1u);i++) {
        const std::string block="blocks."+std::to_string(i)+".";
        Buffer attended;
        {
            auto norm=layerNorm(device,h,vector(weights,block+"norm1.weight",hidden),vector(weights,block+"norm1.bias",hidden),n,hidden,1e-6f,true);
            observe(observer,block+"norm1",norm,n,hidden);
            auto split=qkv(device,weights,norm,rotary,n,block,limits.rowTile);norm={};
            observe(observer,block+"rope_q",split.q,n,hidden);observe(observer,block+"rope_k",split.k,n,hidden);observe(observer,block+"v",split.v,n,hidden);
            attended=attention(device,split,n,geo.frames,limits);
        }
        observe(observer,block+"sdpa",attended,n,hidden);
        auto projected=projection(device,weights,attended,n,hidden,hidden,block+"attn.proj",limits.rowTile);attended={};
        observe(observer,block+"attention_projection",projected,n,hidden);
        h=binary(device,h,projected,n*hidden,0,true);projected={};device.drain();
        observe(observer,block+"attention_residual",h,n,hidden);
        {
            auto norm=layerNorm(device,h,vector(weights,block+"norm2.weight",hidden),vector(weights,block+"norm2.bias",hidden),n,hidden,1e-6f,true);
            observe(observer,block+"norm2",norm,n,hidden);
            auto down=mlp(device,weights,norm,n,block,limits.rowTile,observer);norm={};
            h=binary(device,h,down,n*hidden,0,true);device.drain();
        }
        observe(observer,block+"output",h,n,hidden);
    }
    if(!full)return {h,{},plan};
    Buffer merged=device.floats(count32(uint64_t(n/4)*2560));
    {
        // Processor rows are already 2x2 block-major. Norm each 1024 row before contiguous 4-row reshape.
        auto normalized=layerNorm(device,h,vector(weights,"merger.norm.weight",hidden),vector(weights,"merger.norm.bias",hidden),n,hidden,1e-6f,true);
        observe(observer,"merger.norm",normalized,n,hidden);
        const auto& up=matrix(weights,"merger.linear_fc1.weight",4096,4096);const auto& ub=vector(weights,"merger.linear_fc1.bias",4096);
        const auto& down=matrix(weights,"merger.linear_fc2.weight",2560,4096);const auto& db=vector(weights,"merger.linear_fc2.bias",2560);
        for(uint32_t first=0;first<n/4;first+=limits.rowTile) {
            uint32_t rows=std::min(limits.rowTile,n/4-first);auto input=gather(device,normalized,first,rows,4096);
            auto upValue=linear(device,input,up,rows,true,&ub);
            auto activated=unary(device,upValue,rows*4096,4,true); // nn.GELU() defaults to erf, unlike block MLP.
            auto output=linear(device,activated,down,rows,true,&db);copy(device,output,merged,rows*2560,0,first*2560);device.drain();
            observe(observer,"merger.gelu_erf.tile."+std::to_string(first),activated,rows,4096);
        }
    }
    device.drain();observe(observer,"pooler_output",merged,n/4,2560);
    return {h,merged,plan};
}
VisionOutput VisionModel::firstBlock(const float* p,uint32_t n,const std::vector<VisionGrid>& grids,const VisionObserver& o) {return run(p,n,grids,false,o);}
VisionOutput VisionModel::forward(const float* p,uint32_t n,const std::vector<VisionGrid>& grids,const VisionObserver& o) {return run(p,n,grids,true,o);}
}
