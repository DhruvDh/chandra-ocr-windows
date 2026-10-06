// New code, MPL-2.0. Test-only recording Device/ModelWeights for CPU checks of graph call order.
// No GPU, arithmetic or weight bytes: dispatches are traced, never executed. Readback content is a
// deterministic BF16-representable pattern of (allocation serial, element). Include in one TU only.
#pragma once
#include "api.h"
#include "text_model.h"
#include "vision_model.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace chandra::dc {
struct FakeTrace { std::vector<std::string> lines; uint64_t serial=0, live=0; };
inline FakeTrace& fakeTrace() { static FakeTrace trace; return trace; }
inline std::string fakeDigest(const void* data,size_t bytes) {
    uint64_t h=1469598103934665603ull;auto* p=static_cast<const unsigned char*>(data);
    for(size_t i=0;i<bytes;i++) { h^=p[i];h*=1099511628211ull; }
    char text[17];std::snprintf(text,sizeof(text),"%016llx",static_cast<unsigned long long>(h));return text;
}
inline float fakeValue(uint64_t serial,uint64_t index) { return float(int((serial*131+index*7)%251)-125); }
struct Storage {
    uint64_t serial=0; uint32_t bytes=0;
    ~Storage() { fakeTrace().live-=bytes;fakeTrace().lines.push_back("release "+std::to_string(serial)); }
};
struct Device::Impl { std::vector<std::shared_ptr<Storage>> pending; };
Device::Device(const std::wstring&,const std::string&,const std::string&):impl(std::make_unique<Impl>()) {}
Device::~Device()=default;
namespace fake {
inline Buffer make(uint32_t count,const void* initial) {
    if(!count||uint64_t(count)*4>128ull*1024*1024)throw std::runtime_error("Fake buffer must be nonempty and at most 128 MiB");
    auto s=std::make_shared<Storage>();auto& t=fakeTrace();s->serial=++t.serial;s->bytes=count*4;t.live+=s->bytes;
    t.lines.push_back("alloc "+std::to_string(s->serial)+" words "+std::to_string(count)+" init "+(initial?fakeDigest(initial,size_t(count)*4):"none"));
    return {s,count,count,false};
}
inline void retain(std::vector<std::shared_ptr<Storage>>& pending,const Buffer& b) {
    if(std::none_of(pending.begin(),pending.end(),[&](const auto& p) { return p==b.storage; }))pending.push_back(b.storage);
}
}
Buffer Device::floats(uint32_t count,const float* initial) { return fake::make(count,initial); }
Buffer Device::words(uint32_t count,const uint32_t* initial) { return fake::make(count,initial); }
void Device::upload(Buffer& b,const void* values,uint32_t bytes) {
    fake::retain(impl->pending,b);fakeTrace().lines.push_back("upload "+std::to_string(b.storage->serial)+" "+fakeDigest(values,bytes));
}
void Device::dispatch(const std::string& shader,const std::vector<const Buffer*>& inputs,const std::vector<Buffer*>& outputs,
                             const void* parameters,uint32_t parameterBytes,uint32_t x,uint32_t y,uint32_t z) {
    std::string line="dispatch "+shader+" in";
    for(auto* b:inputs) { if(!b||!b->storage)throw std::runtime_error("Fake null SRV");line+=" "+std::to_string(b->storage->serial);fake::retain(impl->pending,*b); }
    line+=" out";
    for(auto* b:outputs) { if(!b||!b->storage)throw std::runtime_error("Fake null UAV");line+=" "+std::to_string(b->storage->serial);fake::retain(impl->pending,*b); }
    line+=" params "+(parameterBytes?fakeDigest(parameters,parameterBytes):"none")+" groups "+std::to_string(x)+" "+std::to_string(y)+" "+std::to_string(z);
    fakeTrace().lines.push_back(line);
}
void Device::drain(uint32_t) { fakeTrace().lines.push_back("drain");impl->pending.clear(); }
std::vector<uint32_t> Device::readWords(const Buffer& b) {
    if(!b.storage)throw std::runtime_error("Fake readback of empty buffer");
    // Mirrors device.cpp: a readback drains before copying and again before mapping.
    fakeTrace().lines.push_back("read "+std::to_string(b.storage->serial));impl->pending.clear();
    std::vector<uint32_t> result(b.words);
    for(uint32_t i=0;i<b.words;i++) { float v=fakeValue(b.storage->serial,i);std::memcpy(&result[i],&v,4); }
    return result;
}
std::vector<float> Device::readFloats(const Buffer& b) {
    auto words=readWords(b);std::vector<float> result(words.size());std::memcpy(result.data(),words.data(),words.size()*4);return result;
}
void Device::zero(Buffer& b) { fake::retain(impl->pending,b);fakeTrace().lines.push_back("zero "+std::to_string(b.storage->serial)); }
uint64_t Device::trackedBufferBytes() const { return fakeTrace().live; }
std::string Device::identityJson() const { return "{\"fake\":true}"; }
std::string Device::memoryJson() const { return "{\"fake\":true}"; }
void Device::beginProfile() {}
std::string Device::finishProfile() { return "{}"; }

struct ModelWeights::Impl { std::map<std::string,Weight> weights; };
ModelWeights::ModelWeights(Device& d,const std::wstring&,uint64_t,const std::vector<std::string>&):impl(std::make_unique<Impl>()) {
    auto add=[&](const std::string& name,uint32_t rows,uint32_t cols) {
        Weight w;w.rows=rows;w.cols=cols;w.bf16=true;const uint32_t perShard=uint32_t((64ull<<20)/cols);
        for(uint32_t first=0;first<rows;) {
            const uint32_t n=std::min(rows-first,perShard);const uint64_t elements=uint64_t(n)*cols;
            Buffer b=d.words(uint32_t((elements+1)/2));b.logicalElements=elements;b.packedBF16=true;
            w.shards.push_back(b);w.shardFirstRows.push_back(first);first+=n;
        }
        impl->weights.emplace(name,std::move(w));
    };
    const std::string t="model.language_model.";
    for(uint32_t l=0;l<32;l++) {
        const auto b=t+"layers."+std::to_string(l)+".";
        add(b+"input_layernorm.weight",1,2560);add(b+"post_attention_layernorm.weight",1,2560);
        add(b+"mlp.gate_proj.weight",9216,2560);add(b+"mlp.up_proj.weight",9216,2560);add(b+"mlp.down_proj.weight",2560,9216);
        if(l%4==3) {
            add(b+"self_attn.q_proj.weight",8192,2560);add(b+"self_attn.k_proj.weight",1024,2560);add(b+"self_attn.v_proj.weight",1024,2560);
            add(b+"self_attn.q_norm.weight",1,256);add(b+"self_attn.k_norm.weight",1,256);add(b+"self_attn.o_proj.weight",2560,4096);
        } else {
            add(b+"linear_attn.in_proj_qkv.weight",8192,2560);add(b+"linear_attn.in_proj_z.weight",4096,2560);
            add(b+"linear_attn.in_proj_a.weight",32,2560);add(b+"linear_attn.in_proj_b.weight",32,2560);
            add(b+"linear_attn.conv1d.weight",8192,4);add(b+"linear_attn.A_log",1,32);add(b+"linear_attn.dt_bias",1,32);
            add(b+"linear_attn.norm.weight",1,128);add(b+"linear_attn.out_proj.weight",2560,4096);
        }
    }
    add(t+"norm.weight",1,2560);add(t+"embed_tokens.weight",248320,2560);
    const std::string v="model.visual.";
    add(v+"patch_embed.proj.weight",1024,1536);add(v+"patch_embed.proj.bias",1,1024);add(v+"pos_embed.weight",2304,1024);
    for(uint32_t i=0;i<24;i++) {
        const auto b=v+"blocks."+std::to_string(i)+".";
        for(auto n:{"norm1.","norm2."}) { add(b+n+"weight",1,1024);add(b+n+"bias",1,1024); }
        add(b+"attn.qkv.weight",3072,1024);add(b+"attn.qkv.bias",1,3072);add(b+"attn.proj.weight",1024,1024);add(b+"attn.proj.bias",1,1024);
        add(b+"mlp.linear_fc1.weight",4096,1024);add(b+"mlp.linear_fc1.bias",1,4096);add(b+"mlp.linear_fc2.weight",1024,4096);add(b+"mlp.linear_fc2.bias",1,1024);
    }
    add(v+"merger.norm.weight",1,1024);add(v+"merger.norm.bias",1,1024);
    add(v+"merger.linear_fc1.weight",4096,4096);add(v+"merger.linear_fc1.bias",1,4096);
    add(v+"merger.linear_fc2.weight",2560,4096);add(v+"merger.linear_fc2.bias",1,2560);
}
ModelWeights::~ModelWeights()=default;
const Weight& ModelWeights::at(const std::string& name) const {
    auto it=impl->weights.find(name);if(it==impl->weights.end())throw std::runtime_error("Fake weight absent: "+name);return it->second;
}
bool ModelWeights::contains(const std::string& name) const { return impl->weights.count(name)!=0; }
std::string ModelWeights::provenanceJson() const { return "{\"fake\":true}"; }

// Fixed scenario: one [1,4,4] vision frame, 130-token prefill across three 64-row tiles, two cached advances.
// Four image-token rows give a consistent merged geometry; embeddings come from token IDs, not a merge.
struct FakeScenario {
    std::vector<float> pixels=std::vector<float>(16*1536,0.25f); std::vector<VisionGrid> grids{{1,4,4}};
    std::vector<uint32_t> ids, visionRowMap, decodeTokens{5,7}; TextPositions positions; std::vector<TextPositions> decodePositions;
    FakeScenario() {
        auto push=[&](uint32_t id,uint32_t t,uint32_t h,uint32_t w,uint32_t map) {
            ids.push_back(id);positions.temporal.push_back(t);positions.height.push_back(h);positions.width.push_back(w);visionRowMap.push_back(map);
        };
        for(uint32_t i=0;i<5;i++)push(100+i,i,i,i,UINT32_MAX);
        for(uint32_t m=0;m<4;m++)push(248056,5,5+m/2,5+m%2,m);
        for(uint32_t i=0;i<121;i++)push(1000+i,7+i,7+i,7+i,UINT32_MAX);
        for(uint32_t step=0;step<2;step++)decodePositions.push_back({{128+step},{128+step},{128+step}});
    }
};
template<class Hooks> void runFakeScenario(Device& d,ModelWeights& w,const FakeScenario& s,Hooks& hooks) {
    {
        VisionModel vision(d,w);auto output=hooks.vision(vision,s);(void)output;
    }
    TextModel text(d,w);auto request=text.newRequest();
    {
        auto embeddings=embedding(d,w.at("model.language_model.embed_tokens.weight"),s.ids);hooks.embeddings(embeddings,s);
        auto result=hooks.prefill(text,request,embeddings,s.positions);(void)d.readFloats(result.logits);
    }
    for(size_t k=0;k<s.decodeTokens.size();k++) {
        auto result=hooks.advance(text,request,s.decodeTokens[k],s.decodePositions[k]);(void)d.readFloats(result.logits);
    }
    text.retire(request);d.drain();
}
// Calls exactly the predecessor public API: no observer arguments.
struct PlainHooks {
    VisionOutput vision(VisionModel& v,const FakeScenario& s) { return v.forward(s.pixels.data(),16,s.grids); }
    void embeddings(const Buffer&,const FakeScenario&) {}
    TextResult prefill(TextModel& t,TextRequest& r,const Buffer& e,const TextPositions& p) { return t.prefill(r,e,p,true); }
    TextResult advance(TextModel& t,TextRequest& r,uint32_t id,const TextPositions& p) { return t.advance(r,id,p); }
};
}
