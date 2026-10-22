// New code, MPL-2.0. Test-only Device/ModelWeights that EXECUTE the translated runtime HLSL on the CPU.
// The unchanged graph sources (operators.cpp, text_model.cpp, vision_model.cpp) call this Device exactly
// as they call device.cpp; every dispatch is validated like device.cpp, recorded with its complete
// constants and groups, and then emulated (hlsl_cpu.h), executed by a bitwise-equivalent fast replica
// (linear.hlsl and linear_gemv.hlsl only; tests compare both against emulation), or stubbed for trace-only
// production-geometry runs. Weights are synthetic BF16 patterns, never model bytes. Include in one TU only,
// after the generated shaders header. No D3D11, GPU or trained value is involved.
#pragma once
#include "api.h"
#include "hlsl_cpu.h"
#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>

namespace chandra::dc {
namespace emulation {
enum class Mode { Emulate, Fast, Stub };
struct DispatchRecord {
    std::string shader; uint32_t x=0,y=0,z=0; std::vector<uint32_t> params; std::vector<uint64_t> inputs, outputs;
    std::vector<uint32_t> outputWords; Mode mode=Mode::Emulate;
};
struct Event { std::string kind; uint64_t serial=0; uint32_t words=0; bool initial=false; size_t dispatch=SIZE_MAX; std::vector<uint32_t> initialWords; };
struct Config {
    hlsl_cpu::Order order=hlsl_cpu::Order::Ascending; uint64_t seed=1;
    std::map<std::string,Mode> modes; Mode defaultMode=Mode::Emulate;
    std::map<std::string,std::string> overrides; // Graph shader name -> another translated entry (e.g. a predecessor).
    bool keepTrace=true;
    // Synthetic weights: seed and an alias map so many layers can share one bounded tensor.
    uint64_t weightSeed=7; std::function<std::string(const std::string&)> alias;
};
struct State {
    Config config; hlsl_cpu::Findings findings; std::vector<DispatchRecord> dispatches; std::vector<Event> events;
    uint64_t serial=0, live=0, peak=0, drains=0, reads=0;
    std::unordered_map<uint64_t,std::weak_ptr<Storage>> storages; // Lookup of still-live buffers by serial.
    std::map<std::string,double> seconds; // Host time per executed shader, for reporting test cost only.
    void reset(Config c) { config=std::move(c);findings={};dispatches.clear();events.clear();storages.clear();serial=live=peak=drains=reads=0; }
};
inline State& state() { static State s;return s; }
inline std::string canonical(const std::string& name) {
    std::string n=name;if(n.find('.')==std::string::npos)n+=".hlsl";if(n.find('/')==std::string::npos)n="runtime/"+n;return n;
}
inline uint32_t bf16Round(float v) {
    uint32_t u;std::memcpy(&u,&v,4);
    if((u&0x7fffffffu)>0x7f800000u)return (u&0xffff0000u)|0x00400000u;
    return (u+0x7fffu+((u>>16)&1u))&0xffff0000u;
}
inline float asFloat(uint32_t u) { float f;std::memcpy(&f,&u,4);return f; }
inline uint32_t asWord(float f) { uint32_t u;std::memcpy(&u,&f,4);return u; }
inline uint64_t mix(uint64_t x) { x^=x>>33;x*=0xff51afd7ed558ccdull;x^=x>>33;x*=0xc4ceb9fe1a85ec53ull;x^=x>>33;return x; }
inline uint64_t nameHash(const std::string& s) { uint64_t h=1469598103934665603ull;for(unsigned char c:s) { h^=c;h*=1099511628211ull; }return h; }
// Deterministic synthetic value in [lo,hi), BF16-rounded.
inline uint16_t syntheticBF16(uint64_t seed,uint64_t index,float lo,float hi) {
    const double u=double(mix(seed^mix(index+0x9e3779b97f4a7c15ull))>>11)*(1.0/9007199254740992.0);
    return uint16_t(bf16Round(float(lo+(hi-lo)*u))>>16);
}
}
struct Storage {
    hlsl_cpu::Memory memory; uint64_t serial=0;
    ~Storage() { auto& s=emulation::state();s.live-=uint64_t(memory.words)*4;s.storages.erase(serial);if(s.config.keepTrace)s.events.push_back({"release",serial,memory.words,false,SIZE_MAX,{}}); }
};
struct Device::Impl { std::vector<std::shared_ptr<Storage>> pending; };
Device::Device(const std::wstring&,const std::string&,const std::string&):impl(std::make_unique<Impl>()) {}
Device::~Device()=default;
namespace emulation {
inline Buffer make(uint32_t count,const void* initial) {
    if(!count||uint64_t(count)*4>128ull*1024*1024)throw std::runtime_error("D3D11 buffer must be nonempty and at most 128 MiB");
    auto& s=state();
    if(s.live+uint64_t(count)*4>13ull*1024*1024*1024)throw std::runtime_error("Tracked D3D11 allocation cap exceeded");
    auto st=std::make_shared<Storage>();st->serial=++s.serial;st->memory.words=count;st->memory.label="buffer"+std::to_string(st->serial);
    if(initial) { st->memory.data.assign(static_cast<const uint32_t*>(initial),static_cast<const uint32_t*>(initial)+count);st->memory.markAllWritten(); }
    s.live+=uint64_t(count)*4;s.peak=std::max(s.peak,s.live);s.storages[st->serial]=st;
    if(s.config.keepTrace) {
        Event e{"alloc",st->serial,count,initial!=nullptr,SIZE_MAX,{}};
        if(initial&&count<=4096)e.initialWords=st->memory.data; // Small uploads (positions, maps) stay inspectable.
        s.events.push_back(std::move(e));
    }
    return {st,count,count,false};
}
inline void valid(const Buffer& b) {
    if(!b.storage||!b.words||b.storage->memory.words!=b.words)throw std::runtime_error("Invalid buffer or foreign D3D11 device");
    if(!b.logicalElements||b.logicalElements>uint64_t(b.words)*(b.packedBF16?2:1))throw std::runtime_error("Invalid logical buffer extent");
}
// linear.hlsl replica: every output is ((((+0 + a0*w0) + a1*w1) ...) + bias) then optional BF16, in ascending k.
// Eight outputs advance together in independent accumulators; no output's operation order changes.
inline void fastLinear(const hlsl_cpu::Binding& b,bool gemv) {
    const uint32_t* c=b.constants.data();
    const uint32_t inputWidth=c[0],outputWidth=c[1],batchRows=c[2],firstOutput=c[3],shardRows=c[4],rowFirst=c[5],flags=c[6],weightRowFirst=c[7];
    auto* input=b.srv.at(0);auto* weight=b.srv.at(1);auto* bias=b.srv.at(2);auto* output=b.uav.at(0);
    const uint32_t outputs=gemv?std::min<uint32_t>(shardRows,b.groups.x*8):std::min<uint32_t>(shardRows,b.groups.x*16);
    const uint32_t rows=gemv?1:std::min<uint32_t>(batchRows,b.groups.y*16);
    const uint32_t* in=input->words32();const uint32_t* w=weight->words32();const uint32_t* bi=bias->words32();uint32_t* out=output->words32();
    const bool bf=(flags&1)!=0;
    auto weightAt=[&](uint64_t i) { return bf?asFloat(((w[i>>1]>>((i&1)*16))&0xffffu)<<16):asFloat(w[i]); };
    auto biasAt=[&](uint64_t i) { return (flags&8)?asFloat(((bi[i>>1]>>((i&1)*16))&0xffffu)<<16):asFloat(bi[i]); };
    constexpr uint32_t block=8;
    std::vector<float> x(uint64_t(inputWidth)*32),wt(uint64_t(inputWidth)*block),acc(32*block);
    for(uint32_t r0=0;r0<rows;r0+=32) {
        const uint32_t nr=std::min<uint32_t>(32,rows-r0);
        for(uint32_t r=0;r<nr;r++)for(uint32_t k=0;k<inputWidth;k++)x[uint64_t(r)*inputWidth+k]=asFloat(in[uint64_t(rowFirst+r0+r)*inputWidth+k]);
        for(uint32_t o0=0;o0<outputs;o0+=block) {
            const uint32_t no=std::min<uint32_t>(block,outputs-o0);
            for(uint32_t k=0;k<inputWidth;k++)for(uint32_t o=0;o<block;o++)
                wt[uint64_t(k)*block+o]=o<no?weightAt(uint64_t(weightRowFirst+o0+o)*inputWidth+k):0.0f;
            std::fill(acc.begin(),acc.end(),0.0f);
            for(uint32_t r=0;r<nr;r++) {
                float* a=&acc[r*block];const float* xr=&x[uint64_t(r)*inputWidth];
                for(uint32_t k=0;k<inputWidth;k++) {
                    const float xk=xr[k];const float* wk=&wt[uint64_t(k)*block];
                    for(uint32_t o=0;o<block;o++) { const float p=xk*wk[o];a[o]=a[o]+p; }
                }
            }
            for(uint32_t r=0;r<nr;r++)for(uint32_t o=0;o<no;o++) {
                float sum=acc[r*block+o];const uint32_t column=firstOutput+o0+o;
                if(flags&4)sum=sum+biasAt(column);
                const uint32_t bits=(flags&2)?bf16Round(sum):asWord(sum);
                const uint64_t index=gemv?column:uint64_t(rowFirst+r0+r)*outputWidth+column;
                out[index]=bits;output->markWritten(uint32_t(index));
            }
        }
    }
}
}
Buffer Device::floats(uint32_t count,const float* initial) { return emulation::make(count,initial); }
Buffer Device::words(uint32_t count,const uint32_t* initial) { return emulation::make(count,initial); }
void Device::upload(Buffer& b,const void* values,uint32_t bytes) {
    emulation::valid(b);if(!values||bytes!=b.words*4u)throw std::runtime_error("Complete bounded buffer upload required");
    auto& m=b.storage->memory;m.data.assign(static_cast<const uint32_t*>(values),static_cast<const uint32_t*>(values)+b.words);m.markAllWritten();
    if(emulation::state().config.keepTrace)emulation::state().events.push_back({"upload",b.storage->serial,b.words,true,SIZE_MAX,{}});
}
void Device::dispatch(const std::string& name,const std::vector<const Buffer*>& inputs,const std::vector<Buffer*>& outputs,
                      const void* params,uint32_t paramBytes,uint32_t x,uint32_t y,uint32_t z) {
    using namespace emulation;
    if(inputs.size()>16||outputs.empty()||outputs.size()>8||paramBytes>256||(paramBytes&&!params)||!x||!y||!z||x>65535||y>65535||z>65535)
        throw std::runtime_error("Invalid bounded compute dispatch");
    std::set<const Storage*> writable;
    for(auto p:outputs) { if(!p)throw std::runtime_error("Null UAV");valid(*p);if(!writable.insert(p->storage.get()).second)throw std::runtime_error("Aliased UAVs"); }
    for(auto p:inputs) { if(!p)throw std::runtime_error("Null SRV");valid(*p);if(writable.count(p->storage.get()))throw std::runtime_error("SRV/UAV alias forbidden"); }
    auto& s=state();const std::string shader=canonical(name);
    auto overridden=s.config.overrides.find(shader);const std::string executed=overridden==s.config.overrides.end()?shader:overridden->second;
    const hlsl_cpu::Entry* entry=nullptr;
    for(const auto& e:hlsl_cpu::entries())if(executed==e.name)entry=&e;
    if(!entry)throw std::runtime_error("Shader compile "+shader+": not present in the translated runtime set");
    hlsl_cpu::Binding b;b.groups={x,y,z};if(paramBytes)std::memcpy(b.constants.data(),params,paramBytes);
    for(auto p:inputs)b.srv.push_back(&p->storage->memory);
    for(auto p:outputs)b.uav.push_back(&p->storage->memory);
    Mode mode=s.config.defaultMode;auto found=s.config.modes.find(shader);if(found!=s.config.modes.end())mode=found->second;
    if(mode==Mode::Fast&&shader!="runtime/linear.hlsl"&&shader!="runtime/linear_gemv.hlsl")throw std::runtime_error("No fast replica for "+shader);
    DispatchRecord record{shader,x,y,z,b.constants,{},{},{},mode};
    for(auto p:inputs)record.inputs.push_back(p->storage->serial);
    for(auto p:outputs) { record.outputs.push_back(p->storage->serial);record.outputWords.push_back(p->words); }
    if(s.config.keepTrace) { s.dispatches.push_back(record);s.events.push_back({"dispatch",0,0,false,s.dispatches.size()-1,{}}); }
    for(auto p:inputs)if(std::find(impl->pending.begin(),impl->pending.end(),p->storage)==impl->pending.end())impl->pending.push_back(p->storage);
    for(auto p:outputs)if(std::find(impl->pending.begin(),impl->pending.end(),p->storage)==impl->pending.end())impl->pending.push_back(p->storage);
    if(mode==Mode::Stub) { for(auto p:outputs)p->storage->memory.markAllWritten();return; }
    const auto began=std::chrono::steady_clock::now();
    if(mode==Mode::Fast)fastLinear(b,shader=="runtime/linear_gemv.hlsl");
    else {
        hlsl_cpu::Options o;o.order=s.config.order;o.seed=s.config.seed+s.dispatches.size();o.findings=&s.findings;
        entry->run(b,o);
    }
    s.seconds[shader]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
}
void Device::drain(uint32_t) { auto& s=emulation::state();s.drains++;if(s.config.keepTrace)s.events.push_back({"drain",0,0,false,SIZE_MAX,{}});impl->pending.clear(); }
std::vector<uint32_t> Device::readWords(const Buffer& b) {
    emulation::valid(b);auto& s=emulation::state();s.reads++;drain();
    auto& m=b.storage->memory;if(!m.fullyWritten())s.findings.add("readback_uninitialized",m.label);
    if(s.config.keepTrace)s.events.push_back({"read",b.storage->serial,b.words,false,SIZE_MAX,{}});
    const uint32_t* p=m.words32();return std::vector<uint32_t>(p,p+b.words);
}
std::vector<float> Device::readFloats(const Buffer& b) {
    if(b.packedBF16)throw std::runtime_error("Packed BF16 is not float storage");
    auto words=readWords(b);std::vector<float> result(words.size());std::memcpy(result.data(),words.data(),words.size()*4);return result;
}
void Device::zero(Buffer& b) {
    emulation::valid(b);auto& m=b.storage->memory;
    if(!m.data.empty())std::fill(m.data.begin(),m.data.end(),0u);
    m.zeroed=true;m.written.clear();
    if(emulation::state().config.keepTrace)emulation::state().events.push_back({"zero",b.storage->serial,b.words,false,SIZE_MAX,{}});
}
uint64_t Device::trackedBufferBytes() const { return emulation::state().live; }
std::string Device::identityJson() const { return "{\"emulated\":true}"; }
std::string Device::memoryJson() const { return "{\"emulated\":true}"; }
void Device::beginProfile() {}
std::string Device::finishProfile() { return "{}"; }

// Synthetic BF16 weights with the pinned names, shapes and the importer's 128 MiB row sharding. Contents
// are generated lazily per shard; config.alias can map many layer names onto one bounded tensor.
struct ModelWeights::Impl { std::map<std::string,Weight> weights; std::function<std::string(const std::string&)> alias; };
ModelWeights::ModelWeights(Device& d,const std::wstring&,uint64_t,const std::vector<std::string>&):impl(std::make_unique<Impl>()) {
    const auto& config=emulation::state().config;impl->alias=config.alias;
    auto add=[&](const std::string& name,uint32_t rows,uint32_t cols,float lo,float hi) {
        Weight w;w.rows=rows;w.cols=cols;w.bf16=true;
        uint64_t perShard=(128ull<<20)/(uint64_t(cols)*2);if(cols%2&&perShard>1)perShard-=perShard%2;
        const uint64_t seed=config.weightSeed^emulation::nameHash(name);
        for(uint32_t first=0;first<rows;) {
            const uint32_t n=uint32_t(std::min<uint64_t>(rows-first,perShard));const uint64_t elements=uint64_t(n)*cols,base=uint64_t(first)*cols;
            Buffer b=d.words(uint32_t((elements+1)/2));b.logicalElements=elements;b.packedBF16=true;
            b.storage->memory.label=name+"#"+std::to_string(first);
            b.storage->memory.generator=[seed,base,elements,lo,hi](std::vector<uint32_t>& words) {
                for(uint64_t i=0;i<elements;i++) {
                    const uint32_t v=emulation::syntheticBF16(seed,base+i,lo,hi);
                    words[i>>1]|=uint32_t(v)<<((i&1)*16);
                }
            };
            w.shards.push_back(b);w.shardFirstRows.push_back(first);first+=n;
        }
        impl->weights.emplace(name,std::move(w));
    };
    auto matrix=[&](const std::string& name,uint32_t rows,uint32_t cols) { const float s=1.0f/std::sqrt(float(cols));add(name,rows,cols,-s,s); };
    auto vec=[&](const std::string& name,uint32_t n,float lo,float hi) { add(name,1,n,lo,hi); };
    const std::string t="model.language_model.";
    auto wanted=[&](const std::string& name) { return !impl->alias||impl->alias(name)==name; };
    for(uint32_t l=0;l<32;l++) {
        const auto b=t+"layers."+std::to_string(l)+".";
        if(!wanted(b+"mlp.gate_proj.weight"))continue;
        vec(b+"input_layernorm.weight",2560,-0.25f,0.25f);vec(b+"post_attention_layernorm.weight",2560,-0.25f,0.25f);
        matrix(b+"mlp.gate_proj.weight",9216,2560);matrix(b+"mlp.up_proj.weight",9216,2560);matrix(b+"mlp.down_proj.weight",2560,9216);
        if(l%4==3) {
            matrix(b+"self_attn.q_proj.weight",8192,2560);matrix(b+"self_attn.k_proj.weight",1024,2560);matrix(b+"self_attn.v_proj.weight",1024,2560);
            vec(b+"self_attn.q_norm.weight",256,-0.25f,0.25f);vec(b+"self_attn.k_norm.weight",256,-0.25f,0.25f);matrix(b+"self_attn.o_proj.weight",2560,4096);
        } else {
            matrix(b+"linear_attn.in_proj_qkv.weight",8192,2560);matrix(b+"linear_attn.in_proj_z.weight",4096,2560);
            matrix(b+"linear_attn.in_proj_a.weight",32,2560);matrix(b+"linear_attn.in_proj_b.weight",32,2560);
            add(b+"linear_attn.conv1d.weight",8192,4,-0.5f,0.5f);vec(b+"linear_attn.A_log",32,0.0f,2.75f);vec(b+"linear_attn.dt_bias",32,-1.0f,1.0f);
            vec(b+"linear_attn.norm.weight",128,0.5f,1.5f);matrix(b+"linear_attn.out_proj.weight",2560,4096);
        }
    }
    vec(t+"norm.weight",2560,-0.25f,0.25f);add(t+"embed_tokens.weight",248320,2560,-1.0f,1.0f);
    const std::string v="model.visual.";
    matrix(v+"patch_embed.proj.weight",1024,1536);vec(v+"patch_embed.proj.bias",1024,-0.1f,0.1f);add(v+"pos_embed.weight",2304,1024,-1.0f,1.0f);
    for(uint32_t i=0;i<24;i++) {
        const auto b=v+"blocks."+std::to_string(i)+".";
        if(!wanted(b+"attn.qkv.weight"))continue;
        for(auto n:{"norm1.","norm2."}) { vec(b+n+"weight",1024,0.75f,1.25f);vec(b+n+"bias",1024,-0.1f,0.1f); }
        matrix(b+"attn.qkv.weight",3072,1024);vec(b+"attn.qkv.bias",3072,-0.1f,0.1f);matrix(b+"attn.proj.weight",1024,1024);vec(b+"attn.proj.bias",1024,-0.1f,0.1f);
        matrix(b+"mlp.linear_fc1.weight",4096,1024);vec(b+"mlp.linear_fc1.bias",4096,-0.1f,0.1f);matrix(b+"mlp.linear_fc2.weight",1024,4096);vec(b+"mlp.linear_fc2.bias",1024,-0.1f,0.1f);
    }
    vec(v+"merger.norm.weight",1024,0.75f,1.25f);vec(v+"merger.norm.bias",1024,-0.1f,0.1f);
    matrix(v+"merger.linear_fc1.weight",4096,4096);vec(v+"merger.linear_fc1.bias",4096,-0.1f,0.1f);
    matrix(v+"merger.linear_fc2.weight",2560,4096);vec(v+"merger.linear_fc2.bias",2560,-0.1f,0.1f);
}
ModelWeights::~ModelWeights()=default;
const Weight& ModelWeights::at(const std::string& name) const {
    const std::string target=impl->alias?impl->alias(name):name;
    auto it=impl->weights.find(target);if(it==impl->weights.end())throw std::runtime_error("Emulated weight absent: "+name);return it->second;
}
bool ModelWeights::contains(const std::string& name) const { return impl->weights.count(impl->alias?impl->alias(name):name)!=0; }
std::string ModelWeights::provenanceJson() const { return "{\"emulated\":true,\"synthetic_bf16\":true}"; }
}
