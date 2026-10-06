// New code, MPL-2.0. CPU checks of runtime HLSL synchronization and the fast linear replica, executed from
// the unmodified shader text (tests/native/hlsl_cpu.py). Requires a generated header that also contains
// "original/text_attention_softmax.hlsl" (the 2c44182 shader) and the synthetic detector controls
// "control/uav_race.hlsl" and "control/uninitialized.hlsl". Prints one JSON object; exit 0 only if all
// expectations hold. No D3D11, GPU, model bytes or trained values.
#include "shaders.generated.h"
#include "emulated_device.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>

namespace chandra::dc::experimental { const char* gemvB1Selection(); } // Defined in operators.cpp, not api.h.
using namespace chandra::dc;
using hlsl_cpu::Order;
namespace {
int failures=0;
#define EXPECT(condition) do { if(!(condition)) { std::cerr<<__FILE__<<":"<<__LINE__<<": expected " #condition "\n";failures++; } } while(0)
const Order orders[]={Order::Ascending,Order::Descending,Order::Shuffled};
const char* orderName(Order o) { return o==Order::Ascending?"ascending":o==Order::Descending?"descending":"shuffled"; }
const hlsl_cpu::Entry& entry(const std::string& name) {
    for(const auto& e:hlsl_cpu::entries())if(name==e.name)return e;
    throw std::runtime_error("missing translated shader "+name);
}
float f32(uint32_t u) { float f;std::memcpy(&f,&u,4);return f; }
uint32_t u32(float f) { uint32_t u;std::memcpy(&u,&f,4);return u; }
uint64_t splitmix(uint64_t& s) { uint64_t z=(s+=0x9e3779b97f4a7c15ull);z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;z=(z^(z>>27))*0x94d049bb133111ebull;return z^(z>>31); }
float uniform(uint64_t& s,float lo,float hi) { return lo+(hi-lo)*float(double(splitmix(s)>>11)*(1.0/9007199254740992.0)); }
float bf16(float v) { return f32(emulation::bf16Round(v)); }
hlsl_cpu::Memory memory(const std::string& label,const std::vector<uint32_t>& words) {
    hlsl_cpu::Memory m;m.label=label;m.words=uint32_t(words.size());m.data=words;m.markAllWritten();return m;
}
hlsl_cpu::Memory output(const std::string& label,uint32_t words) { hlsl_cpu::Memory m;m.label=label;m.words=words;return m; }
std::vector<uint32_t> packBF16(const std::vector<float>& values) {
    std::vector<uint32_t> w((values.size()+1)/2,0);
    for(size_t i=0;i<values.size();i++)w[i/2]|=(emulation::bf16Round(values[i])>>16)<<((i&1)*16);
    return w;
}
template<class P> std::vector<uint32_t> params(const P& p) { std::vector<uint32_t> c(64,0);std::memcpy(c.data(),&p,sizeof(p));return c; }
struct Run { hlsl_cpu::Findings findings; std::vector<std::vector<uint32_t>> outputs; };
// Fresh copies of the inputs/outputs per schedule, so each schedule starts from identical memory.
Run run(const std::string& shader,const std::vector<hlsl_cpu::Memory>& srv,const std::vector<hlsl_cpu::Memory>& uav,
        const std::vector<uint32_t>& constants,hlsl_cpu::uint3 groups,Order order,uint64_t seed=11) {
    Run r;auto inputs=srv;auto outputs=uav;hlsl_cpu::Binding b;b.groups=groups;b.constants=constants;
    for(auto& m:inputs)b.srv.push_back(&m);
    for(auto& m:outputs)b.uav.push_back(&m);
    hlsl_cpu::Options o;o.order=order;o.seed=seed;o.findings=&r.findings;entry(shader).run(b,o);
    for(auto& m:outputs)r.outputs.push_back(std::vector<uint32_t>(m.words32(),m.words32()+m.words));
    return r;
}
std::string kinds(const hlsl_cpu::Findings& f) {
    std::ostringstream s;s<<"{";bool first=true;
    for(const auto& k:f.kinds) { s<<(first?"":",")<<"\""<<k.first<<"\":"<<k.second;first=false; }
    s<<"}";return s.str();
}
std::ostringstream report;
bool firstCase=true;
void emit(const std::string& json) { report<<(firstCase?"":",\n")<<json;firstCase=false; }

// Text-attention score rows as text_model.cpp writes them: row r of a call with `base` cached rows and
// `count = base + rows` keys; keys after base + r keep the shader's -FLT_MAX causal mask.
struct Rows { uint32_t rows,count,base; std::vector<uint32_t> words; };
Rows causalScores(uint32_t base,uint32_t rows,uint64_t seed) {
    Rows s{rows*16,base+rows,base,{}};s.words.resize(size_t(s.rows)*s.count);
    for(uint32_t r=0;r<s.rows;r++)for(uint32_t k=0;k<s.count;k++) {
        const uint32_t query=r/16;
        // Typical bf16(q.k)/16 magnitudes with one dominant key per row; values are exact BF16 times 1/16.
        float v=k<=base+query?bf16(uniform(seed,-48.0f,48.0f))*0.0625f:-3.402823466e38f;
        if(k==(r*7919u)%(base+query+1))v=bf16(uniform(seed,100.0f,180.0f))*0.0625f;
        s.words[size_t(r)*s.count+k]=u32(v);
    }
    return s;
}
struct Deviation { double absolute=0, relative=0, sum=0; };
Deviation softmaxDeviation(const Rows& s,const std::vector<uint32_t>& got) {
    Deviation d;
    for(uint32_t r=0;r<s.rows;r++) {
        double m=-std::numeric_limits<double>::infinity(),total=0,sum=0;
        for(uint32_t k=0;k<s.count;k++)m=std::max(m,double(f32(s.words[size_t(r)*s.count+k])));
        for(uint32_t k=0;k<s.count;k++)total+=std::exp(double(f32(s.words[size_t(r)*s.count+k]))-m);
        for(uint32_t k=0;k<s.count;k++) {
            const double ref=std::exp(double(f32(s.words[size_t(r)*s.count+k]))-m)/total,g=f32(got[size_t(r)*s.count+k]);
            d.absolute=std::max(d.absolute,std::fabs(g-ref));
            if(ref>1e-30)d.relative=std::max(d.relative,std::fabs(g-ref)/ref);
            sum+=g;
        }
        d.sum=std::max(d.sum,std::fabs(sum-1.0));
    }
    return d;
}
struct TextParams { uint32_t rows,width,offset,stride,base,count,mode,pad; };

void textSoftmax() {
    // (base, rows): first prefill tile, ragged/one-lane counts, the recorded page's last 33-row tile and decode.
    const uint32_t cases[][2]={{0,1},{0,64},{63,1},{64,64},{127,1},{128,2},{3584,33},{3617,1},{3645,1}};
    for(const auto& c:cases) {
        const auto s=causalScores(c[0],c[1],c[0]*131+c[1]);
        const auto constants=params(TextParams{c[1],16,0,0,c[0],s.count,0,0});
        auto memories=std::vector<hlsl_cpu::Memory>{memory("scores",s.words)};
        std::vector<uint32_t> reference;
        for(Order o:orders) {
            // The 2c44182 shader: a barrier-interval race on tmp[0] whatever the schedule.
            auto old=run("original/text_attention_softmax.hlsl",{},memories,constants,{s.rows,1,1},o);
            const auto od=softmaxDeviation(s,old.outputs[0]);
            const uint64_t oldRaces=old.findings.count("groupshared_race_write_read")+old.findings.count("groupshared_race_read_write");
            EXPECT(oldRaces>0);
            EXPECT(old.findings.total==oldRaces);
            // Repaired shader: no finding and schedule-independent bits.
            auto fixed=run("runtime/text_attention_softmax.hlsl",{},memories,constants,{s.rows,1,1},o);
            const auto fd=softmaxDeviation(s,fixed.outputs[0]);
            EXPECT(fixed.findings.clean());
            if(reference.empty())reference=fixed.outputs[0];
            EXPECT(fixed.outputs[0]==reference);
            // BF16 RNE of each probability (2^-9 relative) plus FP32 exp/sum error.
            EXPECT(fd.relative<=1.01/256);EXPECT(fd.sum<=1.0/128); // BF16 RNE is within 2^-8 relative.
            uint64_t differing=0;for(size_t i=0;i<reference.size();i++)differing+=old.outputs[0][i]!=reference[i];
            std::ostringstream j;
            j<<"{\"case\":\"text_softmax\",\"base\":"<<c[0]<<",\"rows\":"<<c[1]<<",\"count\":"<<s.count<<",\"order\":\""<<orderName(o)
             <<"\",\"original_findings\":"<<kinds(old.findings)<<",\"original_max_abs\":"<<od.absolute<<",\"original_max_rel\":"<<od.relative
             <<",\"original_words_differing_from_repaired\":"<<differing<<",\"repaired_findings\":"<<fixed.findings.total
             <<",\"repaired_max_abs\":"<<fd.absolute<<",\"repaired_max_rel\":"<<fd.relative<<"}";
            emit(j.str());
            // A legal ascending schedule lets lane 0 publish its exp partial before later lanes read the maximum.
            if(o==Order::Ascending&&s.count>1) { EXPECT(differing>0);EXPECT(od.relative>0.5); }
        }
    }
}

// Every other runtime shader that uses groupshared memory/barriers: no race, no uninitialized or
// out-of-bounds access, no divergent barrier, identical bits under all three schedules.
void negativeControls() {
    uint64_t seed=99;
    auto floats=[&](size_t n,float lo,float hi,bool round=true) { std::vector<uint32_t> w(n);for(auto& x:w)x=u32(round?bf16(uniform(seed,lo,hi)):uniform(seed,lo,hi));return w; };
    auto bfv=[&](size_t n,float lo,float hi) { std::vector<float> v(n);for(auto& x:v)x=uniform(seed,lo,hi);return packBF16(v); };
    struct Case { std::string name,shader; std::vector<hlsl_cpu::Memory> srv,uav; std::vector<uint32_t> constants; hlsl_cpu::uint3 groups; };
    std::vector<Case> cases;
    struct SoftmaxParams { uint32_t queryHeads,sequenceLength,pad0,pad1; };
    for(uint32_t L:{200u,832u,12096u})
        cases.push_back({"vision_softmax_L"+std::to_string(L),"runtime/vision_attention_softmax.hlsl",{},{memory("scores",floats(size_t(3)*L,-6,6,false))},params(SoftmaxParams{3,L,0,0}),{3,1,1}});
    struct AttentionParams { uint32_t sequenceStart,sequenceLength,queryStart,queryCount,keyStart,keyCount,tileIndex,tileCount,width,heads,headDim,pad; };
    cases.push_back({"vision_scores_L200","runtime/vision_attention_scores.hlsl",{memory("q",floats(200*1024,-2,2)),memory("k",floats(200*1024,-2,2))},
                     {output("scores",3*16*200)},params(AttentionParams{0,200,197,3,0,0,0,2,1024,16,64,0}),{2,48,1}});
    struct NormParams { uint32_t width,rows,rowFirst,flags; float epsilon; uint32_t layerNorm,r0,r1; };
    cases.push_back({"rms_norm_2560","runtime/norm.hlsl",{memory("x",floats(3*2560,-3,3)),memory("w",bfv(2560,-0.5,0.5)),memory("b",bfv(2560,-0.5,0.5))},
                     {output("y",3*2560)},params(NormParams{2560,3,0,1|2|4,1e-6f,0,0,0}),{3,1,1}});
    cases.push_back({"layer_norm_1024","runtime/norm.hlsl",{memory("x",floats(3*1024,-3,3)),memory("w",bfv(1024,0.5,1.5)),memory("b",bfv(1024,-0.5,0.5))},
                     {output("y",3*1024)},params(NormParams{1024,3,0,1|2|8,1e-6f,1,0,0}),{3,1,1}});
    cases.push_back({"gdn_gate","runtime/text_gdn_gate.hlsl",{memory("x",floats(2*4096,-2,2)),memory("z",floats(2*4096,-2,2)),memory("w",bfv(128,0.5,1.5))},
                     {output("y",2*4096)},params(TextParams{2,32,128,0,0,0,0,0}),{32,2,1}});
    cases.push_back({"gdn_prepare","runtime/text_gdn_prepare.hlsl",{memory("cv",floats(2*8192,-2,2)),memory("a",floats(64,-2,2)),memory("b",floats(64,-2,2)),
                     memory("aw",bfv(32,0,2.75)),memory("dt",bfv(32,-1,1))},{output("q",2*4096),output("k",2*4096),output("v",2*4096),output("g",64),output("beta",64)},
                     params(TextParams{2,32,0,0,0,0,0,0}),{32,2,1}});
    for(auto& c:cases) {
        std::vector<std::vector<uint32_t>> reference;uint64_t total=0;
        for(Order o:orders) {
            auto r=run(c.shader,c.srv,c.uav,c.constants,c.groups,o);
            total+=r.findings.total;
            if(!r.findings.clean())std::cerr<<c.name<<" "<<orderName(o)<<" "<<r.findings.first.front()<<"\n";
            if(reference.empty())reference=r.outputs;
            EXPECT(r.outputs==reference);
            for(const auto& out:r.outputs)for(uint32_t w:out)EXPECT(std::isfinite(f32(w)));
        }
        EXPECT(total==0);
        emit("{\"case\":\"negative_control\",\"name\":\""+c.name+"\",\"shader\":\""+c.shader+"\",\"findings\":"+std::to_string(total)+",\"schedules\":3}");
    }
}

// Synthetic controls prove the detector reports real hazards rather than accepting everything.
void detectorControls() {
    auto race=run("control/uav_race.hlsl",{},{output("y",4)},std::vector<uint32_t>(64,0),{2,1,1},Order::Ascending);
    EXPECT(race.findings.count("uav_race_write_write")>0);
    auto uninitialized=run("control/uninitialized.hlsl",{output("x",64)},{output("y",64)},std::vector<uint32_t>(64,0),{1,1,1},Order::Ascending);
    EXPECT(uninitialized.findings.count("uninitialized_read")==64);
    EXPECT(uninitialized.findings.count("groupshared_uninitialized_read")>0);
    emit("{\"case\":\"detector_controls\",\"uav_race\":"+kinds(race.findings)+",\"uninitialized\":"+kinds(uninitialized.findings)+"}");
}

// operators.cpp linear() through the Device with the emulated linear.hlsl (or linear_gemv.hlsl when the
// process selector is "ordered") versus the fast replica used by the larger graph checks: equal bits.
void fastReplica() {
    Device device(L"","emulated","emulated");
    uint64_t seed=5;
    const char* selected=experimental::gemvB1Selection();
    const bool gemv=std::string(selected)=="ordered";
    const uint32_t shapes[][3]={{1,37,20},{1,1536,40},{5,64,33},{33,37,20},{17,1536,1100}}; // rows, K, outputs (1100 spans two 1024-output calls)
    uint64_t compared=0;
    for(const auto& s:shapes)for(bool round:{false,true})for(bool withBias:{false,true}) {
        const uint32_t rows=s[0],K=s[1],O=s[2];
        std::vector<float> x(size_t(rows)*K),w(size_t(O)*K),bv(O);
        for(auto& v:x)v=bf16(uniform(seed,-2,2));for(auto& v:w)v=uniform(seed,-0.2f,0.2f);for(auto& v:bv)v=uniform(seed,-0.5,0.5);
        std::vector<uint32_t> results[2];
        for(int fast=0;fast<2;fast++) {
            auto& st=emulation::state();emulation::Config config;config.keepTrace=false;
            if(fast) { config.modes["runtime/linear.hlsl"]=emulation::Mode::Fast;config.modes["runtime/linear_gemv.hlsl"]=emulation::Mode::Fast; }
            st.reset(config);
            Buffer input=device.floats(uint32_t(x.size()),x.data());
            // Two shards split at an odd row so the second shard starts mid-word for odd K.
            Weight weight;weight.rows=O;weight.cols=K;weight.bf16=true;
            const uint32_t split=O/2|1;
            for(uint32_t part=0;part<2;part++) {
                const uint32_t first=part?split:0,count=part?O-split:split;
                std::vector<float> slice(w.begin()+size_t(first)*K,w.begin()+size_t(first+count)*K);
                auto packed=packBF16(slice);Buffer b=device.words(uint32_t(packed.size()),packed.data());
                b.logicalElements=uint64_t(count)*K;b.packedBF16=true;weight.shards.push_back(b);weight.shardFirstRows.push_back(first);
            }
            Weight bias;bias.rows=1;bias.cols=O;bias.bf16=true;{auto packed=packBF16(bv);Buffer b=device.words(uint32_t(packed.size()),packed.data());b.logicalElements=O;b.packedBF16=true;bias.shards.push_back(b);bias.shardFirstRows.push_back(0);}
            Buffer out=linear(device,input,weight,rows,round,withBias?&bias:nullptr);
            results[fast]=device.readWords(out);
            EXPECT(st.findings.clean());
            if(!st.findings.clean())std::cerr<<"linear findings: "<<st.findings.first.front()<<"\n";
        }
        EXPECT(results[0]==results[1]);compared+=results[0].size();
    }
    emit(std::string("{\"case\":\"fast_linear_replica\",\"selector\":\"")+selected+"\",\"gemv_route_for_rows_1\":"+(gemv?"true":"false")+",\"compared_words\":"+std::to_string(compared)+"}");
}
}

int main(int argc,char** argv) {
    const std::string only=argc>1?argv[1]:"all";
    try {
        if(only=="all"||only=="softmax")textSoftmax();
        if(only=="all"||only=="controls") { negativeControls();detectorControls(); }
        if(only=="all"||only=="linear")fastReplica();
    } catch(const std::exception& e) { std::cerr<<"exception: "<<e.what()<<"\n";failures++; }
    std::cout<<"{\"failures\":"<<failures<<",\"cases\":[\n"<<report.str()<<"\n]}\n";
    return failures?1:0;
}
