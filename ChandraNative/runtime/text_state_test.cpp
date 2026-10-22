// New code, MPL-2.0. Executed CPU controls of the unchanged text graph (text_model.cpp, operators.cpp) with the
// translated runtime HLSL and synthetic BF16 weights: chunk/offset invariance, prefill-versus-cached-decode
// equality, schedule determinism, request-state initialization and isolation, and a graph-level comparison with
// the 2c44182 attention softmax. One Gated DeltaNet layer (0) and one full-attention layer (3) run through
// TextModel::diagnosticLayer on 140 rows with three-axis M-RoPE positions shaped like the recorded prompt:
// text rows, one image block, then text rows continuing after the image's maximum coordinate. Linear and B1 GEMV
// use the fast replica that shader_emulation_test.cpp proves bitwise equal to the emulated HLSL.
// No trained values, D3D11 or GPU; prints one JSON object and exits 0 only if every expectation holds.
#include "shaders.generated.h"
#include "emulated_device.h"
#include "text_model.h"
#include <iostream>
#include <sstream>

namespace chandra::dc::experimental { const char* gemvB1Selection(); }
using namespace chandra::dc;
namespace {
int failures=0;
#define EXPECT(condition) do { if(!(condition)) { std::cerr<<__FILE__<<":"<<__LINE__<<": expected " #condition "\n";failures++; } } while(0)
constexpr uint32_t rowsTotal=140;
std::ostringstream report;
void emit(const std::string& j) { report<<(report.tellp()>0?",\n":"")<<j; }

struct Inputs { std::vector<float> hidden; TextPositions positions; };
Inputs inputs(uint64_t seed) {
    Inputs in;in.hidden.resize(size_t(rowsTotal)*2560);
    for(size_t i=0;i<in.hidden.size();i++)in.hidden[i]=emulation::asFloat(emulation::bf16Round(float(int64_t(emulation::mix(seed+i)%4001)-2000)/1000.0f));
    // 4 text rows, a 6x8 merged image grid (t fixed, h/w varying), then text from max+1, as get_rope_index orders them.
    for(uint32_t r=0;r<rowsTotal;r++) {
        uint32_t t,h,w;
        if(r<4) t=h=w=r;
        else if(r<4+48) { t=4;h=4+(r-4)/8;w=4+(r-4)%8; }
        else t=h=w=12+(r-52);
        in.positions.temporal.push_back(t);in.positions.height.push_back(h);in.positions.width.push_back(w);
    }
    return in;
}
TextPositions slice(const TextPositions& p,uint32_t first,uint32_t n) {
    TextPositions s;
    for(auto [d,src]:{std::pair{&s.temporal,&p.temporal},std::pair{&s.height,&p.height},std::pair{&s.width,&p.width}})d->assign(src->begin()+first,src->begin()+first+n);
    return s;
}
struct Result { std::vector<uint32_t> outputs, state; uint32_t length=0; uint64_t findings=0; std::string first; };
// Runs layer l over all rows in calls of the given sizes on a fresh request cache (newRequest zeroes it).
Result runLayer(Device& d,TextModel& text,uint32_t l,const Inputs& in,const std::vector<uint32_t>& calls,bool poisonState=false) {
    auto& st=emulation::state();st.findings={};
    TextRequest request=text.newRequest();auto& cache=request.layers[l];
    if(poisonState) { // Nonzero carried state: the outputs must change (sensitivity control).
        Buffer& b=l%4==3?cache.keys:cache.recurrent;std::vector<float> v(b.words);for(size_t i=0;i<v.size();i++)v[i]=float(int(i%7)-3)*0.125f;d.upload(b,v.data(),b.words*4);
    }
    Result r;uint32_t first=0;
    for(uint32_t n:calls) {
        Buffer h=d.floats(n*2560,in.hidden.data()+size_t(first)*2560);
        auto out=text.diagnosticLayer(l,h,n,slice(in.positions,first,n),cache);
        auto words=d.readWords(out);r.outputs.insert(r.outputs.end(),words.begin(),words.end());first+=n;
    }
    EXPECT(first==rowsTotal);r.length=cache.length;
    if(l%4==3) { auto k=d.readWords(cache.keys),v=d.readWords(cache.values);r.state.assign(k.begin(),k.begin()+rowsTotal*1024);r.state.insert(r.state.end(),v.begin(),v.begin()+rowsTotal*1024); }
    else { r.state=d.readWords(cache.conv);auto rec=d.readWords(cache.recurrent);r.state.insert(r.state.end(),rec.begin(),rec.end()); }
    r.findings=st.findings.total;if(!st.findings.clean())r.first=st.findings.first.front();
    text.retire(request);d.drain();return r;
}
std::vector<uint32_t> repeat(uint32_t n,uint32_t count) { return std::vector<uint32_t>(count,n); }
uint64_t differing(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b) {
    if(a.size()!=b.size())return UINT64_MAX;uint64_t n=0;for(size_t i=0;i<a.size();i++)n+=a[i]!=b[i];return n;
}
std::vector<std::vector<uint32_t>> chunkings() {
    return {{64,64,12},{12,64,64},{35,35,35,35},{1,63,64,12},repeat(1,rowsTotal)};
}
void configure(hlsl_cpu::Order order,const std::map<std::string,std::string>& overrides={}) {
    emulation::Config c;c.keepTrace=false;c.order=order;c.overrides=overrides;
    c.modes["runtime/linear.hlsl"]=emulation::Mode::Fast;c.modes["runtime/linear_gemv.hlsl"]=emulation::Mode::Fast;
    emulation::state().reset(c);
}

void layerControls(uint32_t layer) {
    const Inputs in=inputs(17);
    configure(hlsl_cpu::Order::Ascending);
    Device d(L"","emulated","emulated");ModelWeights w(d,L"");TextModel text(d,w);
    const auto all=chunkings();Result reference;
    for(size_t c=0;c<all.size();c++) {
        auto r=runLayer(d,text,layer,in,all[c]);
        if(r.findings)std::cerr<<"layer "<<layer<<" finding: "<<r.first<<"\n";
        EXPECT(r.findings==0);EXPECT(r.length==rowsTotal);
        if(c==0)reference=r;
        const uint64_t out=differing(r.outputs,reference.outputs),state=differing(r.state,reference.state);
        EXPECT(out==0);EXPECT(state==0);
        std::string calls;for(auto n:all[c])calls+=(calls.empty()?"":",")+std::to_string(n);
        if(all[c].size()>8)calls=std::to_string(all[c].size())+"x1";
        emit("{\"case\":\"chunk_invariance\",\"layer\":"+std::to_string(layer)+",\"calls\":\""+calls+"\",\"differing_output_words\":"+std::to_string(out)+
             ",\"differing_state_words\":"+std::to_string(state)+",\"findings\":"+std::to_string(r.findings)+"}");
    }
    for(uint32_t w0:reference.outputs)EXPECT(std::isfinite(emulation::asFloat(w0)));
    // Sensitivity: a shifted M-RoPE coordinate on one later row, or a nonzero initial state, changes the result.
    if(layer%4==3) {
        Inputs shifted=in;shifted.positions.height[100]+=1;
        auto r=runLayer(d,text,layer,shifted,all[0]);const uint64_t out=differing(r.outputs,reference.outputs);
        // Rows before 100 cannot see row 100's key; row 100 and later rows must change.
        uint64_t early=0;for(size_t i=0;i<size_t(100)*2560;i++)early+=r.outputs[i]!=reference.outputs[i];
        EXPECT(early==0);EXPECT(out>0);
        emit("{\"case\":\"position_sensitivity\",\"layer\":"+std::to_string(layer)+",\"differing_output_words\":"+std::to_string(out)+",\"differing_before_row_100\":"+std::to_string(early)+"}");
    }
    // Initial state: a nonzero GDN recurrent state must change the outputs; for attention, rows beyond the causal
    // frontier are never read, so nonzero contents there must change nothing.
    auto poisoned=runLayer(d,text,layer,in,all[0],true);const uint64_t changed=differing(poisoned.outputs,reference.outputs);
    if(layer%4==3)EXPECT(changed==0);else EXPECT(changed>0);
    emit("{\"case\":\"initial_state_sensitivity\",\"layer\":"+std::to_string(layer)+",\"poisoned\":\""+(layer%4==3?"key cache":"recurrent state")+
         "\",\"differing_output_words\":"+std::to_string(changed)+"}");
    // A cache allocated but never zeroed: the GDN conv/recurrent state is read before any write (detected), while
    // attention writes every key/value row before its first read (no undefined read).
    {
        auto& st=emulation::state();st.findings={};
        TextLayerCache cache;
        if(layer%4==3) { cache.keys=d.floats(16384*1024);cache.values=d.floats(16384*1024); } else { cache.conv=d.floats(8192*4);cache.recurrent=d.floats(32*128*128); }
        Buffer h=d.floats(64*2560,in.hidden.data());(void)text.diagnosticLayer(layer,h,64,slice(in.positions,0,64),cache);d.drain();
        const uint64_t reads=st.findings.count("uninitialized_read");
        if(layer%4==3)EXPECT(reads==0&&st.findings.clean());else EXPECT(reads>0);
        emit("{\"case\":\"unzeroed_cache\",\"layer\":"+std::to_string(layer)+",\"uninitialized_reads\":"+std::to_string(reads)+"}");
    }
    // Isolation: two requests advanced alternately equal their solo runs.
    {
        const Inputs other=inputs(91);auto soloOther=runLayer(d,text,layer,other,all[0]);
        auto& st=emulation::state();st.findings={};
        TextRequest a=text.newRequest(),b=text.newRequest();std::vector<uint32_t> outA,outB;uint32_t first=0;
        for(uint32_t n:all[0]) {
            for(auto [request,data,out]:{std::tuple{&a,&in,&outA},std::tuple{&b,&other,&outB}}) {
                Buffer h=d.floats(n*2560,data->hidden.data()+size_t(first)*2560);
                auto words=d.readWords(text.diagnosticLayer(layer,h,n,slice(data->positions,first,n),request->layers[layer]));out->insert(out->end(),words.begin(),words.end());
            }
            first+=n;
        }
        EXPECT(differing(outA,reference.outputs)==0);EXPECT(differing(outB,soloOther.outputs)==0);EXPECT(st.findings.clean());
        text.retire(a);text.retire(b);d.drain();
        emit("{\"case\":\"request_isolation\",\"layer\":"+std::to_string(layer)+",\"differing_a\":"+std::to_string(differing(outA,reference.outputs))+
             ",\"differing_b\":"+std::to_string(differing(outB,soloOther.outputs))+"}");
    }
}

// The attention layer under three legal lane schedules: the repaired graph is bit-stable; with the 2c44182
// softmax the same layer, inputs and weights produce schedule-dependent outputs.
void scheduleDeterminism() {
    const Inputs in=inputs(17);const std::vector<uint32_t> calls{64,64,12};
    for(bool original:{false,true}) {
        std::map<std::string,std::string> overrides;
        if(original)overrides["runtime/text_attention_softmax.hlsl"]="original/text_attention_softmax.hlsl";
        std::vector<Result> results;uint64_t races=0;
        for(auto order:{hlsl_cpu::Order::Descending,hlsl_cpu::Order::Ascending,hlsl_cpu::Order::Shuffled}) {
            configure(order,overrides);
            Device d(L"","emulated","emulated");ModelWeights w(d,L"");TextModel text(d,w);
            results.push_back(runLayer(d,text,3,in,calls));races+=results.back().findings;
        }
        const uint64_t ascending=differing(results[1].outputs,results[0].outputs),shuffled=differing(results[2].outputs,results[0].outputs);
        if(original) { EXPECT(races>0);EXPECT(ascending+shuffled>0); }
        else { EXPECT(races==0);EXPECT(ascending==0&&shuffled==0); }
        emit(std::string("{\"case\":\"schedule_determinism\",\"softmax\":\"")+(original?"original_2c44182":"repaired")+"\",\"race_findings\":"+std::to_string(races)+
             ",\"ascending_vs_descending_differing_words\":"+std::to_string(ascending)+",\"shuffled_vs_descending_differing_words\":"+std::to_string(shuffled)+"}");
    }
}
}

int main(int argc,char** argv) {
    const std::string only=argc>1?argv[1]:"all";
    try {
        if(only=="all"||only=="gdn")layerControls(0);
        if(only=="all"||only=="attention")layerControls(3);
        if(only=="all"||only=="schedule")scheduleDeterminism();
    } catch(const std::exception& e) { std::cerr<<"exception: "<<e.what()<<"\n";failures++; }
    std::string cost;for(const auto& [name,seconds]:emulation::state().seconds)cost+=(cost.empty()?"":",")+("\""+name+"\":"+std::to_string(seconds));
    std::cout<<"{\"failures\":"<<failures<<",\"selector\":\""<<experimental::gemvB1Selection()<<"\",\"last_configuration_host_seconds\":{"<<cost<<"},\"cases\":[\n"<<report.str()<<"\n]}\n";
    return failures?1:0;
}
