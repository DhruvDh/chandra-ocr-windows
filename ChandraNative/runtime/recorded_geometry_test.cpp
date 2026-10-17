// New code, MPL-2.0. Production-geometry CPU checks of the unchanged vision/text host graph for the exact
// recorded Zotero request: grid [1,126,96], 12,096 patches, 3,024 merged rows, 3,617 prompt rows.
// Usage: recorded-geometry-test POSITIONS_I64 OUTPUT_DIRECTORY [PLAN_JSON DUMP_DIRECTORY TOKEN_CAP]
// POSITIONS_I64 is the [3,1,3617] little-endian int64 three-axis position tensor; the Python driver checks its
// SHA-256 against the recorded input manifest before calling this program.
//  1. Emulates the actual vision_rope_frequency/vision_rope_positions/vision_position/vision_rows HLSL through
//     VisionModel::forward (projections stubbed) and writes positions, cos/sin and the learned-position
//     interpolation, plus the synthetic BF16 table, for the pinned-upstream comparison.
//  2. Replays the request phases of inference_core.cpp generate() with every dispatch stubbed: complete
//     VisionModel::forward, prompt embedding, the merge() allocation/dispatch loop, a fresh TextRequest, prefill
//     and two cached advances, then retirement. It reports tracked live/peak bytes at each phase boundary and
//     checks tile, chunk, offset, cache-length and position coverage from the exact constants the graph issued.
//  3. Emulates multimodal_merge.hlsl over the full prompt with the dispatch loop of inference_core.cpp merge().
//  4. With a plan: a dry run of the unchanged diagnostics::Recorder wired as inference.cpp wires it, over the same
//     stubbed replay with TOKEN_CAP generated rows, writing a strict v2 dump whose payloads are stub zeros. It proves
//     the plan resolves, executes and closes at this geometry; it contains no numerical evidence.
// No arithmetic of trained weights, D3D11 or GPU is involved; prints one JSON object.
#include "shaders.generated.h"
#include "emulated_device.h"
#include "text_model.h"
#include "vision_model.h"
#include "diagnostics.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

using namespace chandra::dc;
namespace {
int failures=0;
#define EXPECT(condition) do { if(!(condition)) { std::cerr<<__FILE__<<":"<<__LINE__<<": expected " #condition "\n";failures++; } } while(0)
constexpr uint32_t gridH=126,gridW=96,patches=gridH*gridW,merged=patches/4,prompt=3617,firstImage=4,decodeSteps=2;
struct Stop {};
std::string out;
std::ostringstream report;
void field(const std::string& name,const std::string& json) { report<<(report.tellp()>0?",\n":"")<<"\""<<name<<"\":"<<json; }
void save(const std::string& name,const void* data,size_t bytes) {
    std::ofstream f(out+"/"+name,std::ios::binary);f.write(static_cast<const char*>(data),std::streamsize(bytes));
    if(!f)throw std::runtime_error("write failed: "+name);
}
std::vector<uint32_t> memoryOf(uint64_t serial) {
    auto it=emulation::state().storages.find(serial);
    if(it==emulation::state().storages.end())throw std::runtime_error("buffer no longer live");
    auto s=it->second.lock();const uint32_t* p=s->memory.words32();return std::vector<uint32_t>(p,p+s->memory.words);
}
// Disjoint, complete cover of [0,total) by half-open intervals.
bool covers(std::vector<std::pair<uint64_t,uint64_t>> v,uint64_t total) {
    std::sort(v.begin(),v.end());uint64_t next=0;
    for(auto [a,b]:v) { if(a!=next||b<=a)return false;next=b; }
    return next==total;
}
using emulation::DispatchRecord;
std::vector<const DispatchRecord*> select(const std::string& shader) {
    std::vector<const DispatchRecord*> r;for(const auto& d:emulation::state().dispatches)if(d.shader==shader)r.push_back(&d);return r;
}
std::string join(const std::vector<uint32_t>& v) { std::string s="[";for(size_t i=0;i<v.size();i++)s+=(i?",":"")+std::to_string(v[i]);return s+"]"; }

void visionEmulation() {
    emulation::Config c;c.keepTrace=true;
    c.modes["runtime/linear.hlsl"]=emulation::Mode::Stub;c.modes["runtime/elementwise.hlsl"]=emulation::Mode::Stub;
    emulation::state().reset(c);
    Device d(L"","emulated","emulated");ModelWeights w(d,L"");VisionModel vision(d,w);
    std::vector<float> pixels(size_t(patches)*1536,0.0f);bool sawRotary=false,sawPosition=false;
    auto observer=[&](const std::string& name,const Buffer& b,uint32_t rows,uint32_t width) {
        if(name=="rotary_cos_sin") {
            EXPECT(rows==patches&&width==128);auto v=d.readWords(b);save("rotary_cos_sin.f32",v.data(),v.size()*4);
            const auto rope=select("runtime/vision_rope_positions.hlsl");EXPECT(rope.size()==1);
            auto positions=memoryOf(rope.at(0)->inputs.at(0));EXPECT(positions.size()==size_t(patches)*4);
            save("vision_positions.u32",positions.data(),positions.size()*4);sawRotary=true;
        } else if(name=="position_embedding") {
            EXPECT(rows==patches&&width==1024);auto v=d.readWords(b);save("position_embedding.f32",v.data(),v.size()*4);
            sawPosition=true;throw Stop{};
        }
    };
    try { vision.forward(pixels.data(),patches,{{1,gridH,gridW}},observer); } catch(const Stop&) {}
    EXPECT(sawRotary&&sawPosition);
    const auto& table=w.at("model.visual.pos_embed.weight");EXPECT(table.shards.size()==1);
    const auto words=memoryOf(table.shards[0].storage->serial);save("pos_embed_table.bf16x2",words.data(),words.size()*4);
    const auto position=select("runtime/vision_position.hlsl");
    std::vector<std::pair<uint64_t,uint64_t>> spans;for(auto* p:position)spans.push_back({p->params[1],uint64_t(p->params[1])+p->params[0]});
    EXPECT(covers(spans,uint64_t(patches)*1024));
    const auto& f=emulation::state().findings;
    if(!f.clean())std::cerr<<"vision emulation finding: "<<f.first.front()<<"\n";
    EXPECT(f.clean());
    field("vision_emulation","{\"findings\":"+std::to_string(f.total)+",\"position_dispatches\":"+std::to_string(position.size())+
          ",\"position_groups\":"+std::to_string(position.at(0)->x)+"}");
}

std::string phases="[";
void phase(const char* name) {
    const auto& s=emulation::state();
    phases+=std::string(phases.size()>1?",":"")+"{\"name\":\""+name+"\",\"tracked_live\":"+std::to_string(s.live)+",\"tracked_peak\":"+std::to_string(s.peak)+"}";
}
void visionTrace(VisionOutput& result,std::map<std::string,uint64_t>& serials,Device& d,ModelWeights& w) {
    VisionModel vision(d,w);
    std::vector<float> pixels(size_t(patches)*1536,0.0f);
    auto observer=[&](const std::string& name,const Buffer& b,uint32_t,uint32_t) { serials[name]=b.storage->serial; };
    result=vision.forward(pixels.data(),patches,{{1,gridH,gridW}},observer);
    EXPECT(result.poolerOutput.logicalElements==uint64_t(merged)*2560&&result.lastHiddenState.logicalElements==uint64_t(patches)*1024);
    const auto f=VisionModel::forecast(patches,{{1,gridH,gridW}});
    EXPECT(f.mergedRows==merged&&f.maximumFrameRows==patches&&f.attentionDispatchesPerBlock==378*4);
    const auto& all=emulation::state().dispatches;
    uint32_t maxGroups=0;for(const auto& r:all)maxGroups=std::max({maxGroups,r.x,r.y,r.z});
    // Attention: every block covers each of the 378 query tiles once with 95 key tiles and the 64-key tail.
    auto scores=select("runtime/vision_attention_scores.hlsl"),values=select("runtime/vision_attention_values.hlsl");
    auto softmax=select("runtime/vision_attention_softmax.hlsl"),reduce=select("runtime/vision_attention_reduce.hlsl");
    EXPECT(scores.size()==24*378&&values.size()==scores.size()&&softmax.size()==scores.size()&&reduce.size()==scores.size());
    std::map<uint32_t,uint32_t> starts;
    for(size_t i=0;i<scores.size();i++) {
        const auto& p=scores[i]->params;
        EXPECT(p[0]==0&&p[1]==patches&&p[3]==32&&p[7]==95&&p[8]==1024&&p[9]==16&&p[10]==64&&scores[i]->x==95&&scores[i]->y==512);
        EXPECT(values[i]->params==p&&values[i]->x==95&&values[i]->y==512);
        EXPECT(softmax[i]->params[0]==512&&softmax[i]->params[1]==patches&&softmax[i]->x==512);
        EXPECT(reduce[i]->params[0]==32*1024&&reduce[i]->params[1]==p[2]&&reduce[i]->params[2]==95&&reduce[i]->x==128);
        EXPECT(p[2]%32==0&&p[2]<patches);starts[p[2]]++;
    }
    EXPECT(starts.size()==378);for(auto [s,n]:starts)EXPECT(n==24);
    for(uint32_t b=0;b<24;b++) {
        const auto sdpa=serials.at("blocks."+std::to_string(b)+".sdpa");std::vector<std::pair<uint64_t,uint64_t>> spans;
        for(auto* r:reduce)if(r->outputs[0]==sdpa)spans.push_back({uint64_t(r->params[1])*1024,uint64_t(r->params[1])*1024+r->params[0]});
        EXPECT(spans.size()==378&&covers(spans,uint64_t(patches)*1024));
    }
    // Row-tiled projections: 24 rope tiles per block (23 x 512 + 320) and complete scatter coverage.
    auto rope=select("runtime/vision_rope.hlsl");EXPECT(rope.size()==24*24);
    std::map<uint32_t,uint32_t> ropeRows;for(auto* r:rope) { ropeRows[r->params[1]]=r->params[0];EXPECT(r->params[2]==r->params[0]*1024); }
    EXPECT(ropeRows.size()==24&&ropeRows.at(0)==512&&ropeRows.at(11776)==320);
    auto copies=select("runtime/vision_rows.hlsl");
    auto coverage=[&](uint64_t serial,uint64_t total,size_t expected) {
        std::vector<std::pair<uint64_t,uint64_t>> spans;
        for(auto* r:copies)if(r->outputs[0]==serial)spans.push_back({r->params[2],uint64_t(r->params[2])+r->params[0]});
        return spans.size()==expected&&covers(spans,total);
    };
    EXPECT(coverage(serials.at("patch_embedding"),uint64_t(patches)*1024,24));
    EXPECT(coverage(serials.at("pooler_output"),uint64_t(merged)*2560,6));
    for(uint32_t b=0;b<24;b++)EXPECT(coverage(serials.at("blocks."+std::to_string(b)+".attention_projection"),uint64_t(patches)*1024,24));
    // Merger gathers read contiguous 4-row groups of the per-row LayerNorm at first*4096 (multi-tile only at this geometry).
    std::vector<uint32_t> mergerRows;
    for(auto* r:copies)if(!r->inputs.empty()&&r->inputs[0]==serials.at("merger.norm")) {
        EXPECT(r->params[1]%4096==0&&r->params[0]%4096==0&&r->params[2]==0);mergerRows.push_back(r->params[1]/4096);
    }
    EXPECT((mergerRows==std::vector<uint32_t>{0,512,1024,1536,2048,2560}));
    field("vision_trace","{\"dispatches\":"+std::to_string(all.size())+",\"maximum_group_dimension\":"+std::to_string(maxGroups)+
          ",\"query_tiles_per_block\":"+std::to_string(starts.size())+",\"key_tiles\":95,\"merger_tile_first_rows\":"+join(mergerRows)+
          ",\"peak_tracked_bytes\":"+std::to_string(emulation::state().peak)+"}");
}

void textTrace(const std::vector<int64_t>& positions,Device& d,TextModel& text,TextRequest& request,Buffer& embeddings) {
    TextPositions pos;
    for(uint32_t i=0;i<prompt;i++) { pos.temporal.push_back(uint32_t(positions[i]));pos.height.push_back(uint32_t(positions[prompt+i]));pos.width.push_back(uint32_t(positions[2*prompt+i])); }
    const uint32_t maximum=*std::max_element(pos.temporal.begin(),pos.temporal.end());
    EXPECT(maximum==655);
    std::vector<uint64_t> keys,recurrent;
    for(uint32_t l=0;l<32;l++)(l%4==3?keys:recurrent).push_back((l%4==3?request.layers[l].keys:request.layers[l].recurrent).storage->serial);
    const uint64_t input=embeddings.storage->serial;
    const size_t before=emulation::state().dispatches.size();
    TextResult logits=text.prefill(request,embeddings,pos,true);phase("text_prefill");embeddings={};d.drain();
    EXPECT(request.tokens==prompt&&logits.rows==1);for(const auto& layer:request.layers)EXPECT(layer.length==prompt);
    // Decode positions as inference_core.cpp generate(): every axis maximumPosition + index + 1.
    for(uint32_t step=0;step<decodeSteps;step++) {
        const uint32_t coordinate=maximum+step+1;logits=text.advance(request,248046u-step,TextPositions{{coordinate},{coordinate},{coordinate}});EXPECT(logits.rows==1);
    }
    phase("greedy_cached_decode_and_token_readback");
    EXPECT(request.tokens==prompt+decodeSteps&&request.generated==decodeSteps);
    const auto& all=emulation::state().dispatches;
    // Small position uploads keep their words; map serial -> words.
    std::map<uint64_t,std::vector<uint32_t>> uploads;
    for(const auto& e:emulation::state().events)if(e.kind=="alloc"&&e.initial&&!e.initialWords.empty())uploads[e.serial]=e.initialWords;
    std::vector<uint32_t> rows;std::vector<std::pair<uint64_t,uint64_t>> slices;
    for(size_t i=before;i<all.size();i++)if(all[i].shader=="runtime/text_slice.hlsl"&&all[i].inputs[0]==input) {
        const auto& p=all[i].params;EXPECT(p[1]==2560&&p[3]==2560&&p[2]%2560==0);
        slices.push_back({p[2]/2560,p[2]/2560+p[0]});rows.push_back(p[0]);
    }
    EXPECT(slices.size()==57&&covers(slices,prompt)&&rows.back()==33);
    // Attention layers: cache writes at base = consumed rows, scores over base+rows keys with causal base,
    // and RoPE coordinates equal to the exact recorded positions of those rows (then decode coordinates).
    uint64_t checkedRows=0,attentionCalls=0;
    for(uint64_t k:keys) {
        uint32_t expected=0;
        for(size_t i=before;i<all.size();i++) {
            const auto& r=all[i];
            if(r.shader=="runtime/text_cache.hlsl"&&r.outputs[0]==k) {
                const uint32_t n=r.params[0],base=r.params[4];EXPECT(base==expected&&r.params[1]==1024&&r.x==(n*1024+127)/128);
                // The four preceding rope/score uploads belong to this call; check the queries' rope inputs.
                const DispatchRecord* rope=nullptr;for(size_t j=i;j-->before;)if(all[j].shader=="runtime/text_rope.hlsl") { rope=&all[j];break; }
                EXPECT(rope&&rope->params[0]==n);
                if(rope)for(uint32_t axis=0;axis<3;axis++) {
                    const auto& words=uploads.at(rope->inputs.at(1+axis));EXPECT(words.size()==n);
                    for(uint32_t row=0;row<n;row++) {
                        const uint32_t absolute=base+row;
                        const uint32_t want=absolute<prompt?uint32_t(positions[size_t(axis)*prompt+absolute]):maximum+(absolute-prompt)+1;
                        EXPECT(words[row]==want);checkedRows++;
                    }
                }
                const DispatchRecord* score=nullptr;for(size_t j=i+1;j<all.size();j++)if(all[j].shader=="runtime/text_attention_scores.hlsl") { score=&all[j];break; }
                EXPECT(score&&score->inputs[1]==k&&score->params[4]==base&&score->params[5]==base+n&&score->x==(base+n+127)/128&&score->y==n*16);
                expected+=n;attentionCalls++;
            }
        }
        EXPECT(expected==prompt+decodeSteps);
    }
    // Gated DeltaNet layers: the recurrent state carries through 57 prefill tiles and the decode rows in order.
    uint64_t deltaCalls=0;
    for(uint64_t s:recurrent) {
        std::vector<uint32_t> tokens;
        for(size_t i=before;i<all.size();i++)if(all[i].shader=="runtime/text_delta.hlsl"&&all[i].outputs[0]==s) {
            const auto& p=all[i].params;EXPECT(p[0]==32&&p[1]==128&&p[2]==128&&p[4]==0&&all[i].x==32);tokens.push_back(p[3]);
        }
        EXPECT(tokens.size()==57+decodeSteps);
        uint64_t sum=0;for(size_t t=0;t<tokens.size();t++) { sum+=tokens[t];EXPECT(tokens[t]==(t<56?64u:t==56?33u:1u)); }
        EXPECT(sum==prompt+decodeSteps);deltaCalls+=tokens.size();
    }
    uint32_t maxGroups=0;for(size_t i=before;i<all.size();i++)maxGroups=std::max({maxGroups,all[i].x,all[i].y,all[i].z});
    field("text_trace","{\"dispatches\":"+std::to_string(all.size()-before)+",\"prefill_tiles\":"+std::to_string(slices.size())+
          ",\"last_tile_rows\":"+std::to_string(rows.back())+",\"attention_calls\":"+std::to_string(attentionCalls)+
          ",\"delta_calls\":"+std::to_string(deltaCalls)+",\"rope_coordinates_checked\":"+std::to_string(checkedRows)+
          ",\"maximum_group_dimension\":"+std::to_string(maxGroups)+",\"cache_tokens_after_decode\":"+std::to_string(request.tokens)+"}");
    logits={};text.retire(request);d.drain();phase("request_retirement_and_drain");
}

// The phase order and buffer lifetimes of inference_core.cpp generate() (Windows-only, not compiled here).
void replay(const std::vector<int64_t>& positions) {
    emulation::Config c;c.keepTrace=true;c.defaultMode=emulation::Mode::Stub;emulation::state().reset(c);
    Device d(L"","emulated","emulated");ModelWeights w(d,L"");phase("resident_model");
    std::map<std::string,uint64_t> serials;VisionOutput visual;
    visionTrace(visual,serials,d,w);phase("complete_vision_forward");
    // Allocation-identical stand-in IDs: the image token on the recorded image rows, ordinary text IDs elsewhere.
    std::vector<uint32_t> ids(prompt,198),rowMap(prompt,UINT32_MAX);
    for(uint32_t i=0;i<merged;i++) { ids[firstImage+i]=248056;rowMap[firstImage+i]=i; }
    auto tokens=embedding(d,w.at("model.language_model.embed_tokens.weight"),ids);phase("text_embedding");
    Buffer embeddings;
    {
        auto map=d.words(prompt,rowMap.data());auto output=d.floats(prompt*2560);
        struct Params { uint32_t tokens,width,visionRows,firstElement,count,r0,r1,r2; };
        for(uint32_t first=0;first<prompt*2560;) {
            uint32_t count=std::min(prompt*2560-first,65535u*256u);Params p{prompt,2560,merged,first,count,0,0,0};
            d.dispatch("runtime/multimodal_merge.hlsl",{&tokens,&visual.poolerOutput,&map},{&output},&p,sizeof(p),(count+255)/256);first+=count;
        }
        d.drain();embeddings=output;
    }
    phase("ordered_multimodal_merge");tokens={};visual={};d.drain();
    TextModel text(d,w);TextRequest request=text.newRequest();phase("text_request_cache_creation");
    textTrace(positions,d,text,request,embeddings);
    field("phases",phases+"]");
}

namespace dx=chandra::dc::diagnostics;
struct PosixFile final:dx::File {
    int fd; explicit PosixFile(int f):fd(f) {}
    ~PosixFile() override { if(fd>=0)close(fd); }
    void write(const void* data,size_t n) override {
        auto* p=static_cast<const char*>(data);
        while(n) { auto k=::write(fd,p,n);if(k<=0)throw std::runtime_error("dump write failed");p+=k;n-=size_t(k); }
    }
    void flush() override { if(fsync(fd)!=0)throw std::runtime_error("dump fsync failed"); }
};
struct PosixDirectory final:dx::Directory {
    std::string root;
    explicit PosixDirectory(std::string path):root(std::move(path)) { if(mkdir(root.c_str(),0700)!=0)throw std::runtime_error("Fresh dump directory required"); }
    std::unique_ptr<dx::File> createNew(const std::string& name) override {
        if(!dx::safeName(name))throw std::runtime_error("Unsafe dump filename");
        int fd=open((root+"/"+name).c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        if(fd<0)throw std::runtime_error("Exclusive dump file creation failed: "+name);
        return std::make_unique<PosixFile>(fd);
    }
};
// The hook wiring of inference.cpp execute() and the phase order of inference_core.cpp generate(), both Windows-only.
void dryRun(const std::vector<int64_t>& positions,const std::string& planPath,const std::string& dumpPath,uint32_t cap) {
    emulation::Config c;c.keepTrace=false;c.defaultMode=emulation::Mode::Stub;emulation::state().reset(c);
    dx::Geometry g;g.patchRows=patches;g.mergedRows=merged;g.generationLimit=cap;g.grids={{1,gridH,gridW}};
    g.ids.assign(prompt,198);g.visionRowMap.assign(prompt,UINT32_MAX);
    for(uint32_t i=0;i<merged;i++) { g.ids[firstImage+i]=248056;g.visionRowMap[firstImage+i]=i; }
    for(uint32_t i=0;i<prompt;i++) { g.positions.temporal.push_back(uint32_t(positions[i]));g.positions.height.push_back(uint32_t(positions[prompt+i]));g.positions.width.push_back(uint32_t(positions[2*prompt+i])); }
    std::ifstream file(planPath,std::ios::binary);std::string planText((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    const auto requested=dx::Json::parse(planText);auto plan=dx::resolve(&requested,g);
    std::unique_ptr<Device> device;
    dx::Recorder recorder(plan,g,std::make_unique<PosixDirectory>(dumpPath),[&device](const Buffer& b) { return device->readFloats(b); },
        {{"model_revision","af93b47dba1b47b6640c86ccf487ed2260ab9a09"},{"input_manifest_sha256","0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae"},
         {"dry_run","stubbed CPU replay; payloads are zeros"}},
        {{"kind","emulated_trace_dry_run"},{"scope","Recorder/plan execution check only; no trained, native GPU or numerical value"}});
    device=std::make_unique<Device>(L"","emulated","emulated");recorder.boundary("device_creation");
    ModelWeights w(*device,L"");recorder.boundary("authenticated_model_import_upload");
    const std::string synthetic=dx::sha256("emulated synthetic BF16 weights",31);
    recorder.authenticatedModel({{"revision","af93b47dba1b47b6640c86ccf487ed2260ab9a09"},{"model_sha256",synthetic},{"config_sha256",synthetic},{"model_bytes",0}},{{"emulated",true}});
    Device& d=*device;TextModel text(d,w);TextRequest request;
    {
        VisionModel vision(d,w);std::vector<float> pixels(size_t(patches)*1536,0.0f);
        auto visual=vision.forward(pixels.data(),patches,g.grids,[&](const std::string& name,const Buffer& b,uint32_t rows,uint32_t width) { recorder.vision(name,b,rows,width); });
        recorder.boundary("complete_vision_forward");
        auto tokens=embedding(d,w.at("model.language_model.embed_tokens.weight"),g.ids);recorder.boundary("text_embedding");
        auto map=d.words(prompt,g.visionRowMap.data());auto embeddings=d.floats(prompt*2560);
        struct Params { uint32_t tokens,width,visionRows,firstElement,count,r0,r1,r2; };
        for(uint32_t first=0;first<prompt*2560;) {
            uint32_t count=std::min(prompt*2560-first,65535u*256u);Params p{prompt,2560,merged,first,count,0,0,0};
            d.dispatch("runtime/multimodal_merge.hlsl",{&tokens,&visual.poolerOutput,&map},{&embeddings},&p,sizeof(p),(count+255)/256);first+=count;
        }
        d.drain();map={};recorder.merged(embeddings,prompt,2560);recorder.boundary("ordered_multimodal_merge");
        tokens={};visual={};d.drain();
        request=text.newRequest();recorder.boundary("text_request_cache_creation");
        auto observer=[&](const TextObservation& o) { recorder.text(o); };
        auto logits=text.prefill(request,embeddings,g.positions,true,observer);recorder.boundary("text_prefill");embeddings={};d.drain();
        const uint32_t maximum=655;
        for(uint32_t index=0;index<cap;index++) {
            if(index+1==cap)break;
            const uint32_t coordinate=maximum+index+1;logits=text.advance(request,1000u+index,TextPositions{{coordinate},{coordinate},{coordinate}},observer);
        }
        recorder.boundary("greedy_cached_decode_and_token_readback");
    }
    text.retire(request);d.drain();recorder.boundary("request_retirement_and_drain");
    auto summary=recorder.finish(true,"");
    field("dry_run",summary.dump());
}

void mergeEmulation() {
    emulation::Config c;c.keepTrace=false;emulation::state().reset(c);
    Device d(L"","emulated","emulated");
    std::vector<float> textRows(size_t(prompt)*2560),visionRows(size_t(merged)*2560);
    for(size_t i=0;i<textRows.size();i++)textRows[i]=float(i);
    for(size_t i=0;i<visionRows.size();i++)visionRows[i]=-float(i)-1.0f;
    std::vector<uint32_t> map(prompt,UINT32_MAX);for(uint32_t i=0;i<merged;i++)map[firstImage+i]=i;
    Buffer text=d.floats(uint32_t(textRows.size()),textRows.data()),vision=d.floats(uint32_t(visionRows.size()),visionRows.data());
    Buffer rowMap=d.words(prompt,map.data()),output=d.floats(prompt*2560);
    // Loop and constants of inference_core.cpp merge(); that file is Windows-only and is not compiled here.
    struct Params { uint32_t tokens,width,visionRows,firstElement,count,r0,r1,r2; };
    for(uint32_t first=0;first<prompt*2560;) {
        uint32_t count=std::min(prompt*2560-first,65535u*256u);Params p{prompt,2560,merged,first,count,0,0,0};
        d.dispatch("runtime/multimodal_merge.hlsl",{&text,&vision,&rowMap},{&output},&p,sizeof(p),(count+255)/256);first+=count;
    }
    auto got=d.readFloats(output);uint64_t mismatches=0;
    for(uint32_t r=0;r<prompt;r++)for(uint32_t col=0;col<2560;col++) {
        const float want=map[r]==UINT32_MAX?textRows[size_t(r)*2560+col]:visionRows[size_t(map[r])*2560+col];
        mismatches+=got[size_t(r)*2560+col]!=want;
    }
    EXPECT(mismatches==0);EXPECT(emulation::state().findings.clean());
    field("merge_emulation","{\"rows\":"+std::to_string(prompt)+",\"image_rows\":["+std::to_string(firstImage)+","+std::to_string(firstImage+merged-1)+
          "],\"mismatches\":"+std::to_string(mismatches)+",\"findings\":"+std::to_string(emulation::state().findings.total)+"}");
}
}

int main(int argc,char** argv) {
    if(argc!=3&&argc!=6) { std::cerr<<"usage: recorded-geometry-test POSITIONS_I64 OUTPUT_DIRECTORY [PLAN_JSON DUMP_DIRECTORY TOKEN_CAP]\n";return 2; }
    out=argv[2];
    try {
        std::ifstream f(argv[1],std::ios::binary);std::vector<int64_t> positions(size_t(3)*prompt);
        f.read(reinterpret_cast<char*>(positions.data()),std::streamsize(positions.size()*8));
        if(!f||f.peek()!=EOF)throw std::runtime_error("positions must be exactly 3*3617 int64 values");
        if(argc==6)dryRun(positions,argv[3],argv[4],uint32_t(std::stoul(argv[5])));
        else { visionEmulation();replay(positions);mergeEmulation(); }
    } catch(const std::exception& e) { std::cerr<<"exception: "<<e.what()<<"\n";failures++; }
    std::cout<<"{\"failures\":"<<failures<<",\n"<<report.str()<<"\n}\n";
    return failures?1:0;
}
