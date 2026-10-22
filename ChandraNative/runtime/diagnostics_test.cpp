// New code, MPL-2.0. CPU tests for diagnostics.cpp and the text/vision observer seams with the fake
// Device: no GPU, model bytes or numerical claim. POSIX-only file sink. With one argument (a path that
// must not exist), also writes one complete fake-scenario dump for the Python comparator to verify.
#include "diagnostics_fake_device.h"
#include "diagnostics.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

using namespace chandra::dc;
namespace dx=chandra::dc::diagnostics;
using Json=dx::Json;
namespace {
int failures=0;
#define EXPECT(condition) do { if(!(condition)) { std::cerr<<__FILE__<<":"<<__LINE__<<": expected " #condition "\n";failures++; } } while(0)
template<class F> bool throws(F f,const std::string& fragment="") {
    try { f(); } catch(const std::exception& e) {
        if(fragment.empty()||std::string(e.what()).find(fragment)!=std::string::npos)return true;
        std::cerr<<"unexpected exception text: "<<e.what()<<" (wanted "<<fragment<<")\n";return false;
    }
    return false;
}

struct MemoryState { std::map<std::string,std::string> files; int failCreateAt=-1, creates=0; };
struct MemoryFile final:dx::File {
    std::string& target; explicit MemoryFile(std::string& t):target(t) {}
    void write(const void* data,size_t n) override { target.append(static_cast<const char*>(data),n); }
    void flush() override {}
};
struct MemoryDirectory final:dx::Directory {
    std::shared_ptr<MemoryState> state; explicit MemoryDirectory(std::shared_ptr<MemoryState> s):state(std::move(s)) {}
    std::unique_ptr<dx::File> createNew(const std::string& name) override {
        if(state->failCreateAt==state->creates++)throw std::runtime_error("injected exclusive-create failure");
        if(!dx::safeName(name)||!state->files.emplace(name,"").second)throw std::runtime_error("exclusive create refused: "+name);
        return std::make_unique<MemoryFile>(state->files[name]);
    }
};
struct PosixFile final:dx::File {
    int fd; explicit PosixFile(int f):fd(f) {}
    ~PosixFile() override { if(fd>=0)close(fd); }
    void write(const void* data,size_t n) override {
        auto* p=static_cast<const char*>(data);
        while(n) { auto k=::write(fd,p,n);if(k<=0)throw std::runtime_error("diagnostic write failed");p+=k;n-=size_t(k); }
    }
    void flush() override { if(fsync(fd)!=0)throw std::runtime_error("diagnostic fsync failed"); }
};
struct PosixDirectory final:dx::Directory {
    std::string root;
    explicit PosixDirectory(std::string path):root(std::move(path)) { if(mkdir(root.c_str(),0700)!=0)throw std::runtime_error("Fresh diagnostic directory required"); }
    std::unique_ptr<dx::File> createNew(const std::string& name) override {
        if(!dx::safeName(name))throw std::runtime_error("Unsafe diagnostic filename refused");
        int fd=open((root+"/"+name).c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        if(fd<0)throw std::runtime_error("Exclusive diagnostic file creation failed: "+name);
        return std::make_unique<PosixFile>(fd);
    }
};
std::vector<Json> lines(const std::string& text) {
    std::vector<Json> result;std::istringstream in(text);std::string line;
    while(std::getline(in,line))result.push_back(Json::parse(line));
    return result;
}
// Tiny prepared-input shape: 801 tokens, [1,32,26] grid, 208 image rows, next M-RoPE coordinate 609.
dx::Geometry tiny(uint32_t limit=12384) {
    dx::Geometry g;g.patchRows=832;g.mergedRows=208;g.generationLimit=limit;g.grids={{1,32,26}};
    auto push=[&](uint32_t id,uint32_t t,uint32_t h,uint32_t w,uint32_t m) {
        g.ids.push_back(id);g.positions.temporal.push_back(t);g.positions.height.push_back(h);g.positions.width.push_back(w);g.visionRowMap.push_back(m);
    };
    for(uint32_t i=0;i<10;i++)push(1+i,i,i,i,UINT32_MAX);
    for(uint32_t m=0;m<208;m++)push(248056,10,10+m/13,10+m%13,m);
    for(uint32_t i=0;i<583;i++)push(20+i,26+i,26+i,26+i,UINT32_MAX);
    return g;
}
dx::Geometry scenarioGeometry(const FakeScenario& s) {
    dx::Geometry g;g.patchRows=16;g.mergedRows=4;g.generationLimit=3;g.grids=s.grids;g.ids=s.ids;g.visionRowMap=s.visionRowMap;g.positions=s.positions;return g;
}
Json commitments() {
    return {{"model_revision","af93b47dba1b47b6640c86ccf487ed2260ab9a09"},{"input_manifest_sha256",std::string(64,'1')},{"input",{{"fake_scenario",true}}}};
}
Json provenance() {
    return {{"revision","af93b47dba1b47b6640c86ccf487ed2260ab9a09"},{"model_sha256",std::string(64,'2')},{"config_sha256",std::string(64,'3')},{"model_bytes",1}};
}
Json producer() { return {{"kind","fake_device_cpu_test"},{"numerical_values","synthetic pattern, not model arithmetic"}}; }

void testSha() {
    auto hex=[](const std::string& s) { return dx::sha256(s.data(),s.size()); };
    EXPECT(hex("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT(hex("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string two="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    EXPECT(hex(two)=="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    EXPECT(hex(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    for(size_t split=0;split<=two.size();split++) {
        dx::Sha256 s;s.update(two.data(),split);const auto partial=s.finish();s.update(two.data()+split,two.size()-split);
        EXPECT(s.finish()==hex(two)&&s.finish()==hex(two)&&partial==hex(two.substr(0,split)));
    }
}
void testDefaultPlan() {
    const auto g=tiny();const auto p=dx::resolve(nullptr,g);
    EXPECT((p.visionRows==std::vector<uint32_t>{0,831})&&p.visionBlocks.size()==24&&(p.mergerRows==std::vector<uint32_t>{0,207}));
    EXPECT((p.prefillRows==std::vector<uint32_t>{0,10,217,800})&&p.embeddingRows==p.prefillRows&&p.prefillLayers.size()==32);
    EXPECT((p.decodeSteps==std::vector<uint32_t>{0,1,2,3,4,5,6,7})&&p.decodeLayers.size()==32&&p.states.empty());
    const uint64_t row=2560*4,visionRow=1024*4,logits=248320*4;
    const uint64_t records=2*26+2+4+4*33+1+8*34;
    const uint64_t bytes=2*26*visionRow+2*row+4*row+4*33*row+logits+8*(33*row+logits);
    EXPECT(p.records==records&&p.keys.size()==records&&p.payloadBytes==bytes&&p.payloadBytes<=p.byteLimit);
    // Rows 0 and 10 share prefill tile 0; 217 is tile 3; 800 is the final 33-row tile 12.
    const uint64_t readbackBytes=26*832ull*visionRow+208*row+801*row+33*(64+64+33)*row+logits+8*(33*row+logits);
    EXPECT(p.readbacks==26+1+1+33*3+1+8*34&&p.readbackBytes==readbackBytes&&p.largestReadbackBytes==801ull*row); // Whole merged embedding buffer.
    EXPECT(p.keys.count("text.layer_output/i5/prefill/s-/r800/c-")&&p.keys.count("text.logits/i-/decode/s7/r0/c-")&&p.keys.count("vision.block_output/i23/vision/s-/r831/c-"));
    EXPECT(p.sha256.size()==64&&p.resolved.at("explicit_plan")==false&&dx::resolve(nullptr,g).sha256==p.sha256);
    // A nine-token diagnostic cap still captures decode steps 0..7 (generated indices 1..8); a cap of one has no decode step.
    EXPECT(dx::resolve(nullptr,tiny(9)).decodeSteps.size()==8&&dx::resolve(nullptr,tiny(1)).decodeSteps.empty()&&dx::resolve(nullptr,tiny(4)).decodeSteps.size()==3);
}
void testPlanRefusals() {
    const auto g=tiny(9);
    auto refused=[&](const std::string& text,const std::string& fragment) {
        Json j=Json::parse(text);bool ok=throws([&] { dx::resolve(&j,g); },fragment);
        if(!ok)std::cerr<<"plan was not refused as expected: "<<text<<"\n";
        EXPECT(ok);
    };
    const std::string s="\"schema\":\"chandra.directcompute.diagnostic-plan.v1\"";
    refused("{}","schema");
    refused("{\"schema\":\"other\"}","schema");
    refused("{\"schema\":1}","schema");
    refused("{"+s+",\"extra\":1}","Unknown diagnostic plan key: extra");
    refused("{"+s+",\"vision\":{\"row\":[1]}}","Unknown diagnostic plan key: vision.row");
    refused("{"+s+",\"vision\":[]}","must be an object");
    refused("{"+s+",\"vision\":{\"rows\":[832]}}","outside 832");
    refused("{"+s+",\"vision\":{\"rows\":[1,1]}}","duplicate");
    refused("{"+s+",\"vision\":{\"rows\":[1.0]}}","nonnegative integer");
    refused("{"+s+",\"vision\":{\"rows\":[-1]}}","nonnegative integer");
    refused("{"+s+",\"vision\":{\"rows\":\"all\"}}","array");
    refused("{"+s+",\"vision\":{\"patch_embedding\":1}}","boolean");
    Json many=Json::array();for(int i=0;i<65;i++)many.push_back(i);
    refused("{"+s+",\"prefill\":{\"rows\":"+many.dump()+"}}","at most 64");
    refused("{"+s+",\"prefill\":{\"layers\":[32]}}","outside 32");
    refused("{"+s+",\"decode\":{\"steps\":[8]}}","outside 8");
    refused("{"+s+",\"byte_limit\":268435457}","at most");
    refused("{"+s+",\"byte_limit\":0}","positive");
    refused("{"+s+",\"byte_limit\":1000000}","refused before execution");
    refused("{"+s+",\"vision\":{\"rows\":[],\"patch_embedding\":false},\"merger\":{\"rows\":[]},\"embedding\":{\"rows\":[]},\"prefill\":{\"rows\":[],\"logits\":false},\"decode\":{\"steps\":[]}}","no records");
    refused("{"+s+",\"gdn_state\":{}}","array");
    refused("{"+s+",\"gdn_state\":[{\"layer\":3,\"conv\":true,\"cache_lengths\":[64]}]}","Gated Delta Net");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"cache_lengths\":[64]}]}","conv state or at least one recurrent head");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true}]}","explicit layer and cache_lengths");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true,\"cache_lengths\":[65]}]}","not a completed prefill tile");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true,\"cache_lengths\":[810]}]}","not a completed prefill tile");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true,\"cache_lengths\":[0]}]}","not a completed prefill tile");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true,\"cache_lengths\":[64,64]}]}","duplicate");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true,\"cache_lengths\":[64]},{\"layer\":0,\"recurrent_heads\":[1],\"cache_lengths\":[128]}]}","more than once");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"conv\":true,\"cache_lengths\":[64],\"head\":1}]}","Unknown diagnostic plan key: gdn_state.head");
    refused("{"+s+",\"gdn_state\":[{\"layer\":0,\"recurrent_heads\":[32],\"cache_lengths\":[64]}]}","outside 32");
    Json valid=Json::parse("{"+s+",\"gdn_state\":[{\"layer\":4,\"recurrent_heads\":\"all\",\"cache_lengths\":[808]},{\"layer\":0,\"conv\":true,\"recurrent_heads\":[31,0],\"cache_lengths\":[801,64,802]}],\"decode\":{\"steps\":[7,0],\"layers\":[0,31]}}");
    const auto p=dx::resolve(&valid,g);
    EXPECT(p.states.size()==2&&p.states[0].layer==0&&(p.states[0].heads==std::vector<uint32_t>{0,31})&&(p.states[0].cacheLengths==std::vector<uint32_t>{64,801,802}));
    EXPECT(p.states[1].heads.size()==32&&(p.decodeSteps==std::vector<uint32_t>{0,7})&&p.resolved.at("explicit_plan")==true);
    EXPECT(p.keys.count("text.gdn_conv_state/i0/prefill/s-/r-/c64")&&p.keys.count("text.gdn_recurrent_state/i0/decode/s0/r31/c802")&&p.keys.count("text.gdn_recurrent_state/i4/decode/s6/r17/c808"));
    EXPECT(!p.keys.count("text.layer_output/i5/decode/s0/r0/c-")&&p.keys.count("text.layer_output/i31/decode/s7/r0/c-"));
}
struct ObserverHooks {
    VisionObserver visionObserver; TextObserver textObserver; std::function<void(const Buffer&)> embeddingObserver;
    VisionOutput vision(VisionModel& v,const FakeScenario& s) { return v.forward(s.pixels.data(),16,s.grids,visionObserver); }
    void embeddings(const Buffer& b,const FakeScenario&) { if(embeddingObserver)embeddingObserver(b); }
    TextResult prefill(TextModel& t,TextRequest& r,const Buffer& e,const TextPositions& p) { return t.prefill(r,e,p,true,textObserver); }
    TextResult advance(TextModel& t,TextRequest& r,uint32_t id,const TextPositions& p) { return t.advance(r,id,p,textObserver); }
};
std::vector<std::string> runTrace(ObserverHooks* observers,const std::function<void(Device&)>& attach={}) {
    fakeTrace()=FakeTrace{};
    {
        Device device(L"","fake","fake");ModelWeights weights(device,L"");FakeScenario scenario;
        if(attach)attach(device);
        if(observers)runFakeScenario(device,weights,scenario,*observers);
        else { PlainHooks plain;runFakeScenario(device,weights,scenario,plain); }
    }
    EXPECT(fakeTrace().live==0);
    return fakeTrace().lines;
}
std::vector<std::string> withoutReadback(const std::vector<std::string>& in) {
    std::vector<std::string> out;
    for(const auto& l:in)if(l.rfind("read ",0)!=0&&l.rfind("release ",0)!=0)out.push_back(l);
    return out;
}
size_t reads(const std::vector<std::string>& in) { return size_t(std::count_if(in.begin(),in.end(),[](const std::string& l) { return l.rfind("read ",0)==0; })); }
void testObserverMetadataAndEquivalence() {
    const auto plain=runTrace(nullptr);
    struct Seen { TextStage stage; uint32_t layer,tokensBefore,generatedBefore,callTokens,first,count,cacheLength,rows,width,t,h,w; int64_t token; };
    std::vector<Seen> seen;std::vector<std::string> visionNames;
    ObserverHooks noop;
    noop.visionObserver=[&](const std::string& name,const Buffer&,uint32_t,uint32_t) { visionNames.push_back(name); };
    noop.textObserver=[&](const TextObservation& o) {
        const uint32_t row=o.first+o.count-1;
        seen.push_back({o.stage,o.layer,o.tokensBefore,o.generatedBefore,o.callTokens,o.first,o.count,o.cacheLength,o.rows,o.width,
            o.positions.temporal[row],o.positions.height[row],o.positions.width[row],o.tokenIds?int64_t(o.tokenIds->at(0)):-1});
    };
    const auto observed=runTrace(&noop);
    EXPECT(observed==plain); // Observer presence alone adds no allocation, dispatch, drain, readback or release.
    EXPECT(std::count(visionNames.begin(),visionNames.end(),"blocks.23.output")==1&&std::count(visionNames.begin(),visionNames.end(),"pooler_output")==1);
    // Prefill: 3 tiles x (32 layer outputs + 24 GDN layers x 2 states + final norm) + logits; each decode step: one tile.
    const size_t perTile=32+24*2+1;
    EXPECT(seen.size()==3*perTile+1+2*(perTile+1));
    size_t index=0;
    for(uint32_t tile=0;tile<3;tile++) {
        const uint32_t first=tile*64,count=std::min(64u,130-first);
        for(uint32_t l=0;l<32;l++) {
            const auto& o=seen.at(index++);
            EXPECT(o.stage==TextStage::LayerOutput&&o.layer==l&&o.first==first&&o.count==count&&o.cacheLength==first+count&&o.tokensBefore==0&&o.callTokens==130&&o.rows==count&&o.width==2560&&o.token==-1);
            if(l%4!=3) {
                const auto& conv=seen.at(index++);const auto& rec=seen.at(index++);
                EXPECT(conv.stage==TextStage::ConvState&&conv.layer==l&&conv.rows==8192&&conv.width==4&&conv.cacheLength==first+count);
                EXPECT(rec.stage==TextStage::RecurrentState&&rec.layer==l&&rec.rows==32&&rec.width==16384&&rec.cacheLength==first+count);
            }
        }
        const auto& norm=seen.at(index++);
        EXPECT(norm.stage==TextStage::FinalNorm&&norm.first==first&&norm.count==count&&norm.cacheLength==first+count);
    }
    const auto& logits=seen.at(index++);
    EXPECT(logits.stage==TextStage::Logits&&logits.first==129&&logits.count==1&&logits.cacheLength==130&&logits.width==248320&&logits.t==127);
    for(uint32_t step=0;step<2;step++) {
        for(size_t k=0;k<perTile+1;k++) {
            const auto& o=seen.at(index++);
            EXPECT(o.tokensBefore==130+step&&o.generatedBefore==step&&o.callTokens==1&&o.first==0&&o.count==1&&o.cacheLength==131+step&&
                   o.token==(step?7:5)&&o.t==128+step&&o.h==128+step&&o.w==128+step);
        }
    }
    // Recorder mode: only readbacks (and their drain-driven release timing) differ, and exactly as forecast.
    const FakeScenario scenario;const auto geometry=scenarioGeometry(scenario);
    Json plan={{"schema",dx::planSchema},{"gdn_state",{{{"layer",0},{"conv",true},{"recurrent_heads",{1}},{"cache_lengths",{64,130,131}}}}}};
    const auto resolved=dx::resolve(&plan,geometry);
    auto state=std::make_shared<MemoryState>();std::unique_ptr<dx::Recorder> recorder;
    ObserverHooks recording;
    recording.visionObserver=[&](const std::string& n,const Buffer& b,uint32_t r,uint32_t w) { recorder->vision(n,b,r,w); };
    recording.textObserver=[&](const TextObservation& o) { recorder->text(o); };
    recording.embeddingObserver=[&](const Buffer& b) { recorder->merged(b,130,2560); };
    const auto recorded=runTrace(&recording,[&](Device& device) {
        recorder=std::make_unique<dx::Recorder>(resolved,geometry,std::make_unique<MemoryDirectory>(state),
            [&device](const Buffer& b) { return device.readFloats(b); },commitments(),producer());
        recorder->authenticatedModel(provenance(),{{"fake",true}});
    });
    EXPECT(withoutReadback(recorded)==withoutReadback(plain));
    EXPECT(reads(recorded)==reads(plain)+resolved.readbacks);
    const auto summary=recorder->finish(true,"");
    EXPECT(summary.at("dump_complete")==true&&summary.at("written_records")==resolved.records&&summary.at("payload_bytes")==resolved.payloadBytes&&summary.at("readback_bytes")==resolved.readbackBytes);
    EXPECT(summary.at("progress_sha256")==dx::sha256(state->files["progress.jsonl"].data(),state->files["progress.jsonl"].size()));
    const auto progress=lines(state->files["progress.jsonl"]);
    EXPECT(state->files.size()==resolved.records+1&&progress.front().at("event")=="plan"&&progress.back().at("event")=="finished");
    size_t checked=0;
    for(const auto& line:progress) {
        if(line.at("event")!="record")continue;
        const auto& payload=state->files.at(line.at("file").get<std::string>());
        EXPECT(payload.size()==line.at("payload_bytes").get<uint64_t>()&&dx::sha256(payload.data(),payload.size())==line.at("payload_sha256"));
        EXPECT(line.at("commitments").at("model_sha256")==std::string(64,'2')&&line.at("commitments").at("plan_sha256")==resolved.sha256&&line.at("qualified")==false);
        if(line.at("stage")=="text.layer_output"&&line.at("phase")=="prefill"&&line.at("layer")==7&&line.at("selected_rows")==Json{129}) {
            const auto& c=line.at("coordinates").at(0);
            EXPECT(c.at("absolute_row")==129&&c.at("chunk_index")==2&&c.at("chunk_row")==1&&c.at("token_id")==1120&&(c.at("position")==Json{127,127,127}));
            EXPECT(line.at("cache").at("cache_length_after")==130&&line.at("layer_kind")=="full_attention"&&(line.at("logical_shape")==Json{130,2560}));
            checked++;
        }
        if(line.at("stage")=="text.layer_output"&&line.at("phase")=="prefill"&&line.at("layer")==0&&line.at("selected_rows")==Json{8}) {
            const auto& c=line.at("coordinates").at(0);
            EXPECT(c.at("chunk_index")==0&&c.at("token_id")==248056&&c.at("source")=="vision"&&(c.at("position")==Json{5,6,6})&&line.at("cache").at("cache_length_after")==64);
            checked++;
        }
        if(line.at("stage")=="text.logits"&&line.at("phase")=="decode"&&line.at("decode_step")==1) {
            EXPECT(line.at("decode").at("consumed_token_id")==7&&line.at("produces_generated_index")==2&&line.at("complete_tensor")==true&&line.at("cache").at("cache_length_after")==132);
            EXPECT((line.at("coordinates").at(0).at("position")==Json{129,129,129})&&line.at("coordinates").at(0).at("absolute_row")==131);
            checked++;
        }
        if(line.at("stage")=="text.gdn_recurrent_state"&&line.at("cache_length")==131) {
            EXPECT(line.at("phase")=="decode"&&line.at("decode_step")==0&&line.at("selected_rows")==Json{1}&&line.at("bf16_rounding")=="none_fp32_state"&&(line.at("payload_shape")==Json{1,128,128}));
            EXPECT(line.at("conditioning").at("prefix_rows")==131&&line.at("coordinates").at(0).at("last_absolute_row")==130&&(line.at("coordinates").at(0).at("last_position")==Json{128,128,128}));
            checked++;
        }
        if(line.at("stage")=="vision.block_output"&&line.at("vision_block")==23&&line.at("selected_rows")==Json{15}) {
            const auto& c=line.at("coordinates").at(0);
            EXPECT(c.at("grid_row")==3&&c.at("grid_col")==3&&c.at("merged_row")==3&&line.at("bf16_rounding")=="bf16_round_to_nearest_even_at_graph_boundary");
            checked++;
        }
    }
    EXPECT(checked==5);
}
void testObserverFailurePoisonsRequest() {
    fakeTrace()=FakeTrace{};
    {
        Device device(L"","fake","fake");ModelWeights weights(device,L"");const uint64_t weightBytes=device.trackedBufferBytes();
        FakeScenario s;TextModel text(device,weights);auto request=text.newRequest();
        auto embeddings=embedding(device,weights.at("model.language_model.embed_tokens.weight"),s.ids);
        int calls=0;
        TextObserver failing=[&](const TextObservation& o) { calls++;if(o.stage==TextStage::LayerOutput&&o.layer==5&&o.first==64)throw std::runtime_error("observer failure"); };
        EXPECT(throws([&] { text.prefill(request,embeddings,s.positions,true,failing); },"observer failure"));
        EXPECT(request.failed&&!request.retired&&std::all_of(request.layers.begin(),request.layers.end(),[](const TextLayerCache& c) { return c.failed; }));
        EXPECT(throws([&] { text.advance(request,5,s.decodePositions[0]); },"cannot be reused"));
        EXPECT(throws([&] { text.prefill(request,embeddings,s.positions,true); },"cannot be reused"));
        text.retire(request);embeddings={};device.drain();
        EXPECT(request.retired&&!request.layers[0].conv.storage&&!request.layers[3].keys.storage&&device.trackedBufferBytes()==weightBytes&&calls>0);
    }
    EXPECT(fakeTrace().live==0);
}
void testRecorderFailureLeavesDiscoverableProgress() {
    const FakeScenario scenario;const auto geometry=scenarioGeometry(scenario);const auto plan=dx::resolve(nullptr,geometry);
    auto state=std::make_shared<MemoryState>();state->failCreateAt=60; // progress.jsonl is create 0; fails within text prefill records.
    fakeTrace()=FakeTrace{};
    {
        Device device(L"","fake","fake");ModelWeights weights(device,L"");
        dx::Recorder recorder(plan,geometry,std::make_unique<MemoryDirectory>(state),[&device](const Buffer& b) { return device.readFloats(b); },commitments(),producer());
        recorder.authenticatedModel(provenance(),{{"fake",true}});
        TextModel text(device,weights);auto request=text.newRequest();
        auto embeddings=embedding(device,weights.at("model.language_model.embed_tokens.weight"),scenario.ids);
        EXPECT(throws([&] { text.prefill(request,embeddings,scenario.positions,true,[&](const TextObservation& o) { recorder.text(o); }); },"injected"));
        EXPECT(request.failed);text.retire(request);
        const auto summary=recorder.finish(false,"injected exclusive-create failure");
        EXPECT(summary.at("dump_complete")==false&&summary.at("run_status")=="failed"&&summary.at("written_records")==59);
        EXPECT(recorder.finish(false,"again")==summary);
        const auto progress=lines(state->files["progress.jsonl"]);const auto& last=progress.back();
        EXPECT(last.at("event")=="finished"&&last.at("qualified")==false&&last.at("incomplete_record").is_string()&&last.at("missing_records").size()==plan.records-59);
        EXPECT(progress[progress.size()-2].at("event")=="record_started"&&progress[progress.size()-3].at("event")=="record");
        EXPECT(throws([&] { recorder.boundary("after"); recorder.text(TextObservation{TextStage::Logits,UINT32_MAX,embeddings,1,248320,0,0,130,129,1,130,scenario.positions,nullptr}); },"finished or failed"));
    }
}
void testRecorderGuards() {
    const FakeScenario scenario;const auto geometry=scenarioGeometry(scenario);auto plan=dx::resolve(nullptr,geometry);
    fakeTrace()=FakeTrace{};
    Device device(L"","fake","fake");
    std::vector<float> crafted(16*1024,1.0f);
    crafted[0]=std::numeric_limits<float>::quiet_NaN();crafted[1]=1.0000001f;crafted[15*1024+3]=std::numeric_limits<float>::infinity();
    auto reader=[&](const Buffer&) { return crafted; };
    auto state=std::make_shared<MemoryState>();
    dx::Recorder recorder(plan,geometry,std::make_unique<MemoryDirectory>(state),reader,commitments(),producer());
    auto hidden=device.floats(16*1024);
    EXPECT(throws([&] { recorder.vision("patch_embedding",hidden,16,1024); },"authenticated model commitment"));
    EXPECT(throws([&] { recorder.authenticatedModel({{"revision","0"},{"model_sha256","x"},{"config_sha256","y"},{"model_bytes",1}},{}); },"differs"));
    recorder.authenticatedModel(provenance(),{{"fake",true}});
    EXPECT(throws([&] { recorder.authenticatedModel(provenance(),{}); },"once"));
    EXPECT(throws([&] { recorder.vision("patch_embedding",hidden,16,512); },"shape differs"));
    recorder.vision("patch_embedding",hidden,16,1024);
    EXPECT(throws([&] { recorder.vision("patch_embedding",hidden,16,1024); },"Duplicate diagnostic record"));
    recorder.vision("blocks.0.norm1",hidden,16,1024); // Intermediate observations are ignored without readback.
    auto progress=lines(state->files["progress.jsonl"]);const auto& first=progress.at(3);
    EXPECT(first.at("event")=="record"&&first.at("selected_rows")==Json{0}&&first.at("observed").at("nonfinite")==1&&first.at("observed").at("finite_non_bf16_representable")==1);
    EXPECT(progress.at(5).at("observed").at("nonfinite")==1&&progress.at(5).at("selected_rows")==Json{15});
    auto broken=dx::resolve(nullptr,geometry);broken.byteLimit=4096;
    auto other=std::make_shared<MemoryState>();
    dx::Recorder small(broken,geometry,std::make_unique<MemoryDirectory>(other),reader,commitments(),producer());
    small.authenticatedModel(provenance(),{});
    EXPECT(throws([&] { small.vision("position_added",hidden,16,1024); },"budget"));
    EXPECT(throws([&] { dx::Recorder(plan,geometry,std::make_unique<MemoryDirectory>(state),reader,commitments(),producer()); },"exclusive create refused"));
    // Observations whose coordinates contradict the declared call are refused rather than relabeled.
    const TextPositions one{{128},{128},{128}};const std::vector<uint32_t> token{5};auto row=device.floats(2560);
    EXPECT(throws([&] { recorder.text(TextObservation{TextStage::LayerOutput,0,row,1,2560,130,0,1,0,1,999,one,&token}); },"inconsistent"));
    EXPECT(throws([&] { recorder.text(TextObservation{TextStage::LayerOutput,0,row,1,2560,7,0,1,0,1,8,one,&token}); },"neither"));
    // Decode step 0 before the prompt prefill call has no provable conditioning prefix.
    EXPECT(throws([&] { recorder.text(TextObservation{TextStage::LayerOutput,0,row,1,2560,130,0,1,0,1,131,one,&token}); },"prefix cannot be proven"));
    auto tile=device.floats(64*2560);
    EXPECT(throws([&] { recorder.text(TextObservation{TextStage::LayerOutput,0,tile,64,2560,0,0,130,0,64,64,scenario.positions,nullptr}); },"incomplete buffer"));
    EXPECT(!dx::safeName("../x")&&!dx::safeName("a/b")&&!dx::safeName(".hidden")&&!dx::safeName("Upper")&&!dx::safeName("")&&!dx::safeName("a..b")&&dx::safeName("00001.text.logits.decode.s0.r0.f32"));
}
// Canonical consumed-prefix digest computed independently of the recorder's incremental state.
std::string prefixDigest(const FakeScenario& s,const std::vector<uint32_t>& generated,const std::vector<TextPositions>& positions,size_t rows) {
    std::string text=std::string(dx::prefixSchema)+"\ninput_manifest_sha256 "+std::string(64,'1')+"\nprompt_rows "+std::to_string(s.ids.size())+"\n";
    for(size_t r=0;r<rows;r++) {
        const bool prompt=r<s.ids.size();const size_t g=r-s.ids.size();
        const uint32_t token=prompt?s.ids[r]:generated[g];
        const auto& p=prompt?s.positions:positions[g];const size_t i=prompt?r:0;
        text+=std::to_string(r)+" "+std::to_string(token)+" "+std::to_string(p.temporal[i])+" "+std::to_string(p.height[i])+" "+std::to_string(p.width[i])+"\n";
    }
    return dx::sha256(text.data(),text.size());
}
// Runs the fake scenario with a recorder; returns its progress lines and leaves files in state.
std::vector<Json> recordScenario(const FakeScenario& scenario,const Json& plan,std::shared_ptr<MemoryState> state,bool runCompleted=true) {
    auto geometry=scenarioGeometry(scenario);const auto resolved=dx::resolve(&plan,geometry);
    std::unique_ptr<dx::Recorder> recorder;
    ObserverHooks hooks;
    hooks.visionObserver=[&](const std::string& n,const Buffer& b,uint32_t r,uint32_t w) { recorder->vision(n,b,r,w); };
    hooks.textObserver=[&](const TextObservation& o) { recorder->text(o); };
    hooks.embeddingObserver=[&](const Buffer& b) { recorder->merged(b,130,2560); };
    fakeTrace()=FakeTrace{};
    {
        Device device(L"","fake","fake");ModelWeights weights(device,L"");
        recorder=std::make_unique<dx::Recorder>(resolved,geometry,std::make_unique<MemoryDirectory>(state),
            [&device](const Buffer& b) { return device.readFloats(b); },commitments(),producer());
        recorder->authenticatedModel(provenance(),{{"fake",true}});
        runFakeScenario(device,weights,scenario,hooks);
    }
    recorder->finish(runCompleted,runCompleted?"":"synthetic failure");
    return lines(state->files["progress.jsonl"]);
}
// Sparse decode capture: only step 1 is selected, yet its records commit to the complete prefix
// including the unselected step-0 token, and a different earlier token changes that commitment.
Json sparsePlan() {
    return Json::parse(std::string("{\"schema\":\"")+dx::planSchema+"\",\"vision\":{\"rows\":[3],\"blocks\":[],\"position_added\":false},"
        "\"merger\":{\"rows\":[]},\"embedding\":{\"rows\":[]},\"prefill\":{\"rows\":[63,64],\"layers\":[0,3],\"final_norm\":false,\"logits\":false},"
        "\"decode\":{\"steps\":[1],\"layers\":[0]},\"gdn_state\":[{\"layer\":0,\"conv\":true,\"recurrent_heads\":[2],\"cache_lengths\":[64,130,132]}]}");
}
void testConsumedPrefix() {
    const FakeScenario scenario;
    auto state=std::make_shared<MemoryState>();const auto progress=recordScenario(scenario,sparsePlan(),state);
    std::vector<Json> conditioning,records;
    for(const auto& l:progress) { if(l.at("event")=="conditioning")conditioning.push_back(l);if(l.at("event")=="record")records.push_back(l); }
    EXPECT(progress.front().at("conditioning").at("prompt_rows")==130&&progress.front().at("conditioning").at("schema")==dx::prefixSchema);
    // Prompt rows once at the prefill call, then steps 0 and 1 together before the first step-1 record; no trailing rows remain.
    EXPECT(conditioning.size()==2&&conditioning[0].at("first_row")==0&&conditioning[0].at("rows").size()==130&&conditioning[1].at("first_row")==130);
    EXPECT(conditioning[1].at("rows").size()==2&&(conditioning[1].at("rows")[0]==Json{5,128,128,128})&&(conditioning[1].at("rows")[1]==Json{7,129,129,129}));
    EXPECT(conditioning[0].at("prefix_sha256")==prefixDigest(scenario,scenario.decodeTokens,scenario.decodePositions,130));
    EXPECT(conditioning[1].at("prefix_sha256")==prefixDigest(scenario,scenario.decodeTokens,scenario.decodePositions,132));
    const auto& last=progress.back();
    EXPECT(last.at("consumed_rows")==132&&last.at("consumed_prefix_sha256")==conditioning[1].at("prefix_sha256")&&last.at("dump_complete")==true);
    size_t checked=0;
    for(const auto& r:records) {
        const auto& c=r.at("conditioning");
        if(r.at("stage")=="vision.patch_embedding") { EXPECT(c.is_null());continue; }
        const uint64_t n=c.at("prefix_rows");
        EXPECT(c.at("prefix_sha256")==prefixDigest(scenario,scenario.decodeTokens,scenario.decodePositions,n));
        if(r.at("stage")=="text.layer_output"&&r.at("phase")=="decode") { EXPECT(n==132&&r.at("decode").at("consumed_token_id")==7);checked++; }
        if(r.at("stage")=="text.layer_output"&&r.at("phase")=="prefill") { EXPECT(n==r.at("coordinates").at(0).at("absolute_row").get<uint64_t>()+1);checked++; }
        if(r.at("stage")=="text.gdn_recurrent_state") {
            const auto& coordinate=r.at("coordinates").at(0);
            EXPECT(n==r.at("cache_length")&&coordinate.at("last_absolute_row")==n-1&&!coordinate.contains("state_after_absolute_rows"));
            // Provenance keeps the producing call: the 130-row state came from the final 2-row prefill tile.
            if(n==130)EXPECT(r.at("cache").at("call_first_row")==128&&r.at("cache").at("call_row_count")==2);
            checked++;
        }
    }
    EXPECT(checked==1+4+3);
    // Same current step-1 token, different unselected step-0 token: every step-1 commitment changes.
    FakeScenario divergent;divergent.decodeTokens={6,7};
    auto other=std::make_shared<MemoryState>();const auto changed=recordScenario(divergent,sparsePlan(),other);
    std::map<std::string,Json> before,after;
    for(const auto& l:progress)if(l.at("event")=="record")before[l.at("key")]=l;
    for(const auto& l:changed)if(l.at("event")=="record")after[l.at("key")]=l;
    EXPECT(before.size()==after.size());
    for(const auto& [key,record]:before) {
        const bool decode=record.at("phase")=="decode";
        if(record.at("conditioning").is_null())continue;
        EXPECT((record.at("conditioning")==after.at(key).at("conditioning"))==!decode);
        if(decode)EXPECT(record.at("decode")==after.at(key).at("decode")); // Current consumed token 7 is identical.
    }
    // Early stop: the run completes, unreached step-1 records are missing, the dump is incomplete
    // and the terminal line still commits to the actual consumed prefix.
    FakeScenario early;early.decodeTokens={5};early.decodePositions.resize(1);
    auto stopped=std::make_shared<MemoryState>();const auto partial=recordScenario(early,sparsePlan(),stopped);
    const auto& terminal=partial.back();
    EXPECT(terminal.at("run_status")=="completed"&&terminal.at("dump_complete")==false&&terminal.at("consumed_rows")==131&&terminal.at("error").is_null());
    EXPECT(terminal.at("missing_records").size()==terminal.at("planned_records").get<size_t>()-terminal.at("written_records").get<size_t>()&&!terminal.at("missing_records").empty());
    EXPECT(partial[partial.size()-2].at("event")=="conditioning"&&partial[partial.size()-2].at("first_row")==130); // Trailing unselected step 0.
}
// Repeated callbacks must agree with their call; skipped, repeated or reordered calls are refused.
void testCallIdentity() {
    const FakeScenario scenario;const auto geometry=scenarioGeometry(scenario);
    Json plan=Json::parse(std::string("{\"schema\":\"")+dx::planSchema+"\",\"vision\":{\"rows\":[0],\"blocks\":[],\"position_added\":false},"
        "\"merger\":{\"rows\":[]},\"embedding\":{\"rows\":[]},\"prefill\":{\"rows\":[],\"logits\":false,\"final_norm\":false},\"decode\":{\"steps\":[]}}");
    const auto resolved=dx::resolve(&plan,geometry);
    fakeTrace()=FakeTrace{};Device device(L"","fake","fake");auto value=device.floats(64*2560),one=device.floats(2560);
    auto state=std::make_shared<MemoryState>();
    dx::Recorder recorder(resolved,geometry,std::make_unique<MemoryDirectory>(state),[&device](const Buffer& b) { return device.readFloats(b); },commitments(),producer());
    recorder.authenticatedModel(provenance(),{});
    const std::vector<uint32_t> five{5},six{6},seven{7};
    auto observe=[&](uint32_t before,uint32_t generated,uint32_t tokens,uint32_t first,uint32_t count,const TextPositions& p,const std::vector<uint32_t>* ids) {
        recorder.text(TextObservation{TextStage::LayerOutput,0,count==1?one:value,count,2560,before,generated,tokens,first,count,before+first+count,p,ids});
    };
    TextPositions moved=scenario.positions;moved.width[70]++;
    EXPECT(throws([&] { observe(0,0,130,0,64,moved,nullptr); },"differ from the authenticated prompt"));
    observe(0,0,130,0,64,scenario.positions,nullptr);
    observe(0,0,130,64,64,scenario.positions,nullptr); // Same call, later tile: no new rows.
    EXPECT(throws([&] { observe(0,0,130,64,64,moved,nullptr); },"disagrees with the consumed rows"));
    EXPECT(throws([&] { observe(131,1,1,0,1,scenario.decodePositions[1],&seven); },"prefix cannot be proven"));
    observe(130,0,1,0,1,scenario.decodePositions[0],&five);
    EXPECT(throws([&] { observe(130,0,1,0,1,scenario.decodePositions[0],&six); },"disagrees with the consumed rows"));
    EXPECT(throws([&] { observe(0,0,130,0,64,scenario.positions,nullptr); },"prefix cannot be proven"));
    observe(131,1,1,0,1,scenario.decodePositions[1],&seven);
    const auto summary=recorder.finish(false,"");
    EXPECT(summary.at("consumed_rows")==132&&summary.at("run_status")=="failed");
    const auto progress=lines(state->files["progress.jsonl"]);const auto& terminal=progress.back();
    EXPECT(terminal.at("error")=="run failed without an error message"&&terminal.at("consumed_prefix_sha256")==prefixDigest(scenario,{5,7},scenario.decodePositions,132));
    EXPECT(progress[progress.size()-2].at("event")=="conditioning"&&progress[progress.size()-2].at("rows").size()==132);
}
void saveFiles(const MemoryState& state,const std::string& directory) {
    PosixDirectory out(directory);
    for(const auto& [name,data]:state.files) { auto f=out.createNew(name);f->write(data.data(),data.size());f->flush(); }
}
int writeDump(const std::string& directory) {
    const FakeScenario scenario;const auto geometry=scenarioGeometry(scenario);
    Json plan={{"schema",dx::planSchema},{"gdn_state",{{{"layer",0},{"conv",true},{"recurrent_heads",{1}},{"cache_lengths",{64,130,131}}}}}};
    const auto resolved=dx::resolve(&plan,geometry);std::unique_ptr<dx::Recorder> recorder;
    ObserverHooks hooks;
    hooks.visionObserver=[&](const std::string& n,const Buffer& b,uint32_t r,uint32_t w) { recorder->vision(n,b,r,w); };
    hooks.textObserver=[&](const TextObservation& o) { recorder->text(o); };
    hooks.embeddingObserver=[&](const Buffer& b) { recorder->merged(b,130,2560); };
    runTrace(&hooks,[&](Device& device) {
        recorder=std::make_unique<dx::Recorder>(resolved,geometry,std::make_unique<PosixDirectory>(directory),
            [&device](const Buffer& b) { return device.readFloats(b); },commitments(),producer());
        recorder->authenticatedModel(provenance(),{{"fake",true}});
        recorder->boundary("fake_device_created");
    });
    std::cout<<recorder->finish(true,"").dump()<<'\n';return 0;
}
// Complete, sparse-decode, genuine early-stop and failed-record dumps for the Python comparator.
int writeDumps(const std::string& complete,const std::string& sparse,const std::string& early,const std::string& failed) {
    const int status=writeDump(complete);
    auto state=std::make_shared<MemoryState>();recordScenario(FakeScenario{},sparsePlan(),state);saveFiles(*state,sparse);
    FakeScenario stopped;stopped.decodeTokens={5};stopped.decodePositions.resize(1);
    state=std::make_shared<MemoryState>();recordScenario(stopped,sparsePlan(),state);saveFiles(*state,early);
    const FakeScenario scenario;const auto geometry=scenarioGeometry(scenario);const auto plan=dx::resolve(nullptr,geometry);
    state=std::make_shared<MemoryState>();state->failCreateAt=60;
    fakeTrace()=FakeTrace{};
    {
        Device device(L"","fake","fake");ModelWeights weights(device,L"");
        dx::Recorder recorder(plan,geometry,std::make_unique<MemoryDirectory>(state),[&device](const Buffer& b) { return device.readFloats(b); },commitments(),producer());
        recorder.authenticatedModel(provenance(),{{"fake",true}});
        TextModel text(device,weights);auto request=text.newRequest();
        auto embeddings=embedding(device,weights.at("model.language_model.embed_tokens.weight"),scenario.ids);
        try { text.prefill(request,embeddings,scenario.positions,true,[&](const TextObservation& o) { recorder.text(o); }); } catch(const std::exception&) {}
        text.retire(request);recorder.finish(false,"injected exclusive-create failure");
    }
    saveFiles(*state,failed);
    return status;
}
}

int main(int argc,char** argv) {
    try {
        testSha();testDefaultPlan();testPlanRefusals();testObserverMetadataAndEquivalence();
        testObserverFailurePoisonsRequest();testRecorderFailureLeavesDiscoverableProgress();testRecorderGuards();
        testConsumedPrefix();testCallIdentity();
        if(argc==5&&!failures)return writeDumps(argv[1],argv[2],argv[3],argv[4]);
        if(argc==2&&!failures)return writeDump(argv[1]);
    } catch(const std::exception& e) { std::cerr<<"unexpected exception: "<<e.what()<<'\n';return 1; }
    if(failures)std::cerr<<failures<<" diagnostic expectation(s) failed\n";else std::cerr<<"diagnostics tests passed\n";
    return failures?1:0;
}
