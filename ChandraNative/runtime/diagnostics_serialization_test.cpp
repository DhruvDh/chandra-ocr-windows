// New code, MPL-2.0. Portable CPU regression for the JSON shape of progress.jsonl written by diagnostics.cpp
// under the vendored nlohmann/json. Standard C++17 only, so the same file builds with GCC, Clang and MSVC:
// Build09's MSVC recorder wrote bare coordinate objects that the POSIX-only diagnostics_test.cpp never saw.
// It drives the real Recorder through the fake-Device graph across every recorded stage, checks the exact
// JSON type of every field of every emitted line, and pins each dump's progress SHA-256, which must be
// identical on every compiler. With one argument (a path that must not exist) it also writes the dumps and
// summary.json for scripts/native/compare_diagnostics.py. No GPU, D3D11, model bytes or numerical claim:
// payloads are a deterministic bit pattern, not graph values.
#include "diagnostics_fake_device.h"
#include "diagnostics.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>

using namespace chandra::dc;
namespace dx=chandra::dc::diagnostics;
using Json=dx::Json;
namespace {
int failures=0;
#define EXPECT(condition) do { if(!(condition)) { std::cerr<<__FILE__<<":"<<__LINE__<<": expected " #condition "\n";failures++; } } while(0)

// SHA-256 of each scenario's progress.jsonl from GCC 16.2.1 and Clang 23.1.1, byte-identical with the
// pre-fix recorder on those compilers. Any other compiler must reproduce these bytes exactly.
const std::map<std::string,std::string> expectedProgress={
    {"complete","5a1a1768a541e09f2f0067adc26f3c5485777a89a74a83dd4e3318138631c29b"},
    {"early_stop","3463b7484bc79222d1295a792b45f951e42e3a01f886ac72a442e9eea7083564"},
    {"failed","3a2574561ec61ef5a29c58e4a3fb04aee25a038cfa3f2af2d02ab61d3963fe03"}};

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
// Payload words depend only on the recorder's read order and element index, never on fake allocation
// serials, so dump bytes isolate serialization. Some reads start with a NaN; FP32 GDN state reads (conv
// 8192x4 or recurrent 32x16384 words, no BF16 claim) end with a finite non-BF16 word.
dx::Reader patternReader() {
    auto reads=std::make_shared<uint64_t>(0);
    return [reads](const Buffer& b) {
        const uint64_t k=(*reads)++;std::vector<uint32_t> words(b.words);
        for(uint32_t i=0;i<b.words;i++) { const float v=float(int((k*131+uint64_t(i)*7)%251)-125);std::memcpy(&words[i],&v,4); }
        if(k%7==3)words.front()=0x7fc00000u;
        if(b.words==8192*4||b.words==32*16384)words.back()=0x3f800001u;
        std::vector<float> values(words.size());std::memcpy(values.data(),words.data(),words.size()*4);return values;
    };
}
// Synthetic identities shaped like inference.cpp's commitments, producer and model provenance.
Json commitments(const Json& planFile) {
    auto c=Json::parse(R"({"model_revision":"synthetic-revision-0001","input_manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "input":{"prompt_tokens":130,"patch_rows":16,"merged_rows":4,"image_grid_thw":[[1,4,4]],"next_decode_position":128,
        "stop_token_ids":[2],"generation_limit":3,"diagnostic_token_cap":3}})");
    c["plan_file"]=planFile;return c;
}
Json producer() {
    return Json::parse(R"({"kind":"fake_device_cpu_serialization_test","executable_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "shader_root":{"files":0,"tree_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","canonical":"synthetic"},
        "scope":"Synthetic producer identity; no executable or shader tree"})");
}
Json provenance() {
    return Json::parse(R"({"revision":"synthetic-revision-0001","model_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "config_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","model_bytes":1})");
}
dx::Geometry geometry(const FakeScenario& s) {
    dx::Geometry g;g.patchRows=16;g.mergedRows=4;g.generationLimit=3;g.grids=s.grids;g.ids=s.ids;g.visionRowMap=s.visionRowMap;g.positions=s.positions;return g;
}
// Every recorded stage: vision rows, all blocks, merger, text and image embedding rows, prefill rows in
// three tiles over GDN and full-attention layers, both decode steps, and GDN state at a mid-prompt tile,
// the prompt end and both decode steps.
Json explicitPlan() {
    return Json::parse(std::string(R"({"schema":")")+dx::planSchema+R"(","vision":{"rows":[0,15]},"merger":{"rows":[0,3]},
        "embedding":{"rows":[0,5,8,129]},"prefill":{"rows":[0,5,63,64,129],"layers":[0,3,30,31]},"decode":{"steps":[0,1],"layers":[0,3]},
        "gdn_state":[{"layer":0,"conv":true,"recurrent_heads":[0,31],"cache_lengths":[64,130,131,132]},{"layer":4,"recurrent_heads":[1],"cache_lengths":[128]}]})");
}

struct Hooks {
    dx::Recorder* recorder=nullptr;
    TextObserver text() { return [this](const TextObservation& o) { recorder->text(o); }; }
    VisionOutput vision(VisionModel& v,const FakeScenario& s) {
        auto out=v.forward(s.pixels.data(),16,s.grids,[this](const std::string& n,const Buffer& b,uint32_t r,uint32_t w) { recorder->vision(n,b,r,w); });
        recorder->boundary("complete_vision_forward");return out;
    }
    void embeddings(const Buffer& b,const FakeScenario& s) { recorder->merged(b,uint32_t(s.ids.size()),2560);recorder->boundary("ordered_multimodal_merge"); }
    TextResult prefill(TextModel& t,TextRequest& r,const Buffer& e,const TextPositions& p) { auto out=t.prefill(r,e,p,true,text());recorder->boundary("text_prefill");return out; }
    TextResult advance(TextModel& t,TextRequest& r,uint32_t id,const TextPositions& p) { return t.advance(r,id,p,text()); }
};
// Runs the fake graph with a recorder in inference.cpp's order: boundaries before the model commitment,
// so the first record is progress line 6 as in Build09. Returns the finish() summary.
Json record(const FakeScenario& scenario,const Json* plan,std::shared_ptr<MemoryState> state,uint64_t* planned=nullptr) {
    const auto g=geometry(scenario);const auto resolved=dx::resolve(plan,g);if(planned)*planned=resolved.records;
    const Json planFile=plan?Json::parse(R"({"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","bytes":512})"):Json(nullptr);
    dx::Recorder recorder(resolved,g,std::make_unique<MemoryDirectory>(state),patternReader(),commitments(planFile),producer());
    std::string error;
    fakeTrace()=FakeTrace{};
    try {
        Device device(L"","fake","fake");recorder.boundary("device_creation");
        ModelWeights weights(device,L"");recorder.boundary("authenticated_model_import_upload");
        recorder.authenticatedModel(provenance(),Json::parse(R"({"fake":true})"));
        Hooks hooks;hooks.recorder=&recorder;runFakeScenario(device,weights,scenario,hooks);
    } catch(const std::exception& e) { error=e.what(); }
    return recorder.finish(error.empty(),error);
}

// Exact progress schema, checked on the re-parsed text so nonnegative integers are unsigned numbers.
struct Checker {
    std::vector<std::string> errors; std::string where; Json header, model;
    void fail(const std::string& what) { errors.push_back(where+": "+what); }
    void require(bool ok,const std::string& what) { if(!ok)fail(what); }
    static bool natural(const Json& j) { return j.is_number_unsigned(); }
    static bool hex(const Json& j) { return j.is_string()&&j.get<std::string>().size()==64&&j.get<std::string>().find_first_not_of("0123456789abcdef")==std::string::npos; }
    static bool position(const Json& j) { return j.is_array()&&j.size()==3&&std::all_of(j.begin(),j.end(),natural); }
    static bool shape(const Json& j) { return j.is_array()&&!j.empty()&&std::all_of(j.begin(),j.end(),[](const Json& d) { return natural(d)&&d.get<uint64_t>()>0; }); }
    bool fields(const Json& o,std::initializer_list<const char*> names,const std::string& what) {
        bool ok=o.is_object()&&o.size()==names.size();
        for(const char* n:names)ok=ok&&o.contains(n);
        require(ok,what+" must be an object with exactly its schema fields");return ok;
    }
    void naturals(const Json& o,std::initializer_list<const char*> names) { for(const char* n:names)require(natural(o.at(n)),std::string(n)+" must be a nonnegative integer"); }
    void line(const Json& j,size_t number,bool last) {
        where="progress line "+std::to_string(number);
        if(!j.is_object()||!j.contains("event")||!j.at("event").is_string()) { fail("event object required");return; }
        const auto event=j.at("event").get<std::string>();
        require((number==1)==(event=="plan")&&last==(event=="finished"),"plan must be first and finished last");
        if(event=="plan") {
            if(!fields(j,{"event","schema","plan","plan_sha256","commitments","producer","qualified","conditioning","scope"},"plan line"))return;
            require(j.at("schema")==dx::progressSchema&&j.at("plan").is_object()&&j.at("plan").value("schema","")==dx::planSchema&&hex(j.at("plan_sha256")),"plan schema and digest");
            require(j.at("commitments").is_object()&&j.at("commitments").value("plan_sha256","")==j.at("plan_sha256")&&j.at("commitments").at("model_revision").is_string()&&
                    j.at("commitments").at("input_manifest_sha256").is_string()&&j.at("producer").is_object()&&j.at("qualified")==false&&j.at("scope").is_string(),"plan commitments, producer and qualification");
            if(fields(j.at("conditioning"),{"schema","prompt_rows"},"plan conditioning"))require(j.at("conditioning").at("schema")==dx::prefixSchema&&natural(j.at("conditioning").at("prompt_rows")),"conditioning declaration");
            header=j;
        } else if(event=="model_authenticated") {
            if(!fields(j,{"event","model","device"},"model line")||!fields(j.at("model"),{"revision","model_sha256","config_sha256","model_bytes"},"model"))return;
            const auto& m=j.at("model");
            require(m.at("revision").is_string()&&m.at("model_sha256").is_string()&&m.at("config_sha256").is_string()&&natural(m.at("model_bytes"))&&j.at("device").is_object(),"model field types");
            model=m;
        } else if(event=="boundary") {
            if(fields(j,{"event","name","completed_records","payload_bytes"},"boundary line")) { require(j.at("name").is_string(),"boundary name");naturals(j,{"completed_records","payload_bytes"}); }
        } else if(event=="record_started") {
            if(fields(j,{"event","sequence","file","key"},"record start")) { require(j.at("file").is_string()&&j.at("key").is_string(),"start names");naturals(j,{"sequence"}); }
        } else if(event=="conditioning") {
            if(!fields(j,{"event","first_row","rows","consumed_rows","prefix_sha256"},"conditioning line"))return;
            naturals(j,{"first_row","consumed_rows"});require(hex(j.at("prefix_sha256")),"conditioning digest");
            const auto& rows=j.at("rows");
            require(rows.is_array()&&!rows.empty()&&std::all_of(rows.begin(),rows.end(),[](const Json& r) { return r.is_array()&&r.size()==4&&std::all_of(r.begin(),r.end(),natural); }),
                    "conditioning rows must be [token_id, t, h, w] integer arrays");
        } else if(event=="finished") {
            if(!fields(j,{"event","run_status","dump_complete","planned_records","written_records","missing_records","incomplete_record","payload_bytes",
                          "readbacks","readback_bytes","error","consumed_rows","consumed_prefix_sha256","qualified"},"finished line"))return;
            naturals(j,{"planned_records","written_records","payload_bytes","readbacks","readback_bytes","consumed_rows"});
            const auto& missing=j.at("missing_records");const bool completed=j.at("run_status")=="completed";
            require((completed||j.at("run_status")=="failed")&&j.at("dump_complete").is_boolean()&&j.at("qualified")==false&&hex(j.at("consumed_prefix_sha256")),"terminal status types");
            require(missing.is_array()&&std::all_of(missing.begin(),missing.end(),[](const Json& k) { return k.is_string(); }),"missing_records must be an array of keys");
            require(j.at("incomplete_record").is_null()||j.at("incomplete_record").is_string(),"incomplete_record must be null or a key");
            require(completed?j.at("error").is_null():j.at("error").is_string()&&!j.at("error").get<std::string>().empty(),"error must be null exactly for a completed run");
        } else if(event=="record") record(j);
        else fail("unknown event "+event);
    }
    void record(const Json& r) {
        if(!fields(r,{"event","schema","sequence","file","key","stage","component","phase","layer","layer_kind","vision_block","decode_step","cache_length",
                      "logical_shape","selected_axis","selected_rows","payload_shape","complete_tensor","storage_dtype","byte_order","bf16_rounding",
                      "payload_bytes","payload_sha256","observed","coordinates","cache","decode","conditioning","produces_generated_index",
                      "commitments","producer","qualified"},"record"))return;
        if(!r.at("stage").is_string()||!r.at("phase").is_string()) { fail("stage and phase must be strings");return; }
        const auto stage=r.at("stage").get<std::string>(),phase=r.at("phase").get<std::string>();
        const bool text=stage.rfind("text.",0)==0,state=stage=="text.gdn_conv_state"||stage=="text.gdn_recurrent_state";
        const bool rowStage=stage=="text.layer_output"||stage=="text.final_norm"||stage=="text.logits",embedding=stage=="text.merged_embedding";
        const bool patch=stage=="vision.patch_embedding"||stage=="vision.position_added"||stage=="vision.block_output",merger=stage=="vision.merger_output";
        require(patch||merger||embedding||rowStage||state,"unknown stage "+stage);
        const bool decode=phase=="decode",causal=rowStage||state;
        require(causal?decode||phase=="prefill":phase==(embedding?"merge":"vision"),"phase differs from stage");
        require(r.at("schema")==dx::recordSchema&&natural(r.at("sequence"))&&r.at("file").is_string()&&r.at("key").is_string()&&r.at("component")==(text?"text":"vision"),"record identity types");
        const auto& layer=r.at("layer");
        if(stage=="text.layer_output"||state) {
            require(natural(layer)&&layer.get<uint64_t>()<32,"layer must be a text layer index");
            if(natural(layer))require(r.at("layer_kind")==(layer.get<uint64_t>()%4==3?"full_attention":"gated_delta_net"),"layer_kind differs from layer");
        } else require(layer.is_null()&&r.at("layer_kind").is_null(),"layer and layer_kind must be null");
        require(stage=="vision.block_output"?natural(r.at("vision_block")):r.at("vision_block").is_null(),"vision_block type");
        require(decode?natural(r.at("decode_step")):r.at("decode_step").is_null(),"decode_step type");
        require(state?natural(r.at("cache_length")):r.at("cache_length").is_null(),"cache_length type");
        require(r.at("selected_axis")==0u&&natural(r.at("selected_axis")),"selected_axis must be integer 0");
        const auto& selected=r.at("selected_rows");const auto& logical=r.at("logical_shape");const auto& payload=r.at("payload_shape");
        const bool all=stage=="text.gdn_conv_state";
        require(all?selected=="all":selected.is_array()&&selected.size()==1&&natural(selected[0]),"selected_rows must be one-element integer array (\"all\" only for conv state)");
        require(shape(logical)&&shape(payload),"logical and payload shapes must be positive integer arrays");
        if(shape(logical)&&shape(payload)) {
            Json expected=logical;if(!all)expected[0]=1u;
            require(payload==expected,"payload_shape differs from the selection");
            uint64_t elements=1;for(const auto& d:payload)elements*=d.get<uint64_t>();
            require(natural(r.at("payload_bytes"))&&r.at("payload_bytes").get<uint64_t>()==elements*4,"payload_bytes differs from payload_shape");
        }
        require(r.at("complete_tensor").is_boolean()&&r.at("storage_dtype")=="float32"&&r.at("byte_order")=="little"&&hex(r.at("payload_sha256")),"storage field types");
        require(r.at("bf16_rounding")==(state?"none_fp32_state":"bf16_round_to_nearest_even_at_graph_boundary"),"bf16_rounding differs from stage");
        if(fields(r.at("observed"),{"nonfinite","finite_non_bf16_representable"},"observed"))naturals(r.at("observed"),{"nonfinite","finite_non_bf16_representable"});
        // The Build09 defect: coordinates must be an array holding exactly one object, for every stage.
        const auto& coordinates=r.at("coordinates");
        if(!coordinates.is_array()||coordinates.size()!=1||!coordinates[0].is_object()) { fail("coordinates must be an array of exactly one object");return; }
        const auto& c=coordinates[0];
        if(patch) {
            if(fields(c,{"patch_row","frame","grid_row","grid_col","merged_row"},"vision coordinate")) { naturals(c,{"patch_row","frame","grid_row","grid_col","merged_row"});require(c.at("patch_row")==selected[0],"patch_row differs from selection"); }
        } else if(merger) {
            if(fields(c,{"merged_row","prompt_row"},"merger coordinate")) { naturals(c,{"merged_row","prompt_row"});require(c.at("merged_row")==selected[0],"merged_row differs from selection"); }
        } else if(embedding) {
            if(fields(c,{"absolute_row","token_id","source","merged_row","position"},"embedding coordinate")) {
                naturals(c,{"absolute_row","token_id"});const bool image=c.at("source")=="vision";
                require((image||c.at("source")=="text")&&(image?natural(c.at("merged_row")):c.at("merged_row").is_null())&&position(c.at("position"))&&c.at("absolute_row")==selected[0],"embedding coordinate types");
            }
        } else if(rowStage) {
            const bool ok=decode?fields(c,{"absolute_row","call_row","chunk_index","chunk_row","position","token_id"},"decode row coordinate")
                                :fields(c,{"absolute_row","call_row","chunk_index","chunk_row","position","token_id","source"},"prefill row coordinate");
            if(ok) { naturals(c,{"absolute_row","call_row","chunk_index","chunk_row","token_id"});require(position(c.at("position"))&&(decode||c.at("source")=="vision"||c.at("source")=="text"),"row coordinate types"); }
        } else if(fields(c,{"last_absolute_row","last_position"},"state coordinate")) {
            naturals(c,{"last_absolute_row"});require(position(c.at("last_position")),"last_position must be three integers");
        }
        const auto& cache=r.at("cache");
        if(causal) {
            if(fields(cache,{"cache_length_after","cache_length_source","request_tokens_before_call","generated_before_call","call_tokens","call_first_row","call_row_count"},"cache")) {
                naturals(cache,{"cache_length_after","request_tokens_before_call","generated_before_call","call_tokens","call_first_row","call_row_count"});
                require(cache.at("cache_length_source")==(stage=="text.final_norm"?"last_layer":stage=="text.logits"?"request":"layer"),"cache_length_source differs from stage");
            }
            if(fields(r.at("conditioning"),{"prefix_rows","prefix_sha256"},"conditioning"))require(natural(r.at("conditioning").at("prefix_rows"))&&hex(r.at("conditioning").at("prefix_sha256")),"conditioning types");
        } else require(cache.is_null()&&r.at("conditioning").is_null(),"vision and merge records carry null cache and conditioning");
        if(decode) {
            if(fields(r.at("decode"),{"step","consumed_generated_index","consumed_token_id","logits_produce_generated_index"},"decode")) {
                naturals(r.at("decode"),{"step","consumed_generated_index","consumed_token_id","logits_produce_generated_index"});
                require(r.at("decode").at("step")==r.at("decode_step"),"decode.step differs from decode_step");
            }
        } else require(r.at("decode").is_null(),"non-decode records carry null decode");
        require(stage=="text.logits"?natural(r.at("produces_generated_index")):r.at("produces_generated_index").is_null(),"produces_generated_index type");
        Json expected=header.value("commitments",Json());
        for(const char* k:{"model_sha256","config_sha256","model_bytes"})if(model.contains(k))expected[k]=model.at(k);
        require(r.at("commitments")==expected&&r.at("producer")==header.value("producer",Json())&&r.at("qualified")==false,"record commitments, producer and qualification");
    }
};
std::vector<std::string> schemaErrors(const std::string& progress,size_t* lines=nullptr) {
    Checker checker;std::vector<std::string> text;std::string line;
    for(size_t start=0;start<progress.size();) {
        const auto end=progress.find('\n',start);
        if(end==std::string::npos) { checker.errors.push_back("progress ends with a partial line");break; }
        text.push_back(progress.substr(start,end-start));start=end+1;
    }
    Json started;
    for(size_t i=0;i<text.size();i++) {
        Json j;
        try { j=Json::parse(text[i]); } catch(const std::exception& e) { checker.errors.push_back("progress line "+std::to_string(i+1)+": "+e.what());continue; }
        checker.line(j,i+1,i+1==text.size());
        const auto event=j.value("event","");
        if(event=="record_started")started=j;
        if(event=="record") {
            checker.require(!started.is_null()&&started.at("sequence")==j.at("sequence")&&started.at("file")==j.at("file")&&started.at("key")==j.at("key"),"record without its matching start");
            started=nullptr;
        }
    }
    if(lines)*lines=text.size();
    return checker.errors;
}
// The checker must refuse the Build09 shape and its neighbours, or a passing run proves nothing.
size_t negativeControls(const std::string& progress) {
    std::vector<std::string> text;size_t start=0;
    for(auto end=progress.find('\n');end!=std::string::npos;start=end+1,end=progress.find('\n',start))text.push_back(progress.substr(start,end-start));
    auto refused=[&](const std::function<bool(Json&)>& mutate) {
        for(auto& t:text) {
            auto j=Json::parse(t);if(j.at("event")!="record"||!mutate(j))continue;
            const auto original=t;t=j.dump();std::string joined;for(const auto& l:text)joined+=l+"\n";t=original;
            return !schemaErrors(joined).empty();
        }
        return false;
    };
    size_t passed=0;
    auto control=[&](bool ok,const char* name) { if(ok)passed++;else { std::cerr<<"schema checker accepted negative control: "<<name<<"\n";failures++; } };
    control(refused([](Json& j) { if(j.at("stage")!="vision.patch_embedding")return false;j["coordinates"]=Json(j.at("coordinates").at(0));return true; }),"bare vision coordinate object");
    control(refused([](Json& j) { if(j.at("stage")!="text.layer_output"||j.at("phase")!="decode")return false;j["coordinates"]=Json(j.at("coordinates").at(0));return true; }),"bare decode coordinate object");
    control(refused([](Json& j) { if(j.at("stage")!="text.gdn_conv_state")return false;j["coordinates"]=Json::array();return true; }),"empty coordinates");
    control(refused([](Json& j) { if(j.at("stage")!="text.logits")return false;j["selected_rows"]=Json(j.at("selected_rows").at(0));return true; }),"scalar selected row");
    control(refused([](Json& j) { if(j.at("phase")!="decode")return false;j["decode"]=Json::array();j["decode"].push_back(1);return true; }),"array decode metadata");
    control(refused([](Json& j) { if(j.at("stage")!="text.final_norm")return false;j["cache"]["call_tokens"]=-1;return true; }),"negative cache field");
    control(refused([](Json& j) { if(j.at("stage")!="text.merged_embedding")return false;j["coordinates"][0]["position"]=Json::array();j["coordinates"][0]["position"].push_back(1);return true; }),"short position");
    control(refused([](Json& j) { if(j.at("stage")!="vision.merger_output")return false;j["layer"]=0;return true; }),"vision record with layer");
    return passed;
}
// Records this compiler's reading of the construct that broke Build09; the recorder no longer depends on it.
Json objectValue() { return Json::parse(R"({"k":1})"); }
Json braceProbe() {
    Json value=objectValue(),lvalue,prvalue;lvalue={value};prvalue={objectValue()};
    auto kind=[](const Json& j) { return std::string(j.is_array()?"array":j.is_object()?"object":"other"); };
    return {{"lvalue",kind(lvalue)},{"prvalue",kind(prvalue)}};
}
std::string compiler() {
#if defined(__clang__)
    return std::string("clang ")+__clang_version__;
#elif defined(_MSC_VER)
    return "msvc "+std::to_string(_MSC_FULL_VER);
#elif defined(__GNUC__)
    return "gcc "+std::to_string(__GNUC__)+"."+std::to_string(__GNUC_MINOR__)+"."+std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown";
#endif
}
void save(const MemoryState& state,const std::filesystem::path& directory) {
    if(!std::filesystem::create_directory(directory))throw std::runtime_error("Fresh dump directory required: "+directory.string());
    for(const auto& [name,data]:state.files) {
        if(!dx::safeName(name))throw std::runtime_error("Unsafe dump filename: "+name);
        std::ofstream out(directory/name,std::ios::binary);out.write(data.data(),std::streamsize(data.size()));out.close();
        if(!out)throw std::runtime_error("Writing dump file failed: "+name);
    }
}
}

int main(int argc,char** argv) {
    if(argc>2) { std::cerr<<"usage: diagnostics-serialization-test [FRESH_OUTPUT_DIRECTORY]\n";return 2; }
    Json summary={{"test","directcompute_diagnostic_serialization"},{"compiler",compiler()},{"nlohmann_json",std::to_string(NLOHMANN_JSON_VERSION_MAJOR)+"."+
        std::to_string(NLOHMANN_JSON_VERSION_MINOR)+"."+std::to_string(NLOHMANN_JSON_VERSION_PATCH)},{"single_json_brace_assignment",braceProbe()},
        {"gpu_or_d3d11_exercised",false},{"numerical_claim",false}};
    try {
        std::map<std::string,std::shared_ptr<MemoryState>> dumps;
        const auto plan=explicitPlan();uint64_t planned=0;
        // Complete: every stage and both decode steps; failed: an exclusive-create failure two records before the end;
        // early_stop: the default plan with only decode step 0 reached, so step-1 records are missing.
        dumps["complete"]=std::make_shared<MemoryState>();
        const auto complete=record(FakeScenario{},&plan,dumps["complete"],&planned);
        EXPECT(complete.at("dump_complete")==true&&complete.at("written_records")==planned&&complete.at("run_status")=="completed");
        dumps["failed"]=std::make_shared<MemoryState>();dumps["failed"]->failCreateAt=int(planned)-1;
        const auto failed=record(FakeScenario{},&plan,dumps["failed"]);
        EXPECT(failed.at("dump_complete")==false&&failed.at("run_status")=="failed"&&failed.at("written_records")==planned-2);
        FakeScenario stopped;stopped.decodeTokens={5};stopped.decodePositions.resize(1);
        dumps["early_stop"]=std::make_shared<MemoryState>();
        const auto early=record(stopped,nullptr,dumps["early_stop"]);
        EXPECT(early.at("dump_complete")==false&&early.at("run_status")=="completed"&&early.at("missing_records").get<uint64_t>()>0);

        size_t checkedLines=0;
        for(const auto& [name,state]:dumps) {
            const auto& progress=state->files.at("progress.jsonl");size_t lines=0;
            const auto errors=schemaErrors(progress,&lines);checkedLines+=lines;
            for(size_t i=0;i<errors.size()&&i<20;i++)std::cerr<<name<<": "<<errors[i]<<"\n";
            EXPECT(errors.empty());
            const auto digest=dx::sha256(progress.data(),progress.size());
            uint64_t records=0;std::map<std::string,uint64_t> stages;
            for(size_t start=0,end=progress.find('\n');end!=std::string::npos;start=end+1,end=progress.find('\n',start)) {
                const auto j=Json::parse(progress.substr(start,end-start));
                if(j.at("event")=="record") { records++;stages[j.at("stage").get<std::string>()+"/"+j.at("phase").get<std::string>()]++; }
            }
            EXPECT(state->files.size()==records+1); // The failed exclusive create writes no payload file.
            const bool matches=expectedProgress.at(name)==digest;EXPECT(matches);
            summary["dumps"][name]={{"lines",lines},{"records",records},{"progress_sha256",digest},{"expected_progress_sha256",expectedProgress.at(name)},
                {"progress_matches",matches},{"schema_errors",errors.size()},{"stages",stages}};
        }
        // Every coordinate-producing stage appears in the complete dump, in each phase it can occur in.
        const auto& stages=summary["dumps"]["complete"]["stages"];
        for(const char* s:{"vision.patch_embedding/vision","vision.position_added/vision","vision.block_output/vision","vision.merger_output/vision",
                           "text.merged_embedding/merge","text.layer_output/prefill","text.layer_output/decode","text.final_norm/prefill","text.final_norm/decode",
                           "text.logits/prefill","text.logits/decode","text.gdn_conv_state/prefill","text.gdn_conv_state/decode",
                           "text.gdn_recurrent_state/prefill","text.gdn_recurrent_state/decode"})
            if(!stages.contains(s)) { std::cerr<<"complete dump lacks stage "<<s<<"\n";failures++; }
        summary["schema_lines_checked"]=checkedLines;
        summary["schema_negative_controls_refused"]=negativeControls(dumps["complete"]->files.at("progress.jsonl"));
        if(argc==2) {
            const std::filesystem::path root(argv[1]);
            if(!std::filesystem::create_directory(root))throw std::runtime_error("Fresh output directory required");
            for(const auto& [name,state]:dumps)save(*state,root/name);
        }
        summary["failures"]=failures;
        if(argc==2) {
            std::ofstream out(std::filesystem::path(argv[1])/"summary.json",std::ios::binary);out<<summary.dump()<<'\n';out.close();
            if(!out)throw std::runtime_error("Writing summary.json failed");
        }
    } catch(const std::exception& e) { std::cerr<<"unexpected exception: "<<e.what()<<'\n';return 1; }
    std::cout<<summary.dump()<<'\n';
    if(failures)std::cerr<<failures<<" serialization expectation(s) failed\n";else std::cerr<<"diagnostic serialization tests passed\n";
    return failures?1:0;
}
