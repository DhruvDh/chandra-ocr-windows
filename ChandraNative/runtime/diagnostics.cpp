// New code, MPL-2.0. Portable diagnostic plan, SHA-256 and recorder; no D3D11 or Win32 dependency.
#include "diagnostics.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace chandra::dc::diagnostics {
namespace {
void require(bool ok,const std::string& reason) { if(!ok)throw std::invalid_argument(reason); }
void check(bool ok,const std::string& reason) { if(!ok)throw std::runtime_error(reason); }
constexpr uint32_t K[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
uint32_t rotate(uint32_t x,int n) { return (x>>n)|(x<<(32-n)); }

// Stage names are the dump contract shared with scripts/native/compare_diagnostics.py.
const char* const patchStage="vision.patch_embedding";
const char* const positionStage="vision.position_added";
const char* const blockStage="vision.block_output";
const char* const mergerStage="vision.merger_output";
const char* const embeddingStage="text.merged_embedding";
const char* const layerStage="text.layer_output";
const char* const normStage="text.final_norm";
const char* const logitsStage="text.logits";
const char* const convStage="text.gdn_conv_state";
const char* const recurrentStage="text.gdn_recurrent_state";
constexpr uint32_t convChannels=8192, convTaps=4, recurrentHeads=32, recurrentHead=128*128, chunk=64;

std::string key(const std::string& stage,int64_t index,const std::string& phase,int64_t step,int64_t row,int64_t cache) {
    auto part=[](int64_t v) { return v<0?std::string("-"):std::to_string(v); };
    return stage+"/i"+part(index)+"/"+phase+"/s"+part(step)+"/r"+part(row)+"/c"+part(cache);
}
bool contains(const std::vector<uint32_t>& values,uint64_t v) { return std::binary_search(values.begin(),values.end(),v); }
std::vector<uint32_t> range(uint32_t n) { std::vector<uint32_t> v(n);for(uint32_t i=0;i<n;i++)v[i]=i;return v; }
std::vector<uint32_t> unique(std::vector<uint32_t> v) { std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());return v; }
uint64_t natural(const Json& j,const std::string& what) {
    require(j.is_number_unsigned()||(j.is_number_integer()&&j.get<int64_t>()>=0),what+" must be a nonnegative integer");
    return j.get<uint64_t>();
}
bool flag(const Json& object,const char* name,bool fallback,const std::string& where) {
    auto it=object.find(name);if(it==object.end())return fallback;
    require(it->is_boolean(),where+"."+name+" must be a boolean");return it->get<bool>();
}
const Json& section(const Json& plan,const char* name,std::initializer_list<const char*> keys) {
    static const Json empty=Json::object();
    auto it=plan.find(name);if(it==plan.end())return empty;
    require(it->is_object(),std::string("Diagnostic plan section must be an object: ")+name);
    for(auto k=it->begin();k!=it->end();++k)
        require(std::find_if(keys.begin(),keys.end(),[&](const char* known) { return k.key()==known; })!=keys.end(),
                std::string("Unknown diagnostic plan key: ")+name+"."+k.key());
    return *it;
}
// Explicit unique indices below extent; "all" only where allowed. Order is normalized, never truncated.
std::vector<uint32_t> indices(const Json& object,const char* name,uint32_t extent,std::vector<uint32_t> fallback,
                              uint32_t maximum,bool allowAll,const std::string& where) {
    auto it=object.find(name);if(it==object.end())return fallback;
    const std::string what=where+"."+name;
    if(allowAll&&it->is_string()) { require(*it=="all",what+" accepts only \"all\" as a string");return range(extent); }
    require(it->is_array()&&it->size()<=maximum,what+" must be an array of at most "+std::to_string(maximum)+" indices");
    std::vector<uint32_t> result;
    for(const auto& v:*it) { auto n=natural(v,what);require(n<extent,what+" index "+std::to_string(n)+" is outside "+std::to_string(extent));result.push_back(uint32_t(n)); }
    auto sorted=unique(result);require(sorted.size()==result.size(),what+" contains a duplicate index");return sorted;
}
struct Planned { std::string key,group; uint64_t bytes=0,groupBytes=0; };
std::vector<Planned> enumerate(const Plan& p,const Geometry& g) {
    std::vector<Planned> out;const uint32_t prompt=uint32_t(g.ids.size());
    auto add=[&](std::string k,uint64_t bytes,std::string group,uint64_t groupBytes) { out.push_back({std::move(k),std::move(group),bytes,groupBytes}); };
    auto vision=[&](const char* stage,int64_t index) {
        for(auto r:p.visionRows)add(key(stage,index,"vision",-1,r,-1),visionWidth*4ull,key(stage,index,"vision",-1,-1,-1),uint64_t(g.patchRows)*visionWidth*4);
    };
    if(p.patchEmbedding)vision(patchStage,-1);
    if(p.positionAdded)vision(positionStage,-1);
    for(auto b:p.visionBlocks)vision(blockStage,b);
    for(auto m:p.mergerRows)add(key(mergerStage,-1,"vision",-1,m,-1),textWidth*4ull,mergerStage,uint64_t(g.mergedRows)*textWidth*4);
    for(auto r:p.embeddingRows)add(key(embeddingStage,-1,"merge",-1,r,-1),textWidth*4ull,embeddingStage,uint64_t(prompt)*textWidth*4);
    auto prefill=[&](const char* stage,int64_t index) {
        for(auto r:p.prefillRows) {
            uint32_t c=r/chunk,rows=std::min(chunk,prompt-c*chunk);
            add(key(stage,index,"prefill",-1,r,-1),textWidth*4ull,key(stage,index,"prefill",-1,-1,c),uint64_t(rows)*textWidth*4);
        }
    };
    for(auto l:p.prefillLayers)prefill(layerStage,l);
    if(p.prefillFinalNorm)prefill(normStage,-1);
    if(p.prefillLogits)add(key(logitsStage,-1,"prefill",-1,0,-1),vocabulary*4ull,key(logitsStage,-1,"prefill",-1,-1,-1),vocabulary*4ull);
    for(auto d:p.decodeSteps) {
        for(auto l:p.decodeLayers)add(key(layerStage,l,"decode",d,0,-1),textWidth*4ull,key(layerStage,l,"decode",d,-1,-1),textWidth*4ull);
        if(p.decodeFinalNorm)add(key(normStage,-1,"decode",d,0,-1),textWidth*4ull,key(normStage,-1,"decode",d,-1,-1),textWidth*4ull);
        if(p.decodeLogits)add(key(logitsStage,-1,"decode",d,0,-1),vocabulary*4ull,key(logitsStage,-1,"decode",d,-1,-1),vocabulary*4ull);
    }
    for(const auto& s:p.states)for(auto length:s.cacheLengths) {
        const bool decode=length>prompt;const int64_t step=decode?int64_t(length)-prompt-1:-1;const char* phase=decode?"decode":"prefill";
        if(s.conv)add(key(convStage,s.layer,phase,step,-1,length),uint64_t(convChannels)*convTaps*4,key(convStage,s.layer,phase,step,-1,length),uint64_t(convChannels)*convTaps*4);
        for(auto h:s.heads)add(key(recurrentStage,s.layer,phase,step,h,length),recurrentHead*4ull,key(recurrentStage,s.layer,phase,step,-1,length),uint64_t(recurrentHeads)*recurrentHead*4);
    }
    return out;
}
std::vector<uint32_t> imageRows(const Geometry& g) {
    std::vector<uint32_t> rows;for(uint32_t i=0;i<g.visionRowMap.size();i++)if(g.visionRowMap[i]!=UINT32_MAX)rows.push_back(i);return rows;
}
Json list(const std::vector<uint32_t>& v) { return Json(v); }
// One-element schema arrays are built explicitly. Never brace-wrap a single Json value ({value}): GCC and
// Clang build a one-element array, but MSVC copy-initializes the value itself (CWG 1467), so Build09
// wrote bare coordinate objects. Brace lists of non-Json scalars or of key/value pairs are unaffected.
Json one(Json value) { Json array=Json::array();array.push_back(std::move(value));return array; }
}

Sha256::Sha256():state{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19},block{} {}
void Sha256::compress(const uint8_t* p) {
    uint32_t w[64];
    for(int i=0;i<16;i++)w[i]=uint32_t(p[4*i])<<24|uint32_t(p[4*i+1])<<16|uint32_t(p[4*i+2])<<8|uint32_t(p[4*i+3]);
    for(int i=16;i<64;i++) {
        uint32_t s0=rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3),s1=rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++) {
        uint32_t t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+K[i]+w[i];
        uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}
void Sha256::update(const void* data,size_t n) {
    auto* p=static_cast<const uint8_t*>(data);bytes+=n;
    while(n) {
        size_t take=std::min(n,64-used);std::memcpy(block+used,p,take);used+=take;p+=take;n-=take;
        if(used==64) { compress(block);used=0; }
    }
}
std::string Sha256::finish() const {
    Sha256 copy=*this;const uint64_t bits=bytes*8;const uint8_t one=0x80,zero=0;
    copy.update(&one,1);while(copy.used!=56)copy.update(&zero,1);
    uint8_t length[8];for(int i=0;i<8;i++)length[i]=uint8_t(bits>>(56-8*i));copy.update(length,8);
    static const char* hex="0123456789abcdef";std::string text;
    for(uint32_t word:copy.state)for(int shift=28;shift>=0;shift-=4)text.push_back(hex[(word>>shift)&15]);
    return text;
}
std::string sha256(const void* data,size_t n) { Sha256 s;s.update(data,n);return s.finish(); }

bool safeName(const std::string& name) {
    if(name.empty()||name.size()>160||name.front()=='.'||name.find("..")!=std::string::npos)return false;
    return std::all_of(name.begin(),name.end(),[](char c) { return (c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='.'||c=='_'||c=='-'; });
}

Plan resolve(const Json* requested,const Geometry& g) {
    const uint32_t prompt=uint32_t(g.ids.size());
    require(prompt&&g.visionRowMap.size()==prompt&&g.positions.temporal.size()==prompt&&g.positions.height.size()==prompt&&
            g.positions.width.size()==prompt&&g.patchRows&&g.mergedRows&&g.patchRows==4*g.mergedRows&&g.generationLimit&&
            g.generationLimit<=12384&&!g.grids.empty(),"Diagnostic geometry requires authenticated prompt, vision and generation facts");
    const auto images=imageRows(g);require(images.size()==g.mergedRows,"Diagnostic geometry image rows differ from merged rows");
    static const Json defaults={{"schema",planSchema}};
    const Json& j=requested?*requested:defaults;
    require(j.is_object()&&j.contains("schema")&&j.at("schema").is_string()&&j.at("schema")==planSchema,std::string("Diagnostic plan schema must be ")+planSchema);
    for(auto it=j.begin();it!=j.end();++it)
        require(it.key()=="schema"||it.key()=="byte_limit"||it.key()=="vision"||it.key()=="merger"||it.key()=="embedding"||
                it.key()=="prefill"||it.key()=="decode"||it.key()=="gdn_state","Unknown diagnostic plan key: "+it.key());
    Plan p;
    if(j.contains("byte_limit")) {
        p.byteLimit=natural(j.at("byte_limit"),"byte_limit");
        require(p.byteLimit&&p.byteLimit<=maximumByteLimit,"byte_limit must be positive and at most "+std::to_string(maximumByteLimit));
    }
    const auto& vision=section(j,"vision",{"rows","blocks","patch_embedding","position_added"});
    p.visionRows=indices(vision,"rows",g.patchRows,unique({0,g.patchRows-1}),maximumSelectedRows,false,"vision");
    p.visionBlocks=indices(vision,"blocks",24,range(24),24,true,"vision");
    p.patchEmbedding=flag(vision,"patch_embedding",true,"vision");p.positionAdded=flag(vision,"position_added",true,"vision");
    p.mergerRows=indices(section(j,"merger",{"rows"}),"rows",g.mergedRows,unique({0,g.mergedRows-1}),maximumSelectedRows,false,"merger");
    const auto& prefill=section(j,"prefill",{"rows","layers","final_norm","logits"});
    p.prefillRows=indices(prefill,"rows",prompt,unique({0,images.front(),images.back(),prompt-1}),maximumSelectedRows,false,"prefill");
    p.prefillLayers=indices(prefill,"layers",32,range(32),32,true,"prefill");
    p.prefillFinalNorm=flag(prefill,"final_norm",true,"prefill");p.prefillLogits=flag(prefill,"logits",true,"prefill");
    p.embeddingRows=indices(section(j,"embedding",{"rows"}),"rows",prompt,p.prefillRows,maximumSelectedRows,false,"embedding");
    // Decode step d consumes generated token d and produces logits for generated index d+1; it runs only if d+1 < limit.
    const uint32_t decodeExtent=g.generationLimit-1;
    const auto& decode=section(j,"decode",{"steps","layers","final_norm","logits"});
    p.decodeSteps=indices(decode,"steps",decodeExtent,range(std::min(defaultDecodeSteps,decodeExtent)),maximumDecodeSteps,false,"decode");
    p.decodeLayers=indices(decode,"layers",32,range(32),32,true,"decode");
    p.decodeFinalNorm=flag(decode,"final_norm",true,"decode");p.decodeLogits=flag(decode,"logits",true,"decode");
    if(j.contains("gdn_state")) {
        const auto& states=j.at("gdn_state");
        require(states.is_array()&&states.size()<=maximumStateSelections,"gdn_state must be an array of at most "+std::to_string(maximumStateSelections)+" explicit selections");
        std::set<uint32_t> layers;
        for(const auto& entry:states) {
            require(entry.is_object(),"gdn_state entries must be objects");
            for(auto it=entry.begin();it!=entry.end();++it)
                require(it.key()=="layer"||it.key()=="conv"||it.key()=="recurrent_heads"||it.key()=="cache_lengths","Unknown diagnostic plan key: gdn_state."+it.key());
            require(entry.contains("layer")&&entry.contains("cache_lengths"),"gdn_state entries require explicit layer and cache_lengths");
            StateSelection s;const auto layer=natural(entry.at("layer"),"gdn_state.layer");
            require(layer<32&&layer%4!=3,"gdn_state.layer must name one of the 24 Gated Delta Net layers");
            s.layer=uint32_t(layer);require(layers.insert(s.layer).second,"gdn_state names a layer more than once");
            s.conv=flag(entry,"conv",false,"gdn_state");
            s.heads=indices(entry,"recurrent_heads",recurrentHeads,{},recurrentHeads,true,"gdn_state");
            require(s.conv||!s.heads.empty(),"gdn_state selection must request conv state or at least one recurrent head");
            const auto& lengths=entry.at("cache_lengths");
            require(lengths.is_array()&&!lengths.empty()&&lengths.size()<=maximumStateLengths,"gdn_state.cache_lengths must be a nonempty array of at most "+std::to_string(maximumStateLengths));
            for(const auto& v:lengths) {
                auto length=natural(v,"gdn_state.cache_lengths");
                // Observable state exists only after a completed 64-row prefill tile or one completed decode step.
                const bool prefillEnd=length&&length<=prompt&&(length%chunk==0||length==prompt);
                const bool decodeEnd=length>prompt&&length-prompt-1<decodeExtent;
                require(prefillEnd||decodeEnd,"gdn_state cache length "+std::to_string(length)+" is not a completed prefill tile or reachable decode step");
                s.cacheLengths.push_back(uint32_t(length));
            }
            auto sorted=unique(s.cacheLengths);require(sorted.size()==s.cacheLengths.size(),"gdn_state.cache_lengths contains a duplicate");
            s.cacheLengths=sorted;p.states.push_back(std::move(s));
        }
        std::sort(p.states.begin(),p.states.end(),[](const StateSelection& a,const StateSelection& b) { return a.layer<b.layer; });
    }
    std::map<std::string,uint64_t> groups;
    for(const auto& r:enumerate(p,g)) {
        require(p.keys.insert(r.key).second,"Diagnostic plan produced a duplicate record identity: "+r.key);
        p.payloadBytes+=r.bytes;groups.emplace(r.group,r.groupBytes);
    }
    p.records=p.keys.size();
    for(const auto& [name,bytes]:groups) { p.readbacks++;p.readbackBytes+=bytes;p.largestReadbackBytes=std::max(p.largestReadbackBytes,bytes); }
    require(p.records,"Diagnostic plan selects no records");
    require(p.records<=maximumRecords,"Diagnostic plan requests "+std::to_string(p.records)+" records; limit "+std::to_string(maximumRecords));
    require(p.payloadBytes<=p.byteLimit,"Diagnostic plan requests "+std::to_string(p.payloadBytes)+" payload bytes, above byte_limit "+std::to_string(p.byteLimit)+"; refused before execution");
    require(p.readbackBytes<=maximumReadbackBytes&&p.largestReadbackBytes<=maximumSingleReadbackBytes,"Diagnostic readback forecast exceeds bounded limits");
    Json states=Json::array();
    for(const auto& s:p.states)states.push_back({{"layer",s.layer},{"conv",s.conv},{"recurrent_heads",list(s.heads)},{"cache_lengths",list(s.cacheLengths)}});
    p.resolved={{"schema",planSchema},{"byte_limit",p.byteLimit},
        {"vision",{{"rows",list(p.visionRows)},{"blocks",list(p.visionBlocks)},{"patch_embedding",p.patchEmbedding},{"position_added",p.positionAdded}}},
        {"merger",{{"rows",list(p.mergerRows)}}},{"embedding",{{"rows",list(p.embeddingRows)}}},
        {"prefill",{{"rows",list(p.prefillRows)},{"layers",list(p.prefillLayers)},{"final_norm",p.prefillFinalNorm},{"logits",p.prefillLogits}}},
        {"decode",{{"steps",list(p.decodeSteps)},{"layers",list(p.decodeLayers)},{"final_norm",p.decodeFinalNorm},{"logits",p.decodeLogits}}},
        {"gdn_state",states},{"explicit_plan",requested!=nullptr},
        {"forecast",{{"records",p.records},{"payload_bytes",p.payloadBytes},{"readbacks",p.readbacks},{"readback_bytes",p.readbackBytes},
            {"largest_readback_bytes",p.largestReadbackBytes},{"keys",p.keys},
            {"scope","Exact planned records and full-buffer staging readbacks; unreached decode steps are reported missing, never substituted"}}}};
    const auto canonical=p.resolved.dump();p.sha256=sha256(canonical.data(),canonical.size());
    return p;
}

// row is the single selected logical row, or -1 for a complete "all" selection; coordinate is exactly
// one object. put() alone shapes both into the record's selected_rows and one-element coordinates array.
struct Recorder::Spec {
    std::string stage,phase; int64_t index=-1,step=-1,row=-1,cache=-1;
    Json logicalShape,payloadShape,coordinate,cacheState,decode; bool complete=false,state=false; int64_t produces=-1;
    uint64_t prefix=0; // Consumed request rows that condition a text value; 0 for vision and merged embedding.
};
Recorder::Recorder(Plan p,Geometry g,std::unique_ptr<Directory> d,Reader r,Json c,Json producerIdentity)
    :plan(std::move(p)),geometry(std::move(g)),directory(std::move(d)),reader(std::move(r)),commitments(std::move(c)),producer(std::move(producerIdentity)) {
    const uint16_t endian=1;check(*reinterpret_cast<const uint8_t*>(&endian)==1,"Diagnostic FP32 payloads require little-endian host storage");
    check(directory&&reader&&plan.records&&!plan.sha256.empty()&&commitments.is_object()&&producer.is_object()&&
          commitments.contains("input_manifest_sha256")&&commitments.at("input_manifest_sha256").is_string(),
          "Diagnostic recorder requires a resolved plan, directory, reader and input-manifest commitment");
    commitments["plan_sha256"]=plan.sha256;
    const auto header=std::string(prefixSchema)+"\ninput_manifest_sha256 "+commitments.at("input_manifest_sha256").get<std::string>()+
        "\nprompt_rows "+std::to_string(geometry.ids.size())+"\n";
    prefixHash.update(header.data(),header.size());prefixDigests.push_back(prefixHash.finish());
    progress=directory->createNew("progress.jsonl");
    line({{"event","plan"},{"schema",progressSchema},{"plan",plan.resolved},{"plan_sha256",plan.sha256},{"commitments",commitments},
          {"producer",producer},{"qualified",false},{"conditioning",{{"schema",prefixSchema},{"prompt_rows",geometry.ids.size()}}},
          {"scope","Bounded raw evidence of selected completed values; incomplete, failed or unfinished dumps are unqualified and no dump is OCR or numerical acceptance"}});
}
Recorder::~Recorder()=default;
void Recorder::line(const Json& value) {
    check(progress!=nullptr,"Diagnostic progress file is unavailable");
    const auto text=value.dump()+"\n";
    try { progress->write(text.data(),text.size());progress->flush(); }
    catch(...) { broken=true;throw; }
    progressHash.update(text.data(),text.size());
}
void Recorder::authenticatedModel(const Json& provenance,const Json& device) {
    check(!finished&&!modelCommitted,"Model commitment must be recorded once before diagnostic records");
    check(provenance.is_object()&&provenance.contains("revision")&&commitments.contains("model_revision")&&provenance.at("revision")==commitments.at("model_revision")&&
          provenance.at("model_sha256").is_string()&&provenance.at("config_sha256").is_string(),
          "Authenticated model provenance differs from the diagnostic commitment");
    commitments["model_sha256"]=provenance.at("model_sha256");commitments["config_sha256"]=provenance.at("config_sha256");
    commitments["model_bytes"]=provenance.at("model_bytes");
    line({{"event","model_authenticated"},{"model",{{"revision",commitments["model_revision"]},{"model_sha256",commitments["model_sha256"]},
          {"config_sha256",commitments["config_sha256"]},{"model_bytes",commitments["model_bytes"]}}},{"device",device}});
    modelCommitted=true;
}
void Recorder::boundary(const std::string& name) {
    if(finished||broken)return; // After a failed write only the terminal line is attempted.
    line({{"event","boundary"},{"name",name},{"completed_records",written.size()},{"payload_bytes",payloadBytes}});
}
std::vector<float> Recorder::read(const Buffer& b,uint64_t elements) {
    check(b.storage&&!b.packedBF16&&b.logicalElements==elements&&b.words==elements,"Observed buffer differs from its declared diagnostic shape");
    check(readbacks<plan.readbacks&&readbackBytes+elements*4<=plan.readbackBytes,"Diagnostic readback exceeds its forecast");
    auto values=reader(b);check(values.size()==elements,"Diagnostic readback returned an incomplete buffer");
    readbacks++;readbackBytes+=elements*4;return values;
}
void Recorder::put(const Spec& s,const float* data,size_t count) {
    check(!finished&&!broken,"Diagnostic recorder is finished or failed");
    check(modelCommitted,"Diagnostic records require the authenticated model commitment first");
    const auto identity=key(s.stage,s.index,s.phase,s.step,s.row,s.cache);
    check(plan.keys.count(identity)!=0,"Unplanned diagnostic record refused: "+identity);
    check(!written.count(identity),"Duplicate diagnostic record refused: "+identity);
    const uint64_t bytes=uint64_t(count)*4;
    check(count&&payloadBytes+bytes<=plan.byteLimit&&written.size()<plan.records,"Diagnostic byte/record budget would be exceeded");
    uint64_t elements=1;for(const auto& d:s.payloadShape)elements*=d.get<uint64_t>();
    check(elements==count,"Diagnostic payload shape differs from its selected rows");
    check(s.coordinate.is_object(),"Diagnostic record coordinate must be one object: "+identity);
    check(s.prefix<=consumed.size(),"Diagnostic record prefix exceeds the consumed request rows");
    char prefix[16];std::snprintf(prefix,sizeof(prefix),"%05llu",static_cast<unsigned long long>(sequence));
    std::string name=std::string(prefix)+"."+s.stage+(s.index>=0?".i"+std::to_string(s.index):"")+"."+s.phase+
        (s.step>=0?".s"+std::to_string(s.step):"")+(s.row>=0?".r"+std::to_string(s.row):".all")+(s.cache>=0?".c"+std::to_string(s.cache):"")+".f32";
    check(safeName(name),"Unsafe generated diagnostic filename");
    uint64_t nonfinite=0,nonBF16=0;
    for(size_t i=0;i<count;i++) {
        uint32_t bits;std::memcpy(&bits,data+i,4);
        if(!std::isfinite(data[i]))nonfinite++;else if(bits&0xffffu)nonBF16++;
    }
    if(s.prefix&&conditioned<consumed.size())condition(); // The record's whole conditioning precedes it in progress.jsonl.
    started=identity;
    line({{"event","record_started"},{"sequence",sequence},{"file",name},{"key",identity}});
    try {
        auto file=directory->createNew(name);file->write(data,size_t(bytes));file->flush();
    } catch(...) { broken=true;throw; }
    Json record={{"event","record"},{"schema",recordSchema},{"sequence",sequence},{"file",name},{"key",identity},{"stage",s.stage},
        {"component",s.stage.substr(0,s.stage.find('.'))},{"phase",s.phase},
        {"layer",s.stage.rfind("text.",0)==0&&s.index>=0?Json(s.index):Json(nullptr)},
        {"layer_kind",s.stage.rfind("text.",0)==0&&s.index>=0?Json(s.index%4==3?"full_attention":"gated_delta_net"):Json(nullptr)},
        {"vision_block",s.stage==blockStage?Json(s.index):Json(nullptr)},
        {"decode_step",s.step>=0?Json(s.step):Json(nullptr)},{"cache_length",s.cache>=0?Json(s.cache):Json(nullptr)},
        {"logical_shape",s.logicalShape},{"selected_axis",0},{"selected_rows",s.row>=0?one(s.row):Json("all")},{"payload_shape",s.payloadShape},
        {"complete_tensor",s.complete},{"storage_dtype","float32"},{"byte_order","little"},
        {"bf16_rounding",s.state?"none_fp32_state":"bf16_round_to_nearest_even_at_graph_boundary"},
        {"payload_bytes",bytes},{"payload_sha256",sha256(data,size_t(bytes))},
        {"observed",{{"nonfinite",nonfinite},{"finite_non_bf16_representable",nonBF16}}},
        {"coordinates",one(s.coordinate)},{"cache",s.cacheState},{"decode",s.decode},
        {"conditioning",s.prefix?Json{{"prefix_rows",s.prefix},{"prefix_sha256",prefixDigests[s.prefix]}}:Json(nullptr)},
        {"produces_generated_index",s.produces>=0?Json(s.produces):Json(nullptr)},
        {"commitments",commitments},{"producer",producer},{"qualified",false}};
    line(record);
    written.insert(identity);started.clear();payloadBytes+=bytes;sequence++;
}
void Recorder::vision(const std::string& name,const Buffer& b,uint32_t rows,uint32_t width) {
    check(!finished&&!broken,"Diagnostic recorder is finished or failed");
    std::string stage;int64_t index=-1;bool enabled=false;bool merger=false;
    if(name=="patch_embedding") { stage=patchStage;enabled=plan.patchEmbedding; }
    else if(name=="position_added") { stage=positionStage;enabled=plan.positionAdded; }
    else if(name=="pooler_output") { stage=mergerStage;merger=true;enabled=!plan.mergerRows.empty(); }
    else if(name.rfind("blocks.",0)==0&&name.size()>14&&name.compare(name.size()-7,7,".output")==0) {
        const auto digits=name.substr(7,name.size()-14);
        if(digits.empty()||digits.size()>2||!std::all_of(digits.begin(),digits.end(),[](char c) { return c>='0'&&c<='9'; }))return;
        stage=blockStage;index=std::stoi(digits);enabled=contains(plan.visionBlocks,uint64_t(index));
    } else return; // Other vision observations are intermediate and never selected by this plan.
    const auto& selected=merger?plan.mergerRows:plan.visionRows;
    if(!enabled||selected.empty())return;
    const uint32_t expectedRows=merger?geometry.mergedRows:geometry.patchRows,expectedWidth=merger?textWidth:visionWidth;
    check(rows==expectedRows&&width==expectedWidth,"Vision observation shape differs from authenticated geometry: "+name);
    const auto values=read(b,uint64_t(rows)*width);
    std::vector<uint32_t> image;if(merger)image=imageRows(geometry);
    for(auto r:selected) {
        Spec s;s.stage=stage;s.index=index;s.phase="vision";s.row=r;s.logicalShape={rows,width};s.payloadShape={1,width};s.complete=rows==1;
        if(merger)s.coordinate={{"merged_row",r},{"prompt_row",image.at(r)}};
        else {
            // Same block-major frame geometry as vision_model.cpp; metadata only.
            uint32_t first=0,frame=0;Json c;
            for(const auto& grid:geometry.grids)for(uint32_t t=0;t<grid.temporal;t++,frame++) {
                const uint32_t size=grid.height*grid.width;
                if(r>=first&&r<first+size) {
                    const uint32_t within=r-first,blocksW=grid.width/2;
                    c={{"patch_row",r},{"frame",frame},{"grid_row",(within/(4*blocksW))*2+(within/2)%2},{"grid_col",((within/4)%blocksW)*2+within%2},{"merged_row",r/4}};
                }
                first+=size;
            }
            check(!c.is_null(),"Patch row outside authenticated grid");s.coordinate=std::move(c);
        }
        put(s,values.data()+size_t(r)*width,width);
    }
}
void Recorder::merged(const Buffer& b,uint32_t rows,uint32_t width) {
    check(!finished&&!broken,"Diagnostic recorder is finished or failed");
    if(plan.embeddingRows.empty())return;
    check(rows==geometry.ids.size()&&width==textWidth,"Merged embedding shape differs from the authenticated prompt");
    const auto values=read(b,uint64_t(rows)*width);const auto& pos=geometry.positions;
    for(auto r:plan.embeddingRows) {
        const bool image=geometry.visionRowMap[r]!=UINT32_MAX;
        Spec s;s.stage=embeddingStage;s.phase="merge";s.row=r;s.logicalShape={rows,width};s.payloadShape={1,width};s.complete=rows==1;
        s.coordinate={{"absolute_row",r},{"token_id",geometry.ids[r]},{"source",image?"vision":"text"},
            {"merged_row",image?Json(geometry.visionRowMap[r]):Json(nullptr)},{"position",{pos.temporal[r],pos.height[r],pos.width[r]}}};
        put(s,values.data()+size_t(r)*width,width);
    }
}
void Recorder::text(const TextObservation& o) {
    check(!finished&&!broken,"Diagnostic recorder is finished or failed");
    const uint32_t prompt=uint32_t(geometry.ids.size());
    const bool prefill=!o.tokenIds&&o.tokensBefore==0&&o.generatedBefore==0&&o.callTokens==prompt;
    const bool decode=o.tokenIds&&o.tokenIds->size()==1&&o.callTokens==1&&o.tokensBefore==uint64_t(prompt)+o.generatedBefore;
    check(prefill!=decode,"Text observation is neither the planned prompt prefill nor one cached decode step");
    check(o.count&&uint64_t(o.first)+o.count<=o.callTokens&&o.positions.temporal.size()==o.callTokens&&o.positions.height.size()==o.callTokens&&
          o.positions.width.size()==o.callTokens&&uint64_t(o.cacheLength)==uint64_t(o.tokensBefore)+o.first+o.count,
          "Text observation cache/row coordinate is inconsistent");
    enter(o,decode);
    const int64_t step=decode?int64_t(o.generatedBefore):-1;const char* phase=decode?"decode":"prefill";
    Json cache={{"cache_length_after",o.cacheLength},{"cache_length_source",o.stage==TextStage::FinalNorm?"last_layer":o.stage==TextStage::Logits?"request":"layer"},
        {"request_tokens_before_call",o.tokensBefore},{"generated_before_call",o.generatedBefore},{"call_tokens",o.callTokens},
        {"call_first_row",o.first},{"call_row_count",o.count}};
    Json decodeJson=decode?Json{{"step",step},{"consumed_generated_index",step},{"consumed_token_id",o.tokenIds->front()},{"logits_produce_generated_index",step+1}}:Json(nullptr);
    auto coordinate=[&](uint32_t callRow) {
        const uint32_t absolute=o.tokensBefore+callRow;
        Json c={{"absolute_row",absolute},{"call_row",callRow},{"chunk_index",callRow/chunk},{"chunk_row",callRow%chunk},
            {"position",{o.positions.temporal[callRow],o.positions.height[callRow],o.positions.width[callRow]}}};
        if(decode)c["token_id"]=(*o.tokenIds)[callRow];
        else { c["token_id"]=geometry.ids[absolute];c["source"]=geometry.visionRowMap[absolute]!=UINT32_MAX?"vision":"text"; }
        return c;
    };
    if(o.stage==TextStage::LayerOutput||o.stage==TextStage::FinalNorm) {
        const bool layer=o.stage==TextStage::LayerOutput;
        check(o.width==textWidth&&o.rows==o.count&&(layer?o.layer<32:o.layer==UINT32_MAX),"Text hidden observation shape differs");
        bool enabled=layer?contains(decode?plan.decodeLayers:plan.prefillLayers,o.layer):(decode?plan.decodeFinalNorm:plan.prefillFinalNorm);
        if(decode)enabled=enabled&&contains(plan.decodeSteps,uint64_t(step));
        std::vector<uint32_t> local;
        if(enabled) {
            if(decode)local.push_back(0);
            else for(auto r:plan.prefillRows)if(r>=o.first&&r<o.first+o.count)local.push_back(r-o.first);
        }
        if(local.empty())return;
        const auto values=read(o.value,uint64_t(o.rows)*o.width);
        for(auto i:local) {
            const uint32_t callRow=o.first+i;
            Spec s;s.stage=layer?layerStage:normStage;s.index=layer?int64_t(o.layer):-1;s.phase=phase;s.step=step;s.row=decode?0:callRow;
            s.logicalShape={o.callTokens,textWidth};s.payloadShape={1,textWidth};s.complete=o.callTokens==1;
            s.coordinate=coordinate(callRow);s.cacheState=cache;s.decode=decodeJson;s.prefix=uint64_t(o.tokensBefore)+callRow+1;
            put(s,values.data()+size_t(i)*o.width,o.width);
        }
    } else if(o.stage==TextStage::Logits) {
        check(o.width==vocabulary&&o.rows==1&&o.count==1&&o.first+1==o.callTokens&&o.layer==UINT32_MAX,"Only last-row complete vocabulary logits are diagnosable");
        if(!(decode?plan.decodeLogits&&contains(plan.decodeSteps,uint64_t(step)):plan.prefillLogits))return;
        const auto values=read(o.value,vocabulary);
        Spec s;s.stage=logitsStage;s.phase=phase;s.step=step;s.row=0;s.logicalShape={1,vocabulary};s.payloadShape={1,vocabulary};
        s.complete=true;s.coordinate=coordinate(o.first);s.cacheState=cache;s.decode=decodeJson;s.produces=decode?step+1:0;
        s.prefix=uint64_t(o.tokensBefore)+o.first+1;
        put(s,values.data(),vocabulary);
    } else {
        const bool conv=o.stage==TextStage::ConvState;
        check(o.layer<32&&o.layer%4!=3&&(conv?o.rows==convChannels&&o.width==convTaps:o.rows==recurrentHeads&&o.width==recurrentHead),"GDN state observation shape differs");
        auto selection=std::find_if(plan.states.begin(),plan.states.end(),[&](const StateSelection& s) { return s.layer==o.layer; });
        if(selection==plan.states.end()||!contains(selection->cacheLengths,o.cacheLength)||(conv?!selection->conv:selection->heads.empty()))return;
        const auto values=read(o.value,uint64_t(o.rows)*o.width);
        // Carried state is identified by its whole consumed prefix; the producing call's first row stays in cache provenance.
        const uint32_t last=o.first+o.count-1;
        auto base=[&](Spec& s) {
            s.stage=conv?convStage:recurrentStage;s.index=o.layer;s.phase=phase;s.step=step;s.cache=o.cacheLength;s.state=true;
            s.coordinate={{"last_absolute_row",o.tokensBefore+last},{"last_position",{o.positions.temporal[last],o.positions.height[last],o.positions.width[last]}}};
            s.cacheState=cache;s.decode=decodeJson;s.prefix=o.cacheLength;
        };
        if(conv) {
            Spec s;base(s);s.logicalShape={convChannels,convTaps};s.payloadShape={convChannels,convTaps};s.complete=true; // row -1: complete "all" selection.
            put(s,values.data(),values.size());
        } else for(auto h:selection->heads) {
            Spec s;base(s);s.row=h;s.logicalShape={recurrentHeads,128,128};s.payloadShape={1,128,128};
            put(s,values.data()+size_t(h)*recurrentHead,recurrentHead);
        }
    }
}
// Advances the consumed prefix once per graph call. Repeated callbacks of the current call must agree
// with its rows; a call that skips, repeats or reorders request rows cannot be bound and is refused.
void Recorder::enter(const TextObservation& o,bool decode) {
    const uint32_t prompt=uint32_t(geometry.ids.size());
    auto token=[&](uint32_t callRow) { return decode?(*o.tokenIds)[callRow]:geometry.ids[o.tokensBefore+callRow]; };
    if(call.active&&call.before==o.tokensBefore&&call.generated==o.generatedBefore&&call.tokens==o.callTokens) {
        for(uint32_t i=o.first;i<o.first+o.count;i++) {
            const auto& c=consumed[size_t(o.tokensBefore)+i];
            check(c[0]==token(i)&&c[1]==o.positions.temporal[i]&&c[2]==o.positions.height[i]&&c[3]==o.positions.width[i],
                  "Text observation disagrees with the consumed rows of its call");
        }
        return;
    }
    check(o.tokensBefore==consumed.size()&&(decode?consumed.size()==uint64_t(prompt)+o.generatedBefore:consumed.empty()),
          "Text observation skips or repeats a request call; its consumed prefix cannot be proven");
    if(!decode)check(o.positions.temporal==geometry.positions.temporal&&o.positions.height==geometry.positions.height&&
                     o.positions.width==geometry.positions.width,"Prefill positions differ from the authenticated prompt");
    for(uint32_t i=0;i<o.callTokens;i++) {
        const std::array<uint32_t,4> row{token(i),o.positions.temporal[i],o.positions.height[i],o.positions.width[i]};
        const auto text=std::to_string(consumed.size())+" "+std::to_string(row[0])+" "+std::to_string(row[1])+" "+
            std::to_string(row[2])+" "+std::to_string(row[3])+"\n";
        prefixHash.update(text.data(),text.size());consumed.push_back(row);prefixDigests.push_back(prefixHash.finish());
    }
    call={true,o.tokensBefore,o.generatedBefore,o.callTokens};
}
// Appends every consumed row not yet written, so the dump carries its complete conditioning from row 0.
void Recorder::condition() {
    Json rows=Json::array();
    for(size_t i=size_t(conditioned);i<consumed.size();i++)rows.push_back(consumed[i]);
    line({{"event","conditioning"},{"first_row",conditioned},{"rows",rows},{"consumed_rows",consumed.size()},{"prefix_sha256",prefixDigests.back()}});
    conditioned=consumed.size();
}
Json Recorder::finish(bool runCompleted,const std::string& error) {
    if(finished)return summary;
    // A completed run has no error; a failed run always names one, so the terminal line cannot contradict itself.
    const bool completed=runCompleted&&error.empty();
    const std::string reason=completed?std::string():error.empty()?std::string("run failed without an error message"):error;
    std::vector<std::string> missing;
    for(const auto& k:plan.keys)if(!written.count(k))missing.push_back(k);
    const bool complete=completed&&!broken&&missing.empty()&&started.empty();
    finished=true; // A failed terminal write is reported by the caller; it is never retried into a second terminal line.
    if(!broken&&conditioned<consumed.size())condition(); // Rows consumed after the last selected record.
    Json terminal={{"event","finished"},{"run_status",completed?"completed":"failed"},{"dump_complete",complete},
        {"planned_records",plan.records},{"written_records",written.size()},{"missing_records",missing},
        {"incomplete_record",started.empty()?Json(nullptr):Json(started)},{"payload_bytes",payloadBytes},
        {"readbacks",readbacks},{"readback_bytes",readbackBytes},{"error",reason.empty()?Json(nullptr):Json(reason)},
        {"consumed_rows",consumed.size()},{"consumed_prefix_sha256",prefixDigests.back()},{"qualified",false}};
    line(terminal);
    summary={{"directory","diagnostics"},{"progress_file","progress.jsonl"},{"progress_sha256",progressHash.finish()},
        {"plan_sha256",plan.sha256},{"run_status",terminal["run_status"]},{"dump_complete",complete},{"planned_records",plan.records},
        {"written_records",written.size()},{"missing_records",missing.size()},{"payload_bytes",payloadBytes},{"readback_bytes",readbackBytes},
        {"consumed_rows",consumed.size()},{"consumed_prefix_sha256",prefixDigests.back()},{"qualified",false},{"scope","Raw selected values only; compare with an explicit reference and caller-supplied tolerances"}};
    return summary;
}
}
