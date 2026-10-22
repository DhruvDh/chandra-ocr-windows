// New code, MPL-2.0. Small CPU-only parser tests; no Windows/GPU/device path.
#include "model_weights.cpp"
#include <iostream>
#include <fstream>
using namespace chandra::dc::weight_detail;
namespace {
unsigned passed=0;
template<class F> void rejects(F f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"negative fixture accepted");++passed;}
Inventory parse(const std::string& s,uint64_t bytes){return parseHeader(reinterpret_cast<const uint8_t*>(s.data()),s.size(),bytes);}
std::string descriptor(const std::string& name,const std::string& dtype,const Json& shape,uint64_t begin,uint64_t end){return Json{{name,{{"dtype",dtype},{"shape",shape},{"data_offsets",{begin,end}}}}}.dump();}
}
int main(int argc,char** argv){
 try {
    auto vector=parse(descriptor("norm.weight","BF16",Json::array({3}),0,6),6);const auto& v=vector.at("norm.weight");require(v.rows==1&&v.cols==3&&v.elements==3&&v.bf16,"vector flatten differs");auto tail=shardPlan(v);require(tail.size()==1&&tail[0].words==2&&tail[0].storageBytes==8&&tail[0].sourceBytes==6,"odd tail plan differs");++passed;
    auto fp32=parse(descriptor("input_layernorm.weight","F32",Json::array({3}),0,12),12);require(!fp32.at("input_layernorm.weight").bf16&&shardPlan(fp32.begin()->second)[0].words==3,"F32 norm storage differs");++passed;
    auto conv=parse(descriptor("conv.weight","BF16",Json::array({1024,3,2,16,16}),0,1024ull*1536*2),1024ull*1536*2);require(conv.at("conv.weight").rows==1024&&conv.at("conv.weight").cols==1536,"vision flatten differs");++passed;
    auto shapes=pinnedShapes();require(shapes.size()==739,"source inventory count differs");
    Json original=Json::object();uint64_t offset=0;
    for(const auto& item:shapes){uint64_t n=1;for(auto d:item.second)n=multiply(n,d);uint64_t end=add(offset,multiply(n,2));original[item.first]={{"dtype","BF16"},{"shape",item.second},{"data_offsets",{offset,end}}};offset=end;}
    auto source=parse(original.dump(),offset);validatePinned(source);++passed;
    auto full=selectGraph(source,{});require(full.fullGraph&&full.effective.size()==724&&!full.tiedAliasExpansion,"empty selection does not retain complete graph");++passed;
    auto norm=selectGraph(source,{"model.language_model.layers.0.input_layernorm.weight"});require(!norm.fullGraph&&norm.effective.size()==1&&!norm.tiedAliasExpansion&&norm.effective.count("lm_head.weight")==0,"bounded norm selection expanded unexpectedly");++passed;
    for(const std::string alias:{"lm_head.weight","model.language_model.embed_tokens.weight"}){auto tied=selectGraph(source,{alias});require(tied.effective.size()==2&&tied.effective.count("lm_head.weight")&&tied.effective.count("model.language_model.embed_tokens.weight")&&tied.tiedAliasExpansion,"selected tied alias does not have explicit canonical closure");++passed;}
    rejects([&]{selectGraph(source,{"unknown.weight"});});
    rejects([&]{selectGraph(source,{"mtp.fc.weight"});});
    rejects([&]{selectGraph(source,{"model.language_model.norm.weight","model.language_model.norm.weight"});});
    auto embedding=source.at("model.language_model.embed_tokens.weight");auto shards=shardPlan(embedding);uint64_t n=0;uint32_t first=0;for(auto s:shards){require(s.firstRow==first&&s.storageBytes<=maximumShardBytes,"embedding shard boundary differs");first+=s.rows;n+=s.elements;}require(n==embedding.elements&&first==248320&&shards.size()>1,"embedding row split incomplete");++passed;
    rejects([&]{parse("{\"a\":{\"dtype\":\"BF16\",\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,2]}}",2);});
    rejects([&]{parse(descriptor("x","U8",Json::array({2}),0,2),2);});
    rejects([&]{parse(descriptor("linear.weight","F32",Json::array({2,2}),0,16),16);});
    rejects([&]{parse(descriptor("large.norm","F32",Json::array({16385}),0,16385*4),16385*4);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({0}),0,0),0);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({-1}),0,2),2);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({UINT32_MAX,UINT32_MAX,UINT32_MAX}),0,2),2);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({3}),0,5),5);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({3}),2,8),8);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({3}),0,6),8);});
    rejects([&]{parse(descriptor("x","BF16",Json::array({3}),0,8),6);});
    rejects([&]{parse("{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[2,6]}}",6);});
    rejects([&]{auto bad=source;bad.erase(bad.begin());validatePinned(bad);});
    rejects([&]{auto bad=source;bad.begin()->second.shape[0]+=1;validatePinned(bad);});
    rejects([&]{auto bad=source;bad.begin()->second.dtype="F32";validatePinned(bad);});
    rejects([&]{multiply(UINT64_MAX,2);});rejects([&]{add(UINT64_MAX,1);});
    uint8_t prefix[8]={0x70,0x65,0x01,0,0,0,0,0};require(readU64LE(prefix)==91504,"little-endian header prefix differs");++passed;
    Json receipt={{"test","directcompute_model_weights_parser"},{"checks_passed",passed},{"gpu_or_native_import",false},{"full_model_loaded",false}};
    if(argc==2){
        std::ifstream in(argv[1],std::ios::binary|std::ios::ate);require(bool(in),"source header fixture open failed");auto size=in.tellg();require(size>0&&uint64_t(size)<=maximumHeaderBytes,"source header fixture exceeds bound");std::string bytes(static_cast<size_t>(size),'\0');in.seekg(0);in.read(bytes.data(),static_cast<std::streamsize>(bytes.size()));require(bool(in),"source header fixture read failed");
        auto actual=parse(bytes,pinnedModelBytes-8-bytes.size());validatePinned(actual);uint64_t uploaded=0,mtpBytes=0;size_t count=0,shardCount=0;
        for(const auto& item:actual){if(item.first.rfind("mtp.",0)==0){mtpBytes+=item.second.end-item.second.begin;continue;}++count;if(item.first=="lm_head.weight")continue;for(const auto& s:shardPlan(item.second)){uploaded+=s.storageBytes;++shardCount;}}
        receipt["actual_pinned_header_only"]={{"tensors",actual.size()},{"graph_names",count},{"unique_shards",shardCount},{"upload_storage_bytes",uploaded},{"mtp_skipped_bytes",mtpBytes},{"payload_read",false}};
    }else require(argc==1,"one optional bounded header fixture path is supported");
    std::cout<<receipt.dump()<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
