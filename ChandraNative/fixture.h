// New ChandraNative code, MPL-2.0. Hash-checked, bounded external activation fixture.
#pragma once
#include "oracle.h"
#include "vendor/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <climits>
#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/sha.h>
#endif
namespace chandra {
using Json=nlohmann::json;
inline void fixtureRequire(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
inline std::string sha256(const void* data,size_t bytes) {
    unsigned char digest[32];
#ifdef _WIN32
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    fixtureRequire(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider failure");
    ULONG objectBytes=0,received=0;std::vector<unsigned char> object;
    try {
        fixtureRequire(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&objectBytes),sizeof(objectBytes),&received,0)>=0,"SHA256 property failure");object.resize(objectBytes);
        fixtureRequire(BCryptCreateHash(algorithm,&hash,object.data(),objectBytes,nullptr,0,0)>=0,"SHA256 create failure");
        fixtureRequire(bytes<=ULONG_MAX && BCryptHashData(hash,const_cast<PUCHAR>(static_cast<const unsigned char*>(data)),ULONG(bytes),0)>=0,"SHA256 input failure");
        fixtureRequire(BCryptFinishHash(hash,digest,sizeof(digest),0)>=0,"SHA256 finish failure");
    }catch(...) {if(hash)BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);throw;}
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);
#else
    SHA256(static_cast<const unsigned char*>(data),bytes,digest);
#endif
    std::ostringstream s;s<<std::hex<<std::setfill('0');for(unsigned char v:digest)s<<std::setw(2)<<unsigned(v);return s.str();
}
inline std::vector<unsigned char> boundedRead(const std::filesystem::path& path,size_t limit) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);fixtureRequire(bool(in),"fixture file open failed");auto length=in.tellg();fixtureRequire(length>=0 && uint64_t(length)<=limit,"fixture file exceeds bound");std::vector<unsigned char> bytes(static_cast<size_t>(length));in.seekg(0);if(!bytes.empty())in.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()));fixtureRequire(bool(in),"fixture read failed");return bytes;
}
inline void exactKeys(const Json& j,std::initializer_list<const char*> names) {
    fixtureRequire(j.is_object() && j.size()==names.size(),"fixture object fields mismatch");for(auto name:names)fixtureRequire(j.contains(name),"missing fixture field");
}
inline uint64_t unsignedInteger(const Json& j) {fixtureRequire(j.is_number_unsigned(),"fixture requires nonnegative integer JSON number");return j.get<uint64_t>();}
inline bool hashString(const Json& j) {if(!j.is_string())return false;auto s=j.get<std::string>();return s.size()==64 && s.find_first_not_of("0123456789abcdef")==std::string::npos;}
struct ExternalFixture {
    Fixture inputs;std::vector<float> expectedState,expectedOutput;Json metadata;
    std::string metadataHash,payloadHash;
};
inline ExternalFixture loadFixture(const std::filesystem::path& metadataPath) {
    auto raw=boundedRead(metadataPath,1024*1024);ExternalFixture result;result.metadataHash=sha256(raw.data(),raw.size());std::vector<std::set<std::string>> objects;
    auto duplicateCheck=[&](int,Json::parse_event_t event,Json& parsed) {
        if(event==Json::parse_event_t::object_start)objects.emplace_back();
        else if(event==Json::parse_event_t::key)fixtureRequire(!objects.empty() && objects.back().insert(parsed.get<std::string>()).second,"duplicate JSON key");
        else if(event==Json::parse_event_t::object_end){fixtureRequire(!objects.empty(),"JSON object stack mismatch");objects.pop_back();}
        return true;
    };
    result.metadata=Json::parse(raw.begin(),raw.end(),duplicateCheck,true,false);const auto& meta=result.metadata;
    Json root=meta;if(root.contains("context")){fixtureRequire(root.at("context").is_object(),"context must be object");root.erase("context");}
    exactKeys(root,{"schema","payload_file","payload_bytes","payload_sha256","dtype","byte_order","model_revision","source","semantics","tensors"});
    fixtureRequire(meta.at("schema")=="chandra.native.delta.v1" && meta.at("dtype")=="float32" && meta.at("byte_order")=="little","unsupported fixture format");
    fixtureRequire(meta.at("model_revision")=="af93b47dba1b47b6640c86ccf487ed2260ab9a09" && meta.at("source").is_object(),"fixture model/source mismatch");
    fixtureRequire(meta.at("payload_file").is_string(),"payload filename type");auto filename=meta.at("payload_file").get<std::string>();fixtureRequire(!filename.empty() && filename.size()<128 && filename.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")==std::string::npos && filename!="." && filename!="..","payload must be a local basename");
    const auto& sem=meta.at("semantics");exactKeys(sem,{"q","k","g","beta","heads","initial_state","layout"});
    fixtureRequire(sem.at("q")=="l2_normalized_scaled_sqrt_keydim" && sem.at("k")=="l2_normalized" && sem.at("g")=="negative_log_decay" && sem.at("beta")=="post_sigmoid" && sem.at("heads")=="expanded_value_heads" && sem.at("initial_state")=="zeros" && sem.at("layout")=="THD","unsupported recurrence semantics");
    fixtureRequire(hashString(meta.at("payload_sha256")),"invalid payload SHA256");auto bytes=boundedRead(metadataPath.parent_path()/filename,512ull*1024*1024);fixtureRequire(bytes.size()==unsignedInteger(meta.at("payload_bytes")),"payload length mismatch");result.payloadHash=sha256(bytes.data(),bytes.size());fixtureRequire(result.payloadHash==meta.at("payload_sha256"),"payload SHA256 mismatch");
    const auto& tensors=meta.at("tensors");exactKeys(tensors,{"q","k","v","g","beta","expected_output","expected_state"});uint64_t consumed=0;
    auto tensor=[&](const char* name)->std::pair<std::vector<unsigned>,std::vector<float>> {
        const auto& desc=tensors.at(name);exactKeys(desc,{"shape","offset_bytes","byte_length","sha256"});fixtureRequire(desc.at("shape").is_array() && desc.at("shape").size()>=2 && desc.at("shape").size()<=3,"invalid tensor rank");std::vector<unsigned> shape;uint64_t elements=1;
        for(const auto& dimension:desc.at("shape")){uint64_t d=unsignedInteger(dimension);fixtureRequire(d>0 && d<=4096 && elements<=INT_MAX/4/d,"tensor shape overflow");shape.push_back(unsigned(d));elements*=d;}
        uint64_t offset=unsignedInteger(desc.at("offset_bytes")),length=unsignedInteger(desc.at("byte_length"));fixtureRequire(offset==consumed && length==elements*4 && offset<=bytes.size() && length<=bytes.size()-offset,"tensor offsets/lengths mismatch");fixtureRequire(hashString(desc.at("sha256")) && sha256(bytes.data()+offset,size_t(length))==desc.at("sha256"),"tensor SHA256 mismatch");consumed+=length;std::vector<float> values(static_cast<size_t>(elements));std::memcpy(values.data(),bytes.data()+offset,size_t(length));for(float x:values)fixtureRequire(std::isfinite(x),"nonfinite tensor data");return {shape,std::move(values)};
    };
    // Serialization order is part of v1, and covers the entire payload without gaps.
    auto q=tensor("q"),k=tensor("k"),v=tensor("v"),g=tensor("g"),beta=tensor("beta"),out=tensor("expected_output"),state=tensor("expected_state");fixtureRequire(consumed==bytes.size(),"unaccounted payload bytes");
    fixtureRequire(q.first.size()==3 && v.first.size()==3,"q/v rank mismatch");unsigned t=q.first[0],h=q.first[1],keys=q.first[2],values=v.first[2];fixtureRequire(h<=32 && keys<=128 && values<=128,"unsupported kernel dimensions");
    fixtureRequire(k.first==q.first && v.first==std::vector<unsigned>{t,h,values} && out.first==v.first && g.first==std::vector<unsigned>{t,h} && beta.first==g.first && state.first==std::vector<unsigned>{h,keys,values},"incompatible tensor shapes");
    for(float x:g.second)fixtureRequire(x<=0,"positive log decay");
    for(float x:beta.second)fixtureRequire(x>=0 && x<=1,"beta outside [0,1]");
    result.inputs={t,h,keys,values,std::move(q.second),std::move(k.second),std::move(v.second),std::move(g.second),std::move(beta.second)};result.expectedOutput=std::move(out.second);result.expectedState=std::move(state.second);return result;
}
struct Comparison {Error absolute;double maxToleranceRatio=0;bool passed=true;};
template<class A,class B> inline Comparison compare(const std::vector<A>& observed,const std::vector<B>& expected,double atol,double rtol) {
    fixtureRequire(atol>=0 && rtol>=0 && std::isfinite(atol) && std::isfinite(rtol),"invalid numerical tolerance");Comparison c;c.absolute=error(observed,expected);for(size_t i=0;i<observed.size();++i){double difference=std::abs(double(observed[i])-double(expected[i])),bound=atol+rtol*std::abs(double(expected[i]));double ratio=bound>0?difference/bound:(difference==0?0:std::numeric_limits<double>::infinity());c.maxToleranceRatio=std::max(c.maxToleranceRatio,ratio);if(ratio>1)c.passed=false;}return c;
}
}
