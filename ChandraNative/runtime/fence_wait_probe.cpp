// New code, MPL-2.0. Public deterministic, model-free original/fence completion probe.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "api.h"
#include "fence_wait.h"
#include "../vendor/nlohmann/json.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
using Json=nlohmann::json;
constexpr uint32_t words=128,rounds=8;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void emit(const Json& value){const std::string s=value.dump();require(s.size()<=32768,"Probe JSON exceeds 32 KiB contract");std::cout<<s<<'\n';require(bool(std::cout),"Probe output failed");}
Json plan(const std::string& route){return {{"schema","chandra.directcompute.fence_wait_probe.v1"},{"mode","plan"},{"route",route},
 {"model_free",true},{"device_created",false},{"tensor_bytes_bound",1024},{"staging_bytes_bound",512},
 {"host_readback_bytes_bound",512},{"total_readback_bytes",4608},{"dispatches",8},{"dispatch_groups",{1,1,1}},
 {"dispatch_threads",128},{"maximum_dispatch_milliseconds_exclusive",100},{"drains",51},
 {"maximum_drain_milliseconds",10000},{"maximum_event_block_slice_milliseconds",10},
 {"process_commit_bytes",268435456},{"job_commit_bytes",536870912},{"job_active_processes",2},
 {"core",0},{"priority","BelowNormal"},{"stdout_bytes_bound",32769},{"stderr_bytes_bound",8192},
 {"cooperative_wall_seconds",30},{"independent_job_wall_seconds",60},{"lease_seconds",15},
 {"whole_model_correctness",false},{"OCR_acceptance",false},{"throughput_acceptance",false}};}
}
int main(int argc,char** argv){
 bool execute=false,planned=false;std::string route,pci,luid,shader;
 try{
  for(int i=1;i<argc;++i){const std::string key=argv[i];
   if(key=="--plan"){require(!planned&&!execute,"One probe mode required");planned=true;continue;}
   if(key=="--execute"){require(!planned&&!execute,"One probe mode required");execute=true;continue;}
   require(i+1<argc,"Missing probe argument value");const std::string value=argv[++i];
   if(key=="--route"){require(route.empty(),"Duplicate route");route=value;}
   else if(key=="--pci"){require(pci.empty(),"Duplicate PCI");pci=value;}
   else if(key=="--luid"){require(luid.empty(),"Duplicate LUID");luid=value;}
   else if(key=="--shader-root"){require(shader.empty(),"Duplicate shader root");shader=value;}
   else throw std::runtime_error("Unknown probe option");
  }
  require(planned||execute,"One probe mode required");require(route=="query_sleep"||route=="fence_event","Explicit original or experimental route required");
  const auto selected=chandra::dc::fence_wait::selectedByEnvironment();
  require((selected==chandra::dc::fence_wait::Mode::fenceEvent)==(route=="fence_event"),"Environment and explicit probe route differ; refused before Device creation");
  if(planned){require(shader.empty()&&pci.empty()&&luid.empty(),"Plan accepts route only");emit(plan(route));return 0;}
  require(!shader.empty()&&!pci.empty()&&!luid.empty(),"Execute requires current root-selected PCI, LUID and authenticated shader root");
  // argv paths are ASCII in the owned recipe; refuse lossy widening.
  for(const unsigned char c:shader)require(c<128,"Probe shader path must be ASCII");
  const std::wstring wide(shader.begin(),shader.end());chandra::dc::Device device(wide,pci,luid);
  Json profiles=Json::array();const auto start=std::chrono::steady_clock::now();
  auto wall=[&]{require(std::chrono::steady_clock::now()-start<std::chrono::seconds(30),"Probe cooperative wall exceeded; retire owned process without replay");};
  const std::array<uint32_t,8> params{1,words,0,0,0,words,0,0};
  for(uint32_t round=0;round<rounds;++round){
   wall();std::array<uint32_t,words> expected{};
   for(uint32_t i=0;i<words;++i){const float v=float(int32_t(i)-64)*0.125f+float(round);std::memcpy(&expected[i],&v,sizeof(v));}
   device.beginProfile();auto input=device.words(words);auto output=device.words(words);
   const std::weak_ptr<chandra::dc::Storage> lifetime=input.storage;
   device.upload(input,expected.data(),uint32_t(sizeof(expected)));
   device.dispatch("text_copy",{&input},{&output},params.data(),uint32_t(sizeof(params)),1);
   input={};require(!lifetime.expired(),"Submitted input Storage was freed before drain");
   require(device.trackedBufferBytes()==1024,"Unexpected submitted buffer accounting");device.drain();
   require(lifetime.expired(),"Completed drain did not release dropped input Storage");
   require(device.trackedBufferBytes()==512,"Completed drain ownership accounting differs");
   const Json profile=Json::parse(device.finishProfile());const auto& dispatches=profile.at("dispatches");
   require(profile.at("disjoint")==false&&dispatches.size()==1,"Expected exactly one non-disjoint dispatch");
   const double ms=dispatches.at(0).at("gpu_milliseconds").get<double>();
   require(std::isfinite(ms)&&ms>=0&&ms<100,"Tiny dispatch is not strictly below 100 ms; stop without replay or expansion");profiles.push_back(profile);
   const auto actual=device.readWords(output);require(actual.size()==expected.size()&&std::equal(actual.begin(),actual.end(),expected.begin()),"Completed deterministic copy/upload readback mismatch");
   output={};require(device.trackedBufferBytes()==0,"Round left live tensor Storage");
  }
  for(uint32_t i=0;i<8;++i){wall();device.drain();}
  auto zeros=device.words(words);device.zero(zeros);device.drain();const auto actual=device.readWords(zeros);
  require(actual.size()==words&&std::all_of(actual.begin(),actual.end(),[](uint32_t v){return v==0;}),"Completed zero/readback mismatch");zeros={};
  require(device.trackedBufferBytes()==0,"Probe tensor Storage remains after completion");wall();
  Json result=plan(route);result["mode"]="execute";result["device_created"]=true;result["passed"]=true;
  result["scope"]="Tiny deterministic upload/copy/zero readbacks, repeated completed drains and retained input Storage only";
  result["device_identity"]=Json::parse(device.identityJson());result["device_memory"]=Json::parse(device.memoryJson());result["profiles"]=profiles;
  emit(result);return 0;
 }catch(const std::exception& e){
  std::string error=e.what();if(error.size()>1024)error.resize(1024);
  emit(Json{{"schema","chandra.directcompute.fence_wait_probe.v1"},{"mode",execute?"execute":"plan"},{"route",route},{"passed",false},{"error",error},
   {"owned_process_retirement_required",execute},{"replayed",false},{"whole_model_correctness",false},{"OCR_acceptance",false},{"throughput_acceptance",false}});return 3;
 }
}
