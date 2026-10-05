// New ChandraNative code, MPL-2.0. Independent small operand fixture driver.
// No model loader. Expected values are prepared by a separate scalar oracle.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "api.h"
#include "../vendor/nlohmann/json.hpp"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

using Json=nlohmann::json;
using namespace chandra::dc;
namespace {
constexpr const char* fixtureSha="f90c27f581155bdbfcf00d9a8eaf4703446cba7e88151e7ef51dc6f00a7404d7";
constexpr uint64_t fixtureBytes=1014955;
constexpr double screen=2e-5; // Retained selected-kernel screen, declared before GPU results.
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::string ascii(const std::wstring& text){std::string result;for(auto c:text){require(c>0&&c<128,"ASCII PCI/LUID/option required");result.push_back(char(c));}return result;}
std::string hash(const std::vector<uint8_t>& bytes){
 BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE value=nullptr;std::vector<uint8_t> object;std::string result;
 require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA256 provider failed");
 try{ULONG size=0,received=0;require(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&received,0)>=0,"SHA256 property failed");object.resize(size);
  require(BCryptCreateHash(algorithm,&value,object.data(),size,nullptr,0,0)>=0,"SHA256 create failed");
  require(BCryptHashData(value,const_cast<PUCHAR>(bytes.data()),ULONG(bytes.size()),0)>=0,"SHA256 update failed");uint8_t output[32];require(BCryptFinishHash(value,output,32,0)>=0,"SHA256 finish failed");
  std::ostringstream stream;stream<<std::hex<<std::setfill('0');for(auto b:output)stream<<std::setw(2)<<unsigned(b);result=stream.str();
 }catch(...){if(value)BCryptDestroyHash(value);BCryptCloseAlgorithmProvider(algorithm,0);throw;}
 BCryptDestroyHash(value);BCryptCloseAlgorithmProvider(algorithm,0);return result;
}
Json loadFixture(const std::filesystem::path& path){
 require(std::filesystem::file_size(path)==fixtureBytes,"Exact frozen small fixture length required");
 std::ifstream stream(path,std::ios::binary);require(bool(stream),"Cannot open small fixture");std::vector<uint8_t> bytes(fixtureBytes);stream.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(bytes.size()));require(stream.gcount()==std::streamsize(bytes.size()),"Truncated small fixture");
 require(hash(bytes)==fixtureSha,"Exact independent fixture SHA256 required");auto value=Json::parse(bytes);
 require(value.at("schema")=="private.chandra.directcompute.native-operator-fixture.v1"&&value.at("inactive")==true,"Wrong fixture schema/scope");return value;
}
void flatten(const Json& value,std::vector<float>& out){
 if(value.is_array()){for(const auto& item:value)flatten(item,out);return;}
 require(value.is_number(),"Numeric scalar fixture required");double x=value.get<double>();require(std::isfinite(x)&&std::abs(x)<=std::numeric_limits<float>::max(),"Finite FP32 input required");out.push_back(float(x));
}
std::vector<float> floats(const Json& value,size_t count){std::vector<float> out;flatten(value,out);require(out.size()==count,"Exact fixture element count required");return out;}
uint16_t bf16(float x){uint32_t u;std::memcpy(&u,&x,4);if((u&0x7fffffff)>0x7f800000)return uint16_t((u>>16)|0x40);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
Buffer activation(Device& d,const std::vector<float>& x){require(!x.empty()&&x.size()<=1024*1024,"Small complete operand required");return d.floats(uint32_t(x.size()),x.data());}
Buffer packed(Device& d,const std::vector<float>& x){std::vector<uint32_t> words((x.size()+1)/2,0);for(size_t i=0;i<x.size();i++)words[i/2]|=uint32_t(bf16(x[i]))<<((i%2)*16);auto b=d.words(uint32_t(words.size()),words.data());b.logicalElements=x.size();b.packedBF16=true;return b;}
Weight smallWeight(Device& d,const std::vector<float>& x,uint32_t rows,uint32_t cols,bool shard=false){
 require(x.size()==uint64_t(rows)*cols,"Small weight shape differs");Weight w;w.rows=rows;w.cols=cols;w.bf16=true;
 uint32_t first=0;while(first<rows){uint32_t count=shard?std::min(3u,rows-first):rows;std::vector<float> slice(x.begin()+size_t(first)*cols,x.begin()+size_t(first+count)*cols);w.shards.push_back(packed(d,slice));w.shardFirstRows.push_back(first);first+=count;}return w;
}
bool compare(Json& report,const char* name,const std::vector<float>& observed,const Json& expected,bool exact=false){
 // Keep FP64 expected scalars as double; converting before comparison would lose
 // the independent adjudicator's low bits, especially for FP32 recurrent state.
 std::vector<double> ref;auto walk=[&](auto&& self,const Json& value)->void{if(value.is_array()){for(const auto& v:value)self(self,v);}else{require(value.is_number(),"Numeric expected scalar required");double x=value.get<double>();require(std::isfinite(x),"Finite expected scalar required");ref.push_back(x);}};walk(walk,expected);
 require(ref.size()==observed.size(),"Complete output extent differs");double maximum=0,squares=0;uint64_t bad=0;
 for(size_t i=0;i<ref.size();i++){require(std::isfinite(observed[i]),"Native output is nonfinite");double error=std::abs(double(observed[i])-ref[i]);maximum=std::max(maximum,error);squares+=error*error;float expectedFloat=float(ref[i]);uint32_t expectedBits,observedBits;std::memcpy(&expectedBits,&expectedFloat,4);std::memcpy(&observedBits,&observed[i],4);if(exact?observedBits!=expectedBits:error>screen)bad++;}
 bool pass=bad==0;report["checks"].push_back({{"name",name},{"passed",pass},{"elements",ref.size()},{"maximum_absolute",maximum},{"rms",std::sqrt(squares/std::max(size_t(1),ref.size()))},{"violations",bad},{"absolute_screen",exact?0:screen},{"relative_screen",0},{"observed",observed}});return pass;
}
struct P{uint32_t rows,width,offset,stride,base,count,mode,pad;};
struct DeltaP{uint32_t heads,keyDim,valueDim,tokens,start,reserved0,reserved1,reserved2;};
static_assert(sizeof(P)==32&&sizeof(DeltaP)==32,"Native fixture cbuffer ABI");
void run(Device& d,const Json& fixture,Json& report){
 bool passed=true;
 {
  // Separately pinned scalar bit witnesses from oracles.py/check_oracles.py.
  // These exercise conversion only; nonfinite graph operands are not admitted.
  std::vector<uint32_t> input={0x00000000,0x80000000,0x3f808000,0x3f818000,0xbf808000,0x00018000,0x7f800000,0xff800000,0x7f800001,0x7fc10000};
  std::vector<uint32_t> expected={0x00000000,0x80000000,0x3f800000,0x3f820000,0xbf800000,0x00020000,0x7f800000,0xff800000,0x7fc00000,0x7fc10000};
  auto raw=d.words(uint32_t(input.size()),input.data());auto output=unary(d,raw,uint32_t(input.size()),3,false);auto observed=d.readWords(output);bool exact=observed==expected;
  report["checks"].push_back({{"name","BF16_RNE_ties_signed_zero_nan_exact_bits"},{"passed",exact},{"input_fp32_bits",input},{"expected_fp32_bits",expected},{"observed_fp32_bits",observed}});passed=exact&&passed;d.drain();
 }
 {
  const auto& p=fixture.at("projection");auto input=activation(d,floats(p.at("input"),14));auto w=smallWeight(d,floats(p.at("weights"),35),5,7,true);
  auto output=linear(d,input,w,2,false);passed=compare(report,"projection_fp32_vs_fp64",d.readFloats(output),p.at("fp64_accumulator"))&&passed;
  output=linear(d,input,w,2,true);passed=compare(report,"projection_bf16_exact",d.readFloats(output),p.at("bf16_output"),true)&&passed;
  d.drain();
 }
 {
  const auto& n=fixture.at("norm");auto input=activation(d,floats(n.at("input"),16));auto w=smallWeight(d,floats(n.at("weight"),8),1,8);
  auto output=rmsNorm(d,input,w,2,8,1e-6f,true,true);passed=compare(report,"zero_centered_rms_fp64",d.readFloats(output),n.at("expected"))&&passed;d.drain();
 }
 {
  const auto& g=fixture.at("gdn");require(g.at("tokens")==3&&g.at("heads")==2&&g.at("key_dim")==4&&g.at("value_dim")==4,"Exact tiny GDN shape required");
  auto q=activation(d,floats(g.at("q"),24)),k=activation(d,floats(g.at("k"),24)),v=activation(d,floats(g.at("v"),24));auto decay=activation(d,floats(g.at("g"),6)),beta=activation(d,floats(g.at("beta"),6));auto initial=floats(g.at("initial_state"),32);
  auto state=activation(d,initial),output=d.floats(24);d.zero(output);DeltaP p{2,4,4,3,0,0,0,0};
  d.dispatch("runtime/text_delta.hlsl",{&q,&k,&v,&decay,&beta},{&state,&output},&p,sizeof(p),2);auto wholeState=d.readFloats(state),wholeOutput=d.readFloats(output);
  passed=compare(report,"GDN_complete_FP32_state_vs_FP64",wholeState,g.at("fp64").at("final_state"))&&passed;passed=compare(report,"GDN_complete_BF16_output_vs_FP64",wholeOutput,g.at("fp64").at("output"))&&passed;
  auto survivor=activation(d,initial);d.upload(state,initial.data(),uint32_t(initial.size()*4));d.zero(output);
  p.tokens=1;p.start=0;d.dispatch("runtime/text_delta.hlsl",{&q,&k,&v,&decay,&beta},{&state,&output},&p,sizeof(p),2);d.drain();p.tokens=2;p.start=1;d.dispatch("runtime/text_delta.hlsl",{&q,&k,&v,&decay,&beta},{&state,&output},&p,sizeof(p),2);
  passed=compare(report,"GDN_split_state_exact",d.readFloats(state),Json(wholeState),true)&&passed;passed=compare(report,"GDN_split_output_exact",d.readFloats(output),Json(wholeOutput),true)&&passed;
  passed=compare(report,"GDN_unrelated_state_isolation_exact",d.readFloats(survivor),g.at("initial_state"),true)&&passed;d.drain();
 }
 {
  const auto& a=fixture.at("attention");require(a.at("rows")==3&&a.at("query_heads")==16&&a.at("kv_heads")==4&&a.at("head_dim")==256&&a.at("past")==2&&a.at("keys")==5,"Exact native attention ABI shape required");
  auto q=activation(d,floats(a.at("q"),3*16*256)),k=activation(d,floats(a.at("k"),5*4*256)),v=activation(d,floats(a.at("v"),5*4*256));auto scores=d.floats(3*16*5);P p{3,16,0,0,2,5,0,0};
  d.dispatch("runtime/text_attention_scores.hlsl",{&q,&k},{&scores},&p,sizeof(p),1,3*16);passed=compare(report,"attention_scores_FP64",d.readFloats(scores),a.at("fp64").at("scores"))&&passed;
  d.dispatch("runtime/text_attention_softmax.hlsl",{},{&scores},&p,sizeof(p),3*16);auto probabilities=d.readFloats(scores);passed=compare(report,"attention_probabilities_FP64",probabilities,a.at("fp64").at("probabilities"))&&passed;
  bool causal=true;for(uint32_t row=0;row<3;row++)for(uint32_t head=0;head<16;head++)for(uint32_t key=3+row;key<5;key++)causal=causal&&probabilities[(row*16+head)*5+key]==0;
  report["checks"].push_back({{"name","attention_future_keys_zero"},{"passed",causal}});passed=causal&&passed;
  auto partials=d.floats(3*4096),output=d.floats(3*4096);p.base=0;p.offset=0;d.dispatch("runtime/text_attention_values.hlsl",{&scores,&v},{&partials},&p,sizeof(p),16,3);p.count=1;d.dispatch("runtime/text_attention_reduce.hlsl",{&partials},{&output},&p,sizeof(p),(3*4096+127)/128);
  passed=compare(report,"attention_complete_BF16_output_FP64",d.readFloats(output),a.at("fp64").at("output"))&&passed;d.drain();
 }
 d.drain();report["tracked_buffers_after_completed_scopes"]=d.trackedBufferBytes();require(d.trackedBufferBytes()==0,"Fixture scopes did not retire complete buffers");report["passed"]=passed;
}
void writeFresh(HANDLE file,const Json& value){std::string output=value.dump(2)+"\n";require(output.size()<2*1024*1024,"Small retained fixture report exceeds 2 MiB");DWORD written=0;require(WriteFile(file,output.data(),DWORD(output.size()),&written,nullptr)&&written==output.size(),"Fixture receipt write failed");require(FlushFileBuffers(file),"Fixture receipt flush failed");}
}
int wmain(int argc,wchar_t** argv){
 Json report={{"schema","private.chandra.directcompute.native-operator-fixture-run.v1"},{"fixture_sha256",fixtureSha},{"native_executed",false},{"passed",false},{"full_model_accepted",false},{"performance_accepted",false},{"checks",Json::array()}};HANDLE receipt=INVALID_HANDLE_VALUE;
 try{
  std::map<std::wstring,std::wstring> args;bool execute=false;
  for(int i=1;i<argc;i++){std::wstring name=argv[i];if(name==L"--execute"){require(!execute,"Duplicate execute flag");execute=true;}else{require(i+1<argc,"Option value absent");require(args.emplace(name,argv[++i]).second,"Duplicate option");}}
  if(!execute){std::cout<<"{\"state\":\"INACTIVE\",\"device_created\":false,\"model_loaded\":false}\n";return 0;}
  for(const auto& item:args)require(item.first==L"--fixture"||item.first==L"--shader-root"||item.first==L"--pci"||item.first==L"--luid"||item.first==L"--output","Unknown fixture option");
  for(auto required:{L"--fixture",L"--shader-root",L"--pci",L"--output"})require(args.count(required)&&!args.at(required).empty(),"Fixture/root/PCI/fresh-output binding required");
  auto fixture=loadFixture(args.at(L"--fixture"));receipt=CreateFileW(args.at(L"--output").c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);require(receipt!=INVALID_HANDLE_VALUE,"Fresh task-owned receipt required");
  {Device device(args.at(L"--shader-root"),ascii(args.at(L"--pci")),args.count(L"--luid")?ascii(args.at(L"--luid")):std::string());report["native_executed"]=true;report["device_identity"]=Json::parse(device.identityJson());report["memory_before"]=Json::parse(device.memoryJson());device.beginProfile();run(device,fixture,report);report["gpu_profile"]=Json::parse(device.finishProfile());report["memory_after"]=Json::parse(device.memoryJson());}
  report["state"]=report.at("passed")==true?"SELECTED_FIXTURE_PASS":"SELECTED_FIXTURE_FAIL";writeFresh(receipt,report);CloseHandle(receipt);std::cout<<report.dump()<<"\n";return report.at("passed")==true?0:2;
 }catch(const std::exception& error){report["passed"]=false;report["state"]="FAIL_OR_INCOMPLETE";report["error"]=error.what();if(receipt!=INVALID_HANDLE_VALUE){try{writeFresh(receipt,report);}catch(...){}CloseHandle(receipt);}std::cerr<<report.dump()<<"\n";return 1;}
}
