// New code, MPL-2.0. D3D11 compute ownership adapted from Const-me/Whisper.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "api.h"
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include "../adapter_identity.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <unordered_set>
using Microsoft::WRL::ComPtr;
namespace chandra::dc {
static void checked(HRESULT hr,const char* action) {
 if(FAILED(hr))throw std::runtime_error(std::string(action)+" HRESULT="+std::to_string(uint32_t(hr)));
}
struct Accounting {std::atomic<uint64_t> bytes{0},peak{0};};
struct Storage {
 ComPtr<ID3D11Buffer> data;
 ComPtr<ID3D11ShaderResourceView> srv;
 ComPtr<ID3D11UnorderedAccessView> uav;
 std::shared_ptr<Accounting> account;
 uint32_t bytes=0;
 ~Storage(){if(account)account->bytes.fetch_sub(bytes);}
};
struct Device::Impl {
 ComPtr<ID3D11Device> device;
 ComPtr<ID3D11DeviceContext> context;
 ComPtr<IDXGIAdapter3> adapter;
 ComPtr<ID3D11Buffer> constants;
 std::unordered_map<std::string,ComPtr<ID3D11ComputeShader>> shaders;
 std::shared_ptr<Accounting> account=std::make_shared<Accounting>();
 std::filesystem::path shaderRoot;
 std::string identity;
 std::thread::id owner=std::this_thread::get_id();
 std::vector<std::shared_ptr<Storage>> pending;
 std::unordered_set<const Storage*> pendingIds;
 struct Timing {std::string name;uint32_t x,y,z;ComPtr<ID3D11Query> first,last;};
 std::vector<Timing> timings;
 ComPtr<ID3D11Query> disjoint;
 bool profiling=false;
 DXGI_QUERY_VIDEO_MEMORY_INFO memory() const {
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  checked(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info),"Query local video memory");
  return info;
 }
 void retain(const Buffer& b){
  if(pendingIds.count(b.storage.get()))return;
  pending.push_back(b.storage);
  try{pendingIds.insert(b.storage.get());}catch(...){pending.pop_back();throw;}
 }
 void thread() const {if(std::this_thread::get_id()!=owner)throw std::runtime_error("D3D11 actor used from another thread");}
 Buffer make(uint32_t count,const void* initial) {
  thread();
  const uint64_t bytes=uint64_t(count)*4;
  if(!count||bytes>128ull*1024*1024)throw std::runtime_error("D3D11 buffer must be nonempty and at most 128 MiB");
  if(account->bytes.load()+bytes>13ull*1024*1024*1024)throw std::runtime_error("Tracked D3D11 allocation cap exceeded");
  auto observed=memory();
  if(observed.Budget<=observed.CurrentUsage||observed.Budget-observed.CurrentUsage<bytes+2ull*1024*1024*1024)
   throw std::runtime_error("GPU contention or insufficient WDDM budget; retain 2 GiB local headroom");
  auto s=std::make_shared<Storage>();
  D3D11_BUFFER_DESC desc{};desc.ByteWidth=uint32_t(bytes);desc.Usage=D3D11_USAGE_DEFAULT;
  desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
  desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  D3D11_SUBRESOURCE_DATA init{};init.pSysMem=initial;
  checked(device->CreateBuffer(&desc,initial?&init:nullptr,&s->data),"CreateBuffer");
  D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.Format=DXGI_FORMAT_R32_TYPELESS;
  sd.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;sd.BufferEx.NumElements=count;sd.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
  checked(device->CreateShaderResourceView(s->data.Get(),&sd,&s->srv),"Create raw SRV");
  D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;
  ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=count;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
  checked(device->CreateUnorderedAccessView(s->data.Get(),&ud,&s->uav),"Create raw UAV");
  s->account=account;s->bytes=uint32_t(bytes);uint64_t total=account->bytes.fetch_add(bytes)+bytes;
  uint64_t peak=account->peak.load();while(peak<total&&!account->peak.compare_exchange_weak(peak,total)){}
  return {s,count,count,false};
 }
 void valid(const Buffer& b) const {
  if(!b.storage||b.storage->account!=account||!b.words||uint64_t(b.words)*4!=b.storage->bytes)
   throw std::runtime_error("Invalid buffer or foreign D3D11 device");
  const uint64_t capacity=uint64_t(b.words)*(b.packedBF16?2:1);
  if(!b.logicalElements||b.logicalElements>capacity)throw std::runtime_error("Invalid logical buffer extent");
 }
 ID3D11ComputeShader* shader(const std::string& name) {
  if(name.empty()||name.find("..")!=std::string::npos||std::filesystem::path(name).is_absolute())
   throw std::runtime_error("Relative shader name required");
  auto found=shaders.find(name);if(found!=shaders.end())return found->second.Get();
  auto relative=std::filesystem::path(name);
  if(relative.extension().empty())relative.replace_extension(L".hlsl");
  if(!relative.has_parent_path())relative=std::filesystem::path(L"runtime")/relative;
  auto path=shaderRoot/relative;
  ComPtr<ID3DBlob> code,errors;
  HRESULT hr=D3DCompileFromFile(path.c_str(),nullptr,D3D_COMPILE_STANDARD_FILE_INCLUDE,"main","cs_5_0",
   D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&errors);
  if(FAILED(hr)){
   std::string diagnostic=errors?std::string(static_cast<const char*>(errors->GetBufferPointer()),errors->GetBufferSize()):"";
   throw std::runtime_error("Shader compile "+name+": "+diagnostic);
  }
  ComPtr<ID3D11ComputeShader> result;
  checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&result),"CreateComputeShader");
  auto inserted=shaders.emplace(name,std::move(result));return inserted.first->second.Get();
 }
};
Device::Device(const std::wstring& directory,const std::string& pci,const std::string& luid):impl(std::make_unique<Impl>()){
 if(pci.empty())throw std::runtime_error("Explicit physical PCI identity is required");
 auto chosen=chandra::selectAdapter(chandra::intelAdapters(L"A770"),-1,pci,luid);
 impl->identity=chosen.identity.dump();impl->shaderRoot=directory;
 checked(chosen.handle.As(&impl->adapter),"Require WDDM budget interface");
 if(!std::filesystem::is_directory(impl->shaderRoot))throw std::runtime_error("Shader directory absent");
 D3D_FEATURE_LEVEL requested[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0},actual;
 HRESULT hr=D3D11CreateDevice(chosen.handle.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,requested,2,D3D11_SDK_VERSION,
  &impl->device,&actual,&impl->context);
 if(hr==E_INVALIDARG)hr=D3D11CreateDevice(chosen.handle.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,requested+1,1,D3D11_SDK_VERSION,
  &impl->device,&actual,&impl->context);
 checked(hr,"D3D11CreateDevice");
 D3D11_BUFFER_DESC cb{};cb.ByteWidth=256;cb.Usage=D3D11_USAGE_DEFAULT;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 checked(impl->device->CreateBuffer(&cb,nullptr,&impl->constants),"Create constant buffer");
}
Device::~Device()=default;
Buffer Device::floats(uint32_t count,const float* values){return impl->make(count,values);}
Buffer Device::words(uint32_t count,const uint32_t* values){return impl->make(count,values);}
void Device::upload(Buffer& b,const void* values,uint32_t bytes){
 impl->thread();impl->valid(b);if(!values||bytes!=b.storage->bytes)throw std::runtime_error("Complete bounded buffer upload required");
 impl->retain(b);
 impl->context->UpdateSubresource(b.storage->data.Get(),0,nullptr,values,0,0);
}
void Device::dispatch(const std::string& name,const std::vector<const Buffer*>& inputs,const std::vector<Buffer*>& outputs,
 const void* params,uint32_t paramBytes,uint32_t x,uint32_t y,uint32_t z){
 impl->thread();
 if(inputs.size()>16||outputs.empty()||outputs.size()>8||paramBytes>256||(paramBytes&&!params)||!x||!y||!z||x>65535||y>65535||z>65535)
  throw std::runtime_error("Invalid bounded compute dispatch");
 std::unordered_set<const Storage*> writable;
 for(auto p:outputs){if(!p)throw std::runtime_error("Null UAV");impl->valid(*p);if(!writable.insert(p->storage.get()).second)throw std::runtime_error("Aliased UAVs");}
 for(auto p:inputs){if(!p)throw std::runtime_error("Null SRV");impl->valid(*p);if(writable.count(p->storage.get()))throw std::runtime_error("SRV/UAV alias forbidden");}
 std::array<uint32_t,64> constants{};if(paramBytes)std::memcpy(constants.data(),params,paramBytes);
 ID3D11Buffer* cb=impl->constants.Get();impl->context->CSSetConstantBuffers(0,1,&cb);
 std::array<ID3D11ShaderResourceView*,16> srvs{};std::array<ID3D11UnorderedAccessView*,8> uavs{};
 for(size_t i=0;i<inputs.size();++i)srvs[i]=inputs[i]->storage->srv.Get();
 for(size_t i=0;i<outputs.size();++i)uavs[i]=outputs[i]->storage->uav.Get();
 auto shader=impl->shader(name);
 for(auto p:inputs)impl->retain(*p);
 for(auto p:outputs)impl->retain(*p);
 Impl::Timing timing{name,x,y,z,{},{}};
 if(impl->profiling){
  if(impl->timings.size()>=4096)throw std::runtime_error("Profile window exceeds 4096 bounded dispatch records");
  D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP,0};
  checked(impl->device->CreateQuery(&desc,&timing.first),"Create first timestamp");
  checked(impl->device->CreateQuery(&desc,&timing.last),"Create last timestamp");
  impl->timings.push_back(std::move(timing));
 }
 impl->context->UpdateSubresource(impl->constants.Get(),0,nullptr,constants.data(),0,0);
 impl->context->CSSetShader(shader,nullptr,0);
 impl->context->CSSetShaderResources(0,16,srvs.data());impl->context->CSSetUnorderedAccessViews(0,8,uavs.data(),nullptr);
 if(impl->profiling)impl->context->End(impl->timings.back().first.Get());
 impl->context->Dispatch(x,y,z);
 if(impl->profiling)impl->context->End(impl->timings.back().last.Get());
 srvs.fill(nullptr);uavs.fill(nullptr);
 impl->context->CSSetShaderResources(0,16,srvs.data());impl->context->CSSetUnorderedAccessViews(0,8,uavs.data(),nullptr);
}
void Device::drain(uint32_t timeout){
 impl->thread();if(!timeout||timeout>30000)throw std::runtime_error("Bounded drain timeout required");
 D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT,0};ComPtr<ID3D11Query> q;
 checked(impl->device->CreateQuery(&desc,&q),"Create drain query");impl->context->End(q.Get());impl->context->Flush();
 auto start=std::chrono::steady_clock::now();
 for(;;){BOOL complete=FALSE;HRESULT hr=impl->context->GetData(q.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);
  if(hr==S_OK&&complete){impl->pending.clear();impl->pendingIds.clear();return;}checked(hr,"D3D11 drain query");
  if(std::chrono::steady_clock::now()-start>std::chrono::milliseconds(timeout))throw std::runtime_error("D3D11 drain deadline exceeded; retire owned worker");
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
 }
}
std::vector<uint32_t> Device::readWords(const Buffer& b){
 impl->thread();impl->valid(b);drain();D3D11_BUFFER_DESC desc{};b.storage->data->GetDesc(&desc);
 if(impl->account->bytes.load()+desc.ByteWidth>13ull*1024*1024*1024)throw std::runtime_error("Readback staging exceeds tracked allocation cap");
 auto memory=impl->memory();
 if(memory.Budget<=memory.CurrentUsage||memory.Budget-memory.CurrentUsage<desc.ByteWidth+2ull*1024*1024*1024)
  throw std::runtime_error("Readback staging would breach 2 GiB local headroom");
 desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.MiscFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Buffer> staging;checked(impl->device->CreateBuffer(&desc,nullptr,&staging),"Create staging");
 std::vector<uint32_t> result(b.words);
 impl->context->CopyResource(staging.Get(),b.storage->data.Get());drain();D3D11_MAPPED_SUBRESOURCE mapped{};
 checked(impl->context->Map(staging.Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped),"Map completed readback");
 std::memcpy(result.data(),mapped.pData,b.storage->bytes);impl->context->Unmap(staging.Get(),0);return result;
}
std::vector<float> Device::readFloats(const Buffer& b){
 if(b.packedBF16)throw std::runtime_error("Packed BF16 is not float storage");auto words=readWords(b);
 std::vector<float> result(words.size());std::memcpy(result.data(),words.data(),words.size()*4);return result;
}
void Device::zero(Buffer& b){impl->thread();impl->valid(b);const UINT zeros[4]{};impl->retain(b);impl->context->ClearUnorderedAccessViewUint(b.storage->uav.Get(),zeros);}
uint64_t Device::trackedBufferBytes()const{return impl->account->bytes.load();}
std::string Device::identityJson()const{return impl->identity;}
std::string Device::memoryJson()const{
 impl->thread();auto info=impl->memory();
 return chandra::Json{{"local_budget",info.Budget},{"local_usage",info.CurrentUsage},
  {"tracked_live",impl->account->bytes.load()},{"tracked_peak",impl->account->peak.load()},
  {"required_headroom",2ull*1024*1024*1024}}.dump();
}
void Device::beginProfile(){
 impl->thread();if(impl->profiling||!impl->timings.empty())throw std::runtime_error("Profile window already open or uncollected");
 drain();D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP_DISJOINT,0};
 checked(impl->device->CreateQuery(&desc,&impl->disjoint),"Create disjoint query");
 impl->context->Begin(impl->disjoint.Get());impl->profiling=true;
}
std::string Device::finishProfile(){
 impl->thread();if(!impl->profiling)throw std::runtime_error("No profile window");
 impl->context->End(impl->disjoint.Get());impl->profiling=false;drain();
 D3D11_QUERY_DATA_TIMESTAMP_DISJOINT frequency{};
 if(impl->context->GetData(impl->disjoint.Get(),&frequency,sizeof(frequency),D3D11_ASYNC_GETDATA_DONOTFLUSH)!=S_OK||frequency.Disjoint||!frequency.Frequency)
  throw std::runtime_error("GPU timestamps disjoint or unavailable; timing rejected");
 chandra::Json rows=chandra::Json::array();
 for(const auto& record:impl->timings){UINT64 first=0,last=0;
  if(impl->context->GetData(record.first.Get(),&first,sizeof(first),D3D11_ASYNC_GETDATA_DONOTFLUSH)!=S_OK||
     impl->context->GetData(record.last.Get(),&last,sizeof(last),D3D11_ASYNC_GETDATA_DONOTFLUSH)!=S_OK||last<first)
   throw std::runtime_error("Completed GPU timestamp pair unavailable");
  rows.push_back({{"shader",record.name},{"groups",{record.x,record.y,record.z}},
   {"gpu_milliseconds",double(last-first)*1000.0/double(frequency.Frequency)}});
 }
 impl->timings.clear();impl->disjoint.Reset();
 return chandra::Json{{"frequency",frequency.Frequency},{"disjoint",false},{"dispatches",rows}}.dump();
}
}
