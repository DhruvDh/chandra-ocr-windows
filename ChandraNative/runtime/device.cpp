// New code, MPL-2.0. D3D11 compute ownership adapted from Const-me/Whisper.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "api.h"
#include <windows.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include "../adapter_identity.h"
#include "readback_wait.h"
#include "cpu_timing.h"
#include "fence_wait.h"
#include "copy_retention.h"
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
static_assert(weight_srv_only::defaultUsage==D3D11_USAGE_DEFAULT&&weight_srv_only::shaderResourceBind==D3D11_BIND_SHADER_RESOURCE&&weight_srv_only::rawMisc==D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS&&weight_srv_only::rawFormat==DXGI_FORMAT_R32_TYPELESS&&weight_srv_only::bufferExDimension==D3D11_SRV_DIMENSION_BUFFEREX&&weight_srv_only::rawSRVFlag==D3D11_BUFFEREX_SRV_FLAG_RAW,"Portable SRV-only descriptor constants");
static_assert(sizeof(HRESULT)==sizeof(int32_t)&&readback::okStatus==S_OK&&readback::stillDrawingStatus==DXGI_ERROR_WAS_STILL_DRAWING,"Portable Map status constants");
static_assert(fence_wait::signaled==WAIT_OBJECT_0&&fence_wait::waitTimeout==WAIT_TIMEOUT&&fence_wait::failedWaitStatus==WAIT_FAILED,"Portable event wait constants");
static void checked(HRESULT hr,const char* action) {
 if(FAILED(hr))throw std::runtime_error(std::string(action)+" HRESULT="+std::to_string(uint32_t(hr)));
}
static std::chrono::nanoseconds steadyNanoseconds() noexcept {
 return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch());
}
struct Accounting {std::atomic<uint64_t> bytes{0},peak{0};};
struct Storage {
 ComPtr<ID3D11Buffer> data;
 ComPtr<ID3D11ShaderResourceView> srv;
 ComPtr<ID3D11UnorderedAccessView> uav;
 std::shared_ptr<Accounting> account;
 uint32_t bytes=0,viewWords=0;bool readonlyWeight=false;
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
 struct ReadbackMaps {uint64_t waits=0,retried=0,stillDrawing=0;std::chrono::nanoseconds longest{0};} readbackMaps;
 std::unique_ptr<cpu_timing::Recorder> cpu; // Null unless CHANDRA_NATIVE_CPU_TIMING=1 at construction; never reset.
 fence_wait::Mode waitMode=fence_wait::Mode::querySleep;
 ComPtr<ID3D11Device5> fenceDevice;ComPtr<ID3D11DeviceContext4> fenceContext;ComPtr<ID3D11Fence> fence;
 struct Event {
  HANDLE handle=nullptr;
  ~Event(){if(handle)CloseHandle(handle);}
  Event()=default;Event(const Event&)=delete;Event& operator=(const Event&)=delete;
 } fenceEvent;
 fence_wait::State fenceState;std::atomic<bool> fenceCanceled{false};
 copy_retention::Owner<std::shared_ptr<Storage>,ComPtr<ID3D11Buffer>> copyOwner;
 struct FenceStats {
  cpu_timing::Counter calls,successes,failures,waitCalls,checks,signals,registrations,requestedMilliseconds;
  cpu_timing::Accumulator waits,elapsed;fence_wait::Outcome last;
 } fenceStats;
 void usable() const {if(fenceState.poisoned)throw std::runtime_error("Experimental fence Device poisoned; retire owned worker without replay");}
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
 Buffer make(uint32_t count,const void* initial,uint32_t physicalBytes=0) {
  thread();usable();cpu_timing::BufferScope timed(cpu.get());
  const auto extent=weight_storage::geometry(count,physicalBytes);const uint64_t bytes=extent.physicalBytes;
  if(initial&&extent.boxed())throw std::runtime_error("Padded weights require a bounded logical-prefix upload");
  if(!weight_storage::withinBudget(account->bytes.load(),bytes,13ull*1024*1024*1024))throw std::runtime_error("Tracked D3D11 allocation cap exceeded");
  auto observed=memory();timed.lap(cpu_timing::bufferBudgetQuery);
  if(observed.Budget<=observed.CurrentUsage||observed.Budget-observed.CurrentUsage<bytes+2ull*1024*1024*1024)
   throw std::runtime_error("GPU contention or insufficient WDDM budget; retain 2 GiB local headroom");
  auto s=std::make_shared<Storage>();
  D3D11_BUFFER_DESC desc{};desc.ByteWidth=uint32_t(bytes);desc.Usage=D3D11_USAGE_DEFAULT;
  desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
  desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  D3D11_SUBRESOURCE_DATA init{};init.pSysMem=initial;
  checked(device->CreateBuffer(&desc,initial?&init:nullptr,&s->data),"CreateBuffer");
  D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.Format=DXGI_FORMAT_R32_TYPELESS;
  sd.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;sd.BufferEx.NumElements=extent.logicalWords;sd.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
  checked(device->CreateShaderResourceView(s->data.Get(),&sd,&s->srv),"Create raw SRV");
  D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;
  ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;ud.Buffer.NumElements=extent.logicalWords;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
  checked(device->CreateUnorderedAccessView(s->data.Get(),&ud,&s->uav),"Create raw UAV");
  s->account=account;s->bytes=uint32_t(bytes);s->viewWords=extent.logicalWords;uint64_t total=account->bytes.fetch_add(bytes)+bytes;
  uint64_t peak=account->peak.load();while(peak<total&&!account->peak.compare_exchange_weak(peak,total)){}
  timed.created(bytes);
  return {s,count,count,false};
 }
 Buffer makeSrvOnlyWeight(uint32_t count,const void* initial) {
  thread();usable();cpu_timing::BufferScope timed(cpu.get());
  if(!initial)throw std::runtime_error("SRV-only weights require explicit initial data");
  const auto extent=weight_storage::geometry(count);const uint64_t bytes=extent.physicalBytes;
  if(!weight_storage::withinBudget(account->bytes.load(),bytes,13ull*1024*1024*1024))throw std::runtime_error("Tracked D3D11 allocation cap exceeded");
  auto observed=memory();timed.lap(cpu_timing::bufferBudgetQuery);
  if(observed.Budget<=observed.CurrentUsage||observed.Budget-observed.CurrentUsage<bytes+2ull*1024*1024*1024)
   throw std::runtime_error("GPU contention or insufficient WDDM budget; retain 2 GiB local headroom");
  auto s=std::make_shared<Storage>();
  D3D11_BUFFER_DESC desc{};desc.ByteWidth=uint32_t(bytes);desc.Usage=D3D11_USAGE_DEFAULT;
  desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
  D3D11_SUBRESOURCE_DATA init{};init.pSysMem=initial;
  checked(device->CreateBuffer(&desc,&init,&s->data),"CreateBuffer");
  D3D11_SHADER_RESOURCE_VIEW_DESC sd{};sd.Format=DXGI_FORMAT_R32_TYPELESS;
  sd.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;sd.BufferEx.NumElements=extent.logicalWords;sd.BufferEx.Flags=D3D11_BUFFEREX_SRV_FLAG_RAW;
  checked(device->CreateShaderResourceView(s->data.Get(),&sd,&s->srv),"Create raw SRV");
  s->readonlyWeight=true;
  s->account=account;s->bytes=uint32_t(bytes);s->viewWords=extent.logicalWords;uint64_t total=account->bytes.fetch_add(bytes)+bytes;
  uint64_t peak=account->peak.load();while(peak<total&&!account->peak.compare_exchange_weak(peak,total)){}
  timed.created(bytes);
  return {s,count,count,false};
 }
 void valid(const Buffer& b) const {
  if(!b.storage||b.storage->account!=account||!b.words||b.words!=b.storage->viewWords||uint64_t(b.words)*4>b.storage->bytes)
   throw std::runtime_error("Invalid buffer or foreign D3D11 device");
  const uint64_t capacity=uint64_t(b.words)*(b.packedBF16?2:1);
  if(!b.logicalElements||b.logicalElements>capacity)throw std::runtime_error("Invalid logical buffer extent");
 }
 void writable(const Buffer& b) const {
  valid(b);if(!weight_srv_only::writable(b.storage->readonlyWeight,b.storage->uav.Get()!=nullptr))throw std::runtime_error("Read-only weight or missing UAV cannot be a write target");
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
 // Read once; a malformed opt-in fails before adapter selection, D3D11 device or model allocation.
 impl->waitMode=fence_wait::selectedByEnvironment();
 if(cpu_timing::enabledByEnvironment()){
  LARGE_INTEGER frequency{};impl->cpu=std::make_unique<cpu_timing::Recorder>(&steadyNanoseconds,QueryPerformanceFrequency(&frequency)?uint64_t(frequency.QuadPart):0);
 }
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
 if(impl->waitMode==fence_wait::Mode::fenceEvent){
  auto requireFence=[](HRESULT status,const char* action){
   if(status!=S_OK)throw std::runtime_error(std::string("CHANDRA_EXPERIMENTAL_DRAIN_WAIT=fence_event requires ")+action+"; HRESULT="+std::to_string(uint32_t(status))+"; experimental route refused, no fallback; use unset/query_sleep in a separately admitted process");
  };
  requireFence(impl->device.As(&impl->fenceDevice),"ID3D11Device5");
  requireFence(impl->context.As(&impl->fenceContext),"ID3D11DeviceContext4");
  if(impl->context->GetType()!=D3D11_DEVICE_CONTEXT_IMMEDIATE)throw std::runtime_error("Experimental fence requires immediate context; route refused without fallback");
  requireFence(impl->fenceDevice->CreateFence(0,D3D11_FENCE_FLAG_NONE,IID_PPV_ARGS(impl->fence.GetAddressOf())),"CreateFence(D3D11_FENCE_FLAG_NONE)");
  impl->fenceEvent.handle=CreateEventW(nullptr,TRUE,FALSE,nullptr); // Private, non-inheritable, initially nonsignaled.
  if(!impl->fenceEvent.handle)throw std::runtime_error("Experimental fence CreateEvent failed; Win32="+std::to_string(GetLastError())+"; route refused without fallback");
 }
 D3D11_BUFFER_DESC cb{};cb.ByteWidth=256;cb.Usage=D3D11_USAGE_DEFAULT;cb.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 checked(impl->device->CreateBuffer(&cb,nullptr,&impl->constants),"Create constant buffer");
}
Device::~Device(){
 // SetEventOnCompletion has no cancellation API. A failed experimental drain may
 // still own a driver registration and submitted buffers. Quarantine the complete
 // Impl (event, context, fence, constants and retained Storage) until process exit;
 // never recycle/close that handle or pretend that a timeout drained GPU work.
 // This exceptional one-Device retention requires external owned-worker retirement.
 if(impl&&impl->waitMode==fence_wait::Mode::fenceEvent&&(impl->fenceState.poisoned||impl->fenceState.registrationOutstanding||!impl->pending.empty()||impl->copyOwner.active()))(void)impl.release();
}
void Device::cancelExperimentalDrainWait() noexcept {impl->fenceCanceled.store(true,std::memory_order_release);}
Buffer Device::floats(uint32_t count,const float* values){return impl->make(count,values);}
Buffer Device::words(uint32_t count,const uint32_t* values){return impl->make(count,values);}
Buffer Device::srvOnlyWeightWords(uint32_t count,const void* initial){return impl->makeSrvOnlyWeight(count,initial);}
weight_srv_only::Descriptor Device::weightDescriptor(const Buffer& b) const {
 impl->thread();impl->valid(b);weight_srv_only::Descriptor result{};D3D11_BUFFER_DESC bd{};b.storage->data->GetDesc(&bd);
 result.byteWidth=bd.ByteWidth;result.usage=uint32_t(bd.Usage);result.bindFlags=bd.BindFlags;result.miscFlags=bd.MiscFlags;result.cpuAccess=bd.CPUAccessFlags;result.stride=bd.StructureByteStride;
 result.readonlyWeight=b.storage->readonlyWeight;result.srvPresent=b.storage->srv.Get()!=nullptr;result.uavPresent=b.storage->uav.Get()!=nullptr;
 if(result.srvPresent){D3D11_SHADER_RESOURCE_VIEW_DESC sd{};b.storage->srv->GetDesc(&sd);result.srvFormat=uint32_t(sd.Format);result.srvDimension=uint32_t(sd.ViewDimension);
  if(sd.ViewDimension==D3D11_SRV_DIMENSION_BUFFEREX){result.srvFirst=sd.BufferEx.FirstElement;result.srvWords=sd.BufferEx.NumElements;result.srvFlags=sd.BufferEx.Flags;}
  ComPtr<ID3D11Resource> resource;b.storage->srv->GetResource(&resource);result.srvSameResource=resource.Get()==b.storage->data.Get();}
 return result;
}
Buffer Device::paddedWeightWords(uint32_t count,uint32_t physicalBytes){
 if(physicalBytes!=weight_storage::allocationBytes(uint64_t(count)*4,WeightStorageExperiment::page4096))throw std::runtime_error("Padded weight extent must be the page4096 allocation");
 return impl->make(count,nullptr,physicalBytes);
}
void Device::upload(Buffer& b,const void* values,uint32_t bytes){
 impl->thread();impl->usable();cpu_timing::CallScope timed(impl->cpu.get(),&cpu_timing::Recorder::upload);
 impl->writable(b);const auto extent=weight_storage::geometry(b.words,b.storage->bytes);
 if(!values||bytes!=extent.logicalBytes)throw std::runtime_error("Complete bounded buffer upload required");
 impl->retain(b);
 if(extent.boxed()){const auto box=extent.prefix<D3D11_BOX>();impl->context->UpdateSubresource(b.storage->data.Get(),0,&box,values,0,0);}
 else impl->context->UpdateSubresource(b.storage->data.Get(),0,nullptr,values,0,0);
}
void Device::dispatch(const std::string& name,const std::vector<const Buffer*>& inputs,const std::vector<Buffer*>& outputs,
 const void* params,uint32_t paramBytes,uint32_t x,uint32_t y,uint32_t z){
 impl->thread();impl->usable();cpu_timing::DispatchScope timed(impl->cpu.get(),name);
 if(inputs.size()>16||outputs.empty()||outputs.size()>8||paramBytes>256||(paramBytes&&!params)||!x||!y||!z||x>65535||y>65535||z>65535)
  throw std::runtime_error("Invalid bounded compute dispatch");
 std::unordered_set<const Storage*> writable;
 for(auto p:outputs){if(!p)throw std::runtime_error("Null UAV");impl->writable(*p);if(!writable.insert(p->storage.get()).second)throw std::runtime_error("Aliased UAVs");}
 for(auto p:inputs){if(!p)throw std::runtime_error("Null SRV");impl->valid(*p);if(writable.count(p->storage.get()))throw std::runtime_error("SRV/UAV alias forbidden");}
 std::array<uint32_t,64> constants{};if(paramBytes)std::memcpy(constants.data(),params,paramBytes);
 ID3D11Buffer* cb=impl->constants.Get();impl->context->CSSetConstantBuffers(0,1,&cb);
 std::array<ID3D11ShaderResourceView*,16> srvs{};std::array<ID3D11UnorderedAccessView*,8> uavs{};
 for(size_t i=0;i<inputs.size();++i)srvs[i]=inputs[i]->storage->srv.Get();
 for(size_t i=0;i<outputs.size();++i)uavs[i]=outputs[i]->storage->uav.Get();
 const size_t cachedShaders=impl->shaders.size();timed.lap(cpu_timing::dispatchPrepare);
 auto shader=impl->shader(name);timed.lap(impl->shaders.size()!=cachedShaders?cpu_timing::dispatchShaderCompile:cpu_timing::dispatchShaderLookup);
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
 timed.lap(cpu_timing::dispatchRetain);
 impl->context->UpdateSubresource(impl->constants.Get(),0,nullptr,constants.data(),0,0);
 impl->context->CSSetShader(shader,nullptr,0);
 impl->context->CSSetShaderResources(0,16,srvs.data());impl->context->CSSetUnorderedAccessViews(0,8,uavs.data(),nullptr);
 if(impl->profiling)impl->context->End(impl->timings.back().first.Get());
 impl->context->Dispatch(x,y,z);
 if(impl->profiling)impl->context->End(impl->timings.back().last.Get());
 srvs.fill(nullptr);uavs.fill(nullptr);
 impl->context->CSSetShaderResources(0,16,srvs.data());impl->context->CSSetUnorderedAccessViews(0,8,uavs.data(),nullptr);
 timed.recorded(impl->profiling);
}
void Device::drain(uint32_t timeout){
 impl->thread();cpu_timing::DrainScope timed(impl->cpu.get());if(!timeout||timeout>30000)throw std::runtime_error("Bounded drain timeout required");
 if(impl->waitMode==fence_wait::Mode::fenceEvent){
  impl->usable();
  struct Driver {
   Impl& p;uint32_t lastError=0;
   fence_wait::Nanoseconds now() const noexcept{return steadyNanoseconds();}
   bool canceled() const noexcept{return p.fenceCanceled.load(std::memory_order_acquire);}
   int32_t removed() const noexcept{return int32_t(p.device->GetDeviceRemovedReason());}
   bool resetEvent() noexcept {if(ResetEvent(p.fenceEvent.handle))return true;lastError=GetLastError();return false;}
   uint32_t error() const noexcept{return lastError;}
   int32_t signal(uint64_t value) noexcept{return int32_t(p.fenceContext->Signal(p.fence.Get(),value));}
   void flush() noexcept {p.context->Flush();}
   uint64_t completed() const noexcept{return p.fence->GetCompletedValue();}
   int32_t registerEvent(uint64_t value) noexcept{return int32_t(p.fence->SetEventOnCompletion(value,p.fenceEvent.handle));}
   uint32_t wait(uint32_t milliseconds) noexcept {
    const DWORD result=WaitForSingleObject(p.fenceEvent.handle,milliseconds);if(result==WAIT_FAILED)lastError=GetLastError();return result;
   }
  } driver{*impl,0};
  auto& stats=impl->fenceStats;stats.calls.add(1);
  const auto outcome=fence_wait::drain(impl->fenceState,timeout,driver,[&](fence_wait::Phase phase,bool ready,bool failed){
   switch(phase){
   case fence_wait::Phase::prepare:timed.lap(cpu_timing::drainFencePrepare);break;
   case fence_wait::Phase::signalFlush:if(failed)timed.lap(cpu_timing::drainEndFlush);else timed.flushed();break;
   case fence_wait::Phase::registration:timed.lap(cpu_timing::drainFenceRegistration);break;
   case fence_wait::Phase::check:timed.fenceChecked(ready,failed);break;
   case fence_wait::Phase::deadline:timed.checkedDeadline();break;
   case fence_wait::Phase::wait:timed.lap(cpu_timing::drainEventWait);break;
   }
  });
  stats.last=outcome;stats.waitCalls.add(outcome.waits);stats.checks.add(outcome.checks);stats.signals.add(uint64_t(outcome.signaledSubmitted));
  stats.registrations.add(uint64_t(outcome.eventRegistered));stats.requestedMilliseconds.add(outcome.requestedWaitMilliseconds);
  stats.waits.add(outcome.waited);stats.elapsed.add(outcome.elapsed);
  if(outcome.result!=fence_wait::Result::completed){
   stats.failures.add(1);
   throw std::runtime_error(std::string("Experimental fence drain ")+fence_wait::name(outcome.result)+"; target="+std::to_string(outcome.target)+" completed="+std::to_string(outcome.completedValue)+" HRESULT="+std::to_string(uint32_t(outcome.status))+" Win32="+std::to_string(outcome.error)+" Wait="+std::to_string(outcome.lastWait)+"; pending Storage and registered event retained until owned-process retirement; no replay");
  }
  if(!impl->copyOwner.completed(outcome)){
   impl->fenceState.poisoned=true;stats.failures.add(1);
   throw std::runtime_error("Experimental staging-copy ownership fence mismatch; source and staging retained until owned-process retirement");
  }
  stats.successes.add(1);timed.releasing(impl->pending.size());impl->pending.clear();impl->pendingIds.clear();timed.released();return;
 }
 D3D11_QUERY_DESC desc{D3D11_QUERY_EVENT,0};ComPtr<ID3D11Query> q;
 checked(impl->device->CreateQuery(&desc,&q),"Create drain query");timed.lap(cpu_timing::drainQueryCreate);impl->context->End(q.Get());impl->context->Flush();timed.flushed();
 auto start=std::chrono::steady_clock::now();
 for(;;){BOOL complete=FALSE;HRESULT hr=impl->context->GetData(q.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);timed.polled(hr==S_OK&&complete,FAILED(hr));
  if(hr==S_OK&&complete){timed.releasing(impl->pending.size());impl->pending.clear();impl->pendingIds.clear();timed.released();return;}checked(hr,"D3D11 drain query");
  if(std::chrono::steady_clock::now()-start>std::chrono::milliseconds(timeout))throw std::runtime_error("D3D11 drain deadline exceeded; retire owned worker");
  timed.checkedDeadline();std::this_thread::sleep_for(std::chrono::milliseconds(1));timed.slept(std::chrono::milliseconds(1));
 }
}
std::vector<uint32_t> Device::readWords(const Buffer& b){
 impl->thread();cpu_timing::ReadbackScope timed(impl->cpu.get());impl->valid(b);drain();timed.lap(cpu_timing::readbackBeforeCopyDrain);
 const auto extent=weight_storage::geometry(b.words,b.storage->bytes);
 D3D11_BUFFER_DESC desc{};b.storage->data->GetDesc(&desc);
 if(desc.ByteWidth!=extent.physicalBytes)throw std::runtime_error("Readback resource extent differs from tracked storage");
 desc.ByteWidth=extent.logicalBytes; // Staging contains only the exact logical prefix.
 if(!weight_storage::withinBudget(impl->account->bytes.load(),desc.ByteWidth,13ull*1024*1024*1024))throw std::runtime_error("Readback staging exceeds tracked allocation cap");
 auto memory=impl->memory();
 if(memory.Budget<=memory.CurrentUsage||memory.Budget-memory.CurrentUsage<desc.ByteWidth+2ull*1024*1024*1024)
  throw std::runtime_error("Readback staging would breach 2 GiB local headroom");
 desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.MiscFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Buffer> staging;checked(impl->device->CreateBuffer(&desc,nullptr,&staging),"Create staging");
 std::vector<uint32_t> result(b.words);
 const auto now=[]{return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch());};
 timed.lap(cpu_timing::readbackStagingPrepare);
 if(impl->waitMode==fence_wait::Mode::fenceEvent){
  impl->copyOwner.hold(b.storage,staging,extent.logicalBytes,impl->account->bytes.load(),impl->fenceState.lastIssued);
  impl->copyOwner.beforeEnqueue(); // Both source and staging survive every subsequent unwind.
 }
 if(extent.boxed()){const auto box=extent.prefix<D3D11_BOX>();impl->context->CopySubresourceRegion(staging.Get(),0,0,0,0,b.storage->data.Get(),0,&box);}
 else impl->context->CopyResource(staging.Get(),b.storage->data.Get());
 const auto deadline=now()+readback::postCopyBudget;timed.copied();drain();timed.lap(cpu_timing::readbackAfterCopyDrain);
 // Map stays nonblocking. Only WAS_STILL_DRAWING polls this same staging copy again, until the one post-copy deadline.
 const auto outcome=readback::copyWhenReady(deadline,
  [&](void** data){D3D11_MAPPED_SUBRESOURCE mapped{};const HRESULT hr=impl->context->Map(staging.Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);*data=mapped.pData;return int32_t(hr);},
  [&]{impl->context->Unmap(staging.Get(),0);},
  [&](const void* data){std::memcpy(result.data(),data,extent.logicalBytes);},
  now,[cpu=impl->cpu.get()](std::chrono::nanoseconds pause){const cpu_timing::ReadinessSleep timedSleep(cpu,pause);std::this_thread::sleep_for(pause);});
 timed.lap(cpu_timing::readbackMapReady);
 auto& maps=impl->readbackMaps;++maps.waits;maps.retried+=uint64_t(outcome.stillDrawing>0);maps.stillDrawing+=outcome.stillDrawing;maps.longest=std::max(maps.longest,outcome.waited);
 if(outcome.result!=readback::Result::copied)
  throw std::runtime_error(readback::describe(outcome)+"; device removed reason HRESULT="+std::to_string(uint32_t(impl->device->GetDeviceRemovedReason())));
 return result;
}
std::vector<uint32_t> Device::readWordsRange(const Buffer& b,uint32_t firstWord,uint32_t count,direct_head_row::Receipt& receipt){
 receipt={};const auto start=steadyNanoseconds();ComPtr<ID3D11Buffer> staging;
 auto release=[&]{
  if(impl->waitMode==fence_wait::Mode::fenceEvent){
   const bool retained=impl->copyOwner.owns(staging.Get());staging.Reset();receipt.stagingReleased=receipt.stagingCreated&&!retained;
   receipt.operationNanoseconds=(steadyNanoseconds()-start).count();return;
  }
  staging.Reset();receipt.stagingReleased=receipt.stagingCreated;receipt.operationNanoseconds=(steadyNanoseconds()-start).count();
 };
 try{
  impl->thread();impl->valid(b);receipt.extent=direct_head_row::range(b.words,b.storage->bytes,firstWord,count);
  D3D11_BUFFER_DESC source{};b.storage->data->GetDesc(&source);
  receipt.sourceUsage=source.Usage;receipt.sourceBindFlags=source.BindFlags;receipt.sourceMiscFlags=source.MiscFlags;
  receipt.sourceCPUAccess=source.CPUAccessFlags;receipt.sourceStride=source.StructureByteStride;
  if(source.ByteWidth!=receipt.extent.physicalBytes||source.Usage!=D3D11_USAGE_DEFAULT||source.StructureByteStride!=0)
   throw std::runtime_error("Direct row source resource descriptor differs from tracked raw storage");
  drain();receipt.beforeCopyDrainCompleted=true;
  if(!weight_storage::withinBudget(impl->account->bytes.load(),receipt.extent.bytes,13ull*1024*1024*1024))throw std::runtime_error("Direct row staging exceeds tracked allocation cap");
  const auto memory=impl->memory();
  if(memory.Budget<=memory.CurrentUsage||memory.Budget-memory.CurrentUsage<receipt.extent.bytes+2ull*1024*1024*1024)
   throw std::runtime_error("Direct row staging would breach 2 GiB local headroom");
  D3D11_BUFFER_DESC desc{};desc.ByteWidth=receipt.extent.bytes;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  checked(impl->device->CreateBuffer(&desc,nullptr,&staging),"Create direct row staging");receipt.stagingCreated=true;
  D3D11_BUFFER_DESC actual{};staging->GetDesc(&actual);
  if(actual.ByteWidth!=desc.ByteWidth||actual.Usage!=desc.Usage||actual.CPUAccessFlags!=desc.CPUAccessFlags||actual.BindFlags||actual.MiscFlags||actual.StructureByteStride)
   throw std::runtime_error("Direct row staging descriptor differs from exact readback extent");
  std::vector<uint32_t> result(count);const auto box=receipt.extent.box<D3D11_BOX>();
  if(impl->waitMode==fence_wait::Mode::fenceEvent){
   impl->copyOwner.hold(b.storage,staging,receipt.extent.bytes,impl->account->bytes.load(),impl->fenceState.lastIssued);
  }
  impl->retain(b); // Keep the source alive until the completed event, also on a failed drain.
  if(impl->waitMode==fence_wait::Mode::fenceEvent)impl->copyOwner.beforeEnqueue();
  impl->context->CopySubresourceRegion(staging.Get(),0,0,0,0,b.storage->data.Get(),0,&box);++receipt.copiesSubmitted;
  const auto deadline=steadyNanoseconds()+readback::postCopyBudget;
  drain();receipt.afterCopyDrainCompleted=true;receipt.mapAttempted=true;
  receipt.readiness=readback::copyWhenReady(deadline,
   [&](void** data){D3D11_MAPPED_SUBRESOURCE mapped{};const HRESULT hr=impl->context->Map(staging.Get(),0,D3D11_MAP_READ,D3D11_MAP_FLAG_DO_NOT_WAIT,&mapped);*data=mapped.pData;return int32_t(hr);},
   [&]{impl->context->Unmap(staging.Get(),0);receipt.unmapped=true;},
   [&](const void* data){std::memcpy(result.data(),data,receipt.extent.bytes);},
   &steadyNanoseconds,[](std::chrono::nanoseconds pause){std::this_thread::sleep_for(pause);});
  if(receipt.readiness.result!=readback::Result::copied)throw std::runtime_error(readback::describe(receipt.readiness));
  release();return result;
 }catch(...){receipt.deviceRemovedReason=int32_t(impl->device->GetDeviceRemovedReason());receipt.deviceRemovedReasonQueried=true;release();throw;}
}
std::vector<float> Device::readFloats(const Buffer& b){
 if(b.packedBF16)throw std::runtime_error("Packed BF16 is not float storage");auto words=readWords(b);
 std::vector<float> result(words.size());std::memcpy(result.data(),words.data(),words.size()*4);return result;
}
void Device::zero(Buffer& b){impl->thread();impl->usable();cpu_timing::CallScope timed(impl->cpu.get(),&cpu_timing::Recorder::zero);impl->writable(b);const UINT zeros[4]{};impl->retain(b);impl->context->ClearUnorderedAccessViewUint(b.storage->uav.Get(),zeros);}
uint64_t Device::trackedBufferBytes()const{return impl->account->bytes.load();}
std::string Device::identityJson()const{return impl->identity;}
std::string Device::memoryJson()const{
 impl->thread();auto info=impl->memory();
 const auto& f=impl->fenceStats;
 return chandra::Json{{"local_budget",info.Budget},{"local_usage",info.CurrentUsage},
  {"tracked_live",impl->account->bytes.load()},{"tracked_peak",impl->account->peak.load()},
  {"required_headroom",2ull*1024*1024*1024},
  {"drain_wait",{{"schema","chandra.directcompute.experimental_drain_wait.v1"},{"environment_variable",fence_wait::variable},
   {"route",impl->waitMode==fence_wait::Mode::fenceEvent?"fence_event":"query_sleep"},
   {"support",impl->waitMode==fence_wait::Mode::fenceEvent?"Device5_Context4_fence_created":"not_queried"},
   {"maximum_block_slice_milliseconds",fence_wait::sliceMilliseconds},{"poisoned",impl->fenceState.poisoned},
   {"retirement_required",impl->fenceState.poisoned},{"registered_event_outstanding",impl->fenceState.registrationOutstanding},
   {"last_issued_value",impl->fenceState.lastIssued},{"pending_storage_references",impl->pending.size()},
   {"pending_copy_owner",{{"active",impl->copyOwner.active()},{"submitted",impl->copyOwner.submitted()},
    {"staging_bytes",impl->copyOwner.bytes()},{"peak_staging_bytes",impl->copyOwner.peakBytes()},
    {"expected_fence_value",impl->copyOwner.expectedFence()},{"maximum_staging_bytes",weight_storage::maximumBufferBytes}}},
   {"calls",f.calls.json()},{"completed_calls",f.successes.json()},{"failed_calls",f.failures.json()},
   {"signals",f.signals.json()},{"event_registrations",f.registrations.json()},{"completion_checks",f.checks.json()},
   {"blocking_wait_calls",f.waitCalls.json()},{"requested_wait_milliseconds",f.requestedMilliseconds.json()},
   {"actual_wait",f.waits.json()},{"inclusive_drain",f.elapsed.json()},
   {"last_result",f.calls.value?chandra::Json(fence_wait::name(f.last.result)):chandra::Json(nullptr)},
   {"boundary","Host wall time and requested blocking durations; not GPU-active time, VRAM, CPU utilization or page throughput"}}},
  {"readback_map",{{"waits",impl->readbackMaps.waits},{"waits_with_still_drawing",impl->readbackMaps.retried},
   {"still_drawing_results",impl->readbackMaps.stillDrawing},{"longest_wait_nanoseconds",impl->readbackMaps.longest.count()}}},
  {"cpu_timing",impl->cpu?impl->cpu->json():cpu_timing::disabledJson()}}.dump();
}
void Device::beginProfile(){
 impl->thread();if(impl->profiling||!impl->timings.empty())throw std::runtime_error("Profile window already open or uncollected");
 drain();D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP_DISJOINT,0};
 checked(impl->device->CreateQuery(&desc,&impl->disjoint),"Create disjoint query");
 impl->context->Begin(impl->disjoint.Get());impl->profiling=true;
}
std::string Device::finishProfile(){
 impl->thread();if(!impl->profiling)throw std::runtime_error("No profile window");
 if(impl->waitMode==fence_wait::Mode::fenceEvent)impl->usable(); // Refuse before End on a poisoned experimental context.
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
