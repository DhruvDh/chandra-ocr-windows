// New code, MPL-2.0. CPU-only tests of the readback readiness control in
// readback_wait.h. Fakes script Map statuses, a monotonic clock advanced only by
// sleeps/Map cost, and map/unmap/consume lifetime. They do not exercise D3D11,
// a driver, real staging readiness or the GPU; device.cpp is not compiled here.
#include "readback_wait.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
using namespace chandra::dc::readback;
using std::chrono::milliseconds;
using std::chrono::microseconds;
namespace {
unsigned passed=0;
void require(bool condition,const char* reason){if(!condition)throw std::runtime_error(reason);}
constexpr int32_t hresult(uint32_t raw){return int32_t(raw);}
constexpr int32_t sOk=okStatus,stillDrawing=stillDrawingStatus,sFalse=1,dxgiStatusOccluded=hresult(0x087A0001u);
constexpr int32_t deviceRemoved=hresult(0x887A0005u),deviceHung=hresult(0x887A0006u),deviceReset=hresult(0x887A0007u),
 invalidCall=hresult(0x887A0001u),waitTimeout=hresult(0x887A0027u),outOfMemory=hresult(0x8007000Eu),invalidArg=hresult(0x80070057u),genericFail=hresult(0x80004005u);
// One fake staging resource. The status script's last entry repeats. Violations
// count Map while mapped, Unmap/consume without a mapping, and sleeps that are
// nonpositive, above the backoff cap or taken while mapped.
struct Fake {
 std::vector<int32_t> script;size_t next=0;bool nullOnSuccess=false,throwInConsume=false,mapped=false;
 Nanoseconds clock{milliseconds(5000)},oversleep{0},mapCost{0};
 std::vector<Nanoseconds> sleeps,mapTimes;
 uint32_t unmaps=0,consumed=0,violations=0;
 uint32_t source[4]{0x3F800000u,0xFF7FFFFFu,0x00000001u,0x7FC00000u},destination[4]{};
 explicit Fake(std::vector<int32_t> s):script(std::move(s)){}
 int32_t map(void** data){
  require(mapTimes.size()<100000,"fake Map limit reached; control polled without bound");
  if(mapped)++violations;
  mapTimes.push_back(clock);clock+=mapCost;
  const int32_t status=script.at(std::min(next++,script.size()-1));
  if(status>=0){mapped=true;*data=nullOnSuccess?nullptr:static_cast<void*>(source);}
  return status;
 }
 void unmap(){if(!mapped)++violations;mapped=false;++unmaps;}
 void consume(const void* data){
  if(!mapped||data!=source)++violations;
  std::memcpy(destination,data,sizeof(destination));++consumed;
  if(throwInConsume)throw std::runtime_error("consumer failure");
 }
 void sleep(Nanoseconds d){if(d<=Nanoseconds{0}||d>maximumDelay||mapped)++violations;sleeps.push_back(d);clock+=d+oversleep;}
 Nanoseconds slept() const {Nanoseconds total{0};for(auto d:sleeps)total+=d;return total;}
 bool copiedExactly() const {return std::memcmp(source,destination,sizeof(source))==0;}
 bool untouched() const {for(auto w:destination)if(w)return false;return true;}
 bool clean() const {return !mapped&&!violations;}
};
Outcome run(Fake& f,Nanoseconds deadline){
 return copyWhenReady(deadline,[&](void** data){return f.map(data);},[&]{f.unmap();},[&](const void* data){f.consume(data);},
  [&]{return f.clock;},[&](Nanoseconds d){f.sleep(d);});
}
bool contains(const std::string& text,const char* part){return text.find(part)!=std::string::npos;}
}
int main(){
 try {
  require(uint32_t(stillDrawingStatus)==2289696778u&&okStatus==0,"DXGI_ERROR_WAS_STILL_DRAWING/S_OK constants differ");
  require(postCopyBudget==milliseconds(10000)&&firstDelay==milliseconds(1)&&maximumDelay==milliseconds(8),"readiness policy constants differ");++passed;

  {Fake meta({sOk});meta.unmap();void* data=nullptr;meta.map(&data);meta.map(&data);meta.sleep(Nanoseconds{0});
   require(meta.violations==3,"fake lifetime detector is not sensitive");++passed;}

  {Fake f({sOk});const auto deadline=f.clock+postCopyBudget;const auto o=run(f,deadline);
   require(o.result==Result::copied&&o.status==sOk&&o.attempts==1&&o.stillDrawing==0&&o.waited==Nanoseconds{0},"immediate success outcome differs");
   require(f.mapTimes.size()==1&&f.unmaps==1&&f.consumed==1&&f.sleeps.empty()&&f.clean()&&f.copiedExactly(),"immediate success lifetime/bytes differ");++passed;}

  {Fake f({stillDrawing,stillDrawing,stillDrawing,sOk});const auto start=f.clock;const auto o=run(f,start+postCopyBudget);
   require(o.result==Result::copied&&o.status==sOk&&o.attempts==4&&o.stillDrawing==3&&o.waited==milliseconds(7),"transient-then-success outcome differs");
   require(f.sleeps==std::vector<Nanoseconds>{milliseconds(1),milliseconds(2),milliseconds(4)},"transient backoff sequence differs");
   require(f.mapTimes==std::vector<Nanoseconds>{start,start+milliseconds(1),start+milliseconds(3),start+milliseconds(7)},"Map attempt clock differs");
   require(f.unmaps==1&&f.consumed==1&&f.clean()&&f.copiedExactly(),"transient success lifetime/bytes differ");++passed;}

  // Persistent transient: 1+2+4 ms, 1,249 capped 8 ms sleeps, then one sleep clipped to 1 ms ends exactly at the deadline.
  {Fake f({stillDrawing});const auto start=f.clock,deadline=start+postCopyBudget;const auto o=run(f,deadline);
   require(o.result==Result::stillDrawingAtDeadline&&o.status==stillDrawing&&o.attempts==1254&&o.stillDrawing==1254&&o.waited==postCopyBudget,"persistent transient outcome differs");
   require(f.sleeps.size()==1253&&f.slept()==postCopyBudget&&f.clock==deadline&&f.mapTimes.back()==deadline&&f.mapTimes.size()==o.attempts,"persistent deadline accounting differs");
   require(f.sleeps[0]==milliseconds(1)&&f.sleeps[1]==milliseconds(2)&&f.sleeps[2]==milliseconds(4)&&f.sleeps[3]==milliseconds(8)&&f.sleeps[1251]==milliseconds(8)&&f.sleeps[1252]==milliseconds(1),"persistent backoff/clipping differs");
   require(f.unmaps==0&&f.consumed==0&&f.clean()&&f.untouched(),"timed-out readback touched a mapping");++passed;}

  // OS oversleep and Map cost may not reset or extend the deadline: overshoot is at most one oversleep plus one Map.
  {Fake f({stillDrawing});f.oversleep=milliseconds(3);f.mapCost=microseconds(250);const auto start=f.clock,deadline=start+postCopyBudget;const auto o=run(f,deadline);
   require(o.result==Result::stillDrawingAtDeadline&&o.waited==f.clock-start&&f.clock>=deadline&&f.clock-deadline<=f.oversleep+f.mapCost,"oversleep crossed the absolute deadline");
   require(f.slept()<postCopyBudget&&o.attempts==f.mapTimes.size()&&o.attempts==f.sleeps.size()+1&&o.attempts<1254&&f.clean(),"oversleep attempt accounting differs");++passed;}

  // The deadline is the caller's copy-time anchor: a drain that used 9,995 ms leaves exactly 5 ms.
  {Fake f({stillDrawing});const auto anchor=f.clock;f.clock+=milliseconds(9995);const auto o=run(f,anchor+postCopyBudget);
   require(o.result==Result::stillDrawingAtDeadline&&o.attempts==4&&o.waited==milliseconds(5),"anchored remaining window differs");
   require(f.sleeps==std::vector<Nanoseconds>{milliseconds(1),milliseconds(2),milliseconds(2)}&&f.clean(),"anchored final sleep was not clipped");++passed;}
  {Fake f({stillDrawing,stillDrawing,sOk});const auto anchor=f.clock;f.clock+=milliseconds(9995);const auto o=run(f,anchor+postCopyBudget);
   require(o.result==Result::copied&&o.attempts==3&&o.waited==milliseconds(3)&&f.unmaps==1&&f.clean()&&f.copiedExactly(),"success inside remaining window differs");++passed;}

  // An expired deadline still allows the original single Map, then refuses without sleeping.
  {Fake f({stillDrawing});const auto deadline=f.clock-milliseconds(1);const auto o=run(f,deadline);
   require(o.result==Result::stillDrawingAtDeadline&&o.attempts==1&&f.sleeps.empty()&&f.unmaps==0&&f.clean(),"expired deadline retried");++passed;}
  {Fake f({sOk});const auto o=run(f,f.clock-milliseconds(1));
   require(o.result==Result::copied&&o.attempts==1&&f.unmaps==1&&f.clean()&&f.copiedExactly(),"expired deadline refused immediate success");++passed;}

  for(int32_t error:{deviceRemoved,deviceHung,deviceReset,invalidCall,waitTimeout,outOfMemory,invalidArg,genericFail,hresult(0x887A0009u),hresult(0x887A000Bu)}){
   Fake f({error,sOk});const auto o=run(f,f.clock+postCopyBudget);
   require(o.result==Result::mapFailed&&o.status==error&&o.attempts==1&&o.stillDrawing==0&&o.waited==Nanoseconds{0},"nonretry HRESULT was retried or altered");
   require(f.sleeps.empty()&&f.unmaps==0&&f.consumed==0&&f.clean()&&f.untouched(),"failed Map was unmapped or consumed");++passed;
  }
  {Fake f({stillDrawing,stillDrawing,deviceRemoved,sOk});const auto o=run(f,f.clock+postCopyBudget);
   require(o.result==Result::mapFailed&&o.status==deviceRemoved&&o.attempts==3&&o.stillDrawing==2&&o.waited==milliseconds(3),"removal after transient was retried");
   require(f.sleeps==std::vector<Nanoseconds>{milliseconds(1),milliseconds(2)}&&f.unmaps==0&&f.consumed==0&&f.clean(),"removal after transient touched a mapping");++passed;}

  for(const auto& script:{std::vector<int32_t>{sFalse},std::vector<int32_t>{dxgiStatusOccluded},std::vector<int32_t>{stillDrawing,sFalse}}){
   Fake f(script);const auto o=run(f,f.clock+postCopyBudget);
   require(o.result==Result::unexpectedSuccess&&o.status==script.back()&&o.attempts==script.size(),"unexpected success accepted");
   require(f.unmaps==1&&f.consumed==0&&f.clean()&&f.untouched(),"unexpected success mapping not released exactly once");++passed;
  }
  {Fake f({sOk});f.nullOnSuccess=true;const auto o=run(f,f.clock+postCopyBudget);
   require(o.result==Result::nullData&&o.attempts==1&&f.unmaps==1&&f.consumed==0&&f.clean()&&f.untouched(),"null mapping not rejected and released");++passed;}

  {Fake f({stillDrawing,sOk});f.throwInConsume=true;bool caught=false;
   try{run(f,f.clock+postCopyBudget);}catch(const std::runtime_error& e){caught=std::string(e.what())=="consumer failure";}
   require(caught&&f.consumed==1&&f.unmaps==1&&f.clean(),"throwing consumer leaked or double-released the mapping");++passed;}

  {Fake f({stillDrawing});const auto text=describe(run(f,f.clock+postCopyBudget));
   require(contains(text,"still drawing HRESULT=2289696778 (0x887A000A)")&&contains(text,"10000 ms post-copy readiness deadline")&&contains(text,"after 1254 nonblocking attempts over 10000.000 ms")&&contains(text,"not resubmitted"),"timeout diagnostic differs");
   Fake g({deviceRemoved});const auto removed=describe(run(g,g.clock+postCopyBudget));
   require(removed.rfind("Map completed readback HRESULT=2289696773 (0x887A0005) on attempt 1",0)==0&&contains(removed,"not retried"),"nonretry diagnostic differs");
   Fake h({sFalse});require(contains(describe(run(h,h.clock+postCopyBudget)),"unexpected success HRESULT=1 (0x00000001)"),"unexpected success diagnostic differs");++passed;}

  std::cout<<"{\"test\":\"directcompute_readback_wait_control\",\"checks_passed\":"<<passed
   <<",\"cpu_fakes_only\":true,\"d3d11_driver_or_gpu_exercised\":false}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAILED after "<<passed<<" checks: "<<e.what()<<"\n";return 1;}
}
