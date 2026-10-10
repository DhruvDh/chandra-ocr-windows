// New code, MPL-2.0. CPU-only tests of the opt-in timing accounting in
// cpu_timing.h. A fake monotonic clock advances only when a test says so, and a
// counting global operator new detects allocation. The helpers below call the
// same scopes as device.cpp in the same lap order, but device.cpp is not compiled
// here: D3D11, a driver, QueryPerformanceCounter and the GPU are not exercised.
// Runners: cpu_timing_test.sh (GCC/Clang); MSVC /W4 /WX in docs/directcompute-cpu-timing.md.
#include "cpu_timing.h"
#include <iostream>
#include <new>
#include <stdlib.h>
#include <utility>
using namespace chandra::dc::cpu_timing;
using std::chrono::milliseconds;
// Replacement global allocation counts every operator new and can fail the next
// one. malloc/free stay behind non-inlined wrappers so GCC's new/delete pairing
// analysis sees matching replacement functions. MSVC spells the attribute
// __declspec(noinline); it rejects [[gnu::noinline]] as unknown (C5030).
#ifdef _MSC_VER
#define CPU_TIMING_TEST_NOINLINE __declspec(noinline)
#else
#define CPU_TIMING_TEST_NOINLINE [[gnu::noinline]]
#endif
namespace {
size_t allocations=0;bool failNextAllocation=false;
CPU_TIMING_TEST_NOINLINE void* obtain(std::size_t bytes){
 if(failNextAllocation){failNextAllocation=false;throw std::bad_alloc();}
 ++allocations;if(void* p=std::malloc(bytes?bytes:1))return p;throw std::bad_alloc();
}
CPU_TIMING_TEST_NOINLINE void release(void* p) noexcept {std::free(p);}
}
void* operator new(std::size_t bytes){return obtain(bytes);}
void* operator new[](std::size_t bytes){return obtain(bytes);}
void operator delete(void* p) noexcept {release(p);}
void operator delete[](void* p) noexcept {release(p);}
void operator delete(void* p,std::size_t) noexcept {release(p);}
void operator delete[](void* p,std::size_t) noexcept {release(p);}
namespace {
unsigned passed=0;
void require(bool condition,const char* reason){if(!condition)throw std::runtime_error(reason);}
bool contains(const std::string& text,const std::string& part){return text.find(part)!=std::string::npos;}
// The live process environment, read and written on the branch cpu_timing.h
// compiles (the same _MSC_VER test), so enabledByEnvironment runs its own
// _dupenv_s or getenv path. Values are compared and measured, never printed. POSIX
// setenv stores an empty value. No Windows CRT call does: _putenv_s with an empty
// value removes the variable, so that is the only empty encoding checked there.
struct Setting {
 bool present=false;std::string value;
 bool operator==(const Setting& other) const {return present==other.present&&value==other.value;}
};
#ifdef _MSC_VER
constexpr const char* environmentRoute="msvc_dupenv_s";
constexpr const char* emptyEnvironmentValue="not_encodable_putenv_s_removes";
Setting environmentSetting(){
 struct Free {void operator()(char* p) const noexcept {std::free(p);}};
 char* value=nullptr;size_t bytes=0;
 require(_dupenv_s(&value,&bytes,variable)==0,"_dupenv_s failed");
 const std::unique_ptr<char,Free> owned(value);Setting s;
 if(value){s.present=true;s.value=value;}
 return s;
}
void setEnvironment(const char* value){require(value[0]!='\0'&&_putenv_s(variable,value)==0,"_putenv_s could not set a nonempty value");}
// Removes only a present variable, so no result depends on removing an absent one.
void removeEnvironment(){if(environmentSetting().present)require(_putenv_s(variable,"")==0,"_putenv_s could not remove the variable");}
#else
constexpr const char* environmentRoute="posix_getenv";
constexpr const char* emptyEnvironmentValue="rejected";
Setting environmentSetting(){Setting s;if(const char* value=std::getenv(variable)){s.present=true;s.value=value;}return s;}
void setEnvironment(const char* value){require(setenv(variable,value,1)==0,"setenv failed");}
void removeEnvironment(){require(unsetenv(variable)==0,"unsetenv failed");}
#endif
bool environmentRejects(size_t bytes){
 try{enabledByEnvironment();}catch(const std::invalid_argument& e){return contains(e.what(),"rejected a "+std::to_string(bytes)+"-byte value");}
 return false;
}
// enabledByEnvironment returns, or rejects with the same length, exactly as the
// pure enabledBy does for this setting.
bool helperAgrees(const Setting& s){
 bool expected=false,rejects=false;
 try{expected=enabledBy(s.present?s.value.c_str():nullptr);}catch(const std::invalid_argument&){rejects=true;}
 return rejects?environmentRejects(s.value.size()):enabledByEnvironment()==expected;
}
Nanoseconds fakeTime{std::chrono::seconds(1)};uint64_t fakeReads=0;
Nanoseconds fakeNow() noexcept {++fakeReads;return fakeTime;}
void advance(int64_t ns){fakeTime+=Nanoseconds(ns);}
struct Thrown {};
bool same(const Accumulator& a,uint64_t count,uint64_t total){return !a.unavailable&&a.count==count&&a.total==total;}
// Exit lap count equals call count and the laps sum exactly to the inclusive total.
template<size_t N> bool partitioned(const Operation<N>& op){
 uint64_t sum=0;for(const auto& p:op.phases){if(p.unavailable)return false;sum+=p.total;}
 return !op.calls.unavailable&&sum==op.calls.total&&op.phases[N-1].count==op.calls.count;
}
// Device::dispatch lap order: prepare, lookup or compile, retain, record; then locals are destroyed.
void dispatch(Recorder* r,const std::string& name,int64_t prepare,int64_t resolve,bool compile,int64_t retain,int64_t record,int64_t exit,bool gpuTimestamps=false){
 DispatchScope timed(r,name);
 advance(prepare);timed.lap(dispatchPrepare);
 advance(resolve);timed.lap(compile?dispatchShaderCompile:dispatchShaderLookup);
 advance(retain);timed.lap(dispatchRetain);
 advance(record);timed.recorded(gpuTimestamps);
 advance(exit);
}
// Device::drain control flow over a scripted GetData sequence; each sleep requests 1 ms.
enum Poll {notReady,ready,failedStatus};
struct DrainScript {std::vector<Poll> polls;int64_t create=0,flush=0,getData=0,deadline=0,sleep=0,release=0,exit=0;size_t retained=0;bool expire=false;};
void drain(Recorder* r,const DrainScript& s){
 DrainScope timed(r);
 advance(s.create);timed.lap(drainQueryCreate);
 advance(s.flush);timed.flushed();
 for(size_t i=0;;++i){
  const Poll p=s.polls.at(i);advance(s.getData);timed.polled(p==ready,p==failedStatus);
  if(p==ready){timed.releasing(s.retained);advance(s.release);timed.released();advance(s.exit);return;}
  if(p==failedStatus){advance(s.exit);throw Thrown{};}
  advance(s.deadline);
  if(s.expire&&i+1==s.polls.size()){advance(s.exit);throw Thrown{};}
  timed.checkedDeadline();advance(s.sleep);timed.slept(milliseconds(1));
 }
}
// Device::readWords: validation and drain, staging, copy, drain, Map readiness, exit.
struct ReadbackScript {DrainScript before,after;int64_t valid=0,staging=0,copy=0,gap=0,map=0,exit=0;std::vector<std::pair<int64_t,int64_t>> sleeps;bool failAfterCopy=false;};
void readback(Recorder* r,const ReadbackScript& s){
 ReadbackScope timed(r);
 advance(s.valid);drain(r,s.before);timed.lap(readbackBeforeCopyDrain);
 advance(s.staging);timed.lap(readbackStagingPrepare);
 advance(s.copy);timed.copied();
 if(s.failAfterCopy){advance(s.exit);throw Thrown{};}
 drain(r,s.after);advance(s.gap);timed.lap(readbackAfterCopyDrain);
 for(const auto& pause:s.sleeps){const ReadinessSleep timedSleep(r,Nanoseconds(pause.first));advance(pause.second);}
 advance(s.map);timed.lap(readbackMapReady);
 advance(s.exit);
}
void buffer(Recorder* r,int64_t budget,int64_t create,int64_t exit,uint64_t bytes){
 BufferScope timed(r);advance(budget);timed.lap(bufferBudgetQuery);advance(create);timed.created(bytes);advance(exit);
}
void call(Recorder* r,Operation<1> Recorder::* member,int64_t ns){CallScope timed(r,member);advance(ns);}
std::vector<std::string> keys(const Json& object){std::vector<std::string> k;for(const auto& item:object.items())k.push_back(item.key());return k;}
std::vector<std::string> sortedNames(const std::initializer_list<const char*>& names){std::vector<std::string> k(names.begin(),names.end());std::sort(k.begin(),k.end());return k;}
template<size_t N> std::vector<std::string> sortedNames(const std::array<const char*,N>& names){std::vector<std::string> k(names.begin(),names.end());std::sort(k.begin(),k.end());return k;}
}
int main(){
 try {
  // Activation accepts only unset, "0" and "1"; rejections report length, not content.
  require(!enabledBy(nullptr)&&!enabledBy("0")&&enabledBy("1"),"accepted settings differ");
  const std::string huge(4096,'1');
  for(const char* bad:{"","true","TRUE","on","yes","2","-1","01","10"," 1","1 ","1\n","0x1",huge.c_str()}){
   bool rejected=false;
   try{enabledBy(bad);}catch(const std::invalid_argument& e){
    const std::string m=e.what();
    rejected=contains(m,"CHANDRA_NATIVE_CPU_TIMING must be unset, 0 or 1; rejected a "+std::to_string(std::strlen(bad))+"-byte value before D3D11 device creation")&&m.size()<160&&!contains(m,"true")&&!contains(m,"1111");
   }
   require(rejected,"invalid setting accepted or diagnostic differs");
  }
  ++passed;

  // Environment: the inherited setting, then unset, 0, 1 and rejected values through
  // the helper's platform branch; the inherited setting is then restored and read
  // back. A failed check ends the process, so only the passing path restores.
  const char* inherited="absent";
  {const Setting preexisting=environmentSetting();
   if(preexisting.present)inherited=preexisting.value.empty()?"empty":"present";
   require(helperAgrees(preexisting),"inherited setting read differently from enabledBy");
#ifdef _MSC_VER
   // A parent's environment block can carry an empty entry; _putenv_s cannot restore it.
   require(!preexisting.present||!preexisting.value.empty(),"an inherited empty CHANDRA_NATIVE_CPU_TIMING cannot be restored through _putenv_s; unset it and rerun");
#endif
   removeEnvironment();require(!environmentSetting().present&&!enabledByEnvironment(),"unset variable enabled timing");
   setEnvironment("0");require(!enabledByEnvironment(),"0 enabled timing");
   setEnvironment("1");require(enabledByEnvironment(),"1 did not enable timing");
   for(const std::string& bad:{std::string("yes"),std::string(300,'1')}){
    setEnvironment(bad.c_str());require(environmentSetting()==Setting{true,bad}&&environmentRejects(bad.size()),"environment value not rejected");
   }
#ifdef _MSC_VER
   setEnvironment("1");
   require(_putenv_s(variable,"")==0&&!environmentSetting().present&&!enabledByEnvironment(),"_putenv_s with an empty value did not remove the variable");
#else
   setEnvironment("");require(environmentSetting()==Setting{true,""}&&environmentRejects(0),"empty environment value not rejected");
#endif
   if(preexisting.present)setEnvironment(preexisting.value.c_str());else removeEnvironment();
   require(environmentSetting()==preexisting&&helperAgrees(preexisting),"inherited setting not restored");++passed;}

  // Disabled: every scope and method used by device.cpp, including exceptional exits,
  // makes no clock read and no allocation; memoryJson gets only the marker object.
  {const std::string name="runtime/linear.hlsl";
   const DrainScript ok{{notReady,ready},1,1,1,1,1,1,1,3,false},expired{{notReady},1,1,1,1,1,1,1,0,true};
   ReadbackScript rb;rb.before=ok;rb.after=ok;rb.sleeps={{1000000,1000000}};ReadbackScript rbFail=rb;rbFail.failAfterCopy=true;
   const size_t before=allocations;const uint64_t reads=fakeReads;Recorder* none=nullptr;
   for(int i=0;i<3;++i){
    dispatch(none,name,1,1,i==0,1,1,1,true);
    try{DispatchScope timed(none,name);timed.lap(dispatchPrepare);throw Thrown{};}catch(const Thrown&){}
    drain(none,ok);try{drain(none,expired);}catch(const Thrown&){}
    readback(none,rb);try{readback(none,rbFail);}catch(const Thrown&){}
    buffer(none,1,1,1,4096);call(none,&Recorder::upload,1);call(none,&Recorder::zero,1);
   }
   require(allocations==before&&fakeReads==reads,"disabled timing allocated or read the clock");
   const Json off=disabledJson();
   require(off==Json{{"schema","chandra.directcompute.cpu_timing.v1"},{"enabled",false},{"environment_variable","CHANDRA_NATIVE_CPU_TIMING"}},"disabled marker differs");++passed;}

  // Dispatch: compile versus lookup laps, GPU-profiled flag, failure after one lap,
  // per-shader rows for successful calls only and seven clock reads per success.
  {Recorder r(&fakeNow,0);const std::string linear="runtime/linear.hlsl",norm="runtime/norm.hlsl";
   const uint64_t reads=fakeReads;dispatch(&r,linear,3,1000,true,4,10,2);
   require(fakeReads-reads==7,"enabled dispatch clock reads differ");
   dispatch(&r,linear,2,1,false,3,8,1,true);dispatch(&r,norm,5,700,true,6,20,4);
   try{DispatchScope timed(&r,norm);advance(9);timed.lap(dispatchPrepare);advance(11);throw Thrown{};}catch(const Thrown&){}
   const auto& d=r.dispatch;
   require(same(d.op.calls,4,1789)&&d.op.calls.longest==1019&&same(d.op.failed,1,20),"dispatch inclusive totals differ");
   require(same(d.op.phases[dispatchPrepare],4,19)&&same(d.op.phases[dispatchShaderLookup],1,1)&&same(d.op.phases[dispatchShaderCompile],2,1700)&&
    same(d.op.phases[dispatchRetain],3,13)&&same(d.op.phases[dispatchRecord],3,38)&&same(d.op.phases[dispatchExit],4,18)&&partitioned(d.op),"dispatch laps differ");
   require(d.rows.size()==2&&d.rows[0].name==linear&&d.rows[1].name==norm&&d.index.size()==2,"shader rows differ");
   require(same(d.rows[0].calls,2,1034)&&same(d.rows[0].record,2,18)&&same(d.rows[0].compile,1,1000)&&
    same(d.rows[1].calls,1,735)&&same(d.rows[1].record,1,20)&&same(d.rows[1].compile,1,700),"shader row values differ");
   require(d.rows[0].calls.total+d.rows[1].calls.total==d.op.calls.total-d.op.failed.total&&d.gpuProfiled.value==1&&same(d.attribution,3,0)&&d.unattributed.count==0,"shader attribution totals differ");++passed;}

  // Bounded rows: names beyond capacity share one row; steady-state accounting,
  // including that row, never allocates or grows storage.
  {Recorder r(&fakeNow,0);std::vector<std::string> names;
   for(size_t i=0;i<shaderCapacity+3;++i)names.push_back("runtime/generated_shader_"+std::to_string(i)+".hlsl");
   const auto* storage=r.dispatch.rows.data();const size_t capacity=r.dispatch.rows.capacity(),buckets=r.dispatch.index.bucket_count();
   for(const auto& n:names)dispatch(&r,n,1,1,true,1,1,1);
   const size_t before=allocations;for(const auto& n:names)dispatch(&r,n,1,1,false,1,1,1);
   const auto& d=r.dispatch;
   require(allocations==before,"steady-state enabled dispatch accounting allocated");
   require(capacity>=shaderCapacity&&d.rows.size()==shaderCapacity&&d.rows.data()==storage&&d.rows.capacity()==capacity&&d.index.size()==shaderCapacity&&d.index.bucket_count()==buckets,"shader storage grew past its bound");
   require(same(d.beyondCapacity.calls,6,30)&&same(d.beyondCapacity.compile,3,3)&&d.rows.back().name==names[shaderCapacity-1],"beyond-capacity row differs");
   uint64_t rowCalls=0;for(const auto& row:d.rows)rowCalls+=row.calls.count;
   require(rowCalls==2*shaderCapacity&&d.op.calls.count==names.size()*2&&partitioned(d.op),"bounded shader totals differ");
   const Json j=r.json();require(j.at("dispatch").at("shaders").size()==shaderCapacity&&j.at("dispatch").at("shader_row_capacity")==shaderCapacity,"bounded shader JSON differs");++passed;}

  // A failed row insertion is unattributed instead of escaping the destructor.
  {Recorder r(&fakeNow,0);const std::string name="runtime/a_shader_name_longer_than_small_string.hlsl";
   {DispatchScope timed(&r,name);advance(1);timed.lap(dispatchPrepare);advance(1);timed.lap(dispatchShaderCompile);advance(1);timed.lap(dispatchRetain);advance(1);timed.recorded(false);failNextAllocation=true;}
   require(!failNextAllocation&&r.dispatch.rows.empty()&&r.dispatch.index.empty()&&same(r.dispatch.unattributed,1,4)&&same(r.dispatch.op.calls,1,4)&&r.dispatch.op.failed.count==0,"failed insertion was not unattributed");
   dispatch(&r,name,1,1,false,1,1,1);require(r.dispatch.rows.size()==1&&same(r.dispatch.rows[0].calls,1,5),"later insertion after failure differs");++passed;}

  // Drain: immediate-ready and pending EVENT queries, actual versus requested sleep,
  // fence wait from Flush to the completing poll, deadline/GetData/validation failures.
  {Recorder r(&fakeNow,0);
   drain(&r,{{ready},7,30,2,0,0,5,1,4,false});
   drain(&r,{{notReady,notReady,ready},6,20,3,1,1600000,2,1,0,false});
   try{drain(&r,{{notReady,notReady},1,1,1,1,1000000,0,3,0,true});require(false,"expired drain returned");}catch(const Thrown&){}
   try{drain(&r,{{failedStatus},1,1,1,0,0,0,2,0,false});require(false,"failed GetData returned");}catch(const Thrown&){}
   try{DrainScope timed(&r);advance(4);throw Thrown{};}catch(const Thrown&){}
   const auto& s=r.drain(Origin::outsideReadback);const auto& p=s.op.phases;
   require(same(s.op.calls,5,45+3200040+1000009+5+4)&&same(s.op.failed,3,1000009+5+4),"drain inclusive totals differ");
   require(same(p[drainQueryCreate],4,15)&&same(p[drainEndFlush],4,52)&&same(p[drainGetData],7,2+9+2+1)&&same(p[drainDeadlineCheck],3,3)&&
    same(p[drainSleep],3,4200000)&&same(p[drainRelease],2,7)&&same(p[drainExit],5,1+1+4+2+4)&&partitioned(s.op),"drain laps differ");
   require(s.firstPollReady.value==1&&s.firstPollNotReady.value==2&&s.pollsReady.value==2&&s.pollsNotReady.value==4&&s.pollsFailed.value==1,"poll classification differs");
   require(same(s.fenceWait,2,2+3200011)&&s.fenceWait.longest==3200011&&s.requestedSleep.value==3000000&&s.retainedReleased.value==4,"fence wait, sleep or release accounting differs");
   require(r.drain(Origin::readbackBeforeCopy).op.calls.count==0&&r.drain(Origin::readbackAfterCopy).op.calls.count==0,"outside drains were attributed to readback");++passed;}

  // Readback nests its two drains by origin; top-level totals cover every advanced
  // nanosecond exactly once, and origin is restored on success and on failure.
  {const uint64_t reads=fakeReads;Recorder r(&fakeNow,10000000);const Nanoseconds created=fakeTime;
   const std::string linear="runtime/linear.hlsl";
   buffer(&r,50,400,5,1048576);call(&r,&Recorder::upload,30);call(&r,&Recorder::zero,12);
   dispatch(&r,linear,3,900,true,4,10,2);drain(&r,{{ready},5,40,2,0,0,9,1,6,false});
   ReadbackScript rb;rb.valid=3;rb.before={{notReady,ready},5,50,2,1,1100000,8,1,2,false};rb.staging=70;rb.copy=6;
   rb.after={{ready},4,25,2,0,0,1,1,0,false};rb.gap=1;rb.sleeps={{1000000,1300000}};rb.map=90;rb.exit=11;
   readback(&r,rb);
   const auto& before=r.drain(Origin::readbackBeforeCopy);const auto& after=r.drain(Origin::readbackAfterCopy);const auto& rp=r.readback.op.phases;
   require(same(before.op.calls,1,5+50+2+1+1100000+2+8+1)&&before.firstPollNotReady.value==1&&same(after.op.calls,1,4+25+2+1+1)&&after.firstPollReady.value==1,"readback drains not attributed by origin");
   require(same(rp[readbackBeforeCopyDrain],1,3+before.op.calls.total)&&same(rp[readbackStagingPrepare],1,70)&&same(rp[readbackCopyRecord],1,6)&&
    same(rp[readbackAfterCopyDrain],1,after.op.calls.total+1)&&same(rp[readbackMapReady],1,1300090)&&same(rp[readbackExit],1,11)&&partitioned(r.readback.op),"readback laps differ");
   require(same(r.readback.readinessSleep,1,1300000)&&r.readback.readinessRequested.value==1000000&&r.origin==Origin::outsideReadback,"readiness sleep or origin restoration differs");
   require(same(r.buffers.op.phases[bufferBudgetQuery],1,50)&&same(r.buffers.op.phases[bufferCreate],1,400)&&partitioned(r.buffers.op)&&r.buffers.bytes.value==1048576&&same(r.upload.calls,1,30)&&same(r.zero.calls,1,12),"buffer/upload/zero accounting differs");
   const uint64_t elapsed=uint64_t((fakeTime-created).count());
   require(r.accountedTopLevel()==elapsed,"top-level totals do not cover elapsed time exactly once");
   uint64_t naive=r.dispatch.op.calls.total+r.buffers.op.calls.total+r.upload.calls.total+r.zero.calls.total+r.readback.op.calls.total;
   for(const auto& origin:r.drains)naive+=origin.op.calls.total;
   require(naive-elapsed==before.op.calls.total+after.op.calls.total,"adding readback drains to readback should double-count exactly them");
   const Json j=Json::parse(r.json().dump());
   require(j.at("accounted_top_level_nanoseconds")==elapsed&&j.at("elapsed_nanoseconds")==elapsed&&j.at("clock_reads")==fakeReads-reads&&j.at("query_performance_frequency_hz")==10000000,"top-level JSON differs");
   rb.failAfterCopy=true;try{readback(&r,rb);require(false,"failed readback returned");}catch(const Thrown&){}
   require(r.origin==Origin::outsideReadback&&r.readback.op.failed.count==1&&r.readback.op.phases[readbackCopyRecord].count==2&&partitioned(r.readback.op),"failed readback did not restore origin");
   ReadbackScript drainFails=rb;drainFails.failAfterCopy=false;drainFails.before={{failedStatus},1,1,1,0,0,0,1,0,false};
   try{readback(&r,drainFails);require(false,"readback with failed drain returned");}catch(const Thrown&){}
   require(r.origin==Origin::outsideReadback&&r.readback.op.failed.count==2&&before.op.failed.count==1&&r.drain(Origin::outsideReadback).op.calls.count==1,"nested drain failure misattributed");
   drain(&r,{{ready},1,1,1,0,0,1,1,0,false});require(r.drain(Origin::outsideReadback).op.calls.count==2,"drain after readback not outside");++passed;}

  // JSON: versioned schema, exact phase keys, honest unknown frequency, and null
  // with a reason instead of a wrapped or negative value.
  {Recorder r(&fakeNow,0);dispatch(&r,"runtime/norm.hlsl",1,1,true,1,1,1);
   const Json j=Json::parse(r.json().dump());
   require(j.at("schema")=="chandra.directcompute.cpu_timing.v1"&&j.at("enabled")==true&&j.at("query_performance_frequency_hz").is_null()&&j.at("clock")=="std::chrono::steady_clock","JSON header differs");
   require(keys(j)==sortedNames({"schema","enabled","environment_variable","boundary","clock","query_performance_frequency_hz","elapsed_nanoseconds","clock_reads","top_level","accounted_top_level_nanoseconds","dispatch","drain","readback","buffer_create","upload","zero"}),"JSON keys differ");
   require(keys(j.at("dispatch").at("phases"))==sortedNames(dispatchNames)&&keys(j.at("readback").at("phases"))==sortedNames(readbackNames)&&keys(j.at("buffer_create").at("phases"))==sortedNames(bufferNames)&&!j.at("upload").contains("phases"),"phase keys differ");
   require(keys(j.at("drain"))==sortedNames({"outside_readback","readback_before_copy","readback_after_copy"})&&keys(j.at("drain").at("outside_readback").at("phases"))==sortedNames(drainNames),"drain keys differ");
   require(j.at("dispatch").at("shaders").at(0)==Json{{"shader","runtime/norm.hlsl"},{"calls",{{"count",1},{"nanoseconds",5},{"maximum_nanoseconds",5}}},{"record",{{"count",1},{"nanoseconds",1},{"maximum_nanoseconds",1}}},{"compile",{{"count",1},{"nanoseconds",1},{"maximum_nanoseconds",1}}}},"shader row JSON differs");
   require(j.at("top_level")==Json::array({"dispatch","buffer_create","upload","zero","drain.outside_readback","readback"})&&contains(j.at("boundary").get<std::string>(),"Not CPU utilization, GPU execution time, global free VRAM or throughput"),"boundary JSON differs");++passed;}
  {Accumulator a;const Nanoseconds big(std::numeric_limits<Nanoseconds::rep>::max());a.add(big);a.add(big);
   require(same(a,2,maximum-1),"largest representable total differs");a.add(big);a.add(Nanoseconds(1));
   require(a.unavailable&&std::string(a.unavailable)=="overflow"&&a.json().at("count").is_null()&&a.json().at("nanoseconds").is_null()&&a.json().at("unavailable")=="overflow","overflow not unavailable");
   Accumulator c;c.count=maximum;c.add(Nanoseconds(1));require(c.json().at("count").is_null(),"count overflow not unavailable");
   Accumulator n;n.add(Nanoseconds(-1));require(n.json().at("unavailable")=="negative_interval","negative interval accepted");
   Counter k;k.add(maximum);require(k.json()==maximum,"maximum counter differs");k.add(1);require(k.json().is_null(),"counter overflow not null");
   Recorder r(&fakeNow,0);{CallScope timed(&r,&Recorder::upload);advance(-5);}advance(5);
   require(r.upload.calls.unavailable&&r.accountedTopLevel().is_null()&&Json::parse(r.json().dump()).at("accounted_top_level_nanoseconds").is_null(),"clock regression not reported as unavailable");++passed;}

  std::cout<<"{\"test\":\"directcompute_cpu_timing_accounting\",\"checks_passed\":"<<passed
   <<",\"cpu_fakes_only\":true,\"d3d11_driver_or_gpu_exercised\":false,\"environment_route\":\""<<environmentRoute
   <<"\",\"empty_environment_value\":\""<<emptyEnvironmentValue<<"\",\"preexisting_variable\":\""<<inherited<<"\"}\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAILED after "<<passed<<" checks: "<<e.what()<<"\n";return 1;}
}
