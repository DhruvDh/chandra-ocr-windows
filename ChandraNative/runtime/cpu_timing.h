// New code, MPL-2.0. Opt-in CPU-side wall accounting for chandra::dc::Device.
// CHANDRA_NATIVE_CPU_TIMING is read once per Device, before adapter selection,
// D3D11 device creation or model upload: unset or "0" disables, "1" enables and
// any other value is rejected. Disabled scopes hold a null Recorder, so they make
// no clock reads and allocate nothing. Enabled storage is fixed when the Device is
// created (bounded shader rows plus one overflow row); there is no event log.
// A Scope partitions one Device call into contiguous laps that share clock reads,
// so a call's laps sum exactly to its inclusive time. Every interval is host wall
// time on the Device owner thread, including time blocked in D3D11/driver calls,
// waits and sleeps: not CPU utilization, GPU execution time, free VRAM or
// throughput. No Windows types appear here, so CPU tests drive this same code.
// Field boundaries: docs/directcompute-cpu-timing.md.
#pragma once
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace chandra::dc::cpu_timing {
using Nanoseconds=std::chrono::nanoseconds;
using Json=nlohmann::json;
using Clock=Nanoseconds(*)() noexcept;
constexpr const char* variable="CHANDRA_NATIVE_CPU_TIMING";
constexpr const char* schema="chandra.directcompute.cpu_timing.v1";
constexpr size_t shaderCapacity=128;
constexpr uint64_t maximum=std::numeric_limits<uint64_t>::max();
// nullptr means unset. Only the exact strings "0" and "1" are accepted; a rejected
// value is reported by length only.
inline bool enabledBy(const char* value){
 if(!value||!std::strcmp(value,"0"))return false;
 if(!std::strcmp(value,"1"))return true;
 throw std::invalid_argument(std::string(variable)+" must be unset, 0 or 1; rejected a "+std::to_string(std::strlen(value))+"-byte value before D3D11 device creation");
}
inline bool enabledByEnvironment(){
#ifdef _MSC_VER
 struct Free {void operator()(char* p) const noexcept {std::free(p);}};
 char* value=nullptr;size_t bytes=0;
 if(_dupenv_s(&value,&bytes,variable))throw std::runtime_error(std::string(variable)+" could not be read");
 const std::unique_ptr<char,Free> owned(value);
 return enabledBy(value);
#else
 return enabledBy(std::getenv(variable));
#endif
}
inline Json disabledJson(){return Json{{"schema",schema},{"enabled",false},{"environment_variable",variable}};}
// Count, total and maximum of nonnegative intervals. A negative interval or a
// uint64 overflow makes the whole accumulator unavailable (JSON nulls), never wrapped.
struct Accumulator {
 uint64_t count=0,total=0,longest=0;const char* unavailable=nullptr;
 void add(Nanoseconds d) noexcept {
  if(unavailable)return;
  if(d.count()<0){unavailable="negative_interval";return;}
  const uint64_t v=uint64_t(d.count());
  if(count==maximum||v>maximum-total){unavailable="overflow";return;}
  ++count;total+=v;longest=std::max(longest,v);
 }
 Json json() const {
  if(unavailable)return Json{{"count",nullptr},{"nanoseconds",nullptr},{"maximum_nanoseconds",nullptr},{"unavailable",unavailable}};
  return Json{{"count",count},{"nanoseconds",total},{"maximum_nanoseconds",longest}};
 }
};
// Event or quantity count; null in JSON once an addition would overflow uint64.
struct Counter {
 uint64_t value=0;bool overflow=false;
 void add(uint64_t n) noexcept {if(overflow)return;if(n>maximum-value){overflow=true;return;}value+=n;}
 Json json() const {return overflow?Json(nullptr):Json(value);}
};
// One instrumented Device entry point. calls: inclusive time of every call, from
// scope construction just after the owner-thread check until scope destruction
// after every later local is destroyed, whether it returns or throws. failed: the
// subset that left by exception. phases: contiguous laps; the last is "exit".
template<size_t N> struct Operation {
 Accumulator calls,failed;std::array<Accumulator,N> phases;
 Json json([[maybe_unused]] const std::array<const char*,N>& names) const {
  Json j{{"calls",calls.json()},{"failed_calls",failed.json()}};
  if constexpr(N>1){Json p=Json::object();for(size_t i=0;i<N;++i)p[names[i]]=phases[i].json();j["phases"]=p;}
  return j;
 }
};
enum DispatchPhase:size_t {dispatchPrepare,dispatchShaderLookup,dispatchShaderCompile,dispatchRetain,dispatchRecord,dispatchExit,dispatchPhases};
enum DrainPhase:size_t {drainQueryCreate,drainEndFlush,drainGetData,drainDeadlineCheck,drainSleep,drainRelease,drainFencePrepare,drainFenceRegistration,drainFenceCheck,drainEventWait,drainExit,drainPhases};
enum ReadbackPhase:size_t {readbackBeforeCopyDrain,readbackStagingPrepare,readbackCopyRecord,readbackAfterCopyDrain,readbackMapReady,readbackExit,readbackPhases};
enum BufferPhase:size_t {bufferBudgetQuery,bufferCreate,bufferExit,bufferPhases};
constexpr std::array<const char*,dispatchPhases> dispatchNames{"prepare","shader_lookup","shader_compile","retain","record","exit"};
constexpr std::array<const char*,drainPhases> drainNames{"query_create","end_flush","get_data","deadline_check","sleep","release","fence_prepare","fence_registration","fence_check","event_wait","exit"};
constexpr std::array<const char*,readbackPhases> readbackNames{"before_copy_drain","staging_prepare","copy_record","after_copy_drain","map_ready","exit"};
constexpr std::array<const char*,bufferPhases> bufferNames{"budget_query","create","exit"};
constexpr std::array<const char*,1> callNames{"exit"};
// Successful dispatches only: calls is inclusive, record/compile are those laps.
struct ShaderRow {
 std::string name;Accumulator calls,record,compile;
 Json json() const {return Json{{"calls",calls.json()},{"record",record.json()},{"compile",compile.json()}};}
};
struct DispatchStats {
 Operation<dispatchPhases> op;Counter gpuProfiled;
 std::vector<ShaderRow> rows;std::unordered_map<std::string,size_t> index;ShaderRow beyondCapacity;
 Accumulator unattributed,attribution;
};
// Which caller a drain serves. Drains inside Device::readWords are nested in its
// readback interval; only outsideReadback drains are top-level calls.
enum class Origin:uint8_t {outsideReadback,readbackBeforeCopy,readbackAfterCopy};
struct DrainStats {
 Operation<drainPhases> op;Accumulator fenceWait;
 Counter firstPollReady,firstPollNotReady,pollsReady,pollsNotReady,pollsFailed,requestedSleep,retainedReleased;
 Json json() const {
  Json j=op.json(drainNames);
  j["first_poll_ready"]=firstPollReady.json();j["first_poll_not_ready"]=firstPollNotReady.json();
  j["polls"]=Json{{"ready",pollsReady.json()},{"not_ready",pollsNotReady.json()},{"failed",pollsFailed.json()}};
  j["fence_wait"]=fenceWait.json();j["requested_sleep_nanoseconds"]=requestedSleep.json();
  j["retained_buffer_references_released"]=retainedReleased.json();
  return j;
 }
};
struct ReadbackStats {Operation<readbackPhases> op;Accumulator readinessSleep;Counter readinessRequested;};
struct BufferStats {Operation<bufferPhases> op;Counter bytes;};
class Recorder {
 Clock clock;uint64_t frequency;mutable Counter reads;Nanoseconds created;
public:
 DispatchStats dispatch;std::array<DrainStats,3> drains;ReadbackStats readback;BufferStats buffers;Operation<1> upload,zero;
 Origin origin=Origin::outsideReadback;
 // performanceFrequency is the Windows QueryPerformanceFrequency, or 0 if unknown.
 Recorder(Clock source,uint64_t performanceFrequency):clock(source),frequency(performanceFrequency),created(now()){
  dispatch.rows.reserve(shaderCapacity);dispatch.index.reserve(shaderCapacity);
 }
 Recorder(const Recorder&)=delete;Recorder& operator=(const Recorder&)=delete;
 Nanoseconds now() const noexcept {reads.add(1);return clock();}
 const DrainStats& drain(Origin o) const {return drains[size_t(o)];}
 // Sum of the non-overlapping top-level inclusive totals, or null if any is unavailable.
 Json accountedTopLevel() const {
  const Accumulator* top[]={&dispatch.op.calls,&buffers.op.calls,&upload.calls,&zero.calls,&drain(Origin::outsideReadback).op.calls,&readback.op.calls};
  uint64_t sum=0;
  for(const auto* a:top){if(a->unavailable||a->total>maximum-sum)return nullptr;sum+=a->total;}
  return sum;
 }
 Json json() const {
  const Nanoseconds at=now();
  Json shaders=Json::array();
  for(const auto& row:dispatch.rows){Json j=row.json();j["shader"]=row.name;shaders.push_back(std::move(j));}
  Json d=dispatch.op.json(dispatchNames);
  d["gpu_timestamp_profiled_calls"]=dispatch.gpuProfiled.json();d["shader_row_capacity"]=shaderCapacity;d["shaders"]=std::move(shaders);
  d["shaders_beyond_capacity"]=dispatch.beyondCapacity.json();d["unattributed_calls"]=dispatch.unattributed.json();
  d["attribution_overhead"]=dispatch.attribution.json();
  Json r=readback.op.json(readbackNames);
  r["readiness_sleep"]=readback.readinessSleep.json();r["readiness_requested_sleep_nanoseconds"]=readback.readinessRequested.json();
  Json b=buffers.op.json(bufferNames);b["bytes_created"]=buffers.bytes.json();
  Accumulator elapsed;elapsed.add(at-created);
  return Json{{"schema",schema},{"enabled",true},{"environment_variable",variable},
   {"boundary","Cumulative host wall time on the Device owner thread since Device construction; includes time blocked in D3D11/driver calls, fence waits and sleeps. Not CPU utilization, GPU execution time, global free VRAM or throughput. See docs/directcompute-cpu-timing.md"},
   {"clock","std::chrono::steady_clock"},{"query_performance_frequency_hz",frequency?Json(frequency):Json(nullptr)},
   {"elapsed_nanoseconds",elapsed.unavailable?Json(nullptr):Json(elapsed.total)},{"clock_reads",reads.json()},
   {"top_level",Json::array({"dispatch","buffer_create","upload","zero","drain.outside_readback","readback"})},{"accounted_top_level_nanoseconds",accountedTopLevel()},
   {"dispatch",d},
   {"drain",{{"outside_readback",drains[0].json()},{"readback_before_copy",drains[1].json()},{"readback_after_copy",drains[2].json()}}},
   {"readback",r},{"buffer_create",b},{"upload",upload.json(callNames)},{"zero",zero.json(callNames)}};
 }
};
// RAII partition of one call. With a null Recorder every member is a no-op.
template<size_t N> class Scope {
protected:
 Recorder* const recorder;Operation<N>* const operation;Nanoseconds started{0},previous{0};int exceptions=0;bool open=false,threw=false;
 Nanoseconds mark(size_t phase) noexcept {
  if(!open)return Nanoseconds{0};
  const Nanoseconds t=recorder->now(),d=t-previous;operation->phases[phase].add(d);previous=t;return d;
 }
 // Closes the "exit" lap and the inclusive interval once; returns the end time.
 Nanoseconds finish() noexcept {
  if(!open)return previous;
  open=false;threw=std::uncaught_exceptions()>exceptions;
  const Nanoseconds t=recorder->now();operation->phases[N-1].add(t-previous);operation->calls.add(t-started);
  if(threw)operation->failed.add(t-started);
  previous=t;return t;
 }
public:
 Scope(Recorder* r,Operation<N>* o) noexcept:recorder(o?r:nullptr),operation(r?o:nullptr){
  if(recorder){exceptions=std::uncaught_exceptions();started=previous=recorder->now();open=true;}
 }
 Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
 ~Scope(){finish();}
 void lap(size_t phase) noexcept {if(phase<N-1)mark(phase);}
};
class CallScope:public Scope<1> {
public:
 CallScope(Recorder* r,Operation<1> Recorder::* member) noexcept:Scope(r,r?&(r->*member):nullptr){}
};
class BufferScope:public Scope<bufferPhases> {
public:
 explicit BufferScope(Recorder* r) noexcept:Scope(r,r?&r->buffers.op:nullptr){}
 void created(uint64_t bytes) noexcept {if(!open)return;mark(bufferCreate);recorder->buffers.bytes.add(bytes);}
};
// Per-shader attribution runs after the call's end time and is itself timed as
// attribution overhead; a full table or failed insertion never throws.
class DispatchScope:public Scope<dispatchPhases> {
 const std::string& shader;Nanoseconds recordLap{0},compileLap{0};bool compiled=false,haveRecord=false;
public:
 DispatchScope(Recorder* r,const std::string& name) noexcept:Scope(r,r?&r->dispatch.op:nullptr),shader(name){}
 void lap(size_t phase) noexcept {
  if(phase>=dispatchExit)return;
  const Nanoseconds d=mark(phase);
  if(phase==dispatchShaderCompile){compileLap=d;compiled=true;}else if(phase==dispatchRecord){recordLap=d;haveRecord=true;}
 }
 // After the final unbind; GPU timestamp End calls are inside the lap when profiling.
 void recorded(bool gpuTimestamps) noexcept {if(!open)return;lap(dispatchRecord);if(gpuTimestamps)recorder->dispatch.gpuProfiled.add(1);}
 ~DispatchScope(){
  if(!open)return;
  const Nanoseconds end=finish();if(threw)return;
  auto& s=recorder->dispatch;ShaderRow* row=nullptr;
  try{
   const auto found=s.index.find(shader);
   if(found!=s.index.end())row=&s.rows[found->second];
   else if(s.rows.size()<shaderCapacity){
    s.rows.emplace_back();
    try{s.rows.back().name=shader;s.index.emplace(shader,s.rows.size()-1);}catch(...){s.rows.pop_back();throw;}
    row=&s.rows.back();
   }else row=&s.beyondCapacity;
  }catch(...){row=nullptr;}
  if(row){row->calls.add(end-started);if(haveRecord)row->record.add(recordLap);if(compiled)row->compile.add(compileLap);}
  else s.unattributed.add(end-started);
  s.attribution.add(recorder->now()-end);
 }
};
class DrainScope:public Scope<drainPhases> {
 DrainStats* const stats;Nanoseconds flushedAt{0};bool polledOnce=false;
 DrainScope(Recorder* r,DrainStats* s) noexcept:Scope(r,s?&s->op:nullptr),stats(s){}
public:
 explicit DrainScope(Recorder* r) noexcept:DrainScope(r,r?&r->drains[size_t(r->origin)]:nullptr){}
 // After End(EVENT) and Flush: the fence wait starts at this lap.
 void flushed() noexcept {if(!open)return;mark(drainEndFlush);flushedAt=previous;}
 // After each GetData. Completion ends the fence wait; a failed status is counted
 // but never ready. The first poll classifies the drain as ready or not ready.
 void polled(bool ready,bool failedStatus) noexcept {
  if(!open)return;
  mark(drainGetData);
  if(!polledOnce&&!failedStatus)(ready?stats->firstPollReady:stats->firstPollNotReady).add(1);
  polledOnce=true;
  if(ready){stats->pollsReady.add(1);stats->fenceWait.add(previous-flushedAt);}
  else (failedStatus?stats->pollsFailed:stats->pollsNotReady).add(1);
 }
 // Same completion classification, with its own host phase for the event route.
 void fenceChecked(bool ready,bool failedStatus) noexcept {
  if(!open)return;
  mark(drainFenceCheck);
  if(!polledOnce&&!failedStatus)(ready?stats->firstPollReady:stats->firstPollNotReady).add(1);
  polledOnce=true;
  if(ready){stats->pollsReady.add(1);stats->fenceWait.add(previous-flushedAt);}
  else (failedStatus?stats->pollsFailed:stats->pollsNotReady).add(1);
 }
 void releasing(size_t references) noexcept {if(open)stats->retainedReleased.add(uint64_t(references));}
 void released() noexcept {mark(drainRelease);}
 void checkedDeadline() noexcept {mark(drainDeadlineCheck);}
 // After sleep_for(requested) returns: the lap is the actual sleep.
 void slept(Nanoseconds requested) noexcept {if(!open)return;mark(drainSleep);stats->requestedSleep.add(uint64_t(std::max(requested.count(),Nanoseconds::rep{0})));}
};
class ReadbackScope:public Scope<readbackPhases> {
 Origin saved=Origin::outsideReadback;
public:
 explicit ReadbackScope(Recorder* r) noexcept:Scope(r,r?&r->readback.op:nullptr){if(recorder){saved=recorder->origin;recorder->origin=Origin::readbackBeforeCopy;}}
 // After CopyResource is recorded: later drains in this call wait for the copy.
 void copied() noexcept {if(!open)return;mark(readbackCopyRecord);recorder->origin=Origin::readbackAfterCopy;}
 ~ReadbackScope(){if(recorder)recorder->origin=saved;}
};
// One readiness pause inside readback::copyWhenReady: actual versus requested.
class ReadinessSleep {
 Recorder* const recorder;Nanoseconds started{0};
public:
 ReadinessSleep(Recorder* r,Nanoseconds requested) noexcept:recorder(r){
  if(r){r->readback.readinessRequested.add(uint64_t(std::max(requested.count(),Nanoseconds::rep{0})));started=r->now();}
 }
 ReadinessSleep(const ReadinessSleep&)=delete;ReadinessSleep& operator=(const ReadinessSleep&)=delete;
 ~ReadinessSleep(){if(recorder)recorder->readback.readinessSleep.add(recorder->now()-started);}
};
}
