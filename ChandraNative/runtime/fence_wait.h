// New code, MPL-2.0. One-variable experimental drain control, shared with CPU fakes.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
namespace chandra::dc::fence_wait {
using Nanoseconds=std::chrono::nanoseconds;
constexpr const char* variable="CHANDRA_EXPERIMENTAL_DRAIN_WAIT";
constexpr uint64_t removedValue=std::numeric_limits<uint64_t>::max();
constexpr uint32_t sliceMilliseconds=10;
constexpr uint32_t signaled=0,waitTimeout=258,failedWaitStatus=0xffffffffu;
enum class Mode {querySleep,fenceEvent};
inline Mode selectedBy(const char* value){
 if(!value||!std::strcmp(value,"query_sleep"))return Mode::querySleep;
 if(!std::strcmp(value,"fence_event"))return Mode::fenceEvent;
 throw std::invalid_argument(std::string(variable)+" must be unset, query_sleep or fence_event; value refused before adapter selection");
}
inline Mode selectedByEnvironment(){
#ifdef _MSC_VER
 struct Free {void operator()(char* p) const noexcept {std::free(p);}};
 char* value=nullptr;size_t bytes=0;const auto status=_dupenv_s(&value,&bytes,variable);
 const std::unique_ptr<char,Free> owned(value);
 if(status!=0)throw std::runtime_error(std::string(variable)+" secure environment acquisition failed");
 return selectedBy(owned.get());
#else
 return selectedBy(std::getenv(variable));
#endif
}
enum class Result {completed,poisoned,invalidTimeout,exhausted,canceled,removed,deadline,resetFailed,signalFailed,registrationFailed,waitFailed,unexpectedWait,earlyWake,unexpectedRemovalStatus};
struct State {
 uint64_t lastIssued=0;bool poisoned=false,registrationOutstanding=false;
};
struct Outcome {
 Result result=Result::poisoned;uint64_t target=0,completedValue=0;
 int32_t status=0;uint32_t error=0,lastWait=waitTimeout,checks=0,waits=0;
 uint64_t requestedWaitMilliseconds=0;Nanoseconds waited{0},elapsed{0};
 bool signaledSubmitted=false,eventRegistered=false,eventObserved=false;
};
inline const char* name(Result r) noexcept {
 switch(r){
 case Result::completed:return "completed";case Result::poisoned:return "poisoned";
 case Result::invalidTimeout:return "invalid_timeout";case Result::exhausted:return "fence_value_exhausted";
 case Result::canceled:return "canceled";case Result::removed:return "device_removed";
 case Result::deadline:return "deadline";case Result::resetFailed:return "event_reset_failed";
 case Result::signalFailed:return "signal_failed";case Result::registrationFailed:return "event_registration_failed";
 case Result::waitFailed:return "wait_failed";case Result::unexpectedWait:return "unexpected_wait_status";
 case Result::earlyWake:return "event_woke_before_matching_fence";
 case Result::unexpectedRemovalStatus:return "unexpected_device_removed_status";
 }
 return "unknown";
}
enum class Phase {prepare,signalFlush,registration,check,deadline,wait};
// Driver owns an immediate context, one fence and one private manual-reset event.
// It returns exact S_OK (0) from HRESULT operations; all other statuses fail.
// No registration may be reused until its event was observed and target completed.
// On any failure, poison is sticky; caller retains all resources until process exit.
// now() is monotonic. wait(ms) blocks, never spins; Win32 errors are captured by driver.
template<class Driver,class Observe>
Outcome drain(State& state,uint32_t timeout,Driver& d,const Observe& observe){
 Outcome o;const Nanoseconds start=d.now();
 auto finish=[&](Result r){o.result=r;o.elapsed=d.now()-start;if(r!=Result::completed)state.poisoned=true;return o;};
 if(state.poisoned||state.registrationOutstanding)return finish(Result::poisoned);
 if(!timeout||timeout>30000)return finish(Result::invalidTimeout);
 const Nanoseconds budget=std::chrono::milliseconds(timeout);
 if(start.count()<0||start.count()>std::numeric_limits<int64_t>::max()-budget.count())return finish(Result::deadline);
 const Nanoseconds deadline=start+budget;
 if(state.lastIssued>=removedValue-1)return finish(Result::exhausted);
 o.target=++state.lastIssued; // Reserve UINT64_MAX for device removal; never wrap or rewind.
 auto check=[&](){
  if(d.canceled())return Result::canceled;
  o.status=d.removed();
  if(o.status<0)return Result::removed;
  if(o.status!=0)return Result::unexpectedRemovalStatus;
  if(d.now()>=deadline)return Result::deadline;
  return Result::completed; // Checks passed, not yet a proof of GPU completion.
 };
 Result valid=check();observe(Phase::deadline,false,false);
 if(valid!=Result::completed)return finish(valid);
 if(!d.resetEvent()){o.error=d.error();observe(Phase::prepare,false,true);return finish(Result::resetFailed);}
 observe(Phase::prepare,false,false);
 o.status=d.signal(o.target);
 if(o.status!=0){observe(Phase::signalFlush,false,true);return finish(Result::signalFailed);}
 o.signaledSubmitted=true;d.flush();observe(Phase::signalFlush,false,false);
 // Fast completion has no event registration to retire.
 valid=check();observe(Phase::deadline,false,false);if(valid!=Result::completed)return finish(valid);
 o.completedValue=d.completed();++o.checks;
 if(o.completedValue==removedValue){observe(Phase::check,false,true);return finish(Result::removed);}
 valid=check();observe(Phase::deadline,false,false);if(valid!=Result::completed)return finish(valid);
 if(o.completedValue>=o.target){observe(Phase::check,true,false);return finish(Result::completed);}
 observe(Phase::check,false,false);
 // Mark ownership before the registration call: even a failed call is quarantined.
 state.registrationOutstanding=true;o.status=d.registerEvent(o.target);
 if(o.status!=0){observe(Phase::registration,false,true);return finish(Result::registrationFailed);}
 o.eventRegistered=true;observe(Phase::registration,false,false);
 for(;;){
  valid=check();observe(Phase::deadline,false,false);if(valid!=Result::completed)return finish(valid);
  const Nanoseconds remaining=deadline-d.now();if(remaining.count()<=0)return finish(Result::deadline);
  // Round up to a nonzero millisecond. Absolute deadline is checked after each wait.
  const uint64_t ms=uint64_t((remaining.count()+999999)/1000000);
  const uint32_t request=uint32_t(std::min<uint64_t>(sliceMilliseconds,ms));
  const Nanoseconds before=d.now();o.lastWait=d.wait(request);const Nanoseconds after=d.now();
  ++o.waits;o.requestedWaitMilliseconds+=request;o.waited+=after-before;observe(Phase::wait,false,false);
  if(o.lastWait==failedWaitStatus){o.error=d.error();return finish(Result::waitFailed);}
  if(o.lastWait!=waitTimeout&&o.lastWait!=signaled)return finish(Result::unexpectedWait);
  valid=check();observe(Phase::deadline,false,false);if(valid!=Result::completed)return finish(valid);
  o.completedValue=d.completed();++o.checks;
  if(o.completedValue==removedValue){observe(Phase::check,false,true);return finish(Result::removed);}
  valid=check();observe(Phase::deadline,false,false);if(valid!=Result::completed)return finish(valid);
  if(o.lastWait==signaled){
   o.eventObserved=true;
   if(o.completedValue<o.target){observe(Phase::check,false,true);return finish(Result::earlyWake);}
   state.registrationOutstanding=false;observe(Phase::check,true,false);return finish(Result::completed);
  }
  // Completion without observed event delivery cannot recycle the registered handle.
  observe(Phase::check,false,false);
 }
}
}
