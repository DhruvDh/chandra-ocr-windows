// New code, MPL-2.0. Portable control for bounded staging-readback readiness.
// D3D11 Map(READ, DO_NOT_WAIT) may return DXGI_ERROR_WAS_STILL_DRAWING while an
// already-submitted staging copy is not yet CPU-readable. Only that status is
// polled again, on the same staging resource, until one absolute deadline. No
// Windows types appear here so CPU tests drive the same control as device.cpp.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
namespace chandra::dc::readback {
using Nanoseconds=std::chrono::nanoseconds;
constexpr int32_t okStatus=0; // S_OK
constexpr int32_t stillDrawingStatus=int32_t(0x887A000Au); // DXGI_ERROR_WAS_STILL_DRAWING
// One post-copy window, equal to Device::drain's default, covers the existing
// copy drain and every Map attempt; it is anchored once when the copy is recorded.
constexpr Nanoseconds postCopyBudget=std::chrono::milliseconds(10000);
constexpr Nanoseconds firstDelay=std::chrono::milliseconds(1),maximumDelay=std::chrono::milliseconds(8);
enum class Result {copied,stillDrawingAtDeadline,mapFailed,unexpectedSuccess,nullData};
struct Outcome {Result result=Result::mapFailed;int32_t status=okStatus;uint32_t attempts=0,stillDrawing=0;Nanoseconds waited{0};};
template<class Unmap> class Mapped {
 const Unmap& unmap;
public:
 explicit Mapped(const Unmap& u):unmap(u){}
 ~Mapped(){unmap();}
 Mapped(const Mapped&)=delete;Mapped& operator=(const Mapped&)=delete;
};
// map(void**) returns the Map status and sets the data pointer. Any success status
// means mapped, so unmap() runs exactly once, also if consume(const void*) throws;
// failed statuses are never unmapped. unmap() must not throw. now() is monotonic
// and sleep(d) blocks for at least d on it. One attempt is always made, so an
// already-expired deadline still permits the original single nonblocking Map.
template<class Map,class Unmap,class Consume,class Now,class Sleep>
Outcome copyWhenReady(Nanoseconds deadline,const Map& map,const Unmap& unmap,const Consume& consume,const Now& now,const Sleep& sleep){
 Outcome o;const Nanoseconds first=now();Nanoseconds delay=firstDelay;
 for(;;){
  void* data=nullptr;o.status=map(&data);++o.attempts;
  if(o.status==stillDrawingStatus){
   ++o.stillDrawing;const Nanoseconds observed=now();
   if(observed>=deadline){o.result=Result::stillDrawingAtDeadline;o.waited=observed-first;return o;}
   sleep(std::min(delay,deadline-observed));delay=std::min(delay*2,maximumDelay);continue;
  }
  if(o.status<0)o.result=Result::mapFailed;
  else {
   Mapped<Unmap> mapped(unmap);
   if(o.status!=okStatus)o.result=Result::unexpectedSuccess;
   else if(!data)o.result=Result::nullData;
   else {consume(static_cast<const void*>(data));o.result=Result::copied;}
  }
  o.waited=now()-first;return o;
 }
}
inline std::string describe(const Outcome& o){
 char text[384];const unsigned status=unsigned(uint32_t(o.status)),attempts=o.attempts;const double ms=double(o.waited.count())/1e6;
 switch(o.result){
 case Result::copied:
  std::snprintf(text,sizeof(text),"Map completed readback copied on attempt %u after %.3f ms",attempts,ms);break;
 case Result::stillDrawingAtDeadline:
  std::snprintf(text,sizeof(text),"Map completed readback still drawing HRESULT=%u (0x%08X) at the %lld ms post-copy readiness deadline after %u nonblocking attempts over %.3f ms; copy not resubmitted; fail and retire the request and owned worker",
   status,status,static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(postCopyBudget).count()),attempts,ms);break;
 case Result::mapFailed:
  std::snprintf(text,sizeof(text),"Map completed readback HRESULT=%u (0x%08X) on attempt %u after %.3f ms; status is not retried",status,status,attempts,ms);break;
 case Result::unexpectedSuccess:
  std::snprintf(text,sizeof(text),"Map completed readback returned unexpected success HRESULT=%u (0x%08X) on attempt %u; unmapped and rejected",status,status,attempts);break;
 case Result::nullData:
  std::snprintf(text,sizeof(text),"Map completed readback returned S_OK with null data on attempt %u; unmapped and rejected",attempts);break;
 default:
  std::snprintf(text,sizeof(text),"Map completed readback outcome unknown on attempt %u",attempts);break;
 }
 return text;
}
}
