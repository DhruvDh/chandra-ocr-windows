// New code, MPL-2.0. One fixed ownership slot for a submitted staging copy.
#pragma once
#include "fence_wait.h"
#include "weight_storage_experiment.h"
#include <type_traits>
#include <utility>
namespace chandra::dc::copy_retention {
// Source is shared_ptr<Storage>; Staging is ComPtr<ID3D11Buffer>. No allocation
// occurs while adopting those already-created owners. Tests use the same class.
template<class Source,class Staging> class Owner {
 static_assert(std::is_nothrow_move_assignable<Source>::value&&std::is_nothrow_move_assignable<Staging>::value,"Copy owners must move without allocation or exceptions");
 static_assert(std::is_nothrow_default_constructible<Source>::value&&std::is_nothrow_default_constructible<Staging>::value,"Empty copy owners must construct without exceptions");
 Source sourceOwner;Staging stagingOwner;uint32_t live=0,peak=0;uint64_t expected=0;bool queued=false;
public:
 Owner()=default;Owner(const Owner&)=delete;Owner& operator=(const Owner&)=delete;
 bool active() const noexcept{return live!=0;}
 uint32_t bytes() const noexcept{return live;}
 uint32_t peakBytes() const noexcept{return peak;}
 uint64_t expectedFence() const noexcept{return expected;}
 bool submitted() const noexcept{return queued;}
 bool owns(const void* pointer) const noexcept{return active()&&stagingOwner.Get()==pointer;}
 void hold(Source source,Staging staging,uint32_t bytes,uint64_t tracked,uint64_t lastFence){
  if(active())throw std::runtime_error("One pending staging-copy owner already occupied; refuse before enqueue");
  if(!source.get()||!staging.Get()||!bytes||bytes%4||bytes>weight_storage::maximumBufferBytes)
   throw std::invalid_argument("Staging-copy ownership requires valid source/staging and 1..128 MiB whole-word extent");
  if(!weight_storage::withinBudget(tracked,bytes,13ull*1024*1024*1024))throw std::runtime_error("Retained staging copy exceeds original tracked-plus-staging cap; refuse before enqueue");
  if(lastFence>=fence_wait::removedValue-1)throw std::runtime_error("No nonwrapping next fence for staging copy; refuse before enqueue");
  // All refusal/allocation-prone work precedes these noexcept owner moves.
  sourceOwner=std::move(source);stagingOwner=std::move(staging);live=bytes;peak=std::max(peak,bytes);expected=lastFence+1;queued=false;
 }
 void beforeEnqueue(){if(!active()||queued)throw std::runtime_error("Invalid staging-copy submission ownership");queued=true;}
 // The result must prove this exact ownership epoch and registration delivery.
 // A failed/late/unknown drain never releases it; process retirement owns cleanup.
 bool completed(const fence_wait::Outcome& proof) noexcept {
  if(!active())return true;
  if(proof.result!=fence_wait::Result::completed||proof.target!=expected||proof.completedValue<expected||
   proof.completedValue==fence_wait::removedValue||(proof.eventRegistered&&!proof.eventObserved))return false;
  sourceOwner=Source{};stagingOwner=Staging{};live=0;expected=0;queued=false;return true;
 }
};
}
