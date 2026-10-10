// New code, MPL-2.0. One original row, finite import boundaries; no repair or replay.
#pragma once
#include "import_row_watch_state.h"
#include <cstdint>
#include <stdexcept>
namespace chandra::dc::import_row::shard1 {
constexpr uint32_t targetOperation=2, shardIndex=1, globalRow=52427, firstRow=26214, rows=26214, columns=2560;
constexpr uint32_t rowWords=1280, rowBytes=5120, firstWord=(rows-1)*rowWords, resourceBytes=rows*rowBytes;
constexpr uint64_t payloadBegin=91512, tensorBegin=1271398400, sourceOffset=payloadBegin+tensorBegin+uint64_t(firstRow)*rowBytes;
constexpr uint32_t maximumSamples=733, maximumRecords=1472, maximumRecordBytes=4096, maximumPlanBytes=1024*1024;
constexpr uint64_t maximumArtifactBytes=uint64_t(maximumRecords)*maximumRecordBytes+maximumPlanBytes+3*rowBytes;
constexpr const char* expectedSHA="0575f9588d644d6b669a29517b1dda97ea56b4831b5578edffb99913df8e4ab7";
enum class Boundary { uploadDrain, ownViewRelease, constructorReturn };
inline void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
struct Counts {
 uint32_t admitted=0,allocated=0,uploadAdmitted=0,uploadReturned=0,drained=0,samples=0,copyAdmissions=0,phase=0;
 uint64_t allocatedBytes=0,completedUploadBytes=0;bool sourceCaptured=false,ownViewReleased=false,finalSample=false,pending=false,fault=false,refused=false,terminal=false;
 Boundary pendingBoundary=Boundary::uploadDrain;
 void begin(uint32_t ordinal,uint64_t bytes){require(!fault&&!refused&&!terminal&&!pending&&phase==0&&ordinal==admitted+1&&ordinal<=watch::operationCount&&bytes>0&&bytes<=134217728&&
  (ordinal<=targetOperation||ownViewReleased),"Shard1 operation admission order/extent differs");++admitted;phase=1;}
 void allocation(uint64_t bytes){require(phase==1&&bytes>0&&bytes<=watch::allocationBytes-allocatedBytes,"Shard1 allocation order/extent differs");++allocated;allocatedBytes+=bytes;phase=2;}
 void mapped(){require(phase==2,"Shard1 map order differs");phase=3;}
 void source(){require(phase==3&&admitted==targetOperation&&!sourceCaptured,"Shard1 source snapshot boundary differs");sourceCaptured=true;}
 void submit(){require(phase==3&&(admitted!=targetOperation||sourceCaptured),"Shard1 upload admission order differs");++uploadAdmitted;phase=4;}
 void returned(){require(phase==4,"Shard1 upload return order differs");++uploadReturned;phase=5;}
 void drain(uint64_t bytes){require(phase==5&&bytes<=watch::allocationBytes-completedUploadBytes,"Shard1 original drain order/extent differs");++drained;completedUploadBytes+=bytes;phase=6;}
 void skipBeforeTarget(){require(phase==6&&drained==1&&!sourceCaptured&&samples==0,"Shard1 pre-target skip differs");phase=0;}
 void release(bool succeeded){require(!fault&&!refused&&!terminal&&!pending&&phase==0&&drained==targetOperation&&samples==1&&sourceCaptured&&!ownViewReleased,"Shard1 own-view release order differs");
  require(succeeded,"Shard1 own source view unmap failed; release not confirmed");ownViewReleased=true;}
 void copy(Boundary boundary){require(!fault&&!refused&&!terminal&&!pending&&sourceCaptured&&copyAdmissions==samples&&samples<maximumSamples,"Shard1 copy after stop/pending or beyond bound");
  if(boundary==Boundary::uploadDrain)require(phase==6&&drained>=targetOperation&&!finalSample,"Shard1 upload sample boundary differs");
  else if(boundary==Boundary::ownViewRelease)require(phase==0&&drained==targetOperation&&samples==1&&ownViewReleased,"Shard1 release sample boundary differs");
  else require(phase==0&&drained==watch::operationCount&&samples==maximumSamples-1&&ownViewReleased&&!finalSample,"Shard1 constructor sample boundary differs");
  pending=true;pendingBoundary=boundary;++copyAdmissions;
 }
 void observed(Boundary boundary,bool equal){require(pending&&pendingBoundary==boundary&&copyAdmissions==samples+1,"Shard1 completed sample boundary differs");
  pending=false;++samples;fault=!equal;if(boundary==Boundary::uploadDrain)phase=0;if(boundary==Boundary::constructorReturn)finalSample=true;
 }
 void finish(){require(!terminal&&!refused&&!pending&&phase==0&&(fault||(finalSample&&samples==maximumSamples&&drained==watch::operationCount&&allocated==watch::operationCount&&allocatedBytes==watch::allocationBytes&&completedUploadBytes==watch::allocationBytes)),"Shard1 terminal boundary incomplete");terminal=true;}
 void refusal(){refused=true;terminal=false;}
};
} // namespace chandra::dc::import_row::shard1
