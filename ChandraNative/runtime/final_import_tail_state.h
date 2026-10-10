// New code, MPL-2.0. Post-import-only sequencing; no early GPU observations.
#pragma once
#include "head_shard_tail_state.h"
namespace chandra::dc::import_row::final_tails {
constexpr uint32_t maximumOperations=732,maximumCopies=tails::targetCount;
constexpr uint32_t maximumRecords=32,maximumRecordBytes=4096;
constexpr uint64_t rowArrayBytes=uint64_t(tails::targetCount)*tails::rowBytes;
constexpr uint64_t maximumArtifactBytes=uint64_t(maximumRecords)*maximumRecordBytes+2*rowArrayBytes+tails::rowBytes;
struct Counts {
 uint32_t admitted=0,APIcallsAdmitted=0,returned=0,drained=0,sources=0,copyAdmissions=0,copyReturns=0,copiesSubmitted=0,samples=0,matched=0;
 uint64_t storageBytes=0;bool APIpending=false,copyPending=false,fault=false,refused=false,terminal=false;
 void begin(uint32_t ordinal) {tails::require(!APIpending&&!copyPending&&!fault&&!refused&&!terminal&&ordinal==admitted+1&&admitted<maximumOperations,"Final-tail API admission order differs");++admitted;APIpending=true;}
 void source(uint32_t i) {tails::require(APIpending&&i==sources&&i<tails::targetCount,"Final-tail source snapshot order differs");++sources;}
 void beforeCall() {tails::require(APIpending&&APIcallsAdmitted+1==admitted&&returned+1==admitted,"Final-tail API call admission differs");++APIcallsAdmitted;}
 void APIreturned() {tails::require(APIpending&&APIcallsAdmitted==admitted&&returned+1==admitted,"Final-tail API return order differs");++returned;}
 void drain(uint64_t bytes) {tails::require(APIpending&&returned==admitted&&drained+1==admitted&&bytes>0&&bytes<=128ull*1024*1024&&storageBytes<=tails::originalAllocationBytes&&bytes<=tails::originalAllocationBytes-storageBytes,"Final-tail original drain order/extent differs");storageBytes+=bytes;++drained;APIpending=false;}
 void copy(uint32_t i) {tails::require(!APIpending&&!copyPending&&!fault&&!refused&&!terminal&&drained==maximumOperations&&storageBytes==tails::originalAllocationBytes&&sources==tails::targetCount&&i==samples&&copyAdmissions<maximumCopies,"Final-tail copy requires complete import and ordered target");++copyAdmissions;copyPending=true;}
 void copyReturned(uint32_t copies) {tails::require(copyPending&&copyReturns+1==copyAdmissions&&copies==1,"Final-tail direct copy return differs");++copyReturns;copiesSubmitted+=copies;}
 void observed(bool equal) {tails::require(copyPending&&copyReturns==copyAdmissions&&samples<maximumCopies,"Final-tail comparison lacks completed return");++samples;if(equal)++matched;else fault=true;copyPending=false;}
 void finish() {tails::require(!APIpending&&!copyPending&&!terminal&&!refused&&samples>0&&(fault||(samples==maximumCopies&&matched==maximumCopies&&copyAdmissions==maximumCopies&&copyReturns==maximumCopies&&copiesSubmitted==maximumCopies)),"Final-tail terminal boundary incomplete");terminal=true;}
 void refusal() {refused=true;terminal=false;}
};
} // namespace chandra::dc::import_row::final_tails
