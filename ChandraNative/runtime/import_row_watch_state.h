// New code, MPL-2.0. Small portable sequencing core; no driver calls or alternate upload.
#pragma once
#include <cstdint>
#include <stdexcept>
namespace chandra::dc::import_row::watch {
constexpr uint32_t tensorCount=739, graphCount=724, uniqueTensorCount=723, operationCount=732;
constexpr uint64_t allocationBytes=9078531072ull, fileBytes=10591220088ull;
constexpr uint32_t maximumSamples=operationCount+1, maximumJournalRecords=1470, maximumRecordBytes=4096;
constexpr uint32_t maximumPlanBytes=1024*1024, rawBytes=5120;
constexpr uint64_t maximumArtifactBytes=uint64_t(maximumJournalRecords)*maximumRecordBytes+maximumPlanBytes+3*rawBytes;
inline void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
struct Counts {
 uint32_t admitted=0, allocated=0, uploadAdmitted=0, uploadReturned=0, drained=0, sampled=0, copiesAdmitted=0;
 uint64_t allocatedBytes=0, completedUploadBytes=0;
 uint32_t phase=0; bool finalSample=false, fault=false, terminal=false;
 void begin(uint32_t ordinal,uint64_t bytes){require(!terminal&&!fault&&phase==0&&ordinal==admitted+1&&ordinal<=operationCount&&bytes>0&&bytes<=134217728,"Watch operation admission order/extent differs");++admitted;phase=1;}
 void allocation(uint64_t bytes){require(phase==1&&bytes>0&&bytes<=allocationBytes-allocatedBytes,"Watch allocation order/extent differs");++allocated;allocatedBytes+=bytes;phase=2;}
 void mapped(){require(phase==2,"Watch source map order differs");phase=3;}
 void submit(){require(phase==3,"Watch upload admission order differs");++uploadAdmitted;phase=4;}
 void returned(){require(phase==4,"Watch upload return order differs");++uploadReturned;phase=5;}
 void drain(uint64_t bytes){require(phase==5&&bytes<=allocationBytes-completedUploadBytes,"Watch completed drain order/extent differs");++drained;completedUploadBytes+=bytes;phase=6;}
 void copy(){require(!terminal&&!fault&&((phase==6&&!finalSample)||(phase==0&&drained==operationCount&&!finalSample))&&copiesAdmitted==sampled,"Watch read admission order differs; no replay");++copiesAdmitted;}
 void sample(bool equal){require(phase==6&&copiesAdmitted==sampled+1,"Watch sample completion order differs");++sampled;phase=0;fault=!equal;}
 void final(bool equal){require(phase==0&&drained==operationCount&&sampled==operationCount&&copiesAdmitted==maximumSamples&&!finalSample,"Watch final sample order differs");++sampled;finalSample=true;fault=!equal;}
 void finish(){require(!terminal&&phase==0&&(fault||(finalSample&&sampled==maximumSamples&&allocated==operationCount&&allocatedBytes==allocationBytes&&completedUploadBytes==allocationBytes)),"Watch terminal boundary incomplete");terminal=true;}
};
inline void range(uint64_t offset,uint64_t bytes){require(offset<=fileBytes&&bytes<=fileBytes-offset&&bytes>0&&bytes<=134217728,"Watch held-file range differs");}
} // namespace chandra::dc::import_row::watch
