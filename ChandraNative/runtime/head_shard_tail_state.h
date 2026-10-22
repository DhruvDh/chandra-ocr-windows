// New code, MPL-2.0. Fixed ten-tail sequencing; no driver or alternative upload.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
namespace chandra::dc::import_row::tails {
constexpr uint32_t targetCount=10, vocabulary=248320, columns=2560, fullRows=26214;
constexpr uint32_t rowWords=1280, rowBytes=5120, maximumRecords=32, maximumRecordBytes=4096;
constexpr uint32_t maximumAdditionalCopies=18, maximumSamples=21;
constexpr uint64_t fileBytes=10591220088ull, payloadBegin=91512, tensorBegin=1271398400;
constexpr uint64_t tensorBytes=uint64_t(vocabulary)*rowBytes, originalAllocationBytes=9078531072ull;
constexpr uint64_t maximumArtifactBytes=uint64_t(maximumRecords)*maximumRecordBytes+3*rowBytes;
inline void require(bool condition,const char* reason){if(!condition)throw std::runtime_error(reason);}
struct Target {uint32_t index,firstRow,rows,row,words,firstWord,bytes;uint64_t fileOffset,rowFileOffset;};
inline Target target(uint32_t index){
 require(index<targetCount,"Tail index outside pinned original head shards");
 const uint32_t first=index*fullRows,rows=index+1==targetCount?vocabulary-first:fullRows;
 const uint32_t bytes=rows*rowBytes,firstWord=(rows-1)*rowWords;
 const uint64_t offset=payloadBegin+tensorBegin+uint64_t(first)*rowBytes;
 require(rows&&uint64_t(firstWord)*4+rowBytes==bytes&&offset<=fileBytes&&bytes<=fileBytes-offset,"Tail source/resource range differs");
 return {index,first,rows,first+rows-1,rows*rowWords,firstWord,bytes,offset,offset+uint64_t(firstWord)*4};
}
struct Counts {
 std::array<bool,targetCount> sources{},initial{},final{};
 uint32_t sourceCount=0,initialCount=0,finalCount=0,samples=0,directAdmissions=0,directReturns=0,submittedCopies=0,reusedSamples=0;
 bool fault=false,refused=false,terminal=false,pending=false,pendingLast=false,pendingReused=false;uint32_t pendingTarget=0;
 void source(uint32_t i){require(!terminal&&!fault&&!refused&&!pending&&i==sourceCount&&i==initialCount&&i<targetCount&&!sources[i],"Tail CPU source capture order differs");sources[i]=true;++sourceCount;}
 void admit(uint32_t i,bool last,bool reused){
  require(!terminal&&!fault&&!refused&&!pending&&i<targetCount&&sources[i],"Tail read admission after stop or without source");
  require(last?(sourceCount==targetCount&&initialCount==targetCount&&i==finalCount&&!final[i]):(i==initialCount&&!initial[i]),"Tail initial/final read order differs");
  if(!reused){require(i!=0&&directAdmissions<maximumAdditionalCopies,"Tail copy bound or shard0 duplication");++directAdmissions;}
  pending=true;pendingTarget=i;pendingLast=last;pendingReused=reused;
 }
 void observed(uint32_t i,bool last,bool reused,bool equal,uint32_t copies){
  require(!terminal&&!fault&&!refused&&pending&&pendingTarget==i&&pendingLast==last&&pendingReused==reused&&i<targetCount&&samples<maximumSamples&&copies==1,"Tail completed sample boundary differs");
  if(reused)++reusedSamples;else require(directReturns==directAdmissions,"Tail sample lacks observed read return");
  require(last?(i==finalCount&&!final[i]):(i==initialCount&&!initial[i]),"Tail completed sample order differs");
  if(last){final[i]=true;++finalCount;}else{initial[i]=true;++initialCount;}
  ++samples;fault=!equal;pending=false;
 }
 void returned(uint32_t copies){require(pending&&!pendingReused&&directReturns<directAdmissions&&copies<=1,"Tail read return admission/receipt differs");++directReturns;submittedCopies+=copies;}
 void failedCopy(uint32_t copies){require(copies<=1,"Tail failed read copy receipt differs");submittedCopies+=copies;}
 void legacyFault(){require(!terminal&&!fault&&!refused&&!pending&&sources[0]&&initial[0]&&samples<maximumSamples,"Tail legacy fault boundary differs");++samples;++reusedSamples;fault=true;}
 void finish(){require(!terminal&&!refused&&!pending&&samples>0&&(fault||(sourceCount==targetCount&&initialCount==targetCount&&finalCount==targetCount&&samples==2*targetCount&&directAdmissions==maximumAdditionalCopies&&directReturns==maximumAdditionalCopies&&submittedCopies==maximumAdditionalCopies&&reusedSamples==2)),"Tail terminal boundary incomplete");terminal=true;}
 void refusal(){refused=true;terminal=false;}
};
} // namespace chandra::dc::import_row::tails
