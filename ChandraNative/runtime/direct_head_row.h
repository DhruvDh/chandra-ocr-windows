// New code, MPL-2.0. Fixed opt-in import checkpoint observer, no shader or SRV read.
#pragma once
#include "weight_storage_experiment.h"
#include "readback_wait.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace chandra::dc::direct_head_row {
constexpr uint32_t row = 26213, columns = 2560, vocabulary = 248320;
constexpr uint32_t rowWords = 1280, rowBytes = 5120, shardRows = 26214;
constexpr uint32_t logicalWords = shardRows * rowWords, firstWord = row * rowWords;
constexpr const char* file = "after_import.row26213.direct.bf16";
struct Range {
 uint32_t firstByte=0,bytes=0,logicalBytes=0,physicalBytes=0;
 template<class Box> Box box()const{return {firstByte,0,0,firstByte+bytes,1,1};}
};
// uint64 intermediates reject overflow and invisible padding before a D3D call.
inline Range range(uint32_t words,uint32_t physicalBytes,uint32_t begin,uint32_t count){
 const auto extent=weight_storage::geometry(words,physicalBytes);
 if(!count||count>rowWords||uint64_t(begin)+count>words)throw std::invalid_argument("Direct row range outside exact logical extent or 5120-byte cap");
 return {uint32_t(uint64_t(begin)*4),count*4,extent.logicalBytes,extent.physicalBytes};
}
inline void requireKnown(uint32_t rows,uint32_t cols,uint32_t firstShardRow,uint64_t shardElements,uint32_t words,bool bf16){
 if(rows!=vocabulary||cols!=columns||firstShardRow!=0||shardElements!=uint64_t(shardRows)*columns||words!=logicalWords||!bf16)
  throw std::invalid_argument("Direct head-row observer requires pinned shard0 row26213 topology");
}
// This receipt records the synchronous observer only. Worker/Job retirement is external evidence.
struct Receipt {
 Range extent; uint32_t sourceUsage=0,sourceBindFlags=0,sourceMiscFlags=0,sourceCPUAccess=0,sourceStride=0;
 uint32_t copiesSubmitted=0; bool beforeCopyDrainCompleted=false,afterCopyDrainCompleted=false;
 bool stagingCreated=false,stagingReleased=false,mapAttempted=false,unmapped=false;
 readback::Outcome readiness; bool deviceRemovedReasonQueried=false; int32_t deviceRemovedReason=0; int64_t operationNanoseconds=0;
};
struct Observation {
 Receipt transport; bool rawFileWritten=false,sourceReadCompleted=false,complete=false;
 std::string gpuSha256,sourceSha256; uint32_t differingWords=0,firstDifferingWord=UINT32_MAX;
};
// Preserve copy/hash and a successfully flushed raw file before source/comparison failures.
// SourceOffset is relative to the payload of the CLI's already authenticated held model file.
template<class Direct,class Write,class Source,class Digest>
void capture(const Direct& direct,const Write& write,const Source& read,const Digest& sha,uint64_t sourceOffset,Observation& out){
 const auto gpu=direct(out.transport);
 if(gpu.size()!=rowWords)throw std::runtime_error("Direct head row readback incomplete");
 out.gpuSha256=sha(gpu.data(),rowBytes);
 write(file,gpu.data(),rowBytes);out.rawFileWritten=true;
 std::vector<uint32_t> source(rowWords);read(sourceOffset,source.data(),rowBytes);out.sourceReadCompleted=true;
 out.sourceSha256=sha(source.data(),rowBytes);
 for(uint32_t i=0;i<rowWords;++i)if(gpu[i]!=source[i]){++out.differingWords;if(out.firstDifferingWord==UINT32_MAX)out.firstDifferingWord=i;}
 out.complete=true;
}

}
