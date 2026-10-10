// New code, MPL-2.0. Opt-in original shard1 lifetime observer; implementation is out-of-line in model_weights.cpp.
#pragma once
#include "api.h"
#include "import_row_source.h"
#include "shard1_boundary_state.h"
#include <algorithm>
#include <array>
#include <cstring>
namespace chandra::dc::import_row {
constexpr const char* shard1SourceFile="source.row52427.bf16";
constexpr const char* shard1GoodFile="last-good.row52427.direct.bf16";
constexpr const char* shard1BadFile="first-bad.row52427.direct.bf16";
class Shard1Stop final:public std::runtime_error {public:Shard1Stop():std::runtime_error("Import-only shard1 boundary observer stopped at first byte mismatch"){} };
class Shard1Observer {
 Sink sink;Digest digest;Progress progress;shard1::Counts counts;Buffer target;
 Json operations=Json::array(),current=nullptr,mapping=nullptr,sourceRecord=nullptr,lastGood=nullptr,firstBad=nullptr,firstRefusal=nullptr,lastTransport=nullptr;
 std::array<uint32_t,shard1::rowWords> expected{},good{};direct_head_row::Receipt receipt{};
 uint32_t records=0,writeAttempts=0;bool prepared=false,bound=false,goodAvailable=false,complete=false,journalFailed=false,persistenceFailed=false;
 bool sourceAttempted=false,sourceWritten=false,goodAttempted=false,goodWritten=false,badAttempted=false,badWritten=false,targetReleased=false,importStackClosed=false,identitiesChecked=false;
 Json counters()const;
 Json transport()const;
 void emit(Json record);
 void persistGood();
 void finish();
 void sample(Device& device,shard1::Boundary boundary);
public:
 Shard1Observer(Sink s,Digest d,Progress p);
 static Json forecast();
 void prepare(Json plan,const Json& identity);
 void begin(const Json& op);
 void allocated(const Buffer& b);
 void mapped(const Json& value);
 void beforeUpload(const Buffer& b,const void* pointer,uint64_t bytes);
 void submitting(); void returned();
 void afterDrain(Device& device,const Buffer& b);
 bool needsOwnViewRelease()const;
 void afterSourceRelease(Device& device,bool succeeded);
 void afterImport(Device& device,const Weight& embedding,const Weight& head);
 void failure(const std::string& reason)noexcept;
 void stackClosedAndReleaseTarget();
 bool completed()const;
 Json report()const;
};
} // namespace chandra::dc::import_row
