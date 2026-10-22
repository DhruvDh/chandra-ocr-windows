// New code, MPL-2.0. Opt-in import-only observer of one original Storage; no repair or replay.
#pragma once
#include "api.h"
#include "import_row_source.h"
#include "import_row_watch_state.h"
#include "head_shard_tail_audit.h"
#include <algorithm>
#include <array>
#include <memory>
namespace chandra::dc::import_row {
constexpr const char* watchExpectedSHA="e696b6bda9b22dba3ad2a49132d6353302f5b20ad2627909ae5474586a46c07f";
constexpr const char* watchInitialFile="initial.row26213.direct.bf16";
constexpr const char* watchGoodFile="last-good.row26213.direct.bf16";
constexpr const char* watchBadFile="first-bad.row26213.direct.bf16";
class WatchStop final : public std::runtime_error { public: WatchStop():std::runtime_error("Import-only watcher stopped at first observed mismatch"){} };
class WatchObserver {
 Sink sink; Digest digest; Progress progress; watch::Counts counts; HeadTailObserver* headTails=nullptr;
 Buffer target; Json operations=Json::array(), current=nullptr, lastGood=nullptr, firstBad=nullptr;
 Json state={{"schema","chandra.directcompute.import-row-watch.v1"},{"requested",true},{"complete",false},{"outcome","partial_or_refused"},
  {"row",row},{"shard_index",0},{"first_word",direct_head_row::firstWord},{"word_count",direct_head_row::rowWords},{"row_bytes",rowBytes},
  {"expected_source_sha256",watchExpectedSHA},{"allocation_selector","exact"},{"request_replayed",false},{"error",nullptr},
  {"initial_raw_written_and_flushed",false},{"last_good_raw_written_and_flushed",false},{"first_bad_raw_written_and_flushed",false},
  {"plan_written_and_flushed",false},{"journal_records_written",0},{"target_reference_released",false},
  {"scope","Import-only cutoff observations; observer can change order/residency; no repair, arithmetic, driver cause, numerical/OCR or root retirement acceptance"}};
 direct_head_row::Receipt receipt; std::array<uint32_t,direct_head_row::rowWords> initial{},good{};
 uint32_t records=0; bool prepared=false,bound=false,goodAvailable=false,goodWriteAttempted=false,badWriteAttempted=false;
 Json counters()const{return {{"operations_admitted",counts.admitted},{"original_allocations_completed",counts.allocated},{"original_allocation_bytes",counts.allocatedBytes},
  {"original_upload_calls_admitted",counts.uploadAdmitted},{"original_upload_returns_observed",counts.uploadReturned},{"original_upload_drains_completed",counts.drained},
  {"completed_original_upload_bytes",counts.completedUploadBytes},{"completed_samples",counts.sampled},{"direct_read_call_admissions",counts.copiesAdmitted},{"current_operation_phase",counts.phase}};}
 Json transport()const{
  const auto& r=receipt;return {{"source_first_byte",r.extent.firstByte},{"copy_bytes",r.extent.bytes},{"source_logical_bytes",r.extent.logicalBytes},{"source_physical_bytes",r.extent.physicalBytes},
   {"source_descriptor",{{"usage",r.sourceUsage},{"bind_flags",r.sourceBindFlags},{"misc_flags",r.sourceMiscFlags},{"cpu_access_flags",r.sourceCPUAccess},{"structure_byte_stride",r.sourceStride}}},
   {"copies_submitted",r.copiesSubmitted},{"before_copy_drain_completed",r.beforeCopyDrainCompleted},{"after_copy_drain_completed",r.afterCopyDrainCompleted},
   {"staging_created",r.stagingCreated},{"staging_released",r.stagingReleased},{"map_attempted",r.mapAttempted},{"map_status",r.mapAttempted?Json(uint32_t(r.readiness.status)):Json(nullptr)},
   {"map_attempts",r.readiness.attempts},{"still_drawing_results",r.readiness.stillDrawing},{"map_wait_nanoseconds",r.readiness.waited.count()},
   {"map_copied",r.mapAttempted&&r.readiness.result==readback::Result::copied},{"unmapped",r.unmapped},{"operation_nanoseconds",r.operationNanoseconds},
   {"device_removed_reason_queried",r.deviceRemovedReasonQueried},{"device_removed_reason",r.deviceRemovedReasonQueried?Json(uint32_t(r.deviceRemovedReason)):Json(nullptr)},
   {"copy_resubmitted",false},{"retirement_scope","Local staging reference only; root owns process/Job/channel retirement"}};
 }
 void emit(Json record){
  watch::require(records<watch::maximumJournalRecords,"Watch journal record count exceeded");
  record["schema"]="chandra.directcompute.import-row-watch-journal.v1";record["record_index"]=records+1;record["counts"]=counters();
  watch::require(record.dump().size()+1<=watch::maximumRecordBytes,"Watch journal record bytes exceeded");progress(record);++records;state["journal_records_written"]=records;
 }
 void persistGood(){if(goodAvailable&&!goodWriteAttempted){goodWriteAttempted=true;sink(watchGoodFile,good.data(),rowBytes);state["last_good_raw_written_and_flushed"]=true;}}
 void terminal(){persistGood();counts.finish();state["outcome"]=counts.fault?"first_fault":"no_fault_completed_import";
  emit({{"event","terminal"},{"outcome",state.at("outcome")},{"last_good",lastGood},{"first_bad",firstBad},
   {"initial_raw_written_and_flushed",state.at("initial_raw_written_and_flushed")},{"last_good_raw_written_and_flushed",state.at("last_good_raw_written_and_flushed")},{"first_bad_raw_written_and_flushed",state.at("first_bad_raw_written_and_flushed")}});
  state["complete"]=true;
 }
 void sample(Device& device,bool final){
  counts.copy();receipt={};state["observation_in_progress"]=true;
  try{
   const auto words=device.readWordsRange(target,direct_head_row::firstWord,direct_head_row::rowWords,receipt);state["last_transport"]=transport();
   watch::require(words.size()==direct_head_row::rowWords&&receipt.extent.firstByte==rowByteOffset&&receipt.extent.bytes==rowBytes&&receipt.extent.logicalBytes==shardBytes&&receipt.extent.physicalBytes==shardBytes&&
    receipt.copiesSubmitted==1&&receipt.beforeCopyDrainCompleted&&receipt.afterCopyDrainCompleted&&receipt.stagingReleased&&receipt.mapAttempted&&receipt.readiness.result==readback::Result::copied&&receipt.unmapped,"Watch direct-row transport incomplete or extent differs");
   const std::string sha=digest(words.data(),rowBytes);watch::require(sha.size()==64,"Watch digest extent differs");const bool equal=sha==watchExpectedSHA;
   const bool first=counts.sampled==0;
   if(final)counts.final(equal);else counts.sample(equal);
   if(first){std::copy(words.begin(),words.end(),initial.begin());state["initial_sha256"]=sha;sink(watchInitialFile,words.data(),rowBytes);state["initial_raw_written_and_flushed"]=true;}
   Json observed={{"sample_index",counts.sampled},{"at",final?"after_constructor_return_source_maps_released_before_arithmetic":"after_original_upload_drain_current_source_view_alive_before_next_import"},
    {"operation_ordinal",counts.drained},{"source_view_scope_alive",!final},{"sha256",sha},{"equals_expected_CPU_source",equal}};
   observed["counts_at_cutoff"]=counters();
   if(!equal){uint32_t different=0,firstDifferent=UINT32_MAX,lastDifferent=0;for(uint32_t i=0;i<direct_head_row::rowWords;++i)if(words[i]!=initial[i]){++different;firstDifferent=std::min(firstDifferent,i);lastDifferent=i;}observed["different_u32_words_vs_initial"]=different;observed["first_different_word_vs_initial"]=different?Json(firstDifferent):Json(nullptr);observed["last_different_word_vs_initial"]=different?Json(lastDifferent):Json(nullptr);}
   if(!equal){firstBad=observed;firstBad["operation"]=current;badWriteAttempted=true;sink(watchBadFile,words.data(),rowBytes);state["first_bad_raw_written_and_flushed"]=true;}
   else{std::copy(words.begin(),words.end(),good.begin());goodAvailable=true;lastGood=observed;lastGood["operation"]=current;}
   state["observation_in_progress"]=false;
   emit({{"event",final?"after_full_import_sample":"completed_upload_sample"},{"operation",current},{"observation",observed},{"current_source_mapping",state.at("current_source_mapping")},{"transport",transport()}});
   if(headTails&&(first||final||!equal))headTails->reused(words,receipt,first,final,counters());
   if(!equal){terminal();throw WatchStop();}
   if(final)terminal();
  }catch(...){state["last_transport"]=transport();if(headTails)headTails->legacyStackExit(receipt,counters());throw;}
 }
public:
 WatchObserver(Sink s,Digest d,Progress p,HeadTailObserver* tailObserver=nullptr):sink(std::move(s)),digest(std::move(d)),progress(std::move(p)),headTails(tailObserver){watch::require(bool(sink)&&bool(digest)&&bool(progress),"Watch callbacks required");}
 void prepare(Json plan,const Json& deviceIdentity){
  watch::require(!prepared&&plan.at("operations").size()==watch::operationCount&&plan.at("source_tensor_count")==watch::tensorCount&&plan.at("effective_tensor_count")==watch::graphCount&&plan.at("unique_upload_tensor_count")==watch::uniqueTensorCount&&plan.at("allocation_bytes")==watch::allocationBytes&&plan.at("allocation_selector")=="exact","Watch pinned inventory/plan differs");
  operations=plan.at("operations");uint64_t bytes=0;uint32_t ordinal=0,tensorOrdinal=0;std::string previous;
  for(const auto& op:operations){
   watch::require(op.at("operation_ordinal")==++ordinal&&op.at("tensor_name").is_string()&&op.at("tensor_name").get<std::string>().size()<=256,"Watch plan order/name differs");
   const auto name=op.at("tensor_name").get<std::string>();if(name!=previous){watch::require(previous.empty()||previous<name,"Watch tensor order differs");previous=name;++tensorOrdinal;}
   watch::require(op.at("tensor_ordinal")==tensorOrdinal&&op.at("allocation_bytes")==op.at("logical_storage_bytes")&&op.at("logical_storage_bytes").get<uint64_t>()==uint64_t(op.at("words").get<uint32_t>())*4,"Watch plan tensor/extents differ");
   watch::range(op.at("source_file_offset").get<uint64_t>(),op.at("source_bytes").get<uint64_t>());
   const auto n=op.at("allocation_bytes").get<uint64_t>();watch::require(n<=watch::allocationBytes-bytes,"Watch plan cumulative extent differs");bytes+=n;
  }
  watch::require(tensorOrdinal==watch::uniqueTensorCount&&bytes==watch::allocationBytes&&operations.at(0).at("tensor_name")=="model.language_model.embed_tokens.weight"&&operations.at(0).at("chunk_index")==0,"Watch pinned total/first operation differs");
  plan["schema"]="chandra.directcompute.import-row-watch-plan.v1";plan["device_identity"]=deviceIdentity;
  plan["observer_limits"]={{"maximum_operations",watch::operationCount},{"maximum_samples",watch::maximumSamples},{"maximum_plan_bytes",watch::maximumPlanBytes},{"maximum_journal_records",watch::maximumJournalRecords},{"maximum_record_bytes",watch::maximumRecordBytes},{"maximum_raw_files",3},{"raw_file_bytes",rowBytes},{"maximum_artifact_bytes",watch::maximumArtifactBytes}};
  const auto raw=plan.dump()+"\n";watch::require(raw.size()<=watch::maximumPlanBytes,"Watch plan byte bound exceeded");
  sink("plan.json",raw.data(),raw.size());state["plan_written_and_flushed"]=true;state["plan_sha256"]=digest(raw.data(),raw.size());state["plan_bytes"]=raw.size();
  state["model_binding"]=plan.at("model_binding");state["observer_limits"]=plan.at("observer_limits");prepared=true;
  emit({{"event","plan_persisted"},{"plan_sha256",state.at("plan_sha256")},{"plan_bytes",raw.size()},{"maximum_operations",watch::operationCount}});
  if(headTails)headTails->prepare(plan);
 }
 void begin(const Json& actual){watch::require(prepared&&counts.admitted<operations.size()&&actual==operations.at(counts.admitted),"Watch actual upload differs from authenticated ordered plan");
  counts.begin(actual.at("operation_ordinal").get<uint32_t>(),actual.at("allocation_bytes").get<uint64_t>());current=actual;state["source_map_alive"]=false;
  emit({{"event","operation_admitted"},{"operation",current},{"last_good_sample_index",lastGood.is_null()?Json(nullptr):lastGood.at("sample_index")},{"admission_is_not_API_execution_proof",true}});
 }
 void allocated(const Buffer& buffer){watch::require(buffer.storage&&buffer.words==current.at("words")&&buffer.logicalElements==current.at("logical_elements")&&buffer.packedBF16,"Watch allocated logical descriptor differs");counts.allocation(current.at("allocation_bytes").get<uint64_t>());}
 void mapped(const Json& mapping){counts.mapped();state["current_source_mapping"]=mapping;state["source_map_alive"]=true;}
 void beforeUpload(const Buffer& buffer,const void* source,uint64_t bytes){if(headTails)headTails->beforeUpload(current,state.at("current_source_mapping"),buffer,source,bytes,counters());}
 void submitting(){counts.submit();}void returned(){counts.returned();}
 void afterDrain(Device& device,const Buffer& buffer,const Observer& source){
  counts.drain(current.at("allocation_bytes").get<uint64_t>());
  if(!bound){watch::require(source.completed(),"Watch initial sample requires complete CPU source capture");const auto cpu=source.report();
   watch::require(cpu.at("captures").size()==3&&cpu.at("whole_file_hash_and_tie_equality_passed_before_capture")==true,"Watch CPU source authentication incomplete");
   for(const auto& c:cpu.at("captures"))watch::require(c.at("sha256")==watchExpectedSHA&&c.at("bytes")==rowBytes&&c.at("raw_file_written_and_flushed")==true,"Watch CPU source baseline differs");
   direct_head_row::requireKnown(248320,2560,0,buffer.logicalElements,buffer.words,buffer.packedBF16);target=buffer;bound=true;
   state["source_range"]=cpu.at("source_range");state["source_buffer"]={{"words",buffer.words},{"logical_elements",buffer.logicalElements},{"packed_BF16",buffer.packedBF16},
    {"identity_token","original_embedding_shard0_Storage"},{"identity_scope","Retained shared_ptr to original Storage; no integer address or reconstructed resource used"}};
  }
  sample(device,false);
  if(headTails)headTails->afterOriginalDrain(device,buffer,current,counters());
 }
 void afterImport(Device& device,const Buffer& embedding,const Buffer& head){
  watch::require(bound&&target.storage==embedding.storage&&target.storage==head.storage,"Watch original embedding/head Storage identity changed");
  state["same_storage_after_import_checked"]=true;state["same_storage_after_import"]=true;state["source_map_alive"]=false;state["constructor_return_observed"]=true;sample(device,true);
 }
 void failure(const std::string& reason)noexcept{
  try{state["complete"]=false;state["outcome"]="partial_or_refused";state["error"]=reason;persistGood();emit({{"event","failure"},{"operation",current},{"error",reason},{"last_good",lastGood},{"first_bad",firstBad},{"source_map_alive_at_last_observed_boundary",state.value("source_map_alive",false)}});}
  catch(...){try{state["failure_persistence_or_journal_failed"]=true;}catch(...){}}
 }
 void releaseTarget(){target={};state["target_reference_released"]=true;state["source_map_alive"]=false;state["import_stack_unwound_or_returned"]=true;}
 bool completed()const{return counts.terminal&&state.at("complete")==true;}
 Json report()const{Json out=state;out["counts"]=counters();out["current_operation"]=current;out["last_good"]=lastGood;out["first_bad"]=firstBad;return out;}
};
} // namespace chandra::dc::import_row
