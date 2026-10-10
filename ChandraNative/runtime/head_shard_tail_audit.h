// New code, MPL-2.0. Optional original-shard direct observer; never a repair or inference.
#pragma once
#include "api.h"
#include "import_row_source.h"
#include "head_shard_tail_state.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
namespace chandra::dc::import_row {
constexpr const char* tailSourceFile="fault.source.bf16";
constexpr const char* tailGoodFile="fault.last-good.direct.bf16";
constexpr const char* tailBadFile="fault.first-bad.direct.bf16";
class HeadTailStop final:public std::runtime_error {public:HeadTailStop():std::runtime_error("Import-only head-tail observer stopped at first byte mismatch"){} };
class HeadTailObserver {
 Sink sink;Digest digest;Progress progress;tails::Counts counts;
 std::array<Buffer,tails::targetCount> targets{};
 std::array<std::array<uint32_t,tails::rowWords>,tails::targetCount> expected{},good{};
 std::array<bool,tails::targetCount> goodAvailable{},sourceCopied{};uint32_t activeTarget=0;
 Json plan=Json::array(),sourceRecords=Json::array(),lastGoods=Json::array(),firstBad=nullptr,firstRefusal=nullptr;
 Json originalCounts=nullptr,lastTransport=nullptr;
 uint32_t records=0,writeAttempts=0;bool prepared=false,journalFailed=false,failurePersistenceFailed=false,complete=false;
 bool sourceWriteAttempted=false,sourceWritten=false,goodWriteAttempted=false,goodWritten=false,badWriteAttempted=false,badWritten=false;
 bool originalIdentitiesChecked=false,targetsReleased=false,sourceMapsReleased=false,targetContextObserved=false;
 Json counterReport()const{return {{"CPU_source_snapshots_completed",counts.sourceCount},{"initial_samples_completed",counts.initialCount},{"final_samples_completed",counts.finalCount},
  {"completed_selected_samples",counts.samples},{"additional_direct_read_call_admissions",counts.directAdmissions},{"additional_direct_read_returns_observed",counts.directReturns},
  {"additional_copies_submitted_observed",counts.submittedCopies},{"legacy_samples_reused",counts.reusedSamples},{"comparison_boundary_pending",counts.pending},{"journal_records_written",records},{"journal_write_attempts",writeAttempts}};}
 static Json transport(const direct_head_row::Receipt& r){return {{"source_first_byte",r.extent.firstByte},{"copy_bytes",r.extent.bytes},{"source_logical_bytes",r.extent.logicalBytes},{"source_physical_bytes",r.extent.physicalBytes},
  {"source_descriptor",{{"usage",r.sourceUsage},{"bind_flags",r.sourceBindFlags},{"misc_flags",r.sourceMiscFlags},{"cpu_access_flags",r.sourceCPUAccess},{"structure_byte_stride",r.sourceStride}}},
  {"copies_submitted",r.copiesSubmitted},{"before_copy_drain_completed",r.beforeCopyDrainCompleted},{"after_copy_drain_completed",r.afterCopyDrainCompleted},
  {"map_attempted",r.mapAttempted},{"map_status",r.mapAttempted?Json(uint32_t(r.readiness.status)):Json(nullptr)},{"map_attempts",r.readiness.attempts},{"still_drawing_results",r.readiness.stillDrawing},
  {"map_copied",r.mapAttempted&&r.readiness.result==readback::Result::copied},{"unmapped",r.unmapped},{"staging_created",r.stagingCreated},{"staging_released",r.stagingReleased},
  {"device_removed_reason_queried",r.deviceRemovedReasonQueried},{"device_removed_reason",r.deviceRemovedReasonQueried?Json(uint32_t(r.deviceRemovedReason)):Json(nullptr)},
  {"operation_nanoseconds",r.operationNanoseconds},{"copy_resubmitted",false},{"retirement_scope","Local staging reference only; model/native/Job retirement separate"}};}
 void emit(Json record){
  tails::require(!journalFailed&&records<tails::maximumRecords,"Tail journal bound or prior torn/failed write");
  record["schema"]="chandra.directcompute.head-shard-tail-journal.v1";record["record_index"]=records+1;record["counts"]=counterReport();
  tails::require(record.dump().size()+1<=tails::maximumRecordBytes,"Tail journal byte bound exceeded");++writeAttempts;
  try{progress(record);}catch(...){journalFailed=true;throw;}++records;
 }
 void faultRows(uint32_t i,const std::vector<uint32_t>& words){
  tails::require(!sourceWriteAttempted&&!badWriteAttempted,"Tail fault raw persistence never replays");sourceWriteAttempted=true;sink(tailSourceFile,expected[i].data(),tails::rowBytes);sourceWritten=true;
  if(goodAvailable[i]){goodWriteAttempted=true;sink(tailGoodFile,good[i].data(),tails::rowBytes);goodWritten=true;}
  badWriteAttempted=true;sink(tailBadFile,words.data(),tails::rowBytes);badWritten=true;
 }
 void finish(){counts.finish();emit({{"event","terminal"},{"outcome",counts.fault?"first_fault":"no_fault_completed_selected_tails"},{"original_counts",originalCounts},
  {"source_raw_written_and_flushed",sourceWritten},{"last_good_raw_written_and_flushed",goodWritten},{"first_bad_raw_written_and_flushed",badWritten}});complete=true;}
 void observed(uint32_t i,const std::vector<uint32_t>& words,const direct_head_row::Receipt& r,const char* phase,bool reused,bool last,bool legacy){
  const auto t=tails::target(i);lastTransport=transport(r);
  tails::require(words.size()==tails::rowWords&&r.extent.firstByte==t.firstWord*4&&r.extent.bytes==tails::rowBytes&&r.extent.logicalBytes==t.bytes&&r.extent.physicalBytes==t.bytes&&
   r.copiesSubmitted==1&&r.beforeCopyDrainCompleted&&r.afterCopyDrainCompleted&&r.stagingReleased&&r.mapAttempted&&r.readiness.result==readback::Result::copied&&r.unmapped,"Tail direct-row transport incomplete or extent differs");
  const bool equal=std::equal(words.begin(),words.end(),expected[i].begin());const std::string sha=digest(words.data(),tails::rowBytes);tails::require(sha.size()==64,"Tail GPU digest extent differs");
  if(legacy){tails::require(!equal,"Legacy tail-fault reuse requires byte mismatch");counts.legacyFault();}else counts.observed(i,last,reused,equal,r.copiesSubmitted);
  Json sample={{"phase",phase},{"shard_index",i},{"global_row",t.row},{"local_row",t.rows-1},{"source_record_index",i},{"gpu_sha256",sha},{"source_sha256",sourceRecords.at(i).at("sha256")},
   {"byte_equal_to_actual_upload_source",equal},{"reused_original_shard0_sample",reused},{"target_source_view_alive",std::string(phase)=="initial"},
   {"current_upload_source_view_alive",!sourceMapsReleased},{"source_maps_released",sourceMapsReleased},{"original_counts",originalCounts}};
  if(!equal){uint32_t n=0,first=UINT32_MAX,lastWord=0;for(uint32_t k=0;k<tails::rowWords;++k)if(words[k]!=expected[i][k]){++n;first=std::min(first,k);lastWord=k;}
   sample["different_u32_words"]=n;sample["first_different_word"]=first;sample["last_different_word"]=lastWord;firstBad=sample;faultRows(i,words);
  }else{std::copy(words.begin(),words.end(),good[i].begin());goodAvailable[i]=true;lastGoods.at(i)=sample;}
  emit({{"event","sample"},{"observation",sample},{"transport",lastTransport}});if(!equal)finish();
 }
 void direct(Device& device,uint32_t i,bool last){
  activeTarget=i;targetContextObserved=true;counts.admit(i,last,false);direct_head_row::Receipt r{};bool returned=false;
  try{const auto words=device.readWordsRange(targets[i],tails::target(i).firstWord,tails::rowWords,r);counts.returned(r.copiesSubmitted);returned=true;observed(i,words,r,last?"after_import":"initial",false,last,false);}
  catch(...){if(!returned)counts.failedCopy(r.copiesSubmitted);lastTransport=transport(r);throw;}
  if(counts.fault)throw HeadTailStop();
 }
public:
 HeadTailObserver(Sink s,Digest d,Progress p):sink(std::move(s)),digest(std::move(d)),progress(std::move(p)){tails::require(bool(sink)&&bool(digest)&&bool(progress),"Tail callbacks required");for(uint32_t i=0;i<tails::targetCount;++i)lastGoods.push_back(nullptr);}
 static Json forecast(){Json targetsPlan=Json::array();for(uint32_t i=0;i<tails::targetCount;++i){const auto t=tails::target(i);targetsPlan.push_back({{"shard_index",i},{"global_row",t.row},{"first_word",t.firstWord},{"original_resource_bytes",t.bytes},{"source_row_file_offset",t.rowFileOffset}});}
  return {{"schema","chandra.directcompute.head-shard-tail-plan.v1"},{"requested",true},{"import_only",true},{"allocation_selector","exact"},{"selected_targets",targetsPlan},
   {"maximum_additional_direct_copies",tails::maximumAdditionalCopies},{"legacy_watcher_samples_max",733},{"whole_direct_copy_admissions_max",751},{"maximum_journal_records",tails::maximumRecords},{"maximum_record_bytes",tails::maximumRecordBytes},
   {"maximum_artifact_bytes",tails::maximumArtifactBytes},{"CPU_row_storage_bytes",107520},{"staging_bytes_at_one_time",tails::rowBytes},
   {"leaves",{{"progress.jsonl",uint64_t(tails::maximumRecords)*tails::maximumRecordBytes},{tailSourceFile,tails::rowBytes},{tailGoodFile,tails::rowBytes},{tailBadFile,tails::rowBytes}}},
   {"observation_scope","Ordered selected tails only; copies/CPU snapshots change timing/residency; no whole-model integrity, arithmetic, repair or driver cause"}};
 }
 void prepare(const Json& originalPlan){
  tails::require(!prepared&&originalPlan.at("operations").size()==732&&originalPlan.at("allocation_selector")=="exact"&&originalPlan.at("allocation_bytes")==tails::originalAllocationBytes,"Tail original plan binding differs");
  const auto& model=originalPlan.at("model_binding");tails::require(model.at("model_bytes")==tails::fileBytes&&model.at("model_sha256")=="0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847"&&model.at("payload_begin")==tails::payloadBegin&&model.at("whole_file_hash_performed")==true&&model.at("tied_byte_equality_performed")==true,"Tail held-file authentication differs");
  for(uint32_t i=0;i<tails::targetCount;++i){const auto t=tails::target(i);const auto& op=originalPlan.at("operations").at(i);
   tails::require(op.at("operation_ordinal")==i+1&&op.at("tensor_name")=="model.language_model.embed_tokens.weight"&&op.at("dtype")=="BF16"&&op.at("shape")==Json::array({248320,2560})&&op.at("chunk_index")==i&&op.at("first_row")==t.firstRow&&op.at("rows")==t.rows&&op.at("cols")==tails::columns&&
    op.at("words")==t.words&&op.at("logical_elements")==uint64_t(t.rows)*tails::columns&&op.at("source_bytes")==t.bytes&&op.at("logical_storage_bytes")==t.bytes&&op.at("allocation_bytes")==t.bytes&&op.at("source_file_offset")==t.fileOffset&&op.at("tensor_data_offsets")==Json::array({tails::tensorBegin,tails::tensorBegin+tails::tensorBytes}),"Tail original shard/source topology differs");plan.push_back(op);
  }
  emit({{"event","plan"},{"plan",forecast()},{"whole_file_hash_performed",true},{"tied_byte_equality_performed",true},{"model_sha256",model.at("model_sha256")}});prepared=true;
 }
 void beforeUpload(const Json& op,const Json& mapping,const Buffer& b,const void* source,uint64_t bytes,const Json& original){
  originalCounts=original;
  if(op.at("tensor_name")!="model.language_model.embed_tokens.weight")return;
  const uint32_t i=op.at("chunk_index").get<uint32_t>();const auto t=tails::target(i);
  activeTarget=i;targetContextObserved=true;
  tails::require(prepared&&op==plan.at(i)&&source&&bytes==t.bytes&&b.storage&&b.packedBF16&&b.words==t.words&&b.logicalElements==uint64_t(t.rows)*tails::columns,"Tail upload CPU pointer/resource binding differs");
  const auto gran=mapping.at("allocation_granularity").get<uint32_t>();const auto offset=mapping.at("file_offset").get<uint64_t>(),prefix=mapping.at("skipped_prefix").get<uint64_t>(),mapped=mapping.at("mapped_bytes").get<uint64_t>();
  tails::require(gran&&offset==t.fileOffset-t.fileOffset%gran&&prefix==t.fileOffset-offset&&mapped==prefix+bytes&&offset<=tails::fileBytes&&mapped<=tails::fileBytes-offset,"Tail original source mapping extent differs");
  for(uint32_t k=0;k<i;++k)tails::require(targets[k].storage!=b.storage,"Tail original shards share unexpected Storage");
  activeTarget=i;std::memcpy(expected[i].data(),static_cast<const unsigned char*>(source)+uint64_t(t.firstWord)*4,tails::rowBytes);sourceCopied[i]=true;
  const auto sha=digest(expected[i].data(),tails::rowBytes);tails::require(sha.size()==64,"Tail source digest extent differs");
  const Json checkpointSHA=i==0?Json("e696b6bda9b22dba3ad2a49132d6353302f5b20ad2627909ae5474586a46c07f"):i==1?Json("0575f9588d644d6b669a29517b1dda97ea56b4831b5578edffb99913df8e4ab7"):Json(nullptr);
  const bool knownSourceMatches=checkpointSHA.is_null()||sha==checkpointSHA.get<std::string>();
  counts.source(i);targets[i]=b;Json record={{"shard_index",i},{"global_row",t.row},{"local_row",t.rows-1},{"source_file_offset",t.fileOffset},{"row_file_offset",t.rowFileOffset},{"source_bytes",bytes},{"row_bytes",tails::rowBytes},{"sha256",sha},{"source_mapping",mapping},{"original_upload_operation",op.at("operation_ordinal")},
   {"known_checkpoint_row_sha256",checkpointSHA},{"known_checkpoint_row_hash_matches",checkpointSHA.is_null()?Json(nullptr):Json(knownSourceMatches)},
   {"copied_from_actual_submission_pointer",true},{"source_view_alive_at_snapshot",true},{"source_pointer_redirected",false},{"source_snapshot_owned",true},{"original_Storage_retained",true}};sourceRecords.push_back(record);emit({{"event","before_original_upload_source_snapshot"},{"source",record},{"admission_is_not_upload_execution",true}});
  tails::require(knownSourceMatches,"Tail actual CPU source differs from an independently authenticated checkpoint row; upload refused");
 }
 void reused(const std::vector<uint32_t>& words,const direct_head_row::Receipt& r,bool first,bool last,const Json& original){
  activeTarget=0;targetContextObserved=true;originalCounts=original;if(last)sourceMapsReleased=true;
  if(first||last){counts.admit(0,last,true);observed(0,words,r,last?"after_import":"initial",true,last,false);}
  else observed(0,words,r,"legacy_row0_cutoff",true,false,true);
 }
 void afterOriginalDrain(Device& device,const Buffer& buffer,const Json& op,const Json& original){
  originalCounts=original;if(op.at("tensor_name")!="model.language_model.embed_tokens.weight")return;
  const auto i=op.at("chunk_index").get<uint32_t>();const auto t=tails::target(i);activeTarget=i;targetContextObserved=true;
  tails::require(targets[i].storage==buffer.storage&&buffer.words==t.words&&buffer.logicalElements==uint64_t(t.rows)*tails::columns&&buffer.packedBF16,"Tail actual uploaded Buffer identity/descriptor changed");if(i)direct(device,i,false);
 }
 void afterImport(Device& device,const Weight& embedding,const Weight& head,const Json& original){
  originalCounts=original;sourceMapsReleased=true;
  tails::require(prepared&&!counts.fault&&!counts.refused&&embedding.rows==tails::vocabulary&&embedding.cols==tails::columns&&embedding.bf16&&head.rows==embedding.rows&&head.cols==embedding.cols&&head.bf16&&embedding.shards.size()==tails::targetCount&&head.shards.size()==tails::targetCount&&embedding.shardFirstRows.size()==tails::targetCount&&head.shardFirstRows==embedding.shardFirstRows,"Tail final embedding/head topology differs");
  tails::require(original.at("original_upload_drains_completed")==732&&original.at("original_allocations_completed")==732&&original.at("original_allocation_bytes")==tails::originalAllocationBytes&&original.at("completed_original_upload_bytes")==tails::originalAllocationBytes&&original.at("completed_samples")==733,"Tail original full import cutoff incomplete");
  for(uint32_t i=0;i<tails::targetCount;++i){const auto t=tails::target(i);activeTarget=i;targetContextObserved=true;tails::require(targets[i].storage==embedding.shards[i].storage&&targets[i].storage==head.shards[i].storage&&embedding.shardFirstRows[i]==t.firstRow&&targets[i].words==t.words&&targets[i].logicalElements==uint64_t(t.rows)*tails::columns&&targets[i].packedBF16&&embedding.shards[i].words==t.words&&head.shards[i].words==t.words&&embedding.shards[i].logicalElements==targets[i].logicalElements&&head.shards[i].logicalElements==targets[i].logicalElements&&embedding.shards[i].packedBF16&&head.shards[i].packedBF16,"Tail original Storage or descriptor changed after import");}originalIdentitiesChecked=true;
  for(uint32_t i=1;i<tails::targetCount;++i)direct(device,i,true);finish();
 }
 void failure(const std::string& reason)noexcept{
  try{counts.refusal();complete=false;if(firstRefusal.is_null())firstRefusal={{"reason",reason.substr(0,1024)},{"selected_target_context_observed",targetContextObserved},{"last_observed_target_shard_index",targetContextObserved?Json(activeTarget):Json(nullptr)},
   {"target_context_scope","Last selected-tail or original watcher sample stack boundary; generic importer error can precede the next hook, so this is not proof of failed API target"},{"original_counts",originalCounts},{"last_transport",lastTransport},{"first_bad_retained",!firstBad.is_null()}};
   if(sourceCopied[activeTarget]&&!sourceWriteAttempted){sourceWriteAttempted=true;sink(tailSourceFile,expected[activeTarget].data(),tails::rowBytes);sourceWritten=true;}
   if(goodAvailable[activeTarget]&&!goodWriteAttempted){goodWriteAttempted=true;sink(tailGoodFile,good[activeTarget].data(),tails::rowBytes);goodWritten=true;}
   if(!journalFailed&&records<tails::maximumRecords)emit({{"event","partial_or_refused"},{"first_refusal",firstRefusal}});
  }catch(...){failurePersistenceFailed=true;}
 }
 void legacyStackExit(const direct_head_row::Receipt& r,const Json& original){activeTarget=0;targetContextObserved=true;originalCounts=original;lastTransport=transport(r);}
 void importStackClosed(){sourceMapsReleased=true;}
 void releaseTargets(){for(auto& b:targets)b={};targetsReleased=true;}
 bool completed()const{return complete&&counts.terminal&&!counts.refused;}
 Json report()const{return {{"schema","chandra.directcompute.head-shard-tail-audit.v1"},{"requested",true},{"complete",completed()},{"outcome",completed()?(counts.fault?"first_fault":"no_fault_completed_selected_tails"):"partial_or_refused"},
  {"selected_tails_equal",completed()&&!counts.fault},{"arithmetic_gate_eligible",false},{"import_only",true},{"selected_target_count",tails::targetCount},{"counts",counterReport()},{"original_counts_at_last_observed_boundary",originalCounts},{"source_records",sourceRecords},{"last_good_selected_samples",lastGoods},{"first_bad",firstBad},{"first_refusal",firstRefusal},{"last_transport",lastTransport},
  {"source_raw_write_attempted",sourceWriteAttempted},{"source_raw_written_and_flushed",sourceWritten},{"last_good_raw_write_attempted",goodWriteAttempted},{"last_good_raw_written_and_flushed",goodWritten},{"first_bad_raw_write_attempted",badWriteAttempted},{"first_bad_raw_written_and_flushed",badWritten},{"journal_write_failed_or_torn",journalFailed},{"failure_evidence_persistence_failed",failurePersistenceFailed},
  {"same_original_embedding_head_Storage_checked",originalIdentitiesChecked},{"target_references_released",targetsReleased},{"source_maps_released_observed",sourceMapsReleased},{"root_physical_retirement_accepted",false},{"request_replayed",false},{"plan",forecast()}};}
};
} // namespace chandra::dc::import_row
