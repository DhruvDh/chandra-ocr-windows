// New code, MPL-2.0. Narrow post-import observer; implementation in existing model_weights.cpp.
#pragma once
#include "api.h"
#include "import_row_source.h"
#include "final_import_tail_state.h"
namespace chandra::dc::import_row {
constexpr const char* finalTailSourceFile="source.tails.bf16";
constexpr const char* finalTailGoodFile="matched-prefix.direct.bf16";
constexpr const char* finalTailBadFile="first-bad.direct.bf16";
class FinalTailObserver {
 struct Impl;std::unique_ptr<Impl> impl;
public:
 FinalTailObserver(WeightImportAPI,Sink,Digest,Progress);
 FinalTailObserver(WeightImportAPI,Sink,Digest,Progress,bool weightSrvOnly);~FinalTailObserver();
 FinalTailObserver(const FinalTailObserver&)=delete;FinalTailObserver& operator=(const FinalTailObserver&)=delete;
 WeightImportAPI importAPI()const noexcept;
 bool weightSrvOnly()const noexcept;
 static Json forecast(WeightImportAPI,bool weightSrvOnly=false);
 void prepare(const Json&);void begin(const Json&);
 void beforeAPI(const Json& mapping,const void* pointer,uint64_t bytes);
 void returned(const Buffer&);void returned(Device&,const Buffer&);void afterDrain();
 void afterImport(Device&,const Weight&,const Weight&);
 void failure(const std::string&)noexcept;void releaseTargets();
 bool completed()const;Json report()const;
};
Json runFinalTailImport(Device&,const std::wstring&,WeightImportAPI,FinalTailObserver&);
Json runFinalTailImport(Device&,const std::wstring&,WeightImportAPI,FinalTailObserver&,bool weightSrvOnly);
} // namespace chandra::dc::import_row
