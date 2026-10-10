// New code, MPL-2.0. Explicit DEFAULT initialization experiment; no numerical policy.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace chandra::dc {
enum class WeightImportAPI { updateSubresource, defaultInitial };
namespace weight_import {
inline const char* name(WeightImportAPI api) {
 if(api==WeightImportAPI::updateSubresource)return "update-subresource";
 if(api==WeightImportAPI::defaultInitial)return "default-initial";
 throw std::invalid_argument("Unknown weight import API");
}
inline WeightImportAPI parse(const std::string& value) {
 if(value=="update-subresource")return WeightImportAPI::updateSubresource;
 if(value=="default-initial")return WeightImportAPI::defaultInitial;
 throw std::invalid_argument("Unknown weight import API selector");
}
// Scratch belongs to the caller's source-view scope and survives its original drain.
inline const void* initialPayload(const void* source,uint64_t sourceBytes,uint64_t storageBytes,uint32_t words,std::vector<uint32_t>& scratch) {
 if(!source||!sourceBytes||!words||storageBytes!=uint64_t(words)*4||sourceBytes>storageBytes||storageBytes>128ull*1024*1024||!scratch.empty())
  throw std::runtime_error("Complete bounded initial-data payload required");
 if(sourceBytes==storageBytes)return source;
 scratch.assign(words,0);std::memcpy(scratch.data(),source,static_cast<size_t>(sourceBytes));return scratch.data();
}
template<class DeviceT> auto createInitial(DeviceT& device,bool bf16,uint32_t words,const void* values) {
 if(!words||!values)throw std::runtime_error("Initial-data allocation requires complete payload");
 return bf16?device.words(words,static_cast<const uint32_t*>(values)):device.floats(words,static_cast<const float*>(values));
}
} // namespace weight_import
} // namespace chandra::dc
