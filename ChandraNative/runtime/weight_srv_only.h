// New code, MPL-2.0. Weight-only DEFAULT raw SRV/no-UAV diagnostic policy.
#pragma once
#include <cstdint>
namespace chandra::dc::weight_srv_only {
constexpr uint32_t defaultUsage=0,shaderResourceBind=8,rawMisc=32,rawFormat=39,bufferExDimension=11,rawSRVFlag=1;
inline bool admitted(bool requested,bool finalOnly,bool exact,bool defaultInitial,bool explicitAPI) noexcept {
 return !requested || (finalOnly && exact && defaultInitial && explicitAPI);
}
inline bool writable(bool readonlyWeight,bool uavPresent) noexcept {return !readonlyWeight && uavPresent;}
struct Descriptor {
 uint32_t byteWidth=0,usage=0,bindFlags=0,miscFlags=0,cpuAccess=0,stride=0;
 uint32_t srvFormat=0,srvDimension=0,srvFirst=0,srvWords=0,srvFlags=0;
 bool readonlyWeight=false,srvPresent=false,uavPresent=false,srvSameResource=false;
};
inline bool matches(const Descriptor& d,uint32_t words) noexcept {
 return words>0 && words<=33554432 && uint64_t(words)*4==d.byteWidth && d.usage==defaultUsage && d.bindFlags==shaderResourceBind && d.miscFlags==rawMisc && d.cpuAccess==0 && d.stride==0 && d.srvFormat==rawFormat && d.srvDimension==bufferExDimension && d.srvFirst==0 && d.srvWords==words && d.srvFlags==rawSRVFlag && d.readonlyWeight && d.srvPresent && !d.uavPresent && d.srvSameResource;
}
} // namespace chandra::dc::weight_srv_only
