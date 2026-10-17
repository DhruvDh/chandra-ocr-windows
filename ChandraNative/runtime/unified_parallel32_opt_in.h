// New code, MPL-2.0. Separate inactive route; never an ordered2 alias or fallback.
#pragma once
#include <cstdint>
#include <cstring>
namespace chandra::unified_parallel32_opt_in {
inline bool supported(uint32_t rows,uint32_t inputWidth,bool bf16Weight) noexcept {
 return rows==2 && bf16Weight && (inputWidth==2560 || inputWidth==9216);
}
inline bool selectionAllowed(bool padded,bool b2,bool ordered2,const char* captured,const char* live) noexcept {
 return padded && b2 && !ordered2 && captured && live &&
     std::strcmp(captured,"parallel32")==0 && std::strcmp(live,"parallel32")==0;
}
inline constexpr const char* shader="runtime/linear_gemm_parallel32_b2.hlsl";
inline constexpr const char* shaderSha256="57d88c96ea7e913e374fb830d0b84788d23a34471f27cbc7f13f770a35c3f97e";
inline constexpr uint32_t shaderBytes=5670;
inline constexpr const char* singleShader="runtime/linear_gemv_parallel32.hlsl";
inline constexpr const char* singleShaderSha256="8d3a4293f5c96ab5e689c70e2168bb6a570b17e998ab26437118d18d1f6ca847";
inline constexpr uint32_t singleShaderBytes=4803;
inline constexpr uint32_t outputsPerGroup=8,lanes=32,tile=512,stride=513,threads=256,staged=16,sharedBytes=20512;
static_assert(sharedBytes==(8*513+2*512)*4 && sharedBytes<32768,"SM5 shared budget");
}
namespace chandra::dc::experimental {
void enableUnifiedParallel32B2();
bool unifiedParallel32B2Enabled() noexcept;
uint64_t unifiedParallel32B2Dispatches() noexcept;
}
