// New code, MPL-2.0. Genuine B2 shared FFN/head candidate, inactive by default.
#pragma once
#include <cstdint>
namespace chandra::ordered2_opt_in {
inline bool supported(uint32_t rows,uint32_t inputWidth,bool bf16Weight) noexcept {
    return rows==2 && bf16Weight && (inputWidth==2560 || inputWidth==9216);
}
inline constexpr const char* shader="runtime/linear_gemm_ordered2.hlsl";
inline constexpr const char* shaderSha256="df14d59cbea50c41f150e259d9453b92486637d0b589354b88527c9ab8448ea2";
inline constexpr uint32_t shaderBytes=5212;
inline constexpr uint32_t outputsPerGroup=8, tile=512, stride=513, sharedBytes=20520;
static_assert(sharedBytes==(8*513+2*513)*4 && sharedBytes<32768,"cs_5_0 shared storage");
}
namespace chandra::dc::experimental {
void enableGemmOrdered2();
bool gemmOrdered2Enabled() noexcept;
// Successful Device::dispatch returns for this candidate; no completion/acceptance claim.
uint64_t gemmOrdered2Dispatches() noexcept;
}
