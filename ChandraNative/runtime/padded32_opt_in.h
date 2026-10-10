// New ChandraNative code, MPL-2.0. Explicit ordered multirow staging experiment.
// No kernel change, B1 selection, device/model work, or execution authority.
#pragma once
#include <cstdint>
namespace chandra::padded32_opt_in {
inline bool supported(uint32_t rows, uint32_t inputWidth, bool bf16Weight) noexcept {
    return bf16Weight && rows >= 2 && rows <= 32 && inputWidth >= 32 &&
           inputWidth <= 9216 && inputWidth % 32 == 0;
}
inline constexpr const char* shader = "runtime/linear_gemm_padded32.hlsl";
inline constexpr const char* fallbackShader = "runtime/linear.hlsl";
inline constexpr const char* shaderSha256 = "93206bb0eae75929f86a81775360245705f6e3655639c054f912b4247f2a4d75";
inline constexpr const char* fallbackSha256 = "8fa9fb68f69b35a812f4a5821db11148e665611b0b5f560a5a0186ac13343d5f";
inline const char* schema(bool fullPage, bool parallel32) noexcept {
    return fullPage ? (parallel32 ? "chandra.directcompute.final-import-parallel32-padded32-full-page-result.v1" :
                                  "chandra.directcompute.final-import-padded32-full-page-result.v1") :
                     (parallel32 ? "chandra.directcompute.final-import-parallel32-padded32-cap2-result.v1" :
                                  "chandra.directcompute.final-import-padded32-cap2-result.v1");
}
inline const char* mode(bool fullPage, bool parallel32) noexcept {
    return fullPage ? (parallel32 ? "final_import_then_parallel32_padded32_full_page_generation" :
                                  "final_import_then_ordered_B1_padded32_full_page_generation") :
                     (parallel32 ? "final_import_then_parallel32_padded32_cap2_arithmetic" :
                                  "final_import_then_ordered_B1_padded32_cap2_arithmetic");
}
}
namespace chandra::dc::experimental {
// CLI-only latch, enabled once before the first linear() call; worker stays inactive.
void enableGemmPadded32();
bool gemmPadded32Enabled() noexcept;
struct GemmPadded32Counts { uint64_t padded32 = 0, fallback = 0; };
// Successful Device::dispatch() returns, not proof of GPU completion or arithmetic acceptance.
GemmPadded32Counts gemmPadded32Counts() noexcept;
}
