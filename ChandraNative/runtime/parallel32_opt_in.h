// New ChandraNative code, MPL-2.0. Pure admission for an explicit arithmetic experiment.
// No device, environment reads, model loading, allocation, or external execution authority.
#pragma once
#include <cstring>
namespace chandra::parallel32_opt_in {
inline bool admitted(bool optIn, bool fullPageRoute, bool cap2Route, const char* environment,
                     const char* capturedOperatorSelector) {
    return optIn && (fullPageRoute != cap2Route) && environment && capturedOperatorSelector &&
           std::strcmp(environment, "parallel32") == 0 && std::strcmp(capturedOperatorSelector, "parallel32") == 0;
}
inline constexpr const char* fullPageSchema = "chandra.directcompute.final-import-parallel32-full-page-result.v1";
inline constexpr const char* fullPageMode = "final_import_then_parallel32_full_page_generation";
inline constexpr const char* cap2Schema = "chandra.directcompute.final-import-parallel32-cap2-result.v1";
inline constexpr const char* cap2Mode = "final_import_then_parallel32_cap2_arithmetic";
inline constexpr const char* refusedSchema = "chandra.directcompute.parallel32-refused-result.v1";
inline constexpr const char* shaderSha256 = "8d3a4293f5c96ab5e689c70e2168bb6a570b17e998ab26437118d18d1f6ca847";
} // namespace chandra::parallel32_opt_in
