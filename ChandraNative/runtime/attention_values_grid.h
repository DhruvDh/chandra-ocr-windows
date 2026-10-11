// New ChandraNative code, MPL-2.0. Explicit independent attention-value tile dispatch experiment.
#pragma once
#include <cstdint>
#include <stdexcept>
namespace chandra::attention_values_grid {
inline bool selected=false, modelConstructed=false;
inline void enable() {
 if(selected||modelConstructed)throw std::invalid_argument("Attention-values grid must be selected once before the first TextModel");
 selected=true;
}
inline bool enabled() noexcept { return selected; }
inline void lockModelSelection() noexcept { modelConstructed=true; }
inline uint32_t checkedTiles(uint32_t rows,uint32_t keys) {
 if(!rows||rows>64||keys<rows||keys>16384)
  throw std::invalid_argument("Attention-values grid requires 1<=rows<=64 and rows<=keys<=16384");
 const uint32_t tiles=(keys+127)/128;
 if(16>65535||rows>65535||!tiles||tiles>65535)
  throw std::invalid_argument("Attention-values grid exceeds D3D11 dispatch dimensions");
 return tiles;
}
inline constexpr const char* shader="runtime/text_attention_values_tile_grid.hlsl";
inline constexpr const char* serialShader="runtime/text_attention_values.hlsl";
inline constexpr const char* reduceShader="runtime/text_attention_reduce.hlsl";
inline constexpr const char* shaderSha256="bd9a31e1e09c264ec34684dfb592e679383d5e55073fd0242568081773acbe5e";
inline constexpr const char* serialSha256="32128314fc79d6b1fb9f37bb2097c004aa754ad1c905cda35366a3c51427f8b6";
inline constexpr const char* reduceSha256="f0548298b831ab6d47adf4d0ebde9f6209d49134ca1aaddfbf60102db9c99e26";
inline constexpr uint32_t shaderBytes=783,serialBytes=746,reduceBytes=527;
template<class Json> Json metadata() {
 return {{"schema","chandra.directcompute.attention-values-tile-grid-selection.v1"},{"selector","tile-grid-3d"},{"explicit_cli_opt_in",true},
  {"cli_flag","--experimental-attention-values-grid"},{"shader",shader},{"shader_source_sha256",shaderSha256},
  {"serial_shader",serialShader},{"serial_shader_source_sha256",serialSha256},{"reduce_shader",reduceShader},{"reduce_shader_source_sha256",reduceSha256},
  {"groups","16,rows,ceil(totalKeys/128)"},{"tile_axis","SV_GroupID.z"},{"keys_per_group",128},
  {"supported_rows",{1,64}},{"maximum_total_keys",16384},{"partial_layout","((tile*rows+row)*16+head)*256+channel"},
  {"arithmetic","Original ascending k within each 128-key tile; original ascending tile reduction and BF16 RNE"},
  {"operation_order_changed",false},{"precision_changed",false},{"other_operator_selectors_changed",false},
  {"scope","Explicit pre-model source-pinned selection; compiled arithmetic and trained equivalence remain unqualified"},
  {"trained_numerics_accepted",false},{"complete_OCR_or_PPS_accepted",false}};
}
}
