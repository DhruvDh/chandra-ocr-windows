# Resolve native vision attention before further GPU trials

The normal-page B1 trial and a narrower diagnostic both reported device loss, with temporally correlated Windows engine-timeout archives. The diagnostic reached text-mask preparation before failing a synchronization; it never entered the first text layer. That identifies where the error surfaced, not the causal kernel. All test-owned processes subsequently closed and the ordinary endpoint returned cold. No full native-page inference or Intel speedup is qualified.

The pinned XPU selector has a material admission risk: an enabled `EFFICIENT_ATTENTION` branch can select math even when explicit math is disabled. The serving context also permits the oneDNN `OVERRIDEABLE` backend, which has higher default priority; its actual selection and workspace were not captured. The retained stderr has no efficient-fallback warning, so the source finding does not establish an observed fallback. [Pinned selector](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mkldnn/xpu/Attention.cpp), [oneDNN bridge](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mkldnn/xpu/detail/Attention.cpp).

Flash-only would not provide an assumed remedy: the pinned Flash architecture whitelist omits Alchemist, the A770 architecture. The existing query-chunk candidate intercepts text attention; vision still uses full-image attention across 24 blocks. [Flash gates](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/xpu/sdp_utils.cpp).

| Attention geometry | Query rows | Key rows | FP32 score bytes if materialized |
| --- | ---: | ---: | ---: |
| Accepted Tiny vision | 832 | 832 | 44,302,336 |
| Native-page vision | 13,464 | 13,464 | 11,601,874,944 |
| Proposed bounded native call | 32 | 13,464 | 27,574,272 |

These calculations use 16 heads. Native attention requires about 262 times Tiny's attention arithmetic. A possible 11.60 GB score buffer would accompany approximately 9.18 GB of live model allocations; neither the buffer nor the actual peak was observed. One-second resource samples cannot exclude brief workspace peaks. Fencing between whole vision blocks does not cap one attention call's memory or duration.

The next diagnostic should retain the actual first vision Q/K/V descriptors and backend choice, then stop before invoking SDPA. Its preceding projection work still needs admission. A later query-only partition must retain all keys, values, pixels, scaling and output allowance, with completion fences between calls. The smaller per-call score plane is a preparation bound, not proof of numerical equality or safety. Tiny numerical and complete-output checks, native-geometry primitive checks, clean process closure and independent admission precede any whole native-page retry. Both diagnostic and partition remain uninstalled source preparation. The [evidence index](../benchmarks/evidence/hillclimb-2026-10-04.json) records source commitments and keeps these findings separate from GPU acceptance.
