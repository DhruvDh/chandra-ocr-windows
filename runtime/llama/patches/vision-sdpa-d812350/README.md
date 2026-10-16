# Experimental oneDNN vision attention

[candidate.patch](candidate.patch) is the exact reviewed and built two-file patch against llama.cpp [d81235049384534c167caea52b85a694f6103d14](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14) (b11429). It adds a default-off, narrowly bounded unmasked vision-attention route using oneDNN Graph on Intel Arc. The built leaf passed the [retained operator experiment](../../../../benchmarks/evidence/sycl-vision-attention-operator-2026-10-15.json). The subsequent [first complete-page comparison](../../../../benchmarks/evidence/sycl-vision-page-throughput-2026-10-16.json) passed all 30 source checks and observed 52.4% more pages/s than the matched SYCL baseline. Repeated measurements and accumulated numerical-drift qualification remain pending before promotion; Vulkan remains selected. [provenance.json](provenance.json) preserves the earlier build/operator boundary and exact source, SDK, compiler, settings, output identities and evidence hashes; raw machine-specific receipts and binaries are not included.

Both `GGML_SYCL_FA_ONEDNN_UNMASKED=1` and, on DG2, `GGML_SYCL_FA_ONEDNN_DG2_D64=1` are required. Without them, upstream routing remains unchanged. The admitted operation is unmasked self-attention with F32 Q, directly bindable F16 K/V, batch 1, 16 heads, dimension 64, equal sequence lengths 32–12,096, scale 0.125, no sinks/bias/softcap and the existing F32 accumulation request. The graph omits the absent mask and adds mask presence and engine identity to its cache key. Present-mask behavior remains unchanged. Q, probabilities and output still use F16 staging with strict matmul accumulation; this is not all-F32 attention.

The experimental Graph allocator limits each internal allocation to 64 MiB and aggregate live reservations to 128 MiB. A denied allocation or unmasked graph/runtime failure aborts the experiment rather than silently falling back. This blocks the actual shape's multi-gigabyte quadratic scratch allocation, but does not bound ggml buffers, driver/JIT allocations or total GPU/process memory. Event and registered-queue waits precede frees and are included in the measured cost. Keep one submitting host thread and retain the backend for the entire owned worker process: the inherited compiled-partition map is unsynchronized and allocator state deliberately survives until process exit.

## Apply and build in a private project directory

Use a fresh, short Windows working root, an initialized VS 2022 x64 environment (observed MSVC 14.44.35207 / Windows SDK 10.0.26100.0), CMake 3.31.6 and Ninja. Obtain the exact Intel oneAPI 2026.1.1.32 offline kit and official b11429 Windows SYCL archive from the URLs and hashes in the provenance. The recorded SDK was extracted as archive data: select the 19 component IDs/versions listed there, verify their kit manifests, and merge only ordinary `_installdir/` members after stripping that prefix into a fresh `sdk/`; reject traversal, Windows path aliases, links and conflicting files, and retain licenses. No installer/custom action, registration, global PATH or driver change is needed. The tested extraction/containment helpers are not redistributed here; the provenance supplies the inputs and layout, not a portable automated installer.

Define `$Source`, `$Build`, `$Sdk` and `$PatchDir` as operator-chosen absolute paths in that private workspace, and `$CMake`, `$Ninja`, `$Cl` as the selected tool paths. Before configuration, initialize the child-local environment described by `build.environment`: prepend its SDK-relative PATH/INCLUDE/LIB entries to the VS environment, set its SDK root variables, and preserve CMake's Windows exception defaults. In particular, set `CXXFLAGS=-fsycl-max-parallel-link-jobs=1`; do not replace `CMAKE_CXX_FLAGS` with that value. The following commands are an application/configuration recipe, not a resource supervisor or an instruction to run the candidate:

```powershell
git -C $Source checkout --detach d81235049384534c167caea52b85a694f6103d14
git -C $Source apply --check "$PatchDir/candidate.patch"
git -C $Source apply "$PatchDir/candidate.patch"

$p = Get-Content -Raw "$PatchDir/provenance.json" | ConvertFrom-Json
$flags = @($p.build.cmake_cache.PSObject.Properties | ForEach-Object { "-D$($_.Name)=$($_.Value)" })
$paths = @($p.build.sdk_cmake_paths.PSObject.Properties | ForEach-Object { "-D$($_.Name)=$Sdk/$($_.Value)" })
& $CMake -S $Source -B $Build -G Ninja @flags @paths "-DCMAKE_C_COMPILER=$Cl" "-DCMAKE_MAKE_PROGRAM=$Ninja"
& $Ninja -C $Build -j1 ggml-sycl
```

Check each command's success before continuing and compare the two patched source hashes with the provenance. The actual build first compiled/linked a tiny SYCL/oneDNN/MKL header-and-link smoke plus a C++ exception control, without running either; the upstream IntelSYCL host-only language feature check ran during CMake configuration. It then built only `ggml-sycl` and its base dependencies with F16 and GPU oneDNN enabled, direct Level Zero API disabled, and one compile/link job. Require actual `GGML_SYCL_DNNL=1`, successful IntelSYCL discovery, `DNNL_GPU_RUNTIME=SYCL`, `DNNL_GPU_VENDOR=INTEL`, preserved exceptions and the one-job flag in the module link rule. Build success alone does not verify these conditions. The recorded build used one CPU, below-normal priority, a 2,400-second work limit, 4 GiB process/Job private caps, live 4 GiB physical/commit reserves and more than 12 GiB additional free disk. Those controls are separate from CMake/Ninja and must be supplied by the operator's owned launcher.

Copy only the resulting `bin/ggml-sycl.dll` into a fresh copy of the authenticated official b11429 runtime. Preserve all other runtime files, compare the leaf's exports and required base imports, and authenticate the complete resulting executable/DLL inventory. The observed leaf is 65,078,272 bytes, SHA256 `90e65f3b67017a598a7fc0c467bf08b15157ee521d27f1e1921126461ae641c7`; it is an observed artifact identity, not a promise of bit-identical rebuilds. The SDK's SYCL/oneDNN/MKL DLLs matched the official runtime. The full VS/Windows SDK installation was not recursively inventoried, and the private smoke/extraction/ownership helpers are not included, so this publication does not claim a fully hermetic reproduction.

## Enable the actual experimental worker

Use a separately authenticated SYCL adapter configuration and candidate runtime inventory, following [the adapter contract](../../README.md). Parent-shell variables alone do **not** enable this experiment: `LlamaConfig.environment()` scrubs inherited overrides. The tested isolated service source copy adds exactly the following two entries to its SYCL child `env.update(...)`, leaving the public/default adapter unchanged, then rebinds that source copy and configuration to their new hashes:

```python
GGML_SYCL_FA_ONEDNN_UNMASKED="1",
GGML_SYCL_FA_ONEDNN_DG2_D64="1",
```

Retain the verified physical-device mapping, explicit Level Zero selector, all three existing Sysman exclusions, one submitting CPU thread, private persistent cache, process ownership/retirement, memory guards, Q8_0 text/BF16 projector and KV, 16,384 context per slot and 12,384 output allowance. These two flags are not currently JSON configuration fields. Neither adding the patch nor setting an activation marker promotes this candidate or changes the deployed backend.

The completed operator run passed four small masked/unmasked CPU-reference controls in each arm at the pre-existing NMSE threshold 0.0005, using independently randomized operands. At 12,096 tokens it identified oneDNN `sdp_primitive_kernel_t` and measured one warmed 232-repeat aggregate per arm: 359.137 ms default versus 20.421 ms enabled (17.59×). That actual-shape timing includes packing/conversion/allocation/deallocation waits, but no actual-shape CPU correctness comparison; it repeats the same buffers and excludes initialization, JIT, upload and final cache destruction. The operator experiment establishes its own timing and numerical scope; the later page evidence above separately records source-content/localization and throughput at four concurrent requests. Neither alone establishes general numerical equivalence or deployment qualification.

Upstream llama.cpp material is MIT-licensed; its notice is retained in [LICENSE.llama.cpp](LICENSE.llama.cpp). This repository's license is unchanged. No runtime binaries, SDK payloads, weights or private machine paths are shipped in this directory.
