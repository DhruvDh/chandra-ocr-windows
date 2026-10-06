# DirectCompute B1 GEMV occupancy variants

The [closed Windows calibration record](../benchmarks/evidence/directcompute-gemv-occupancy-2026-10-06.json) rejects these two variants as performance nominees. Strict native compilation and all 70,524 independent output-word comparisons across six serial GPU points pass, but both forward and reversed cohorts are slower than ordered8 on the dominant shapes. The code remains opt-in experimental source; it has no trained, complete OCR, endpoint or sustained page-throughput acceptance. Every point released owned buffers, observed zero Job descendants and restored the ordinary worker cold.

| Mean GPU milliseconds per dispatch, range across two cohorts | Ordered8 | Ordered32 | Ordered64 |
| --- | --- | --- | --- |
| K=2560, N=1024, BF16 weights | 0.16539–0.16602 | 0.31513–0.31586 | 0.47661–0.47672 |
| K=9216, N=1024, dyadic BF16 weights | 0.54448–0.54563 | 1.07458–1.07516 | 1.66198–1.66216 |

The separate FP32 order-sensitive phase retains an ordered32 timing outlier; no result was removed. These short kernel observations do not measure a page-throughput ceiling. Raw timestamp endpoints and runtime bytecode are unavailable, and physical host RAM was checked at admission rather than continuously traced. Build13's failed default-selector assertion remains unchanged; fresh Build14 passed the corrected checks.

`ordered32` and `ordered64` are two experimental, inactive-by-default schedules for batch-row-1 `linear()` calls on the Windows Intel Arc A770. They follow the [ordered candidate](directcompute-gemv-candidate.md) (`ordered`, here also called ordered8), which calibrated K=9216, N=1024 at about 0.545 ms against the predecessor's 3.56 ms and preserved all 395 tiny-page token IDs and scalar journal rows. The variants keep ordered8's arithmetic exactly and change only how many output lanes accumulate per workgroup, the K tile, and how weights are staged. The closed record above now establishes strict compilation and model-free GPU checks for both variants. Neither passes the performance nomination gate, and no trained or page-throughput speedup is accepted.

## Selector and routes

`CHANDRA_EXPERIMENTAL_GEMV_B1` is still read once before `main` and never re-read. Unset or `0` keeps the predecessor `linear.hlsl` route for every batch size, byte-for-byte; `ordered` keeps commit `0935926`'s `linear_gemv.hlsl` dispatches byte-for-byte; `ordered32` and `ordered64` select their own shaders for rows equal to one only. Matching is exact and case-sensitive. Any other value, including `ordered8`, `Ordered32` or a value with spaces, makes every `linear()` call throw `std::invalid_argument` before validation, allocation or dispatch, whatever its batch size. There is no fallback, and nothing selects a variant automatically. Rows other than one keep `linear.hlsl`, and every call keeps the existing bound of at most 1,024 outputs per dispatch with the complete K in each dispatch. [gemv_variants.h](../ChandraNative/runtime/gemv_variants.h) holds the selector parser and the shape table used by [operators.cpp](../ChandraNative/runtime/operators.cpp), the calibration and the tests; `api.h` is unchanged.

| Selector | Shader | Outputs per 256-thread group | K tile | Padded row stride (floats) | Groupshared bytes | Groups per 1,024 outputs | Tiles at K=9,216 |
|---|---|---|---|---|---|---|---|
| `ordered` | `runtime/linear_gemv.hlsl` (unchanged) | 8 | 512 | 513 | 18,464 | 128 | 18 |
| `ordered32` | [linear_gemv_ordered32.hlsl](../ChandraNative/shaders/runtime/linear_gemv_ordered32.hlsl) | 32 | 128 | 129 | 17,024 | 32 | 72 |
| `ordered64` | [linear_gemv_ordered64.hlsl](../ChandraNative/shaders/runtime/linear_gemv_ordered64.hlsl) | 64 | 64 | 65 | 16,896 | 16 | 144 |

## What ordered8 does, and what the variants change

In ordered8, all 256 threads stage one 8 × 512 weight tile (32 lanes per output row load consecutive 32-bit words) and the 512-element input slice into groupshared memory, then one barrier releases threads 0–7. Each continues its own FP32 sum over 512 ascending k while the other 248 threads wait at the second barrier. So per tile, 8 of 256 lanes do arithmetic, and each group re-stages the complete input vector. The measured 0.545 ms at K=9,216 equals about 59 ns, or about 124 cycles at the nominal 2.1 GHz clock, per sequential k step, if all 128 groups were resident together. The source alone cannot say whether that time is the latency of one owner's dependent step (two groupshared reads, a multiply and a dependent add), or groups waiting for residency, load-issue or groupshared throughput, staging latency and barriers.

The variants raise the owners per group to 32 or 64. ROWS×TILE stays 4,096 elements, so each tile stages the same 16 KiB of FP32 weights and each thread holds at most 16 FP32 or 9 packed-BF16 words, as in ordered8. Staging is row-major. Word slot `i < 16` of a thread holds word `f % width` of tile row `f / width`, where `f = thread + 256·i` and `width` is TILE/2 packed BF16 words or TILE FP32 words. A SIMD16 or SIMD32 message therefore reads 64 or 128 contiguous bytes of one weight row. When a row's first element in a full tile is odd (possible only for odd K), its tile spans TILE/2+1 packed words; thread `r` (below ROWS) loads that extra word for row `r` in slot 16. Rows past the group, rows past the shard and words past the tile tail have count zero and never load or store. Each thread issues all its global loads before any groupshared store. Packed word `j` supplies positions `2j−parity` and `2j+1−parity`, under the same guards as ordered8. Rows are padded to TILE+1 floats, so owner `o` reading `o·STRIDE+k` maps to bank `(o+k) mod banks` instead of every owner hitting one bank. The input slice is loaded by threads below TILE.

Every thread runs the same K-loop trip count, computed only from the cbuffer, and reaches both barriers of every tile. The barriers are top-level statements of that loop, `main` has no `return`, `break`, `continue` or `discard`, and owners are threads below ROWS whose output is inside the shard. Output tails, shard offsets (`weightRowFirst`) and `firstOutput` are used exactly as in ordered8. Host validation already keeps every touched word inside the shard's uint32 byte range.

## Arithmetic order

Each output is still one `precise` FP32 `sum` from +0, then `sum = sum + a[k]*w[k]` in ascending k over the complete K, then bias, then the same optional BF16 round-to-nearest-even with NaN quieting. The `packed`, `bf16_rne`, owner-loop and epilogue source text is byte-identical to `linear_gemv.hlsl`, and a CPU test enforces this. There is no split-K, reassociation, FP16, wave intrinsic, `mad`/`fma`, transposition, prepacking or alternate checkpoint. A variant result should therefore be bit-identical to ordered8 and the predecessor, NaN payloads aside. Like ordered8, this rests on fxc and the driver honouring `precise` and D3D11 float rules, and the order-sensitive calibration phases exist to expose any departure.

## Source-derived roofline and occupancy

These are structural counts from the pinned decode shapes and the 128 MiB shard plan (`gemv-variants-test SHADER_DIRECTORY --forecast`). The predecessor and ordered8 rows reproduce the published forecast exactly. They are not measurements.

| Per B1 decode token | Predecessor | ordered8 | ordered32 | ordered64 |
|---|---|---|---|---|
| Linear dispatches | 1,431 | 1,431 | 1,431 | 1,431 |
| Workgroups | 86,278 | 172,547 | 43,144 | 21,596 |
| Lanes accumulating per group | 16 of 256 (all 256 run the loop) | 8 of 256 | 32 of 256 | 64 of 256 |
| Group barriers | 16,425,920 | 2,053,150 | 2,053,440 | 2,055,360 |
| Thread multiply-adds issued | 67,280,568,320 | 4,204,789,760 | 4,204,789,760 | 4,204,789,760 |
| Global input-vector staging bytes | 1,051,258,880 | 2,102,425,600 | 525,680,640 | 263,086,080 |
| Groupshared bytes written | 33,640,284,160 | 18,921,584,640 | 17,344,839,680 | 17,082,245,120 |
| Groupshared bytes read by the dot loops | 538,244,546,560 | 33,638,318,080 | 33,638,318,080 | 33,638,318,080 |

Every route issues the same useful arithmetic: 4,204,789,760 multiply-adds (8,409,579,520 FP32 FLOPs) against 8,409,579,520 unique BF16 weight bytes, about one FLOP per weight byte. All three ordered routes load each packed weight word once per dispatch for even K; odd K also reloads words that straddle a tile or row boundary. By weight bytes, K=2,560 is 74.1% of a token, K=9,216 18.0% and K=4,096 8.0%. Of the 1,431 dispatches, 1,309 have 1,024 outputs, 64 have 512, 48 have 32, and ten are LM-head tails of 614 or 106 outputs.

For one K=9,216 × 1,024 dispatch, the unique traffic is 18,874,368 weight bytes, 36,864 input bytes and 4,096 output bytes, for 18,874,368 FLOPs. At Intel's nominal 560 GB/s for the A770 16 GB, the weight bytes alone take 33.7 µs; the measured ordered8 0.545 ms gives a logical weight-byte/time ratio of 34.5–34.7 GB/s, about 6% of that nominal figure. This derived ratio is not measured DRAM bandwidth. At that rate, one token's B1 weights would take about 243 ms, against about 709 ms per cached advance of observed tiny decode/readback (279.396 s for 394 cached advances, with the initial token readback also inside the phase). That rate has not been measured at K=2,560 or K=4,096, so this is only a sizing estimate. Even an infinitely fast GEMV would leave the rest of the token's time, which this change does not address.

Several limits follow from the source alone:

- **Chains per dispatch.** Without split-K, each output is one dependent chain of K FP32 additions, and a dispatch has at most 1,024 chains. Owner count only changes how those chains are packed into groups and hardware threads. Against the nominal 4,096 FP32 lanes, the useful-chain/nominal-ALU ratio is at most 25% in any ordered route, or 0.8% for the 32-output projections. This source-derived ratio is not measured hardware occupancy or lane utilization.
- **Step latency.** If ordered8's roughly 59 ns per step is the latency of one owner's dependent step, the variants cannot shorten it.
- **Fewer groups.** At 1,024 outputs, ordered32 launches 32 groups and ordered64 launches 16 against 32 Xe-cores, so a one-group-per-core placement assumption would put ordered64 on at most half the Xe-cores per dispatch. Actual placement and residency have not been measured. Fewer groups can reduce aggregate load issue and L1 bandwidth.
- **More tiles.** Each tile exposes one global-load round trip and two barriers, with no double-buffering. At K=9,216 a group has 18, 72 or 144 such intervals, each covering 512, 128 or 64 owner steps. If these intervals dominate, the variants will be slower.
- **What they save.** They cut input re-staging 4× or 8× and fill 32 or 64 lanes of the owner hardware threads instead of 8. If ordered8 is limited by too many co-resident groups or by per-message groupshared/load throughput, fewer, fuller groups help.
- **Staging overhead.** The variants recompute per-slot row, parity, base and count for 17 slots per thread per tile; ordered8 needs one row per thread. The cost of that integer work is unmeasured.

The source cannot tell us the SIMD width or register allocation the Intel compiler chooses, spills, how many groups are resident per Xe-core, groupshared capacity, or whether groupshared reads are hoisted across the `precise` additions. Nor does it give groupshared, L1 or DRAM latency and achieved bandwidth, barrier cost, clocks and power state, or how D3D11 serialises consecutive dispatches that write one output buffer. Whether either variant is faster, slower or equal can only be established on the A770. If all three ordered routes measure about the same, per-step latency is the likely limit. Candidates for that case are vectorised groupshared reads (float4 per owner), deeper load batching ahead of the add chain, and overlapping the next tile's global loads with the current tile's chain. All keep the ascending order, and none is implemented here.

## Calibration

[gemv_candidate_calibration.h](../ChandraNative/runtime/gemv_candidate_calibration.h) runs one plan for every ordered route; the expected shader and groups come from the selected route. The receipt schema is now `...gemv-b1-candidate-calibration.v2`. It records `candidate`, `shader` and `route_geometry`, plus `variant_confirmed_by_argument`. The six v1 phases keep their names, operands, order and 9,446 compared output words.

Two phases are added. The first is an order-sensitive odd-K phase (K=1,001, outputs 130 in shards of 65 and 65, BF16 bias). It covers odd-parity full tiles, K tails, one-output group tails for 8, 32 and 64 outputs per group, and a half-used last packed word in each shard. The second is an order-sensitive K=2,560 × 1,024 chunk, the dominant decode width. Each route therefore compares 11,754 output words in 32 dispatches and 18 readbacks.

Order-sensitive expectations are sequential CPU binary32 dots from generator values, and dyadic expectations are exact binary64. The CPU never reads packed words or device buffers. Nine malformed admissions must still be refused before allocation or dispatch. The 100 ms per-dispatch gate, the 120-second source deadline, the 256 MiB process commit cap, the per-phase tracked-plus-readback forecast (maximum 18,919,424 bytes), the 8 MiB fresh `CREATE_NEW` receipt, explicit PCI/LUID and the stop-without-rerun rule are unchanged. Without `--execute` it prints the inactive state and creates no device. Execution requires `ordered`, `ordered32` or `ordered64` at process start and the selected shader file under the absolute shader root. An optional `--variant NAME` must equal the selector or the run is refused before any device exists. A dispatch with any other shader or group count is refused as a fallback.

| Phase | K × outputs | Weights, shards, bias | Calls | Groups per call: ordered8 | ordered32 | ordered64 |
|---|---|---|---|---|---|---|
| specials | 4 × 7 | BF16 [7], BF16 bias | 4 | 1 | 1 | 1 |
| ragged odd K | 37 × 21 | BF16 [6, 9, 6], BF16 bias | 2 | 1, 2, 1 | 1, 1, 1 | 1, 1, 1 |
| FP32 weights | 67 × 40 | FP32 [17, 23], FP32 bias | 2 | 3, 3 | 1, 1 | 1, 1 |
| odd K order-sensitive (new) | 1,001 × 130 | BF16 [65, 65], BF16 bias | 2 | 9, 9 | 3, 3 | 2, 2 |
| multi-chunk | 1,001 × 2,600 | BF16 [1,500, 1,100] | 2 | 128, 60, 128, 10 | 32, 15, 32, 3 | 16, 8, 16, 2 |
| K=2,560 order-sensitive (new) | 2,560 × 1,024 | BF16 [1,024] | 2 | 128 | 32 | 16 |
| full width dyadic | 9,216 × 1,024 | BF16 [1,024] | 2 | 128 | 32 | 16 |
| full width order-sensitive | 9,216 × 1,024 | BF16 [1,024] | 2 | 128 | 32 | 16 |

## Root commands

`ChandraNative\runtime\build.cmd` is unchanged. Its `shader-compile.exe` step compiles every `shaders\runtime\*.hlsl` except `vision_common.hlsl`, including both new files, with `cs_5_0`, optimisation level 3 and IEEE strictness, which is how `Device` compiles them. For a direct fxc listing with the same flags, run this at an interactive Developer Command Prompt (use `%%S` in a batch file), with `CHANDRA_SOURCE` set to the checkout and `CHANDRA_FXC_OUT` set to a fresh task-owned directory:

```bat
for %S in (linear_gemv linear_gemv_ordered32 linear_gemv_ordered64) do fxc /nologo /T cs_5_0 /E main /O3 /Gis /Fo "%CHANDRA_FXC_OUT%\%S.cso" /Fc "%CHANDRA_FXC_OUT%\%S.asm" "%CHANDRA_SOURCE%\ChandraNative\shaders\runtime\%S.hlsl"
findstr /R /C:"^ *mad " /C:"^ *sync_" /C:"dcl_tgsm" /C:"dcl_thread_group" "%CHANDRA_FXC_OUT%\*.asm"
```

Expect no compiler errors. Each listing should show `dcl_thread_group 256, 1, 1`, groupshared declarations of 4,128 + 128 four-byte elements (17,024 bytes, ordered32) or 4,160 + 64 (16,896 bytes, ordered64), and exactly two `sync_g_t` instructions inside the K loop. There should be no floating-point `mad`; integer `imad`/`umad` for addressing is expected. Report warnings verbatim, and treat a floating `mad` in the owner loop as a stop.

After `build.cmd` has produced `device.obj` and `operators.obj`, build the calibration and the portable helper test in the build directory. `gemv_variants.h` sits next to the sources, so no new include path is needed. Neither has been compiled with MSVC in this task; report any warning rather than suppressing it:

```bat
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%CHANDRA_SOURCE%\ChandraNative\runtime\gemv_candidate_calibration.cpp" device.obj operators.obj /Fe:gemv-candidate-calibration.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%CHANDRA_SOURCE%\ChandraNative\runtime\gemv_variants_test.cpp" /Fe:gemv-variants-test.exe
gemv-variants-test.exe "%CHANDRA_SOURCE%\ChandraNative\shaders\runtime"
```

Each calibration is a fresh process in its own externally owned Job, with a fresh receipt path (one core, below-normal priority, 256 MiB commit cap). Check the inactive state first; an invalid value must exit 1 without creating a device:

```bat
set CHANDRA_EXPERIMENTAL_GEMV_B1=ordered32
gemv-candidate-calibration.exe
gemv-candidate-calibration.exe --execute --variant ordered32 --shader-root "%CHANDRA_SHADER_ROOT%" --pci "%CHANDRA_PCI%" --luid "%CHANDRA_LUID%" --output "%CHANDRA_FRESH_GEMV_CALIBRATION_FILE%"
```

The same two commands apply with `ordered` and `ordered64`. The ordered8 command in the [candidate guide](directcompute-gemv-candidate.md) remains valid without `--variant`.

## Validation sequence and current disposition

The closed record completes steps 1–3. Both candidates fail the performance condition for step 4, so the subsequent trained promotion tests were not run for them. The sequence remains the method for evaluating a future candidate.

1. Offline fxc compilation of both variant files and the unchanged `linear_gemv.hlsl` with the flags above, plus the listing review, before any GPU admission.
2. Inactive checks for unset, `0`, `ordered`, `ordered32`, `ordered64` and one invalid value. No device may be created.
3. Same-session paired synthetic calibration in the order `ordered`, `ordered32`, `ordered64`, `ordered64`, `ordered32`, `ordered`, with six fresh processes, Jobs and receipts. Every phase must be bit-exact and every dispatch below 100 ms. Compare per-dispatch GPU milliseconds for the K=2,560 and both K=9,216 phases per route. The spread between repeated runs of one route is the noise floor; the earlier 0.545 ms ordered8 figure is from a separate session and is not the baseline.
4. Only if a variant beats ordered8 at both K=2,560 and K=9,216 by more than that spread: run the original tiny request under that selector and under `ordered` with identical Jobs and limits. All 395 IDs and the complete scalar journal must be identical; record cold and cached-decode seconds.
5. The trained projection/logit vector comparison under separate finite Jobs. It is expected to be bit-identical to ordered8 by construction, which must be shown.
6. The complete original corpus against the primary AMD OCR reference, and exact-client acceptance.
7. Repeated, uninstrumented sustained page throughput at safe concurrency against the ordered8 and predecessor routes, before any production selection.

A mismatch, a dispatch at or above 100 ms, an unexpected shader or geometry, or a forecast or deadline breach stops that route without a rerun or a larger phase. These comparisons do not involve AMD service or GPU tuning, XPU/OpenVINO, driver, TDR or power changes.

## CPU checks

`ChandraNative/runtime/gemv_candidate_test.sh FRESH_DIRECTORY` runs everything on the CPU and makes no GPU, model or network access. Every build uses `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror -ffp-contract=off -fno-fast-math`. The suite does the following:

- Checks that `linear_gemv.hlsl` and `linear.hlsl` are unchanged from `0935926` (and `linear.hlsl` from `888ab33`).
- Builds this checkout's `operators.cpp`, `888ab33`'s and `0935926`'s against the fake device.
- Requires the default-route transcripts (unset and `0`) to equal `888ab33`'s, and the `ordered` transcript (shaders, groups, cbuffer words, SRV/UAV ids, output hashes) to equal `0935926`'s.
- For each ordered route: checks twelve shapes through the real `linear()` bit-equal to an independent ascending FP32 reference, and requires all 24 output hashes to equal the predecessor's. Batch-2 and batch-33 transcripts must equal the predecessor's.
- For each ordered route: passes the full calibration plan on the emulated device (11,754 words), and checks that the plan for each other ordered route refuses this route's dispatches.
- Checks that fourteen invalid selector values are refused before allocation or dispatch.
- Repeats the numerical observations of the [candidate guide](directcompute-gemv-candidate.md), which are unchanged; both variant models are bit-equal to ascending FP32 in every scenario.

[gemv_variants_test.cpp](../ChandraNative/runtime/gemv_variants_test.cpp) checks four things:

- **Selector table.** Exact selectors, sixteen invalid spellings and the shape table, against an independent statement of it.
- **Staging algebra.** The `staging::stage()` index algebra is checked exhaustively over every K tail 1…2·TILE+1, the pinned widths, both weight types, row offsets 0/1/1,024/1,025, six output-tail sizes and the largest uint32-addressable shard. Each live row's tile words must load exactly once, each position must be stored exactly once with the element it denotes, and nothing else may load or store. Every output of 1…1,024-output dispatches must have exactly one owner.
- **HLSL structure.** The barrier rule (only at the top of the uniform K loop, never governed by a condition), no early exit, no FP16/`mad`/`fma`/`dot`/wave/atomic tokens, defines equal to the helper, and arithmetic text equal to `linear_gemv.hlsl`. The two variants must differ only in selector and geometry defines. Each rule is shown to reject eight mutated sources.
- **Forecast.** The per-token forecast above.

Deliberately mutating `stage()` (dropping the parity word, a short word count, forced even parity) makes the staging checks fail.

The fake device runs a C++ transliteration of ordered8 and a C++ model of the variants built on `staging::stage()`. Neither executes HLSL. They establish algorithm, indexing, coverage, barrier-arrival and host-integration properties, not fxc, driver or A770 behaviour. The correspondence between the model and the HLSL text rests on review, the source-structure checks and the native order-sensitive calibration. Clang and AddressSanitizer/UndefinedBehaviorSanitizer builds repeat the helper and candidate checks. On one core, the complete script took 337 seconds; the largest child process (a compiler) peaked at 1.19 GB resident.
