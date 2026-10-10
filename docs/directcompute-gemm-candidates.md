# DirectCompute GEMM candidates (inactive)

Two standalone tiled GEMM kernels, [linear_gemm_padded32.hlsl](../ChandraNative/shaders/runtime/linear_gemm_padded32.hlsl) and [linear_gemm_padded64.hlsl](../ChandraNative/shaders/runtime/linear_gemm_padded64.hlsl), target the multi-row `linear()` dispatches of Chandra OCR 2 text prefill and vision projections on the Intel Arc A770. They have exactly the binding, constant-buffer and dispatch ABI of production [linear.hlsl](../ChandraNative/shaders/runtime/linear.hlsl). Nothing selects these GEMM candidates: this GEMM change adds no operator, API, worker or serving selection. Multi-row model dispatches retain `linear.hlsl`, and the existing opt-in single-row GEMV routes are unaffected. A model-free calibration, [gemm_candidate_calibration.cpp](../ChandraNative/runtime/gemm_candidate_calibration.cpp), dispatches production and both candidates directly through the unchanged `Device` on identical public operands. Every output word must match an exact CPU oracle and the predecessor, and every untouched word its sentinel. Source review, CPU checks, strict Windows compilation and the closed model-free GPU calibration are complete. The calibration found exact output equality and faster synthetic GEMM dispatches, as recorded below. Driver ISA and occupancy remain unmeasured. No trained model, OCR or page-throughput result exists for these kernels.

## Production predecessor

`operators.cpp` splits every `linear()` call into dispatches of at most 32 batch rows and at most 1,024 outputs of one weight shard, always with the complete K (at most 9,216). Each dispatch binds `t0` FP32 activations, `t1` the weight shard (packed BF16 or FP32), `t2` the bias (or the weight again when there is none), `u0` the FP32 output, and an eight-word cbuffer `{inputWidth, outputWidth, batchRows, firstOutput, shardRows, rowFirst, flags, weightRowFirst}`. Flag bits are BF16 weights (1), BF16 output rounding (2), bias present (4) and BF16 bias (8). Groups are `ceil(shardRows/16) × ceil(batchRows/16)` of 16×16 threads, each thread owning one output. Prefill runs 64-token tiles, that is two 32-row dispatches, and vision projections use 32-row dispatches of their row tiles; their K is 1,024, 1,536, 2,560, 4,096 or 9,216. `linear.hlsl` stages 16×32 input and weight tiles in `tileInput[16][32]` and `tileWeight[16][32]`. Each thread runs `precise float sum = 0`, then `precise product = a*w; sum = sum + product` for ascending k, then adds the bias, applies BF16 round-to-nearest-even if requested and stores. The closed public calibration in [directcompute-2026-10-05.json](../benchmarks/evidence/directcompute-2026-10-05.json) measured K=9,216, 1,024-output dispatches at 3.561/3.560 ms (1 row), 3.076/3.054 ms (16 rows) and 6.458/6.437 ms (32 rows). The single-row phase has the same 64 groups as 16 rows but ran first in that session, so its extra 16% is consistent with a clock ramp. That interpretation is not established.

## Candidates

| Kernel | K tile | Groupshared layout | Groupshared bytes | Barriers per group at K=9,216 |
|---|---|---|---|---|
| `linear.hlsl` (production) | 32 | `[16][32]` × 2 | 4,096 | 576 |
| `linear_gemm_padded32.hlsl` | 32 | `[16][33]` × 2 | 4,224 | 576 |
| `linear_gemm_padded64.hlsl` | 64 | `[16][65]` × 2 | 8,320 | 288 |

Both candidates keep production's declarations, `packed()`, `bf16_rne()` and epilogue (bias, rounding, store address) byte-for-byte. The CPU suite checks that textually. They also keep `[numthreads(16,16,1)]`, the dispatch geometry, `precise float sum = 0.0`, ascending tiles, `n = min(TILE_K, inputWidth - begin)` and the identical statement pair `precise float product = tileInput[lane.y][k] * tileWeight[lane.x][k]; sum = sum + product;`. Every output therefore runs the predecessor's exact sequence of binary32 products and sums, in increasing k, by construction. Ragged rows and columns stage zeros and never store, but they run every iteration. The loop count comes only from the cbuffer, there is no `return` in `main` and staging has no branch, so all 256 threads reach both barriers of every tile. Packed BF16 elements, including odd indices, input row offsets, output offsets and weight shard offsets use production's formulas unchanged. There is no split-K, reassociation, FP16, wave intrinsic, DPAS/XMX, DirectML/XPU, weight conversion, transposition or tolerance.

Three things change, and none changes any value placed in groupshared memory. First, each tile row is padded to an odd stride (33 or 65 words). Second, weights are staged like inputs: lane.y selects the tile row and lane.x + 16·part the k column, where production uses lane.x for the weight row. All of a thread's loads are issued into registers before its groupshared stores. Third, `padded64` stages K in 64-element tiles. Production versus `padded32` therefore measures the padding and the staging remap together. `padded32` versus `padded64` isolates tile depth. Separating padding from remapping would need a third kernel, which this task does not provide.

## Structural forecasts and hypotheses

These are counts from the source and stated assumptions, not compiler output or measurements.

| Access (per SIMD lane group, lane.x fastest) | Production | Candidates |
|---|---|---|
| Dot-loop weight read `tileWeight[lane.x][k]` | word `32·lane.x + k`: 16 distinct words in one bank modulo 16/32 banks, or two banks with eight words each modulo 64 banks | word `33·lane.x + k` or `65·lane.x + k`: 16 distinct banks |
| Dot-loop input read `tileInput[lane.y][k]` | one word per lane.y (broadcast); two rows 32 words apart share a bank under SIMD32 with 16/32 banks, and use different banks with 64 banks | rows 33/65 words apart use different banks |
| Weight staging store | `tileWeight[lane.x][lane.y+16p]`: 16-way same-bank with 16/32 banks, or eight-way across two banks with 64 banks | `tileWeight[lane.y][lane.x+16p]`: consecutive words |
| Weight global load, one message | 16 lanes read 16 rows K elements apart (18 KiB at K=9,216): 16 cache lines | 16 consecutive BF16 of one row: 32 bytes in one 64-byte line when K and the tile start are multiples of 32 |
| Input global load and store | 16 consecutive FP32 | unchanged |

Each staged input element is read by the 16 output lanes of its row, and each staged weight element by the 16 row lanes of its output: 16 logical uses per staged element in all three kernels. One barrier pair covers 8,192 terms (K32) or 16,384 (K64) per group. Across a dispatch, each input element is loaded by `ceil(N/16)` groups (64 at N=1,024) and each weight element by `ceil(R/16)` groups (1 at 16 rows, 2 at 32). Groupshared use is 4,224 or 8,320 bytes per group, within the 32 KiB `cs_5_0` per-group limit. Four resident groups use at most 33,280 bytes of an Xe-core's shared local memory. The forecast is that this does not limit residency, but this repository has not measured the per-Xe-core partition. A770 has 32 Xe-cores of 16 vector engines with 8 hardware threads each (public Xe-HPG figures). A 256-thread group then occupies 16 SIMD16 or 8 SIMD32 hardware threads. At these dispatch sizes (64 or 128 groups) that is 2 or 4 groups per Xe-core, at most 25% or 50% of its thread slots at SIMD16, regardless of kernel. The ABI forbids changing this, so latency hiding stays limited. Source inspection gives roughly 14–18 live 32-bit values per lane in production. The candidates add 4 (K32) or 8 (K64) staged values, for about 18–26. At SIMD16 each value occupies two 32-byte GRFs, so about 36–52 of 128 GRFs, with no spill expected. SIMD32 doubles that and might be tight for K64 if the driver also unrolls the dot loop. Which SIMD width, unrolling and SLM access widths the Intel driver chooses is unknown until root inspects the ISA.

## Roofline for K=9,216, N=1,024, BF16 weights

Nominal inputs are the repository's [reference A770 figures](roofline-scenarios.md): 560 GB/s DRAM, 2.1 GHz and 4,096 FP32 lanes. `precise` forbids fused multiply-add, so each term issues one multiply and one add. The vector floor is therefore 8.6 T operations/s, not the 17.2 TFLOP/s FMA figure. The groupshared model assumes 16 four-byte banks per Xe-core, one word per bank per clock, same-word broadcast, scalar accesses and SIMD16 lanes with lane.x fastest. No source or measurement in this repository establishes those properties for the A770 under D3D11.

| Quantity | 16 rows | 32 rows |
|---|---|---|
| Terms (multiply+add pairs) | 150,994,944 | 301,989,888 |
| Unique bytes: weights / input / output | 18,874,368 / 589,824 / 65,536 | 18,874,368 / 1,179,648 / 131,072 |
| Logical global loads: input FP32 / weight BF16 elements | 37,748,736 / 18,874,368 B | 75,497,472 / 37,748,736 B |
| Groupshared reads (two floats per term) | 1,207,959,552 B | 2,415,919,104 B |
| Vector floor (no FMA) | 35.1 µs | 70.2 µs |
| DRAM floor (with / without L2 reuse of weights) | 34.9 µs | 36.0 / 69.7 µs |
| Measured predecessor (mean) | 3.065 ms, 98.5 GFLOP/s, 87× floor | 6.447 ms, 93.7 GFLOP/s, 92× floor |
| Bank model, production (16-way weight reads) | 2.39 ms | 4.78 ms |
| Bank model, padded (conflict-free) | 0.28 ms | 0.56 ms |

The bank model alone accounts for about 74–78% of the measured predecessor times. It also explains why the time doubles with the group count while weight traffic barely changes, and why the single-row dispatch costs as much as 16 rows. That fits a per-Xe-core shared resource saturated by 16-way serialization. It is still a hypothesis. If the driver compiles SIMD32 with same-address merging, or issues vector SLM loads, or uses a different bank organization, the conflict degree and the prediction change. The candidates' conflict-free floor of 0.28/0.56 ms is about 11× below the measured predecessor (8.5× below the model's own production estimate), but it ignores staging, barriers, global-load latency at 25–50% thread residency, loop overhead and clocks. Even a confirmed speedup would bound only these dispatches. Prefill and vision also contain other operators, and nothing here measures pages per second.

Timing interpretation is hardware-dependent for further reasons. D3D11 timestamp pairs bracket the dispatch on the GPU queue. The final timestamp precedes the host drain and profile collection; host compilation and readback are outside the interval. Clocks ramp from idle and vary with power and thermal state. The first use of a shader includes driver compilation on the CPU, though not in the GPU interval. Other GPU clients and driver versions differ. The calibration therefore isolates each dispatch in its own drained profile window and counterbalances order. It reports every sample, and its synthetic repetition is not model or page acceptance.

## Calibration harness

[gemm_candidate_fixture.h](../ChandraNative/runtime/gemm_candidate_fixture.h) holds the portable plan, generators, exact oracle, admission rules and the run loop over a supplied `Device`. [gemm_candidate_calibration.cpp](../ChandraNative/runtime/gemm_candidate_calibration.cpp) adds strict arguments, Job admission, BCrypt SHA-256, producer commitments and the receipt. It links only `device.obj` and never `operators.obj`, so no serving selector is read. Without `--execute` it validates any given options, prints `{"state":"INACTIVE","device_created":false,"model_loaded":false,...}` with the case names and dispatch counts, and exits 0. With `--execute` it requires all of the following, and otherwise exits 2 without creating a receipt or device. Unknown, duplicate, empty or option-like values are refused.

- `--plan`, exactly `correctness` or `correctness-then-timing`.
- An absolute existing `--shader-root` containing `runtime/linear.hlsl` and both candidates.
- Lowercase `--pci bb:dd.f` and `--luid hhhhhhhh:hhhhhhhh`.
- An absolute `--output` in an existing directory.
- Membership of a Job that kills on close and bounds process or job memory to at most 1 GiB, checked with `IsProcessInJob` and `QueryInformationJobObject`.

The receipt is created with `CREATE_NEW`, bounded to 8 MiB, rewritten after each case and on failure. Before the device exists it records SHA-256 for the executable and the three shader files, and the shader hashes are rechecked after the run. After `Device` selects the adapter by exact PCI/LUID, its reported PCI, LUID, Intel vendor ID and A770 description are rechecked. The source deadline is 120 seconds. Only the external Job can stop a running dispatch.

Operands are public functions of their absolute indices. `splitmix_dyadic_v1` draws integers in [−16, 16] over 32; hashing makes rows, columns and k positions distinguishable, unlike a mod-33 linear form that repeats every 33 rows or outputs. Its products and every partial sum, in any order, are exact in binary32 (K·256+512 < 2^24). The two order generators use full 24-bit FP32 significands against 8-bit BF16 significands, either FP32 inputs × BF16 weights or BF16-exact inputs × FP32 weights, with 1/64 exact zeros. Their 32-bit products make both binary32 products and sums round. The oracle represents every quantity as an int64 multiple of 2^−42 and applies binary32 round-to-nearest-even after each product and each addition in ascending k, then after the bias. It derives BF16 by integer rounding to 8 significant bits. Admission proves every magnitude below 2^61, including rounding growth over K. Every nonzero value is at least 2^−42 and below 2^19, so nothing is subnormal or overflows and D3D11 denormal flushing cannot apply. Before any dispatch, each output is cross-checked against a separate CPU binary32 loop and against the shader's BF16 bit formula.

| Case | Generator | K | Rows (first+count of total) | Outputs (first+count of width) | Shard first / weightRowFirst / rows | Weights | Bias | Groups |
|---|---|---|---|---|---|---|---|---|
| `k1_single_term_b1_offsets` | dyadic | 1 | 2+1 of 3 | 3+3 of 7 | 1 / 2 / 5 | BF16 | BF16 | 1×1 |
| `k37_odd_bf16_ragged_offsets_bf16_bias` | dyadic | 37 | 3+19 of 23 | 12+45 of 61 | 5 / 7 / 55 | BF16 | BF16 | 3×2 |
| `k67_odd_fp32_weights_fp32_bias` | dyadic | 67 | 1+7 of 9 | 17+23 of 44 | 9 / 8 / 33 | FP32 | FP32 | 2×1 |
| `k96_k64_tail_bf16_fp32_bias` | dyadic | 96 | 0+32 of 32 | 16+33 of 50 | 0 / 16 / 50 | BF16 | FP32 | 3×2 |
| `k1001_odd_second_row_chunk` | dyadic | 1,001 | 32+31 of 64 | 25+1,000 of 1,030 | 24 / 1 / 1,003 | BF16 | none | 63×2 |
| `order_k2560_fp32_input_bf16_weight_bf16_bias` | order | 2,560 | 5+32 of 40 | 67+300 of 400 | 64 / 3 / 320 | BF16 | BF16 | 19×2 |
| `order_k4096_bf16_input_fp32_weight_fp32_bias` | order | 4,096 | 2+16 of 20 | 517+1,024 of 1,545 | 512 / 5 / 1,031 | FP32 | FP32 | 64×1 |
| `b1_k9216_n1024` | dyadic | 9,216 | 1+1 of 2 | 0+1,024 of 1,024 | 0 / 0 / 1,024 | BF16 | none | 64×1 |
| `k9216_n1024_r16` (timing) | dyadic | 9,216 | 1+16 of 17 | 1+1,024 of 1,027 | 0 / 1 / 1,025 | BF16 | none | 64×1 |
| `k9216_n1024_r32_second_chunks` (timing) | dyadic | 9,216 | 32+32 of 64 | 1,024+1,024 of 2,560 | 0 / 1,024 / 2,560 | BF16 | none | 64×2 |
| `order_k9216_n1024_r32_bf16_bias` | order | 9,216 | 0+32 of 32 | 0+1,024 of 1,024 | 0 / 0 / 1,024 | BF16 | BF16 | 64×2 |

Each output buffer also has 1–7 trailing guard words. The cases cover:

- K=1, odd K with odd shard-row starts (packed odd-index BF16), and K tails inside both tile sizes, including a K64 tail of exactly 32.
- Ragged rows and columns, and nonzero `rowFirst`, `firstOutput`, `weightRowFirst` and shard first rows.
- Rows and columns on both sides of the dispatched region, and BF16 bias at an odd column.
- FP32 and BF16 weights and biases, and no bias.
- The single-row fallback geometry, and K=9,216/N=1,024 at 16 and 32 rows. The 32-row dyadic case mirrors the text `down_proj` second output chunk of a prefill tile's second row chunk.

The largest case needs 50,855,944 device bytes including its readback staging copy, against a 128 MiB per-case cap that is checked exactly against `Device`'s tracked bytes. Across the plan, CPU observations in the summary include 3,803 BF16 ties rounding down to even and 3,693 rounding up. In the order cases, which are deliberately sensitive to order, fused products change 80–88% of FP32 outputs, descending k 96%, 32-wide split-K blocks 95% and a single rounding of the exact dot 95%. In the dyadic cases these alternatives change nothing. A candidate that reassociated or contracted would therefore fail the order cases.

Cases run in this order. For each case the oracle runs first, then the uploads. Two calls follow, FP32 raw and then BF16-rounded with the case's bias. Each call dispatches production, `padded32` and `padded64` in that order. Every dispatch proceeds as follows:

1. Upload a full sentinel image of NaN words, salted per dispatch.
2. Open a drained profile window and issue exactly one dispatch with the same eight constants and groups.
3. Close the window and require one completed, non-disjoint timestamp pair from the expected shader and groups. The time must be finite, nonnegative and strictly below 100 ms.
4. Read back the whole buffer. Every region word must equal the oracle and every other word that dispatch's sentinel. A candidate's region must also equal production's.

Before each larger case, each kernel's slowest dispatch is scaled by the work ratio (groups × K) with a 2× margin, and the forecast must stay below 100 ms. Any mismatch, missing or disjoint profile, unexpected shader or groups, gate breach, memory mismatch or deadline stops the run immediately, with no rerun, replay or fallback. The 66 correctness dispatches stay within production's own per-dispatch bound; nothing dispatches a whole model. After each case its buffers are released and drained, and tracked bytes must return to zero.

`correctness-then-timing` adds a timing phase only after every correctness case has passed. The two timing shapes are regenerated and hash-checked against their correctness operands. Three rounds then dispatch, for each shape, the six orders of the three kernels, alternating which shape runs first: 108 dispatches. Every timed dispatch keeps its own profile window and the same complete correctness and predecessor checks. The receipt records every sample, plus minimum, median, mean, maximum and median ratio to production per shape and kernel, labelled as synthetic repetition rather than model or page acceptance. Per dispatch, the receipt keeps the kernel, call, eight constants, groups, salt, raw profile, GPU milliseconds, expected and observed image SHA-256, observed region SHA-256, mismatch counts with up to 16 examples, and the predecessor equality flag. Per case it keeps admission bounds, oracle observations, operand hashes, expected region hashes, memory snapshots and forecasts.

## CPU checks

`ChandraNative/runtime/gemm_candidate_fixture_test.sh FRESH_DIRECTORY` needs no GPU, D3D11, fxc, model or network. It checks eight production files against `HEAD` and pinned SHA-256 values. It then builds [gemm_candidate_fixture_test.cpp](../ChandraNative/runtime/gemm_candidate_fixture_test.cpp) with GCC 16.2 and Clang 23.1 using `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror -ffp-contract=off`, and with GCC ASan/UBSan. If `glslangValidator` is already installed, it also parses the three shaders with its HLSL front end. The final recorded run, pinned to one core, passed in 145 seconds with peak child resident memory of 833 MiB. GCC and Clang each ran 60,070,014 checks over the full plan and wrote byte-identical deterministic summaries (SHA-256 `b3268b0a748e43ac5636e52e09f45c922bb891a63c1cb921d3816154f7693a0e`). The sanitized build ran 12,147,273 checks on cases of at most 50M terms. glslang accepted all three shaders, and its SPIR-V marks the candidates' products and sums `NoContraction`. That is a secondary front end, not fxc.

The test exercises the real header:

- Integer rounding and binary32/BF16 bit derivation against the CPU's conversions over 200,000 random values.
- Plan coverage and 12 admission refusals.
- Every uploaded word finite and normal or zero, with signed inputs.
- The oracle on every case against the CPU binary32 loop and the BF16 formula.
- Exactness and order insensitivity of the dyadic cases, and order sensitivity of the order cases.
- Sentinel, flip, missing-store and stale-image detection.
- Fifteen profile refusals, the growth gate, schedule balance, 26 argument refusals, identity mismatches and Job limits.
- A source-structure check that each candidate keeps production's declarations, helpers, epilogue and dot statement, with an odd padded stride, two barriers, no `return` in `main`, no branch in staging and no forbidden constructs.

A separate C++ statement of `linear.hlsl`'s per-output ABI semantics, using uint32 index arithmetic and bounds-checked loads, reproduces each oracle image and its sentinels from the uploaded words. That validates packing and addressing, not a kernel. A test-only `FakeDevice` implements `chandra::dc::Device` by mapping all three shader names to that model. It runs the full correctness plan (66 dispatches), and a reduced plan with timing (138 dispatches, largest receipt about 208 KB). It also injects ten faults: a candidate region bit flip, a sentinel store, a missing store, a predecessor/oracle disagreement, a 100 ms dispatch, a disjoint window, a fallback shader name, a growth-gate refusal and two timing-phase faults. Each must stop at that dispatch with the record retained and every buffer released. Because the fake device produces identical results for all three names, it says nothing about the HLSL, fxc, the driver or the A770.

## Root build and GPU plan

Build from a reviewed checkout in a fresh directory. `build.cmd` runs `shader-compile.exe` over every `runtime/*.hlsl`, so both candidates are compiled with the runtime's own `cs_5_0`, `OPTIMIZATION_LEVEL3 | IEEE_STRICTNESS` flags before anything executes. Save as a `.cmd` file and run it from a plain prompt:

```bat
@echo off
setlocal
if "%CHANDRA_SOURCE%"=="" (echo Set CHANDRA_SOURCE & exit /b 2)
if "%CHANDRA_BUILD%"=="" (echo Set CHANDRA_BUILD & exit /b 2)
if exist "%CHANDRA_BUILD%" (echo CHANDRA_BUILD must be fresh & exit /b 2)
call "%CHANDRA_SOURCE%\ChandraNative\runtime\build.cmd" "%CHANDRA_BUILD%" || exit /b 1
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
pushd "%CHANDRA_BUILD%" || exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%CHANDRA_SOURCE%\ChandraNative\runtime\gemm_candidate_calibration.cpp" device.obj /Fe:gemm-candidate-calibration.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib || (popd & exit /b 1)
for %%S in (linear linear_gemm_padded32 linear_gemm_padded64) do (
  fxc /nologo /T cs_5_0 /E main /O3 /Gis /Fo "%CHANDRA_BUILD%\%%S.cso" /Fc "%CHANDRA_BUILD%\%%S.asm" "%CHANDRA_SOURCE%\ChandraNative\shaders\runtime\%%S.hlsl" || (popd & exit /b 1)
)
gemm-candidate-calibration.exe || (popd & exit /b 1)
popd
```

Keep all compiler warnings. In each listing, check for `dcl_thread_group 16, 16, 1`, two groupshared declarations of 512 (production), 528 (`padded32`) or 1,040 (`padded64`) four-byte elements, and two `sync_g_t` barriers. The dot loop must be a `mul` followed by an `add`, both marked precise, with no floating-point `mad`. Driver ISA, SIMD width, registers, spills and residency need Intel tooling of root's choice; this task prescribes none.

The GPU run plan:

1. Coordinate the A770 window with the owner. No other model may be loaded on it; `Device` also refuses less than 2 GiB of WDDM headroom.
2. Launch the executable inside a fresh Job with kill-on-close, a job memory limit of 1 GiB (at most 1 GiB is admitted), an active-process limit of 1, one CPU and below-normal priority. Add an external 180-second wall-clock watchdog that closes the Job. Write receipts to fresh paths under ignored `results/`.
3. Run `gemm-candidate-calibration.exe --execute --plan correctness --shader-root "%CHANDRA_SHADER_ROOT%" --pci "%CHANDRA_PCI%" --luid "%CHANDRA_LUID%" --output "%CHANDRA_FRESH_GEMM_RECEIPT%"`. Exit 0 means `GEMM_CANDIDATE_CORRECTNESS_PASS`. Exit 2 means admission was refused before any receipt or device. Exit 1 means failure; a receipt records `FAIL_OR_INCOMPLETE` when it can be written, but receipt-creation failure returns 1 without creating a receipt or device. Do not rerun a failure; diagnose it.
4. Accept the correctness run only if all of the following hold:
   - all 66 dispatches passed, with zero region and sentinel mismatches;
   - every candidate has `matches_predecessor_region: true`;
   - the shader hashes before and after are equal and match the reviewed files;
   - the identity matches;
   - tracked memory returned to zero;
   - every growth forecast and dispatch stayed below 100 ms.

   Production's 16- and 32-row dyadic times should resemble the 2026-10-05 calibration; a large deviation calls for investigation before timing.
5. Only then run `--plan correctness-then-timing` into a new receipt (174 dispatches). Report every sample, the per-position spread and the median ratios as synthetic dispatch evidence.

Read the outcomes as hypotheses tested, not conclusions about pages:

- `padded32` near production: the bank model is wrong for this driver or hardware, and the ISA should be examined.
- `padded32` much faster: SLM serialization, global staging or both dominated; separating them needs a third kernel.
- `padded64` faster than `padded32`: barrier or latency exposure matters at this residency.
- `padded64` slower: its larger tiles or register pressure cost more than they save.

## Closed Windows calibration

The independently reviewed Build15 compiled the full runtime and standalone calibration with MSVC, and retained strict `cs_5_0 /O3 /Gis` listings for all three kernels. The listings show the intended group and shared-memory geometry, separate precise multiply/add operations and no floating-point `mad`. Ten inactive/refusal cases created no Device. A correctness-only GPU point then passed 66 dispatches, and a fresh correctness-plus-timing point passed 174, including 108 counterbalanced timing samples. Independent regeneration reproduced all 240 operand/output commitments and 13,778,862 checked words, with zero mismatches. The maximum observed dispatch was 6.446823 ms. Both points returned tracked allocations and Job descendants to zero, and restored the ordinary worker cold.

| K=9,216, N=1,024 | Predecessor median | Padded32 median | Padded32 speed ratio | Padded64 median |
| --- | --- | --- | --- | --- |
| 16 rows | 3.053802 ms | 1.835260 ms | 1.66× | 2.254661 ms |
| 32 rows | 5.309531 ms | 2.100052 ms | 2.53× | 3.230781 ms |

Each cell uses 18 samples across all six kernel orders and three rounds. All individual samples, including outliers, remain in the retained receipts. These are synthetic dispatch improvements. Padded32 is the nominee for a later explicit selector and trained checks; neither kernel is currently selectable by the model.

The reusable CPU runner now pins the integrated predecessor: the reviewed expanded GEMV selectors changed `operators.cpp`, and the separately reviewed resident-worker extraction changed `runtime/build.cmd`. Neither changes this experiment's multi-row operator body or production shader. The guard remains strict against the integrated source bytes and task-local tracked changes; earlier frozen source/test evidence remains intact. The guard repair's Claude process stopped on its subscription limit before producing its requested guide paragraph or final handoff, so independent review establishes the code/test boundary and this paragraph records the later integration.

## Outstanding qualifications

The following are root's, in order:

1. Driver ISA, register and residency inspection.
2. An explicit experimental selector for the padded32 nominee.
3. Trained numerical and complete OCR checks before any serving promotion.

Any integration then needs an explicit selector. Operator selection belongs to root and the separate selection task, not to this change. After selection, a candidate needs:

- bitwise equality with the predecessor on trained vision and prefill activations across every dispatch shape, shard split and K of the pinned graph;
- complete OCR equality on the pinned pages;
- sustained page-throughput measurement in the owner's serving configuration.

Until then the candidates are unselectable by the model, and the synthetic dispatch gains establish no model speed, page-rate or OCR improvement.
