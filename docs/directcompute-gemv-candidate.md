# DirectCompute B1 GEMV candidate

This is an experimental, inactive-by-default projection kernel for batch-row-1 `linear()` calls on the Windows Intel Arc A770. The [October 6 checkpoint](../benchmarks/evidence/directcompute-2026-10-06.json) records successful MSVC/fxc compilation, native synthetic calibration and an original tiny caller-EOS run. All 395 token IDs and the complete scalar journal match the readback-fixed baseline; the observed cold interval fell from 616 to 359 seconds. This is one sequential point per route, without full trained vectors or sustained-throughput qualification. The immutable predecessor is commit `888ab33`, whose complete original graph produced nine greedy tokens matching the CPU FP32 reference. With the selector unset, this source issues the same dispatches as that commit.

## Selector

`CHANDRA_EXPERIMENTAL_GEMV_B1` is read once, during static initialization of `operators.cpp` (before `main`), and never re-read; later environment changes have no effect. Valid values are exactly: unset or `0` for the predecessor route, and `ordered` for the candidate. Every other value, including an empty value, `1`, different case or surrounding spaces, makes every subsequent `linear()` call throw `std::invalid_argument` before shape validation, output allocation or dispatch, whatever its batch size. With `ordered`, rows equal to one always use `runtime/linear_gemv.hlsl` and other batch sizes keep `runtime/linear.hlsl`; shader compile or dispatch failures propagate, and there is no fallback. `chandra::dc::experimental::gemvB1Selection()` in `operators.cpp` exposes the parsed state without changing `api.h`.

The unchanged inference CLI first reaches `linear()` after model upload, so an invalid value there is refused only at that point. Root can check the value first without creating a device: `gemv-candidate-calibration.exe` without `--execute` prints `{"state":"INACTIVE",...,"gemv_b1_selection":...}` and exits 0, or exits 1 for an invalid value.

## Kernel and order

The predecessor schedules 16 output columns × 16 batch rows per group. For B1, 15 of every 16 rows are inactive, adjacent lanes load weights K elements apart, and each 32-element K step costs two group barriers (576 per group at K=9216). The model-free calibration measured about 3.56 ms for one K=9216, 1,024-output B1 dispatch, about 5.3 GB/s of weight traffic.

[linear_gemv.hlsl](../ChandraNative/shaders/runtime/linear_gemv.hlsl) uses 256-thread `cs_5_0` groups owning 8 consecutive outputs, so it needs at most 128 groups per ≤1,024-output dispatch. For each 512-element K tile, 32 lanes per output row load consecutive 32-bit weight words. All loads are issued into registers before any groupshared store, together with the FP32 input slice. The loaded data is expanded into a padded `8 × 513` float tile (18,464 bytes of groupshared memory with the input tile), and then lanes 0–7 continue their own sums. The sum is the predecessor's operation sequence unchanged: `precise` FP32 `sum` from +0, `sum = sum + a[k] * w[k]` in ascending k for the complete K, bias after the complete dot, then the same explicit BF16 RNE (NaN quieting, signed-zero and Inf preservation). Packed BF16 rows may start at odd elements; word `j` of a tile supplies positions `2j - parity` and `2j + 1 - parity`. FP32 weights stage one word per element. The kernel uses the same 32-byte cbuffer, SRV/UAV bindings, shard offsets, ≤1,024 outputs and complete-K dispatch boundary as the predecessor. It uses no FP16, split-K, reassociation, wave intrinsics, DirectML, transposition or prepacking.

I chose complete ascending order over a K-lane groupshared reduction. The predecessor exposes strided weight loads, inactive rows and frequent barriers; the ordered tile design addresses those costs. Their individual contribution to trained decode time has not been measured. An ordered tile kernel removes the wasted rows and most barriers while keeping each output bit-identical to the predecessor's arithmetic by construction, so qualification can compare against the predecessor exactly. Its cost is one serial chain of up to 9,216 dependent FP32 additions per output, with only 8 of 256 lanes summing between barriers. A 32-lane split with a groupshared tree would stream more parallel loads but changes FP32 results (below), so it would need higher precision, trained and complete OCR qualification. The CPU harness evaluates that split only as a rejected alternative; it is not implemented on the GPU.

## Forecasts for one decode token

These are structural counts from the pinned shapes and the 128 MiB shard plan, not measurements:

| Quantity | Predecessor | Candidate |
|---|---|---|
| Linear dispatches | 1,431 | 1,431 |
| Unique linear weight bytes read | 8,409,579,520 | 8,409,579,520 |
| Workgroups | 86,278 (16×16 threads) | 172,547 (256 threads) |
| Group barriers | 16,425,920 | 2,053,150 |
| Thread multiply-adds issued | 67,280,568,320 | 4,204,789,760 |
| Input vector staging reads | 1,051,258,880 bytes from row-0 lanes (its ternary also evaluates loads for inactive rows) | 2,102,425,600 bytes; the ≤36 KiB vector is expected to stay cache-resident |

No device buffer is added. Each dispatch still writes the same output words, and the candidate loads each packed weight word once per dispatch where the predecessor loads each BF16 word once per element. These forecasts do not establish faster dispatches, decode or sustained pages per second.

## Public calibration

[gemv_candidate_calibration.cpp](../ChandraNative/runtime/gemv_candidate_calibration.cpp) is the Windows entry and [gemv_candidate_calibration.h](../ChandraNative/runtime/gemv_candidate_calibration.h) holds the portable plan. It is inactive without `--execute`. Execution requires `CHANDRA_EXPERIMENTAL_GEMV_B1=ordered` at process start, an absolute shader root containing `runtime/linear_gemv.hlsl`, the exact PCI and LUID, and a fresh absolute receipt path. The owner supplies the external Job, the 256 MiB process commit cap and one below-normal-priority core. The source deadline is 120 seconds; the external Job enforces retirement, because the source cannot preempt a running dispatch.

First, nine malformed B1 admissions must throw `std::invalid_argument` without tracked allocation or dispatch: K=9,217, activation extent, packed activation, shard offset gap, dtype flag, uncovered rows, packed word count, partial row and bias width. Six phases then run in increasing size, each in one nondisjoint timestamp window, with every output word compared against CPU expectations prepared before dispatch:

| Phase | K × outputs | Weights, shards, bias | Calls | Dispatch groups per call |
|---|---|---|---|---|
| specials | 4 × 7 | BF16 [7], BF16 bias | 4 | 1 |
| ragged odd K | 37 × 21 | BF16 [6, 9, 6], BF16 bias | 2 | 1, 2, 1 |
| FP32 weights | 67 × 40 | FP32 [17, 23], FP32 bias | 2 | 3, 3 |
| multi-chunk | 1,001 × 2,600 | BF16 [1,500, 1,100] | 2 | 128, 60, 128, 10 |
| full width dyadic | 9,216 × 1,024 | BF16 [1,024] | 2 | 128 |
| full width order-sensitive | 9,216 × 1,024 | BF16 [1,024] | 2 | 128 |

Calls cover FP32 and BF16 returns. Dyadic phases use exact binary64 expectations whose products and partial sums are exact in FP32 for any order, so they cannot distinguish summation orders. The full-width dyadic phase reuses `dispatch_calibration.cpp`'s B1 generator for comparison with its 3.56 ms observation. The specials phase covers +0 from negative-zero products, overflow to ±Inf, Inf−Inf NaN, BF16 ties in both directions, and −0/−Inf bias. Its expectations are sequential CPU FP32 with at most one nonzero product (or one ±Inf pair) per output, so they are order-independent; an expected NaN requires an observed NaN, which after BF16 rounding has a zero low half and the quiet bit. The last phase uses public non-dyadic operands that never produce subnormal or nonfinite intermediates. It requires bitwise equality with the CPU sequential ascending FP32 dot, so the native kernel must reproduce the predecessor's order and rounding.

The receipt records operand FNV-1a hashes, expected and observed output words, mismatch indices, exact BF16 tie counts, profiled shader names, groups and per-dispatch GPU milliseconds. Each phase forecasts at most 18,919,424 tracked bytes plus readback. A mismatch, a dispatch at or above 100 ms, unexpected shader or geometry, a forecast breach or a deadline breach stops before any larger phase, and there is no rerun.

After `ChandraNative\runtime\build.cmd` has produced `device.obj` and `operators.obj` in the build directory (its `shader-compile.exe` step also compiles `linear_gemv.hlsl`), build and run from that directory:

```bat
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%CHANDRA_SOURCE%\ChandraNative\runtime\gemv_candidate_calibration.cpp" device.obj operators.obj /Fe:gemv-candidate-calibration.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib
set CHANDRA_EXPERIMENTAL_GEMV_B1=ordered
gemv-candidate-calibration.exe --execute --shader-root "%CHANDRA_SHADER_ROOT%" --pci "%CHANDRA_PCI%" --luid "%CHANDRA_LUID%" --output "%CHANDRA_FRESH_GEMV_CALIBRATION_FILE%"
```

## CPU checks

`ChandraNative/runtime/gemv_candidate_test.sh FRESH_DIRECTORY` builds [gemv_candidate_test.cpp](../ChandraNative/runtime/gemv_candidate_test.cpp) with g++ against this checkout's `operators.cpp`, and against the predecessor `operators.cpp` read from Git. It runs each selector state in its own process; the run is single-threaded, deterministic, and peaked at 95,124 KiB resident in the recorded run. A fake `Device` keeps `device.cpp`'s dispatch-validation rules and executes C++ transliterations of `linear.hlsl` and `linear_gemv.hlsl`. The transliterations use D3D11 float32 rules (sign-preserving flushing of subnormal operands and results), bounds-check every raw load and store, and fail on unstaged or stale groupshared reads, groupshared store races, out-of-row tile positions and duplicate or missing output stores. They test the algorithm, addressing and host integration, not fxc, the driver or the A770.

The suite checks four things. First, the default-route dispatch transcript (shader, groups, cbuffer words, SRV/UAV ids and output hashes) for unset and `0` is byte-identical to the predecessor build. Second, twelve shapes through the real `linear()`, all bit-equal to an independent ascending FP32 reference computed from generator values: K from 1 to 9,216, ragged groups and chunks, odd-K BF16 parity, FP32 weights, BF16/FP32 bias, specials, two-chunk K=9,216, and batches 2 and 33. Third, candidate dispatch parameters and geometry against an independent forecast, unchanged batch>1 transcripts under `ordered`, eight invalid selector values refused before allocation or dispatch, and the full calibration plan passing on the emulated device and refusing the predecessor route. Fourth, the numerical observations below.

On 64 representative outputs per K, the CPU observations compare ascending FP32 (the predecessor and the candidate, bit-equal in every scenario) with the rejected split32 tree, both against compensated binary64 (exact products, Neumaier summation). These are observations, not thresholds:

| Scenario | split32 FP32 bits differ | split32 BF16 output differs | max \|error\| / Σ\|terms\|, ascending | same, split32 |
|---|---|---|---|---|
| random K=2,560 | 57/64 | 0/64 | 5.6e-8 | 7.7e-9 |
| random K=4,096 | 63/64 | 1/64 | 5.1e-8 | 7.5e-9 |
| random K=9,216 | 59/64 | 0/64 | 5.6e-8 | 4.5e-9 |
| ±2^20 pair absorbs small terms | 62/64 | 62/64 | 1.4e-6 | 2.5e-7 |
| alternating ±2^12 | 64/64 | 44/64 | 5.5e-8 | 1.3e-8 |
| signed-zero products | 0/16 | 0/16 | 0 | 0 |
| exponents −60…60 | 55/64 | 0/64 | 3.6e-7 | 6.5e-8 |
| near BF16 tie 1+2^-8 | 250/256 | 27/256 | 3.9e-6 | 7.3e-7 |
| MAX, MAX, −MAX, −MAX | 1/1 | 1/1 | +Inf | NaN |

The tree is usually closer to binary64 but changes predecessor bits, flips BF16 outputs near ties and under cancellation, and can turn an overflow +Inf into NaN. Taking it would be a numerical change requiring qualification, not an equivalent schedule.

## Validation and remaining work

Build08 compiled the source and all runtime shaders with MSVC/fxc. Native calibration passed six phases, nine malformed admissions and all 9446 independent output-word checks across 26 dispatches. The full-width dyadic dispatches measured 0.5466 and 0.5447 ms; the maximum across all candidate dispatches was 1.3073 ms, below the 100 ms admission gate. The earlier separate-session predecessor B1 dyadic measurements were about 3.56 ms. This synthetic kernel comparison is not a paired page-throughput result. The normal tiny request preserved output allowance 12384 and all 395 baseline tokens/scalar records, with clean owned closure. See the [closed-point checkpoint](../benchmarks/evidence/directcompute-2026-10-06.json) for identities, resources and exact measurements.

Full trained projection/logit vectors, complete original corpus and AMD/exact-client acceptance remain pending. Device CPU accounting is being implemented to locate costs outside this kernel. Repeated uninstrumented sustained page measurements at safe concurrency are required before promoting a node throughput gain. Geometry differences inherited from the baseline, native endpoint integration and lifecycle qualification remain open.

The serial chain, input re-staging, 8-output groups and 512-element tiles are untuned; if native timing is poor, change the tile shape before admitting any reassociated order. If hardware were to contract `precise` multiply-add or treat subnormals differently from the D3D11 rule, the order-sensitive phase is designed to expose it rather than hide it.
