# DirectCompute exact-input correctness: text softmax barrier and multiwave regression

This guide documents the uniform group barrier in [text_attention_softmax.hlsl](../ChandraNative/shaders/runtime/text_attention_softmax.hlsl) and its standalone model-free regression. Finished source review, strict Windows compilation and the A770 run pass: all 17 cases and 102 dispatches meet their numerical, timing and resource gates. The longest GPU interval is 0.691771 ms; no model is loaded. The [redacted evidence record](../benchmarks/evidence/directcompute-text-softmax-regression-2026-10-06.json) binds source, build, hardware, independent reviews and current-runtime CPU integration. This repairs a demonstrated shared-memory hazard; it does not establish the cause of the [second original page’s numerical failure](directcompute-exact-client-recovery.md), trained numerics, full OCR, endpoint acceptance or throughput.

The control-only `bf16ModelBits` helper has a retained P3 defect: it treats finite `2^127` as infinity. Current generated probabilities and sensitivity controls never reach that range, so the defect does not affect this finite regression; correct it before general reuse. The independent hardware review rebuilt all input hashes and oracle metadata, plus complete nearest-BF16 output hashes for twelve cases. Five cases use derived adjacent-word gates and retain native full-buffer checks and hash commitments without raw output tensors. The original control showed no physical failure in this sample, which is allowed and does not disprove the source hazard.

## Status at this source cutoff

| Item | Outcome at this cutoff | Scope |
|---|---|---|
| Shared-maximum barrier in `text_attention_softmax.hlsl` | Present, SHA-256 `283d3356ea3c882517e96ddff26706bae21bfe3cd04a523fc065e9aa8183169e`; independently reviewed | Source, CPU and finite model-free GPU |
| Original shader at `2c44182` | Preserved as `ChandraNative/shaders/control/text_attention_softmax_2c44182.hlsl`, SHA-256 `5b6c0445dcebe2acdc8104bf1c0fd2b37e1c05665d30d960ca4a0e83d6432369` | Negative control, never production |
| Translated-HLSL emulation, layer state and recorded-geometry tests | Pass on GCC 16.2.1 and Clang 23.1.1, warning-free; see [CPU results](#cpu-results) | Linux/POSIX CPU seam |
| Pinned upstream geometry comparisons | Pass with Python 3.12.14, Torch 2.14.1+cpu, Transformers 5.18.0 on both compilers | Indexing, positions and interpolation; no trained arithmetic |
| Compiler classification defect (`clang++` received a GCC-only option) | Repaired: the family comes from predefined macros | Test driver only |
| Rotary description | Corrected to the actual uniform absolute 2^-23 gate; it is not a ULP bound or ULP measurement | Test text only |
| New multiwave regression driver | Strict GCC/Clang/sanitizer and MSVC builds pass; all 17 A770 cases and 102 dispatches pass | Model-free kernel only |
| Root's Build20 instrumented normal CLI run | 556 IDs through caller EOS 248046; 18 of 18 literal content checks pass; 55 selected records equal to the capped run | Independently reviewed functional and baseline-relative scope; no numerical oracle |
| Build16 uninstrumented worker, normal allowance | Failed: 29 IDs decoding to one Blank-Page block; 0 of 18 content checks | Preserved failure |
| Interrupted worker request after the barrier | Interrupted by NorthStone's reboot; no terminal result | No replay or acceptance |
| Fresh ordinary worker, first original page | 556 IDs through caller EOS; output and journals equal the CLI; all 18 content checks pass | One functional fixture, independently reviewed |
| Ordinary worker, second original page | 1,952 alternating unrelated IDs; no EOS; 0 of 18 content checks | Rejected; safely canceled and externally closed |
| Fresh trained numerics, remaining corpus, native endpoint and throughput | Not qualified | Root-owned and open |

## Producer record

The first Claude producer of this assignment stopped with exit 1 after 4,258.076418 seconds, when its OAuth session expired and could not be refreshed. It wrote no final report. Its frozen 10-file partial source (127,214 bytes) went to independent review, which accepted the barrier argument and found the Clang classification defect, the incorrect ULP wording and the missing guide. A resumed producer began those test-text repairs, but NorthStone rebooted during it. That producer has no completion, exit code or report, and root froze its 10-file source. The source implementation and its retained final report come from a third, fresh producer. It verified all 10 files against that reboot-closed manifest before editing. Root subsequently added the finished Windows, hardware and integration evidence summarized here. Neither earlier boundary is converted into a completion here.

## Workload and preserved runs

The recorded Zotero CLI request uses decoded RGB pixels of 1,624 × 2,100 and 3,617 prompt tokens. Its grid is `[1,126,96]`, with 12,096 original patches and 3,024 merged rows; image rows occupy prompt rows 4 through 3,027. The pinned model is `datalab-to/chandra-ocr-2` at `af93b47dba1b47b6640c86ccf487ed2260ab9a09`, with context 16,384 and normal output allowance 12,384. The input manifest SHA-256 is `0aa588473930f110cc359ba369623f29765a85f4e05ffcbca2ac509933b787ae`. Its three-axis prompt positions hash to `698cadcea94a3ce34a9e33848f591dfcea1786ef8dd32a6db707bd6c087c493a`. None of these values, BF16 cast boundaries, FP32 arithmetic, selectors, diagnostic opt-in behavior, leases or capacity ownership change here.

The Build16 worker request with the normal allowance returned 29 IDs including caller stop 248046. They decode to `<div data-bbox="0 0 1000 1000" data-label="Blank-Page"></div>`, and all 18 literal content checks fail. That sealed failure remains a failure. Its 14-token diagnostic prefix and the tiny page's 395-ID result keep their earlier scopes, without OCR, trained numerical, endpoint or performance acceptance.

Root's closed Build20 instrumented CLI run includes the barrier. With the original fixture and no diagnostic cap, it produced 556 IDs through caller EOS 248046 and passed all 18 literal text, equation, table and end-marker checks. All 55 selected diagnostic records are exactly equal between the 14-token capped run and the normal run over identical consumed tokens and positions. That equality is baseline-relative: both sides are this native graph, so it is not an independent numerical oracle. The run also carried the CLI diagnostic observer, which the uninstrumented worker does not. Its success therefore cannot be attributed to the barrier alone. Independent review accepts that first-fixture functional observation. A later worker request was interrupted by NorthStone's reboot and remains interrupted without replay. A distinct fresh ordinary-worker request subsequently produced the same 556 IDs, journal and decoded bytes. The second original native page then failed from its first prediction, with extreme and all-zero tied logits, no EOS and no expected content. These distinct outcomes remain preserved in the [original-page record](directcompute-exact-client-recovery.md).

## The shared-maximum hazard and the repair

The shader runs one 128-lane group per query-head row and keeps one `groupshared float tmp[128]`. Each lane takes the maximum over its strided keys, and a seven-level tree reduces those maxima into `tmp[0]` behind group barriers. Every lane then reads `m=tmp[0]` and sums `exp(score-m)` over its keys. Each lane stores that partial in `tmp[lane]` for a second tree, then divides and writes each probability rounded to BF16. Before the repair, nothing ordered the read `m=tmp[0]` against lane 0's later store `tmp[0]=total` in the same barrier interval. A lane in another wave that read `tmp[0]` after that store used lane 0's exponential partial as its maximum. That changed its exponentials, the shared denominator and its stored probabilities.

The repair inserts `GroupMemoryBarrierWithGroupSync()` immediately after `m=tmp[0]`. The barrier is on the uniform path, so every lane completes the read before any lane overwrites shared storage. Microsoft documents this intrinsic as completing group shared memory accesses and synchronizing all threads in the group ([GroupMemoryBarrierWithGroupSync](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync)). The final denominator read `total=tmp[0]` needs no second barrier, because this shader never writes `tmp` again. The production vision softmax already uses the same read-then-barrier sequence. The tracked diff is exactly that barrier plus one leading comment line; a CPU test rebuilds the original bytes from the repaired file.

Lanes of one hardware wave execute each instruction together, so a wave cannot overwrite `tmp[0]` before its own lanes read it. The hazard can therefore change an output only if a lane outside lane 0's wave owns a key, which means `count` exceeds the wave width. Intel compute waves have 8, 16 or 32 lanes, so the existing operator fixture's five keys always lie in lane 0's wave and cannot expose the hazard. Rows with at least 128 keys give every lane of every wave a key.

## CPU evidence and its boundaries

[tests/native/hlsl_cpu.py](../tests/native/hlsl_cpu.py) translates the unmodified runtime HLSL into C++. [hlsl_cpu.h](../ChandraNative/runtime/hlsl_cpu.h) runs each group's lanes as cooperative fibers under ascending, descending or seeded-shuffle schedules, with each lane's barrier interval atomic. It reports conflicting groupshared or UAV accesses, uninitialized reads, out-of-bounds addresses and divergent barriers. Arithmetic is host FP32 without contraction; HLSL `exp`/`sin`/`cos` implementation error and driver compilation are not modelled. These lane-granular schedules include orders that hardware waves cannot produce, so they demonstrate a source hazard, not a hardware observation. The fiber emulator uses POSIX `ucontext`, and the recorded-geometry driver uses POSIX file APIs. Both are Linux CPU seams. Neither is an MSVC test, a GPU test or part of `runtime/build.cmd`.

[test_exact_input_correctness.py](../tests/native/test_exact_input_correctness.py) builds four drivers against the unchanged graph sources. Each test sits at one of three distinct levels:

- Synthetic numerical layer tests ([text_state_test.cpp](../ChandraNative/runtime/text_state_test.cpp)). These execute layer 0 (Gated DeltaNet) and layer 3 (full attention) on 140 synthetic rows with synthetic BF16 weights. They check five chunkings, including 140 cached single-row calls, on both decode routes. Output and carried state stay exact, causality and initialization controls hold, and interleaved requests isolate. Under the 2c44182 softmax the attention layer depends on the schedule, while the repaired layer does not. Only two synthetic layers and 140 rows are evaluated numerically.
- The 3,617-row dispatch/state-length trace ([recorded_geometry_test.cpp](../ChandraNative/runtime/recorded_geometry_test.cpp)). This drives the unchanged graph at the recorded geometry with most large arithmetic stubbed. It checks tiles, offsets, cache lengths, RoPE coordinates, merge rows and the seven tracked allocation phases of the failed Build16 result. The Windows-only merge and driver lifetime sequence is reproduced in the test from source review, not compiled from `inference_core.cpp`. Its dry-run diagnostic dumps carry declared zero payloads. It validates geometry, dispatch, index and accounting paths, not image content, GPU residency, trained arithmetic or a cause.
- Shader-level hazard tests ([shader_emulation_test.cpp](../ChandraNative/runtime/shader_emulation_test.cpp)). These cover 27 text-softmax cases (nine shapes under three schedules), including the recorded 33-row tail at base 3,584 over 3,617 keys. The repaired shader has zero findings and identical bits across schedules, while the original has conflicting accesses and is wrong under the ascending schedule. Five other barrier shaders serve as clean controls, and the detector flags deliberate hazards.

[test_recorded_geometry_upstream.py](../tests/native/test_recorded_geometry_upstream.py) runs against the actual pinned upstream source: `modeling_qwen3_5.py` SHA-256 `d0c8561d91e31e50d9a57b74196c5a7ad023e42ebe9f84004d0a535310879939` and `vision_utils.py` SHA-256 `bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356`. Vision position IDs, text M-RoPE, cached decode positions and image-row alignment are equal. Learned-position interpolation of a synthetic BF16 table is bit-identical after the upstream BF16 cast. Native FP32 cos/sin use host libm, not HLSL; they pass a uniform absolute 2^-23 gate at every element. That gate equals two FP32 ulps only for magnitudes in [0.5, 1), and no ULP distance is measured. Only synthetic tensors and model metadata are used.

The compiler-classification repair runs the selected compiler with `-dM -E` and classifies it by predefined macros: `__clang__` means Clang, `__GNUC__` without `__clang__` means GCC, and anything else gets no family-specific option. Only GCC receives `-Wno-dangling-reference`, which silences a GCC false positive in the unchanged `vision_model.cpp`. The independent reviewer's failed stock-Clang gate remains part of the record. The later manual Clang build without that option did not erase it; this repair is what makes the Clang suite pass unmodified.

## Multiwave regression driver

The driver dispatches the actual shader through the unchanged production `Device`, using production cbuffer words and groups. It checks every output word against an independent binary64 normalizer under prospective BF16 gates. It also runs the byte-identical original as a non-gating comparison control. It reads no model, weight, pixel or private asset.

| File | Role |
|---|---|
| [text_softmax_regression.h](../ChandraNative/runtime/text_softmax_regression.h) | Portable core: plan, generator, oracle, gates, sensitivity controls, run loop, admission and receipt logic |
| [text_softmax_regression.cpp](../ChandraNative/runtime/text_softmax_regression.cpp) | Windows host and `wmain`: CREATE_NEW receipt, Job and identity queries, console cancellation (MSVC only) |
| [text_softmax_regression_fake_device.h](../ChandraNative/runtime/text_softmax_regression_fake_device.h) | Test-only Device with a wave-granular model of both shaders and fault injection |
| [text_softmax_regression_test.cpp](../ChandraNative/runtime/text_softmax_regression_test.cpp), [text_softmax_regression_test.sh](../ChandraNative/runtime/text_softmax_regression_test.sh) | Strict GCC/Clang/sanitizer unit and lifecycle suite |
| [text_softmax_regression_emulation_test.cpp](../ChandraNative/runtime/text_softmax_regression_emulation_test.cpp) | Translated-HLSL harness and production transcript check, run from the Python suite |
| [text_attention_softmax_2c44182.hlsl](../ChandraNative/shaders/control/text_attention_softmax_2c44182.hlsl) | Byte copy of the original shader for the control |

It reuses the reviewed helpers of [vision_dispatch_calibration.h](../ChandraNative/runtime/vision_dispatch_calibration.h), SHA-256 `05e08c107c9c96856a84604dbd594087df9987ca2330a90d1f20bea33e5a78ef`. Those are SHA-256, strict arguments and pins, bounded receipt excerpts and the D3D11 error constants. It calls only the unchanged Device API ([api.h](../ChandraNative/runtime/api.h) `563c7555da0b5c06c7a17cadc0722b8e252b7c9b867283dfe1a684e14bf5f9d1`, [device.cpp](../ChandraNative/runtime/device.cpp) `0fff6aa94cfe41f0738682718b1fedec5ef547a9fbfeab470114554e12e36d73`). Device admission, the 128 MiB buffer limit, the 13 GiB tracked cap, the 2 GiB WDDM headroom rule and the bounded readback wait all stay unchanged. No production file, selector or build script changes.

### Plan

Each case is one production attention call: `rows` query rows after `base` cached tokens, `count = base + rows` keys and `16·rows` groups. Row `r` admits keys `0..base+r`, and later keys hold the scores shader's `-3.402823466e38` mask. The cbuffer is `{rows,16,0,0,base,count,0,0}`, exactly as `TextModel::mix` writes it. The translated harness confirms the shader name, all eight words, the groups and the plane size against the unchanged `TextModel` attention layer, run record-only at every planned call.

| Case | base | rows | count | groups | valid keys per row | waves with keys (32-lane) |
|---|---:|---:|---:|---:|---|---:|
| `single_key_decode_c1` | 0 | 1 | 1 | 16 | 1 | 1 |
| `operator_fixture_like_c5` | 2 | 3 | 5 | 48 | 3–5 | 1 |
| `first_prefill_tile_c64` | 0 | 64 | 64 | 1,024 | 1–64 | 2 |
| `decode_c127` | 126 | 1 | 127 | 16 | 127 | 4 |
| `decode_c128` | 127 | 1 | 128 | 16 | 128 | 4 |
| `decode_c129` | 128 | 1 | 129 | 16 | 129 | 4 |
| `ragged_tail_c129` | 96 | 33 | 129 | 528 | 97–129 | 4 |
| `second_prefill_tile_c128` | 64 | 64 | 128 | 1,024 | 65–128 | 4 |
| `tiny_prompt_tail_c801` | 768 | 33 | 801 | 528 | 769–801 | 4 |
| `recorded_full_tile_c3584` | 3,520 | 64 | 3,584 | 1,024 | 3,521–3,584 | 4 |
| `recorded_tail_c3617` | 3,584 | 33 | 3,617 | 528 | 3,585–3,617 | 4 |
| `recorded_last_row_c3617` | 3,616 | 1 | 3,617 | 16 | 3,617 | 4 |
| `recorded_decode0_c3618` | 3,617 | 1 | 3,618 | 16 | 3,618 | 4 |
| `recorded_decode4_c3622` | 3,621 | 1 | 3,622 | 16 | 3,622 | 4 |
| `normal_allowance_last_c16000` | 15,999 | 1 | 16,000 | 16 | 16,000 | 4 |
| `context_bound_tail_c16384` | 16,382 | 2 | 16,384 | 32 | 16,383–16,384 | 4 |
| `context_bound_decode_c16384` | 16,383 | 1 | 16,384 | 16 | 16,384 | 4 |

The 801-key case is the last prefill tile of the original tiny page. The recorded cases cover the last full 64-row tile, the 33-row causal tail and a full 3,617-key row. They also cover decode step 0 and decode step 4, which consumes generated token 4 and produces generated index 5, the earlier first divergence. The 16,000-key case is the largest count this prompt reaches within its normal allowance (3,617 + 12,384 − 1). The 16,384-key cases sit at the context bound. Each case runs the repaired shader three times, then, after every repaired case has passed, the original three times: 102 dispatches in all.

All sizes are derived with checked 64-bit arithmetic and refused before any action if any bound fails: 1–64 rows, at most 16,384 keys, at most 1,024 groups, at most 4,194,304 score words per dispatch, at most 128 keys per lane, at most a 128 MiB buffer, at most 32 MiB of device buffer plus staging, and at most 128 MiB of host vectors. The plan holds 7,489,984 score words (7,367,808 valid and 122,176 masked). The largest case, `recorded_full_tile_c3584`, needs 29,361,152 device-plus-staging bytes and a forecast of 80,857,088 host bytes. Each dispatch's per-lane work is at most 128 keys. The closed vision calibration's 512-group, 12,096-key softmax took 0.341 ms on the A770, so these dispatches are roughly three orders of magnitude below a TDR. A measured dispatch at or above 100 ms still stops the run.

### Generator, oracle and gates

Generator `public_lowbias32_dyadic_causal_text_scores_v1` writes valid scores as `j/16` with `|j| ≤ 255`. These are BF16-exact values times 1/16, like the scores shader's `bf(q·k)·0.0625`. Row kinds cycle so that every case, including each 16-group decode, has spread rows (`j` in −128..127), uniform rows (one constant per row) and spike rows. A spike row's maximum sits on the last valid key, key 0, lane 127 or a hashed key, so the maximum moves between lanes and waves. A spike exceeds the other scores by about 20 to 28, so probabilities reach down to 7.8e-13 without approaching FP32 subnormals.

The oracle reads only the generator's integers. For each row it computes `p_k = exp(x_k)/Σexp(x_j)` in binary64 with a Neumaier sum, where `x_k = (j_k − j_max)/16` is exact. It never reads device output. The gate for each valid word is derived beforehand from the shader's operation sequence and named assumptions, which the closed vision calibration also uses. FP32 add, subtract and multiply round to nearest even. The base-2 exponential has relative error at most 2^-21. FP32 division is within 2.5 ULP, as D3D11 specifies. With `u = 2^-24`:

- Exponential argument: `|t̂ − x·log2 e| ≤ Δ = log2(e)(|x| + 2|s| + 2|m|)·4u`. This covers `fl(fl(s−m)·c)` and a fused or unfused `mad(s, c, −fl(m·c))` with `c = fl32(log2 e)`, because the shader has no `precise`. The term bound is `τ = 2^Δ(1 + 2^-21) − 1`. The final loop recomputes the exponential independently of the sum loop, and the bound treats the two computations independently.
- Sum: each lane adds at most `⌈count/128⌉` terms from +0, followed by a seven-level tree. Every term passes through at most `n = ⌈count/128⌉ − 1 + 7` roundings under any association inside a lane, giving `γ_n = nu/(1 − nu)`. Masked terms are exact zeros.
- Probability: `B = max{(1+τ)(1+φ)/((1−r)(1−γ_n)) − 1, 1 − (1−τ)(1−φ)/((1+r)(1+γ_n))} + 2^-36`, where `r` is the row's largest `τ` and `φ = 2.5·2^-23`. The largest B over the whole plan is 4.90e-5, so it is about 1/80 of BF16's 2^-8 relative rounding.
- Stored word: BF16 round-to-nearest-even is monotone, so a correct shader stores a value in `[RNE(p(1−B)), RNE(p(1+B))]`. The oracle rounds binary64 to BF16 exactly. For 99.62% of valid words this interval holds one BF16 value; for 28,110 words (0.38%), lying within B of a BF16 midpoint, it holds two adjacent values. No gate admits three.

Every word of every readback is checked: the score plane plus 128 trailing guard words. Valid words must be finite, positive, BF16-stored (low 16 bits zero) and inside their interval. Masked words must be exactly +0, since `exp2(−∞) = +0` under D3D11, and guard words must be unchanged. Uniform rows must be bit-identical across their valid keys. Each row's sum must satisfy `|Σ p̂ − 1| ≤ 2^-8(1 + B_row) + B_row + valid·2^-52`; this bound uses no oracle probability. The three repaired repetitions must be bit-identical. A different GPU transcendental rounding can legitimately move a word to the adjacent value within a two-value interval. The receipt reports each case's count of words equal to the correctly rounded binary64 probability and its count of admitted adjacent words, instead of widening any gate. If the A770's `exp` or division falls outside these named assumptions, the driver reports a failed case; no tolerance is relaxed.

After the first repaired readback of each case passes, the same check must refuse a set of deliberately wrong outputs, or the run fails as non-discriminating:

- One word one BF16 step above or below its interval.
- A NaN word, and a word with its low bits set.
- A masked word set to the smallest normal BF16.
- An overwritten guard word.
- The longest row raised by one BF16 step.
- Binary64 models of the original hazard for 32-lane and 16-lane waves running in ascending order.
- Lane 127's partial dropped from the denominator.

Each hazard model applies only when a key-owning lane lies outside lane 0's wave. It is reported as not applicable for the one-key and five-key cases, which is the measured reason the old fixture cannot see the hazard.

### Original-shader control

The control dispatches `control/text_attention_softmax_2c44182.hlsl` on exactly the same inputs; the driver verifies that their SHA-256 matches the repaired case's input. Its results are recorded but never gate the run: oracle rejection, bits that differ from the repaired output, and repetitions that disagree. A physical schedule exposes the hazard only if a later wave reads `tmp[0]` after lane 0's wave has finished its exponential loop. That window is short and depends on scheduling, so a clean control on the A770 is expected to be common and is not evidence that the original is correct. Lifecycle failures during the control pass, such as a disjoint timer, a 100 ms dispatch, tracked bytes or a deadline, still fail the run.

### Execution lifecycle

Without `--execute`, the executable prints `INACTIVE` with the 17 case names, 102 planned dispatches and the plan hash. It creates no Device and no file, and loads no model. The options are exactly `--execute`, `--shader-root`, `--pci`, `--luid` and `--output`, parsed by the vision calibration's strict parser: each at most once, values present, nonempty and not option-shaped. The PCI pin must be lowercase `bb:dd.f` and the LUID `hhhhhhhh:llllllll`, and paths must be absolute. Execution additionally requires that:

- the shader root exists;
- the receipt path is fresh, in an existing directory and outside the shader root;
- the process already runs in an external Job with kill-on-close and a memory limit of at most 1 GiB;
- no cancellation has been requested;
- `CreateFileW(CREATE_NEW)` creates the receipt.

The driver only queries the Job; it never creates, joins or modifies one, and the receipt records `created_by_this_process: false`. Unknown, duplicate or malformed arguments, repeated or existing output paths, a missing Job and early cancellation all exit 2 (3 for cancellation) without a receipt or Device.

After the receipt exists, the driver performs these steps in order:

1. It records the executable's SHA-256 and `_MSC_FULL_VER`.
2. It verifies both shader files against their frozen sizes and SHA-256 values before any Device exists.
3. It reads the user-mode driver version for the pinned LUID through DXGI.
4. It creates the Device with the exact PCI and LUID, then re-authenticates the reported PCI, LUID, Intel vendor and A770 description.

Each of the 102 dispatches then follows the same sequence:

1. Allocate a fresh score buffer from the uploaded inputs and check tracked bytes against the forecast.
2. Persist the receipt and open a profile window, which drains first.
3. Dispatch once and close the window.
4. Require a nondisjoint timer, exactly one record with the planned shader and groups, and a finite duration below 100 ms.
5. Read back the buffer, release it, drain and require zero tracked bytes.
6. Run the host checks.

Cancellation and the absolute 120-second source deadline are checked at each case boundary, before each allocation, after each pre-dispatch receipt persistence, after the profile window opens, before Device creation, and every 64 score rows during host work. The deadline is never reset. Ctrl+C or Ctrl+Break requests cancellation, and close, logoff and shutdown wait up to 4 seconds for the receipt. After a failure or cancellation, the driver persists the receipt first, then attempts exactly one bounded drain and records `tracked_after`; nothing is resubmitted. After the Device is destroyed, it re-hashes both shaders, which must be unchanged, and records the loaded `d3dcompiler_47.dll` path, size and SHA-256. Receipts use the vision calibration's bounded serialization: a report above 8 MiB becomes a failing `RECEIPT_OVERSIZE` summary. Exit codes are 0 for pass or inactive, 1 for failure, 2 for refusal and 3 for cancellation. A device call that is already running stays bounded by the Device's own 10-second drain and readback limits. Root's external timer and kill-on-close Job remain the hard bound, and the driver never touches services.

## CPU results

The finished source producer ran the following CPU commands in its isolated checkout on one CPU through a task-owned runner with a bounded timeout, recording exit, wall time and the largest waited descendant's peak RSS. No GPU, D3D11, fxc, model, network, dependency installation or service action was involved. A host daemon re-niced new shell processes in this sandbox to −4 shortly after they started. The final runner therefore sets itself to niceness 15 before forking and raises any process found below 15 back to 15 every 0.2 seconds. It counts those corrections, so a process could spend at most about 0.2 seconds below niceness 15 before correction.

| Final command | Result | Wall, peak RSS | Niceness corrections |
|---|---|---|---|
| `ChandraNative/runtime/text_softmax_regression_test.sh FRESH` | g++ and clang++ strict builds emit no diagnostics; each passes 9,166,418 checks with byte-identical summaries; the ASan/UBSan quick build passes 9,165,300 | 103.3 s, 881,188 KiB | 10 |
| `CXX=g++ python -I tests/native/test_exact_input_correctness.py -v` | 12 of 12 tests pass | 158.5 s, 708,724 KiB | 2 |
| `CXX=clang++ python -I tests/native/test_exact_input_correctness.py -v` | 12 of 12 tests pass | 172.3 s, 576,464 KiB | 3 |
| `CXX=g++ python -I tests/native/test_recorded_geometry_upstream.py -v` | 3 of 3 tests run and pass (none skipped) | 14.4 s, 781,092 KiB | 2 |
| `CXX=clang++ python -I tests/native/test_recorded_geometry_upstream.py -v` | 3 of 3 tests run and pass (none skipped) | 9.0 s, 781,108 KiB | 2 |

`python` is the supplied pinned CPU environment (Python 3.12.14, Torch 2.14.1+cpu, Transformers 5.18.0); GCC is 16.2.1 and Clang is 23.1.1. The upstream test skips unless both upstream source hashes and both package versions match, so each test appears by name rather than as a skip. Three earlier runs used the first runner, which logged niceness only at start, so parts of them may have run at −4: an 11-test GCC baseline before the driver existed (149.4 s), a first pass of the strict script and a first GCC suite pass. A complete earlier pass of all five final commands under the enforcing runner also passed, before two include-only header edits; its logs are retained, and the final runs above, on the frozen source, supersede all of them. A first Clang suite pass was deliberately terminated midway once the −4 niceness was noticed. Its partial log and exit −15 are retained; that termination is not a test failure.

In the unit suite, the fake Device's wave model shows the repaired shader with identical bits for 8-, 16-, 32-, 64- and 128-lane waves and both wave orders. With ascending wave order, the original is rejected in exactly the cases whose count exceeds the wave width: 15 cases at 8, 16 and 32 lanes, and 14 at 64 lanes. It is rejected in none at 128 lanes or in descending order. On the translated actual HLSL, all 17 production transcripts equal the driver's calls. The repaired shader passes the oracle with zero findings and identical bits under all three schedules. The original produces race findings in every case. Under the lane-granular ascending schedule it is rejected in all 16 cases with at least two keys, and under the descending schedule it equals the repaired output. For each fault, the lifecycle matrix stops at the expected dispatch and retains the receipt. Once a Device exists, it also records one bounded failure drain; the failing-drain case records that drain as failed:

- a repaired shader without the barrier, at dispatch 7, the first 64-key case;
- a dropped lane partial, at dispatch 13, the 128-key case;
- a guard store, a masked-word write, a NaN, and an admissible but unrepeatable word;
- six timer faults, and a 100 ms dispatch in each pass;
- a failing drain, and an 8 MiB driver message;
- cancellation inside a dispatch, inside profile preparation and before Device creation;
- deadline expiry inside profile preparation, during the receipt persistence after a completed dispatch (stopping before the next allocation) and after the last dispatch;
- missing driver, producer or compiler identity, a refused Device and a mismatched PCI or vendor;
- shader bytes changed before or during the run.

## Root commands

Build after the authenticated runtime build has produced `device.obj` in `%CHANDRA_RUNTIME_BUILD%`. `CHANDRA_SOURCE` is the authenticated LF checkout and `CHANDRA_FXC_OUT` is a fresh task-owned directory:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
cd /d "%CHANDRA_RUNTIME_BUILD%" || exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 /WX "%CHANDRA_SOURCE%\ChandraNative\runtime\text_softmax_regression.cpp" device.obj /Fe:text-softmax-regression.exe /link d3d11.lib dxgi.lib d3dcompiler.lib bcrypt.lib gdi32.lib > text-softmax-regression-compiler.log 2>&1 || exit /b 1
fxc /nologo /T cs_5_0 /E main /O3 /Gis /Fo "%CHANDRA_FXC_OUT%\text_attention_softmax.cso" /Fc "%CHANDRA_FXC_OUT%\text_attention_softmax.asm" "%CHANDRA_SOURCE%\ChandraNative\shaders\runtime\text_attention_softmax.hlsl" || exit /b 1
fxc /nologo /T cs_5_0 /E main /O3 /Gis /Fo "%CHANDRA_FXC_OUT%\text_attention_softmax_2c44182.cso" /Fc "%CHANDRA_FXC_OUT%\text_attention_softmax_2c44182.asm" "%CHANDRA_SOURCE%\ChandraNative\shaders\control\text_attention_softmax_2c44182.hlsl" || exit /b 1
text-softmax-regression.exe
```

`/W4 /WX` is the strict gate. If an inherited header warns, retain the log and report it; do not drop `/WX` or suppress the warning without review. `/O3 /Gis` matches the Device's `D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS` runtime compilation. The listings should show the repaired shader's extra `sync_g_t` after the `tmp[0]` load. The listings are review evidence; the Device compiles from source at run time. The last command must exit 0 and print `"state":"INACTIVE"`, `"device_created":false`, `"dispatches":102` and `"plan_sha256":"e19e8f0a82b201e769e2d60069e485dca4e9374d4b92cdf5f29717d83575d213"`, the CPU suite's hash. Stop if the hash differs.

Execute once inside a root-owned kill-on-close Job. The proposed limits are a 768 MiB process commit, a 1 GiB Job commit (the driver refuses a memory limit above 1 GiB), one core at below-normal priority and an external wall timer of about 180 seconds. No other model may hold the GPU. `CHANDRA_SHADER_ROOT` is the absolute `ChandraNative\shaders` directory of the authenticated LF source, containing both `runtime` and `control`:

```bat
text-softmax-regression.exe --execute --shader-root "%CHANDRA_SHADER_ROOT%" --pci "%CHANDRA_PCI%" --luid "%CHANDRA_LUID%" --output "%CHANDRA_FRESH_TEXT_SOFTMAX_RECEIPT%"
```

Hardware admission of the barrier regression requires all of the following:

- The process exits 0 with state `TEXT_SOFTMAX_REGRESSION_PASS` and the plan hash above.
- Both shader commitments are equal before and after the run, and the executable, MSVC version, UMD driver version and `d3dcompiler_47.dll` hash are recorded.
- All 102 dispatches are submitted, completed and measured, each in a nondisjoint window strictly below 100 ms.
- All 17 repaired cases pass every word, normalization and repetition check, and all applicable sensitivity controls are refused.
- Each dispatch and the final drain leave zero tracked bytes.
- The Job ends with zero active processes and commit peaks within its caps, and root restores the ordinary worker.

Record the original control's outcome, but do not require it to fail. Any other outcome retains the receipt, requires owned retirement and is not rerun automatically. Passing establishes the repaired kernel's multiwave synchronization and BF16 output on the A770 for these synthetic inputs only, not trained numerics, OCR or throughput.

## Next trained comparison

This follows the independent review and stays root-owned and separately admitted.

1. Use the closed native-0 capped/normal pair as a baseline-relative control: its selected full-vocabulary rows and shared-prefix state agree, including decode step 4. For the rejected native-1 page, begin at last-prefill/first-prediction values, since generated index zero is already invalid. Authenticate the consumed prompt, complete prefix and all three position axes before comparing values; differing prefixes make later logits incomparable.
2. Compare those rows with a fresh, independently and correctly attested pinned upstream trained producer, using a higher-precision oracle where needed. Preserve the original BF16 cast boundaries and continuously carried caches. Start with last-prefill logits plus decode steps 0 through 4, and prompt rows 63, 64, 3,583, 3,584 and 3,616 at layers 0 and 3. Bisect later text layers only when the first disagreement requires it. Compare GDN state at completed lengths 64, 3,584, 3,617, 3,618 and 3,622.
3. If merged conditioning already disagrees, compare patch, position, selected vision block and merger boundaries against the same trained producer; sparse matches cannot certify all vision rows.
4. Reuse the strict selected-row recorder with a newly authenticated narrow plan and its byte and readback forecast. Do not use the broad 1,023-record dry-run plan (`c37044dd…`, resolved `0e4832ac…`, 1,552,353,280 forecast readback bytes) as the first trained capture.
5. After the hardware and trained gates pass, run the original uncapped 1,624 × 2,100 page through the uninstrumented worker. It must complete through caller EOS with its prose, equations, table associations, layout and all 18 content checks.

Historical custody, toy agreement, natural EOS, a lower-resolution page, an output cap or an allocation-size match cannot replace any of these steps. A model-attestation or reference-custody gap blocks numerical promotion.

## Remaining gates

- Diagnose and repair the second original page’s first-prediction failure, preserving the exact original input and normal generation policy.
- A fresh, correctly attested upstream trained reference on the same original input, with tokens and all position axes authenticated before values are compared.
- Complete original native/scanned corpus verification after the trained numerical gates; the first fixture’s closed functional observations do not qualify the corpus.
- Endpoint integration and sustained correctly-completed-pages-per-second measurement. Both remain out of scope until correctness qualifies.
