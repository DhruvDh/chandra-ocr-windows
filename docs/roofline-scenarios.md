# Native geometry and conditional traffic scenarios

The [offline model](../benchmarks/roofline.md) separates current serial B1 geometry, hypothetical true batching and queued client demand. Run it without importing a model or using a device:

```sh
python benchmarks/roofline.py --queue-concurrency 15 --prefix 3617 --patches 12096 --completion 397
python benchmarks/roofline.py --batch 15 --prefix 3617 --patches 12096 --completion 397
```

The first command records queue demand with unchanged B1 arithmetic; it does not admit 15 requests. The second assumes 15 active rows in one model invocation, which the current native engine does not implement. `native_engine` reports serial B1 with one active and one waiting slot. The `chandra.roofline-inputs.v1` and `chandra.roofline-estimates.v1` schemas remain unchanged; `conditional_cached_decode_dram_scenario` reports conditional traffic, while `capacity_terms.native_request_resident_cache` reports source-defined native allocation. Measured timing, effective bandwidth, compatible compute, clock, hardware ceiling and sustained page rate remain unknown. No XMX peak or driver compatibility is assumed.

## Native storage and initial prefill

The original Zotero page is 1,624×2,100 pixels with 3,617 prompt tokens, context 16,384 and output allowance 12,384. Completion 397 is an analytical scenario, not an observed native completion length. Weight storage is BF16; KV and recurrent storage are four-byte FP32 floats through raw `R32_TYPELESS` views. `TextModel::newRequest` reserves every KV position immediately:

| Fixed native request cache component | Bytes |
| --- | ---: |
| Eight K/V pairs, each buffer 16,384×1,024×4 bytes | 1,073,741,824 |
| Twenty-four recurrent matrices, each 32×128×128×4 bytes | 50,331,648 |
| Twenty-four convolution histories, each 8,192×4×4 bytes | 3,145,728 |
| `bytes_per_request.total` | 1,127,219,200 |

The separate `recurrent_plus_final_kv_bytes` and `recurrent_plus_allowance_kv_bytes` fields describe occupied recurrent/KV storage for hypothetical model rows, exclude convolution history and carry `logical_occupied_storage_not_resident_allocation` status. Occupied context is `P + G − 1`, since the last predicted token is not consumed. Neither those terms nor the fixed cache describes total process memory, admission or observed fit; model weights, activations, temporary buffers and other memory remain separate.

`phases.text_prefill.native_chunked_prefill_per_request` records the native0 source schedule: 56 full 64-token chunks plus a 33-token tail, or 57 successive chunks. Each chunk visits 7,138,181,120 bytes of distinct text projection weights; one 1,271,398,400-byte vocabulary-head sweep follows all chunks. `projection_weight_sweep_bytes` totals 408,147,722,240 bytes under a conditional one-sweep-per-chunk assumption. The separate ideal whole-prefix `projection_weight_bytes_once` is 8,409,579,520 bytes. Repeated shader row-group requests and cache reuse across groups/chunks prevent either number from being equated with off-chip traffic; native prefill `dram_traffic_bytes` remains null.

Whole-prefix full-attention and softmax lifetime outputs retain explicit ideal alternate-schedule labels. They do not describe the current chunked native buffer peak. BF16 weight storage also does not establish a BF16 matrix compute roof; any later calibration must match actual instructions, precision and shapes.

## Cached decoding, with prefill counted once

Let P be prefix tokens, G completion tokens, B hypothetical simultaneous model rows, W text projection/vocabulary weights in bytes, K stored KV bytes per page token, and R FP32 recurrent matrix bytes per page. Current native execution has B=1 regardless of queued demand. Prefill produces completion token one. Thus D = G − 1 cached forwards remain, whose attention context lengths are P+1 through P+D. At cached forward j:

```text
Q_batch(j) = W + B*K*(P+j) + B*K + 2*B*R + Q_other(j), j=1,...,D
Q_page = D*W/B + K*(D*P + D*(D+1)/2) + D*K + 2*D*R + sum(Q_other)/B
```

The terms represent shared projection weights, attention KV reads, new KV writes, recurrent matrix reads/writes, and omitted traffic. Attention KV reads include the newly appended token: this scenario assumes cache write followed by attention read; fusion can avoid that immediate off-chip read. The new JSON field reports integer batch totals and per-page amortization separately. Completion=1 produces zero cached-decode traffic, while vision and prefill work remain positive.

This is logical traffic **conditionally assumed to stream through DRAM**, not a proved DRAM lower bound. Cache reuse and fusion can reduce off-chip bytes; rereads, head expansion, concatenation and scratch copies can increase them. Shared weight amortization at B>1 requires actual engine batching, which current native source does not implement. The recurrence term assumes a full matrix read and write per forward and omits convolution history. It also omits other activations, scratch, nonlinear work, prefill, vision, transfers and host/client costs.

From the pinned model dimensions and native four-byte KV storage, W=8,409,579,520 bytes, K=65,536 bytes and R=50,331,648 bytes. With P=3617, G=397 and D=396, the `bytes_per_completed_page` scenarios are as follows. Decimal GB denotes 1,000,000,000 bytes:

| Conditional cached-decode term | Current B1 geometry, decimal GB/page | Hypothetical B15, decimal GB/page |
| --- | ---: | ---: |
| Shared weight reads | 3330.193489920 | 222.012899328 |
| Attention KV reads | 99.020832768 | 99.020832768 |
| New KV writes | 0.025952256 | 0.025952256 |
| Recurrent matrix reads and writes | 39.862665216 | 39.862665216 |
| Subtotal before other traffic | 3469.102940160 | 360.922349568 |

Queue concurrency 15 keeps the B1 column unchanged; the B15 column assumes an unimplemented multi-row invocation. Neither subtotal supplies timing or pages/sec because actual DRAM traffic and compatible empirical roofs are unknown, and prefill, vision and host work remain separate. Weight amortization varies with hypothetical B; per-page KV and recurrent terms do not. A future variable batch needs active B_j and individual context lengths at each invocation. Static batches drain, padding wastes work and replenishment adds prefill; queued requests cannot be treated as permanently active equal rows.

## Completing the model and testing its implications

The source chunk schedule accounts for projection operand sweeps, while prefill traffic still needs KV writes, projected Q/K/V and MLP activations, recurrence chunk scratch/state and materialized attention-score/probability traffic. Do not multiply R by every prefix token without inspecting the chunk implementation. Peak allocation is not summed traffic. Vision needs patch embedding/merger, activations and attention by each image/window segment; never square patches concatenated across unrelated pages. Prefill's vocabulary head produces the first completion token and must not also be charged as a cached forward.

A later admitted measurement should establish ordinary phase time, actual B1 invocation shapes and queue demand on a numerically qualified native worker. Paired unprofiled timing and low-perturbation device timelines test submission gaps. Counter-measured off-chip bytes and achieved bandwidth test memory limitation; precision/shape-compatible arithmetic and instruction selection test compute limitation. High device busy coverage alone does not distinguish them. Admission, CPU image preparation, transfers and finished-row waste can also create a plateau.

Use separate arithmetic/resource demands and actual memory levels before applying `max(F/C, Q/BW)` to a phase. Include serialized launch/host service costs on the critical path, preserve overlap, and divide complete correct pages by steady-state system elapsed time. No invented occupancy or bandwidth-efficiency percentage closes the gap between this scenario and measured sustained throughput.

The following **historical nominal-input appendix (checked 2026-10-04)** reproduces the original dated section verbatim so the [GEMM component methodology](directcompute-gemm-candidates.md), [vision calibration](directcompute-vision-calibration.md) and [retained throughput history](sustained-throughput-history-2026-10-05.md) keep their source route. The unchanged GEMM methodology additionally records the nominal 4,096 FP32-lane assumption; vision calibration uses the nominal 560 GB/s and 34.4 TFLOP/s FP16 context below. These historical reference assumptions are separate from the current unknown `empirical_roofs` and `analytical_hardware_ceiling`; they establish no compatible native FP32/XMX rate, driver compatibility, observed fit, timing or PPS. The active vendor-derived page-rate estimates remain removed.

## Nominal vendor inputs

These inputs were checked on 2026-10-04. They describe reference hardware arithmetic and off-chip bandwidth, not measured sustained performance. Decimal GB/s and TFLOP/s are used; multiply/add counts as two operations.

| Nominal input | RX 7900 XTX | Arc A770 16 GB reference |
| --- | --- | --- |
| DRAM bandwidth | 960 GB/s | 560 GB/s |
| Clock used in arithmetic derivation | Up to 2.5 GHz boost | 2.1 GHz graphics clock |
| BF16 matrix throughput | 512 FLOP/clock/CU × 96 CUs = 49,152 FLOP/clock; 122.88 TFLOP/s at that boost clock | 65,536 XMX BF16 operations/clock; 137.6256 TFLOP/s at that graphics clock |
| FP16 matrix throughput | Same architectural rate; product page rounds to 123 TFLOP/s | Same XMX rate as BF16 |
| FP16 vector throughput | Product page labels 61.4 TFLOP/s | 16,384 FLOP/clock; 34.4064 TFLOP/s at the reference clock |
| BF16 vector or FP32 recurrent-state compatible ceiling | Unknown in this model | Unknown in this model |
| Effective bandwidth, shape-compatible compute and ordinary launch cost | Unknown | Unknown |

AMD's [product specifications](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html) give DRAM bandwidth, CU count, boost clock and separately labelled FP16 vector/matrix rates. Its [RDNA 3 WMMA article](https://gpuopen.com/learn/wmma_on_rdna3/) explicitly gives 512 FLOP/clock/CU for both FP16 and BF16, with BF16 input/FP32 accumulation instructions. The derived BF16 matrix figure therefore uses a precision-specific architectural rate, not FP32 or marketing TOPS.

Intel's [A770 16 GB specifications](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html) give bandwidth, 512 vector engines and the reference graphics clock. Its [Xe-HPG architecture tables](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html) distinguish vector FP16 from DPAS XMX FP16/BF16 and give their per-clock rates.

A reference graphics clock is not a universal maximum; maximum boost is not a sustained clock commitment. Partner-board memory clocks can differ. Installed kernels must actually use compatible DPAS/WMMA instructions and matrix shapes; recurrent FP32 arithmetic, softmax and vector/nonlinear operations require separate compatible roofs. These numbers are not calibration inputs to the executable model.
