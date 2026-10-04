# Conditional hardware and cached-decode scenarios

The offline model can explain how batching changes resource demand without pretending to measure the GPU. Run it without importing a model or using a device:

```sh
python benchmarks/roofline.py --batch 15 --prefix 3617 --patches 12096 --completion 397
```

Existing partial phase costs and persistent-state outputs remain available. The additive `conditional_cached_decode_dram_scenario` reports a separately labelled traffic scenario. Empirical bandwidth, precision/shape-compatible compute, ordinary launch cost and sustained page rate remain null. Queue concurrency is not the `batch` argument: it means active rows within one actual model invocation.

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

## Cached decoding, with prefill counted once

Let P be prefix tokens, G completion tokens, B simultaneous active rows, W text projection/vocabulary weights in bytes, K stored KV bytes per page token, and R FP32 recurrent matrix bytes per page. Prefill produces completion token one. Thus D = G − 1 cached forwards remain, whose attention context lengths are P+1 through P+D. At cached forward j:

```text
Q_batch(j) = W + B*K*(P+j) + B*K + 2*B*R + Q_other(j), j=1,...,D
Q_page = D*W/B + K*(D*P + D*(D+1)/2) + D*K + 2*D*R + sum(Q_other)/B
```

The terms represent shared projection weights, attention KV reads, new KV writes, recurrent matrix reads/writes, and omitted traffic. Attention KV reads include the newly appended token: this scenario assumes cache write followed by attention read; fusion can avoid that immediate off-chip read. The new JSON field reports integer batch totals and per-page amortization separately. Completion=1 produces zero cached-decode traffic, while vision and prefill work remain positive.

This is logical traffic **conditionally assumed to stream through DRAM**, not a proved DRAM lower bound. Cache reuse and fusion can reduce off-chip bytes; rereads, head expansion, concatenation and scratch copies can increase them. One shared weight read requires actual fused batching. The recurrence term assumes a full matrix read and write per forward and omits convolution history. It also omits other activations, scratch, nonlinear work, prefill, vision, transfers and host/client costs.

From the pinned model dimensions, W=8,409,579,520 bytes, K=32,768 bytes and R=50,331,648 bytes. With B=15, P=3617, G=397 and D=396:

| Conditional cached-decode term | Decimal GB per page |
| --- | --- |
| Shared weight reads | 222.012899328 |
| Attention KV reads | 49.510416384 |
| New KV writes | 0.012976128 |
| Recurrent matrix reads and writes | 39.862665216 |
| Subtotal before other traffic | 311.398957056 |

Nominal bandwidth divided by that subtotal gives illustrative cached-decode-only ratios of 3.083 pages/s for AMD and 1.798 for Intel. They are conditional memory scenarios, not full-page throughput predictions, measured plateaus or proof that batch 15 fits. Prefill, vision and serialized host work must be charged separately. Weight amortization improves with B; per-page KV and recurrent terms do not. A variable batch needs active B_j and individual context lengths at every invocation. Static batches drain, padding wastes work, and continuous replenishment introduces new prefill work. None can be represented by treating 15 queued requests as 15 permanently active equal rows.

## Completing the model and testing its implications

Prefill traffic needs KV writes, projected Q/K/V and MLP activations, recurrence chunk scratch/state, and any materialized attention-score/probability traffic. Do not multiply R by every prefix token without inspecting the chunk implementation. Peak allocation is not summed traffic; fused attention can avoid full quadratic score materialization. Vision needs patch embedding/merger, activations and attention by each image/window segment; never square patches concatenated across unrelated pages. Prefill's vocabulary head produces the first completion token and must not also be charged as a cached forward.

A later admitted measurement should establish ordinary phase time and active rows at safe batches, including the neighborhood of 15 if capacity permits. Paired unprofiled timing and low-perturbation device timelines test submission gaps. Counter-measured off-chip bytes and achieved bandwidth test memory limitation; shape-compatible BF16 throughput and instruction selection test matrix limitation. High device busy coverage alone does not distinguish compute from memory. Admission, CPU image preparation, transfers and finished-row waste can also create a plateau.

Use separate arithmetic/resource demands and actual memory levels before applying `max(F/C, Q/BW)` to a phase. Include serialized launch/host service costs on the critical path, preserve overlap, and divide complete correct pages by steady-state system elapsed time. No invented occupancy or bandwidth-efficiency percentage closes the gap between this scenario and measured sustained throughput.
