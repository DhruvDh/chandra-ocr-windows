# Parallel32 GPU dispatch measurement

An actual model-free comparison on the Intel Arc A770 found ordered/parallel32 GPU duration ratios of 2.0204–2.4931 for the larger K = 2,560 and 9,216, N = 1,024 linear calls. K = 1,001, N = 9 measured 1.6063–1.7321, while K = 67, N = 9 slightly regressed. These are exploratory single-sample dispatch results: each output choice has one measurement per arm, with no warmup or repetitions.

| K | N | Output | Ordered GPU ms | Parallel32 GPU ms | Ordered / parallel32 |
| --- | --- | --- | --- | --- | --- |
| 67 | 9 | FP32 | 0.02052083 | 0.02223958 | 0.9227 |
| 67 | 9 | BF16-rounded | 0.02093750 | 0.02192708 | 0.9549 |
| 1001 | 9 | FP32 | 0.07036458 | 0.04062500 | 1.7321 |
| 1001 | 9 | BF16-rounded | 0.07119792 | 0.04432292 | 1.6063 |
| 2560 | 1024 | FP32 | 0.16531250 | 0.08182292 | 2.0204 |
| 2560 | 1024 | BF16-rounded | 0.16557292 | 0.07781250 | 2.1278 |
| 9216 | 1024 | FP32 | 0.54817708 | 0.22875000 | 2.3964 |
| 9216 | 1024 | BF16-rounded | 0.56458333 | 0.22645833 | 2.4931 |

Both arms used the same accepted executable, fixed synthetic operands and phase contract, their accepted ordered or parallel32 shader, the same physical A770 and original driver 32.0.101.8991. Four shapes and two output choices produced eight dispatches per arm. The timestamps use a 19,200,000 Hz frequency, and every measured interval was non-disjoint. GPU durations come from timestamp differences rather than host wall time; the optional host drain, copy and readback timing recorder was disabled, so those durations are unavailable rather than zero.

The actual measured outputs passed the independent exact dyadic-product error-bound checks and exact FP32-to-BF16 conversion checks. Parallel32 changes arithmetic order through a fixed 32-lane reduction tree; this qualification covers the documented finite synthetic inputs. The route applies only to genuine one-row linear calls, including cached decoding and the final vocabulary head. Multirow vision and text prefill are outside this measured scope.

Finished review conditionally accepts the component measurement, numerical provenance and actual custody. Separate closure evidence confirms natural native retirement, retired Jobs with zero active processes and closed handles, joined watchdogs, exited exact owners and closed transport/local groups. The numerical certificate itself retains false admission, external-closure, full-OCR and page-performance flags; it does not supply those separate authorities. The [redacted evidence receipt](../benchmarks/evidence/directcompute-parallel32-measurement-2026-10-07.json) binds the accepted metadata and both captures by byte count and SHA-256 without publishing payloads.

An earlier candidate was refused at preflight because the historical endpoint was already absent, before that candidate existed. A supported one-time cold restoration preceded the fresh measured candidate; no inference was automatically replayed and no exit cause was established. A wrong restoration filename failed before execution, and the first offline comparison refused a missing CPU-set property before comparing captures; the corrected comparison supplied that property, retained the same gate and performed no hardware work. These early tool failures remain root-observed narratives, with no durable success receipts invented for them.

One sample provides no uncertainty estimate, statistical significance, sustained performance or ceiling measurement. This result qualifies neither whole-model correctness, complete OCR, page throughput nor deployment. The [original-weight import discrepancy](directcompute-import-boundaries-2026-10-07.md) still occurs before inference arithmetic and holds whole-model OCR progression. Nothing was deployed.
