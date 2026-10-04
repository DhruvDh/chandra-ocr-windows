# Improving Windows OCR performance

These experiments start from the qualified hybrid BF16 implementation. No optimized candidate has been promoted. Follow the [numerical gates](../../verification/hillclimb-numerical-gates.md), obtain an exclusive GPU lease and run through the project's owned Windows Job Object channel before allocating a model. The ordinary Windows service stays stopped during those experiments; the independent ROCm gateway remains available.

The baseline representative page completed correctly with 397 output tokens. Its separate unprofiled warm request took 85.497 seconds; the instrumented request took 126.375 seconds. The trace covers one prefill and eight cached forwards and reveals approximately 2,740 GPU kernels per decoding token. Instrumentation substantially changes execution time, so it identifies opportunities without establishing an attainable speedup.

| Experiment | Current evidence | Method and tools |
| --- | --- | --- |
| Baseline profile | Complete correct output; many small recurrent and normalization operations | [Method](method.md), [runner](profile_baseline.py) |
| Explicit-mask text attention | Original numerical gate failed; early probability distributions also moved further from CPU FP32. Set aside. | [Finding and method](attention_candidate.md) |
| Compiled recurrent updates | Tiny trained-operator errors, but the complete cached trajectory failed the frozen numerical policy at six positions. Set aside. | [Method](recurrent_method.md), [candidate](recurrent_candidate.py) |
| Cached normalization gains | All 395 logit rows and 25,280 cache records match baseline exactly; complete OCR and performance remain pending | [Method](norm_method.md), [candidate](norm_candidate.py) |
| Native fused normalization | Trained operator replay completed; 35 of 2,935,296 outputs differed, all during prefill. Full-model qualification pending. | [Method](norm_method.md), [exporter](norm_export.py) |
| Native XPU graph replay | The installed A770 runtime reports graph recording/replay unsupported; no replay comparisons ran | [Probe method](graph_method.md) |

The [compact evidence record](../../benchmarks/evidence/hillclimb-2026-10-04.json) commits to retained private receipts. Raw traces and trained tensors remain outside Git. Compiler diagnostics and partial runs are evidence of those attempts, never successful qualification or speed measurements. The [independent synthetic pages](../../benchmarks/qualification-v1/README.md) broaden the original development template's regression scope.
