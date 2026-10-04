# Improving Windows OCR performance

These experiments start from the qualified hybrid BF16 implementation. No optimized candidate has been promoted. New GPU trials are currently on hold while important work continues on the shared machines. The qualified Windows endpoint is available on demand, with its model unloaded; the independent ROCm gateway remains available. Current optimization work uses source, retained traces and small CPU-only tests. A process deadline does not protect the host from a driver or kernel hang.

Before any later GPU trial, follow the [numerical gates](../../verification/hillclimb-numerical-gates.md), establish an appropriate work window and exclusive GPU lease, and run through the project's owned Windows Job Object channel. Stop the ordinary Windows worker only for the duration of that owned trial and restore its qualified on-demand service afterward. Do not change host drivers or run stress, deliberate crash, custom-kernel or compilation experiments during concurrent priority work.

The baseline representative page completed correctly with 397 output tokens. Its separate unprofiled warm request took 85.497 seconds; the instrumented request took 126.375 seconds. The trace covers one prefill and eight cached forwards and reveals approximately 2,740 GPU kernels per decoding token. Instrumentation substantially changes execution time, so it identifies opportunities without establishing an attainable speedup.

| Experiment | Current evidence | Method and tools |
| --- | --- | --- |
| Baseline profile | Complete correct output; many small recurrent and normalization operations | [Method](method.md), [runner](profile_baseline.py) |
| Explicit-mask text attention | Original numerical gate failed; early probability distributions also moved further from CPU FP32. Set aside. | [Finding and method](attention_candidate.md) |
| Compiled recurrent updates | Both variants failed full cached numerical checks: six positions for v1 and eleven with contraction disabled. Set aside. | [Method](recurrent_method.md), [candidate](recurrent_candidate.py) |
| Cached normalization gains | All 395 logit rows, 25,280 cache records and five complete OCR pages match exactly. Two of five planned timing pairs completed; extra peak allocation leaves memory qualification unresolved. Further pairs and promotion are held. | [Method](norm_method.md), [candidate](norm_candidate.py) |
| Production gain ownership | CPU lifetime checks and all 162 retained trained fixtures pass; no XPU peak-memory or speed evidence yet | [Prospective protocol](../../verification/gain-production-protocol.md), [CPU replay](../../verification/replay_norm_gain_cpu.py) |
| Native fused normalization | Complete cached trajectory failed the frozen policy at 17 positions, including an unapproved digit decision. Set aside. | [Method](norm_method.md), [exporter](norm_export.py) |
| Native XPU graph replay | The installed A770 runtime reports graph recording/replay unsupported; no replay comparisons ran | [Probe method](graph_method.md) |

The [compact evidence record](../../benchmarks/evidence/hillclimb-2026-10-04.json) commits to retained private receipts. Raw traces and trained tensors remain outside Git. Compiler diagnostics and partial runs are evidence of those attempts, never successful qualification or speed measurements. The [independent synthetic pages](../../benchmarks/qualification-v1/README.md) broaden the original development template's regression scope.
