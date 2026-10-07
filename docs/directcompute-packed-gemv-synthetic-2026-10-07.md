# Packed GEMV numerical qualification

The experimental `ordered_packed` DirectCompute kernel passed one model-free Intel Arc A770 calibration on October 7, 2026. It stages packed BF16 words in shared memory while retaining the original eight-output geometry, barriers, ascending-K FP32 arithmetic, bias and output rounding. The experimental source remains outside the serving branch pending its applicable full-model qualification.

The strict Windows C++ build and both FXC shader checks passed before execution. The GPU completed eight native numerical phases covering 11,754 output words, rejected nine malformed admissions without dispatch, and passed 21 boundary cases comprising 42 dispatches and 378 output words. Finite outputs match their expected storage bits. NaNs follow the declared class rule; rounded NaNs also require the quiet bit and a zero low half. The cases cover ragged and odd widths, shards, BF16 and FP32 weights, bias, order-sensitive accumulation, subnormals, underflow, signed zero, infinities, NaNs and nearest-even rounding boundaries.

| Native phase | K | Outputs | Largest dispatch, ms |
| --- | ---: | ---: | ---: |
| Order-sensitive BF16 | 2,560 | 1,024 | 0.266406 |
| Dyadic BF16 | 9,216 | 1,024 | 0.902292 |
| Order-sensitive FP32 | 9,216 | 1,024 | 0.916354 |

The 0.916354 ms maximum covers the eight native calibration phases. Boundary dispatches independently pass their strict 100 ms limit. These are exploratory kernel timings, with no paired original-kernel comparison. They establish no speedup or sustained page-throughput result.

The declared largest phase plan bounds tracked tensors plus readback at 18,919,424 bytes; it is not a measured resident-RAM or VRAM peak. Observed Windows process and aggregate Job commit peaked at 168,759,296 and 182,538,240 bytes under fixed 256 MiB and 512 MiB caps. Tracked tensors returned to zero after every phase and final release. The Job, exact owner, transport, local process group and root units closed; the historical worker was restored and confirmed cold. Source, tools, compiled artifacts and retained results authenticated after completion.

The [summary receipt](../benchmarks/evidence/directcompute-packed-gemv-synthetic-2026-10-07.json) pins source and patch hashes and the independently reviewed acceptance. The original offline qualifier's protocol-name mistake and failed result remain preserved; correction qualified the original captures without GPU replay. Full-model imported-weight integrity remains held by the [second-shard row discrepancy](directcompute-interrupted-watch-arithmetic-2026-10-07.md). Trained numerics, complete OCR, the native endpoint and sustained correct pages per second remain unaccepted. No candidate was deployed, and NorthStone's AMD throughput work remains stopped.
