# Numerical evidence, October 3, 2026

The retained evidence supports the hybrid Intel profile—BF16 weights, vision SDPA and text eager—for the checked synthetic PNGs. All-SDPA text decoding is excluded. [The machine-readable record](numerical-2026-10-03.json) contains exact source/input/payload hashes, runtime versions, full scalar measurements, mismatched token positions and raw run identifiers. Raw tensors and endpoint responses remain outside Git for cross-backend regression work; this public record contains no private paths or configuration.

The checkpoint is `datalab-to/chandra-ocr-2` revision `af93b47dba1b47b6640c86ccf487ed2260ab9a09`: the retained integrity inventory covers 17 files and 739 BF16 tensors, with byte-identical input embedding/output-head payloads. The matched CPU oracle uses official PyTorch `2.14.1+cpu`, torchvision `0.29.1+cpu` and Transformers `5.18.0`; Intel uses PyTorch `2.14.1+xpu` and the same Transformers graph source hash. The tiny image and production processor profile produce 801 prefill tokens and 832 vision patches. Comparisons cover every vocabulary entry at all 394 retokenized response-body positions, with exact matching processed-input hashes and conditioning tokens; EOS is measured independently through free generation.

| CPU FP32 reference versus Intel BF16 | Greedy agreement | Maximum full-vocabulary KL | Maximum target log-probability difference |
| --- | ---: | ---: | ---: |
| Text/vision eager | 391/394 | 0.481333 | 0.545531 |
| Strict text/vision SDPA | 393/394 | 0.225625 | 0.176540 |
| Hybrid vision SDPA/text eager | 391/394 | 0.145344 | 0.613627 |

The differing greedy choices are bounding-box digits, including a reference top-two margin of approximately 0.00357. These measurements do not establish identical distributions. Hybrid cached/full final logits differ by maximum 0.0625 and RMS 0.00966, with finite tensors and the same greedy token. Strict all-SDPA instead has maximum 5.875 and RMS approximately 1.017, with divergent cached digits. Its good full-forward result therefore cannot validate generation.

Independent CPU FP32 greedy generation stopped correctly after 395 tokens. The hybrid tiny endpoint also stopped after 395 tokens and matched CPU HTML exactly outside bounding-box attributes: all six ordered block labels, prose, equations and associated table cells were identical. Maximum coordinate difference was two normalized units (0.816 pixels horizontally); vertical changes were at most one unit (0.528 pixels), and minimum box IoU was 0.954545. Tiny, small and representative synthetic PNG endpoint gates all passed with stop reasons; their completion counts were 395, 395 and 397. These are three variants of one content/layout template, not a broad independent corpus or a repeated throughput benchmark.

The bounded SDPA probe isolates the failure without loading a model. It uses identical BF16-rounded PCG64 arrays, 16 query heads, four KV heads, dimension 256 and an independent CPU FP32 eager softmax/matmul oracle. Intel execution explicitly excludes the math backend. All GQA/manual-KV-repeat output pairs are bit-identical, ruling out grouped-query repetition as the distinguishing condition in this probe.

| Query/KV length | Mask and causal flag | Intel maximum error | Intel RMS error |
| --- | --- | ---: | ---: |
| 1/801 | None, false | 2.070807 | 0.445546 |
| 1/809 | None, false | 2.060387 | 0.443676 |
| 1/809 | Explicit all-true boolean mask, false | 0.000682 | 0.000138 |
| 1/809 | None, true; deliberate upper-left alignment control | 0 | 0 |
| 801/801 | None, true | 0.009232 | 0.000273 |
| 801/809 | Explicit causal-offset boolean mask, false | 0.008308 | 0.000277 |

The frozen probe inputs have SHA-256 `b1a19550b9a7aebbce84b9e22a04fcf9f7036f10d3c54aaeb485bbc32ca8856f`; its source has SHA-256 `63ddcbbbb78f9981bedcc602aa58629e98f961f055cf7726dfc5cbadb24c7480`. Every retained probe payload hash was rechecked during curation. The CPU math implementation produces bit-identical GQA/manual-repeat results in all six cases. The evidence localizes the affected maskless single-query dispatch condition; it does not identify the internal fused kernel cause. Host FP32 oracle arithmetic differs slightly despite identical input bytes.

An explicit-mask workaround merits a later, separate opt-in candidate because the equivalent masked call restores small BF16-scale error. Keep serving on hybrid while that candidate is assessed. Promotion requires trained failing-layer Q/K/V fixtures across cache lengths, correct boolean/float-mask semantics, whole-model cached/full and complete teacher-forced comparisons, long greedy termination, complete OCR/geometry and state-isolation/cancellation checks, and measured allocation/performance costs. This record makes no serving change, deployment claim or full native-graph claim; excluded wrong-MRoPE and wrong-stop experiments remain failed runs. The older PyTorch 2.13 CPU FP32 cached/full measurement is identified separately from the matched-version full teacher and successful greedy references.
