# Numerical gates for performance experiments

An optimization must preserve accurate complete OCR before its speed matters. Freeze the checkpoint, installed model graph, runtime, processed pixels, prompt and conditioning tokens for every comparison. Record the exact candidate source and prove that its intended operators executed. A trained operator check, teacher-forced export and free generation establish different properties; none substitutes for the others. The [qualified reference evidence](evidence/README.md) and [experiment index](../experiments/xpu-hillclimb/README.md) identify the retained results.

## Compare against the reference

The matched PyTorch 2.14.1 CPU FP32 reference now covers all 394 conditioned response tokens plus the terminal distribution. Every comparable cached argmax agrees with the existing full-sequence reference; the maximum logit discrepancy is 6.15×10⁻⁵ and maximum KL divergence is 2.57×10⁻¹¹. These measurements calibrate the FP32 paths on one frozen page. They do not define a universal tolerance for BF16 arithmetic or prove accuracy on other documents.

For future full-trajectory candidate trials, collect the unchanged Windows hybrid baseline first and freeze its per-position errors against that CPU reference. The coordinator approved the following engineering budgets before those candidate trials: candidate total-variation error against CPU may exceed the corresponding hybrid error by at most 0.01, and absolute target log-probability error may exceed the corresponding hybrid error by at most 0.02. Total variation bounds the error in any token event's probability; these allowances permit at most one additional percentage point and approximately a 2% multiplicative target-probability discrepancy, respectively. They are explicit policy choices, not mathematical guarantees about BF16 or OCR quality. Preserve every failing row; do not increase allowances after observing a candidate.

Keep raw and centered logit errors, directional KL divergence, top-two margins and token disagreements as diagnostics. Meaningful text, equation, table, structure and stop-token changes fail qualification. Previously reviewed bounding-box digit differences require an explicit exception tied to the same token role and acceptable observed geometry; they do not authorize arbitrary new differences. Compare several cache lengths and the complete response, because a satisfactory early prefix can conceal later drift.

## Check the changed operation

Use actual trained inputs and the exact pinned source for operator comparisons. Distinguish execution error on BF16-rounded inputs from the additional error introduced by a complete BF16 model. Check shape, dtype, finiteness, input mutation and interception coverage. A recurrent implementation must retain FP32 state and be checked across all 24 affected layers, including later positions; BF16 output agreement cannot excuse incorrect persistent state. Compare CPU FP32 and unchanged XPU eager results before interpreting the candidate's errors. Report per-layer absolute and normalized errors with the reference scale. The earlier native primitive's 2×10⁻⁵ state threshold applied to its first-layer testcase and does not automatically transfer to other layers.

Gain-only normalization caching promises unchanged arithmetic, so it requires bit-identical trained outputs, full exported logits and cache tensors against the fresh hybrid baseline. Native fused normalization and compiled recurrence can change reduction or rounding order; they require the broader reference comparisons above. Capability receipts may collect metrics without passing qualification. Instrumented fixture capture, state hashing and compiler warmup are excluded from performance measurements.

## Require complete output and a usable worker

Run complete greedy OCR under the ordinary 12,384-token output allowance and exact checkpoint termination rules. Check ordered prose, equations, row/column table associations, labels, page order, terminal markers, actual stop reasons and bounding boxes. Retain the original PNG/PDF regression corpus and the independently composed [qualification pages](../benchmarks/qualification-v1/README.md). These are synthetic regression sets; passing them does not establish general document accuracy. Repeated pages, cancellation, worker recovery and idle unload must also preserve state isolation and release GPU allocations before deployment.

Only qualified candidates proceed to paired, uninstrumented warm timing. Keep loading, compilation and cold requests separate. Require equivalent complete outputs and actual interception counts, and retain failures alongside successful runs. Promote a measurable improvement only after the deployed path passes the same boundaries.

## Preserve the rejected attention trial

The explicit-mask text SDPA candidate failed its original eight-step cached/full gate: maximum error 0.125 and RMS 0.012622 exceeded the predeclared hybrid envelope of 0.0625 and 0.009662. That result remains a failure. Matched CPU BF16 SDPA showed a similar cached/full floor, demonstrating why one observed hybrid maximum is not a universal numerical bound. It did not validate the candidate. On the same nine conditioned rows, candidate error against CPU FP32 was also worse: maximum KL 0.021515 versus hybrid 0.002377, and mean KL 0.003735 versus 0.000422, despite identical argmax choices. The candidate is set aside.

The earlier maskless SDPA implementation had apparently satisfactory full-sequence teacher measurements but divergent cached decoding and hallucinated complete OCR. Together these cases explain the protocol: neither a scalar rounding explanation, a matching early argmax nor a good teacher-forced result establishes a correct inference path.
