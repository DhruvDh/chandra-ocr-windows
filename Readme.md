# Native Chandra OCR for Windows

This public fork of [Const-me/Whisper](https://github.com/Const-me/Whisper) preserves its history and MPL-2.0 source while developing Chandra OCR 2 in C++ and HLSL through DirectCompute/D3D11. The target is a correctness-qualified native runtime with measured sustained page throughput. NorthStone's AMD Radeon RX 7900 XTX endpoint is the primary end-to-end reference; independent CPU higher precision calculations are the operator oracle. The original project documentation remains in [WHISPER-README.md](WHISPER-README.md).

October 10 UTC: the experimental current144 DirectCompute CLI has completed four public synthetic fixture pages, native-0, native-1, scanned-0 and scanned-1, at the ordinary Zotero workload. All reach natural EOS 248046; complete prose, both columns, reading order, equations and every table cell match the retained AMD substantive markup exactly after removing only `data-bbox`. Geometry differs in 5/8/1/9 blocks, by at most 1/2/1/8 normalized units respectively, with no accepted geometry tolerance. The [redacted current144 record](benchmarks/evidence/directcompute-current144-2026-10-10.json) preserves those scoped acceptances and original control failures.

The same frozen source also passed a narrow trained-B2 differential check: three retained paired/singleton final-normalization, complete-vocabulary logit and selected-state comparisons are bit exact, as are four predecessor norm/logit rows at two causal positions. Complete carried-state and inactive-neighbor equality remain native-producer hash/metadata observations where full buffers were not persisted. CPU FP32 comparisons are descriptive; no independent higher-precision whole-graph oracle or numerical tolerance has been accepted. This diagnostic does not qualify complete B2 OCR pages.

Native HTTP/client acceptance, arbitrary-document OCR and sustained correct pages per second remain open. The installed CPU environment is accepted, but the same-context resident preflight failed before HTTP with `STATUS_QUOTA_EXCEEDED` at about 313 MiB process/Job commit, far below the unchanged memory limits; the specific quota cause remains unresolved. Failure and subsequent affirmative retirement are separate observations. There is no accepted native endpoint or deployment. The [current work record](docs/work-record.md) gives the next correctness and lifecycle gates.

At the October 8 cutoff, the [weight-import record](docs/directcompute-weight-import-2026-10-08.md) now includes a useful SRV-only follow-up: one DEFAULT-initial import completed all 732 original uploads/drains and matched all ten selected output-head tails after constructor return. The earlier two-API tail failures remain preserved. The [redacted evidence](benchmarks/evidence/directcompute-weight-import-2026-10-08.json) accepts selected-row observations and cleanup only; whole-model integrity, a production fix, trained numerical correctness, complete OCR and native page throughput remain unqualified. The matching arithmetic bridge has compiled and passed independent artifact review. Its matching runtime and fresh CPU forecast are now reviewed; the [next original cap2 GPU attempt was interrupted before a complete arithmetic result](docs/directcompute-cap2-status-2026-10-08.md), then retired and restored cold. Same-live numerical capture remains required. The experimental source remains unpromoted; earlier first-page and second-page results below retain their original scope.

The earlier Build20 second-page attempt failed from its first numerical prediction, alternating an enormous token-52427 logit with an all-zero vocabulary. It reached no EOS and passed none of the 18 content checks. Its bounded request closed without replay; the [failure record](benchmarks/evidence/directcompute-native-second-page-failure-2026-10-06.json) retains the evidence. Correctness diagnosis precedes further native performance or endpoint promotion.

The [native runtime](ChandraNative/runtime) connects the authenticated original model, complete vision graph, ordered image-language merge, text prefill and cached greedy generation. The earlier full-size Zotero page completed in the frozen Build20 CLI and ordinary resident worker, each producing 556 IDs through caller EOS with complete prose, columns, equations and table values. Their output and scalar journals match exactly; content and non-geometric markup also match the retained primary AMD response. Five bounding boxes differ by at most one normalized unit, with no accepted tolerance. The [recovery guide](docs/directcompute-exact-client-recovery.md) records cold wrapper intervals of 797.484 and 795.859 seconds and the separate independent reviews. The [shared-memory repair and model-free regression](docs/directcompute-exact-input-correctness.md) pass finished source review, strict Windows compilation and all 17 A770 cases; no native runtime service is deployed. Fresh trained numerics, broader OCR, native HTTP/client serving and sustained throughput remain open. The [earlier tiny-page measurements](benchmarks/evidence/directcompute-2026-10-06.json), [runtime guide](docs/directcompute-runtime.md) and [initial checkpoint](benchmarks/evidence/directcompute-2026-10-05.json) retain their original scope and failed attempts.

The PyTorch XPU endpoint on Intel Arc A770 has retained functional OCR and client/lifecycle evidence. It remains a historical reference with **0% qualified inference-engine speedup**, rather than success of the native implementation. Earlier DirectCompute BF16, normalization and recurrence tests establish only their recorded primitive scope. XPU and OpenVINO performance work are outside the current target. The retained [work history](docs/work-record-history-2026-10-05.md) and [measured evidence](benchmarks/evidence/README.md) preserve those results and failed experiments.

| Work | Start here |
| --- | --- |
| Understand the native target and references | [Design](docs/design.md), [DirectCompute runtime](docs/directcompute-runtime.md) |
| Review current CLI and B2 evidence | [Current144 aggregate](benchmarks/evidence/directcompute-current144-2026-10-10.json), [work record](docs/work-record.md) |
| Inspect native graph and shaders | [Runtime API](ChandraNative/runtime/api.h), [text graph](ChandraNative/runtime/text_model.cpp), [HLSL](ChandraNative/shaders/runtime) |
| Authenticate model weights | [Model manifest](provenance/model.json), [native importer](ChandraNative/runtime/model_weights.cpp), [inventory verifier](scripts/model_inventory.py) |
| Review operator oracles and prior primitive results | [CPU oracle](ChandraNative/oracle.h), [primitive guide and evidence](ChandraNative/README.md) |
| Freeze inputs and qualify complete OCR | [Benchmark workflow](benchmarks/README.md), [numerical workflow](verification/README.md) |
| Inspect the fresh exact-input numerical reference | [October 7 reference and preparation boundary](docs/directcompute-reference-2026-10-07.md) |
| Resume after the interrupted watched-model run | [October 7 interruption and physical closure](docs/directcompute-interrupted-watch-arithmetic-2026-10-07.md) |
| Inspect opt-in native HTTP source | [Native endpoint guide](docs/directcompute-native-endpoint.md) |
| Inspect historical endpoint behavior | [XPU runtime guide](runtime/waystone/README.md), [service package](chandra_service), [operations](docs/operations.md) |
| Recover earlier work and failures | [Retained work history](docs/work-record-history-2026-10-05.md), [evidence index](benchmarks/evidence/hillclimb-2026-10-04.json), [agent guidance](AGENTS.md) |

## Preserve the workload

Use `datalab-to/chandra-ocr-2` revision `af93b47dba1b47b6640c86ccf487ed2260ab9a09`, the exact processor/tokenizer/prompt, original page pixels, a 16,384-token context and the normal 12,384-token output allowance. The recorded Zotero CLI pages use 1,624 × 2,100 decoded pixels and 3,617 prompt tokens; older frozen renders use 1,632 × 2,112 and 3,959. These are distinct inputs. No shortened output, image resize, lower precision or alternate checkpoint can be credited as an unchanged-model speedup. Preserve complete table associations, equations, layout, metadata, raw responses and genuine EOS alongside numerical outputs and carried state.

Model weights stay outside Git. The downloader checks pinned file sizes and hashes, refuses existing mismatches and keeps partial files out of accepted names. The inventory verifier checks tensor shapes/ranges and the embedding/output-head bytes. The native importer performs its own authenticated, bounded import; it is not authorized by a model name alone. The pinned model and upstream source are recorded under [provenance](provenance).

## Source and offline checks

The service/client test environment is separate from the inference environments. Its existing offline checks are:

```sh
uv sync --locked --group test --group benchmark
uv run --no-sync pytest
python scripts/model_inventory.py /path/to/pinned/model --output .runtime/model-inventory.json
```

These commands do not qualify the new native graph. [ChandraNative/runtime/build.cmd](ChandraNative/runtime/build.cmd) defines the separate Windows C++ library and offline shader build, while the older [primitive guide](ChandraNative/README.md) documents its standalone operator runner and CPU oracle. Native execution and full-model import require separately owned, resource-bounded qualification; build products, models and raw experiments belong in ignored project-local locations.

## Endpoint compatibility

The retained service contract uses model alias `chandra`, `/v1/models`, `/v1/chat/completions` and health/model listings that do not load the model. The separate router accepts `X-Chandra-Backend: northstone`, `waystone` or `auto`, and backend-specific base URLs. It submits a request once, holds capacity until actual completion or cancellation drain, and reports failure without replay. These existing routes provide the compatibility target for a future qualified native worker.

The existing Chandra CLI invocation is:

```sh
MAX_VLLM_RETRIES=0 VLLM_API_BASE=http://127.0.0.1:8002/v1 VLLM_MODEL_NAME=chandra uv run --no-sync chandra input.pdf output --method vllm --batch-size 2 --no-html
```

This retained command addresses the historical routed endpoint. The native CLI consumes authenticated prepared tensors, and native HTTP conformance remains unaccepted. `MAX_VLLM_RETRIES=0` disables Chandra's application retries; its OpenAI SDK can still retry transport failures, so a controlled native comparison must also preserve the single-submission obligation. After numerical and complete OCR qualification, measure repeated unprofiled sustained page throughput against the AMD reference at safe concurrency, with unchanged inputs and complete resource/closure evidence. No native throughput or page ceiling is claimed here.

The model is published separately by Datalab under its supplied OpenRAIL license. This repository does not redistribute its weights. [Upstream provenance](provenance/upstream.json) records the fork parent, retained source history and pinned model revision.
