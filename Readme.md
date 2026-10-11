# Native Chandra OCR for Windows

This public fork of [Const-me/Whisper](https://github.com/Const-me/Whisper) preserves its history and MPL-2.0 source while developing Chandra OCR 2 in C++ and HLSL through DirectCompute/D3D11. The target is a correctness-qualified native runtime with measured sustained page throughput. NorthStone's AMD Radeon RX 7900 XTX endpoint is the primary end-to-end reference; independent CPU higher precision calculations are the operator oracle. The original project documentation remains in [WHISPER-README.md](WHISPER-README.md).

Completed October 11 reviews qualify the experimental pipeline worker’s native-0 HTTP singleton and two complete native-0/native-1 pairs on the unchanged 1,624 × 2,100 pixels, 3,617-token prompt, 16,384-token context and 12,384-token allowance. The pages reach natural EOS after 556/560 IDs, preserve substantive text, tables, equations, reading order and non-geometric markup against the retained AMD responses, and cover native-1 surviving in both slots across distinct episodes. The unchanged stopped pair-proof reader accepts the actual two-episode bundle. Five/eight geometry differences remain unaccepted; the native-1 standalone current2415 journal is unavailable. The [October 11 progress receipt](benchmarks/evidence/directcompute-progress-2026-10-11.json) binds the completed source, build, capture and review identities.

The public source now includes the reviewed own-row mix-input implementation and compatible endpoint modules. Its current3036 build preserves the saved trained payloads and state joins against the pipeline reference, and later full-page checks accepted all pages in the first one-request and two-request windows. Three and four measured pages respectively finished within 20 minutes; later completions are excluded. These are early concurrency observations, with repeats still needed, rather than an accepted optimization gain or throughput ceiling. The [same-build window receipt](benchmarks/evidence/directcompute-current3036-windows-2026-10-11.json) separates page correctness, cutoff counts, cohort cadence and the remaining numerical limits.

These results establish their finite page or numerical scope. Sustained correct pages per second, a throughput improvement, broader endpoint/client lifecycle qualification, layout truth and permanent deployment remain unaccepted. The [reviewed production subset](docs/directcompute-source-integration.md) is now present in this fork; historical build identities remain separate from a fresh public-tree build and deployment. The [current work record](docs/work-record.md) preserves earlier accepted and failed outcomes and gives the remaining gates.

| Work | Start here |
| --- | --- |
| Understand the native target and references | [Design](docs/design.md), [DirectCompute runtime](docs/directcompute-runtime.md) |
| Review current CLI, HTTP and trained-capture evidence | [October 11 progress](benchmarks/evidence/directcompute-progress-2026-10-11.json), [Current144 aggregate](benchmarks/evidence/directcompute-current144-2026-10-10.json), [HTTP observations](benchmarks/evidence/directcompute-native-http-2026-10-10.json), [work record](docs/work-record.md) |
| Inspect native graph and shaders | [Runtime API](ChandraNative/runtime/api.h), [text graph](ChandraNative/runtime/text_model.cpp), [HLSL](ChandraNative/shaders/runtime) |
| Authenticate model weights | [Model manifest](provenance/model.json), [native importer](ChandraNative/runtime/model_weights.cpp), [inventory verifier](scripts/model_inventory.py) |
| Review operator oracles and prior primitive results | [CPU oracle](ChandraNative/oracle.h), [primitive guide and evidence](ChandraNative/README.md) |
| Freeze inputs and qualify complete OCR | [Benchmark workflow](benchmarks/README.md), [numerical workflow](verification/README.md) |
| Inspect the fresh exact-input numerical reference | [October 7 reference and preparation boundary](docs/directcompute-reference-2026-10-07.md) |
| Resume after the interrupted watched-model run | [October 7 interruption and physical closure](docs/directcompute-interrupted-watch-arithmetic-2026-10-07.md) |
| Inspect opt-in native HTTP source | [Native endpoint guide](docs/directcompute-native-endpoint.md) |
| Inspect historical endpoint behavior | [XPU runtime guide](runtime/waystone/README.md), [service package](chandra_service), [operations](docs/operations.md) |
| Recover earlier work and failures | [Retained work history](docs/work-record-history-2026-10-05.md), [evidence index](benchmarks/evidence/hillclimb-2026-10-04.json), [agent guidance](AGENTS.md) |

Historical milestones and failures—including four current144 CLI fixture captures, narrow B2 differentials, selected SRV-only import tails, the Build20 first-page recovery and rejected second page—remain in the [current144 aggregate](benchmarks/evidence/directcompute-current144-2026-10-10.json), [work record](docs/work-record.md), [retained history](docs/work-record-history-2026-10-05.md) and [evidence index](benchmarks/evidence/README.md). Those records preserve their original numerical, geometry, custody and failure boundaries: selected tails and producer state hashes do not establish whole-model accuracy, and earlier primitive and XPU results retain their historical scope. XPU and OpenVINO performance work remain outside the current native target; original failed outcomes remain unchanged.

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

This retained command addresses the historical routed endpoint. The native CLI consumes authenticated prepared tensors; the page-specific HTTP results above do not establish general client/lifecycle conformance. `MAX_VLLM_RETRIES=0` disables Chandra's application retries; its OpenAI SDK can still retry transport failures, so a controlled native comparison must also preserve the single-submission obligation. After numerical and complete OCR qualification, measure repeated unprofiled sustained page throughput against the AMD reference at safe concurrency, with unchanged inputs and complete resource/closure evidence. No native throughput or page ceiling is claimed here.

The model is published separately by Datalab under its supplied OpenRAIL license. This repository does not redistribute its weights. [Upstream provenance](provenance/upstream.json) records the fork parent, retained source history and pinned model revision.
