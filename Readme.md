# Native Chandra OCR for Windows

This public fork of [Const-me/Whisper](https://github.com/Const-me/Whisper) preserves its history and MPL-2.0 source while developing Chandra OCR 2 inference for Windows Intel GPUs. The objective is correct, complete OCR at the best sustained concurrency, measured against an empirical hardware roofline. NorthStone's AMD Radeon RX 7900 XTX endpoint is the primary end-to-end reference; independent CPU higher precision calculations are the operator oracle. The original project documentation remains in [WHISPER-README.md](WHISPER-README.md).

As of October 15, the selected Vulkan configuration on the original four-page synthetic corpus delivers **0.0511 correct pages/s, about 184 pages/hour, at four concurrent requests**. Larger batches and greater concurrency raised this rate by 86.2% over the original Vulkan C2 result. Eight concurrent requests were slower at 0.0441 pages/s. All retained C4 and C8 responses passed the original content checks and an independently authored source text/layout contract. A broader check passed routes and ledgers but found identifier substitutions in both dense pages. A targeted unquantized FP32 comparison makes the same first wrong choice under common preceding text, narrowing its interpretation without establishing whole-page equivalence. The blank case exposed an overly restrictive verifier; classification and downstream artifacts remain separate from truthful blank recognition. The candidate uses Q8_0 text weights with the unchanged BF16 projector and cache, a full 16,384-token context per request and the normal 12,384-token output allowance. These finite throughput rates include the final submitted requests draining, but exclude image preparation, loading and warmup. Broader numerical acceptance, the throughput ceiling and full CLI compatibility remain unfinished. The [engine trials](docs/sycl-engine-trial.md) retain exact measurements, pins and limitations.

With original images submitted through the complete endpoint adapter, C4-b128 completed 20 correct measured pages in 402.8078 seconds: **0.04965 pages/s, about 179 pages/hour**, including image preparation, queueing and inference. All four warmups and twenty measured responses passed independent source-content and localization checks. This finite run excludes PDF rendering, loading and warmup. A separate prepared-image batch-256 trial reached 0.05118 pages/s, only 0.24% above batch 128, with the same sixteen completions inside the measurement window. That single-run difference does not establish an improvement; batch 128 remains the selected candidate.

The [CPU preparation package](runtime/sycl/README.md) reproduces all four reference prepared images and requests exactly, removing the initial padding mismatch. Its public synthetic regressions pass all 14 tests, including a fresh Windows run. The [endpoint adapter](runtime/llama/README.md) supports one, two or four slots through an explicit opt-in CLI; its combined offline regression passed 82 tests plus 44 subtests. Finite Windows tests now cover shared startup, four overlapping requests, idle unload, cancellation with explicit peer failure, recovery, forced worker failure and controller loss during generation, with confirmed process retirement. The ordinary Waystone route is now installed: its cold-start page and four overlapping pages passed source checks, a fifth request received an explicit capacity refusal, and its normal 300-second idle policy unloaded despite passive health/model queries. Exact unload latency and full CLI compatibility remain unmeasured. A cooperative-matrix kernel experiment was substantially slower and was rejected. Stock operator measurements also found the F16 vision FFN slower than BF16; the separately audited F16 projector has not undergone model inference. DirectCompute/HLSL and the unsuccessful SYCL starts remain retained references. The [work record](docs/work-record.md) provides the route to current allocations; inspect actual process state before taking over.

## Retained DirectCompute results

The following October 11 findings retain their original candidate and observation scope. Their outstanding gates apply to those recorded builds; current exploratory trials and promotion criteria follow [the design](docs/design.md).

Completed October 11 reviews qualify the experimental pipeline worker’s native-0 HTTP singleton and two complete native-0/native-1 pairs on the unchanged 1,624 × 2,100 pixels, 3,617-token prompt, 16,384-token context and 12,384-token allowance. The pages reach natural EOS after 556/560 IDs, preserve substantive text, tables, equations, reading order and non-geometric markup against the retained AMD responses, and cover native-1 surviving in both slots across distinct episodes. The unchanged stopped pair-proof reader accepts the actual two-episode bundle. Five/eight geometry differences remain unaccepted; the native-1 standalone current2415 journal is unavailable. The [October 11 progress receipt](benchmarks/evidence/directcompute-progress-2026-10-11.json) binds the completed source, build, capture and review identities.

The public source now includes the reviewed own-row mix-input implementation and compatible endpoint modules. Its current3036 build preserves the saved trained payloads and state joins against the pipeline reference, and later full-page checks accepted all pages in the first one-request and two-request windows. Three and four measured pages respectively finished within 20 minutes; later completions are excluded. These are early concurrency observations, with repeats still needed, rather than an accepted optimization gain or throughput ceiling. The [same-build window receipt](benchmarks/evidence/directcompute-current3036-windows-2026-10-11.json) separates page correctness, cutoff counts, cohort cadence and the remaining numerical limits.

These results establish their finite page or numerical scope. Sustained correct pages per second, a throughput improvement, broader endpoint/client lifecycle qualification, layout truth and permanent deployment remain unaccepted. The [reviewed production subset](docs/directcompute-source-integration.md) is now present in this fork; historical build identities remain separate from a fresh public-tree build and deployment. The [current work record](docs/work-record.md) preserves earlier accepted and failed outcomes and gives the remaining gates.

## Find the work

| Work | Start here |
| --- | --- |
| Resume the current engine trial | [Engine trials](docs/sycl-engine-trial.md), [design](docs/design.md), [work record](docs/work-record.md) |
| Understand the retained DirectCompute implementation | [DirectCompute runtime](docs/directcompute-runtime.md) |
| Review historical CLI, HTTP and trained-capture evidence | [October 11 progress](benchmarks/evidence/directcompute-progress-2026-10-11.json), [Current144 aggregate](benchmarks/evidence/directcompute-current144-2026-10-10.json), [HTTP observations](benchmarks/evidence/directcompute-native-http-2026-10-10.json) |
| Inspect native graph and shaders | [Runtime API](ChandraNative/runtime/api.h), [text graph](ChandraNative/runtime/text_model.cpp), [HLSL](ChandraNative/shaders/runtime) |
| Authenticate model weights | [Model manifest](provenance/model.json), [native importer](ChandraNative/runtime/model_weights.cpp), [inventory verifier](scripts/model_inventory.py) |
| Review operator oracles and prior primitive results | [CPU oracle](ChandraNative/oracle.h), [primitive guide and evidence](ChandraNative/README.md) |
| Freeze inputs and qualify complete OCR | [Benchmark workflow](benchmarks/README.md), [numerical workflow](verification/README.md) |
| Inspect the fresh exact-input numerical reference | [October 7 reference and preparation boundary](docs/directcompute-reference-2026-10-07.md) |
| Resume after the interrupted watched-model run | [October 7 interruption and physical closure](docs/directcompute-interrupted-watch-arithmetic-2026-10-07.md) |
| Inspect opt-in native HTTP source | [Native endpoint guide](docs/directcompute-native-endpoint.md) |
| Inspect historical endpoint behavior | [XPU runtime guide](runtime/waystone/README.md), [service package](chandra_service), [operations](docs/operations.md) |
| Recover earlier work and failures | [Retained work history](docs/work-record-history-2026-10-05.md), [evidence index](benchmarks/evidence/hillclimb-2026-10-04.json), [agent guidance](AGENTS.md) |

Historical milestones and failures—including four current144 CLI fixture captures, narrow B2 differentials, selected SRV-only import tails, the Build20 first-page recovery and rejected second page—remain in the [current144 aggregate](benchmarks/evidence/directcompute-current144-2026-10-10.json), [work record](docs/work-record.md), [retained history](docs/work-record-history-2026-10-05.md) and [evidence index](benchmarks/evidence/README.md). Those records preserve their original numerical, geometry, custody and failure boundaries: selected tails and producer state hashes do not establish whole-model accuracy, and earlier primitive and XPU results retain their historical scope. The October 15 backend decision supersedes the earlier exclusion of XPU and OpenVINO performance work; original failed outcomes remain unchanged.

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

These commands check the retained service and DirectCompute tooling; they do not qualify the SYCL engine. [ChandraNative/runtime/build.cmd](ChandraNative/runtime/build.cmd) defines the separate Windows C++ library and offline shader build, while the older [primitive guide](ChandraNative/README.md) documents its standalone operator runner and CPU oracle. Engine experiments use separately owned, resource-bounded execution; build products, models and raw experiments belong in ignored project-local locations.

## Endpoint compatibility

The retained service contract uses model alias `chandra`, `/v1/models`, `/v1/chat/completions` and health/model listings that do not load the model. The separate router accepts `X-Chandra-Backend: northstone`, `waystone` or `auto`, and backend-specific base URLs. It submits a request once, holds capacity until actual completion or cancellation drain, and reports failure without replay. These existing routes provide the compatibility target for a future qualified native worker.

The existing Chandra CLI invocation is:

```sh
MAX_VLLM_RETRIES=0 VLLM_API_BASE=http://127.0.0.1:8002/v1 VLLM_MODEL_NAME=chandra uv run --no-sync chandra input.pdf output --method vllm --batch-size 2 --no-html
```

This retained command addresses the historical routed endpoint. The DirectCompute CLI consumes authenticated prepared tensors; the page-specific HTTP results above do not establish general client/lifecycle conformance. `MAX_VLLM_RETRIES=0` disables Chandra's application retries; its OpenAI SDK can still retry transport failures, so a controlled comparison must also preserve the single-submission obligation. Bounded complete-page trials and provisional timings may guide engineering before qualification is complete. Promote a throughput gain only with numerical and complete OCR evidence, cancellation and concurrency checks, and repeated sustained measurements under unchanged conditioning. No SYCL throughput or page ceiling is claimed here.

The model is published separately by Datalab under its supplied OpenRAIL license. This repository does not redistribute its weights. [Upstream provenance](provenance/upstream.json) records the fork parent, retained source history and pinned model revision.
