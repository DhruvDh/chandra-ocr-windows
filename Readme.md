# Native Chandra OCR for Windows

This public fork of [Const-me/Whisper](https://github.com/Const-me/Whisper) preserves its history and MPL-2.0 source while developing Chandra OCR 2 inference for Windows Intel GPUs. The objective is correct, complete OCR at the best sustained concurrency, measured against an empirical hardware roofline. NorthStone's AMD Radeon RX 7900 XTX endpoint is the primary end-to-end reference; independent CPU higher precision calculations are the operator oracle. The original project documentation remains in [WHISPER-README.md](WHISPER-README.md).

The installed Q8_0-KV A770 profile delivers **0.0929139 correct original-endpoint pages/s—about 334 pages/hour—at six concurrent requests** on the retained four-case synthetic corpus in its [normal-service sustained repeat](benchmarks/evidence/sycl-q8-normal-service-2026-10-16.json). It completed 72 measured pages in 774.9111 seconds including drain, with 66 inside the 720-second window; all 76 warmup and measured responses passed independent source-content and localization checks. This is 0.36% above the earlier private Q8 C6 observation, about 25.1% above the prior BF16 C4 repeat and 87.1% above the single Vulkan observation. These finite comparisons include original-image preparation, queueing and inference, exclude PDF rendering, cold loading, warmup and offline assessment, and establish no paired confidence interval or hardware ceiling.

This installed profile uses the pinned llama.cpp SYCL backend with [fused vision attention](runtime/llama/patches/vision-sdpa-d812350/README.md), Q8_0 text and KV, the unchanged BF16/F32 projector, batch/microbatch 128, and four logical CPUs available to the owned processes while inference thread counts remain one. Each request retains a full 16,384-token context and 12,384-token output allowance. Exact source/configuration readback, cold six-request completion, seventh-request refusal, cancellation with explicit failure of unfinished peers and fresh recovery passed. The normal 300-second idle policy was followed by an asleep/unloaded observation 358.184 seconds after the last page; exact retirement latency was not measured. Both native and scanned two-page CLI checks reproduce the prior BF16 and Vulkan Markdown and metadata hashes. The [deployment guide](docs/llama-deployment.md) explains authenticated selection, admission and rollback; the service still reports `qualified:false`.

The preceding [Q8 experiments](benchmarks/evidence/sycl-q8-kv-2026-10-16.json) reached 0.0813662 pages/s at C4 and 0.0925798 at C6, with all 64 and 76 source checks passing and clean closure. The earlier [BF16 C4 observations](benchmarks/evidence/sycl-vision-affinity-repeat-2026-10-16.json) reached 0.0742634 and 0.0742734 pages/s, with all 120 checks passing. Ten selected decoder decisions and a complete-output attention screen provide scoped numerical evidence for Q8; broad equivalence remains unestablished. Q8_0 is block INT8, not FP8. C6 is explicitly selected with conservative admission forecasts; omitting the KV option continues to select BF16, and no higher concurrency is authorized by these results.

An [independent CPU FP32 reference](benchmarks/evidence/sycl-vision-cpu-reference-2026-10-16.json) places the faster vision path substantially closer to FP32 on native-0: final-conditioning relative L2 error is **0.853%, versus 12.282% for the original GPU path**. This resolves the specific [12.6% OFF/ON discrepancy](benchmarks/evidence/sycl-vision-conditioning-2026-10-16.json); it covers one complete vision tensor, with residual token-level error and slightly different normalization arithmetic. On a separate dense page, Q8 SYCL, BF16 SYCL, CPU FP32 and Vulkan produce identical non-geometric content including two wrong identifier characters. Both SYCL arms have the same eight strict whitespace-corridor failures of approximately 0.1–1.9 original pixels while improving core containment and preserving all word centers. The [engine guide](docs/sycl-engine-trial.md#numerical-and-ocr-scope) explains the limited text-consumer relevance and the matched singleton's scope. These preserved failures provide no evidence of a Q8-only regression on that page and no general OCR acceptance.

## Find the work

| Work | Start here |
| --- | --- |
| Understand the installed profile, measurements and limitations | [Engine trials](docs/sycl-engine-trial.md), [work record](docs/work-record.md), [design](docs/design.md) |
| Configure the on-demand endpoint and exact image preparation | [Endpoint adapter](runtime/llama/README.md), [CPU preparation](runtime/sycl/README.md), [operations](docs/operations.md) |
| Reproduce the fused-attention source experiment | [Patch, build provenance and opt-in flags](runtime/llama/patches/vision-sdpa-d812350/README.md) |
| Authenticate weights and qualify complete OCR | [Model manifest](provenance/model.json), [inventory verifier](scripts/model_inventory.py), [benchmark workflow](benchmarks/README.md), [numerical workflow](verification/README.md) |
| Inspect retained DirectCompute code and evidence | [Runtime](docs/directcompute-runtime.md), [source integration](docs/directcompute-source-integration.md), [native HTTP](docs/directcompute-native-endpoint.md), [operator guide](ChandraNative/README.md) |
| Recover earlier accepted and failed results | [Work record](docs/work-record.md), [retained history](docs/work-record-history-2026-10-05.md), [evidence index](benchmarks/evidence/README.md), [agent guidance](AGENTS.md) |

The [CPU preparation package](runtime/sycl/README.md) reproduces all four retained reference images and requests exactly, correcting the initial aspect-padding mismatch. The [adapter](runtime/llama/README.md) supports one, two or four slots and conditional explicit SYCL Q8 C6, with shared demand startup, passive health/model listings and idle unloading. The engine guide separates the installed profile's lifecycle checks from the [completed Q8 KV experiments](docs/sycl-engine-trial.md#q8-kv-experiment), numerical comparisons and earlier isolated Vulkan controller-loss tests. C6 preserves six fully reserved contexts and all resource guards; the earlier private C6 run's roughly 371 MiB process-cap margin is neither VRAM nor a future memory guarantee, and no new Windows peak was collected for the installed repeat.

## Preserve the workload

Use `datalab-to/chandra-ocr-2` revision `af93b47dba1b47b6640c86ccf487ed2260ab9a09`, the exact processor/tokenizer/prompt, original page pixels, a 16,384-token context and the normal 12,384-token output allowance. The retained throughput requests use 1,624 × 2,100 decoded pixels and 3,617 prompt tokens; older frozen renders use 1,632 × 2,112 and 3,959. These are distinct inputs. Record precision changes explicitly: the maintained candidate uses block-INT8 Q8_0 text, not FP8 or unchanged-BF16 inference. Do not credit shorter output, smaller images or another checkpoint as an unchanged-workload speedup. Preserve complete table associations, equations, layout, metadata, raw responses and natural EOS alongside numerical outputs and carried state.

Model weights stay outside Git. The downloader checks pinned file sizes and hashes, refuses existing mismatches and keeps partial files out of accepted names. The inventory verifier checks tensor shapes/ranges and the embedding/output-head bytes. The native importer performs its own authenticated, bounded import; it is not authorized by a model name alone. The pinned model and upstream source are recorded under [provenance](provenance).

## Endpoint compatibility

The service uses model alias `chandra`, `/v1/models` and `/v1/chat/completions`; health and model listings do not load inference or extend its idle lifetime. The router accepts `X-Chandra-Backend: northstone`, `waystone` or `auto`, and backend-specific base URLs. A submitted request is never replayed automatically, and capacity remains held until generation has actually completed or cancellation has retired the worker.

The following CLI form completed native and scanned two-page PDFs through the installed SYCL Waystone route, as well as the prior Vulkan route:

```sh
MAX_VLLM_RETRIES=0 VLLM_API_BASE=http://127.0.0.1:8002/backends/waystone/v1 VLLM_MODEL_NAME=chandra uv run --no-sync chandra input.pdf output --method vllm --batch-size 4 --no-html
```

The [installed Q8 compatibility record](benchmarks/evidence/sycl-q8-normal-service-2026-10-16.json) retains the two successful invocations, four single-submission requests, natural stops and passing saved-output/source/metadata checks. The native and scanned runs took 39.688 and 39.141 seconds. The isolated validation environment also disabled OpenAI SDK retries and recorded a bounded transport journal; `MAX_VLLM_RETRIES=0` alone disables only Chandra's application retries. Each PDF supplied two requests, so these finite checks do not demonstrate six-request CLI overlap or a new sustained rate. The [prior BF16 CLI results](benchmarks/evidence/sycl-cli-compatibility-2026-10-16.json) retain their own timings and provenance.

## Source and offline checks

The service/client test environment is separate from the inference environments. Its existing offline checks are:

```sh
uv sync --locked --group test --group benchmark
uv run --no-sync pytest
python scripts/model_inventory.py /path/to/pinned/model --output .runtime/model-inventory.json
```

These commands check the retained service and DirectCompute tooling; they do not qualify the SYCL engine. [ChandraNative/runtime/build.cmd](ChandraNative/runtime/build.cmd) defines the separate Windows C++ library and offline shader build. Engine experiments use separately owned, resource-bounded execution; build products, models and raw runs belong in ignored project-local locations.

## Retained DirectCompute work

The October 11 experimental pipeline completed a native-0 HTTP singleton and two native-0/native-1 pairs with natural EOS, matching substantive AMD text, tables, equations, order and non-geometric markup. Five/eight geometry differences remained unaccepted, and the native-1 standalone current2415 journal was unavailable. The later current3036 own-row mix-input build preserved the saved trained payloads and state joins, then completed three and four correct pages inside separate 20-minute C1 and C2 windows. These are historical page and concurrency observations, without an accepted sustained optimization gain or deployment. The [progress receipt](benchmarks/evidence/directcompute-progress-2026-10-11.json), [same-build windows](benchmarks/evidence/directcompute-current3036-windows-2026-10-11.json) and [work record](docs/work-record.md) preserve their scope, failures and remaining questions. Their progression holds do not govern the maintained-engine experiments authorized by the current design.

The model is published separately by Datalab under its supplied OpenRAIL license. This repository does not redistribute its weights. [Upstream provenance](provenance/upstream.json) records the fork parent, retained source history and pinned model revision.
