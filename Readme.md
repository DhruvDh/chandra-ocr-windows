# Chandra OCR for Windows Intel GPUs

A public derivative of [Const-me/Whisper](https://github.com/Const-me/Whisper), preserving its history and MPL-2.0 source, with a correctness-first workflow for Chandra OCR 2 and an endpoint router for Windows Intel and Linux ROCm machines. The original project documentation is retained in [WHISPER-README.md](WHISPER-README.md).

Full-model Intel validation is in progress. The repository contains a synthetic OCR benchmark, numerical comparison tools, a Transformers XPU adapter, an OpenAI-compatible worker/router and native DirectCompute primitive tests. The native primitives are a foundation for Chandra's Qwen3.5 graph; they do not yet implement the complete model. See the [current work record](docs/work-record.md) for the tested boundary.

| Work | Start here |
| --- | --- |
| Understand the approach | [Design](docs/design.md) |
| Reproduce inputs and evaluate results | [Benchmark workflow](benchmarks/README.md) |
| Inspect the checkpoint | [Download manifest](provenance/model.json), [inventory verifier](scripts/model_inventory.py) |
| Run native operator checks | [DirectCompute guide](ChandraNative/README.md) |
| Inspect the Intel reference | [Runtime project](runtime/waystone/pyproject.toml), [adapter](runtime/waystone/backend.py) |
| Serve or route OCR | [Service package](chandra_service), [operations](docs/operations.md) |
| Resume implementation | [Work record](docs/work-record.md), [agent guidance](AGENTS.md) |

## Start with correctness

The lightweight service/client environment is separate from the Intel inference environment. Install Python 3.12.14 and dependencies locally with uv, then run the offline checks:

```sh
uv sync --locked --group test --group benchmark
uv run --no-sync pytest
python scripts/model_inventory.py /path/to/pinned/model --output .runtime/model-inventory.json
```

The model verifier checks every download hash, tensor shape and byte range, then verifies the supposedly tied embedding and output head by hashing their actual bytes. Model weights stay outside Git. The exact checkpoint and upstream commits are recorded under [provenance](provenance).

Benchmark timing is meaningful only alongside verified inputs, complete output and numerical acceptance. The checked-in corpus is synthetic development data, not a hidden quality benchmark. The workflow preserves raw failures and partial output, separates cold and warm measurements, and refuses to promote timing results with incomplete provenance.

## Use the endpoint contract

Both workers use model alias `chandra` and `/v1/chat/completions`; model listings and health checks do not load the model. The router accepts `X-Chandra-Backend: northstone`, `waystone` or `auto`. Automatic routing estimates finish time from available capacity and configured page timings, rotating equal-cost choices. It does not replay an already submitted request after a failure.

The existing Chandra CLI contract is:

```sh
VLLM_API_BASE=http://127.0.0.1:8002/v1 VLLM_MODEL_NAME=chandra uv run --no-sync chandra input.pdf output --method vllm --batch-size 2 --no-html
```

That command requires a configured, validated endpoint; it is not evidence that the Windows deployment has passed acceptance. The normal output allowance is 12,384 tokens. Private host configuration, downloaded runtimes, models and raw experiments belong in ignored project-local directories.

The model is published separately by Datalab under its supplied OpenRAIL license. This repository does not redistribute its weights. [Upstream provenance](provenance/upstream.json) distinguishes this history-preserving derivative from a GitHub fork-network entry.
