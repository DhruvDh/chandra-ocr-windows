# Chandra OCR 2 on Windows

The goal is accurate Chandra OCR 2 on an Intel Arc A770, exposed through the same OpenAI-compatible interface used by the existing ROCm endpoint. A separate router will let callers choose either machine or distribute independent OCR pages between them. Correctness is a prerequisite for comparing speed.

## Engine and reference

The pinned checkpoint is `datalab-to/chandra-ocr-2` at `af93b47dba1b47b6640c86ccf487ed2260ab9a09`. It is Qwen3.5, with a vision encoder, 24 Gated Delta Net layers and eight full-attention layers. Whisper's speech graph cannot execute it by changing a model name. Its Direct3D resource management, dispatch, readback and profiling provide a starting point for the new native operators.

The first full-model reference uses the checkpoint's Transformers processor and graph on PyTorch XPU, with explicit Intel device selection and BF16 weights. This route makes preprocessing, tokenization and tensor comparisons accessible. The native DirectCompute path begins with independently checked BF16 decoding, normalization and recurrent-state kernels. It becomes a serving option only after the complete graph passes the same corpus and state-isolation checks. A maintained OpenVINO backend remains a candidate if exact-model XPU execution is unsuitable; conversion and altered precision each require new validation.

The safetensors payload is approximately 9.86 GiB and stores both the input embedding and output head. Sharing them reduces resident weight storage only after byte equality is verified; the config flag alone is insufficient. Recurrent state remains FP32. Full attention and vision scratch must be bounded and measured, especially for large page images. A 16 GiB adapter does not justify copying another machine's concurrency settings.

## Measurement workflow

Freeze synthetic page pixels, expected content, the OCR prompt and source hashes before collecting a baseline. Check table cells with their row and column associations, equation representations, page count, metadata, end markers and stop reasons. Preserve raw outputs so a plausible-looking answer cannot hide truncation or missing content. Teacher-forced logits and recurrent-state comparisons locate numerical divergence independently of greedy generation.

Record cold loading, first-token latency, generation time, total page latency, output tokens, memory and repeated warm throughput separately. Keep output length and correctness gates alongside every speed result. Tune one material variable at a time and retain the baseline, candidate, change and result. A public synthetic corpus supports regression work; it is not a hidden generalization test.

## Serving and routing

The worker serves model alias `chandra`, `/v1/models`, `/v1/chat/completions` and a non-waking health route. It starts with one active inference and bounded admission. The Chandra client normally permits 12,384 output tokens; a short feasibility probe must be clearly labeled and must not become the production output limit.

The router is separate from both inference engines. It selects a backend before submitting a request, retains a lease through streamed completion or actual cancellation, and reports failures without silently replaying a submitted POST. Explicit selection supports comparisons and maintenance; automatic selection uses configured capacity and observed availability. Management endpoints are not forwarded. Existing ROCm service configuration stays independent of this project's deployment.

## Provenance

The repository retains Const-me/Whisper history at `1ce7399704d8bea92429fc429819d32991fe9152`, with `upstream` pointing to the original repository. It was created as an independent derivative because the initial request was private, then made public with the owner's approval. It is not a GitHub fork-network entry. The original source is MPL-2.0; the downloaded model retains its own OpenRAIL license.
