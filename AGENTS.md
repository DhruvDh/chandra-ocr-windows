# Work on Chandra for Windows

This repository preserves Const-me/Whisper's history while adding Chandra OCR 2 inference, numerical validation and endpoint routing for Windows Intel GPUs. Read [Readme.md](Readme.md), [the design](docs/design.md) and [the work record](docs/work-record.md) before resuming. Keep the original MPL-2.0 license and upstream attribution; model weights have their own license.

Correctness precedes performance. Pin the model revision, processor, runtime and input pixels; compare numerical outputs and complete OCR before promoting a faster implementation. A kernel probe is evidence for that kernel, not proof that the full model works. Record failures and missing measurements as such. Never silently fall back to a CPU, change precision, trim image dimensions or output allowance, or substitute a different checkpoint.

Keep installed tools, environments, models, private host configuration and raw runs under ignored project-local locations. Use synthetic or explicitly authorized public benchmark documents. Do not copy research files, live Zotero state, credentials or machine-private paths into this public repository. Preserve the owner's other services and coordinate GPU allocation before loading another model.

The HTTP boundary accepts only supported model and inference routes. Health and model listings must not start inference. Hold capacity until generation has actually stopped, including after cancellation, and never replay a submitted request automatically. Test concurrency, cancellation and failure with isolated workers before deployment.

Write ordinary prose as one physical line per paragraph. Keep source, setup, measurements and remaining work discoverable here, with concrete commands and honest evidence boundaries. Retain useful failed experiments and necessary build inputs; stop task-owned producers deliberately when they are no longer needed.

Claude delegation is disabled for this work until the owner explicitly asks to resume it. Use GPT-6.1 Sol at xhigh for implementation and finished review, with Luna monitoring quota.
