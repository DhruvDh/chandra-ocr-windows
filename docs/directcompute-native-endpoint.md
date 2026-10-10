# DirectCompute native endpoint bridge

The native bridge implements the existing Chandra HTTP service contract through `runtime/native` and the additive `native` launcher mode. Its Python source and fake controls cover strict worker protocol handling, bounded source traversal, failure ownership, acquisition/registration failure and session retirement contention. These checks establish a source boundary under CPython 3.12 with the GIL; actual Windows Job/pipe/handle/CRT/custody behavior, processor and client conformance remain unqualified. No native HTTP service is deployed, and the bridge reports `qualified_full_graph`, `qualified_OCR` and `performance_claim` as false.

The [native runtime](directcompute-runtime.md), [resident worker](directcompute-resident-worker.md), [original native-1 failure](directcompute-native1-head-audit.md) and [native-0 final-page control](directcompute-final-page-probe.md) retain their separate evidence and holds. The failed native-1 page and failed native-0 audit control still block whole-model progression. Source integration does not clear either failure, trained numerical correctness, complete OCR/corpus correctness, endpoint/client acceptance or sustained throughput. GPU/model admission, Windows validation, service transitions and deployment remain root-owned work.

## Compatibility and ownership

The existing service keeps alias `chandra`, `/health`, `/v1/models`, `/v1/chat/completions`, streaming framing, deadlines, cancellation-capacity accounting and idle policy. The separate router and historical XPU mode remain unchanged. Imports, backend construction, health and model listings start no worker, processor, device or weight load. A request is submitted once, with no automatic replay or CPU fallback. Source integration changes no C++/HLSL, dependency lock, checkpoint, precision, image dimensions, context or output allowance.

The native request shape is one user message containing an `image_url` PNG/JPEG/WebP data URL followed by a `text` part, temperature zero, and `max_tokens` 12384 or omitted. Wrong media declarations, multiple frames, excessive decoded pixels/processed patch rows, reordered content or a prompt that cannot preserve the full 12,384-token allowance within the 16,384-token context are refused before submission. The checkpoint remains `datalab-to/chandra-ocr-2` revision `af93b47dba1b47b6640c86ccf487ed2260ab9a09`. Preparation uses the authenticated processor/tokenizer/configs, pinned Chandra caller and established serving profile; it writes the existing input-package ABI, re-reads file lengths/hashes and refuses initialized CUDA/XPU contexts. The processor path has not received real-input acceptance through this bridge.

Startup authenticates the configured executable, shader snapshot and every required worker `ready` commitment: model/config hashes, revision, full graph/import identity, Arc A770 PCI/LUID, context, allowance, stop IDs, selector, lease, protocol/capacity and finite Job limits. A strict reader bounds each event to 65,536 bytes, rejects malformed UTF-8, duplicate keys and nonfinite JSON, and checks sequence, request/output correlation, token indexes, stop flags and terminal release against streamed output. Stalled channels, malformed output, crash, failed source verification or unclean retirement poison the backend. Subsequent work is refused rather than replayed.

Cancellation retains capacity and prepared input until the terminal event affirmatively releases work or whole-worker retirement is confirmed. A canceled request leaves a resident survivor only after clean release. Process exit plus an empty Job are required before closing its controlling handles. Idle unload tries the session condition without waiting under the backend lock; contention skips that tick. Health similarly reports conservative contention state, while shutdown and publication/finalization waits retain ownership within their configured source bounds. These controls do not preempt a GPU dispatch or establish a hard real-time guarantee, arbitrary OS-stall bounds or free-threaded Python behavior.

The only production transport creates a fresh finite kill-on-close, non-breakaway Windows Job, verifies its memory/affinity/single-process limits, creates a suspended worker with only three inherited pipe ends, verifies Job assignment and then resumes. Fixed slots own both pipe outputs, both process-information handles, the Job return value and each held-file result before fallible registry bookkeeping. Held-file names/paths are prepared before acquisition. CRT adoption saves the descriptor before changing raw ownership; raw aliases are not closed after transfer. Failed raw closes remain recoverable, and an ambiguous descriptor number is recorded without retry after possible reuse. `handles_closed` remains false while disposal or child retirement is unknown.

Shader work is globally bounded at 256 files, 1 MiB per file, 8 MiB total, 64 directories including the root, 512 entries, depth four, 240 relative-path characters, 65,536 source lines, 1,024 includes and 4,096 expanded includes per starting file. Enumeration stops at the first excess entry and cyclic or unsafe includes are refused without weakening file/hash authentication. Snapshot removal plans a bounded traversal and checks identity before mutation. Clean custody requires verification, successful held-file release and successful snapshot removal. Failed startup, closure or removal retains a recoverable owner and failed health; shutdown returns false while it remains pending. Fresh startup refuses a prior `h-` snapshot, preventing unattended failed-snapshot accumulation.

`cleanup_retained()` is an explicit operator action on the owning backend after startup, requests and finalization have settled. It records a new cleanup disposition while preserving the original failed receipt and failed admission. It does not submit inference, replay a request or expose an HTTP management route. Health, unload and close do not retry failed disposal. Ambiguous CRT numbers require operator disposition; after the owner process exits, establish OS/process/Job closure and snapshot disposition before fresh startup.

## Explicit configuration

`python -m chandra_service native` requires `--native-config` and `--native-config-sha256`; XPU model/limit options are refused in native mode. The strict, SHA-authenticated JSON is at most 64 KiB and rejects unknown fields. Every path is absolute, normalized, existing and free of symlink/reparse components. Input/output roots are distinct from each other and model/shader roots. The executable and all eight processor/config files must match their pinned commitments; the worker authenticates the original model weights. The example below contains every required field. Zero hashes and LUID are illustrative values that must be replaced with the exact admitted build, reconciled shader tree and host identity; they authorize no launch. Root must admit the CPU, memory/patch limits and private paths and authenticate the completed JSON bytes.

```json
{
  "schema": "chandra.native-endpoint.config.v1",
  "activate": "serve-unqualified-directcompute-native-chandra",
  "executable": "D:\\chandra\\build\\chandra-worker.exe",
  "executable_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "model_dir": "D:\\chandra\\model",
  "shader_root": "D:\\chandra\\repo\\ChandraNative\\shaders",
  "shader_tree_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "input_root": "D:\\chandra\\private\\native-inputs",
  "output_root": "D:\\chandra\\private\\native-outputs",
  "caller_source": "D:\\chandra\\runtime\\.venv\\Lib\\site-packages\\chandra\\model\\hf.py",
  "pci": "03:00.0",
  "luid": "00000000:00000000",
  "gemv_b1_selection": "predecessor",
  "lease_ms": 60000,
  "worker_cpu": 0,
  "process_memory_limit_bytes": 12884901888,
  "job_memory_limit_bytes": 13958643712,
  "max_patch_rows": 12288
}
```

Optional defaults are `max_image_pixels` 4,000,000, `startup_seconds` 900, `admission_seconds` 60, `cancel_grace_seconds` 300, `shutdown_seconds` 120, `terminate_seconds` 30, `retained_outputs` 16 and `processor_dll_bootstrap` false. `worker_cpu` is exactly one admitted logical CPU; `lease_ms` is 1,000–600,000, job memory is at least process memory, and `max_patch_rows` cannot exceed the serving-profile ceiling of 12,288. The configured ceiling is an admission bound, not proof that the hardware can fit or correctly execute that geometry. Store completed configuration, model/build inputs and request/output roots outside public source.

## Source checks and held acceptance

The offline bridge checks use the existing test environment and fake processor/worker/API controls. They cover protocol/reader failures, no replay, request/package commitments, cancellation ownership, publication/retirement contention, global shader bounds, failed cleanup, acquisition/registration failures and descriptor reuse. They run no model, GPU or real Windows worker. Preserve original failed controls and separate source/fake review from native acceptance.

```bash
python -I -m pytest -q -p no:cacheprovider tests/native/test_native_endpoint.py tests/native/test_native_config_and_job.py tests/native/test_native_ownership.py tests/native/test_native_registration.py
```

Before any runtime promotion, root must establish actual Windows containment, inherited-handle/pipe EOF behavior, CRT ownership/recovery and held-source sharing; exact real processor/pixel/tensor/prompt agreement through `runtime.native.prepare_check`; worker protocol and client/HTTP streaming/nonstreaming/cancellation/idle/shutdown behavior with one submission and no SDK replay; native-0 clean-control and native-1 failure disposition; fresh trained numerical and complete-OCR/corpus correctness; and sustained unchanged-workload throughput with complete resource/closure evidence. Existing single-page observations and model-free probes retain their own limited scope. No listener, service or GPU command is authorized by this guide's configuration example or fake test results.
