# DirectCompute resident worker

`chandra-worker.exe` keeps one D3D11 `Device` and one authenticated complete `ModelWeights` resident for its process lifetime, so repeated original OCR requests do not reimport about 10 GB of weights per page. It runs B1 requests one at a time on the Device-owning thread through exactly the same request path as `chandra-inference.exe`, behind a versioned, bounded, newline-JSON stdin/stdout protocol. It is inactive by default. It is not an HTTP endpoint, not batched or concurrent GPU execution, not an XPU/OpenVINO/DirectML backend and not a throughput result. HTTP adaptation is a later task. The [October 6 CPU record](../benchmarks/evidence/directcompute-resident-worker-cpu-2026-10-06.json) now qualifies actual MSVC compilation, 12 inactive/refusal probes and seven Windows protocol/input probes, with all owned Jobs observed empty after retirement. A separate plan authenticates the exact recorded Zotero input: 1,624 × 2,100 pixels, 3,617 prompt tokens, 12,096 patches and the normal 12,384-token output allowance. These are CPU checks; worker GPU execution, complete OCR, numerical correctness and sustained throughput remain open. Qualified worker speedup remains 0%.

## Source and process model

[inference_core.h](../ChandraNative/runtime/inference_core.h) and [inference_core.cpp](../ChandraNative/runtime/inference_core.cpp) hold input authentication, the forecast and execution report builders, greedy readback, the ordered multimodal merge and `generate()`, the complete one-page body: vision forward, text embedding, merge, a fresh `TextRequest` prefill, greedy cached decode to the caller stop IDs or the allowance, and retirement and drain on success or failure. This code was moved out of [inference.cpp](../ChandraNative/runtime/inference.cpp) unchanged except for parameters and optional hooks. Every hook is empty in the CLI, which keeps its options, diagnostics, Device and model lifetime and `wmain`, so its modes, limits, arithmetic, files and exit codes are unchanged. [worker.cpp](../ChandraNative/runtime/worker.cpp) is the worker. No operator, device, shader, diagnostics or `api.h` source changed.

The main thread constructs the `Device`, whose existing guard rejects use from any other thread, imports the weights once and then runs requests serially. A reader thread reads stdin into a bounded line buffer and performs admission, cancellation notification, status and lease renewal. It never calls the `Device` and never touches a GPU buffer. At exit, after the drain, the main thread cancels a still-pending read and joins the reader (see Shutdown and exit). One mutex guards the shared state and serializes every stdout write, so events from both threads form one totally ordered sequence. Each request gets its own `TextRequest`, vision workspace, embeddings, logits and output directory. All of them are created inside `generate()` and retired and drained before its terminal event. No cache or state survives between requests apart from the resident weights.

## Modes and startup

With neither `--plan` nor `--execute`, the worker prints one `inactive` event naming the GEMV selection and exits 0. It does not read stdin. The process-scoped `CHANDRA_EXPERIMENTAL_GEMV_B1` selector is checked first in every mode, before options, Device or model. The exact valid values in the integrated runtime are unset, `0`, `ordered`, `ordered32` and `ordered64`, reported canonically as `predecessor` or the selected name. The wider ordered32/64 kernels retain their experimental source routes but were slower in the closed model-free calibration and were rejected as performance nominees. Any other value, including an empty value, `1` or different case, produces a `fatal` event and exit 1.

| Option | Plan | Execute |
|---|---|---|
| `--model-dir`, `--shader-root` | absolute existing directories, never opened | same; the model is imported once |
| `--input-root`, `--output-root` | absolute existing plain directories; no component of either path may be a reparse point or symlink; canonical forms must be distinct and not nested | same |
| `--pci`, `--luid` | required, recorded, not opened | required; the existing `Device` adapter guard verifies both |
| `--lease-ms` | optional, 1,000–600,000 | required, 1,000–600,000 |
| external Job | observed and reported | required: the immediate Job must have kill-on-close, a finite process or job memory limit and no breakaway |

Every refusal above happens before any `Device` exists. In execute mode the worker creates the `Device`, imports the complete graph (`full_graph_requested`, no omitted tensors, verified tied-head byte equality, the same check the CLI makes) and records the tracked resident bytes as the baseline for every later accounting check. A `Device` or import failure emits `fatal` with stage `startup`, releases whatever was created, emits `exit` and returns 1. `--plan` creates no `Device` and reads no model payload. It authenticates each request's input on the CPU and writes the CLI's forecast report. Plan mode exercises the whole protocol and lifecycle without the GPU. It does not wake the GPU and does not qualify anything.

## Protocol version 1

Input is UTF-8 newline-delimited JSON with at most 4,096 bytes per line. Every message is a JSON object with `"schema": "chandra.directcompute.worker-request.v1"` and a `type`. Duplicate keys and unknown fields are refused. Any well-formed message renews the lease.

| type | fields | effect |
|---|---|---|
| `submit` | `id`, `model`, `input_manifest`, `input_sha256`, `output`, optional `diagnostic_token_cap` | admit one page request or refuse it |
| `cancel` | `id` | cancel a waiting or active request |
| `status` | — | report state from the reader thread; never touches the Device |
| `lease` | — | renew the parent lease only |
| `shutdown` | — | stop admission and retire all work, then exit |

Every stdout line is one event object, at most 65,536 bytes, with `"schema": "chandra.directcompute.worker-event.v1"` and a `seq` that counts up from 0 without gaps. Diagnostics go to stderr, and complete reports go to the request's own `result.json`. An `id` field appears only on events about an admitted request, and admitted IDs are unique for the life of the process. A refusal carries `submitted_id` instead.

| event | meaning |
|---|---|
| `ready` | mode, protocol limits, model alias/revision/context/allowance/caller stops, resident model hashes and bytes (execute), device identity, GEMV selection, observed Job, lease and obligations |
| `admitted` | slot `active` or `waiting`, request index, output name, allowance |
| `rejected` | a submission refused before admission: `reason`, bounded `detail` |
| `started` | the Device thread took the request |
| `phase` | a completed phase with wall seconds and the Device memory observation |
| `token` | ordered `index`, `token_id`, `stop`, best and runner-up logit, margin and tie count, after the row is durably in `generated-tokens.jsonl` |
| `cancel_requested` | acknowledgement with `state` (`waiting`, `admitted`, `active`) and `origin`; `released` is always false |
| `cancel_rejected` | `unknown_id`, `already_terminal`, `cancel_already_requested`, `invalid_id` or `unknown_field` |
| `terminal` | the single final event for a request (see below) |
| `input_channel_failed` | a stdin read failed with a status that is not end of input: `win32_error`, `win32_name` (null for codes outside the known stdin set), `unterminated_bytes` of a discarded partial line and a bounded `detail`; no further input is read |
| `status`, `protocol_error`, `closing`, `exit`, `fatal`, `inactive` | worker state, malformed input, stopped admission, final release, startup refusal, inactive default |

A `terminal` event has `status` set to `completed`, `planned`, `failed` or `canceled`, plus `failure_class`, bounded `error`, `dispatched`, `cancel_origin`, `cancel_boundary`, `generated_tokens`, `stop_reason`, `stop_token_id`, `request_retired`, `drained`, `tracked_buffer_bytes_after`, `resident_model_bytes`, `released`, `worker_poisoned`, `request_replayed` (always false) and the SHA-256 of the exact `result.json` bytes. Capacity is free only after a `terminal` event with `released: true`.

## Admission

The reader thread checks each submission in this order: message shape and fields; an `id` of 1–64 characters from `[A-Za-z0-9._:-]` that this process has never admitted; `model` equal to `chandra`; `input_sha256` as 64 lowercase hex digits; an optional cap that is an integer from 1 to 12,383 (omit it for the normal 12,384-token allowance); and an `input_manifest` relative path whose `/`-separated components are safe names (no `.`, `..`, backslash, colon, absolute form or Windows device name), each existing below the input root and none a reparse point or symlink, ending at a regular file. `output` must be one new safe directory name. Next come closing or poisoned state, the 65,536-admission lifetime bound and capacity: one active plus one waiting request, with any further submission refused as `busy`. Only then does the worker create the output directory exclusively, which a concurrent reuse cannot win, and emit `admitted`. A refused submission creates nothing.

The complete input authentication of the CLI (manifest SHA-256, schema and pinned revision, caller/config/tokenizer pins, typed raw tensors, geometry, positions, full allowance and finite pixels) runs on the Device thread when the request starts, before any dispatch. The worker additionally requires the authenticated caller stop set to be exactly {248044, 248046}. A failure here ends the request with `failure_class: input_refused`, `dispatched: false` and `released: true`, and the worker continues. Inputs are read and hashed at use and never cached.

## Cancellation, failure and survivors

Cancelling the waiting request retires it immediately without dispatch (`cancel_requested` with state `waiting`, then terminal `canceled`, `dispatched: false`). Cancelling the active request only sets a flag, which the Device thread checks at established safe boundaries: before and after input authentication; after the vision `patch_embedding`, `position_added`, every `blocks.N.output` and `pooler_output` observations (each after the graph's own drain); after every text layer output and the logits observation; between the host phases; and before each decode advance. The check throws inside `generate()`, so the request is poisoned, retired and drained by the same path as any graph failure. These observers only read a flag. They add no dispatch, drain, readback or allocation. The CPU tests show the worker's request call sequence is identical to the CLI's. Source cannot preempt an in-flight dispatch or a drain. Its bound is the existing 10-second drain and readback deadline, and beyond that the external Job.

A request survives cancellation, and later requests run, only when its drain completed and tracked buffers returned exactly to the resident model bytes. Any other failure after dispatch poisons the worker: a device, graph, readback, nonfinite-logit or token-file error, or an accounting mismatch. The worker refuses further work, fails the waiting request with `failure_class: worker_poisoned` and `dispatched: false`, releases the model, attempts a final drain and exits 2. Nothing is ever replayed. A `receipt_write_failed` after a completed, drained and accounted request does not poison, because the device work is already retired.

## Shutdown and exit

Stdin EOF, `shutdown`, lease expiry, 16 protocol errors, a stdin read failure, a broken event channel and poisoning all close the worker. Closing stops admission, retires the waiting request, asks the active request to stop at its next safe boundary, releases the model buffers after the request is retired, drains, checks tracked bytes return to zero, destroys the `Device` and writes the final `exit` event, which carries `reason` (the first closing cause), `clean`, `model_released`, `drained`, `tracked_buffer_bytes_after_release`, counts and an `input` record. Exit code 0 means a clean release after EOF or `shutdown` during which input neither failed nor reached the protocol-error ceiling and every event was written. 1 means a configuration or startup refusal, 2 means a poisoned worker or an unclean release, and 3 means a clean release after lease expiry, the protocol-error ceiling, a stdin read failure or a broken event channel. These facts are kept independently of their order: `reason` stays the first closing cause, but a ceiling, read failure or failed event write that follows an EOF or `shutdown` close still makes the exit 3, and the ceiling is judged from the error count as well as the input state. A worker whose event channel broke cannot write its `exit` event, so its exit code is then the only record. The code is settled after that last write: when writing `exit` is itself the first failure, a clean EOF or `shutdown` exits 3 instead of 0 while 1 and 2 keep precedence, so an `exit` event that was written always carries the process exit code. The parent must keep stdin open while it wants work to continue, and must read stdout continuously, because stdout writes block under backpressure.

The reader keeps `GetLastError()` from the statement after a failed `ReadFile`, before any other call. End of input is exactly a successful zero-byte read (a file or the null device at its end; a parent must therefore never write a zero-length message) or `ERROR_BROKEN_PIPE` (109, the parent closed its end of an anonymous pipe). Every other status is a channel failure: the worker emits `input_channel_failed` with the code, closes with reason `input_channel_failed` (retiring the waiting request without dispatch and asking the active request to stop at its next safe boundary), drains and releases as above and exits 3. `ERROR_OPERATION_ABORTED` (995) is not a failure only when it is the completion of the worker's own exit-time cancellation described below; an abort the worker did not request is reported like any other failure. Nothing is replayed on any path.

The 16-error ceiling is absolute. The error that reaches it closes the worker if it is not already closing and ends input at once: lines already buffered in the same read, and anything written later, are never parsed, so no further `protocol_error`, `status` or `rejected` event follows, whichever reason closed the worker first. An unterminated final line is counted as `incomplete_final_line` before input is marked ended, so when it is the sixteenth error the `input` state is `protocol_error_limit` rather than `eof`, `win32_error` keeps the status of that last read (109 for a closed pipe, null for a zero-byte read) and the exit is 3, including when `shutdown` closed the worker first. The reader thread exits immediately and the active request is retired and drained by the Device thread as for any other closing; the parent need not close stdin. Because stdin is no longer read, a parent that keeps writing more than the pipe buffer holds blocks until the worker exits and its end of the pipe closes. After a requested `shutdown` the reader keeps serving `status` and `cancel` while the active request drains. Once the release is complete, the main thread sets a stop flag and calls `CancelSynchronousIo` on the reader thread, whose pending `ReadFile` then completes with `ERROR_OPERATION_ABORTED`. Between reads there is nothing to cancel and the reader sees the flag instead, so the call is repeated every 20 ms up to 50 times; a reader still blocked after that is left to end with the process. Input the reader has not processed when the flag is set, including an end of input behind an unterminated line, is neither parsed nor counted, so the exit code always agrees with the reported `protocol_errors`. The worker then writes `exit` and joins the reader. The `exit` event's `input` record has `state` (`eof`, `failed`, `protocol_error_limit`, `canceled_at_exit`, `not_started` after a startup failure, or `open` if the reader could not be stopped), `win32_error` (the read status, null for a zero-byte read or when no read failed), `read_cancel_requested` and `reader_stopped`. So an ordinary `shutdown` or lease expiry with stdin still open reports `canceled_at_exit` with 995 and keeps exit code 0 or 3 respectively, without an `input_channel_failed` event.

## Outputs

An executed request directory holds `generated-tokens.jsonl`, with the same rows as the CLI, and `result.json`, the CLI's `chandra.directcompute.result.v1` report (including the complete model provenance and device identity) plus a `worker` section and accounting fields. A planned request's `result.json` is the CLI's forecast report plus the `worker` section. A request that never reached the Device gets a small `chandra.directcompute.worker-result.v1` receipt. Every receipt has `passed` false unless execution completed with retirement and accounting, and the full-graph, OCR and performance qualification flags stay false.

## Obligations and limitations

The worker observes but cannot enforce process closure, adapter admission, ACLs or driver behaviour. Root must run it inside a finite kill-on-close Job created by a live parent that holds stdin, use private input and output roots that no other principal can rewrite between check and use, and keep owning service, GPU and closure evidence. The existing [windows_lease.py](../scripts/host/windows_lease.py) sets kill-on-close but no memory limit, so the worker refuses that Job as it stands. Root must add a process or job memory limit. A diagnostic dump (`--diagnostic-dump`) is not offered by the worker; use the CLI. Status, plan and inactive modes never create a `Device`.

## CPU tests

```bash
python3 -I tests/native/test_resident_worker.py
CHANDRA_TEST_SANITIZE=thread TSAN_OPTIONS="halt_on_error=1 exitcode=66" python3 -I tests/native/test_resident_worker.py
CHANDRA_TEST_SANITIZE=address,undefined ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -I tests/native/test_resident_worker.py
```

The suite compiles the predecessor `inference.cpp` from commit `caefe03`, the refactored CLI and the worker with the host C++17 compiler against a test-only [Win32 shim](../tests/native/resident_worker/win32_shim.cpp) and a [recording fake Device](../tests/native/resident_worker/fake_device.cpp). The fake executes no arithmetic, checks the owning thread on every call and returns scripted argmax logits as a control-flow seam. The shim's `ReadFile` follows the synchronous Win32 contract the reader depends on: a pipe whose writer closed fails with `ERROR_BROKEN_PIPE`, a file or device ends with a successful zero-byte read, a read on a directory fails with `ERROR_INVALID_FUNCTION`, one on a write-only descriptor with `ERROR_ACCESS_DENIED`, and an errno with no Win32 counterpart becomes the application-defined `0x20000000|errno`. Its `CancelSynchronousIo` completes a pipe read blocked in that thread with `ERROR_OPERATION_ABORTED` and reports `ERROR_NOT_FOUND` when nothing is pending. `CHANDRA_SHIM_READ_FAIL=N:CODE` fails the Nth stdin read with a Win32 code Linux cannot produce on demand, such as an unrequested abort or `ERROR_INVALID_HANDLE`. `CHANDRA_SHIM_READ_GATE=N:PATH` only delays the Nth stdin read until `PATH` exists, so a test can grow a regular-file stdin and still end it with that file's genuine zero-byte read, and the fake Device's `CHANDRA_FAKE_GATE_DRAIN=N` holds its Nth drain, such as the exit-time drain of an idle worker, until the gate file exists. The pinned caller, generation-config and tokenizer files are not in Git, so synthetic stand-ins replace only those three SHA-256 literals, identically in predecessor and current sources.

The predecessor and refactored CLIs must produce byte-identical fake-device traces, identical stderr and exit codes, and identical `result.json`, `generated-tokens.jsonl` and diagnostic files apart from wall-clock and executable hashes. This holds for the default normal-allowance generation, an immediate stop, an explicit cap, forecast with and without a dump, an executed dump, injected dispatch and drain failures, nonfinite logits, `ordered` and invalid selectors, and a wrong manifest hash.

The worker tests cover:

- selector and configuration refusals with no Device;
- three sequential requests sharing one import, each releasing every request-owned buffer inside its own trace section, with normalized request traces equal to each other and to the CLI's request section;
- a deterministic event stream;
- busy overflow, waiting and active cancellation, and refused duplicate or unknown cancels;
- a later survivor after cancellation at a decode step and at a vision block boundary;
- EOF and `shutdown` with active and waiting work;
- worker poisoning by injected device failure and by nonfinite logits, without dispatching the waiting request;
- input refusal without poisoning;
- 33 admission refusals that create no output;
- bounded malformed protocol handling up to the error limit;
- the 16-error ceiling reached by one 64-line burst while the active request is held in a readback, and again after `shutdown`: exactly 16 errors, the reader thread exits while generation is still held, later input is never parsed, the waiting request retires without dispatch, the active one cancels at a decode step and drains, and the worker exits 3 with stdin still open;
- end of input by pipe close (`ERROR_BROKEN_PIPE`), by a regular file with an unterminated last line and by the null device, all exiting 0;
- stdin read failures at idle startup (directory, write-only descriptor, injected `ERROR_INVALID_HANDLE`, `ERROR_NO_DATA` and an unnamed code), each giving `input_channel_failed` and exit 3, and an unrequested `ERROR_OPERATION_ABORTED` with active and waiting work, which retires both without replay;
- `shutdown` and lease expiry with stdin held open, where the worker cancels its own pending read and reports `canceled_at_exit` without an error;
- the independent counterexample (active and waiting work held, 15 malformed lines, `shutdown`, then an unterminated byte whose end of input is error 16) ended by a closed pipe and by a regular file's zero-byte read, each exiting 3 with `reason` `shutdown_requested`, `input` state `protocol_error_limit` and `win32_error` 109 or null, after the waiting request retired without dispatch and the active one canceled at a decode step and drained;
- the same sixteenth error at end of input in an idle worker, by pipe and by file, which closes with `protocol_error_limit`, and after `shutdown` while the idle worker is held in its exit-time drain, each exiting 3;
- 15 errors after `shutdown` ended by EOF (`eof`, 109) or by the worker's own read cancellation with stdin open (`canceled_at_exit`, 995), each exiting 0;
- a ceiling reached before end of input by pipe, by file and after `shutdown`, where the unterminated byte and EOF behind it are never read (`win32_error` null) and the exit is 3;
- an injected `ERROR_NO_DATA` read failure after `shutdown` and 15 errors, which reports one unterminated byte without counting it, keeps state `failed` with 232 and exits 3;
- a parent that stops reading stdout after `shutdown` or after closing stdin, where the worker still retires and drains the active request, releases the model to zero tracked bytes and exits 3;
- a parent that stops reading stdout only while an idle worker is held in its exit-time drain, so `exit` is the first failed write: EOF and `shutdown` exit 3, while lease expiry (3), an unclean release (2) and a startup failure (1) keep their codes; with stdout still read, each `exit` event carries the process exit code, and both runs leave identical release traces;
- lease expiry;
- plan mode without a Device;
- `ordered` selection and a failed model import;
- the session exerciser, including raw malformed lines, an `exit` observed with stdin still open and the sixteenth error supplied by the end of input it causes.

The ThreadSanitizer and AddressSanitizer/UndefinedBehaviorSanitizer runs pass, the latter with leak detection on now that the reader is joined at exit. These checks do not compile with MSVC, exercise D3D11 or reparse-point semantics, read trained values or establish numerical equivalence.

## Root-owned verification

Run these only after review, on the owner's Windows host, with the existing GPU coordination. The first commands use no GPU.

```bat
ChandraNative\runtime\build.cmd "%CHANDRA_FRESH_BUILD_DIRECTORY%"
"%CHANDRA_FRESH_BUILD_DIRECTORY%\chandra-worker.exe"
set CHANDRA_EXPERIMENTAL_GEMV_B1=1
"%CHANDRA_FRESH_BUILD_DIRECTORY%\chandra-worker.exe"
set CHANDRA_EXPERIMENTAL_GEMV_B1=
python -I scripts\native\worker_session.py --session "%CHANDRA_PLAN_SESSION%" --events "%CHANDRA_FRESH_PLAN_EVENTS%" -- "%CHANDRA_FRESH_BUILD_DIRECTORY%\chandra-worker.exe" --plan --model-dir "%CHANDRA_MODEL_DIR%" --shader-root "%CHANDRA_SHADER_ROOT%" --input-root "%CHANDRA_WORKER_INPUT_ROOT%" --output-root "%CHANDRA_WORKER_OUTPUT_ROOT%" --pci "%CHANDRA_PCI%" --luid "%CHANDRA_LUID%"
```

Check that the build emits `chandra-worker.exe` and review the /W4 output. The second command must print one `inactive` line and exit 0, and the third must print `fatal` and exit 1. In plan mode, the session file is JSON lines of `{"send": ...}` and `{"expect": ...}` steps, for example a `submit` with `input_manifest` such as `tiny/manifest.json` relative to the input root, its SHA-256 and a fresh `output` name, followed by `{"expect": {"event": "terminal", "id": "..."}}`. Expect terminal `planned`, a forecast `result.json`, `exit` with reason `input_eof`, and no Device in the `ready` event.

The input-channel repair adds four more CPU-only plan-mode checks, which create no `Device`. The closed Windows probes now cover pipe EOF, shutdown, the error ceiling, the incomplete final line, a genuine stdin read failure and lease expiry; their accepted receipts are linked above. The recipe below remains a reproducible route, with regular-file/NUL zero-byte EOF variants still unmeasured. Write `%CHANDRA_SHUTDOWN_SESSION%` with the two lines `{"send": {"schema": "chandra.directcompute.worker-request.v1", "type": "shutdown"}}` and `{"expect": {"event": "exit"}}`, and `%CHANDRA_CEILING_SESSION%` with the three lines `{"send_raw": "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n"}`, `{"expect": {"event": "closing", "reason": "protocol_error_limit"}}` and `{"expect": {"event": "exit"}}`. The ordering repair adds two more such checks: write `%CHANDRA_EOF_CEILING_SESSION%` with the one line `{"send_raw": "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{"}` (15 malformed lines and one unterminated byte), after which the exerciser closes stdin, and create `%CHANDRA_FRESH_EOF_CEILING_STDIN%` with the same 46 bytes using the command below. All `%CHANDRA_FRESH_*%` paths must not exist yet.

```bat
set CHANDRA_WORKER=%CHANDRA_FRESH_BUILD_DIRECTORY%\chandra-worker.exe
set CHANDRA_PLAN_ARGS=--plan --model-dir "%CHANDRA_MODEL_DIR%" --shader-root "%CHANDRA_SHADER_ROOT%" --input-root "%CHANDRA_WORKER_INPUT_ROOT%" --output-root "%CHANDRA_WORKER_OUTPUT_ROOT%" --pci "%CHANDRA_PCI%" --luid "%CHANDRA_LUID%"
"%CHANDRA_WORKER%" %CHANDRA_PLAN_ARGS% < NUL
python -I scripts\native\worker_session.py --session "%CHANDRA_SHUTDOWN_SESSION%" --events "%CHANDRA_FRESH_SHUTDOWN_EVENTS%" -- "%CHANDRA_WORKER%" %CHANDRA_PLAN_ARGS%
python -I scripts\native\worker_session.py --session "%CHANDRA_CEILING_SESSION%" --events "%CHANDRA_FRESH_CEILING_EVENTS%" -- "%CHANDRA_WORKER%" %CHANDRA_PLAN_ARGS%
python -I -c "import subprocess,sys; sink=open(sys.argv[1],'xb'); r=subprocess.run(sys.argv[2:],stdin=sink,stdout=subprocess.PIPE); sys.stdout.buffer.write(r.stdout); print('exit code', r.returncode)" "%CHANDRA_FRESH_WRITE_ONLY_STDIN%" "%CHANDRA_WORKER%" %CHANDRA_PLAN_ARGS%
python -I scripts\native\worker_session.py --session "%CHANDRA_EOF_CEILING_SESSION%" --events "%CHANDRA_FRESH_EOF_CEILING_EVENTS%" -- "%CHANDRA_WORKER%" %CHANDRA_PLAN_ARGS%
python -I -c "import sys; open(sys.argv[1],'xb').write(b'{}\n'*15+b'{')" "%CHANDRA_FRESH_EOF_CEILING_STDIN%"
"%CHANDRA_WORKER%" %CHANDRA_PLAN_ARGS% < "%CHANDRA_FRESH_EOF_CEILING_STDIN%"
echo exit code %ERRORLEVEL%
```

Expected results:

- The earlier plan session, which the exerciser ends by closing the pipe, reports `input` `{"state": "eof", "win32_error": 109, ...}`.
- `< NUL` gives `ready`, `closing` with `input_eof`, then `exit` code 0 with `input` state `eof` and a null `win32_error` (a successful zero-byte read).
- The shutdown session exits 0 while stdin is still open, and no `input_channel_failed` may appear. A pending read can report `canceled_at_exit` with status 995; shutdown can also reach the stop path before a read begins. The closed root probe took that earlier path. The separate lease-expiry probe observed status 995 and proves cancellation of a real pending anonymous-pipe read.
- The ceiling session reports `worker_exit_code` 3 (so the exerciser returns 1) with stdin still open. The events file holds exactly 16 `protocol_error` lines, and the `input` state is `protocol_error_limit` with `read_cancel_requested` false.
- Write-only stdin must give `ready`, `input_channel_failed` with a nonzero `win32_error`, `closing` with `input_channel_failed`, then `exit` code 3. `ERROR_ACCESS_DENIED` (5) is the expected status, but this is a prediction for Windows that the run must confirm.
- The end-of-input ceiling session reports `worker_exit_code` 3 (so the exerciser returns 1). Its events file holds exactly 16 `protocol_error` lines, the last with reason `incomplete_final_line`, then `closing` with `protocol_error_limit`, and `exit_event.input` must be `{"state": "protocol_error_limit", "win32_error": 109, "read_cancel_requested": false, "reader_stopped": true}`.
- The same 46 bytes redirected from a file give the same 16 errors, `closing` and `exit` with reason and `input` state `protocol_error_limit`, a null `win32_error` (a successful zero-byte read) and exit code 3.

For execution, run the same exerciser from inside the root-owned Job (kill-on-close plus a finite memory limit) with the admitted adapter. For example, use `--lease-seconds 10 --timeout-seconds 3600` with `--execute ... --lease-ms 60000`, keeping the selector state identical to the retained CLI run being compared. A useful first session has five steps:

1. Submit the retained tiny input twice in sequence with the normal allowance.
2. Submit it again, cancel it after its first `token` event and wait for its terminal.
3. Submit a fourth copy and wait for its terminal.
4. Submit, receive a `token`, then end the session so EOF retires the active request.
5. Optionally, before step 1, submit a second request while one is active, then a third (expect `busy`), and cancel the waiting one.

Acceptance requires all of the following:

- `ready` reports the expected PCI/LUID identity, model hashes, `imports: 1`, GEMV selection and Job limits.
- Each completed request's `generated-tokens.jsonl` equals the retained CLI journal for the same input and selector. For example, the closed tiny run ended at caller EOS 248046 after 395 IDs.
- The repeated and post-cancellation requests reproduce it exactly.
- The canceled request reports `released: true`, `drained: true` and tracked bytes equal to the resident model bytes.
- `exit` reports `clean: true` and zero tracked bytes.
- The Job and host observations show closure.

Compare journals with, for example:

```bat
python -I -c "import json,sys; a,b=([json.loads(l) for l in open(p,encoding='utf-8')] for p in sys.argv[1:3]); print(a==b); sys.exit(a!=b)" "%CHANDRA_CLI_RUN%\generated-tokens.jsonl" "%CHANDRA_WORKER_OUTPUT_ROOT%\%CHANDRA_REQUEST_OUTPUT%\generated-tokens.jsonl"
```

Phase wall seconds are observations of one loaded worker, not page throughput. Sustained throughput, broader OCR, the exact Zotero inputs, full trained vectors and the concurrency ceiling remain unqualified.
