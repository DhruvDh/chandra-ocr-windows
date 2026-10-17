# DirectCompute CPU-side timing

Ordered B1 projection measured about 6.5x faster in correctness-checked public A770 calibration, yet complete tiny-model decode improved only 1.91x (533.5653 s to 279.3955 s) with all 395 scalar logit/margin/token journal rows unchanged. This instrumentation measures where the `chandra::dc::Device` owner thread spends wall time in command recording, buffer creation, drains, readiness polling and sleeps, so later Windows runs can replace guesses about the remaining decode cost with observations. The production implementation compiled on Windows, and its [repaired portable test passed strict MSVC compilation and four inherited environment states](../benchmarks/evidence/directcompute-cpu-timing-msvc-2026-10-06.json), alongside independent GCC/Clang review. The [closed original-tiny 14-token A770 observation](../benchmarks/evidence/directcompute-host-timing-2026-10-06.json) passes 1,712 independent accounting and lifecycle checks with all scalar rows unchanged. Its bounded phase differences retain nested drains and the post-prefill drain boundary; it supplies no GPU execution, throughput or observer-overhead qualification.

## Boundary

Every interval is host wall time on the single Device owner thread, read from `std::chrono::steady_clock` (QueryPerformanceCounter-backed in the MSVC standard library). An interval includes time the thread spends blocked inside D3D11 and driver calls, waiting on the EVENT fence or sleeping. It is therefore not CPU utilization, not GPU execution time, not global or free VRAM, and not throughput. A fence wait is the time between the thread's return from `Flush` and the poll that observed completion: it includes GPU execution of previously recorded work, driver submission latency, the 1 ms polling sleep and its oversleep, and excludes GPU work that finished while the thread was still recording. A drain whose first poll is ready shows that the thread did not wait, not that the GPU did no work. D3D11 drivers may defer work recorded by immediate-context calls into `Flush` or into later calls, so time recorded for one shader is the host cost of that call, not the cost of that shader on the GPU. No GPU timestamp queries, synchronization, GPU buffers or copies are added; the existing `beginProfile`/`finishProfile` GPU timestamps remain the only GPU-side timing route.

## Activation

Set `CHANDRA_NATIVE_CPU_TIMING=1` in the environment of the process before it constructs a `Device`. Unset or `0` disables timing; every other value, including an empty value, `true`, `01` or `1 ` with trailing whitespace, is rejected with `CHANDRA_NATIVE_CPU_TIMING must be unset, 0 or 1; rejected a N-byte value before D3D11 device creation`. The value itself is never echoed. The variable is read once, as the first action of the Device constructor, before adapter selection, `D3D11CreateDevice` or model upload, so a rejected value creates no D3D11 device and allocates no GPU memory. A Device's setting stays fixed for its lifetime; no API can enable, disable or reset it. In `chandra-inference.exe`, input authentication still precedes Device construction, so a malformed value fails after authentication; `execute` catches the constructor exception, records the message as `result.json`'s `error`, and has no Device for `final_memory`.

```bat
set CHANDRA_NATIVE_CPU_TIMING=1
chandra-inference.exe ...existing arguments...
set CHANDRA_NATIVE_CPU_TIMING=
```

On Windows, `set CHANDRA_NATIVE_CPU_TIMING=` in `cmd` removes the variable rather than storing an empty value, as does the CRT's `_putenv_s` with an empty value, so those routes disable timing. Whether an empty entry that a parent process places in the environment block it passes reaches the Device's `_dupenv_s` read as an empty string has not been exercised.

When disabled, the Device keeps a null recorder pointer. Each instrumentation point builds a stack object that tests that pointer and does nothing else. It makes no clock read, allocates nothing, adds no D3D11 call and changes nothing about allocation, binding order, the drain cadence, query flags, the 1 ms drain sleep, staging copies, the readback sequence or error handling. The one disabled-mode difference visible outside the Device is the additive marker described below. When enabled, the only allocations are the recorder, made once at construction with its 128-row shader table and hash index reserved in advance, plus one name copy and one index node the first time each shader name succeeds, and JSON serialization inside `memoryJson`.

## JSON route

`Device::memoryJson()` gains one key, `cpu_timing`. When disabled it is exactly `{"enabled":false,"environment_variable":"CHANDRA_NATIVE_CPU_TIMING","schema":"chandra.directcompute.cpu_timing.v1"}`. When enabled it is the cumulative object below, counted from Device construction. Existing callers expose it unchanged. In `chandra-inference.exe` it appears in each `memory_observations[i].observed.cpu_timing` taken after the named phase (`device_creation`, `authenticated_model_import_upload`, `complete_vision_forward`, `text_embedding`, `ordered_multimodal_merge`, `text_request_cache_creation`, `text_prefill`, `greedy_cached_decode_and_token_readback`, `request_retirement_and_drain`, `model_buffer_release_and_drain`), and in `final_memory` after a failed run. `operator-fixture.exe` reports it in `memory_before`/`memory_after`, and `dispatch-calibration.exe` in each phase's and the report's `memory_before`/`memory_after`. Subtract consecutive observations to attribute counts and nanoseconds to a phase. For example, `greedy_cached_decode_and_token_readback` minus `text_prefill` isolates cached decode together with its per-token logits readback. Maxima are cumulative and cannot be subtracted.

## Interval model

Each instrumented Device entry point opens one scope immediately after its existing owner-thread check. The scope's inclusive interval (`calls`) runs from that point until the scope is destroyed after every later local variable, so it includes destroying temporaries such as COM references and the staging buffer. It covers calls that return and calls that throw, and `failed_calls` is the subset that left by exception. Within a call, contiguous laps share clock reads: each lap ends where the next begins and the final `exit` lap runs from the last named lap to scope destruction. For every operation, the sum of its phase `nanoseconds` equals its `calls.nanoseconds` exactly, and `exit.count` equals `calls.count`. A phase's `count` is the number of laps taken, so loop phases such as `get_data` count polls and phases a call skipped are not counted.

The top-level operations are `dispatch`, `buffer_create`, `upload`, `zero`, `drain.outside_readback` and `readback`. They never overlap, because the Device runs on one thread and none of them calls another, except that `readback` calls `drain` twice. Those two drains are recorded separately as `drain.readback_before_copy` and `drain.readback_after_copy` and are also contained in readback's `before_copy_drain` and `after_copy_drain` laps. Adding them to `readback` therefore double-counts them. `accounted_top_level_nanoseconds` is the sum of the six top-level inclusive totals. Between two observations, the change in `elapsed_nanoseconds` minus the change in `accounted_top_level_nanoseconds` is owner-thread wall time outside instrumented Device calls, such as graph code, host argmax and JSON, `readFloats`' final conversion copy, buffer releases outside a drain, `memoryJson` itself and the gaps between calls. That remainder is not attributed further.

## Fields

Interval fields are `{"count":N,"nanoseconds":T,"maximum_nanoseconds":M}`, where T is the total and M the longest single interval. If an addition would overflow uint64 or an interval is negative, which `steady_clock` should never produce, the three values become `null` and `unavailable` gives `overflow` or `negative_interval`; values never wrap. Counter fields are uint64 numbers, and `null` means an addition would have overflowed. `accounted_top_level_nanoseconds` is `null` if any summand is unavailable or the sum overflows.

| Field | Boundary |
| --- | --- |
| `schema`, `enabled`, `environment_variable` | `chandra.directcompute.cpu_timing.v1`; additive, so later fields need a new schema only if a meaning changes. |
| `boundary` | Fixed text restating the limits above. |
| `clock`, `query_performance_frequency_hz` | Clock source and the value QueryPerformanceFrequency reported at construction (`null` if unknown); 1/frequency is the tick, not a measured clock-read cost. |
| `elapsed_nanoseconds` | From recorder creation at the start of the Device constructor to this `memoryJson` call. The first observation includes adapter selection and device creation, which no operation accounts. |
| `clock_reads` | Clock reads made by the recorder so far, including this observation's own read. |
| `top_level`, `accounted_top_level_nanoseconds` | The six non-overlapping operations and the sum of their inclusive totals. |
| `dispatch.phases.prepare` | Scope start to just before `shader(name)`: argument validation, alias checks and their hash set, constant copy, `CSSetConstantBuffers`, binding arrays. |
| `dispatch.phases.shader_lookup` / `shader_compile` | The `shader(name)` call when the cache already held the name, or when it grew (`D3DCompileFromFile` plus `CreateComputeShader`). A failed compile throws and lands in `exit`. |
| `dispatch.phases.retain` | Pending-buffer retention, the per-call record struct and, while a `beginProfile` window is open, creation of two GPU timestamp queries. |
| `dispatch.phases.record` | Constant `UpdateSubresource`, `CSSetShader`, SRV/UAV binding, `Dispatch`, unbinding and, when profiling, the two timestamp `End` calls. This is host immediate-context recording, not GPU execution or a guaranteed kernel submission. |
| `dispatch.phases.exit` | Destruction of locals and return, or the remainder of a failed call. |
| `dispatch.gpu_timestamp_profiled_calls` | Dispatches recorded inside a `beginProfile` window, whose `retain` and `record` laps include timestamp-query work. |
| `dispatch.shaders[]` | Successful dispatches only, one row per distinct caller-supplied name in first-use order, at most `shader_row_capacity` (128). `calls` is inclusive, `record` is that lap and `compile` counts cache misses. Names are the literal cache keys, so `text_delta` and `runtime/text_delta.hlsl` are separate rows, as they are separate compiled shaders. |
| `dispatch.shaders_beyond_capacity` | Successful dispatches of names first seen after 128 rows were full, aggregated without names. |
| `dispatch.unattributed_calls` | Successful dispatches whose row could not be created because allocation failed. Row `calls`, plus the beyond-capacity and unattributed values, equal `calls` minus `failed_calls`. |
| `dispatch.attribution_overhead` | Per successful dispatch, the time after the call's end spent finding or creating its row. It lies outside every operation interval. |
| `drain.<origin>` | `outside_readback` covers explicit drains from graph code, the importer and inference, plus the drains inside `beginProfile`/`finishProfile`. `readback_before_copy` and `readback_after_copy` are the two drains in `readWords`. |
| `drain.*.phases.query_create` | Scope start through `CreateQuery(EVENT)`, including the timeout argument check. |
| `drain.*.phases.end_flush` | `End(query)` and `Flush()`. The fence wait starts when this lap ends. |
| `drain.*.phases.get_data` | One lap per `GetData(DONOTFLUSH)` poll, including loop setup and, on the first poll, the existing deadline anchor read. |
| `drain.*.phases.deadline_check` | Per not-ready poll, the existing failure check and deadline comparison before sleeping. An expiring deadline throws and lands in `exit`. |
| `drain.*.phases.sleep` | Actual duration of each `sleep_for(1 ms)`. `requested_sleep_nanoseconds` is the requested total, so actual minus requested is oversleep. |
| `drain.*.phases.release` | After a ready poll, clearing the pending retention vector and set, which can run buffer and COM destructors. `retained_buffer_references_released` counts entries cleared, not buffers freed. |
| `drain.*.first_poll_ready` / `first_poll_not_ready` | Drains whose first `GetData` reported completion, or a success status without completion. A failed first poll counts as neither. |
| `drain.*.polls` | Every poll classified as `ready`, `not_ready` (S_FALSE or success without completion) or `failed` (failing HRESULT, which then throws). |
| `drain.*.fence_wait` | One interval per completed drain, from the end of `end_flush` to the end of the completing `get_data` lap. It equals that drain's `get_data`, `deadline_check` and `sleep` laps. Failed drains add none. |
| `readback.phases.before_copy_drain` | Buffer validation and the complete first drain. |
| `readback.phases.staging_prepare` | Size and cap checks, `QueryVideoMemoryInfo`, staging `CreateBuffer` and the zero-initialized host result vector. |
| `readback.phases.copy_record` | `CopyResource` and the existing post-copy deadline anchor read. |
| `readback.phases.after_copy_drain` | The complete second drain. |
| `readback.phases.map_ready` | The unchanged `readback::copyWhenReady`: every nonblocking `Map`, readiness sleeps, the `memcpy` and `Unmap`. |
| `readback.phases.exit` | Existing `readback_map` counter updates, failure diagnostics if any, and staging/result destruction or move. |
| `readback.readiness_sleep`, `readiness_requested_sleep_nanoseconds` | Actual and requested durations of the WAS_STILL_DRAWING backoff sleeps inside `map_ready`. The existing `readback_map` counters keep their meaning. |
| `buffer_create.phases.budget_query` | Scope start through `QueryVideoMemoryInfo` in `floats`/`words`, including size and cap checks. |
| `buffer_create.phases.create` | Budget check, `CreateBuffer` (with initial-data copy if supplied), raw SRV and UAV creation, and tracked accounting. `bytes_created` sums successful creations. |
| `upload`, `zero` | Inclusive `Device::upload` (`UpdateSubresource`) and `Device::zero` (`ClearUnorderedAccessViewUint`) calls; no phases. |

## Observer effects

Enabled timing adds clock reads inside the intervals it measures, and their cost is unmeasured. A successful dispatch makes 7 reads: start, 4 laps, end and one after attribution. A drain that is ready on its first poll makes 6, and each not-ready poll adds 3. A readback makes 7 plus those of its two drains and 2 per readiness sleep. Buffer creation makes 4, upload and zero 2 each, and `memoryJson` 1. `clock_reads` multiplied by a Windows-measured per-read cost bounds that overhead. Per-shader attribution is timed separately as `attribution_overhead`. Building and serializing up to 128 shader rows in `memoryJson` happens after `chandra-inference.exe` stamps a phase, so it lands in the next phase's `wall_seconds`. Sleep requests, poll flags and timeouts are unchanged, but the extra work can shift when polls happen relative to GPU completion.

## Validation

[cpu_timing_test.sh](../ChandraNative/runtime/cpu_timing_test.sh) builds [cpu_timing_test.cpp](../ChandraNative/runtime/cpu_timing_test.cpp) with GCC or Clang, `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror` and AddressSanitizer/UndefinedBehaviorSanitizer into a fresh temporary directory, then runs it four times, with `CHANDRA_NATIVE_CPU_TIMING` absent, empty, `1` and `yes` beforehand. It drives the actual [cpu_timing.h](../ChandraNative/runtime/cpu_timing.h) scopes in the same lap order as [device.cpp](../ChandraNative/runtime/device.cpp), with a fake monotonic clock and a counting replacement `operator new`. It covers:

- accepted and rejected settings, both through the pure `enabledBy`, including the empty string, and through live environment reads by the helper's own branch, `_dupenv_s` under MSVC or `getenv` elsewhere;
- a disabled run of every scope, including exceptional exits, with zero clock reads and zero allocations;
- exact dispatch, drain, readback, buffer, upload and zero partitions;
- compile versus lookup classification;
- immediate-ready, pending, deadline, failed-GetData and invalid-timeout drains;
- drain origin attribution and restoration after readback success and failure;
- top-level totals covering a scripted elapsed interval exactly once;
- the 128-row bound, with no steady-state allocation or growth;
- allocation failure during attribution;
- overflow and negative intervals;
- the JSON keys.

```sh
ChandraNative/runtime/cpu_timing_test.sh
CXX=clang++ ChandraNative/runtime/cpu_timing_test.sh
ChandraNative/runtime/readback_wait_test.sh
```

These tests do not compile `device.cpp` or exercise D3D11, a driver, QueryPerformanceCounter, Windows timer resolution or the GPU. The existing readback readiness test is unchanged and still passes.

The environment check reads and writes the variable on the same `_MSC_VER` branch as the helper, so each compiler runs the read path it builds. It first checks that the helper reads the inherited setting exactly as `enabledBy` reads an independent copy, then removes the variable, sets `0`, `1`, `yes` and a 300-byte value, and finally restores the inherited setting and reads it back. Values are compared and measured, never printed. POSIX `setenv` can store an empty value, so the POSIX branch checks that an empty environment value is rejected as a 0-byte value. No Windows CRT call can store one, because `_putenv_s` with an empty value removes the variable, so the MSVC branch instead checks that this removal leaves the helper reporting timing disabled. On both branches the empty string itself stays covered by `enabledBy("")`. Under MSVC the test refuses an inherited empty value before changing anything, because `_putenv_s` could not restore it. Each run prints one JSON line with `checks_passed`, `environment_route` (`posix_getenv` or `msvc_dupenv_s`), `empty_environment_value` (`rejected` or `not_encodable_putenv_s_removes`) and `preexisting_variable`.

Under MSVC, from a batch file run in the repository root, with `BUILD` set to a fresh directory, the CPU-only route keeps the strict flags that Build10 used:

```bat
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 /WX "ChandraNative\runtime\cpu_timing_test.cpp" /Fe:"%BUILD%\cpu-timing-test.exe" /Fo:"%BUILD%\cpu-timing-test.obj"
if %ERRORLEVEL% neq 0 exit /b 1
set "CHANDRA_NATIVE_CPU_TIMING="
"%BUILD%\cpu-timing-test.exe"
if %ERRORLEVEL% neq 0 exit /b 1
set "CHANDRA_NATIVE_CPU_TIMING=1"
"%BUILD%\cpu-timing-test.exe"
if %ERRORLEVEL% neq 0 exit /b 1
set "CHANDRA_NATIVE_CPU_TIMING=yes"
"%BUILD%\cpu-timing-test.exe"
if %ERRORLEVEL% neq 0 exit /b 1
endlocal
```

It should print three lines with `"checks_passed":10` and `"environment_route":"msvc_dupenv_s"`, the first with `"preexisting_variable":"absent"` and the others `present`. The checks use `neq 0` rather than `errorlevel 1` because a crashed process exits with a negative NTSTATUS code, which `if errorlevel 1` does not catch. The [closed MSVC point](../benchmarks/evidence/directcompute-cpu-timing-msvc-2026-10-06.json) independently passes strict compilation and all ten checks under four inherited states: absent, 0, 1 and yes. It separately observes three compiler descendants before the kill-on-close handle closes; post-close zero is unavailable. Build10's original production compile and failed test compile remain preserved; the repair replaces the GNU attribute and POSIX environment calls.

## Remaining Windows verification

Build11 compiled the production Device and helper, and the standalone repaired test passes strict MSVC `/W4 /WX` with no new warnings. The real A770 host-timing observation passes independent attribution review within its explicitly capped host-wall scope. Complete the remaining activation and observer-overhead checks before treating these measurements as a comparison:

1. With the variable unset, confirm unchanged generated tokens, logits journal and readback bytes against a prior run, and the disabled marker in every observation.
2. With `yes`, confirm rejection before any device identity or model upload is recorded. An empty value cannot be set from `cmd` (see Activation).
3. With `1`, check on real data that phase sums equal inclusive totals, `fence_wait` equals its laps, and `accounted_top_level_nanoseconds` never exceeds `elapsed_nanoseconds`.
4. Record `query_performance_frequency_hz`, a measured per-read `steady_clock` cost, and paired enabled and disabled wall times on the same input to bound the observer effect.

An opt-in event-wait experiment is in separate source implementation. It needs its own capability, lifecycle, numerical/OCR and paired timing checks before promotion; the current host waits alone establish no removable overhead or speed gain.
