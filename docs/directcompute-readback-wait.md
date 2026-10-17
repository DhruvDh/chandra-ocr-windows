# DirectCompute staging-readback readiness wait

## Observation and scope

At commit 888ab33 the complete D3D11 graph ran on the Intel Arc A770 and its first nine free greedy tokens matched the retained CPU FP32 reference. Root's later full tiny-page attempt stopped after 40 tokens when `Device::readWords` failed with HRESULT 2289696778 (0x887A000A, `DXGI_ERROR_WAS_STILL_DRAWING`). The request was marked failed and retired, the final drain completed, tracked buffers returned to zero, the process exited 1, the Windows Job closed and the ordinary service was restored cold; nothing was replayed. Complete OCR remains unqualified. The implementation addresses only that readback status and preserves arithmetic. The later closed tiny result below supplies its separately bounded recovery and caller-EOS evidence.

## Cause

[device.cpp](../ChandraNative/runtime/device.cpp) at 888ab33 recorded `CopyResource` into a staging buffer, drained an event query, then called `Map(D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT)` exactly once and treated every failure as fatal. Microsoft documents that `D3D11_MAP_FLAG_DO_NOT_WAIT` makes [Map](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-map) return `DXGI_ERROR_WAS_STILL_DRAWING` instead of blocking while the GPU still prevents CPU access ([D3D11_MAP_FLAG](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_map_flag)). That status means the staging resource is not yet readable; it is not device removal, which Map reports with its own removal or reset codes. The driver's reason for reporting it after a completed event query is unknown here.

## Change

[readback_wait.h](../ChandraNative/runtime/readback_wait.h) holds a portable `copyWhenReady` control with no Windows types. `readWords` passes it callables for the existing nonblocking Map, Unmap, the existing `memcpy`, `steady_clock` and `sleep_for`. Exactly `0x887A000A` is retried, on the same staging buffer and the same already-recorded copy. No copy or dispatch is resubmitted, no flush is added, Map is never made blocking and nothing is replayed. Every other failing HRESULT, including device removed, hung or reset, invalid call and `DXGI_ERROR_WAIT_TIMEOUT`, fails on the attempt that returned it. Shaders, operators, `api.h`, `drain`, buffer and staging limits, thread and identity guards and queued-buffer retention are unchanged. When the first Map returns `S_OK`, the bytes copied and the work submitted are the same as at 888ab33.

## Deadline and clocks

| Clock | Starts | Covers | Bound |
| --- | --- | --- | --- |
| Existing pre-copy `drain()` | Its own `steady_clock` start | All previously queued work | Fails after 10,000 ms, checked every 1 ms |
| Post-copy deadline `D` | `steady_clock` immediately after `CopyResource` is recorded | The existing post-copy `drain()`, whose own 10,000 ms clock starts microseconds later, and every Map attempt | `D` = copy time + 10,000 ms, computed once and never reset |
| Map backoff | After each `WAS_STILL_DRAWING` before `D` | Yielding sleeps between nonblocking attempts | 1, 2 and 4 ms, then 8 ms; the last sleep is clipped to end at `D` |
| Root's Windows Job | External | The whole process, including creation, import and every readback | Owned and enforced by root; not replaced |

At least one Map is always attempted, so a drain that ends after `D` still gets the original single nonblocking Map. After `D` a further `WAS_STILL_DRAWING` fails. With the whole window available and exact sleeps, that is 1,254 attempts and 1,253 sleeps; real Windows sleeps can overshoot to the timer granularity, which lowers the count. The post-copy phase therefore ends by about `D` plus one OS sleep overshoot and one nonblocking Map call. The worst case for one `readWords` stays about 20 s, 10 s before the copy plus 10 s after it, plus a few timer-granularity overshoots and the synchronous staging creation. This is the same bound as 888ab33, whose post-copy drain alone could take 10 s. Sharing the drain's existing 10-second policy instead of adding a third window keeps the bound root's Job must allow unchanged.

## Mapped-resource lifetime

The staging `ComPtr` and the result vector are created before the copy and outlive every attempt. A failed Map, including `WAS_STILL_DRAWING`, is never unmapped. Any success status counts as mapped and gets exactly one `Unmap` from a scope guard, even if the copy step throws; only `memcpy` runs while the buffer is mapped, and error text is built after `Unmap`. Only `S_OK` with a non-null `pData` is copied. At 888ab33 any other success status would have been accepted and a null `pData` dereferenced; both are now unmapped and rejected. Failures raise the existing exception type, so the [inference CLI](../ChandraNative/runtime/inference.cpp) still marks the request failed, retires it, drains and records the error without replay.

## Diagnostics

A timeout message reads, for example, `Map completed readback still drawing HRESULT=2289696778 (0x887A000A) at the 10000 ms post-copy readiness deadline after N nonblocking attempts over T ms; copy not resubmitted; fail and retire the request and owned worker`. A non-retried failure keeps the earlier `Map completed readback HRESULT=<decimal>` prefix and adds the hex value, the attempt number and the elapsed time. Unexpected success statuses and null `pData` have their own `unmapped and rejected` messages. Every failure message ends with `device removed reason HRESULT=<decimal>` from `GetDeviceRemovedReason`, which is 0 when the device is not removed. `Device::memoryJson()` adds four cumulative counters under `readback_map`: `waits`, `waits_with_still_drawing`, `still_drawing_results` and `longest_wait_nanoseconds`, measured from the first Map to the final result. The inference CLI copies that object into a `memory_observations[].observed` entry after each phase completes, so a successful run reports the counters at `memory_observations[].observed.readback_map`, and its last entry, `model_buffer_release_and_drain`, covers every readback and shows whether the retry was used. A successful run has no `final_memory`. That key is written only after a failure, and only when the final `drain()` and memory query succeed, in which case `final_drain_completed` is true; `final_memory.readback_map` then also covers the failing phase, which has no `memory_observations` entry of its own. If the final drain or memory query fails, `final_memory` is absent and only the counters from phases completed before the failure remain.

## CPU tests

[readback_wait_test.cpp](../ChandraNative/runtime/readback_wait_test.cpp) drives the same template that `readWords` uses, with single-threaded fakes for the Map status sequence, a monotonic clock advanced only by sleeps and Map cost, and map/unmap/copy lifetime. It covers:

- immediate success and transient-then-success, with the exact backoff and clock accounting;
- persistent `WAS_STILL_DRAWING` ending exactly at the original deadline, including with OS oversleep and with a caller-anchored deadline that the drain has partly used;
- an already-expired deadline;
- ten non-retried error HRESULTs, and device removal after transients;
- unexpected success statuses, null `pData`, and a throwing copy step.

The fakes do not exercise D3D11, the Intel driver, real staging readiness, GPU timing or `device.cpp` itself.

```bash
ChandraNative/runtime/readback_wait_test.sh             # uses $CXX, default c++
CXX=clang++ ChandraNative/runtime/readback_wait_test.sh
```

```bat
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 ChandraNative\runtime\readback_wait_test.cpp /Fo"%CHANDRA_FRESH_BUILD_DIRECTORY%\readback_wait_test.obj" /Fe"%CHANDRA_FRESH_BUILD_DIRECTORY%\readback-wait-test.exe"
"%CHANDRA_FRESH_BUILD_DIRECTORY%\readback-wait-test.exe"
```

## Closed Windows and GPU validation

Build07 compiled the full native graph with MSVC and all runtime shaders with fxc; all 27 portable wait tests passed under MSVC. The next normal tiny request completed 395 tokens through caller EOS, retained the failed request’s first 40 scalar records exactly and retired owned buffers to zero. One readback invocation encountered and recovered from a real WAS_STILL_DRAWING result on the same copy. The longest recorded wait across all invocations was 9.608 ms; the affected token and its individual elapsed time are not available. Native exit 0, Job active 0 and ordinary cold-worker restoration were independently checked. The [October 6 closed-point checkpoint](../benchmarks/evidence/directcompute-2026-10-06.json) records source/build/input/model/device identities and resource limits.

That observation supports real recovery without request replay; it does not explain the driver’s initial post-drain status or establish that all readiness failures recover. Full trained numerical comparison, AMD/exact-client acceptance, broader OCR, native serving and sustained throughput remain open. The original failed attempt remains retained separately.
