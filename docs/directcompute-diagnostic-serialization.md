# DirectCompute diagnostic serialization

## Observation and scope

Root's MSVC Build09 (Visual Studio 2022 17.14, `/std:c++17`) of the finished native recorder ran the tiny 14-token trained baseline capture to completion: 639 records and 20,514,816 payload bytes, buffers retired, Job drained and closed. The unchanged strict parser in [compare_diagnostics.py](../scripts/native/compare_diagnostics.py) then refused the dump at progress line 6 with `one coordinate object per record required`, because that record's `coordinates` field was a JSON object instead of the required one-element array. That dump stays immutable and unqualified; nothing here rewrites it, accepts it or makes a numerical claim from it. A fresh capture from a corrected build is required.

## Cause

The frozen Build09 source is byte-identical to the recorder at commit caefe03. In the captured `progress.jsonl`, 627 of 639 records carry a bare coordinate object and 12 carry a one-element array. The 627 are exactly the records whose coordinates [diagnostics.cpp](../ChandraNative/runtime/diagnostics.cpp) built by brace-wrapping one existing `nlohmann::json` value: vision patch, position and block rows (`s.coordinates={c};`) and text layer, final-norm and logits rows (`s.coordinates={coordinate(callRow)};`). The 12 arrays (merger, merged embedding and GDN state) were brace literals of key/value pairs, and every `selected_rows` built from an integer (`{r}`, `{s.row}`, `{h}`) was an array.

For a class with an initializer-list constructor, `x={v}` where `v` has that same class type is read two ways. GCC 16.2.1 and Clang 23.1.1 select nlohmann's initializer-list constructor and build `[v]` (the CWG 2137 reading). MSVC, as Build09 shows, copy-initializes `v` itself (the CWG 1467 reading). Brace lists of non-JSON scalars or of key/value pairs are not affected. The existing [diagnostics_test.cpp](../ChandraNative/runtime/diagnostics_test.cpp) is POSIX-only, so the 25 portable tests ran only with GCC and Clang, where the recorder was correct.

Two investigation-only checks bound the defect. Reading the retained dump read-only, an in-memory copy with those 627 objects wrapped passed every other strict-parser check: plan authentication, every consumed-prefix digest, payload SHA-256s, BF16 declarations and the terminal line. The independent field-type check added below flags exactly those 627 lines, starting at line 6, and nothing else in the real producer, commitment, device, conditioning or terminal lines. Neither check wrote a file, and neither is acceptance.

## Change

[diagnostics.cpp](../ChandraNative/runtime/diagnostics.cpp) no longer brace-wraps a JSON value anywhere. Each stage stores exactly one coordinate object in the recorder's internal `Spec`. `put()` alone builds the one-element `coordinates` array with an explicit `Json::array()` and `push_back`, and builds `selected_rows` from the same row that already forms the record key: `[row]`, or `"all"` for the complete conv state. `put()` refuses a non-object coordinate before writing `record_started`. Plan resolution, record keys and filenames, readbacks and budgets, payload bytes, observation counts, consumed-prefix semantics, terminal integrity, [diagnostics.h](../ChandraNative/runtime/diagnostics.h), [api.h](../ChandraNative/runtime/api.h), the observer interfaces, `inference.cpp`, shaders, device code and the strict parser are unchanged. With GCC and Clang, the fixed recorder writes byte-identical dumps to caefe03. The diagnostics-off graph trace binary does not link `diagnostics.cpp`, and none of its sources changed.

## Portable regression

[diagnostics_serialization_test.cpp](../ChandraNative/runtime/diagnostics_serialization_test.cpp) uses standard C++17 only (no POSIX or Win32 headers), so the same file builds with GCC, Clang and MSVC. It drives the real recorder through the fake-Device graph in `inference.cpp` boundary order, so the first record is line 6 as in Build09. Payloads are a deterministic bit pattern that is independent of fake allocation order. Some payloads start with a NaN; FP32 state payloads end with a finite non-BF16 word. It writes three dumps:

| Dump | Plan and run | Lines | Records | `progress.jsonl` SHA-256 |
| --- | --- | --- | --- | --- |
| `complete` | Explicit plan covering every stage: all 15 stage/phase pairs, with GDN state at cache lengths 64, 128, 130, 131 and 132 | 221 | 105 | `5a1a1768a541e09f2f0067adc26f3c5485777a89a74a83dd4e3318138631c29b` |
| `early_stop` | Default plan; only decode step 0 is reached | 460 | 225 | `3463b7484bc79222d1295a792b45f951e42e3a01f886ac72a442e9eea7083564` |
| `failed` | Explicit plan; exclusive create fails two records before the end | 218 | 103 | `3a2574561ec61ef5a29c58e4a3fb04aee25a038cfa3f2af2d02ab61d3963fe03` |

The test re-parses every line and checks the exact field set and JSON type of each event. For records this includes the per-stage coordinate fields, `selected_rows`, both shapes, cache, decode, conditioning, `produces_generated_index`, commitments, producer and qualification. Each start line must match its record. Eight negative controls must be refused, including the Build09 bare vision and decode coordinate objects. It pins the three digests, which GCC and Clang reproduce with both the fixed and the caefe03 recorder, and exits nonzero on any difference. It also reports, without asserting, how this compiler reads `x={jsonValue}`.

[test_native_diagnostic_serialization.py](../tests/native/test_native_diagnostic_serialization.py) builds the test with every host compiler present (g++ and clang++) and loads every dump with the unchanged strict parser. It applies an independent Python field-type check and compares every file byte-for-byte with a build of the caefe03 recorder. It also rebuilds the Build09 shape synthetically: the 86 affected records of the `complete` dump are rewritten as bare objects, and the parser must refuse with exactly `progress line 6: one coordinate object per record required`. It never builds or runs MSVC. Setting `CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT` applies the same parser, schema and digest checks to an output directory written by another compiler.

```bash
python3 -I tests/native/test_native_diagnostic_serialization.py   # g++ and clang++ when present
python3 -I tests/native/test_compare_diagnostics.py
python3 -I tests/native/test_native_diagnostics.py                # existing tests; $CXX, else g++
CXX=clang++ python3 -I tests/native/test_native_diagnostics.py
```

## Root-owned MSVC validation before fresh captures

Root must build and run the same test with MSVC before admitting a fresh GPU capture; no Linux run here compiled or exercised Windows. `CHANDRA_SOURCE` is the checkout root and `CHANDRA_FRESH_BUILD_DIRECTORY` names a directory that does not exist yet.

```bat
@echo off
setlocal EnableExtensions
rem Review artifact only. Root owns execution in a bounded Windows CPU-only stage.
rem Set CHANDRA_SOURCE and CHANDRA_FRESH_BUILD_DIRECTORY to absolute paths before calling.
if not defined CHANDRA_SOURCE exit /b 2
if not defined CHANDRA_FRESH_BUILD_DIRECTORY exit /b 2
if exist "%CHANDRA_FRESH_BUILD_DIRECTORY%" exit /b 2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
mkdir "%CHANDRA_FRESH_BUILD_DIRECTORY%" || exit /b 1
pushd "%CHANDRA_FRESH_BUILD_DIRECTORY%" || exit /b 1
set "CHANDRA_RUNTIME_SOURCE=%CHANDRA_SOURCE%\ChandraNative\runtime"
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%CHANDRA_RUNTIME_SOURCE%\diagnostics_serialization_test.cpp" "%CHANDRA_RUNTIME_SOURCE%\diagnostics.cpp" "%CHANDRA_RUNTIME_SOURCE%\operators.cpp" "%CHANDRA_RUNTIME_SOURCE%\text_model.cpp" "%CHANDRA_RUNTIME_SOURCE%\vision_model.cpp" /Fe:diagnostics-serialization-test.exe > serialization-compiler.log 2>&1 || exit /b 1
diagnostics-serialization-test.exe "%CHANDRA_FRESH_BUILD_DIRECTORY%\serialization-output" > serialization-stdout.json 2> serialization-stderr.log || exit /b 1
cl /nologo /std:c++17 /O2 /fp:strict /EHsc /W4 "%CHANDRA_RUNTIME_SOURCE%\diagnostics_trace.cpp" "%CHANDRA_RUNTIME_SOURCE%\operators.cpp" "%CHANDRA_RUNTIME_SOURCE%\text_model.cpp" "%CHANDRA_RUNTIME_SOURCE%\vision_model.cpp" /Fe:diagnostics-trace.exe > trace-compiler.log 2>&1 || exit /b 1
diagnostics-trace.exe diagnostics-off.trace > trace-stdout.log 2> trace-stderr.log || exit /b 1
set "CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT=%CHANDRA_FRESH_BUILD_DIRECTORY%\serialization-output"
python -I "%CHANDRA_SOURCE%\tests\native\test_native_diagnostic_serialization.py" -v > strict-parser-tests.log 2>&1 || exit /b 1
python -I -c "import hashlib,json,os,pathlib; p=pathlib.Path(os.environ['CHANDRA_DIAGNOSTIC_SERIALIZATION_OUTPUT']); s=json.loads((p/'summary.json').read_text(encoding='utf-8')); assert s['compiler'].startswith('msvc 1944'),s['compiler']; assert s['failures']==0; assert s['schema_lines_checked']==899; assert s['schema_negative_controls_refused']==8; assert s['single_json_brace_assignment']=={'lvalue':'object','prvalue':'object'},s['single_json_brace_assignment']; t=pathlib.Path('diagnostics-off.trace').read_bytes(); assert t.count(b'\n')==22107; assert hashlib.sha256(t).hexdigest()=='9dc656964bfead8f5dd4010fc46d0afd1a5e486e3c96f19b18d44378b025b25a'; print(json.dumps({'compiler':s['compiler'],'schema_lines_checked':s['schema_lines_checked'],'trace_lines':t.count(b'\n'),'trace_sha256':hashlib.sha256(t).hexdigest(),'native_gpu_execution':False}))" > windows-assertions.json 2> windows-assertions-stderr.log || exit /b 1
popd
exit /b 0
```

Expected results:

- `serialization_exit=0`, and stderr ends with `diagnostic serialization tests passed`.
- `serialization-stdout.json` reports a `compiler` beginning with `msvc 1944`, `failures` 0, `nlohmann_json` `3.12.0`, `schema_lines_checked` 899 and `schema_negative_controls_refused` 8.
- For each dump, `serialization-stdout.json` reports `schema_errors` 0, `progress_matches` true and the `progress_sha256` from the table above.
- `single_json_brace_assignment` is expected to be `{"lvalue":"object","prvalue":"object"}`; that is the prediction from Build09's dump, not an assertion. Record the observed value. If MSVC reports `array`, the cause above is contradicted and must be re-examined before admission, even if every other check passes.
- The Python run reports `test_external_output_matches_the_pinned_schema_and_bytes ... ok`. The host-compiled class is skipped on Windows.
- The diagnostics-off trace is expected to have 22,107 lines and SHA-256 `9dc656964bfead8f5dd4010fc46d0afd1a5e486e3c96f19b18d44378b025b25a`, the value from both GCC and Clang. None of its sources changed, so a difference would come from MSVC itself (for example, argument evaluation order changing allocation order), not from this repair. Explain any difference with an MSVC build of the predecessor before claiming the trace unchanged on Windows.

If any `progress_sha256` differs, do not admit a GPU capture. Compare that dump's `progress.jsonl` with the same dump regenerated on Linux by `diagnostics-serialization-test` with GCC or Clang. The first differing byte names the construct that MSVC serializes differently.

## Remaining work

The [Build11 guarded MSVC CPU regression](../benchmarks/evidence/directcompute-diagnostic-serialization-cpu-2026-10-06.json) completed and independently matched GCC/Clang dump bytes. The subsequent [selected trained pair](../benchmarks/evidence/directcompute-selected-trained-pair-2026-10-06.json) passes strict loading and baseline/ordered8 bit equality across 639 payloads at an explicit 14-token prefix. This accepts no CPU/AMD oracle agreement, full 395-token vectors or exact Zotero-page OCR; those reference and corpus gates remain root-owned work. The Build09 dump remains unqualified evidence of this defect only. `runtime/build.cmd` integration of this test belongs to separate work and is not changed here.
