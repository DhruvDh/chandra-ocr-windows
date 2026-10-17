# Current work

Waystone serves Chandra OCR 2 on its Intel Arc A770 using the normal eager Q8 C5/b256 profile. Its accepted sustained result is **80 source-correct measured pages in 758.413182 seconds including drain: 0.1054834 pages/s, about 380 pages/hour**. Seventy-five finished inside the 720-second window; all 84 warmup and measured source reports passed. C5 remains the accepted installed reference and selected profile; this result establishes no hardware ceiling.

| Purpose | Start here |
| --- | --- |
| Use and operate the endpoint | [Project guide](../Readme.md) and [host controls](../scripts/host) |
| Inspect the accepted sustained result | [C5 evidence](../benchmarks/evidence/sycl-q8-c5-normal-service-2026-10-17.json) and [benchmark guide](../benchmarks/README.md) |
| Understand the maintained engine | [SYCL trial and limitations](sycl-engine-trial.md) |
| Recover detailed milestones and earlier failures | [Preserved work history through October 17](work-history-2026-10-17.md) |

The current configuration retains source 90c00f384383e1f2c6815505dda188dba04ac9c3, Q8_0 text/KV, the BF16/F32 projector, five full 16,384-token slots and the 12,384-token output allowance. It remains `qualified:false`. Use live supported controls to determine process state; a benchmark receipt is not a live health check.

Current-stack native and scanned CLI checks pass for the exact historical NorthStone PDF pair: all four saved pages have Markdown and metadata byte-identical to their AMD counterparts, exact parsed requests, natural stops and independent content, table, equation and localization checks. These are four rendered cases of two logical authored pages, without extracted figures. Raw boxes differ, historical AMD provenance remains incomplete, and the shared dense-page errors persist. This establishes neither broad OCR accuracy nor general numerical equivalence. Detailed receipts remain in the operator-local cross-machine-e2e-current-q8-01 bundle, as recorded in the preserved history.

The matched eager C6/b256 trial passed all 82 source checks and its cold capacity gate. Its drain-inclusive rate was 0.1073738 pages/s, 1.7921% above C5, but it completed only 72 pages inside 720 seconds versus C5's 75. This unpaired, boundary-sensitive result establishes no robust gain or higher ceiling; C5 remains selected.

NorthStone's completed private C8 validation measured **208 source-correct pages in 732.915150430 seconds including drain, 0.2837982 pages/s, about 1,022 pages/hour**; 200 finished inside 720 seconds, and all eight warmup plus 208 measured source reports passed. The monitor and client joined zero, all safety guards passed and the owner closed cleanly; the normal AMD service was restored cold before the unchanged source assessment. This confirms the short C8 pilot's rate over the same 720-second window used for Intel, without establishing optimal concurrency, a global plateau or a causal hardware ranking: the stacks and precision differ, the runs are unpaired, and four repeated fixtures represent only two logical authored pages. Normal AMD remains C2.

The earlier C2 reference remains fourteen measured pages in 124.180329 seconds, 0.1127393 pages/s, with all eighteen source reports passing, BF16, full context/output and per-request vision computation preserved. Its two failed optional TVM startup-build children and successful fallback remain separate from the serving worker's clean retirement. The supported optional-builder disable worked in private validation; canonical AMD commit cef58c5 now applies it, with normal cold checks and no new normal model qualification. C15 failed during startup under pressure and produced no measured throughput.

The recurrent-projection candidate passed synthetic operator comparison against instrumented baseline B2: all 2,457,600 output words per arm matched across eight patterns and 24 real Q8 matrices. Eight native files differ from the normal runtime. Qualified timing, representative recurrent activations, complete-model correctness and sustained gain remain unproved; no candidate kernel is deployed. The oneDNN prefill numerical gate is staged, and the Q8 feed-forward fusion build and operator routes are prepared. The Uvicorn short-circuit source fix is tested but undeployed. These preparations establish no endpoint speedup.

Further Windows runtime work is held for fresh adequate physical/global-commit headroom; the latest oneDNN model admission was withheld for insufficient observed reserves. Aggregate kernel pools rose while OCR was unloaded, but no allocation owner or cause is established. Existing forecasts, caps and floors remain unchanged.

Keep source changes in their named worktrees, current decisions in the project record and detailed evidence with its experiment. Retain failed trials that explain a decision, reproduce a defect or support rollback. Use temporary space for disposable fake outputs and retire them before freezing a successful packet; do not create another model, SDK or runtime copy when a retained input can be referenced. Preserve active bundles and unique changes until their actual consumer and retirement boundary are settled.

The chronological history was moved without changing its bytes: SHA256 806589cee0e6dc5f5e62792efa020cd6b5e9110df82c179a1473e9636374850e. Its dated observations retain their original scope; this shorter page is the route to the current work.
