# Original-weight import boundaries

The latest import-only diagnostic found a direct-copy discrepancy in original weights before inference execution. All ten selected initial tail readbacks matched snapshots of the actual submission pointers. After 732 original uploads totaling 9,078,531,072 bytes and constructor return, the second head shard's selected tail failed byte equality. This is a correctness finding, not a repaired model.

| Selected tail | Initial direct readback | Final direct readback after import |
| --- | --- | --- |
| Shard 0, global row 26,213 | Equal to its submission source | Still equal |
| Shard 1, global row 52,427 | Equal to its submission source | 64 differing u32 words spanning 256 bytes |
| Other eight selected tails | All equal to their submission sources | Unobserved: the first fault stopped further checks |

The shard 1 difference occupies u32 words 768 through 831, corresponding to BF16 words 1,536 through 1,663 and row byte interval [3,072, 3,328). Independent checkpoint authentication freshly rehashed the original 10,591,220,088-byte file and found both tied head and embedding row 52,427 values exactly equal to the captured upload source. All 128 differing BF16 words in the bad direct copy decode to finite values: 33 zeros, 62 subnormals and 33 normals. Finiteness alone therefore cannot establish weight integrity.

The diagnostic ran no inference shader or vision/text arithmetic and generated no tokens. It retained the original 3,617-token input, 16,384-token context and normal 12,384-token output allowance. The total native process path of 59.0336828 seconds covers import and observer work; it is not inference latency, cold-startup performance or a pages-per-second measurement. Root authenticated 18 retained files totaling 2,904,765 bytes. Separate evidence confirms exact process and transport retirement, closed local and guardian ownership, and restoration of the unchanged historical endpoint cold.

The [earlier interrupted observation](directcompute-interrupted-watch-arithmetic-2026-10-07.md) had a discrepant SRV-derived row 52,427 without a direct-copy counterpart. This later point supplies a direct-copy discrepancy for that row before inference; it leaves the earlier point's original evidence unchanged. The observation narrows the investigation to the interval between the initial and final shard 1 readbacks, including remaining imports, observer operations and source-view release. Resident storage, direct-copy/staging/readback, synchronization or observer effects remain alternatives, and no application or driver cause is established.

The next source-only observer targets per-upload boundaries and release of the importer's own mapped source view. Current source scopes each `View` to a shard while model and file handles outlive it. These source findings identify observation boundaries; no newly compiled or executed observer, runtime cause or production fix is claimed. The [redacted evidence receipt](../benchmarks/evidence/directcompute-original-weight-import-2026-10-07.json) binds the accepted observation, checkpoint metadata and closure reviews by logical label, byte count and SHA-256. Finished review accepts observation integrity and custody only. Whole-model integrity, complete OCR, native endpoint, timing, page throughput and deployment remain unaccepted.
