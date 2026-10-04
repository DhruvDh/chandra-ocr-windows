"""Distinct v2 entry point; preserve the committed v1 exporter unchanged."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import recurrent_run_export
from recurrent_v2_candidate import compiled_recurrent as v2_scope

receipts = []


@contextmanager
def compiled_recurrent(**kwargs):
    with v2_scope(**kwargs) as receipt:
        receipts.append(receipt)
        yield receipt


if __name__ == "__main__":
    invocation = list(sys.argv)
    recurrent_run_export.compiled_recurrent = compiled_recurrent
    try:
        recurrent_run_export.main()
    finally:
        # The shared exporter writes inside its scope. Write this distinct
        # receipt again after restoration, including both v2 executed sources.
        flag = "--replay-output" if "--replay-output" in invocation else "--output"
        if flag in invocation and receipts:
            output = Path(invocation[invocation.index(flag) + 1])
            if output.exists():
                receipt = receipts[-1]
                receipt["source_sha256"].update({name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("recurrent_v2_candidate.py", "recurrent_v2_run_export.py")})
                (output / "recurrent-candidate.json").write_text(
                    json.dumps(receipt, indent=2), encoding="utf-8")
