"""Verify supplied model files against a revision-pinned SHA256 manifest."""
import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
a = p.parse_args(); root = Path(a.model)
manifest = json.loads(Path(a.manifest).read_text())
rows = []
for path in sorted(root.glob("*")):
    if not path.is_file(): continue
    match = next(x for x in manifest["files"] if x["file"] == path.name)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    assert digest == match["sha256"], path.name
    assert path.stat().st_size == match["size"], path.name
    rows.append({"file": path.name, "size": path.stat().st_size, "sha256": digest})
result = {"model": manifest["model"], "revision": manifest["revision"], "files": rows}
Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Verified", len(rows), "files at", result["revision"], flush=True)
