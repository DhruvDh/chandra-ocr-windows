"""Verify the supplied checkpoint before allocating model memory on the GPU."""
import hashlib
import json
import threading
import time
from pathlib import Path

REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
CRITICAL = {"model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "video_preprocessor_config.json"}
_cache = {}
_lock = threading.Lock()


def verify(model_path: Path) -> dict:
    start = time.perf_counter()
    manifest_path = Path(__file__).resolve().parents[2] / "provenance/model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or manifest["model"] != "datalab-to/chandra-ocr-2":
        raise RuntimeError("Pinned checkpoint provenance has an unexpected identity")
    entries = {row["file"]: row for row in manifest["files"]}
    if not CRITICAL <= entries.keys():
        raise RuntimeError("Pinned checkpoint provenance is incomplete")
    hashed = 0
    with _lock:
        for name in sorted(CRITICAL):
            path = (model_path / name).resolve(strict=True)
            stat = path.stat()
            expected = entries[name]
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            key = (str(path), expected["sha256"])
            if stat.st_size != expected["size"]:
                raise RuntimeError(f"Checkpoint identity mismatch: {name}")
            if _cache.get(key) != signature:
                with path.open("rb") as file:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                if digest != expected["sha256"]:
                    raise RuntimeError(f"Checkpoint identity mismatch: {name}")
                after = path.stat()
                if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature:
                    raise RuntimeError(f"Checkpoint changed during verification: {name}")
                _cache[key] = signature
                hashed += 1
    return {"revision": REVISION, "files_checked": len(CRITICAL), "files_hashed": hashed, "elapsed_seconds": time.perf_counter() - start}
