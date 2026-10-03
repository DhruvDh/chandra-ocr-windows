"""Fetch the exact public checkpoint, verifying each file before accepting it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request

from model_inventory import digest

MANIFEST = Path(__file__).resolve().parents[1] / "provenance/model.json"


def fetch(destination: Path, manifest: dict):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        target = (destination / entry["file"]).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Manifest path escapes destination")
        if target.exists():
            if target.stat().st_size != entry["size"] or digest(target) != entry["sha256"]:
                raise ValueError(f"Existing file differs from the pin; preserve or relocate it before retrying: {target}")
            print(f"Verified {entry['file']}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < entry["size"] + 512 * 1024 * 1024:
            raise ValueError(f"Insufficient space for pinned download and 512 MiB reserve: {entry['file']}")
        # Use only the public pinned revision; do not read tokens or login state.
        url = "https://huggingface.co/" + manifest["model"] + "/resolve/" + manifest["revision"] + "/" + urllib.parse.quote(entry["file"], safe="/")
        request = urllib.request.Request(url, headers={"User-Agent": "chandra-ocr-windows-pinned-download/1"})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".partial", dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                print(f"Downloading {entry['file']} ({entry['size']} bytes)", flush=True)
                h = hashlib.sha256()
                count = 0
                with urllib.request.urlopen(request, timeout=120) as response:
                    while block := response.read(8 * 1024 * 1024):
                        count += len(block)
                        if count > entry["size"]:
                            raise ValueError(f"Download larger than pinned size: {entry['file']}")
                        h.update(block)
                        output.write(block)
                if count != entry["size"] or h.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Download integrity failed: {entry['file']}")
            if target.exists():
                raise ValueError(f"Destination appeared during download; preserving it: {target}")
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print("All pinned files verified. Model execution requires separate device and OCR validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    fetch(args.destination, json.loads(MANIFEST.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
