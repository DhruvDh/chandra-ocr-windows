"""Observe native Windows worker RAM without importing Torch or waking inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def snapshot():
    if os.name != "nt":
        raise RuntimeError("Run this observer with the installed native Windows runtime")
    import psutil

    workers = []
    for process in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        # Inspect only the dedicated worker module; never emit its full arguments.
        arguments = process.info.get("cmdline") or []
        if arguments[1:4] != ["-m", "chandra_service", "waystone"]:
            continue
        try:
            memory = process.memory_info()
        except psutil.NoSuchProcess:
            continue
        workers.append({
            "pid": process.pid,
            "parent_pid": process.info["ppid"],
            "created_unix_seconds": process.info["create_time"],
            "resident_working_set_bytes": memory.rss,
            "peak_working_set_bytes": getattr(memory, "peak_wset", None),
            "private_committed_bytes": getattr(memory, "private", None),
        })
    system = psutil.virtual_memory()
    return {
        "schema": "chandra-windows-ram-observation-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workers": sorted(workers, key=lambda worker: worker["pid"]),
        "system_total_bytes": system.total,
        "system_available_bytes": system.available,
        "scope": "Windows process RAM snapshot; commit is not resident RAM. Includes a matching venv launcher when present. No process totals are summed; no GPU or driver memory measurement is made.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a new evidence file; refuse replacement")
    args = parser.parse_args()
    result = snapshot()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    print(encoded, end="")
    return 0 if result["workers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
