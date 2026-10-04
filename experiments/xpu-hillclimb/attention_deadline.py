"""Inner deadline; run only inside run_channel's outer Windows Job Object.

The authenticated outer lease owns cleanup after interruption or SSH disconnect.
This helper alone does not establish that ownership.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", required=True, type=int)
parser.add_argument("--receipt", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or not 1 <= args.seconds <= 600:
    parser.error("Explicit command and 1..600 second deadline required")
receipt = Path(args.receipt)
receipt.parent.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
with receipt.with_suffix(".stdout.log").open("w", encoding="utf-8") as stdout, receipt.with_suffix(".stderr.log").open("w", encoding="utf-8") as stderr:
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
    report = {"pid": process.pid, "command": command, "deadline_seconds": args.seconds, "state": "running"}
    receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        process.wait(timeout=args.seconds)
        report["state"] = "completed" if process.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        report["state"] = "deadline_exceeded"
        killed = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, text=True)
        report["tree_stop"] = {"returncode": killed.returncode, "stdout": killed.stdout, "stderr": killed.stderr}
        process.wait(timeout=30)
    report.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started)
    receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report), flush=True)
sys.exit(0 if report["state"] == "completed" else 1)
