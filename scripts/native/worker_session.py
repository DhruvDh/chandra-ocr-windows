"""Run one scripted stdin/stdout session against chandra-worker.exe and retain every event.

Standard library only. This is a local protocol exerciser for root-owned acceptance, not an endpoint:
it opens no listener, creates no Job and never replays a request. The caller must already run inside
the required external kill-on-close Job (the worker itself refuses --execute otherwise). The session
file is JSON lines; each line is one step:
  {"send": {...protocol message...}}            write one message line
  {"send_raw": "text"}                          write this UTF-8 text verbatim (e.g. malformed lines)
  {"expect": {"event": "terminal", "id": "a"}}  wait for an event containing these key/values
  {"expect": {...}, "timeout_seconds": 3600}    optional per-step deadline (default --timeout-seconds)
While waiting, a lease message is sent every --lease-seconds. After the last step (or any failure) stdin
is closed, which makes the worker retire remaining work, release the model and drain before exit. A final
{"expect": {"event": "exit"}} step instead observes an exit the worker reached with stdin still open.
"""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

REQUEST_SCHEMA = "chandra.directcompute.worker-request.v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, type=Path, help="JSON-lines step file")
    parser.add_argument("--events", required=True, type=Path, help="fresh file receiving every worker stdout line")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--lease-seconds", type=float, default=0.0, help="send a lease message at this interval while waiting (0 disables)")
    parser.add_argument("worker", nargs=argparse.REMAINDER, help="-- followed by the worker command line")
    args = parser.parse_args()
    command = args.worker[1:] if args.worker[:1] == ["--"] else args.worker
    if not command:
        parser.error("pass the worker command after --")
    steps = [json.loads(line) for line in args.session.read_text(encoding="utf-8").splitlines() if line.strip()]
    for step in steps:
        if set(step) - {"send", "send_raw", "expect", "timeout_seconds"} or sum(k in step for k in ("send", "send_raw", "expect")) != 1:
            parser.error(f"invalid step {step}")
    with open(args.events, "xb") as sink:  # Exclusive: never overwrite earlier evidence.
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        events = queue.Queue()

        def pump():
            for raw in process.stdout:
                sink.write(raw)
                sink.flush()
                try:
                    events.put(json.loads(raw))
                except ValueError:
                    events.put({"event": "unparseable_line"})
            events.put(None)
        threading.Thread(target=pump, daemon=True).start()
        failure, seen, lease = None, [], time.monotonic()

        def send(message):
            process.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
            process.stdin.flush()
        try:
            for step in steps:
                if "send" in step:
                    send(step["send"])
                    continue
                if "send_raw" in step:
                    process.stdin.write(step["send_raw"].encode("utf-8"))
                    process.stdin.flush()
                    continue
                deadline = time.monotonic() + float(step.get("timeout_seconds", args.timeout_seconds))
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        raise TimeoutError(f"no event matching {step['expect']}")
                    if args.lease_seconds and now - lease >= args.lease_seconds:
                        send({"schema": REQUEST_SCHEMA, "type": "lease"})
                        lease = now
                    try:
                        event = events.get(timeout=min(0.2, deadline - now))
                    except queue.Empty:
                        continue
                    if event is None:
                        raise RuntimeError(f"worker exited before an event matching {step['expect']}")
                    seen.append(event)
                    if all(event.get(k) == v for k, v in step["expect"].items()):
                        break
        except (OSError, RuntimeError, TimeoutError) as error:
            failure = str(error)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            code = process.wait()
        while True:
            event = events.get()
            if event is None:
                break
            seen.append(event)
    terminals = {e.get("id"): e.get("status") for e in seen if e.get("event") == "terminal"}
    final = seen[-1] if seen else {}
    summary = {"worker_exit_code": code, "session_failure": failure, "terminal_status": terminals, "events": len(seen),
               "exit_event": final if final.get("event") == "exit" else None, "request_replayed": False}
    print(json.dumps(summary, sort_keys=True))
    return 0 if failure is None and code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
