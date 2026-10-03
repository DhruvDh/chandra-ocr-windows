# Run and manage the endpoints

Deployment acceptance is still in progress. The following commands describe the implemented interfaces; consult the work record before directing research work to a newly installed backend.

The worker imports its GPU runtime only when inference begins. Run it from the repository root using the interpreter in the Intel environment, after the model download and device checks pass:

```powershell
& $RuntimePython -m chandra_service waystone --model-path $ModelDirectory --host 127.0.0.1 --port 18001 --max-input-tokens 16384 --max-output-tokens 12384
```

The worker admits one active inference and two waiting requests. It supports one user message with an image data URL and text, deterministic generation, normal Chandra output length and optional SSE streaming. Unsupported options are rejected. Images are processed with the explicit serving profile `shortest_edge=3136, longest_edge=3145728`, matching the established ROCm endpoint. This is a processor pixel-area profile, not a change to the uploaded image bytes. Input plus requested output must fit the context limit.

Health and model listings are passive. Five minutes without active or waiting work triggers model unload under the same worker lock. This releases GPU allocations; the next request reloads from local storage. It differs from ROCm's RAM-retaining level-1 sleep and needs its own latency and memory measurements.

The router uses a JSON file containing named private or loopback origins:

```json
{
  "backends": {
    "northstone": {"url": "http://127.0.0.1:8000", "capacity": 2, "expected_page_seconds": 1},
    "waystone": {"url": "http://127.0.0.1:18001", "capacity": 1, "expected_page_seconds": 1}
  },
  "cooldown_seconds": 15,
  "inference_seconds": 1800
}
```

```sh
uv run --no-sync python -m chandra_service router --config .local/router.json --host 127.0.0.1 --port 8002
```

`X-Chandra-Backend` selects `northstone`, `waystone` or `auto`; responses identify the chosen backend. Automatic selection estimates finish cost as `(active + 1) / capacity × expected_page_seconds`, using the larger of local and reported activity. Equal costs rotate. Replace the example costs with measured page medians; identical costs can make a fast client wait unnecessarily for a slower machine. These static estimates do not predict page difficulty, cold loading or remaining work. The router submits once. If a downstream caller cancels, it drains the already submitted upstream request while retaining that backend's capacity. A transport timeout or disconnect leaves execution uncertain, so the backend is quarantined until its cooldown and an idle health observation. Client libraries can implement their own retries; disable those when measuring failures or when duplicate work matters.

The worker requests cooperative cancellation at its total inference deadline and keeps ownership until generation stops. During a GPU or driver hang, process supervision is the final recovery boundary. Stopping a service must stop only its task-owned process tree; do not kill unrelated Python processes or alter drivers to recover this worker.

The intended local deployment uses an SSH channel to Waystone with a loopback port forward, keeping its worker private and avoiding Windows firewall or administrator changes. A separate NorthStone user service owns the router. Repeated `--host` arguments bind loopback and a specific LAN interface in the same process, sharing one scheduling pool. Existing ROCm listeners remain independent. Installation, restart, cancellation, idle unload and rollback will be recorded here after that deployment has been exercised.
