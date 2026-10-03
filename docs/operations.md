# Run and manage the endpoints

Deployment acceptance is still in progress. The following commands describe the implemented interfaces; consult the work record before directing research work to a newly installed backend.

The worker imports its GPU runtime only when inference begins. Run it from the repository root using the interpreter in the Intel environment, after the model download and device checks pass:

```powershell
& $RuntimePython -m chandra_service waystone --model-path $ModelDirectory --host 127.0.0.1 --port 18001 --max-input-tokens 16384 --max-output-tokens 12384 --attention hybrid
```

The worker admits one active inference and two waiting requests. It supports one user message with an image data URL and text, deterministic generation, normal Chandra output length and optional SSE streaming. Unsupported options are rejected. Images are processed with the explicit serving profile `shortest_edge=3136, longest_edge=3145728`, matching the established ROCm endpoint. This is a processor pixel-area profile, not a change to the uploaded image bytes. Processed input must fit 4,096 tokens, and input plus requested output must fit the 16,384-token context limit. Hybrid attention uses fused vision attention and eager text attention; full text SDPA failed cached decoding on this runtime and is excluded from serving.

Health and model listings are passive. Five minutes without active or waiting work triggers model unload under the same worker lock. This releases GPU allocations; the next request reloads from local storage. It differs from ROCm's RAM-retaining level-1 sleep and needs its own latency and memory measurements.

After the runtime has initialized, `/health` reports PyTorch's current allocated and reserved GPU bytes and its peak allocated bytes. A fresh worker reports these as null without initializing the GPU. These counters describe PyTorch allocations, not total driver or desktop GPU memory.

The router uses a JSON file containing named private or loopback origins:

```json
{
  "backends": {
    "northstone": {"url": "http://127.0.0.1:8000", "capacity": 2, "expected_page_seconds": 1},
    "waystone": {"url": "http://127.0.0.1:18001", "capacity": 1, "queue_limit": 2, "expected_page_seconds": 1}
  },
  "cooldown_seconds": 15,
  "inference_seconds": 1800
}
```

```sh
uv run --no-sync python -m chandra_service router --config .local/router.json --host 127.0.0.1 --port 8002
```

`X-Chandra-Backend` selects `northstone`, `waystone` or `auto`; responses identify the chosen backend. Automatic selection estimates finish cost as `(active + 1) / capacity × expected_page_seconds`, using the larger of local and reported activity. Equal costs rotate. Replace the example costs with measured page medians; identical costs can make a fast client wait unnecessarily for a slower machine. These static estimates do not predict page difficulty, cold loading or remaining work. The router submits once. If a downstream caller cancels, it drains the already submitted upstream request while retaining that backend's capacity. A transport timeout or disconnect leaves execution uncertain, so the backend is quarantined until its cooldown and an idle health observation. Client libraries can implement their own retries; disable those when measuring failures or when duplicate work matters.

For clients such as the Chandra CLI that expose a base URL, select `http://HOST:8002/backends/waystone/v1` or `http://HOST:8002/backends/northstone/v1`; `http://HOST:8002/v1` chooses automatically. Model listings under every prefix remain passive. Management endpoints are not forwarded, and a path/header disagreement is rejected before submission.

Set `MAX_VLLM_RETRIES=0` when using the stock Chandra CLI with this deterministic endpoint. Chandra otherwise retries failed or repetitive output with increasing temperature, which the worker rejects rather than silently ignoring. This setting disables Chandra's application retries; its OpenAI SDK can still retry transport errors independently. Successful first-pass CLI compatibility does not establish replay-free behavior for that client. Use the benchmark's direct HTTP runner when testing failure or cancellation semantics without client retries.

Explicit Waystone selection permits two waiting requests behind its single active inference, matching the worker's bounded queue and allowing the ordinary two-page CLI batch. Automatic routing considers free execution capacity so it can distribute work across machines before queuing. Execution capacity and waiting capacity are separate; the speed estimate divides by execution capacity only.

The worker requests cooperative cancellation at its total inference deadline and keeps ownership until generation stops. During a GPU or driver hang, process supervision is the final recovery boundary. Stopping a service must stop only its task-owned process tree; do not kill unrelated Python processes or alter drivers to recover this worker.

The deployment uses an SSH channel to Waystone with loopback forwards for the worker and an authenticated heartbeat lease, keeping its worker private and avoiding Windows firewall or administrator changes. The Windows supervisor assigns a suspended child to a Job Object before it can execute; lease EOF, expiry or supervisor exit closes the owned process tree. A separate NorthStone user service owns the router. Repeated `--host` arguments bind loopback and a specific LAN interface in the same process, sharing one scheduling pool. Existing ROCm listeners remain independent. The [work record](work-record.md) records exercised deployment and supervisor checks. GPU cancellation, crash recovery, idle unload and final handoff remain under acceptance.
