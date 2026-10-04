"""Explicitly admitted direct-node HTTP throughput; disabled without --execute.

Only GET /health and POST /v1/chat/completions are used. No lifecycle routes,
redirects, proxies, POST retries, router selection or hardware ownership release.
A fresh external monitor supplies worker/resource evidence absent from some node
health APIs. Neither this adapter nor client cancellation proves process closure.
"""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

try:
    from . import throughput as core
except ImportError:
    import throughput as core

IDENTITY = 'direct-node-throughput-live-v1'
SOURCE_FILES = ('benchmarks/throughput_live.py', 'benchmarks/throughput.py',
                'benchmarks/endpoint.py', 'benchmarks/evaluate.py', 'benchmarks/prompt.txt')


class AdmissionError(RuntimeError):
    pass


def read_json(path, limit=65536):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise AdmissionError('Receipt exceeds byte bound')
    return json.loads(raw), core.digest(raw)


def positive(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
        raise AdmissionError('Invalid bounded ' + name)
    return value


def validate_admission(admission, url, corpus, monitor, *, execute):
    if execute is not True or admission.get('root_live_throughput_lease') is not True or admission.get('candidate') != IDENTITY:
        raise AdmissionError('Explicit execute and fresh matching root admission required')
    parsed = urlsplit(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise AdmissionError('Explicit single-node origin URL required; no router or API-path mode')
    if admission['direct_url'].rstrip('/') != url.rstrip('/'):
        raise AdmissionError('Admitted direct URL differs')
    now = time.time()
    issued = admission['issued_unix']
    expires = admission['expires_unix']
    if not isinstance(issued, (int, float)) or not math.isfinite(issued) or not 0 <= now - issued <= 300:
        raise AdmissionError('Root admission is stale or future dated')
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', admission['lease_id']):
        raise AdmissionError('Explicit safe unique lease ID required')
    if Path(admission['monitor_path']).resolve() != Path(monitor).resolve():
        raise AdmissionError('External monitor path differs')
    if not admission['node_id'] or not isinstance(admission['expected_worker'], dict) or not admission['expected_worker']:
        raise AdmissionError('Explicit node and worker identity required')
    profile = admission.get('health_profile')
    if profile not in ('waystone', 'northstone'):
        raise AdmissionError('Explicit real node health profile required')
    if not re.fullmatch('[0-9a-f]{64}', admission.get('node_source_sha256', '')):
        raise AdmissionError('Explicit node runtime source SHA256 required')
    expected = admission['expected_health']
    if not isinstance(expected, dict) or not all(expected.get(k) for k in ('backend', 'model', 'revision')):
        raise AdmissionError('Pinned node backend/model/revision required')
    if expected['backend'] == 'router' or 'backends' in expected:
        raise AdmissionError('Router is not a direct-node admission')
    if profile == 'northstone' and admission.get('initial_state') not in ('awake', 'asleep'):
        raise AdmissionError('NorthStone requires an identified awake/asleep worker; cold start identity is not prebound')
    if profile == 'northstone' and admission['initial_state'] == 'asleep' and not admission.get('allow_request_wake'):
        raise AdmissionError('Sleeping worker ordinary-request wake not admitted')
    if type(admission['initial_loaded']) is not bool or type(admission['allow_request_wake']) is not bool:
        raise AdmissionError('Explicit initial load state and wake permission required')
    if not admission['initial_loaded'] or admission['allow_request_wake']:
        raise AdmissionError('Prepared sustained adapter is warm-only; cold/wake studies remain held')
    if profile == 'northstone' and admission['initial_state'] != 'awake':
        raise AdmissionError('Prepared NorthStone adapter requires awake identified worker')
    root = Path(__file__).resolve().parents[1]
    hashes = admission['source_sha256']
    if set(hashes) != set(SOURCE_FILES) or any(core.digest((root / name).read_bytes()) != hashes[name] for name in SOURCE_FILES):
        raise AdmissionError('Benchmark/validator/prompt source binding differs')
    configurations = admission['concurrency']
    if not isinstance(configurations, list) or not configurations or len(set(configurations)) != len(configurations):
        raise AdmissionError('Explicit unique concurrency sequence required')
    for concurrency in configurations:
        core.Bounds(concurrency, admission['request_seconds'], admission['cell_seconds'], admission['response_bytes'],
                    admission['work_limit']).check()
    cycles = admission['warmup_cycles']
    if type(cycles) is not int or not 1 <= cycles <= 10:
        raise AdmissionError('Separately admitted positive warmup required')
    work, metadata = core.corpus_work(corpus, admission['repeat'])
    if metadata != admission['corpus_metadata'] or len(work) != admission['planned_pages'] or len(work) > admission['work_limit']:
        raise AdmissionError('Frozen corpus, order, repeat or work limit differs')
    if len(metadata['fixture_order']) * cycles > admission['work_limit']:
        raise AdmissionError('Warmup exceeds admitted work limit')
    if admission.get('drain_contract') not in ('service-pending-holds-inference', 'external-engine-metrics'):
        raise AdmissionError('Explicit source-supported backend drain contract required')
    if not re.fullmatch('[0-9a-f]{64}', admission.get('drain_contract_source_sha256', '')):
        raise AdmissionError('Backend drain semantics source commitment required')
    if admission['drain_contract'] == 'external-engine-metrics' and not admission.get('engine_metrics_source'):
        raise AdmissionError('Explicit external engine metric source required')
    if profile == 'northstone' and admission['drain_contract'] != 'external-engine-metrics':
        raise AdmissionError('NorthStone gateway requires actual engine drain evidence')
    positive(admission['drain_seconds'], 300, 'drain deadline')
    positive(admission['poll_seconds'], 5, 'monitor interval')
    if admission['poll_seconds'] < .01:
        raise AdmissionError('Monitor polling interval below 10 ms')
    positive(admission['monitor_max_age_seconds'], 30, 'monitor freshness')
    n = len(configurations)
    total_requests = (len(work) + len(metadata['fixture_order']) * cycles) * n
    sample_forecast = (total_requests
        + 2 * n * (math.ceil(admission['cell_seconds'] / admission['poll_seconds']) + 2)
        + (2 * n + 1) * (math.ceil(admission['drain_seconds'] / admission['poll_seconds']) + 1)
        + 4 * n + 1)
    if sample_forecast > 10000:
        raise AdmissionError('Whole admitted envelope exceeds fixed 10000 passive witness samples')
    forecast = (len(configurations) * (2 * admission['cell_seconds'] + 2 * admission['drain_seconds'] + 4)
                + admission['drain_seconds'] + 12)
    if not isinstance(expires, (int, float)) or not math.isfinite(expires) or not now + forecast <= expires <= now + 86400:
        raise AdmissionError('Lease does not cover whole warmup/cells/final drain envelope')
    positive(admission['max_output_bytes'], 2 * 1024**3, 'artifact disk budget')
    if type(admission['max_output_bytes']) is not int or admission['max_output_bytes'] <= 8 * 1024**2:
        raise AdmissionError('Integer output budget must include 8 MiB atomic-summary reserve')
    positive(admission['max_client_bytes'], 512 * 1024**2, 'client working budget')
    requests = {id(item.request): item.request for item in work}
    working = sum(map(len, requests.values())) + 4 * max(configurations) * admission['response_bytes'] + 8 * 1024**2
    if working > admission['max_client_bytes']:
        raise AdmissionError('Conservative input/response working envelope exceeds client budget')
    raw_forecast = (len(work) + len(metadata['fixture_order']) * cycles) * len(configurations) * admission['response_bytes']
    if raw_forecast + sum(map(len, requests.values())) + 8 * 1024**2 > admission['max_output_bytes']:
        raise AdmissionError('Worst-case raw/input disk forecast exceeds admitted output budget')
    return work, metadata


class Artifacts:
    """Count actual file growth; reserve a bounded terminal summary separately."""
    def __init__(self, output, maximum):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.maximum = maximum
        self.sizes = {}
        self.used = 0

    def put(self, name, data, *, terminal=False, append=False):
        if Path(name).name != name:
            raise AdmissionError('Unsafe artifact name')
        old = self.sizes.get(name, 0)
        size = old + len(data) if append else len(data)
        ceiling = self.maximum if terminal else self.maximum - 8 * 1024**2
        # Atomic replace temporarily holds old+new files; append grows directly.
        transient_peak = self.used + len(data)
        if transient_peak > ceiling or (terminal and size > 4 * 1024**2):
            raise AdmissionError('Actual artifact byte budget exceeded')
        path = self.output / name
        if append:
            with path.open('ab') as stream:
                stream.write(data)
        else:
            temporary = path.with_suffix(path.suffix + '.pending')
            temporary.write_bytes(data)
            temporary.replace(path)
        self.used += size - old
        self.sizes[name] = size

    def json(self, name, value, *, terminal=False):
        self.put(name, (json.dumps(value, indent=2) + '\n').encode(), terminal=terminal)


class Witness:
    def __init__(self, admission, monitor, artifacts):
        self.admission, self.monitor, self.artifacts = admission, monitor, artifacts
        self.samples = 0

    def sample(self, *, admission_required, warm_required=False):
        value, sha = read_json(self.monitor)
        now = time.time()
        a = self.admission
        for key, expected in (('lease_id', a['lease_id']), ('node_id', a['node_id']),
                              ('worker_identity', a['expected_worker']), ('source_sha256', a['node_source_sha256']),
                              ('model_revision', a['expected_health']['revision']),
                              ('model', a['expected_health']['model']), ('backend', a['expected_health']['backend'])):
            if value.get(key) != expected:
                raise AdmissionError('External passive witness differs: ' + key)
        issued = value.get('issued_unix')
        if not isinstance(issued, (int, float)) or not math.isfinite(issued) or not 0 <= now - issued <= a['monitor_max_age_seconds']:
            raise AdmissionError('External passive witness stale or future dated')
        if admission_required and (now >= a['expires_unix'] or value.get('allow_admission') is not True
                                   or value.get('resources_passed') is not True or value.get('abort') is not False):
            raise AdmissionError('Lease/resource/external abort prevents admission')
        allowed_states = ('ready', 'cold') if a['health_profile'] == 'waystone' else ('awake', 'asleep')
        warm_state = 'ready' if a['health_profile'] == 'waystone' else 'awake'
        if value.get('health_state') not in allowed_states:
            raise AdmissionError('External witness has unavailable/transitional node load state')
        if admission_required and (warm_required or not a['allow_request_wake']) and value['health_state'] != warm_state:
            raise AdmissionError('Unadmitted node wake/load transition')
        self.last_value = value
        self.samples += 1
        if self.samples > 10000:
            raise AdmissionError('Passive monitor sample count bound reached')
        # Engine values are external observed telemetry; pending is never renamed engine activity.
        entry = {'observed_unix': now, 'receipt_sha256': sha, 'worker_identity': value['worker_identity'],
                 'allow_admission': value.get('allow_admission'), 'resources_passed': value.get('resources_passed'),
                 'abort': value.get('abort'), 'health_state': value['health_state'], 'engine_telemetry': value.get('engine_telemetry'),
                 'engine_telemetry_status': 'reported_external' if value.get('engine_telemetry') is not None else 'unavailable'}
        self.artifacts.put('monitor.jsonl', (json.dumps(entry) + '\n').encode(), append=True)
        return value


def check_health(info, admission, *, initial=False):
    if not isinstance(info, dict) or 'backends' in info:
        raise AdmissionError('Passive health is not a single-node health response')
    if admission['health_profile'] == 'waystone' and any(info.get(key) != value for key, value in admission['expected_health'].items()):
        raise AdmissionError('Passive node model/backend identity differs')
    if type(info.get('pending_or_active')) is not int or info['pending_or_active'] < 0:
        raise AdmissionError('Passive pending count unavailable')
    worker = admission['expected_worker']
    for key, expected_key in (('worker_pid', 'pid'), ('starts', 'starts')):
        if key in info and (expected_key not in worker or info[key] != worker[expected_key]):
            raise AdmissionError('Passive worker identity changed: ' + key)
    if admission['health_profile'] == 'waystone':
        if type(info.get('loaded')) is not bool or info.get('state') not in ('ready', 'cold'):
            raise AdmissionError('Passive Waystone load state unavailable/unhealthy')
        if initial and (info['loaded'] != admission['initial_loaded'] or info.get('accepting') is not True):
            raise AdmissionError('Initial Waystone load/accepting state differs')
    else:
        required = ('state', 'pending_or_active', 'worker_pid', 'starts', 'wakes', 'sleeps',
                    'last_start_seconds', 'last_wake_seconds', 'last_error')
        if any(key not in info for key in required) or info['state'] not in ('awake', 'asleep'):
            raise AdmissionError('Passive NorthStone gateway state contract differs')
        if initial and info['state'] != admission['initial_state']:
            raise AdmissionError('Initial NorthStone awake/asleep state differs')
    if initial and info['pending_or_active'] != 0:
        raise AdmissionError('Initial node is not idle')
    return info


async def run(admission, url, corpus, monitor, output, *, execute=False, http_transport=None):
    work, metadata = validate_admission(admission, url, corpus, monitor, execute=execute)
    import httpx
    if httpx.__version__ != '0.28.1':
        raise AdmissionError('Pinned HTTPX 0.28.1 required')
    artifacts = Artifacts(output, admission['max_output_bytes'])
    witness = Witness(admission, monitor, artifacts)
    report = {'schema': 'chandra-direct-node-throughput-v1', 'candidate': IDENTITY,
              'lease_id': admission['lease_id'], 'node_id': admission['node_id'], 'direct_url': url,
              'admission_sha256': core.digest(json.dumps(admission, sort_keys=True).encode()),
              'corpus_metadata': metadata, 'source_sha256': admission['source_sha256'],
              'expected_worker': admission['expected_worker'], 'valid': False,
              'hardware_ownership_released': False, 'backend_drain_verified': False,
              'configurations': [], 'unattempted_concurrency': [],
              'phase_occupancy_status': 'unavailable; scheduler telemetry is not GPU phase occupancy',
              'scope': 'Finite frozen work throughput; no automatic ceiling, first-seen or GPU roofline claim'}
    def save():
        report['artifact_bytes_written_current'] = artifacts.used
        artifacts.json('summary.json', report, terminal=True)
    written = set()
    for item in work:
        if id(item.request) not in written:
            artifacts.put(core.digest(item.request) + '.request.json', item.request)
            written.add(id(item.request))
    save()
    limits = httpx.Limits(max_connections=max(admission['concurrency']), max_keepalive_connections=max(admission['concurrency']))
    transport_events = {}
    async with httpx.AsyncClient(transport=http_transport, trust_env=False, follow_redirects=False,
                                 limits=limits, timeout=httpx.Timeout(2)) as client:
        async def health(stage):
            async with asyncio.timeout(2):
                async with client.stream('GET', url.rstrip('/') + '/health', headers={'Accept-Encoding': 'identity'}) as response:
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(body) + len(chunk) > 65536:
                            raise AdmissionError('Passive health exceeds byte bound')
                        body.extend(chunk)
                    if response.status_code != 200:
                        raise AdmissionError('Passive health request failed')
            info = check_health(json.loads(body), admission, initial=(stage == 'initial'))
            if info['state'] != witness.last_value['health_state']:
                raise AdmissionError('Direct health state differs from external passive witness')
            artifacts.put('health.jsonl', (json.dumps({'stage': stage, 'observed_unix': time.time(), 'health': info}) + '\n').encode(), append=True)
            return info

        async def drain(stage):
            deadline = time.monotonic() + admission['drain_seconds']
            try:
                async with asyncio.timeout(admission['drain_seconds']):
                    while time.monotonic() < deadline:
                        observed = witness.sample(admission_required=False)
                        info = await health(stage)
                        engine_zero = True
                        if admission['drain_contract'] == 'external-engine-metrics':
                            metrics = observed.get('engine_telemetry')
                            engine_zero = (isinstance(metrics, dict)
                                and metrics.get('source') == admission['engine_metrics_source']
                                and type(metrics.get('running')) is int and metrics['running'] == 0
                                and type(metrics.get('waiting')) is int and metrics['waiting'] == 0
                                and isinstance(metrics.get('observed_unix'), (int, float))
                                and 0 <= time.time() - metrics['observed_unix'] <= admission['monitor_max_age_seconds'])
                        if info['pending_or_active'] == 0 and engine_zero:
                            return True
                        await asyncio.sleep(admission['poll_seconds'])
            except Exception as error:
                report['drain_failure'] = type(error).__name__ + ': ' + str(error)
            return False

        warm_required = False
        async def transport(item, request_id, deadline, maximum):
            witness.sample(admission_required=True, warm_required=warm_required)
            event = {'request_id': request_id, 'post_attempts': 0, 'raw_bytes': 0, 'status': None, 'error': None}
            transport_events[request_id] = event
            raw = bytearray()
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionError('Request deadline already expired')
                event['post_attempts'] = 1
                async with client.stream('POST', url.rstrip('/') + '/v1/chat/completions', content=item.request,
                    headers={'Content-Type': 'application/json', 'X-Request-ID': request_id, 'Accept-Encoding': 'identity'},
                    timeout=httpx.Timeout(remaining)) as response:
                    event['status'] = response.status_code
                    async for chunk in response.aiter_raw():
                        remaining_bytes = maximum - len(raw)
                        raw.extend(chunk[:remaining_bytes])
                        if len(chunk) > remaining_bytes:
                            raise AdmissionError('Raw response byte limit exceeded')
                return core.Reply(bytes(raw), event['status'])
            except asyncio.CancelledError:
                event['error'] = 'Client transport cancelled; remote inference drain unverified'
                raise
            except Exception as error:
                event['error'] = type(error).__name__ + ': ' + str(error)
                return core.Reply(bytes(raw), 599)  # Preserve partial bytes, never pass ambiguous execution.
            finally:
                event['raw_bytes'] = len(raw)
                event['response_sha256'] = core.digest(raw)
                artifacts.put(request_id + '.response.raw', bytes(raw))

        async def cell(items, concurrency, prefix):
            nonlocal warm_required
            warm_required = 'measured' in prefix
            cell_report = {'phase': prefix, 'transport_completed_records': 0, 'result': None}
            def receive(record, raw):
                event = transport_events.pop(record['request_id'], None)
                if event is None:
                    artifacts.put(record['request_id'] + '.response.raw', raw)
                artifacts.json(record['request_id'] + '.receipt.json', dict(record, transport=event))
                cell_report['transport_completed_records'] += 1
            bounds = core.Bounds(concurrency, admission['request_seconds'], admission['cell_seconds'],
                                 admission['response_bytes'], admission['work_limit'])
            begun = time.monotonic()
            task = asyncio.create_task(core.measure(items, transport, core.endpoint_validator, bounds,
                        request_prefix=admission['lease_id'] + '-' + prefix, on_record=receive))
            async def watch():
                while True:
                    witness.sample(admission_required=True, warm_required=warm_required)
                    await asyncio.sleep(admission['poll_seconds'])
            watcher = asyncio.create_task(watch())
            try:
                done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    await watcher
                result = await task
                witness.sample(admission_required=True, warm_required=warm_required)  # Catch abort/expiry even after the last response.
                result.pop('records')  # Complete per-request receipts live on disk.
                cell_report['result'] = result
            except asyncio.CancelledError:
                cell_report['abort'] = 'Caller cancelled; no backend closure claim'
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception as error:
                cell_report['abort'] = type(error).__name__ + ': ' + str(error)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            drained = await drain(prefix + '-final')
            cell_report['backend_drain_verified'] = drained
            cell_report['cell_entry_to_passive_backend_drain_seconds'] = time.monotonic() - begun
            valid = (cell_report['result'] is not None and cell_report['result']['valid'] and drained and 'abort' not in cell_report)
            cell_report['valid'] = valid
            cell_report['completed_correct_pages_per_second_including_setup_and_verified_drain'] = (
                len(items) / cell_report['cell_entry_to_passive_backend_drain_seconds'] if valid else None)
            return cell_report

        try:
            witness.sample(admission_required=True)
            report['initial_health'] = await health('initial')
            fixture_count = len(metadata['fixture_order'])
            warmup = tuple(replace(work[i], work_id=f'warmup{cycle}:{work[i].fixture_id}')
                           for cycle in range(admission['warmup_cycles']) for i in range(fixture_count))
            for index, concurrency in enumerate(admission['concurrency']):
                witness.sample(admission_required=True)
                config = {'concurrency': concurrency, 'warmup': None, 'measured': None}
                report['configurations'].append(config)
                before_warmup = await health('pre-warmup')
                if index == 0:
                    check_health(before_warmup, admission, initial=True)
                elif before_warmup['state'] != ('ready' if admission['health_profile'] == 'waystone' else 'awake'):
                    raise AdmissionError('Subsequent warmup node unexpectedly cold/asleep')
                config['warmup'] = await cell(warmup, concurrency, f'c{index}-warmup')
                save()
                if config['warmup']['valid']:
                    witness.sample(admission_required=True, warm_required=True)
                    before_measured = await health('pre-measured')
                    if before_measured['pending_or_active'] != 0 or before_measured['state'] != ('ready' if admission['health_profile'] == 'waystone' else 'awake'):
                        raise AdmissionError('Measured phase requires unchanged warm idle worker')
                    config['measured'] = await cell(work, concurrency, f'c{index}-measured')
                    save()
                if config['measured'] is None or not config['measured']['valid']:
                    report['unattempted_concurrency'] = admission['concurrency'][index + 1:]
                    break
            report['valid'] = (len(report['configurations']) == len(admission['concurrency'])
                               and all(c['measured'] is not None and c['measured']['valid'] for c in report['configurations']))
        except asyncio.CancelledError:
            report['failure'] = 'Caller cancelled; hardware ownership retained'
            raise
        except Exception as error:
            report['unattempted_concurrency'] = admission['concurrency'][len(report['configurations']):]
            report['failure'] = type(error).__name__ + ': ' + str(error)
        finally:
            report['backend_drain_verified'] = await drain('terminal')
            report['valid'] = report['valid'] and report['backend_drain_verified']
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('direct-url', 'corpus', 'monitor', 'admission', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--execute', action='store_true', help='Still requires a fresh matching external root lease')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Live adapter held: explicit --execute and external admission required')
    admission, _ = read_json(args.admission)
    report = asyncio.run(run(admission, args.direct_url, args.corpus, args.monitor, args.output, execute=args.execute))
    return 0 if report['valid'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
