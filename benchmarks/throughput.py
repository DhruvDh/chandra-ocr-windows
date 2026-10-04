"""Bounded rolling-concurrency measurement and OFFLINE replay; no live adapter.

Callable transports must honor cancellation and enforce the supplied deadline and
response limit while reading. Finite complete-work measurements need enough
repetitions and independent steady-state analysis to estimate a throughput ceiling.
A future live adapter needs external hardware and
owned-process admission. This module imports no GPU runtime and manages no service.
"""
import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Work:
    work_id: str
    fixture_id: str
    request: bytes
    expected_json: str


@dataclass(frozen=True)
class Reply:
    raw: bytes
    status: int = 200


@dataclass(frozen=True)
class Bounds:
    concurrency: int
    request_seconds: float = 900
    run_seconds: float = 3600
    response_bytes: int = 8 * 1024 * 1024
    max_work: int = 10000

    def check(self):
        if type(self.concurrency) is not int or not 1 <= self.concurrency <= 64:
            raise ValueError('Concurrency must be an integer from 1 to 64')
        for value in (self.request_seconds, self.run_seconds):
            if isinstance(value, bool) or not 0 < value <= 86400:
                raise ValueError('Time bounds must be finite, positive and at most one day')
        if type(self.response_bytes) is not int or not 1 <= self.response_bytes <= 64 * 1024 * 1024:
            raise ValueError('Response limit must be 1 byte to 64 MiB')
        if type(self.max_work) is not int or not 1 <= self.max_work <= 10000:
            raise ValueError('Work limit must be 1 to 10000')


def digest(data):
    return hashlib.sha256(data).hexdigest()


async def measure(work, transport, validator, bounds, *, request_prefix, on_record=None):
    """Measure one fixed work sequence, refill freed slots, stop+drain on error.

    transport(item, request_id, deadline_monotonic, max_response_bytes) is async;
    validator(item, Reply) returns a JSON-serializable dict with passed=True only
    after complete response validation. on_record(record, bounded_raw) may persist
    replies. No retries. Cancellation is cooperative; no hard process kill is
    possible in this core. The external caller owns a hard stop for live work.
    """
    bounds.check()
    work = tuple(work)
    if not work or len(work) > bounds.max_work:
        raise ValueError('Frozen workload is empty or exceeds count bound')
    if any(not isinstance(item, Work) or not isinstance(item.request, bytes)
           or not isinstance(item.expected_json, str)
           or any(not isinstance(value, str) or not value or len(value.encode()) > 256
                  for value in (item.work_id, item.fixture_id)) for item in work):
        raise ValueError('Work must contain bounded IDs and immutable request/expectation bytes')
    if len({item.work_id for item in work}) != len(work):
        raise ValueError('Duplicate work IDs')
    if not request_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in request_prefix):
        raise ValueError('Request prefix must be a safe explicit identifier')
    unique_requests = {id(item.request): item.request for item in work}
    unique_expectations = {id(item.expected_json): item.expected_json for item in work}
    if (sum(map(len, unique_requests.values())) > 128 * 1024 * 1024
            or sum(len(value.encode()) for value in unique_expectations.values()) > 4 * 1024 * 1024):
        raise ValueError('Unique workload requests/expectations exceed 128/4 MiB bounds')
    request_digests = {key: digest(value) for key, value in unique_requests.items()}
    expected_digests = {key: digest(value.encode()) for key, value in unique_expectations.items()}
    started = time.monotonic()
    end_deadline = started + bounds.run_seconds
    records = []
    pending = set()
    admitted = 0
    peak = 0
    stopped = None

    async def execute(item, index):
        request_id = f'{request_prefix}-{index:05d}'
        beginning = time.monotonic()
        raw = b''
        status = None
        error = None
        quality = {'passed': False, 'failures': ['not_validated']}
        received = None
        try:
            deadline = min(end_deadline, beginning + bounds.request_seconds)
            reply = await asyncio.wait_for(transport(item, request_id, deadline, bounds.response_bytes),
                                           timeout=max(0, deadline - time.monotonic()))
            received = time.monotonic()
            if not isinstance(reply, Reply) or not isinstance(reply.raw, bytes):
                raise TypeError('Transport must return Reply with raw bytes')
            raw = reply.raw[:bounds.response_bytes]
            status = reply.status
            if received > deadline:
                raise TimeoutError('Transport completed beyond the admitted deadline')
            if len(reply.raw) > bounds.response_bytes:
                raise ValueError('Response byte limit exceeded; retained prefix only')
            if status != 200:
                raise ValueError('HTTP status is not 200')
            quality = validator(item, reply)
            if not isinstance(quality, dict) or quality.get('passed') is not True:
                raise ValueError('Response quality failed')
        except asyncio.CancelledError:
            error = 'Cancelled: run deadline reached; local transport task cancellation completed; remote inference drain unverified'
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        finished = time.monotonic()
        record = {'request_id': request_id, 'work_id': item.work_id, 'fixture_id': item.fixture_id,
                  'request_sha256': request_digests[id(item.request)], 'response_sha256': digest(raw),
                  'response_bytes_retained': len(raw), 'http_status': status, 'error': error,
                  'correctness': quality, 'started_seconds': beginning - started,
                  'response_finished_seconds': None if received is None else received - started,
                  'finished_seconds': finished - started, 'elapsed_seconds': finished - beginning,
                  'passed': error is None and quality.get('passed') is True}
        records.append(record)
        if on_record is not None:
            try:
                on_record(record, raw)
            except Exception as exc:
                record['error'] = 'ReceiptWriteError: ' + str(exc)
                record['passed'] = False
        return record

    def fill():
        nonlocal admitted, peak, stopped
        if stopped is None and time.monotonic() >= end_deadline:
            stopped = 'run_deadline'
        while stopped is None and admitted < len(work) and len(pending) < bounds.concurrency:
            pending.add(asyncio.create_task(execute(work[admitted], admitted)))
            admitted += 1
        peak = max(peak, len(pending))

    try:
        fill()
        while pending:
            remaining = end_deadline - time.monotonic()
            if remaining <= 0:
                stopped = stopped or 'run_deadline'
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending)
                pending.clear()
                break
            done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                continue
            completed = [task.result() for task in done]
            if any(not record['passed'] for record in completed):
                stopped = stopped or 'request_error_or_quality_failure'
            fill()
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    elapsed = time.monotonic() - started
    correct = sum(record['passed'] for record in records)
    valid = stopped is None and admitted == len(work) and len(records) == len(work) and correct == len(work)
    return {'schema': 'chandra-throughput-configuration-v1', 'valid': valid,
            'concurrency': bounds.concurrency, 'planned_pages': len(work), 'admitted_pages': admitted,
            'completed_requests': len(records), 'completed_correct_pages': correct,
            'unadmitted_work_ids': [item.work_id for item in work[admitted:]],
            'stop_reason': stopped, 'peak_in_flight': peak, 'elapsed_seconds': elapsed,
            'client_tasks_drained': True, 'backend_drain_verified': False,
            'first_admission_to_last_response_seconds': max(
                (r['response_finished_seconds'] for r in records if r['response_finished_seconds'] is not None), default=None),
            'completed_correct_pages_per_second': correct / elapsed if valid else None,
            'timing_scope': 'First admission through final local transport completion, validation and receipt callbacks; warmup excluded; no remote drain proof',
            'bounds': vars(bounds), 'work_sequence': [item.work_id for item in work],
            'workload_sha256': digest(json.dumps([(item.work_id, item.fixture_id, request_digests[id(item.request)],
                                                   expected_digests[id(item.expected_json)]) for item in work],
                                                 separators=(',', ':')).encode()),
            'records': sorted(records, key=lambda record: record['request_id'])}


def endpoint_validator(item, reply):
    # Local imports keep the core usable with injected validators and as a script.
    import sys
    directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, directory)
    try:
        from endpoint import validate, parse_sse
    finally:
        sys.path.remove(directory)
    payload = json.loads(item.request)
    if payload.get('stream'):
        result = parse_sse(reply.raw)
    else:
        data = json.loads(reply.raw)
        choice = data['choices'][0]
        result = {'content': choice['message']['content'], 'finish_reason': choice['finish_reason'],
                  'usage': data.get('usage')}
    return validate(json.loads(item.expected_json), result, max_tokens=payload['max_tokens'])


def corpus_work(corpus, repeat):
    import base64
    if type(repeat) is not int or not 1 <= repeat <= 10000:
        raise ValueError('Repeat must be 1 to 10000')
    corpus = Path(corpus).resolve()
    with (corpus / 'manifest.json').open('rb') as stream:
        manifest_bytes = stream.read(4 * 1024 * 1024 + 1)
    if len(manifest_bytes) > 4 * 1024 * 1024:
        raise ValueError('Corpus manifest exceeds 4 MiB')
    manifest = json.loads(manifest_bytes)
    fixtures = manifest['fixtures']
    if not fixtures or len(fixtures) > 64 or len(fixtures) * repeat > 10000 or len({f['id'] for f in fixtures}) != len(fixtures):
        raise ValueError('Invalid corpus IDs or bounded work count')
    required = ('table', 'numbers', 'text', 'end_markers', 'ordered_text', 'required_labels')
    for fixture in fixtures:
        expected = fixture.get('expected')
        if not isinstance(expected, dict) or any(not isinstance(expected.get(key), list) for key in required):
            raise ValueError('Unsupported expected schema: CLI requires existing endpoint inputs-v2 quality fields')
    prompt = Path(__file__).with_name('prompt.txt').read_bytes()
    if len(prompt) > 1024 * 1024:
        raise ValueError('Prompt exceeds 1 MiB')
    requests = {}
    expectations = {}
    for fixture in fixtures:
        path = (corpus / fixture['image']).resolve()
        if not path.is_relative_to(corpus):
            raise ValueError('Image path escapes corpus')
        with path.open('rb') as stream:
            image = stream.read(16 * 1024 * 1024 + 1)
        if len(image) > 16 * 1024 * 1024 or digest(image) != fixture['image_sha256']:
            raise ValueError('Corpus image exceeds byte bound or hash differs')
        payload = {'model': 'chandra', 'temperature': 0, 'top_p': .1, 'max_tokens': 12384, 'stream': False,
                   'messages': [{'role': 'user', 'content': [
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image).decode()}},
                       {'type': 'text', 'text': prompt.decode()}]}]}
        requests[fixture['id']] = json.dumps(payload, separators=(',', ':')).encode()
        expectations[fixture['id']] = json.dumps(fixture['expected'], sort_keys=True, separators=(',', ':'))
        if sum(len(value) for value in requests.values()) > 128 * 1024 * 1024:
            raise ValueError('Unique corpus request bytes exceed 128 MiB')
    work = tuple(Work(f'{rep:04d}:{f["id"]}', f['id'], requests[f['id']],
                      expectations[f['id']])
                 for rep in range(repeat) for f in fixtures)
    return work, {'corpus_manifest_sha256': digest(manifest_bytes), 'prompt_sha256': digest(prompt),
                  'repeat': repeat, 'fixture_order': [f['id'] for f in fixtures]}


async def replay_cli(args):
    url = urlsplit(args.direct_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Explicit credential-free direct backend HTTP URL required')
    if len(set(args.concurrency)) != len(args.concurrency):
        raise ValueError('Concurrency configurations must be unique')
    work, metadata = corpus_work(args.corpus, args.repeat)
    replay_path = Path(args.replay).resolve()
    with replay_path.open('rb') as stream:
        replay_manifest = stream.read(1024 * 1024 + 1)
    if len(replay_manifest) > 1024 * 1024:
        raise ValueError('Replay manifest exceeds 1 MiB')
    entries = json.loads(replay_manifest)
    if set(entries) != set(metadata['fixture_order']):
        raise ValueError('Replay must cover exactly the frozen corpus fixtures')
    replies = {}
    for fixture_id, entry in entries.items():
        delay = entry['delay_seconds']
        if isinstance(delay, bool) or not 0 <= delay <= 86400:
            raise ValueError('Replay delays must be bounded and nonnegative')
        path = (replay_path.parent / entry['response_file']).resolve()
        if not path.is_relative_to(replay_path.parent):
            raise ValueError('Replay response path escapes replay directory')
        with path.open('rb') as stream:
            raw = stream.read(args.max_response_bytes + 1)
        replies[fixture_id] = (delay, Reply(raw, entry.get('http_status', 200)))
        if sum(len(value[1].raw) for value in replies.values()) > 128 * 1024 * 1024:
            raise ValueError('Aggregate replay replies exceed 128 MiB')
    async def transport(item, request_id, deadline, response_limit):
        delay, reply = replies[item.fixture_id]
        await asyncio.sleep(delay)
        return reply
    # Exact replay raw writes are forecast before output creation, including warmup.
    raw_disk_limit = getattr(args, 'max_replay_raw_bytes', 512 * 1024 * 1024)
    if type(raw_disk_limit) is not int or not 1 <= raw_disk_limit <= 2 * 1024 * 1024 * 1024:
        raise ValueError('Replay raw disk limit must be positive and at most 2 GiB')
    planned_raw_bytes = sum(min(len(value[1].raw), args.max_response_bytes) for value in replies.values())
    planned_raw_bytes *= (args.repeat + args.warmup_cycles) * len(args.concurrency)
    if planned_raw_bytes > raw_disk_limit:
        raise ValueError('Planned replay raw writes exceed explicit disk bound')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    written = set()
    for item in work:
        if id(item.request) not in written:
            request_sha = digest(item.request)
            (output / (request_sha + '.request.json')).write_bytes(item.request)
            written.add(id(item.request))
    summary = {'schema': 'chandra-throughput-offline-replay-v1', 'mode': 'offline-replay',
               'direct_backend_url_metadata_only': args.direct_url, 'hardware_measurement': False,
               'live_adapter_available': False, 'metadata': metadata, 'source_sha256': digest(Path(__file__).read_bytes()),
               'replay_manifest_sha256': digest(replay_manifest), 'planned_concurrency': args.concurrency,
               'unattempted_concurrency': [], 'planned_raw_reply_bytes': planned_raw_bytes,
               'max_replay_raw_bytes': raw_disk_limit, 'configurations': []}
    def save():
        path = output / 'summary.json'
        temporary = path.with_suffix('.pending')
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(path)
    for config_index, concurrency in enumerate(args.concurrency):
        bounds = Bounds(concurrency, args.request_seconds, args.run_seconds, args.max_response_bytes)
        bounds.check()
        configuration = {'concurrency': concurrency, 'warmup': None, 'measured': None}
        summary['configurations'].append(configuration)
        save()
        def writer():
            def record(receipt, raw):
                prefix = output / receipt['request_id']
                prefix.with_suffix('.response.raw').write_bytes(raw)
                prefix.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            return record
        if args.warmup_cycles:
            warmup, _ = corpus_work(args.corpus, args.warmup_cycles)
            configuration['warmup'] = await measure(warmup, transport, endpoint_validator, bounds,
                request_prefix=f'c{config_index}-warmup', on_record=writer())
            save()
            if not configuration['warmup']['valid']:
                configuration['invalid_reason'] = 'warmup_failed; measurement not admitted'
                summary['stop_reason'] = 'warmup_failed; remaining configurations not admitted'
                summary['unattempted_concurrency'] = args.concurrency[config_index + 1:]
                save()
                break
        configuration['measured'] = await measure(work, transport, endpoint_validator, bounds,
            request_prefix=f'c{config_index}-measured', on_record=writer())
        if not configuration['measured']['valid']:
            summary['stop_reason'] = 'measured_cell_failed; remaining configurations not admitted'
            summary['unattempted_concurrency'] = args.concurrency[config_index + 1:]
            save()
            break
        save()
    return 0 if all(c['measured'] is not None and c['measured']['valid'] for c in summary['configurations']) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--replay', required=True, help='Offline JSON fixture map: response_file, delay_seconds, optional http_status')
    parser.add_argument('--direct-url', required=True, help='Single backend URL recorded only; never contacted')
    parser.add_argument('--corpus', required=True, help='Frozen endpoint inputs-v2 expected schema; qualification-v1 schemas rejected')
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrency', type=int, nargs='+', required=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--warmup-cycles', type=int, default=1)
    parser.add_argument('--request-seconds', type=float, default=900)
    parser.add_argument('--run-seconds', type=float, default=3600)
    parser.add_argument('--max-replay-raw-bytes', type=int, default=512 * 1024 * 1024,
                        help='Total forecast replay raw replies, including warmup/all cells; at most 2 GiB')
    parser.add_argument('--max-response-bytes', type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    if not 0 <= args.warmup_cycles <= 10:
        parser.error('Warmup cycles must be from 0 to 10')
    # Validate every bound before reading replay files or creating output.
    for concurrency in args.concurrency:
        Bounds(concurrency, args.request_seconds, args.run_seconds, args.max_response_bytes).check()
    return asyncio.run(replay_cli(args))

if __name__ == '__main__':
    raise SystemExit(main())
