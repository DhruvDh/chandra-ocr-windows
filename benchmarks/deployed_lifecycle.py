"""Explicit installed-service acceptance: overflow, owned worker crash, and real idle unload.

This runner never retries inference POSTs or starts/stops services. The crash phase
kills only the receipt-verified worker unit's main process; systemd owns recovery.
"""
import argparse
import asyncio
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import httpx

from live_integration import Harness, occupancy, sha
from evaluate import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
WORKER_UNIT = 'chandra-waystone-worker.service'


def complete_on(report, backend):
    return report.get('correctness', {}).get('passed') is True and report.get('response_headers', {}).get('x-chandra-backend') == backend


def unloaded(health):
    memory = health.get('torch_xpu_memory') or {}
    return (health.get('state') == 'cold' and health.get('loaded') is False
            and occupancy(health, 'waystone') == 0
            and type(memory.get('allocated_bytes')) is int and memory['allocated_bytes'] == 0
            and type(memory.get('reserved_bytes')) is int and memory['reserved_bytes'] == 0)


class Lifecycle:
    def __init__(self, args, client, report):
        self.args = args
        self.client = client
        self.report = report
        self.output = args.output

    def save(self):
        # Every observation survives interruption; raw inference evidence is written by Harness.
        destination = self.output / 'lifecycle.json'
        temporary = destination.with_suffix('.pending')
        temporary.write_text(json.dumps(self.report, indent=2) + '\n')
        temporary.replace(destination)

    def harness(self, stage, backend):
        output = self.output / stage / backend
        output.mkdir(parents=True, exist_ok=True)
        harness = Harness(self.client, self.args.base_url, self.args.corpus, output, backend,
                          self.args.request_seconds, self.args.drain_seconds)
        harness.is_router = True
        return harness

    async def worker_health(self):
        response = await self.client.get(self.args.worker_base_url.rstrip('/').removesuffix('/v1') + '/health', timeout=5)
        response.raise_for_status()
        return response.json()

    async def drain(self, harness, receipt):
        receipt['drain'] = await harness.idle()
        self.save()
        if not receipt['drain']['passed']:
            raise ValueError('Backend ownership did not drain before the next phase')

    async def overflow(self):
        receipt = {'passed': False, 'observations': [], 'runs': []}
        self.report['phases']['overflow'] = receipt
        self.save()
        north = self.harness('overflow', 'northstone')
        auto = self.harness('overflow', 'auto')
        tasks = [asyncio.create_task(north.request(self.args.overflow_fixture, True)) for _ in range(2)]
        try:
            deadline = time.monotonic() + min(30, self.args.request_seconds)
            while time.monotonic() < deadline:
                health = await auto.health()
                receipt['observations'].append({'seconds': time.monotonic(), 'health': health})
                self.save()
                entry = health.get('backends', {}).get('northstone', {})
                if entry.get('active') == 2 and all(not task.done() for task in tasks):
                    receipt['observed_northstone_active_two'] = True
                    break
                if any(task.done() for task in tasks):
                    raise ValueError('NorthStone requests completed before real overlap was established')
                await asyncio.sleep(.1)
            else:
                raise ValueError('NorthStone active=2 was not observed within the overlap deadline')
            third = asyncio.create_task(auto.request('tiny', True))
            tasks.append(third)
            runs = await asyncio.gather(*tasks)
            receipt['runs'] = runs
            receipt['passed'] = complete_on(runs[0], 'northstone') and complete_on(runs[1], 'northstone') and complete_on(runs[2], 'waystone')
            self.save()
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.drain(auto, receipt)
        return receipt

    async def control(self, *arguments):
        process = await asyncio.create_subprocess_exec('systemctl', '--user', *arguments,
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode:
            raise RuntimeError('Bounded systemctl operation failed: ' + stderr.decode(errors='replace')[:512])
        return stdout.decode().strip()

    async def restart_count(self):
        value = await self.control('show', '--property=NRestarts', '--value', WORKER_UNIT)
        if not value.isdecimal(): raise ValueError('Worker restart count was not observable')
        return int(value)

    async def verify_owned_unit(self):
        spec = importlib.util.spec_from_file_location('deployed_host_manage', ROOT / 'scripts/host/manage.py')
        manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manager)
        directory = Path.home() / '.config/systemd/user'
        owned = await asyncio.wait_for(asyncio.to_thread(manager.check_unit_ownership, directory, WORKER_UNIT), 5)
        if not owned: raise ValueError('No receipt-verified installed worker unit; crash mutation refused')
        return {'unit': WORKER_UNIT, 'receipt_sha256': sha((manager.LOCAL / 'installation.json').read_bytes()),
                'unit_sha256': sha((directory / WORKER_UNIT).read_bytes())}

    async def crash(self):
        receipt = {'passed': False, 'observations': [], 'runner_post_retries': 0}
        self.report['phases']['crash'] = receipt
        self.save()
        worker = self.harness('crash', 'waystone')
        auto = self.harness('crash', 'auto')
        first_content = asyncio.Event()
        task = asyncio.create_task(worker.request('tiny', True, first_content=first_content))
        try:
            await asyncio.wait_for(first_content.wait(), self.args.request_seconds)
            health = await worker.health()
            if task.done() or occupancy(health, 'waystone') != 1:
                raise ValueError('No active stream after actual content; owned worker crash refused')
            receipt['pre_crash_health'] = health
            receipt['ownership'] = await self.verify_owned_unit()
            receipt['restarts_before'] = before = await self.restart_count()
            if task.done() or occupancy(await worker.health(), 'waystone') != 1:
                raise ValueError('Stream ended during ownership checks; worker crash refused')
            self.save()
            receipt['kill_command'] = ['systemctl', '--user', 'kill', '--kill-whom=main', '--signal=SIGKILL', WORKER_UNIT]
            await self.control(*receipt['kill_command'][2:])
            receipt['kill_submitted'] = True
            deadline = time.monotonic() + 90
            self.save()
            run = await asyncio.wait_for(task, max(.01, deadline - time.monotonic()))
            receipt['interrupted_run'] = run
            raw = (worker.output / run['raw_file']).read_bytes()
            receipt['incomplete_stream_verified'] = (run['ttft_seconds'] is not None and b'[DONE]' not in raw
                                                    and not run['correctness']['passed'])
            while time.monotonic() < deadline:
                count = await self.restart_count()
                if count > before + 1: raise ValueError('Worker restarted more than once after the deliberate crash')
                try:
                    remote = await self.worker_health()
                    health = await auto.health()
                    observation = {'restarts': count, 'worker_health': remote, 'router_health': health}
                    entry = health.get('backends', {}).get('waystone', {})
                    ready = (count == before + 1 and occupancy(remote, 'waystone') == 0
                             and remote.get('state') in {'cold', 'ready'} and occupancy(health, 'waystone') == 0
                             and entry.get('state') in {'cold', 'ready', 'loaded', 'unloaded'}
                             and entry.get('uncertain') is False)
                except (httpx.HTTPError, ValueError) as exc:
                    observation = {'restarts': count, 'error': type(exc).__name__}
                    ready = False
                receipt['observations'].append(observation)
                self.save()
                if ready: break
                await asyncio.sleep(1)
            else:
                raise ValueError('Automatic worker restart and observable idle health exceeded 90s')
            receipt['restarts_after'] = await self.restart_count()
            recovery = await worker.request('tiny', True)
            receipt['recovery'] = recovery
            receipt['restarts_after_recovery'] = await self.restart_count()
            receipt['passed'] = (receipt['incomplete_stream_verified'] and receipt['restarts_after'] == before + 1
                                 and receipt['restarts_after_recovery'] == before + 1 and complete_on(recovery, 'waystone'))
            self.save()
        finally:
            if not task.done(): task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.drain(auto, receipt)
        return receipt

    async def idle(self):
        receipt = {'passed': False, 'cycles': []}
        self.report['phases']['idle'] = receipt
        self.save()
        worker = self.harness('idle', 'waystone')
        auto = self.harness('idle', 'auto')
        for index in range(2):
            cycle = {'index': index + 1, 'passed': False, 'observations': []}
            receipt['cycles'].append(cycle)
            self.save()
            cycle['run'] = await worker.request('tiny', True)
            self.save()
            if not complete_on(cycle['run'], 'waystone'):
                await self.drain(auto, cycle)
                raise ValueError('Idle-cycle boundary request did not complete correctly')
            boundary = time.monotonic()
            cycle['boundary'] = boundary
            while time.monotonic() - boundary <= 330:
                health = await self.worker_health()
                models = await self.client.get(self.args.worker_base_url.rstrip('/') + '/models', timeout=5)
                models.raise_for_status()
                elapsed = time.monotonic() - boundary
                cycle['observations'].append({'elapsed_seconds': elapsed, 'health': health, 'models': models.json()})
                self.save()
                if unloaded(health):
                    cycle['unload_seconds'] = elapsed
                    cycle['passed'] = 295 <= elapsed <= 315
                    self.save()
                    break
                if occupancy(health, 'waystone') != 0:
                    raise ValueError('Unexpected active work during passive idle acceptance')
                await asyncio.sleep(5)
            else:
                raise ValueError('Real 300s idle unload exceeded the 330s observation bound')
            if not cycle['passed']: raise ValueError('Idle unload occurred outside the 295..315s timing window')
        receipt['passed'] = all(cycle['passed'] for cycle in receipt['cycles'])
        receipt['final_worker_health'] = await self.worker_health()
        receipt['passed'] = receipt['passed'] and unloaded(receipt['final_worker_health'])
        self.save()
        return receipt


async def execute(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.corpus = args.corpus.resolve()
    report = {'schema': 'chandra-deployed-lifecycle-v1', 'base_url': args.base_url,
              'worker_base_url': args.worker_base_url, 'phases': {}, 'passed': False,
              'runner_post_retries': 0, 'provenance_verified': False, 'budget_seconds': args.budget_seconds,
              'reserved_final_drain_seconds': args.drain_seconds,
              'provenance': None, 'corpus_sha256': None}
    started = time.monotonic()
    runner = None
    try:
        report['provenance'] = json.loads(args.provenance.read_text())
        report['corpus_sha256'] = sha((args.corpus / 'manifest.json').read_bytes())
        if not verify_manifest(args.corpus / 'manifest.json')['passed']: raise ValueError('Corpus commitment mismatch')
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            runner = Lifecycle(args, client, report)
            runner.save()
            try:
                async with asyncio.timeout(args.budget_seconds - args.drain_seconds):
                    harness = runner.harness('preflight', 'auto')
                    report['initial_router_health'] = health = await harness.health()
                    report['initial_worker_health'] = await runner.worker_health()
                    runner.save()
                    if 'backends' not in health or occupancy(health, 'auto') != 0:
                        raise ValueError('Installed router is busy or capacity is unobservable; no POST submitted')
                    for phase in args.phases:
                        method = getattr(runner, phase)
                        value = await method()
                        runner.save()
                        if not value['passed']: break
                    report['passed'] = len(report['phases']) == len(args.phases) and all(value['passed'] for value in report['phases'].values())
            finally:
                # Reserve the configured drain allowance inside the total budget.
                # Cleanup makes no POST and cannot free backend-owned generation.
                cleanup = runner.harness('final-drain', 'auto')
                try:
                    report['final_drain'] = await cleanup.idle()
                    if not report['final_drain']['passed']: report['passed'] = False
                except Exception as exc:
                    report['final_drain'] = {'passed': False, 'error': type(exc).__name__}
                    report['passed'] = False
                runner.save()
    except (Exception, asyncio.CancelledError) as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)[:512]
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if runner: runner.save()
        else: (args.output / 'lifecycle.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--worker-base-url', required=True)
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('inputs-v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--phases', nargs='+', choices=['overflow', 'crash', 'idle'], default=['overflow', 'crash', 'idle'])
    parser.add_argument('--overflow-fixture', default='representative')
    parser.add_argument('--request-seconds', type=float, default=900)
    parser.add_argument('--drain-seconds', type=float, default=120)
    parser.add_argument('--budget-seconds', type=float, default=1800)
    args = parser.parse_args()
    for url in (args.base_url, args.worker_base_url):
        if not url.startswith(('http://', 'https://')) or not url.rstrip('/').endswith('/v1'):
            parser.error('Explicit HTTP(S) /v1 URLs are required')
    if len(set(args.phases)) != len(args.phases): parser.error('Phases must be unique')
    if any(not math.isfinite(value) or value <= 0 for value in (args.request_seconds, args.drain_seconds, args.budget_seconds)):
        parser.error('Deadlines must be positive finite numbers')
    if args.budget_seconds <= args.drain_seconds: parser.error('Total budget must exceed the reserved final drain allowance')
    result = asyncio.run(execute(args))
    print(json.dumps({'passed': result['passed'], 'phases': list(result['phases']), 'error': result.get('error')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
