"""Real sockets verify disconnect/drain, idle lifecycle, and process-crash no replay."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name('tcp_fixture.py')


def payload():
    return {'model': 'chandra', 'stream': True, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'ocr_layout'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}},
    ]}]}


@unittest.skipUnless(os.name == "posix", "POSIX inherited-socket subprocess fixture")
class TcpLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_tcp_cancel_idle_and_crash_without_replay(self):
        processes = []
        logs = []
        with tempfile.TemporaryDirectory(prefix='chandra-tcp-test-') as directory:
            events_path = Path(directory) / 'events.jsonl'

            def events():
                return [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []

            def spawn(mode, *extra):
                with socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    listener.listen(16)
                    port = listener.getsockname()[1]
                    log = open(Path(directory) / (mode + '.log'), 'wb')
                    logs.append(log)
                    env = dict(os.environ, PYTHONPATH=str(ROOT))
                    process = subprocess.Popen([sys.executable, str(FIXTURE), mode, '--fd', str(listener.fileno()), *extra],
                                               cwd=ROOT, env=env, pass_fds=(listener.fileno(),), stdout=log, stderr=log)
                    processes.append(process)
                    return process, f'http://127.0.0.1:{port}'

            async def eventually(predicate, timeout=3):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if await predicate():
                        return
                    await asyncio.sleep(.025)
                self.fail('TCP lifecycle condition exceeded bounded deadline')

            try:
                backend, backend_url = spawn('backend', '--events', str(events_path))
                router, router_url = spawn('router', '--upstream', backend_url)
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    async def ready():
                        self.assertIsNone(backend.poll(), 'fake backend exited during startup')
                        self.assertIsNone(router.poll(), 'router exited during startup')
                        try:
                            return (await client.get(router_url + '/health')).json()['state'] == 'ready'
                        except httpx.HTTPError:
                            return False
                    await eventually(ready)

                    # Close the actual downstream socket while upstream generation runs.
                    async with client.stream('POST', router_url + '/v1/chat/completions', json=payload()) as response:
                        self.assertEqual(response.status_code, 200)
                        async for line in response.aiter_lines():
                            if '"content":"first"' in line:
                                break
                    health = (await client.get(router_url + '/health')).json()
                    self.assertEqual(health['backends']['waystone']['active'], 1)
                    rejected = await client.post(router_url + '/v1/chat/completions', json=payload())
                    self.assertEqual(rejected.status_code, 503)
                    self.assertFalse(any(e['event'] == 'unload' for e in events()))

                    # Poll status/models throughout idle: neither may postpone unload.
                    async def idle_unloaded():
                        await client.get(backend_url + '/health')
                        await client.get(backend_url + '/v1/models')
                        await client.get(router_url + '/health')
                        return any(e['event'] == 'unload' for e in events())
                    await eventually(idle_unloaded)
                    recorded = events()
                    self.assertEqual(sum(e['event'] == 'start' for e in recorded), 1)
                    finish = next(e['time'] for e in recorded if e['event'] == 'finish')
                    unload = next(e['time'] for e in recorded if e['event'] == 'unload')
                    self.assertGreaterEqual(unload - finish, .18)
                    self.assertEqual((await client.get(router_url + '/health')).json()['backends']['waystone']['active'], 0)

                    # Crash after one submitted POST: router emits bounded failure and never replays.
                    async with client.stream('POST', router_url + '/v1/chat/completions', json=payload()) as response:
                        lines = response.aiter_lines()
                        async for line in lines:
                            if '"content":"first"' in line:
                                break
                        backend.kill()
                        await asyncio.to_thread(backend.wait, 2)
                        remaining = '\n'.join([line async for line in lines])
                    self.assertIn('Request was not replayed', remaining)
                    self.assertLess(len(remaining), 1024)
                    self.assertEqual(sum(e['event'] == 'start' for e in events()), 2)
                    rejected = await client.post(router_url + '/v1/chat/completions', json=payload())
                    self.assertEqual(rejected.status_code, 503)
                    self.assertEqual(sum(e['event'] == 'start' for e in events()), 2)
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, 2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait, 2)
                for log in logs:
                    log.close()


if __name__ == '__main__':
    unittest.main()
