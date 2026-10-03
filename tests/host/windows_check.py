"""Explicit harmless Windows Job Object acceptance; no model or service is imported."""
import argparse
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import socket
import secrets
import time
import sys
import tempfile
import zlib

SPEC = importlib.util.spec_from_file_location('host_manage', Path(__file__).resolve().parents[2] / 'scripts/host/manage.py')
manage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage)


def encoded_command(script, host):
    return ['ssh', '-T', '-o', 'BatchMode=yes', host, 'pwsh', '-NoProfile', '-EncodedCommand', base64.b64encode(script.encode('utf-16le')).decode()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    reports = []
    for mode in ('eof', 'timeout', 'channel_stop'):
        fake = "import json,os,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); print(json.dumps({'parent':os.getpid(),'child':p.pid}),flush=True); time.sleep(60)"
        lease_token = secrets.token_urlsafe(32)
        remote_port = 18002
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            local_port = reservation.getsockname()[1]
        settings = base64.b64encode(json.dumps({'argv': [args.python, '-c', fake], 'cwd': args.root, 'lease_seconds': 5 if mode == 'channel_stop' else 1, 'lease_port': remote_port, 'lease_token': lease_token}).encode()).decode()
        source = base64.b64encode(zlib.compress(manage.REMOTE_SUPERVISOR.encode())).decode()
        code = 'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(source) + ')))'
        script = '& ' + manage.ps_quote(args.python) + ' -c ' + manage.ps_quote(code) + ' ' + manage.ps_quote(settings) + '; exit $LASTEXITCODE'
        ssh_argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-L', f'127.0.0.1:{local_port}:127.0.0.1:{remote_port}', args.host, '"' + args.python + '" -c "' + code + '" ' + settings]
        temporary = tempfile.TemporaryDirectory(prefix='chandra-owned-channel-')
        if mode == 'channel_stop':
            channel_path = Path(temporary.name) / 'channel.json'
            channel_path.write_text(json.dumps({'ssh_argv': ssh_argv, 'lease_port': local_port, 'lease_token': lease_token}))
            argv = [sys.executable, str(Path(manage.__file__)), '_channel', '--config', str(channel_path)]
        else:
            argv = ssh_argv
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        connection = None
        try:
            # A harmless worker always exits under its job even if this SSH caller dies.
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('Windows supervisor failed: ' + process.stderr.read()[-2048:])
            pids = json.loads(line)
            if mode == 'channel_stop':
                time.sleep(.5)
                started = time.monotonic()
                process.terminate()
            else:
                connection = socket.create_connection(('127.0.0.1', local_port), timeout=2)
                connection.sendall(lease_token.encode() + b'\n')
                if manage.read_exact(connection, 6) != b'owned\n': raise RuntimeError('Lease acknowledgement failed')
                connection.sendall(b'alive\n')
                started = time.monotonic()
            if mode == 'eof':
                time.sleep(.25)
                if process.poll() is not None: raise RuntimeError('Lease ended while heartbeat socket remained healthy')
                connection.close(); connection = None
            elif mode == 'timeout':
                time.sleep(1.3)
                connection.close(); connection = None
            result = process.wait(timeout=5)
            elapsed = time.monotonic() - started
            probe = '$ids=@(' + str(pids['parent']) + ',' + str(pids['child']) + '); @($ids | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count'
            remaining = subprocess.run(encoded_command(probe, args.host), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5, check=True)
            if remaining.stdout.strip() != '0':
                raise RuntimeError('Owned worker or descendant remained: ' + remaining.stdout)
            diagnostic = process.stderr.read()
            expected = 'heartbeat deadline' if mode == 'timeout' else 'lease EOF'
            if expected not in diagnostic: raise RuntimeError('Wrong lease end reason: ' + diagnostic)
            if mode == 'timeout' and elapsed < .9: raise RuntimeError('Lease did not survive until heartbeat deadline')
            reports.append({'case': mode, 'owned_pids': pids, 'exit_code': result, 'seconds_after_last_heartbeat': round(elapsed, 3), 'remaining_owned_processes': 0, 'lease_end_reason': expected})
        finally:
            if connection is not None: connection.close()
            if process.poll() is None:
                process.kill(); process.wait(timeout=3)
            if not process.stdin.closed: process.stdin.close()
            process.stdout.close(); process.stderr.close()
            temporary.cleanup()
    print(json.dumps({'no_gpu_or_model_import': True, 'cases': reports}, indent=2))


if __name__ == '__main__': main()
