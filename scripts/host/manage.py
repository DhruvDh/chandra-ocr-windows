"""Install and control task-owned Linux user services for a Windows worker and router."""
from __future__ import annotations

import argparse
import base64
import ipaddress
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import secrets
import socket
import threading
import time
import tempfile
import zlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local"
UNITS = ("chandra-waystone-worker.service", "chandra-router.service")
MARKER = "# Managed by chandra-ocr-windows/scripts/host/manage.py"
REMOTE_SUPERVISOR = (Path(__file__).with_name("windows_lease.py")).read_text()


def unit_quote(value):
    # systemd.exec quoting, with literal specifier '%' escaped separately.
    value = str(value)
    if any(c in value for c in "\r\n\0"):
        raise ValueError("Control characters are not valid service arguments")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def command(arguments):
    return " ".join(unit_quote(value) for value in arguments)


def directory_value(value):
    # WorkingDirectory takes one path value; ExecStart's token quotes are literal here.
    value = str(value)
    if not value.startswith("/") or any(c in value for c in "\r\n\0"):
        raise ValueError("WorkingDirectory must be an absolute path without control characters")
    return value.replace("\\", "\\\\").replace("%", "%%")


def ps_quote(value):
    if any(c in value for c in "\r\n\0"):
        raise ValueError("Control characters are not valid remote paths")
    return "'" + value.replace("'", "''") + "'"


def remote_command(args, lease_token):
    argv = [args.remote_python, "-m", "chandra_service", "waystone", "--model-path", args.model_path,
            "--host", "127.0.0.1", "--port", str(args.worker_port), "--max-input-tokens", "16384",
            "--max-output-tokens", "12384", "--attention", "hybrid"]
    settings = base64.b64encode(json.dumps({"argv": argv, "cwd": args.remote_root, "lease_port": args.lease_port, "lease_token": lease_token}).encode()).decode()
    source = base64.b64encode(zlib.compress(REMOTE_SUPERVISOR.encode())).decode()
    # Invoke native Python directly: PowerShell encoded commands exceed the
    # Windows SSH shell command-length limit once the supervisor is embedded.
    # Paths inside settings are base64 JSON; the sole shell-visible path is quoted.
    if any(character in args.remote_python for character in '\r\n\0"%!'):
        raise ValueError("Remote Python path contains unsupported shell characters")
    code = "import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(source) + ")))"
    remote = '"' + args.remote_python + '" -c "' + code + '" ' + settings
    if len(remote) > 7500:
        raise ValueError("Encoded remote ownership command exceeds the bounded SSH command length")
    return ["/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{args.worker_port}:127.0.0.1:{args.worker_port}",
            "-L", f"127.0.0.1:{args.lease_port}:127.0.0.1:{args.lease_port}",
            args.ssh_host, remote]


def read_exact(connection, count):
    data = bytearray()
    while len(data) < count:
        chunk = connection.recv(count - len(data))
        if not chunk:
            raise OSError("Lease closed before ownership acknowledgement")
        data.extend(chunk)
    return bytes(data)


def run_channel(argv, lease_port, lease_token, stop=None):
    """Socket EOF or a 15s missing heartbeat ends the remote owned job."""
    stop = stop or threading.Event()
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    connection = None
    try:
        deadline = time.monotonic() + 15
        while process.poll() is None and not stop.is_set():
            if connection is None:
                try:
                    connection = socket.create_connection(("127.0.0.1", lease_port), timeout=1)
                    connection.sendall(lease_token.encode() + b"\n")
                    if read_exact(connection, 6) != b"owned\n":
                        raise OSError("Lease ownership acknowledgement missing")
                except OSError:
                    if connection is not None:
                        connection.close()
                        connection = None
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Owned Windows lease did not become available within 15s")
                    stop.wait(.2)
                    continue
            try:
                connection.sendall(b"alive\n")
            except OSError:
                break
            stop.wait(2)
    finally:
        if connection is not None:
            connection.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
    return process.returncode


def private_host(value):
    address = ipaddress.ip_address(value)
    if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
        raise ValueError("Choose a concrete private or loopback address")
    return value


def systemctl(*arguments, check=True):
    return subprocess.run(["systemctl", "--user", *arguments], check=check)


def check_unit_ownership(unit_dir, name):
    path = unit_dir / name
    if path.is_symlink():
        raise ValueError(f"Unit path is a symlink; preserve its target: {path}")
    if path.exists() and not path.read_text().startswith(MARKER):
        raise ValueError(f"Existing unmanaged unit must be preserved: {path}")
    result = subprocess.run(["systemctl", "--user", "show", "--property=LoadState,FragmentPath,DropInPaths", name],
                            check=True, stdout=subprocess.PIPE, text=True)
    properties = dict(line.split("=", 1) for line in (result.stdout or "").splitlines() if "=" in line)
    fragment = properties.get("FragmentPath", "")
    if properties.get("DropInPaths"):
        raise ValueError(f"Unit has external drop-ins; preserve its effective configuration: {name}")
    if fragment and Path(fragment).resolve() != path.resolve():
        raise ValueError(f"Effective unit belongs to another location: {name}")
    if properties.get("LoadState") == "loaded" and (not fragment or not path.exists()):
        raise ValueError(f"Loaded unit has no owned local source: {name}")
    if path.exists():
        receipt_path = LOCAL / "installation.json"
        if not receipt_path.exists():
            raise ValueError(f"Owned unit requires an installation receipt: {name}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("source_root") != str(ROOT) or receipt.get("units") != list(UNITS):
            raise ValueError(f"Unit receipt belongs to another owner: {name}")
        expected = receipt.get("unit_sha256", {}).get(name)
        if expected != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Unit differs from its installation receipt; preserve and review it: {name}")
    return path.exists()


def install(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.ssh_host):
        raise ValueError("Use a configured SSH host alias")
    if not all(1024 <= port <= 65535 for port in (args.worker_port, args.router_port, args.lease_port)) or len({args.worker_port, args.router_port, args.lease_port}) != 3:
        raise ValueError("Choose distinct unprivileged worker, lease, and router ports")
    hosts = ["127.0.0.1"]
    if args.lan_address and private_host(args.lan_address) not in hosts:
        hosts.append(args.lan_address)
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("Install the local service environment with uv sync --locked first")
    unit_dir = Path.home() / ".config/systemd/user"
    for name in UNITS:
        check_unit_ownership(unit_dir, name)
    if LOCAL.is_symlink() or any((LOCAL / name).is_symlink() for name in ("router.json", "channel.json", "installation.json")):
        raise ValueError("Managed local configuration paths must not be symlinks")
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    config = {"backends": {"northstone": {"url": args.northstone_url, "capacity": 2, "queue_limit": 0, "expected_page_seconds": args.northstone_page_seconds},
                            "waystone": {"url": f"http://127.0.0.1:{args.worker_port}", "capacity": 1, "queue_limit": 2, "expected_page_seconds": args.waystone_page_seconds}},
              "cooldown_seconds": 15, "inference_seconds": 1800}
    sys.path.insert(0, str(ROOT))
    from chandra_service.router import Endpoint
    for name, settings in config["backends"].items():
        Endpoint(name, **settings)
    config_path = LOCAL / "router.json"
    channel_path = LOCAL / "channel.json"
    receipt_path = LOCAL / "installation.json"
    if any(path.exists() for path in (config_path, channel_path, receipt_path)):
        if not receipt_path.exists():
            raise ValueError("Existing local configuration has no ownership receipt; preserve it")
        previous = json.loads(receipt_path.read_text())
        if previous.get("source_root") != str(ROOT) or previous.get("units") != list(UNITS):
            raise ValueError("Existing installation receipt belongs to another owner")
    base = MARKER + "\n[Unit]\nDescription={description}\nAfter=network-online.target\n\n[Service]\nType=simple\nWorkingDirectory=" + directory_value(ROOT) + "\nExecStart={exec_start}\nRestart=on-failure\nRestartSec=15\nTimeoutStopSec=35\nKillMode=control-group\n\n[Install]\nWantedBy=default.target\n"
    router_args = [python, "-m", "chandra_service", "router", "--config", config_path, "--port", args.router_port]
    for host in hosts:
        router_args += ["--host", host]
    channel_args = [python, ROOT / "scripts/host/manage.py", "_channel", "--config", channel_path]
    content = [base.format(description="Chandra Windows worker over an owned SSH lease", exec_start=command(channel_args)),
               base.format(description="Chandra OCR router for ROCm and Windows Intel", exec_start=command(router_args))]
    receipt = {"schema_version": 2, "source_root": str(ROOT), "units": list(UNITS), "hosts": hosts,
               "router_port": args.router_port, "worker_port": args.worker_port, "ssh_host": args.ssh_host,
               "remote_root": args.remote_root, "remote_python": args.remote_python, "model_path": args.model_path,
               "lease_port": args.lease_port, "remote_ownership": "unnamed Windows Job Object; authenticated loopback TCP lease; kill on job close",
               "unit_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in zip(UNITS, content)}}
    lease_token = secrets.token_urlsafe(32)
    files = {config_path: json.dumps(config, indent=2) + "\n",
             channel_path: json.dumps({"ssh_argv": remote_command(args, lease_token), "lease_port": args.lease_port, "lease_token": lease_token}, indent=2) + "\n",
             receipt_path: json.dumps(receipt, indent=2) + "\n"}
    files.update({unit_dir / name: text for name, text in zip(UNITS, content)})
    # Verify staged units before changing installed files or running services.
    with tempfile.TemporaryDirectory(prefix="chandra-unit-stage-", dir=LOCAL) as stage:
        staged = []
        for name, text in zip(UNITS, content):
            path = Path(stage) / name
            path.write_text(text)
            staged.append(str(path))
        subprocess.run(["systemd-analyze", "--user", "verify", *staged], check=True)
    previous_files = {path: (path.read_bytes(), path.stat().st_mode & 0o777) if path.exists() else None for path in files}
    def state(action, name):
        result = subprocess.run(["systemctl", "--user", action, name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return result.returncode == 0
    old_active = {name: state("is-active", name) for name in UNITS}
    old_enabled = {name: state("is-enabled", name) for name in UNITS}
    unit_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Replacement must stop the old lease before acquiring a new remote job.
        existing_units = [name for name in UNITS if previous_files[unit_dir / name] is not None]
        if existing_units:
            systemctl("stop", *existing_units)
        for path, text in files.items():
            path.write_text(text)
            path.chmod(0o600 if path.parent == LOCAL else 0o644)
        systemctl("daemon-reload")
        systemctl("enable", *UNITS)
        systemctl("start", *UNITS)
    except BaseException:
        systemctl("stop", *UNITS, check=False)
        systemctl("disable", *UNITS, check=False)
        for path, previous in previous_files.items():
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous[0])
                path.chmod(previous[1])
        systemctl("daemon-reload", check=False)
        for name in UNITS:
            systemctl("enable" if old_enabled[name] else "disable", name, check=False)
            if old_active[name]:
                systemctl("start", name, check=False)
        raise
    print("Installed user services. Check status and /health; installation is not model acceptance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    install_parser = sub.add_parser("install")
    install_parser.add_argument("--ssh-host", default="waystone")
    install_parser.add_argument("--remote-root", required=True)
    install_parser.add_argument("--remote-python", required=True)
    install_parser.add_argument("--model-path", required=True)
    install_parser.add_argument("--northstone-url", default="http://127.0.0.1:8000")
    install_parser.add_argument("--worker-port", type=int, default=18001)
    install_parser.add_argument("--lease-port", type=int, default=18002)
    install_parser.add_argument("--router-port", type=int, default=8002)
    install_parser.add_argument("--lan-address")
    install_parser.add_argument("--northstone-page-seconds", type=float, default=1)
    install_parser.add_argument("--waystone-page-seconds", type=float, default=1)
    channel_parser = sub.add_parser("_channel", help=argparse.SUPPRESS)
    channel_parser.add_argument("--config", type=Path, required=True)
    for action in ("status", "start", "stop", "restart", "uninstall", "logs"):
        sub.add_parser(action)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This manager runs on the Linux router host; the inference worker remains native Windows")
    if args.action == "_channel":
        stop = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())
        settings = json.loads(args.config.read_text())
        return_code = run_channel(settings["ssh_argv"], settings["lease_port"], settings["lease_token"], stop)
        raise SystemExit(0 if stop.is_set() else return_code)
    if args.action == "install":
        install(args)
    elif args.action == "logs":
        subprocess.run(["journalctl", "--user", "-u", UNITS[0], "-u", UNITS[1], "-n", "100", "--no-pager"], check=True)
    elif args.action == "uninstall":
        directory = Path.home() / ".config/systemd/user"
        names = [name for name in UNITS if check_unit_ownership(directory, name)]
        if names:
            systemctl("stop", *names)
            systemctl("disable", *names)
        for name in names:
            (directory / name).unlink()
        systemctl("daemon-reload")
        print("Removed task-owned user services. Models, environments, inputs and results were retained.")
    else:
        if args.action in {"start", "stop", "restart"}:
            directory = Path.home() / ".config/systemd/user"
            for name in UNITS:
                if not check_unit_ownership(directory, name):
                    raise ValueError(f"No owned installed unit: {name}")
        systemctl(args.action, *UNITS, *( ["--no-pager"] if args.action == "status" else []), check=args.action != "status")


if __name__ == "__main__":
    main()
