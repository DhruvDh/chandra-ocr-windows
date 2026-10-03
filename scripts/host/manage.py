"""Install and control task-owned Linux user services for a Windows worker and router."""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local"
UNITS = ("chandra-waystone-worker.service", "chandra-router.service")
MARKER = "# Managed by chandra-ocr-windows/scripts/host/manage.py"


def unit_quote(value):
    # systemd.exec quoting, with literal specifier '%' escaped separately.
    value = str(value)
    if any(c in value for c in "\r\n\0"):
        raise ValueError("Control characters are not valid service arguments")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def command(arguments):
    return " ".join(unit_quote(value) for value in arguments)


def ps_quote(value):
    if any(c in value for c in "\r\n\0"):
        raise ValueError("Control characters are not valid remote paths")
    return "'" + value.replace("'", "''") + "'"


def remote_command(args):
    script = "$ErrorActionPreference='Stop'; Set-Location -LiteralPath " + ps_quote(args.remote_root) + "; "
    script += "$env:PYTHONPATH=" + ps_quote(args.remote_root) + "; & " + ps_quote(args.remote_python)
    script += " -m chandra_service waystone --model-path " + ps_quote(args.model_path)
    script += f" --host 127.0.0.1 --port {args.worker_port} --max-input-tokens 16384 --max-output-tokens 12384; exit $LASTEXITCODE"
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return ["/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{args.worker_port}:127.0.0.1:{args.worker_port}",
            args.ssh_host, "pwsh", "-NoProfile", "-EncodedCommand", encoded]


def private_host(value):
    address = ipaddress.ip_address(value)
    if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
        raise ValueError("Choose a concrete private or loopback address")
    return value


def systemctl(*arguments, check=True):
    return subprocess.run(["systemctl", "--user", *arguments], check=check)


def install(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.ssh_host):
        raise ValueError("Use a configured SSH host alias")
    if not all(1024 <= port <= 65535 for port in (args.worker_port, args.router_port)) or args.worker_port == args.router_port:
        raise ValueError("Choose distinct unprivileged worker and router ports")
    hosts = ["127.0.0.1"]
    if args.lan_address and private_host(args.lan_address) not in hosts:
        hosts.append(args.lan_address)
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("Install the local service environment with uv sync --locked first")
    unit_dir = Path.home() / ".config/systemd/user"
    for name in UNITS:
        path = unit_dir / name
        if path.exists() and not path.read_text().startswith(MARKER):
            raise ValueError(f"Existing unmanaged unit must be preserved: {path}")
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    config = {"backends": {"northstone": {"url": args.northstone_url, "capacity": 2, "expected_page_seconds": args.northstone_page_seconds},
                            "waystone": {"url": f"http://127.0.0.1:{args.worker_port}", "capacity": 1, "expected_page_seconds": args.waystone_page_seconds}},
              "cooldown_seconds": 15, "inference_seconds": 1800}
    sys.path.insert(0, str(ROOT))
    from chandra_service.router import Endpoint
    for name, settings in config["backends"].items():
        Endpoint(name, **settings)
    config_path = LOCAL / "router.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    config_path.chmod(0o600)
    base = MARKER + "\n[Unit]\nDescription={description}\nAfter=network-online.target\n\n[Service]\nType=simple\nWorkingDirectory=" + unit_quote(ROOT) + "\nExecStart={exec_start}\nRestart=on-failure\nRestartSec=15\nTimeoutStopSec=35\nKillMode=control-group\n\n[Install]\nWantedBy=default.target\n"
    router_args = [python, "-m", "chandra_service", "router", "--config", config_path, "--port", args.router_port]
    for host in hosts:
        router_args += ["--host", host]
    content = [base.format(description="Chandra Windows worker over a private SSH channel", exec_start=command(remote_command(args))),
               base.format(description="Chandra OCR router for ROCm and Windows Intel", exec_start=command(router_args))]
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name, text in zip(UNITS, content):
        (unit_dir / name).write_text(text)
    receipt = {"schema_version": 1, "source_root": str(ROOT), "units": list(UNITS), "hosts": hosts,
               "router_port": args.router_port, "worker_port": args.worker_port, "ssh_host": args.ssh_host,
               "remote_root": args.remote_root, "remote_python": args.remote_python, "model_path": args.model_path}
    receipt_path = LOCAL / "installation.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_path.chmod(0o600)
    subprocess.run(["systemd-analyze", "--user", "verify", *(str(unit_dir / name) for name in UNITS)], check=True)
    systemctl("daemon-reload")
    systemctl("enable", "--now", *UNITS)
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
    install_parser.add_argument("--router-port", type=int, default=8002)
    install_parser.add_argument("--lan-address")
    install_parser.add_argument("--northstone-page-seconds", type=float, default=1)
    install_parser.add_argument("--waystone-page-seconds", type=float, default=1)
    for action in ("status", "start", "stop", "restart", "uninstall", "logs"):
        sub.add_parser(action)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This manager runs on the Linux router host; the inference worker remains native Windows")
    if args.action == "install":
        install(args)
    elif args.action == "logs":
        subprocess.run(["journalctl", "--user", "-u", UNITS[0], "-u", UNITS[1], "-n", "100", "--no-pager"], check=True)
    elif args.action == "uninstall":
        directory = Path.home() / ".config/systemd/user"
        paths = [directory / name for name in UNITS]
        if any(path.exists() and not path.read_text().startswith(MARKER) for path in paths):
            raise ValueError("Refusing to remove an unmanaged service")
        systemctl("disable", "--now", *UNITS, check=False)
        for path in paths:
            path.unlink(missing_ok=True)
        systemctl("daemon-reload")
        print("Removed task-owned user services. Models, environments, inputs and results were retained.")
    else:
        systemctl(args.action, *UNITS, *( ["--no-pager"] if args.action == "status" else []), check=args.action != "status")


if __name__ == "__main__":
    main()
