"""Copy only serving source to an existing Windows checkout, preserving replaced files."""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def ps_quote(value):
    if any(c in value for c in "\r\n\0"):
        raise ValueError("Control characters are not valid remote paths")
    return "'" + value.replace("'", "''") + "'"


def powershell(host, source):
    encoded = base64.b64encode(("$ProgressPreference='SilentlyContinue';" + source).encode("utf-16le")).decode("ascii")
    return subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", host, "pwsh", "-NoProfile", "-EncodedCommand", encoded], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="waystone")
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.ssh_host):
        parser.error("Use a configured SSH host alias")
    files = sorted([*ROOT.glob("chandra_service/*.py"), *ROOT.glob("runtime/waystone/*.py"), ROOT / "provenance/model.json"])
    if not all(path.is_file() for path in files):
        raise ValueError("Serving source is incomplete")
    entries = [{"file": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    name = "serving-source-" + stamp
    remote = args.remote_root.rstrip("\\/") + "\\.runtime\\source-snapshots\\" + name
    powershell(args.ssh_host, "$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Path " + ps_quote(remote) + " | Out-Null")
    with tempfile.TemporaryDirectory(prefix="chandra-serving-source-") as temporary:
        manifest = Path(temporary) / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "files": entries}, indent=2) + "\n")
        archive = Path(temporary) / "source.tar"
        with tarfile.open(archive, "w") as output:
            output.add(manifest, arcname="manifest.json")
            for path in files:
                output.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        # SFTP-mode scp treats the remote pathname as data, not a shell expression.
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(archive), args.ssh_host + ":" + remote.replace("\\", "/") + "/source.tar"], check=True)
        script = f"""
$ErrorActionPreference='Stop'
$root={ps_quote(args.remote_root)}
$stage={ps_quote(remote)}
$archive=Join-Path $stage 'source.tar'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -ne '{archive_hash}') {{ throw 'Source archive hash mismatch' }}
$incoming=Join-Path $stage 'incoming'
New-Item -ItemType Directory -Path $incoming | Out-Null
& tar -xf $archive -C $incoming
if ($LASTEXITCODE -ne 0) {{ throw 'Source extraction failed' }}
$manifest=Get-Content -Raw -LiteralPath (Join-Path $incoming 'manifest.json') | ConvertFrom-Json
$prior=@()
foreach ($entry in $manifest.files) {{
    $source=Join-Path $incoming $entry.file
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant() -ne $entry.sha256) {{ throw 'Staged source hash mismatch' }}
    $target=Join-Path $root $entry.file
    if (Test-Path -LiteralPath $target) {{
        $backup=Join-Path (Join-Path $stage 'before') $entry.file
        New-Item -ItemType Directory -Force -Path (Split-Path $backup) | Out-Null
        Copy-Item -LiteralPath $target -Destination $backup
        $prior+=@{{file=$entry.file;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $backup).Hash.ToLowerInvariant()}}
    }}
}}
ConvertTo-Json -Depth 5 -InputObject @{{files=$prior}} | Set-Content -Encoding utf8 -LiteralPath (Join-Path $stage 'before.json')
foreach ($entry in $manifest.files) {{
    $target=Join-Path $root $entry.file
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath (Join-Path $incoming $entry.file) -Destination $target -Force
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant() -ne $entry.sha256) {{ throw 'Installed source hash mismatch' }}
}}
@{{state='verified';files=$manifest.files.Count;receipt=$stage}} | ConvertTo-Json -Compress
"""
        powershell(args.ssh_host, script)


if __name__ == "__main__":
    main()
