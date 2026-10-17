"""Opt-in latest startup evidence; one private snapshot and one fixed atomic temp."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

from .config import plain_path

FILE_NAME = "llama-startup.private.json"
MAX_BYTES = 128 * 1024
TAIL_CHARACTERS = 8192


class StartupDiagnosticError(RuntimeError):
    pass


def _number(value):
    return value if type(value) is int and -(1 << 63) <= value < (1 << 64) else None


class StartupDiagnostic:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_absolute() or self.path.name != FILE_NAME:
            raise ValueError("Startup evidence requires its fixed private filename and absolute path")
        plain_path(str(self.path.parent))
        self.temp = self.path.with_name(FILE_NAME + ".tmp")
        self._lock = threading.Lock()
        self._secrets, self._frozen, self._generation = (), False, 0
        self._last = None
        self._data = {"schema": "llama-startup-private-v1", "adapter_pid": os.getpid(),
                      "generation": 0, "startup_stage": "configured", "retirement_stage": None,
                      "native_pid": None, "stdout_tail": "", "stderr_tail": "",
                      "native_output_frozen": False, "complete_lines_only": True}
        # Refuse an earlier service's evidence, symlink, or abandoned atomic temp.
        if self.temp.exists() or self.temp.is_symlink():
            raise StartupDiagnosticError("Earlier private startup temporary evidence must be preserved")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as destination:
                destination.write(self._encoded())
                destination.flush()
                os.fsync(destination.fileno())
        except OSError:
            raise StartupDiagnosticError("Cannot create exclusive private startup evidence") from None

    def _encoded(self):
        data = {**self._data, "written_utc": datetime.now(timezone.utc).isoformat()}
        raw = (json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")
        if len(raw) > MAX_BYTES:
            raise StartupDiagnosticError("Private startup evidence exceeded its fixed bound")
        return raw

    def _write(self):
        raw = self._encoded()
        created = False
        try:
            plain_path(str(self.path))
            fd = os.open(self.temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            with os.fdopen(fd, "wb") as destination:
                destination.write(raw)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(self.temp, self.path)
        except (OSError, ValueError):
            raise StartupDiagnosticError("Private startup evidence write failed") from None
        finally:
            if created and self.temp.exists():
                try:
                    self.temp.unlink()
                except OSError:
                    pass

    def begin(self, config_sha256, api_key, alias):
        with self._lock:
            self._secrets = (api_key, alias)
            self._frozen = False
            self._generation += 1
            self._data.update(config_sha256=config_sha256, generation=self._generation,
                              startup_stage="worker_start", retirement_stage=None,
                              stdout_tail="", stderr_tail="", native_output_frozen=False,
                              native_pid=None)
            self._last = None

    def _tail(self, text, clipped):
        # Never persist a token fragment at either end of the bounded RAM tail.
        lines = text.splitlines(keepends=True)
        if clipped and lines:
            lines = lines[1:]
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines.pop()
        text = "".join(lines)
        for secret in self._secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        if len(text) > TAIL_CHARACTERS:
            text = text[-TAIL_CHARACTERS:]
            text = text.partition("\n")[2]  # Do not retain a partial first line.
        return text

    def record(self, snapshot, *, stage=None, retirement=None, freeze=False):
        with self._lock:
            if stage is not None:
                self._data["startup_stage"] = stage
            if retirement is not None:
                self._data["retirement_stage"] = retirement
            for name in ("started", "starting", "stop_requested", "closed", "pending"):
                self._data[name] = snapshot.get(name) if type(snapshot.get(name)) is bool else None
            pid = _number(snapshot.get("pid"))
            if pid is not None and pid > 0:
                self._data["native_pid"] = pid  # Keep the last actual PID through cleanup-owner release.
            self._data.update(exit_code=_number(snapshot.get("exit_code")),
                              fault_present=bool(snapshot.get("fault")))
            for name in ("closure", "cleanup", "launch_evidence"):
                evidence = snapshot.get(name) or {}
                self._data[name] = {key: evidence.get(key) if type(evidence.get(key)) is bool else _number(evidence.get(key))
                                    for key in ("handles_closed", "job_active_processes", "process_exit_code",
                                                "child_created", "child_retired", "child_exited", "contained", "resumed")}
            release = snapshot.get("file_release") or {}
            self._data["held_files_released"] = release.get("released_all") if type(release.get("released_all")) is bool else None
            self._data["partial_launch_pending"] = snapshot.get("partial_launch_pending") is True
            if not self._frozen:
                for name in ("stdout", "stderr"):
                    self._data[name + "_tail"] = self._tail(snapshot.get(name + "_tail", ""),
                                                           snapshot.get("tail_clipped", {}).get(name, False))
            if freeze:
                self._frozen = self._data["native_output_frozen"] = True
            if self._last == self._data:
                return
            self._write()
            self._last = dict(self._data)
