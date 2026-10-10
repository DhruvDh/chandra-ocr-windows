"""Host custody of the HLSL source the native worker compiles lazily (stdlib only; starts nothing).

chandra-worker.exe compiles each shader at its first dispatch with D3DCompileFromFile, no macros and the
standard include handler, from whatever --shader-root names. Quoted includes resolve against the including
file's directory, the initial file's directory or the working directory. The worker reports no identity for
the bytes it compiled, so the executable hash alone does not identify the arithmetic. The endpoint therefore:

- commits to a canonical tree digest in its authenticated configuration, in the same form as the CLI's
  diagnostic producer record ("sorted generic relative path, tab, file SHA-256, newline");
- bounds the tree and refuses unexpected paths, file types and content, and any include that could leave the
  committed tree or resolve differently under the include handler's lookup rules;
- verifies the configured tree at configuration load and again at every launch, then gives the worker a fresh
  task-owned snapshot of exactly the verified bytes instead of the configured directory, with its working
  directory set to the snapshot directory that holds every including file;
- has the transport hold the snapshot files (and the executable) for the worker's lifetime, and re-verifies
  the snapshot before every submission, after every terminal and at retirement.

All of this is host custody evidence. It shows which bytes were present at --shader-root when the host looked,
not which bytes the worker compiled; native compile evidence would need a worker protocol field that the
finished worker does not supply.
"""
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat

CANONICAL = "sorted generic relative path, tab, file SHA-256, newline"
MAX_FILES = 256
MAX_FILE_BYTES = 1 << 20
MAX_TREE_BYTES = 8 << 20
MAX_DEPTH = 4
MAX_RELATIVE = 240
MAX_DIRECTORIES = 64  # Includes the root; bounds empty-directory work independently of file/byte limits.
MAX_ENTRIES = 512     # Across the entire walk, not per directory.
MAX_SOURCE_LINES = 65536
MAX_INCLUDES = 1024
MAX_INCLUDE_EXPANSIONS = 4096
MAX_WORK_ROOT_ENTRIES = 2048
SEGMENT = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9_-])?")
RESERVED = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {f"{name}{i}" for name in ("com", "lpt") for i in range(10)}
TEXT = re.compile(rb"[\t\n\r\x20-\x7e]*")
INCLUDE = re.compile(rb'[ \t]*#[ \t]*include[ \t]+"([A-Za-z0-9_][A-Za-z0-9_.-]{0,62}\.hlsl)"[ \t]*')
LINE_DIRECTIVE = re.compile(rb"[ \t]*#[ \t]*line\b")
REPARSE = 0x400
HEX64 = re.compile(r"[0-9a-f]{64}")


class ShaderSourceRefused(RuntimeError):
    """The HLSL tree differs from its commitment or is outside the bounded, closed form the endpoint admits."""


def _refuse(condition, message):
    if not condition:
        raise ShaderSourceRefused(message)


class ShaderTree:
    """Exactly the bytes read from one tree, with their canonical digest and include closure."""

    def __init__(self, files, include_directory, includes):
        self.files = tuple(files)  # ((relative generic path, bytes), ...) sorted by path
        self.include_directory = include_directory  # the one directory whose files include, or None
        self.includes = includes
        self.canonical = "".join(f"{path}\t{hashlib.sha256(data).hexdigest()}\n" for path, data in self.files).encode("ascii")
        self.tree_sha256 = hashlib.sha256(self.canonical).hexdigest()
        self.total_bytes = sum(len(data) for _, data in self.files)

    def identity(self):
        return {"tree_sha256": self.tree_sha256, "files": len(self.files), "bytes": self.total_bytes, "includes": self.includes,
                "include_directory": self.include_directory, "canonical": CANONICAL}


def _plain(info, name):
    _refuse(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & REPARSE),
            f"Shader tree entry {name} is a symlink or reparse point")


def _segment(name, relative):
    _refuse(SEGMENT.fullmatch(name) is not None and name.split(".")[0].lower() not in RESERVED,
            f"Shader tree entry name is not admitted: {relative[:80]!r}")


def directory_names(directory, limit):
    """Enumerate at most limit+1 entries before refusal; never allocate an unbounded listdir result."""
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            _refuse(len(names) < limit, f"Directory traversal exceeds its entry budget {limit}")
            names.append(entry.name)
    names.sort()
    return names


def _read(path, relative, info):
    """Read one regular file once, refusing a swapped, special or oversized object."""
    _refuse(info.st_size <= MAX_FILE_BYTES, f"Shader file exceeds {MAX_FILE_BYTES} bytes: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        # POSIX identities are stable; on Windows lstat and fstat may report file IDs differently, and the digest decides.
        same = os.name == "nt" or (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino)
        _refuse(stat.S_ISREG(opened.st_mode) and same, f"Shader file changed while it was read: {relative}")
        chunks, total = [], 0
        while chunk := os.read(descriptor, 65536):
            total += len(chunk)
            _refuse(total <= MAX_FILE_BYTES, f"Shader file exceeds {MAX_FILE_BYTES} bytes: {relative}")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _includes(relative, data):
    """Bare same-directory include names; any other appearance of an include or #line directive is refused."""
    _refuse(TEXT.fullmatch(data) is not None, f"Shader file is not printable ASCII source: {relative}")
    _refuse(b"\\\n" not in data and b"\\\r\n" not in data, f"Shader file contains a line splice: {relative}")
    _refuse(b"??=" not in data and b"??/" not in data, f"Shader file contains a trigraph: {relative}")
    names = []
    for line in data.split(b"\n"):
        line = line[:-1] if line.endswith(b"\r") else line
        _refuse(LINE_DIRECTIVE.match(line) is None, f"Shader file contains a #line directive: {relative}")
        if b"include" in line:
            match = INCLUDE.fullmatch(line)
            _refuse(match is not None, f"Shader include is not a bare quoted same-directory .hlsl name on its own line: {relative}")
            _refuse(len(names) < MAX_INCLUDES, "Shader includes exceed their total work bound")
            names.append(match.group(1).decode("ascii"))
    return names


def read_tree(root):
    """Bounded walk of a plain directory tree of .hlsl files; returns the exact bytes read."""
    root = Path(root)
    info = os.lstat(root)
    _plain(info, "root")
    _refuse(stat.S_ISDIR(info.st_mode), "Shader root is not a plain directory")
    files, folded, pending, total = [], set(), [("", root, 0)], 0
    visited, directories_seen = 0, 1
    while pending:
        prefix, directory, depth = pending.pop()
        names = directory_names(directory, min(MAX_FILES, MAX_ENTRIES - visited))
        visited += len(names)
        for name in names:
            relative = f"{prefix}/{name}" if prefix else name
            _segment(name, relative)
            _refuse(len(relative) <= MAX_RELATIVE, "Shader path exceeds its length bound")
            _refuse(relative.lower() not in folded, f"Shader paths collide without case: {relative}")
            folded.add(relative.lower())
            path = directory / name
            entry = os.lstat(path)
            _plain(entry, relative)
            if stat.S_ISDIR(entry.st_mode):
                _refuse(depth + 1 <= MAX_DEPTH, f"Shader tree is deeper than {MAX_DEPTH} directories")
                directories_seen += 1
                _refuse(directories_seen <= MAX_DIRECTORIES, "Shader tree exceeds its global directory bound")
                pending.append((relative, path, depth + 1))
            else:
                _refuse(stat.S_ISREG(entry.st_mode), f"Shader tree entry is not a regular file: {relative}")
                _refuse(name.endswith(".hlsl"), f"Shader tree holds a file that is not .hlsl: {relative}")
                _refuse(len(files) < MAX_FILES, "Shader tree exceeds its file or byte bound")
                data = _read(path, relative, entry)
                total += len(data)
                _refuse(len(files) < MAX_FILES and total <= MAX_TREE_BYTES, "Shader tree exceeds its file or byte bound")
                files.append((relative, data))
    _refuse(files, "Shader tree holds no .hlsl file")
    files.sort()
    present = {path for path, _ in files}
    directories, count, lines, graph = set(), 0, 0, {}
    for path, data in files:
        lines += data.count(b"\n") + 1
        _refuse(lines <= MAX_SOURCE_LINES, "Shader source exceeds its global line work bound")
        names = _includes(path, data)
        graph[path] = []
        if not names:
            continue
        directory = path.rpartition("/")[0]
        directories.add(directory)
        for name in names:
            target = f"{directory}/{name}" if directory else name
            _refuse(target in present and target != path, f"Shader include does not name another committed file: {path} -> {name}")
            count += 1
            _refuse(count <= MAX_INCLUDES, "Shader includes exceed their total work bound")
            graph[path].append(target)
    # A closed include graph must also finish: no compiler-dependent recursive include ceiling is assumed.
    done, expansion = set(), {}
    for start in graph:
        active, walk = set(), [(start, False)]
        while walk:
            node, exiting = walk.pop()
            if exiting:
                active.remove(node)
                expansion[node] = 1 + sum(expansion[target] for target in graph[node])
                _refuse(expansion[node] <= MAX_INCLUDE_EXPANSIONS, "Shader include expansion exceeds its work bound")
                done.add(node)
            elif node not in done:
                _refuse(node not in active, "Shader include graph contains a cycle")
                active.add(node)
                walk.append((node, True))
                walk.extend((target, False) for target in graph[node])
    # One including directory makes the initial-file, parent-file and working-directory lookups the same file.
    _refuse(len(directories) <= 1, "Shader includes must all originate in one directory")
    return ShaderTree(files, directories.pop() if directories else None, count)


def verified_tree(root, expected_sha256):
    tree = read_tree(root)
    _refuse(tree.tree_sha256 == expected_sha256, f"Shader tree digest {tree.tree_sha256} differs from the configured shader_tree_sha256")
    return tree


class Snapshot:
    """A task-owned copy of one verified tree under the private input root: the worker's --shader-root."""

    def __init__(self, directory, tree):
        self.directory, self.tree = Path(directory), tree
        self.name = self.directory.name
        self.shaders = self.directory / "shaders"
        self.cwd = self.shaders.joinpath(*tree.include_directory.split("/")) if tree.include_directory else self.shaders

    @classmethod
    def create(cls, input_root, tree, on_created=None):
        """Write the verified bytes exclusively into a fresh directory, mark them read-only and authenticate them."""
        names = directory_names(input_root, MAX_WORK_ROOT_ENTRIES)
        _refuse(not any(name.startswith("h-") for name in names), "A prior shader snapshot remains owned under input_root; operator disposition is required")
        directory = Path(input_root) / ("h-" + secrets.token_hex(16))
        os.mkdir(directory, 0o700)  # Exclusive; never an existing directory.
        snapshot = cls(directory, tree)
        try:
            if on_created is not None:
                on_created(snapshot)  # The backend owns this snapshot even if writing or cleanup later fails.
            os.mkdir(snapshot.shaders, 0o700)
            for relative, data in tree.files:
                path = snapshot.shaders.joinpath(*relative.split("/"))
                for parent in reversed(path.relative_to(snapshot.shaders).parents[:-1]):
                    if not os.path.lexists(snapshot.shaders / parent):
                        os.mkdir(snapshot.shaders / parent, 0o700)
                with open(path, "xb") as output:
                    output.write(data)
                os.chmod(path, stat.S_IREAD)
            snapshot.verify()
        except BaseException as error:
            snapshot.creation_removal_error = snapshot.remove()
            error.cleanup_snapshot = snapshot  # A direct caller also retains the recoverable failed-creation owner.
            raise
        return snapshot

    def files(self):
        return [self.shaders.joinpath(*relative.split("/")) for relative, _ in self.tree.files]

    def verify(self):
        """Re-read the snapshot; it must be exactly the committed tree and nothing else."""
        _refuse(directory_names(self.directory, 1) == ["shaders"], "Shader snapshot directory holds unexpected entries")
        found = read_tree(self.shaders)
        _refuse(found.tree_sha256 == self.tree.tree_sha256, f"Shader snapshot digest {found.tree_sha256} differs from the committed tree")
        return found

    def remove(self):
        """Plan a finite removal before mutation; refuse a substituted root and never traverse a link/reparse point."""
        try:
            if not os.path.lexists(self.directory):
                return None
            root_info = os.lstat(self.directory)
            _plain(root_info, "snapshot root")
            _refuse(stat.S_ISDIR(root_info.st_mode), "Snapshot root is no longer a directory")
            pending, directories, leaves, visited, seen_dirs = [(self.directory, 0)], [], [], 0, 1
            while pending:
                current, depth = pending.pop()
                directories.append((current, os.lstat(current)))
                names = directory_names(current, MAX_ENTRIES + 1 - visited)
                visited += len(names)
                for name in names:
                    path = current / name
                    _refuse(len(str(path.relative_to(self.directory))) <= MAX_RELATIVE + len("shaders/"), "Snapshot cleanup path exceeds its bound")
                    info = os.lstat(path)
                    _refuse(not (getattr(info, "st_file_attributes", 0) & REPARSE), "Snapshot cleanup refuses a reparse point")
                    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                        _refuse(depth + 1 <= MAX_DEPTH + 1, "Snapshot cleanup exceeds its depth bound")
                        seen_dirs += 1
                        _refuse(seen_dirs <= MAX_DIRECTORIES + 1, "Snapshot cleanup exceeds its global directory bound")
                        pending.append((path, depth + 1))
                    else:
                        _refuse(stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode), "Snapshot cleanup refuses a special file")
                        leaves.append((path, info))
            current_root = os.lstat(self.directory)
            _refuse((current_root.st_dev, current_root.st_ino) == (root_info.st_dev, root_info.st_ino), "Snapshot root changed before cleanup")
            def check_directories():
                # Recheck all components before each mutation: a changed ancestor must not redirect cleanup elsewhere.
                for path, original in directories:
                    current = os.lstat(path)
                    _plain(current, "snapshot cleanup directory")
                    _refuse((current.st_dev, current.st_ino) == (original.st_dev, original.st_ino), "Snapshot directory changed before cleanup")

            for path, original in leaves:
                check_directories()
                current = os.lstat(path)
                _refuse((current.st_dev, current.st_ino, current.st_mode) == (original.st_dev, original.st_ino, original.st_mode), "Snapshot entry changed before cleanup")
                if not stat.S_ISLNK(current.st_mode):
                    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
                os.unlink(path)
            for path, original in reversed(directories):
                current = os.lstat(path)
                _plain(current, "snapshot cleanup directory")
                _refuse((current.st_dev, current.st_ino) == (original.st_dev, original.st_ino), "Snapshot directory changed before cleanup")
                os.rmdir(path)
        except (OSError, ShaderSourceRefused) as error:
            return f"{type(error).__name__}: {str(error)[:200]}"
        return None


class SourceCustody:
    """The custody record bound to one worker: committed identity, snapshot, held files and every verification."""

    def __init__(self, snapshot, held):
        self.snapshot, self.held = snapshot, held
        self.verifications = []
        self.retirement = None
        self.cleanup = None          # Latest operator cleanup disposition; the original retirement receipt stays immutable.

    @property
    def pending(self):
        result = self.cleanup or self.retirement
        return result is None or not ((result.get("held") or {}).get("released_all") is True and result.get("removed") is True)

    def verify(self, stage):
        self.snapshot.verify()
        self.verifications = (self.verifications + [stage])[-8:]

    def identity(self):
        return {**self.snapshot.tree.identity(), "snapshot": self.snapshot.name, "custody": "host", "held_by": self.held.description,
                "verified": list(self.verifications), "native_compile_evidence": None, "retirement": self.retirement,
                "cleanup_pending": self.retirement is not None and self.pending, "cleanup": self.cleanup}

    def retire(self):
        """After confirmed process closure: verify once more, release the held files, then remove the snapshot."""
        if self.retirement is not None:
            return self.retirement
        result = {"verified_at_retirement": False, "verification_error": None, "held": None, "removed": False, "removal_error": None,
                  "snapshot": self.snapshot.name, "disposition": "retained: cleanup is unconfirmed"}
        try:
            self.snapshot.verify()
            result["verified_at_retirement"] = True
        except (ShaderSourceRefused, OSError) as error:
            result["verification_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        try:
            result["held"] = self.held.release()
        except Exception as error:
            result["held"] = {"released_all": False, "failures": [f"{type(error).__name__}: {str(error)[:200]}"]}
        result["removal_error"] = self.snapshot.remove() if result["held"].get("released_all") is True else "retained: held file handles did not all close"
        result["removed"] = result["removal_error"] is None
        result["clean"] = result["verified_at_retirement"] and result["held"].get("released_all") is True and result["removed"]
        result["disposition"] = "removed" if result["removed"] else "retained: operator cleanup required; new native startup is refused"
        self.retirement = result
        return result

    def retry_cleanup(self):
        """Explicit operator action after confirmed process retirement; never changes the original receipt or restarts inference."""
        result = {"snapshot": self.snapshot.name, "held": None, "removed": False, "removal_error": None}
        try:
            result["held"] = self.held.release()
        except Exception as error:
            result["held"] = {"released_all": False, "failures": [f"{type(error).__name__}: {str(error)[:200]}"]}
        result["removal_error"] = self.snapshot.remove() if result["held"].get("released_all") is True else "retained: held file handles did not all close"
        result["removed"] = result["removal_error"] is None
        result["disposition"] = "removed" if result["removed"] else "retained: operator cleanup required"
        self.cleanup = result
        return result
