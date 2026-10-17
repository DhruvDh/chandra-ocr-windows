"""Contemporary weight custody for fresh CPU FP32 teacher exports (standard library only).

verification/export_logits.py --custody uses this module to bind one explicit local model directory to the pinned provenance/model.json inventory around its explicit Transformers loads. It retains the exact producer bytes and requires them to compile to the module code objects the process is already executing, so that the retained exporter and helper are the code that runs; the exporter then performs the export in a module executed from the retained exporter compilation. Before the processor load it walks the model directory by file descriptor without following links, refuses unexpected, missing, symlinked or non-regular entries and loader-significant names that the custody route does not authenticate, and streams every member through SHA-256 within its pinned extent and a monotonic deadline. It records file identities, extents and digests, an in-process library source manifest and the declared load arguments, and it refuses a load whose arguments differ from the declaration. Every file it hashes is opened without blocking and must be a regular file whose identity does not change while it is read. Before final metadata is accepted it repeats every check, requires identical identities, extents, timestamps and digests, and reopens every retained evidence file to require the very file it wrote with exactly the committed bytes. The evidence is the before/after state of named files, the code-object binding of the producer and the arguments passed to the loaders; it does not show which bytes a library read.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import types

SCHEMA = "chandra.cpu-export-custody.v2"
RECEIPT_SCHEMA = "chandra.cpu-export-custody-receipt.v2"
MIB = 1024 * 1024
BLOCK = 8 * MIB  # Fixed streaming buffer; hashing memory does not grow with file size.
INVENTORY_LIMIT, INDEX_LIMIT, PRODUCER_LIMIT = MIB, 16 * MIB, 4 * MIB
MEMBER_LIMIT, ENTRY_LIMIT, DEPTH_LIMIT = 256, 4096, 8
SOURCE_FILES, SOURCE_FILE_BYTES, SOURCE_BYTES = 20000, 2048 * MIB, 4096 * MIB
HARD_DEADLINE = 3600.0
ERROR_LIMIT = 4096
SINGLE, INDEX = "model.safetensors", "model.safetensors.index.json"
NAME = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.+-]{0,254}")
HEX, REVISION = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")
# Files a Transformers local load may resolve besides config, processor and tokenizer files: other weight formats and indexes,
# adapters and remote code. The custody route authenticates safetensors only, so an inventory naming any of these is refused.
UNSUPPORTED_SUFFIXES = (".py", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack", ".gguf", ".pkl", ".pickle")
UNSUPPORTED_NAMES = {"adapter_config.json", "adapter_model.safetensors", "pytorch_model.bin.index.json", "tf_model.h5.index.json",
                     "flax_model.msgpack.index.json", "model.safetensors.index.json.partial"}
ROLES = {"config.json": "model_config", "generation_config.json": "generation_config", INDEX: "weights_index",
         "preprocessor_config.json": "processor", "processor_config.json": "processor", "video_preprocessor_config.json": "processor",
         "tokenizer.json": "tokenizer", "tokenizer_config.json": "tokenizer", "special_tokens_map.json": "tokenizer",
         "added_tokens.json": "tokenizer", "vocab.json": "tokenizer", "merges.txt": "tokenizer",
         "chat_template.jinja": "chat_template", "chat_template.json": "chat_template"}
SOURCE_PACKAGES = ("transformers", "tokenizers", "safetensors", "huggingface_hub", "accelerate", "torch", "torchvision", "PIL", "numpy")
FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOCTTY", 0)
DIRECTORY_FLAGS = FLAGS | getattr(os, "O_DIRECTORY", 0)
FOLLOW_FLAGS = FLAGS & ~getattr(os, "O_NOFOLLOW", 0)  # Only for the exporter's metadata digests of paths it was given, which may be links.
EVIDENCE = ("Before anything was loaded, the exact producer bytes were retained and required to compile to the module code objects the "
            "process was already executing, and the export ran from the retained exporter compilation. Before the processor load, every "
            "directory entry under model_root was enumerated by file descriptor without following links and every pinned member was streamed "
            "through SHA-256 and matched provenance/model.json; the declared from_pretrained arguments were compared with the arguments actually "
            "passed; after the export and before metadata.json was written, the same walk, identities, extents, timestamps and digests were "
            "required to be identical, and every retained evidence file was reopened and required to be the unchanged file the session wrote. "
            "This is evidence about named files before and after explicit loads, not proof of which bytes Transformers, safetensors or the "
            "operating system read.")
EXECUTION = ("Each retained producer file was compiled from exactly its retained bytes and required to equal, by CPython code-object equality "
             "(bytecode, constants including nested function code, names, line and column table, exception table), the module code object the "
             "process captured from its own executing frame; export_logits.py --custody then executes the retained exporter compilation in a "
             "fresh module for the export itself.")
LIMITATIONS = [
    "A change made and fully reverted between the two passes, including its timestamps, is not detectable by before/after observation.",
    "Library sources are hashed on disk; Python may execute cached bytecode, torch/lib is listed as the directory holds it when sources are collected, and the interpreter, its standard library and system libraries outside torch/lib are named only by version.",
    "Producer binding is CPython code-object equality: the bytes Python first compiled may differ from the retained bytes only in text that does not change the compiled code, such as a comment on an unchanged line; interpreter start-up hooks and library code are outside the binding.",
    "Custody opens only regular files, without blocking, and checks its deadline before every open and 8 MiB block; a read the kernel itself stalls, the exporter's own payload writes, and reads by the loaders, Pillow and the exporter's prompt and response reads, are bounded only by an outer timeout.",
    "Arguments the producer does not pass take the defaults of the recorded Transformers sources; nothing here inspects their resolution.",
    "Custody concerns the checkpoint, producer and library files only; it qualifies no logits, numerical agreement or OCR result.",
]
clock = time.monotonic  # One monotonic clock for every deadline; tests may substitute it.
MODULE_CODE = sys._getframe(0).f_code if hasattr(sys, "_getframe") else None  # This module as Python compiled it, for producer binding.


class CustodyError(Exception):
    """Custody could not be established or was lost; the export must not be accepted."""


def require(condition, message):
    if not condition:
        raise CustodyError(message)


class Deadline:
    def __init__(self, seconds, what):
        require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= HARD_DEADLINE,
                f"The custody deadline must be within (0, {HARD_DEADLINE:g}] seconds")
        self.seconds, self.what, self.start = float(seconds), what, clock()
        self.end = self.start + self.seconds

    def check(self, stage):
        require(clock() < self.end, f"The {self.seconds:g}-second custody deadline for {self.what} expired during {stage}")

    def elapsed(self):
        return clock() - self.start


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def strict_json(data, what):
    """UTF-8 JSON without duplicate keys, NaN/Infinity tokens or overflowing numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{what}: duplicate JSON key {key!r:.80}")
            result[key] = value
        return result

    def constant(token):
        raise CustodyError(f"{what}: nonfinite JSON constant {token}")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise CustodyError(f"{what} is not strict UTF-8 JSON: {type(error).__name__}") from None
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float):
            require(math.isfinite(item), f"{what}: a number overflows to a nonfinite value")
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return value


def member_parts(text):
    """A pinned member name: relative POSIX components without '.', '..', separators, controls or spaces."""
    require(isinstance(text, str) and 0 < len(text) <= 1024, f"Inventory member name is not a short string: {text!r:.80}")
    parts = text.split("/")
    require(len(parts) <= DEPTH_LIMIT and all(NAME.fullmatch(p) and p not in (".", "..") for p in parts),
            f"Inventory member name is unsafe: {text!r:.120}")
    return tuple(parts)


def role(name):
    base = name.rsplit("/", 1)[-1]
    if base.endswith(".safetensors"):
        return "weights"
    return ROLES.get(name, "other") if "/" not in name else "other"


def load_inventory(data, model=None, revision=None):
    """Validate an inventory in the provenance/model.json format and classify its members."""
    require(len(data) <= INVENTORY_LIMIT, f"The model inventory exceeds {INVENTORY_LIMIT} bytes")
    inventory = strict_json(data, "model inventory")
    require(isinstance(inventory, dict) and set(inventory) == {"model", "revision", "files"}, "Model inventory fields differ from provenance/model.json")
    require(isinstance(inventory["model"], str) and 0 < len(inventory["model"]) <= 200, "Model inventory model name is invalid")
    require(isinstance(inventory["revision"], str) and REVISION.fullmatch(inventory["revision"]), "Model inventory revision must be a 40-digit commit")
    require(model is None or inventory["model"] == model, f"Model inventory names {inventory['model']!r:.80}, not {model}")
    require(revision is None or inventory["revision"] == revision, f"Model inventory revision {inventory['revision']} is not the pinned {revision}")
    files = inventory["files"]
    require(isinstance(files, list) and 0 < len(files) <= MEMBER_LIMIT, f"Model inventory must list 1..{MEMBER_LIMIT} files")
    members, folded = {}, set()
    for record in files:
        require(isinstance(record, dict) and set(record) == {"file", "size", "sha256"}, "Model inventory file record is malformed")
        name, size, digest = record["file"], record["size"], record["sha256"]
        member_parts(name)
        require(type(size) is int and 0 <= size < 2 ** 63, f"Inventory size of {name} must be a nonnegative integer")
        require(isinstance(digest, str) and HEX.fullmatch(digest), f"Inventory SHA-256 of {name} must be lowercase hexadecimal")
        require(name.casefold() not in folded, f"Inventory names {name} twice (case-insensitively)")
        folded.add(name.casefold())
        members[name] = {"file": name, "role": role(name), "size": size, "sha256": digest}
    for name in members:
        base = name.rsplit("/", 1)[-1]
        require(base not in UNSUPPORTED_NAMES and not base.endswith(UNSUPPORTED_SUFFIXES),
                f"Inventory member {name} could change Transformers resolution and is outside the safetensors custody route")
        require(not any(other.startswith(name + "/") for other in members), f"Inventory member {name} is also a directory of another member")
    weights = sorted(n for n, m in members.items() if m["role"] == "weights")
    require(all("/" not in n for n in weights), f"Weights must be top-level files of the model root: {weights}")
    require((SINGLE in members) != (INDEX in members),
            f"The inventory must name exactly one of {SINGLE} and {INDEX}; both makes the loaded weights ambiguous and neither names none")
    if SINGLE in members:
        require(weights == [SINGLE], f"Inventory weights {weights} include files a single-file {SINGLE} load never resolves")
        layout = "single-file safetensors"
    else:
        require(weights, f"{INDEX} names shards but the inventory pins none")
        layout = "sharded safetensors with index"
    require("config.json" in members, "The inventory does not pin config.json")
    ordered = [members[n] for n in sorted(members)]
    return {"model": inventory["model"], "revision": inventory["revision"], "sha256": sha256(data), "bytes": len(data),
            "layout": layout, "members": ordered, "member_bytes": sum(m["size"] for m in ordered)}


def check_index(data, shards):
    """The pinned index must map tensors onto exactly the pinned shards."""
    index = strict_json(data, INDEX)
    require(isinstance(index, dict) and "weight_map" in index and set(index) <= {"metadata", "weight_map"},
            f"{INDEX} fields differ from a safetensors index")
    require("metadata" not in index or isinstance(index["metadata"], dict), f"{INDEX} metadata must be an object")
    mapping = index["weight_map"]
    require(isinstance(mapping, dict) and mapping and all(isinstance(v, str) for v in mapping.values()), f"{INDEX} weight_map is malformed")
    named = set(mapping.values())
    for shard in named:
        require(NAME.fullmatch(shard) and shard not in (".", "..") and shard.endswith(".safetensors") and shard != SINGLE,
                f"{INDEX} names an unsafe or unsupported shard {shard!r:.120}")
    require(named == set(shards), f"{INDEX} shards differ from the pinned shards: unpinned {sorted(named - set(shards))[:8]}, "
                                  f"unreferenced {sorted(set(shards) - named)[:8]}")
    return {"file": INDEX, "tensors": len(mapping), "shards": sorted(named)}


def resolved_directory(text, what):
    """An absolute, normalized directory path none of whose components is a symlink."""
    require(isinstance(text, str) and os.path.isabs(text) and os.path.normpath(text) == text and text != os.sep,
            f"{what} must be an absolute normalized path: {text!r:.200}")
    real = os.path.realpath(text)
    require(real == text, f"{what} traverses a symlink; pass the resolved directory {real!r:.200} explicitly")
    try:
        info = os.lstat(text)
    except OSError as error:
        raise CustodyError(f"{what} cannot be inspected: {error.strerror}") from None
    require(stat.S_ISDIR(info.st_mode), f"{what} is not a directory: {text}")
    return text


def identity(info):
    return {"device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode, "links": info.st_nlink, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def open_at(directory_fd, name, flags, what):
    try:
        return os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise CustodyError(f"{what} cannot be opened without following links: {error.strerror}") from None


def open_member(root_fd, parts):
    """Open a member component by component relative to the root descriptor, never following a link."""
    name, opened, current = "/".join(parts), [], root_fd
    try:
        for part in parts[:-1]:
            current = open_at(current, part, DIRECTORY_FLAGS, f"Directory of {name}")
            opened.append(current)
        return open_at(current, parts[-1], FLAGS, name)
    finally:
        for fd in opened:
            os.close(fd)


def walk(root_fd, files, directories, deadline):
    """Every entry under the root without following links: {member: identity, directory: identity}."""
    found, folders, pending, count = {}, {"": identity(os.fstat(root_fd))}, [("", root_fd, False)], 0
    try:
        while pending:
            prefix, fd, owned = pending.pop()
            try:
                with os.scandir(fd) as listing:
                    for entry in listing:
                        count += 1
                        require(count <= ENTRY_LIMIT, f"The model root holds more than {ENTRY_LIMIT} entries")
                        deadline.check("directory enumeration")
                        name = prefix + entry.name
                        if entry.is_symlink():
                            raise CustodyError(f"Model root entry {name!r:.200} is a symlink; custody never follows links")
                        if entry.is_dir(follow_symlinks=False):
                            require(name in directories, f"Model root holds an unexpected directory {name!r:.200}")
                            child = open_at(fd, entry.name, DIRECTORY_FLAGS, f"Directory {name}")
                            folders[name] = identity(os.fstat(child))
                            pending.append((name + "/", child, True))
                        elif entry.is_file(follow_symlinks=False):
                            require(name in files, f"Model root holds an unexpected file {name!r:.200}; Transformers could resolve it")
                            found[name] = identity(entry.stat(follow_symlinks=False))
                        else:
                            raise CustodyError(f"Model root entry {name!r:.200} is not a regular file or directory")
            finally:
                if owned:
                    os.close(fd)
    finally:
        for _, fd, owned in pending:
            if owned:
                os.close(fd)
    missing = sorted(set(files) - set(found))
    require(not missing, f"Pinned members are missing from the model root: {missing[:8]}")
    require(set(folders) == set(directories) | {""}, f"Pinned member directories are missing: {sorted(set(directories) - set(folders))[:8]}")
    return found, folders


def stream(handle, size, deadline, what, keep=False):
    """SHA-256 of exactly size bytes from an open file, never reading beyond size + 1 bytes."""
    hasher, total, kept = hashlib.sha256(), 0, []
    buffer = bytearray(min(BLOCK, size + 1))
    view = memoryview(buffer)
    while True:
        deadline.check(f"hashing {what}")
        count = handle.readinto(view[:min(len(buffer), size - total + 1)])
        if not count:
            break
        total += count
        require(total <= size, f"{what} grew beyond its {size} pinned bytes while it was hashed")
        hasher.update(view[:count])
        if keep:
            kept.append(bytes(view[:count]))
    require(total == size, f"{what} is truncated: {total} of {size} bytes")
    return hasher.hexdigest(), b"".join(kept) if keep else None


def hash_member(root_fd, member, deadline):
    name = member["file"]
    fd = open_member(root_fd, member_parts(name))
    with os.fdopen(fd, "rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        require(stat.S_ISREG(before.st_mode), f"{name} is not a regular file")
        require(before.st_size == member["size"], f"{name} has {before.st_size} bytes; the inventory pins {member['size']}")
        require(name != INDEX or member["size"] <= INDEX_LIMIT, f"{INDEX} exceeds {INDEX_LIMIT} bytes")
        digest, data = stream(handle, member["size"], deadline, name, keep=name == INDEX)
        after = os.fstat(handle.fileno())
    require(identity(before) == identity(after), f"{name} changed while it was hashed")
    return identity(after), digest, data


def verify_directory(root, inventory, deadline):
    """One complete custody pass over the model root; any deviation from the inventory refuses."""
    resolved_directory(root, "The model root")
    started, members = clock(), {m["file"]: m for m in inventory["members"]}
    directories = {"/".join(member_parts(n)[:i]) for n in members for i in range(1, len(member_parts(n)))}
    fd = open_at(None, root, DIRECTORY_FLAGS, "The model root")
    try:
        opened, named = os.fstat(fd), os.lstat(root)
        require((opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino), "The model root changed while it was opened")
        found, folders = walk(fd, members, directories, deadline)
        records, index = [], None
        for name in sorted(members):
            member = members[name]
            observed, digest, data = hash_member(fd, member, deadline)
            require(observed == found[name], f"{name} was replaced or changed between enumeration and hashing")
            require(digest == member["sha256"], f"{name} SHA-256 {digest} differs from the inventory pin {member['sha256']}")
            if name == INDEX:
                index = check_index(data, [m["file"] for m in inventory["members"] if m["role"] == "weights"])
            records.append({"file": name, "role": member["role"], "sha256": digest, **observed})
        after_folders = {"": identity(os.fstat(fd))}
        for name in sorted(directories):
            child = open_member(fd, member_parts(name))
            try:
                after_folders[name] = identity(os.fstat(child))
            finally:
                os.close(child)
        require(after_folders == folders, "A model root directory changed while it was hashed")
    finally:
        os.close(fd)
    return {"directories": [{"directory": n or ".", **folders[n]} for n in sorted(folders)], "members": records,
            "index": index, "bytes_hashed": sum(m["size"] for m in members.values()), "elapsed_seconds": clock() - started}


def compare_passes(before, after):
    """The post-load pass must observe exactly what the pre-load pass observed."""
    differences = []
    for key, label in (("directories", "directory"), ("members", "file")):
        old = {r[label]: {k: v for k, v in r.items() if k != label} for r in before[key]}
        new = {r[label]: {k: v for k, v in r.items() if k != label} for r in after[key]}
        for name in sorted(set(old) | set(new)):
            if old.get(name) != new.get(name):
                fields = sorted(k for k in set(old.get(name, {})) | set(new.get(name, {})) if old.get(name, {}).get(k) != new.get(name, {}).get(k))
                differences.append(f"{label} {name}: {', '.join(fields) or 'presence'}")
    require(before["index"] == after["index"], "The weights index mapping changed")
    require(not differences, "Model root custody changed between the pre-load and post-load passes: " + "; ".join(differences[:12]))


def changed_fields(old, new):
    return ", ".join(sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))) or "presence"


def regular_handle(fd, what):
    """An unbuffered reader and the identity of fd if it is a regular file; otherwise close fd and refuse."""
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise CustodyError(f"{what} is not a regular file")
    return os.fdopen(fd, "rb", buffering=0), identity(info)


def hash_regular(path, what, deadline, limit, keep=False, follow=False, dir_fd=None):
    """(identity, SHA-256, bytes if keep) of a regular file of at most limit bytes, opened without blocking on a FIFO or (unless follow) following a final link; its identity must not change while it is read within the deadline."""
    deadline.check(f"opening {what}")
    try:
        fd = os.open(path, FOLLOW_FLAGS if follow else FLAGS, dir_fd=dir_fd)
    except OSError as error:
        raise CustodyError(f"{what} cannot be opened{'' if follow else ' without following links'}: {error.strerror}") from None
    handle, before = regular_handle(fd, what)
    with handle:
        require(before["size"] <= limit, f"{what} exceeds {limit} bytes")
        digest, data = stream(handle, before["size"], deadline, what, keep)
        after = identity(os.fstat(handle.fileno()))
    require(before == after, f"{what} changed while it was hashed: {changed_fields(before, after)}")
    return after, digest, data


def read_regular(path, limit, what, deadline=None):
    """Bytes of a regular non-symlink file of at most limit bytes."""
    return hash_regular(path, what, deadline or Deadline(HARD_DEADLINE, what), limit, keep=True)[2]


def directory_identity(path, what):
    """(device, inode) of a directory that is not a symlink."""
    try:
        info = os.lstat(path)
    except OSError as error:
        raise CustodyError(f"{what} cannot be inspected: {error.strerror}") from None
    require(stat.S_ISDIR(info.st_mode), f"{what} is not a directory: {path}")
    return [info.st_dev, info.st_ino]


def verify_files(directory, directory_id, files, deadline, what):
    """Reopen each committed file below directory by descriptor without following links: it must be the very regular file that was written (identity unchanged, so still a single link) and hold exactly its committed bytes. Returns {name: {bytes, sha256}}."""
    fd = open_at(None, directory, DIRECTORY_FLAGS, what)
    checked = {}
    try:
        info = os.fstat(fd)
        require([info.st_dev, info.st_ino] == directory_id, f"{what} {directory} was replaced after it was created")
        for name, record in sorted(files.items()):
            deadline.check(f"opening {name}")
            handle, before = regular_handle(open_member(fd, tuple(name.split("/"))), f"{name} in {what}")
            with handle:
                require(before == record["identity"], f"{name} in {what} changed after it was written: {changed_fields(record['identity'], before)}")
                digest = stream(handle, record["bytes"], deadline, f"{name} in {what}")[0]
                after = identity(os.fstat(handle.fileno()))
            require(after == before, f"{name} in {what} changed while it was hashed: {changed_fields(before, after)}")
            require(digest == record["sha256"], f"{name} in {what} SHA-256 {digest} differs from the committed {record['sha256']}")
            checked[name] = {"bytes": record["bytes"], "sha256": digest}
    finally:
        os.close(fd)
    return checked


def describe_argument(value, where):
    """JSON form of one load argument; only plain values, paths and Torch dtypes are accepted."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        require(math.isfinite(value), f"{where} is not finite")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if type(value) is dict:
        require(all(type(k) is str for k in value), f"{where} has a non-string key")
        return {k: describe_argument(v, f"{where}.{k}") for k, v in sorted(value.items())}
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    raise CustodyError(f"{where} has unsupported load-argument type {type(value).__qualname__}")


def describe_loader(loader):
    owner = getattr(loader, "__self__", None)
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__qualname__}.{loader.__name__}"
    return f"{getattr(loader, '__module__', '?')}.{getattr(loader, '__qualname__', repr(loader))}"


def describe_load(loader, path, kwargs):
    return {"callable": describe_loader(loader), "pretrained_model_name_or_path": describe_argument(path, "pretrained_model_name_or_path"),
            "kwargs": describe_argument(dict(kwargs), "kwargs")}


def module_sources(modules, packages=SOURCE_PACKAGES, library_directories=()):
    """{package-relative path: absolute path} for source or extension files of loaded modules in the named packages.

    A module without a string __file__ (built-in, frozen, namespace package or extension submodule) and a relative pseudo-path (such as a torch namespace object's "_ops.py") name no file and are not listed. Every absolute __file__ names the file a loaded module came from and is listed whatever is there now, so that hashing refuses one that has disappeared or is no longer a regular file (a directory, FIFO, socket, device or symlink) rather than silently omitting a loaded module. A library directory under a loaded package must still be a directory."""
    bases, files = {}, {}
    for top in packages:
        path = getattr(modules.get(top), "__file__", None)
        if isinstance(path, str):
            package = Path(path).parent if Path(path).name.startswith("__init__.") else Path(path)
            bases[top] = package.parent
    for name, module in sorted(modules.items()):
        top = name.partition(".")[0]
        path = getattr(module, "__file__", None)
        if top in bases and isinstance(path, str) and os.path.isabs(path):
            path = Path(path)
            try:
                files[path.relative_to(bases[top]).as_posix()] = path
            except ValueError:
                files[f"{name} (outside {top})"] = path
    for directory in library_directories:
        directory = Path(directory)
        base = next((b for b in bases.values() if directory.is_relative_to(b)), None)
        if base is not None:
            require(directory.is_dir(), f"Library directory {directory.relative_to(base).as_posix()} of a loaded package is not a directory")
            for path in sorted(directory.iterdir()):
                if ".so" in path.name and not os.path.isdir(path):
                    files[path.relative_to(base).as_posix()] = path
    return files


def hash_sources(files, deadline):
    """Size and SHA-256 of each library file, each opened without following a final link or blocking (a FIFO, socket, device or symlink is refused, never waited on) and required to be a regular file whose identity does not change while it is read."""
    require(len(files) <= SOURCE_FILES, f"More than {SOURCE_FILES} library source files are loaded")
    records, total = {}, 0
    for name in sorted(files):
        observed, digest, _ = hash_regular(files[name], f"Library source {name}", deadline, min(SOURCE_FILE_BYTES, SOURCE_BYTES - total))
        total += observed["size"]
        records[name] = {"size": observed["size"], "sha256": digest}
    return records


def source_manifest(records):
    return "".join(f"{r['sha256']}  {r['size']}  {n}\n" for n, r in sorted(records.items())).encode("utf-8")


def write_new(path, data):
    """Create path exclusively and make its bytes durable; return the identity of the file written."""
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        return identity(os.fstat(handle.fileno()))


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_durably(directory, name, data):
    """Write name inside directory through a temporary sibling and one rename, so name never holds partial bytes."""
    temporary = Path(directory) / f".{name}.partial"
    if os.path.lexists(temporary):  # Left by an interrupted write of this same name in a directory this producer created.
        os.unlink(temporary)
    write_new(temporary, data)
    os.replace(temporary, Path(directory) / name)
    sync_directory(directory)


def bounded(text):
    text = " ".join(str(text).split())
    return text if len(text) <= ERROR_LIMIT else text[:ERROR_LIMIT - 1] + "…"


class Session:
    """One custody-qualified export: retained and bound producer, pre-load pass, declared loads, admission, post-load pass and acceptance."""

    def __init__(self, model_root, inventory_path, output, producer, hash_seconds, model=None, revision=None, inventory_sha256=None,
                 inventory_name=None):
        self.started = clock()
        require(os.name == "posix" and os.open in os.supports_dir_fd and os.scandir in os.supports_fd,
                "--custody needs POSIX descriptor-relative opens and listings; this platform cannot provide them")
        require(sys.implementation.name == "cpython", "--custody binds the producer by CPython code-object equality; this interpreter is not CPython")
        self.root = resolved_directory(model_root, "--model")
        require(isinstance(output, str) and os.path.isabs(output) and os.path.normpath(output) == output and output != os.sep,
                "--output must be an absolute normalized path under --custody")
        parent = os.path.dirname(output)
        resolved_directory(parent, "The --output parent")
        self.output, self.evidence = output, output + ".custody"
        require(not os.path.lexists(self.output), "--output must name a fresh path")
        require(not os.path.lexists(self.evidence), f"The custody evidence directory {self.evidence} already exists")
        Deadline(hash_seconds, "argument validation")
        self.hash_seconds = float(hash_seconds)
        data = read_regular(inventory_path, INVENTORY_LIMIT, "The model inventory")
        require(inventory_sha256 is None or sha256(data) == inventory_sha256,
                f"The model inventory SHA-256 {sha256(data)} differs from the producer's pinned {inventory_sha256}")
        self.inventory = load_inventory(data, model, revision)
        self.inventory_file = inventory_name or Path(inventory_path).name
        require(producer and all(isinstance(k, str) and isinstance(v, tuple) and len(v) == 2 for k, v in producer.items()),
                "Producer files must map repository paths to (path, bytes)")
        require(len({Path(name).name for name in producer}) == len(producer), "Producer file names must be distinct")
        self.producer = {name: {"path": str(path), "bytes": data, "sha256": sha256(data)} for name, (path, data) in producer.items()}
        self.before = self.after = self.sources = self.admission = self.execution = None
        self.declared, self.loads, self.status = {}, [], "started"
        self.retained, self.evidence_id, self.output_id = {}, None, None  # Evidence-relative name: {bytes, sha256, identity} as written.
        self.metadata_sha256 = self.receipt_sha256 = self.receipt_bytes = None

    # ---------------------------------------------------------------- pre-load

    def begin(self):
        """Create the evidence directory and retain the exact producer bytes before anything is loaded."""
        os.mkdir(self.evidence)
        self.evidence_id = directory_identity(self.evidence, "The custody evidence directory")
        os.mkdir(os.path.join(self.evidence, "producer"))
        for name, record in self.producer.items():
            record["copy"] = f"producer/{Path(name).name}"
            self.retain(record["copy"], record["bytes"])
        self.status = "begun"

    def retain(self, name, data):
        """Write one evidence file exclusively and commit its bytes, digest and identity for every later check."""
        self.retained[name] = {"bytes": len(data), "sha256": sha256(data), "identity": write_new(os.path.join(self.evidence, name), data)}

    def bind_execution(self, executing):
        """Require each retained producer file to compile to the module code object the caller captured from its executing frame.

        executing maps every producer name to that code object. Returns the compilations of the retained bytes, which are equal to the executing code; the caller executes the exporter's compilation so that the export itself runs exactly the retained bytes."""
        require(self.status == "begun", "Producer execution is bound once, after begin() and before the pre-load pass")
        require(isinstance(executing, dict) and set(executing) == set(self.producer),
                f"Executing code must be given for exactly the retained producer files {sorted(self.producer)}")
        compiled = {}
        for name, record in sorted(self.producer.items()):
            require(isinstance(executing[name], types.CodeType), f"The executing code of {name} is not a code object")
            try:
                compiled[name] = compile(record["bytes"], record["path"], "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as error:
                raise CustodyError(f"Retained producer {name} does not compile: {type(error).__name__}: {error}") from None
            require(compiled[name] == executing[name],
                    f"Retained producer {name} (SHA-256 {record['sha256']}) does not compile to the code this process is executing; "
                    f"the file changed after Python compiled it, or a stale bytecode cache was executed")
        self.execution = {"interpreter": f"{sys.implementation.name} {'.'.join(map(str, sys.version_info[:3]))}", "optimize": sys.flags.optimize,
                          "files": sorted(compiled), "rule": EXECUTION}
        self.status = "bound"
        return compiled

    def verify_retained(self, when):
        """Every retained evidence file, and admission.json once written, must still be the unchanged file the session wrote."""
        deadline = Deadline(self.hash_seconds, f"retained evidence {when}")
        try:
            checked = verify_files(self.evidence, self.evidence_id, self.retained, deadline, "the custody evidence directory")
            if self.admission is not None:
                checked["../" + os.path.basename(self.output) + "/admission.json"] = verify_files(
                    self.output, self.output_id, {"admission.json": self.admission}, deadline, "the export directory")["admission.json"]
        except CustodyError as error:
            raise CustodyError(f"Retained evidence {when}: {error}") from None
        return checked

    def verify_before_load(self):
        require(self.status == "bound" and self.before is None, "The pre-load pass runs once, after begin() and bind_execution()")
        self.before = dict(verify_directory(self.root, self.inventory, Deadline(self.hash_seconds, "the pre-load pass")),
                           deadline_seconds=self.hash_seconds)
        return self.before

    def producer_sha256(self, name):
        require(name in self.producer, f"{name} is not a retained producer file")
        return self.producer[name]["sha256"]

    def require_producer(self, name, value, what):
        require(value == self.producer_sha256(name), f"{what} {value} differs from the retained producer {name} SHA-256 {self.producer_sha256(name)}")

    def check_producer(self, when):
        """The original producer files must still hold the retained bytes."""
        deadline = Deadline(self.hash_seconds, f"producer files {when}")
        for name, record in self.producer.items():
            digest = hash_regular(record["path"], f"Producer {name}", deadline, PRODUCER_LIMIT)[1]
            require(digest == record["sha256"], f"Producer {name} changed {when}")

    def file_sha256(self, path):
        """SHA-256 of a file the exporter names in its metadata; the path may be a link, but the file must be regular and is never waited on."""
        return hash_regular(path, f"{os.fspath(path)}", Deadline(self.hash_seconds, f"hashing {os.fspath(path)}"), SOURCE_FILE_BYTES, follow=True)[1]

    def open_output(self):
        """A descriptor of the --output directory, opened without following links; it must be the directory admission.json was written into."""
        fd = open_at(None, self.output, DIRECTORY_FLAGS, "The --output directory")
        info = os.fstat(fd)
        if [info.st_dev, info.st_ino] != self.output_id:
            os.close(fd)
            raise CustodyError(f"The --output directory {self.output} was replaced after admission")
        return fd

    def payload_sha256(self, directory, name):
        """SHA-256 of a payload the exporter has just written, for its metadata, read as acceptance reads payloads: by descriptor in the admitted --output directory, never following a link or waiting, as a regular file whose identity does not change while it is read within the deadline."""
        require(self.admission is not None and os.path.realpath(directory) == self.output, "Payloads are hashed in the session's --output directory after admission")
        fd = self.open_output()
        try:
            return hash_regular(name, f"Payload {name}", Deadline(self.hash_seconds, f"payload {name}"), 2 ** 63, dir_fd=fd)[1]
        finally:
            os.close(fd)

    def digest(self, name):
        require(self.before is not None, "No pre-load pass has run")
        record = next((m for m in self.before["members"] if m["file"] == name), None)
        require(record is not None, f"{name} is not a pinned member")
        return record["sha256"]

    def require_digest(self, name, value, what):
        require(value == self.digest(name), f"{what} {value} differs from the verified {name} SHA-256 {self.digest(name)}")

    def declare(self, role, loader, path, **kwargs):
        """Record the exact explicit load before any load happens."""
        require(self.before is not None and not self.loads, "Loads are declared after the pre-load pass and before any load")
        require(role not in self.declared, f"The {role} load is declared twice")
        load = describe_load(loader, path, kwargs)
        require(load["pretrained_model_name_or_path"] == self.root, f"The {role} load names a path other than the verified model root")
        require(load["kwargs"].get("local_files_only") is True, f"The {role} load must pass local_files_only=True")
        self.declared[role] = load

    def load(self, role, loader, path, **kwargs):
        """Call the loader only with exactly the declared arguments, after the pre-load pass."""
        require(self.before is not None and self.after is None, "Loads happen after the pre-load pass and before the post-load pass")
        require(role in self.declared, f"The {role} load was never declared")
        require(all(entry["role"] != role for entry in self.loads), f"The {role} load already happened")
        actual = describe_load(loader, path, kwargs)
        require(actual == self.declared[role], f"The {role} load arguments differ from the declaration: {canonical(actual).decode()[:600]}")
        require(role != "model" or self.admission is not None, "The model load happens only after admission.json is written")
        entry = {"role": role, "started_seconds": clock() - self.started, "elapsed_seconds": None, "completed": False}
        self.loads.append(entry)
        start = clock()
        result = loader(path, **kwargs)
        entry.update(elapsed_seconds=clock() - start, completed=True)
        return result

    @staticmethod
    def loaded_sources(modules, library_directories=()):
        return module_sources(modules, library_directories=library_directories)

    def record_sources(self, files, classes):
        """Hash library sources now in the process and name the classes that perform the computation."""
        require(self.before is not None and self.admission is None and self.sources is None, "Sources are recorded once, before admission")
        records = hash_sources(files, Deadline(self.hash_seconds, "library sources"))
        manifest = source_manifest(records)
        self.retain("sources.txt", manifest)
        paths = {os.path.abspath(path): name for name, path in files.items()}
        named = {}
        for label, (qualified, path) in sorted(classes.items()):
            name = paths.get(os.path.abspath(path)) if path else None
            named[label] = {"class": qualified, "source": name, "sha256": records[name]["sha256"] if name else None}
        self.sources = {"files": files, "records": records, "summary": {
            "files": len(records), "bytes": sum(r["size"] for r in records.values()), "manifest": "sources.txt",
            "manifest_sha256": sha256(manifest), "classes": named}}

    def require_source(self, label, value, what):
        """A digest the exporter recorded for a computation class's source must equal the manifest's record of that file."""
        named = (self.sources or {}).get("summary", {}).get("classes", {}).get(label)
        require(named is not None and named["sha256"] is not None, f"No recorded library source names the {label} class")
        require(value == named["sha256"], f"{what} {value} differs from the recorded {named['source']} SHA-256 {named['sha256']}")

    def admission_record(self):
        require(self.execution is not None and self.before is not None and self.sources is not None and set(self.declared) == {"processor", "model"},
                "The admission custody record needs the producer binding, the pre-load pass, sources and both declared loads")
        weights = [m for m in self.before["members"] if m["role"] == "weights"]
        return {
            "schema": SCHEMA, "model": self.inventory["model"], "revision": self.inventory["revision"],
            "inventory": {"file": self.inventory_file, "sha256": self.inventory["sha256"], "bytes": self.inventory["bytes"],
                          "members": len(self.inventory["members"]), "member_bytes": self.inventory["member_bytes"]},
            "model_root": self.root,
            "checkpoint": {"layout": self.inventory["layout"], "weights": [{"file": m["file"], "size": m["size"], "sha256": m["sha256"]} for m in weights],
                           "index": self.before["index"]},
            "before_load": self.before,
            "loads": {role: self.declared[role] for role in sorted(self.declared)},
            "processor_load_completed_before_admission": any(e["role"] == "processor" and e["completed"] for e in self.loads),
            "producer": {name: {"sha256": r["sha256"], "bytes": len(r["bytes"]), "copy": r["copy"]} for name, r in sorted(self.producer.items())},
            "producer_execution": self.execution,
            "sources": self.sources["summary"],
            "evidence_directory": f"../{os.path.basename(self.evidence)}",
            "acceptance": "metadata.json is written only after the post-load pass matches before_load, every retained evidence file is reopened unchanged, and the evidence directory holds an accepted receipt.json naming its SHA-256",
            "evidence": EVIDENCE,
        }

    def write_admission(self, directory, meta):
        """Write admission.json exactly as the exporter serializes it, with this session's custody record."""
        require(self.admission is None, "admission.json is written once")
        require(os.path.realpath(directory) == self.output, "admission.json belongs in the session's --output directory")
        record = self.admission_record()
        require(isinstance(meta.get("runtime"), dict) and meta["runtime"].get("custody") == record, "Admission runtime lacks this custody record")
        self.check_producer("after it was retained")
        self.verify_retained("before admission")  # Corrupted evidence refuses before the model load, not only at acceptance.
        self.output_id = directory_identity(self.output, "The --output directory")
        data = json.dumps(meta, indent=2).encode("utf-8")
        written = write_new(os.path.join(self.output, "admission.json"), data)
        self.admission = {"sha256": sha256(data), "custody_sha256": sha256(canonical(record)), "bytes": len(data), "identity": written, "data": data}
        return self.admission

    # ---------------------------------------------------------------- post-load

    def recheck(self, sources_now):
        """Repeat every pre-load observation; record library files first imported after admission."""
        require(self.admission is not None and any(e["role"] == "model" and e["completed"] for e in self.loads),
                "The post-load pass follows admission and a completed model load")
        deadline = Deadline(self.hash_seconds, "the post-load pass")
        after = dict(verify_directory(self.root, self.inventory, deadline), deadline_seconds=self.hash_seconds)
        compare_passes(self.before, after)
        now = hash_sources(self.sources["files"], Deadline(self.hash_seconds, "library sources after loading"))
        changed = sorted(n for n in now if now[n] != self.sources["records"][n])
        require(not changed, f"Library sources changed after admission: {changed[:8]}")
        later = {n: p for n, p in sources_now.items() if n not in self.sources["records"]}
        later_records = hash_sources(later, Deadline(self.hash_seconds, "library sources imported after admission"))
        self.check_producer("during the export")
        self.after = after
        return {"model_root": after, "sources_unchanged": len(now), "imported_after_admission": [
            {"file": n, **later_records[n]} for n in sorted(later_records)]}

    def check_final(self, meta):
        """Final metadata keeps every admission field the importer compares, including this custody record unchanged."""
        admission = json.loads(self.admission["data"])
        runtime = {k: v for k, v in meta["runtime"].items() if k != "attention_config"}
        require(runtime == admission["runtime"], "Final runtime differs from admission runtime beyond attention_config")
        require(sha256(canonical(meta["runtime"]["custody"])) == self.admission["custody_sha256"], "The custody record changed after admission")
        for key in ("schema_version", "inputs", "response_sha256"):
            require(meta.get(key) == admission[key], f"Final metadata {key} differs from admission.json")

    def accept(self, directory, meta, payloads, sources_now, reported=None):
        """Recheck, then write an accepted receipt, then metadata.json; any failure leaves no accepted metadata.json."""
        require(os.path.realpath(directory) == self.output, "metadata.json belongs in the session's --output directory")
        require(not os.path.lexists(os.path.join(directory, "metadata.json")), "metadata.json already exists")
        recheck = self.recheck(sources_now)
        self.check_final(meta)
        fd = self.open_output()
        try:
            verified, deadline = {}, Deadline(self.hash_seconds, "payload verification")
            for name, expected in sorted(payloads.items()):
                observed, digest, _ = hash_regular(name, f"Payload {name}", deadline, 2 ** 63, dir_fd=fd)
                require(digest == expected, f"Payload {name} SHA-256 {digest} differs from the metadata's {expected}")
                verified[name] = {"bytes": observed["size"], "sha256": digest}
            entries = sorted(os.listdir(fd))
        finally:
            os.close(fd)
        data = json.dumps(meta, indent=2).encode("utf-8")
        self.metadata_sha256 = sha256(data)
        retained = self.verify_retained("before acceptance")
        self.write_receipt("accepted", None, recheck=recheck, payloads=verified, reported=reported, entries=entries, retained=retained)
        try:
            replace_durably(directory, "metadata.json", data)
            require(sha256(read_regular(os.path.join(directory, "metadata.json"), len(data), "metadata.json")) == self.metadata_sha256,
                    "metadata.json differs from the bytes the receipt names")
            # The accepted receipt names only evidence that is still unchanged, and is itself intact, once metadata.json exists.
            self.verify_retained("after metadata.json was written")
            require(sha256(read_regular(os.path.join(self.evidence, "receipt.json"), self.receipt_bytes, "receipt.json")) == self.receipt_sha256,
                    "receipt.json differs from the accepted receipt that was written")
        except BaseException as error:
            for name in ("metadata.json", ".metadata.json.partial"):
                try:
                    os.unlink(os.path.join(directory, name))
                except OSError:
                    pass
            self.status = "accepting"
            self.refuse(error)
            raise
        return self.metadata_sha256

    # ---------------------------------------------------------------- receipt

    def write_receipt(self, status, error, recheck=None, payloads=None, reported=None, entries=None, retained=None):
        receipt = {
            "schema": RECEIPT_SCHEMA, "status": status, "error": None if error is None else bounded(error),
            "export_directory": self.output, "model_root": self.root,
            "inventory": {"file": self.inventory_file, "sha256": self.inventory["sha256"], "layout": self.inventory["layout"]},
            "admission_sha256": self.admission["sha256"] if self.admission else None,
            "custody_sha256": self.admission["custody_sha256"] if self.admission else None,
            "metadata_sha256": self.metadata_sha256 if status == "accepted" else None,
            "payloads": payloads, "export_entries_before_metadata": entries,
            "before_load": self.before, "loads_declared": self.declared, "loads": self.loads,
            "after_load": recheck, "reported_by_loaded_objects": reported,
            "producer": {name: {"sha256": r["sha256"], "bytes": len(r["bytes"]), "copy": r.get("copy")} for name, r in sorted(self.producer.items())},
            "producer_execution": self.execution,
            "sources": self.sources["summary"] if self.sources else None,
            "retained_evidence": retained,
            "elapsed_seconds": clock() - self.started, "evidence": EVIDENCE, "limitations": LIMITATIONS,
        }
        data = json.dumps(receipt, indent=2).encode("utf-8")
        replace_durably(self.evidence, "receipt.json", data)
        self.status, self.receipt_sha256, self.receipt_bytes = status, sha256(data), len(data)
        return receipt

    def refuse(self, error):
        """Best-effort refused receipt; never raises and never replaces an accepted receipt that has its metadata."""
        if self.status in ("started", "accepted", "refused") or not os.path.isdir(self.evidence):
            return None
        try:
            self.metadata_sha256 = None
            return self.write_receipt("refused", f"{type(error).__name__}: {error}")
        except BaseException:
            return None
