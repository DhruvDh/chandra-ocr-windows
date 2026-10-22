"""Standard-library checks of verification/cpu_custody.py on small synthetic model directories.

No model, Torch, Transformers, device or network: loaders are fake callbacks, checkpoints are a few bytes, and importer compatibility is checked by running the reviewed scripts/native/import_cpu_diagnostics.py unchanged on metadata the custody session writes.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


custody = load("cpu_custody", ROOT / "verification/cpu_custody.py")
CustodyError = custody.CustodyError
FLOAT32 = type("dtype", (), {"__module__": "torch", "__str__": lambda self: "torch.float32"})()
REVISION = "a" * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def bounded(seconds=10):
    """Fail instead of hanging if the block waits longer than seconds; SIGALRM interrupts a blocking open of a FIFO."""
    def expire(signum, frame):
        raise AssertionError(f"blocked for more than {seconds} seconds")
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class FakeAutoProcessor:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return "processor"


class FakeModel:
    calls, hook = [], None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        if cls.hook:
            cls.hook()
        return "model"


SINGLE = {"config.json": b'{"model_type": "qwen3_5"}', "tokenizer.json": b'{"version": "1.0"}', "chat_template.jinja": b"{{ messages }}",
          "model.safetensors": bytes(range(64)), ".eval_results/bench.yaml": b"score: 1\n", "README.md": b"# fixture\n"}
SHARDED = {"config.json": SINGLE["config.json"], "model-00001-of-00002.safetensors": b"shard one",
           "model-00002-of-00002.safetensors": b"shard two",
           "model.safetensors.index.json": json.dumps({"metadata": {"total_size": 18}, "weight_map": {
               "a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}}).encode()}


class Case(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chandra-custody-")
        self.base = Path(os.path.realpath(self.temporary.name))
        self.count = 0
        FakeAutoProcessor.calls, FakeModel.calls, FakeModel.hook = [], [], None
        producer = self.base / "producer"
        producer.mkdir()
        (producer / "export_logits.py").write_bytes(b"# exporter bytes\n")
        (producer / "cpu_custody.py").write_bytes(b"# helper bytes\n")
        self.producer = {f"verification/{n}": (producer / n, (producer / n).read_bytes()) for n in ("export_logits.py", "cpu_custody.py")}
        library = self.base / "site" / "transformers"
        (library / "models").mkdir(parents=True)
        (library / "__init__.py").write_bytes(b"# transformers\n")
        (library / "models" / "modeling.py").write_bytes(b"class Model: pass\n")
        self.modules = {"transformers": types.SimpleNamespace(__file__=str(library / "__init__.py")),
                        "transformers.models.modeling": types.SimpleNamespace(__file__=str(library / "models" / "modeling.py")),
                        "json": json}

    def tearDown(self):
        self.temporary.cleanup()

    def fresh(self, name):
        self.count += 1
        return self.base / f"{name}{self.count}"

    def model(self, files=SINGLE, listed=None, mutate=None):
        root = self.fresh("model")
        for name, data in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
        listed = files if listed is None else listed
        inventory = {"model": "test/model", "revision": REVISION,
                     "files": [{"file": n, "size": len(d), "sha256": sha(d)} for n, d in sorted(listed.items())]}
        if mutate:
            mutate(inventory)
        path = self.fresh("inventory")
        path.write_bytes(json.dumps(inventory, indent=2).encode())
        return str(root), path

    def session(self, root, inventory, seconds=60.0, **pins):
        return custody.Session(root, inventory, str(self.fresh("export")), self.producer, seconds, **pins)

    def bound(self, s):
        """begin(), then bind the retained fake producer bytes to their own compilations, as the exporter binds the code it is executing."""
        s.begin()
        s.bind_execution({name: compile(data, str(path), "exec", dont_inherit=True) for name, (path, data) in self.producer.items()})
        return s

    def admitted(self, files=SINGLE, hook=None):
        """A session through admission and the model load, with the payload written; returns (session, export, meta)."""
        root, inventory = self.model(files)
        s = self.session(root, inventory)
        self.bound(s)
        s.verify_before_load()
        s.declare("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.declare("model", FakeModel.from_pretrained, Path(root), dtype=FLOAT32, device_map="cpu", local_files_only=True,
                  attn_implementation={"vision_config": "sdpa", "text_config": "eager"})
        s.load("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.record_sources(s.loaded_sources(self.modules), {"model": ("transformers.models.modeling.Model", self.modules["transformers.models.modeling"].__file__)})
        export = Path(s.output)
        export.mkdir()
        meta = {"schema_version": 1, "context": {"model_revision": REVISION}, "inputs": {"input_ids": {"sha256": "1" * 64}},
                "runtime": {"device": "cpu", "dtype": "torch.float32", "script_sha256": s.producer_sha256("verification/export_logits.py")},
                "response_sha256": "2" * 64, "states": []}
        meta["runtime"]["custody"] = s.admission_record()
        s.write_admission(export, meta)
        FakeModel.hook = hook
        s.load("model", FakeModel.from_pretrained, Path(root), dtype=FLOAT32, device_map="cpu", local_files_only=True,
               attn_implementation={"text_config": "eager", "vision_config": "sdpa"})
        (export / "logits.f32").write_bytes(b"\0" * 16)
        meta["runtime"]["attention_config"] = {"vision": "sdpa", "text": "eager"}
        meta.update(logits={"file": "logits.f32", "sha256": sha(b"\0" * 16)}, elapsed_seconds=1.5)
        return s, export, meta

    def accept(self, s, export, meta, payloads=None):
        return s.accept(export, meta, payloads or {"logits.f32": meta["logits"]["sha256"]}, s.loaded_sources(self.modules), {"model.dtype": "torch.float32"})

    def receipt(self, s):
        return json.loads(Path(s.evidence, "receipt.json").read_text())

    def refused_after_load(self, hook, fragment, files=SINGLE):
        s, export, meta = self.admitted(files, hook=hook)
        with self.assertRaises(CustodyError) as caught:
            self.accept(s, export, meta)
        s.refuse(caught.exception)
        self.assertFalse(os.path.lexists(export / "metadata.json"))
        receipt = self.receipt(s)
        self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("refused", None))
        for part in fragment.split(" & "):  # Coarse filesystem clocks may or may not also move ctime, so fields are matched one by one.
            self.assertIn(part, str(caught.exception))
            self.assertIn(part, receipt["error"])
        return receipt


class InventoryTests(Case):
    def test_pinned_repository_inventory_is_accepted_as_single_file(self):
        data = (ROOT / "provenance/model.json").read_bytes()
        inventory = custody.load_inventory(data, "datalab-to/chandra-ocr-2", "af93b47dba1b47b6640c86ccf487ed2260ab9a09")
        self.assertEqual(inventory["layout"], "single-file safetensors")
        self.assertEqual(len(inventory["members"]), 17)
        weights = [m for m in inventory["members"] if m["role"] == "weights"]
        self.assertEqual(weights, [{"file": "model.safetensors", "role": "weights", "size": 10591220088,
                                    "sha256": "0804568be9f099d6479fad9ed77a4da4611f3c1e7bc6e009af7dce45e8aa3847"}])
        roles = {m["file"]: m["role"] for m in inventory["members"]}
        self.assertEqual((roles["config.json"], roles["tokenizer.json"], roles["chat_template.jinja"], roles["processor_config.json"]),
                         ("model_config", "tokenizer", "chat_template", "processor"))
        # The exporter's pin is the digest of exactly these tracked bytes (eol=lf in .gitattributes).
        exporter = (ROOT / "verification/export_logits.py").read_text()
        self.assertIn(f'INVENTORY_SHA256 = "{sha(data)}"', exporter)

    def test_unsafe_or_ambiguous_inventories_are_refused(self):
        base = {"model": "test/model", "revision": REVISION, "files": [
            {"file": "config.json", "size": 1, "sha256": "0" * 64}, {"file": "model.safetensors", "size": 1, "sha256": "1" * 64}]}
        cases = [
            ("unsafe", lambda f: f.append({"file": "../escape", "size": 1, "sha256": "2" * 64})),
            ("unsafe", lambda f: f.append({"file": "/absolute", "size": 1, "sha256": "2" * 64})),
            ("unsafe", lambda f: f.append({"file": "a//b", "size": 1, "sha256": "2" * 64})),
            ("unsafe", lambda f: f.append({"file": "a/./b", "size": 1, "sha256": "2" * 64})),
            ("unsafe", lambda f: f.append({"file": "a\\b", "size": 1, "sha256": "2" * 64})),
            ("unsafe", lambda f: f.append({"file": "space name", "size": 1, "sha256": "2" * 64})),
            ("twice", lambda f: f.append({"file": "CONFIG.json", "size": 1, "sha256": "2" * 64})),
            ("twice", lambda f: f.append(dict(f[0]))),
            ("nonnegative integer", lambda f: f[0].update(size=True)),
            ("nonnegative integer", lambda f: f[0].update(size=1.0)),
            ("nonnegative integer", lambda f: f[0].update(size=-1)),
            ("lowercase hexadecimal", lambda f: f[0].update(sha256="A" * 64)),
            ("malformed", lambda f: f[0].update(extra=1)),
            ("Transformers resolution", lambda f: f.append({"file": "modeling_custom.py", "size": 1, "sha256": "2" * 64})),
            ("Transformers resolution", lambda f: f.append({"file": "pytorch_model.bin", "size": 1, "sha256": "2" * 64})),
            ("Transformers resolution", lambda f: f.append({"file": "adapter_config.json", "size": 1, "sha256": "2" * 64})),
            ("Transformers resolution", lambda f: f.append({"file": "pytorch_model.bin.index.json", "size": 1, "sha256": "2" * 64})),
            ("top-level", lambda f: f.append({"file": "sub/extra.safetensors", "size": 1, "sha256": "2" * 64})),
            ("never resolves", lambda f: f.append({"file": "other.safetensors", "size": 1, "sha256": "2" * 64})),
            ("exactly one", lambda f: f.append({"file": "model.safetensors.index.json", "size": 1, "sha256": "2" * 64})),
            ("exactly one", lambda f: f.pop(1)),
            ("config.json", lambda f: f.pop(0)),
            ("also a directory", lambda f: f.append({"file": "config.json/x", "size": 1, "sha256": "2" * 64})),
        ]
        for fragment, mutate in cases:
            with self.subTest(fragment=fragment):
                inventory = json.loads(json.dumps(base))
                mutate(inventory["files"])
                with self.assertRaises(CustodyError) as caught:
                    custody.load_inventory(json.dumps(inventory).encode())
                self.assertIn(fragment, str(caught.exception))
        for fragment, data in (("duplicate JSON key", b'{"model": "a", "model": "b", "revision": "", "files": []}'),
                               ("nonfinite", b'{"model": NaN}'), ("nonfinite", b'{"model": 1e999}'), ("strict UTF-8", b'\xff')):
            with self.subTest(fragment=fragment), self.assertRaises(CustodyError) as caught:
                custody.load_inventory(data)
            self.assertIn(fragment, str(caught.exception))
        with self.assertRaisesRegex(CustodyError, "not the pinned"):
            custody.load_inventory(json.dumps(base).encode(), revision="b" * 40)
        with self.assertRaisesRegex(CustodyError, "names 'test/model'"):
            custody.load_inventory(json.dumps(base).encode(), model="datalab-to/chandra-ocr-2")

    def test_inventory_digest_pin_is_enforced(self):
        root, inventory = self.model()
        with self.assertRaisesRegex(CustodyError, "differs from the producer's pinned"):
            self.session(root, inventory, inventory_sha256="0" * 64)
        self.session(root, inventory, inventory_sha256=sha(inventory.read_bytes()), model="test/model", revision=REVISION)


class DirectoryTests(Case):
    def verify(self, root, inventory):
        s = self.session(root, inventory)
        self.bound(s)
        return s.verify_before_load()

    def test_valid_single_file_and_sharded_layouts(self):
        root, inventory = self.model()
        before = self.verify(root, inventory)
        self.assertEqual([m["file"] for m in before["members"]], sorted(SINGLE))
        self.assertEqual(before["bytes_hashed"], sum(len(d) for d in SINGLE.values()))
        self.assertIsNone(before["index"])
        self.assertEqual([d["directory"] for d in before["directories"]], [".", ".eval_results"])
        record = next(m for m in before["members"] if m["file"] == "model.safetensors")
        info = os.stat(Path(root) / "model.safetensors")
        self.assertEqual((record["sha256"], record["size"], record["inode"], record["mtime_ns"], record["ctime_ns"]),
                         (sha(SINGLE["model.safetensors"]), 64, info.st_ino, info.st_mtime_ns, info.st_ctime_ns))
        root, inventory = self.model(SHARDED)
        before = self.verify(root, inventory)
        self.assertEqual(before["index"], {"file": "model.safetensors.index.json", "tensors": 2,
                                           "shards": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]})

    def test_index_must_name_exactly_the_pinned_shards(self):
        def index(mapping, extra=None):
            files = dict(SHARDED)
            files["model.safetensors.index.json"] = json.dumps(dict({"weight_map": mapping}, **(extra or {}))).encode()
            return files
        one, two = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
        cases = [("unpinned ['model-00003", index({"a": one, "b": two, "c": "model-00003-of-00002.safetensors"})),
                 ("unreferenced ['model-00002", index({"a": one})),
                 ("unsafe or unsupported shard", index({"a": one, "b": two, "c": "../model-00001-of-00002.safetensors"})),
                 ("unsafe or unsupported shard", index({"a": one, "b": two, "c": "sub/x.safetensors"})),
                 ("unsafe or unsupported shard", index({"a": one, "b": two, "c": "model.safetensors"})),
                 ("fields differ", index({"a": one, "b": two}, {"extra": 1})),
                 ("weight_map is malformed", index({}))]
        for fragment, files in cases:
            with self.subTest(fragment=fragment):
                root, inventory = self.model(files)
                with self.assertRaises(CustodyError) as caught:
                    self.verify(root, inventory)
                self.assertIn(fragment, str(caught.exception))

    def test_mismatched_unexpected_missing_and_linked_members_are_refused(self):
        same_size = dict(SINGLE, **{"model.safetensors": bytes(reversed(range(64)))})
        other_config = dict(SINGLE, **{"config.json": b'{"model_type": "other!!"}'})
        cases = [
            ("differs from the inventory pin", same_size, SINGLE),
            ("differs from the inventory pin", other_config, SINGLE),
            ("the inventory pins 64", dict(SINGLE, **{"model.safetensors": bytes(65)}), SINGLE),
            ("unexpected file 'pytorch_model.bin'", dict(SINGLE, **{"pytorch_model.bin": b"x"}), SINGLE),
            ("unexpected file 'model.safetensors.index.json'", dict(SINGLE, **{"model.safetensors.index.json": b"{}"}), SINGLE),
            ("unexpected file 'adapter_config.json'", dict(SINGLE, **{"adapter_config.json": b"{}"}), SINGLE),
            ("unexpected file '.eval_results/extra'", dict(SINGLE, **{".eval_results/extra": b"x"}), SINGLE),
            ("unexpected directory '.cache'", dict(SINGLE, **{".cache/huggingface/x": b"x"}), SINGLE),
            ("missing from the model root: ['README.md']", {k: v for k, v in SINGLE.items() if k != "README.md"}, SINGLE),
        ]
        for fragment, files, listed in cases:
            with self.subTest(fragment=fragment):
                root, inventory = self.model(files, listed)
                with self.assertRaises(CustodyError) as caught:
                    self.verify(root, inventory)
                self.assertIn(fragment, str(caught.exception))
        # A member symlinked to identical bytes, a symlinked member directory, a FIFO and an empty listed directory.
        root, inventory = self.model()
        os.replace(Path(root) / "README.md", self.base / "README.md")
        os.symlink(self.base / "README.md", Path(root) / "README.md")
        with self.assertRaisesRegex(CustodyError, "'README.md' is a symlink"):
            self.verify(root, inventory)
        root, inventory = self.model()
        shutil.move(str(Path(root) / ".eval_results"), str(self.base / "elsewhere"))
        os.symlink(self.base / "elsewhere", Path(root) / ".eval_results")
        with self.assertRaisesRegex(CustodyError, "'.eval_results' is a symlink"):
            self.verify(root, inventory)
        root, inventory = self.model()
        os.mkfifo(Path(root) / "pipe")
        with self.assertRaisesRegex(CustodyError, "'pipe' is not a regular file or directory"):
            self.verify(root, inventory)

    def test_model_root_must_be_explicit_and_resolved(self):
        root, inventory = self.model()
        link = self.base / "link"
        os.symlink(root, link)
        nested = self.base / "parent-link"
        os.symlink(self.base, nested)
        cases = [("traverses a symlink", str(link)), ("traverses a symlink", str(nested / Path(root).name)),
                 ("absolute normalized", os.path.relpath(root)), ("absolute normalized", root + "/"),
                 ("absolute normalized", root + "/../" + Path(root).name), ("not a directory", str(Path(root) / "config.json")),
                 ("cannot be inspected", str(self.base / "absent"))]
        for fragment, path in cases:
            with self.subTest(path=path), self.assertRaises(CustodyError) as caught:
                self.session(path, inventory)
            self.assertIn(fragment, str(caught.exception))
        s = self.session(root, inventory)
        self.bound(s)
        os.rename(root, root + ".moved")
        os.symlink(root + ".moved", root)
        with self.assertRaisesRegex(CustodyError, "traverses a symlink"):
            s.verify_before_load()

    def test_output_and_evidence_paths_are_fresh(self):
        root, inventory = self.model()
        export = self.fresh("export")
        export.mkdir()
        with self.assertRaisesRegex(CustodyError, "fresh path"):
            custody.Session(root, inventory, str(export), self.producer, 60)
        export = self.fresh("export")
        Path(str(export) + ".custody").mkdir()
        with self.assertRaisesRegex(CustodyError, "already exists"):
            custody.Session(root, inventory, str(export), self.producer, 60)
        with self.assertRaisesRegex(CustodyError, "absolute normalized"):
            custody.Session(root, inventory, "relative/export", self.producer, 60)


class ChangeTests(Case):
    def test_accepted_export_binds_admission_metadata_receipt_and_producer(self):
        s, export, meta = self.admitted()
        metadata_sha = self.accept(s, export, meta)
        data = (export / "metadata.json").read_bytes()
        self.assertEqual(sha(data), metadata_sha)
        self.assertEqual(data, json.dumps(meta, indent=2).encode())
        admission = json.loads((export / "admission.json").read_text())
        final = json.loads(data)
        self.assertEqual(admission["runtime"], {k: v for k, v in final["runtime"].items() if k != "attention_config"})
        receipt = self.receipt(s)
        self.assertEqual((receipt["status"], receipt["metadata_sha256"], receipt["admission_sha256"]),
                         ("accepted", metadata_sha, sha((export / "admission.json").read_bytes())))
        self.assertEqual(receipt["custody_sha256"], sha(custody.canonical(final["runtime"]["custody"])))
        self.assertEqual(receipt["after_load"]["model_root"]["members"], admission["runtime"]["custody"]["before_load"]["members"])
        self.assertEqual(receipt["payloads"], {"logits.f32": {"bytes": 16, "sha256": sha(b"\0" * 16)}})
        self.assertEqual(receipt["export_entries_before_metadata"], ["admission.json", "logits.f32"])
        self.assertEqual(sorted(os.listdir(export)), ["admission.json", "logits.f32", "metadata.json"])
        self.assertEqual(sorted(os.listdir(s.evidence)), ["producer", "receipt.json", "sources.txt"])
        record = final["runtime"]["custody"]
        self.assertEqual(record["evidence_directory"], f"../{Path(s.evidence).name}")
        for name, (path, data) in self.producer.items():
            copy = Path(s.evidence) / record["producer"][name]["copy"]
            self.assertEqual((copy.read_bytes(), record["producer"][name]["sha256"]), (data, sha(data)))
        manifest = (Path(s.evidence) / "sources.txt").read_bytes()
        self.assertEqual(record["sources"]["manifest_sha256"], sha(manifest))
        self.assertEqual(manifest.decode().splitlines()[1].split()[2], "transformers/models/modeling.py")
        self.assertEqual(record["sources"]["classes"]["model"]["source"], "transformers/models/modeling.py")
        self.assertEqual(record["checkpoint"], {"layout": "single-file safetensors", "index": None, "weights": [
            {"file": "model.safetensors", "size": 64, "sha256": sha(SINGLE["model.safetensors"])}]})
        self.assertTrue(record["processor_load_completed_before_admission"])

    def test_explicit_load_arguments_are_recorded_and_stable(self):
        s, export, meta = self.admitted()
        root = s.root
        self.assertEqual(FakeAutoProcessor.calls, [(Path(root), {"local_files_only": True})])
        self.assertEqual(FakeModel.calls, [(Path(root), {"dtype": FLOAT32, "device_map": "cpu", "local_files_only": True,
                                                         "attn_implementation": {"vision_config": "sdpa", "text_config": "eager"}})])
        loads = meta["runtime"]["custody"]["loads"]
        self.assertEqual(loads["model"], {"callable": f"{__name__}.FakeModel.from_pretrained", "pretrained_model_name_or_path": root,
                                          "kwargs": {"attn_implementation": {"text_config": "eager", "vision_config": "sdpa"},
                                                     "device_map": "cpu", "dtype": "torch.float32", "local_files_only": True}})
        self.assertEqual(loads["processor"]["kwargs"], {"local_files_only": True})
        self.assertEqual(json.loads((export / "admission.json").read_text())["runtime"]["custody"]["loads"], loads)
        # Every deviation is refused before the loader runs.
        root2, inventory = self.model()
        t = self.session(root2, inventory)
        self.bound(t)
        with self.assertRaisesRegex(CustodyError, "after the pre-load pass"):
            t.declare("processor", FakeAutoProcessor.from_pretrained, Path(root2), local_files_only=True)
        t.verify_before_load()
        with self.assertRaisesRegex(CustodyError, "other than the verified model root"):
            t.declare("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        with self.assertRaisesRegex(CustodyError, "local_files_only=True"):
            t.declare("processor", FakeAutoProcessor.from_pretrained, Path(root2))
        with self.assertRaisesRegex(CustodyError, "unsupported load-argument type"):
            t.declare("processor", FakeAutoProcessor.from_pretrained, Path(root2), local_files_only=True, extra=object())
        t.declare("processor", FakeAutoProcessor.from_pretrained, Path(root2), local_files_only=True)
        t.declare("model", FakeModel.from_pretrained, Path(root2), dtype=FLOAT32, local_files_only=True)
        with self.assertRaisesRegex(CustodyError, "declared twice"):
            t.declare("model", FakeModel.from_pretrained, Path(root2), dtype=FLOAT32, local_files_only=True)
        calls = len(FakeAutoProcessor.calls)
        with self.assertRaisesRegex(CustodyError, "differ from the declaration"):
            t.load("processor", FakeAutoProcessor.from_pretrained, Path(root2), local_files_only=True, trust_remote_code=True)
        with self.assertRaisesRegex(CustodyError, "differ from the declaration"):
            t.load("processor", FakeModel.from_pretrained, Path(root2), local_files_only=True)
        with self.assertRaisesRegex(CustodyError, "never declared"):
            t.load("tokenizer", FakeAutoProcessor.from_pretrained, Path(root2), local_files_only=True)
        with self.assertRaisesRegex(CustodyError, "only after admission.json"):
            t.load("model", FakeModel.from_pretrained, Path(root2), dtype=FLOAT32, local_files_only=True)
        self.assertEqual(len(FakeAutoProcessor.calls), calls)
        t.load("processor", FakeAutoProcessor.from_pretrained, str(root2), local_files_only=True)  # A str and a Path name the same root.
        with self.assertRaisesRegex(CustodyError, "already happened"):
            t.load("processor", FakeAutoProcessor.from_pretrained, Path(root2), local_files_only=True)

    def test_changes_during_load_refuse_without_metadata(self):
        def root_of():
            return Path(FakeModel.calls[-1][0])
        def rewrite_identical():
            path = root_of() / "config.json"
            info = path.stat()
            path.write_bytes(path.read_bytes())
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000))
        def replace_identical():
            path = root_of() / "model.safetensors"
            shutil.copy2(path, str(path) + ".copy")
            os.replace(str(path) + ".copy", path)
        def touch():
            path = root_of() / "tokenizer.json"
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
        def chmod():
            os.chmod(root_of() / "README.md", 0o600)
        def grow():
            with open(root_of() / "model.safetensors", "ab") as stream:
                stream.write(b"x")
        def corrupt():
            with open(root_of() / "model.safetensors", "r+b") as stream:
                stream.write(b"\xff")
        def extra_shard():
            (root_of() / "model-00001-of-00001.safetensors").write_bytes(b"x")
        def remove_and_restore():
            path = root_of() / "chat_template.jinja"
            data = path.read_bytes()
            path.unlink()
            (root_of() / "chat_template.jinja").write_bytes(data)
        def link():
            path = root_of() / "README.md"
            os.replace(path, self.base / "moved-readme")
            os.symlink(self.base / "moved-readme", path)
        cases = [(rewrite_identical, "file config.json: & mtime_ns"), (replace_identical, "file model.safetensors: & inode"),
                 (touch, "file tokenizer.json: & mtime_ns"), (chmod, "file README.md: & mode"),
                 (grow, "model.safetensors has 65 bytes"), (corrupt, "model.safetensors SHA-256"),
                 (extra_shard, "unexpected file 'model-00001-of-00001.safetensors'"),
                 (remove_and_restore, "file chat_template.jinja:"), (link, "'README.md' is a symlink")]
        for hook, fragment in cases:
            with self.subTest(hook=hook.__name__):
                receipt = self.refused_after_load(hook, fragment)
                self.assertIsNotNone(receipt["admission_sha256"])
                self.assertEqual(receipt["loads"][-1]["role"], "model")

    def test_changed_admission_payload_custody_producer_or_sources_refuse(self):
        def admission(s, export, meta):
            path = export / "admission.json"
            path.write_bytes(path.read_bytes() + b" ")
        def payload(s, export, meta):
            (export / "logits.f32").write_bytes(b"\1" * 16)
        def custody_record(s, export, meta):
            meta["runtime"]["custody"]["model_root"] = "/elsewhere"
        def runtime(s, export, meta):
            meta["runtime"]["threads"] = 7
        def inputs(s, export, meta):
            meta["inputs"]["input_ids"]["sha256"] = "3" * 64
        def producer(s, export, meta):
            self.producer["verification/cpu_custody.py"][0].write_bytes(b"# edited helper\n")
        def source(s, export, meta):
            Path(self.modules["transformers.models.modeling"].__file__).write_bytes(b"class Model: edited\n")
        cases = [(admission, "admission.json in the export directory changed after it was written"), (payload, "Payload logits.f32 SHA-256"), (custody_record, "Final runtime differs"),
                 (runtime, "Final runtime differs"), (inputs, "inputs differs"), (producer, "Producer verification/cpu_custody.py changed"),
                 (source, "Library sources changed after admission: ['transformers/models/modeling.py']")]
        for change, fragment in cases:
            with self.subTest(change=change.__name__):
                s, export, meta = self.admitted()
                change(s, export, meta)
                with self.assertRaises(CustodyError) as caught:
                    self.accept(s, export, meta)
                self.assertIn(fragment, str(caught.exception))
                self.assertFalse(os.path.lexists(export / "metadata.json"))
                self.setUp_library_again()

    def test_recorded_model_source_digest_is_joined_to_the_manifest(self):
        s, export, meta = self.admitted()
        recorded = s.sources["summary"]["classes"]["model"]
        self.assertEqual(recorded["sha256"], sha(b"class Model: pass\n"))
        s.require_source("model", recorded["sha256"], "runtime model_source_sha256")
        with self.assertRaisesRegex(CustodyError, "runtime model_source_sha256 0{64} differs from the recorded transformers/models/modeling.py SHA-256"):
            s.require_source("model", "0" * 64, "runtime model_source_sha256")
        with self.assertRaisesRegex(CustodyError, "No recorded library source names the tokenizer class"):
            s.require_source("tokenizer", recorded["sha256"], "tokenizer source")

    def setUp_library_again(self):
        Path(self.modules["transformers.models.modeling"].__file__).write_bytes(b"class Model: pass\n")
        self.producer["verification/cpu_custody.py"][0].write_bytes(self.producer["verification/cpu_custody.py"][1])

    def test_failed_metadata_write_leaves_a_refused_receipt_and_no_metadata(self):
        s, export, meta = self.admitted()
        original = custody.replace_durably

        def failing(directory, name, data):
            if name == "metadata.json":
                (Path(directory) / ".metadata.json.partial").write_bytes(data[:10])
                raise OSError(28, "No space left on device")
            return original(directory, name, data)

        with mock.patch.object(custody, "replace_durably", failing), self.assertRaises(OSError):
            self.accept(s, export, meta)
        self.assertEqual(sorted(os.listdir(export)), ["admission.json", "logits.f32"])
        receipt = self.receipt(s)
        self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("refused", None))
        self.assertIn("No space left on device", receipt["error"])

    def test_loader_failure_and_late_refusal(self):
        root, inventory = self.model()
        s = self.session(root, inventory)
        self.bound(s)
        s.verify_before_load()
        s.declare("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)

        def broken(path, **kwargs):
            raise RuntimeError("loader exploded")

        s.declare("model", broken, Path(root), local_files_only=True)
        s.load("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.record_sources(s.loaded_sources(self.modules), {})
        s.admission = {"sha256": "0" * 64, "custody_sha256": "0" * 64, "bytes": 2, "identity": None, "data": b"{}"}  # Admission written elsewhere in this test.
        with self.assertRaisesRegex(RuntimeError, "loader exploded") as caught:
            s.load("model", broken, Path(root), local_files_only=True)
        s.refuse(caught.exception)
        receipt = self.receipt(s)
        self.assertEqual((receipt["status"], receipt["loads"][-1]["completed"]), ("refused", False))
        self.assertIn("RuntimeError: loader exploded", receipt["error"])
        with self.assertRaisesRegex(CustodyError, "completed model load"):
            s.recheck({})
        # An accepted receipt is never replaced by a later refusal.
        t, export, meta = self.admitted()
        self.accept(t, export, meta)
        self.assertIsNone(t.refuse(RuntimeError("after acceptance")))
        self.assertEqual(self.receipt(t)["status"], "accepted")


class BoundTests(Case):
    def test_deadline_and_bounds_are_validated(self):
        for seconds in (0, -1, 3600.5, float("nan"), float("inf"), True, "60"):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(CustodyError, "deadline must be within"):
                custody.Deadline(seconds, "test")
        root, inventory = self.model()
        with self.assertRaisesRegex(CustodyError, "deadline must be within"):
            self.session(root, inventory, seconds=7200)

    def test_hashing_stops_at_the_deadline(self):
        root, inventory = self.model()
        s = self.session(root, inventory, seconds=5)
        self.bound(s)
        ticks = iter(range(10 ** 6))
        with mock.patch.object(custody, "clock", lambda: float(next(ticks))), mock.patch.object(custody, "BLOCK", 4):
            with self.assertRaisesRegex(CustodyError, "5-second custody deadline for the pre-load pass expired during"):
                s.verify_before_load()
        self.assertIsNone(s.before)

    def test_streaming_is_bounded_and_detects_concurrent_change(self):
        root, inventory = self.model()
        member = {"file": "model.safetensors", "size": 64, "sha256": sha(SINGLE["model.safetensors"])}
        path = Path(root) / "model.safetensors"
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with mock.patch.object(custody, "BLOCK", 8):
                reads = []

                class Recording:
                    def __init__(self, handle):
                        self.handle = handle

                    def readinto(self, view):
                        reads.append(len(view))
                        return self.handle.readinto(view)

                with open(path, "rb", buffering=0) as handle:
                    digest, _ = custody.stream(Recording(handle), 64, custody.Deadline(60, "test"), "weights")
                self.assertEqual(digest, member["sha256"])
                self.assertEqual(reads, [8] * 8 + [1])  # Eight full blocks, then one byte past the pinned extent to observe end of file.
                with open(path, "rb", buffering=0) as handle, self.assertRaisesRegex(CustodyError, "truncated: 64 of 65 bytes"):
                    custody.stream(handle, 65, custody.Deadline(60, "test"), "weights")
                with open(path, "rb", buffering=0) as handle, self.assertRaisesRegex(CustodyError, "grew beyond its 63 pinned bytes"):
                    custody.stream(handle, 63, custody.Deadline(60, "test"), "weights")
                observed, digest, _ = custody.hash_member(fd, member, custody.Deadline(60, "test"))
                self.assertEqual((digest, observed["size"]), (member["sha256"], 64))
                calls = []

                def mutating_clock():
                    calls.append(1)
                    if len(calls) == 4:
                        with open(path, "ab") as stream:
                            stream.write(b"!")
                    return 0.0

                with mock.patch.object(custody, "clock", mutating_clock), self.assertRaisesRegex(CustodyError, "grew beyond its 64 pinned bytes"):
                    custody.hash_member(fd, member, custody.Deadline(60, "test"))
                path.write_bytes(SINGLE["model.safetensors"])
                calls.clear()

                def touching_clock():
                    calls.append(1)
                    if len(calls) == 4:
                        os.utime(path, ns=(0, 10 ** 9))
                    return 0.0

                with mock.patch.object(custody, "clock", touching_clock), self.assertRaisesRegex(CustodyError, "changed while it was hashed"):
                    custody.hash_member(fd, member, custody.Deadline(60, "test"))
        finally:
            os.close(fd)

    def test_module_sources_name_loaded_packages_and_libraries(self):
        lib = self.base / "site" / "torch" / "lib"
        lib.mkdir(parents=True)
        (self.base / "site" / "torch" / "__init__.py").write_bytes(b"# torch\n")
        (lib / "libtorch_cpu.so").write_bytes(b"\x7fELF")
        (lib / "README").write_bytes(b"not a library")
        modules = dict(self.modules, torch=types.SimpleNamespace(__file__=str(self.base / "site" / "torch" / "__init__.py")),
                       unrelated=types.SimpleNamespace(__file__=str(self.base / "site" / "unrelated.py")),
                       **{"transformers.lazy": types.SimpleNamespace()})
        files = custody.module_sources(modules, library_directories=[lib])
        self.assertEqual(sorted(files), ["torch/__init__.py", "torch/lib/libtorch_cpu.so", "transformers/__init__.py", "transformers/models/modeling.py"])
        records = custody.hash_sources(files, custody.Deadline(60, "test"))
        self.assertEqual(records["torch/lib/libtorch_cpu.so"], {"size": 4, "sha256": sha(b"\x7fELF")})
        library_digest = sha(b"\x7fELF")
        self.assertEqual(custody.source_manifest(records).decode().splitlines()[1], f"{library_digest}  4  torch/lib/libtorch_cpu.so")
        # A library directory under a loaded package that has disappeared is refused rather than contributing no files.
        with self.assertRaisesRegex(CustodyError, "Library directory torch/lib-gone of a loaded package is not a directory"):
            custody.module_sources(modules, library_directories=[lib.parent / "lib-gone"])
        self.assertEqual(custody.module_sources(modules, library_directories=[self.base / "elsewhere" / "lib"]), custody.module_sources(modules))


class ProducerBindingTests(Case):
    """The retained producer bytes must compile to the code objects the process is executing, before the pre-load pass."""

    def executing(self, **changes):
        codes = {name: compile(data, str(path), "exec", dont_inherit=True) for name, (path, data) in self.producer.items()}
        return dict(codes, **{name: compile(source, "elsewhere", "exec", dont_inherit=True) for name, source in changes.items()})

    def test_retained_bytes_must_compile_to_the_executing_code(self):
        root, inventory = self.model()
        s = self.session(root, inventory)
        with self.assertRaisesRegex(CustodyError, "after begin"):
            s.bind_execution(self.executing())
        s.begin()
        with self.assertRaisesRegex(CustodyError, "after begin\\(\\) and bind_execution\\(\\)"):
            s.verify_before_load()
        exporter = "verification/export_logits.py"
        cases = [({exporter: b"# exporter bytes\nthreads = 3\n"}, None, f"Retained producer {exporter} (SHA-256 {sha(self.producer[exporter][1])}) does not compile"),
                 ({"verification/cpu_custody.py": b"# helper bytes\nx = 1\n"}, None, "Retained producer verification/cpu_custody.py"),
                 ({}, lambda codes: codes.pop(exporter), "exactly the retained producer files"),
                 ({}, lambda codes: codes.update(extra=codes[exporter]), "exactly the retained producer files"),
                 ({}, lambda codes: codes.update({exporter: None}), "is not a code object")]
        for changes, edit, fragment in cases:
            with self.subTest(fragment=fragment):
                codes = self.executing(**{k: v for k, v in changes.items()})
                if edit:
                    edit(codes)
                with self.assertRaises(CustodyError) as caught:
                    s.bind_execution(codes)
                self.assertIn(fragment, str(caught.exception))
                self.assertEqual((s.status, s.execution), ("begun", None))
        compiled = s.bind_execution(self.executing())
        self.assertEqual(compiled, self.executing())
        self.assertEqual((s.status, s.execution["files"], s.execution["optimize"]), ("bound", sorted(self.producer), sys.flags.optimize))
        with self.assertRaisesRegex(CustodyError, "bound once"):
            s.bind_execution(self.executing())
        s.verify_before_load()

    def test_mismatch_keeps_the_retained_bytes_with_a_refused_receipt(self):
        root, inventory = self.model()
        s = self.session(root, inventory)
        s.begin()
        with self.assertRaises(CustodyError) as caught:
            s.bind_execution(self.executing(**{"verification/export_logits.py": b"threads = 8\n"}))
        s.refuse(caught.exception)
        receipt = self.receipt(s)
        self.assertEqual((receipt["status"], receipt["producer_execution"], receipt["before_load"]), ("refused", None, None))
        self.assertIn("does not compile to the code this process is executing", receipt["error"])
        self.assertEqual((Path(s.evidence) / "producer" / "export_logits.py").read_bytes(), self.producer["verification/export_logits.py"][1])
        self.assertFalse(os.path.lexists(s.output))
        # Bytes that do not compile at all are refused the same way.
        self.producer["verification/export_logits.py"] = (self.producer["verification/export_logits.py"][0], b"def (\n")
        t = self.session(root, inventory)
        t.begin()
        with self.assertRaisesRegex(CustodyError, "does not compile: SyntaxError"):
            t.bind_execution({name: compile(b"", "x", "exec") for name in self.producer})

    def test_loaded_helper_binds_to_its_file_and_not_to_an_edit(self):
        data = (ROOT / "verification/cpu_custody.py").read_bytes()
        self.assertIsInstance(custody.MODULE_CODE, types.CodeType)
        self.assertEqual(compile(data, "any name", "exec", dont_inherit=True), custody.MODULE_CODE)
        edited = data.replace(b"BLOCK = 8 * MIB", b"BLOCK = 4 * MIB")
        self.assertNotEqual(edited, data)
        self.assertNotEqual(compile(edited, "any name", "exec", dont_inherit=True), custody.MODULE_CODE)

    def test_main_script_compilation_equals_the_retained_compilation(self):
        """A script run as __main__ by this interpreter captures the code its retained bytes compile to; an edit after start-up does not."""
        script = self.base / "main_probe.py"
        script.write_bytes(b"import sys, time\n"
                           b"CODE = sys._getframe(0).f_code\n"
                           b"open(sys.argv[1], 'wb').write(open(__file__, 'rb').read().replace(b'threads = 8', b'threads = 3'))\n"
                           b"threads = 8\n"
                           b"data = open(__file__, 'rb').read()\n"
                           b"original = open(sys.argv[2], 'rb').read()\n"
                           b"print(compile(original, __file__, 'exec', dont_inherit=True) == CODE, compile(data, __file__, 'exec', dont_inherit=True) == CODE)\n")
        shutil.copyfile(script, self.base / "main_probe.original")
        result = subprocess.run([sys.executable, "-I", "-B", str(script), str(script), str(self.base / "main_probe.original")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual((result.returncode, result.stdout.split()), (0, ["True", "False"]), result.stderr)


class RetainedEvidenceTests(Case):
    """Retained producer copies, sources.txt and admission.json are re-authenticated, separately from the original files."""

    def corruptions(self):
        def rewrite(path):  # Same extent, other bytes, the original mtime restored.
            info = path.stat()
            path.write_bytes(bytes(b ^ 1 for b in path.read_bytes()))
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        def append(path):
            with open(path, "ab") as stream:
                stream.write(b"#")
        def identical_copy(path):
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)
        def symlink(path):
            moved = self.fresh("moved")
            os.replace(path, moved)
            os.symlink(moved, path)
        def fifo(path):
            path.unlink()
            os.mkfifo(path)
        def remove(path):
            path.unlink()
        def hard_link(path):
            os.link(path, self.fresh("hard-link"))
        def touch(path):
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
        return [(rewrite, "(changed after it was written|differs from the committed)"), (append, "changed after it was written: .*size"),
                (identical_copy, "changed after it was written"), (symlink, "cannot be opened without following links"),
                (fifo, "is not a regular file"), (remove, "cannot be opened without following links"),
                (hard_link, "changed after it was written: .*links"), (touch, "changed after it was written: .*mtime_ns")]

    def test_corrupted_retained_evidence_is_refused_without_metadata(self):
        for target in ("producer/export_logits.py", "producer/cpu_custody.py", "sources.txt"):
            for corrupt, fragment in self.corruptions():
                with self.subTest(target=target, corruption=corrupt.__name__):
                    s, export, meta = self.admitted()
                    corrupt(Path(s.evidence) / target)
                    with bounded(), self.assertRaises(CustodyError) as caught:
                        self.accept(s, export, meta)
                    s.refuse(caught.exception)
                    self.assertRegex(str(caught.exception), f"{target}.* {fragment}" if "cannot" not in fragment else f"{target} {fragment}")
                    self.assertFalse(os.path.lexists(export / "metadata.json"))
                    receipt = self.receipt(s)
                    self.assertEqual((receipt["status"], receipt["metadata_sha256"], receipt["retained_evidence"]), ("refused", None, None))
                    # The originals were untouched: this is not the original-source mutation control.
                    for name, (path, data) in self.producer.items():
                        self.assertEqual(path.read_bytes(), data)

    def test_replaced_evidence_or_admission_is_refused(self):
        def evidence_copy(s, export):
            os.rename(s.evidence, s.evidence + ".moved")
            shutil.copytree(s.evidence + ".moved", s.evidence)
        def evidence_link(s, export):
            os.rename(s.evidence, s.evidence + ".moved")
            os.symlink(s.evidence + ".moved", s.evidence)
        def admission_fifo(s, export):
            (export / "admission.json").unlink()
            os.mkfifo(export / "admission.json")
        def admission_copy(s, export):
            data = (export / "admission.json").read_bytes()
            (export / "admission.json").unlink()
            (export / "admission.json").write_bytes(data)
        def export_copy(s, export):
            os.rename(export, str(export) + ".moved")
            shutil.copytree(str(export) + ".moved", export)
        cases = [(evidence_copy, "the custody evidence directory .* was replaced after it was created"),
                 (evidence_link, "the custody evidence directory cannot be opened without following links"),
                 (admission_fifo, "admission.json in the export directory is not a regular file"),
                 (admission_copy, "admission.json in the export directory changed after it was written"),
                 (export_copy, "The --output directory .* was replaced after admission")]
        for change, fragment in cases:
            with self.subTest(change=change.__name__):
                s, export, meta = self.admitted()
                change(s, export)
                with bounded(), self.assertRaises(CustodyError) as caught:
                    self.accept(s, export, meta)
                self.assertRegex(str(caught.exception), fragment)
                self.assertFalse(os.path.lexists(export / "metadata.json"))

    def test_corruption_before_admission_refuses_before_the_model_load(self):
        root, inventory = self.model()
        s = self.bound(self.session(root, inventory))
        s.verify_before_load()
        s.declare("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.declare("model", FakeModel.from_pretrained, Path(root), local_files_only=True)
        s.load("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.record_sources(s.loaded_sources(self.modules), {})
        (Path(s.evidence) / "sources.txt").write_bytes(b"changed source manifest\n")
        export = Path(s.output)
        export.mkdir()
        meta = {"runtime": {"custody": s.admission_record()}}
        with self.assertRaisesRegex(CustodyError, "sources.txt in the custody evidence directory changed after it was written"):
            s.write_admission(export, meta)
        self.assertEqual(os.listdir(export), [])
        with self.assertRaisesRegex(CustodyError, "only after admission.json"):
            s.load("model", FakeModel.from_pretrained, Path(root), local_files_only=True)
        self.assertEqual(FakeModel.calls, [])

    def test_corruption_during_acceptance_withdraws_metadata(self):
        s, export, meta = self.admitted()
        original = custody.replace_durably

        def corrupting(directory, name, data):
            result = original(directory, name, data)
            if name == "metadata.json":
                (Path(s.evidence) / "producer" / "cpu_custody.py").write_bytes(b"# changed during acceptance\n")
            return result

        with mock.patch.object(custody, "replace_durably", corrupting), self.assertRaisesRegex(CustodyError, "producer/cpu_custody.py"):
            self.accept(s, export, meta)
        self.assertEqual(sorted(os.listdir(export)), ["admission.json", "logits.f32"])
        receipt = self.receipt(s)
        self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("refused", None))
        self.assertIn("after metadata.json was written", receipt["error"])

    def test_accepted_receipt_names_every_retained_file_by_its_bytes(self):
        s, export, meta = self.admitted()
        self.accept(s, export, meta)
        receipt_bytes = (Path(s.evidence) / "receipt.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        self.assertEqual(s.receipt_sha256, sha(receipt_bytes))
        retained = receipt["retained_evidence"]
        self.assertEqual(sorted(retained), [f"../{export.name}/admission.json", "producer/cpu_custody.py", "producer/export_logits.py", "sources.txt"])
        for name, record in retained.items():
            data = (Path(s.evidence) / name).read_bytes()
            self.assertEqual(record, {"bytes": len(data), "sha256": sha(data)})
        self.assertEqual(retained[f"../{export.name}/admission.json"]["sha256"], receipt["admission_sha256"])
        self.assertEqual(retained["sources.txt"]["sha256"], receipt["sources"]["manifest_sha256"])
        for name in self.producer:
            self.assertEqual(retained["producer/" + Path(name).name]["sha256"], receipt["producer"][name]["sha256"])
        self.assertEqual(receipt["producer_execution"], json.loads((export / "metadata.json").read_text())["runtime"]["custody"]["producer_execution"])
        self.assertEqual(receipt["producer_execution"]["rule"], custody.EXECUTION)


class BoundedReadTests(Case):
    """Every file custody hashes is opened without blocking; a FIFO or other special file is refused at once, never waited on."""

    CHILD = ("import importlib.util, sys, time\n"
             "spec = importlib.util.spec_from_file_location('custody', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
             "start = time.monotonic()\n"
             "try:\n"
             "    m.hash_sources({'transformers/models/modeling.py': sys.argv[2]}, m.Deadline(0.02, 'FIFO test'))\n"
             "except m.CustodyError as error:\n"
             "    print('refused', round(time.monotonic() - start, 3), error)\n")

    def test_library_source_replaced_by_fifo_is_refused_within_bounds(self):
        files = custody.module_sources(self.modules)
        source = Path(self.modules["transformers.models.modeling"].__file__)
        self.assertEqual(files["transformers/models/modeling.py"], source)
        source.unlink()
        os.mkfifo(source)
        with bounded():
            with self.assertRaisesRegex(CustodyError, "Library source transformers/models/modeling.py is not a regular file"):
                custody.hash_sources(files, custody.Deadline(0.02, "FIFO test"))
        self.assertEqual(custody.module_sources(self.modules)["transformers/models/modeling.py"], source)  # Listed afresh, never omitted.
        started = time.monotonic()
        result = subprocess.run([sys.executable, "-I", "-B", "-c", self.CHILD, str(ROOT / "verification/cpu_custody.py"), str(source)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, "^refused [0-9.]+ Library source transformers/models/modeling.py is not a regular file")
        self.assertLess(float(result.stdout.split()[1]), 1.0)
        self.assertLess(time.monotonic() - started, 60)

    def test_session_refuses_special_or_linked_sources_and_producers(self):
        def fifo(path):
            path.unlink()
            os.mkfifo(path)
        def link(path):
            moved = self.fresh("moved")
            os.replace(path, moved)
            os.symlink(moved, path)
        modeling = lambda: Path(self.modules["transformers.models.modeling"].__file__)
        helper = lambda: self.producer["verification/cpu_custody.py"][0]
        cases = [(modeling, fifo, "Library source transformers/models/modeling.py is not a regular file"),
                 (modeling, link, "Library source transformers/models/modeling.py cannot be opened without following links"),
                 (helper, fifo, "Producer verification/cpu_custody.py is not a regular file"),
                 (helper, link, "Producer verification/cpu_custody.py cannot be opened without following links")]
        for target, change, fragment in cases:
            with self.subTest(fragment=fragment):
                s, export, meta = self.admitted()
                path = target()
                data = path.read_bytes()
                change(path)
                try:
                    with bounded(), self.assertRaises(CustodyError) as caught:
                        self.accept(s, export, meta)
                    s.refuse(caught.exception)
                    self.assertIn(fragment, str(caught.exception))
                    self.assertEqual(self.receipt(s)["status"], "refused")
                    self.assertFalse(os.path.lexists(export / "metadata.json"))
                finally:
                    path.unlink()
                    path.write_bytes(data)
        # Before admission, a FIFO source refuses the manifest itself.
        root, inventory = self.model()
        s = self.bound(self.session(root, inventory))
        s.verify_before_load()
        fifo(modeling())
        try:
            with bounded(), self.assertRaisesRegex(CustodyError, "is not a regular file"):
                s.record_sources(s.loaded_sources(self.modules), {})
            self.assertFalse(os.path.lexists(Path(s.evidence) / "sources.txt"))
        finally:
            modeling().unlink()
            modeling().write_bytes(b"class Model: pass\n")

    def test_source_listing_and_hashing_bounds(self):
        library = Path(self.modules["transformers"].__file__).parent
        (library / "package_dir").mkdir()
        modules = dict(self.modules, **{"transformers.pseudo": types.SimpleNamespace(__file__="_ops.py"),
                                        "transformers.namespace": types.SimpleNamespace(__file__=None), "transformers.builtin": types.SimpleNamespace(),
                                        "transformers.package_dir": types.SimpleNamespace(__file__=str(library / "package_dir")),
                                        "transformers.gone": types.SimpleNamespace(__file__=str(library / "gone.py"))})
        # A module naming no file or a relative pseudo-path is not a source; every absolute file is listed whatever is there now, so hashing refuses it.
        listed = custody.module_sources(modules)
        self.assertEqual(sorted(listed), ["transformers/__init__.py", "transformers/gone.py", "transformers/models/modeling.py", "transformers/package_dir"])
        for name, fragment in (("transformers/gone.py", "cannot be opened without following links: No such file or directory"),
                               ("transformers/package_dir", "is not a regular file")):
            with self.subTest(name=name), self.assertRaisesRegex(CustodyError, f"Library source {name} {fragment}"):
                custody.hash_sources({name: listed[name]}, custody.Deadline(60, "test"))
        files = custody.module_sources(self.modules)
        with mock.patch.object(custody, "SOURCE_BYTES", 20):
            with self.assertRaisesRegex(CustodyError, "Library source transformers/models/modeling.py exceeds 5 bytes"):
                custody.hash_sources(files, custody.Deadline(60, "test"))
        calls = []

        def appending_clock():
            calls.append(1)
            if len(calls) == 3:
                with open(files["transformers/__init__.py"], "ab") as stream:
                    stream.write(b"#")
            return 0.0

        with mock.patch.object(custody, "clock", appending_clock), self.assertRaisesRegex(CustodyError, "grew beyond|changed while it was hashed"):
            custody.hash_sources(files, custody.Deadline(60, "test"))

    def test_loaded_source_that_disappears_is_refused_not_omitted(self):
        modeling = Path(self.modules["transformers.models.modeling"].__file__)
        classes = {"model": ("transformers.models.modeling.Model", str(modeling))}
        def removed():
            modeling.unlink()
        def directory():
            modeling.unlink()
            modeling.mkdir()
        for change, fragment in ((removed, "cannot be opened without following links: No such file or directory"), (directory, "is not a regular file")):
            with self.subTest(change=change.__name__):
                root, inventory = self.model()
                s = self.bound(self.session(root, inventory))
                s.verify_before_load()
                change()
                try:
                    with self.assertRaisesRegex(CustodyError, f"Library source transformers/models/modeling.py {fragment}"):
                        s.record_sources(s.loaded_sources(self.modules), classes)
                finally:
                    if modeling.is_dir():
                        modeling.rmdir()
                    modeling.write_bytes(b"class Model: pass\n")
                self.assertIsNone(s.sources)
                self.assertFalse(os.path.lexists(Path(s.evidence) / "sources.txt"))
        # A module first imported during the model load is hashed at acceptance; if its file is gone by then the export is refused, not accepted without it.
        late, data = modeling.parent.parent / "late.py", b"# imported during the load\n"
        modules = dict(self.modules, **{"transformers.late": types.SimpleNamespace(__file__=str(late))})
        for present in (True, False):
            with self.subTest(present=present):
                late.write_bytes(data)
                s, export, meta = self.admitted(hook=None if present else late.unlink)
                if present:
                    s.accept(export, meta, {"logits.f32": meta["logits"]["sha256"]}, s.loaded_sources(modules))
                    self.assertEqual(self.receipt(s)["after_load"]["imported_after_admission"],
                                     [{"file": "transformers/late.py", "size": len(data), "sha256": sha(data)}])
                    continue
                with self.assertRaisesRegex(CustodyError, "Library source transformers/late.py cannot be opened without following links: No such file or directory"):
                    s.accept(export, meta, {"logits.f32": meta["logits"]["sha256"]}, s.loaded_sources(modules))
                self.assertFalse(os.path.lexists(export / "metadata.json"))

    def test_exporter_metadata_digests_are_bounded_and_may_follow_links(self):
        root, inventory = self.model()
        s = self.session(root, inventory)
        target = self.fresh("prompt")
        target.write_bytes(b"prompt")
        link = self.fresh("prompt-link")
        os.symlink(target, link)
        self.assertEqual((s.file_sha256(target), s.file_sha256(link)), (sha(b"prompt"), sha(b"prompt")))
        pipe = self.fresh("pipe")
        os.mkfifo(pipe)
        os.symlink(pipe, self.fresh("pipe-link"))
        for path in (pipe, self.base / f"pipe-link{self.count}", self.base):
            with self.subTest(path=path), bounded(), self.assertRaisesRegex(CustodyError, "is not a regular file"):
                s.file_sha256(path)
        with self.assertRaisesRegex(CustodyError, "cannot be opened"):
            s.file_sha256(self.base / "absent")


class PayloadDigestTests(Case):
    """The exporter's first digest of a payload it wrote reads it as acceptance does: by descriptor in the admitted --output directory, never following a link or waiting."""

    def test_regular_payload_digest_equals_the_ordinary_digest(self):
        s, export, meta = self.admitted()
        with open(export / "logits.f32", "rb") as handle:
            ordinary = hashlib.file_digest(handle, "sha256").hexdigest()
        self.assertEqual(s.payload_sha256(export, "logits.f32"), ordinary)
        self.assertEqual(ordinary, meta["logits"]["sha256"])
        self.accept(s, export, meta)
        self.assertEqual(self.receipt(s)["payloads"], {"logits.f32": {"bytes": 16, "sha256": ordinary}})

    def test_substituted_changed_late_or_misplaced_payloads_are_refused(self):
        def fifo(path):
            path.unlink()
            os.mkfifo(path)
        def link(path):
            os.replace(path, path.with_name("copy.f32"))
            os.symlink(path.with_name("copy.f32"), path)
        def removed(path):
            path.unlink()
        def directory(path):
            path.unlink()
            path.mkdir()
        cases = [(fifo, "Payload logits.f32 is not a regular file"), (link, "Payload logits.f32 cannot be opened without following links"),
                 (removed, "Payload logits.f32 cannot be opened without following links: No such file or directory"),
                 (directory, "Payload logits.f32 is not a regular file")]
        for change, fragment in cases:
            with self.subTest(change=change.__name__):
                s, export, meta = self.admitted()
                payload = export / "logits.f32"
                change(payload)
                try:
                    with bounded(), self.assertRaisesRegex(CustodyError, fragment):
                        s.payload_sha256(export, "logits.f32")
                finally:
                    if os.path.lexists(payload) and not payload.is_symlink() and not payload.is_file():
                        payload.rmdir() if payload.is_dir() else payload.unlink()
        s, export, meta = self.admitted()
        os.replace(export, self.fresh("moved-export"))
        export.mkdir()
        (export / "logits.f32").write_bytes(b"\0" * 16)
        with self.assertRaisesRegex(CustodyError, "The --output directory .* was replaced after admission"):
            s.payload_sha256(export, "logits.f32")
        # Replaced after the directory check, the payload is still read from the admitted directory, and acceptance refuses the replacement.
        s, export, meta = self.admitted()
        original = custody.hash_regular
        def replace_then_hash(*args, **kwargs):
            os.replace(export, self.fresh("moved-export"))
            export.mkdir()
            (export / "logits.f32").write_bytes(b"\1" * 16)
            return original(*args, **kwargs)
        with mock.patch.object(custody, "hash_regular", replace_then_hash):
            self.assertEqual(s.payload_sha256(export, "logits.f32"), meta["logits"]["sha256"])
        with self.assertRaisesRegex(CustodyError, "The --output directory .* was replaced after admission"):
            self.accept(s, export, meta)
        s, export, meta = self.admitted()
        original = custody.stream
        def touch_after_read(*args, **kwargs):
            result = original(*args, **kwargs)
            info = (export / "logits.f32").stat()
            os.utime(export / "logits.f32", ns=(info.st_atime_ns, info.st_mtime_ns + 1))
            return result
        with mock.patch.object(custody, "stream", touch_after_read), self.assertRaisesRegex(CustodyError, "Payload logits.f32 changed while it was hashed: .*mtime_ns"):
            s.payload_sha256(export, "logits.f32")
        ticks = iter([0.0] + [61.0] * 8)
        with mock.patch.object(custody, "clock", lambda: next(ticks)), \
                self.assertRaisesRegex(CustodyError, "60-second custody deadline for payload logits.f32 expired during opening Payload logits.f32"):
            s.payload_sha256(export, "logits.f32")
        with self.assertRaisesRegex(CustodyError, "Payloads are hashed in the session's --output directory after admission"):
            s.payload_sha256(self.base, "logits.f32")
        root, inventory = self.model()
        t = self.bound(self.session(root, inventory))
        with self.assertRaisesRegex(CustodyError, "Payloads are hashed in the session's --output directory after admission"):
            t.payload_sha256(t.output, "logits.f32")


class ImporterCompatibilityTests(Case):
    """Metadata a custody session writes imports through the reviewed importer exactly as it stands."""

    def setUp(self):
        super().setUp()
        self.native = load("native_import_tests", ROOT / "tests/native/test_import_cpu_diagnostics.py")
        self.imp = self.native.imp
        self.fx = self.native.Fixture(self.base)
        self.manifest, self.manifest_sha = self.fx.manifest()

    def export(self, route):
        n = self.native
        root, inventory = self.model()
        s = self.session(root, inventory)
        self.bound(s)
        s.verify_before_load()
        s.declare("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.declare("model", FakeModel.from_pretrained, Path(root), dtype=FLOAT32, device_map="cpu", local_files_only=True, attn_implementation="eager")
        s.load("processor", FakeAutoProcessor.from_pretrained, Path(root), local_files_only=True)
        s.record_sources(s.loaded_sources(self.modules), {})
        teacher = n.TEACHER[:3]
        context = {"vocabulary_size": n.V, "model_revision": self.imp.PIN, "tokenizer_sha256": n.PINS["tokenizer.json"],
                   "config_sha256": n.PINS["config.json"], "image_sha256": sha(b"synthetic png"), "prompt_sha256": sha(b"prompt"),
                   "processor_profile": n.PROFILE, "positions": list(range(n.P0 - 1, n.P0 + len(teacher))), "prefix_token_ids": list(n.PROMPT),
                   "target_token_ids": list(teacher)}
        raw = {"input_ids": n.i64(n.PROMPT), "attention_mask": n.i64([1] * n.P0), "mm_token_type_ids": n.i64([int(t == n.IMG) for t in n.PROMPT]),
               "pixel_values": n.PIXELS, "image_grid_thw": n.i64(n.GRID)}
        shapes = {"pixel_values": [16, 1536], "image_grid_thw": [1, 3]}
        inputs = {k: {"shape": shapes.get(k, [1, n.P0]), "dtype": "torch.float32" if k == "pixel_values" else "torch.int64", "finite": True,
                      "sha256": sha(v)} for k, v in raw.items()}
        runtime = {"torch": "2.14.1+cpu", "transformers": "5.18.0", "python": "3.12.14", "device": "cpu", "dtype": "torch.float32", "threads": 8,
                   "attention": "eager", "model_source_sha256": self.imp.packager.ROPE_SOURCE_SHA,
                   "script_sha256": s.producer_sha256("verification/export_logits.py")}
        meta = {"schema_version": 1, "context": context, "inputs": inputs, "runtime": runtime, "response_sha256": "6" * 64, "states": []}
        meta["runtime"]["custody"] = s.admission_record()
        export = Path(s.output)
        export.mkdir()
        s.write_admission(export, meta)
        s.load("model", FakeModel.from_pretrained, Path(root), dtype=FLOAT32, device_map="cpu", local_files_only=True, attn_implementation="eager")
        meta["memory_admission"] = {"available_kib": 1}
        meta["runtime"]["attention_config"] = {"vision": "eager", "text": "eager"}
        if route == self.imp.CACHED:
            rows = list(range(n.P0 - 1, n.P0 + len(teacher)))
            meta["states"] = [n.snapshot(n.P0 + i) for i in range(len(teacher) + 1)]
        else:
            rows = list(range(n.P0 - 1, n.P0 - 1 + len(teacher)))
            meta["context"].update(positions=rows, teacher_token_ids=list(teacher), target_token_ids=list(teacher))
            meta["states"] = [n.snapshot(n.P0 + len(teacher))]
        payload = b"".join(n.finite_row(i) for i in range(len(rows)))
        (export / "logits.f32").write_bytes(payload)
        payloads = {"logits.f32": sha(payload)}
        if route == self.imp.CACHED:
            (export / "logits.partial.f32").write_bytes(payload)
            payloads["logits.partial.f32"] = sha(payload)
        meta["logits"] = {"file": "logits.f32", "shape": [len(rows), n.V], "dtype": "float32", "byte_order": "little", "sha256": sha(payload)}
        meta.update(elapsed_seconds=2.0, peak_rss_kib=3)
        metadata_sha = s.accept(export, meta, payloads, s.loaded_sources(self.modules))
        return s, export, metadata_sha, payload

    def test_cached_and_monolithic_custody_exports_import_unchanged(self):
        for route, rows in ((self.imp.CACHED, "0,1,2,3"), (self.imp.MONOLITHIC, "0")):
            with self.subTest(route=route):
                s, export, metadata_sha, payload = self.export(route)
                self.assertTrue(set(os.listdir(export)) <= self.imp.EXPORT_ENTRIES)
                self.assertNotIn(Path(s.evidence), [export] + list(export.iterdir()))
                output = self.fx.path("out")
                args = ["--export-dir", export, "--metadata-sha256", metadata_sha, "--input-manifest", self.manifest, "--input-sha256",
                        self.manifest_sha, "--route", route, "--rows", rows, "--output", output, "--max-input-bytes", 64 * 1024 * 1024,
                        "--max-rows", 64, "--max-output-bytes", 64 * 1024 * 1024, "--deadline-seconds", 600,
                        "--attest-model-sha256", self.imp.MODEL_SHA256, "--attestation-basis", f"synthetic custody receipt {sha(Path(s.evidence, 'receipt.json').read_bytes())}",
                        "--producer-source", Path(s.evidence) / "producer" / "export_logits.py"]
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = self.imp.main([str(a) for a in args])
                report = json.loads(out.getvalue())
                self.assertEqual((code, report["status"], report["error"]), (0, "imported", None))
                dump = self.native.cd.load_dump(Path(output) / "reference")
                runtime = dump["header"]["producer"]["runtime"]
                self.assertEqual(runtime["custody"]["before_load"], self.receipt(s)["before_load"])
                record = dump["records"]["text.logits/i-/prefill/s-/r0/c-"]
                self.assertEqual((Path(output) / "reference" / record["file"]).read_bytes(), payload[:self.native.ROW])


if __name__ == "__main__":
    unittest.main()
