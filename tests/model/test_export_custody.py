"""Run verification/export_logits.py itself, with and without --custody, on fake Torch, Transformers and Pillow modules.

No framework, model or device is imported. The fakes implement only the tensor operations the cached and all-teacher routes call, on the twelve-token synthetic prompt of tests/native/test_import_cpu_diagnostics.py; loaders are callbacks that record their arguments and the custody state they observe. Memory gates run on synthetic /proc readings. The real importer then converts the custody exports; because no synthetic file can reproduce the pinned modeling source, tokenizer and config bytes, only those three importer pins are substituted with the fixture digests for that step. Producer-binding controls run copies of the exporter whose disk bytes change after Python compiled them, including one run as a real __main__ script in a child interpreter.
"""
from array import array
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import py_compile
import shutil
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load("native_import_tests_for_exporter", ROOT / "tests/native/test_import_cpu_diagnostics.py")
imp = native.imp
V, P0, IMG, TEACHER = native.V, native.P0, native.IMG, native.TEACHER[:3]
REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
NS = types.SimpleNamespace


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


# Synthetic /proc readings within every memory gate, so the exporter's own cpu_memory runs in every route, including the retained module.
PROC = {"/proc/meminfo": "MemAvailable:   67108864 kB\n", "/proc/self/smaps_rollup": "Rss: 1 kB\nAnonymous: 1 kB\nPss_File: 0 kB\nSwap: 0 kB\n"}
READ_TEXT = Path.read_text


def read_text(path, *args, **kwargs):
    return PROC[str(path)] if str(path) in PROC else READ_TEXT(path, *args, **kwargs)


# ---------------------------------------------------------------- fake torch


class dtype:
    def __init__(self, name, code):
        self.name, self.code = name, code

    def __str__(self):
        return f"torch.{self.name}"


dtype.__module__ = "torch"
float32, bfloat16, int64, uint8 = dtype("float32", "f"), dtype("bfloat16", "f"), dtype("int64", "q"), dtype("uint8", "B")


class Tensor:
    def __init__(self, data, shape, kind):
        self.data, self.shape, self.dtype = data, tuple(shape), kind

    def detach(self):
        return self

    cpu = contiguous = detach

    def float(self):
        return self.to(float32)

    def to(self, target):
        if isinstance(target, dtype) and target is not self.dtype:
            return Tensor(array(target.code, self.data), self.shape, target)
        return self

    def view(self, target):
        assert target is uint8
        raw = self.data.tobytes()
        return Tensor(array("B", raw), (len(raw),), uint8)

    def numpy(self):
        return Numpy(self.data)

    def tolist(self):
        def nest(values, shape):
            if len(shape) == 1:
                return list(values)
            step = len(values) // shape[0]
            return [nest(values[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]
        return nest(self.data, self.shape)

    def __getitem__(self, index):
        data, shape = self.data, self.shape
        for i in index if isinstance(index, tuple) else (index,):
            step = len(data) // shape[0]
            i %= shape[0]
            data, shape = data[i * step:(i + 1) * step], shape[1:]
        return Tensor(data, shape, self.dtype)

    def argmax(self):
        return max(range(len(self.data)), key=self.data.__getitem__)

    def __add__(self, other):
        assert len(other.data) == 1
        return Tensor(array(self.data.typecode, [v + other.data[0] for v in self.data]), self.shape, self.dtype)


class Numpy:
    def __init__(self, data):
        self.data = data

    def astype(self, code):
        assert code == "<f4" and sys.byteorder == "little"
        return Numpy(array("f", self.data))

    def tobytes(self):
        return self.data.tobytes()

    def tofile(self, target):
        if hasattr(target, "write"):
            target.write(self.data.tobytes())
        else:
            Path(target).write_bytes(self.data.tobytes())


def flatten(data):
    if not isinstance(data, list):
        return [data], []
    parts = [flatten(x) for x in data]
    return [v for p in parts for v in p[0]], [len(data)] + (parts[0][1] if parts else [])


def tensor(data, device=None, dtype=None):
    flat, shape = flatten(data)
    kind = dtype or (float32 if any(isinstance(v, float) for v in flat) else int64)
    return Tensor(array(kind.code, flat), shape, kind)


def cat(tensors, dim):
    assert dim == 1 and all(t.shape[0] == 1 and len(t.shape) == 2 for t in tensors)
    data = array(tensors[0].data.typecode)
    for t in tensors:
        data.extend(t.data)
    return Tensor(data, (1, len(data)), tensors[0].dtype)


def stack(rows):
    data = array("f")
    for row in rows:
        data.extend(row.data)
    return Tensor(data, (len(rows),) + rows[0].shape, float32)


def zeros(shape, dtype=float32, device=None):
    return Tensor(array(dtype.code, [0]) * math.prod(shape), shape, dtype)


def full(shape, value, device=None, dtype=int64):
    return Tensor(array(dtype.code, [value]) * math.prod(shape), shape, dtype)


def ones_like(t):
    return Tensor(array(t.data.typecode, [1]) * len(t.data), t.shape, t.dtype)


def isfinite(t):
    return NS(all=lambda: all(math.isfinite(v) for v in t.data))


ROWS = {}


def row(absolute):
    """A deterministic finite full-vocabulary row for one absolute sequence row."""
    if absolute not in ROWS:
        ROWS[absolute] = array("f")
        ROWS[absolute].frombytes(native.finite_row(absolute))
    return ROWS[absolute]


class Cache:
    def __init__(self, length):
        self.length = length
        linear = [NS(conv_states=zeros((1, 8, 4)), recurrent_states=zeros((1, 2, 4, 4))) for _ in range(3)]
        self.layers = linear + [NS(keys=zeros((1, 4, length, 8)), values=zeros((1, 4, length, 8)))]


# ---------------------------------------------------------------- fake transformers and Pillow


class Observed:
    """What the fake loaders saw on disk when they were called."""
    output, events, processor_hook, model_hook, forward_hook = None, [], None, None, None

    @classmethod
    def note(cls, event):
        output = Path(cls.output)
        cls.events.append((event, sorted(os.listdir(output)) if output.exists() else None,
                           sorted(os.listdir(str(output) + ".custody")) if os.path.exists(str(output) + ".custody") else None))


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return list(TEACHER)


class ImageProcessor:
    size = None


class Processor:
    def __init__(self):
        self.tokenizer, self.image_processor = Tokenizer(), ImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": True, "return_tensors": "pt"}
        return {"input_ids": tensor([native.PROMPT]), "attention_mask": tensor([[1] * P0]),
                "mm_token_type_ids": tensor([[int(t == IMG) for t in native.PROMPT]]),
                "pixel_values": Tensor(array("f", range(1536)) * 16, (16, 1536), float32), "image_grid_thw": tensor([native.GRID])}


class AutoProcessor:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        Observed.note("processor")
        if Observed.processor_hook:
            Observed.processor_hook()
        return Processor()


class Qwen3_5Config:
    pass


class Model:
    def __init__(self, path, kwargs):
        attention = kwargs["attn_implementation"]
        vision, text = (attention["vision_config"], attention["text_config"]) if isinstance(attention, dict) else (attention, attention)
        self.config = NS(vision_config=NS(_attn_implementation=vision), text_config=NS(_attn_implementation=text), _name_or_path=str(path))
        weight = NS(data_ptr=lambda: 4096)
        self.lm_head = NS(weight=weight)
        self.model = NS(language_model=NS(embed_tokens=NS(weight=weight)), rope_deltas=tensor([[native.NEXT - P0]]))
        self.generation_config = NS(eos_token_id=248044)
        self.name_or_path, self.dtype = str(path), kwargs["dtype"]

    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1, **other):
        if Observed.forward_hook:
            Observed.forward_hook()
        length = input_ids.shape[1] + (past_key_values.length if past_key_values else 0)
        rows = list(logits_to_keep.data) if isinstance(logits_to_keep, Tensor) else [length - 1]
        data = array("f")
        for r in rows:
            data.extend(row(r))
        return NS(logits=Tensor(data, (1, len(rows), V), float32), past_key_values=Cache(length))


class Qwen3_5ForConditionalGeneration:
    config_class = Qwen3_5Config
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        Observed.note("model")
        if Observed.model_hook:
            Observed.model_hook()
        return Model(path, kwargs)


def fake_modules(site):
    """Fake packages whose module files exist, so source hashing and inspect.getsourcefile see real bytes."""
    layout = {"torch": "torch/__init__.py", "transformers": "transformers/__init__.py", "PIL": "PIL/__init__.py", "PIL.Image": "PIL/Image.py",
              "transformers.models.auto.processing_auto": "transformers/models/auto/processing_auto.py",
              "transformers.models.qwen3_5.modeling_qwen3_5": "transformers/models/qwen3_5/modeling_qwen3_5.py",
              "transformers.models.qwen3_5.configuration_qwen3_5": "transformers/models/qwen3_5/configuration_qwen3_5.py",
              "transformers.models.qwen3_vl.processing_qwen3_vl": "transformers/models/qwen3_vl/processing_qwen3_vl.py"}
    modules = {}
    for name, relative in layout.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# fake {name}\n".encode())
        modules[name] = types.ModuleType(name)
        modules[name].__file__ = str(path)
    (site / "torch" / "lib").mkdir()
    (site / "torch" / "lib" / "libtorch_cpu.so").write_bytes(b"\x7fELF fake")
    torch = modules["torch"]
    for name in ("Tensor", "tensor", "cat", "stack", "zeros", "full", "ones_like", "isfinite", "float32", "bfloat16", "uint8"):
        setattr(torch, name, globals()[name])
    torch.long = torch.int64 = int64
    torch.__version__, torch.inference_mode = "2.14.1+cpu", contextlib.nullcontext
    torch.set_num_threads = torch.set_num_interop_threads = lambda n: None
    placed = {AutoProcessor: "transformers.models.auto.processing_auto", Qwen3_5ForConditionalGeneration: "transformers.models.qwen3_5.modeling_qwen3_5",
              Qwen3_5Config: "transformers.models.qwen3_5.configuration_qwen3_5", Processor: "transformers.models.qwen3_vl.processing_qwen3_vl",
              Tokenizer: "transformers.models.qwen3_vl.processing_qwen3_vl", ImageProcessor: "transformers.models.qwen3_vl.processing_qwen3_vl"}
    for cls, module in placed.items():
        cls.__module__ = module
        setattr(modules[module], cls.__name__, cls)
    transformers = modules["transformers"]
    transformers.__version__, transformers.AutoProcessor = "5.18.0", AutoProcessor
    transformers.Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGeneration
    modules["PIL.Image"].open = lambda path: NS(convert=lambda mode: f"{mode} image")
    modules["PIL"].Image = modules["PIL.Image"]
    return modules


def install_fakes(config):
    """sitecustomize hook of the __main__ child: the fakes and synthetic /proc readings, the observed output, a thread-count report, and optionally an edit of the exporter file while Pillow is imported, after Python has compiled and started executing the exporter."""
    modules = fake_modules(Path(config["site"]))
    pillow = {name: modules.pop(name) for name in ("PIL", "PIL.Image")}
    sys.modules.update(modules)
    Path.read_text = read_text
    Observed.output = config["output"]
    modules["torch"].set_num_threads = lambda n: print(json.dumps({"phase": "test_threads", "threads": n}), flush=True)

    class Pillow:
        def find_spec(self, name, path=None, target=None):
            return importlib.util.spec_from_loader(name, self) if name in pillow else None

        def create_module(self, spec):
            return pillow[spec.name]

        def exec_module(self, module):
            if module.__name__ == "PIL" and config.get("edit"):
                exporter = Path(config["edit"])
                exporter.write_bytes(exporter.read_bytes().replace(b"torch.set_num_threads(8)", b"torch.set_num_threads(3)"))

    sys.meta_path.insert(0, Pillow())


class ExporterCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="chandra-export-custody-")
        self.base = Path(os.path.realpath(self.temporary.name))
        self.modules = fake_modules(self.base / "site")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(sys.modules, self.modules))
        stack.enter_context(mock.patch.dict(os.environ))
        stack.enter_context(mock.patch.object(Path, "read_text", read_text))
        self.mod = load("export_logits_under_test", ROOT / "verification/export_logits.py")
        self.root = self.base / "model"
        self.files = {"config.json": b'{"model_type": "qwen3_5"}', "tokenizer.json": b'{"version": "1.0"}', "chat_template.jinja": b"{{ m }}",
                      "model.safetensors": bytes(range(256)) * 4, ".eval_results/bench.yaml": b"score: 1\n"}
        for name, data in self.files.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(data)
        inventory = {"model": "datalab-to/chandra-ocr-2", "revision": REVISION,
                     "files": [{"file": n, "size": len(d), "sha256": sha(d)} for n, d in sorted(self.files.items())]}
        self.inventory = self.base / "model.json"
        self.inventory.write_bytes(json.dumps(inventory, indent=2).encode())
        self.pin = self.mod.INVENTORY_SHA256
        self.mod.INVENTORY, self.mod.INVENTORY_SHA256 = self.inventory, sha(self.inventory.read_bytes())
        for name, data in (("image.png", b"synthetic png"), ("prompt.txt", b"prompt"), ("response.txt", b"teacher text")):
            (self.base / name).write_bytes(data)
        AutoProcessor.calls, Qwen3_5ForConditionalGeneration.calls = [], []
        Observed.events, Observed.processor_hook, Observed.model_hook, Observed.forward_hook = [], None, None, None
        self.count = 0
        self.addCleanup(self.temporary.cleanup)

    def output(self):
        self.count += 1
        return self.base / f"export{self.count}"

    def exporter_copy(self, name="repo"):
        """An exporter and helper checkout under base whose provenance/model.json is this fixture's inventory, pinned in its INVENTORY_SHA256."""
        repo = self.base / name
        (repo / "verification").mkdir(parents=True)
        (repo / "provenance").mkdir()
        shutil.copyfile(self.inventory, repo / "provenance" / "model.json")
        exporter = (ROOT / "verification/export_logits.py").read_bytes()
        pinned = exporter.replace(f'INVENTORY_SHA256 = "{self.pin}"'.encode(), f'INVENTORY_SHA256 = "{sha(self.inventory.read_bytes())}"'.encode())
        self.assertNotEqual(pinned, exporter)
        (repo / "verification" / "export_logits.py").write_bytes(pinned)
        shutil.copyfile(ROOT / "verification/cpu_custody.py", repo / "verification" / "cpu_custody.py")
        return repo / "verification" / "export_logits.py", repo / "verification" / "cpu_custody.py"

    def run_export(self, *extra, output=None, model=None):
        output = output or self.output()
        Observed.output = output
        argv = ["export_logits.py", "--model", str(model or self.root), "--image", str(self.base / "image.png"), "--prompt", str(self.base / "prompt.txt"),
                "--response", str(self.base / "response.txt"), "--output", str(output)] + list(extra)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = self.mod.main() or 0
            except SystemExit as error:
                code = error.code
        lines = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith("{")]
        return code, lines, Path(output), err.getvalue()

    def import_export(self, export, route, rows):
        """The reviewed importer, unchanged except for the three pinned digests no synthetic file can reproduce."""
        pins = {"tokenizer.json": sha(self.files["tokenizer.json"]), "config.json": sha(self.files["config.json"])}
        modeling = sha(Path(self.modules["transformers.models.qwen3_5.modeling_qwen3_5"].__file__).read_bytes())
        with mock.patch.object(imp.packager, "ROPE_SOURCE_SHA", modeling), mock.patch.dict(imp.packager.PROCESSOR_PINS, pins):
            fx = native.Fixture(self.base)
            manifest, manifest_sha = fx.manifest()
            receipt = Path(str(export) + ".custody") / "receipt.json"
            args = ["--export-dir", export, "--metadata-sha256", sha((export / "metadata.json").read_bytes()), "--input-manifest", manifest,
                    "--input-sha256", manifest_sha, "--route", route, "--rows", rows, "--output", fx.path("import"),
                    "--max-input-bytes", 64 * 1024 * 1024, "--max-rows", 64, "--max-output-bytes", 64 * 1024 * 1024, "--deadline-seconds", 600,
                    "--attest-model-sha256", imp.MODEL_SHA256, "--attestation-basis", f"synthetic custody receipt {sha(receipt.read_bytes())}",
                    "--producer-source", Path(str(export) + ".custody") / "producer" / "export_logits.py"]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = imp.main([str(a) for a in args])
        return code, json.loads(out.getvalue())

    def check_accepted(self, code, lines, output):
        self.assertEqual(code, 0, lines)
        evidence = Path(str(output) + ".custody")
        receipt = json.loads((evidence / "receipt.json").read_text())
        metadata = (output / "metadata.json").read_bytes()
        self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("accepted", sha(metadata)))
        admission, meta = json.loads((output / "admission.json").read_text()), json.loads(metadata)
        self.assertEqual(admission["runtime"], {k: v for k, v in meta["runtime"].items() if k != "attention_config"})
        record = meta["runtime"]["custody"]
        self.assertEqual(record["model_root"], str(self.root))
        self.assertEqual(record["inventory"]["sha256"], sha(self.inventory.read_bytes()))
        self.assertEqual(record["checkpoint"]["weights"], [{"file": "model.safetensors", "size": 1024, "sha256": sha(self.files["model.safetensors"])}])
        self.assertEqual(record["loads"]["model"], {"callable": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration.from_pretrained",
                                                    "pretrained_model_name_or_path": str(self.root), "kwargs": {
                                                        "attn_implementation": "eager", "device_map": "cpu", "dtype": "torch.float32", "local_files_only": True}})
        self.assertEqual(record["loads"]["processor"]["kwargs"], {"local_files_only": True})
        self.assertEqual(Qwen3_5ForConditionalGeneration.calls, [(self.root, {"dtype": float32, "device_map": "cpu", "local_files_only": True, "attn_implementation": "eager"})])
        self.assertEqual(AutoProcessor.calls, [(self.root, {"local_files_only": True})])
        # The processor loads after the pre-load pass and producer retention but before admission.json; the model loads after admission.json.
        self.assertEqual(Observed.events, [("processor", [], ["producer"]), ("model", ["admission.json"], ["producer", "sources.txt"])])
        classes = record["sources"]["classes"]
        self.assertEqual(classes["model"]["source"], "transformers/models/qwen3_5/modeling_qwen3_5.py")
        self.assertEqual(classes["model"]["sha256"], meta["runtime"]["model_source_sha256"])
        self.assertEqual((classes["tokenizer"]["class"], classes["model_config"]["source"]),
                         ("transformers.models.qwen3_vl.processing_qwen3_vl.Tokenizer", "transformers/models/qwen3_5/configuration_qwen3_5.py"))
        manifest = (evidence / "sources.txt").read_text().splitlines()
        self.assertIn("torch/lib/libtorch_cpu.so", [line.split("  ")[2] for line in manifest])
        self.assertEqual(sha((evidence / "producer" / "export_logits.py").read_bytes()), meta["runtime"]["script_sha256"])
        self.assertEqual((evidence / "producer" / "export_logits.py").read_bytes(), (ROOT / "verification/export_logits.py").read_bytes())
        self.assertEqual((evidence / "producer" / "cpu_custody.py").read_bytes(), (ROOT / "verification/cpu_custody.py").read_bytes())
        self.assertEqual(receipt["reported_by_loaded_objects"], {"model.name_or_path": str(self.root), "model.config._name_or_path": str(self.root),
                                                                 "model.dtype": "torch.float32"})
        self.assertEqual([line["phase"] for line in lines if line["phase"].startswith("custody")],
                         ["custody_verified_before_load", "custody_accepted"])
        self.assertTrue(set(os.listdir(output)) <= imp.EXPORT_ENTRIES)
        # The producer binding and every retained file are named in the record and receipt, and the printed receipt digest is the file's.
        execution = record["producer_execution"]
        self.assertEqual((execution["files"], receipt["producer_execution"]), (sorted(record["producer"]), execution))
        self.assertIn("code-object equality", execution["rule"])
        retained = receipt["retained_evidence"]
        self.assertEqual(sorted(retained), [f"../{output.name}/admission.json", "producer/cpu_custody.py", "producer/export_logits.py", "sources.txt"])
        for name, entry in retained.items():
            self.assertEqual(entry["sha256"], sha((evidence / name).read_bytes()))
        accepted = next(line for line in lines if line["phase"] == "custody_accepted")
        self.assertEqual((accepted["receipt_sha256"], accepted["metadata_sha256"]), (sha((evidence / "receipt.json").read_bytes()), sha(metadata)))
        return meta


class ExporterCustodyTests(ExporterCase):
    def test_cached_custody_export_is_accepted_and_imports(self):
        code, lines, output, _ = self.run_export("--custody")
        meta = self.check_accepted(code, lines, output)
        self.assertEqual(sorted(os.listdir(output)), ["admission.json", "logits.f32", "logits.partial.f32", "metadata.json"])
        self.assertEqual(meta["logits"]["shape"], [len(TEACHER) + 1, V])
        code, report = self.import_export(output, imp.CACHED, "0,1,2,3")
        self.assertEqual((code, report["status"], report["error"]), (0, "imported", None))

    def test_all_teacher_custody_export_is_accepted_and_imports(self):
        code, lines, output, _ = self.run_export("--custody", "--all-teacher", "--custody-hash-seconds", "120")
        meta = self.check_accepted(code, lines, output)
        self.assertEqual(sorted(os.listdir(output)), ["admission.json", "logits.f32", "metadata.json"])
        self.assertEqual(meta["runtime"]["custody"]["before_load"]["deadline_seconds"], 120.0)
        code, report = self.import_export(output, imp.MONOLITHIC, "0")
        self.assertEqual((code, report["status"], report["error"]), (0, "imported", None))

    def test_change_during_model_load_leaves_no_metadata(self):
        def touch():
            path = self.root / "config.json"
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
        Observed.model_hook = touch
        code, lines, output, _ = self.run_export("--custody")
        self.assertEqual(code, 2)
        self.assertEqual((lines[-1]["phase"], lines[-1]["metadata_written"]), ("custody_refused", False))
        self.assertIn("file config.json:", lines[-1]["error"])
        self.assertEqual(sorted(os.listdir(output)), ["admission.json", "logits.f32", "logits.partial.f32"])
        receipt = json.loads((Path(str(output) + ".custody") / "receipt.json").read_text())
        self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("refused", None))
        self.assertIn("mtime_ns", receipt["error"])

    def test_forward_failure_records_refusal_and_reraises(self):
        def fail():
            raise RuntimeError("synthetic forward failure")
        Observed.forward_hook = fail
        output = self.output()
        Observed.output = output
        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            argv = ["export_logits.py", "--model", str(self.root), "--image", str(self.base / "image.png"), "--prompt", str(self.base / "prompt.txt"),
                    "--response", str(self.base / "response.txt"), "--output", str(output), "--custody", "--all-teacher"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.mod.main()
        self.assertEqual(sorted(os.listdir(output)), ["admission.json"])
        receipt = json.loads((Path(str(output) + ".custody") / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("RuntimeError: synthetic forward failure", receipt["error"])

    def test_unsupported_or_unsafe_custody_runs_are_refused_before_output(self):
        link = self.base / "model-link"
        os.symlink(self.root, link)
        cases = [(("--custody", "--device", "xpu"), None, "CPU FP32 exports only"), (("--custody", "--precision", "bfloat16"), None, "CPU FP32 exports only"),
                 (("--custody", "--activation-fixture"), None, "--activation-fixture"), (("--custody", "--greedy-max-tokens", "16"), None, "--greedy-max-tokens"),
                 (("--custody",), link, "traverses a symlink"),
                 (("--custody", "--custody-hash-seconds", "7200"), None, "deadline must be within")]
        for extra, model, fragment in cases:
            with self.subTest(extra=extra, model=model):
                code, lines, output, _ = self.run_export(*extra, model=model)
                self.assertEqual((code, lines[-1]["phase"], lines[-1]["output_created"]), (2, "custody_refused", False))
                self.assertIn(fragment, lines[-1]["error"])
                self.assertFalse(os.path.lexists(output) or os.path.lexists(str(output) + ".custody"))
        code, lines, output, _ = self.run_export("--custody", output=Path(os.path.relpath(self.output())))
        self.assertEqual((code, lines[-1]["phase"]), (2, "custody_refused"))
        self.assertIn("absolute normalized", lines[-1]["error"])
        self.mod.INVENTORY_SHA256 = "0" * 64
        code, lines, output, _ = self.run_export("--custody")
        self.assertEqual(code, 2)
        self.assertIn("differs from the producer's pinned", lines[-1]["error"])
        code, lines, output, err = self.run_export("--custody-hash-seconds", "60")
        self.assertEqual(code, 2)
        self.assertIn("--custody-hash-seconds requires --custody", err)
        self.assertFalse(os.path.lexists(output))

    def test_route_without_custody_is_unchanged(self):
        code, lines, output, _ = self.run_export("--all-teacher")
        self.assertEqual(code, 0)
        self.assertFalse(os.path.lexists(str(output) + ".custody"))
        admission, meta = json.loads((output / "admission.json").read_text()), json.loads((output / "metadata.json").read_text())
        self.assertNotIn("custody", admission["runtime"])
        self.assertEqual(set(meta["runtime"]), {"torch", "transformers", "python", "device", "dtype", "threads", "attention", "model_source_sha256",
                                                "script_sha256", "attention_config"})
        self.assertEqual(AutoProcessor.calls, [(self.root, {"local_files_only": True})])
        self.assertEqual(Qwen3_5ForConditionalGeneration.calls, [(self.root, {"dtype": float32, "device_map": "cpu", "local_files_only": True, "attn_implementation": "eager"})])
        self.assertEqual(imp.check_metadata.__name__, "check_metadata")
        with mock.patch.object(imp.packager, "ROPE_SOURCE_SHA", meta["runtime"]["model_source_sha256"]):
            imp.check_admission(admission, meta, imp.MONOLITHIC)
            self.assertEqual(imp.check_metadata(meta, imp.MONOLITHIC)["lengths"], [P0 + len(TEACHER)])
        self.assertEqual([line["phase"] for line in lines], ["prepared", "complete_sparse"])
        # A plain-text path, relative output and an unverified directory remain usable without --custody, as before.
        link = self.base / "model-link"
        os.symlink(self.root, link)
        relative = Path(os.path.relpath(self.output()))
        code, lines, output, _ = self.run_export("--all-teacher", model=link, output=relative)
        self.assertEqual(code, 0)
        self.assertTrue((Path.cwd() / relative / "metadata.json").is_file())


SITECUSTOMIZE = """import importlib.util, json, os
config = json.loads(os.environ["CHANDRA_EXPORT_CUSTODY_FAKES"])
spec = importlib.util.spec_from_file_location("export_custody_fakes", config["tests"])
fakes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fakes)
fakes.install_fakes(config)
"""


class ProducerBindingTests(ExporterCase):
    """--custody runs only exporter bytes it retains, bound to the code Python compiled and was executing before main() ran."""

    def load_copy(self, name):
        exporter, helper = self.exporter_copy(name)
        self.mod = load(f"export_logits_{name}", exporter)  # The copy's own INVENTORY and pin name this fixture; nothing is patched.
        return exporter, helper

    def run_threads(self, *extra):
        threads = []
        with mock.patch.object(self.modules["torch"], "set_num_threads", threads.append):
            code, lines, output, _ = self.run_export(*extra)
        return code, lines, output, threads

    def test_disk_edit_after_compilation_is_refused_with_the_retained_bytes(self):
        exporter, _ = self.load_copy("edited")
        compiled = exporter.read_bytes()
        substituted = compiled.replace(b"torch.set_num_threads(8)", b"torch.set_num_threads(3)")
        self.assertNotEqual(substituted, compiled)
        exporter.write_bytes(substituted)  # Python has compiled the eight-thread exporter; disk now holds a three-thread one.
        code, lines, output, threads = self.run_threads("--custody", "--all-teacher")
        self.assertEqual((code, lines[-1]["phase"], lines[-1]["metadata_written"]), (2, "custody_refused", False))
        self.assertIn("Retained producer verification/export_logits.py", lines[-1]["error"])
        self.assertIn("does not compile to the code this process is executing", lines[-1]["error"])
        evidence = Path(str(output) + ".custody")
        receipt = json.loads((evidence / "receipt.json").read_text())
        self.assertEqual((receipt["status"], receipt["producer_execution"], receipt["before_load"], receipt["metadata_sha256"]), ("refused", None, None, None))
        self.assertEqual((evidence / "producer" / "export_logits.py").read_bytes(), substituted)  # Kept as what disk held, under a refusal.
        self.assertEqual(receipt["producer"]["verification/export_logits.py"]["sha256"], sha(substituted))
        self.assertEqual((threads, Observed.events, os.path.lexists(output)), ([], [], False))  # No export ran and nothing was loaded.
        # The same copy, unedited between compilation and retention, is accepted and runs its eight threads.
        exporter.write_bytes(compiled)
        self.mod = load("export_logits_restored", exporter)
        code, lines, output, threads = self.run_threads("--custody", "--all-teacher")
        self.assertEqual((code, threads), (0, [8]), lines)
        self.assertEqual((Path(str(output) + ".custody") / "producer" / "export_logits.py").read_bytes(), compiled)

    def test_stale_bytecode_is_refused(self):
        exporter, _ = self.exporter_copy("stale")
        py_compile.compile(str(exporter), cfile=importlib.util.cache_from_source(str(exporter)), doraise=True)
        info = exporter.stat()
        exporter.write_bytes(exporter.read_bytes().replace(b"torch.set_num_threads(8)", b"torch.set_num_threads(3)"))
        os.utime(exporter, ns=(info.st_atime_ns, info.st_mtime_ns))  # Same size and mtime, so the import system trusts the stale cache.
        self.mod = load("export_logits_stale", exporter)
        code, lines, output, threads = self.run_threads("--custody", "--all-teacher")
        self.assertEqual((code, threads, os.path.lexists(output)), (2, [], False))
        self.assertIn("stale bytecode cache", lines[-1]["error"])
        code, lines, output, threads = self.run_threads("--all-teacher")  # Without --custody the cached eight-thread code runs, as before.
        self.assertEqual((code, threads), (0, [8]))

    def test_custody_export_runs_the_retained_module_not_the_imported_one(self):
        self.mod.PROFILE = {"size": {"shortest_edge": 1, "longest_edge": 2}}  # The imported module's state changed after compilation.
        code, lines, output, _ = self.run_export("--custody", "--all-teacher")
        meta = self.check_accepted(code, lines, output)
        self.assertEqual(meta["context"]["processor_profile"], {"size": {"shortest_edge": 3136, "longest_edge": 3145728}})
        code, lines, output, _ = self.run_export("--all-teacher")
        self.assertEqual((code, json.loads((output / "metadata.json").read_text())["context"]["processor_profile"]), (0, self.mod.PROFILE))

    def test_main_script_child_binds_and_refuses_an_edit_during_import(self):
        """python verification/export_logits.py in a child interpreter: accepted unchanged, refused when edited while Pillow imports."""
        boot = self.base / "boot"
        boot.mkdir()
        (boot / "sitecustomize.py").write_text(SITECUSTOMIZE)
        for case in ("unchanged", "edited"):
            with self.subTest(case=case):
                exporter, _ = self.exporter_copy(case)
                compiled, output = exporter.read_bytes(), self.output()
                config = {"tests": str(Path(__file__).resolve()), "site": str(self.base / f"site-{case}"), "output": str(output),
                          "edit": str(exporter) if case == "edited" else None}
                env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONPATH": str(boot), "CHANDRA_EXPORT_CUSTODY_FAKES": json.dumps(config)}
                argv = [sys.executable, "-s", "-B", str(exporter), "--custody", "--all-teacher", "--model", str(self.root), "--image", str(self.base / "image.png"),
                        "--prompt", str(self.base / "prompt.txt"), "--response", str(self.base / "response.txt"), "--output", str(output)]
                result = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env, cwd=self.base)
                lines = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                evidence = Path(str(output) + ".custody")
                receipt = json.loads((evidence / "receipt.json").read_text())
                retained = (evidence / "producer" / "export_logits.py").read_bytes()
                threads = [line["threads"] for line in lines if line["phase"] == "test_threads"]
                if case == "unchanged":
                    self.assertEqual((result.returncode, threads, receipt["status"], retained), (0, [8], "accepted", compiled), result.stderr)
                    self.assertEqual(receipt["metadata_sha256"], sha((output / "metadata.json").read_bytes()))
                else:
                    self.assertEqual((result.returncode, threads, receipt["status"], os.path.lexists(output)), (2, [], "refused", False), result.stderr)
                    self.assertEqual(retained, compiled.replace(b"torch.set_num_threads(8)", b"torch.set_num_threads(3)"))
                    self.assertIn("does not compile to the code this process is executing", lines[-1]["error"])

    def test_unreadable_producer_files_are_refused_before_output(self):
        def fifo(path):
            path.unlink()
            os.mkfifo(path)
        def link(path):
            moved = path.with_name("moved.py")
            os.replace(path, moved)
            os.symlink(moved, path)
        def garbage(path):
            path.write_bytes(b"def (\n")
        cases = [("helper-fifo", 1, fifo, "The custody helper cannot be read and compiled: OSError"),
                 ("helper-link", 1, link, "The custody helper cannot be read and compiled: OSError"),
                 ("helper-syntax", 1, garbage, "The custody helper cannot be read and compiled: SyntaxError"),
                 ("exporter-fifo", 0, fifo, "is not a regular file")]
        for name, which, change, fragment in cases:
            with self.subTest(case=name):
                files = self.load_copy(name)
                original = files[which].read_bytes()
                change(files[which])
                try:
                    with bounded():
                        code, lines, output, _ = self.run_export("--custody", "--all-teacher")
                finally:  # A failure's traceback reads these files' lines; never leave it a FIFO to block on.
                    files[which].unlink()
                    files[which].write_bytes(original)
                self.assertEqual((code, lines[-1]["phase"], lines[-1]["output_created"]), (2, "custody_refused", False))
                self.assertIn(fragment, lines[-1]["error"])
                self.assertFalse(os.path.lexists(output) or os.path.lexists(str(output) + ".custody"))

    def test_model_file_replaced_by_fifo_after_the_pre_load_pass_is_refused_finitely(self):
        def swap():
            (self.root / "tokenizer.json").unlink()
            os.mkfifo(self.root / "tokenizer.json")
        Observed.processor_hook = swap
        with bounded():
            code, lines, output, _ = self.run_export("--custody")
        self.assertEqual((code, lines[-1]["phase"], lines[-1]["metadata_written"]), (2, "custody_refused", False))
        self.assertIn("tokenizer.json is not a regular file", lines[-1]["error"])
        self.assertEqual(os.listdir(output), [])
        self.assertEqual(json.loads((Path(str(output) + ".custody") / "receipt.json").read_text())["status"], "refused")


def touch(path):
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))


class PayloadAndSourceAdmissionTests(ExporterCase):
    """Under --custody the exporter's first logits.f32 digest is read as acceptance reads payloads, and a loaded library file that disappears is refused, not omitted."""

    ROUTES = {"cached": (), "full-teacher": ("--full-teacher",), "all-teacher": ("--all-teacher",)}

    def reset_observed(self):
        AutoProcessor.calls, Qwen3_5ForConditionalGeneration.calls, Observed.events = [], [], []

    def test_regular_payload_digests_and_bytes_are_unchanged_and_never_read_by_a_blocking_open(self):
        real, opened = hashlib.file_digest, []
        def recording(handle, digest):
            opened.append(Path(handle.name).name)
            return real(handle, digest)
        for route, extra in self.ROUTES.items():
            with self.subTest(route=route):
                exports = {}
                for custody in (False, True):
                    self.reset_observed()
                    opened.clear()
                    with mock.patch.object(hashlib, "file_digest", recording):
                        code, lines, output, _ = self.run_export(*(("--custody",) if custody else ()), *extra)
                    if custody:
                        meta = self.check_accepted(code, lines, output)
                        self.assertEqual(opened, [])  # No custody digest, the payload's included, goes through the blocking sha().
                    else:
                        self.assertEqual(code, 0, lines)
                        meta = json.loads((output / "metadata.json").read_text())
                        self.assertIn("logits.f32", opened)  # Without --custody the ordinary sha() digests the payload, as before.
                    exports[custody] = (output, meta)
                (plain, plain_meta), (kept, kept_meta) = exports[False], exports[True]
                data = (kept / "logits.f32").read_bytes()
                self.assertEqual(data, (plain / "logits.f32").read_bytes())
                self.assertEqual(plain_meta["logits"], kept_meta["logits"])
                self.assertEqual(kept_meta["logits"]["sha256"], sha(data))
                receipt = json.loads((Path(str(kept) + ".custody") / "receipt.json").read_text())
                payloads = {"logits.f32": {"bytes": len(data), "sha256": sha(data)}}
                if route == "cached":
                    self.assertEqual((kept / "logits.partial.f32").read_bytes(), data)
                    payloads["logits.partial.f32"] = payloads["logits.f32"]
                self.assertEqual(receipt["payloads"], payloads)

    def test_payload_substituted_before_its_first_digest_is_refused_within_the_deadline(self):
        def fifo(path):
            path.unlink()
            os.mkfifo(path)
        def link_to_fifo(path):
            pipe = path.with_name("unaccepted-pipe")
            os.mkfifo(pipe)
            path.unlink()
            os.symlink(pipe, path)
        def link_to_copy(path):
            copy = path.with_name("unaccepted-logits-copy.f32")
            os.replace(path, copy)
            os.symlink(copy, path)
        def removed(path):
            path.unlink()
        changes = [(fifo, "Payload logits.f32 is not a regular file"), (link_to_fifo, "Payload logits.f32 cannot be opened without following links"),
                   (link_to_copy, "Payload logits.f32 cannot be opened without following links"),
                   (removed, "Payload logits.f32 cannot be opened without following links: No such file or directory")]
        original = Numpy.tofile
        for route, extra in self.ROUTES.items():
            for change, fragment in changes:
                with self.subTest(route=route, change=change.__name__):
                    self.reset_observed()
                    swapped, output = [], self.output()
                    def tofile(values, target):
                        original(values, target)
                        if isinstance(target, Path) and target.name == "logits.f32":
                            change(target)
                            touch(self.root / "config.json")  # Only the post-load pass would see this, so a payload refusal comes from the first digest.
                            swapped.append(target)
                    try:
                        with mock.patch.object(Numpy, "tofile", tofile), bounded(5):
                            code, lines, output, _ = self.run_export("--custody", "--custody-hash-seconds", "1", *extra, output=output)
                    finally:  # A blocked or failed run must not leave a FIFO for a later read to wait on.
                        for pipe in (output / "logits.f32", output / "unaccepted-pipe"):
                            if os.path.lexists(pipe) and not pipe.is_symlink() and not pipe.is_file():
                                pipe.unlink()
                    self.assertEqual(len(swapped), 1)
                    self.assertEqual(code, 2, lines)
                    self.assertEqual((lines[-1]["phase"], lines[-1]["metadata_written"]), ("custody_refused", False))
                    self.assertIn(fragment, lines[-1]["error"])
                    receipt = json.loads((Path(str(output) + ".custody") / "receipt.json").read_text())
                    self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("refused", None))
                    self.assertIn(fragment, receipt["error"])
                    self.assertFalse(os.path.lexists(output / "metadata.json"))
                    self.assertIn("admission.json", os.listdir(output))
                    self.assertEqual(len(Qwen3_5ForConditionalGeneration.calls), 1)

    def library_module(self, name, body=b"def numerical_helper():\n    return 1\n"):
        """A module of the fake transformers package loaded from its own regular file and registered as imported (setUp restores sys.modules)."""
        path = self.base / "site" / "transformers" / f"{name}.py"
        path.write_bytes(body)
        module = types.ModuleType(f"transformers.{name}")
        module.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        sys.modules[module.__name__] = module
        return module, path

    def test_loaded_library_file_removed_before_admission_is_refused(self):
        module, path = self.library_module("numerical_helper")
        executed = []
        def processor_hook():
            executed.append(module.numerical_helper())
            path.unlink()
        Observed.processor_hook = processor_hook
        for route, extra in self.ROUTES.items():
            with self.subTest(route=route):
                self.reset_observed()
                path.write_bytes(b"def numerical_helper():\n    return 1\n")
                code, lines, output, _ = self.run_export("--custody", *extra)
                fragment = "Library source transformers/numerical_helper.py cannot be opened without following links: No such file or directory"
                self.assertEqual(code, 2, lines)
                self.assertEqual((lines[-1]["phase"], lines[-1]["metadata_written"]), ("custody_refused", False))
                self.assertIn(fragment, lines[-1]["error"])
                evidence = Path(str(output) + ".custody")
                receipt = json.loads((evidence / "receipt.json").read_text())
                self.assertEqual((receipt["status"], receipt["sources"], receipt["admission_sha256"]), ("refused", None, None))
                self.assertIn(fragment, receipt["error"])
                # Refused at the manifest, before admission.json, sources.txt or the model load.
                self.assertEqual((os.listdir(output), sorted(os.listdir(evidence))), ([], ["producer", "receipt.json"]))
                self.assertEqual(Qwen3_5ForConditionalGeneration.calls, [])
        self.assertEqual(executed, [1, 1, 1])

    def test_library_file_imported_during_the_model_load_and_removed_is_refused(self):
        executed = []
        def model_hook():
            module, path = self.library_module("late_helper")
            executed.append(module.numerical_helper())
            path.unlink()
        Observed.model_hook = model_hook
        for route, extra in self.ROUTES.items():
            with self.subTest(route=route):
                self.reset_observed()
                sys.modules.pop("transformers.late_helper", None)  # Imported, and its file removed, by the previous route's model load.
                code, lines, output, _ = self.run_export("--custody", *extra)
                fragment = "Library source transformers/late_helper.py cannot be opened without following links: No such file or directory"
                self.assertEqual(code, 2, lines)
                self.assertEqual((lines[-1]["phase"], lines[-1]["metadata_written"]), ("custody_refused", False))
                self.assertIn(fragment, lines[-1]["error"])
                receipt = json.loads((Path(str(output) + ".custody") / "receipt.json").read_text())
                self.assertEqual((receipt["status"], receipt["metadata_sha256"]), ("refused", None))
                self.assertFalse(os.path.lexists(output / "metadata.json"))
                self.assertIn("admission.json", os.listdir(output))
        self.assertEqual(executed, [1, 1, 1])

    def test_modules_without_a_file_are_not_sources_and_still_accept(self):
        extra = {"transformers.pseudo": types.ModuleType("transformers.pseudo"), "transformers.namespace": types.ModuleType("transformers.namespace"),
                 "torch._C._nn": types.ModuleType("torch._C._nn")}
        extra["transformers.pseudo"].__file__ = "_ops.py"  # A relative pseudo-path, as torch's namespace objects declare, names no file.
        extra["transformers.namespace"].__file__ = None  # A namespace package; torch._C._nn, like an extension submodule, has no __file__.
        sys.modules.update(extra)
        code, lines, output, _ = self.run_export("--custody", "--all-teacher")
        self.check_accepted(code, lines, output)
        listed = [line.split("  ")[2] for line in (Path(str(output) + ".custody") / "sources.txt").read_text().splitlines()]
        self.assertFalse([name for name in listed if "pseudo" in name or "namespace" in name or "_ops" in name or "_nn" in name])


if __name__ == "__main__":
    unittest.main()
