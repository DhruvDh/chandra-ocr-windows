"""CPU tests for the resident native worker and the refactored CLI request path.

Builds the Windows sources (predecessor inference.cpp from Git, the refactored inference.cpp plus
inference_core.cpp, and worker.cpp) with the host C++17 compiler against a test-only Win32 shim and a
recording fake Device that executes no arithmetic. Synthetic input packages stand in for the pinned
public caller/config files, which are not in this repository: only the three pinned SHA-256 literals are
replaced, identically for predecessor and current sources. Logits are a scripted control-flow seam, so
nothing here is numerical evidence. MSVC, HLSL, D3D11 and every GPU run remain root-owned.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "ChandraNative/runtime"
SUPPORT = ROOT / "tests/native/resident_worker"
PREDECESSOR = "caefe03"  # Baseline commit before the core extraction and worker.
COMPILER = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
# Optional, e.g. CHANDRA_TEST_SANITIZE=thread or address,undefined, to rebuild every test binary with a sanitizer.
SANITIZE = ["-g", f"-fsanitize={os.environ['CHANDRA_TEST_SANITIZE']}"] if os.environ.get("CHANDRA_TEST_SANITIZE") else []
PINS = {
    "caller": "185ef7cbff08dea1a196f7fd96993f596d5943b90c2edcdaba001f0487cd1921",
    "generation": "0c35bb39fbaed1ac0656baabc4f4e9bda20214e12336d0e4e8755aac1f487c2e",
    "tokenizer": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
}
EVENT_SCHEMA = "chandra.directcompute.worker-event.v1"
REQUEST_SCHEMA = "chandra.directcompute.worker-request.v1"
REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
STOPS = [248046, 248044]
VOLATILE = {"duration_wall_seconds", "input_authentication_wall_seconds", "total_process_path_wall_seconds", "wall_seconds", "executable_sha256", "progress_sha256", "remaining_ms"}

# Synthetic public stand-ins for the pinned caller/config files (their SHA-256 replaces the pins in test builds only).
CALLER = (b"# Synthetic test stand-in for the pinned Chandra caller source; not the original file.\n" + b"#" * 4000)[:3177] + b"\n"
GENERATION = json.dumps({"eos_token_id": STOPS}).encode()
TOKENIZER = json.dumps({"added_tokens_decoder": {"248046": {"content": "<|im_end|>"}, "248044": {"content": "<|endoftext|>"}}}).encode()
SYNTHETIC = {"caller": hashlib.sha256(CALLER).hexdigest(), "generation": hashlib.sha256(GENERATION).hexdigest(), "tokenizer": hashlib.sha256(TOKENIZER).hexdigest()}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor(directory, name, dtype, shape, values):
    data = struct.pack("<%d%s" % (len(values), "q" if dtype == "I64" else "f"), *values)
    (directory / f"{name}.bin").write_bytes(data)
    strides, stride = [], 1
    for n in reversed(shape):
        strides.insert(0, stride)
        stride *= n
    return {"file": f"{name}.bin", "sha256": sha(data), "dtype": dtype, "byte_order": "little", "shape": shape, "bytes": len(data), "strides": strides}


def make_package(directory, tail=61):
    """B1 page: 5 text rows, 4 image rows from one [1,4,4] grid, then `tail` text rows (70 rows by default, two prefill tiles)."""
    directory.mkdir(parents=True)
    (directory / "chandra-caller-hf.py").write_bytes(CALLER)
    (directory / "generation_config.json").write_bytes(GENERATION)
    (directory / "tokenizer_config.json").write_bytes(TOKENIZER)
    ids, t, h, w = [], [], [], []
    for i in range(5):
        ids.append(100 + i); t.append(i); h.append(i); w.append(i)
    for m in range(4):
        ids.append(248056); t.append(5); h.append(5 + m // 2); w.append(5 + m % 2)
    for i in range(tail):
        ids.append(1000 + i); t.append(7 + i); h.append(7 + i); w.append(7 + i)
    n, top = len(ids), 7 + tail - 1
    tensors = {
        "input_ids": tensor(directory, "input_ids", "I64", [1, n], ids),
        "pixel_values": tensor(directory, "pixel_values", "F32", [16, 1536], [((i * 37) % 101) / 101.0 - 0.5 for i in range(16 * 1536)]),
        "image_grid_thw": tensor(directory, "image_grid_thw", "I64", [1, 3], [1, 4, 4]),
        "position_ids": tensor(directory, "position_ids", "I64", [3, 1, n], t + h + w),
        "attention_mask": tensor(directory, "attention_mask", "I64", [1, n], [1] * n),
        "mm_token_type_ids": tensor(directory, "mm_token_type_ids", "I64", [1, n], [1 if i == 248056 else 0 for i in ids]),
        "text_position_ids": tensor(directory, "text_position_ids", "I64", [1, n], list(range(n))),
        "rope_deltas": tensor(directory, "rope_deltas", "I64", [1, 1], [top + 1 - n]),
    }
    manifest = {
        "schema": "chandra.directcompute.input.v1", "model": "datalab-to/chandra-ocr-2", "revision": REVISION, "image_token_id": 248056,
        "image": {"sha256": "a" * 64, "pixels_sha256": "b" * 64, "dimensions": [64, 64]}, "prompt": {"sha256": "c" * 64},
        "processor_profile": "synthetic-test", "processor_kwargs": {},
        "generation": {"context_limit": 16384, "max_output_tokens": 12384, "stop_token_ids": STOPS,
                       "stop_token_provenance": {"caller_source_file": "chandra-caller-hf.py", "caller_source_sha256": SYNTHETIC["caller"],
                                                 "generation_config_sha256": SYNTHETIC["generation"], "tokenizer_config_sha256": SYNTHETIC["tokenizer"],
                                                 "caller_stop_token_ids": STOPS}},
        "tensors": tensors, "prompt_tokens": n, "image_token_positions": [5, 6, 7, 8], "image_grid_thw": [[1, 4, 4]],
        "positions": {"next_decode_position": top + 1},
    }
    data = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(data)
    return directory / "manifest.json", sha(data)


def substitute(text, name):
    for key, pin in PINS.items():
        if text.count(pin) != 1:
            raise AssertionError(f"{name}: expected exactly one pinned {key} literal")
        text = text.replace(pin, SYNTHETIC[key])
    return text


def compile_object(source, output, includes, strict):
    flags = ["-std=c++17", "-O1", "-Wall", "-Wextra", "-c", *SANITIZE] + (["-Werror"] if strict else [])
    if "g++" in Path(COMPILER).name and "clang" not in Path(COMPILER).name:
        flags.append("-Wno-dangling-reference")
    result = subprocess.run([COMPILER, *flags, *[f"-I{i}" for i in includes], "-o", str(output), str(source)], capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise AssertionError(f"compile {source.name} failed:\n{result.stderr[-4000:]}")
    return output


def link(objects, output):
    result = subprocess.run([COMPILER, *SANITIZE, "-o", str(output), *map(str, objects), "-lpthread"], capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise AssertionError(f"link {output.name} failed:\n{result.stderr[-4000:]}")
    return output


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def read_trace(path):
    return path.read_text().splitlines() if path.exists() else []


def request_sections(lines):
    """Split a worker trace into per-request sections bounded by the worker's tracked-bytes queries."""
    marks = [i for i, line in enumerate(lines) if line.startswith("tracked ")][1:-1]
    return [lines[marks[i] + 1:marks[i + 1]] for i in range(0, len(marks) - 1, 2)]


def canonical(section, weight_last):
    """Renumber request-owned serials by first appearance (weight serials stay fixed) and drop the
    process-global logits counter, so sections from different requests or processes compare exactly."""
    mapping, out = {}, []

    def rename(word):
        return word if int(word) <= weight_last else f"r{mapping.setdefault(int(word), len(mapping))}"
    for line in section:
        words = line.split()
        if words[0] == "memory":
            continue
        if words[0] in ("alloc", "release", "read", "zero", "upload"):
            words[1] = rename(words[1])
        elif words[0] == "dispatch":
            stop = words.index("params")
            words = [rename(w) if 2 < i < stop and w.isdigit() else w for i, w in enumerate(words)]
        elif words[0] == "logits":
            words = ["logits", words[2]]
        out.append(" ".join(words))
    return out


def live_after(section, weight_last):
    live = set()
    for line in section:
        words = line.split()
        if words[0] == "alloc":
            live.add(int(words[1]))
        elif words[0] == "release":
            live.discard(int(words[1]))
    return {s for s in live if s > weight_last}


class Worker:
    """A worker process with a stdout pump that validates every line as a bounded schema-tagged event."""

    def __init__(self, test, args, env, stdin=subprocess.PIPE):
        self.test, self.events, self.bad = test, [], []
        self.proc = subprocess.Popen([str(test.worker), *args], stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.queue, self.stderr = queue.Queue(), []
        self.threads = [threading.Thread(target=self._pump, daemon=True), threading.Thread(target=lambda: self.stderr.extend(self.proc.stderr), daemon=True)]
        for thread in self.threads:
            thread.start()

    def _pump(self):
        for raw in self.proc.stdout:
            try:
                event = json.loads(raw)
                if event.get("schema") != EVENT_SCHEMA or len(raw) > 65536 or not raw.endswith(b"\n"):
                    self.bad.append(raw)
            except ValueError:
                self.bad.append(raw)
                continue
            self.queue.put(event)
        self.queue.put(None)

    def send(self, message):
        data = message if isinstance(message, bytes) else (json.dumps(message) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def submit(self, ident, output, manifest="pkg/manifest.json", digest=None, **extra):
        self.send({"schema": REQUEST_SCHEMA, "type": "submit", "id": ident, "model": "chandra", "input_manifest": manifest,
                   "input_sha256": digest or self.test.digest, "output": output, **extra})

    def control(self, kind, **extra):
        self.send({"schema": REQUEST_SCHEMA, "type": kind, **extra})

    def expect(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out; last events {self.events[-5:]}")
            event = self.queue.get(timeout=remaining)
            if event is None:
                raise AssertionError(f"worker exited first; last events {self.events[-5:]}; stderr {b''.join(self.stderr)[-2000:]}")
            self.events.append(event)
            if predicate(event):
                return event

    def finish(self, close=True, timeout=60):
        if close and self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        code = self.proc.wait(timeout=timeout)
        while True:
            event = self.queue.get(timeout=timeout)
            if event is None:
                break
            self.events.append(event)
        for thread in self.threads:
            thread.join(timeout)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()
        self.test.assertEqual(self.bad, [])
        self.test.assertEqual([e["seq"] for e in self.events], list(range(len(self.events))))
        return code


def parked(gate, timeout=60):
    """Waits until the fake Device thread is blocked at the gate, so a following cancel is deterministic."""
    marker, deadline = Path(str(gate) + ".waiting"), time.monotonic() + timeout
    while not marker.exists():
        if time.monotonic() > deadline:
            raise AssertionError("fake Device never reached its gate")
        time.sleep(0.01)


def of(kind, ident=None, **match):
    return lambda e: e["event"] == kind and (ident is None or e.get("id") == ident) and all(e.get(k) == v for k, v in match.items())


def threads(pid):
    """Live threads of a process (Linux /proc), used to observe the reader thread ending."""
    return len(os.listdir(f"/proc/{pid}/task"))


def eventually(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition never held")
        time.sleep(0.01)


@unittest.skipUnless(COMPILER and os.name == "posix", "host C++17 compiler and POSIX shim required")
class ResidentWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="chandra-resident-worker-")
        cls.root = Path(cls.temporary.name)
        src = cls.root / "src/ChandraNative"
        (src / "runtime").mkdir(parents=True)
        (src / "vendor").symlink_to(ROOT / "ChandraNative/vendor")
        for path in RUNTIME.iterdir():
            if path.suffix in (".h", ".cpp"):
                text = path.read_text()
                (src / "runtime" / path.name).write_text(substitute(text, path.name) if path.name == "inference_core.h" else text)
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{PREDECESSOR}:ChandraNative/runtime/inference.cpp"], capture_output=True, text=True, timeout=60)
        if shown.returncode:
            raise unittest.SkipTest(f"predecessor {PREDECESSOR} unavailable")
        (src / "runtime/inference_predecessor.cpp").write_text(substitute(shown.stdout, "predecessor inference.cpp"))
        objects, shim = cls.root / "obj", [SUPPORT / "win32"]
        objects.mkdir()
        common = [compile_object(src / "runtime" / name, objects / (name + ".o"), shim, False)
                  for name in ("operators.cpp", "text_model.cpp", "vision_model.cpp", "diagnostics.cpp")]
        common += [compile_object(SUPPORT / name, objects / (name + ".o"), shim, True) for name in ("fake_device.cpp", "win32_shim.cpp")]
        core = compile_object(src / "runtime/inference_core.cpp", objects / "inference_core.o", shim, True)
        cls.cli_old = link([compile_object(src / "runtime/inference_predecessor.cpp", objects / "old.o", shim, False), *common], cls.root / "cli-predecessor")
        cls.cli_new = link([compile_object(src / "runtime/inference.cpp", objects / "new.o", shim, True), core, *common], cls.root / "cli-current")
        cls.worker = link([compile_object(src / "runtime/worker.cpp", objects / "worker.o", shim, True), core, *common], cls.root / "worker")
        cls.model, cls.shaders = cls.root / "model", cls.root / "shaders"
        cls.model.mkdir(); cls.shaders.mkdir()
        cls.inputs, cls.outputs = cls.root / "inputs", cls.root / "outputs"
        cls.inputs.mkdir(); cls.outputs.mkdir()
        cls.manifest, cls.digest = make_package(cls.inputs / "pkg")
        cls.counter = 0

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fresh(self, stem):
        type(self).counter += 1
        return self.root / f"{stem}-{self.counter}"

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CHANDRA_")}
        env.update({k: str(v) for k, v in extra.items()})
        return env

    # ----- CLI default-path equivalence with the predecessor -----
    def run_cli(self, binary, mode, output, env, extra=(), digest=None):
        trace = self.fresh("trace")
        args = [str(binary), mode, "--model-dir", str(self.model), "--shader-root", str(self.shaders), "--input-manifest", str(self.manifest),
                "--input-sha256", digest or self.digest, "--output-dir", str(output), "--pci", "fake-pci", "--luid", "fake-luid", *extra]
        result = subprocess.run(args, capture_output=True, timeout=600, env=self.env(CHANDRA_FAKE_TRACE=trace, **env))
        return result, read_trace(trace)

    def compare_cli(self, mode="--execute", env=None, extra=(), digest=None):
        outputs = [self.fresh("cli-old"), self.fresh("cli-new")]
        runs = [self.run_cli(binary, mode, out, env or {}, extra, digest) for binary, out in zip((self.cli_old, self.cli_new), outputs)]
        (old, old_trace), (new, new_trace) = runs
        self.assertEqual(old.returncode, new.returncode, new.stderr)
        self.assertEqual(old_trace, new_trace)
        self.assertEqual(old.stderr, new.stderr)
        if old.stdout:
            self.assertEqual(normalize(json.loads(old.stdout)), normalize(json.loads(new.stdout)))
        old_files = sorted(p.relative_to(outputs[0]) for p in outputs[0].rglob("*")) if outputs[0].exists() else None
        new_files = sorted(p.relative_to(outputs[1]) for p in outputs[1].rglob("*")) if outputs[1].exists() else None
        self.assertEqual(old_files, new_files)
        for name in old_files or []:
            a, b = outputs[0] / name, outputs[1] / name
            if a.is_dir():
                continue
            if a.suffix in (".json", ".jsonl"):
                self.assertEqual([normalize(json.loads(x)) for x in a.read_text().splitlines()], [normalize(json.loads(x)) for x in b.read_text().splitlines()], name)
            else:
                self.assertEqual(a.read_bytes(), b.read_bytes(), name)
        return new, new_trace, outputs[1]

    def test_cli_default_generation_matches_predecessor(self):
        result, trace, output = self.compare_cli(env={"CHANDRA_FAKE_TOKENS": "5,7,248046"})
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "result.json").read_text())
        self.assertEqual((report["passed"], report["mode"], report["generated_token_ids"], report["stop_reason"], report["diagnostic_token_cap"]),
                         (True, "full_page_generation", [5, 7, 248046], "stop_token", None))
        self.assertEqual([json.loads(x)["token_id"] for x in (output / "generated-tokens.jsonl").read_text().splitlines()], [5, 7, 248046])
        self.assertGreater(len(trace), 15000)
        self.assertEqual(sum(line.startswith("logits ") for line in trace), 3)
        self.assertNotIn("runtime/linear_gemv.hlsl", "\n".join(trace))

    def test_cli_cap_failure_forecast_dump_and_refusal_match_predecessor(self):
        cases = [
            ("cap", "--execute", {"CHANDRA_FAKE_TOKENS": "5,7"}, ["--diagnostic-token-cap", "2"], None, 0),
            ("stop-first", "--execute", {"CHANDRA_FAKE_TOKENS": "248044"}, [], None, 0),
            ("forecast", "--forecast", {}, [], None, 0),
            ("forecast-dump", "--forecast", {}, ["--diagnostic-dump"], None, 0),
            ("dump", "--execute", {"CHANDRA_FAKE_TOKENS": ",".join(["5"] * 12)}, ["--diagnostic-dump", "--diagnostic-token-cap", "2"], None, 0),
            ("dispatch-failure", "--execute", {"CHANDRA_FAKE_TOKENS": "5", "CHANDRA_FAKE_FAIL": "dispatch:900"}, [], None, 1),
            ("drain-failure", "--execute", {"CHANDRA_FAKE_TOKENS": "5", "CHANDRA_FAKE_FAIL": "drain:40"}, [], None, 1),
            ("nonfinite", "--execute", {"CHANDRA_FAKE_TOKENS": "nan"}, [], None, 1),
            ("ordered", "--execute", {"CHANDRA_FAKE_TOKENS": "5,248046", "CHANDRA_EXPERIMENTAL_GEMV_B1": "ordered"}, [], None, 0),
            ("invalid-selector", "--execute", {"CHANDRA_FAKE_TOKENS": "5", "CHANDRA_EXPERIMENTAL_GEMV_B1": "1"}, [], None, 1),
            ("wrong-hash", "--execute", {}, [], "0" * 64, 1),
        ]
        for name, mode, env, extra, digest, code in cases:
            with self.subTest(name):
                result, trace, output = self.compare_cli(mode, env, extra, digest)
                self.assertEqual(result.returncode, code, result.stderr)
                if name == "ordered":
                    self.assertIn("runtime/linear_gemv.hlsl", "\n".join(trace))
                if name == "wrong-hash":
                    self.assertFalse(output.exists())
                    self.assertEqual(trace, [])
                if name == "invalid-selector":
                    self.assertIn("CHANDRA_EXPERIMENTAL_GEMV_B1", json.loads((output / "result.json").read_text())["error"])

    # ----- Worker -----
    def worker_args(self, mode="--execute", lease=60000, **override):
        args = {"--model-dir": self.model, "--shader-root": self.shaders, "--input-root": self.inputs, "--output-root": self.outputs,
                "--pci": "fake-pci", "--luid": "fake-luid", "--lease-ms": lease}
        args.update({"--" + k.replace("_", "-"): v for k, v in override.items()})
        out = [mode] if mode else []
        for key, value in args.items():
            if value is not None:
                out += [key, str(value)]
        return out

    def start(self, env=None, mode="--execute", job="kill,process_memory=4294967296", stdin=subprocess.PIPE, **args):
        trace = self.fresh("worker-trace")
        extra = dict(env or {})
        if job is not None:
            extra["CHANDRA_SHIM_JOB"] = job
        worker = Worker(self, self.worker_args(mode, **args), self.env(CHANDRA_FAKE_TRACE=trace, **extra), stdin)
        worker.trace = trace
        return worker

    def weight_last(self, trace):
        line = next(line for line in trace if line.startswith("model_import_end"))
        return int(line.split()[4])

    def test_inactive_and_selector_refusals_create_no_device(self):
        for env, code, selection in (({}, 0, "predecessor"), ({"CHANDRA_EXPERIMENTAL_GEMV_B1": "0"}, 0, "predecessor"),
                                     ({"CHANDRA_EXPERIMENTAL_GEMV_B1": "ordered"}, 0, "ordered"), ({"CHANDRA_EXPERIMENTAL_GEMV_B1": "1"}, 1, None),
                                     ({"CHANDRA_EXPERIMENTAL_GEMV_B1": ""}, 1, None), ({"CHANDRA_EXPERIMENTAL_GEMV_B1": "Ordered"}, 1, None)):
            with self.subTest(env=env):
                trace = self.fresh("trace")
                result = subprocess.run([str(self.worker)], capture_output=True, timeout=60, env=self.env(CHANDRA_FAKE_TRACE=trace, **env))
                self.assertEqual(result.returncode, code)
                event = json.loads(result.stdout)
                self.assertEqual((event["schema"], event["seq"], event["device_created"], event["model_loaded"]), (EVENT_SCHEMA, 0, False, False))
                self.assertEqual(event["event"], "inactive" if code == 0 else "fatal")
                self.assertEqual(event["gemv_b1_selection"], selection)
                self.assertEqual(read_trace(trace), [])
                if code == 0:  # The inactive mode does not read stdin or wait.
                    self.assertEqual(result.stdout.count(b"\n"), 1)
        # An invalid selector refuses an explicit execute request before Device creation too.
        worker = self.start({"CHANDRA_EXPERIMENTAL_GEMV_B1": "1"})
        self.assertEqual(worker.finish(), 1)
        self.assertEqual((worker.events[0]["event"], worker.events[0]["stage"]), ("fatal", "configuration"))
        self.assertEqual(read_trace(worker.trace), [])

    def test_execute_configuration_refusals_precede_device_creation(self):
        link = self.root / "linked-inputs"
        link.symlink_to(self.inputs)
        cases = [
            ("no job", {"job": None}, "kill-on-close Job"),
            ("job without kill", {"job": "process_memory=1000"}, "kill-on-close Job"),
            ("job without memory", {"job": "kill"}, "kill-on-close Job"),
            ("breakaway", {"job": "kill,job_memory=1000,breakaway"}, "kill-on-close Job"),
            ("no lease", {"lease": None}, "--lease-ms"),
            ("short lease", {"lease": 999}, "between 1000"),
            ("relative input root", {"input_root": "inputs"}, "absolute"),
            ("symlinked input root", {"input_root": link}, "reparse"),
            ("nested roots", {"output_root": self.inputs / "pkg"}, "not nested"),
            ("missing output root", {"output_root": self.root / "absent"}, "missing or reparse"),
            ("trailing separator", {"output_root": str(self.outputs) + "/"}, "normalized"),
            ("no pci", {"pci": None}, "PCI"),
            ("relative model", {"model_dir": "model"}, "model and shader"),
        ]
        for name, change, fragment in cases:
            with self.subTest(name):
                job = change.pop("job", "kill,process_memory=4294967296")
                worker = self.start(job=job, **change)
                self.assertEqual(worker.finish(), 1)
                self.assertEqual(len(worker.events), 1)
                self.assertEqual((worker.events[0]["event"], worker.events[0]["device_created"]), ("fatal", False))
                self.assertIn(fragment, worker.events[0]["error"])
                self.assertEqual(read_trace(worker.trace), [])
        both = Worker(self, ["--plan", *self.worker_args()], self.env(CHANDRA_SHIM_JOB="kill,process_memory=4294967296"))
        self.assertEqual(both.finish(), 1)
        self.assertIn("exactly one", both.events[0]["error"])

    def test_sequential_requests_share_one_import_and_match_cli_request_path(self):
        worker = self.start({"CHANDRA_FAKE_TOKENS": "5,7,248046,5,7,248046,9,248044"})
        ready = worker.expect(of("ready"))
        self.assertEqual((ready["mode"], ready["device_created"], ready["model_loaded"], ready["gemv_b1_selection"]), ("execute", True, True, "predecessor"))
        self.assertEqual(ready["model"], {"alias": "chandra", "revision": REVISION, "context_limit": 16384, "normal_output_allowance": 12384, "caller_stop_token_ids": [248044, 248046]})
        self.assertEqual((ready["job"]["kill_on_job_close"], ready["job"]["process_memory_limit_bytes"], ready["lease_ms"]), (True, 4294967296, 60000))
        self.assertEqual((ready["qualified_OCR"], ready["performance_claim"], ready["resident_model"]["imports"]), (False, False, 1))
        outputs = {}
        for ident, expected in (("r1", [5, 7, 248046]), ("r2", [5, 7, 248046]), ("r3", [9, 248044])):
            outputs[ident] = self.fresh("o").name
            worker.submit(ident, outputs[ident])
            worker.expect(of("admitted", ident, slot="active"))
            terminal = worker.expect(of("terminal", ident))
            tokens = [e["token_id"] for e in worker.events if e["event"] == "token" and e["id"] == ident]
            self.assertEqual(tokens, expected)
            self.assertEqual((terminal["status"], terminal["released"], terminal["drained"], terminal["request_retired"], terminal["worker_poisoned"]),
                             ("completed", True, True, True, False))
            self.assertEqual((terminal["tracked_buffer_bytes_after"], terminal["stop_reason"], terminal["generated_tokens"]),
                             (terminal["resident_model_bytes"], "stop_token", len(expected)))
            out = self.outputs / outputs[ident]
            self.assertEqual(sha((out / "result.json").read_bytes()), terminal["result_sha256"])
            report = json.loads((out / "result.json").read_text())
            self.assertEqual((report["passed"], report["generated_token_ids"], report["worker"]["request_id"], report["request_replayed"]), (True, expected, ident, False))
            rows = [json.loads(x) for x in (out / "generated-tokens.jsonl").read_text().splitlines()]
            events = [e for e in worker.events if e["event"] == "token" and e["id"] == ident]
            self.assertEqual([(r["generated_index"], r["token_id"], r["best_logit"], r["top_margin"]) for r in rows],
                             [(e["index"], e["token_id"], e["best_logit"], e["top_margin"]) for e in events])
        worker.control("status")
        status = worker.expect(of("status"))
        self.assertEqual((status["state"], status["admitted_total"], status["terminal_counts"]["completed"], status["status_wakes_device"]), ("idle", 3, 3, False))
        self.assertEqual(worker.finish(), 0)
        exit_event = worker.events[-1]
        self.assertEqual((exit_event["event"], exit_event["reason"], exit_event["clean"], exit_event["model_released"], exit_event["drained"], exit_event["tracked_buffer_bytes_after_release"]),
                         ("exit", "input_eof", True, True, True, 0))
        trace = read_trace(worker.trace)
        self.assertEqual([line.split()[0] for line in trace if line.split()[0] in ("device_create", "model_import_end", "model_release_end", "device_destroy")],
                         ["device_create", "model_import_end", "model_release_end", "device_destroy"])
        self.assertNotIn("foreign_thread", trace)
        last = self.weight_last(trace)
        sections = request_sections(trace)
        self.assertEqual(len(sections), 3)
        for section in sections:
            self.assertEqual(live_after(section, last), set())  # Every request-owned buffer is released inside its own section.
        self.assertEqual(canonical(sections[0], last), canonical(sections[1], last))
        # The CLI's request section (after import, before model release) is the same call sequence.
        _, cli_trace = self.run_cli(self.cli_new, "--execute", self.fresh("cli"), {"CHANDRA_FAKE_TOKENS": "5,7,248046"})
        start = next(i for i, line in enumerate(cli_trace) if line.startswith("model_import_end")) + 1
        end = next(i for i, line in enumerate(cli_trace) if line.startswith("model_release_begin"))
        self.assertEqual(canonical(cli_trace[start:end], self.weight_last(cli_trace)), canonical(sections[0], last))

    def test_event_stream_is_deterministic(self):
        streams = []
        for _ in range(2):
            worker = self.start({"CHANDRA_FAKE_TOKENS": "5,248046,7,248044"})
            worker.expect(of("ready"))
            for ident in ("a", "b"):
                worker.submit(ident, self.fresh("o").name)
                worker.expect(of("terminal", ident))
            self.assertEqual(worker.finish(), 0)
            streams.append([normalize({k: v for k, v in e.items() if k not in ("output", "result_sha256", "device_memory", "lease")}) for e in worker.events])
        self.assertEqual(streams[0], streams[1])
        kinds = [e["event"] for e in streams[0] if e.get("id") == "a"]
        self.assertEqual(kinds[:2], ["admitted", "started"])
        self.assertEqual(kinds[-1], "terminal")
        self.assertEqual([e["index"] for e in streams[0] if e["event"] == "token" and e["id"] == "a"], [0, 1])

    def test_queue_busy_and_cancellation_with_survivor(self):
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "5,7,9,248046", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_READ": 2})
        worker.expect(of("ready"))
        worker.submit("A", self.fresh("o").name)
        worker.expect(of("token", "A", index=0))  # A is now blocked inside its second logits readback.
        worker.submit("B", "out-b")
        self.assertEqual(worker.expect(of("admitted", "B"))["slot"], "waiting")
        worker.submit("C", self.fresh("o").name)
        busy = worker.expect(of("rejected"))
        self.assertEqual((busy["submitted_id"], busy["reason"]), ("C", "busy"))
        worker.control("status")
        status = worker.expect(of("status"))
        self.assertEqual((status["active"]["id"], status["active"]["started"], status["waiting"]["id"], status["accepting"]), ("A", True, "B", True))
        worker.control("cancel", id="B")
        self.assertEqual(worker.expect(of("cancel_requested", "B"))["state"], "waiting")
        canceled = worker.expect(of("terminal", "B"))
        self.assertEqual((canceled["status"], canceled["dispatched"], canceled["released"], canceled["generated_tokens"]), ("canceled", False, True, 0))
        self.assertEqual(json.loads((self.outputs / "out-b/result.json").read_text())["terminal_status"], "canceled")
        worker.control("cancel", id="nobody")
        self.assertEqual(worker.expect(of("cancel_rejected"))["reason"], "unknown_id")
        worker.submit("D", self.fresh("o").name)
        self.assertEqual(worker.expect(of("admitted", "D"))["slot"], "waiting")
        parked(gate)
        worker.control("cancel", id="A")
        acknowledged = worker.expect(of("cancel_requested", "A"))
        self.assertEqual((acknowledged["state"], acknowledged["released"]), ("active", False))
        worker.control("cancel", id="A")
        self.assertEqual(worker.expect(of("cancel_rejected"))["reason"], "cancel_already_requested")
        gate.write_text("open")
        a = worker.expect(of("terminal", "A"))
        self.assertEqual((a["status"], a["dispatched"], a["cancel_origin"], a["cancel_boundary"], a["generated_tokens"]), ("canceled", True, "client", "decode_step", 2))
        self.assertEqual((a["released"], a["drained"], a["request_retired"], a["worker_poisoned"], a["tracked_buffer_bytes_after"]), (True, True, True, False, a["resident_model_bytes"]))
        d = worker.expect(of("terminal", "D"))
        self.assertEqual((d["status"], d["generated_tokens"], d["stop_token_id"]), ("completed", 2, 248046))
        worker.control("cancel", id="A")
        self.assertEqual(worker.expect(of("cancel_rejected"))["reason"], "already_terminal")
        self.assertEqual(worker.finish(), 0)
        seq = {(e["event"], e.get("id")): e["seq"] for e in worker.events}
        self.assertLess(seq[("terminal", "A")], seq[("started", "D")])
        self.assertLess(seq[("cancel_requested", "A")], seq[("terminal", "A")])
        trace = read_trace(worker.trace)
        last = self.weight_last(trace)
        sections = request_sections(trace)
        self.assertEqual(len(sections), 2)  # B never reached the Device.
        self.assertTrue(all(live_after(s, last) == set() for s in sections))
        self.assertEqual(sum(line.startswith("logits ") for line in sections[0]), 2)
        canceled = sections[0]  # The cancellation drained after the last dispatch, before capacity was released.
        self.assertGreater(max(i for i, line in enumerate(canceled) if line == "drain"), max(i for i, line in enumerate(canceled) if line.startswith("dispatch ")))

    def test_cancel_during_vision_at_block_boundary_then_survivor(self):
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_DISPATCH": 40})
        worker.expect(of("ready"))
        worker.submit("v", self.fresh("o").name)
        parked(gate)
        worker.control("cancel", id="v")
        worker.expect(of("cancel_requested", "v"))
        gate.write_text("open")
        v = worker.expect(of("terminal", "v"))
        self.assertEqual((v["status"], v["generated_tokens"], v["released"], v["worker_poisoned"]), ("canceled", 0, True, False))
        self.assertTrue(v["cancel_boundary"].startswith("vision."), v["cancel_boundary"])
        worker.submit("w", self.fresh("o").name)
        self.assertEqual(worker.expect(of("terminal", "w"))["status"], "completed")
        self.assertEqual(worker.finish(), 0)
        trace = read_trace(worker.trace)
        first = request_sections(trace)[0]
        self.assertFalse(any(line.startswith("logits ") or "text_" in line for line in first))  # Stopped before any text graph work.
        self.assertEqual(live_after(first, self.weight_last(trace)), set())

    def test_eof_and_shutdown_retire_active_and_waiting_work_then_release_model(self):
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "5,7", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_READ": 1})
        worker.expect(of("ready"))
        worker.submit("x", self.fresh("o").name)
        parked(gate)
        worker.submit("y", self.fresh("o").name)
        worker.expect(of("admitted", "y", slot="waiting"))
        worker.proc.stdin.close()
        closing = worker.expect(of("closing"))
        self.assertEqual(closing["reason"], "input_eof")
        y = worker.expect(of("terminal", "y"))
        self.assertEqual((y["status"], y["dispatched"], y["cancel_origin"]), ("canceled", False, "input_eof"))
        worker.expect(of("cancel_requested", "x", origin="input_eof"))
        gate.write_text("open")
        x = worker.expect(of("terminal", "x"))
        self.assertEqual((x["status"], x["released"], x["generated_tokens"]), ("canceled", True, 1))
        self.assertEqual(worker.finish(close=False), 0)
        exit_event = worker.events[-1]
        self.assertEqual((exit_event["reason"], exit_event["clean"], exit_event["tracked_buffer_bytes_after_release"], exit_event["terminal_counts"]["canceled"]),
                         ("input_eof", True, 0, 2))
        trace = read_trace(worker.trace)
        self.assertEqual(trace[-1], "device_destroy")
        # Explicit shutdown with active work: admission stops at once, capacity is held until the cancel drains.
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "5", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_READ": 1})
        worker.expect(of("ready"))
        worker.submit("s", self.fresh("o").name)
        parked(gate)
        worker.control("shutdown")
        worker.expect(of("closing", reason="shutdown_requested"))
        self.assertEqual(worker.expect(of("cancel_requested", "s"))["origin"], "shutdown_requested")
        worker.submit("late", self.fresh("o").name)
        self.assertEqual(worker.expect(of("rejected"))["reason"], "closing")
        worker.control("status")
        status = worker.expect(of("status"))
        self.assertEqual((status["state"], status["accepting"], status["active"]["id"]), ("closing", False, "s"))
        gate.write_text("open")
        self.assertEqual(worker.expect(of("terminal", "s"))["status"], "canceled")
        self.assertEqual(worker.finish(), 0)
        self.assertEqual((worker.events[-1]["reason"], worker.events[-1]["clean"]), ("shutdown_requested", True))

    def test_device_failure_poisons_worker_without_replay(self):
        for name, env in (("dispatch", {"CHANDRA_FAKE_TOKENS": "5", "CHANDRA_FAKE_FAIL": "dispatch:900"}), ("nonfinite", {"CHANDRA_FAKE_TOKENS": "nan"})):
            with self.subTest(name):
                gate = self.fresh("gate")
                worker = self.start({**env, "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_DISPATCH": 10})
                worker.expect(of("ready"))
                worker.submit("p", self.fresh("o").name)
                worker.expect(of("started", "p"))
                worker.submit("q", self.fresh("o").name)
                worker.expect(of("admitted", "q", slot="waiting"))
                gate.write_text("open")
                p = worker.expect(of("terminal", "p"))
                self.assertEqual((p["status"], p["failure_class"], p["worker_poisoned"], p["dispatched"], p["request_replayed"]), ("failed", "device_or_graph_failure", True, True, False))
                worker.expect(of("closing", reason="worker_poisoned"))
                q = worker.expect(of("terminal", "q"))
                self.assertEqual((q["status"], q["failure_class"], q["dispatched"], q["request_replayed"]), ("failed", "worker_poisoned", False, False))
                self.assertEqual(worker.finish(), 2)
                self.assertEqual((worker.events[-1]["reason"], worker.events[-1]["clean"]), ("worker_poisoned", False))
                self.assertEqual((p["request_retired"], p["drained"], p["released"], p["tracked_buffer_bytes_after"]), (True, True, True, p["resident_model_bytes"]))
                trace = read_trace(worker.trace)
                self.assertEqual(len([line for line in trace if line.startswith("tracked ")]), 4)  # Resident, p before/after, final release: q never ran.
                failed = request_sections(trace)[0]
                self.assertEqual(live_after(failed, self.weight_last(trace)), set())  # The failed request's state retired with it.
                self.assertGreater(max(i for i, line in enumerate(failed) if line == "drain"), max(i for i, line in enumerate(failed) if line.startswith("dispatch ")))

    def test_cancel_without_proven_drain_poisons_instead_of_admitting_survivor(self):
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "5,7", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_READ": 2, "CHANDRA_FAKE_DRAIN_FAILS_AFTER_GATE": 1})
        worker.expect(of("ready"))
        worker.submit("h", self.fresh("o").name)
        parked(gate)
        worker.submit("next", self.fresh("o").name)
        worker.expect(of("admitted", "next", slot="waiting"))
        worker.control("cancel", id="h")
        worker.expect(of("cancel_requested", "h"))
        gate.write_text("open")
        h = worker.expect(of("terminal", "h"))
        self.assertEqual((h["status"], h["drained"], h["released"], h["worker_poisoned"]), ("canceled", False, False, True))
        nxt = worker.expect(of("terminal", "next"))
        self.assertEqual((nxt["status"], nxt["failure_class"], nxt["dispatched"]), ("failed", "worker_poisoned", False))
        self.assertEqual(worker.finish(), 2)
        self.assertEqual((worker.events[-1]["clean"], worker.events[-1]["drained"]), (False, False))
        self.assertEqual(len(request_sections(read_trace(worker.trace))), 1)  # No survivor reached the Device.

    def test_input_refusal_does_not_poison_and_admission_refusals_are_bounded(self):
        (self.inputs / "dir.json").mkdir()
        (self.inputs / "link.json").symlink_to(self.manifest)
        (self.inputs / "linkdir").symlink_to(self.inputs / "pkg")
        (self.outputs / "taken").mkdir()
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"})
        worker.expect(of("ready"))
        worker.submit("bad-hash", self.fresh("o").name, digest="f" * 64)
        refused = worker.expect(of("terminal", "bad-hash"))
        self.assertEqual((refused["status"], refused["failure_class"], refused["dispatched"], refused["worker_poisoned"], refused["released"]), ("failed", "input_refused", False, False, True))
        self.assertIn("SHA-256", refused["error"])
        worker.submit("good", self.fresh("o").name)
        self.assertEqual(worker.expect(of("terminal", "good"))["status"], "completed")
        base = {"schema": REQUEST_SCHEMA, "type": "submit", "id": "z", "model": "chandra", "input_manifest": "pkg/manifest.json", "input_sha256": self.digest, "output": "fresh-z"}
        cases = [
            ({"id": "good"}, "duplicate_id"), ({"id": ""}, "invalid_id"), ({"id": "a/b"}, "invalid_id"), ({"id": "x" * 65}, "invalid_id"), ({"id": 5}, "invalid_id"),
            ({"model": "chandra-2"}, "invalid_model"), ({"input_sha256": "F" * 64}, "invalid_sha256"), ({"input_sha256": "ab"}, "invalid_sha256"),
            ({"diagnostic_token_cap": 0}, "invalid_limit"), ({"diagnostic_token_cap": 12384}, "invalid_limit"), ({"diagnostic_token_cap": "5"}, "invalid_limit"),
            ({"diagnostic_token_cap": True}, "invalid_limit"), ({"diagnostic_token_cap": -1}, "invalid_limit"), ({"diagnostic_token_cap": 2.5}, "invalid_limit"),
            ({"input_manifest": "../inputs/pkg/manifest.json"}, "invalid_input_path"), ({"input_manifest": str(self.manifest)}, "invalid_input_path"),
            ({"input_manifest": "pkg\\manifest.json"}, "invalid_input_path"), ({"input_manifest": "pkg/./manifest.json"}, "invalid_input_path"),
            ({"input_manifest": "link.json"}, "invalid_input_path"), ({"input_manifest": "linkdir/manifest.json"}, "invalid_input_path"),
            ({"input_manifest": "dir.json"}, "invalid_input_path"), ({"input_manifest": "pkg/absent.json"}, "invalid_input_path"), ({"input_manifest": "CON"}, "invalid_input_path"),
            ({"output": ""}, "invalid_output_name"), ({"output": ".."}, "invalid_output_name"), ({"output": "a/b"}, "invalid_output_name"), ({"output": "con.txt"}, "invalid_output_name"),
            ({"output": ".hidden"}, "invalid_output_name"), ({"output": "x" * 65}, "invalid_output_name"), ({"output": "bad name"}, "invalid_output_name"),
            ({"output": "taken"}, "output_exists"), ({"extra": 1}, "unknown_field"),
        ]
        before = sorted(p.name for p in self.outputs.iterdir())
        for change, reason in cases:
            message = {**base, **change}
            worker.send(message)
            event = worker.expect(of("rejected"))
            self.assertEqual(event["reason"], reason, change)
        missing = dict(base)
        del missing["output"]
        worker.send(missing)
        self.assertEqual(worker.expect(of("rejected"))["reason"], "missing_field")
        self.assertEqual(sorted(p.name for p in self.outputs.iterdir()), before)  # No refusal created an output.
        self.assertEqual(worker.finish(), 0)
        self.assertEqual(len(request_sections(read_trace(worker.trace))), 1)  # Only "good" touched the Device.

    def test_malformed_protocol_is_bounded_and_closes_after_limit(self):
        worker = self.start(mode="--plan", job=None)
        worker.expect(of("ready", mode="plan"))
        worker.send(b"x" * 5000 + b"\n")
        self.assertEqual(worker.expect(of("protocol_error"))["reason"], "line_too_long")
        worker.control("status")
        self.assertEqual(worker.expect(of("status"))["protocol_errors"], 1)
        for raw, reason in ((b"{not json}\n", "malformed_json"), (b"[1,2]\n", "malformed_json"), (b'{"schema":"x","type":"status"}\n', "schema"),
                            (b'{"schema":"%s","type":"status","type":"status"}\n' % REQUEST_SCHEMA.encode(), "malformed_json"),
                            (b'{"schema":"%s","type":"reload"}\n' % REQUEST_SCHEMA.encode(), "unknown_type"),
                            (b'{"schema":"%s","type":"status","verbose":true}\n' % REQUEST_SCHEMA.encode(), "unknown_field"),
                            (b"\n", "malformed_json"), (b"\xff\xfe\n", "malformed_json")):
            worker.send(raw)
            self.assertEqual(worker.expect(of("protocol_error"))["reason"], reason, raw)
        for _ in range(7):
            worker.send(b"{}\n")
        closing = worker.expect(of("closing"))
        self.assertEqual(closing["reason"], "protocol_error_limit")
        self.assertEqual(worker.finish(), 3)
        self.assertEqual(max(e["count"] for e in worker.events if e["event"] == "protocol_error"), 16)

    def test_lease_expiry_releases_and_exits(self):
        worker = self.start(lease=1000)
        worker.expect(of("ready"))
        started = time.monotonic()
        worker.expect(of("closing", reason="lease_expired"), timeout=30)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.assertEqual(worker.finish(close=False), 3)
        self.assertEqual((worker.events[-1]["model_released"], worker.events[-1]["tracked_buffer_bytes_after_release"]), (True, 0))

    def test_plan_mode_forecasts_without_device_or_model(self):
        worker = self.start(mode="--plan", job=None, lease=None, pci="planned-pci")
        ready = worker.expect(of("ready"))
        self.assertEqual((ready["mode"], ready["device_created"], ready["model_loaded"], ready["resident_model"], ready["job"]["in_job"]), ("plan", False, False, None, False))
        worker.control("status")
        status = worker.expect(of("status"))
        self.assertEqual((status["device_created"], status["status_wakes_device"], status["qualified_OCR"]), (False, False, False))
        name = self.fresh("o").name
        worker.submit("plan-1", name, diagnostic_token_cap=9)
        terminal = worker.expect(of("terminal", "plan-1"))
        self.assertEqual((terminal["status"], terminal["dispatched"], terminal["released"]), ("planned", False, True))
        report = json.loads((self.outputs / name / "result.json").read_text())
        self.assertEqual((report["schema"], report["actual_inference"], report["device_created"], report["diagnostic_token_cap"], report["worker"]["resident_model"]),
                         ("chandra.directcompute.forecast.v1", False, False, 9, False))
        self.assertEqual(worker.finish(), 0)
        self.assertFalse(any(line.startswith(("device_create", "model_import")) for line in read_trace(worker.trace)))

    def test_ordered_selection_and_import_failure(self):
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046", "CHANDRA_EXPERIMENTAL_GEMV_B1": "ordered"})
        self.assertEqual(worker.expect(of("ready"))["gemv_b1_selection"], "ordered")
        worker.submit("g", self.fresh("o").name)
        self.assertEqual(worker.expect(of("terminal", "g"))["status"], "completed")
        self.assertEqual(worker.finish(), 0)
        self.assertIn("runtime/linear_gemv.hlsl", "\n".join(read_trace(worker.trace)))
        worker = self.start({"CHANDRA_FAKE_IMPORT_FAIL": "1"})
        self.assertEqual(worker.finish(), 1)
        self.assertEqual([e["event"] for e in worker.events], ["fatal", "exit"])
        self.assertEqual((worker.events[0]["device_created"], worker.events[0]["model_loaded"], worker.events[1]["device_destroyed"]), (True, False, True))

    def test_session_exerciser_records_events_and_closes_on_timeout(self):
        script = ROOT / "scripts/native/worker_session.py"
        session = self.fresh("session")
        steps = [{"send": {"schema": REQUEST_SCHEMA, "type": "submit", "id": "s1", "model": "chandra", "input_manifest": "pkg/manifest.json",
                           "input_sha256": self.digest, "output": self.fresh("o").name}},
                 {"expect": {"event": "terminal", "id": "s1"}}, {"send": {"schema": REQUEST_SCHEMA, "type": "status"}}, {"expect": {"event": "status"}}]
        session.write_text("".join(json.dumps(step) + "\n" for step in steps))
        env = self.env(CHANDRA_SHIM_JOB="kill,process_memory=4294967296", CHANDRA_FAKE_TOKENS="5,248046")
        events = self.fresh("events")
        result = subprocess.run([os.sys.executable, "-I", str(script), "--session", str(session), "--events", str(events), "--lease-seconds", "0.5",
                                 "--", str(self.worker), *self.worker_args()], capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual((summary["worker_exit_code"], summary["terminal_status"], summary["exit_event"]["reason"]), (0, {"s1": "completed"}, "input_eof"))
        lines = [json.loads(x) for x in events.read_text().splitlines()]
        self.assertEqual(([e["event"] for e in lines][0], [e["seq"] for e in lines]), ("ready", list(range(len(lines)))))
        again = subprocess.run([os.sys.executable, "-I", str(script), "--session", str(session), "--events", str(events), "--", str(self.worker)],
                               capture_output=True, text=True, timeout=60, env=env)
        self.assertNotEqual(again.returncode, 0)  # Existing evidence is never overwritten.
        session.write_text(json.dumps({"expect": {"event": "never"}, "timeout_seconds": 1}) + "\n")
        result = subprocess.run([os.sys.executable, "-I", str(script), "--session", str(session), "--events", str(self.fresh("events")),
                                 "--", str(self.worker), *self.worker_args()], capture_output=True, text=True, timeout=300, env=env)
        summary = json.loads(result.stdout)
        self.assertEqual((result.returncode, summary["worker_exit_code"], summary["exit_event"]["clean"]), (1, 0, True))
        self.assertIn("no event matching", summary["session_failure"])

    # ----- Input channel: protocol-error ceiling, end of input, read failures and read cancellation -----
    def assert_released(self, worker, code, reason, state, win32_error, cancel_requested):
        """Exit `code` after a clean release, with the exit event's input record; returns the exit event."""
        self.assertEqual(worker.finish(close=False), code)
        last = worker.events[-1]
        self.assertEqual((last["event"], last["reason"], last["exit_code"], last["clean"], last["model_released"], last["drained"], last["tracked_buffer_bytes_after_release"], last["request_replayed"]),
                         ("exit", reason, code, True, True, True, 0, False))
        self.assertEqual(last["input"], {"state": state, "win32_error": win32_error, "read_cancel_requested": cancel_requested, "reader_stopped": True})
        trace = read_trace(worker.trace)
        self.assertEqual(trace[-1], "device_destroy")
        self.assertNotIn("foreign_thread", trace)
        return last

    def hold_active_and_waiting(self, env=None, **args):
        """A worker whose active request is parked in its first logits readback and whose second request waits."""
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "5,7,248046", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_READ": 1, **(env or {})}, **args)
        worker.expect(of("ready"))
        worker.submit("held", self.fresh("o").name)
        parked(gate)
        worker.submit("queued", self.fresh("o").name)
        worker.expect(of("admitted", "queued", slot="waiting"))
        return worker, gate

    def assert_retired_without_replay(self, worker, origin):
        """The waiting request never dispatched; the active one stopped at a token boundary, drained and released capacity."""
        queued = next(e for e in worker.events if e["event"] == "terminal" and e["id"] == "queued")
        self.assertEqual((queued["status"], queued["dispatched"], queued["cancel_origin"], queued["released"], queued["request_replayed"]), ("canceled", False, origin, True, False))
        acknowledged = next(e for e in worker.events if e["event"] == "cancel_requested" and e["id"] == "held")
        self.assertEqual((acknowledged["state"], acknowledged["origin"], acknowledged["released"]), ("active", origin, False))
        held = next(e for e in worker.events if e["event"] == "terminal" and e["id"] == "held")
        self.assertEqual((held["status"], held["cancel_origin"], held["cancel_boundary"], held["dispatched"], held["request_replayed"]), ("canceled", origin, "decode_step", True, False))
        self.assertEqual((held["request_retired"], held["drained"], held["released"], held["worker_poisoned"], held["tracked_buffer_bytes_after"]), (True, True, True, False, held["resident_model_bytes"]))
        self.assertFalse(any(e["event"] == "started" and e.get("id") == "queued" for e in worker.events))
        trace = read_trace(worker.trace)
        sections = request_sections(trace)
        self.assertEqual(len(sections), 1)  # Only the held request reached the Device; nothing was replayed.
        self.assertEqual(live_after(sections[0], self.weight_last(trace)), set())
        self.assertGreater(max(i for i, line in enumerate(sections[0]) if line == "drain"), max(i for i, line in enumerate(sections[0]) if line.startswith("dispatch ")))

    def test_protocol_error_ceiling_stops_input_while_active_generation_is_held(self):
        worker, gate = self.hold_active_and_waiting()
        pid, before = worker.proc.pid, threads(worker.proc.pid)
        # One write of 64 malformed lines (3 over-long, 61 without a schema) while the Device thread is parked.
        worker.send((b"x" * 5000 + b"\n") * 3 + b"{}\n" * 61)
        self.assertEqual(worker.expect(of("closing"))["reason"], "protocol_error_limit")
        worker.expect(of("terminal", "queued"))
        worker.expect(of("cancel_requested", "held"))
        # The reader thread ends at the ceiling while the active request is still gated: no EOF is needed.
        eventually(lambda: threads(pid) == before - 1)
        self.assertFalse(gate.exists())
        worker.send(b"{}\n" * 64)  # Later input, including a valid status request, is never parsed.
        worker.control("status")
        gate.write_text("open")
        worker.expect(of("terminal", "held"))
        worker.proc.wait(timeout=60)  # Exits with stdin still open.
        last = self.assert_released(worker, 3, "protocol_error_limit", "protocol_error_limit", None, False)
        errors = [e for e in worker.events if e["event"] == "protocol_error"]
        self.assertEqual([e["count"] for e in errors], list(range(1, 17)))
        self.assertEqual([e["reason"] for e in errors], ["line_too_long"] * 3 + ["schema"] * 13)
        self.assertEqual((last["protocol_errors"], [e["event"] for e in worker.events].count("closing")), (16, 1))
        self.assertFalse(any(e["event"] in ("status", "rejected", "input_channel_failed") for e in worker.events))
        self.assert_retired_without_replay(worker, "protocol_error_limit")

    def test_protocol_error_ceiling_still_bounds_input_after_shutdown(self):
        worker, gate = self.hold_active_and_waiting()
        pid, before = worker.proc.pid, threads(worker.proc.pid)
        worker.control("shutdown")
        worker.expect(of("closing", reason="shutdown_requested"))
        worker.expect(of("cancel_requested", "held"))
        worker.send(b"{}\n" * 40)
        worker.expect(of("protocol_error", count=16))
        eventually(lambda: threads(pid) == before - 1)  # The reader ends at the ceiling, before the gated request finishes.
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        # The first close reason stands, but input that hit the ceiling is not an ordinary shutdown: exit 3.
        last = self.assert_released(worker, 3, "shutdown_requested", "protocol_error_limit", None, False)
        self.assertEqual((last["protocol_errors"], max(e["count"] for e in worker.events if e["event"] == "protocol_error")), (16, 16))
        self.assertEqual([e["event"] for e in worker.events].count("closing"), 1)
        self.assert_retired_without_replay(worker, "shutdown_requested")

    def test_end_of_input_branches_exit_zero(self):
        # The parent closes its pipe: the shim, like a Win32 anonymous pipe, fails the read with ERROR_BROKEN_PIPE (109).
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"})
        worker.expect(of("ready"))
        worker.proc.stdin.close()
        worker.expect(of("closing", reason="input_eof"))
        self.assert_released(worker, 0, "input_eof", "eof", 109, False)
        # A regular file ends with a successful zero-byte read; its unterminated last line is still a protocol error.
        messages = self.fresh("messages")
        messages.write_bytes((json.dumps({"schema": REQUEST_SCHEMA, "type": "status"}) + "\n" + '{"schema":').encode())
        with open(messages, "rb") as source:
            worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"}, stdin=source)
        worker.expect(of("exit"))
        self.assertEqual([e["event"] for e in worker.events], ["ready", "status", "protocol_error", "closing", "exit"])
        self.assertEqual(worker.events[2]["reason"], "incomplete_final_line")
        self.assert_released(worker, 0, "input_eof", "eof", None, False)
        # The null device also ends with a successful zero-byte read.
        worker = self.start(mode="--plan", job=None, stdin=subprocess.DEVNULL)
        worker.expect(of("closing", reason="input_eof"))
        self.assertEqual(worker.finish(), 0)
        self.assertEqual(worker.events[-1]["input"], {"state": "eof", "win32_error": None, "read_cancel_requested": False, "reader_stopped": True})

    def test_input_read_failures_at_idle_startup_report_status_and_exit_3(self):
        directory = os.open(self.inputs, os.O_RDONLY | os.O_DIRECTORY)
        write_only = open(self.fresh("write-only"), "wb")
        cases = [("directory", {}, directory, 1, "ERROR_INVALID_FUNCTION"), ("write-only", {}, write_only, 5, "ERROR_ACCESS_DENIED"),
                 ("invalid handle", {"CHANDRA_SHIM_READ_FAIL": "1:6"}, subprocess.PIPE, 6, "ERROR_INVALID_HANDLE"),
                 ("no data", {"CHANDRA_SHIM_READ_FAIL": "1:232"}, subprocess.PIPE, 232, "ERROR_NO_DATA"),
                 ("unnamed status", {"CHANDRA_SHIM_READ_FAIL": "1:1117"}, subprocess.PIPE, 1117, None)]
        try:
            for name, env, stdin, code, symbol in cases:
                with self.subTest(name):
                    worker = self.start({"CHANDRA_FAKE_TOKENS": "248046", **env}, stdin=stdin)
                    worker.expect(of("ready"))
                    failure = worker.expect(of("input_channel_failed"))
                    self.assertEqual((failure["win32_error"], failure["win32_name"], failure["unterminated_bytes"]), (code, symbol, 0))
                    self.assertIn("exits 3", failure["detail"])
                    self.assertLessEqual(len(json.dumps(failure)), 1024)
                    self.assertEqual(worker.expect(of("closing"))["reason"], "input_channel_failed")
                    worker.proc.wait(timeout=60)  # Ends without EOF, whatever the parent does with its end.
                    last = self.assert_released(worker, 3, "input_channel_failed", "failed", code, False)
                    self.assertEqual([e["event"] for e in worker.events], ["ready", "input_channel_failed", "closing", "exit"])
                    self.assertEqual((last["protocol_errors"], last["admitted_total"]), (0, 0))
        finally:
            os.close(directory)
            write_only.close()

    def test_input_read_failure_retires_waiting_and_cancels_active_without_replay(self):
        # Reads 1 and 2 deliver the two submissions; read 3 fails with ERROR_OPERATION_ABORTED that the worker never requested.
        worker, gate = self.hold_active_and_waiting({"CHANDRA_SHIM_READ_FAIL": "3:995"})
        failure = worker.expect(of("input_channel_failed"))
        self.assertEqual((failure["win32_error"], failure["win32_name"]), (995, "ERROR_OPERATION_ABORTED"))
        self.assertEqual(worker.expect(of("closing"))["reason"], "input_channel_failed")
        worker.expect(of("terminal", "queued"))
        worker.expect(of("cancel_requested", "held"))
        gate.write_text("open")
        worker.expect(of("terminal", "held"))
        worker.proc.wait(timeout=60)  # The reader ended at the failure by itself: finish() had no read to cancel.
        self.assert_released(worker, 3, "input_channel_failed", "failed", 995, False)
        self.assert_retired_without_replay(worker, "input_channel_failed")
        seq = {(e["event"], e.get("id")): e["seq"] for e in worker.events}
        self.assertLess(seq[("input_channel_failed", None)], seq[("closing", None)])
        self.assertLess(seq[("cancel_requested", "held")], seq[("terminal", "held")])

    def test_shutdown_and_lease_expiry_cancel_the_pending_read_without_eof_or_error(self):
        worker, gate = self.hold_active_and_waiting()
        worker.control("shutdown")
        worker.expect(of("closing", reason="shutdown_requested"))
        gate.write_text("open")
        worker.expect(of("terminal", "held"))
        worker.proc.wait(timeout=60)  # Stdin stays open: the worker canceled its own pending read after the drain.
        self.assert_released(worker, 0, "shutdown_requested", "canceled_at_exit", 995, True)
        self.assertFalse(any(e["event"] in ("input_channel_failed", "protocol_error") for e in worker.events))
        self.assert_retired_without_replay(worker, "shutdown_requested")
        worker = self.start(lease=1000)
        worker.expect(of("ready"))
        worker.expect(of("closing", reason="lease_expired"), timeout=30)
        worker.proc.wait(timeout=60)
        self.assert_released(worker, 3, "lease_expired", "canceled_at_exit", 995, True)
        self.assertFalse(any(e["event"] == "input_channel_failed" for e in worker.events))

    def test_session_exerciser_sends_raw_lines_and_observes_exit_with_stdin_open(self):
        script = ROOT / "scripts/native/worker_session.py"
        shutdown = {"schema": REQUEST_SCHEMA, "type": "shutdown"}
        for steps, code, state, errors in (([{"send_raw": "{}\n" * 20}, {"expect": {"event": "closing", "reason": "protocol_error_limit"}}, {"expect": {"event": "exit"}}], 3, "protocol_error_limit", 16),
                                           ([{"send": shutdown}, {"expect": {"event": "exit"}}], 0, "canceled_at_exit", 0)):
            with self.subTest(state):
                session, events = self.fresh("session"), self.fresh("events")
                session.write_text("".join(json.dumps(step) + "\n" for step in steps))
                result = subprocess.run([os.sys.executable, "-I", str(script), "--session", str(session), "--events", str(events), "--timeout-seconds", "60",
                                         "--", str(self.worker), *self.worker_args("--plan", lease=None)], capture_output=True, text=True, timeout=120, env=self.env())
                summary = json.loads(result.stdout)
                self.assertEqual((summary["session_failure"], summary["worker_exit_code"], result.returncode), (None, code, 0 if code == 0 else 1))
                self.assertEqual((summary["exit_event"]["input"]["state"], summary["exit_event"]["protocol_errors"]), (state, errors))
                self.assertEqual(sum(json.loads(x)["event"] == "protocol_error" for x in events.read_text().splitlines()), errors)

    # ----- Ordering: a recorded hard input or channel fact outlives whichever cause closed the worker first -----
    SHUTDOWN = (json.dumps({"schema": REQUEST_SCHEMA, "type": "shutdown"}) + "\n").encode()

    def submission(self, ident, output):
        return (json.dumps({"schema": REQUEST_SCHEMA, "type": "submit", "id": ident, "model": "chandra", "input_manifest": "pkg/manifest.json",
                            "input_sha256": self.digest, "output": output}) + "\n").encode()

    def lifecycle(self, worker):
        """Event kinds in order, without the startup and request-progress events whose interleaving with stdin is not fixed."""
        return [e["event"] for e in worker.events if e["event"] not in ("ready", "admitted", "started", "phase")]

    def test_sixteenth_error_supplied_by_end_of_input_after_shutdown_exits_3(self):
        after_shutdown = ["protocol_error"] * 15 + ["closing", "terminal", "cancel_requested", "protocol_error", "token", "terminal", "exit"]
        # The independent counterexample: 15 malformed lines, shutdown, then an unterminated byte whose EOF is error 16.
        worker, gate = self.hold_active_and_waiting()
        worker.send(b"{}\n" * 15 + self.SHUTDOWN + b"{")
        worker.proc.stdin.close()  # The shim, like an anonymous pipe, fails the next read with ERROR_BROKEN_PIPE (109).
        worker.expect(of("protocol_error", count=16, reason="incomplete_final_line"))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 3, "shutdown_requested", "protocol_error_limit", 109, False)
        self.assertEqual((last["protocol_errors"], self.lifecycle(worker)), (16, after_shutdown))
        self.assertEqual([e["reason"] for e in worker.events if e["event"] == "closing"], ["shutdown_requested"])
        self.assert_retired_without_replay(worker, "shutdown_requested")
        # The same ordering ended by a regular file's genuine successful zero-byte read, which has no Win32 status.
        gate, read_gate, source = self.fresh("gate"), self.fresh("read-gate"), self.fresh("stdin")
        source.write_bytes(self.submission("held", self.fresh("o").name) + self.submission("queued", self.fresh("o").name))
        with open(source, "rb") as stdin:
            worker = self.start({"CHANDRA_FAKE_TOKENS": "5,7,248046", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_READ": 1,
                                 "CHANDRA_SHIM_READ_GATE": f"2:{read_gate}"}, stdin=stdin)
        worker.expect(of("admitted", "queued", slot="waiting"))
        parked(gate)       # The active request is held in its first logits readback.
        parked(read_gate)  # The first read consumed both submissions; the second waits until the file grows.
        with open(source, "ab") as more:
            more.write(b"{}\n" * 15 + self.SHUTDOWN + b"{")
        read_gate.write_text("open")
        worker.expect(of("protocol_error", count=16, reason="incomplete_final_line"))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 3, "shutdown_requested", "protocol_error_limit", None, False)
        self.assertEqual((last["protocol_errors"], self.lifecycle(worker)), (16, after_shutdown))
        self.assertEqual([e["reason"] for e in worker.events if e["event"] == "closing"], ["shutdown_requested"])
        self.assert_retired_without_replay(worker, "shutdown_requested")

    def test_sixteenth_error_supplied_by_end_of_input_while_idle_exits_3(self):
        # With no work and no earlier close, the unterminated final line reaches the ceiling and is the close reason.
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"})
        worker.expect(of("ready"))
        worker.send(b"{}\n" * 15 + b"{")
        worker.proc.stdin.close()
        worker.proc.wait(timeout=60)
        self.assert_released(worker, 3, "protocol_error_limit", "protocol_error_limit", 109, False)
        self.assertEqual(self.lifecycle(worker), ["protocol_error"] * 16 + ["closing", "exit"])
        source = self.fresh("stdin")
        source.write_bytes(b"{}\n" * 15 + b"{")
        with open(source, "rb") as stdin:
            worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"}, stdin=stdin)
        worker.proc.wait(timeout=60)
        self.assert_released(worker, 3, "protocol_error_limit", "protocol_error_limit", None, False)
        self.assertEqual((worker.events[16]["reason"], self.lifecycle(worker)), ("incomplete_final_line", ["protocol_error"] * 16 + ["closing", "exit"]))
        # Idle after shutdown: error 16 arrives while the Device thread is releasing the model in its exit path, before it stops input.
        gate = self.fresh("gate")
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046", "CHANDRA_FAKE_GATE": gate, "CHANDRA_FAKE_GATE_DRAIN": 1})
        worker.expect(of("ready"))
        worker.send(b"{}\n" * 15 + self.SHUTDOWN + b"{")
        parked(gate)  # The exit-time drain after the model release.
        worker.proc.stdin.close()
        worker.expect(of("protocol_error", count=16, reason="incomplete_final_line"))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 3, "shutdown_requested", "protocol_error_limit", 109, False)
        self.assertEqual((last["protocol_errors"], self.lifecycle(worker)), (16, ["protocol_error"] * 15 + ["closing", "protocol_error", "exit"]))
        self.assertIn("gate_open drain 1", read_trace(worker.trace))

    def test_fifteen_errors_after_shutdown_end_cleanly_by_end_of_input_or_read_cancellation(self):
        # Genuine EOF after shutdown below the ceiling: 14 malformed lines and an unterminated one are 15 errors, exit 0.
        worker, gate = self.hold_active_and_waiting()
        worker.send(b"{}\n" * 14 + self.SHUTDOWN + b"{")
        worker.proc.stdin.close()
        worker.expect(of("protocol_error", count=15, reason="incomplete_final_line"))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 0, "shutdown_requested", "eof", 109, False)
        self.assertEqual(last["protocol_errors"], 15)
        self.assert_retired_without_replay(worker, "shutdown_requested")
        # Deliberate exit-time read cancellation after shutdown and 15 errors with stdin held open, exit 0.
        worker, gate = self.hold_active_and_waiting()
        worker.send(self.SHUTDOWN + b"{}\n" * 15)
        worker.expect(of("protocol_error", count=15))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 0, "shutdown_requested", "canceled_at_exit", 995, True)
        self.assertEqual(last["protocol_errors"], 15)
        self.assertFalse(any(e["event"] == "input_channel_failed" for e in worker.events))
        self.assert_retired_without_replay(worker, "shutdown_requested")

    def test_end_of_input_after_the_ceiling_is_never_read_or_counted(self):
        # The ceiling ended input first: the unterminated byte and EOF behind it are never read, so no read status is recorded.
        worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"})
        worker.expect(of("ready"))
        worker.send(b"{}\n" * 16 + b"{")
        worker.proc.stdin.close()
        worker.proc.wait(timeout=60)
        self.assert_released(worker, 3, "protocol_error_limit", "protocol_error_limit", None, False)
        self.assertEqual(self.lifecycle(worker), ["protocol_error"] * 16 + ["closing", "exit"])
        source = self.fresh("stdin")
        source.write_bytes(b"{}\n" * 16 + b"{")
        with open(source, "rb") as stdin:
            worker = self.start({"CHANDRA_FAKE_TOKENS": "248046"}, stdin=stdin)
        worker.proc.wait(timeout=60)
        self.assert_released(worker, 3, "protocol_error_limit", "protocol_error_limit", None, False)
        self.assertFalse(any(e.get("reason") == "incomplete_final_line" for e in worker.events))
        worker, gate = self.hold_active_and_waiting()
        worker.send(self.SHUTDOWN + b"{}\n" * 16 + b"{")
        worker.proc.stdin.close()
        worker.expect(of("protocol_error", count=16))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 3, "shutdown_requested", "protocol_error_limit", None, False)
        self.assertEqual((last["protocol_errors"], [e["reason"] for e in worker.events if e["event"] == "protocol_error"]), (16, ["schema"] * 16))
        self.assert_retired_without_replay(worker, "shutdown_requested")

    def test_read_failure_after_shutdown_keeps_its_status_and_exits_3(self):
        # Reads 1 and 2 deliver the submissions, read 3 the shutdown, 15 malformed lines and one unterminated byte; read 4
        # fails with ERROR_NO_DATA. The failure, not EOF, ended input, so the partial line is reported but not counted.
        worker, gate = self.hold_active_and_waiting({"CHANDRA_SHIM_READ_FAIL": "4:232"})
        worker.send(self.SHUTDOWN + b"{}\n" * 15 + b"{")
        failure = worker.expect(of("input_channel_failed"))
        self.assertEqual((failure["win32_error"], failure["win32_name"], failure["unterminated_bytes"]), (232, "ERROR_NO_DATA", 1))
        gate.write_text("open")
        worker.proc.wait(timeout=60)
        last = self.assert_released(worker, 3, "shutdown_requested", "failed", 232, False)
        self.assertEqual((last["protocol_errors"], self.lifecycle(worker)),
                         (15, ["closing", "terminal", "cancel_requested"] + ["protocol_error"] * 15 + ["input_channel_failed", "token", "terminal", "exit"]))
        self.assert_retired_without_replay(worker, "shutdown_requested")

    def test_broken_event_channel_after_shutdown_or_eof_drains_and_exits_3(self):
        for closing in ("shutdown_requested", "input_eof"):
            with self.subTest(closing):
                gate, trace = self.fresh("gate"), self.fresh("broken-trace")
                env = self.env(CHANDRA_SHIM_JOB="kill,process_memory=4294967296", CHANDRA_FAKE_TRACE=trace, CHANDRA_FAKE_TOKENS="5,7,248046",
                               CHANDRA_FAKE_GATE=gate, CHANDRA_FAKE_GATE_READ=1)
                proc = subprocess.Popen([str(self.worker), *self.worker_args()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
                self.assertEqual(json.loads(proc.stdout.readline())["event"], "ready")
                proc.stdin.write(self.submission("held", self.fresh("o").name))
                proc.stdin.flush()
                parked(gate)
                if closing == "shutdown_requested":
                    proc.stdin.write(self.SHUTDOWN)
                    proc.stdin.flush()
                else:
                    proc.stdin.close()
                seen = []
                while not seen or seen[-1]["event"] != "cancel_requested":
                    seen.append(json.loads(proc.stdout.readline()))
                self.assertEqual([(e["event"], e.get("reason")) for e in seen if e["event"] == "closing"], [("closing", closing)])
                proc.stdout.close()  # The parent stops reading: the held request's next event write fails.
                gate.write_text("open")
                code = proc.wait(timeout=60)
                if not proc.stdin.closed:
                    proc.stdin.close()
                # The first close reason stands, but the events after it were lost, so this is not an ordinary exit 0.
                self.assertEqual(code, 3)
                lines = read_trace(trace)
                sections = request_sections(lines)
                self.assertEqual(len(sections), 1)
                self.assertEqual(live_after(sections[0], self.weight_last(lines)), set())
                self.assertEqual(lines[-4:], ["model_release_end", "drain", "tracked 0", "device_destroy"])

    def test_first_failed_write_at_the_final_exit_event_turns_only_a_clean_exit_into_3(self):
        # An idle worker closes and is held in its exit-time drain, so `exit` is its only remaining stdout write. Each case runs
        # twice from there: read, the event is delivered and carries the process code; with the reader closed, that final write
        # is the first to fail, so a clean EOF or shutdown exits 3 while startup (1), unclean release (2) and lease (3) are kept.
        cases = [  # name, extra env, close, events before exit, delivered exit (reason, clean, input state), delivered code, lost code
            ("input_eof", {}, "eof", ["ready", "closing"], ("input_eof", True, "eof"), 0, 3),
            ("shutdown_requested", {}, "shutdown", ["ready", "closing"], ("shutdown_requested", True, "canceled_at_exit"), 0, 3),
            ("lease_expired", {}, "lease", ["ready", "closing"], ("lease_expired", True, "canceled_at_exit"), 3, 3),
            ("unclean_release", {"CHANDRA_FAKE_FAIL": "drain:1"}, "eof", ["ready", "closing"], ("input_eof", False, "eof"), 2, 2),
            ("startup_failed", {"CHANDRA_FAKE_IMPORT_FAIL": 1}, None, ["fatal"], ("startup_failed", False, "not_started"), 1, 1),
        ]
        for name, extra, close, before, (reason, clean, state), delivered_code, lost_code in cases:
            traces = []
            for delivered in (True, False):
                with self.subTest(name, exit_delivered=delivered):
                    gate, trace = self.fresh("gate"), self.fresh("final-write-trace")
                    env = self.env(CHANDRA_SHIM_JOB="kill,process_memory=4294967296", CHANDRA_FAKE_TRACE=trace, CHANDRA_FAKE_TOKENS="248046",
                                   CHANDRA_FAKE_GATE=gate, CHANDRA_FAKE_GATE_DRAIN=1, **extra)
                    proc = subprocess.Popen([str(self.worker), *self.worker_args(lease=1000 if close == "lease" else 60000)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                    watchdog = threading.Timer(120, proc.kill)  # Bounds every blocking read below.
                    watchdog.start()
                    try:
                        events = [json.loads(proc.stdout.readline())]
                        if close == "eof":
                            proc.stdin.close()
                        elif close == "shutdown":
                            proc.stdin.write(self.SHUTDOWN)
                            proc.stdin.flush()
                        while len(events) < len(before):
                            events.append(json.loads(proc.stdout.readline()))
                        self.assertEqual([(e["event"], e["seq"]) for e in events], list(zip(before, range(len(before)))))
                        self.assertEqual(events[-1]["reason"] if close else events[-1]["stage"], reason if close else "startup")
                        parked(gate)  # The exit-time drain after the model release: only `exit` remains to be written.
                        if delivered:
                            gate.write_text("open")
                            raw = proc.stdout.readline()
                            self.assertTrue(raw.endswith(b"\n"))
                            final = json.loads(raw)
                            self.assertEqual(proc.stdout.read(), b"")  # `exit` is the last line.
                            proc.stdout.close()
                        else:
                            proc.stdout.close()  # The parent stops reading: the final write is the first to fail.
                            gate.write_text("open")
                        code = proc.wait(timeout=60)
                    finally:
                        watchdog.cancel()
                        if proc.poll() is None:  # Only after a failed assertion above.
                            proc.kill()
                            proc.wait(timeout=60)
                    if not proc.stdin.closed:
                        proc.stdin.close()
                    stderr = proc.stderr.read()
                    proc.stderr.close()
                    lines = read_trace(trace)
                    traces.append(lines)
                    self.assertIn("gate_open drain 1", lines)
                    self.assertNotIn("foreign_thread", lines)
                    self.assertEqual(lines[-2:], ["tracked 0", "device_destroy"])
                    self.assertEqual((code, stderr), (delivered_code if delivered else lost_code, b""))
                    if delivered:
                        self.assertEqual((final["event"], final["seq"], final["reason"], final["exit_code"], final["clean"], final["input"]["state"],
                                          final["tracked_buffer_bytes_after_release"], final["device_destroyed"], final["request_replayed"]),
                                         ("exit", len(before), reason, code, clean, state, 0, True, False))
            with self.subTest(name, compare="release traces"):
                # Losing the final event changes only the reported status: release, drain and destruction are identical.
                self.assertEqual(len(traces), 2)
                self.assertEqual(traces[0], traces[1])

    def test_session_exerciser_reports_the_ceiling_reached_at_end_of_input(self):
        # The pending Windows plan-mode check: the session's only step writes 15 malformed lines and an unterminated byte,
        # then the exerciser closes stdin, so EOF supplies error 16.
        script, session, events = ROOT / "scripts/native/worker_session.py", self.fresh("session"), self.fresh("events")
        session.write_text(json.dumps({"send_raw": "{}\n" * 15 + "{"}) + "\n")
        result = subprocess.run([os.sys.executable, "-I", str(script), "--session", str(session), "--events", str(events), "--timeout-seconds", "60",
                                 "--", str(self.worker), *self.worker_args("--plan", lease=None)], capture_output=True, text=True, timeout=120, env=self.env())
        summary = json.loads(result.stdout)
        self.assertEqual((summary["session_failure"], summary["worker_exit_code"], result.returncode), (None, 3, 1))
        self.assertEqual((summary["exit_event"]["reason"], summary["exit_event"]["protocol_errors"], summary["exit_event"]["input"]),
                         ("protocol_error_limit", 16, {"state": "protocol_error_limit", "win32_error": 109, "read_cancel_requested": False, "reader_stopped": True}))
        errors = [e for e in map(json.loads, events.read_text().splitlines()) if e["event"] == "protocol_error"]
        self.assertEqual(([e["count"] for e in errors], errors[-1]["reason"]), (list(range(1, 17)), "incomplete_final_line"))


if __name__ == "__main__":
    unittest.main()
