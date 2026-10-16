"""Small configuration/authentication tests; no model, Windows API or network is used."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from runtime.llama.config import (ACTIVATION, SYCL_ACTIVATION, SYCL_DECODE_GRAPH_LEAF_SHA256,
                                 SYCL_DECODE_GRAPH_MODEL_SHA256, load_config)
from runtime.native.winjob import command_line


class ConfigTests(unittest.TestCase):
    def test_imports_are_from_the_checkout_containing_these_tests(self):
        from runtime.llama import backend, config
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(Path(config.__file__).resolve(), root / "runtime/llama/config.py")
        self.assertEqual(Path(backend.__file__).resolve(), root / "runtime/llama/backend.py")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()

        def file(path):
            value = path.name.encode()
            path.write_bytes(value)
            return {"path": str(path), "size": len(value), "sha256": hashlib.sha256(value).hexdigest()}

        self.raw = {"activate": ACTIVATION, "runtime_dir": str(self.runtime),
                    "runtime_files": [file(self.runtime / "llama-server.exe"), file(self.runtime / "ggml-vulkan.dll")],
                    "model": file(self.root / "model.gguf"), "projector": file(self.root / "projector.gguf"),
                    "template": file(self.root / "template.jinja"), "device_index": 0, "worker_cpu": 0,
                    "startup_seconds": 1200, "request_seconds": 1200, "max_image_pixels": 4_000_000, "parallel_requests": 1}

    def config(self):
        data = json.dumps(self.raw).encode()
        path = self.root / "config.json"
        path.write_bytes(data)
        return load_config(path, hashlib.sha256(data).hexdigest())

    def select_sycl(self):
        self.raw["activate"] = SYCL_ACTIVATION
        self.raw["sycl_device_index"] = self.raw.pop("device_index", 0)
        item = self.raw["runtime_files"][1]
        path = self.runtime / "ggml-sycl.dll"
        Path(item["path"]).rename(path)
        item["path"] = str(path)

    def select_c5_b256(self):
        self.select_sycl()
        cache = self.root / "private-cache"
        cache.mkdir()
        self.raw.update(parallel_requests=5, sycl_batch_size=256, sycl_kv_cache_type="q8_0",
                        admission_physical_bytes=21 * (1 << 28), admission_commit_bytes=49 * (1 << 28),
                        sycl_host_cpu_count=4, sycl_cache_dir=str(cache),
                        sycl_vision_leaf_sha256=self.raw["runtime_files"][1]["sha256"])

    def select_decode_graph(self):
        self.select_c5_b256()
        self.raw.update(sycl_decode_graph=True, sycl_vision_leaf_sha256=SYCL_DECODE_GRAPH_LEAF_SHA256)
        self.raw["runtime_files"][1]["sha256"] = SYCL_DECODE_GRAPH_LEAF_SHA256
        self.raw["model"]["sha256"] = SYCL_DECODE_GRAPH_MODEL_SHA256

    def test_authentication_covers_code_weights_projector_template_and_added_dll(self):
        config = self.config()
        config.authenticate()
        self.assertEqual(len(config.files()), 5)
        self.assertEqual(config.admission_physical_bytes, 6 << 30)
        self.assertEqual(config.admission_commit_bytes, 11 << 30)
        extra = self.runtime / "unexpected.dll"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            config.authenticate()
        extra.unlink()
        config.model.path.write_bytes(b"x" * config.model.size)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            config.authenticate()

    def test_argv_fits_existing_transport_and_preserves_selected_profile(self):
        for parallel in (1, 2, 4):
            self.raw["parallel_requests"] = parallel
            config = self.config()
            argv = config.argv(18003, "secret", "private-alias")
            command_line(argv)
            self.assertLessEqual(len(argv), 64)
            self.assertEqual("--no-kv-unified" in argv, parallel > 1)
            for flag, value in (("--ctx-size", str(16384 * parallel)), ("--parallel", str(parallel)),
                                ("--batch-size", "128"), ("--ubatch-size", "128"), ("--n-predict", "12384"),
                                ("--cache-type-k", "bf16"), ("--cache-type-v", "bf16"), ("--fit", "off"), ("--load-mode", "none"),
                                ("--host", "127.0.0.1"), ("--device", "Vulkan0"), ("--mmproj-device", "Vulkan0")):
                self.assertEqual(argv[argv.index(flag) + 1], value)
        limits = config.job_limits()
        self.assertEqual((limits.process_memory, limits.job_memory, limits.affinity), (21 * (1 << 29), 43 * (1 << 28), 1))

    def test_parallel_profile_is_explicit_and_bounded(self):
        for parallel in (True, 0, 3, 8, "2"):
            self.raw["parallel_requests"] = parallel
            with self.assertRaisesRegex(ValueError, "parallel_requests"):
                self.config()

    def test_q8_kv_changes_only_both_cache_arguments_and_config_identity(self):
        from unittest.mock import patch
        self.select_sycl()
        self.raw["parallel_requests"] = 4
        cache = self.root / "cache"
        cache.mkdir()
        self.raw.update(sycl_cache_dir=str(cache), sycl_host_cpu_count=4,
                        sycl_vision_leaf_sha256=self.raw["runtime_files"][1]["sha256"])
        default = self.config()
        self.assertEqual(default.sycl_kv_cache_type, "bf16")
        self.raw["sycl_kv_cache_type"] = "bf16"
        explicit = self.config()
        self.assertEqual(default.argv(18003, "key", "alias"), explicit.argv(18003, "key", "alias"))
        self.raw["sycl_kv_cache_type"] = "q8_0"
        q8 = self.config()
        expected = default.argv(18003, "key", "alias")
        for flag in ("--cache-type-k", "--cache-type-v"):
            expected[expected.index(flag) + 1] = "q8_0"
        self.assertEqual(q8.argv(18003, "key", "alias"), expected)
        self.assertNotEqual(q8.config_sha256, default.config_sha256)
        self.assertEqual(q8.files(), default.files())
        self.assertEqual(q8.job_limits(), default.job_limits())
        with patch.dict("os.environ", {"SYSTEMROOT": str(self.root), "LLAMA_ARG_CACHE_TYPE_K": "f32",
                                       "LLAMA_ARG_CACHE_TYPE_V": "f32"}, clear=True):
            self.assertEqual(q8.environment(), default.environment())
            self.assertNotIn("LLAMA_ARG_CACHE_TYPE_K", q8.environment())
            self.assertEqual(os.environ["LLAMA_ARG_CACHE_TYPE_K"], "f32")

    def test_six_slots_require_explicit_q8_and_both_conservative_forecasts(self):
        from dataclasses import replace
        from runtime.llama.backend import LlamaBackend
        from runtime.llama.memory import FLOOR, check_memory
        self.select_sycl()
        self.raw.update(parallel_requests=6, sycl_kv_cache_type="q8_0",
                        admission_physical_bytes=21 * (1 << 28), admission_commit_bytes=49 * (1 << 28))
        good = dict(self.raw)
        for field, value in (("sycl_kv_cache_type", None), ("sycl_kv_cache_type", "bf16"),
                             ("admission_physical_bytes", None), ("admission_commit_bytes", None),
                             ("admission_physical_bytes", good["admission_physical_bytes"] - 1),
                             ("admission_commit_bytes", good["admission_commit_bytes"] - 1),
                             ("admission_physical_bytes", True), ("admission_commit_bytes", 49.0 * (1 << 28))):
            self.raw = dict(good)
            if value is None:
                del self.raw[field]
            else:
                self.raw[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.config()
        self.raw = good
        config = self.config()
        argv = config.argv(18003, "key", "alias")
        command_line(argv)
        self.assertLessEqual(len(argv), 64)
        for flag, value in (("--ctx-size", "98304"), ("--parallel", "6"), ("--n-predict", "12384"),
                            ("--batch-size", "128"), ("--ubatch-size", "128"),
                            ("--cache-type-k", "q8_0"), ("--cache-type-v", "q8_0")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--no-kv-unified", argv)
        limits = config.job_limits()
        self.assertEqual((limits.process_memory, limits.job_memory, limits.affinity),
                         (21 * (1 << 29), 43 * (1 << 28), 1))
        self.assertEqual(FLOOR, 4 << 30)
        for key, forecast in (("available_physical", config.admission_physical_bytes),
                              ("available_commit", config.admission_commit_bytes)):
            observed = {"available_physical": 40 << 30, "available_commit": 40 << 30}
            observed[key] = forecast + FLOOR
            with self.assertRaises(RuntimeError):
                check_memory(observed, config.admission_physical_bytes, config.admission_commit_bytes)
        for changes in ({"sycl_kv_cache_type": "bf16"}, {"engine": "vulkan"},
                        {"admission_commit_bytes": 11 << 30}, {"parallel_requests": 8}):
            invalid = replace(config, **changes)
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    invalid.argv(18003, "key", "alias")
                with self.assertRaises(ValueError):
                    LlamaBackend(invalid)

    def test_c5_b256_preserves_every_other_argument_environment_and_resource_limit(self):
        from dataclasses import replace
        from unittest.mock import patch
        from runtime.llama.backend import LlamaBackend
        from runtime.llama.memory import FLOOR, check_memory
        self.select_c5_b256()
        config = self.config()
        config.authenticate()
        c6 = replace(config, parallel_requests=6, sycl_batch_size=128)
        expected = c6.argv(18003, "key", "alias")
        for flag, value in (("--ctx-size", "81920"), ("--parallel", "5"),
                            ("--batch-size", "256"), ("--ubatch-size", "256")):
            expected[expected.index(flag) + 1] = value
        self.assertEqual(config.argv(18003, "key", "alias"), expected)
        command_line(expected)
        self.assertLessEqual(len(expected), 64)
        self.assertIn("--no-kv-unified", expected)
        self.assertEqual(config.files(), c6.files())
        self.assertEqual(config.job_limits(), c6.job_limits())
        limits = config.job_limits()
        self.assertEqual((limits.process_memory, limits.job_memory, limits.affinity),
                         (21 * (1 << 29), 43 * (1 << 28), 15))
        with patch.dict("os.environ", {"SYSTEMROOT": str(self.root), "LLAMA_ARG_BATCH": "512",
                                       "GGML_SYCL_ENABLE_GRAPH": "1"}, clear=True):
            self.assertEqual(config.environment(), c6.environment())
            self.assertNotIn("LLAMA_ARG_BATCH", config.environment())
            self.assertNotIn("GGML_SYCL_ENABLE_GRAPH", config.environment())
            self.assertEqual(config.environment()["OMP_NUM_THREADS"], "1")
            self.assertEqual(config.environment()["MKL_NUM_THREADS"], "1")
        self.assertEqual(LlamaBackend(config).parallel_requests, 5)
        for key, forecast in (("available_physical", config.admission_physical_bytes),
                              ("available_commit", config.admission_commit_bytes)):
            observed = {"available_physical": 40 << 30, "available_commit": 40 << 30}
            observed[key] = forecast + FLOOR
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_memory(observed, config.admission_physical_bytes, config.admission_commit_bytes)
            observed[key] += 1
            check_memory(observed, config.admission_physical_bytes, config.admission_commit_bytes)

    def test_c5_b256_refuses_unmeasured_shapes_precision_affinity_and_missing_forecasts(self):
        from dataclasses import replace
        from runtime.llama.backend import LlamaBackend
        self.select_c5_b256()
        good = dict(self.raw)
        config = self.config()
        changes = [("sycl_batch_size", value) for value in (None, True, 128, 256.0, "256", 512)]
        changes += [("parallel_requests", value) for value in (True, 3, 7, 8)]
        changes += [("sycl_kv_cache_type", value) for value in (None, "bf16", "f16")]
        changes += [("sycl_host_cpu_count", value) for value in (None, 1, 8, True)]
        changes += [("worker_cpu", value) for value in (True, 1)]
        changes += [("sycl_cache_dir", None), ("sycl_vision_leaf_sha256", None),
                    ("sycl_vision_leaf_sha256", "0" * 64), ("context", 8192), ("max_output_tokens", 512)]
        for field in ("admission_physical_bytes", "admission_commit_bytes"):
            changes += [(field, value) for value in (None, good[field] - 1, True, float(good[field]), 33 << 30)]
        for field, value in changes:
            self.raw = dict(good)
            if value is None:
                del self.raw[field]
            else:
                self.raw[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.config()
        for changes in ({"parallel_requests": 4}, {"parallel_requests": 6}, {"engine": "vulkan"},
                        {"sycl_batch_size": 128}, {"sycl_batch_size": True}, {"sycl_kv_cache_type": "bf16"},
                        {"sycl_host_cpu_count": 1}, {"worker_cpu": True}, {"sycl_cache_dir": None},
                        {"sycl_vision_leaf_sha256": None}, {"admission_physical_bytes": 5 << 30},
                        {"admission_commit_bytes": 12 << 30}):
            invalid = replace(config, **changes)
            with self.subTest(direct=changes):
                with self.assertRaises(ValueError):
                    invalid.argv(18003, "key", "alias")
                with self.assertRaises(ValueError):
                    LlamaBackend(invalid)

    def test_batch_option_is_sycl_only_and_256_is_exclusive_to_c5(self):
        for value in (128, 256):
            self.raw["sycl_batch_size"] = value
            with self.subTest(engine="vulkan", value=value), self.assertRaisesRegex(ValueError, "engine-specific"):
                self.config()
        del self.raw["sycl_batch_size"]
        self.select_sycl()
        for parallel in (1, 2, 4, 6):
            self.raw.update(parallel_requests=parallel, sycl_batch_size=256, sycl_kv_cache_type="q8_0",
                            admission_physical_bytes=21 * (1 << 28), admission_commit_bytes=49 * (1 << 28))
            with self.subTest(parallel=parallel), self.assertRaisesRegex(ValueError, "Batch 256"):
                self.config()

    def test_decode_graph_adds_only_early_info_option_and_explicit_graph_environment(self):
        from dataclasses import replace
        from unittest.mock import patch
        from runtime.llama.backend import LlamaBackend
        self.select_decode_graph()
        graph = self.config()
        eager = replace(graph, sycl_decode_graph=False)
        argv = graph.argv(18003, "key", "alias")
        self.assertEqual(argv[:3], [str(self.runtime / "llama-server.exe"), "--log-verbosity", "4"])
        self.assertLess(argv.index("--log-verbosity"), argv.index("--device"))
        self.assertLess(argv.index("--log-verbosity"), argv.index("--mmproj-device"))
        self.assertEqual(argv[:1] + argv[3:], eager.argv(18003, "key", "alias"))
        command_line(argv)
        self.assertEqual(len(argv), 64)
        for flag, value in (("--ctx-size", "81920"), ("--n-predict", "12384"), ("--parallel", "5"),
                            ("--batch-size", "256"), ("--ubatch-size", "256"),
                            ("--threads", "1"), ("--threads-batch", "1"),
                            ("--cache-type-k", "q8_0"), ("--cache-type-v", "q8_0")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        with patch.dict("os.environ", {"SYSTEMROOT": str(self.root), "GGML_SYCL_ENABLE_GRAPH": "1",
                                       "GGML_SYCL_DECODE_GRAPH": "1", "GGML_SYCL_DEBUG": "1",
                                       "GGML_SYCL_DEV_DEBUG": "1"}, clear=True):
            expected = eager.environment()
            for key in ("GGML_SYCL_ENABLE_GRAPH", "GGML_SYCL_DECODE_GRAPH", "GGML_SYCL_DEBUG", "GGML_SYCL_DEV_DEBUG"):
                self.assertNotIn(key, expected)
            expected.update(GGML_SYCL_ENABLE_GRAPH="0", GGML_SYCL_DECODE_GRAPH="3")
            self.assertEqual(graph.environment(), expected)
        self.assertEqual(graph.files(), eager.files())
        self.assertEqual(graph.job_limits(), eager.job_limits())
        self.assertEqual(LlamaBackend(graph).parallel_requests, 5)
        # Pinned names/hashes do not authenticate these deliberately synthetic fixture bytes.
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            graph.authenticate()

    def test_decode_graph_omission_and_false_preserve_every_existing_profile(self):
        from unittest.mock import patch
        vulkan = self.config()
        vulkan_raw = dict(self.raw)
        self.assertIs(vulkan.sycl_decode_graph, False)
        self.assertNotIn("--log-verbosity", vulkan.argv(18003, "key", "alias"))
        self.select_sycl()
        for slots, kv, batch in ((1, "bf16", 128), (2, "q8_0", 128), (4, "bf16", 128), (6, "q8_0", 128)):
            self.raw.update(parallel_requests=slots, sycl_kv_cache_type=kv, sycl_batch_size=batch,
                            admission_physical_bytes=21 * (1 << 28), admission_commit_bytes=49 * (1 << 28))
            omitted = self.config()
            self.raw["sycl_decode_graph"] = False
            explicit = self.config()
            self.assertIs(omitted.sycl_decode_graph, False)
            self.assertEqual(omitted.argv(18003, "key", "alias"), explicit.argv(18003, "key", "alias"))
            with patch.dict("os.environ", {"SYSTEMROOT": str(self.root)}, clear=True):
                self.assertEqual(omitted.environment(), explicit.environment())
            self.assertEqual(omitted.job_limits(), explicit.job_limits())
            del self.raw["sycl_decode_graph"]
        for value in (False, True):
            self.raw = {**vulkan_raw, "sycl_decode_graph": value}
            with self.assertRaisesRegex(ValueError, "engine-specific"):
                self.config()

    def test_decode_graph_rejects_other_shapes_identities_types_and_context_overrides(self):
        import copy
        from dataclasses import replace
        from types import SimpleNamespace
        from runtime.llama.backend import LlamaBackend
        self.select_decode_graph()
        good = copy.deepcopy(self.raw)
        config = self.config()
        changes = [("sycl_decode_graph", value) for value in (None, 0, 1, "true", [], {})]
        changes += [("parallel_requests", value) for value in (1, 2, 4, 6, True)]
        changes += [("sycl_batch_size", value) for value in (128, 512, True, 256.0)]
        changes += [("sycl_kv_cache_type", value) for value in ("bf16", "f16", True)]
        changes += [("sycl_host_cpu_count", value) for value in (1, 8, True)]
        changes += [("worker_cpu", value) for value in (1, True)]
        changes += [("sycl_vision_leaf_sha256", "a" * 64), ("sycl_cache_dir", None), ("sycl_cache_dir", True),
                    ("context", 8192), ("max_output_tokens", 512)]
        for key in ("admission_physical_bytes", "admission_commit_bytes"):
            changes += [(key, value) for value in (None, good[key] - 1, True, float(good[key]), 33 << 30)]
        for key, value in changes:
            self.raw = copy.deepcopy(good)
            # JSON null is distinct from omission for the boolean selector.
            if value is None and key != "sycl_decode_graph":
                del self.raw[key]
            else:
                self.raw[key] = value
            with self.subTest(load=(key, value)), self.assertRaises(ValueError):
                self.config()
        for field in ("model", "runtime_files"):
            self.raw = copy.deepcopy(good)
            target = self.raw[field] if field == "model" else self.raw[field][1]
            target["sha256"] = "a" * 64
            with self.subTest(identity=field), self.assertRaises(ValueError):
                self.config()
        direct = [{key: value} for key, value in changes if key not in ("context", "max_output_tokens")]
        direct += [{"engine": "vulkan"}, {"model": None}, {"model": replace(config.model, sha256="a" * 64)},
                   {"runtime_files": (config.runtime_files[0],)}]
        for changed in direct:
            with self.subTest(construction=changed), self.assertRaises(ValueError):
                replace(config, **changed)
            invalid = SimpleNamespace(**{**vars(config), **changed})
            with self.subTest(backend=changed), self.assertRaises(ValueError):
                LlamaBackend(invalid)
            # Bypass construction deliberately to verify argv independently rechecks the profile.
            invalid_config = object.__new__(type(config))
            for key, value in vars(invalid).items():
                object.__setattr__(invalid_config, key, value)
            with self.subTest(argv=changed), self.assertRaises(ValueError):
                invalid_config.argv(18003, "key", "alias")

    def test_kv_type_is_sycl_only_and_never_coerced(self):
        for value in ("bf16", "q8_0"):
            self.raw["sycl_kv_cache_type"] = value
            with self.assertRaisesRegex(ValueError, "engine-specific"):
                self.config()
        del self.raw["sycl_kv_cache_type"]
        self.raw["parallel_requests"] = 6
        with self.assertRaises(ValueError):
            self.config()
        self.raw["parallel_requests"] = 4
        self.select_sycl()
        for value in (None, True, 8, "Q8_0", "f16", "q4_0", [], {}):
            self.raw["sycl_kv_cache_type"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "sycl_kv_cache_type"):
                self.config()

    def test_configuration_digest_and_opt_in_are_required(self):
        self.config()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_config(self.root / "config.json", "0" * 64)
        self.raw["activate"] = "enabled"
        with self.assertRaisesRegex(ValueError, "Explicit"):
            self.config()

    def test_sycl_preserves_profile_and_limits_except_explicit_device_selection(self):
        from dataclasses import replace
        self.select_sycl()
        for parallel in (1, 2, 4):
            with self.subTest(parallel=parallel):
                self.raw["parallel_requests"] = parallel
                config = self.config()
                self.assertEqual(config.engine, "sycl")
                argv = config.argv(18003, "secret", "private-alias")
                command_line(argv)
                self.assertLessEqual(len(argv), 64)
                self.assertEqual(argv[argv.index("--device") + 1], "SYCL0")
                self.assertEqual(argv[argv.index("--mmproj-device") + 1], "SYCL0")
                split = argv.index("--tensor-split")
                self.assertEqual(argv[split + 1], "1")
                del argv[split:split + 2]
                argv = ["Vulkan0" if value == "SYCL0" else value for value in argv]
                vulkan = replace(config, engine="vulkan")
                self.assertEqual(argv, vulkan.argv(18003, "secret", "private-alias"))
                self.assertEqual(config.job_limits(), vulkan.job_limits())

    def test_sycl_requires_its_authenticated_dll_and_refuses_replacement_or_extra_code(self):
        self.raw["activate"] = SYCL_ACTIVATION
        self.raw["sycl_device_index"] = self.raw.pop("device_index")
        with self.assertRaisesRegex(ValueError, "ggml-sycl.dll"):
            self.config()
        self.select_sycl()
        config = self.config()
        config.authenticate()
        dll = self.runtime / "ggml-sycl.dll"
        original = dll.read_bytes()
        dll.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            config.authenticate()
        dll.write_bytes(original)
        (self.runtime / "extra.dll").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            config.authenticate()

    def test_sycl_environment_is_explicit_and_scrubs_inherited_arithmetic_overrides(self):
        from dataclasses import replace
        from unittest.mock import patch
        self.select_sycl()
        self.raw["sycl_device_index"] = 2
        config = self.config()
        with patch.dict("os.environ", {"SYSTEMROOT": str(self.root), "PATH": "/untrusted",
                                       "ONEAPI_DEVICE_SELECTOR": "opencl:*", "GGML_VK_VISIBLE_DEVICES": "7",
                                       "GGML_SYCL_F16": "0", "GGML_SYCL_ENABLE_FLASH_ATTN": "0",
                                       "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "0",
                                       "ZES_ENABLE_SYSMAN": "1", "UR_L0_ENABLE_SYSMAN_ENV_DEFAULT": "1",
                                       "UR_L0_ENABLE_ZESINIT_DEFAULT": "1",
                                       "MKL_NUM_THREADS": "64", "OMP_NUM_THREADS": "64"}, clear=True):
            env = config.environment()
            vulkan = replace(config, engine="vulkan").environment()
            for key in ("ZES_ENABLE_SYSMAN", "UR_L0_ENABLE_SYSMAN_ENV_DEFAULT", "UR_L0_ENABLE_ZESINIT_DEFAULT"):
                self.assertEqual(env.pop(key), "0")
                self.assertNotIn(key, vulkan)
                self.assertEqual(os.environ[key], "1")
        self.assertEqual(env.pop("ONEAPI_DEVICE_SELECTOR"), "level_zero:2")
        self.assertEqual(env.pop("MKL_NUM_THREADS"), "1")
        self.assertEqual(env.pop("GGML_SYCL_FA_ONEDNN_UNMASKED"), "0")
        self.assertEqual(env.pop("GGML_SYCL_FA_ONEDNN_DG2_D64"), "0")
        self.assertEqual(vulkan.pop("GGML_VK_VISIBLE_DEVICES"), "2")
        self.assertEqual(env, vulkan)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertFalse(any(key.startswith(("GGML_", "SYCL_", "ONEAPI_")) for key in env))
        self.assertNotIn("untrusted", env["PATH"])

    def test_explicit_sycl_profile_binds_leaf_cache_and_four_logical_cpus(self):
        from unittest.mock import patch
        self.select_sycl()
        default = self.config()
        cache = self.root / "private-cache"
        cache.mkdir()
        self.raw.update(sycl_cache_dir=str(cache), sycl_host_cpu_count=4,
                        sycl_vision_leaf_sha256=self.raw["runtime_files"][1]["sha256"])
        selected = self.config()
        selected.authenticate()
        with patch.dict("os.environ", {"SYSTEMROOT": str(self.root), "SYCL_CACHE_DIR": "/foreign",
                                       "SYCL_CACHE_MAX_SIZE": "999", "GGML_SYCL_FA_ONEDNN_UNMASKED": "foreign"}, clear=True):
            env = selected.environment()
            original = default.environment()
        self.assertEqual({key: env[key] for key in ("SYCL_CACHE_PERSISTENT", "SYCL_CACHE_TRACE", "SYCL_CACHE_MAX_SIZE",
                                                   "SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE", "SYCL_CACHE_DIR")},
                         {"SYCL_CACHE_PERSISTENT": "1", "SYCL_CACHE_TRACE": "1", "SYCL_CACHE_MAX_SIZE": "1073741824",
                          "SYCL_CACHE_MIN_DEVICE_IMAGE_SIZE": "0", "SYCL_CACHE_DIR": str(cache)})
        self.assertFalse(any(key.startswith("SYCL_CACHE_") for key in original))
        self.assertEqual((original["GGML_SYCL_FA_ONEDNN_UNMASKED"], original["GGML_SYCL_FA_ONEDNN_DG2_D64"]), ("0", "0"))
        self.assertEqual((env["GGML_SYCL_FA_ONEDNN_UNMASKED"], env["GGML_SYCL_FA_ONEDNN_DG2_D64"]), ("1", "1"))
        self.assertEqual((env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"]), ("1", "1"))
        self.assertEqual(selected.argv(18003, "key", "alias"), default.argv(18003, "key", "alias"))
        self.assertEqual(selected.job_limits().affinity, 15)
        self.assertEqual(default.job_limits().affinity, 1)
        self.assertEqual((selected.job_limits().process_memory, selected.job_limits().job_memory),
                         (default.job_limits().process_memory, default.job_limits().job_memory))

    def test_sycl_vision_refuses_unbound_or_malformed_leaf_identity(self):
        self.select_sycl()
        for digest in (None, True, "0" * 64, "A" * 64, "short"):
            with self.subTest(digest=digest):
                self.raw["sycl_vision_leaf_sha256"] = digest
                with self.assertRaisesRegex(ValueError, "vision leaf"):
                    self.config()

    def test_sycl_cache_is_an_existing_nonlink_directory_rechecked_at_startup(self):
        self.select_sycl()
        cache = self.root / "private-cache"
        cache.mkdir()
        link = self.root / "cache-link"
        link.symlink_to(cache, target_is_directory=True)
        for path in ("relative", str(self.root / "absent"), self.raw["model"]["path"], str(link), None):
            with self.subTest(path=path):
                self.raw["sycl_cache_dir"] = path
                with self.assertRaises((ValueError, OSError)):
                    self.config()
        self.raw["sycl_cache_dir"] = str(cache)
        config = self.config()
        cache.rmdir()
        cache.symlink_to(self.runtime, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            config.authenticate()

    def test_four_cpu_profile_is_explicit_sycl_only_and_retains_cpu_zero(self):
        for key, value in (("sycl_host_cpu_count", 4), ("sycl_cache_dir", str(self.root)),
                           ("sycl_vision_leaf_sha256", "0" * 64)):
            self.raw[key] = value
            with self.assertRaisesRegex(ValueError, "engine-specific"):
                self.config()
            del self.raw[key]
        self.select_sycl()
        for count, cpu in ((True, 0), (1, 0), (2, 0), (8, 0), ("4", 0), (4, 1), (4, False)):
            with self.subTest(count=count, cpu=cpu):
                self.raw.update(sycl_host_cpu_count=count, worker_cpu=cpu)
                with self.assertRaisesRegex(ValueError, "host CPU profile"):
                    self.config()

    def test_sycl_selector_requires_its_own_explicit_namespace(self):
        self.raw["activate"] = SYCL_ACTIVATION
        with self.assertRaisesRegex(ValueError, "engine-specific"):
            self.config()
        self.select_sycl()
        for index in (True, -1, 16, "0", None):
            with self.subTest(index=index):
                self.raw["sycl_device_index"] = index
                with self.assertRaisesRegex(ValueError, "sycl_device_index"):
                    self.config()
        self.raw["sycl_device_index"] = 0
        self.raw["device_index"] = 0
        with self.assertRaisesRegex(ValueError, "engine-specific"):
            self.config()

    def test_environment_drops_inherited_selection_precision_and_path(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"SYSTEMROOT": str(self.root), "PATH": "/untrusted", "GGML_VK_FORCE_F16": "1",
                                       "LLAMA_ARG_MODEL": "other", "VK_DRIVER_FILES": "other", "ONEAPI_DEVICE_SELECTOR": "other"}, clear=True):
            env = self.config().environment()
        self.assertEqual(env["GGML_VK_VISIBLE_DEVICES"], "0")
        self.assertNotIn("GGML_VK_FORCE_F16", env)
        self.assertNotIn("LLAMA_ARG_MODEL", env)
        self.assertNotIn("VK_DRIVER_FILES", env)
        self.assertNotIn("ONEAPI_DEVICE_SELECTOR", env)
        self.assertNotIn("untrusted", env["PATH"])


if __name__ == "__main__":
    unittest.main()
