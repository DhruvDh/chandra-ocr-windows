"""Small configuration/authentication tests; no model, Windows API or network is used."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from runtime.llama.config import ACTIVATION, SYCL_ACTIVATION, load_config
from runtime.native.winjob import command_line


class ConfigTests(unittest.TestCase):
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
        self.assertEqual(vulkan.pop("GGML_VK_VISIBLE_DEVICES"), "2")
        self.assertEqual(env, vulkan)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertFalse(any(key.startswith(("GGML_", "SYCL_", "ONEAPI_")) for key in env))
        self.assertNotIn("untrusted", env["PATH"])

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
