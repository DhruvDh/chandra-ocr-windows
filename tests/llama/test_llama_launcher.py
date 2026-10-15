"""CPU-only positional llama launcher checks; every listener and backend is mocked."""

from contextlib import ExitStack, redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from chandra_service.__main__ import main
from runtime.native.winjob import ContainmentError


class LlamaLauncherTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.order, self.listeners = [], []
        self.configuration, self.backend, self.app = object(), object(), object()
        self.sha = "a" * 64
        self.arguments = ["llama", "--llama-config", "llama.json", "--llama-config-sha256", self.sha]

        def configured(path, digest):
            self.order.append("configuration")
            return self.configuration

        def capable():
            self.order.append("containment")
            return object()

        def backend(config):
            self.order.append("backend")
            return self.backend

        def application(instance, **settings):
            self.order.append("application")
            return self.app

        def listener(*args):
            self.order.append("listener")
            result = Mock()
            self.listeners.append(result)
            return result

        async def serve(**kwargs):
            pass

        def consume(coroutine):
            coroutine.close()  # The launcher hands off a coroutine; no event loop or listener runs.

        def mock(name, **options):
            return self.stack.enter_context(patch(name, **options))

        self.load = mock("runtime.llama.load_config", side_effect=configured)
        self.capability = mock("runtime.native.winjob.kernel32", side_effect=capable)
        self.make_backend = mock("runtime.llama.LlamaBackend", side_effect=backend)
        self.make_app = mock("runtime.llama.app.create_llama_app", side_effect=application)
        self.xpu = mock("runtime.waystone.XPUBackend", return_value=object())
        self.generic_app = mock("chandra_service.service.create_app", return_value=object())
        self.native_config = mock("runtime.native.load_config", return_value=object())
        self.native_backend = mock("runtime.native.NativeBackend", return_value=object())
        self.native_app = mock("runtime.native.app.create_native_app", return_value=object())
        self.router = mock("chandra_service.router.create_router", return_value=object())
        self.server = Mock()
        self.server.serve = Mock(side_effect=serve)
        self.server_factory = mock("uvicorn.Server", return_value=self.server)
        self.server_config = mock("uvicorn.Config", side_effect=lambda app, **options: SimpleNamespace(app=app))
        self.sockets = mock("chandra_service.__main__.socket.socket", side_effect=listener)
        mock("chandra_service.__main__.asyncio.run", side_effect=consume)

    def invoke(self, arguments, refused=False):
        with patch("sys.argv", ["chandra_service", *arguments]), redirect_stderr(io.StringIO()):
            if refused:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
            else:
                main()

    def assert_not_listening(self):
        self.sockets.assert_not_called()
        self.server_factory.assert_not_called()

    def test_llama_requires_both_configuration_and_digest(self):
        cases = [["llama"], ["llama", "--llama-config", "llama.json"],
                 ["llama", "--llama-config-sha256", self.sha]]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.invoke(arguments, refused=True)
        self.load.assert_not_called()
        self.capability.assert_not_called()
        self.assert_not_listening()

    def test_configuration_digest_refusal_precedes_containment_and_listeners(self):
        self.load.side_effect = ValueError("configuration SHA-256 differs")
        self.invoke(self.arguments, refused=True)
        self.load.assert_called_once_with(Path("llama.json"), self.sha)
        self.capability.assert_not_called()
        self.make_backend.assert_not_called()
        self.make_app.assert_not_called()
        self.assert_not_listening()

    def test_malformed_digest_is_refused_before_listening(self):
        self.load.side_effect = ValueError("configuration SHA-256 is required")
        self.invoke(["llama", "--llama-config", "llama.json", "--llama-config-sha256", "bad-digest"], refused=True)
        self.make_backend.assert_not_called()
        self.assert_not_listening()

    def test_windows_containment_refusal_precedes_backend_and_listeners(self):
        self.capability.side_effect = ContainmentError("Windows Job containment is unavailable")
        self.invoke(self.arguments, refused=True)
        self.load.assert_called_once()
        self.capability.assert_called_once_with()
        self.make_backend.assert_not_called()
        self.make_app.assert_not_called()
        self.assert_not_listening()

    def test_only_owned_llama_app_is_created_after_authenticated_capability(self):
        self.invoke(self.arguments)
        self.assertEqual(self.order, ["configuration", "containment", "backend", "application", "listener"])
        self.load.assert_called_once_with(Path("llama.json"), self.sha)
        self.make_backend.assert_called_once_with(self.configuration)
        self.make_app.assert_called_once_with(self.backend, queue_limit=2, inference_seconds=1800, idle_seconds=300)
        for factory in (self.xpu, self.generic_app, self.native_config, self.native_backend, self.native_app, self.router):
            factory.assert_not_called()
        self.server_config.assert_called_once_with(self.app, access_log=False)
        self.server.serve.assert_called_once_with(sockets=self.listeners)
        self.listeners[0].bind.assert_called_once_with(("127.0.0.1", 8081))
        self.listeners[0].close.assert_called_once_with()

    def test_ordinary_service_settings_are_forwarded_unchanged(self):
        self.invoke([*self.arguments, "--host", "127.0.0.2", "--port", "18002", "--queue-limit", "5",
                     "--inference-seconds", "123.5", "--idle-seconds", "12.5"])
        self.make_app.assert_called_once_with(self.backend, queue_limit=5, inference_seconds=123.5, idle_seconds=12.5)
        self.listeners[0].bind.assert_called_once_with(("127.0.0.2", 18002))

    def test_llama_refuses_unrelated_paths_and_conditioning_overrides(self):
        overrides = [["--model-path", "other-model"], ["--config", "router.json"],
                     ["--native-config", "native.json"], ["--native-config-sha256", "b" * 64],
                     ["--max-pixels", "1"], ["--max-input-tokens", "8192"], ["--max-output-tokens", "100"],
                     ["--attention", "eager"], ["--normalization", "gain-only"],
                     ["--precision", "bf16"], ["--dtype", "float16"], ["--ctx-size", "32768"]]
        for override in overrides:
            with self.subTest(override=override):
                self.invoke([*self.arguments, *override], refused=True)
        self.load.assert_not_called()
        self.make_backend.assert_not_called()
        self.assert_not_listening()

    def test_llama_flags_are_refused_in_every_other_mode(self):
        modes = [["waystone", "--model-path", "model"], ["router", "--config", "router.json"],
                 ["native", "--native-config", "native.json", "--native-config-sha256", "b" * 64]]
        flags = [["--llama-config", "llama.json"], ["--llama-config-sha256", self.sha]]
        with patch("chandra_service.__main__.Path.read_text", return_value="{}"):
            for mode in modes:
                for flag in flags:
                    with self.subTest(mode=mode[0], flag=flag):
                        self.invoke([*mode, *flag], refused=True)
        for factory in (self.load, self.make_backend, self.xpu, self.router, self.native_backend):
            factory.assert_not_called()
        self.assert_not_listening()

    def test_waystone_default_limits_and_backend_remain_unchanged(self):
        self.invoke(["waystone", "--model-path", "local-model"])
        self.xpu.assert_called_once_with("local-model", max_pixels=4_000_000, max_input_tokens=16384,
                                         max_output_tokens=12384, attention_backend="hybrid", normalization="original")
        self.generic_app.assert_called_once_with(self.xpu.return_value, queue_limit=2,
                                                 inference_seconds=1800, idle_seconds=300)
        self.load.assert_not_called()
        self.make_backend.assert_not_called()
        self.make_app.assert_not_called()
