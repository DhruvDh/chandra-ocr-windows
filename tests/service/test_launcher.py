import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from chandra_service.__main__ import main, validate_host

class LauncherTests(unittest.TestCase):
    def test_concrete_addresses(self):
        for host in ('127.0.0.1','192.168.8.109','::1'): validate_host(host)
        for host in ('0.0.0.0','::','8.8.8.8','224.0.0.1','example.com'):
            with self.assertRaises(ValueError): validate_host(host)

    def test_repeated_hosts_use_one_server_and_shared_app(self):
        listeners=[Mock(),Mock()]
        app=object(); server=Mock()
        async def serve(**kwargs): pass
        server.serve=Mock(side_effect=serve)
        def consume(coro): coro.close()
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'router.json';config.write_text(json.dumps({'backends':{'northstone':{'url':'http://127.0.0.1:8000'}}}))
            argv=['chandra_service','router','--config',str(config),'--host','127.0.0.1','--host','192.168.8.109']
            with patch('sys.argv',argv),patch('chandra_service.router.create_router',return_value=app) as create,patch('chandra_service.__main__.socket.socket',side_effect=listeners),patch('uvicorn.Server',return_value=server) as factory,patch('chandra_service.__main__.asyncio.run',side_effect=consume):
                main()
            create.assert_called_once();factory.assert_called_once();self.assertIs(factory.call_args.args[0].app,app)
            server.serve.assert_called_once_with(sockets=listeners)
            listeners[0].bind.assert_called_once_with(('127.0.0.1',8081));listeners[1].bind.assert_called_once_with(('192.168.8.109',8081))
            for listener in listeners: listener.close.assert_called_once()

    def test_backend_import_is_cpu_only_and_default_context(self):
        from runtime.waystone import XPUBackend
        backend=XPUBackend('unprovisioned')
        self.assertIsNone(backend.model)
        self.assertEqual(backend.max_input_tokens,16384)
        self.assertEqual(backend.health()['state'],'cold')
