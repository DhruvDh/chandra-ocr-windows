import argparse
import base64
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('host_manage', Path(__file__).resolve().parents[2] / 'scripts/host/manage.py')
manage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage)


def arguments():
    return argparse.Namespace(ssh_host='waystone', remote_root=r'C:\owned checkout', remote_python=r'C:\owned\python.exe',
                              model_path=r'C:\owned\model', worker_port=18001, lease_port=18002, router_port=8002, lan_address=None,
                              northstone_url='http://127.0.0.1:8000', northstone_page_seconds=1, waystone_page_seconds=2, normalization='original')


class ManagerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('systemd-analyze'), 'systemd-analyze is required for the real unit parser')
    def test_real_systemd_parser_accepts_working_directory_with_spaces_and_percent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'working space 100%'
            directory.mkdir()
            unit = Path(temporary) / 'chandra-parser-check.service'
            unit.write_text('[Service]\nType=oneshot\nWorkingDirectory=' + manage.directory_value(directory) + '\nExecStart=/usr/bin/true\n')
            result = subprocess.run(['systemd-analyze', '--user', 'verify', str(unit)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_remote_launch_assigns_suspended_process_to_job_before_resume(self):
        source = manage.REMOTE_SUPERVISOR
        self.assertLess(source.index('K.CreateProcessW(None, command'), source.index('K.AssignProcessToJobObject(job, process.hProcess)'))
        self.assertLess(source.index('K.AssignProcessToJobObject(job, process.hProcess)'), source.index('K.ResumeThread(process.hThread)'))
        self.assertIn('0x2000 # KILL_ON_JOB_CLOSE', source)
        self.assertNotIn('taskkill', source)
        launch = manage.remote_command(arguments(), 'test-token')
        encoded_settings = launch[-1].rsplit(' ', 1)[1]
        self.assertEqual(json.loads(base64.b64decode(encoded_settings))['cwd'], r'C:\owned checkout')
        self.assertIn('127.0.0.1:18001:127.0.0.1:18001', launch)

    def test_normalization_selection_reaches_owned_worker(self):
        for mode in ('original', 'gain-only'):
            settings = arguments()
            settings.normalization = mode
            launch = manage.remote_command(settings, 'test-token')
            argv = json.loads(base64.b64decode(launch[-1].rsplit(' ', 1)[1]))['argv']
            self.assertEqual(argv[argv.index('--normalization') + 1], mode)

    def test_channel_closes_lease_and_waits_before_ssh_termination(self):
        order = []
        stop = threading.Event()
        class Connection:
            def sendall(self, text):
                order.append(('send', text))
                if text == b'alive\n': stop.set()
            def recv(self, count): return b'owned\n'
            def close(self): order.append(('close',))
        class Process:
            returncode = 0
            def poll(self): return None
            def wait(self, timeout): order.append(('wait', timeout)); return 0
        with patch.object(manage.subprocess, 'Popen', return_value=Process()), patch.object(manage.socket, 'create_connection', return_value=Connection()):
            self.assertEqual(manage.run_channel(['owned-ssh'], 18002, 'test-token', stop), 0)
        self.assertEqual(order, [('send', b'test-token\n'), ('send', b'alive\n'), ('close',), ('wait', 10)])

    def test_fragmented_acknowledgement_is_read_completely(self):
        chunks = [b'o', b'wn', b'ed\n']
        class Connection:
            def recv(self, count): return chunks.pop(0)
        self.assertEqual(manage.read_exact(Connection(), 6), b'owned\n')

    def test_effective_foreign_unit_and_dropins_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            unit_dir = Path(directory)
            for output in ['LoadState=loaded\nFragmentPath=/foreign/worker.service\nDropInPaths=\n',
                           'LoadState=loaded\nFragmentPath=\nDropInPaths=\n',
                           'LoadState=loaded\nFragmentPath=\nDropInPaths=/foreign/override.conf\n']:
                result = subprocess.CompletedProcess([], 0, stdout=output)
                with patch.object(manage.subprocess, 'run', return_value=result), self.assertRaises(ValueError):
                    manage.check_unit_ownership(unit_dir, manage.UNITS[0])

    def test_unit_symlink_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            foreign = directory / 'foreign.service'
            foreign.write_text(manage.MARKER + '\nforeign config')
            (directory / manage.UNITS[0]).symlink_to(foreign)
            with self.assertRaises(ValueError):
                manage.check_unit_ownership(directory, manage.UNITS[0])
            self.assertEqual(foreign.read_text(), manage.MARKER + '\nforeign config')

    def test_marker_alone_cannot_authorize_edited_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); local = root / '.local'; local.mkdir()
            directory = root / 'units'; directory.mkdir()
            path = directory / manage.UNITS[0]
            original = manage.MARKER + '\noriginal'
            path.write_text(original + '\nforeign edit')
            (local / 'installation.json').write_text(json.dumps({'source_root': str(root), 'units': list(manage.UNITS),
                'unit_sha256': {manage.UNITS[0]: hashlib.sha256(original.encode()).hexdigest()}}))
            result = subprocess.CompletedProcess([], 0, stdout='LoadState=loaded\nFragmentPath=' + str(path) + '\nDropInPaths=\n')
            with patch.object(manage, 'ROOT', root), patch.object(manage, 'LOCAL', local), patch.object(manage.subprocess, 'run', return_value=result):
                with self.assertRaises(ValueError): manage.check_unit_ownership(directory, manage.UNITS[0])
            self.assertTrue(path.read_text().endswith('foreign edit'))

    def test_install_failure_restores_files_and_prior_service_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo'; local = root / '.local'; home = Path(temporary) / 'home'
            directory = home / '.config/systemd/user'
            (root / '.venv/bin').mkdir(parents=True); (root / '.venv/bin/python').touch()
            local.mkdir(); directory.mkdir(parents=True)
            old_files = {directory / name: manage.MARKER + '\nold unit\n' for name in manage.UNITS}
            old_files[local / 'router.json'] = '{"old": true}\n'
            old_files[local / 'installation.json'] = json.dumps({'source_root': str(root), 'units': list(manage.UNITS), 'unit_sha256': {name: hashlib.sha256(old_files[directory / name].encode()).hexdigest() for name in manage.UNITS}})
            for path, text in old_files.items(): path.write_text(text)
            commands = []; failed = False
            def run(argv, **kwargs):
                nonlocal failed
                commands.append(argv)
                if 'verify' in argv:
                    self.assertTrue(all(path.read_text() == text for path, text in old_files.items()))
                if 'start' in argv and len(argv) > 4 and not failed:
                    failed = True
                    raise subprocess.CalledProcessError(1, argv)
                return subprocess.CompletedProcess(argv, 0)
            with patch.object(manage, 'ROOT', root), patch.object(manage, 'LOCAL', local), patch.object(manage.Path, 'home', return_value=home), patch.object(manage.subprocess, 'run', side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError): manage.install(arguments())
            for path, text in old_files.items(): self.assertEqual(path.read_text(), text)
            self.assertFalse((local / 'channel.json').exists())
            self.assertTrue(any(command[-2:] == ['start', manage.UNITS[0]] for command in commands))
            self.assertTrue(any(command[-2:] == ['start', manage.UNITS[1]] for command in commands))

    def test_unmanaged_unit_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo'; home = Path(temporary) / 'home'; local = root / '.local'
            (root / '.venv/bin').mkdir(parents=True); (root / '.venv/bin/python').touch()
            directory = home / '.config/systemd/user'; directory.mkdir(parents=True)
            owned = directory / manage.UNITS[0]; owned.write_text('unrelated owner')
            with patch.object(manage, 'ROOT', root), patch.object(manage, 'LOCAL', local), patch.object(manage.Path, 'home', return_value=home):
                with self.assertRaises(ValueError): manage.install(arguments())
            self.assertEqual(owned.read_text(), 'unrelated owner')
            self.assertFalse(local.exists())


class NativeManagerTests(unittest.TestCase):
    """Pure native route/transaction controls; every service and SSH action is a fake."""

    def settings(self, **changes):
        values = arguments()
        values.worker_backend = 'native'
        values.model_path = None
        values.native_config = r'C:\owned endpoint\native.json'
        values.native_config_sha256 = 'a' * 64
        for name, value in changes.items():
            setattr(values, name, value)
        return values

    def decoded(self, values):
        launch = manage.remote_command(values, 'synthetic-lease-token')
        return launch, json.loads(base64.b64decode(launch[-1].rsplit(' ', 1)[1]))

    def test_explicit_native_route_authentication_and_exact_context_arguments(self):
        values = self.settings()
        launch, config = self.decoded(values)
        self.assertEqual(config['argv'], [values.remote_python, '-m', 'chandra_service', 'native',
            '--native-config', values.native_config, '--native-config-sha256', values.native_config_sha256,
            '--host', '127.0.0.1', '--port', '18001', '--max-input-tokens', '16384', '--max-output-tokens', '12384'])
        self.assertEqual(config['cwd'], values.remote_root)
        self.assertEqual((config['lease_port'], config['lease_token']), (18002, 'synthetic-lease-token'))
        self.assertIn('127.0.0.1:18001:127.0.0.1:18001', launch)
        self.assertIn('127.0.0.1:18002:127.0.0.1:18002', launch)
        self.assertNotIn('--model-path', config['argv'])
        self.assertNotIn('--attention', config['argv'])

    def test_native_missing_mixed_or_unauthenticated_route_is_refused(self):
        cases = [
            {'native_config': None}, {'native_config_sha256': None},
            {'native_config': 'native.json'}, {'native_config': r'C:native.json'},
            {'native_config': 'C:/owned/native.json'}, {'native_config': r'C:\owned\..\native.json'},
            {'native_config': 'C:\\owned\\native.json\\'}, {'native_config': 'C:\\owned\\bad\nname.json'},
            {'native_config_sha256': 'A' * 64}, {'native_config_sha256': 'a' * 63},
            {'model_path': r'C:\model'}, {'normalization': 'gain-only'},
            {'worker_backend': 'unselected'}, {'worker_backend': 'waystone'},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                manage.remote_command(self.settings(**changes), 'synthetic-lease-token')

    def test_host_cli_requires_explicit_native_inputs_before_installation(self):
        import sys
        base = ['manage.py', 'install', '--worker-backend', 'native', '--remote-root', r'C:\owned',
                '--remote-python', r'C:\python.exe']
        for extra in ([], ['--native-config', r'C:\native.json'],
                      ['--native-config', r'C:\native.json', '--native-config-sha256', 'A' * 64]):
            with self.subTest(extra=extra), patch.object(sys, 'argv', base + extra), \
                    patch.object(manage, 'install') as install, patch.object(manage.subprocess, 'run') as run:
                with self.assertRaises(SystemExit) as refusal:
                    manage.main()
                self.assertEqual(refusal.exception.code, 2)
                install.assert_not_called()
                run.assert_not_called()
        with patch.object(sys, 'argv', base + ['--native-config', r'C:\native.json', '--native-config-sha256', 'b' * 64]), \
                patch.object(manage, 'install') as install:
            manage.main()
            values = install.call_args.args[0]
        self.assertEqual(values.worker_backend, 'native')
        self.assertIsNone(values.model_path)
        self.assertEqual(manage.worker_argv(values)[3], 'native')

    def test_generated_arguments_reach_unchanged_native_config_reader_before_listening(self):
        import sys
        from chandra_service.__main__ import main
        argv = manage.worker_argv(self.settings())
        with patch.object(sys, 'argv', ['chandra_service', *argv[3:]]), \
                patch('runtime.native.load_config', side_effect=ValueError('synthetic authentication refusal')) as load, \
                patch('runtime.native.winjob.kernel32') as kernel, patch('chandra_service.__main__.socket.socket') as listen:
            with self.assertRaises(SystemExit) as refusal:
                main()
            self.assertEqual(refusal.exception.code, 2)
            load.assert_called_once()
            self.assertEqual(str(load.call_args.args[0]), self.settings().native_config)
            self.assertEqual(load.call_args.args[1], 'a' * 64)
            kernel.assert_not_called()
            listen.assert_not_called()

    def fixture(self, temporary, existing=True):
        root = Path(temporary) / 'repo'
        home = Path(temporary) / 'home'
        local = root / '.local'
        directory = home / '.config/systemd/user'
        (root / '.venv/bin').mkdir(parents=True)
        (root / '.venv/bin/python').touch()
        local.mkdir()
        directory.mkdir(parents=True)
        config = {'backends': {
            'northstone': {'url': 'http://127.0.0.1:8000', 'capacity': 2, 'queue_limit': 1,
                          'auto_queue_limit': 1, 'expected_page_seconds': 11.25},
            'waystone': {'url': 'http://127.0.0.1:18001', 'capacity': 1, 'queue_limit': 2, 'expected_page_seconds': 85.5}},
            'cooldown_seconds': 19, 'inference_seconds': 1700}
        old = {}
        if existing:
            old = {directory / name: manage.MARKER + '\nold unit\n' for name in manage.UNITS}
            old[local / 'router.json'] = json.dumps(config, separators=(',', ':')) + '\n'
            old[local / 'channel.json'] = '{"retained_private_lease":true}\n'
            old[local / 'installation.json'] = json.dumps({'source_root': str(root), 'units': list(manage.UNITS),
                'unit_sha256': {name: hashlib.sha256(old[directory / name].encode()).hexdigest() for name in manage.UNITS}})
            for path, text in old.items():
                path.write_text(text)
                path.chmod(0o600 if path.parent == local else 0o644)
        return root, home, local, directory, config, old

    def install_fake(self, fixture, values, fail_start=False):
        root, home, local, directory, config, old = fixture
        commands = []
        failed = False
        def run(argv, **kwargs):
            nonlocal failed
            commands.append(argv)
            if 'show' in argv:
                path = directory / argv[-1]
                text = 'LoadState=loaded\nFragmentPath=' + str(path) + '\nDropInPaths=\n' if path.exists() else 'LoadState=not-found\nFragmentPath=\nDropInPaths=\n'
                return subprocess.CompletedProcess(argv, 0, stdout=text)
            if 'verify' in argv:
                self.assertTrue(all(path.read_bytes() == text.encode() for path, text in old.items()))
            if 'is-active' in argv:
                return subprocess.CompletedProcess(argv, 0 if argv[-1] == manage.UNITS[0] else 1)
            if fail_start and 'start' in argv and len(argv) > 4 and not failed:
                failed = True
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(manage, 'ROOT', root), patch.object(manage, 'LOCAL', local), \
                patch.object(manage.Path, 'home', return_value=home), patch.object(manage.subprocess, 'run', side_effect=run):
            if fail_start:
                with self.assertRaises(subprocess.CalledProcessError):
                    manage.install(values)
            else:
                manage.install(values)
        return commands

    def test_native_reinstall_preserves_owned_router_bytes_and_private_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = self.fixture(temporary)
            self.install_fake(f, self.settings())
            root, home, local, directory, config, old = f
            self.assertEqual((local / 'router.json').read_text(), old[local / 'router.json'])
            receipt = json.loads((local / 'installation.json').read_text())
            self.assertEqual(receipt['worker_backend'], 'native')
            self.assertEqual(receipt['native_config_sha256'], 'a' * 64)
            self.assertEqual(receipt['native_config'], self.settings().native_config)
            settings = json.loads((local / 'channel.json').read_text())
            remote = json.loads(base64.b64decode(settings['ssh_argv'][-1].rsplit(' ', 1)[1]))
            self.assertEqual(remote['argv'][3], 'native')
            self.assertEqual(remote['argv'][remote['argv'].index('--native-config-sha256') + 1], 'a' * 64)
            self.assertEqual((local / 'channel.json').stat().st_mode & 0o777, 0o600)
            for name in manage.UNITS:
                self.assertEqual(receipt['unit_sha256'][name], hashlib.sha256((directory / name).read_bytes()).hexdigest())

    def test_explicit_port_change_updates_only_windows_url_and_fresh_defaults_are_conservative(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = self.fixture(temporary)
            self.install_fake(f, self.settings(worker_port=18003))
            expected = f[4]
            expected['backends']['waystone']['url'] = 'http://127.0.0.1:18003'
            self.assertEqual(json.loads((f[2] / 'router.json').read_text()), expected)
            self.assertEqual(expected['backends']['waystone']['capacity'], 1)
        with tempfile.TemporaryDirectory() as temporary:
            f = self.fixture(temporary, existing=False)
            self.install_fake(f, self.settings())
            config = json.loads((f[2] / 'router.json').read_text())
            self.assertEqual(config['backends']['waystone']['capacity'], 1)
            self.assertEqual(config['backends']['northstone']['capacity'], 2)
            self.assertEqual(config['backends']['northstone']['queue_limit'], 0)

    def test_native_start_failure_restores_every_private_file_mode_and_prior_active_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            f = self.fixture(temporary)
            old_modes = {p: p.stat().st_mode & 0o777 for p in f[5]}
            commands = self.install_fake(f, self.settings(), fail_start=True)
            for path, text in f[5].items():
                self.assertEqual(path.read_text(), text)
                self.assertEqual(path.stat().st_mode & 0o777, old_modes[path])
            self.assertIn(['systemctl', '--user', 'start', manage.UNITS[0]], commands)
            self.assertNotIn(['systemctl', '--user', 'start', manage.UNITS[1]], commands)
            self.assertFalse(any(p.name.startswith('chandra-unit-stage-') for p in f[2].iterdir()))

    def test_native_foreign_or_edited_unit_is_preserved_before_mutation(self):
        for foreign in (True, False):
            with self.subTest(foreign=foreign), tempfile.TemporaryDirectory() as temporary:
                f = self.fixture(temporary)
                path = f[3] / manage.UNITS[0]
                path.write_text('foreign owner' if foreign else manage.MARKER + '\nedited owned unit')
                before = {p: p.read_bytes() for p in f[5]}
                with patch.object(manage, 'ROOT', f[0]), patch.object(manage, 'LOCAL', f[2]), \
                        patch.object(manage.Path, 'home', return_value=f[1]), \
                        patch.object(manage.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='LoadState=loaded\nFragmentPath=' + str(path) + '\nDropInPaths=\n')):
                    with self.assertRaises(ValueError):
                        manage.install(self.settings())
                self.assertEqual({p: p.read_bytes() for p in f[5]}, before)


    def test_native_same_url_preserves_lf_and_crlf_router_bytes_and_mode(self):
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)), tempfile.TemporaryDirectory() as temporary:
                f = self.fixture(temporary)
                path = f[2] / 'router.json'
                original = (json.dumps(f[4], indent=2) + '\n').replace('\n', newline).encode()
                f[5][path] = original.decode()
                path.write_bytes(original)
                before_mode = path.stat().st_mode & 0o777
                self.install_fake(f, self.settings())
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(path.stat().st_mode & 0o777, before_mode)
                self.assertEqual(before_mode, 0o600)

    def test_native_changed_url_preserves_policy_and_uses_canonical_json_for_both_newlines(self):
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)), tempfile.TemporaryDirectory() as temporary:
                f = self.fixture(temporary)
                path = f[2] / 'router.json'
                original = (json.dumps(f[4], indent=2) + '\n').replace('\n', newline).encode()
                f[5][path] = original.decode()
                path.write_bytes(original)
                self.install_fake(f, self.settings(worker_port=18003))
                expected = json.loads(original)
                expected['backends']['waystone']['url'] = 'http://127.0.0.1:18003'
                actual = path.read_bytes()
                self.assertEqual(actual, (json.dumps(expected, indent=2) + '\n').encode())
                self.assertEqual(json.loads(actual), expected)
                self.assertEqual(json.loads(actual)['backends']['northstone'], f[4]['backends']['northstone'])
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_native_failed_reinstall_restores_lf_and_crlf_bytes_and_original_modes(self):
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)), tempfile.TemporaryDirectory() as temporary:
                f = self.fixture(temporary)
                path = f[2] / 'router.json'
                original = (json.dumps(f[4], indent=2) + '\n').replace('\n', newline).encode()
                f[5][path] = original.decode()
                path.write_bytes(original)
                path.chmod(0o640)
                before = {p: (p.read_bytes(), p.stat().st_mode & 0o777) for p in f[5]}
                commands = self.install_fake(f, self.settings(), fail_start=True)
                self.assertEqual({p: (p.read_bytes(), p.stat().st_mode & 0o777) for p in f[5]}, before)
                self.assertIn(['systemctl', '--user', 'start', manage.UNITS[0]], commands)
                self.assertNotIn(['systemctl', '--user', 'start', manage.UNITS[1]], commands)
                self.assertFalse(any(p.name.startswith('chandra-unit-stage-') for p in f[2].iterdir()))



if __name__ == '__main__': unittest.main()
