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


if __name__ == '__main__': unittest.main()
