"""Only extracted pure helpers and fake Win32 calls; never import/run the remote supervisor."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / 'scripts/host/windows_lease.py').read_text()
HELPERS = ast.Module(body=[node for node in ast.parse(SOURCE).body if isinstance(node, ast.FunctionDef)
                          and node.name in ('configured_affinity', 'set_job_limits')], type_ignores=[])


class LeaseAffinityTests(unittest.TestCase):
    def helpers(self, *, sticky=True, query=True):
        calls = []
        saved = {}
        def limit():
            return SimpleNamespace(BasicLimitInformation=SimpleNamespace(LimitFlags=0, Affinity=0))
        def set_limits(job, kind, value, size):
            calls.append('set')
            saved.update(flags=value.BasicLimitInformation.LimitFlags, affinity=value.BasicLimitInformation.Affinity)
            return True
        def query_limits(job, kind, value, size, returned):
            calls.append('query')
            value.BasicLimitInformation.LimitFlags = saved['flags']
            value.BasicLimitInformation.Affinity = saved['affinity'] if sticky else 1
            return query
        namespace = {'LIMIT': limit, 'K': SimpleNamespace(SetInformationJobObject=set_limits, QueryInformationJobObject=query_limits),
                     'ctypes': SimpleNamespace(byref=lambda value: value, sizeof=lambda value: 1,
                                               WinError=lambda error: OSError(error), get_last_error=lambda: 5)}
        exec(compile(HELPERS, 'windows_lease_helpers.py', 'exec'), namespace)
        return namespace, calls, saved

    def test_opt_in_affinity_is_exactly_verified_before_process_creation(self):
        helpers, calls, saved = self.helpers()
        helpers['set_job_limits'](1, helpers['configured_affinity']({'job_affinity': 15}))
        self.assertEqual(calls, ['set', 'query'])
        self.assertEqual(saved, {'flags': 0x2010, 'affinity': 15})
        self.assertLess(SOURCE.index('affinity = configured_affinity(config)'), SOURCE.index('listener = socket.socket()'))
        self.assertLess(SOURCE.index('    set_job_limits(job, affinity)'), SOURCE.index('    if not K.CreateProcessW'))
        for sticky, query, error in ((False, True, RuntimeError), (True, False, OSError)):
            with self.subTest(sticky=sticky, query=query):
                helpers, calls, _ = self.helpers(sticky=sticky, query=query)
                with self.assertRaises(error):
                    helpers['set_job_limits'](1, 15)
                self.assertEqual(calls, ['set', 'query'])

    def test_default_lease_and_invalid_settings(self):
        helpers, calls, saved = self.helpers()
        helpers['set_job_limits'](1, helpers['configured_affinity']({}))
        self.assertEqual(calls, ['set'])
        self.assertEqual(saved, {'flags': 0x2000, 'affinity': 0})
        for value in (True, 1, 3, 7, 31, '15', 15.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                helpers['configured_affinity']({'job_affinity': value})


if __name__ == '__main__':
    unittest.main()
