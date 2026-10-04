"""Exercise meaningful failure cases without inference or endpoint traffic."""
import html
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('qualification_validate', ROOT / 'validate.py')
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


def ideal(expected):
    blocks = []
    table_text = ' '.join(' '.join(row) for row in expected['table'])
    for text in expected['ordered_blocks']:
        if expected['table'] and text == table_text:
            body = '<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(cell) + '</td>' for cell in row) + '</tr>' for row in expected['table']) + '</table>'
        else:
            body = html.escape(text)
        blocks.append('<div data-bbox="10 20 900 40" data-label="Text">' + body + '</div>')
    return {'content': ''.join(blocks), 'finish_reason': 'stop', 'usage': {'completion_tokens': 500}}, blocks


class QualificationChecks(unittest.TestCase):
    def test_complete_output_and_failures(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for fixture in manifest['fixtures']:
            with self.subTest(fixture=fixture['id']):
                expected = fixture['expected']; actual, blocks = ideal(expected)
                self.assertTrue(VALIDATE.check(expected, actual)['passed'])
                mutations = [
                    {**actual, 'content': actual['content'].replace(expected['end_marker'], '')},
                    {**actual, 'finish_reason': 'length'},
                    {**actual, 'usage': None},
                    {**actual, 'usage': {'completion_tokens': 12384}},
                    {**actual, 'stream_errors': [{'type': 'backend_failure'}]},
                    {**actual, 'content': ''.join(reversed(blocks))},
                    {**actual, 'content': actual['content'].replace(expected['ordered_blocks'][0], 'UNEXPECTED HEADING')},
                    {**actual, 'content': actual['content'] + blocks[0]},
                ]
                if expected['table']:
                    harmless = {**actual, 'content': actual['content'].replace('-2.75', '\u2212 2.75').replace('Rate (g/h)', 'Rate\n (g/h)').replace('+18.50', '+ 18.50')}
                    self.assertTrue(VALIDATE.check(expected, harmless)['passed'])
                    mutations += [
                        {**actual, 'content': actual['content'].replace('-2.75', '+2.75')},
                        {**actual, 'content': actual['content'].replace('0.125', '0.375')},
                        {**actual, 'content': actual['content'].replace('+18.50', '-6.25', 1)},
                    ]
                for mutation in mutations:
                    self.assertFalse(VALIDATE.check(expected, mutation)['passed'])
                comparison = VALIDATE.compare(expected, actual, actual, geometry_space='normalized_1000')
                self.assertTrue(comparison['passed'])
                self.assertIsNone(comparison['geometry_comparison']['acceptance_tolerance'])
                # A candidate agreeing with an erroneous baseline still fails.
                self.assertFalse(VALIDATE.compare(expected, mutations[0], mutations[0])['passed'])

    def test_frozen_manifest(self):
        self.assertTrue(VALIDATE.verify(ROOT / 'manifest.json')['passed'])


if __name__ == '__main__':
    unittest.main()
