import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import cli_acceptance
from cli_acceptance import validate_cli

class CLIGates(unittest.TestCase):
    def fixture(self):
        e=dict(file_name='native.pdf',pages=2,table=[['12.50'],['-2.50']],equations=['x = 5'],end_markers=['END1','END2'],ordered_text=['END1','END2'])
        text='<table><tr><td>12.50</td></tr><tr><td>-2.50</td></tr></table>\nEquation: x = 5\nEND1\nEND2'
        m=dict(file_name='native.pdf',num_pages=2,total_token_count=9,total_chunks=2,pages=[dict(page_num=0,token_count=4,num_chunks=1,page_box=[0,0,10,10]),dict(page_num=1,token_count=5,num_chunks=1,page_box=[0,0,10,10])])
        return e,text,m
    def test_valid_metadata_does_not_invent_stop(self):
        e,t,m=self.fixture();r=validate_cli(e,t,m,[])
        self.assertTrue(r['passed']);self.assertIsNone(r['finish_reasons']);self.assertFalse(r['full_ocr_acceptance']);self.assertFalse(r['truncation_verified'])
    def test_missing_page_table_and_html_fail(self):
        e,t,m=self.fixture();m['pages'].pop();self.assertFalse(validate_cli(e,t,m,[])['passed'])
        e,t,m=self.fixture();self.assertFalse(validate_cli(e,t.replace('-2.50','2.50'),m,[])['passed'])
        self.assertFalse(validate_cli(e,t,m,['unwanted.html'])['passed'])
    def test_cli_errors_cannot_hide_in_metadata(self):
        e,t,m=self.fixture();m['total_chunks']=999;self.assertIn('chunk_sum',validate_cli(e,t,m,[])['failures'])
        self.assertFalse(validate_cli(e,'',None,[])['passed'])
    def test_isolated_runner_with_mock_cli(self):
        corpus=Path(__file__).parent/'cli-inputs-v1'
        fixtures=json.loads((corpus/'manifest.json').read_text())['fixtures']
        def process(command, **kwargs):
            fixture=next(f for f in fixtures if Path(command[4]).name==f['pdf'])
            expected=fixture['expected'];output=Path(command[5])/Path(fixture['pdf']).stem;output.mkdir(parents=True)
            text='\n'.join(expected['ordered_text']+expected['text'])+'\n'
            text+='\n'.join('Equation: '+v for v in expected['equations'])+'\n'
            text+='<table>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in expected['table'])+'</table>'
            (output/(fixture['id']+'.md')).write_text(text)
            metadata=dict(file_name=fixture['pdf'],num_pages=2,total_token_count=8,total_chunks=2,pages=[dict(page_num=i,token_count=4,num_chunks=1,page_box=[0,0,1632,2112]) for i in range(2)])
            (output/(fixture['id']+'_metadata.json')).write_text(json.dumps(metadata))
            self.assertEqual(command[-5:], ['--method','vllm','--batch-size','2','--no-html'])
            self.assertNotEqual(kwargs['env']['HOME'], str(Path.home()))
            self.assertEqual(kwargs['env']['VLLM_API_BASE'], 'http://mock.invalid/v1')
            self.assertEqual(kwargs['env']['MAX_VLLM_RETRIES'], '0')
            return subprocess.CompletedProcess(command,0,b'CLI completed',b'')
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys,'argv',['cli_acceptance','--base-url','http://mock.invalid/v1','--corpus',str(corpus),'--output',str(Path(tmp)/'run')]), patch('cli_acceptance.subprocess.run',side_effect=process):
            self.assertEqual(cli_acceptance.main(),0)
            reports=json.loads((Path(tmp)/'run/runs.json').read_text())
            self.assertEqual(len(reports),2)
            self.assertTrue(all(r['environment']['MAX_VLLM_RETRIES']=='0' for r in reports))
            self.assertTrue(all(r['finish_reasons'] is None and not r['provenance_verified'] for r in reports))
if __name__=='__main__':unittest.main()
