import json
import unittest
from endpoint import validate, parse_sse

class EndpointGates(unittest.TestCase):
    def fixture(self):
        expected=dict(table=[['12.50']],numbers=['12.50'],text=['TITLE'],end_markers=['END'],ordered_text=['TITLE','END'],required_labels=['Table'])
        result=dict(content='<div data-label="Table" data-bbox="0 0 1000 1000">TITLE<table><tr><td>12.50</td></tr></table>END</div>',finish_reason='stop',usage=dict(prompt_tokens=2,completion_tokens=3,total_tokens=5))
        return expected,result
    def test_numeric_and_finish_fail(self):
        e,r=self.fixture(); self.assertTrue(validate(e,r)['passed']); r['content']=r['content'].replace('12.50','12.5'); self.assertFalse(validate(e,r)['passed']); e,r=self.fixture(); r['finish_reason']='length'; self.assertFalse(validate(e,r)['passed'])
    def test_unknown_and_metadata_fail(self):
        e,r=self.fixture(); r['usage']=None; self.assertFalse(validate(e,r)['passed']); self.assertFalse(validate(e,{})['passed']); e,r=self.fixture(); r['content']=r['content'].replace('0 0 1000 1000','0 0 1001 1000'); self.assertFalse(validate(e,r)['passed'])
    def test_equation_suffix_regression(self):
        e, r = self.fixture(); e['equations'] = ['y = 22']
        r['content'] += '<div data-label="Math" data-bbox="0 0 100 100">y = 22</div>'
        self.assertTrue(validate(e, r)['passed'])
        r['content'] = r['content'].replace('y = 22', 'y = 220')
        self.assertFalse(validate(e, r)['passed'])
    def test_stream_usage_done(self):
        events=[{'choices':[{'index':0,'delta':{'content':'hello'},'finish_reason':None}]},{'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},{'choices':[],'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}]
        raw=b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)
        self.assertIsNone(parse_sse(raw)['finish_reason']); parsed=parse_sse(raw+b'data: [DONE]\n\n'); self.assertEqual(parsed['content'],'hello'); self.assertEqual(parsed['finish_reason'],'stop'); self.assertEqual(parsed['usage']['total_tokens'],5)
if __name__=='__main__': unittest.main()
