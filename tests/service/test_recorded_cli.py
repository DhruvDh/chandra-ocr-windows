import base64
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('recorded_cli',ROOT/'benchmarks/recorded_cli.py')
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)

class RecordedTests(unittest.TestCase):
    def payload(self):
        return json.dumps({'max_tokens':10,'messages':[{'content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,eA=='}}]}]}).encode()
    def completion(self,finish='stop'):
        return json.dumps({'choices':[{'message':{'content':'synthetic'},'finish_reason':finish}],'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}).encode()
    def test_metadata_fail_closed(self):
        self.assertTrue(r.observed_response(200,self.completion(),self.payload())['passed'])
        for status,body in [(200,self.completion('length')),(503,self.completion()),(200,b'{"choices":[]}'),(200,b'{}')]:
            self.assertFalse(r.observed_response(status,body,self.payload())['passed'])
    def test_unique_pixels_not_request_arrival_order(self):
        facts=[dict(fixture_id='native',page_num=i,pixels_sha256=str(i),width=10,height=10,mode='RGB') for i in range(2)]
        records=[dict(id=i,method='POST',image=facts[1-i],completion={'passed':True,'usage':{'completion_tokens':3}}) for i in range(2)]
        result=r.gate(records,facts,[{'fixture_id':'native','correctness':{'passed':True},'metadata':{'pages':[{'page_num':0,'token_count':3},{'page_num':1,'token_count':3}]}}]);self.assertTrue(result['passed']);self.assertEqual(result['associations'][0]['matches'][0]['page_num'],1)
        records.append(records[0]);self.assertFalse(r.gate(records,facts,[{'fixture_id':'native','correctness':{'passed':True},'metadata':{'pages':[{'page_num':0,'token_count':3},{'page_num':1,'token_count':3}]}}])['passed'])
        facts[0]['pixels_sha256']='1';self.assertFalse(r.gate(records,facts,[])['passed'])
    def test_image_hash_matches_actual_decoded_pixels(self):
        from PIL import Image
        image=Image.new('RGB',(3,2),'white');buffer=io.BytesIO();image.save(buffer,format='PNG');data=buffer.getvalue()
        request=json.dumps({'messages':[{'content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}}]}]}).encode()
        fact=r.image_fact(request)
        self.assertEqual(fact['pixels_sha256'],r.sha(image.tobytes()));self.assertEqual((fact['width'],fact['height']),(3,2))

    def test_cli_token_binding_and_missing_page_fail(self):
        expected=[dict(fixture_id='native',page_num=0,pixels_sha256='x',width=10,height=10,mode='RGB')]
        record=dict(id=0,method='POST',image=expected[0],completion={'passed':True,'usage':{'completion_tokens':3}})
        cli=[dict(fixture_id='native',correctness={'passed':True},metadata={'pages':[{'page_num':0,'token_count':4}]})]
        self.assertFalse(r.gate([record],expected,cli)['passed']);self.assertFalse(r.gate([],expected,cli)['passed'])

    def test_response_cap_preserves_prefix_and_actual_upstream_status(self):
        response=b'0123456789abcdefghij';posts=[]
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                posts.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200);self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
        upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream);thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory,patch.object(r,'image_fact',return_value={}),patch.object(r,'RESPONSE_LIMIT',10):
                recorder=r.Recorder(f'http://127.0.0.1:{upstream.server_port}/v1',Path(directory)/'http',2).start()
                try:
                    request=urllib.request.Request(recorder.base_url+'/chat/completions',data=self.payload())
                    with self.assertRaises(urllib.error.HTTPError) as raised:urllib.request.urlopen(request,timeout=3)
                    self.assertEqual(raised.exception.code,502);record=recorder.records[0]
                    self.assertEqual(record['upstream_status'],200);self.assertEqual(record['delivered_status'],502)
                    self.assertEqual((recorder.output/record['response_file']).read_bytes(),response[:10]);self.assertEqual(len(posts),1)
                    self.assertFalse(record['completion']['passed'])
                finally:recorder.close()
        finally:upstream.shutdown();upstream.server_close();thread.join()

    def test_proxy_preserves_bytes_subpath_status_and_no_redirect_retry(self):
        posts=[];response=self.completion();payload=self.payload()
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=self.rfile.read(int(self.headers['Content-Length']));posts.append((self.path,body))
                self.send_response(307);self.send_header('Location','http://127.0.0.1:1/unwanted');self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(response)));self.end_headers();self.wfile.write(response)
        upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream);thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory,patch.object(r,'image_fact',return_value={'pixels_sha256':'x'}):
                recorder=r.Recorder(f'http://127.0.0.1:{upstream.server_port}/backends/waystone/v1',Path(directory)/'http',2).start()
                try:
                    class NoRedirect(urllib.request.HTTPRedirectHandler):
                        def redirect_request(self,*args,**kwargs):return None
                    opener=urllib.request.build_opener(NoRedirect(),urllib.request.ProxyHandler({}))
                    request=urllib.request.Request(recorder.base_url+'/chat/completions',data=payload,headers={'Content-Type':'application/json'})
                    with self.assertRaises(urllib.error.HTTPError) as raised:opener.open(request,timeout=3)
                    self.assertEqual(raised.exception.code,307);self.assertEqual(raised.exception.read(),response)
                    self.assertEqual(posts,[('/backends/waystone/v1/chat/completions',payload)])
                    record=recorder.records[0];self.assertEqual(record['request_sha256'],r.sha(payload));self.assertEqual(record['response_sha256'],r.sha(response));self.assertFalse(record['completion']['passed'])
                finally:recorder.close()
        finally:upstream.shutdown();upstream.server_close();thread.join()
