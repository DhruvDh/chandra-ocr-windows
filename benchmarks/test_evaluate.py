import unittest
from evaluate import numerical, logits, ocr, summary, check_equations

class Gates(unittest.TestCase):
    def test_nonfinite_and_drift(self):
        self.assertFalse(numerical([1.0], [float('nan')], 1, 1)['passed'])
        self.assertFalse(numerical([1.0], [1.2], .01, .01)['passed'])
    def test_logits_margin(self):
        r = logits([[1, 1.0001]], [[1.0001, 1]], .01, .01)
        self.assertFalse(r['passed'])
        self.assertEqual(r['argmax_agreement'], 0)
    def test_logits_identity_and_target_probabilities(self):
        context = dict(vocabulary_size=2, model_revision='a'*40, tokenizer_sha256='a'*64, config_sha256='b'*64, image_sha256='c'*64, prompt_sha256='d'*64, processor_profile={}, positions=[1], prefix_token_ids=[0,1], target_token_ids=[1])
        self.assertTrue(logits([[0.,1.]], [[0.,1.]], 0, 0, context, dict(context))['passed'])
        report = logits([[0.,1.]], [[0.,1.]], 0, 0, context, dict(context))
        self.assertAlmostEqual(report['target_token_log_probabilities'][0]['reference_log_probability'], -0.3132616875)
        self.assertFalse(logits([[0.,1.]], [[0.,1.]], 0, 0)['passed'])
        candidate = dict(context, prefix_token_ids=[1,1])
        self.assertFalse(logits([[0.,1.]], [[0.,1.]], 0, 0, context, candidate)['passed'])
        candidate = dict(context, vocabulary_size=3)
        self.assertFalse(logits([[0.,1.]], [[0.,1.]], 0, 0, context, candidate)['passed'])
    def test_sparse_teacher_positions(self):
        context = dict(vocabulary_size=2, model_revision='a'*40, tokenizer_sha256='a'*64, config_sha256='b'*64, image_sha256='c'*64, prompt_sha256='d'*64, processor_profile={}, positions=[1,3], prefix_token_ids=[0,1], teacher_token_ids=[1,0,1], target_token_ids=[1,1])
        self.assertTrue(logits([[0.,1.],[0.,1.]], [[0.,1.],[0.,1.]], 0, 0, context, dict(context))['passed'])
        context['target_token_ids'] = [1,0]
        self.assertFalse(logits([[0.,1.],[0.,1.]], [[0.,1.],[0.,1.]], 0, 0, context, dict(context))['passed'])
    def test_truncation_and_cell_error(self):
        expected = dict(pages=1, file_name='synthetic.pdf', table=[['12.50']], end_markers=['END'])
        actual = dict(text='<table><tr><td>12.50</td></tr></table>END', max_output_tokens=12384, finish_reasons=['stop'], metadata=dict(num_pages=1, file_name='synthetic.pdf', total_token_count=4, pages=[dict(page_num=0, token_count=4, num_chunks=1, page_box=[0,0,10,10])]))
        self.assertTrue(ocr(expected,actual)['passed'])
        actual['finish_reasons'] = ['length']
        self.assertIn('truncation_or_finish',ocr(expected,actual)['failures'])
        actual['text'] = actual['text'].replace('12.50','12.5')
        self.assertIn('table_cells_and_order',ocr(expected,actual)['failures'])
    def metadata_fixture(self):
        return (dict(pages=1, file_name='synthetic.pdf', table=[['12.50']], ordered_text=['12.50', 'END']),
                dict(text='<table><tr><td>12.50</td></tr></table>END', max_output_tokens=12384, finish_reasons=['stop'], metadata=dict(num_pages=1, file_name='synthetic.pdf', total_token_count=4, pages=[dict(page_num=0, token_count=4, num_chunks=1, page_box=[0,0,10,10])])))
    def test_infinite_geometry_fails(self):
        e, a = self.metadata_fixture(); a['metadata']['pages'][0]['page_box'][2] = float('inf')
        self.assertIn('page_geometry_chunks', ocr(e, a)['failures'])
    def test_boolean_token_and_chunk_counts_fail(self):
        e, a = self.metadata_fixture(); a['metadata']['pages'][0]['token_count'] = True
        self.assertIn('page_tokens', ocr(e, a)['failures'])
        a['metadata']['pages'][0]['num_chunks'] = True
        self.assertIn('page_geometry_chunks', ocr(e, a)['failures'])
    def test_reading_order_fails(self):
        e, a = self.metadata_fixture(); e['ordered_text'] = ['END', '12.50']
        self.assertIn('reading_order', ocr(e, a)['failures'])
    def test_equations_require_complete_nodes(self):
        self.assertEqual(check_equations(['y = 22'], '<div data-label="Math">y = 22</div>'), [])
        self.assertTrue(check_equations(['y = 22'], '<div data-label="Math">y = 220</div>'))
        self.assertTrue(check_equations(['y = 22'], '<div data-label="Math">y = 22 + 1</div>'))
        self.assertEqual(check_equations(['x = 5', 'y = 22'], '<div>Equation: x = 5; y = 22</div>'), [])
        self.assertEqual(check_equations(['y = 22'], r'\(y = 22\)'), [])
    def test_performance_gate(self):
        with self.assertRaises(ValueError): summary([dict(correctness={'passed':False})])
        with self.assertRaises(ValueError): summary([dict(correctness={'passed':True},profiled=True)])

if __name__ == '__main__': unittest.main()
