"""Offline, stdlib-only correctness gates; never invokes a model or endpoint."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
import re
from pathlib import Path
import statistics

SCHEMA = 'chandra-evaluation-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Tables(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = None; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        if tag in ('th', 'td'): self.cell = []
    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            if self.row is not None: self.row.append(''.join(self.cell).strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row); self.row = None

class EquationNodes(HTMLParser):
    """Retain math nodes and explicit Equation: text blocks without substring matching."""
    def __init__(self):
        super().__init__(); self.stack = []; self.nodes = []
    def handle_starttag(self, tag, attrs):
        if tag in ('div', 'p', 'math'):
            attrs = dict(attrs)
            self.stack.append([tag, attrs.get('data-label') in ('Math', 'Equation', 'Formula') or tag == 'math', []])
    def handle_data(self, data):
        for node in self.stack: node[2].append(data)
    def handle_endtag(self, tag):
        if self.stack and self.stack[-1][0] == tag:
            _, is_math, data = self.stack.pop(); value = ''.join(data).strip()
            if is_math: self.nodes.append(value)
            elif value.startswith('Equation:'): self.nodes.append(value[len('Equation:'):].strip())

def equation_representations(text):
    parser = EquationNodes(); parser.feed(text)
    nodes = parser.nodes
    nodes += re.findall(r'^Equation:\s*(.+)$', text, flags=re.M)
    # Explicit TeX math delimiters also define a complete representation boundary.
    nodes += re.findall(r'\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|(?<!\$)\$([^$]+)\$(?!\$)', text, flags=re.S)
    flattened = []
    for node in nodes:
        if isinstance(node, tuple): node = next((v for v in node if v), '')
        # The synthetic fixture prints three equations separated by semicolons.
        flattened.extend(re.sub(r'\s+', '', value.strip()) for value in node.split(';') if value.strip())
    return flattened

def check_equations(expected, text):
    nodes = equation_representations(text)
    return ['equations:' + value for value in expected if re.sub(r'\s+', '', value) not in nodes]

def ocr(expected, actual):
    """Actual contains text, metadata, and one finish_reason per page."""
    text = actual['text']; meta = actual['metadata']; pages = meta['pages']
    failures = []
    def check(ok, label):
        if not ok: failures.append(label)
    check(meta['num_pages'] == expected['pages'] == len(pages), 'page_count')
    check([p['page_num'] for p in pages] == list(range(expected['pages'])), 'page_order')
    check(meta['file_name'] == expected['file_name'], 'file_name')
    check(len(actual['finish_reasons']) == expected['pages'] and all(r == 'stop' for r in actual['finish_reasons']), 'truncation_or_finish')
    check(type(actual['max_output_tokens']) is int and all(type(p['token_count']) is int and 0 < p['token_count'] < actual['max_output_tokens'] for p in pages), 'page_tokens')
    check(meta['total_token_count'] == sum(p['token_count'] for p in pages), 'token_sum')
    check(all(type(p['num_chunks']) is int and p['num_chunks'] > 0 and len(p['page_box']) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in p['page_box']) and p['page_box'][2] > p['page_box'][0] and p['page_box'][3] > p['page_box'][1] for p in pages), 'page_geometry_chunks')
    for field in ('text', 'numbers', 'end_markers'):
        for value in expected.get(field, []): check(value in text, field + ':' + value)
    failures.extend(check_equations(expected.get('equations', []), text))
    positions = [text.find(value) for value in expected.get('ordered_text', [])]
    check(all(v >= 0 for v in positions) and positions == sorted(positions), 'reading_order')
    parsed = Tables(); parsed.feed(text)
    rows = actual.get('table_rows', parsed.rows)
    check(rows == expected['table'], 'table_cells_and_order')
    # Math node representations compare exactly after whitespace normalization.
    return {'passed': not failures, 'failures': failures}

def numerical(reference, candidate, atol, rtol):
    if len(reference) != len(candidate) or not reference:
        return {'passed': False, 'failures': ['shape_or_empty']}
    if not all(math.isfinite(x) for x in reference + candidate):
        return {'passed': False, 'failures': ['nonfinite']}
    diffs = [abs(a-b) for a,b in zip(reference,candidate)]
    return {'passed': all(d <= atol + rtol*abs(a) for a,d in zip(reference,diffs)),
            'max_absolute_error': max(diffs), 'rms_error': math.sqrt(sum(d*d for d in diffs)/len(diffs)),
            'max_relative_error': max(d/max(abs(a),1e-30) for a,d in zip(reference,diffs)), 'atol': atol, 'rtol': rtol}

def logits(reference, candidate, atol, rtol, reference_context=None, candidate_context=None):
    if len(reference) != len(candidate) or not reference: return {'passed': False, 'failures':['positions']}
    identity_failures = []
    fields = ('vocabulary_size', 'model_revision', 'tokenizer_sha256', 'config_sha256', 'image_sha256', 'prompt_sha256', 'processor_profile', 'positions', 'prefix_token_ids', 'target_token_ids')
    if not isinstance(reference_context, dict) or not isinstance(candidate_context, dict):
        identity_failures.append('missing_teacher_forced_identity')
    else:
        fields = fields + (('teacher_token_ids',) if 'teacher_token_ids' in reference_context or 'teacher_token_ids' in candidate_context else ())
        for field in fields:
            if field not in reference_context or field not in candidate_context or reference_context[field] != candidate_context[field]:
                identity_failures.append('context:' + field)
        c = reference_context
        vocab = c.get('vocabulary_size')
        prefix = c.get('prefix_token_ids'); positions = c.get('positions'); targets = c.get('target_token_ids')
        integer_tokens = lambda values: isinstance(values, list) and all(type(v) is int and 0 <= v < vocab for v in values)
        if type(vocab) is not int or vocab <= 1:
            identity_failures.append('vocabulary_size')
        else:
            if any(len(row) != vocab for row in reference + candidate): identity_failures.append('full_vocabulary_coverage')
            if not integer_tokens(prefix) or not prefix: identity_failures.append('prefix_token_ids')
            if not integer_tokens(targets) or len(targets) not in (len(reference), len(reference)-1): identity_failures.append('target_token_ids')
            if 'teacher_token_ids' in c:
                teacher = c['teacher_token_ids']
                if not integer_tokens(teacher) or not teacher:
                    identity_failures.append('teacher_token_ids')
                    teacher = []
                if not isinstance(positions, list) or len(positions) != len(reference) or any(type(v) is not int for v in positions) or not isinstance(prefix, list) or positions != sorted(set(positions)) or any(not len(prefix)-1 <= v < len(prefix)-1+len(teacher) for v in positions):
                    identity_failures.append('positions')
                elif not isinstance(targets, list) or len(targets)!=len(positions) or targets != [teacher[v-len(prefix)+1] for v in positions]: identity_failures.append('targets_at_positions')
            elif not isinstance(positions, list) or len(positions) != len(reference) or any(type(v) is not int for v in positions) or not isinstance(prefix, list) or positions != list(range(len(prefix)-1, len(prefix)-1+len(reference))): identity_failures.append('positions')
        for field in ('tokenizer_sha256', 'config_sha256', 'image_sha256', 'prompt_sha256'):
            if not isinstance(c.get(field), str) or not re.fullmatch('[0-9a-f]{64}', c[field]): identity_failures.append('hash:' + field)
        if not isinstance(c.get('model_revision'), str) or not re.fullmatch('[0-9a-f]{40}', c['model_revision']): identity_failures.append('model_revision')
        if not isinstance(c.get('processor_profile'), dict): identity_failures.append('processor_profile')
    rows = [numerical(a,b,atol,rtol) for a,b in zip(reference,candidate)]
    valid = all('failures' not in r for r in rows)
    agreements = []
    divergences = []
    margins = []
    target_probabilities = []
    if valid:
        for position, (a,b) in enumerate(zip(reference,candidate)):
            # Stable softmax and KL over the FULL vocabulary, never top-k-only logits.
            la = max(a) + math.log(sum(math.exp(v-max(a)) for v in a))
            lb = max(b) + math.log(sum(math.exp(v-max(b)) for v in b))
            divergences.append(sum(math.exp(x-la)*((x-la)-(y-lb)) for x,y in zip(a,b)))
            agreements.append(max(range(len(a)),key=a.__getitem__) == max(range(len(b)),key=b.__getitem__))
            if not identity_failures and position < len(reference_context['target_token_ids']):
                target = reference_context['target_token_ids'][position]
                target_probabilities.append(dict(target_token_id=target, reference_log_probability=a[target]-la, candidate_log_probability=b[target]-lb, absolute_error=abs((a[target]-la)-(b[target]-lb))))
            ordered = sorted(a, reverse=True); margins.append(ordered[0]-ordered[1] if len(a)>1 else None)
    return {'passed': not identity_failures and valid and all(r['passed'] for r in rows) and all(agreements), 'identity_failures': identity_failures, 'target_token_log_probabilities': target_probabilities, 'positions': rows,
            'argmax_agreement': sum(agreements)/len(agreements) if agreements else None,
            'max_kl': max(divergences) if divergences else None, 'reference_top1_top2_margins': margins}

def verify_manifest(path):
    manifest = json.loads(Path(path).read_text()); root = Path(path).resolve().parent
    failures = []
    items = manifest.get('files')
    if items is None:
        items = [{'path':f['image'],'sha256':f['image_sha256']} for f in manifest.get('fixtures',[])]
    if not items: return {'passed':False,'failures':['empty_manifest']}
    for item in items:
        p = (root/item['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != item['sha256']: failures.append(item['path'])
    return {'passed': not failures, 'failures': failures, 'manifest_sha256': sha(path)}

def summary(runs):
    groups = {}
    for r in runs:
        if r.get('correctness',{}).get('passed') is not True: raise ValueError('Performance promotion blocked by failed correctness')
        if r.get('provenance_verified') is not True: raise ValueError('Performance promotion blocked by unverified provenance')
        if r.get('profiled'): raise ValueError('Profiled run cannot enter unprofiled performance baseline')
        key = r['phase']; groups.setdefault(key, []).append(r)
    result = {}
    for key, group in groups.items():
        if len(group) < 5: raise ValueError('Performance promotion requires at least five repeats per phase')
        values = sorted(r['elapsed_seconds'] for r in group)
        if any(v <= 0 or not math.isfinite(v) for v in values): raise ValueError('Invalid timing')
        result[key] = {'n':len(values),'median_seconds':statistics.median(values), 'min_seconds':min(values),
                       'max_seconds':max(values), 'p95_seconds':values[max(0,math.ceil(.95*len(values))-1)],
                       'pages_per_second':sum(r['pages'] for r in group)/sum(values),
                       'peak_device_vram_bytes':max((r.get('peak_device_vram_bytes') or 0 for r in group)) or None}
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['ocr','tensor','logits','manifest','summary']); p.add_argument('input'); p.add_argument('--expected'); p.add_argument('--atol',type=float,default=0); p.add_argument('--rtol',type=float,default=0)
    a = p.parse_args(); data = json.loads(Path(a.input).read_text())
    if a.atol < 0 or a.rtol < 0 or not math.isfinite(a.atol+a.rtol): p.error('Tolerances must be finite and nonnegative')
    if a.mode == 'ocr': result = ocr(json.loads(Path(a.expected).read_text()),data)
    elif a.mode == 'tensor': result = numerical(data['reference'],data['candidate'],a.atol,a.rtol)
    elif a.mode == 'logits': result = logits(data['reference'],data['candidate'],a.atol,a.rtol,data.get('reference_context'),data.get('candidate_context'))
    elif a.mode == 'manifest': result = verify_manifest(a.input)
    else: result = summary(data)
    print(json.dumps({'schema':SCHEMA,'result':result},indent=2,allow_nan=False))
    return 0 if result.get('passed',True) else 1

if __name__ == '__main__': raise SystemExit(main())
