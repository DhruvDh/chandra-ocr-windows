"""Offline synthetic content gates and separate geometry observations; no inference."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re


def canonical(value):
    # Ignore whitespace/layout wrapping and explicit math delimiters only.
    # Preserve all printed signs, decimal places, punctuation and word characters.
    value = value.replace('\u2212', '-')
    for delimiter in ['\\(', '\\)', '\\[', '\\]', '$']:
        value = value.replace(delimiter, '')
    return re.sub(r'\s+', '', value)


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []; self.rows = []; self.row = None; self.cell = None
        self.stack = []; self.geometry = []

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        if tag in ('th', 'td'): self.cell = []
        if tag == 'div': self.stack.append({'attrs': dict(attrs), 'text': []})

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None: self.cell.append(data)
        for block in self.stack: block['text'].append(data)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            if self.row is not None: self.row.append(''.join(self.cell).strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row); self.row = None
        if tag == 'div' and self.stack:
            block = self.stack.pop(); attrs = block['attrs']
            raw = attrs.get('data-bbox'); box = None
            if raw:
                try:
                    values = [float(v) for v in re.split(r'[\s,]+', raw.strip())]
                    if len(values) == 4 and all(math.isfinite(v) for v in values): box = values
                except ValueError:
                    pass
            self.geometry.append({'text': ''.join(block['text']), 'label': attrs.get('data-label'), 'raw_bbox': raw, 'bbox': box})


def unpack(actual):
    if 'choices' in actual:
        choice = actual['choices'][0]
        return {'content': choice['message']['content'], 'finish_reason': choice.get('finish_reason'), 'usage': actual.get('usage'), 'stream_errors': actual.get('stream_errors', [])}
    result = actual.get('result', actual)
    return {'content': result.get('content', result.get('text', '')), 'finish_reason': result.get('finish_reason'), 'usage': result.get('usage'), 'stream_errors': result.get('stream_errors', [])}


def check(expected, actual, max_output_tokens=12384, geometry_space=None):
    result = unpack(actual); parser = Extract(); parser.feed(result['content'])
    observed = canonical(''.join(parser.text)); blocks = expected['ordered_blocks']
    failures = []; positions = []
    for index, block in enumerate(blocks):
        token = canonical(block); position = observed.find(token); positions.append(position)
        if position < 0: failures.append('missing_or_changed_block:' + str(index))
        elif observed.count(token) != 1: failures.append('duplicated_block:' + str(index))
    if any(v < 0 for v in positions) or positions != sorted(positions): failures.append('reading_order')
    if observed != canonical(''.join(blocks)): failures.append('complete_content_or_order')
    observed_rows = [[canonical(cell) for cell in row] for row in parser.rows]
    expected_rows = [[canonical(cell) for cell in row] for row in expected['table']]
    if observed_rows != expected_rows: failures.append('exact_table_row_column_associations')
    if not observed.endswith(canonical(expected['end_marker'])): failures.append('terminal_sentinel')
    if result['finish_reason'] != 'stop': failures.append('terminal_stop')
    if result['stream_errors']: failures.append('stream_error')
    usage = result['usage'] or {}; count = usage.get('completion_tokens')
    if type(count) is not int or not 0 < count < max_output_tokens: failures.append('usage_or_output_allowance')
    return {'passed': not failures, 'failures': failures, 'finish_reason': result['finish_reason'], 'completion_tokens': count, 'max_output_tokens': max_output_tokens, 'geometry': {'coordinate_space': geometry_space, 'observed_blocks': parser.geometry, 'draw_regions_pixels': expected['geometry_reference'], 'acceptance_tolerance': None, 'scope': 'Recorded separately; generated draw regions are not OCR ground-truth boxes. No geometric pass/fail criterion or inferred coordinate conversion.'}}


def compare(expected, reference, candidate, max_output_tokens=12384, geometry_space=None):
    baseline = check(expected, reference, max_output_tokens, geometry_space)
    trial = check(expected, candidate, max_output_tokens, geometry_space)
    def keyed(report):
        result = {}
        for block in report['geometry']['observed_blocks']:
            key = canonical(block['text'])
            result.setdefault(key, []).append(block)
        return result
    left, right = keyed(baseline), keyed(trial); geometry = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key, []), right.get(key, [])
        unique = len(a) == len(b) == 1
        item = {'content_key': key, 'reference_matches': len(a), 'candidate_matches': len(b), 'unique_content_match': unique}
        if unique:
            item.update({'reference_bbox': a[0]['bbox'], 'candidate_bbox': b[0]['bbox'], 'reference_label': a[0]['label'], 'candidate_label': b[0]['label']})
            if geometry_space and a[0]['bbox'] and b[0]['bbox']:
                item['candidate_minus_reference'] = [y - x for x, y in zip(a[0]['bbox'], b[0]['bbox'])]
        geometry.append(item)
    return {'passed': baseline['passed'] and trial['passed'], 'reference': baseline, 'candidate': trial, 'geometry_comparison': {'coordinate_space': geometry_space, 'blocks': geometry, 'acceptance_tolerance': None}, 'scope': 'Both outputs must independently satisfy frozen expected content. Baseline agreement alone is insufficient; geometry differences are observations, not qualified tolerances.'}


def verify(manifest_path):
    path = Path(manifest_path); manifest = json.loads(path.read_text()); failures = []
    for fixture in manifest['fixtures']:
        image = path.parent / fixture['image']
        if hashlib.sha256(image.read_bytes()).hexdigest() != fixture['image_sha256']: failures.append(fixture['id'])
    return {'passed': not failures, 'failures': failures, 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('fixture', choices=['routes', 'ledger']); p.add_argument('candidate'); p.add_argument('--reference'); p.add_argument('--max-output-tokens', type=int, default=12384); p.add_argument('--geometry-space'); a = p.parse_args()
    manifest_path = Path(__file__).with_name('manifest.json'); integrity = verify(manifest_path)
    if not integrity['passed']: raise SystemExit('Frozen image hash mismatch')
    manifest = json.loads(manifest_path.read_text()); expected = next(f['expected'] for f in manifest['fixtures'] if f['id'] == a.fixture)
    candidate = json.loads(Path(a.candidate).read_text())
    if a.reference:
        result = compare(expected, json.loads(Path(a.reference).read_text()), candidate, a.max_output_tokens, a.geometry_space)
    else:
        result = check(expected, candidate, a.max_output_tokens, a.geometry_space)
    result['corpus_integrity'] = integrity; print(json.dumps(result, indent=2)); raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
