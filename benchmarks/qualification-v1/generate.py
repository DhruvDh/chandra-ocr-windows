"""Create an independently composed synthetic regression set without inference."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    if (root / 'manifest.json').exists():
        raise SystemExit('Refusing to replace a frozen manifest')
    font_path = Path(args.font)
    font_hash = digest(font_path.read_bytes())
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in [22, 24, 30, 38]}
    fixtures = []

    def page():
        image = Image.new('RGB', (1000, 1200), 'white')
        return image, ImageDraw.Draw(image), []

    def block(draw, blocks, text, x, y, width, size=24, label='Text'):
        words = text.split(); lines = []; line = ''
        for word in words:
            proposal = (line + ' ' + word).strip()
            if draw.textlength(proposal, font=fonts[size]) > width and line:
                lines.append(line); line = word
            else:
                line = proposal
        if line:
            lines.append(line)
        for index, value in enumerate(lines):
            draw.text((x, y + index * (size + 9)), value, font=fonts[size], fill='black')
        box = [x, y, x + max(draw.textlength(v, font=fonts[size]) for v in lines), y + len(lines) * (size + 9)]
        blocks.append({'text': text, 'label': label, 'draw_region_pixels': box})
        return box[3]

    def freeze(identifier, image, blocks, table=None):
        path = root / (identifier + '.png')
        image.save(path, compress_level=9)
        expected = {'ordered_blocks': [b['text'] for b in blocks], 'table': table or [], 'end_marker': blocks[-1]['text'], 'geometry_reference': blocks}
        fixtures.append({'id': identifier, 'image': path.name, 'width': image.width, 'height': image.height, 'mode': 'RGB', 'image_sha256': digest(path.read_bytes()), 'pixels_sha256': digest(image.tobytes()), 'expected': expected, 'completion_token_budget_target': [400, 650], 'token_budget_scope': 'Planning estimate including OCR layout markup; actual token counts must be measured, not asserted.'})

    image, draw, blocks = page()
    block(draw, blocks, 'FIELD NOTE: TWO ROUTES', 55, 40, 890, 38)
    block(draw, blocks, 'A synthetic account of a valley survey', 55, 103, 890, 24)
    draw.line((55, 155, 945, 155), fill='black', width=2)
    y = block(draw, blocks, 'WESTERN ROUTE', 55, 190, 420, 30)
    for text in [
        'Mira began at the cedar gate. Her notebook listed a blue bridge, a stone basin, and a narrow path beside the creek.',
        'At noon she marked the shaded bend. The water moved slowly there, although the open channel beyond the ridge remained clear.',
        'She kept the original sequence of observations. A later summary must preserve that sequence rather than merge the two routes.',
    ]:
        y = block(draw, blocks, text, 55, y + 24, 420)
    y = block(draw, blocks, 'EASTERN ROUTE', 525, 190, 420, 30)
    for text in [
        'Oren left the eastern arch after breakfast. He carried a red flag and stopped first at the orchard, then at the empty mill.',
        'Clouds gathered above the slope. He recorded the changed light before measuring the final distance to the wooden shelter.',
        'The route names identify separate accounts. Read the western column completely before reading the eastern column.',
    ]:
        y = block(draw, blocks, text, 525, y + 24, 420)
    block(draw, blocks, 'Equation: q = 2r - 5; r = 8; q = 11', 55, 970, 890, 24)
    block(draw, blocks, 'END-ROUTES-VERIFIED', 55, 1090, 890, 24)
    freeze('routes', image, blocks)

    image, draw, blocks = page()
    block(draw, blocks, 'SIGNED BALANCE LEDGER', 55, 40, 890, 38)
    block(draw, blocks, 'Fictional measurements from four sampling stations', 55, 109, 890, 24)
    block(draw, blocks, 'Keep each signed quantity with its station and column. A negative balance is a recorded result, not a missing value.', 55, 185, 890, 24)
    table = [['Station', 'Shift (mm)', 'Rate (g/h)', 'Balance (g)'], ['Birch', '-2.75', '0.125', '+18.50'], ['Cobalt', '+4.20', '-0.375', '-6.25'], ['Dune', '0.00', '+1.250', '+0.75'], ['Ember', '-0.40', '0.050', '-12.00']]
    xs = [55, 280, 505, 720, 945]; top = 335; height = 83
    for row, cells in enumerate(table):
        for col, cell in enumerate(cells):
            draw.text((xs[col] + 14, top + row * height + 24), cell, font=fonts[24 if row else 22], fill='black')
    for row in range(len(table) + 1):
        draw.line((55, top + row * height, 945, top + row * height), fill='black', width=2)
    for x in xs:
        draw.line((x, top, x, top + len(table) * height), fill='black', width=2)
    blocks.append({'text': ' '.join(' '.join(row) for row in table), 'label': 'Table', 'draw_region_pixels': [55, top, 945, top + len(table) * height]})
    block(draw, blocks, 'Footnote: The plus sign is explicit where printed. Rates retain three decimal places; balances retain two. Zero shift does not imply zero balance.', 55, 800, 890, 24)
    block(draw, blocks, 'Audit note: Cobalt has a positive shift and a negative rate. Ember has the smallest positive rate. These statements must agree with their table rows.', 55, 920, 890, 24)
    block(draw, blocks, 'END-LEDGER-VERIFIED', 55, 1090, 890, 24)
    freeze('ledger', image, blocks, table)
    manifest = {'schema': 'chandra-independent-synthetic-qualification-v1', 'split': 'independent_synthetic_regression', 'scope': 'New synthetic content and layout; no real-world general accuracy claim. Frozen before baseline or optimized candidate inference.', 'generator_sha256': digest(Path(__file__).read_bytes()), 'font_name': font_path.name, 'font_sha256': font_hash, 'pillow_version': __import__('PIL').__version__, 'fixtures': fixtures}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
