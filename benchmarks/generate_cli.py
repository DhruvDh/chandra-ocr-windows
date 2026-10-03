"""Freeze two-page public native-text and scanned CLI fixtures; never overwrite a corpus."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import textwrap
from PIL import ImageFilter
import pypdfium2 as pdfium
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader


def sha(data): return hashlib.sha256(data).hexdigest()

TABLES = [
    [['Sample', 'Mass (g)', 'Count', 'Value'], ['Alpha', '12.50', '17', '203.25'], ['Beta', '7.25', '29', '118.75'], ['Total', '19.75', '46', '322.00']],
    [['Trial', 'Offset', 'Rate', 'Result'], ['Gamma', '-2.50', '0.125', '16.00'], ['Delta', '+3.75', '1.250', '27.50'], ['Total', '1.25', '1.375', '43.50']],
]

def generate(output, font):
    output.mkdir(parents=True, exist_ok=False)
    pdfmetrics.registerFont(TTFont('Fixture', str(font)))
    native = output / 'native.pdf'
    c = canvas.Canvas(str(native), pagesize=(612, 792), invariant=1, pageCompression=1)
    c.setTitle('Public synthetic Chandra CLI acceptance')
    ordered = []; prose = []; equations = ['y = 3x + 7', 'x = 5', 'y = 22', 'z = 2a - 1', 'a = 4', 'z = 7']
    for page in range(2):
        title = f'Synthetic measurement - page {page+1}'
        c.setFont('Fixture', 19); c.drawString(42, 744, title)
        c.setFont('Fixture', 11)
        intro = 'This public fixture tests complete scientific transcription across two pages.'
        c.drawString(42, 715, intro)
        ordered.append(title); prose.append(intro)
        for column in range(2):
            heading = f'COLUMN {page+1}{"A" if column == 0 else "B"}'
            c.setFont('Fixture', 13); c.drawString(42+column*270, 676, heading)
            ordered.append(heading)
            paragraph = ('Observations are recorded before interpretation. Each sample keeps its name, quantity, sign and decimal precision.' if column == 0 else 'Independent readings follow the left column. A complete page preserves every table cell and the terminal marker.')
            c.setFont('Fixture', 11)
            for line, value in enumerate(textwrap.wrap(paragraph, width=35)):
                c.drawString(42+column*270, 653-line*18, value)
        c.setFont('Fixture', 12); c.drawString(42, 518, 'Equation: ' + '; '.join(equations[page*3:page*3+3]))
        table = TABLES[page]; left=42; top=460; widths=[156,120,102,150]
        c.setLineWidth(.6)
        for row, cells in enumerate(table):
            x=left
            for cell, width in zip(cells, widths):
                c.rect(x, top-(row+1)*35, width, 35)
                c.setFont('Fixture', 11); c.drawString(x+8, top-row*35-23, cell); x += width
        marker = f'END-CLI-PAGE-{page+1}-COMPLETE'
        c.setFont('Fixture', 12); c.drawString(42, 250, marker); ordered.append(marker)
        c.showPage()
    c.save()
    document = pdfium.PdfDocument(str(native)); scan=canvas.Canvas(str(output/'scanned.pdf'), pagesize=(612,792), invariant=1, pageCompression=1)
    for page in document:
        image = page.render(scale=2).to_pil().convert('RGB').filter(ImageFilter.GaussianBlur(.25))
        scan.drawImage(ImageReader(image), 0, 0, 612, 792); scan.showPage(); page.close()
    document.close(); scan.save()
    files = []; fixtures = []
    for name in ('native', 'scanned'):
        path = output/(name+'.pdf'); doc = pdfium.PdfDocument(str(path)); pages=[]
        for i, page in enumerate(doc):
            image = page.render(scale=192/72).to_pil().convert('RGB'); png=output/f'{name}-page-{i+1}.png'; image.save(png)
            textpage = page.get_textpage(); native_text=textpage.get_text_range(); textpage.close()
            pages.append(dict(page_num=i,width=image.width,height=image.height,mode='RGB',pixels_sha256=sha(image.tobytes()),image=png.name,image_sha256=sha(png.read_bytes()),native_text_characters=len(native_text)))
            page.close()
        doc.close()
        if (name=='native' and any(p['native_text_characters']==0 for p in pages)) or (name=='scanned' and any(p['native_text_characters']!=0 for p in pages)):
            raise ValueError('Unexpected native/scanned text-layer boundary')
        fixtures.append(dict(id=name,pdf=path.name,pdf_sha256=sha(path.read_bytes()),rendered_pages=pages,expected=dict(file_name=path.name,pages=2,table=TABLES[0]+TABLES[1],text=prose,numbers=[v for table in TABLES for row in table[1:] for v in row[1:]],equations=equations,end_markers=['END-CLI-PAGE-1-COMPLETE','END-CLI-PAGE-2-COMPLETE'],ordered_text=ordered)))
    for path in sorted(output.iterdir()): files.append(dict(path=path.name,sha256=sha(path.read_bytes())))
    manifest=dict(schema='chandra-cli-corpus-v1',split='development',license='CC0-1.0 (synthetic content)',generator_sha256=sha(Path(__file__).read_bytes()),font_name=font.name,font_sha256=sha(font.read_bytes()),runtime={p:importlib.metadata.version(p) for p in ('pillow','reportlab','pypdfium2')},pdfium_version=str(pdfium.PDFIUM_INFO),render_dpi=192,files=files,fixtures=fixtures)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path); p.add_argument('--font',type=Path,required=True); a=p.parse_args(); generate(a.output,a.font)
if __name__=='__main__': main()
