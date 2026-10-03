"""Generate public synthetic PNG fixtures; requires already-installed Pillow."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def digest(b): return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('output'); p.add_argument('--font',required=True); a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): raise SystemExit('Refusing to overwrite frozen corpus')
    font_bytes=Path(a.font).read_bytes(); fixtures=[]
    for name,w,h,size in [('tiny',408,528,10),('small',816,1056,20),('representative',1632,2112,40)]:
        img=Image.new('RGB',(w,h),'white'); draw=ImageDraw.Draw(img); font=ImageFont.truetype(a.font,size)
        lines=['Synthetic OCR measurement', 'REFERENCE '+name.upper(), 'Scientific quantities must remain exact.', 'Equation: y = 3x + 7; x = 5; y = 22']
        for n,line in enumerate(lines): draw.text((int(w*.06),int(h*.06)+n*size*2),line,font=font,fill='black')
        table=[['Sample','Mass (g)','Count','Value'],['Alpha','12.50','17','203.25'],['Beta','7.25','29','118.75'],['Total','19.75','46','322.00']]
        x0=int(w*.06); y0=int(h*.37); cellw=int(w*.22); cellh=size*3
        for row,cells in enumerate(table):
            for col,cell in enumerate(cells): draw.text((x0+col*cellw+size//3,y0+row*cellh+size//2),cell,font=font,fill='black')
        for r in range(5): draw.line((x0,y0+r*cellh,x0+4*cellw,y0+r*cellh),fill='black',width=2)
        for c in range(5): draw.line((x0+c*cellw,y0,x0+c*cellw,y0+4*cellh),fill='black',width=2)
        marker='END-'+name.upper()+'-COMPLETE'; draw.text((x0,int(h*.85)),marker,font=font,fill='black')
        path=root/(name+'.png'); img.save(path,compress_level=9)
        fixtures.append(dict(id=name,image=path.name,width=w,height=h,mode='RGB',image_sha256=digest(path.read_bytes()),pixels_sha256=digest(img.tobytes()),expected=dict(table=table,numbers=['12.50','17','203.25','7.25','29','118.75','19.75','46','322.00'],text=['REFERENCE '+name.upper()],end_markers=[marker],ordered_text=['Synthetic OCR measurement','REFERENCE '+name.upper(),marker],required_labels=['Table'])))
    manifest=dict(schema='chandra-synthetic-corpus-v1',split='development',generator_sha256=digest(Path(__file__).read_bytes()),font_sha256=digest(font_bytes),font_name=Path(a.font).name,pillow_version=__import__('PIL').__version__,fixtures=fixtures)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
