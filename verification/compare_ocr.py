"""Separate exact OCR content/structure changes from normalized box geometry."""
import argparse
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re

class Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks=[]
        self.stack=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="div":
            node=None
            if "data-bbox" in attrs:
                node={"label":attrs.get("data-label"),"box":[float(x) for x in attrs["data-bbox"].split()],"text":""}
                self.blocks.append(node)
            self.stack.append(node)
    def handle_data(self,data):
        for node in self.stack:
            if node is not None:
                node["text"] += data
    def handle_endtag(self,tag):
        if tag=="div" and self.stack:
            self.stack.pop()

def text(path):
    raw=Path(path).read_text()
    try:
        return json.loads(raw)["choices"][0]["message"]["content"]
    except (json.JSONDecodeError,KeyError,TypeError):
        return raw

def compare(reference,candidate):
    source=Path(__file__).parents[1]/"benchmarks"/"evaluate.py"
    spec=importlib.util.spec_from_file_location("chandra_evaluation",source)
    evaluation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    a,b=text(reference),text(candidate)
    ap,bp=Blocks(),Blocks()
    ap.feed(a); bp.feed(b)
    at,bt=evaluation.Tables(),evaluation.Tables()
    at.feed(a);bt.feed(b)
    boxes=[]
    for i,(x,y) in enumerate(zip(ap.blocks,bp.blocks)):
        xb,yb=x["box"],y["box"]
        intersect=max(0,min(xb[2],yb[2])-max(xb[0],yb[0]))*max(0,min(xb[3],yb[3])-max(xb[1],yb[1]))
        union=(xb[2]-xb[0])*(xb[3]-xb[1])+(yb[2]-yb[0])*(yb[3]-yb[1])-intersect
        boxes.append({"block":i,"label_equal":x["label"]==y["label"],"text_equal":x["text"]==y["text"],"reference_box":xb,"candidate_box":yb,"coordinate_delta":[v-u for u,v in zip(xb,yb)],"max_coordinate_delta_1000":max(abs(v-u) for u,v in zip(xb,yb)),"iou":intersect/union if union>0 else None})
    without_boxes=lambda s:re.sub(r'data-bbox="[^"]*"','data-bbox=""',s)
    return {"exact_html_equal":a==b,"html_except_boxes_equal":without_boxes(a)==without_boxes(b),"block_counts":[len(ap.blocks),len(bp.blocks)],"block_text_and_labels_equal":len(ap.blocks)==len(bp.blocks) and all(x["text"]==y["text"] and x["label"]==y["label"] for x,y in zip(ap.blocks,bp.blocks)),"table_rows_equal":at.rows==bt.rows,"reference_table_rows":at.rows,"candidate_table_rows":bt.rows,"equations_equal":evaluation.equation_representations(a)==evaluation.equation_representations(b),"boxes":boxes,"max_coordinate_delta_1000":max((x["max_coordinate_delta_1000"] for x in boxes),default=None),"minimum_box_iou":min((x["iou"] for x in boxes if x["iou"] is not None),default=None),"scope":"ordered block comparison; reports measurements, no implicit geometry tolerance"}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("reference");p.add_argument("candidate");p.add_argument("--output")
    a=p.parse_args();result=json.dumps(compare(a.reference,a.candidate),indent=2)
    if a.output:Path(a.output).write_text(result)
    else:print(result)
