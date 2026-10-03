"""Export CPU-only pinned processor tensors for the frozen public corpus (no model load)."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

PROFILES = {'checkpoint-default': {}, 'northstone-serving': {'size': {'shortest_edge': 3136, 'longest_edge': 3145728}}}
PIN = 'af93b47dba1b47b6640c86ccf487ed2260ab9a09'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def export(model, corpus, output, manifest_path, profile="checkpoint-default"):
    if sys.byteorder != 'little':
        raise ValueError('This export format requires a little-endian host')
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    inventory = json.loads(manifest_path.read_text())
    if inventory['model'] != 'datalab-to/chandra-ocr-2' or inventory['revision'] != PIN:
        raise ValueError('Unexpected processor pin')
    # Verify every non-weight processor/config/tokenizer input before local loading.
    files = {}
    for record in inventory['files']:
        name = record['file']
        if name.endswith('.json') or name.endswith('.jinja'):
            path = (model / name).resolve()
            if not path.is_relative_to(model.resolve()):
                raise ValueError('Model path escape')
            data = path.read_bytes()
            if len(data) != record['size'] or sha(data) != record['sha256']:
                raise ValueError('Processor integrity mismatch: ' + name)
            files[name] = sha(data)
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    manifest_bytes = (corpus / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    prompt = Path(__file__).with_name('prompt.txt').read_text()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    seen = set()
    for fixture in manifest['fixtures']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', fixture['id']) or fixture['id'] in seen:
            raise ValueError('Invalid or duplicate fixture identifier')
        seen.add(fixture['id'])
        path = (corpus / fixture['image']).resolve()
        if not path.is_relative_to(corpus.resolve()):
            raise ValueError('Corpus path escape')
        encoded = path.read_bytes()
        if sha(encoded) != fixture['image_sha256']:
            raise ValueError('Image commitment mismatch')
        with Image.open(path) as source:
            image = source.convert('RGB')
        pixels = image.tobytes()
        if sha(pixels) != fixture['pixels_sha256'] or image.size != (fixture['width'], fixture['height']) or fixture['mode'] != 'RGB':
            raise ValueError('Pixel commitment mismatch')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[rendered], images=[image], return_tensors='pt', **PROFILES[profile])
        tensors = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
                raise ValueError('Expected CPU tensor: ' + key)
            contiguous = value.detach().contiguous()
            raw = contiguous.view(torch.uint8).numpy().tobytes()
            filename = fixture['id'] + '-' + key + '.raw'
            (output / filename).write_bytes(raw)
            tensors[key] = dict(file=filename, sha256=sha(raw), dtype=str(value.dtype), shape=list(value.shape), strides=list(value.stride()), bytes=len(raw), byte_order='little')
        ids = batch['input_ids'][0].tolist()
        image_token_id = json.loads((model / 'config.json').read_text())['image_token_id']
        records.append(dict(fixture_id=fixture['id'], encoded_sha256=sha(encoded), pixels_sha256=sha(pixels), dimensions=list(image.size), tensors=tensors, prompt_token_ids=ids, image_token_id=image_token_id, image_token_positions=[i for i, v in enumerate(ids) if v == image_token_id], image_grid_thw=batch['image_grid_thw'].tolist(), rendered_prompt=rendered, rendered_prompt_sha256=sha(rendered.encode())))
    report = dict(schema='chandra-processor-reference-v1', model=inventory['model'], revision=PIN, processor_files=files, processor_class=type(processor).__name__, image_processor_class=type(processor.image_processor).__name__, runtime={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'pillow', 'tokenizers')}, device='cpu', processor_profile=profile, processor_kwargs=PROFILES[profile], corpus_sha256=sha(manifest_bytes), prompt_sha256=sha(prompt.encode()), provenance_verified=False, evidence_scope='Processor only; no weights, logits, inference runtime or numerical model acceptance', fixtures=records)
    (output / 'processor-reference.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=PROFILES, default='checkpoint-default')
    p.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'provenance/model.json')
    a = p.parse_args()
    report = export(a.model, a.corpus, a.output, a.manifest, a.profile)
    print(json.dumps({'fixtures': len(report['fixtures']), 'device': report['device'], 'provenance_verified': False}))

if __name__ == '__main__':
    main()
