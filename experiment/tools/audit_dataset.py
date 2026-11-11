"""Read-only YOLO image/label audit; output is written to a separate directory."""
import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from common import write_csv, write_json


def dhash(im):
    data = list(im.convert('L').resize((9, 8)).getdata())
    h = 0
    for y in range(8):
        for x in range(8):
            h = (h << 1) | (data[y*9+x] > data[y*9+x+1])
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--overlays', type=int, default=20)
    p.add_argument('--near', action='store_true')
    p.add_argument('--distance', type=int, default=5)
    a = p.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    if out.exists():
        raise SystemExit('Output already exists. Choose a new --out directory.')
    if out == root or root in out.parents:
        raise SystemExit('Keep output outside the dataset root.')
    classes = (root/'classes.txt').read_text(encoding='utf-8-sig').splitlines()
    if not classes or any(not c.strip() for c in classes):
        raise SystemExit('classes.txt must contain one non-empty class name per line.')
    images = sorted(p for p in (root/'images').rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
    if not images:
        raise SystemExit('No images found under root/images.')
    out.mkdir(parents=True)
    rows, issues, candidates, parsed, box_rows = [], [], [], {}, []
    label_used = set()
    hashes = defaultdict(list)
    def issue(image_id, severity, code, detail):
        issues.append(dict(image_id=image_id, severity=severity, code=code, detail=detail))
    for image in images:
        rel = image.relative_to(root/'images')
        iid = rel.as_posix()
        label = (root/'labels'/rel).with_suffix('.txt')
        if label in label_used:
            issue(iid,'ERROR','LABEL_COLLISION','Multiple images map to the same label filename.')
        label_used.add(label)
        boxes, counts = [], [0]*len(classes)
        try:
            with Image.open(image) as source:
                source.load()
                if source.getexif().get(274, 1) != 1:
                    issue(iid,'ERROR','EXIF_ORIENTATION','Normalize orientation and verify labels before splitting.')
                im = source.convert('RGB')
            w,h = im.size
            pixel_hash = hashlib.sha256(f'{w}x{h}:'.encode()+im.tobytes()).hexdigest()
        except Exception as e:
            issue(iid,'ERROR','UNREADABLE_IMAGE',str(e)); continue
        if not label.exists():
            issue(iid,'ERROR','MISSING_LABEL','Confirm whether this is background or an annotation omission.')
        else:
            try:
                lines = label.read_text(encoding='utf-8-sig').splitlines()
            except Exception as e:
                issue(iid,'ERROR','UNREADABLE_LABEL',str(e)); lines=[]
            if not any(s.strip() for s in lines):
                issue(iid,'REVIEW','EMPTY_LABEL','A human must confirm intentional background.')
            for no,line in enumerate(lines,1):
                if not line.strip(): continue
                try:
                    vals = [float(v) for v in line.split()]
                    if len(vals)!=5 or not all(math.isfinite(v) for v in vals): raise ValueError('Expected 5 finite numbers.')
                    c,x,y,bw,bh = vals
                    if int(c)!=c or not 0<=int(c)<len(classes): raise ValueError('Invalid class ID.')
                    if not (0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1): raise ValueError('Invalid normalized coordinates.')
                    if min(x-bw/2,y-bh/2)<-1e-6 or max(x+bw/2,y+bh/2)>1+1e-6: raise ValueError('Box extends beyond image bounds.')
                    boxes.append((int(c),x,y,bw,bh)); counts[int(c)]+=1
                except Exception as e:
                    issue(iid,'ERROR','INVALID_LABEL',f'line {no}: {e}')
        row = dict(image_id=iid, image_path=str(image), label_path=str(label),
            original_split=rel.parts[0] if len(rel.parts)>1 else 'unspecified',
            width=w,height=h,object_count=len(boxes),class_counts=';'.join(map(str,counts)),
            pixel_sha256=pixel_hash,dhash=f'{dhash(im):016x}',
            flip_dhash=f'{dhash(ImageOps.mirror(im)):016x}',
            is_demo=str((root/'SYNTHETIC_DEMO.txt').exists()).lower())
        rows.append(row); hashes[pixel_hash].append(iid); parsed[iid]=boxes
        for c,x,y,bw,bh in boxes:
            box_rows.append(dict(image_id=iid,class_id=c,width_fraction=bw,height_fraction=bh,area_fraction=bw*bh))
    exact=[]
    for digest, ids in hashes.items():
        if len(ids)>1:
            for iid in ids:
                exact.append(dict(pixel_sha256=digest,image_id=iid))
    if a.near:
        if len(rows)>4000:
            raise SystemExit('Near search limited to 4000 readable images. Use a scalable method for larger data.')
        for i,left in enumerate(rows):
            for right in rows[i+1:]:
                if left['pixel_sha256']==right['pixel_sha256']: continue
                b=int(right['dhash'],16)
                dist=min((int(left['dhash'],16)^b).bit_count(),(int(left['flip_dhash'],16)^b).bit_count())
                if dist<=a.distance:
                    candidates.append(dict(image_a=left['image_id'], image_b=right['image_id'],
                        dhash_distance=dist, decision='',reviewer='',evidence=''))
    for label in sorted((root/'labels').rglob('*.txt')):
        if label not in label_used:
            issue(label.relative_to(root).as_posix(),'ERROR','ORPHAN_LABEL','No matching supported image.')
    overlay_dir=out/'overlays'; overlay_dir.mkdir()
    # Cycle across original splits rather than taking the first 20 paths.
    pools=defaultdict(list)
    for r in rows: pools[r['original_split']].append(r)
    selected=[]
    while len(selected)<min(a.overlays,len(rows)):
        for key in sorted(pools):
            if pools[key] and len(selected)<a.overlays: selected.append(pools[key].pop(0))
    for index,r in enumerate(selected):
        with Image.open(r['image_path']) as src: im=src.convert('RGB')
        d=ImageDraw.Draw(im)
        for c,x,y,bw,bh in parsed[r['image_id']]:
            w,h=im.size
            rect=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
            d.rectangle(rect,outline='red',width=2)
            d.text((rect[0],max(0,rect[1]-12)),str(c),fill='red')
        im.save(overlay_dir/f'{index:03d}.png')
        r['overlay_file']=f'overlays/{index:03d}.png'
    fields=['image_id','image_path','label_path','original_split','width','height','object_count','class_counts','pixel_sha256','dhash','flip_dhash','is_demo','overlay_file']
    write_csv(out/'images.csv',rows,fields)
    write_csv(out/'boxes.csv',box_rows,['image_id','class_id','width_fraction','height_fraction','area_fraction'])
    write_csv(out/'issues.csv',issues,['image_id','severity','code','detail'])
    write_csv(out/'exact_duplicates.csv',exact,['pixel_sha256','image_id'])
    write_csv(out/'near_candidates.csv',candidates,['image_a','image_b','dhash_distance','decision','reviewer','evidence'])
    write_csv(out/'groups_template.csv',[dict(image_id=r['image_id'],group_id='',reviewer='',evidence='') for r in rows],['image_id','group_id','reviewer','evidence'])
    write_json(out/'summary.json',dict(readable_images=len(rows),discovered_images=len(images),objects=sum(r['object_count'] for r in rows),
        error_count=sum(x['severity']=='ERROR' for x in issues),review_count=sum(x['severity']=='REVIEW' for x in issues),
        exact_duplicate_groups=sum(len(v)>1 for v in hashes.values()),near_candidates=len(candidates),
        note='dHash is only a candidate search; rotation/crop/compression duplicates may be missed. Group identity requires review.'))
    print(f'Audit saved to {out}. Review issues.csv and overlays; a clean script output does not prove labels are correct.')
    if any(x['severity']=='ERROR' for x in issues): raise SystemExit(2)


if __name__=='__main__': main()
