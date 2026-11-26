"""Build immutable four-condition inputs from human-reviewed CSV and review evidence."""
import argparse
import shutil
import tempfile
from pathlib import Path
from PIL import Image, ImageOps
from .common import (CONDITIONS, boxes, digest, read_csv, read_json, require, write_csv, write_json)

FIELDS = ('image_id', 'group_id', 'split', 'ab', 'control_id', 'image_path', 'label_path',
          'image_sha256', 'label_sha256', 'review_status', 'reviewer')
GATES = ('groups_reviewed', 'gt_reviewed', 'split_reviewed', 'control_reviewed', 'train_known_errors_resolved')


def prepare(manifest, review, out):
    manifest, review, out = map(lambda p: Path(p).resolve(), (manifest, review, out))
    require(not out.exists(), 'Output exists; use a new bundle version')
    rows = read_csv(manifest)
    require(rows and set(FIELDS).issubset(rows[0]), f'CSV needs fields: {FIELDS}')
    ids = {r['image_id']: r for r in rows}
    require(len(ids) == len(rows) and all(ids), 'Duplicate/empty image_id')
    gate = read_json(review)
    require(gate.get('manifest_sha256') == digest(manifest), 'Review manifest hash mismatch')
    for key in GATES:
        item = gate.get(key, {})
        require(item.get('approved') is True and str(item.get('reviewer', '')).strip(), f'Not reviewed: {key}')
        evidence = (review.parent/str(item.get('evidence', ''))).resolve()
        require(evidence.is_file(), f'Missing review evidence: {key}')
    group_splits, group_ab, hashes, paths = {}, {}, {}, set()
    for r in rows:
        require(r['split'] in ('train', 'val', 'test') and r['group_id'].strip(), 'Invalid split/group')
        g = r['group_id']
        require(group_splits.setdefault(g, r['split']) == r['split'], f'Group crosses split: {g}')
        if r['split'] == 'test':
            require(r['ab'] in ('A', 'B'), f'Test needs A/B: {r["image_id"]}')
            require(group_ab.setdefault(g, r['ab']) == r['ab'], f'Group crosses A/B: {g}')
        else:
            require(not r['ab'] and not r['control_id'], 'Only Test rows have ab/control_id')
        for field in ('image_path', 'label_path'):
            p = (manifest.parent/r[field]).resolve()
            require(p.is_file() and digest(p) == r[field.replace('_path', '_sha256')], f'Hash mismatch: {p}')
            r[field] = str(p)
        require(r['image_path'] not in paths, 'Same image path repeated')
        paths.add(r['image_path'])
        # Byte-identical originals must not cross the reviewed splits.
        require(hashes.setdefault(r['image_sha256'], r['split']) == r['split'], 'Exact image crosses splits')
        r['_boxes'] = boxes(r['label_path'])
        with Image.open(r['image_path']) as im:
            im.load()
            require(im.getexif().get(274, 1) == 1, 'Normalize EXIF orientation before review')
            r['_size'] = list(im.size)
    test = [r for r in rows if r['split'] == 'test']
    val = [r for r in rows if r['split'] == 'val']
    require(test and val and {r['ab'] for r in test} == {'A', 'B'}, 'Need Val and Test A/B')
    controls = [r['control_id'] for r in test]
    require(len(set(controls)) == len(test), 'A control is used more than once')
    require(all(c in ids and ids[c]['split'] == 'train' for c in controls), 'Control must be independent Train')
    for r in rows:
        if r['split'] in ('val', 'test') or r['image_id'] in controls:
            require(r['review_status'] == 'reviewed' and r['reviewer'].strip(), f'GT not reviewed: {r["image_id"]}')
    require(sum(len(r['_boxes']) for r in test) and sum(len(r['_boxes']) for r in val), 'No GT for AP')
    common = [r for r in rows if r['split'] == 'train' and r['image_id'] not in controls]
    require(common, 'No common training images left')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)/'bundle'
        (stage/'images').mkdir(parents=True)
        (stage/'labels').mkdir()
        for i, r in enumerate(sorted(rows, key=lambda x: x['image_id'])):
            alias = f'im_{i:05d}'
            image = 'images/'+alias+Path(r['image_path']).suffix.lower()
            label = 'labels/'+alias+'.txt'
            shutil.copyfile(r['image_path'], stage/image)
            shutil.copyfile(r['label_path'], stage/label)
            require(digest(stage/image) == r['image_sha256'] and digest(stage/label) == r['label_sha256'],
                    'Source changed during copy')
            r['image'], r['label'] = image, label
        # Both control and exposed slot images use exactly the same PNG/flip processing.
        for r in rows:
            if r['split'] == 'test' or r['image_id'] in controls:
                alias = Path(r['image']).stem+'_flip'
                r['flip_image'] = 'images/'+alias+'.png'
                with Image.open(stage/r['image']) as im:
                    ImageOps.mirror(im.convert('RGB')).save(stage/r['flip_image'])
                flipped = [[c, 1-x, y, w, h] for c,x,y,w,h in r['_boxes']]
                (stage/'labels'/f'{alias}.txt').write_text(
                    ''.join(' '.join(f'{v:.12g}' for v in b)+'\n' for b in flipped), encoding='utf-8')
                require(len(boxes(stage/'labels'/f'{alias}.txt')) == len(r['_boxes']), 'Flip count mismatch')
        lists = stage/'lists'
        lists.mkdir()
        for split, selected in [('val', val), ('test', test)]:
            (lists/f'{split}.txt').write_text(''.join('./../'+r['image']+'\n' for r in selected), encoding='utf-8')
        mappings = []
        for condition in CONDITIONS:
            train = [r['image'] for r in common]
            for r in test:
                exposed = condition == '100' or condition == '50'+r['ab']
                source = r if exposed else ids[r['control_id']]
                train.append(source['flip_image'])
                mappings.append(dict(condition=condition, test_id=r['image_id'], group_id=r['group_id'],
                                     ab=r['ab'], control_id=r['control_id'], source_id=source['image_id'],
                                     exposed=exposed, image=source['flip_image']))
            require(len(train) == len(set(train)), 'Repeated training file')
            (lists/f'train_{condition}.txt').write_text(''.join('./../'+p+'\n' for p in train), encoding='utf-8')
        write_csv(stage/'slot_mapping.csv', mappings)
        records = [{k: r[k] for k in ('image_id','group_id','split','ab','control_id','image','label','image_sha256',
                                     'label_sha256','review_status','reviewer')} | {'size':r['_size']} for r in rows]
        write_json(stage/'plan.json', dict(schema_version=1, conditions=list(CONDITIONS), records=records,
                   train_count=len(common)+len(test), gt_version=digest(manifest), review_sha256=digest(review),
                   sizes={s:sum(r['split']==s for r in rows) for s in ('train','val','test')},
                   exposure_counts={c:sum(m['condition']==c and m['exposed'] for m in mappings) for c in CONDITIONS}))
        shutil.copyfile(manifest, stage/'reviewed_manifest.csv')
        shutil.copyfile(review, stage/'review_approvals.json')
        for key in GATES:
            p=(review.parent/gate[key]['evidence']).resolve()
            dest=stage/'evidence'/f'{key}{p.suffix}'
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(p,dest)
        write_json(stage/'bundle.lock.json', {'files':{p.relative_to(stage).as_posix():digest(p)
                   for p in sorted(stage.rglob('*')) if p.is_file()}})
        stage.rename(out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(prepare(a.manifest,a.review,a.out))


if __name__=='__main__':
    main()
