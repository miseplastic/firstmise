"""Create reproducible image/group splits from audited data and reviewed groups."""
import argparse
import hashlib
import random
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from common import read_csv, write_csv, write_json


def partition(items, seed):
    if len(items)<3: raise ValueError('At least 3 independent items/groups are required.')
    items=list(items); random.Random(seed).shuffle(items)
    n=len(items); nval=max(1,round(.15*n)); ntest=max(1,round(.15*n))
    return dict(train=items[:n-nval-ntest], val=items[n-nval-ntest:n-ntest], test=items[n-ntest:])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',required=True)
    p.add_argument('--groups',required=True)
    p.add_argument('--classes',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--review-confirmed',action='store_true')
    a=p.parse_args()
    if not a.review_confirmed:
        raise SystemExit('Human review required: resolve label issues, verify groups, and review candidates before --review-confirmed.')
    audit=Path(a.audit); out=Path(a.out).resolve()
    if out.exists(): raise SystemExit('Output already exists; choose a new --out folder.')
    issues=read_csv(audit/'issues.csv')
    if any(r['severity']=='ERROR' for r in issues): raise SystemExit('Audit has errors. Fix the working copy and rerun audit first.')
    rows=read_csv(audit/'images.csv'); group_rows=read_csv(a.groups)
    if not rows: raise SystemExit('Empty images.csv.')
    rows.sort(key=lambda row: row['image_id'])
    ids=[r['image_id'] for r in rows]
    if len(set(ids))!=len(ids): raise SystemExit('Duplicate image IDs.')
    if len(group_rows)!=len(rows) or {r['image_id'] for r in group_rows}!=set(ids):
        raise SystemExit('groups.csv must contain every image exactly once, no missing or extra IDs.')
    if len({r['image_id'] for r in group_rows})!=len(group_rows): raise SystemExit('Duplicate group records.')
    if any(not all(r.get(k,'').strip() for k in ('group_id','reviewer','evidence')) for r in group_rows):
        raise SystemExit('Every group record needs group_id, reviewer and evidence (including reviewed singletons).')
    groups={r['image_id']:r['group_id'] for r in group_rows}
    if 'G_ISOLATE_PENDING_POOL' in groups.values():
        raise SystemExit('Resolve pending group records before generating R/G splits.')
    by_hash=defaultdict(set)
    for r in rows: by_hash[r['pixel_sha256']].add(groups[r['image_id']])
    if any(len(v)>1 for v in by_hash.values()): raise SystemExit('Identical pixel images assigned to different groups.')
    classes=Path(a.classes).read_text(encoding='utf-8-sig').splitlines()
    nclasses=len(classes)
    if not classes or any(not x.strip() for x in classes): raise SystemExit('Invalid classes.txt.')
    if any(len(r['class_counts'].split(';'))!=nclasses for r in rows): raise SystemExit('Class list differs from audit; audit again.')
    for r in rows:
        if not Path(r['image_path']).exists(): raise SystemExit('Image paths no longer exist. Rerun audit on target computer/server.')
        r['group_id']=groups[r['image_id']]
    by_group=defaultdict(list)
    for r in rows: by_group[r['group_id']].append(r)
    boxes=defaultdict(list)
    for b in read_csv(audit/'boxes.csv'): boxes[b['image_id']].append(b)
    manifests=[]; summaries=[]; drops=[]
    split_boxes=[]
    for repeat,seed in enumerate((11,22,33),1):
        rparts=partition(rows,seed)
        gkeys=partition(sorted(by_group),seed)
        gparts={s:[r for g in gs for r in by_group[g]] for s,gs in gkeys.items()}
        target=min(len(rparts['train']),len(gparts['train']))
        if target==0: raise SystemExit('Empty training set.')
        for cond,parts in [('R',rparts),('G',gparts)]:
            if len(parts['train'])>target:
                candidates=list(parts['train']); random.Random(9000+seed).shuffle(candidates)
                parts['train']=candidates[:target]
                for r in candidates[target:]: drops.append(dict(condition=cond,repeat=repeat,image_id=r['image_id'],reason='match train image count'))
            dest=out/f'{cond}_{repeat}'; dest.mkdir(parents=True)
            for split,items in parts.items():
                if not items: raise SystemExit('Empty split.')
                paths = [
                    './' + Path(os.path.relpath(r['image_path'], dest)).as_posix()
                    for r in items
                ]
                (dest / f'{split}.txt').write_text('\n'.join(paths) + '\n', encoding='utf-8')
                counts=[sum(int(r['class_counts'].split(';')[i]) for r in items) for i in range(nclasses)]
                areas=[float(b['area_fraction']) for r in items for b in boxes[r['image_id']]]
                summaries.append(dict(condition=cond,repeat=repeat,split=split,images=len(items),groups=len({r['group_id'] for r in items}),
                    objects=sum(int(r['object_count']) for r in items),class_counts=';'.join(map(str,counts)),missing_classes=';'.join(str(i) for i,n in enumerate(counts) if n==0),
                    box_area_median=statistics.median(areas) if areas else ''))
                for r in items:
                    for b in boxes[r['image_id']]:split_boxes.append(dict(condition=cond,repeat=repeat,split=split,**b))
                for r in items:
                    manifests.append(dict(image_id=r['image_id'],group_id=r['group_id'],condition=cond,repeat=repeat,split=split,split_seed=seed,train_seed=100+repeat))
            if cond=='G':
                sets=[{r['group_id'] for r in parts[s]} for s in ('train','val','test')]
                assert not any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
            # run_training.py anchors this portable config to its own folder for YOLOv5.
            config = {split: f'{split}.txt' for split in ('train', 'val', 'test')}
            config['names'] = classes
            (dest / 'data.yaml').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    write_csv(out/'manifest.csv',manifests,['image_id','group_id','condition','repeat','split','split_seed','train_seed'])
    write_csv(out/'distribution.csv',summaries,['condition','repeat','split','images','groups','objects','class_counts','missing_classes','box_area_median'])
    write_csv(out/'box_sizes.csv',split_boxes,['condition','repeat','split','image_id','class_id','width_fraction','height_fraction','area_fraction'])
    write_csv(out/'excluded_train.csv',drops,['condition','repeat','image_id','reason'])
    write_json(out/'review_gate.json',dict(status='SPLITS_GENERATED',
        grouping_sha256=hashlib.sha256(Path(a.groups).read_bytes()).hexdigest(),
        is_demo=any(r['is_demo']=='true' for r in rows),
        note='Inspect distribution.csv and box_sizes.csv. R/G test sets differ, so score differences also reflect test-set composition.'))
    print(f'Saved six R/G splits to {out}.')


if __name__=='__main__': main()
