"""Paired fixed-Test analysis and per-seed group bootstrap (no GPU required)."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics
import numpy as np
from .common import CONDITIONS,SEEDS,check_bundle,digest,read_json,require,source_hashes,write_csv,write_json
from .metrics import summarize,choose_threshold,count_metrics

METRICS=('mAP50_95','mAP50','MAE','Bias','FP_per_image','FN_per_image')


def load_records(project,bundle,plan,bhash,condition,seed,split,registry):
    run=f'c{condition}_s{seed}';entry=registry.get(run,{})
    require(entry.get('status')=='completed',f'Training incomplete: {run}')
    checkpoint=project/entry['directory']/'weights/last.pt'
    require(digest(checkpoint)==entry['checkpoint_sha256'],'Checkpoint changed')
    folder=project/'predictions'/run/split
    meta=read_json(folder/'summary.json')
    require(meta['status']=='completed' and meta['bundle_sha256']==bhash and
            meta['checkpoint_sha256']==entry['checkpoint_sha256'] and
            meta['pipeline_sources']==source_hashes() and meta['records_sha256']==digest(folder/'records.jsonl'),
            f'Export provenance mismatch: {run}/{split}')
    require(meta['split']==split and meta['condition']==condition and meta['seed']==seed,'Wrong export identity')
    require(meta['numpy_version']==np.__version__,'Use export NumPy version for confidence-tie reproducibility')
    require(meta['conf_filter']==.001 and meta['nms_iou']==.6 and meta['max_det']==300
            and meta['save_hybrid'] is False and meta['half'] is False,'Wrong inference configuration')
    rows=[json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup={r['image_id']:r for r in rows}
    expected=[r for r in plan['records'] if r['split']==split]
    require(len(lookup)==len(rows)==len(expected) and set(lookup)=={r['image_id'] for r in expected},'Missing/duplicate image export')
    ordered=[]
    for row in expected:
        r=lookup[row['image_id']]
        require(r['evaluation_status']=='completed' and r['gt_version']==plan['gt_version'] and
                r['group_id']==row['group_id'] and r['ab']==row['ab'] and
                r['condition']==condition and r['seed']==seed and r['split']==split,'Wrong record identity')
        pred=np.asarray(r['predictions'],float).reshape(-1,6)
        gt=np.asarray(r['gt'],float).reshape(-1,5)
        correct=np.asarray(r['correct'],bool).reshape(-1,10)
        require(len(correct)==len(pred) and np.isfinite(pred).all() and np.isfinite(gt).all(),'Invalid prediction arrays')
        require((pred[:,5]==0).all() and (gt[:,0]==0).all(),'Not single-class')
        ordered.append(r)
    return ordered,meta


def group_draw(records,rng,stratified):
    strata=defaultdict(lambda:defaultdict(list))
    for r in records:strata[r['ab'] if stratified else 'T'][r['group_id']].append(r['image_id'])
    sample=[];groups=[]
    for name,members in sorted(strata.items()):
        keys=sorted(members)
        drawn=rng.choice(keys,size=len(keys),replace=True).tolist()
        groups.extend(drawn)
        for g in drawn:sample.extend(members[g])
    return sample,groups


def compare(data,thresholds,whole_ids=None,ab_ids=None):
    original=list(data['0'])
    whole_ids=original if whole_ids is None else whole_ids
    ab_ids=original if ab_ids is None else ab_ids
    scores={}
    for c in CONDITIONS:
        for subset in ('T','A','B'):
            ids=whole_ids if subset=='T' else [i for i in ab_ids if data[c][i]['ab']==subset]
            scores[c,subset]=summarize([data[c][i] for i in ids],thresholds[c])
    results={}
    def difference(name,left,right):
        results[name]={m:scores[left][m]-scores[right][m] for m in METRICS}
    for c in ('50A','50B','100'):
        difference(f'{c}-0_T',(c,'T'),('0','T'))
    for c in ('50A','50B'):
        for s in ('A','B'):difference(f'{c}-0_{s}',(c,s),('0',s))
    difference('C_A',('50A','A'),('50B','A'))
    difference('C_B',('50B','B'),('50A','B'))
    for name,left,right in [('D_A','50A-0_A','50A-0_B'),('D_B','50B-0_B','50B-0_A')]:
        results[name]={m:results[left][m]-results[right][m] for m in METRICS}
    results['C_mean_equals_D_mean']={m:(results['C_A'][m]+results['C_B'][m])/2 for m in METRICS}
    return scores,results


def bootstrap(data,thresholds,repeats=2000,random_seed=20261124,sample_path=None):
    rng=np.random.default_rng(random_seed)
    records=list(data['0'].values())
    enough={s:len({r['group_id'] for r in records if s=='T' or r['ab']==s})>=2 for s in ('T','A','B')}
    distributions=defaultdict(lambda:defaultdict(list))
    writer=gzip.open(sample_path,'wt',encoding='utf-8') if sample_path else None
    try:
        for b in range(repeats):
            whole,wgroups=group_draw(records,rng,False)
            ab,abgroups=group_draw(records,rng,True)
            if writer:writer.write(json.dumps(dict(iteration=b,whole_groups=wgroups,ab_groups=abgroups))+'\n')
            _,contrasts=compare(data,thresholds,whole,ab)
            for name,values in contrasts.items():
                valid=enough['T'] if name.endswith('_T') else (enough['A'] if name.endswith('_A') and name!='D_A' else
                      enough['B'] if name.endswith('_B') and name!='D_B' else enough['A'] and enough['B'])
                if valid:
                    for metric,value in values.items():distributions[name][metric].append(value)
    finally:
        if writer:writer.close()
    return {name:{m:((float(np.percentile(v,2.5)),float(np.percentile(v,97.5))) if np.isfinite(v).all() else (None,None))
                  for m,v in values.items()}
            for name,values in distributions.items()}


def analyze(bundle,project,out,repeats=2000,pilot=False):
    bundle,project,out=map(lambda p:Path(p).resolve(),(bundle,project,out))
    plan,bhash=check_bundle(bundle);config=read_json(project/'config.json')
    require(config['bundle_sha256']==bhash and config['pipeline_sources']==source_hashes(),'Project inputs/code changed')
    require(not out.exists(),'Analysis output exists; choose a new output directory')
    if pilot:
        require(config['pilot'] is True,'--pilot requires a pilot project')
    else:require(config['pilot'] is False and config['epochs']==30 and repeats==2000,'Main analysis requires 30 epochs and 2000 resamples')
    registry=read_json(project/'registry.json')
    loaded={};metadata=[]
    pairs=[('0',101)] if pilot else [(c,s) for s in SEEDS for c in CONDITIONS]
    for c,s in pairs:
        for split in ('val','test'):
            rows,meta=load_records(project,bundle,plan,bhash,c,s,split,registry)
            loaded[c,s,split]=rows;metadata.append(meta)
    # All models must use identical GT, not merely the same GT version string.
    for split in ('val','test'):
        ref=loaded['0',101,split]
        for c,s in pairs:
            require([r['gt'] for r in loaded[c,s,split]]==[r['gt'] for r in ref],'GT differs between models')
    out.mkdir(parents=True)
    threshold_rows=[];image_rows=[];score_rows=[];contrast_rows=[]
    for seed in ((101,) if pilot else SEEDS):
        conditions=('0',) if pilot else CONDITIONS
        chosen={}
        for c in conditions:
            threshold,grid=choose_threshold(loaded[c,seed,'val']);chosen[c]=threshold
            threshold_rows.extend(dict(condition=c,seed=seed,threshold=t,val_MAE=mae,selected=t==threshold) for t,mae in grid)
        for mode in ('baseline_common','model_specific'):
            thresholds={c:chosen['0'] if mode=='baseline_common' else chosen[c] for c in conditions}
            data={c:{r['image_id']:r for r in loaded[c,seed,'test']} for c in conditions}
            for c in conditions:
                for subset in ('T','A','B'):
                    rows=[r for r in data[c].values() if subset=='T' or r['ab']==subset]
                    scores=summarize(rows,thresholds[c])
                    score_rows.append(dict(mode=mode,condition=c,seed=seed,subset=subset,threshold=thresholds[c],**scores))
                for r in data[c].values():
                    exposed=c=='100' or c=='50'+r['ab']
                    image_rows.append(dict(mode=mode,condition=c,seed=seed,image_id=r['image_id'],group_id=r['group_id'],
                                           ab=r['ab'],exposed=exposed,evaluation_status='completed',gt_version=r['gt_version'],
                                           threshold=thresholds[c],**count_metrics(r,thresholds[c])))
            if not pilot:
                _,contrasts=compare(data,thresholds)
                intervals=bootstrap(data,thresholds,repeats,sample_path=out/f'samples_{mode}_s{seed}.jsonl.gz')
                for name,values in contrasts.items():
                    for metric,value in values.items():
                        ci=intervals.get(name,{}).get(metric)
                        factor=100 if metric.startswith('mAP') else 1
                        valid_ci=ci is not None and ci[0] is not None
                        contrast_rows.append(dict(mode=mode,seed=seed,comparison=name,metric=metric,
                            unit='percentage_points' if factor==100 else 'particles_per_image',
                            difference=value*factor,ci_low=ci[0]*factor if valid_ci else '',
                            ci_high=ci[1]*factor if valid_ci else '',
                            ci_status='reported' if valid_ci else 'not_reported_undefined_AP_draw' if ci else 'not_reported_less_than_two_groups'))
    write_csv(out/'threshold_grid.csv',threshold_rows)
    write_csv(out/'per_image.csv',image_rows)
    write_csv(out/'scores.csv',score_rows)
    if contrast_rows:
        write_csv(out/'contrasts.csv',contrast_rows)
        groups=defaultdict(list)
        for r in contrast_rows:groups[r['mode'],r['comparison'],r['metric'],r['unit']].append(r['difference'])
        write_csv(out/'seed_summary.csv',[dict(mode=k[0],comparison=k[1],metric=k[2],unit=k[3],seeds=len(v),
                  mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),seed101=v[0],seed102=v[1],seed103=v[2])
                  for k,v in groups.items()])
    write_json(out/'analysis.json',dict(status='completed',is_pilot=pilot,bundle_sha256=bhash,
               project_config=config,numpy_version=np.__version__,artifacts={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},
               config_sha256=digest(project/'config.json'),pipeline_sources=source_hashes(),
               repeats=0 if pilot else repeats,random_seed=20261124,ci='per-seed paired group percentile 95%',
               bootstrap_whole='unstratified groups',bootstrap_ab='stratified within A/B groups',
               saturated_exports=[dict(condition=m['condition'],seed=m['seed'],split=m['split'],images=m['saturated_images'])
                                  for m in metadata if m['saturated_images']],
               notes=['mAP raw scores use 0-1; mAP differences use percentage points.',
                      'C mean equals D mean; not independent evidence.',
                      'Intervals condition on fixed models and thresholds, not training/selection uncertainty.']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();analyze(a.bundle,a.project,a.out,pilot=a.pilot)


if __name__=='__main__':main()
