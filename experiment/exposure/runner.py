"""Train four exposure conditions and export audited Val/Test predictions."""
import argparse
from contextlib import contextmanager
import csv
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from .common import (CONDITIONS, SEEDS, check_bundle, check_yolo, digest, load_settings,
                     read_json, require, source_hashes, write_json)


@contextmanager
def project_lock(project):
    project.mkdir(parents=True,exist_ok=True)
    path=project/'.pipeline.lock'
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,f'pid={os.getpid()}\n'.encode());os.close(fd)
    try: yield
    finally: path.unlink()


def execute(command, log, cwd):
    print(subprocess.list2cmdline(list(map(str,command))),flush=True)
    with Path(log).open('w',encoding='utf-8') as stream:
        subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,cwd=cwd,check=True)


def validate_training(directory, epochs):
    opt=load_settings(directory/'opt.yaml');hyp=load_settings(directory/'hyp.yaml')
    expected=dict(epochs=epochs,batch_size=4,imgsz=640,noautoanchor=True,patience=0)
    for key,value in expected.items(): require(opt.get(key)==value,f'Training option mismatch: {key}')
    require(hyp.get('fliplr')==0,'Stored fliplr is not zero')
    with (directory/'results.csv').open(newline='') as stream: rows=list(csv.DictReader(stream))
    require(len(rows)==epochs and int({k.strip():v for k,v in rows[-1].items()}['epoch'])==epochs-1,
            'Training did not complete the requested epochs')
    require(all(math.isfinite(float(v)) for r in rows for v in r.values()),'Non-finite training metrics')
    require((directory/'weights/last.pt').is_file(),'Missing last.pt')


def validate_pilot(report,config):
    require(report is not None,'Main training requires --pilot-report from successful pilot analysis')
    report=Path(report).resolve();meta=read_json(report)
    require(meta.get('status')=='completed' and meta.get('is_pilot') is True,'Not a completed pilot report')
    previous=meta['project_config']
    for key in ('bundle_sha256','yolo_commit','weights_sha256','pipeline_sources','batch','imgsz',
                'base_hyp_sha256','anchors_sha256','device','workers'):
        require(previous[key]==config[key],f'Pilot no longer matches: {key}')
    for path,sha in meta['artifacts'].items():
        require(digest(report.parent/path)==sha,'Pilot analysis output changed')


def train(a):
    plan,bhash=check_bundle(a.bundle)
    commit=check_yolo(a.yolov5,a.weights)
    epochs=1 if a.pilot else 30
    project=a.project.resolve();yolo=a.yolov5.resolve();bundle=a.bundle.resolve()
    config=dict(bundle_sha256=bhash,yolo_commit=commit,weights_sha256=digest(a.weights),
                pipeline_sources=source_hashes(),pilot=a.pilot,epochs=epochs,batch=4,imgsz=640,workers=a.workers,
                device=a.device,anchors_sha256=digest(yolo/'models/yolov5s.yaml'),
                base_hyp_sha256=digest(yolo/'data/hyps/hyp.scratch-low.yaml'))
    if not a.pilot:validate_pilot(a.pilot_report,config)
    with project_lock(project):
        if (project/'config.json').exists():require(read_json(project/'config.json')==config,'Project configuration changed; use a new project')
        else: write_json(project/'config.json',config)
        registry=read_json(project/'registry.json') if (project/'registry.json').exists() else {}
        hyp=load_settings(yolo/'data/hyps/hyp.scratch-low.yaml');hyp['fliplr']=0.0
        # JSON is valid YAML; .yaml suffix is required by YOLO's check_yaml.
        write_json(project/'hyp.yaml',hyp)
        selected=[(c,s) for s in (a.seed or SEEDS) for c in (a.condition or CONDITIONS)]
        if a.pilot:selected=[('0',101)]
        for condition,seed in selected:
            check_bundle(bundle)
            run=f'c{condition}_s{seed}'
            previous=registry.get(run)
            if previous and previous['status']=='completed':
                directory=project/previous['directory']
                validate_training(directory,epochs)
                require(digest(directory/'weights/last.pt')==previous['checkpoint_sha256'],'Checkpoint changed')
                print(f'SKIP completed {run}',flush=True);continue
            if previous:require(a.retry_failed,f'{run} failed/started; inspect log then use --retry-failed')
            attempt=(previous['attempt']+1) if previous else 1
            parent=project/'train'/run;parent.mkdir(parents=True,exist_ok=True)
            attempt_name=f'attempt_{attempt:03d}'
            directory=parent/attempt_name
            require(not directory.exists(),'Attempt directory exists')
            data=project/f'data_{condition}.json'
            write_json(data,dict(train=str(bundle/f'lists/train_{condition}.txt'),val=str(bundle/'lists/val.txt'),
                                 test=str(bundle/'lists/test.txt'),nc=1,names=['Microplastic']))
            entry=dict(status='started',attempt=attempt,directory=directory.relative_to(project).as_posix(),
                       condition=condition,seed=seed,started=datetime.datetime.now(datetime.timezone.utc).isoformat())
            registry[run]=entry;write_json(project/'registry.json',registry)
            command=[sys.executable,'-u',yolo/'train.py','--data',data,'--weights',a.weights.resolve(),
                     '--cfg',yolo/'models/yolov5s.yaml','--hyp',project/'hyp.yaml','--epochs',str(epochs),
                     '--batch-size','4','--imgsz','640','--workers',str(a.workers),'--device',a.device,
                     '--seed',str(seed),'--noautoanchor','--patience','0','--project',parent,'--name',attempt_name]
            start=datetime.datetime.now(datetime.timezone.utc)
            try:
                execute(command,parent/f'{attempt_name}.log',yolo)
                validate_training(directory,epochs)
                opt=load_settings(directory/'opt.yaml')
                require(opt.get('seed')==seed and not opt.get('resume'),'Wrong seed or resumed model')
                check_bundle(bundle)
                entry.update(status='completed',checkpoint_sha256=digest(directory/'weights/last.pt'))
            except BaseException as exc:
                entry.update(status='failed',error=str(exc));raise
            finally:
                entry['elapsed_seconds']=(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds()
                write_json(parent/f'{attempt_name}.json',entry)
                write_json(project/'registry.json',registry)


def predict(a):
    plan,bhash=check_bundle(a.bundle);yolo=a.yolov5.resolve();project=a.project.resolve()
    check_yolo(yolo)
    with project_lock(project):
        config=read_json(project/'config.json')
        require(config['bundle_sha256']==bhash and config['pipeline_sources']==source_hashes(),'Inputs/code changed')
        registry=read_json(project/'registry.json')
        selected=[(c,s) for s in (a.seed or SEEDS) for c in (a.condition or CONDITIONS)]
        if config['pilot']:selected=[('0',101)]
        for condition,seed in selected:
            run=f'c{condition}_s{seed}';entry=registry.get(run,{})
            require(entry.get('status')=='completed',f'Training incomplete: {run}')
            checkpoint=project/entry['directory']/'weights/last.pt'
            require(digest(checkpoint)==entry['checkpoint_sha256'],'Checkpoint changed')
            for split in ('val','test'):
                dest=project/'predictions'/run/split
                if (dest/'summary.json').exists():
                    meta=read_json(dest/'summary.json')
                    require(meta['bundle_sha256']==bhash and meta['checkpoint_sha256']==digest(checkpoint)
                            and meta['records_sha256']==digest(dest/'records.jsonl'),'Prediction archive changed')
                    print(f'SKIP completed {run}/{split}');continue
                if dest.exists():
                    require(a.retry_failed,f'Partial prediction output: {dest}; inspect then use --retry-failed')
                    # Preserve failed export; never reuse its records as completed.
                    archive=dest.with_name(split+'_failed_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
                    dest.rename(archive)
                dest.mkdir(parents=True)
                command=[sys.executable,'-u',Path(__file__).with_name('worker.py'),'--yolov5',yolo,
                         '--bundle',a.bundle.resolve(),'--checkpoint',checkpoint,'--split',split,'--out',dest,
                         '--condition',condition,'--seed',str(seed),'--device',a.device,'--workers',str(a.workers)]
                execute(command,dest/'worker.log',yolo)
                require((dest/'summary.json').is_file(),'Worker did not certify completion')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--check',action='store_true',help='Check sealed input; no training')
    p.add_argument('--predict',action='store_true',help='Export Val and Test for completed runs')
    p.add_argument('--yolov5',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--project',type=Path)
    p.add_argument('--condition',choices=CONDITIONS,action='append')
    p.add_argument('--seed',type=int,choices=SEEDS,action='append')
    p.add_argument('--device',default='0')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--pilot',action='store_true',help='Separate 1-epoch zero-exposure/101 project; never counted as main')
    p.add_argument('--pilot-report',type=Path,help='Completed pilot analysis.json, required before main training')
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    require(a.workers>=0,'Invalid workers')
    if a.check:
        plan,sha=check_bundle(a.bundle)
        print(json.dumps({'bundle_sha256':sha,'train_count':plan['train_count'],'sizes':plan['sizes'],
                          'exposure_counts':plan['exposure_counts']},indent=2));return
    require(a.yolov5 and a.project,'Need --yolov5 and --project')
    if a.predict:predict(a)
    else:
        require(a.weights,'Need --weights')
        train(a)


if __name__=='__main__':main()
