import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiment'))
from exposure.common import (CONDITIONS,SEEDS,YOLO_COMMIT,boxes,check_bundle,digest,read_csv,read_json,
                             source_hashes,write_csv,write_json)
from exposure.prepare import prepare,GATES,FIELDS
from exposure.metrics import average_precision,matches,count_metrics,choose_threshold
from exposure.analyze import bootstrap,compare,group_draw,analyze
from exposure import runner


def fixture(directory):
    rows=[]
    for i in range(16):
        image=directory/f'{i}.png';label=directory/f'{i}.txt'
        Image.new('RGB',(64,48),(i*13,i*7,i*3)).save(image)
        label.write_text('0 0.25 0.5 0.125 0.25\n')
        split='train' if i<8 else 'val' if i<10 else 'test'
        rows.append(dict(image_id=f'id{i}',group_id=f'g{i}',split=split,
                         ab=('A' if i<13 else 'B') if split=='test' else '',
                         control_id=f'id{i-10}' if split=='test' else '',image_path=image.name,label_path=label.name,
                         image_sha256=digest(image),label_sha256=digest(label),review_status='reviewed',reviewer='fixture'))
    manifest=directory/'manifest.csv';write_csv(manifest,rows,FIELDS)
    evidence=directory/'evidence.txt';evidence.write_text('SYNTHETIC TEST ONLY; NOT HUMAN APPROVAL')
    gate=dict(manifest_sha256=digest(manifest),**{k:dict(approved=True,reviewer='fixture',evidence=evidence.name) for k in GATES})
    review=directory/'review.json';write_json(review,gate)
    return manifest,review


def record(image_id='i',group='g',ab='A',predictions=None,gt=None):
    predictions=[[0,0,10,10,.9,0]] if predictions is None else predictions
    gt=[[0,0,0,10,10]] if gt is None else gt
    correct=np.column_stack([matches(predictions,gt,t) for t in np.linspace(.5,.95,10)]).tolist()
    return dict(image_id=image_id,group_id=group,ab=ab,gt=gt,predictions=predictions,correct=correct)


class MetricsTests(unittest.TestCase):
    def test_cli_help(self):
        root=Path(__file__).resolve().parents[1]
        for command in ([sys.executable,'experiment/run_training.py','--help'],
                        [sys.executable,'-m','experiment.exposure.prepare','--help'],
                        [sys.executable,'-m','experiment.exposure.analyze','--help']):
            completed=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)

    def test_perfect_and_zero(self):
        r=record();self.assertAlmostEqual(average_precision([r])[1],.995)
        self.assertEqual(count_metrics(r,.5)['abs_count_error'],0)
        r=record(predictions=[])
        self.assertEqual(average_precision([r]),(0.,0.))
        self.assertEqual(count_metrics(r,.5)['FN'],1)

    def test_false_positive_false_negative_cancellation(self):
        gt=[[0,i*20,0,i*20+10,10] for i in range(10)]
        pred=[[i*20,0,i*20+10,10,.9,0] for i in range(6)]
        pred += [[i*20,30,i*20+10,40,.8,0] for i in range(4)]
        result=count_metrics(record(predictions=pred,gt=gt),.5)
        self.assertEqual((result['TP'],result['FP'],result['FN'],result['abs_count_error']),(6,4,4,0))

    def test_duplicate_detections_one_to_one(self):
        r=record(predictions=[[0,0,10,10,.9,0],[0,0,10,10,.8,0]])
        v=count_metrics(r,.5)
        self.assertEqual((v['TP'],v['FP'],v['FN']),(1,1,0))

    def test_threshold_uses_lowest_tie(self):
        t,grid=choose_threshold([record()])
        self.assertEqual(t,.05);self.assertEqual(len(grid),19)

    def test_ap_does_not_average_image_ap(self):
        r=record();miss=record('j',predictions=[])
        self.assertNotAlmostEqual(average_precision([r,miss])[1],(average_precision([r])[1]+average_precision([miss])[1])/2)

    def test_groups_repeated_together(self):
        rows=[record('a','g1'),record('b','g1'),record('c','g2')]
        duplicate=False
        for seed in range(20):
            ids,groups=group_draw(rows,np.random.default_rng(seed),False)
            self.assertEqual(ids.count('a'),ids.count('b'))
            if groups==['g1','g1']:
                duplicate=True;self.assertEqual(ids.count('a'),2)
        self.assertTrue(duplicate)

    def test_contrasts_identity_and_bootstrap(self):
        records=[record(str(i),f'g{i}','A' if i<2 else 'B') for i in range(4)]
        data={c:{r['image_id']:dict(r) for r in records} for c in CONDITIONS}
        data['50B']['0']=record('0','g0','A',predictions=[])
        thresholds={c:.5 for c in CONDITIONS}
        _,values=compare(data,thresholds)
        for metric in values['D_A']:
            self.assertAlmostEqual((values['D_A'][metric]+values['D_B'][metric])/2,
                                   values['C_mean_equals_D_mean'][metric])
        one=bootstrap(data,thresholds,repeats=12)
        self.assertEqual(one,bootstrap(data,thresholds,repeats=12))
        self.assertIn('100-0_T',one)

    def test_bootstrap_omits_single_group_subset(self):
        rows=[record('a','one','A'),record('b','two','B')]
        data={c:{r['image_id']:dict(r) for r in rows} for c in CONDITIONS}
        ci=bootstrap(data,{c:.5 for c in CONDITIONS},repeats=5)
        self.assertIn('100-0_T',ci)
        self.assertNotIn('C_A',ci)
        self.assertNotIn('D_A',ci)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.manifest,self.review=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def build(self):return prepare(self.manifest,self.review,self.root/'bundle')
    def update(self,rows):
        write_csv(self.manifest,rows,FIELDS)
        gate=read_json(self.review);gate['manifest_sha256']=digest(self.manifest);write_json(self.review,gate)

    def test_all_four_conditions_and_flip(self):
        bundle=self.build();plan,_=check_bundle(bundle)
        mapping=read_csv(bundle/'slot_mapping.csv')
        for c,exposed in [('0',0),('50A',3),('50B',3),('100',6)]:
            rows=[r for r in mapping if r['condition']==c]
            self.assertEqual(sum(r['exposed']=='True' for r in rows),exposed)
            listing=(bundle/f'lists/train_{c}.txt').read_text().splitlines()
            self.assertEqual(len(listing),8)
            self.assertTrue(all(x.startswith('./') for x in listing))
            self.assertTrue(all((bundle/'lists'/x).resolve().is_file() for x in listing))
        flip=next((bundle/'labels').glob('*_flip.txt'))
        self.assertAlmostEqual(boxes(flip)[0][1],.75)
        common=set((bundle/'lists/train_0.txt').read_text().splitlines()) & set((bundle/'lists/train_100.txt').read_text().splitlines())
        self.assertEqual(len(common),2)  # No control originals left in common Train.

    def test_unapproved_and_hash_change_rejected(self):
        gate=read_json(self.review);gate['gt_reviewed']['approved']=False;write_json(self.review,gate)
        with self.assertRaisesRegex(ValueError,'Not reviewed'):self.build()
        gate['gt_reviewed']['approved']=True;write_json(self.review,gate)
        (self.root/'0.txt').write_text('')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.build()

    def test_group_cross_split_rejected(self):
        rows=read_csv(self.manifest);rows[0]['group_id']=rows[-1]['group_id'];self.update(rows)
        with self.assertRaisesRegex(ValueError,'Group crosses split'):self.build()

    def test_group_cross_ab_rejected(self):
        rows=read_csv(self.manifest);rows[10]['group_id']=rows[13]['group_id'];self.update(rows)
        with self.assertRaisesRegex(ValueError,'Group crosses A/B'):self.build()

    def test_control_reuse_rejected(self):
        rows=read_csv(self.manifest);rows[10]['control_id']=rows[11]['control_id'];self.update(rows)
        with self.assertRaisesRegex(ValueError,'control is used'):self.build()

    def test_bundle_change_rejected(self):
        b=self.build();(b/'lists/train_0.txt').write_text('')
        with self.assertRaisesRegex(ValueError,'Bundle changed'):check_bundle(b)

    def test_mocked_training_and_pilot_analysis(self):
        from argparse import Namespace
        b=self.build();plan,bhash=check_bundle(b)
        yolo=self.root/'yolo';(yolo/'models').mkdir(parents=True);(yolo/'data/hyps').mkdir(parents=True)
        write_json(yolo/'models/yolov5s.yaml',{'anchors':[[10,13]]})
        write_json(yolo/'data/hyps/hyp.scratch-low.yaml',{'fliplr':.5})
        weights=self.root/'initial.pt';weights.write_bytes(b'fake')
        project=self.root/'pilot'
        args=Namespace(bundle=b,yolov5=yolo,weights=weights,project=project,pilot=True,workers=0,device='cpu',seed=None,condition=None,retry_failed=False)
        def fake_execute(command,log,cwd):
            directory=Path(command[command.index('--project')+1])/command[command.index('--name')+1]
            (directory/'weights').mkdir(parents=True)
            write_json(directory/'opt.yaml',dict(epochs=1,batch_size=4,imgsz=640,noautoanchor=True,patience=0,seed=101,resume=False))
            write_json(directory/'hyp.yaml',{'fliplr':0})
            write_csv(directory/'results.csv',[{'epoch':0}]);(directory/'weights/last.pt').write_bytes(b'fake trained')
        with patch.object(runner,'check_yolo',return_value=YOLO_COMMIT),patch.object(runner,'execute',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):runner.train(args)
        self.assertEqual(read_json(project/'registry.json')['c0_s101']['status'],'failed')
        with patch.object(runner,'check_yolo',return_value=YOLO_COMMIT):
            with self.assertRaisesRegex(ValueError,'retry-failed'):runner.train(args)
        args.retry_failed=True
        with patch.object(runner,'check_yolo',return_value=YOLO_COMMIT),patch.object(runner,'execute',side_effect=fake_execute) as execute:
            runner.train(args);runner.train(args);self.assertEqual(execute.call_count,1)
        registry=read_json(project/'registry.json');entry=registry['c0_s101']
        self.assertEqual(entry['attempt'],2)
        self.assertTrue((project/'train/c0_s101/attempt_001.json').is_file())
        for split in ('val','test'):
            folder=project/'predictions/c0_s101'/split;folder.mkdir(parents=True)
            rows=[]
            for source in plan['records']:
                if source['split']!=split:continue
                r=record(source['image_id'],source['group_id'],source['ab'])
                r.update(condition='0',seed=101,split=split,evaluation_status='completed',gt_version=plan['gt_version'])
                rows.append(r)
            (folder/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write_json(folder/'summary.json',dict(status='completed',split=split,condition='0',seed=101,bundle_sha256=bhash,
                checkpoint_sha256=entry['checkpoint_sha256'],pipeline_sources=source_hashes(),
                records_sha256=digest(folder/'records.jsonl'),numpy_version=np.__version__,conf_filter=.001,nms_iou=.6,max_det=300,save_hybrid=False,half=False,saturated_images=[]))
        out=self.root/'analysis';analyze(b,project,out,pilot=True)
        self.assertEqual(len(read_csv(out/'scores.csv')),6)
        self.assertTrue(read_json(out/'analysis.json')['is_pilot'])
        config=read_json(project/'config.json')
        runner.validate_pilot(out/'analysis.json',config)
        with self.assertRaisesRegex(ValueError,'no longer matches'):
            runner.validate_pilot(out/'analysis.json',dict(config,workers=99))
        with self.assertRaisesRegex(ValueError,'Main analysis requires'):analyze(b,project,self.root/'bad')
        folder=project/'predictions/c0_s101/test';(folder/'records.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError,'provenance'):analyze(b,project,self.root/'tampered',pilot=True)

    def test_main_analysis_all_conditions_and_seeds(self):
        b=self.build();plan,bhash=check_bundle(b);project=self.root/'main';project.mkdir()
        write_json(project/'config.json',dict(bundle_sha256=bhash,pipeline_sources=source_hashes(),pilot=False,epochs=30))
        registry={}
        for seed in SEEDS:
            for c in CONDITIONS:
                run=f'c{c}_s{seed}';directory=project/'train'/run
                (directory/'weights').mkdir(parents=True)
                checkpoint=directory/'weights/last.pt';checkpoint.write_bytes(b'test')
                registry[run]=dict(status='completed',directory=directory.relative_to(project).as_posix(),checkpoint_sha256=digest(checkpoint))
                for split in ('val','test'):
                    folder=project/'predictions'/run/split;folder.mkdir(parents=True)
                    rows=[]
                    for source in plan['records']:
                        if source['split']!=split:continue
                        r=record(source['image_id'],source['group_id'],source['ab'],predictions=[] if c=='0' else None)
                        r.update(condition=c,seed=seed,split=split,evaluation_status='completed',gt_version=plan['gt_version'])
                        rows.append(r)
                    file=folder/'records.jsonl';file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    write_json(folder/'summary.json',dict(status='completed',split=split,condition=c,seed=seed,bundle_sha256=bhash,
                        checkpoint_sha256=digest(checkpoint),pipeline_sources=source_hashes(),records_sha256=digest(file),
                        numpy_version=np.__version__,conf_filter=.001,nms_iou=.6,max_det=300,save_hybrid=False,half=False,saturated_images=[]))
        write_json(project/'registry.json',registry)
        with patch('exposure.analyze.bootstrap',return_value={}) as boot:
            out=self.root/'analysis';analyze(b,project,out)
            self.assertEqual(boot.call_count,6)
            self.assertTrue(all(call.args[2]==2000 for call in boot.call_args_list))
        scores=read_csv(out/'scores.csv');self.assertEqual(len(scores),72)
        summary=read_csv(out/'seed_summary.csv')
        self.assertTrue(all(int(row['seeds'])==3 for row in summary))
        primary=[r for r in summary if r['comparison']=='100-0_T' and r['metric']=='mAP50_95']
        self.assertEqual(len(primary),2)
        self.assertTrue(all(float(r['mean'])>0 for r in primary))


if __name__=='__main__':unittest.main()
