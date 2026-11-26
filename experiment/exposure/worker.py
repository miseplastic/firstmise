"""GPU export worker; runs only in the verified server YOLO environment."""
import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np

# Keep the offline pipeline import separate from YOLO's `utils` package.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exposure.common import check_bundle,check_yolo,digest,require,source_hashes,write_json
from exposure.metrics import average_precision,matches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('yolov5','bundle','checkpoint','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--condition',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='0');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    plan,bhash=check_bundle(a.bundle);check_yolo(a.yolov5)
    sys.path.insert(0,str(a.yolov5.resolve()))
    import torch
    from models.common import DetectMultiBackend
    from utils.dataloaders import create_dataloader
    from utils.general import LOGGER,check_img_size,non_max_suppression,scale_boxes,xywh2xyxy
    from utils.metrics import process_batch,ap_per_class
    from utils.torch_utils import select_device
    device=select_device(a.device,batch_size=4)
    model=DetectMultiBackend(str(a.checkpoint),device=device,fp16=False)
    require(model.pt and int(model.model.nc)==1,'Need a one-class PyTorch model')
    stride=int(model.stride);size=check_img_size(640,s=stride)
    model.eval();model.warmup(imgsz=(1,3,size,size))
    selected=[r for r in plan['records'] if r['split']==a.split]
    by_path={str((a.bundle/r['image']).resolve()):r for r in selected}
    loader=create_dataloader(str((a.bundle/f'lists/{a.split}.txt').resolve()),size,4,stride,
                             single_cls=False,pad=0.5,rect=True,workers=a.workers)[0]
    iouv=torch.linspace(.5,.95,10,device=device)
    nms_warnings=[]
    class NMSGuard(logging.Handler):
        def emit(self,record):
            message=record.getMessage()
            if 'NMS' in message and 'time limit' in message.lower():nms_warnings.append(message)
    LOGGER.addHandler(NMSGuard())
    records=[];seen=set();saturated=[]
    outfile=a.out/'records.jsonl'
    with outfile.open('x',encoding='utf-8') as stream, torch.no_grad():
        for im,targets,paths,shapes in loader:
            im=im.to(device,non_blocking=True).float()/255
            targets=targets.to(device)
            targets[:,2:]*=torch.tensor([im.shape[3],im.shape[2],im.shape[3],im.shape[2]],device=device)
            raw=model(im,augment=False)
            raw_prediction=raw[0] if isinstance(raw,(list,tuple)) else raw
            require(bool(torch.isfinite(raw_prediction).all()),'Non-finite model output; cannot certify zero detections')
            predictions=non_max_suppression(raw,conf_thres=.001,iou_thres=.6,labels=(),
                                           multi_label=True,agnostic=False,max_det=300)
            require(not nms_warnings,'NMS timed out; remaining images cannot be marked as zero predictions')
            for i,pred in enumerate(predictions):
                path=str(Path(paths[i]).resolve())
                require(path in by_path and path not in seen,'Unexpected/duplicate evaluation image')
                seen.add(path);r=by_path[path]
                labels=targets[targets[:,0]==i,1:]
                box=xywh2xyxy(labels[:,1:5])
                scale_boxes(im[i].shape[1:],box,shapes[i][0],shapes[i][1])
                gt=torch.cat((labels[:,0:1],box),1)
                native=pred.clone()
                scale_boxes(im[i].shape[1:],native[:,:4],shapes[i][0],shapes[i][1])
                correct=process_batch(native,gt,iouv) if len(gt) else torch.zeros((len(native),10),dtype=torch.bool,device=device)
                require(bool((native[:,5]==0).all()) and bool((gt[:,0]==0).all()),'Unexpected class')
                record=dict(image_id=r['image_id'],group_id=r['group_id'],ab=r['ab'],condition=a.condition,
                            seed=a.seed,split=a.split,evaluation_status='completed',gt_version=plan['gt_version'],
                            gt=gt.cpu().tolist(),predictions=native.cpu().tolist(),correct=correct.cpu().tolist())
                # Validate the count matcher against official matching at each candidate threshold.
                for threshold in (i/20 for i in range(1,20)):
                    filtered=native[native[:,4].double()>=threshold]
                    official=process_batch(filtered,gt,iouv[:1]).cpu().numpy()[:,0] if len(gt) else np.zeros(len(filtered),bool)
                    require(np.array_equal(matches(filtered.cpu().tolist(),record['gt']),official),'FP/FN matcher parity failure')
                if len(native)==300:saturated.append(r['image_id'])
                records.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n')
    require(seen==set(by_path),'Some evaluation images were skipped; export is incomplete')
    order={r['image_id']:i for i,r in enumerate(selected)}
    records.sort(key=lambda r:order[r['image_id']])
    require(sum(len(r['gt']) for r in records)>0,'No GT')
    # Reuse official AP routine and compare our CPU recomputation before certifying export.
    confidence=np.asarray([v[4] for r in records for v in r['predictions']])
    correctness=np.concatenate([np.asarray(r['correct'],bool).reshape(-1,10) for r in records])
    classes=np.zeros(len(confidence));target=np.zeros(sum(len(r['gt']) for r in records))
    official=(0.,0.)
    if len(confidence) and correctness.any():
        ap=ap_per_class(correctness,confidence,classes,target,plot=False,names={0:'Microplastic'})[5]
        official=(float(ap[:,0].mean()),float(ap.mean()))
    offline=average_precision(records)
    require(np.allclose(official,offline,atol=1e-7,rtol=0),'Official/offline AP mismatch')
    check_bundle(a.bundle)
    write_json(a.out/'summary.json',dict(status='completed',split=a.split,condition=a.condition,seed=a.seed,
               bundle_sha256=bhash,checkpoint_sha256=digest(a.checkpoint),records_sha256=digest(outfile),
               pipeline_sources=source_hashes(),images=len(records),official_mAP50=official[0],official_mAP50_95=official[1],
               conf_filter=.001,nms_iou=.6,max_det=300,half=False,augment=False,save_hybrid=False,
               saturated_images=saturated,torch_version=torch.__version__,numpy_version=np.__version__,device=str(device),
               gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'))
    print(json.dumps({'images':len(records),'official_mAP':official,'saturated_images':len(saturated)}))


if __name__=='__main__':main()
