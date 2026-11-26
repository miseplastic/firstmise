"""Single-class metrics, with AP integration/matching compatible with verified YOLOv5.

Reference: ultralytics/yolov5 9777cd00 utils/metrics.py (AGPL-3.0).
AP correctness flags are produced by that official process_batch in the GPU worker.
"""
import numpy as np


def matches(predictions, gt, threshold=0.5):
    p=np.asarray(predictions,dtype=np.float32).reshape(-1,6)
    g=np.asarray(gt,dtype=np.float32).reshape(-1,5)
    correct=np.zeros(len(p),dtype=bool)
    if not len(p) or not len(g): return correct
    lt=np.maximum(g[:,None,1:3],p[None,:,0:2])
    rb=np.minimum(g[:,None,3:5],p[None,:,2:4])
    wh=np.maximum(rb-lt,0);inter=wh[:,:,0]*wh[:,:,1]
    ga=(g[:,3]-g[:,1])*(g[:,4]-g[:,2]);pa=(p[:,2]-p[:,0])*(p[:,3]-p[:,1])
    iou=inter/(ga[:,None]+pa[None,:]-inter+1e-7)
    gi,pi=np.where((iou>=threshold)&(g[:,None,0]==p[None,:,5]))
    if len(gi):
        m=np.column_stack((gi,pi,iou[gi,pi]))
        if len(m)>1:
            m=m[m[:,2].argsort()[::-1]]
            m=m[np.unique(m[:,1],return_index=True)[1]]
            m=m[np.unique(m[:,0],return_index=True)[1]]
        correct[m[:,1].astype(int)]=True
    return correct


def count_metrics(record, threshold):
    cache=record.setdefault('_counts',{})
    if threshold in cache: return cache[threshold]
    p=[v for v in record['predictions'] if v[4]>=threshold]
    n=len(record['gt']);tp=int(matches(p,record['gt']).sum())
    result=dict(gt_count=n,pred_count=len(p),TP=tp,FP=len(p)-tp,FN=n-tp,
                signed_count_error=len(p)-n,abs_count_error=abs(len(p)-n),image_recall=tp/n if n else None)
    cache[threshold]=result
    return result


def average_precision(records):
    """Recompute 101-point AP, including duplicate bootstrap occurrences."""
    nt=sum(len(r['gt']) for r in records)
    if not nt: return float('nan'),float('nan')
    conf=np.asarray([p[4] for r in records for p in r['predictions']])
    if not len(conf): return 0.,0.
    tp=np.concatenate([np.asarray(r['correct'],dtype=bool).reshape(-1,10) for r in records])
    tp=tp[np.argsort(-conf)]
    tpc=tp.cumsum(0);fpc=(1-tp).cumsum(0)
    recall=tpc/(nt+1e-16);precision=tpc/(tpc+fpc)
    ap=[]
    for i in range(10):
        rec=np.concatenate(([0.],recall[:,i],[1.]))
        pre=np.concatenate(([1.],precision[:,i],[0.]))
        pre=np.flip(np.maximum.accumulate(np.flip(pre)))
        x=np.linspace(0,1,101)
        integrate=getattr(np,'trapezoid',None) or np.trapz
        ap.append(float(integrate(np.interp(x,rec,pre),x)))
    return ap[0],float(np.mean(ap))


def summarize(records, threshold):
    values=[count_metrics(r,threshold) for r in records]
    ap50,ap=average_precision(records)
    return dict(images=len(records),groups=len({r['group_id'] for r in records}),mAP50=ap50,mAP50_95=ap,
                MAE=float(np.mean([v['abs_count_error'] for v in values])),
                Bias=float(np.mean([v['signed_count_error'] for v in values])),
                FP=sum(v['FP'] for v in values),FN=sum(v['FN'] for v in values),
                FP_per_image=float(np.mean([v['FP'] for v in values])),
                FN_per_image=float(np.mean([v['FN'] for v in values])))


def choose_threshold(records):
    grid=[i/20 for i in range(1,20)]
    scores=[(t,float(np.mean([abs(sum(p[4]>=t for p in r['predictions'])-len(r['gt'])) for r in records]))) for t in grid]
    return min(scores,key=lambda x:(x[1],x[0]))[0],scores
