import json, numpy as np, glob, os
from PIL import Image
from ultralytics.utils.metrics import ap_per_class, box_iou
import torch
MODELS=['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']
NICE={'yolov5nu':'YOLOv5nu','yolov8n':'YOLOv8n','yolo11n':'YOLO11n','yolo12n':'YOLO12n','yolo26n':'YOLO26n'}
IOUV=np.linspace(0.5,0.95,10)
def load_gt():
    gt={}
    for f in sorted(glob.glob('mar20_yolo/test/labels/*.txt')):
        stem=os.path.basename(f)[:-4]
        w,h=Image.open(f'mar20_yolo/test/images/{stem}.jpg').size
        a=np.loadtxt(f,ndmin=2)
        c=a[:,0].astype(int); xc,yc,bw,bh=a[:,1]*w,a[:,2]*h,a[:,3]*w,a[:,4]*h
        gt[stem]=(c,np.stack([xc-bw/2,yc-bh/2,xc+bw/2,yc+bh/2],1))
    return gt
def load_pred(m):
    d=json.load(open(f'runs/detect/val_cpu/{m}/predictions.json'))
    P={}
    for r in d:
        s=os.path.splitext(r['file_name'])[0] if 'file_name' in r else str(r['image_id'])
        x,y,w,h=r['bbox']; P.setdefault(s,[]).append((x,y,x+w,y+h,r['score'],r['category_id']-1))
    return {k:np.array(v) for k,v in P.items()}
def match(gtc,gtb,pb,pc):
    """ultralytics-style matching; returns correct (npred,10)"""
    correct=np.zeros((len(pb),10),bool)
    if len(pb)==0 or len(gtb)==0: return correct
    iou=box_iou(torch.tensor(gtb),torch.tensor(pb)).numpy()
    iou=iou*(gtc[:,None]==pc[None,:])
    for i,t in enumerate(IOUV):
        mt=np.nonzero(iou>=t)
        mt=np.array(mt).T
        if mt.shape[0]:
            if mt.shape[0]>1:
                mt=mt[iou[mt[:,0],mt[:,1]].argsort()[::-1]]
                mt=mt[np.unique(mt[:,1],return_index=True)[1]]
                mt=mt[np.unique(mt[:,0],return_index=True)[1]]
            correct[mt[:,1].astype(int),i]=True
    return correct
def per_image_stats(gt,P):
    S={}
    for s,(gc,gb) in gt.items():
        p=P.get(s,np.zeros((0,6)))
        cor=match(gc,gb,p[:,:4],p[:,5].astype(int))
        S[s]=(cor,p[:,4],p[:,5].astype(int),gc)
    return S
def compute_map(S,stems):
    tp=np.concatenate([S[s][0] for s in stems]);conf=np.concatenate([S[s][1] for s in stems])
    pc=np.concatenate([S[s][2] for s in stems]);tc=np.concatenate([S[s][3] for s in stems])
    out=ap_per_class(tp,conf,pc,tc)
    ap=out[5]; return ap[:,0].mean(), ap.mean(), out
