import cv2, numpy as np, glob, os, json, pickle, torch, time
torch.set_num_threads(1)
from gradcam import YoloGradCAM, class_cam, box_mask
MODELS=['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']
stems=sorted(os.path.basename(f)[:-4] for f in glob.glob('../mar20_yolo/test/labels/*.txt'))
rng=np.random.default_rng(0); sub=sorted(rng.choice(stems,150,replace=False).tolist())
G={m:YoloGradCAM(f'../runs_mar20/{m}/weights/best.pt') for m in MODELS}
rows=[]; t0=time.time()
for n,s in enumerate(sub):
    img=cv2.imread(f'../mar20_yolo/test/images/{s}.jpg'); H,W=img.shape[:2]
    L=np.loadtxt(f'../mar20_yolo/test/labels/{s}.txt',ndmin=2)
    cls=L[:,0].astype(int); bx=np.stack([(L[:,1]-L[:,3]/2)*W,(L[:,2]-L[:,4]/2)*H,(L[:,1]+L[:,3]/2)*W,(L[:,2]+L[:,4]/2)*H],1)
    for c in np.unique(cls):
        mc=box_mask((H,W),bx[cls==c]); mo=box_mask((H,W),bx[cls!=c])&~mc
        for m in MODELS:
            cam,nsel=class_cam(G[m],img,int(c))
            tot=cam.sum()+1e-12
            yx=np.unravel_index(cam.argmax(),cam.shape)
            rows.append(dict(stem=s,cls=int(c),model=m,e_in=float(cam[mc].sum()/tot),e_other=float(cam[mo].sum()/tot),
                             area_in=float(mc.mean()),hit=bool(mc[yx]),nsel=nsel))
    if n%10==0: print(n,len(rows),round(time.time()-t0),flush=True)
json.dump(rows,open('cam_metrics.json','w'))
