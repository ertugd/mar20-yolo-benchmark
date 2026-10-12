import pickle, numpy as np, torch, os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator as D
os.chdir('/home/claude/work')
REC=[]
orig=D.update_metrics
def patched(self,preds,batch):
    for si,pred in enumerate(preds):
        pb=self._prepare_batch(si,batch); pn=self._prepare_pred(pred)
        tp=self._process_batch(pn,pb)['tp']
        REC.append(dict(stem=Path(pb['im_file']).stem,tp=tp,conf=pn['conf'].cpu().numpy(),pcls=pn['cls'].cpu().numpy().astype(int),
            pbox=pn['bboxes'].cpu().numpy(),tcls=pb['cls'].cpu().numpy().astype(int),tbox=pb['bboxes'].cpu().numpy()))
    return orig(self,preds,batch)
D.update_metrics=patched
res={}
for m in ['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']:
    REC.clear()
    r=YOLO(f'runs_mar20/{m}/weights/best.pt').val(data='mar20_yolo/data.yaml',split='test',imgsz=640,batch=16,conf=0.001,iou=0.7,max_det=300,device='cpu',plots=False,verbose=False,project='/home/claude/work/runs/cap',name=m,exist_ok=True)
    res[m]=dict(rec=list(REC),map50=r.box.map50,map=r.box.map,mp=r.box.mp,mr=r.box.mr,ap50=r.box.ap50.copy(),ap=r.box.ap.copy(),
                speed=r.speed)
    print(m,r.box.map50,r.box.map,len(REC),flush=True)
pickle.dump(res,open('an/capture.pkl','wb'))
