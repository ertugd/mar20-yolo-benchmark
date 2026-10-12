import json, sys
from ultralytics import YOLO
out={}
for m in ['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']:
    model=YOLO(f'runs_mar20/{m}/weights/best.pt')
    r=model.val(data='mar20_yolo/data.yaml',split='test',imgsz=640,batch=8,conf=0.001,iou=0.7,max_det=300,device='cpu',save_json=True,project='val_cpu',name=m,exist_ok=True,plots=False,verbose=False)
    out[m]={'mAP50':r.box.map50,'mAP':r.box.map,'P':r.box.mp,'R':r.box.mr}
    print(m,out[m],flush=True)
json.dump(out,open('val_cpu/summary.json','w'),indent=1)
