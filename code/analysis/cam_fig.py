import cv2,numpy as np,matplotlib,torch
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gradcam import YoloGradCAM,class_cam
MODELS=['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']
NICE={'yolov5nu':'YOLOv5nu','yolov8n':'YOLOv8n','yolo11n':'YOLO11n','yolo12n':'YOLO12n','yolo26n':'YOLO26n'}
G={m:YoloGradCAM(f'../runs_mar20/{m}/weights/best.pt') for m in MODELS}
CASES=[('1387',10),('1336',12),('864',18)]
fig,axs=plt.subplots(len(CASES),6,figsize=(7.4,3.9))
for r,(s,c) in enumerate(CASES):
    img=cv2.imread(f'../mar20_yolo/test/images/{s}.jpg'); H,W=img.shape[:2]
    L=np.loadtxt(f'../mar20_yolo/test/labels/{s}.txt',ndmin=2); cls=L[:,0].astype(int)
    bx=np.stack([(L[:,1]-L[:,3]/2)*W,(L[:,2]-L[:,4]/2)*H,(L[:,1]+L[:,3]/2)*W,(L[:,2]+L[:,4]/2)*H],1)
    x1,y1=max(0,bx[:,0].min()-40),max(0,bx[:,1].min()-40); x2,y2=min(W,bx[:,2].max()+40),min(H,bx[:,3].max()+40)
    # square crop
    side=max(x2-x1,y2-y1); cx,cy=(x1+x2)/2,(y1+y2)/2; x1,x2=int(max(0,cx-side/2)),int(min(W,cx+side/2)); y1,y2=int(max(0,cy-side/2)),int(min(H,cy+side/2))
    rgb=img[:,:,::-1]
    ax=axs[r,0]; ax.imshow(rgb[y1:y2,x1:x2])
    for b,k in zip(bx,cls):
        col='#ffd400' if k==c else 'white'; ls='-' if k==c else '--'
        ax.add_patch(plt.Rectangle((b[0]-x1,b[1]-y1),b[2]-b[0],b[3]-b[1],fill=False,ec=col,lw=1.0 if k==c else 0.6,ls=ls))
        ax.text(b[0]-x1,b[1]-y1-3,f'A{k+1}',color=col,fontsize=5)
    ax.set_xlim(0,x2-x1); ax.set_ylim(y2-y1,0)
    ax.set_ylabel(f'Target A{c+1}',fontsize=7)
    for j,m in enumerate(MODELS):
        cam,_=class_cam(G[m],img,c)
        hm=plt.cm.jet(cam)[:,:,:3]; ov=0.55*rgb/255.+0.45*hm
        ax=axs[r,j+1]; ax.imshow(ov[y1:y2,x1:x2])
        for b,k in zip(bx,cls):
            if k==c: ax.add_patch(plt.Rectangle((b[0]-x1,b[1]-y1),b[2]-b[0],b[3]-b[1],fill=False,ec='white',lw=0.6))
        ax.set_xlim(0,x2-x1); ax.set_ylim(y2-y1,0)
        if r==0: ax.set_title(NICE[m],fontsize=7.5)
    if r==0: axs[r,0].set_title('Image and labels',fontsize=7.5)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
plt.subplots_adjust(wspace=0.03,hspace=0.05,left=0.04,right=0.995,top=0.94,bottom=0.01)
plt.savefig('/home/claude/paper/figs/fig_gradcam.pdf',dpi=300); plt.savefig('/tmp/claude-0/gc.png',dpi=220)
