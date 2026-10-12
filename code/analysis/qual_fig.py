import numpy as np, matplotlib, cv2, torch
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
m=YOLO('runs_mar20/yolo12n/weights/best.pt')
SC=['341','725','680','750']
TT={'341':'(a)','725':'(b)','680':'(c)','750':'(d)'}
rows=[]
for s in SC:
    img=cv2.imread(f'mar20_yolo/test/images/{s}.jpg')[:,:,::-1]; H,W=img.shape[:2]
    L=np.loadtxt(f'mar20_yolo/test/labels/{s}.txt',ndmin=2); gc=L[:,0].astype(int)
    gb=np.stack([(L[:,1]-L[:,3]/2)*W,(L[:,2]-L[:,4]/2)*H,(L[:,1]+L[:,3]/2)*W,(L[:,2]+L[:,4]/2)*H],1)
    r=m.predict(img[:,:,::-1].copy(),imgsz=640,conf=0.25,iou=0.7,device='cpu',verbose=False)[0]
    pb=r.boxes.xyxy.numpy(); pc=r.boxes.cls.numpy().astype(int); pp=r.boxes.conf.numpy()
    ok=np.zeros(len(pb),bool)
    if len(pb):
        iou=box_iou(torch.tensor(gb),torch.tensor(pb)).numpy(); used=set()
        for j in np.argsort(-pp):
            cand=[i for i in range(len(gb)) if i not in used and iou[i,j]>=0.5 and gc[i]==pc[j]]
            if cand: i=max(cand,key=lambda i:iou[i,j]); used.add(i); ok[j]=True
    allb=np.concatenate([gb,pb]) if len(pb) else gb
    if s=='680': allb=allb[(allb[:,0]+allb[:,2])/2<W*0.42]
    x1,y1=max(0,allb[:,0].min()-45),max(0,allb[:,1].min()-45); x2,y2=min(W,allb[:,2].max()+45),min(H,allb[:,3].max()+45)
    rows.append((s,img,gb,gc,pb,pc,pp,ok,(int(x1),int(y1),int(x2),int(y2))))
hr=[(c[3]-c[1])/(c[2]-c[0]) for *_,c in rows]
fig=plt.figure(figsize=(7.2,7.2*0.5*sum(hr)+0.4))
gs=fig.add_gridspec(len(rows),2,height_ratios=hr,hspace=0.06,wspace=0.03,left=0.045,right=0.995,top=0.97,bottom=0.01)
def lab(ax,x,y,t,col):
    ax.text(x,y,t,fontsize=6.5,color='black' if col in ('#ffd400','#7CFC00') else 'white',va='bottom',ha='left',
            bbox=dict(boxstyle='square,pad=0.08',fc=col,ec='none',alpha=0.9))
for k,(s,img,gb,gc,pb,pc,pp,ok,(x1,y1,x2,y2)) in enumerate(rows):
    for j in range(2):
        ax=fig.add_subplot(gs[k,j]); ax.imshow(img[y1:y2,x1:x2]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        if j==0:
            for b,c in zip(gb,gc):
                ax.add_patch(plt.Rectangle((b[0]-x1,b[1]-y1),b[2]-b[0],b[3]-b[1],fill=False,ec='#ffd400',lw=0.9)); lab(ax,b[0]-x1,b[1]-y1,f'A{c+1}','#ffd400')
            ax.set_ylabel(TT[s],fontsize=8,rotation=0,labelpad=10,va='center')
            if k==0: ax.set_title('Ground truth',fontsize=8.5)
        else:
            for b,c,p,o in zip(pb,pc,pp,ok):
                col='#1a9e4b' if o else '#d62728'
                ax.add_patch(plt.Rectangle((b[0]-x1,b[1]-y1),b[2]-b[0],b[3]-b[1],fill=False,ec=col,lw=0.9)); lab(ax,b[0]-x1,b[1]-y1,f'A{c+1} '+f'{p:.2f}'[1:],col)
            if k==0: ax.set_title('YOLOv12n prediction',fontsize=8.5)
        ax.set_xlim(0,x2-x1); ax.set_ylim(y2-y1,0)
    print(s, 'pred',[(f'A{c+1}',round(float(p),2),bool(o)) for c,p,o in zip(pc,pp,ok)])
plt.savefig('/home/claude/paper/figs/fig_qualitative.pdf',dpi=300); plt.savefig('/tmp/claude-0/qual.png',dpi=110)
