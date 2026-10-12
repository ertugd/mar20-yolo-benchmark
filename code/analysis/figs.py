import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
OUT='/home/claude/paper/figs/'
MODELS=['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']
NICE={'yolov5nu':'YOLOv5n','yolov8n':'YOLOv8n','yolo11n':'YOLOv11n','yolo12n':'YOLOv12n','yolo26n':'YOLOv26n'}
COL={'yolov5nu':'#2a78d6','yolov8n':'#eb6834','yolo11n':'#1baf7a','yolo12n':'#4a3aa7','yolo26n':'#e34948'}
MK={'yolov5nu':'o','yolov8n':'s','yolo11n':'^','yolo12n':'D','yolo26n':'v'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
  'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,
  'grid.color':'#e6e5e0','grid.linewidth':0.6,'savefig.dpi':300,'legend.frameon':False})
S=json.load(open('stats.json'))
TRAIN={'yolov5nu':74.48,'yolov8n':77.07,'yolo11n':79.66,'yolo12n':110.71,'yolo26n':98.97}
PAPER={'yolov5nu':79.78,'yolov8n':79.77,'yolo11n':80.04,'yolo12n':80.45,'yolo26n':80.28}
# --- curves montage
names=['BoxP_curve','BoxR_curve','BoxF1_curve','BoxPR_curve']
ims=[Image.open(f'/mnt/user-data/uploads/mar20v1/runs_mar20/test_yolo12n/{n}.png').convert('RGB') for n in names]
w,h=ims[0].size; M=Image.new('RGB',(2*w,2*h),'white')
for i,im in enumerate(ims): M.paste(im.resize((w,h)),((i%2)*w,(i//2)*h))
M.resize((2*w//2,2*h//2)).save(OUT+'fig_curves.png')
# --- confusion matrix original
Image.open('/mnt/user-data/uploads/mar20v1/runs_mar20/test_yolo12n/confusion_matrix.png').convert('RGB').save(OUT+'fig_confusion.png')
# --- qualitative
a=Image.open('/mnt/user-data/uploads/mar20v1/runs_mar20/test_yolo12n/val_batch0_labels.jpg');b=Image.open('/mnt/user-data/uploads/mar20v1/runs_mar20/test_yolo12n/val_batch0_pred.jpg')
Q=Image.new('RGB',(a.width+b.width+20,a.height),'white');Q.paste(a,(0,0));Q.paste(b,(a.width+20,0));Q.save(OUT+'fig_qualitative.jpg',quality=90)
# --- training curves + time
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6),gridspec_kw={'width_ratios':[1.15,1.15,1]})
for m in MODELS:
    d=pd.read_csv(f'../runs_mar20/{m}/results.csv'); d.columns=[c.strip() for c in d.columns]
    y=d['metrics/mAP50-95(B)']*100
    ax[0].plot(d['epoch'],y,color=COL[m],lw=1.0,label=NICE[m])
    ax[1].plot(d['epoch'],y,color=COL[m],lw=0.5,alpha=0.35)
    ax[1].plot(d['epoch'],y.rolling(10,min_periods=1).mean(),color=COL[m],lw=1.4,label=NICE[m])
ax[0].set_ylim(30,83); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Validation mAP@0.5:0.95 (%)'); ax[0].legend(loc='lower right',fontsize=6.5)
ax[0].set_title('(a) All epochs',loc='left',fontsize=8.5)
ax[1].set_xlim(100,200); ax[1].set_ylim(78.5,81.8); ax[1].set_xlabel('Epoch'); ax[1].set_title('(b) Epochs 100-200, 10-epoch mean',loc='left',fontsize=8.5)
off={'yolov5nu':(-8,6),'yolov8n':(4,-11),'yolo11n':(6,-3),'yolo12n':(-14,7),'yolo26n':(6,-3)}
for m in MODELS:
    ax[2].plot(TRAIN[m],PAPER[m],marker=MK[m],ms=7,color=COL[m],ls='none',mec='white',mew=1)
    ax[2].annotate(NICE[m],(TRAIN[m],PAPER[m]),xytext=off[m],textcoords='offset points',fontsize=6.5,color='#0b0b0b')
ax[2].set_xlabel('Training time (min)'); ax[2].set_ylabel('Test mAP@0.5:0.95 (%)'); ax[2].set_xlim(68,122); ax[2].set_ylim(79.6,80.6)
ax[2].set_title('(c)',loc='left',fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT+'fig_training.pdf'); plt.close()
# --- bootstrap CI forest
fig,ax=plt.subplots(1,2,figsize=(7.2,2.3))
for j,(k,ck,lab) in enumerate([('map50','ci50','mAP@0.5 (%)'),('map','ci','mAP@0.5:0.95 (%)')]):
    for i,m in enumerate(MODELS[::-1]):
        r=S['R'][m]; lo,hi=[x*100 for x in r[ck]]; v=r[k]*100
        ax[j].plot([lo,hi],[i,i],color=COL[m],lw=2,solid_capstyle='round'); ax[j].plot(v,i,'o',color=COL[m],ms=7,mec='white',mew=1)
        ax[j].annotate(f'{v:.2f} [{lo:.2f}, {hi:.2f}]',(hi,i),xytext=(5,-3),textcoords='offset points',fontsize=6.5,color='#52514e')
    ax[j].set_yticks(range(5)); ax[j].set_yticklabels([NICE[m] for m in MODELS[::-1]]); ax[j].set_xlabel(lab); ax[j].grid(axis='y',visible=False)
    ax[j].set_title('(%s)'%'ab'[j],loc='left',fontsize=9)
ax[0].set_xlim(94.8,100.2); ax[1].set_xlim(77.8,85.4)
plt.tight_layout(); plt.savefig(OUT+'fig_bootstrap.pdf'); plt.close()
# --- McNemar matrices
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
for ax,(s,title) in zip(axs,[('opt','(a) F1-optimal threshold'),('0.25','(b) Fixed threshold 0.25')]):
    P=np.full((5,5),np.nan); T=[['']*5 for _ in range(5)]
    for r in S['mcnemar'][s]['pairs']:
        i,j=MODELS.index(r['a']),MODELS.index(r['b'])
        P[j,i]=-np.log10(max(r['p_holm'],1e-6))
        T[j][i]=f"{r['a_only']}/{r['b_only']}\np={r['p_holm']:.3f}" if r['p_holm']>=0.001 else f"{r['a_only']}/{r['b_only']}\np<0.001"
    im=ax.imshow(P,cmap='Blues',vmin=0,vmax=3)
    for i in range(5):
        for j in range(5):
            if T[i][j]: ax.text(j,i,T[i][j],ha='center',va='center',fontsize=6.5,color='white' if P[i,j]>1.8 else '#0b0b0b')
    ax.set_xticks(range(5)); ax.set_xticklabels([NICE[m] for m in MODELS],rotation=30,ha='right'); ax.set_yticks(range(5)); ax.set_yticklabels([NICE[m] for m in MODELS])
    ax.grid(False); ax.set_title(title,loc='left',fontsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
cb=fig.colorbar(im,ax=axs,shrink=0.8); cb.set_label(r'$-\log_{10}$ (Holm-adjusted $p$)')
plt.savefig(OUT+'fig_mcnemar.pdf',bbox_inches='tight'); plt.close()
# --- per-class AP heatmap
A=np.array([S['R_orig_cls_ap'][m] for m in MODELS]).T*100
A50=np.array([S['R_orig_cls_ap50'][m] for m in MODELS]).T*100
fig,axs=plt.subplots(1,2,figsize=(7.2,4.6))
for ax,X,t,vmin in [(axs[0],A50,'(a) AP@0.5 (%)',80),(axs[1],A,'(b) AP@0.5:0.95 (%)',60)]:
    im=ax.imshow(X,cmap='Blues',vmin=vmin,vmax=100,aspect='auto')
    for i in range(20):
        for j in range(5): ax.text(j,i,f'{X[i,j]:.1f}',ha='center',va='center',fontsize=6,color='white' if X[i,j]>vmin+0.7*(100-vmin) else '#0b0b0b')
    ax.set_yticks(range(20)); ax.set_yticklabels([f'A{i+1}' for i in range(20)]); ax.set_xticks(range(5)); ax.set_xticklabels([NICE[m] for m in MODELS],rotation=30,ha='right')
    ax.grid(False); ax.set_title(t,loc='left',fontsize=9); fig.colorbar(im,ax=ax,shrink=0.6)
    for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout(); plt.savefig(OUT+'fig_perclass.pdf'); plt.close()
print('ok')
# --- CAM focus
import json as _j
d=pd.DataFrame(_j.load(open('cam_metrics.json')))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.5),gridspec_kw={'width_ratios':[1.5,1]})
data=[d[d.model==m].e_in.values*100 for m in MODELS]
bp=ax[0].boxplot(data,widths=0.5,patch_artist=True,showfliers=False,medianprops=dict(color='#0b0b0b',lw=1.2))
for p,m in zip(bp['boxes'],MODELS): p.set_facecolor(COL[m]); p.set_alpha(0.55); p.set_edgecolor(COL[m])
for i,m in enumerate(MODELS): ax[0].plot(i+1,data[i].mean(),'D',color='white',mec='#0b0b0b',ms=4)
ax[0].axhline(d.area_in.mean()*100,color='#52514e',ls='--',lw=0.8); ax[0].text(5.45,d.area_in.mean()*100+2,'box area',fontsize=6.5,color='#52514e',ha='right')
ax[0].set_xticks(range(1,6)); ax[0].set_xticklabels([NICE[m] for m in MODELS]); ax[0].set_ylabel('CAM energy in target boxes (%)'); ax[0].set_title('(a)',loc='left',fontsize=9)
h=[d[d.model==m].hit.mean()*100 for m in MODELS]
ax[1].bar(range(5),h,color=[COL[m] for m in MODELS],width=0.6)
for i,v in enumerate(h): ax[1].text(i,v+1.5,f'{v:.1f}',ha='center',fontsize=6.5,color='#0b0b0b')
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([NICE[m] for m in MODELS],rotation=30,ha='right'); ax[1].set_ylabel('Pointing-game hit rate (%)'); ax[1].set_ylim(0,100)
ax[1].set_title('(b)',loc='left',fontsize=9); ax[1].grid(axis='x',visible=False)
plt.tight_layout(); plt.savefig(OUT+'fig_camstats.pdf'); plt.close()
print('cam ok')
