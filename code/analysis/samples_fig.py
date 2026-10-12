import numpy as np, matplotlib, cv2
matplotlib.use('Agg'); import matplotlib.pyplot as plt
SC=['3633','2715','2591','2545','3679','66','601','417']
fig,axs=plt.subplots(2,4,figsize=(7.2,3.75))
for ax,s,lt in zip(axs.ravel(),SC,'abcdefgh'):
    img=cv2.imread(f'mar20_yolo/test/images/{s}.jpg')[:,:,::-1]; H,W=img.shape[:2]
    L=np.loadtxt(f'mar20_yolo/test/labels/{s}.txt',ndmin=2); gc=L[:,0].astype(int)
    gb=np.stack([(L[:,1]-L[:,3]/2)*W,(L[:,2]-L[:,4]/2)*H,(L[:,1]+L[:,3]/2)*W,(L[:,2]+L[:,4]/2)*H],1)
    x1,y1,x2,y2=gb[:,0].min()-40,gb[:,1].min()-40,gb[:,2].max()+40,gb[:,3].max()+40
    side=min(max(x2-x1,y2-y1),W,H); cx,cy=(x1+x2)/2,(y1+y2)/2
    x1=int(np.clip(cx-side/2,0,max(0,W-side))); y1=int(np.clip(cy-side/2,0,max(0,H-side))); x2=int(min(W,x1+side)); y2=int(min(H,y1+side))
    ax.imshow(img[y1:y2,x1:x2])
    for b,c in zip(gb,gc):
        ax.add_patch(plt.Rectangle((b[0]-x1,b[1]-y1),b[2]-b[0],b[3]-b[1],fill=False,ec='#ffd400',lw=0.7))
        top=(b[1]-y1)<14
        ax.text(b[0]-x1,b[1]-y1+(2 if top else 0),f'A{c+1}',fontsize=5,color='black',va='top' if top else 'bottom',bbox=dict(boxstyle='square,pad=0.05',fc='#ffd400',ec='none',alpha=0.9),clip_on=True)
    ax.set_xlim(0,x2-x1); ax.set_ylim(y2-y1,0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f'({lt}) '+', '.join(f'A{c+1}' for c in sorted(set(gc))),fontsize=7)
plt.subplots_adjust(wspace=0.04,hspace=0.14,left=0.005,right=0.995,top=0.94,bottom=0.005)
plt.savefig('/home/claude/paper/figs/fig_samples.pdf',dpi=300); plt.savefig('/tmp/claude-0/samples.png',dpi=110)
