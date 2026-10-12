import torch, numpy as np, cv2
from ultralytics.data.augment import LetterBox
STRIDES=[8,16,32]
class YoloGradCAM:
    def __init__(self, weight):
        ck=torch.load(weight,weights_only=False,map_location='cpu')
        self.net=ck['model'].float().eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        self.det=self.net.model[-1]; self.e2e=bool(getattr(self.det,'end2end',False))
        self.acts={}; self.grads={}
        for k,li in enumerate(self.det.f):
            layer=self.net.model[li]
            layer.register_forward_hook(self._fh(k))
        # anchor centers
        c=[];s=[]
        for st in STRIDES:
            n=640//st; ys,xs=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
            c.append(np.stack([(xs.ravel()+.5)*st,(ys.ravel()+.5)*st],1)); s+= [STRIDES.index(st)]*n*n
        self.centers=np.concatenate(c); self.scale=np.array(s)
    def _fh(self,k):
        def h(mod,inp,out):
            self.acts[k]=out
        return h
    def prep(self,img_bgr):
        lb=LetterBox((640,640),auto=False)
        im=lb(image=img_bgr)
        x=torch.from_numpy(im[:,:,::-1].copy()).permute(2,0,1).float()[None]/255.
        h0,w0=img_bgr.shape[:2]; r=min(640/h0,640/w0)
        pw,ph=(640-round(w0*r))/2,(640-round(h0*r))/2
        return x,(r,pw,ph)
    def __call__(self,img_bgr,box_xyxy,cls):
        """box in original image coords. returns cam (H0,W0) in [0,1], score, anchor"""
        x,(r,pw,ph)=self.prep(img_bgr); x.requires_grad_(True)
        out=self.net(x)
        d=out[1]['one2one'] if self.e2e else out[1]
        scores=d['scores'][0]  # (nc,8400) logits
        b=np.array(box_xyxy,float)*r+np.array([pw,ph,pw,ph])
        cx,cy=self.centers[:,0],self.centers[:,1]
        inside=(cx>=b[0])&(cx<=b[2])&(cy>=b[1])&(cy<=b[3])
        idx=np.nonzero(inside)[0]
        if len(idx)==0: idx=np.array([np.argmin((cx-(b[0]+b[2])/2)**2+(cy-(b[1]+b[3])/2)**2)])
        sc=scores[cls,idx].detach().numpy(); a=int(idx[sc.argmax()])
        k=int(self.scale[a])
        off=[0,6400,8000][k]; n=640//STRIDES[k]
        head=(self.det.one2one if self.e2e else self.det.one2many)['cls_head'][k]
        Afull=self.acts[k]; logits=head(Afull)[0].reshape(self.det.nc,-1)
        target=logits[cls,a-off]
        G=torch.autograd.grad(target,Afull)[0][0]; A=Afull[0].detach()
        w=G.mean(dim=(1,2)); cam=torch.relu((w[:,None,None]*A).sum(0)).detach().numpy()
        cam=cv2.resize(cam,(640,640),interpolation=cv2.INTER_LINEAR)
        h0,w0=img_bgr.shape[:2]
        cam=cam[int(round(ph)):int(round(ph))+round(h0*r), int(round(pw)):int(round(pw))+round(w0*r)]
        cam=cv2.resize(cam,(w0,h0))
        if cam.max()>0: cam=cam/cam.max()
        for k2 in list(self.acts): self.acts[k2]=None
        return cam,float(torch.sigmoid(target)),a,k
def energy_in_box(cam,box,scale=1.0):
    x1,y1,x2,y2=box; cx,cy,w,h=(x1+x2)/2,(y1+y2)/2,(x2-x1)*scale,(y2-y1)*scale
    H,W=cam.shape; X1,Y1,X2,Y2=[int(round(v)) for v in (max(0,cx-w/2),max(0,cy-h/2),min(W,cx+w/2),min(H,cy+h/2))]
    tot=cam.sum(); return float(cam[Y1:Y2,X1:X2].sum()/tot) if tot>0 else np.nan

def class_cam(g, img_bgr, cls, thr=0.25):
    """Class-level Grad-CAM: target = sum of class logits over anchors with score>thr (fallback: top anchor).
    CAMs from the three detection-head input layers are upsampled and summed."""
    x,(r,pw,ph)=g.prep(img_bgr)
    with torch.enable_grad():
        out=g.net(x.requires_grad_(True))
        head=(g.det.one2one if g.e2e else g.det.one2many)['cls_head']
        feats=[g.acts[k] for k in range(3)]
        logits=[head[k](feats[k])[0].reshape(g.det.nc,-1)[cls] for k in range(3)]
        allg=torch.cat(logits); sel=torch.sigmoid(allg)>thr
        if sel.sum()==0: sel=allg==allg.max()
        target=allg[sel].sum()
        grads=torch.autograd.grad(target,feats,allow_unused=True)
    h0,w0=img_bgr.shape[:2]; total=np.zeros((640,640),np.float32)
    for k in range(3):
        if grads[k] is None: continue
        A=feats[k][0].detach(); G=grads[k][0]
        w=G.mean(dim=(1,2)); cam=torch.relu((w[:,None,None]*A).sum(0)).numpy()
        total+=cv2.resize(cam,(640,640),interpolation=cv2.INTER_LINEAR)
    cam=total[int(round(ph)):int(round(ph))+round(h0*r), int(round(pw)):int(round(pw))+round(w0*r)]
    cam=cv2.resize(cam,(w0,h0))
    if cam.max()>0: cam=cam/cam.max()
    for k2 in list(g.acts): g.acts[k2]=None
    return cam, int(sel.sum())
def box_mask(shape,boxes):
    m=np.zeros(shape,bool)
    for x1,y1,x2,y2 in boxes:
        m[max(0,int(y1)):int(np.ceil(y2)),max(0,int(x1)):int(np.ceil(x2))]=True
    return m
