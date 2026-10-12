import pickle, numpy as np, json, itertools, pandas as pd, pingouin as pg
from ultralytics.utils.metrics import ap_per_class, box_iou
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.multitest import multipletests
import torch
MODELS=['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']
NICE={'yolov5nu':'YOLOv5nu','yolov8n':'YOLOv8n','yolo11n':'YOLO11n','yolo12n':'YOLO12n','yolo26n':'YOLO26n'}
D=pickle.load(open('capture.pkl','rb'))
stems=[r['stem'] for r in D['yolov5nu']['rec']]
for m in MODELS: assert [r['stem'] for r in D[m]['rec']]==stems
R={}
def mapof(recs,idx):
    tp=np.concatenate([recs[i]['tp'] for i in idx]); conf=np.concatenate([recs[i]['conf'] for i in idx])
    pc=np.concatenate([recs[i]['pcls'] for i in idx]); tc=np.concatenate([recs[i]['tcls'] for i in idx])
    o=ap_per_class(tp,conf,pc,tc); ap=o[5]; return ap[:,0].mean(), ap.mean(), o
allidx=np.arange(len(stems))
for m in MODELS:
    m50,m95,o=mapof(D[m]['rec'],allidx)
    f1=o[9].mean(0); from ultralytics.utils.metrics import smooth
    i=smooth(f1,0.1).argmax(); conf_star=float(o[10][i])
    R[m]=dict(map50=m50,map=m95,P=float(o[2].mean()),R=float(o[3].mean()),F1=float(o[4].mean()),conf_star=conf_star,
              ap50=o[5][:,0].tolist(),ap=o[5].mean(1).tolist(),classes=o[6].tolist())
    print(m,round(m50*100,2),round(m95*100,2),'P',round(R[m]['P']*100,2),'R',round(R[m]['R']*100,2),'conf*',round(conf_star,3))
# ---------- bootstrap ----------
rng=np.random.default_rng(0); B=1000
boot={m:[] for m in MODELS}
for b in range(B):
    idx=rng.integers(0,len(stems),len(stems))
    for m in MODELS:
        m50,m95,_=mapof(D[m]['rec'],idx); boot[m].append((m50,m95))
boot={m:np.array(v) for m,v in boot.items()}
for m in MODELS:
    R[m]['ci50']=np.percentile(boot[m][:,0],[2.5,97.5]).tolist(); R[m]['ci']=np.percentile(boot[m][:,1],[2.5,97.5]).tolist()
pair_boot={}
for a,b in itertools.combinations(MODELS,2):
    d50=boot[a][:,0]-boot[b][:,0]; d=boot[a][:,1]-boot[b][:,1]
    pair_boot[f'{a}|{b}']=dict(d50=float(R[a]['map50']-R[b]['map50']),ci50=np.percentile(d50,[2.5,97.5]).tolist(),
        d=float(R[a]['map']-R[b]['map']),ci=np.percentile(d,[2.5,97.5]).tolist(),
        p_boot=float(2*min((d<=0).mean(),(d>=0).mean())), p_boot50=float(2*min((d50<=0).mean(),(d50>=0).mean())))
# ---------- operating point matching ----------
def greedy(tb,tc,pb,pc,thr=0.5):
    """returns gt_ok (bool per gt), gt_loc_wrongcls, n_tp, n_fp"""
    ok=np.zeros(len(tb),bool); used=np.zeros(len(pb),bool)
    if len(pb) and len(tb):
        iou=box_iou(torch.tensor(tb),torch.tensor(pb)).numpy()
        same=(tc[:,None]==pc[None,:])
        cand=np.argwhere((iou>=thr)&same)
        cand=cand[np.argsort(-iou[cand[:,0],cand[:,1]])]
        for g,p in cand:
            if not ok[g] and not used[p]: ok[g]=True; used[p]=True
    # wrong-class localisation: unmatched gt overlapped (iou>=thr) by an unused pred of other class
    wrong=np.zeros(len(tb),bool)
    if len(pb) and len(tb):
        for g in np.nonzero(~ok)[0]:
            if ((iou[g]>=thr)&(~used)).any(): wrong[g]=True
    ntp=int(ok.sum()); nfp=int(len(pb)-used.sum())
    return ok,wrong,ntp,nfp
def op_eval(m,conf_thr):
    gt_ok=[];gt_wrong=[];gt_cls=[];per_img=[]
    for r in D[m]['rec']:
        k=r['conf']>=conf_thr
        ok,wr,ntp,nfp=greedy(r['tbox'],r['tcls'],r['pbox'][k],r['pcls'][k])
        gt_ok.append(ok);gt_wrong.append(wr);gt_cls.append(r['tcls'])
        fn=len(r['tbox'])-ntp; per_img.append((ntp,nfp,fn))
    return np.concatenate(gt_ok),np.concatenate(gt_wrong),np.concatenate(gt_cls),np.array(per_img)
OP={}
for setting in ['opt','0.25']:
    OP[setting]={}
    for m in MODELS:
        thr=R[m]['conf_star'] if setting=='opt' else 0.25
        OP[setting][m]=op_eval(m,thr)
def mcn(setting):
    rows=[]
    for a,b in itertools.combinations(MODELS,2):
        A=OP[setting][a][0];Bb=OP[setting][b][0]
        n01=int((A&~Bb).sum()); n10=int((~A&Bb).sum())
        t=[[int((A&Bb).sum()),n01],[n10,int((~A&~Bb).sum())]]
        p=mcnemar(t,exact=True).pvalue
        rows.append(dict(a=a,b=b,a_only=n01,b_only=n10,p=p,acc_a=float(A.mean()),acc_b=float(Bb.mean())))
    ps=[r['p'] for r in rows]; adj=multipletests(ps,method='holm')[1]
    for r,q in zip(rows,adj): r['p_holm']=float(q)
    X=np.stack([OP[setting][m][0] for m in MODELS],1).astype(int)
    q=cochrans_q(X)
    return rows,dict(Q=float(q.statistic),p=float(q.pvalue),df=4)
MC={s:mcn(s) for s in OP}
# ---------- ANOVA ----------
def anova_long(vals,subj_name):
    df=pd.DataFrame([(s,m,v) for m in MODELS for s,v in enumerate(vals[m])],columns=[subj_name,'model','y'])
    a=pg.rm_anova(df,dv='y',within='model',subject=subj_name,correction=True,detailed=True,effsize='np2')
    f=pg.friedman(df,dv='y',within='model',subject=subj_name)
    post=pg.pairwise_tests(df,dv='y',within='model',subject=subj_name,padjust='holm',parametric=True)
    return a,f,post,df
AN={}
per_img_f1={}
for m in MODELS:
    pi=OP['opt'][m][3]; tp,fp,fn=pi[:,0],pi[:,1],pi[:,2]; per_img_f1[m]=2*tp/np.maximum(2*tp+fp+fn,1)
AN['img_f1']=anova_long(per_img_f1,'image')
AN['cls_ap']=anova_long({m:np.array(R[m]['ap']) for m in MODELS},'cls')
AN['cls_ap50']=anova_long({m:np.array(R[m]['ap50']) for m in MODELS},'cls')
out=dict(R=R,pair_boot=pair_boot,mcnemar={s:dict(pairs=MC[s][0],cochran=MC[s][1]) for s in MC},
         anova={k:dict(rm=v[0].to_dict(orient='records'),friedman=v[1].to_dict(orient='records'),post=v[2].to_dict(orient='records'),
                       means=v[3].groupby('model').y.mean().to_dict(), sds=v[3].groupby('model').y.std().to_dict()) for k,v in AN.items()},
         op_counts={s:{m:dict(n_gt=int(len(OP[s][m][0])),ok=int(OP[s][m][0].sum()),wrongcls=int(OP[s][m][1].sum()),
                   missed=int((~OP[s][m][0]&~OP[s][m][1]).sum()),fp=int(OP[s][m][3][:,1].sum())) for m in MODELS} for s in OP})
json.dump(out,open('stats.json','w'),indent=1,default=float)
pickle.dump(dict(OP=OP,boot=boot,per_img_f1=per_img_f1),open('stats_aux.pkl','wb'))
for s in MC:
    print('== McNemar',s, MC[s][1])
    for r in MC[s][0]: print(f"{r['a']:9s} {r['b']:9s} {r['a_only']:3d} {r['b_only']:3d} p={r['p']:.4f} holm={r['p_holm']:.4f} acc {r['acc_a']:.4f} {r['acc_b']:.4f}")
for k,v in AN.items():
    print('== ANOVA',k); print(v[0].round(4).to_string()); print(v[1].round(4).to_string())
    print(v[2][['A','B','T','p-unc','p-corr']].round(4).to_string())
print(json.dumps(out['op_counts'],indent=0))
for m in MODELS: print(m,[round(x*100,2) for x in R[m]['ci50']],[round(x*100,2) for x in R[m]['ci']])
