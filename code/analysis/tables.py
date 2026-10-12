import json, itertools
S=json.load(open('stats.json')); C=json.load(open('cam_stats.json'))
MODELS=['yolov5nu','yolov8n','yolo11n','yolo12n','yolo26n']
N={'yolov5nu':'YOLOv5nu','yolov8n':'YOLOv8n','yolo11n':'YOLO11n','yolo12n':'YOLO12n','yolo26n':'YOLO26n'}
T='/home/claude/paper/tables/'
def p(x):
    return '$<$0.001' if x<0.001 else f'{x:.3f}'
# bootstrap table
L=[]
for m in MODELS:
    r=S['R'][m]
    L.append(f"{N[m]} & {r['map50']*100:.2f} & [{r['ci50'][0]*100:.2f}, {r['ci50'][1]*100:.2f}] & {r['map']*100:.2f} & [{r['ci'][0]*100:.2f}, {r['ci'][1]*100:.2f}] \\\\")
open(T+'boot.tex','w').write('\n'.join(L))
L=[]
for a,b in itertools.combinations(MODELS,2):
    v=S['pair_boot'][f'{a}|{b}']
    L.append(f"{N[a]} -- {N[b]} & {v['d50']*100:+.2f} & [{v['ci50'][0]*100:+.2f}, {v['ci50'][1]*100:+.2f}] & {v['d']*100:+.2f} & [{v['ci'][0]*100:+.2f}, {v['ci'][1]*100:+.2f}] \\\\")
open(T+'bootpairs.tex','w').write('\n'.join(L))
# mcnemar
L=[]
po={(r['a'],r['b']):r for r in S['mcnemar']['opt']['pairs']}; pf={(r['a'],r['b']):r for r in S['mcnemar']['0.25']['pairs']}
for a,b in itertools.combinations(MODELS,2):
    o=po[(a,b)]; f=pf[(a,b)]
    sig=lambda q: '$^{*}$' if q<0.05 else ''
    L.append(f"{N[a]} -- {N[b]} & {o['a_only']} / {o['b_only']} & {p(o['p'])} & {p(o['p_holm'])}{sig(o['p_holm'])} & {f['a_only']} / {f['b_only']} & {p(f['p'])} & {p(f['p_holm'])}{sig(f['p_holm'])} \\\\")
open(T+'mcnemar.tex','w').write('\n'.join(L))
# error taxonomy
L=[]
for m in MODELS:
    o=S['op_counts']['opt'][m]; f=S['op_counts']['0.25'][m]
    L.append(f"{N[m]} & {S['R'][m]['conf_star']:.3f} & {o['ok']} ({o['ok']/o['n_gt']*100:.2f}) & {o['wrongcls']} & {o['missed']} & {o['fp']} & {f['ok']} ({f['ok']/f['n_gt']*100:.2f}) & {f['wrongcls']} & {f['missed']} & {f['fp']} \\\\")
open(T+'errors.tex','w').write('\n'.join(L))
# anova
def arow(key,label,n):
    a=S['anova'][key]; rm=a['rm'][0]; fr=a['friedman'][0]
    return f"{label} & {n} & $F({rm['DF']},{a['rm'][1]['DF']})={rm['F']:.2f}$ & {rm['eps']:.3f} & {p(rm['p_GG_corr'])} & {rm['np2']:.3f} & {fr['Q']:.2f} & {p(fr['p_unc'])} & {fr['W']:.3f} \\\\"
L=[arow('cls_ap50','Per-class AP@0.5',20),arow('cls_ap','Per-class AP@0.5:0.95',20),arow('img_f1','Per-image F1',381)]
c=C['e_in']; rm=c['rm'][0]; fr=c['friedman'][0]
L.append(f"Grad-CAM $E_\\mathrm{{in}}$ & 211 & $F({rm['ddof1']},{rm['ddof2']})={rm['F']:.2f}$ & {rm['eps']:.3f} & {p(rm['p_GG_corr'])} & {C['e_in']['rm_np2']:.3f} & {fr['Q']:.2f} & {p(fr['p_unc'])} & {fr['W']:.3f} \\\\")
open(T+'anova.tex','w').write('\n'.join(L))
# means table for anova
L=[]
for m in MODELS:
    L.append(f"{N[m]} & {S['anova']['cls_ap50']['means'][m]*100:.2f} & {S['anova']['cls_ap']['means'][m]*100:.2f} & {S['anova']['img_f1']['means'][m]*100:.2f} ({S['anova']['img_f1']['sds'][m]*100:.2f}) \\\\")
open(T+'anova_means.tex','w').write('\n'.join(L))
# cam table
L=[]
for m in MODELS:
    s=C['summary'][m]
    L.append(f"{N[m]} & {s['e_in']*100:.1f} ({s['e_in_sd']*100:.1f}) & {s['e_other']*100:.1f} & {s['e_bg']*100:.1f} & {s['hit']*100:.1f} & {C['hard']['hard'][m]*100:.1f} & {C['hard']['other'][m]*100:.1f} \\\\")
open(T+'cam.tex','w').write('\n'.join(L))
for f in ['boot','bootpairs','mcnemar','errors','anova','anova_means','cam']: print(f); print(open(T+f+'.tex').read())
