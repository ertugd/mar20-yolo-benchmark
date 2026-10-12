from common import *
import os; os.chdir('/home/claude/work')
gt=load_gt()
for m in MODELS:
    P=load_pred(m); S=per_image_stats(gt,P)
    m50,m95,out=compute_map(S,list(gt))
    print(m, round(m50*100,2), round(m95*100,2), 'npred',sum(len(v) for v in P.values()), 'P',out[2].mean().round(4),'R',out[3].mean().round(4))
