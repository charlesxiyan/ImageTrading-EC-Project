import numpy as np, pandas as pd, sys, runpy
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
P=sys.argv[1]
X=np.array(np.load(P+'/features.npy',mmap_mode='r')); V=np.load(P+'/valid.npy')
lr=np.log(X[:,:,3]); ret=np.diff(lr,axis=1,prepend=np.nan)
T=lr.shape[1]
def feats(s,t):
    c={f'r{k}':ret[s,t-k+1] for k in range(1,13)}
    for w in (6,12,39,77): c[f'cum{w}']=lr[s,t]-lr[s,t-w]
    o,h,l,cl,v=[X[s,t,i] for i in range(5)]
    c['pos']=(cl-l)/np.maximum(h-l,1e-12); c['rng']=np.log(h/l)
    c['rv']=np.nanstd(np.stack([ret[s,t-k] for k in range(77)]),axis=0)
    c['absr']=np.nanmean(np.abs(np.stack([ret[s,t-k] for k in range(12)])),axis=0)
    c['tod']=t%78
    return pd.DataFrame(c).fillna(0)
tr=pd.read_parquet(P+'/train.parquet').sample(300000,random_state=1)
va=pd.read_parquet(P+'/validation.parquet').sample(200000,random_state=1)
Ftr=feats(tr.security_index.values,tr.end_index.values); Fva=feats(va.security_index.values,va.end_index.values)
def target(df,h,kind):
    s,t=df.security_index.values,df.end_index.values; t2=np.minimum(t+h,T-1)
    r=lr[s,t2]-lr[s,t]; ok=np.isfinite(r)&(t+h<T)&(r!=0)
    if kind=='dir': y=(r>0)
    else:
        y=np.abs(r)
    return y,ok
for h in (1,6,12,78):
    for kind in ('dir','vol'):
        ytr,oktr=target(tr,h,kind); yva,okva=target(va,h,kind)
        if kind=='vol':
            # big-move vs small-move: above per-stock train median |r|
            med=pd.Series(ytr[oktr]).groupby(tr.security_index.values[oktr]).median()
            ytr=ytr>med.reindex(tr.security_index.values).values; yva=yva>med.reindex(va.security_index.values).values
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,min_samples_leaf=500)
        m.fit(Ftr[oktr],ytr[oktr]); p=m.predict_proba(Fva[okva])[:,1]
        print(f'h={h:3d} bars  {kind:3s}  val AUC={roc_auc_score(yva[okva],p):.4f}  n={okva.sum()}')
