import numpy as np, pandas as pd, sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
P=sys.argv[1]
X=np.load(P+'/features.npy',mmap_mode='r')
X=np.array(X)  # 200MB
lr=np.log(X[:,:,3]); lr_ret=np.diff(lr,axis=1,prepend=np.nan)
def feats(df):
    s=df.security_index.values; t=df.end_index.values
    cols={}
    for k in range(1,13): cols[f'r{k}']=lr_ret[s,t-k+1]
    for w in (6,12,39,77): cols[f'cum{w}']=lr[s,t]-lr[s,t-w]
    o,h,l,c,v=[X[s,t,i] for i in range(5)]
    cols['pos']=(c-l)/np.maximum(h-l,1e-12); cols['rng']=np.log(h/l)
    vol=X[:,:,4]
    cols['vrel']=np.log(v/np.nanmean(np.stack([vol[s,t-k] for k in range(1,78)]),axis=0))
    rv=np.nanstd(np.stack([lr_ret[s,t-k] for k in range(0,77)]),axis=0)
    cols['rv']=rv; cols['r1z']=cols['r1']/np.maximum(rv,1e-9)
    cols['tod']=(t%78); cols['logp']=np.log(c)
    return pd.DataFrame(cols)
tr=pd.read_parquet(P+'/train.parquet'); va=pd.read_parquet(P+'/validation.parquet'); te=pd.read_parquet(P+'/test.parquet')
tr=tr.sample(400000,random_state=0)
Ftr,Fva,Fte=feats(tr),feats(va),feats(te)
for name,m in [('logit',LogisticRegression(max_iter=500,C=1.0)),('hgb',HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=500))]:
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    mm=make_pipeline(StandardScaler(),m) if name=='logit' else m
    mm.fit(Ftr.fillna(0),tr.label)
    for nm,F,d in [('val',Fva,va),('test',Fte,te)]:
        p=mm.predict_proba(F.fillna(0))[:,1]
        print(name,nm,'AUC=%.4f acc=%.4f prior=%.4f'%(roc_auc_score(d.label,p),accuracy_score(d.label,p>=.5),d.label.mean()))
    if name=='logit':
        print(dict(zip(Ftr.columns,np.round(mm[-1].coef_[0],4))))
# last-bar reversal by time-of-day & size of last move
va2=va.assign(r1=Fva.r1.values,tod=Fva.tod.values)
print('val: P(up | last bar down) =',va2[va2.r1<0].label.mean(),' P(up|last up)=',va2[va2.r1>0].label.mean())
print('val tod==0 (target is first bar? ) count',(va2.tod==77).sum())
