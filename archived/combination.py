#!/usr/bin/env python3
"""
Simple rank-ensemble (AP-first) of classic detectors (non-DL)
- Trains all detectors on train normals
- Tunes tiny grids on valid-train; reports on valid-holdout
- Combines IF, LOF, PCA-Rec, KMeansDist via rank-average (equal weights)
- Prints AUPRC (primary) and AUROC (secondary)
- Writes Kaggle submission for test set
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ------------------ shared preprocessing ------------------

def detect_cols(df):
    s,t=[],[]
    for c in df.columns:
        if c in ['index','Id','target']: continue
        if isinstance(c,str) and '_' in c and c.split('_',1)[0].isdigit(): t.append(c)
        else: s.append(c)
    return s,t

def group_temporal(cols):
    g={}
    for c in cols:
        _,ft=c.split('_',1); g.setdefault(ft,[]).append(c)
    for k in g: g[k]=sorted(g[k], key=lambda x:int(x.split('_',1)[0]))
    return g

def per_loan_ffill_bfill(A):
    mask=np.isnan(A); idx=np.where(~mask, np.arange(A.shape[1]), 0)
    np.maximum.accumulate(idx,axis=1,out=idx)
    ff=A[np.arange(A.shape[0])[:,None],idx]
    mask2=np.isnan(ff); idx2=np.where(~mask2, np.arange(ff.shape[1]), ff.shape[1]-1)
    np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
    bf=ff[np.arange(A.shape[0])[:,None],idx2]
    return np.nan_to_num(bf, nan=0.0)

class PP:
    def __init__(self):
        self.enc:Dict[str,LabelEncoder]={}
        self.imp:Dict[str,float]={}
        self.scaler=RobustScaler()
        self.final_static:List[str]=[]
    def fit(self, df):
        data=df.copy()
        if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
        scols,tcols=detect_cols(data)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in data.columns: data[col]=data[col].replace(code,np.nan)
        cat=[c for c in scols if data[c].dtype=='object']
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); self.enc[c]=enc
            data[c]=enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in scols if c not in cat]:
            self.imp[c]=float(data[c].median()) if data[c].notna().any() else 0.0
            data[c]=data[c].fillna(self.imp[c])
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate']=data['OriginalInterestRate']/100.0/12.0
            data['payment_burden']=data['OriginalUPB']*data['monthly_rate']
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers','credit_ltv_ratio','dti_credit_ratio','monthly_rate','payment_burden']
        self.final_static=[c for c in core if c in data.columns and data[c].nunique()>1]
        Xs=self.scaler.fit_transform(data[self.final_static].to_numpy(float))
        g=group_temporal(tcols)
        feats=[]
        for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
            if f in g:
                arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first
                vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(data),1))
        return np.hstack([Xs,Xtemp])
    def transform(self, df):
        data=df.copy()
        if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
        scols,tcols=detect_cols(data)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in data.columns: data[col]=data[col].replace(code,np.nan)
        for c,enc in self.enc.items():
            data[c]=data[c].fillna('MISSING').astype(str)
            data[c]=data[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            data[c]=enc.transform(data[c])
        for c,val in self.imp.items():
            if c in data.columns: data[c]=data[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate']=data['OriginalInterestRate']/100.0/12.0
            data['payment_burden']=data['OriginalUPB']*data['monthly_rate']
        Xs=self.scaler.transform(data[self.final_static].to_numpy(float))
        g=group_temporal(tcols)
        feats=[]
        for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
            if f in g:
                arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first
                vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(data),1))
        return np.hstack([Xs,Xtemp])

# ------------------ detectors ------------------

def rank01(x):
    r = pd.Series(x).rank(method='average')
    r = (r - r.min())/(r.max()-r.min()+1e-12)
    return r.to_numpy()

def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    pp=PP(); Xtr=pp.fit(train)
    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr=pp.transform(vtr); Xvho=pp.transform(vho)

    # IF
    best_auc,bn=-1,400
    for n in [300,400,600]:
        IF=IsolationForest(n_estimators=n, contamination='auto', random_state=42, n_jobs=-1).fit(Xtr)
        s=-IF.score_samples(Xvtr)
        a=roc_auc_score(vtr['target'], s)
        if a>best_auc: best_auc,bn=a,n
    IF=IsolationForest(n_estimators=bn, contamination='auto', random_state=42, n_jobs=-1).fit(Xtr)
    s_if=-IF.score_samples(Xvho)

    # LOF
    best_auc,bk=-1,20
    for k in [10,20,30]:
        lof=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto').fit(Xtr)
        s=-lof.score_samples(Xvtr)
        a=roc_auc_score(vtr['target'], s)
        if a>best_auc: best_auc,bk=a,k
    lof=LocalOutlierFactor(n_neighbors=bk, novelty=True, contamination='auto').fit(Xtr)
    s_lof=-lof.score_samples(Xvho)

    # PCA-Rec
    pfull=PCA().fit(Xtr)
    cum=np.cumsum(pfull.explained_variance_ratio_)
    nc=max(2, int(np.argmax(cum>=0.95))+1)
    pca=PCA(n_components=nc).fit(Xtr)
    Xh=pca.inverse_transform(pca.transform(Xvho))
    s_pca=np.mean((Xvho-Xh)**2, axis=1)

    # KMeansDist
    best_auc,bk2=-1,8
    for k in [5,8,12]:
        km=KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xtr)
        centers=km.cluster_centers_
        d2=np.min(((Xvtr[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1)
        a=roc_auc_score(vtr['target'], np.sqrt(d2))
        if a>best_auc: best_auc,bk2=a,k
    km=KMeans(n_clusters=bk2, n_init=10, random_state=42).fit(Xtr)
    centers=km.cluster_centers_
    d2=np.min(((Xvho[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1)
    s_km=np.sqrt(d2)

    # Rank-average ensemble
    S = np.vstack([rank01(s_if), rank01(s_lof), rank01(s_pca), rank01(s_km)]).T
    s_ens = S.mean(axis=1)

    # Metrics
    auprc = average_precision_score(vho['target'], s_ens)
    auc = roc_auc_score(vho['target'], s_ens)
    print(f'[Rank-Ensemble IF+LOF+PCA+KMeans] HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

    # Test submission
    try:
        test=pd.read_csv('Data/loans_test.csv')
        Xt=pp.transform(test)
        s_if_test=-IF.score_samples(Xt)
        s_lof_test=-lof.score_samples(Xt)
        Xh_t=pca.inverse_transform(pca.transform(Xt)); s_pca_test=np.mean((Xt-Xh_t)**2, axis=1)
        d2_t=np.min(((Xt[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1); s_km_test=np.sqrt(d2_t)
        S_t=np.vstack([rank01(s_if_test), rank01(s_lof_test), rank01(s_pca_test), rank01(s_km_test)]).T
        s_test=S_t.mean(axis=1)
        id_col='Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({id_col:test[id_col], 'anomaly_score':s_test}).to_csv('SUB_rank_ensemble.csv', index=False)
        print('Saved SUB_rank_ensemble.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()
