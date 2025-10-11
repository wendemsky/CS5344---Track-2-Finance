# =============================================
# Script 1/2: lof_pca_tuned.py
# =============================================
#!/usr/bin/env python3
"""
LOF after PCA (leak-free, non-DL) with fine k-grid and optional whitening
- Fit PCA on train normals to retain 90–95% variance (cap components)
- Transform valid/test with the same PCA; LOF runs in PCA space
- Tiny search over variance target, n_components cap, and k
- Reports AUPRC (primary) and AUROC on a valid holdout
- Writes SUB_lof_pca_tuned.csv for test
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# ------------------ utilities ------------------

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
        d=df.copy()
        if 'Id' in d.columns and 'index' not in d.columns: d['index']=d['Id']
        sc,tc=detect_cols(d)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in d.columns: d[col]=d[col].replace(code, np.nan)
        cat=[c for c in sc if d[c].dtype=='object']
        for c in cat:
            enc=LabelEncoder(); vals=d[c].fillna('MISSING').astype(str).unique().tolist();
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); self.enc[c]=enc
            d[c]=enc.transform(d[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in sc if c not in cat]:
            self.imp[c]=float(d[c].median()) if d[c].notna().any() else 0.0
            d[c]=d[c].fillna(self.imp[c])
        if set(['CreditScore','OriginalLTV']).issubset(d.columns): d['credit_ltv_ratio']=d['CreditScore']/(d['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(d.columns): d['dti_credit_ratio']=d['OriginalDTI']/(d['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(d.columns):
            d['monthly_rate']=d['OriginalInterestRate']/100.0/12.0; d['payment_burden']=d['OriginalUPB']*d['monthly_rate']
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers','credit_ltv_ratio','dti_credit_ratio','monthly_rate','payment_burden']
        self.final_static=[c for c in core if c in d.columns and d[c].nunique()>1]
        Xs=self.scaler.fit_transform(d[self.final_static].to_numpy(float))
        g=group_temporal(tc)
        feats=[]
        for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
            if f in g:
                arr=per_loan_ffill_bfill(d[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first; vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(d),1))
        return np.hstack([Xs,Xtemp])
    def transform(self, df):
        d=df.copy()
        if 'Id' in d.columns and 'index' not in d.columns: d['index']=d['Id']
        sc,tc=detect_cols(d)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in d.columns: d[col]=d[col].replace(code, np.nan)
        for c,enc in self.enc.items():
            d[c]=d[c].fillna('MISSING').astype(str)
            d[c]=d[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            d[c]=enc.transform(d[c])
        for c,val in self.imp.items():
            if c in d.columns: d[c]=d[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(d.columns): d['credit_ltv_ratio']=d['CreditScore']/(d['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(d.columns): d['dti_credit_ratio']=d['OriginalDTI']/(d['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(d.columns): d['monthly_rate']=d['OriginalInterestRate']/100.0/12.0; d['payment_burden']=d['OriginalUPB']*d['monthly_rate']
        Xs=self.scaler.transform(d[self.final_static].to_numpy(float))
        g=group_temporal(tc)
        feats=[]
        for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
            if f in g:
                arr=per_loan_ffill_bfill(d[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first; vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(d),1))
        return np.hstack([Xs,Xtemp])

# ------------------ training ------------------

def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    pp=PP(); Xtr=pp.fit(train)
    y=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,y):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr=pp.transform(vtr); Xvho=pp.transform(vho)

    # PCA search: variance target and cap
    var_targets=[0.90, 0.95]
    max_caps=[30, 40, 50]
    best=(None,None,None,-1.0)
    for vt in var_targets:
        pfull=PCA().fit(Xtr)
        cum=np.cumsum(pfull.explained_variance_ratio_)
        nc=int(np.argmax(cum>=vt))+1
        for cap in max_caps:
            n_use=max(2, min(nc, cap))
            pca=PCA(n_components=n_use).fit(Xtr)
            # whiten (standardize PCA scores)
            scaler_pca=StandardScaler(with_mean=False)  # PCA already centered
            Ztr=scaler_pca.fit_transform(pca.transform(Xtr))
            Zv=scaler_pca.transform(pca.transform(Xvtr))
            # fine k grid
            for k in [5,7,9,11,13,15]:
                lof=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
                lof.fit(Ztr)
                s=-lof.score_samples(Zv)
                ap=average_precision_score(vtr['target'], s)
                if ap>best[3]:
                    best=(vt, n_use, k, ap)
    vt_best, n_best, k_best, ap_best = best
    print(f'[Tuning] best var={vt_best}, n_comp={n_best}, k={k_best}, valid-train AUPRC={ap_best:.4f}')

    # Train final with best on full train
    pca=PCA(n_components=n_best).fit(Xtr)
    scaler_pca=StandardScaler(with_mean=False)
    Ztr=scaler_pca.fit_transform(pca.transform(Xtr))
    Zvho=scaler_pca.transform(pca.transform(Xvho))
    lof=LocalOutlierFactor(n_neighbors=k_best, novelty=True, contamination='auto').fit(Ztr)
    s_hold=-lof.score_samples(Zvho)
    auprc=average_precision_score(vho['target'], s_hold)
    auc=roc_auc_score(vho['target'], s_hold)
    print(f'[LOF-PCA] HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

    # submission
    try:
        test=pd.read_csv('Data/loans_test.csv')
        Xt=pp.transform(test)
        Zt=scaler_pca.transform(pca.transform(Xt))
        s_test=-lof.score_samples(Zt)
        s_test=(s_test - s_test.min())/(s_test.max()-s_test.min()+1e-12)
        idc='Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({idc:test[idc], 'anomaly_score':s_test}).to_csv('SUB_lof_pca_tuned.csv', index=False)
        print('Saved SUB_lof_pca_tuned.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()


# =============================================
# Script 2/2: lof_weighted_multik.py
# =============================================
#!/usr/bin/env python3
"""
Weighted multi-k LOF (rank-fusion) 
- Learns weights from valid-train AUPRC, applies to valid-holdout
- Optionally uses max-rank fusion (good for early precision)
- Reports AUPRC/AUROC on holdout and writes SUB_lof_weighted_multik.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# --------- reuse simple preprocessing (same as above, without PCA) ---------

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

def rank01(x):
    s=pd.Series(x); r=(s.rank(method='average')-1)/(len(s)-1+1e-12); return r.to_numpy()

class PP:
    def __init__(self):
        self.enc:Dict[str,LabelEncoder]={}
        self.imp:Dict[str,float]={}
        self.scaler=RobustScaler()
        self.final_static:List[str]=[]
    def fit(self, df):
        d=df.copy()
        if 'Id' in d.columns and 'index' not in d.columns: d['index']=d['Id']
        sc,tc=detect_cols(d)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in d.columns: d[col]=d[col].replace(code, np.nan)
        cat=[c for c in sc if d[c].dtype=='object']
        for c in cat:
            enc=LabelEncoder(); vals=d[c].fillna('MISSING').astype(str).unique().tolist();
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); self.enc[c]=enc
            d[c]=enc.transform(d[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in sc if c not in cat]:
            self.imp[c]=float(d[c].median()) if d[c].notna().any() else 0.0
            d[c]=d[c].fillna(self.imp[c])
        if set(['CreditScore','OriginalLTV']).issubset(d.columns): d['credit_ltv_ratio']=d['CreditScore']/(d['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(d.columns): d['dti_credit_ratio']=d['OriginalDTI']/(d['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(d.columns):
            d['monthly_rate']=d['OriginalInterestRate']/100.0/12.0; d['payment_burden']=d['OriginalUPB']*d['monthly_rate']
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers','credit_ltv_ratio','dti_credit_ratio','monthly_rate','payment_burden']
        self.final_static=[c for c in core if c in d.columns and d[c].nunique()>1]
        Xs=self.scaler.fit_transform(d[self.final_static].to_numpy(float))
        g=group_temporal(tc)
        feats=[]
        for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
            if f in g:
                arr=per_loan_ffill_bfill(d[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first; vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(d),1))
        return np.hstack([Xs,Xtemp])
    def transform(self, df):
        d=df.copy()
        if 'Id' in d.columns and 'index' not in d.columns: d['index']=d['Id']
        sc,tc=detect_cols(d)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in d.columns: d[col]=d[col].replace(code, np.nan)
        for c,enc in self.enc.items():
            d[c]=d[c].fillna('MISSING').astype(str)
            d[c]=d[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            d[c]=enc.transform(d[c])
        for c,val in self.imp.items():
            if c in d.columns: d[c]=d[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(d.columns): d['credit_ltv_ratio']=d['CreditScore']/(d['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(d.columns): d['dti_credit_ratio']=d['OriginalDTI']/(d['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(d.columns): d['monthly_rate']=d['OriginalInterestRate']/100.0/12.0; d['payment_burden']=d['OriginalUPB']*d['monthly_rate']
        Xs=self.scaler.transform(d[self.final_static].to_numpy(float))
        g=group_temporal(tc)
        feats=[]
        for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
            if f in g:
                arr=per_loan_ffill_bfill(d[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first; vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(d),1))
        return np.hstack([Xs,Xtemp])

# ------------------ training ------------------

def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    pp=PP(); Xtr=pp.fit(train)
    y=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,y):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr=pp.transform(vtr); Xvho=pp.transform(vho)

    ks=[5,7,9,10,11,13,15]
    models=[]; weights=[]
    # learn weights on valid-train by AUPRC
    ap_sum=0.0
    for k in ks:
        lof=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto').fit(Xtr)
        s=-lof.score_samples(Xvtr)
        ap=average_precision_score(vtr['target'], s)
        models.append(lof); weights.append(ap)
        ap_sum+=ap
        print(f'[valid-train] k={k} AUPRC={ap:.4f}')
    # normalize weights (avoid zero)
    w=np.array(weights); w = w / (w.sum()+1e-12)

    # score holdout via rank-fusion
    R=[]
    for lof in models:
        R.append(rank01(-lof.score_samples(Xvho)))
    R=np.vstack(R).T
    s_w=(R * w).sum(axis=1)

    # also compute max-rank fusion
    s_max=R.max(axis=1)

    for name,scores in [('WeightedRank', s_w), ('MaxRank', s_max)]:
        auprc=average_precision_score(vho['target'], scores)
        auc=roc_auc_score(vho['target'], scores)
        print(f'[Multi-k LOF {name}] HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

    # choose the better of the two for submission
    use_scores = s_w if average_precision_score(vho['target'], s_w) >= average_precision_score(vho['target'], s_max) else s_max

    try:
        test=pd.read_csv('Data/loans_test.csv')
        Xt=pp.transform(test)
        Rt=[]
        for lof in models:
            Rt.append(rank01(-lof.score_samples(Xt)))
        Rt=np.vstack(Rt).T
        st=(Rt * w).sum(axis=1) if use_scores is s_w else Rt.max(axis=1)
        idc='Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({idc:test[idc], 'anomaly_score':st}).to_csv('SUB_lof_weighted_multik.csv', index=False)
        print('Saved SUB_lof_weighted_multik.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()
