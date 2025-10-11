# =============================================
# Script 2/2: clusterwise_lof_and_kmeans.py
# =============================================
#!/usr/bin/env python3
"""
Cluster-conditional LOF + whitened KMeans distance + optional GMM (non-DL)
- KMeans on train normals → per-cluster LOF models
- Whitened (within-cluster) Mahalanobis distance to centroid
- Optional GMM on train normals for -loglik score
- Rank-fusion of scores; tuned on valid-train; report on valid-holdout
- Writes Kaggle submission
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import List, Dict
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# Reuse same preprocessing as above (copy for standalone)

def detect_cols(df):
    s,t=[],[]
    for c in df.columns:
        if c in ['index','Id','target']: continue
        if isinstance(c,str) and '_' in c and c.split('_',1)[0].isdigit(): t.append(c)
        else: s.append(c)
    return s,t

def group_temporal(temp_cols: List[str]):
    g={}
    for c in temp_cols:
        idx,ft=c.split('_',1)
        g.setdefault(ft,[]).append(c)
    for k in g: g[k]=sorted(g[k], key=lambda x:int(x.split('_',1)[0]))
    return g

def per_loan_ffill_bfill(A: np.ndarray) -> np.ndarray:
    mask=np.isnan(A)
    idx=np.where(~mask, np.arange(A.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    ff=A[np.arange(A.shape[0])[:,None], idx]
    mask2=np.isnan(ff)
    idx2=np.where(~mask2, np.arange(ff.shape[1]), ff.shape[1]-1)
    np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
    bf=ff[np.arange(A.shape[0])[:,None], idx2]
    return np.nan_to_num(bf, nan=0.0)

def rank01(x: np.ndarray) -> np.ndarray:
    s=pd.Series(x)
    r=(s.rank(method='average')-1)/(len(s)-1+1e-12)
    return r.to_numpy()

class Preprocessor:
    def __init__(self):
        self.enc:Dict[str,LabelEncoder]={}
        self.imp:Dict[str,float]={}
        self.scaler=RobustScaler()
        self.final_static:List[str]=[]
        self.static_kept=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        self.temp_types=['CurrentActualUPB','EstimatedLTV','InterestBearingUPB','CurrentNonInterestBearingUPB']
    def fit(self, df):
        data=df.copy()
        if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
        scols,tcols=detect_cols(data)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in data.columns: data[col]=data[col].replace(code, np.nan)
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
        core=self.static_kept+['credit_ltv_ratio','dti_credit_ratio','monthly_rate','payment_burden']
        self.final_static=[c for c in core if c in data.columns and data[c].nunique()>1]
        Xs=self.scaler.fit_transform(data[self.final_static].to_numpy(float))
        return self
    def transform(self, df):
        data=df.copy()
        if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
        scols,tcols=detect_cols(data)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in data.columns: data[col]=data[col].replace(code, np.nan)
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
        _,tcols=detect_cols(data)
        g=group_temporal(tcols)
        feats=[]
        for f in self.temp_types:
            if f in g:
                arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
                first=np.abs(arr[:,0])+1.0
                mean_abs=np.abs(arr.mean(axis=1))+1.0
                trend=(arr[:,-1]-arr[:,0])/first
                vol=arr.std(axis=1)/mean_abs
                recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
                feats.append(arr[:,::3]); feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None])
        Xtemp=np.concatenate(feats, axis=1) if feats else np.zeros((len(data),1))
        return np.hstack([Xs, Xtemp])

# ---------------------- cluster-wise modeling ----------------------

def whitened_distance(x: np.ndarray, center: np.ndarray, cov: np.ndarray) -> float:
    # Mahalanobis distance with pseudo-inverse for stability
    try:
        inv=np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv=np.linalg.pinv(cov)
    d=x-center
    return float(np.sqrt(d @ inv @ d.T))


def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    pp=Preprocessor().fit(train)
    Xtr=pp.transform(train)

    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr=pp.transform(vtr); Xvho=pp.transform(vho)

    # tune KMeans k on valid-train
    best_auc, best_k=-1,5
    for k in [5,8,12]:
        km=KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xtr)
        labels=km.predict(Xvtr)
        # score by distance to nearest centroid
        d=np.linalg.norm(Xvtr - km.cluster_centers_[labels], axis=1)
        auc=roc_auc_score(vtr['target'], d)
        if auc>best_auc: best_auc, best_k=auc,k
    km=KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xtr)
    print(f'[Clusterwise] selected k={best_k}')

    # build per-cluster LOF and covariance
    clusters={}
    for c in range(best_k):
        idx=(km.labels_==c)
        Xc=Xtr[idx]
        if Xc.shape[0] < 20:
            continue
        # LOF: tune k locally using valid-train restricted to this cluster assignment
        best_klof, best_auc_loc=20,-1
        for klof in [10,20,35,50]:
            lof=LocalOutlierFactor(n_neighbors=klof, novelty=True, contamination='auto').fit(Xc)
            # evaluate on valid-train points closest to this centroid
            lab_v=np.argmin(((Xvtr[:,None,:]-km.cluster_centers_[None,:,:])**2).sum(axis=2), axis=1)
            sel=(lab_v==c)
            if sel.sum()==0: continue
            s=-lof.score_samples(Xvtr[sel])
            auc=roc_auc_score(vtr['target'].to_numpy()[sel], s)
            if auc>best_auc_loc: best_auc_loc, best_klof=auc,klof
        # store
        lof=LocalOutlierFactor(n_neighbors=best_klof, novelty=True, contamination='auto').fit(Xc)
        center=km.cluster_centers_[c]
        cov=np.cov(Xc.T) + 1e-6*np.eye(Xc.shape[1])
        clusters[c]={'lof':lof,'center':center,'cov':cov}

    # Optional GMM on normals
    best_m, best_bic=None, np.inf
    for m in [3,5,8]:
        try:
            gmm=GaussianMixture(n_components=m, covariance_type='full', random_state=42).fit(Xtr)
            if gmm.bic(Xtr) < best_bic:
                best_bic, best_m = gmm.bic(Xtr), m
        except Exception:
            pass
    gmm = GaussianMixture(n_components=best_m, covariance_type='full', random_state=42).fit(Xtr) if best_m else None

    # scoring on holdout
    labs=np.argmin(((Xvho[:,None,:]-km.cluster_centers_[None,:,:])**2).sum(axis=2), axis=1)
    s_lof=np.zeros(len(Xvho)); s_wh=np.zeros(len(Xvho)); s_gmm=np.zeros(len(Xvho))
    for i,(x,c) in enumerate(zip(Xvho, labs)):
        if c in clusters:
            s_lof[i]=-clusters[c]['lof'].score_samples(x[None,:])[0]
            s_wh[i]=whitened_distance(x, clusters[c]['center'], clusters[c]['cov'])
        else:
            s_lof[i]=0.0; s_wh[i]=0.0
        if gmm is not None:
            s_gmm[i]=-gmm.score_samples(x[None,:])[0]
    # rank-fusion (weights can be tuned on valid-train; here equal weights)
    S=np.vstack([rank01(s_lof), rank01(s_wh), rank01(s_gmm) if gmm is not None else rank01(np.zeros_like(s_lof))]).T
    w=np.array([0.45, 0.35, 0.20]) if gmm is not None else np.array([0.6,0.4,0.0])
    s_ens=(S * w).sum(axis=1)/max(w.sum(),1e-12)
    auprc=average_precision_score(vho['target'], s_ens)
    auc=roc_auc_score(vho['target'], s_ens)
    print(f'[Clusterwise LOF + Whitened KMeans (+GMM)] HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

    # submission
    try:
        test=pd.read_csv('Data/loans_test.csv')
        Xt=pp.transform(test)
        labs_t=np.argmin(((Xt[:,None,:]-km.cluster_centers_[None,:,:])**2).sum(axis=2), axis=1)
        s_lof_t=np.zeros(len(Xt)); s_wh_t=np.zeros(len(Xt)); s_gmm_t=np.zeros(len(Xt))
        for i,(x,c) in enumerate(zip(Xt, labs_t)):
            if c in clusters:
                s_lof_t[i]=-clusters[c]['lof'].score_samples(x[None,:])[0]
                s_wh_t[i]=whitened_distance(x, clusters[c]['center'], clusters[c]['cov'])
            if gmm is not None:
                s_gmm_t[i]=-gmm.score_samples(x[None,:])[0]
        S_t=np.vstack([rank01(s_lof_t), rank01(s_wh_t), rank01(s_gmm_t) if gmm is not None else rank01(np.zeros_like(s_lof_t))]).T
        s_test=(S_t * w).sum(axis=1)/max(w.sum(),1e-12)
        id_col='Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({id_col:test[id_col], 'anomaly_score':s_test}).to_csv('SUB_clusterwise_lof_kmeans.csv', index=False)
        print('Saved SUB_clusterwise_lof_kmeans.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()
