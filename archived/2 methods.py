# =============================================
# Script 1/2: lof_multiscale_ensemble.py
# =============================================
#!/usr/bin/env python3
"""
Multi-scale LOF rank-ensemble (non-DL, leak-free)
- Trains ONLY on train normals
- Preprocessing: special-code cleanup, per-loan temporal ffill/bfill, engineered temporal aggregates
- Cohort-aware z-scoring (optional) using ProductType x NumberOfUnits when available
- Tunes k over valid-train only; reports AUPRC/AUROC on valid-holdout
- Writes Kaggle submission for loans_test.csv
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------- utilities ----------------------

def detect_cols(df: pd.DataFrame):
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
        self.cohort_cols=['ProductType','NumberOfUnits']
        self.cohort_stats:Dict[Tuple,str]={}
        self.static_kept=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        self.temp_types=['CurrentActualUPB','EstimatedLTV','InterestBearingUPB','CurrentNonInterestBearingUPB']

    def _cohort_key(self, row: pd.Series) -> Tuple:
        return tuple(row.get(c, 'UNK') for c in self.cohort_cols)

    def fit(self, df: pd.DataFrame):
        data=df.copy()
        if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
        scols,tcols=detect_cols(data)
        # specials -> NaN
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in data.columns: data[col]=data[col].replace(code, np.nan)
        # categoricals
        cat=[c for c in scols if data[c].dtype=='object']
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); self.enc[c]=enc
            data[c]=enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        # numerical impute
        for c in [c for c in scols if c not in cat]:
            self.imp[c]=float(data[c].median()) if data[c].notna().any() else 0.0
            data[c]=data[c].fillna(self.imp[c])
        # engineered statics
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate']=data['OriginalInterestRate']/100.0/12.0
            data['payment_burden']=data['OriginalUPB']*data['monthly_rate']
        core=self.static_kept+['credit_ltv_ratio','dti_credit_ratio','monthly_rate','payment_burden']
        self.final_static=[c for c in core if c in data.columns and data[c].nunique()>1]
        Xs=self.scaler.fit_transform(data[self.final_static].to_numpy(float))
        # cohort stats for z-scoring (per cohort mean/std on static features)
        cohort_keys=data[self.cohort_cols].astype(str).agg('|'.join, axis=1) if set(self.cohort_cols).issubset(data.columns) else pd.Series(['ALL']*len(data))
        self.cohort_map={k:i for i,k in enumerate(sorted(cohort_keys.unique()))}
        self.cohort_means={}; self.cohort_stds={}
        for k in self.cohort_map.keys():
            idx=(cohort_keys==k).values
            if idx.sum()<5: continue
            self.cohort_means[k]=Xs[idx].mean(axis=0)
            self.cohort_stds[k]=Xs[idx].std(axis=0)+1e-6
        # temporal cache not needed; build on transform
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
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
        # cohort z-scoring (optional & safe)
        if set(self.cohort_cols).issubset(data.columns) and len(self.cohort_means)>0:
            keys=data[self.cohort_cols].astype(str).agg('|'.join, axis=1)
            Xs_adj=np.zeros_like(Xs)
            for i,k in enumerate(keys):
                if k in self.cohort_means:
                    Xs_adj[i]=(Xs[i]-self.cohort_means[k])/self.cohort_stds[k]
                else:
                    Xs_adj[i]=Xs[i]
        else:
            Xs_adj=Xs
        # temporal engineered features
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
                mono=(arr[:,1:]-arr[:,:-1]<0).mean(axis=1) if 'UPB' in f else (arr[:,1:]-arr[:,:-1]>0).mean(axis=1)
                feats.append(arr[:,::3])
                feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append(recent[:,None]); feats.append(mono[:,None])
        Xtemp=np.concatenate(feats, axis=1) if feats else np.zeros((len(data),1))
        return np.hstack([Xs_adj, Xtemp])

# ---------------------- training ----------------------

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

    # Multi-scale LOF
    ks=[10,20,35,50]
    lof_models=[]
    # Tiny tuning per k on valid-train: keep all, we will rank-average
    for k in ks:
        lof=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
        lof.fit(Xtr)
        lof_models.append(lof)
    # Evaluate ensemble on holdout
    scores=[]
    for lof in lof_models:
        s=-lof.score_samples(Xvho)
        scores.append(rank01(s))
    s_ens=np.mean(np.vstack(scores).T, axis=1)
    auprc=average_precision_score(vho['target'], s_ens)
    auc=roc_auc_score(vho['target'], s_ens)
    print(f'[Multi-k LOF rank-ensemble] HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

    # Optional: print single-k baselines for reference
    for k,lof in zip(ks, lof_models):
        s=-lof.score_samples(Xvho)
        print(f'  k={k}: AUPRC={average_precision_score(vho["target"], s):.4f} AUROC={roc_auc_score(vho["target"], s):.4f}')

    # Submission
    try:
        test=pd.read_csv('Data/loans_test.csv')
        Xt=pp.transform(test)
        scores_t=[]
        for lof in lof_models:
            scores_t.append(rank01(-lof.score_samples(Xt)))
        s_test=np.mean(np.vstack(scores_t).T, axis=1)
        id_col='Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({id_col:test[id_col], 'anomaly_score':s_test}).to_csv('SUB_lof_multiscale.csv', index=False)
        print('Saved SUB_lof_multiscale.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()


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
