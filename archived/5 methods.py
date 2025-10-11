# =============================================
# Script 1/5: isolation_forest_baseline.py
# =============================================
#!/usr/bin/env python3
"""
Isolation Forest baseline (leak-free, non-DL)
- Fits ONLY on train normals
- Tunes lightweight params on valid-train; reports on valid-holdout
- Prints AUPRC (primary) and AUROC (secondary)
- Writes optional Kaggle submission for test set
"""
import warnings; warnings.filterwarnings('ignore')
from typing import Tuple, Dict, List
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# ---------- utils (minimal, repeated per script for standalone) ----------

def detect_cols(df: pd.DataFrame):
    s, t = [], []
    for c in df.columns:
        if c in ['index','Id','target']: continue
        if isinstance(c,str) and '_' in c and c.split('_',1)[0].isdigit(): t.append(c)
        else: s.append(c)
    return s, t

def group_temporal(temp_cols: List[str]):
    g = {}
    for c in temp_cols:
        _,ft = c.split('_',1)
        g.setdefault(ft,[]).append(c)
    for k in g: g[k] = sorted(g[k], key=lambda x:int(x.split('_',1)[0]))
    return g

def per_loan_ffill_bfill(A: np.ndarray) -> np.ndarray:
    mask = np.isnan(A)
    idx = np.where(~mask, np.arange(A.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    ff = A[np.arange(A.shape[0])[:,None], idx]
    mask2 = np.isnan(ff)
    idx2 = np.where(~mask2, np.arange(ff.shape[1]), ff.shape[1]-1)
    np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
    bf = ff[np.arange(ff.shape[0])[:,None], idx2]
    return np.nan_to_num(bf, nan=0.0)

def preprocess(df: pd.DataFrame, fit: bool, state: Dict=None):
    if state is None:
        state = {'enc':{}, 'imp':{}, 'scaler':RobustScaler(), 'final_static':[]}
    data = df.copy()
    if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
    scols, tcols = detect_cols(data)
    # specials -> NaN
    for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
        if col in data.columns: data[col] = data[col].replace(code, np.nan)
    # categorical
    cat = [c for c in scols if data[c].dtype=='object']
    if fit:
        for c in cat:
            enc = LabelEncoder(); vals = data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); state['enc'][c]=enc
            data[c] = enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        # numerical impute
        for c in [c for c in scols if c not in cat]:
            state['imp'][c] = float(data[c].median()) if data[c].notna().any() else 0.0
            data[c] = data[c].fillna(state['imp'][c])
        # engineered statics
        new = []
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0); new.append('credit_ltv_ratio')
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0); new.append('dti_credit_ratio')
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate']=data['OriginalInterestRate']/100.0/12.0; new.append('monthly_rate')
            data['payment_burden']=data['OriginalUPB']*data['monthly_rate']; new.append('payment_burden')
        core = ['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        state['final_static'] = [c for c in core+new if c in data.columns and data[c].nunique()>1]
        Xs = state['scaler'].fit_transform(data[state['final_static']].to_numpy(float))
    else:
        for c,enc in state['enc'].items():
            data[c]=data[c].fillna('MISSING').astype(str)
            data[c]=data[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            data[c]=enc.transform(data[c])
        for c,val in state['imp'].items():
            if c in data.columns: data[c]=data[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate']=data['OriginalInterestRate']/100.0/12.0
            data['payment_burden']=data['OriginalUPB']*data['monthly_rate']
        Xs = state['scaler'].transform(data[state['final_static']].to_numpy(float))
    # temporal sampling (every 3rd) for key types
    g = group_temporal(tcols)
    feats = []
    for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
        if f in g:
            arr = per_loan_ffill_bfill(data[g[f]].to_numpy(float))
            # aggregates
            first = np.abs(arr[:,0])+1.0
            mean_abs = np.abs(arr.mean(axis=1))+1.0
            recent = arr[:,-1]-arr[:,max(arr.shape[1]-3,1)]
            trend = (arr[:,-1]-arr[:,0])/first
            vol = arr.std(axis=1)/mean_abs
            feats.append(arr[:,::3])
            feats.append(trend[:,None]); feats.append(vol[:,None]); feats.append((recent/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0))[:,None])
    Xtemp = np.concatenate(feats, axis=1) if feats else np.zeros((len(data),1))
    X = np.hstack([Xs, Xtemp])
    return X, state


def main():
    train = pd.read_csv('Data/loans_train.csv')
    valid = pd.read_csv('Data/loans_valid.csv')
    Xtr, st = preprocess(train, fit=True)
    yv = valid['target'].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri, hoi in sss.split(valid, yv):
        vtr = valid.iloc[tri]; vho = valid.iloc[hoi]
    Xvtr,_ = preprocess(vtr, fit=False, state=st)
    Xvho,_ = preprocess(vho, fit=False, state=st)

    # tiny tuning on valid-train (no leakage)
    grid = [300, 400, 600]
    best_auc, best_n = -1, grid[0]
    for n in grid:
        IF = IsolationForest(n_estimators=n, contamination='auto', random_state=42, n_jobs=-1)
        IF.fit(Xtr)
        s = -IF.score_samples(Xvtr)
        auc = roc_auc_score(vtr['target'], s)
        if auc > best_auc: best_auc, best_n = auc, n
    IF = IsolationForest(n_estimators=best_n, contamination='auto', random_state=42, n_jobs=-1).fit(Xtr)
    s_hold = -IF.score_samples(Xvho)
    auprc = average_precision_score(vho['target'], s_hold)
    auc = roc_auc_score(vho['target'], s_hold)
    print(f'[IsolationForest] best_n={best_n}  HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

    # optional test submission
    try:
        test = pd.read_csv('Data/loans_test.csv')
        Xt,_ = preprocess(test, fit=False, state=st)
        s_test = -IF.score_samples(Xt)
        s_test = (s_test - s_test.min())/(s_test.max()-s_test.min()+1e-12)
        id_col = 'Id' if 'Id' in test.columns else 'index'
        pd.DataFrame({id_col:test[id_col], 'anomaly_score':s_test}).to_csv('SUB_isolation_forest.csv', index=False)
        print('Saved SUB_isolation_forest.csv')
    except Exception as e:
        print('Test skipped:', e)

if __name__=='__main__':
    main()


# =============================================
# Script 2/5: lof_novelty_baseline.py
# =============================================
#!/usr/bin/env python3
"""LOF (novelty=True) baseline, leak-free."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# reuse utils
# (for brevity, identical implementations as script 1; in a real repo, factor into a shared module)

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

def preprocess(df, fit=False, state=None):
    if state is None: state={'enc':{},'imp':{},'scaler':RobustScaler(),'final_static':[]}
    data=df.copy()
    if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
    scols,tcols=detect_cols(data)
    for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
        if col in data.columns: data[col]=data[col].replace(code,np.nan)
    cat=[c for c in scols if data[c].dtype=='object']
    from sklearn.preprocessing import LabelEncoder
    if fit:
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); state['enc'][c]=enc
            data[c]=enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in scols if c not in cat]:
            state['imp'][c]=float(data[c].median()) if data[c].notna().any() else 0.0
            data[c]=data[c].fillna(state['imp'][c])
        new=[]
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0); new.append('credit_ltv_ratio')
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0); new.append('dti_credit_ratio')
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        state['final_static']=[c for c in core+new if c in data.columns and data[c].nunique()>1]
        Xs=state['scaler'].fit_transform(data[state['final_static']].to_numpy(float))
    else:
        for c,enc in state['enc'].items():
            data[c]=data[c].fillna('MISSING').astype(str)
            data[c]=data[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            data[c]=enc.transform(data[c])
        for c,val in state['imp'].items():
            if c in data.columns: data[c]=data[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        Xs=state['scaler'].transform(data[state['final_static']].to_numpy(float))
    g=group_temporal(tcols)
    feats=[]
    for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
        if f in g:
            arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
            feats.append(arr[:,::3])
    Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(data),1))
    X=np.hstack([Xs,Xtemp])
    return X, state


def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    Xtr, st=preprocess(train, fit=True)
    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr,_=preprocess(vtr, fit=False, state=st)
    Xvho,_=preprocess(vho, fit=False, state=st)

    best_auc, best_k = -1, None
    for k in [10, 20, 30]:
        lof=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
        lof.fit(Xtr)
        s = -lof.score_samples(Xvtr)
        auc = roc_auc_score(vtr['target'], s)
        if auc > best_auc: best_auc, best_k = auc, k
    lof=LocalOutlierFactor(n_neighbors=best_k, novelty=True, contamination='auto').fit(Xtr)
    s_hold = -lof.score_samples(Xvho)
    auprc = average_precision_score(vho['target'], s_hold)
    auc = roc_auc_score(vho['target'], s_hold)
    print(f'[LOF] best_k={best_k}  HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

if __name__=='__main__':
    main()


# =============================================
# Script 3/5: pca_reconstruction_baseline.py
# =============================================
#!/usr/bin/env python3
"""PCA reconstruction error baseline, leak-free."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# utils (same pattern)

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

def preprocess(df, fit=False, state=None):
    if state is None: state={'enc':{},'imp':{},'scaler':RobustScaler(),'final_static':[]}
    data=df.copy()
    if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
    scols,tcols=detect_cols(data)
    for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
        if col in data.columns: data[col]=data[col].replace(code,np.nan)
    cat=[c for c in scols if data[c].dtype=='object']
    if fit:
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); state['enc'][c]=enc
            data[c]=enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in scols if c not in cat]:
            state['imp'][c]=float(data[c].median()) if data[c].notna().any() else 0.0
            data[c]=data[c].fillna(state['imp'][c])
        new=[]
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0); new.append('credit_ltv_ratio')
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0); new.append('dti_credit_ratio')
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        state['final_static']=[c for c in core+new if c in data.columns and data[c].nunique()>1]
        Xs=state['scaler'].fit_transform(data[state['final_static']].to_numpy(float))
    else:
        for c,enc in state['enc'].items():
            data[c]=data[c].fillna('MISSING').astype(str)
            data[c]=data[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            data[c]=enc.transform(data[c])
        for c,val in state['imp'].items():
            if c in data.columns: data[c]=data[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        Xs=state['scaler'].transform(data[state['final_static']].to_numpy(float))
    g=group_temporal(tcols)
    feats=[]
    for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
        if f in g:
            arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
            feats.append(arr[:,::3])
    Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(data),1))
    X=np.hstack([Xs,Xtemp])
    return X, state


def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    Xtr, st = preprocess(train, fit=True)
    yv = valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr,_=preprocess(vtr, fit=False, state=st)
    Xvho,_=preprocess(vho, fit=False, state=st)

    # choose n_components for 95% var on train, then verify on valid-train AUC
    pfull = PCA().fit(Xtr)
    cum = np.cumsum(pfull.explained_variance_ratio_)
    n_comp = max(2, int(np.argmax(cum>=0.95))+1)
    pca = PCA(n_components=n_comp).fit(Xtr)
    # score on valid-train
    Xtr_v = pca.inverse_transform(pca.transform(Xvtr))
    s_v = np.mean((Xvtr-Xtr_v)**2, axis=1)
    auc_v = roc_auc_score(vtr['target'], s_v)
    # holdout
    Xh = pca.inverse_transform(pca.transform(Xvho))
    s_h = np.mean((Xvho-Xh)**2, axis=1)
    # normalize scores
    s_h = (s_h - s_h.min())/(s_h.max()-s_h.min()+1e-12)
    auprc = average_precision_score(vho['target'], s_h)
    auc = roc_auc_score(vho['target'], s_h)
    print(f'[PCA-Rec] n_comp={n_comp}  HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}  (valid-train AUC={auc_v:.4f})')

if __name__=='__main__':
    main()


# =============================================
# Script 4/5: kmeans_distance_baseline.py
# =============================================
#!/usr/bin/env python3
"""KMeans distance-to-centroid baseline, leak-free."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# utils as above

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

def preprocess(df, fit=False, state=None):
    if state is None: state={'enc':{},'imp':{},'scaler':RobustScaler(),'final_static':[]}
    data=df.copy()
    if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
    scols,tcols=detect_cols(data)
    for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
        if col in data.columns: data[col]=data[col].replace(code,np.nan)
    cat=[c for c in scols if data[c].dtype=='object']
    if fit:
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); state['enc'][c]=enc
            data[c]=enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in scols if c not in cat]:
            state['imp'][c]=float(data[c].median()) if data[c].notna().any() else 0.0
            data[c]=data[c].fillna(state['imp'][c])
        new=[]
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0); new.append('credit_ltv_ratio')
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0); new.append('dti_credit_ratio')
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        state['final_static']=[c for c in core+new if c in data.columns and data[c].nunique()>1]
        Xs=state['scaler'].fit_transform(data[state['final_static']].to_numpy(float))
    else:
        for c,enc in state['enc'].items():
            data[c]=data[c].fillna('MISSING').astype(str)
            data[c]=data[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            data[c]=enc.transform(data[c])
        for c,val in state['imp'].items():
            if c in data.columns: data[c]=data[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        Xs=state['scaler'].transform(data[state['final_static']].to_numpy(float))
    g=group_temporal(tcols)
    feats=[]
    for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
        if f in g:
            arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
            feats.append(arr[:,::3])
    Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(data),1))
    X=np.hstack([Xs,Xtemp])
    return X, state


def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    Xtr, st=preprocess(train, fit=True)
    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr,_=preprocess(vtr, fit=False, state=st)
    Xvho,_=preprocess(vho, fit=False, state=st)

    best_auc, best_k = -1, None
    for k in [5, 8, 12]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xtr)
        centers = km.cluster_centers_
        # score valid-train
        d2 = np.min(((Xvtr[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1)
        s = np.sqrt(d2)
        auc = roc_auc_score(vtr['target'], s)
        if auc > best_auc: best_auc, best_k = auc, k
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xtr)
    centers = km.cluster_centers_
    d2 = np.min(((Xvho[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1)
    s = (np.sqrt(d2) - d2.min())/(np.sqrt(d2).max()-np.sqrt(d2).min()+1e-12)
    auprc = average_precision_score(vho['target'], s)
    auc = roc_auc_score(vho['target'], s)
    print(f'[KMeansDist] k={best_k}  HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

if __name__=='__main__':
    main()


# =============================================
# Script 5/5: elliptic_envelope_mcd.py
# =============================================
#!/usr/bin/env python3
"""EllipticEnvelope (robust covariance / MCD) as statistical baseline."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# utils

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

def preprocess(df, fit=False, state=None):
    if state is None: state={'enc':{},'imp':{},'scaler':RobustScaler(),'final_static':[]}
    data=df.copy()
    if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
    scols,tcols=detect_cols(data)
    for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
        if col in data.columns: data[col]=data[col].replace(code,np.nan)
    cat=[c for c in scols if data[c].dtype=='object']
    if fit:
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); state['enc'][c]=enc
            data[c]=enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        for c in [c for c in scols if c not in cat]:
            state['imp'][c]=float(data[c].median()) if data[c].notna().any() else 0.0
            data[c]=data[c].fillna(state['imp'][c])
        new=[]
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0); new.append('credit_ltv_ratio')
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0); new.append('dti_credit_ratio')
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        state['final_static']=[c for c in core+new if c in data.columns and data[c].nunique()>1]
        Xs=state['scaler'].fit_transform(data[state['final_static']].to_numpy(float))
    else:
        for c,enc in state['enc'].items():
            data[c]=data[c].fillna('MISSING').astype(str)
            data[c]=data[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            data[c]=enc.transform(data[c])
        for c,val in state['imp'].items():
            if c in data.columns: data[c]=data[c].fillna(val)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns): data['credit_ltv_ratio']=data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns): data['dti_credit_ratio']=data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        Xs=state['scaler'].transform(data[state['final_static']].to_numpy(float))
    g=group_temporal(tcols)
    feats=[]
    for f in ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']:
        if f in g:
            arr=per_loan_ffill_bfill(data[g[f]].to_numpy(float))
            feats.append(arr[:,::3])
    Xtemp=np.concatenate(feats,axis=1) if feats else np.zeros((len(data),1))
    X=np.hstack([Xs,Xtemp])
    return X, state


def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    Xtr, st=preprocess(train, fit=True)
    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr,_=preprocess(vtr, fit=False, state=st)
    Xvho,_=preprocess(vho, fit=False, state=st)

    best_auc, best_c = -1, None
    for c in [0.05, 0.1, 0.15]:
        ee = EllipticEnvelope(contamination=c, support_fraction=0.8, random_state=42).fit(Xtr)
        s = -ee.score_samples(Xvtr)
        auc = roc_auc_score(vtr['target'], s)
        if auc > best_auc: best_auc, best_c = auc, c
    ee = EllipticEnvelope(contamination=best_c, support_fraction=0.8, random_state=42).fit(Xtr)
    s_hold = -ee.score_samples(Xvho)
    auprc = average_precision_score(vho['target'], s_hold)
    auc = roc_auc_score(vho['target'], s_hold)
    print(f'[EllipticEnvelope/MCD] contamination={best_c}  HOLDOUT  AUPRC={auprc:.4f}  AUROC={auc:.4f}')

if __name__=='__main__':
    from sklearn.model_selection import StratifiedShuffleSplit
    main()
