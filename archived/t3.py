# =============================
# Script 3 of 4: isolation_forest_baseline.py
# =============================
#!/usr/bin/env python3
"""
Isolation Forest Baseline (leak-free)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

# Reuse same simple preprocess (duplicated for standalone)
from typing import List, Tuple, Dict

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
    np.minimum.accumulate(idx2[:,::-1],axis=1,out=idx2[:,::-1])
    bf=ff[np.arange(A.shape[0])[:,None],idx2]
    return np.nan_to_num(bf, nan=0.0)


def preprocess(df, fit=False, state=None):
    if state is None:
        state={'enc':{},'imp':{},'scaler':RobustScaler(), 'final_static':[]}
    data=df.copy()
    if 'Id' in data.columns and 'index' not in data.columns: data['index']=data['Id']
    scols,tcols=detect_cols(data)
    for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
        if col in data.columns: data[col]=data[col].replace(code,np.nan)
    cat=[c for c in scols if data[c].dtype=='object']
    if fit:
        for c in cat:
            enc=LabelEncoder(); vals=data[c].fillna('MISSING').astype(str).unique().tolist();
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
    Xtemp=np.concatenate(feats, axis=1) if feats else np.zeros((len(data),1))
    X=np.hstack([Xs, Xtemp])
    return X, state


def main():
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')
    Xtr, st=preprocess(train, fit=True)
    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tr,ho in sss.split(valid,yv):
        vtr=valid.iloc[tr]; vho=valid.iloc[ho]
    Xvho,_=preprocess(vho, fit=False, state=st)
    # Fit IF on train normals
    IF=IsolationForest(n_estimators=400, contamination='auto', random_state=42, n_jobs=-1)
    IF.fit(Xtr)
    s_hold = -IF.score_samples(Xvho)
    auroc=roc_auc_score(vho['target'], s_hold)
    auprc=average_precision_score(vho['target'], s_hold)
    print('[IF] Holdout AUROC=%.4f AUPRC=%.4f'%(auroc,auprc))

if __name__=='__main__':
    from sklearn.model_selection import StratifiedShuffleSplit
    main()
