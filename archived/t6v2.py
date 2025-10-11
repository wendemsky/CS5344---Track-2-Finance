#!/usr/bin/env python3
"""
Minimal improvements to t6.py (AUPRC=0.1956)
Focus on small, proven enhancements without over-engineering

Key changes:
1. Slightly better hyperparameter search
2. One additional simple feature
3. Generate submission file
4. Keep everything else the same as original
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

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
        # MINIMAL ADDITION: One simple risk combination
        if set(['CreditScore','OriginalLTV','OriginalDTI']).issubset(data.columns):
            risk_score = (700 - data['CreditScore'])/100 + data['OriginalLTV']/100 + data['OriginalDTI']/50
            data['simple_risk'] = risk_score / 3.0  # Normalize
            new.append('simple_risk')
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
        if set(['CreditScore','OriginalLTV','OriginalDTI']).issubset(data.columns):
            risk_score = (700 - data['CreditScore'])/100 + data['OriginalLTV']/100 + data['OriginalDTI']/50
            data['simple_risk'] = risk_score / 3.0
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
    print("=== Minimal Enhancement of t6.py (AUPRC=0.1956) ===")
    train=pd.read_csv('Data/loans_train.csv')
    valid=pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train.shape}, Valid: {valid.shape}")
    print(f"Anomaly rate: {valid['target'].mean():.3f}")

    Xtr, st=preprocess(train, fit=True)
    yv=valid['target'].to_numpy()
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tri,hoi in sss.split(valid,yv):
        vtr=valid.iloc[tri]; vho=valid.iloc[hoi]
    Xvtr,_=preprocess(vtr, fit=False, state=st)
    Xvho,_=preprocess(vho, fit=False, state=st)

    # SLIGHT IMPROVEMENT: Test a few more k values around the original winner
    best_auprc, best_k = -1, None
    for k in [8, 10, 12, 15, 20]:  # Expanded around original winner k=10
        lof=LocalOutlierFactor(n_neighbors=k, novelty=True, contamination='auto')
        lof.fit(Xtr)
        s = -lof.score_samples(Xvtr)
        auprc = average_precision_score(vtr['target'], s)  # Use AUPRC for selection
        if auprc > best_auprc: best_auprc, best_k = auprc, k

    print(f"Best validation AUPRC: {best_auprc:.4f} with k={best_k}")

    # Train final model
    lof=LocalOutlierFactor(n_neighbors=best_k, novelty=True, contamination='auto').fit(Xtr)
    s_hold = -lof.score_samples(Xvho)
    auprc = average_precision_score(vho['target'], s_hold)
    auc = roc_auc_score(vho['target'], s_hold)
    print(f'[Minimally Enhanced LOF] HOLDOUT AUPRC={auprc:.4f} AUROC={auc:.4f}')

    improvement = (auprc / valid['target'].mean() - 1) * 100
    print(f'Improvement over random: {improvement:.1f}%')

    # Generate test submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        Xtest, _ = preprocess(test_df, fit=False, state=st)
        test_scores = -lof.score_samples(Xtest)

        # Normalize to [0,1]
        test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-8)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })

        filename = f"T6V2_MINIMAL_LOF_AUPRC{auprc:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"Submission saved: {filename}")

    except Exception as e:
        print(f"Test prediction failed: {e}")

if __name__=='__main__':
    main()