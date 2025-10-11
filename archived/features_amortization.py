# =============================================================
# Script 1/4: features_amortization.py
# (shared lean features + amortization deviation signals)
# =============================================================
#!/usr/bin/env python3
import warnings; warnings.filterwarnings('ignore')
from typing import Dict, List
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder

# We'll focus on the strongest temporal types for neighbourhood methods
TEMP_TYPES = ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB']

class FeatureBuilderAmort:
    """Lean feature builder with amortization deviation signals.
    Fits ONLY on train normals; safe for anomaly detection.
    """
    def __init__(self):
        self.enc:Dict[str,LabelEncoder]={}
        self.imp:Dict[str,float]={}
        self.scaler=RobustScaler()
        self.final_static:List[str]=[]

    # ------------ helpers ------------
    @staticmethod
    def _detect_cols(df: pd.DataFrame):
        stat, temp = [], []
        for c in df.columns:
            if c in ['index','Id','target']: 
                continue
            if isinstance(c,str) and '_' in c and c.split('_',1)[0].isdigit():
                temp.append(c)
            else:
                stat.append(c)
        return stat, temp

    @staticmethod
    def _group_temporal(tcols: List[str]):
        g={}
        for c in tcols:
            _,ft=c.split('_',1)
            g.setdefault(ft,[]).append(c)
        for k in g:
            g[k]=sorted(g[k], key=lambda x:int(x.split('_',1)[0]))
        return g

    @staticmethod
    def _ffill_bfill_rowwise(A: np.ndarray) -> np.ndarray:
        mask=np.isnan(A)
        idx=np.where(~mask, np.arange(A.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        ff=A[np.arange(A.shape[0])[:,None], idx]
        mask2=np.isnan(ff)
        idx2=np.where(~mask2, np.arange(ff.shape[1]), ff.shape[1]-1)
        np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
        bf=ff[np.arange(A.shape[0])[:,None], idx2]
        return np.nan_to_num(bf, nan=0.0)

    # ------------ core ------------
    def fit(self, df: pd.DataFrame):
        d=df.copy()
        if 'Id' in d.columns and 'index' not in d.columns:
            d['index']=d['Id']
        stat, temp = self._detect_cols(d)
        # specials -> NaN
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in d.columns:
                d[col]=d[col].replace(code, np.nan)
        # encode categoricals
        cats=[c for c in stat if d[c].dtype=='object']
        for c in cats:
            enc=LabelEncoder(); vals=d[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals); self.enc[c]=enc
            d[c]=enc.transform(d[c].fillna('MISSING').astype(str).apply(lambda x:x if x in enc.classes_ else 'UNKNOWN'))
        # numeric impute
        for c in [c for c in stat if c not in cats]:
            self.imp[c]=float(d[c].median()) if d[c].notna().any() else 0.0
            d[c]=d[c].fillna(self.imp[c])
        # engineered static ratios
        if set(['CreditScore','OriginalLTV']).issubset(d.columns):
            d['credit_ltv_ratio']=d['CreditScore']/(d['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(d.columns):
            d['dti_credit_ratio']=d['OriginalDTI']/(d['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(d.columns):
            d['monthly_rate']=d['OriginalInterestRate']/100.0/12.0
            d['payment_burden']=d['OriginalUPB']*d['monthly_rate']
        core=['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI',
              'OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers','credit_ltv_ratio',
              'dti_credit_ratio','monthly_rate','payment_burden']
        self.final_static=[c for c in core if c in d.columns and d[c].nunique()>1]
        self.scaler.fit(d[self.final_static].to_numpy(float))
        return self

    def _amortization_signals(self, d: pd.DataFrame, g: Dict[str,List[str]]):
        """Compute amortization deviation signals using OriginalUPB/InterestRate and UPB/ELTV paths."""
        feats=[]
        if 'CurrentActualUPB' in g:
            upb=self._ffill_bfill_rowwise(d[g['CurrentActualUPB']].to_numpy(float))
            rate = d['OriginalInterestRate'].to_numpy(float)/1200.0 if 'OriginalInterestRate' in d.columns else np.zeros(len(d))
            orig = d['OriginalUPB'].to_numpy(float) if 'OriginalUPB' in d.columns else np.maximum(upb[:,0],0)
            pay = orig * rate
            # deltas
            du=np.diff(upb, axis=1)
            # expected principal each month ≈ payment - interest
            # interest approx using previous month balance
            interest = upb[:,:-1]*rate[:,None]
            exp_principal = np.clip(pay[:,None]-interest, a_min=0.0, a_max=None)
            # bad month if principal reduced less than alpha*expected
            alphas=[0.7, 1.0]
            for a in alphas:
                bad=(du > -a*exp_principal)
                frac_bad=bad.mean(axis=1)
                # longest run length
                runs=np.zeros(len(d))
                for i in range(len(d)):
                    b=bad[i]
                    maxrun=0; cur=0
                    for flag in b:
                        if flag:
                            cur+=1; maxrun=max(maxrun,cur)
                        else:
                            cur=0
                    runs[i]=maxrun/ max(1,b.shape[0])
                feats.append(frac_bad[:,None]); feats.append(runs[:,None])
            # first bad month index normalized
            first_bad=np.ones(len(d))
            for i in range(len(d)):
                b=(du[i]>-1.0*exp_principal[i])
                idx=np.argmax(b)
                first_bad[i]= (idx/(du.shape[1]-1)) if b.any() else 1.0
            feats.append(first_bad[:,None])
            # ELTV spike overlap during bad months
            if 'EstimatedLTV' in g:
                eltv=self._ffill_bfill_rowwise(d[g['EstimatedLTV']].to_numpy(float))
                med=np.median(eltv, axis=1, keepdims=True)
                iqr=np.subtract(*np.percentile(eltv,[75,25],axis=1, keepdims=True)) + 1e-6
                spikes=(eltv > (med + 1.5*iqr))[:,:-1]
                bad1=(du > -1.0*exp_principal)
                overlap=(spikes & bad1).mean(axis=1)
                feats.append(overlap[:,None])
        return np.hstack(feats) if feats else np.zeros((len(d),1))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        d=df.copy()
        if 'Id' in d.columns and 'index' not in d.columns:
            d['index']=d['Id']
        stat, temp = self._detect_cols(d)
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in d.columns:
                d[col]=d[col].replace(code, np.nan)
        for c,enc in self.enc.items():
            d[c]=d[c].fillna('MISSING').astype(str)
            d[c]=d[c].apply(lambda x:x if x in enc.classes_ else 'UNKNOWN')
            d[c]=enc.transform(d[c])
        for c,val in self.imp.items():
            if c in d.columns:
                d[c]=d[c].fillna(val)
        # engineered statics again
        if set(['CreditScore','OriginalLTV']).issubset(d.columns):
            d['credit_ltv_ratio']=d['CreditScore']/(d['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(d.columns):
            d['dti_credit_ratio']=d['OriginalDTI']/(d['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(d.columns):
            d['monthly_rate']=d['OriginalInterestRate']/100.0/12.0
            d['payment_burden']=d['OriginalUPB']*d['monthly_rate']
        Xs=self.scaler.transform(d[self.final_static].to_numpy(float))
        # temporal signals (lean)
        _, tcols = self._detect_cols(d)
        g=self._group_temporal(tcols)
        feats=[]
        for f in TEMP_TYPES:
            if f not in g: continue
            arr=self._ffill_bfill_rowwise(d[g[f]].to_numpy(float))
            first=np.abs(arr[:,0])+1.0; mean_abs=np.abs(arr.mean(axis=1))+1.0
            trend=(arr[:,-1]-arr[:,0])/first
            vol=arr.std(axis=1)/mean_abs
            recent=(arr[:,-1]-arr[:,max(arr.shape[1]-3,1)])/(np.abs(arr[:,max(arr.shape[1]-3,1)])+1.0)
            feats.extend([arr[:,::3], trend[:,None], vol[:,None], recent[:,None]])
        Xtemp=np.concatenate(feats, axis=1) if feats else np.zeros((len(d),1))
        # amortization extras
        Xamort=self._amortization_signals(d,g)
        return np.hstack([Xs, Xtemp, Xamort])





