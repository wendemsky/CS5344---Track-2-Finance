# =============================
# Script 1 of 4: lstm_ae_static_conditioned.py
# =============================
#!/usr/bin/env python3
"""
Sequence Reconstruction Autoencoder (LSTM) with Static Conditioning
- Trains ONLY on normal loans from loans_train.csv
- Uses temporal panels as multivariate sequences; static features condition the latent
- Masked MSE reconstruction loss (optionally + first-difference loss)
- Evaluates on a CLEAN holdout split from loans_valid.csv (AUROC, AUPRC)
- Outputs a Kaggle-ready submission for loans_test.csv

Requirements: torch, numpy, pandas, scikit-learn
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------- utils (shared) -------------------

def detect_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    static_cols, temp_cols = [], []
    for c in df.columns:
        if c in ['index','Id','target']:
            continue
        if isinstance(c, str) and '_' in c and c.split('_',1)[0].isdigit():
            temp_cols.append(c)
        else:
            static_cols.append(c)
    return static_cols, temp_cols


def group_temporal(temp_cols: List[str]) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {}
    for c in temp_cols:
        t, ftype = c.split('_',1)
        g.setdefault(ftype, []).append(c)
    for k,v in g.items():
        g[k] = sorted(v, key=lambda x: int(x.split('_',1)[0]))
    return g


def per_loan_ffill_bfill(mat: np.ndarray) -> np.ndarray:
    mask = np.isnan(mat)
    idx = np.where(~mask, np.arange(mat.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    ff = mat[np.arange(mat.shape[0])[:,None], idx]
    mask2 = np.isnan(ff)
    idx2 = np.where(~mask2, np.arange(ff.shape[1]), ff.shape[1]-1)
    np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
    bf = ff[np.arange(ff.shape[0])[:,None], idx2]
    return np.nan_to_num(bf, nan=0.0)


class Preprocessor:
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.num_impute: Dict[str, float] = {}
        self.scaler = RobustScaler()
        self.static_cols: List[str] = []
        self.temp_cols: List[str] = []
        self.final_static: List[str] = []
        self.seq_types = ['CurrentActualUPB','EstimatedLTV','InterestBearingUPB','LoanAge']
        self.max_T = None

    def fit_transform_train(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = df.copy()
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']
        self.static_cols, self.temp_cols = detect_cols(data)
        # special codes
        for col,code in [('CreditScore',9999),('OriginalDTI',999),('OriginalLTV',999)]:
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)
        # categoricals
        cat = [c for c in self.static_cols if data[c].dtype=='object']
        for c in cat:
            enc = LabelEncoder()
            vals = data[c].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in vals: vals.append('UNKNOWN')
            enc.fit(vals)
            self.encoders[c] = enc
            data[c] = enc.transform(data[c].fillna('MISSING').astype(str).apply(lambda x: x if x in enc.classes_ else 'UNKNOWN'))
        # numericals
        num = [c for c in self.static_cols if c not in cat]
        for c in num:
            self.num_impute[c] = float(data[c].median()) if data[c].notna().any() else 0.0
            data[c] = data[c].fillna(self.num_impute[c])
        # engineered statics
        new = []
        if set(['CreditScore','OriginalLTV']).issubset(data.columns):
            data['credit_ltv_ratio'] = data['CreditScore']/(data['OriginalLTV']+1.0); new.append('credit_ltv_ratio')
        if set(['OriginalDTI','CreditScore']).issubset(data.columns):
            data['dti_credit_ratio'] = data['OriginalDTI']/(data['CreditScore']/100.0+1.0); new.append('dti_credit_ratio')
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate'] = data['OriginalInterestRate']/100.0/12.0; new.append('monthly_rate')
            data['payment_burden'] = data['OriginalUPB']*data['monthly_rate']; new.append('payment_burden')
        core_static = ['CreditScore','OriginalUPB','OriginalLTV','OriginalInterestRate','OriginalDTI','OriginalLoanTerm','NumberOfUnits','NumberOfBorrowers']
        self.final_static = [c for c in core_static+new if c in data.columns and data[c].nunique()>1]
        X_static = data[self.final_static].to_numpy(dtype=float)
        # sequences
        groups = group_temporal(self.temp_cols)
        used_types = [t for t in self.seq_types if t in groups]
        seq_mats = [per_loan_ffill_bfill(data[groups[t]].to_numpy(dtype=float)) for t in used_types]
        if not seq_mats:
            raise RuntimeError('No temporal sequence types found.')
        seq = np.stack(seq_mats, axis=2)  # shape: (N, T, F)
        self.max_T = seq.shape[1]
        # scale statics
        X_static = self.scaler.fit_transform(X_static)
        return seq, X_static, used_types

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        data = df.copy()
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']
        # apply encoders/imputers
        for c, enc in self.encoders.items():
            data[c] = data[c].fillna('MISSING').astype(str)
            data[c] = data[c].apply(lambda x: x if x in enc.classes_ else 'UNKNOWN')
            data[c] = enc.transform(data[c])
        for c, val in self.num_impute.items():
            if c in data.columns:
                data[c] = data[c].fillna(val)
        # engineered statics (same as train)
        if set(['CreditScore','OriginalLTV']).issubset(data.columns):
            data['credit_ltv_ratio'] = data['CreditScore']/(data['OriginalLTV']+1.0)
        if set(['OriginalDTI','CreditScore']).issubset(data.columns):
            data['dti_credit_ratio'] = data['OriginalDTI']/(data['CreditScore']/100.0+1.0)
        if set(['OriginalUPB','OriginalInterestRate']).issubset(data.columns):
            data['monthly_rate'] = data['OriginalInterestRate']/100.0/12.0
            data['payment_burden'] = data['OriginalUPB']*data['monthly_rate']
        X_static = data[self.final_static].to_numpy(dtype=float)
        X_static = self.scaler.transform(X_static)
        # sequences
        _, temp_cols = detect_cols(data)
        groups = group_temporal(temp_cols)
        seq_mats = []
        for t in self.seq_types:
            if t in groups:
                arr = per_loan_ffill_bfill(data[groups[t]].to_numpy(dtype=float))
                # align length if differs
                if arr.shape[1] != self.max_T:
                    # pad or trim to max_T
                    if arr.shape[1] < self.max_T:
                        pad = np.zeros((arr.shape[0], self.max_T-arr.shape[1]))
                        arr = np.concatenate([arr, pad], axis=1)
                    else:
                        arr = arr[:, :self.max_T]
                seq_mats.append(arr)
        if not seq_mats:
            raise RuntimeError('No temporal types available at inference.')
        seq = np.stack(seq_mats, axis=2)
        return seq, X_static


class LSTMAE(nn.Module):
    def __init__(self, in_dim: int, static_dim: int, hidden: int = 128, layers: int = 2, cond_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.static_mlp = nn.Sequential(nn.Linear(static_dim, 128), nn.ReLU(), nn.Linear(128, cond_dim), nn.ReLU())
        self.dec_init = nn.Linear(hidden + cond_dim, hidden)
        self.decoder = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x, s):
        # x: (B,T,F), s: (B, S)
        z, (hT, cT) = self.encoder(x)
        z_last = z[:, -1, :]  # (B, hidden)
        h_s = self.static_mlp(s)  # (B, cond_dim)
        zc = torch.cat([z_last, h_s], dim=1)
        h0 = torch.tanh(self.dec_init(zc))  # (B, hidden)
        h0 = h0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(x, (h0, c0))
        y = self.out(dec_out)
        return y


def masked_mse(y, yhat, mask):
    # mask: (B,T,F) with 1 for observed
    diff = (y - yhat) * mask
    return (diff.pow(2).sum() / (mask.sum() + 1e-8))


def train_ae(model, train_seq, train_static, epochs=50, lr=1e-3, batch=128, device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    patience, bad = 7, 0
    # create a mask where original entries are non-zero after ffill/bfill? Here, assume all observed after fill.
    mask = np.ones_like(train_seq, dtype=np.float32)
    tensor_seq = torch.tensor(train_seq, dtype=torch.float32)
    tensor_static = torch.tensor(train_static, dtype=torch.float32)
    tensor_mask = torch.tensor(mask, dtype=torch.float32)
    n = tensor_seq.shape[0]
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            x = tensor_seq[b].to(device)
            s = tensor_static[b].to(device)
            m = tensor_mask[b].to(device)
            opt.zero_grad()
            yhat = model(x, s)
            loss = masked_mse(x, yhat, m)
            # optional trend loss
            dx = x[:,1:,:]-x[:,:-1,:]
            dy = yhat[:,1:,:]-yhat[:,:-1,:]
            loss_trend = ((dx-dy)**2).mean()
            loss = loss + 0.1*loss_trend
            loss.backward()
            opt.step()
            total += loss.item()*len(b)
        avg = total/n
        if avg < best_loss - 1e-5:
            best_loss = avg; bad = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model


def main():
    # Load
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    # Preprocess
    pp = Preprocessor()
    seq_train, stat_train, used_types = pp.fit_transform_train(train_df)

    # Build model
    in_dim = seq_train.shape[2]
    static_dim = stat_train.shape[1]
    model = LSTMAE(in_dim=in_dim, static_dim=static_dim, hidden=128, layers=2, cond_dim=64, dropout=0.1)

    # Train on train normals only
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_ae(model, seq_train, stat_train, epochs=50, lr=1e-3, batch=128, device=device)

    # Prepare validation split (clean holdout)
    yv = valid_df['target'].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for tr, ho in sss.split(valid_df, yv):
        valid_train_df = valid_df.iloc[tr]
        valid_hold_df = valid_df.iloc[ho]
    # transform
    seq_vt, stat_vt = pp.transform(valid_train_df)
    seq_vh, stat_vh = pp.transform(valid_hold_df)

    # Score function: reconstruction error per step → per-loan median
    with torch.no_grad():
        model.eval()
        def score_block(seq, stat):
            tensor_seq = torch.tensor(seq, dtype=torch.float32).to(device)
            tensor_stat = torch.tensor(stat, dtype=torch.float32).to(device)
            preds = []
            for i in range(0, tensor_seq.shape[0], 256):
                yhat = model(tensor_seq[i:i+256], tensor_stat[i:i+256])
                err = (tensor_seq[i:i+256]-yhat).pow(2).mean(dim=2)  # (B,T)
                preds.append(err.detach().cpu().numpy())
            E = np.concatenate(preds, axis=0)
            loan_score = np.median(E, axis=1)
            # normalize within block
            m, M = loan_score.min(), loan_score.max()
            if M-m < 1e-12: return np.zeros_like(loan_score)
            return (loan_score - m)/(M-m)
        s_val_train = score_block(seq_vt, stat_vt)
        s_val_hold = score_block(seq_vh, stat_vh)

    # Evaluate on holdout only
    y_hold = valid_hold_df['target'].to_numpy()
    auroc = roc_auc_score(y_hold, s_val_hold)
    auprc = average_precision_score(y_hold, s_val_hold)
    print("[LSTM-AE] Holdout AUROC=%.4f AUPRC=%.4f" % (auroc, auprc))

    # Test submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        seq_te, stat_te = pp.transform(test_df)
        with torch.no_grad():
            s_test = score_block(seq_te, stat_te)
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        out = pd.DataFrame({id_col: test_df[id_col], 'anomaly_score': s_test})
        out.to_csv('SUB_LSTM_AE.csv', index=False)
        print('Saved SUB_LSTM_AE.csv')
    except Exception as e:
        print('Test scoring skipped:', e)

if __name__ == '__main__':
    main()