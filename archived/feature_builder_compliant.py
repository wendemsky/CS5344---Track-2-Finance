#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

class FeatureBuilderCompliant:
    """
    Train-only feature builder (compliant):
    - Sentinel mapping + missing indicators
    - Categorical label encoding (UNKNOWN for unseen)
    - Early-window temporal engineering (trend, volatility)
    - Amortization residuals via annuity (FRM only; skip IO/balloon/ARM)
    - ONE global StandardScaler on full feature matrix (train-only)
    - Optional PCA (train-only) for distance/clustering models
    Returns both scaled-full and PCA-projected matrices; exposes stable slices.
    """

    def __init__(self, use_pca=True, pca_comps=100, keep_month_idx=(0,1,2,3,6,9,12)):
        self.use_pca = use_pca
        self.pca_comps = pca_comps
        self.keep_month_idx = set(keep_month_idx)

        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.impute_vals: Dict[str, float] = {}
        self.global_scaler: StandardScaler = None
        self.pca: PCA = None

        self.static_cols: List[str] = []
        self.temporal_cols: List[str] = []
        self.amort_slice: slice = slice(0,0)
        self._fitted = False

    # ---------- helpers ----------
    def _is_temporal(self, col: str) -> bool:
        if "_" not in col: return False
        left = col.split("_", 1)[0]
        return left.isdigit()

    def _sentinel_map(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # numeric sentinels -> NaN + missing flags
        SENT = {
            "CreditScore": 9999,
            "OriginalDTI": 999,
            "OriginalLTV": 999,
            "MI Pct": 999,
            # temporal EstimatedLTV has panel vars; static version also appears in some schemas
            "EstimatedLTV": 999
        }
        for col, bad in SENT.items():
            if col in df.columns:
                miss = (df[col] == bad)
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        # category '9'/'99' means not available
        FLAG_9 = [
            "FirstTimeHomebuyerFlag", "OccupancyStatus", "LoanPurpose",
            "Channel", "PropertyType"
        ]
        for col in FLAG_9:
            if col in df.columns:
                x = df[col].astype(str).str.strip()
                miss = (x == "9") | (x == "99")
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        return df

    def _collect_columns(self, df: pd.DataFrame):
        self.static_cols = []
        self.temporal_cols = []
        for c in df.columns:
            if c in ("index","Id","target"):
                continue
            if self._is_temporal(c):
                m = int(c.split("_",1)[0])
                if m in self.keep_month_idx:
                    self.temporal_cols.append(c)
            else:
                self.static_cols.append(c)

    def _fit_cat_and_impute(self, df: pd.DataFrame):
        # label encoders on train-only
        for c in self.static_cols:
            if df[c].dtype == "object":
                enc = LabelEncoder()
                vals = df[c].fillna("MISSING").astype(str).unique().tolist()
                if "UNKNOWN" not in vals:
                    vals.append("UNKNOWN")
                enc.fit(vals)
                self.cat_encoders[c] = enc
        # numeric medians train-only
        for c in self.static_cols:
            if c not in self.cat_encoders:
                v = df[c].dropna()
                self.impute_vals[c] = float(v.median() if len(v) else 0.0)

    def _transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df[self.static_cols].copy()
        for c, enc in self.cat_encoders.items():
            x = data[c].fillna("MISSING").astype(str)
            x = x.where(x.isin(enc.classes_), "UNKNOWN")
            data[c] = enc.transform(x)
        for c in self.static_cols:
            if c not in self.cat_encoders:
                data[c] = data[c].fillna(self.impute_vals.get(c, 0.0))
        return data

    def _build_temporal_block(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.temporal_cols:
            return pd.DataFrame(index=df.index)
        tmp = df[self.temporal_cols].copy()
        tmp = tmp.ffill(axis=1).bfill(axis=1).fillna(0.0)

        feat = {}
        by_type: Dict[str, List[str]] = {}
        for c in self.temporal_cols:
            ftype = c.split("_",1)[1]
            by_type.setdefault(ftype, []).append(c)
        for ftype, cols in by_type.items():
            cols_sorted = sorted(cols, key=lambda x: int(x.split("_",1)[0]))
            a = tmp[cols_sorted]
            first, last = a.iloc[:,0], a.iloc[:,-1]
            trend = safe_div(last - first, first.abs() + 1.0)
            vol = safe_div(a.std(axis=1), a.mean(axis=1).abs() + 1.0)
            feat[f"{ftype}_trend"] = trend
            feat[f"{ftype}_vol"] = vol

        return pd.DataFrame(feat, index=df.index)

    @staticmethod
    def _annuity_payment(orig_upb, annual_rate, term_m):
        r = (annual_rate/100.0)/12.0
        T = np.maximum(term_m, 1.0)
        if np.isnan(r) or r <= 0:
            return orig_upb / T
        num = r * (1.0 + r)**T
        den = (1.0 + r)**T - 1.0
        if den <= 0:
            return orig_upb / T
        return orig_upb * (num/den)

    def _amort_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        need = ["OriginalUPB","OriginalInterestRate","OriginalLoanTerm",
                "ProductType","InterestOnlyFlag","BalloonIndicator"]
        for c in need:
            if c not in df.columns:
                return pd.DataFrame(index=df.index)

        # pick panel cols
        def col_for(suffix):
            cands = [c for c in self.temporal_cols if c.endswith(f"_{suffix}")]
            return sorted(cands, key=lambda x: int(x.split("_",1)[0]))

        ib_cols  = col_for("InterestBearingUPB")
        rt_cols  = col_for("CurrentInterestRate")
        rem_cols = col_for("RemainingMonthsToLegalMaturity")
        if not ib_cols:
            return pd.DataFrame(index=df.index)
        n_steps = len(ib_cols)

        is_frm = (df["ProductType"].astype(str).str.upper() == "FRM")
        is_io  = df.get("InterestOnlyFlag", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        is_bal = df.get("BalloonIndicator", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        use_mask = is_frm & (~is_io) & (~is_bal)

        IB = df[ib_cols].values.astype(float)
        if rt_cols:
            RT = df[rt_cols].values.astype(float)/1200.0
        else:
            RT = np.tile((df["OriginalInterestRate"].values/1200.0)[:,None], (1,n_steps))
        if rem_cols:
            RM = df[rem_cols].values.astype(float)
        else:
            T0 = df["OriginalLoanTerm"].values.astype(float)
            RM = np.maximum(T0[:,None] - np.arange(n_steps)[None,:], 1.0)

        ExpPrin = np.zeros_like(IB)
        for t in range(1, n_steps):
            upb_prev = IB[:, t-1]
            r_t = RT[:, t]
            M_t = np.maximum(RM[:, t], 1.0)
            with np.errstate(over="ignore"):
                num = r_t * np.power(1.0 + r_t, M_t)
                den = np.power(1.0 + r_t, M_t) - 1.0
            P_t = np.where((den>0) & (r_t>0), upb_prev * (num/den), safe_div(upb_prev, M_t))
            exp_prin = np.maximum(P_t - r_t * upb_prev, 0.0)
            ExpPrin[:, t] = exp_prin

        ObsPrin = np.maximum(IB[:, :-1] - IB[:, 1:], 0.0)
        ExpPrin_use = ExpPrin[:, 1:]
        Short = safe_div(ExpPrin_use - ObsPrin, np.abs(ExpPrin_use) + 1e-9)
        Short = np.clip(Short, 0.0, 1.0)

        short_mean = Short.mean(axis=1)
        short_pct_hi = (Short > 0.5).mean(axis=1)

        short_mean = np.where(use_mask.values, short_mean, np.nan)
        short_pct_hi = np.where(use_mask.values, short_pct_hi, np.nan)

        out = pd.DataFrame({
            "amort_short_mean": short_mean,
            "amort_short_pct_hi": short_pct_hi,
            "amort_mask_not_applicable": (~use_mask).astype(int),
        }, index=df.index)
        out[["amort_short_mean","amort_short_pct_hi"]] = out[["amort_short_mean","amort_short_pct_hi"]].fillna(0.0)
        return out

    # ---------- public API ----------
    def fit(self, df_train: pd.DataFrame):
        df = self._sentinel_map(df_train.copy())
        self._collect_columns(df)
        self._fit_cat_and_impute(df)

        Xs = self._transform_static(df)
        Xt = self._build_temporal_block(df)
        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))

        X_full = pd.concat([Xs, Xt, Xa], axis=1).astype(float)
        self.global_scaler = StandardScaler().fit(X_full.values)
        X_scaled = self.global_scaler.transform(X_full.values)

        if self.use_pca:
            k = min(self.pca_comps, X_scaled.shape[1])
            self.pca = PCA(n_components=k, random_state=42).fit(X_scaled)

        start = Xs.shape[1] + Xt.shape[1]
        end = start + Xa.shape[1]
        self.amort_slice = slice(start, end)

        self._fitted = True

    def transform(self, df_any: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, slice], np.ndarray]:
        assert self._fitted, "Call fit() first."
        df = self._sentinel_map(df_any.copy())

        Xs = self._transform_static(df)
        Xt = self._build_temporal_block(df)
        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))

        X_full = pd.concat([Xs, Xt, Xa], axis=1).astype(float)
        X_scaled = self.global_scaler.transform(X_full.values)
        X_det = self.pca.transform(X_scaled) if self.use_pca else X_scaled

        idx = {
            "static": slice(0, Xs.shape[1]),
            "temporal": slice(Xs.shape[1], Xs.shape[1] + Xt.shape[1]),
            "amort": self.amort_slice
        }
        return X_scaled, idx, X_det
