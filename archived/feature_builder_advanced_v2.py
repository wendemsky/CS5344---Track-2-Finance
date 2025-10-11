#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats

# ---------- utilities ----------
def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

def pct_change(a, axis=1, eps=1e-9):
    return (a[:, 1:] - a[:, :-1]) / (np.abs(a[:, :-1]) + eps)

def rolling_window(a, window, func):
    # Applies func along axis=1 with a rolling window
    if a.shape[1] < window:
        # If time series is too short, apply func to the whole series
        if a.shape[1] == 0:
            return np.full((a.shape[0], 0), np.nan)
        return np.apply_along_axis(func, axis=1, arr=a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.apply_along_axis(func, axis=-1, arr=rolled)

# ---------- feature builder ----------
class FeatureBuilderAdvancedV2:
    """
    Train-only feature builder (V2):
      - sentinel -> NaN + missing flags (static)
      - label-encode categorical with UNKNOWN (train-only fit)
      - temporal engineering on multiple early windows (trend/vol/first-diff stats)
      - advanced temporal features: change point detection, volatility clustering, higher-order differences
      - interaction features: static * temporal trends
      - amortization residuals (FRM & not IO/balloon if possible; otherwise masked zeros)
      - ONE global scaler (train-only) across the final tabular feature matrix
      - Optional PCA (train-only) to provide an embedding for distance/clustering detectors

    transform(...) returns:
      X_scaled  : scaled full feature matrix (for “engineered signals” like amort-resid)
      slices    : dict with slices {static, temporal, amort, interaction, advanced_temporal}
      X_embed   : PCA embedding (or X_scaled if PCA disabled) used by LOF/KMeans/kNN/IF/etc.
    """

    def __init__(
        self,
        keep_month_idx_main=(0,3,6,9,12),
        keep_month_idx_alt1=(0,2,4,6,8,10,12),
        keep_month_idx_alt2=(0,3,6,9),
        use_pca=True,
        pca_comps=80,
    ):
        self.keep_month_idx_main = set(keep_month_idx_main)
        self.keep_month_idx_alt1 = set(keep_month_idx_alt1)
        self.keep_month_idx_alt2 = set(keep_month_idx_alt2)
        self.use_pca = use_pca
        self.pca_comps = pca_comps

        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.impute_vals: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

        self.static_cols: List[str] = []
        self.temporal_cols_all: List[str] = []
        self._fitted = False

        # Slices for different feature groups
        self._static_slice = slice(0,0)
        self._temporal_slice = slice(0,0)
        self._amort_slice = slice(0,0)
        self._interaction_slice = slice(0,0)
        self._advanced_temporal_slice = slice(0,0)

    # ---------- helpers ----------
    @staticmethod
    def _is_temporal(col: str) -> bool:
        if "_" not in col: return False
        left = col.split("_", 1)[0]
        return left.isdigit()

    def _sentinel_map(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Numeric sentinels
        SENT = {
            "CreditScore": 9999,
            "OriginalDTI": 999,
            "OriginalLTV": 999,
            "MI Pct": 999,
            "EstimatedLTV": 999,
        }
        for col, bad in SENT.items():
            if col in df.columns:
                miss = (df[col] == bad)
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        # Flags '9'/'99' => missing
        FLAG_9 = ["FirstTimeHomebuyerFlag","OccupancyStatus","LoanPurpose","Channel","PropertyType"]
        for col in FLAG_9:
            if col in df.columns:
                x = df[col].astype(str).str.strip()
                miss = (x == "9") | (x == "99")
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        return df

    def _collect_columns(self, df: pd.DataFrame):
        self.static_cols = []
        self.temporal_cols_all = []
        for c in df.columns:
            if c in ("index","Id","target"): continue
            if self._is_temporal(c):
                self.temporal_cols_all.append(c)
            else:
                self.static_cols.append(c)

    def _fit_cat_and_impute(self, df: pd.DataFrame):
        for c in self.static_cols:
            if df[c].dtype == "object":
                enc = LabelEncoder()
                vals = df[c].fillna("MISSING").astype(str).unique().tolist()
                if "UNKNOWN" not in vals: vals.append("UNKNOWN")
                enc.fit(vals)
                self.cat_encoders[c] = enc
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

    def _temporal_block_multiwindow(self, df: pd.DataFrame) -> pd.DataFrame:
        # collect temporal by suffix type
        by_type: Dict[str, List[str]] = {}
        for c in self.temporal_cols_all:
            if "_" in c:
                by_type.setdefault(c.split("_",1)[1], []).append(c)
        if not by_type:
            return pd.DataFrame(index=df.index)

        feats = {}
        for ftype, cols in by_type.items():
            cols_sorted = sorted(cols, key=lambda x: int(x.split("_",1)[0]))
            M = df[cols_sorted].to_numpy(float)
            # forward/backward fill per row
            mask = np.isnan(M)
            idx = np.where(~mask, np.arange(M.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            M = M[np.arange(M.shape[0])[:,None], idx]
            mask = np.isnan(M)
            idx2 = np.where(~mask, np.arange(M.shape[1]), M.shape[1]-1)
            np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
            M = M[np.arange(M.shape[0])[:,None], idx2]
            M = np.nan_to_num(M, nan=0.0)

            # windows
            def select_steps(step_set):
                step_idx = []
                for c in cols_sorted:
                    m = int(c.split("_",1)[0])
                    if m in step_set:
                        step_idx.append(cols_sorted.index(c))
                if not step_idx:
                    return None
                return M[:, step_idx]

            for name, step_set in [
                ("w_main", self.keep_month_idx_main),
                ("w_alt1", self.keep_month_idx_alt1),
                ("w_alt2", self.keep_month_idx_alt2),
            ]:
                A = select_steps(step_set)
                if A is None or A.shape[1] < 2:
                    continue
                first, last = A[:, 0], A[:, -1]
                trend = safe_div(last - first, np.abs(first) + 1.0)
                vol   = safe_div(A.std(axis=1), np.abs(A.mean(axis=1)) + 1.0)
                # first differences stats
                d = pct_change(A)  # shape [n, T-1]
                d_mean = np.nanmean(d, axis=1)
                d_std  = np.nanstd(d, axis=1)

                feats[f"{ftype}_{name}_trend"] = trend
                feats[f"{ftype}_{name}_vol"]   = vol
                feats[f"{ftype}_{name}_dmean"] = d_mean
                feats[f"{ftype}_{name}_dstd"]  = d_std

        return pd.DataFrame(feats, index=df.index)

    def _advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Collect temporal by suffix type
        by_type: Dict[str, List[str]] = {}
        for c in self.temporal_cols_all:
            if "_" in c:
                by_type.setdefault(c.split("_",1)[1], []).append(c)
        
        feats = {}
        for ftype, cols in by_type.items():
            if "UPB" in ftype or "Balance" in ftype: # Focus on UPB/Principal for change point detection
                cols_sorted = sorted(cols, key=lambda x: int(x.split("_",1)[0]))
                M = df[cols_sorted].to_numpy(float)
                
                # Forward/backward fill per row
                mask = np.isnan(M)
                idx = np.where(~mask, np.arange(M.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                M = M[np.arange(M.shape[0])[:,None], idx]
                mask = np.isnan(M)
                idx2 = np.where(~mask, np.arange(M.shape[1]), M.shape[1]-1)
                np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
                M = M[np.arange(M.shape[0])[:,None], idx2]
                M = np.nan_to_num(M, nan=0.0)

                # 1. Change Point Detection (simplified, using mean shift)
                # Calculate variance of differences over time
                diffs = pct_change(M) # Shape [n, T-1]
                # Look for points where variance of recent differences changes significantly
                # Use a rolling window to calculate local variance
                if diffs.shape[1] >= 4:
                    local_var = rolling_window(diffs, 4, np.var)
                    # Max local variance as a feature
                    max_local_var = np.nanmax(local_var, axis=1)
                    # Mean local variance
                    mean_local_var = np.nanmean(local_var, axis=1)
                    feats[f"{ftype}_max_local_var"] = max_local_var
                    feats[f"{ftype}_mean_local_var"] = mean_local_var

                # 2. Higher-order differences (acceleration in change)
                if diffs.shape[1] >= 2:
                    acc = pct_change(diffs) # Shape [n, T-2] (acceleration of change)
                    acc_mean = np.nanmean(acc, axis=1)
                    acc_std = np.nanstd(acc, axis=1)
                    feats[f"{ftype}_acc_mean"] = acc_mean
                    feats[f"{ftype}_acc_std"] = acc_std

                # 3. Stability metrics (e.g., fraction of time UPB is within X% of mean)
                if M.shape[1] > 1:
                    means = np.mean(M, axis=1, keepdims=True)
                    stds = np.std(M, axis=1, keepdims=True) + 1e-9
                    # Z-score the series
                    z_scores = np.abs((M - means) / stds)
                    # Fraction of time within 1 std dev (or 0.5 std dev)
                    frac_within_1std = (z_scores <= 1.0).mean(axis=1)
                    frac_within_05std = (z_scores <= 0.5).mean(axis=1)
                    feats[f"{ftype}_frac_within_1std"] = frac_within_1std
                    feats[f"{ftype}_frac_within_05std"] = frac_within_05std

        return pd.DataFrame(feats, index=df.index)

    def _interaction_features(self, df_static: pd.DataFrame, df_temporal: pd.DataFrame) -> pd.DataFrame:
        # Example: Static feature * temporal trend
        interactions = {}
        static_numeric = df_static.select_dtypes(include=[np.number]).columns
        temporal_trend_cols = [c for c in df_temporal.columns if "_trend" in c]

        for s_col in static_numeric:
            for t_col in temporal_trend_cols:
                # Interaction feature: static value * temporal trend
                interactions[f"int_{s_col}_x_{t_col}"] = df_static[s_col] * df_temporal[t_col]
        
        return pd.DataFrame(interactions, index=df_static.index)


    @staticmethod
    def _annuity_payment_for_state(upb_prev, rt, rem_m):
        # rt is monthly rate (already divided by 12)
        rem_m = np.maximum(rem_m, 1.0)
        ok = (rt > 0) & (rem_m > 0)
        out = np.zeros_like(upb_prev)
        with np.errstate(over="ignore", invalid="ignore"):
            num = rt * np.power(1.0 + rt, rem_m)
            den = np.power(1.0 + rt, rem_m) - 1.0
            P = np.where((den > 0) & ok, upb_prev * (num/den), safe_div(upb_prev, rem_m))
        out = P
        return out

    def _amort_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # requires temporal UPB & optionally CurrentInterestRate & RemainingMonthsToLegalMaturity
        # Use all available months; combine with multi-window temporal slices implicitly (we use the actual panel)
        tcols = [c for c in self.temporal_cols_all if c.endswith("_InterestBearingUPB")]
        if not tcols:
            return pd.DataFrame(index=df.index)
        tcols_sorted = sorted(tcols, key=lambda x: int(x.split("_",1)[0]))
        IB = df[tcols_sorted].to_numpy(float)

        # monthly rate
        rcols = [c for c in self.temporal_cols_all if c.endswith("_CurrentInterestRate")]
        if rcols:
            rcols_sorted = sorted(rcols, key=lambda x: int(x.split("_",1)[0]))
            RT = df[rcols_sorted].to_numpy(float)/1200.0
        else:
            if "OriginalInterestRate" not in df.columns:
                return pd.DataFrame(index=df.index)
            RT = np.tile((df["OriginalInterestRate"].to_numpy(float)/1200.0)[:,None], (1, IB.shape[1]))

        # remaining months
        mcols = [c for c in self.temporal_cols_all if c.endswith("_RemainingMonthsToLegalMaturity")]
        if mcols:
            mcols_sorted = sorted(mcols, key=lambda x: int(x.split("_",1)[0]))
            RM = df[mcols_sorted].to_numpy(float)
        else:
            if "OriginalLoanTerm" not in df.columns:
                return pd.DataFrame(index=df.index)
            T0 = df["OriginalLoanTerm"].to_numpy(float)
            RM = np.maximum(T0[:,None] - np.arange(IB.shape[1])[None,:], 1.0)

        # applicability mask
        is_frm = df.get("ProductType", pd.Series("FRM", index=df.index)).astype(str).str.upper().eq("FRM")
        is_io  = df.get("InterestOnlyFlag", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        is_bal = df.get("BalloonIndicator", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        use_mask = (is_frm & (~is_io) & (~is_bal)).to_numpy()

        # compute expected principal reduction per step using annuity
        n = IB.shape[0]; T = IB.shape[1]
        ExpPrin = np.zeros((n, T))
        for t in range(1, T):
            upb_prev = IB[:, t-1]
            rt = RT[:, t]
            rem = RM[:, t]
            P = self._annuity_payment_for_state(upb_prev, rt, rem)
            exp_prin = np.maximum(P - rt*upb_prev, 0.0)
            ExpPrin[:, t] = exp_prin

        ObsPrin = np.maximum(IB[:, :-1] - IB[:, 1:], 0.0)
        E = ExpPrin[:, 1:]
        S = (E - ObsPrin) / (np.abs(E) + 1e-9)
        S = np.clip(S, 0.0, 1.0)

        short_mean = S.mean(axis=1)
        short_70   = (S > 0.70).mean(axis=1)
        short_50   = (S > 0.50).mean(axis=1)

        # mask non-applicable loans → 0, but expose mask flag
        short_mean = np.where(use_mask, short_mean, 0.0)
        short_70   = np.where(use_mask, short_70, 0.0)
        short_50   = np.where(use_mask, short_50, 0.0)

        return pd.DataFrame({
            "amort_short_mean": short_mean,
            "amort_short_70": short_70,
            "amort_short_50": short_50,
            "amort_mask_not_applicable": (~use_mask).astype(int)
        }, index=df.index)

    # ---------- public API ----------
    def fit(self, df_train: pd.DataFrame):
        df = self._sentinel_map(df_train.copy())
        self._collect_columns(df)
        self._fit_cat_and_impute(df)

        Xs = self._transform_static(df)
        Xt = self._temporal_block_multiwindow(df)
        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))
        Xat = self._advanced_temporal_features(df)
        Xi = self._interaction_features(Xs, Xt)

        X_full = pd.concat([Xs, Xt, Xa, Xi, Xat], axis=1).astype(float)
        self.scaler = StandardScaler().fit(X_full.values)
        X_scaled = self.scaler.transform(X_full.values)

        if self.use_pca:
            k = min(self.pca_comps, X_scaled.shape[1])
            self.pca = PCA(n_components=k, random_state=42).fit(X_scaled)

        # remember slices
        start_temporal = Xs.shape[1]
        end_temporal   = start_temporal + Xt.shape[1]
        start_amort    = end_temporal
        end_amort      = start_amort + Xa.shape[1]
        start_interaction = end_amort
        end_interaction   = start_interaction + Xi.shape[1]
        start_adv_temp = end_interaction
        end_adv_temp   = start_adv_temp + Xat.shape[1]

        self._static_slice = slice(0, Xs.shape[1])
        self._temporal_slice = slice(start_temporal, end_temporal)
        self._amort_slice    = slice(start_amort, end_amort)
        self._interaction_slice = slice(start_interaction, end_interaction)
        self._advanced_temporal_slice = slice(start_adv_temp, end_adv_temp)

        self._fitted = True

    def transform(self, df_any: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, slice], np.ndarray]:
        assert self._fitted, "Call fit() first."
        df = self._sentinel_map(df_any.copy())

        Xs = self._transform_static(df)
        Xt = self._temporal_block_multiwindow(df)
        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))
        Xat = self._advanced_temporal_features(df)
        Xi = self._interaction_features(Xs, Xt)

        X_full = pd.concat([Xs, Xt, Xa, Xi, Xat], axis=1).astype(float)
        X_scaled = self.scaler.transform(X_full.values)
        X_embed = self.pca.transform(X_scaled) if self.use_pca else X_scaled

        slices = {
            "static": self._static_slice,
            "temporal": self._temporal_slice,
            "amort": self._amort_slice,
            "interaction": self._interaction_slice,
            "advanced_temporal": self._advanced_temporal_slice
        }
        return X_scaled, slices, X_embed
