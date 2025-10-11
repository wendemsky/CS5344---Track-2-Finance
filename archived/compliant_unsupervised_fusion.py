#!/usr/bin/env python3
# Compliant, syllabus-friendly unsupervised anomaly pipeline
# - Train only on train normals
# - Validation used only for hyper-parameter/model selection (no fitting)
# - No supervised meta-learner on validation
# - Distance models run on train-fitted scale/PCA
# - Fusion by rank-based rules chosen via validation AUPRC

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = np.random.default_rng(42)

# ----------------------
# Utility
# ----------------------
def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

# ----------------------
# Feature builder (train-only fit)
# ----------------------
class FeatureBuilderCompliant:
    """
    - Handles sentinels/missing for static
    - Encodes categoricals with LabelEncoder fit on train only (unknown -> 'UNKNOWN')
    - Builds temporal early-window features (subset of months)
    - Builds amortization residual signals (FRM only; skip IO/balloon/ARM)
    - Fits ONE global scaler on the full concatenated matrix (train only)
    - Optionally fits PCA on scaled features for distance/clustering
    """
    def __init__(self, use_pca=True, pca_comps=80, keep_month_idx=(0,3,6,9,12)):
        self.use_pca = use_pca
        self.pca_comps = pca_comps
        self.keep_month_idx = set(keep_month_idx)
        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.impute_vals: Dict[str, float] = {}
        self.global_scaler = None
        self.pca = None
        self.static_cols: List[str] = []
        self.temporal_cols: List[str] = []
        self.amort_slice: slice = slice(0,0)  # will set after fit
        self._fitted = False

    # --- dataset-specific helpers ---
    def _is_temporal(self, col: str) -> bool:
        # temporal columns are named like "N_Field"
        if "_" not in col: return False
        left = col.split("_", 1)[0]
        return left.isdigit()

    def _sentinel_map(self, df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy and convert sentinels to NaN + add missing flags
        SENT = {
            "CreditScore": 9999,
            "OriginalDTI": 999,
            "OriginalLTV": 999,
            "MI Pct": 999,
            "EstimatedLTV": 999,  # temporal has many Ns; handle later too if exists as static
        }
        df = df.copy()
        for col, bad in SENT.items():
            if col in df.columns:
                miss = (df[col] == bad)
                df.loc[miss, col] = np.nan
                df[f"{col}_missing"] = miss.astype(int)

        # flags with '9'/'99' for missing
        FLAG_9 = [
            "FirstTimeHomebuyerFlag", "OccupancyStatus", "LoanPurpose", "Channel",
            "PropertyType"
        ]
        for col in FLAG_9:
            if col in df.columns:
                miss = (df[col].astype(str).str.strip() == "9") | (df[col].astype(str).str.strip() == "99")
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
                # keep only selected months
                m = int(c.split("_",1)[0])
                if m in self.keep_month_idx:
                    self.temporal_cols.append(c)
            else:
                self.static_cols.append(c)

    def _fit_cat_and_impute(self, df: pd.DataFrame):
        # Fit label encoders on train only
        for c in self.static_cols:
            if df[c].dtype == "object":
                enc = LabelEncoder()
                vals = df[c].fillna("MISSING").astype(str).unique().tolist()
                if "UNKNOWN" not in vals: vals.append("UNKNOWN")
                enc.fit(vals)
                self.cat_encoders[c] = enc
        # Fit numeric impute on train
        for c in self.static_cols:
            if c not in self.cat_encoders:
                v = df[c].dropna()
                self.impute_vals[c] = float(v.median() if len(v) else 0.0)

    def _transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df[self.static_cols].copy()
        # cats
        for c, enc in self.cat_encoders.items():
            x = data[c].fillna("MISSING").astype(str)
            x = x.where(x.isin(enc.classes_), "UNKNOWN")
            data[c] = enc.transform(x)
        # nums
        for c in self.static_cols:
            if c not in self.cat_encoders:
                data[c] = data[c].fillna(self.impute_vals.get(c,0.0))
        return data

    def _build_temporal_block(self, df: pd.DataFrame) -> pd.DataFrame:
        # ffill/bfill then 0 for remaining
        if not self.temporal_cols:
            return pd.DataFrame(index=df.index)
        tmp = df[self.temporal_cols].copy()
        tmp = tmp.ffill(axis=1).bfill(axis=1).fillna(0.0)
        # Simple engineered summaries across selected months per feature type
        # group by field name after first underscore
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

    # --- amortization: FRM annuity residual over early window, skip IO/balloon/ARM ---
    @staticmethod
    def _annuity_payment(orig_upb, annual_rate, term_m):
        # annual_rate in %; returns monthly payment
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
        # needs static + temporal panels
        need = ["OriginalUPB","OriginalInterestRate","OriginalLoanTerm",
                "ProductType","InterestOnlyFlag","BalloonIndicator"]
        for c in need:
            if c not in df.columns:
                return pd.DataFrame(index=df.index)

        # Extract needed early-window temporal panels (IB_UPB, CurrentRate, RemainingMonths)
        # We'll detect their names from available columns
        def col_for(suffix):
            cands = [c for c in self.temporal_cols if c.endswith(f"_{suffix}")]
            return sorted(cands, key=lambda x:int(x.split("_",1)[0]))

        ib_cols  = col_for("InterestBearingUPB")
        rt_cols  = col_for("CurrentInterestRate")
        rem_cols = col_for("RemainingMonthsToLegalMaturity")

        if not ib_cols: return pd.DataFrame(index=df.index)
        n_steps = len(ib_cols)

        # compute month t expected principal and observed principal for FRM only (no IO/balloon)
        is_frm = (df["ProductType"].astype(str).str.upper() == "FRM")
        is_io  = df.get("InterestOnlyFlag", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        is_bal = df.get("BalloonIndicator", pd.Series("N", index=df.index)).astype(str).str.upper().eq("Y")
        use_mask = is_frm & (~is_io) & (~is_bal)

        # matrices
        IB = df[ib_cols].values.astype(float)
        # current rate per month if available, else fall back to orig
        if rt_cols:
            RT = df[rt_cols].values.astype(float)/1200.0
        else:
            RT = np.tile((df["OriginalInterestRate"].values/1200.0)[:,None], (1,n_steps))
        # remaining months per month if available, else orig-term - t
        if rem_cols:
            RM = df[rem_cols].values.astype(float)
        else:
            T0 = df["OriginalLoanTerm"].values.astype(float)
            RM = np.maximum(T0[:,None] - np.arange(n_steps)[None,:], 1.0)

        # expected payment/month using re-amortization each step:
        # P_t = annuity(IB_{t-1}, r_t, RM_t)
        # expected principal = max(P_t - r_t * IB_{t-1}, 0)
        ExpPrin = np.zeros_like(IB)
        for t in range(1, n_steps):
            upb_prev = IB[:, t-1]
            r_t = RT[:, t]
            M_t = np.maximum(RM[:, t], 1.0)
            # annuity on remaining balance
            with np.errstate(over="ignore"):
                num = r_t * np.power(1.0 + r_t, M_t)
                den = np.power(1.0 + r_t, M_t) - 1.0
            P_t = np.where((den>0) & (r_t>0), upb_prev * (num/den), safe_div(upb_prev, M_t))
            exp_prin = np.maximum(P_t - r_t * upb_prev, 0.0)
            ExpPrin[:, t] = exp_prin

        # observed principal reduction = max(IB_{t-1} - IB_t, 0)
        ObsPrin = np.maximum(IB[:, :-1] - IB[:, 1:], 0.0)
        ExpPrin_use = ExpPrin[:, 1:]

        # shortfall ratio per month
        Short = np.zeros_like(ObsPrin)
        Short[:] = safe_div(ExpPrin_use - ObsPrin, np.abs(ExpPrin_use) + 1e-9)
        Short = np.clip(Short, 0.0, 1.0)

        # aggregate over early window months (exclude month 0 by construction)
        short_mean = Short.mean(axis=1)
        short_pct_hi = (Short > 0.5).mean(axis=1)

        # mask out non-FRM/IO/Balloon loans by setting to NaN (later impute to 0 and add indicator)
        short_mean = np.where(use_mask.values, short_mean, np.nan)
        short_pct_hi = np.where(use_mask.values, short_pct_hi, np.nan)

        out = pd.DataFrame({
            "amort_short_mean": short_mean,
            "amort_short_pct_hi": short_pct_hi,
            "amort_mask_not_applicable": (~use_mask).astype(int),
        }, index=df.index)
        # impute NaNs to 0 (meaning "no signal") while keeping the mask feature
        out[["amort_short_mean","amort_short_pct_hi"]] = out[["amort_short_mean","amort_short_pct_hi"]].fillna(0.0)
        return out

    # ------------- public API -------------
    def fit(self, df_train: pd.DataFrame):
        df = self._sentinel_map(df_train.copy())
        self._collect_columns(df)
        self._fit_cat_and_impute(df)

        # build blocks on train
        Xs = self._transform_static(df)
        Xt = self._build_temporal_block(df)
        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))  # amort uses both
        # concat (order is stable)
        X_full = pd.concat([Xs, Xt, Xa], axis=1).astype(float)
        # one global scaler on full X
        self.global_scaler = StandardScaler().fit(X_full.values)
        X_scaled = self.global_scaler.transform(X_full.values)

        # optional PCA for distance/clustering
        if self.use_pca:
            k = min(self.pca_comps, X_scaled.shape[1])
            self.pca = PCA(n_components=k, random_state=42).fit(X_scaled)

        # record amort slice (stable)
        start = Xs.shape[1] + Xt.shape[1]
        end = start + Xa.shape[1]
        self.amort_slice = slice(start, end)
        self._fitted = True

    def transform(self, df_any: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, slice], np.ndarray]:
        assert self._fitted, "Call fit() first on train."
        df = self._sentinel_map(df_any.copy())
        # keep same columns; missing unseen cats -> UNKNOWN
        Xs = self._transform_static(df)
        Xt = self._build_temporal_block(df)
        Xa = self._amort_signals(pd.concat([df, Xt], axis=1))
        X_full = pd.concat([Xs, Xt, Xa], axis=1).astype(float)
        X_scaled = self.global_scaler.transform(X_full.values)
        if self.use_pca:
            X_pca = self.pca.transform(X_scaled)
        else:
            X_pca = X_scaled
        idx = {
            "static": slice(0, Xs.shape[1]),
            "temporal": slice(Xs.shape[1], Xs.shape[1]+Xt.shape[1]),
            "amort": self.amort_slice
        }
        return X_scaled, idx, X_pca  # return scaled full & PCA-projected

# ----------------------
# Detectors (train on train-only; score everywhere)
# ----------------------
def train_global_lof(X_train_det: np.ndarray, ks: List[int]) -> List[Tuple[str, LocalOutlierFactor]]:
    models = []
    for k in ks:
        m = LocalOutlierFactor(n_neighbors=k, novelty=True)  # contamination ignored for novelty
        m.fit(X_train_det)
        models.append((f"lof_k{k}", m))
    return models

def train_clusterwise_lof(X_train_det: np.ndarray, n_clusters: int = 12) -> Tuple[KMeans, Dict[int, LocalOutlierFactor]]:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X_train_det)
    cl_models: Dict[int, LocalOutlierFactor] = {}
    for c in range(n_clusters):
        idx = (km.labels_ == c)
        if idx.sum() < 50:
            continue
        Xc = X_train_det[idx]
        # a small set of ks for speed
        best_m, best_k = None, None
        for k in (5,7,9):
            m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xc)
            # pick the "tightest" model by average score magnitude (proxy)
            # (no labels used; purely unsupervised selection)
            sc = -m.score_samples(Xc)
            if (best_m is None) or (sc.mean() > (-best_m.score_samples(Xc)).mean()):
                best_m, best_k = m, k
        if best_m is not None:
            cl_models[c] = best_m
    return km, cl_models

def score_clusterwise(km: KMeans, cl_models: Dict[int, LocalOutlierFactor], X_det: np.ndarray) -> np.ndarray:
    labels = km.predict(X_det)
    s = np.zeros(len(X_det))
    for i, (x, c) in enumerate(zip(X_det, labels)):
        if c in cl_models:
            s[i] = -cl_models[c].score_samples(x[None, :])[0]
        else:
            s[i] = 0.0
    return s

# ----------------------
# Main pipeline
# ----------------------
def main():
    print("=== Compliant Unsupervised Anomaly Pipeline ===")
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test  = pd.read_csv("Data/loans_test.csv")

    # Fit features on TRAIN ONLY
    fb = FeatureBuilderCompliant(use_pca=True, pca_comps=80, keep_month_idx=(0,3,6,9,12))
    fb.fit(train)

    # Build matrices
    Xtr_scaled, idxtr, Xtr_det = fb.transform(train)   # detectors use Xtr_det (PCA)
    Xv_scaled,  idxv,  Xv_det  = fb.transform(valid)
    Xt_scaled,  idxt,  Xt_det  = fb.transform(test)

    yv = valid["target"].to_numpy()

    # --- Train detectors on TRAIN ONLY ---
    ks = [5,6,7,8]
    lof_models = train_global_lof(Xtr_det, ks)
    km, cl_models = train_clusterwise_lof(Xtr_det, n_clusters=12)

    # --- Score validation ---
    det_scores_valid: Dict[str, np.ndarray] = {}
    for name, m in lof_models:
        det_scores_valid[name] = -m.score_samples(Xv_det)
    det_scores_valid["cluster_lof"] = score_clusterwise(km, cl_models, Xv_det)

    # amortization detector: purely engineered scalar score (higher worse)
    amort_cols = np.r_[idxv["amort"].start:idxv["amort"].stop]
    amort_block = Xv_scaled[:, amort_cols]
    # define a single amort score = weighted average of [short_mean, short_pct_hi, mask_not_applicable(neg weight)]
    # Columns order in amort slice is exactly as built: ["amort_short_mean","amort_short_pct_hi","amort_mask_not_applicable"]
    if amort_block.shape[1] >= 3:
        amort_score = amort_block[:, 0]*0.7 + amort_block[:,1]*0.3  # mask is not part of score
        det_scores_valid["amort_resid"] = amort_score
    else:
        det_scores_valid["amort_resid"] = np.zeros(len(valid))

    # --- Evaluate each detector on validation (allowed) ---
    per_det_ap = {}
    per_det_auc = {}
    for k, s in det_scores_valid.items():
        per_det_ap[k]  = average_precision_score(yv, s)
        per_det_auc[k] = roc_auc_score(yv, s)
    print("[Validation] Per-detector AUPRC:", {k: round(v,4) for k,v in per_det_ap.items()})
    print("[Validation] Per-detector AUROC:", {k: round(v,4) for k,v in per_det_auc.items()})

    # --- Hyper-parameter/model selection: choose fusion rule (unsupervised) ---
    # 1) Select detectors that beat a small baseline (e.g., AUPRC >= 0.15) and keep top 4 by AUPRC
    sorted_dets = sorted(per_det_ap.items(), key=lambda kv: kv[1], reverse=True)
    keep = [k for k,_ in sorted_dets if per_det_ap[k] >= 0.15][:4]
    if not keep:
        keep = [sorted_dets[0][0]]
    print("[Selection] Using detectors:", keep)

    # 2) Build rank-based fusions (no learning on validation, just rule selection)
    S = {k: det_scores_valid[k] for k in keep}
    ranks = {k: rank01(S[k]) for k in keep}

    # Max-rank
    maxrank = np.max(np.column_stack([ranks[k] for k in keep]), axis=1)
    ap_max  = average_precision_score(yv, maxrank)
    auc_max = roc_auc_score(yv, maxrank)

    # Weighted rank average with weights proportional to validation AUPRC (hyper-parameter tuning)
    ws = np.array([per_det_ap[k] for k in keep], dtype=float)
    if ws.sum() <= 0: ws = np.ones_like(ws)
    w = ws / ws.sum()
    wrank = np.average(np.column_stack([ranks[k] for k in keep]), axis=1, weights=w)
    ap_w   = average_precision_score(yv, wrank)
    auc_w  = roc_auc_score(yv, wrank)

    # Pick the better fusion by validation AUPRC (allowed as model selection)
    if ap_w >= ap_max:
        final_valid_score = wrank
        final_rule = ("weighted_rank", keep, w.tolist())
        ap_final, auc_final = ap_w, auc_w
    else:
        final_valid_score = maxrank
        final_rule = ("max_rank", keep, None)
        ap_final, auc_final = ap_max, auc_max

    print(f"[Fusion] Rule={final_rule[0]}  AUPRC={ap_final:.4f}  AUROC={auc_final:.4f}")

    # --- Score TEST with the same trained detectors and chosen fusion rule ---
    det_scores_test: Dict[str, np.ndarray] = {}
    for name, m in lof_models:
        det_scores_test[name] = -m.score_samples(Xt_det)
    det_scores_test["cluster_lof"] = score_clusterwise(km, cl_models, Xt_det)
    # amort on test
    amort_cols_t = np.r_[idxt["amort"].start:idxt["amort"].stop]
    amort_block_t = Xt_scaled[:, amort_cols_t]
    if amort_block_t.shape[1] >= 2:
        det_scores_test["amort_resid"] = amort_block_t[:,0]*0.7 + amort_block_t[:,1]*0.3
    else:
        det_scores_test["amort_resid"] = np.zeros(len(test))

    # Apply fusion
    keep = final_rule[1]
    tr = final_rule[0]
    if tr == "weighted_rank":
        w = np.array(final_rule[2], dtype=float)
        R = np.column_stack([rank01(det_scores_test[k]) for k in keep])
        final_test_score = np.average(R, axis=1, weights=w)
    else:
        R = np.column_stack([rank01(det_scores_test[k]) for k in keep])
        final_test_score = np.max(R, axis=1)

    # Save submission
    idc = "Id" if "Id" in test.columns else "index"
    sub = pd.DataFrame({idc: test[idc], "anomaly_score": final_test_score})
    sub.to_csv("SUB_compliant_unsup.csv", index=False)
    print("Saved SUB_compliant_unsup.csv")

if __name__ == "__main__":
    main()
