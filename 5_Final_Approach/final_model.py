#!/usr/bin/env python3
# Ultra Hybrid RFOD-AGD-UWA v4 (Heavier RFOD + Extended EVT + Strong Amort Gate + CLOF + OCSVM)
# - TRAIN-only fitting (unsupervised, compliant)
# - VALID only for hyperparameter / fusion selection
# - TEST for final submission
#
# Key fusion detectors:
#   amort, rfod_temporal, rfod_temporal_topq, rp_lof_embed,
#   cohort_lof_embed, ocsvm_embed
#
# Output: submission.csv

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_builder import FeatureBuilderAdvanced

RNG = np.random.default_rng(42)

CFG = dict(
    PCA_K=80,

    # RFOD-style on scaled features (heavier)
    RFOD_TOPK=48,
    RFOD_POOL_MAX=48,
    RFOD_KEEP=24,
    RFOD_T_TOPK=48,
    RFOD_T_POOL_MAX=48,
    RFOD_T_KEEP=20,
    RF_N_EST=160,
    RF_ROWSUB=0.6,
    RFOD_AGD_ALPHA=0.10,
    RFOD_TOPQ_Q=0.20,

    # CAD (heavier)
    CAD_TOPK=48,
    CAD_PAIRS=800,

    # RP-LOF (non-spherical)
    RP_BAGS=40,
    RP_DIM=60,
    RP_K=7,

    # multi-k k-distance
    KDIST_MULTI=(5, 7, 9),

    # KMeans (for km_dist + CLOF)
    KM_K=15,

    # EVT / gamma tuning (extended)
    EVT_Q_GRID=(0.85, 0.90, 0.95, 0.97, 0.98),
    GAMMA_GRID=(1.0, 1.1, 1.25, 1.4, 1.6),

    # inverse-cov shrinkage
    SHRK=2e-3,

    # fusion gates (amort weights + thresholds) extended
    FUSION_AMORT_WEIGHTS=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    FUSION_GATE_TAUS=(0.75, 0.80, 0.85, 0.90, 0.92, 0.95),

    # CLOF params
    CLOF_MIN_CLUSTER_SIZE=60,
    CLOF_KS=(5, 7, 9),

    # OCSVM params
    OCSVM_NU=0.03,
    OCSVM_GAMMA="scale",
)

# ---------- helpers ----------
def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    # Replace NaN with median; if all NaN, fall back to zeros
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=float)
    med = np.nanmedian(x)
    x = np.nan_to_num(x, nan=med)
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def train_cdf(tr: np.ndarray):
    tr = np.asarray(tr, float)
    # Clean NaNs
    if not np.isfinite(tr).any():
        # Degenerate: return identity 0.5 mapping
        def cdf(x):
            return np.full_like(x, 0.5, dtype=float)
        return np.vectorize(cdf)
    med = np.nanmedian(tr[np.isfinite(tr)])
    tr = np.nan_to_num(tr, nan=med)
    s = np.sort(tr.copy())
    def cdf(x):
        x = np.asarray(x, float)
        # Fill NaNs in query as well
        xm = np.nan_to_num(x, nan=med)
        idx = np.searchsorted(s, xm, side="right")
        return idx / (len(s) + 1e-12)
    return np.vectorize(cdf)

def invcov_weights(P_train: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    P_train = np.asarray(P_train, float)
    # Clean NaNs
    if not np.isfinite(P_train).any():
        m = P_train.shape[1]
        return np.ones(m) / m
    P_train = np.nan_to_num(P_train, nan=np.nanmedian(P_train[np.isfinite(P_train)]))
    mu = P_train.mean(axis=0, keepdims=True)
    X = P_train - mu
    Sigma = (X.T @ X) / max(1, X.shape[0] - 1)
    Sigma = Sigma + lam * np.eye(Sigma.shape[0])
    inv = np.linalg.pinv(Sigma)
    ones = np.ones((Sigma.shape[0], 1))
    w = inv @ ones
    w = w / (ones.T @ inv @ ones + 1e-12)
    return w.ravel()

# ---------- EVT + gamma ----------
def fit_evt_tail(tr: np.ndarray, q: float):
    tr = np.asarray(tr, float)
    # Clean NaNs
    if not np.isfinite(tr).any():
        return {"u": 0.0, "alpha": None, "q": q}
    med = np.nanmedian(tr[np.isfinite(tr)])
    tr = np.nan_to_num(tr, nan=med)
    s = np.sort(tr)
    u = s[int(np.floor(q * (len(s) - 1)))]
    excess = s[s > u] - u
    if len(excess) < 10 or np.all(excess <= 0):
        return {"u": u, "alpha": None, "q": q}
    logs = np.log((u + excess) / (u + 1e-12) + 1e-12)
    alpha = 1.0 / (logs.mean() + 1e-12)
    return {"u": u, "alpha": alpha, "q": q}

def evt_tail_prob(x: np.ndarray, params) -> np.ndarray:
    x = np.asarray(x, float)
    u = params["u"]
    alpha = params.get("alpha", None)
    q = params.get("q", 0.90)
    if alpha is None:
        return rank01(x)
    p = np.zeros_like(x, dtype=float)
    below = (x <= u)
    if np.any(below):
        xmin = x[below].min()
        denom = (u - xmin) + 1e-12
        p[below] = q * (x[below] - xmin) / denom
    if np.any(~below):
        z = x[~below]
        tail_surv = np.power(np.maximum(z / (u + 1e-12), 1e-12), -alpha)
        p[~below] = 1.0 - q * tail_surv
        p[~below] = np.clip(p[~below], q + 1e-6, 1.0 - 1e-9)
    return p

def _nan_safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    if not np.isfinite(y_score).any():
        return 0.0
    med = np.nanmedian(y_score[np.isfinite(y_score)])
    y_score = np.nan_to_num(y_score, nan=med)
    return average_precision_score(y_true, y_score)

def calibrate_evt_gamma(tr, va, te, q_grid, g_grid, yv):
    tr = np.asarray(tr, float)
    va = np.asarray(va, float)
    te = np.asarray(te, float)

    # If everything is NaN for this detector, just use rank-based scores.
    if not np.isfinite(tr).any():
        va_clean = rank01(va)
        te_clean = rank01(te)
        return va_clean, te_clean

    med_tr = np.nanmedian(tr[np.isfinite(tr)])
    tr_clean = np.nan_to_num(tr, nan=med_tr)

    # For validation/test, fill NaNs with their median (if all NaN, fallback to zeros)
    if np.isfinite(va).any():
        med_va = np.nanmedian(va[np.isfinite(va)])
    else:
        med_va = 0.0
    if np.isfinite(te).any():
        med_te = np.nanmedian(te[np.isfinite(te)])
    else:
        med_te = 0.0

    va_clean = np.nan_to_num(va, nan=med_va)
    te_clean = np.nan_to_num(te, nan=med_te)

    cdf = train_cdf(tr_clean)
    base_va = cdf(va_clean)
    base_te = cdf(te_clean)

    best_ap = -1
    best_va = None
    best_te = None

    for q in q_grid:
        params = fit_evt_tail(tr_clean, q=q)
        va_evt = np.maximum(base_va, evt_tail_prob(va_clean, params))
        te_evt = np.maximum(base_te, evt_tail_prob(te_clean, params))
        for g in g_grid:
            va_g = np.power(np.clip(va_evt, 0, 1), g)
            ap = _nan_safe_ap(yv, va_g)
            if ap > best_ap:
                best_ap = ap
                best_va = va_g
                best_te = np.power(np.clip(te_evt, 0, 1), g)

    # Safety: clean any residual NaNs
    if best_va is None:
        best_va = base_va
        best_te = base_te
    best_va = np.nan_to_num(best_va, nan=np.nanmedian(best_va))
    best_te = np.nan_to_num(best_te, nan=np.nanmedian(best_te))

    return best_va, best_te

# ---------- RFOD training ----------
def rfod_train_oob_select(
    Xtr: np.ndarray,
    n_estimators: int,
    target_pool_max: int,
    keep_top_targets: int,
    row_subsample: float,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    n, d = Xtr.shape
    if d == 0:
        return [], [], [], []

    if row_subsample < 1.0:
        idx_rows = rng.choice(n, size=int(max(1, n * row_subsample)), replace=False)
        Xfit = Xtr[idx_rows]
    else:
        Xfit = Xtr

    if d <= target_pool_max:
        pool = list(range(d))
    else:
        pool = rng.choice(d, size=target_pool_max, replace=False).tolist()

    models, masks, targets, oobs = [], [], [], []
    for t in pool:
        mask = np.ones(d, dtype=bool)
        mask[t] = False
        Xin = Xfit[:, mask]
        y = Xfit[:, t]
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            oob_score=True,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state + t,
        ).fit(Xin, y)
        models.append(rf)
        masks.append(mask)
        targets.append(t)
        oobs.append(float(getattr(rf, "oob_score_", -1.0)))

    order = np.argsort(np.array(oobs))[::-1]
    keep = order[:min(keep_top_targets, len(order))]
    models = [models[i] for i in keep]
    masks  = [masks[i]  for i in keep]
    targets= [targets[i] for i in keep]
    oobs   = [oobs[i]   for i in keep]
    return models, masks, targets, oobs

def rfod_agd_uwa_scores(
    models: List[RandomForestRegressor],
    masks: List[np.ndarray],
    targets: List[int],
    Xtr_rf: np.ndarray,
    X_rf: np.ndarray,
    alpha: float = 0.10,
    topq: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not models:
        n = X_rf.shape[0]
        z = np.zeros(n, dtype=float)
        return z, z

    n = X_rf.shape[0]
    m = len(models)

    AGD = np.zeros((n, m), dtype=float)
    UNC = np.zeros((n, m), dtype=float)

    q_low = []
    q_high = []
    for t in targets:
        col = Xtr_rf[:, t]
        if not np.isfinite(col).any():
            ql = 0.0
            qh = 1.0
        else:
            ql = np.quantile(col[np.isfinite(col)], alpha)
            qh = np.quantile(col[np.isfinite(col)], 1.0 - alpha)
            if qh <= ql:
                qh = ql + 1e-6
        q_low.append(ql)
        q_high.append(qh)
    q_low = np.array(q_low)
    q_high = np.array(q_high)

    for j, (rf, mask, t_idx) in enumerate(zip(models, masks, targets)):
        Xin = X_rf[:, mask]
        y = X_rf[:, t_idx]

        est_preds = np.column_stack([est.predict(Xin) for est in rf.estimators_])
        avg_pred = est_preds.mean(axis=1)
        std_pred = est_preds.std(axis=1)

        scale = (q_high[j] - q_low[j]) + 1e-12
        agd = np.abs(y - avg_pred) / scale

        AGD[:, j] = agd
        UNC[:, j] = std_pred

    unc_min = UNC.min(axis=1, keepdims=True)
    unc_max = UNC.max(axis=1, keepdims=True)
    denom = (unc_max - unc_min) + 1e-12
    unc_norm = (UNC - unc_min) / denom
    W = 1.0 - unc_norm

    row_mean = (AGD * W).sum(axis=1) / float(m)

    if topq is None or m == 1:
        return row_mean, np.zeros_like(row_mean)

    k = max(1, int(np.ceil(m * topq)))
    part = np.partition(AGD, -k, axis=1)
    row_topq = part[:, -k:].mean(axis=1)
    return row_mean, row_topq

# ---------- CAD ----------
def choose_top_variance_cols(X: np.ndarray, top_k: int) -> np.ndarray:
    v = np.var(X, axis=0)
    return np.argsort(v)[::-1][:min(top_k, X.shape[1])]

def sample_pairs(cols: np.ndarray, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    from itertools import combinations
    pairs = np.array(list(combinations(cols.tolist(), 2)), dtype=int)
    if len(pairs) <= max_pairs:
        return pairs
    sel = rng.choice(len(pairs), size=max_pairs, replace=False)
    return pairs[sel]

def cad_train(Xtr: np.ndarray, top_k: int, max_pairs: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    feat_idx = choose_top_variance_cols(Xtr, top_k)
    pair_idx = sample_pairs(feat_idx, max_pairs, rng)
    Zi = Xtr[:, pair_idx[:, 0]]
    Zj = Xtr[:, pair_idx[:, 1]]
    P = Zi * Zj
    mu = P.mean(axis=0)
    sd = P.std(axis=0) + 1e-12
    return pair_idx, mu, sd

def cad_score(X: np.ndarray, pair_idx: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Zi = X[:, pair_idx[:, 0]]
    Zj = X[:, pair_idx[:, 1]]
    P = Zi * Zj
    Z = np.abs((P - mu) / sd)
    return Z.mean(axis=1)

def cad_score_rich(X: np.ndarray, pair_idx: np.ndarray, mu: np.ndarray, sd: np.ndarray, q: float = 0.20) -> np.ndarray:
    Zi = X[:, pair_idx[:, 0]]
    Zj = X[:, pair_idx[:, 1]]
    P = Zi * Zj
    Z = np.abs((P - mu) / sd)
    m = Z.shape[1]
    k = max(1, int(np.ceil(m * q)))
    part = np.partition(Z, -k, axis=1)
    return part[:, -k:].mean(axis=1)

# ---------- RP-LOF ----------
def rp_lof_scores(Xtr: np.ndarray, X: np.ndarray, B: int = 40, dim: int = 60, k: int = 7, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    dd = min(dim, d)
    S = []
    for b in range(B):
        R = rng.normal(0, 1 / np.sqrt(dd), size=(d, dd))
        Ztr = Xtr @ R
        Z = X @ R
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Ztr)
        s = -lof.score_samples(Z)
        S.append(rank01(s))
    S = np.column_stack(S)
    return np.max(S, axis=1)

# ---------- kdist multi ----------
def kdist_multi_scores(Xtr: np.ndarray, X: np.ndarray, ks: Tuple[int, ...]) -> np.ndarray:
    scores = []
    for k in ks:
        nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
        d, _ = nn.kneighbors(X)
        scores.append(rank01(d[:, -1]))
    S = np.column_stack(scores)
    return S.mean(axis=1)

# ---------- KMeans centroid z-distance + CLOF ----------
def kmeans_centroid_zdist_train(Xtr_embed: np.ndarray, k: int = 15, seed: int = 42):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xtr_embed)
    centers = km.cluster_centers_
    labs = km.labels_
    dists = np.linalg.norm(Xtr_embed - centers[labs], axis=1)
    mu = np.zeros(k)
    sd = np.ones(k)
    for c in range(k):
        dc = dists[labs == c]
        if len(dc) > 1:
            mu[c] = dc.mean()
            sd[c] = dc.std() + 1e-12
        elif len(dc) == 1:
            mu[c] = dc.mean()
            sd[c] = 1.0
    return km, mu, sd

def kmeans_centroid_zdist_score(km: KMeans, mu: np.ndarray, sd: np.ndarray, X_embed: np.ndarray) -> np.ndarray:
    labs = km.predict(X_embed)
    dists = np.linalg.norm(X_embed - km.cluster_centers_[labs], axis=1)
    z = (dists - mu[labs]) / (sd[labs] + 1e-12)
    return np.maximum(z, 0.0)

def train_cohort_lof_on_embed(
    Xtr_embed: np.ndarray,
    km: KMeans,
    lof_ks=(5, 7, 9),
    min_cluster_size: int = 60,
):
    labs = km.labels_
    clof = {}
    for c in range(km.n_clusters):
        idx = (labs == c)
        if idx.sum() < min_cluster_size:
            continue
        Xc = Xtr_embed[idx]
        best_model = None
        best_mu = -np.inf
        for k in lof_ks:
            m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xc)
            mu = (-m.score_samples(Xc)).mean()
            if mu > best_mu:
                best_mu = mu
                best_model = m
        if best_model is not None:
            clof[c] = best_model
    return clof

def cohort_lof_embed_score(km: KMeans, clof: Dict[int, LocalOutlierFactor], X_embed: np.ndarray) -> np.ndarray:
    labs = km.predict(X_embed)
    s = np.zeros(X_embed.shape[0], dtype=float)
    for i, (x, c) in enumerate(zip(X_embed, labs)):
        if c in clof:
            s[i] = -clof[c].score_samples(x[None, :])[0]
        else:
            s[i] = 0.0
    return s

# ---------- OCSVM ----------
def ocsvm_embed_train(Xtr_embed: np.ndarray, nu=0.03, gamma="scale"):
    oc = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(Xtr_embed)
    return oc

def ocsvm_embed_score(oc: OneClassSVM, X_embed: np.ndarray) -> np.ndarray:
    return -oc.decision_function(X_embed)

# ---------- main ----------
def main():
    print("=== Ultra Hybrid RFOD-AGD-UWA v4 (Heavier + CLOF + OCSVM, NaN-safe) ===")

    train = pd.read_csv("1_Data/loans_train.csv")
    valid = pd.read_csv("1_Data/loans_valid.csv")
    test  = pd.read_csv("1_Data/loans_test.csv")
    yv = valid["target"].to_numpy()

    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=CFG["PCA_K"])
    fb.fit(train)
    Xtr_scaled, sl_tr, Xtr = fb.transform(train)   # Xtr is PCA embed
    Xv_scaled,  sl_v,  Xv  = fb.transform(valid)
    Xt_scaled,  sl_t,  Xt  = fb.transform(test)

    det_tr: Dict[str, np.ndarray] = {}
    det_v : Dict[str, np.ndarray] = {}
    det_t : Dict[str, np.ndarray] = {}

    # ---- amort detector
    am_slice = sl_v["amort"]
    def amort_sc(X_scaled: np.ndarray) -> np.ndarray:
        block = X_scaled[:, np.r_[am_slice.start:am_slice.stop]]
        c = block.shape[1]
        if c >= 6:
            mean_, p70, p50, smax, ratio, runlen = [block[:, i] for i in range(6)]
            return 0.45 * mean_ + 0.22 * p70 + 0.13 * p50 + 0.10 * smax + 0.07 * ratio + 0.03 * runlen
        mean_, p70, p50 = block[:, 0], (block[:, 1] if c >= 2 else 0), (block[:, 2] if c >= 3 else 0)
        return 0.6 * mean_ + 0.25 * p70 + 0.15 * p50

    det_tr["amort"] = amort_sc(Xtr_scaled)
    det_v ["amort"] = amort_sc(Xv_scaled)
    det_t ["amort"] = amort_sc(Xt_scaled)
    print("Added amortization detector.")

    # ---- RFOD global (scaled)
    v = np.var(Xtr_scaled, axis=0)
    idx_top = np.argsort(v)[::-1][:min(CFG["RFOD_TOPK"], Xtr_scaled.shape[1])]
    mask_top = np.zeros(Xtr_scaled.shape[1], dtype=bool)
    mask_top[idx_top] = True
    Xtr_rf = Xtr_scaled[:, mask_top]
    Xv_rf  = Xv_scaled[:, mask_top]
    Xt_rf  = Xt_scaled[:, mask_top]

    rf_models, rf_masks, rf_targets, rf_oobs = rfod_train_oob_select(
        Xtr_rf,
        n_estimators=CFG["RF_N_EST"],
        target_pool_max=min(CFG["RFOD_POOL_MAX"], Xtr_rf.shape[1]),
        keep_top_targets=min(CFG["RFOD_KEEP"], Xtr_rf.shape[1]),
        row_subsample=CFG["RF_ROWSUB"],
        random_state=42,
    )
    rfod_tr_mean, rfod_tr_topq = rfod_agd_uwa_scores(
        rf_models, rf_masks, rf_targets,
        Xtr_rf, Xtr_rf,
        alpha=CFG["RFOD_AGD_ALPHA"],
        topq=CFG["RFOD_TOPQ_Q"],
    )
    rfod_v_mean, rfod_v_topq = rfod_agd_uwa_scores(
        rf_models, rf_masks, rf_targets,
        Xtr_rf, Xv_rf,
        alpha=CFG["RFOD_AGD_ALPHA"],
        topq=CFG["RFOD_TOPQ_Q"],
    )
    rfod_t_mean, rfod_t_topq = rfod_agd_uwa_scores(
        rf_models, rf_masks, rf_targets,
        Xtr_rf, Xt_rf,
        alpha=CFG["RFOD_AGD_ALPHA"],
        topq=CFG["RFOD_TOPQ_Q"],
    )
    det_tr["rfod"]       = rfod_tr_mean
    det_v ["rfod"]       = rfod_v_mean
    det_t ["rfod"]       = rfod_t_mean
    det_tr["rfod_topq"]  = rfod_tr_topq
    det_v ["rfod_topq"]  = rfod_v_topq
    det_t ["rfod_topq"]  = rfod_t_topq
    print(f"Added RFOD global with {len(rf_targets)} targets (AGD+UWA).")

    # ---- RFOD temporal
    tsl = sl_tr["temporal"]
    if isinstance(tsl, slice) and (tsl.stop - tsl.start) > 8:
        Xtr_tmp = Xtr_scaled[:, tsl]
        Xv_tmp  = Xv_scaled[:, tsl]
        Xt_tmp  = Xt_scaled[:, tsl]

        vtmp = np.var(Xtr_tmp, axis=0)
        idx_top_t = np.argsort(vtmp)[::-1][:min(CFG["RFOD_T_TOPK"], Xtr_tmp.shape[1])]
        mtmp = np.zeros(Xtr_tmp.shape[1], dtype=bool)
        mtmp[idx_top_t] = True

        Xtr_t = Xtr_tmp[:, mtmp]
        Xv_t  = Xv_tmp[:, mtmp]
        Xt_t  = Xt_tmp[:, mtmp]

        rft_models, rft_masks, rft_targets, rft_oobs = rfod_train_oob_select(
            Xtr_t,
            n_estimators=CFG["RF_N_EST"],
            target_pool_max=min(CFG["RFOD_T_POOL_MAX"], Xtr_t.shape[1]),
            keep_top_targets=min(CFG["RFOD_T_KEEP"], Xtr_t.shape[1]),
            row_subsample=CFG["RF_ROWSUB"],
            random_state=123,
        )
        rft_tr_mean, rft_tr_topq = rfod_agd_uwa_scores(
            rft_models, rft_masks, rft_targets,
            Xtr_t, Xtr_t,
            alpha=CFG["RFOD_AGD_ALPHA"],
            topq=CFG["RFOD_TOPQ_Q"],
        )
        rft_v_mean, rft_v_topq = rfod_agd_uwa_scores(
            rft_models, rft_masks, rft_targets,
            Xtr_t, Xv_t,
            alpha=CFG["RFOD_AGD_ALPHA"],
            topq=CFG["RFOD_TOPQ_Q"],
        )
        rft_t_mean, rft_t_topq = rfod_agd_uwa_scores(
            rft_models, rft_masks, rft_targets,
            Xtr_t, Xt_t,
            alpha=CFG["RFOD_AGD_ALPHA"],
            topq=CFG["RFOD_TOPQ_Q"],
        )
        det_tr["rfod_temporal"]      = rft_tr_mean
        det_v ["rfod_temporal"]      = rft_v_mean
        det_t ["rfod_temporal"]      = rft_t_mean
        det_tr["rfod_temporal_topq"] = rft_tr_topq
        det_v ["rfod_temporal_topq"] = rft_v_topq
        det_t ["rfod_temporal_topq"] = rft_t_topq
        print(f"Added RFOD temporal with {len(rft_targets)} targets (AGD+UWA).")

    # ---- CAD
    if isinstance(tsl, slice) and (tsl.stop - tsl.start) > 8:
        Xtr_c = Xtr_scaled[:, tsl]
        Xv_c  = Xv_scaled[:, tsl]
        Xt_c  = Xt_scaled[:, tsl]
    else:
        Xtr_c = Xtr_scaled
        Xv_c  = Xv_scaled
        Xt_c  = Xt_scaled

    cad_pairs, cad_mu, cad_sd = cad_train(
        Xtr_c,
        top_k=CFG["CAD_TOPK"],
        max_pairs=CFG["CAD_PAIRS"],
    )
    det_tr["cad_corr"]      = cad_score(Xtr_c, cad_pairs, cad_mu, cad_sd)
    det_v ["cad_corr"]      = cad_score(Xv_c , cad_pairs, cad_mu, cad_sd)
    det_t ["cad_corr"]      = cad_score(Xt_c , cad_pairs, cad_mu, cad_sd)
    det_tr["cad_corr_topq"] = cad_score_rich(Xtr_c, cad_pairs, cad_mu, cad_sd, q=CFG["RFOD_TOPQ_Q"])
    det_v ["cad_corr_topq"] = cad_score_rich(Xv_c , cad_pairs, cad_mu, cad_sd, q=CFG["RFOD_TOPQ_Q"])
    det_t ["cad_corr_topq"] = cad_score_rich(Xt_c , cad_pairs, cad_mu, cad_sd, q=CFG["RFOD_TOPQ_Q"])
    print(f"Added CAD corr-change (pairs={len(cad_pairs)}).")

    # ---- RP-LOF on embed
    det_tr["rp_lof_embed"] = rp_lof_scores(
        Xtr, Xtr,
        B=CFG["RP_BAGS"],
        dim=min(CFG["RP_DIM"], Xtr.shape[1]),
        k=CFG["RP_K"],
        seed=41,
    )
    det_v ["rp_lof_embed"] = rp_lof_scores(
        Xtr, Xv,
        B=CFG["RP_BAGS"],
        dim=min(CFG["RP_DIM"], Xtr.shape[1]),
        k=CFG["RP_K"],
        seed=42,
    )
    det_t ["rp_lof_embed"] = rp_lof_scores(
        Xtr, Xt,
        B=CFG["RP_BAGS"],
        dim=min(CFG["RP_DIM"], Xtr.shape[1]),
        k=CFG["RP_K"],
        seed=43,
    )
    print("Added RP-LOF (non-spherical) on embed.")

    # ---- kdist multi (embed)
    det_tr["kdist_multi_mean"] = kdist_multi_scores(Xtr, Xtr, CFG["KDIST_MULTI"])
    det_v ["kdist_multi_mean"] = kdist_multi_scores(Xtr, Xv , CFG["KDIST_MULTI"])
    det_t ["kdist_multi_mean"] = kdist_multi_scores(Xtr, Xt , CFG["KDIST_MULTI"])

    # ---- KMeans centroid z-distance + CLOF
    km, km_mu, km_sd = kmeans_centroid_zdist_train(Xtr, k=CFG["KM_K"], seed=321)
    det_tr["km_dist_embed"] = kmeans_centroid_zdist_score(km, km_mu, km_sd, Xtr)
    det_v ["km_dist_embed"] = kmeans_centroid_zdist_score(km, km_mu, km_sd, Xv)
    det_t ["km_dist_embed"] = kmeans_centroid_zdist_score(km, km_mu, km_sd, Xt)

    clof_models = train_cohort_lof_on_embed(
        Xtr,
        km,
        lof_ks=CFG["CLOF_KS"],
        min_cluster_size=CFG["CLOF_MIN_CLUSTER_SIZE"],
    )
    det_tr["cohort_lof_embed"] = cohort_lof_embed_score(km, clof_models, Xtr)
    det_v ["cohort_lof_embed"] = cohort_lof_embed_score(km, clof_models, Xv)
    det_t ["cohort_lof_embed"] = cohort_lof_embed_score(km, clof_models, Xt)
    print("Added CLOF (cohort LOF) on embed.")

    # ---- OCSVM on embed
    oc_model = ocsvm_embed_train(
        Xtr,
        nu=CFG["OCSVM_NU"],
        gamma=CFG["OCSVM_GAMMA"],
    )
    det_tr["ocsvm_embed"] = ocsvm_embed_score(oc_model, Xtr)
    det_v ["ocsvm_embed"] = ocsvm_embed_score(oc_model, Xv)
    det_t ["ocsvm_embed"] = ocsvm_embed_score(oc_model, Xt)
    print("Added OCSVM on embed.")

    # ---- EVT + gamma calibration per detector (NaN-safe)
    names = list(det_v.keys())
    prob_train: Dict[str, np.ndarray] = {}
    prob_valid: Dict[str, np.ndarray] = {}
    prob_test : Dict[str, np.ndarray] = {}

    for n in names:
        va_p, te_p = calibrate_evt_gamma(
            det_tr[n],
            det_v[n],
            det_t[n],
            CFG["EVT_Q_GRID"],
            CFG["GAMMA_GRID"],
            yv,
        )
        prob_train[n] = train_cdf(det_tr[n])(det_tr[n])
        prob_valid[n] = va_p
        prob_test [n] = te_p

    print("[Per-detector AUPRC]:", {k: round(_nan_safe_ap(yv, prob_valid[k]), 4) for k in names})
    print("[Per-detector AUROC]:", {k: round(roc_auc_score(yv, np.nan_to_num(prob_valid[k], nan=np.nanmedian(prob_valid[k]))), 4) for k in names})

    # ---- AUPRC-tuned fusion on "golden" set
    preferred_order = [
        "amort",
        "cohort_lof_embed",
        "rp_lof_embed",
        "ocsvm_embed",
        "rfod_temporal",
        "rfod_temporal_topq",
    ]
    det_list = [d for d in preferred_order if d in prob_valid]

    if "amort" not in det_list:
        Ptrain_all = np.column_stack([prob_train[k] for k in names])
        Pv_all     = np.column_stack([prob_valid[k] for k in names])
        Pt_all     = np.column_stack([prob_test[k]  for k in names])
        w_iv_all   = invcov_weights(Ptrain_all, lam=CFG["SHRK"])
        best_valid = Pv_all @ w_iv_all
        best_test  = Pt_all @ w_iv_all
        best_name  = "ivcov_fallback"
        best_ap    = _nan_safe_ap(yv, best_valid)
        best_auc   = roc_auc_score(yv, best_valid)
        print(f"\n[Fusion best (fallback)] {best_name}  AUPRC={best_ap:.4f}  AUROC={best_auc:.4f}")
    else:
        print(f"[Fusion candidate detectors]: {det_list}")

        Ptrain = np.column_stack([prob_train[k] for k in det_list])
        Pv     = np.column_stack([prob_valid[k] for k in det_list])
        Pt     = np.column_stack([prob_test[k]  for k in det_list])

        idx_amort = det_list.index("amort")

        w_iv = invcov_weights(Ptrain, lam=CFG["SHRK"])
        cand_valid: Dict[str, np.ndarray] = {}
        cand_test : Dict[str, np.ndarray] = {}

        cand_valid["ivcov"] = Pv @ w_iv
        cand_test["ivcov"]  = Pt @ w_iv

        cand_valid["avg"] = Pv.mean(axis=1)
        cand_test ["avg"] = Pt.mean(axis=1)
        cand_valid["max"] = Pv.max(axis=1)
        cand_test ["max"] = Pt.max(axis=1)

        am_v = prob_valid["amort"]
        am_t = prob_test["amort"]
        m = Pv.shape[1]

        for a_w in CFG["FUSION_AMORT_WEIGHTS"]:
            pat_name = f"amort_{a_w:.2f}"
            w = np.zeros(m, dtype=float)
            w[idx_amort] = a_w
            if m > 1:
                rem = (1.0 - a_w) / (m - 1)
                for j in range(m):
                    if j != idx_amort:
                        w[j] = rem

            base_v = Pv @ w
            base_t = Pt @ w

            key_lin = f"wlin::{pat_name}"
            cand_valid[key_lin] = base_v
            cand_test[key_lin]  = base_t

            for tau in CFG["FUSION_GATE_TAUS"]:
                key_g = f"wgate::{pat_name}::tau{tau:.2f}"
                cand_valid[key_g] = np.where(am_v >= tau, am_v, base_v)
                cand_test[key_g]  = np.where(am_t >= tau, am_t, base_t)

        cand_valid["amort_only"] = am_v
        cand_test ["amort_only"] = am_t
        cand_valid["amort_ivcov_mix_0.5"] = 0.5 * am_v + 0.5 * cand_valid["ivcov"]
        cand_test ["amort_ivcov_mix_0.5"] = 0.5 * am_t + 0.5 * cand_test["ivcov"]

        best_name = None
        best_ap = -1.0
        best_auc = -1.0
        best_valid = None
        best_test = None

        for name, sv in cand_valid.items():
            ap  = _nan_safe_ap(yv, sv)
            auc = roc_auc_score(yv, np.nan_to_num(sv, nan=np.nanmedian(sv)))
            if ap > best_ap:
                best_ap = ap
                best_auc = auc
                best_name = name
                best_valid = sv
                best_test = cand_test[name]

        print(f"\n[Fusion best AUPRC-tuned v4] {best_name}  AUPRC={best_ap:.4f}  AUROC={best_auc:.4f}")
        print(f"[Fusion keep detectors]: {det_list}")

    idc = "Id" if "Id" in test.columns else "index"
    sub = pd.DataFrame({idc: test[idc], "anomaly_score": best_test})
    sub.to_csv("5_Final_Approach/submission.csv", index=False)
    print("Saved submission.csv to 5_Final_Approach/")

if __name__ == "__main__":
    main()
