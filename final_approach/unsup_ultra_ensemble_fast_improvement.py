#!/usr/bin/env python3
# Ultra ensemble Fast Improvement (unsupervised, compliant):
# - Fit everything on TRAIN (normals only)
# - VALID used only to choose hyperparams & fusion rule (no fitting on valid)
# - Uses ORIGINAL feature builder (fast)
# - Adds ONE fast, potentially powerful detector: Mahalanobis on SCALED features (using standard cov)
# - Adds new fusion rules (top-3, wavg-top-2) alongside original winner (top-2)
# - Saves SUB_ultra_unsup_fast_impr.csv

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

from sklearn.metrics import average_precision_score, roc_auc_score

# Import the ORIGINAL, fast feature builder
from feature_builder_advanced import FeatureBuilderAdvanced

RNG = np.random.default_rng(42)

# ---------- helpers ----------
def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def train_cdf(scores_train: np.ndarray):
    # empirical CDF over train scores -> maps score to tail probability in [0,1]
    s = np.sort(scores_train.copy())
    def cdf(x):
        idx = np.searchsorted(s, x, side='right')
        return idx / (len(s) + 1e-12)
    return np.vectorize(cdf)

def zscore_by_cluster(train_scores: np.ndarray, train_labels: np.ndarray):
    # returns dict: cluster -> (mean, std)
    out = {}
    for c in np.unique(train_labels):
        sc = train_scores[train_labels == c]
        mu = np.mean(sc); sd = np.std(sc) + 1e-12
        out[int(c)] = (float(mu), float(sd))
    return out

# --- ADDED MISSING FUNCTION ---
def pca_recon_error_train(Xtr_scaled: np.ndarray, n_comp=50) -> Tuple[PCA, np.ndarray]:
    p = PCA(n_components=min(n_comp, Xtr_scaled.shape[1]), random_state=42).fit(Xtr_scaled)
    Xr = p.inverse_transform(p.transform(Xtr_scaled))
    train_err = np.mean((Xtr_scaled - Xr)**2, axis=1)
    return p, train_err
# --- END ADDED MISSING FUNCTION ---

def ee_train(Xtr: np.ndarray, support_frac=0.9) -> EllipticEnvelope:
    # robust covariance on PCA space
    return EllipticEnvelope(contamination=1.0-support_frac, support_fraction=support_frac, random_state=42).fit(Xtr)

# ---------- detectors (train-only fit) ----------
def lof_models_train(Xtr: np.ndarray, ks: List[int]) -> Dict[str, LocalOutlierFactor]:
    models = {}
    for k in ks:
        m = LocalOutlierFactor(n_neighbors=k, novelty=True)  # unsup novelty
        m.fit(Xtr)
        models[f"lof_k{k}"] = m
    return models

def kdist_model_train(Xtr: np.ndarray, ks: List[int]) -> Dict[str, NearestNeighbors]:
    models = {}
    for k in ks:
        nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
        models[f"kdist_k{k}"] = nn
    return models

def kdist_score(nn: NearestNeighbors, X: np.ndarray) -> np.ndarray:
    d, _ = nn.kneighbors(X)
    return d[:, -1]

def iforest_train(Xtr: np.ndarray, n_estimators=500, max_samples=1.0, rng=42) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators, max_samples=max_samples,
        contamination="auto", random_state=rng, n_jobs=-1
    ).fit(Xtr)

# --- NEW FAST DETECTOR: Mahalanobis on SCALED features using standard covariance ---
def mahal_score_train_fast(Xtr_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: # Use scaled features
    # Standard (non-robust) covariance for Mahalanobis distance - FASTER than MinCovDet
    mu = np.mean(Xtr_scaled, axis=0)
    cov = np.cov(Xtr_scaled, rowvar=False)
    # Ensure covariance matrix is invertible (add small regularization if needed)
    if not np.all(np.linalg.eigvals(cov) > 0):
        # Add small regularization if covariance is singular or near-singular
        reg_factor = 1e-6
        while not np.all(np.linalg.eigvals(cov) > 0):
            cov += reg_factor * np.eye(cov.shape[0])
            reg_factor *= 10
    inv_cov = np.linalg.inv(cov)
    return mu, inv_cov

def mahal_score_fast(X_scaled: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    diff = X_scaled - mu
    # Calculate Mahalanobis distance squared
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)
    # Return square root (standard Mahalanobis distance)
    return np.sqrt(md_sq)

# --- END NEW FAST DETECTOR ---

def random_proj_lof_scores(Xtr: np.ndarray, X: np.ndarray, B=40, dim=60, k=7, seed=42):
    """Random Projection bagging for LOF. Train-only projections; apply to X."""
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    scores = []
    for b in range(B):
        # Gaussian random projection
        R = rng.normal(0, 1/np.sqrt(dim), size=(d, dim))
        Ztr = Xtr @ R
        Z   = X  @ R
        m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Ztr)
        s = -m.score_samples(Z)
        scores.append(rank01(s))
    S = np.column_stack(scores)
    return np.max(S, axis=1)  # MaxRank across bags

# ---------- fusion ----------
def fuse_candidates(score_map_valid: Dict[str, np.ndarray], weights: Dict[str, float]) -> Dict[str, np.ndarray]:
    # rank-based
    names = list(score_map_valid.keys())
    R = np.column_stack([rank01(score_map_valid[n]) for n in names])
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / (w.sum() + 1e-12)

    cand = {}
    cand["max_rank"]    = np.max(R, axis=1)
    cand["median_rank"] = np.median(R, axis=1)
    cand["wavg_rank"]   = np.average(R, axis=1, weights=w)

    # top-2 max (original winner)
    top2 = sorted(names, key=lambda n: weights[n], reverse=True)[:2]
    R2   = np.column_stack([rank01(score_map_valid[n]) for n in top2])
    cand["max_rank_top2"] = np.max(R2, axis=1)

    # top-3 max (new)
    top3 = sorted(names, key=lambda n: weights[n], reverse=True)[:3]
    R3   = np.column_stack([rank01(score_map_valid[n]) for n in top3])
    cand["max_rank_top3"] = np.max(R3, axis=1)

    # top-2 weighted avg (new)
    top2_w = np.array([weights[n] for n in top2])
    top2_w = top2_w / top2_w.sum()
    cand["wavg_rank_top2"] = np.average(R2, axis=1, weights=top2_w)

    return cand

def fuse_candidates_prob(prob_map_valid: Dict[str, np.ndarray], weights: Dict[str, float]) -> Dict[str, np.ndarray]:
    names = list(prob_map_valid.keys())
    P = np.column_stack([prob_map_valid[n] for n in names])
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / (w.sum() + 1e-12)

    cand = {}
    # average probability
    cand["p_avg"] = np.average(P, axis=1, weights=w)
    # noisy-OR
    cand["p_noisy_or"] = 1.0 - np.prod(1.0 - P, axis=1)
    # max prob
    cand["p_max"] = np.max(P, axis=1)
    return cand

# ---------- main ----------
def main():
    print("=== Ultra Unsupervised Ensemble Fast Improvement (Compliant) ===")
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test  = pd.read_csv("Data/loans_test.csv")

    yv = valid["target"].to_numpy()

    # Use the winning PCA setting from original run
    pca_k = 80
    print(f"\n--- Using PCA setting: {pca_k} comps (from original winner) ---")

    # Fit the ORIGINAL, FAST feature builder
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=pca_k)
    fb.fit(train)

    Xtr_scaled, sl_tr, Xtr = fb.transform(train)
    Xv_scaled,  sl_v,  Xv  = fb.transform(valid)
    Xt_scaled,  sl_t,  Xt  = fb.transform(test)

    # ---------------- train detectors on TRAIN only ----------------
    detectors_scores_valid: Dict[str, np.ndarray] = {}
    detectors_scores_test : Dict[str, np.ndarray] = {}
    detectors_scores_train: Dict[str, np.ndarray] = {}

    # LOF multi-k (Original)
    lof_ks = [4,5,6,7,8,10,12] # Keep original range
    lof_models = lof_models_train(Xtr, lof_ks)
    for name, m in lof_models.items():
        detectors_scores_valid[name] = -m.score_samples(Xv)
        detectors_scores_test[name]  = -m.score_samples(Xt)
        detectors_scores_train[name] = -m.score_samples(Xtr)

    # KMeans + cluster-wise LOF (Original)
    km = KMeans(n_clusters=12, n_init=10, random_state=42).fit(Xtr)
    # choose per-cluster LOF k by internal magnitude
    cl_models = {}
    for c in range(km.n_clusters):
        idx = (km.labels_ == c)
        if idx.sum() < 60: continue
        Xc = Xtr[idx]
        best, best_mu = None, -np.inf
        for k in (5,7,9): # Keep original k range for cluster LOF
            m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xc)
            mu = (-m.score_samples(Xc)).mean()
            if mu > best_mu:
                best_mu, best = mu, m
        if best is not None:
            cl_models[c] = best

    def score_clusterwise(X):
        labs = km.predict(X)
        s = np.zeros(len(X))
        for i, (x, c) in enumerate(zip(X, labs)):
            if c in cl_models:
                s[i] = -cl_models[c].score_samples(x[None,:])[0]
        return s

    detectors_scores_valid["cluster_lof"] = score_clusterwise(Xv)
    detectors_scores_test["cluster_lof"]  = score_clusterwise(Xt)
    detectors_scores_train["cluster_lof"] = score_clusterwise(Xtr)

    # k-distance multi-k (Original)
    kd_ks = [3,5,7,9,11] # Keep original range
    kd_models = kdist_model_train(Xtr, kd_ks)
    for name, nn in kd_models.items():
        detectors_scores_valid[name] = kdist_score(nn, Xv)
        detectors_scores_test[name]  = kdist_score(nn, Xt)
        detectors_scores_train[name] = kdist_score(nn, Xtr)

    # Isolation Forest (PCA space) (Original)
    if_model = iforest_train(Xtr, n_estimators=500, max_samples=1.0, rng=42)
    detectors_scores_valid["iforest"] = -if_model.score_samples(Xv)
    detectors_scores_test["iforest"]  = -if_model.score_samples(Xt)
    detectors_scores_train["iforest"] = -if_model.score_samples(Xtr)

    # Elliptic Envelope (PCA space) (Original)
    try:
        ee_model = ee_train(Xtr, support_frac=0.9)
        detectors_scores_valid["ell_env"] = -ee_model.score_samples(Xv)
        detectors_scores_test["ell_env"]  = -ee_model.score_samples(Xt)
        detectors_scores_train["ell_env"] = -ee_model.score_samples(Xtr)
    except Exception:
        pass # Keep original exception handling

    # PCA reconstruction error (train-only PCA on scaled full) - ORIGINAL
    pre_pca, pre_train_err = pca_recon_error_train(Xtr_scaled, n_comp=min(80, Xtr_scaled.shape[1]))
    def recon_err(X_scaled):
        Xr = pre_pca.inverse_transform(pre_pca.transform(X_scaled))
        return np.mean((X_scaled - Xr)**2, axis=1)
    detectors_scores_train["pca_recon"] = recon_err(Xtr_scaled)
    detectors_scores_valid["pca_recon"] = recon_err(Xv_scaled)
    detectors_scores_test["pca_recon"]  = recon_err(Xt_scaled)

    # Random-Projection LOF bagging (MaxRank within bags) (Original)
    detectors_scores_valid["rp_lof"] = random_proj_lof_scores(Xtr, Xv, B=40, dim=min(60, Xtr.shape[1]), k=7, seed=42)
    detectors_scores_test["rp_lof"]  = random_proj_lof_scores(Xtr, Xt, B=40, dim=min(60, Xtr.shape[1]), k=7, seed=43)
    detectors_scores_train["rp_lof"] = random_proj_lof_scores(Xtr, Xtr, B=40, dim=min(60, Xtr.shape[1]), k=7, seed=41)

    # amortization residual (from scaled full features) - ORIGINAL
    am_slice = sl_v["amort"]
    if am_slice.stop - am_slice.start >= 3:
        def amort_sc(X_scaled):
            block = X_scaled[:, np.r_[am_slice.start:am_slice.stop]]
            # order: mean, 70, 50, mask_not_applicable (depending on builder)
            # build a simple monotone score
            base = 0.6*block[:,0] + 0.25*block[:,1] + 0.15*block[:,2]
            return base
        detectors_scores_valid["amort"] = amort_sc(Xv_scaled)
        detectors_scores_test["amort"]  = amort_sc(Xt_scaled)
        detectors_scores_train["amort"] = amort_sc(Xtr_scaled)

    # --- NEW FAST DETECTOR: Mahalanobis on SCALED features ---
    try:
        mahal_mu, mahal_inv_cov = mahal_score_train_fast(Xtr_scaled) # Use the SCALED features from the builder
        detectors_scores_valid["mahal_fast"] = mahal_score_fast(Xv_scaled, mahal_mu, mahal_inv_cov)
        detectors_scores_test["mahal_fast"]  = mahal_score_fast(Xt_scaled, mahal_mu, mahal_inv_cov)
        detectors_scores_train["mahal_fast"] = mahal_score_fast(Xtr_scaled, mahal_mu, mahal_inv_cov)
        print("Added FAST Mahalanobis detector (on scaled features).")
    except Exception as e:
        print(f"FAST Mahalanobis failed: {e}")
        # If it fails, the detector won't be included in 'keep' later

    # ---------------- per-detector VALID metrics ----------------
    per_ap = {k: average_precision_score(yv, v) for k, v in detectors_scores_valid.items()}
    per_auc = {k: roc_auc_score(yv, v) for k, v in detectors_scores_valid.items()}
    # Print top 10 to see if mahal_fast made a difference
    print("[Per-detector AUPRC (Top 10)]:", {k: round(v,4) for k,v in sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)[:10]})
    print("[Per-detector AUROC (Top 10)]:", {k: round(v,4) for k,v in sorted(per_auc.items(), key=lambda kv: kv[1], reverse=True)[:10]})

    # keep top detectors (threshold + cap) - Keep original logic
    sorted_dets = sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)
    keep = [k for k,_ in sorted_dets if per_ap[k] >= 0.16][:10] # Keep original threshold and cap
    if not keep:
        keep = [sorted_dets[0][0]]
    print("[Selected detectors]:", keep)

    # ---------------- train-only calibration (CDF) ----------------
    cdf_map = {}
    prob_valid = {}
    prob_test  = {}
    weights = {}
    for name in keep:
        tr = detectors_scores_train.get(name, None)
        if tr is None:
            # fallback: use rank on valid as pseudo-prob
            cdf_map[name] = None
            prob_valid[name] = rank01(detectors_scores_valid[name])
            prob_test[name]  = rank01(detectors_scores_test[name])
        else:
            cdf = train_cdf(tr)
            cdf_map[name] = cdf
            prob_valid[name] = cdf(detectors_scores_valid[name])
            prob_test[name]  = cdf(detectors_scores_test[name])
        # weight by VALID AUPRC
        weights[name] = max(per_ap[name], 1e-6)

    # ---------------- cohort normalization (train-only stats) ----------------
    # Use the same KMeans from cluster_LOF for consistency
    # labs_tr = km.labels_ # Already computed for cluster_LOF
    labs_v  = km.predict(Xv)
    labs_t  = km.predict(Xt)

    z_valid_map = {}
    z_test_map  = {}

    for name in keep:
        # compute train cluster stats for this detector
        mu_sd = {}
        sc_tr = detectors_scores_train[name]
        stats = zscore_by_cluster(sc_tr, km.labels_) # Use km.labels_ which is from training
        mu_sd = stats

        # z-score on valid/test
        zv = np.zeros_like(detectors_scores_valid[name], dtype=float)
        zt = np.zeros_like(detectors_scores_test[name], dtype=float)
        for c in np.unique(labs_v):
            mu, sd = mu_sd.get(int(c), (0.0, 1.0))
            sel = (labs_v == c)
            zv[sel] = (detectors_scores_valid[name][sel] - mu) / sd
        for c in np.unique(labs_t):
            mu, sd = mu_sd.get(int(c), (0.0, 1.0))
            sel = (labs_t == c)
            zt[sel] = (detectors_scores_test[name][sel] - mu) / sd
        z_valid_map[name] = zv
        z_test_map[name]  = zt

    # ---------------- fusion candidates ----------------
    # 1) Rank-based on raw scores (includes original winner and new rules)
    rank_fuse = fuse_candidates({k: detectors_scores_valid[k] for k in keep}, weights)

    # 2) Probability-based (train-calibrated CDF)
    prob_fuse = fuse_candidates_prob(prob_valid, weights)

    # 3) Rank-based on cohort-normalized z-scores
    rank_fuse_cohort = fuse_candidates(z_valid_map, weights)
    # 4) Probability via cohort z-scores (map z to ranks as proxy prob)
    prob_fuse_cohort = fuse_candidates_prob({k: rank01(z_valid_map[k]) for k in keep}, weights)

    # collect all candidates and pick the best by VALID AUPRC
    all_cands = {}
    all_cands.update({f"rank::{k}": v for k,v in rank_fuse.items()})
    all_cands.update({f"prob::{k}": v for k,v in prob_fuse.items()})
    all_cands.update({f"rankCoh::{k}": v for k,v in rank_fuse_cohort.items()})
    all_cands.update({f"probCoh::{k}": v for k,v in prob_fuse_cohort.items()})

    best_rule, best_val_s, best_ap_local, best_auc_local = None, None, -1, -1
    for rname, sv in all_cands.items():
        ap = average_precision_score(yv, sv)
        auc = roc_auc_score(yv, sv)
        if ap > best_ap_local:
            best_rule, best_val_s, best_ap_local, best_auc_local = rname, sv, ap, auc

    print(f"\n[Fusion best with Fast Improvement] Rule={best_rule}  AUPRC={best_ap_local:.4f}  AUROC={best_auc_local:.4f}")

    # build corresponding TEST fused score
    def build_fused_test(rule_name: str):
        kind, sub = rule_name.split("::")
        if kind == "rank":
            R = np.column_stack([rank01(detectors_scores_test[k]) for k in keep])
            w = np.array([weights[k] for k in keep]); w = w / (w.sum()+1e-12)
            if sub == "max_rank":
                return np.max(R, axis=1)
            elif sub == "median_rank":
                return np.median(R, axis=1)
            elif sub == "wavg_rank":
                return np.average(R, axis=1, weights=w)
            elif sub == "max_rank_top2":
                top2 = sorted(keep, key=lambda n: weights[n], reverse=True)[:2]
                R2 = np.column_stack([rank01(detectors_scores_test[n]) for n in top2])
                return np.max(R2, axis=1)
            elif sub == "max_rank_top3": # NEW
                top3 = sorted(keep, key=lambda n: weights[n], reverse=True)[:3]
                R3 = np.column_stack([rank01(detectors_scores_test[n]) for n in top3])
                return np.max(R3, axis=1)
            elif sub == "wavg_rank_top2": # NEW
                top2 = sorted(keep, key=lambda n: weights[n], reverse=True)[:2]
                R2 = np.column_stack([rank01(detectors_scores_test[n]) for n in top2])
                top2_w = np.array([weights[n] for n in top2])
                top2_w = top2_w / top2_w.sum()
                return np.average(R2, axis=1, weights=top2_w)
        elif kind == "prob":
            P = np.column_stack([prob_test[k] for k in keep])
            w = np.array([weights[k] for k in keep]); w = w / (w.sum()+1e-12)
            if sub == "p_avg":
                return np.average(P, axis=1, weights=w)
            elif sub == "p_noisy_or":
                return 1.0 - np.prod(1.0 - P, axis=1)
            elif sub == "p_max":
                return np.max(P, axis=1)
        elif kind == "rankCoh":
            R = np.column_stack([rank01(z_test_map[k]) for k in keep])
            w = np.array([weights[k] for k in keep]); w = w / (w.sum()+1e-12)
            if sub == "max_rank":
                return np.max(R, axis=1)
            elif sub == "median_rank":
                return np.median(R, axis=1)
            elif sub == "wavg_rank":
                return np.average(R, axis=1, weights=w)
            elif sub == "max_rank_top2":
                top2 = sorted(keep, key=lambda n: weights[n], reverse=True)[:2]
                R2 = np.column_stack([rank01(z_test_map[n]) for n in top2])
                return np.max(R2, axis=1)
            elif sub == "max_rank_top3": # NEW
                top3 = sorted(keep, key=lambda n: weights[n], reverse=True)[:3]
                R3 = np.column_stack([rank01(z_test_map[n]) for n in top3])
                return np.max(R3, axis=1)
            elif sub == "wavg_rank_top2": # NEW
                top2 = sorted(keep, key=lambda n: weights[n], reverse=True)[:2]
                R2 = np.column_stack([rank01(z_test_map[n]) for n in top2])
                top2_w = np.array([weights[n] for n in top2])
                top2_w = top2_w / top2_w.sum()
                return np.average(R2, axis=1, weights=top2_w)
        elif kind == "probCoh":
            P = np.column_stack([rank01(z_test_map[k]) for k in keep])
            w = np.array([weights[k] for k in keep]); w = w / (w.sum()+1e-12)
            if sub == "p_avg":
                return np.average(P, axis=1, weights=w)
            elif sub == "p_noisy_or":
                return 1.0 - np.prod(1.0 - P, axis=1)
            elif sub == "p_max":
                return np.max(P, axis=1)
        # fallback
        R = np.column_stack([rank01(detectors_scores_test[k]) for k in keep])
        return np.max(R, axis=1)

    # Use the best rule found
    best_submission = build_fused_test(best_rule)

    # final report + save
    print("\n=== BEST VALID Fast Improvement ===")
    print(f"AUPRC={best_ap_local:.4f}  AUROC={best_auc_local:.4f}")
    print(f"Detectors used={keep}")
    print(f"Rule={best_rule}")

    # Build submission from stored best state
    test = pd.read_csv("Data/loans_test.csv")
    idc = "Id" if "Id" in test.columns else "index"
    sub = pd.DataFrame({idc: test[idc], "anomaly_score": best_submission})
    sub.to_csv("SUB_ultra_unsup_fast_impr.csv", index=False)
    print("Saved SUB_ultra_unsup_fast_impr.csv")

if __name__ == "__main__":
    main()