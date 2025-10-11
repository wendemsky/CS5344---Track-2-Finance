#!/usr/bin/env python3
# Ultra ensemble (unsupervised, compliant):
# - Fit everything on TRAIN (normals only)
# - VALID used only to choose hyperparams & fusion rule (no fitting on valid)
# - Rich detector set + train-only calibration + cohort normalization + fusion rules
# - Saves SUB_ultra_unsup.csv

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

from sklearn.metrics import average_precision_score, roc_auc_score

from feature_builder_advanced import FeatureBuilderAdvanced

RNG = np.random.default_rng(42)

# ---------- helpers ----------
def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def train_cdf(scores_train: np.ndarray):
    # empirical CDF over train scores → maps score to tail probability in [0,1]
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

def iforest_train(Xtr: np.ndarray, n_estimators=400, max_samples=1.0, rng=42) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators, max_samples=max_samples,
        contamination="auto", random_state=rng, n_jobs=-1
    ).fit(Xtr)

def ee_train(Xtr: np.ndarray, support_frac=0.9) -> EllipticEnvelope:
    # robust covariance on PCA space
    return EllipticEnvelope(contamination=1.0-support_frac, support_fraction=support_frac, random_state=42).fit(Xtr)

def pca_recon_error_train(Xtr_scaled: np.ndarray, n_comp=50) -> Tuple[PCA, np.ndarray]:
    p = PCA(n_components=min(n_comp, Xtr_scaled.shape[1]), random_state=42).fit(Xtr_scaled)
    Xr = p.inverse_transform(p.transform(Xtr_scaled))
    train_err = np.mean((Xtr_scaled - Xr)**2, axis=1)
    return p, train_err

def random_proj_lof_scores(Xtr: np.ndarray, X: np.ndarray, B=50, dim=60, k=7, seed=42):
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

    # top-2 max
    top2 = sorted(names, key=lambda n: weights[n], reverse=True)[:2]
    R2   = np.column_stack([rank01(score_map_valid[n]) for n in top2])
    cand["max_rank_top2"] = np.max(R2, axis=1)

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
    print("=== Ultra Unsupervised Ensemble (Compliant) ===")
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test  = pd.read_csv("Data/loans_test.csv")

    yv = valid["target"].to_numpy()

    # Try multiple PCA settings (train-fit only); keep the best by validation AUPRC
    pca_settings = [40, 60, 80, 100]

    best_val_ap, best_val_auc = -1, -1
    best_submission = None
    best_log = None

    for pca_k in pca_settings:
        print(f"\n--- PCA setting: {pca_k} comps ---")

        fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=pca_k)
        fb.fit(train)

        Xtr_scaled, sl_tr, Xtr = fb.transform(train)
        Xv_scaled,  sl_v,  Xv  = fb.transform(valid)
        Xt_scaled,  sl_t,  Xt  = fb.transform(test)

        # ---------------- train detectors on TRAIN only ----------------
        detectors_scores_valid: Dict[str, np.ndarray] = {}
        detectors_scores_test : Dict[str, np.ndarray] = {}
        detectors_scores_train: Dict[str, np.ndarray] = {}

        # LOF multi-k
        lof_ks = [4,5,6,7,8,10,12]
        lof_models = lof_models_train(Xtr, lof_ks)
        for name, m in lof_models.items():
            detectors_scores_valid[name] = -m.score_samples(Xv)
            detectors_scores_test[name]  = -m.score_samples(Xt)
            detectors_scores_train[name] = -m.score_samples(Xtr)

        # KMeans + cluster-wise LOF
        km = KMeans(n_clusters=12, n_init=10, random_state=42).fit(Xtr)
        # choose per-cluster LOF k by internal magnitude
        cl_models = {}
        for c in range(km.n_clusters):
            idx = (km.labels_ == c)
            if idx.sum() < 60: continue
            Xc = Xtr[idx]
            best, best_mu = None, -np.inf
            for k in (5,7,9):
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

        # k-distance multi-k
        kd_ks = [3,5,7,9,11]
        kd_models = kdist_model_train(Xtr, kd_ks)
        for name, nn in kd_models.items():
            detectors_scores_valid[name] = kdist_score(nn, Xv)
            detectors_scores_test[name]  = kdist_score(nn, Xt)
            detectors_scores_train[name] = kdist_score(nn, Xtr)

        # Isolation Forest (PCA space)
        if_model = iforest_train(Xtr, n_estimators=500, max_samples=1.0, rng=42)
        detectors_scores_valid["iforest"] = -if_model.score_samples(Xv)
        detectors_scores_test["iforest"]  = -if_model.score_samples(Xt)
        detectors_scores_train["iforest"] = -if_model.score_samples(Xtr)

        # Elliptic Envelope (PCA space)
        try:
            ee_model = ee_train(Xtr, support_frac=0.9)
            detectors_scores_valid["ell_env"] = -ee_model.score_samples(Xv)
            detectors_scores_test["ell_env"]  = -ee_model.score_samples(Xt)
            detectors_scores_train["ell_env"] = -ee_model.score_samples(Xtr)
        except Exception:
            pass

        # PCA reconstruction error (train-only PCA on scaled full)
        pre_pca, pre_train_err = pca_recon_error_train(Xtr_scaled, n_comp=min(80, Xtr_scaled.shape[1]))
        def recon_err(X_scaled):
            Xr = pre_pca.inverse_transform(pre_pca.transform(X_scaled))
            return np.mean((X_scaled - Xr)**2, axis=1)
        detectors_scores_train["pca_recon"] = recon_err(Xtr_scaled)
        detectors_scores_valid["pca_recon"] = recon_err(Xv_scaled)
        detectors_scores_test["pca_recon"]  = recon_err(Xt_scaled)

        # Random-Projection LOF bagging (MaxRank within bags)
        detectors_scores_valid["rp_lof"] = random_proj_lof_scores(Xtr, Xv, B=40, dim=min(60, Xtr.shape[1]), k=7, seed=42)
        detectors_scores_test["rp_lof"]  = random_proj_lof_scores(Xtr, Xt, B=40, dim=min(60, Xtr.shape[1]), k=7, seed=43)
        detectors_scores_train["rp_lof"] = random_proj_lof_scores(Xtr, Xtr, B=40, dim=min(60, Xtr.shape[1]), k=7, seed=41)

        # amortization residual (from scaled full features) — combine short_mean/short_70/short_50
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

        # ---------------- per-detector VALID metrics ----------------
        per_ap = {k: average_precision_score(yv, v) for k, v in detectors_scores_valid.items()}
        per_auc = {k: roc_auc_score(yv, v) for k, v in detectors_scores_valid.items()}
        print("[Per-detector AUPRC]:", {k: round(v,4) for k,v in sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)[:8]})
        print("[Per-detector AUROC]:", {k: round(v,4) for k,v in sorted(per_auc.items(), key=lambda kv: kv[1], reverse=True)[:8]})

        # keep top detectors (threshold + cap)
        sorted_dets = sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)
        keep = [k for k,_ in sorted_dets if per_ap[k] >= 0.16][:10]
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
        # We compute per-cluster z-score for each detector using TRAIN cluster stats,
        # then apply those z-scores to VALID/TEST within their cluster.
        km_cohort = KMeans(n_clusters=12, n_init=10, random_state=123).fit(Xtr)
        labs_tr = km_cohort.labels_
        labs_v  = km_cohort.predict(Xv)
        labs_t  = km_cohort.predict(Xt)

        z_valid_map = {}
        z_test_map  = {}

        for name in keep:
            # compute train cluster stats for this detector
            mu_sd = {}
            sc_tr = detectors_scores_train[name]
            stats = zscore_by_cluster(sc_tr, labs_tr)
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
        # 1) Rank-based on raw scores
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

        print(f"[Fusion best @ PCA={pca_k}] Rule={best_rule}  AUPRC={best_ap_local:.4f}  AUROC={best_auc_local:.4f}")

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

        # record best across PCA settings
        if best_ap_local > best_val_ap:
            best_val_ap, best_val_auc = best_ap_local, best_auc_local
            best_rule_name = best_rule
            best_keep = keep
            best_weights = {k: weights[k] for k in keep}
            best_fb = fb
            best_state = {
                "det_valid": detectors_scores_valid,
                "det_test":  detectors_scores_test,
                "det_train": detectors_scores_train,
                "z_valid":   z_valid_map,
                "z_test":    z_test_map,
                "prob_valid": prob_valid,
                "prob_test":  prob_test,
                "rule": best_rule_name
            }
            best_submission = build_fused_test(best_rule_name)
            best_log = (pca_k, keep, best_rule_name)

    # final report + save
    print("\n=== BEST VALID ===")
    print(f"AUPRC={best_val_ap:.4f}  AUROC={best_val_auc:.4f}")
    print(f"Chosen PCA={best_log[0]}, detectors={best_log[1]}")
    print(f"Rule={best_log[2]}")

    # Build submission from stored best state
    test = pd.read_csv("Data/loans_test.csv")
    idc = "Id" if "Id" in test.columns else "index"
    sub = pd.DataFrame({idc: test[idc], "anomaly_score": best_submission})
    sub.to_csv("SUB_ultra_unsup.csv", index=False)
    print("Saved SUB_ultra_unsup.csv")

if __name__ == "__main__":
    main()
