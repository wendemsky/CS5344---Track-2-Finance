#!/usr/bin/env python3
# Fully compliant unsupervised fusion:
# - Fit encoders/imputers/scaler/PCA and detectors on TRAIN ONLY
# - Use VALID only for hyperparameter / rule selection (no fitting)
# - No meta-learner on validation
# - Save SUB_compliant_unsup.csv

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_builder_compliant import FeatureBuilderCompliant

RNG = np.random.default_rng(42)

def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

# ---- detectors (train on TRAIN) ----
def train_global_lof(X_train_det: np.ndarray, ks: List[int]) -> List[Tuple[str, LocalOutlierFactor]]:
    models = []
    for k in ks:
        m = LocalOutlierFactor(n_neighbors=k, novelty=True)  # contamination ignored in novelty mode
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
        # pick best k by internal score magnitude (unsupervised proxy)
        best_m, best_mean = None, -np.inf
        for k in (5,7,9):
            m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xc)
            sc = -m.score_samples(Xc)
            mu = sc.mean()
            if mu > best_mean:
                best_mean = mu
                best_m = m
        if best_m is not None:
            cl_models[c] = best_m
    return km, cl_models

def score_clusterwise(km: KMeans, cl_models: Dict[int, LocalOutlierFactor], X_det: np.ndarray) -> np.ndarray:
    labels = km.predict(X_det)
    s = np.zeros(len(X_det))
    for i, (x, c) in enumerate(zip(X_det, labels)):
        if c in cl_models:
            s[i] = -cl_models[c].score_samples(x[None,:])[0]
        else:
            s[i] = 0.0
    return s

def kdist_scores(X_train_det: np.ndarray, X_det: np.ndarray, k: int = 7) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(X_train_det)
    dists, _ = nn.kneighbors(X_det)
    return dists[:, -1]  # distance to k-th neighbor

def local_density_ratio(X_train_det: np.ndarray, X_det: np.ndarray, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(X_train_det)
    dists_train, _ = nn.kneighbors(X_train_det)
    dists_det, idx = nn.kneighbors(X_det)

    # avg neighbor density - need to average over neighbors (axis=2)
    neighbor_densities = 1.0 / (dists_train[idx].mean(axis=2).mean(axis=1) + 1e-9)
    query_density = 1.0 / (dists_det.mean(axis=1) + 1e-9)
    return query_density / (neighbor_densities + 1e-9)

def train_iforest(X_train_det: np.ndarray, n_estimators: int = 100, max_samples: int = 256) -> IsolationForest:
    ifo = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                          contamination=0.1, random_state=42)
    ifo.fit(X_train_det)
    return ifo

# ---- main ----
def main():
    print("=== Compliant Unsupervised Anomaly Pipeline ===")

    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test  = pd.read_csv("Data/loans_test.csv")
    yv = valid["target"].to_numpy()

    # 1) Train-only feature builder
    fb = FeatureBuilderCompliant(
        use_pca=True,
        pca_comps=100,                # enhanced: more components
        keep_month_idx=(0,1,2,3,6,9,12)   # enhanced: more temporal months
    )
    fb.fit(train)

    Xtr_scaled, idxtr, Xtr_det = fb.transform(train)
    Xv_scaled,  idxv,  Xv_det  = fb.transform(valid)
    Xt_scaled,  idxt,  Xt_det  = fb.transform(test)

    # 2) Train detectors on TRAIN ONLY
    ks = [3,4,5,6,7,8,10,12,15,20]  # enhanced: wider range
    lof_models = train_global_lof(Xtr_det, ks)
    km, cl_models = train_clusterwise_lof(Xtr_det, n_clusters=20)  # enhanced: more clusters
    ifo = train_iforest(Xtr_det, n_estimators=100, max_samples=256)

    # 3) Score validation (allowed for selection)
    det_scores_valid: Dict[str, np.ndarray] = {}

    # LOFs
    for name, m in lof_models:
        det_scores_valid[name] = -m.score_samples(Xv_det)

    # Cluster-wise LOF
    det_scores_valid["cluster_lof"] = score_clusterwise(km, cl_models, Xv_det)

    # Isolation Forest
    det_scores_valid["iforest"] = -ifo.score_samples(Xv_det)

    # k-distance detectors (multiple k values)
    det_scores_valid["kdist_k5"] = kdist_scores(Xtr_det, Xv_det, k=5)
    det_scores_valid["kdist_k7"] = kdist_scores(Xtr_det, Xv_det, k=7)
    det_scores_valid["kdist_k10"] = kdist_scores(Xtr_det, Xv_det, k=10)
    det_scores_valid["kdist_k15"] = kdist_scores(Xtr_det, Xv_det, k=15)

    # local density ratio detectors
    det_scores_valid["ldr_k10"] = local_density_ratio(Xtr_det, Xv_det, k=10)
    det_scores_valid["ldr_k15"] = local_density_ratio(Xtr_det, Xv_det, k=15)

    # amortization residual (engineered scalar, from scaled-full block)
    amort_block = Xv_scaled[:, np.r_[idxv["amort"].start:idxv["amort"].stop]]
    # expected order: ["amort_short_mean","amort_short_pct_hi","amort_mask_not_applicable"]
    if amort_block.shape[1] >= 2:
        amort_score = amort_block[:, 0]*0.7 + amort_block[:, 1]*0.3
    else:
        amort_score = np.zeros(len(valid))
    det_scores_valid["amort_resid"] = amort_score

    # 4) Evaluate per-detector on validation
    per_ap = {k: average_precision_score(yv, s) for k, s in det_scores_valid.items()}
    per_auc = {k: roc_auc_score(yv, s) for k, s in det_scores_valid.items()}
    print("[Validation] Per-detector AUPRC:", {k: round(v,4) for k,v in per_ap.items()})
    print("[Validation] Per-detector AUROC:", {k: round(v,4) for k,v in per_auc.items()})

    # 5) Select detectors to fuse (by validation AUPRC threshold & top-N)
    sorted_dets = sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)
    keep = [k for k,_ in sorted_dets if per_ap[k] >= 0.15][:10]  # enhanced: keep up to 10 strong ones
    if not keep:
        keep = [sorted_dets[0][0]]
    print("[Selection] Using detectors:", keep)

    # 6) Rule-based fusions (no fitting) -> pick the best by validation AUPRC
    R_keep = np.column_stack([rank01(det_scores_valid[k]) for k in keep])
    weights = np.array([per_ap[k] for k in keep], dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    candidates = {}
    candidates["max_rank"]      = np.max(R_keep, axis=1)
    candidates["wavg_rank"]     = np.average(R_keep, axis=1, weights=weights)
    candidates["median_rank"]   = np.median(R_keep, axis=1)

    # top-2 max
    top2 = sorted(keep, key=lambda k: per_ap[k], reverse=True)[:2]
    R_top2 = np.column_stack([rank01(det_scores_valid[k]) for k in top2])
    candidates["max_rank_top2"] = np.max(R_top2, axis=1)

    # enhanced fusion rules
    candidates["geom_mean"] = np.exp(np.log(R_keep + 1e-9).mean(axis=1))
    candidates["p75_rank"] = np.percentile(R_keep, 75, axis=1)

    # weighted max top-3
    if len(keep) >= 3:
        top3 = sorted(keep, key=lambda k: per_ap[k], reverse=True)[:3]
        R_top3 = np.column_stack([rank01(det_scores_valid[k]) for k in top3])
        top3_weights = np.array([per_ap[k] for k in top3], dtype=float)
        top3_weights /= (top3_weights.sum() + 1e-12)
        candidates["weighted_max_top3"] = np.max(R_top3 * top3_weights, axis=1)

    best_rule, best_score, best_ap, best_auc = None, None, -1, -1
    for name, s in candidates.items():
        ap = average_precision_score(yv, s)
        auc = roc_auc_score(yv, s)
        if ap > best_ap:
            best_rule, best_score, best_ap, best_auc = name, s, ap, auc

    print(f"[Fusion] Rule={best_rule}  AUPRC={best_ap:.4f}  AUROC={best_auc:.4f}")

    # 7) Score TEST with the same detectors and chosen fusion rule
    det_scores_test: Dict[str, np.ndarray] = {}

    for name, m in lof_models:
        det_scores_test[name] = -m.score_samples(Xt_det)
    det_scores_test["cluster_lof"] = score_clusterwise(km, cl_models, Xt_det)
    det_scores_test["iforest"] = -ifo.score_samples(Xt_det)

    det_scores_test["kdist_k5"]    = kdist_scores(Xtr_det, Xt_det, k=5)
    det_scores_test["kdist_k7"]    = kdist_scores(Xtr_det, Xt_det, k=7)
    det_scores_test["kdist_k10"]   = kdist_scores(Xtr_det, Xt_det, k=10)
    det_scores_test["kdist_k15"]   = kdist_scores(Xtr_det, Xt_det, k=15)

    det_scores_test["ldr_k10"] = local_density_ratio(Xtr_det, Xt_det, k=10)
    det_scores_test["ldr_k15"] = local_density_ratio(Xtr_det, Xt_det, k=15)

    amort_block_t = Xt_scaled[:, np.r_[idxt["amort"].start:idxt["amort"].stop]]
    if amort_block_t.shape[1] >= 2:
        det_scores_test["amort_resid"] = amort_block_t[:, 0]*0.7 + amort_block_t[:, 1]*0.3
    else:
        det_scores_test["amort_resid"] = np.zeros(len(test))

    R_test = np.column_stack([rank01(det_scores_test[k]) for k in keep])

    if best_rule == "max_rank":
        final_test_score = np.max(R_test, axis=1)
    elif best_rule == "wavg_rank":
        final_test_score = np.average(R_test, axis=1, weights=weights)
    elif best_rule == "median_rank":
        final_test_score = np.median(R_test, axis=1)
    elif best_rule == "max_rank_top2":
        R_t2 = np.column_stack([rank01(det_scores_test[k]) for k in top2])
        final_test_score = np.max(R_t2, axis=1)
    elif best_rule == "geom_mean":
        final_test_score = np.exp(np.log(R_test + 1e-9).mean(axis=1))
    elif best_rule == "p75_rank":
        final_test_score = np.percentile(R_test, 75, axis=1)
    elif best_rule == "weighted_max_top3":
        if len(keep) >= 3:
            R_t3 = np.column_stack([rank01(det_scores_test[k]) for k in top3])
            final_test_score = np.max(R_t3 * top3_weights, axis=1)
        else:
            final_test_score = np.max(R_test, axis=1)
    else:
        final_test_score = np.max(R_test, axis=1)  # fallback

    idc = "Id" if "Id" in test.columns else "index"
    sub = pd.DataFrame({idc: test[idc], "anomaly_score": final_test_score})
    sub.to_csv("SUB_compliant_unsup.csv", index=False)
    print("Saved SUB_compliant_unsup.csv")

if __name__ == "__main__":
    main()
