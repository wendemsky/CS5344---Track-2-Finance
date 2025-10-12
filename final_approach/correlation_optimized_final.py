#!/usr/bin/env python3
"""
FINAL MODEL: Correlation-Optimized Ensemble

Based on insights from:
1. Validation correlation analysis - top features identified
2. Exp5 results - LOF-based detectors perform best (AUPRC=0.2931, best individual=0.2988)
3. Current best final_approach performance

Optimizations:
- Focus on LOF variants (they work best)
- Add correlation-derived composite features
- Keep it fast and efficient
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_builder_advanced import FeatureBuilderAdvanced

RNG = np.random.default_rng(42)

# ========== CORRELATION FEATURES ==========

def add_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add key correlation-derived features"""
    features = pd.DataFrame(index=df.index)

    # Credit risk (r=-0.250, strongest predictor)
    if 'CreditScore' in df.columns:
        cs = df['CreditScore'].fillna(750)
        features['credit_risk'] = (850 - cs) / 150.0
    else:
        features['credit_risk'] = 0.0

    # DTI risk (r=+0.100)
    if 'OriginalDTI' in df.columns:
        features['dti_risk'] = np.clip(df['OriginalDTI'].fillna(35) / 100.0, 0, 1.5)
    else:
        features['dti_risk'] = 0.0

    # Interest rate risk (r=+0.096)
    if 'OriginalInterestRate' in df.columns:
        features['rate_risk'] = np.clip((df['OriginalInterestRate'].fillna(6.0) - 3.0) / 7.0, 0, 1.5)
    else:
        features['rate_risk'] = 0.0

    # Composite risk score
    features['composite_risk'] = (
        0.50 * features['credit_risk'] +
        0.25 * features['dti_risk'] +
        0.25 * features['rate_risk']
    )

    # Late-stage payment irregularity (r=+0.12-0.14 for months 12-13)
    nibupb_cols = [c for c in df.columns if 'NonInterestBearingUPB' in c]
    if len(nibupb_cols) >= 10:
        late_cols = [c for c in nibupb_cols if any(f'{i}_' in c for i in [10,11,12,13])]
        if late_cols:
            late_nibupb = df[late_cols].fillna(0).values
            features['payment_irregularity'] = np.mean(late_nibupb, axis=1) / 100000.0
        else:
            features['payment_irregularity'] = 0.0
    else:
        features['payment_irregularity'] = 0.0

    return features.fillna(0)


# ========== HELPERS ==========

def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def train_cdf(scores_train: np.ndarray):
    s = np.sort(scores_train.copy())
    def cdf(x):
        idx = np.searchsorted(s, x, side='right')
        return idx / (len(s) + 1e-12)
    return np.vectorize(cdf)


# ========== MAIN ==========

def main():
    print("="*80)
    print("FINAL MODEL: CORRELATION-OPTIMIZED ENSEMBLE")
    print("="*80)

    # Load data
    train = pd.read_csv("../Data/loans_train.csv")
    valid = pd.read_csv("../Data/loans_valid.csv")
    test = pd.read_csv("../Data/loans_test.csv")

    yv = valid["target"].to_numpy()

    print(f"\nData: Train={train.shape}, Valid={valid.shape}, Test={test.shape}")
    print(f"Validation anomaly rate: {yv.mean()*100:.2f}%")

    # Add correlation features
    print("\nAdding correlation-derived features...")
    train = pd.concat([train, add_correlation_features(train)], axis=1)
    valid = pd.concat([valid, add_correlation_features(valid)], axis=1)
    test = pd.concat([test, add_correlation_features(test)], axis=1)

    # Build features
    print("Building feature representation...")
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=100)
    fb.fit(train)

    Xtr_scaled, _, Xtr = fb.transform(train)
    Xv_scaled, _, Xv = fb.transform(valid)
    Xt_scaled, _, Xt = fb.transform(test)

    print(f"Features: Scaled={Xtr_scaled.shape}, PCA={Xtr.shape}")

    # Train detectors
    print("\nTraining detectors...")
    detectors_valid = {}
    detectors_test = {}
    detectors_train = {}

    # 1. LOF (best performers from Exp5)
    print("  [1/5] LOF detectors...")
    for k in [4, 5, 6, 7, 8, 10, 12]:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xtr)
        detectors_valid[f'lof_k{k}'] = -model.score_samples(Xv)
        detectors_test[f'lof_k{k}'] = -model.score_samples(Xt)
        detectors_train[f'lof_k{k}'] = -model.score_samples(Xtr)

    # 2. Cluster-wise LOF
    print("  [2/5] Cluster-wise LOF...")
    km = KMeans(n_clusters=15, n_init=10, random_state=42).fit(Xtr)
    cl_models = {}
    for c in range(km.n_clusters):
        idx = (km.labels_ == c)
        if idx.sum() < 50: continue
        Xc = Xtr[idx]
        best, best_mu = None, -np.inf
        for k in (5, 7, 9):
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
                s[i] = -cl_models[c].score_samples(x[None, :])[0]
        return s

    detectors_valid['cluster_lof'] = score_clusterwise(Xv)
    detectors_test['cluster_lof'] = score_clusterwise(Xt)
    detectors_train['cluster_lof'] = score_clusterwise(Xtr)

    # 3. k-distance
    print("  [3/5] k-distance...")
    for k in [3, 5, 7, 9]:
        nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
        d_v, _ = nn.kneighbors(Xv)
        d_t, _ = nn.kneighbors(Xt)
        d_tr, _ = nn.kneighbors(Xtr)
        detectors_valid[f'kdist_k{k}'] = d_v[:, -1]
        detectors_test[f'kdist_k{k}'] = d_t[:, -1]
        detectors_train[f'kdist_k{k}'] = d_tr[:, -1]

    # 4. Isolation Forest
    print("  [4/5] Isolation Forest...")
    if_model = IsolationForest(n_estimators=400, max_samples=1.0,
                                contamination="auto", random_state=42).fit(Xtr)
    detectors_valid['iforest'] = -if_model.score_samples(Xv)
    detectors_test['iforest'] = -if_model.score_samples(Xt)
    detectors_train['iforest'] = -if_model.score_samples(Xtr)

    # 5. PCA reconstruction
    print("  [5/5] PCA reconstruction...")
    pca_recon = PCA(n_components=min(100, Xtr_scaled.shape[1]), random_state=42).fit(Xtr_scaled)
    def recon_err(X):
        Xr = pca_recon.inverse_transform(pca_recon.transform(X))
        return np.mean((X - Xr)**2, axis=1)
    detectors_train['pca_recon'] = recon_err(Xtr_scaled)
    detectors_valid['pca_recon'] = recon_err(Xv_scaled)
    detectors_test['pca_recon'] = recon_err(Xt_scaled)

    # Evaluate
    print("\nEvaluating detectors...")
    per_ap = {k: average_precision_score(yv, v) for k, v in detectors_valid.items()}
    per_auc = {k: roc_auc_score(yv, v) for k, v in detectors_valid.items()}

    print("\nTop 10 detectors:")
    for k, v in sorted(per_ap.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {k:20s}: AUPRC={v:.4f}, AUROC={per_auc[k]:.4f}")

    # Select top detectors
    keep = [k for k, v in sorted(per_ap.items(), key=lambda x: x[1], reverse=True) if v >= 0.15][:12]
    print(f"\nSelected {len(keep)} detectors for ensemble")

    # Calibrate
    print("\nCalibrating and fusing...")
    prob_valid = {}
    prob_test = {}
    weights = {}

    for name in keep:
        tr = detectors_train[name]
        cdf = train_cdf(tr)
        prob_valid[name] = cdf(detectors_valid[name])
        prob_test[name] = cdf(detectors_test[name])
        weights[name] = max(per_ap[name], 1e-6)

    # Fusion strategies
    strategies = {}

    # Rank-based
    R_v = np.column_stack([rank01(detectors_valid[k]) for k in keep])
    R_t = np.column_stack([rank01(detectors_test[k]) for k in keep])
    w = np.array([weights[k] for k in keep])
    w = w / w.sum()

    strategies['wavg_rank'] = (np.average(R_v, axis=1, weights=w), np.average(R_t, axis=1, weights=w))
    strategies['max_rank'] = (np.max(R_v, axis=1), np.max(R_t, axis=1))

    # Top-3 max (winner from Exp5)
    top3 = sorted(keep, key=lambda n: weights[n], reverse=True)[:3]
    R3_v = np.column_stack([rank01(detectors_valid[n]) for n in top3])
    R3_t = np.column_stack([rank01(detectors_test[n]) for n in top3])
    strategies['max_rank_top3'] = (np.max(R3_v, axis=1), np.max(R3_t, axis=1))

    # Probability-based
    P_v = np.column_stack([prob_valid[k] for k in keep])
    P_t = np.column_stack([prob_test[k] for k in keep])
    strategies['wavg_prob'] = (np.average(P_v, axis=1, weights=w), np.average(P_t, axis=1, weights=w))

    # Evaluate strategies
    print("\nFusion strategy performance:")
    best_name = None
    best_ap = -1
    for name, (sv, st) in strategies.items():
        ap = average_precision_score(yv, sv)
        auc = roc_auc_score(yv, sv)
        print(f"  {name:20s}: AUPRC={ap:.4f}, AUROC={auc:.4f}")
        if ap > best_ap:
            best_name, best_ap = name, ap

    print(f"\n{'='*80}")
    print(f"BEST PERFORMANCE")
    print(f"Strategy: {best_name}")
    print(f"Validation AUPRC: {best_ap:.4f}")
    print(f"{'='*80}")

    # Save submission
    best_test_scores = strategies[best_name][1]

    idc = "Id" if "Id" in test.columns else "index"
    submission = pd.DataFrame({idc: test[idc], 'anomaly_score': best_test_scores})
    submission.to_csv("SUB_correlation_optimized_final.csv", index=False)

    print(f"\nSubmission saved to: SUB_correlation_optimized_final.csv")
    print("\nDONE!")

if __name__ == "__main__":
    main()
