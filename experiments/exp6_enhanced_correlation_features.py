#!/usr/bin/env python3
"""
Experiment 6: Enhanced Correlation Features with Optimized Ensemble

Building on Exp5 results (AUPRC=0.2931), we now:
1. Add more sophisticated correlation-derived features
2. Create interaction terms between top correlated features
3. Use optimized hyperparameters based on Exp5 insights
4. Focus on LOF-based detectors (they performed best)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.append('final_approach')
from feature_builder_advanced import FeatureBuilderAdvanced

RNG = np.random.default_rng(42)

# ========== ENHANCED FEATURE ENGINEERING ==========

def create_enhanced_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced features based on correlation analysis + domain knowledge
    """
    features = pd.DataFrame(index=df.index)

    # === PRIMARY RISK INDICATORS (from correlation analysis) ===

    # 1. CreditScore features (r=-0.250, STRONGEST signal)
    if 'CreditScore' in df.columns:
        cs = df['CreditScore'].fillna(750)
        features['credit_risk_score'] = (850 - cs) / 150.0  # normalize
        features['credit_risk_sq'] = features['credit_risk_score'] ** 2
        features['credit_below_700'] = (cs < 700).astype(float)
        features['credit_below_650'] = (cs < 650).astype(float)
        features['credit_above_800'] = (cs > 800).astype(float)  # safe borrowers
    else:
        for col in ['credit_risk_score', 'credit_risk_sq', 'credit_below_700',
                    'credit_below_650', 'credit_above_800']:
            features[col] = 0.0

    # 2. DTI features (r=+0.100)
    if 'OriginalDTI' in df.columns:
        dti = df['OriginalDTI'].fillna(35)
        features['dti_risk'] = np.clip(dti / 100.0, 0, 1.5)
        features['dti_high'] = (dti > 43).astype(float)  # FHA limit
        features['dti_very_high'] = (dti > 50).astype(float)
    else:
        for col in ['dti_risk', 'dti_high', 'dti_very_high']:
            features[col] = 0.0

    # 3. Interest Rate features (r=+0.096)
    if 'OriginalInterestRate' in df.columns:
        rate = df['OriginalInterestRate'].fillna(6.0)
        features['rate_risk'] = np.clip((rate - 3.0) / 7.0, 0, 1.5)
        features['rate_high'] = (rate > 7.0).astype(float)
        features['rate_very_high'] = (rate > 8.0).astype(float)
    else:
        for col in ['rate_risk', 'rate_high', 'rate_very_high']:
            features[col] = 0.0

    # 4. NonInterestBearingUPB features (r=+0.12 to +0.14 for months 12-13)
    nibupb_cols = [c for c in df.columns if 'NonInterestBearingUPB' in c]
    if len(nibupb_cols) >= 10:
        # Focus on late-stage (months 10-13)
        late_cols = [c for c in nibupb_cols if any(f'{i}_' in c for i in [10, 11, 12, 13])]
        if late_cols:
            late_nibupb = df[late_cols].fillna(0).values
            features['late_nibupb_mean'] = np.mean(late_nibupb, axis=1) / 100000.0
            features['late_nibupb_max'] = np.max(late_nibupb, axis=1) / 100000.0
            features['late_nibupb_trend'] = (late_nibupb[:, -1] - late_nibupb[:, 0]) / 100000.0
            features['late_nibupb_spike'] = (np.max(late_nibupb, axis=1) > 50000).astype(float)
        else:
            for col in ['late_nibupb_mean', 'late_nibupb_max', 'late_nibupb_trend', 'late_nibupb_spike']:
                features[col] = 0.0

        # Early vs late comparison
        early_cols = [c for c in nibupb_cols if any(f'{i}_' in c for i in [0, 1, 2, 3])]
        if early_cols and late_cols:
            early_nibupb = df[early_cols].fillna(0).values
            features['nibupb_growth'] = (np.mean(late_nibupb, axis=1) - np.mean(early_nibupb, axis=1)) / 100000.0
    else:
        for col in ['late_nibupb_mean', 'late_nibupb_max', 'late_nibupb_trend',
                    'late_nibupb_spike', 'nibupb_growth']:
            features[col] = 0.0

    # === INTERACTION FEATURES (combining top predictors) ===

    # 5. Credit x DTI interaction (low credit + high DTI = very risky)
    features['credit_dti_risk'] = features['credit_risk_score'] * features['dti_risk']

    # 6. Credit x Rate interaction (low credit + high rate = subprime)
    features['credit_rate_risk'] = features['credit_risk_score'] * features['rate_risk']

    # 7. DTI x Rate interaction (stressed finances + expensive loan)
    features['dti_rate_risk'] = features['dti_risk'] * features['rate_risk']

    # 8. Triple interaction (credit, DTI, rate)
    features['triple_risk'] = (features['credit_risk_score'] *
                               features['dti_risk'] *
                               features['rate_risk'])

    # === COMPOSITE SCORES ===

    # 9. Weighted composite (based on correlation magnitudes)
    features['composite_risk'] = (
        0.50 * features['credit_risk_score'] +  # r=0.25
        0.20 * features['dti_risk'] +           # r=0.10
        0.20 * features['rate_risk'] +          # r=0.096
        0.10 * features['late_nibupb_mean']     # r=0.12
    )

    # 10. Categorical risk tiers
    risk_bins = [0, 0.3, 0.6, 1.0, 10.0]
    features['risk_tier'] = pd.cut(features['composite_risk'], bins=risk_bins, labels=False).fillna(0)

    # === ADDITIONAL PREDICTORS (weaker but still significant) ===

    # 11. Number of borrowers (r=-0.073)
    if 'NumberOfBorrowers' in df.columns:
        features['single_borrower'] = (df['NumberOfBorrowers'] == 1).astype(float)
        features['multi_borrower'] = (df['NumberOfBorrowers'] > 1).astype(float)
    else:
        features['single_borrower'] = 0.0
        features['multi_borrower'] = 0.0

    # 12. Property valuation method (r=+0.051)
    if 'PropertyValMethod' in df.columns:
        pvm = df['PropertyValMethod'].fillna(0)
        features['prop_val_risk'] = (pvm / 10.0).clip(0, 1)
    else:
        features['prop_val_risk'] = 0.0

    # 13. Original UPB (r=-0.047, smaller loans slightly riskier)
    if 'OriginalUPB' in df.columns:
        upb = df['OriginalUPB'].fillna(200000)
        features['upb_normalized'] = upb / 500000.0
        features['small_loan'] = (upb < 150000).astype(float)
    else:
        features['upb_normalized'] = 0.0
        features['small_loan'] = 0.0

    return features.fillna(0)


# ========== HELPER FUNCTIONS ==========

def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def train_cdf(scores_train: np.ndarray):
    s = np.sort(scores_train.copy())
    def cdf(x):
        idx = np.searchsorted(s, x, side='right')
        return idx / (len(s) + 1e-12)
    return np.vectorize(cdf)


# ========== MAIN EXPERIMENT ==========

def main():
    print("="*80)
    print("EXPERIMENT 6: ENHANCED CORRELATION FEATURES + OPTIMIZED ENSEMBLE")
    print("="*80)

    # Load data
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test = pd.read_csv("Data/loans_test.csv")

    yv = valid["target"].to_numpy()

    print(f"\nTrain: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
    print(f"Validation anomaly rate: {yv.mean()*100:.2f}%")

    # ========== Create enhanced features ==========
    print("\n" + "="*80)
    print("Creating enhanced correlation-based features")
    print("="*80)

    corr_features_train = create_enhanced_correlation_features(train)
    corr_features_valid = create_enhanced_correlation_features(valid)
    corr_features_test = create_enhanced_correlation_features(test)

    print(f"\nCreated {corr_features_train.shape[1]} enhanced features:")
    print(f"  - Primary risk indicators: 11")
    print(f"  - Interaction features: 4")
    print(f"  - Composite scores: 2")
    print(f"  - Additional predictors: {corr_features_train.shape[1] - 17}")

    # Add to datasets
    train_aug = pd.concat([train, corr_features_train], axis=1)
    valid_aug = pd.concat([valid, corr_features_valid], axis=1)
    test_aug = pd.concat([test, corr_features_test], axis=1)

    # ========== Build features ==========
    print("\n" + "="*80)
    print("Building feature representation")
    print("="*80)

    # Use higher PCA components based on expanded feature set
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=120)
    fb.fit(train_aug)

    Xtr_scaled, _, Xtr = fb.transform(train_aug)
    Xv_scaled, _, Xv = fb.transform(valid_aug)
    Xt_scaled, _, Xt = fb.transform(test_aug)

    print(f"Scaled features: {Xtr_scaled.shape}")
    print(f"PCA embedding: {Xtr.shape}")

    # ========== Train optimized ensemble (focused on LOF) ==========
    print("\n" + "="*80)
    print("Training optimized ensemble (LOF-focused)")
    print("="*80)

    detectors_valid = {}
    detectors_test = {}
    detectors_train = {}

    # 1. LOF with expanded k range (Exp5 showed LOF works best)
    print("\n[1/6] LOF detectors with optimized k values...")
    for k in [4, 5, 6, 7, 8, 10, 12, 15]:
        print(f"  Training LOF k={k}...")
        model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(Xtr)
        detectors_valid[f'lof_k{k}'] = -model.score_samples(Xv)
        detectors_test[f'lof_k{k}'] = -model.score_samples(Xt)
        detectors_train[f'lof_k{k}'] = -model.score_samples(Xtr)

    # 2. KMeans + cluster-wise LOF (optimized)
    print("\n[2/6] Cluster-wise LOF detectors...")
    for n_clusters in [12, 15, 18]:
        print(f"  Training with {n_clusters} clusters...")
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(Xtr)
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

        detectors_valid[f'cluster_lof_k{n_clusters}'] = score_clusterwise(Xv)
        detectors_test[f'cluster_lof_k{n_clusters}'] = score_clusterwise(Xt)
        detectors_train[f'cluster_lof_k{n_clusters}'] = score_clusterwise(Xtr)

    # 3. k-distance detectors
    print("\n[3/6] k-distance detectors...")
    for k in [3, 5, 7, 9, 11]:
        nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
        d_v, _ = nn.kneighbors(Xv)
        d_t, _ = nn.kneighbors(Xt)
        d_tr, _ = nn.kneighbors(Xtr)

        detectors_valid[f'kdist_k{k}'] = d_v[:, -1]
        detectors_test[f'kdist_k{k}'] = d_t[:, -1]
        detectors_train[f'kdist_k{k}'] = d_tr[:, -1]

    # 4. Isolation Forest ensemble
    print("\n[4/6] Isolation Forest ensemble...")
    for n_est in [300, 500]:
        if_model = IsolationForest(
            n_estimators=n_est, max_samples=1.0,
            contamination="auto", random_state=42, n_jobs=-1
        ).fit(Xtr)
        detectors_valid[f'iforest_{n_est}'] = -if_model.score_samples(Xv)
        detectors_test[f'iforest_{n_est}'] = -if_model.score_samples(Xt)
        detectors_train[f'iforest_{n_est}'] = -if_model.score_samples(Xtr)

    # 5. PCA reconstruction
    print("\n[5/6] PCA reconstruction error...")
    pca_recon = PCA(n_components=min(100, Xtr_scaled.shape[1]), random_state=42).fit(Xtr_scaled)

    def recon_err(X_scaled):
        Xr = pca_recon.inverse_transform(pca_recon.transform(X_scaled))
        return np.mean((X_scaled - Xr)**2, axis=1)

    detectors_train['pca_recon'] = recon_err(Xtr_scaled)
    detectors_valid['pca_recon'] = recon_err(Xv_scaled)
    detectors_test['pca_recon'] = recon_err(Xt_scaled)

    # 6. Mahalanobis distance (fast)
    print("\n[6/6] Mahalanobis distance...")
    try:
        mu = np.mean(Xtr_scaled, axis=0)
        cov = np.cov(Xtr_scaled, rowvar=False)
        reg = 1e-6
        while not np.all(np.linalg.eigvals(cov + reg * np.eye(cov.shape[0])) > 0):
            reg *= 10
        inv_cov = np.linalg.inv(cov + reg * np.eye(cov.shape[0]))

        def mahal_dist(X):
            diff = X - mu
            return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        detectors_train['mahal'] = mahal_dist(Xtr_scaled)
        detectors_valid['mahal'] = mahal_dist(Xv_scaled)
        detectors_test['mahal'] = mahal_dist(Xt_scaled)
    except Exception as e:
        print(f"  Mahalanobis failed: {e}")

    # ========== Evaluate and select ==========
    print("\n" + "="*80)
    print("Evaluating detectors")
    print("="*80)

    per_ap = {k: average_precision_score(yv, v) for k, v in detectors_valid.items()}
    per_auc = {k: roc_auc_score(yv, v) for k, v in detectors_valid.items()}

    print("\nTop 15 detectors by AUPRC:")
    for k, v in sorted(per_ap.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {k:30s}: AUPRC={v:.4f}, AUROC={per_auc[k]:.4f}")

    # Select top detectors
    sorted_dets = sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)
    keep = [k for k, _ in sorted_dets if per_ap[k] >= 0.15][:15]  # Top 15
    if not keep:
        keep = [sorted_dets[0][0]]

    print(f"\nSelected {len(keep)} detectors for ensemble")

    # ========== Fusion ==========
    print("\n" + "="*80)
    print("Ensemble fusion")
    print("="*80)

    # Calibrate with train CDF
    prob_valid = {}
    prob_test = {}
    weights = {}

    for name in keep:
        tr = detectors_train.get(name)
        if tr is None:
            prob_valid[name] = rank01(detectors_valid[name])
            prob_test[name] = rank01(detectors_test[name])
        else:
            cdf = train_cdf(tr)
            prob_valid[name] = cdf(detectors_valid[name])
            prob_test[name] = cdf(detectors_test[name])

        weights[name] = max(per_ap[name], 1e-6)

    # Fusion strategies
    fusion_results = {}

    # 1. Rank-based weighted average
    R_valid = np.column_stack([rank01(detectors_valid[k]) for k in keep])
    R_test = np.column_stack([rank01(detectors_test[k]) for k in keep])
    w = np.array([weights[k] for k in keep])
    w = w / w.sum()

    fusion_results['wavg_rank'] = {
        'valid': np.average(R_valid, axis=1, weights=w),
        'test': np.average(R_test, axis=1, weights=w)
    }

    # 2. Max rank (top-3)
    top3 = sorted(keep, key=lambda n: weights[n], reverse=True)[:3]
    R3_v = np.column_stack([rank01(detectors_valid[n]) for n in top3])
    R3_t = np.column_stack([rank01(detectors_test[n]) for n in top3])
    fusion_results['max_rank_top3'] = {
        'valid': np.max(R3_v, axis=1),
        'test': np.max(R3_t, axis=1)
    }

    # 3. Max rank (top-5)
    top5 = sorted(keep, key=lambda n: weights[n], reverse=True)[:5]
    R5_v = np.column_stack([rank01(detectors_valid[n]) for n in top5])
    R5_t = np.column_stack([rank01(detectors_test[n]) for n in top5])
    fusion_results['max_rank_top5'] = {
        'valid': np.max(R5_v, axis=1),
        'test': np.max(R5_t, axis=1)
    }

    # 4. Weighted average probability
    P_v = np.column_stack([prob_valid[k] for k in keep])
    P_t = np.column_stack([prob_test[k] for k in keep])
    fusion_results['wavg_prob'] = {
        'valid': np.average(P_v, axis=1, weights=w),
        'test': np.average(P_t, axis=1, weights=w)
    }

    # 5. Noisy-OR
    fusion_results['noisy_or'] = {
        'valid': 1.0 - np.prod(1.0 - P_v, axis=1),
        'test': 1.0 - np.prod(1.0 - P_t, axis=1)
    }

    # Evaluate
    print("\nFusion strategy performance:")
    best_strategy = None
    best_ap = -1
    best_auc = -1

    for name, scores in fusion_results.items():
        ap = average_precision_score(yv, scores['valid'])
        auc = roc_auc_score(yv, scores['valid'])
        print(f"  {name:20s}: AUPRC={ap:.4f}, AUROC={auc:.4f}")

        if ap > best_ap:
            best_strategy = name
            best_ap = ap
            best_auc = auc

    print(f"\n{'='*80}")
    print(f"BEST ENSEMBLE PERFORMANCE")
    print(f"Strategy: {best_strategy}")
    print(f"Validation AUPRC: {best_ap:.4f}")
    print(f"Validation AUROC: {best_auc:.4f}")
    print(f"{'='*80}")

    # ========== Save submission ==========
    best_test_scores = fusion_results[best_strategy]['test']

    idc = "Id" if "Id" in test.columns else "index"
    submission = pd.DataFrame({
        idc: test[idc],
        'anomaly_score': best_test_scores
    })

    submission.to_csv("experiments/SUB_exp6_enhanced_features.csv", index=False)
    print(f"\nSubmission saved to: experiments/SUB_exp6_enhanced_features.csv")

    # Save results
    import json
    results = {
        'experiment': 'exp6_enhanced_correlation_features',
        'best_strategy': best_strategy,
        'validation_auprc': float(best_ap),
        'validation_auroc': float(best_auc),
        'n_detectors': len(keep),
        'detectors_used': keep,
        'top_detector_auprc': max(per_ap.values()),
        'top_detector': max(per_ap, key=per_ap.get)
    }

    with open("experiments/exp6_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Detailed results saved to: experiments/exp6_results.json")
    print("\n" + "="*80)
    print("EXPERIMENT 6 COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
