#!/usr/bin/env python3
"""
Experiment 5: Correlation-Guided Ensemble with Feature Weighting

Based on validation correlation analysis, we know:
1. CreditScore (r=-0.250) - STRONGEST predictor
2. NonInterestBearingUPB (r=+0.14) - Payment structure anomalies
3. OriginalDTI (r=+0.100) - Financial stress
4. OriginalInterestRate (r=+0.096) - Risk profile

Strategy:
- Create specialized detectors for high-correlation features
- Weight ensemble members by feature importance
- Add correlation-specific composite features
- Use top correlated features for distance metrics
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.append('final_approach')
from feature_builder_advanced import FeatureBuilderAdvanced

RNG = np.random.default_rng(42)

# ========== CORRELATION-GUIDED FEATURE ENGINEERING ==========

def create_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features based on top correlated individual features
    These are domain-driven risk indicators
    """
    features = pd.DataFrame(index=df.index)

    # 1. Credit Risk Score (combines CreditScore, DTI, InterestRate)
    if 'CreditScore' in df.columns:
        # Lower credit score = higher risk (flip sign for risk score)
        features['risk_credit'] = (850 - df['CreditScore']) / 150.0  # normalize to ~[0,1]
    else:
        features['risk_credit'] = 0.0

    if 'OriginalDTI' in df.columns:
        # Higher DTI = higher risk
        features['risk_dti'] = np.clip(df['OriginalDTI'] / 100.0, 0, 2)  # normalize
    else:
        features['risk_dti'] = 0.0

    if 'OriginalInterestRate' in df.columns:
        # Higher interest rate = higher risk
        features['risk_rate'] = np.clip((df['OriginalInterestRate'] - 3.0) / 7.0, 0, 2)
    else:
        features['risk_rate'] = 0.0

    # Composite risk score (weighted by correlation strength)
    features['composite_risk_score'] = (
        0.50 * features['risk_credit'] +    # CreditScore weight (r=0.25)
        0.25 * features['risk_dti'] +       # DTI weight (r=0.10)
        0.25 * features['risk_rate']        # InterestRate weight (r=0.096)
    )

    # 2. Payment Irregularity Score (NonInterestBearingUPB pattern)
    nibupb_cols = [c for c in df.columns if 'NonInterestBearingUPB' in c]
    if len(nibupb_cols) >= 3:
        # Focus on later periods (12, 13 show strongest correlation)
        late_cols = [c for c in nibupb_cols if any(f'{i}_' in c for i in [10,11,12,13])]
        if late_cols:
            late_nibupb = df[late_cols].fillna(0).values
            features['payment_irregularity'] = np.mean(late_nibupb, axis=1) / 100000.0  # normalize
            features['payment_irregularity_max'] = np.max(late_nibupb, axis=1) / 100000.0
        else:
            features['payment_irregularity'] = 0.0
            features['payment_irregularity_max'] = 0.0
    else:
        features['payment_irregularity'] = 0.0
        features['payment_irregularity_max'] = 0.0

    # 3. Borrower Quality (Number of borrowers correlation r=-0.073)
    if 'NumberOfBorrowers' in df.columns:
        # Single borrower = slightly higher risk
        features['single_borrower'] = (df['NumberOfBorrowers'] == 1).astype(float)
    else:
        features['single_borrower'] = 0.0

    # 4. Property valuation method risk
    if 'PropertyValMethod' in df.columns:
        # Method correlation r=+0.051 suggests certain methods indicate risk
        features['prop_val_risk'] = df['PropertyValMethod'].fillna(0) / 10.0
    else:
        features['prop_val_risk'] = 0.0

    return features.fillna(0)


def get_top_correlated_feature_indices(df: pd.DataFrame) -> List[int]:
    """
    Returns indices of top correlated features for specialized processing
    """
    top_features = [
        'CreditScore',
        '13_CurrentNonInterestBearingUPB',
        '12_CurrentNonInterestBearingUPB',
        'OriginalDTI',
        'OriginalInterestRate',
        'NumberOfBorrowers'
    ]

    indices = []
    for feat in top_features:
        if feat in df.columns:
            indices.append(df.columns.get_loc(feat))

    return indices


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

# ========== SPECIALIZED DETECTORS ==========

def lof_on_top_features(Xtr: np.ndarray, X: np.ndarray, top_indices: List[int], k=7) -> np.ndarray:
    """LOF using only top correlated features"""
    if len(top_indices) == 0:
        return np.zeros(len(X))

    Xtr_sub = Xtr[:, top_indices]
    X_sub = X[:, top_indices]

    model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xtr_sub)
    return -model.score_samples(X_sub)


def distance_on_composite_risk(Xtr_risk: np.ndarray, X_risk: np.ndarray, k=7) -> np.ndarray:
    """k-distance on composite risk scores"""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr_risk.reshape(-1, 1))
    d, _ = nn.kneighbors(X_risk.reshape(-1, 1))
    return d[:, -1]


def isolation_forest_weighted(Xtr: np.ndarray, X: np.ndarray,
                               feature_weights: np.ndarray,
                               n_estimators=300) -> np.ndarray:
    """
    Isolation Forest with feature-weighted sampling
    Weight features by their absolute correlation with anomalies
    """
    # Sample columns with probability proportional to feature weights
    n_features = Xtr.shape[1]
    n_selected = min(int(n_features * 0.8), n_features)

    # Normalize weights
    weights_norm = feature_weights / (feature_weights.sum() + 1e-12)

    scores = []
    for _ in range(n_estimators):
        # Sample features weighted by importance
        selected_features = RNG.choice(
            n_features,
            size=n_selected,
            replace=False,
            p=weights_norm
        )

        Xtr_sub = Xtr[:, selected_features]
        X_sub = X[:, selected_features]

        # Fit IF on subset
        model = IsolationForest(
            n_estimators=1,
            max_samples=min(512, len(Xtr_sub)),
            contamination="auto",
            random_state=RNG.integers(0, 100000)
        ).fit(Xtr_sub)

        scores.append(-model.score_samples(X_sub))

    # Average scores across all iterations
    return np.mean(scores, axis=0)


# ========== MAIN EXPERIMENT ==========

def main():
    print("="*80)
    print("EXPERIMENT 5: CORRELATION-GUIDED ENSEMBLE")
    print("="*80)

    # Load data
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test = pd.read_csv("Data/loans_test.csv")

    yv = valid["target"].to_numpy()

    print(f"\nTrain: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
    print(f"Validation anomaly rate: {yv.mean()*100:.2f}%")

    # ========== STEP 1: Create correlation-based composite features ==========
    print("\n" + "="*80)
    print("STEP 1: Creating correlation-guided composite features")
    print("="*80)

    corr_features_train = create_correlation_features(train)
    corr_features_valid = create_correlation_features(valid)
    corr_features_test = create_correlation_features(test)

    print(f"\nCreated {corr_features_train.shape[1]} composite features:")
    print(corr_features_train.columns.tolist())

    # Add to datasets
    train_aug = pd.concat([train, corr_features_train], axis=1)
    valid_aug = pd.concat([valid, corr_features_valid], axis=1)
    test_aug = pd.concat([test, corr_features_test], axis=1)

    # ========== STEP 2: Build feature representation ==========
    print("\n" + "="*80)
    print("STEP 2: Building feature representation with PCA")
    print("="*80)

    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=100)
    fb.fit(train_aug)

    Xtr_scaled, sl_tr, Xtr = fb.transform(train_aug)
    Xv_scaled, sl_v, Xv = fb.transform(valid_aug)
    Xt_scaled, sl_t, Xt = fb.transform(test_aug)

    print(f"Scaled features: {Xtr_scaled.shape}")
    print(f"PCA embedding: {Xtr.shape}")

    # Extract composite risk scores for specialized detectors
    risk_score_idx = [i for i, col in enumerate(corr_features_train.columns) if 'composite_risk_score' in col]
    if risk_score_idx:
        # These are appended to end of feature matrix
        offset = Xtr_scaled.shape[1] - len(corr_features_train.columns)
        risk_idx = offset + risk_score_idx[0]
        Xtr_risk = Xtr_scaled[:, risk_idx]
        Xv_risk = Xv_scaled[:, risk_idx]
        Xt_risk = Xt_scaled[:, risk_idx]
    else:
        Xtr_risk = np.zeros(len(Xtr_scaled))
        Xv_risk = np.zeros(len(Xv_scaled))
        Xt_risk = np.zeros(len(Xt_scaled))

    # ========== STEP 3: Train ensemble detectors ==========
    print("\n" + "="*80)
    print("STEP 3: Training correlation-guided ensemble detectors")
    print("="*80)

    detectors_valid: Dict[str, np.ndarray] = {}
    detectors_test: Dict[str, np.ndarray] = {}
    detectors_train: Dict[str, np.ndarray] = {}

    # 1. Standard LOF on full embedding (baseline)
    print("\n[1/10] LOF on full PCA embedding (k=5,7,10)...")
    for k in [5, 7, 10]:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xtr)
        detectors_valid[f'lof_k{k}'] = -model.score_samples(Xv)
        detectors_test[f'lof_k{k}'] = -model.score_samples(Xt)
        detectors_train[f'lof_k{k}'] = -model.score_samples(Xtr)

    # 2. LOF on TOP CORRELATED FEATURES ONLY (specialized)
    print("[2/10] LOF on top correlated features only...")
    top_feat_idx = get_top_correlated_feature_indices(train_aug)
    if len(top_feat_idx) > 0:
        detectors_valid['lof_top_corr'] = lof_on_top_features(Xtr_scaled, Xv_scaled, top_feat_idx, k=7)
        detectors_test['lof_top_corr'] = lof_on_top_features(Xtr_scaled, Xt_scaled, top_feat_idx, k=7)
        detectors_train['lof_top_corr'] = lof_on_top_features(Xtr_scaled, Xtr_scaled, top_feat_idx, k=7)

    # 3. K-distance on composite risk score
    print("[3/10] k-distance on composite risk score...")
    detectors_valid['kdist_risk'] = distance_on_composite_risk(Xtr_risk, Xv_risk, k=9)
    detectors_test['kdist_risk'] = distance_on_composite_risk(Xtr_risk, Xt_risk, k=9)
    detectors_train['kdist_risk'] = distance_on_composite_risk(Xtr_risk, Xtr_risk, k=9)

    # 4. Isolation Forest (standard)
    print("[4/10] Isolation Forest on PCA embedding...")
    if_model = IsolationForest(n_estimators=400, max_samples=1.0, contamination="auto",
                                random_state=42, n_jobs=-1).fit(Xtr)
    detectors_valid['iforest'] = -if_model.score_samples(Xv)
    detectors_test['iforest'] = -if_model.score_samples(Xt)
    detectors_train['iforest'] = -if_model.score_samples(Xtr)

    # 5. Feature-weighted Isolation Forest (specialized)
    print("[5/10] Feature-weighted Isolation Forest...")
    # Create feature weights based on correlation (use uniform for PCA space, or based on top features)
    feature_weights = np.ones(Xtr_scaled.shape[1])
    # Upweight indices corresponding to top correlated features if available
    for idx in top_feat_idx:
        if idx < len(feature_weights):
            feature_weights[idx] = 2.5  # 2.5x weight for top features

    detectors_valid['iforest_weighted'] = isolation_forest_weighted(
        Xtr_scaled, Xv_scaled, feature_weights, n_estimators=200
    )
    detectors_test['iforest_weighted'] = isolation_forest_weighted(
        Xtr_scaled, Xt_scaled, feature_weights, n_estimators=200
    )
    detectors_train['iforest_weighted'] = isolation_forest_weighted(
        Xtr_scaled, Xtr_scaled, feature_weights, n_estimators=200
    )

    # 6. KMeans + cluster-wise LOF
    print("[6/10] KMeans clustering + cluster-wise LOF...")
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

    # 7. k-distance multi-k
    print("[7/10] k-distance detectors (k=3,5,7,9)...")
    for k in [3, 5, 7, 9]:
        nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
        d, _ = nn.kneighbors(Xv)
        detectors_valid[f'kdist_k{k}'] = d[:, -1]
        d, _ = nn.kneighbors(Xt)
        detectors_test[f'kdist_k{k}'] = d[:, -1]
        d, _ = nn.kneighbors(Xtr)
        detectors_train[f'kdist_k{k}'] = d[:, -1]

    # 8. Elliptic Envelope (robust covariance)
    print("[8/10] Elliptic Envelope...")
    try:
        ee_model = EllipticEnvelope(contamination=0.1, support_fraction=0.9, random_state=42).fit(Xtr)
        detectors_valid['ell_env'] = -ee_model.score_samples(Xv)
        detectors_test['ell_env'] = -ee_model.score_samples(Xt)
        detectors_train['ell_env'] = -ee_model.score_samples(Xtr)
    except Exception as e:
        print(f"  Elliptic Envelope failed: {e}")

    # 9. PCA reconstruction error
    print("[9/10] PCA reconstruction error...")
    pca_recon = PCA(n_components=min(100, Xtr_scaled.shape[1]), random_state=42).fit(Xtr_scaled)
    def recon_err(X_scaled):
        Xr = pca_recon.inverse_transform(pca_recon.transform(X_scaled))
        return np.mean((X_scaled - Xr)**2, axis=1)

    detectors_train['pca_recon'] = recon_err(Xtr_scaled)
    detectors_valid['pca_recon'] = recon_err(Xv_scaled)
    detectors_test['pca_recon'] = recon_err(Xt_scaled)

    # 10. Direct composite risk score (as detector)
    print("[10/10] Direct composite risk score...")
    detectors_valid['composite_risk'] = Xv_risk
    detectors_test['composite_risk'] = Xt_risk
    detectors_train['composite_risk'] = Xtr_risk

    # ========== STEP 4: Evaluate individual detectors ==========
    print("\n" + "="*80)
    print("STEP 4: Evaluating individual detectors on validation set")
    print("="*80)

    per_ap = {k: average_precision_score(yv, v) for k, v in detectors_valid.items()}
    per_auc = {k: roc_auc_score(yv, v) for k, v in detectors_valid.items()}

    print("\nTop 10 detectors by AUPRC:")
    for k, v in sorted(per_ap.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {k:30s}: AUPRC={v:.4f}, AUROC={per_auc[k]:.4f}")

    # Select top detectors
    sorted_dets = sorted(per_ap.items(), key=lambda kv: kv[1], reverse=True)
    keep = [k for k, _ in sorted_dets if per_ap[k] >= 0.15][:12]  # Top 12 with threshold
    if not keep:
        keep = [sorted_dets[0][0]]

    print(f"\nSelected {len(keep)} detectors for ensemble:")
    for k in keep:
        print(f"  {k}: {per_ap[k]:.4f}")

    # ========== STEP 5: Ensemble fusion ==========
    print("\n" + "="*80)
    print("STEP 5: Ensemble fusion with correlation-based weighting")
    print("="*80)

    # Calibrate using train CDF
    prob_valid = {}
    prob_test = {}
    weights = {}

    for name in keep:
        tr = detectors_train.get(name, None)
        if tr is None:
            prob_valid[name] = rank01(detectors_valid[name])
            prob_test[name] = rank01(detectors_test[name])
        else:
            cdf = train_cdf(tr)
            prob_valid[name] = cdf(detectors_valid[name])
            prob_test[name] = cdf(detectors_test[name])

        # Weight by AUPRC performance
        weights[name] = max(per_ap[name], 1e-6)

    # Try multiple fusion strategies
    fusion_strategies = {}

    # 1. Weighted average (rank-based)
    R_valid = np.column_stack([rank01(detectors_valid[k]) for k in keep])
    R_test = np.column_stack([rank01(detectors_test[k]) for k in keep])
    w = np.array([weights[k] for k in keep])
    w = w / w.sum()

    fusion_strategies['wavg_rank'] = {
        'valid': np.average(R_valid, axis=1, weights=w),
        'test': np.average(R_test, axis=1, weights=w)
    }

    # 2. Max rank
    fusion_strategies['max_rank'] = {
        'valid': np.max(R_valid, axis=1),
        'test': np.max(R_test, axis=1)
    }

    # 3. Top-3 max
    top3 = sorted(keep, key=lambda n: weights[n], reverse=True)[:3]
    R3_valid = np.column_stack([rank01(detectors_valid[n]) for n in top3])
    R3_test = np.column_stack([rank01(detectors_test[n]) for n in top3])
    fusion_strategies['max_rank_top3'] = {
        'valid': np.max(R3_valid, axis=1),
        'test': np.max(R3_test, axis=1)
    }

    # 4. Probability-based weighted average
    P_valid = np.column_stack([prob_valid[k] for k in keep])
    P_test = np.column_stack([prob_test[k] for k in keep])
    fusion_strategies['wavg_prob'] = {
        'valid': np.average(P_valid, axis=1, weights=w),
        'test': np.average(P_test, axis=1, weights=w)
    }

    # 5. Noisy-OR probability
    fusion_strategies['noisy_or'] = {
        'valid': 1.0 - np.prod(1.0 - P_valid, axis=1),
        'test': 1.0 - np.prod(1.0 - P_test, axis=1)
    }

    # Evaluate all fusion strategies
    print("\nFusion strategy performance:")
    best_strategy = None
    best_ap = -1
    best_auc = -1

    for name, scores in fusion_strategies.items():
        ap = average_precision_score(yv, scores['valid'])
        auc = roc_auc_score(yv, scores['valid'])
        print(f"  {name:20s}: AUPRC={ap:.4f}, AUROC={auc:.4f}")

        if ap > best_ap:
            best_strategy = name
            best_ap = ap
            best_auc = auc

    print(f"\n{'='*80}")
    print(f"BEST STRATEGY: {best_strategy}")
    print(f"Validation AUPRC: {best_ap:.4f}")
    print(f"Validation AUROC: {best_auc:.4f}")
    print(f"{'='*80}")

    # ========== STEP 6: Generate submission ==========
    best_test_scores = fusion_strategies[best_strategy]['test']

    idc = "Id" if "Id" in test.columns else "index"
    submission = pd.DataFrame({
        idc: test[idc],
        'anomaly_score': best_test_scores
    })

    output_path = "experiments/SUB_exp5_correlation_guided.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")

    # Save detailed results
    results = {
        'experiment': 'exp5_correlation_guided_ensemble',
        'best_strategy': best_strategy,
        'validation_auprc': float(best_ap),
        'validation_auroc': float(best_auc),
        'n_detectors': len(keep),
        'detectors_used': keep,
        'individual_auprc': {k: float(per_ap[k]) for k in keep},
        'strategy_performance': {
            name: {
                'auprc': float(average_precision_score(yv, scores['valid'])),
                'auroc': float(roc_auc_score(yv, scores['valid']))
            }
            for name, scores in fusion_strategies.items()
        }
    }

    import json
    with open("experiments/exp5_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to: experiments/exp5_results.json")
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
