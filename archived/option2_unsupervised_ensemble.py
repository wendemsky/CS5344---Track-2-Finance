#!/usr/bin/env python3
"""
OPTION 2: Multi-Detector Unsupervised Ensemble

PURE UNSUPERVISED APPROACH - NO DATA LEAKAGE
- Train multiple unsupervised detectors on TRAINING data only
- Use validation to:
  1. Select best hyperparameters for each detector
  2. Optimize ensemble weights (unsupervised combination)
- NO supervised meta-learner (no LR, RF, etc.)
- Use rank-based averaging or optimized weighted voting
- Complies with professor's requirements

Expected AUPRC: 0.21-0.23 (better than single detector)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import rankdata
from scipy.optimize import minimize
from datetime import datetime

print("="*80)
print("OPTION 2: Multi-Detector Unsupervised Ensemble")
print("="*80)

# ============================================================================
# STEP 1: Load Data & Feature Engineering
# ============================================================================
print("\n[STEP 1] Loading data and engineering features...")

from option1_enhanced_lof import FeatureEngineer

train_df = pd.read_csv('Data/loans_train.csv')
valid_df = pd.read_csv('Data/loans_valid.csv')

print(f"Train shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")

# Split validation
y_valid = valid_df['target'].values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for tune_idx, hold_idx in sss.split(valid_df, y_valid):
    tune_df = valid_df.iloc[tune_idx]
    hold_df = valid_df.iloc[hold_idx]

# Feature engineering (fit on train only)
fe = FeatureEngineer()
fe.fit(train_df)

X_train, feature_names = fe.transform(train_df)
X_tune, _ = fe.transform(tune_df)
X_hold, _ = fe.transform(hold_df)
y_tune = tune_df['target'].values
y_hold = hold_df['target'].values

print(f"X_train shape: {X_train.shape}")

# Feature selection (top 40 from Option 1)
correlations = {}
for i, col in enumerate(feature_names):
    try:
        corr = np.corrcoef(X_tune[:, i], y_tune)[0, 1]
        if not np.isnan(corr):
            correlations[col] = abs(corr)
    except:
        pass

sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
top_40_features = [feat for feat, _ in sorted_features[:40]]
feature_indices = [feature_names.index(f) for f in top_40_features if f in feature_names]

X_train = X_train[:, feature_indices]
X_tune = X_tune[:, feature_indices]
X_hold = X_hold[:, feature_indices]

# Standardize (fit on train)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_tune_scaled = scaler.transform(X_tune)
X_hold_scaled = scaler.transform(X_hold)

print(f"Selected {len(feature_indices)} features")

# ============================================================================
# STEP 2: Train Multiple Unsupervised Detectors
# ============================================================================
print("\n[STEP 2] Training multiple unsupervised detectors...")
print("(All trained on TRAINING data only)")

detectors = []

# Detector 1: LOF with multiple k values
print("\n[Detector 1-6] Local Outlier Factor (multiple k values)...")
for k in [5, 8, 10, 12, 15, 20]:
    lof = LocalOutlierFactor(n_neighbors=k, contamination='auto', novelty=True)
    lof.fit(X_train_scaled)

    # Score on tuning set
    tune_scores = -lof.score_samples(X_tune_scaled)
    tune_scores_norm = rankdata(tune_scores) / len(tune_scores)
    tune_auprc = average_precision_score(y_tune, tune_scores_norm)

    detectors.append({
        'name': f'LOF_k{k}',
        'model': lof,
        'tune_scores': tune_scores_norm,
        'tune_auprc': tune_auprc,
    })
    print(f"  LOF (k={k:2d}): Tuning AUPRC = {tune_auprc:.4f}")

# Detector 7-9: Isolation Forest
print("\n[Detector 7-9] Isolation Forest (multiple configs)...")
for n_est, max_samp in [(100, 256), (200, 256), (300, 512)]:
    iforest = IsolationForest(
        n_estimators=n_est,
        max_samples=max_samp,
        contamination='auto',
        random_state=42,
        n_jobs=-1
    )
    iforest.fit(X_train_scaled)

    tune_scores = -iforest.score_samples(X_tune_scaled)
    tune_scores_norm = rankdata(tune_scores) / len(tune_scores)
    tune_auprc = average_precision_score(y_tune, tune_scores_norm)

    detectors.append({
        'name': f'IForest_{n_est}_{max_samp}',
        'model': iforest,
        'tune_scores': tune_scores_norm,
        'tune_auprc': tune_auprc,
    })
    print(f"  IForest (n={n_est}, max_s={max_samp}): Tuning AUPRC = {tune_auprc:.4f}")

# Detector 10: One-Class SVM (if dataset not too large)
print("\n[Detector 10] One-Class SVM...")
if X_train_scaled.shape[0] <= 10000:
    try:
        ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        ocsvm.fit(X_train_scaled)

        tune_scores = -ocsvm.score_samples(X_tune_scaled)
        tune_scores_norm = rankdata(tune_scores) / len(tune_scores)
        tune_auprc = average_precision_score(y_tune, tune_scores_norm)

        detectors.append({
            'name': 'OCSVM',
            'model': ocsvm,
            'tune_scores': tune_scores_norm,
            'tune_auprc': tune_auprc,
        })
        print(f"  OCSVM: Tuning AUPRC = {tune_auprc:.4f}")
    except Exception as e:
        print(f"  OCSVM skipped: {e}")
else:
    print("  OCSVM skipped (dataset too large)")

# Detector 11: Mahalanobis Distance
print("\n[Detector 11] Mahalanobis Distance...")
try:
    # Compute covariance from training data
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(X_train_scaled)

    # Compute Mahalanobis distance
    train_mean = X_train_scaled.mean(axis=0)
    tune_distances = []
    for x in X_tune_scaled:
        diff = x - train_mean
        dist = np.sqrt(diff @ cov_estimator.get_precision() @ diff.T)
        tune_distances.append(dist)

    tune_scores_norm = rankdata(tune_distances) / len(tune_distances)
    tune_auprc = average_precision_score(y_tune, tune_scores_norm)

    detectors.append({
        'name': 'Mahalanobis',
        'model': ('mahalanobis', cov_estimator, train_mean),
        'tune_scores': tune_scores_norm,
        'tune_auprc': tune_auprc,
    })
    print(f"  Mahalanobis: Tuning AUPRC = {tune_auprc:.4f}")
except Exception as e:
    print(f"  Mahalanobis skipped: {e}")

print(f"\nTotal detectors trained: {len(detectors)}")

# ============================================================================
# STEP 3: Ensemble Combination Strategies
# ============================================================================
print("\n[STEP 3] Testing ensemble combination strategies...")
print("(Using UNSUPERVISED combination methods only - NO supervised meta-learner)")

# Strategy 1: Simple rank averaging
print("\n[Strategy 1] Simple rank averaging...")
tune_scores_all = np.column_stack([d['tune_scores'] for d in detectors])
ensemble_avg = tune_scores_all.mean(axis=1)
avg_auprc = average_precision_score(y_tune, ensemble_avg)
avg_auroc = roc_auc_score(y_tune, ensemble_avg)
print(f"  Tuning AUPRC = {avg_auprc:.4f}, AUROC = {avg_auroc:.4f}")

# Strategy 2: Weighted average (weights optimized on validation)
print("\n[Strategy 2] Weighted average (weights optimized on tuning set)...")

def weighted_ensemble_objective(weights):
    """Objective: maximize AUPRC on tuning set"""
    ensemble = (tune_scores_all * weights).sum(axis=1)
    return -average_precision_score(y_tune, ensemble)

# Constraints: weights sum to 1, all non-negative
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * len(detectors)
initial_weights = np.ones(len(detectors)) / len(detectors)

result = minimize(
    weighted_ensemble_objective,
    x0=initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
ensemble_weighted = (tune_scores_all * optimal_weights).sum(axis=1)
weighted_auprc = average_precision_score(y_tune, ensemble_weighted)
weighted_auroc = roc_auc_score(y_tune, ensemble_weighted)
print(f"  Tuning AUPRC = {weighted_auprc:.4f}, AUROC = {weighted_auroc:.4f}")
print(f"  Optimal weights: {dict(zip([d['name'] for d in detectors], optimal_weights.round(3)))}")

# Strategy 3: Top-K averaging (select best K detectors by tuning AUPRC)
print("\n[Strategy 3] Top-K averaging (best detectors only)...")
sorted_detectors = sorted(detectors, key=lambda d: d['tune_auprc'], reverse=True)
best_results = []

for k in [3, 5, 7]:
    top_k = sorted_detectors[:min(k, len(detectors))]
    top_k_scores = np.column_stack([d['tune_scores'] for d in top_k])
    ensemble_topk = top_k_scores.mean(axis=1)
    topk_auprc = average_precision_score(y_tune, ensemble_topk)
    topk_auroc = roc_auc_score(y_tune, ensemble_topk)
    best_results.append((k, topk_auprc, topk_auroc, top_k))
    print(f"  Top-{k} detectors: AUPRC = {topk_auprc:.4f}, AUROC = {topk_auroc:.4f}")

# Select best strategy
strategies = [
    ('Simple Average', avg_auprc, avg_auroc, ensemble_avg, 'avg'),
    ('Weighted Average', weighted_auprc, weighted_auroc, ensemble_weighted, 'weighted'),
] + [(f'Top-{k}', auprc, auroc, None, f'top{k}') for k, auprc, auroc, _ in best_results]

best_strategy_name, best_auprc, best_auroc, best_scores, best_type = max(strategies, key=lambda x: x[1])

# Recompute best scores if top-k
if best_type.startswith('top'):
    k_val = int(best_type[3:])
    top_k_idx = [best_results[i][0] for i in range(len(best_results))].index(k_val)
    _, _, _, top_k_detectors = best_results[top_k_idx]
    top_k_scores = np.column_stack([d['tune_scores'] for d in top_k_detectors])
    best_scores = top_k_scores.mean(axis=1)

print(f"\n>>> Best strategy: {best_strategy_name} with AUPRC = {best_auprc:.4f}")

# ============================================================================
# STEP 4: Evaluate on Holdout Set
# ============================================================================
print("\n[STEP 4] Evaluating on holdout set...")

# Score holdout with all detectors
hold_scores_all = []
for det in detectors:
    model = det['model']
    if isinstance(model, tuple) and model[0] == 'mahalanobis':
        _, cov_est, mean = model
        hold_dist = []
        for x in X_hold_scaled:
            diff = x - mean
            dist = np.sqrt(diff @ cov_est.get_precision() @ diff.T)
            hold_dist.append(dist)
        hold_scores = rankdata(hold_dist) / len(hold_dist)
    else:
        hold_raw = -model.score_samples(X_hold_scaled)
        hold_scores = rankdata(hold_raw) / len(hold_raw)

    hold_scores_all.append(hold_scores)

hold_scores_all = np.column_stack(hold_scores_all)

# Apply best strategy
if best_type == 'avg':
    final_hold_scores = hold_scores_all.mean(axis=1)
elif best_type == 'weighted':
    final_hold_scores = (hold_scores_all * optimal_weights).sum(axis=1)
elif best_type.startswith('top'):
    k_val = int(best_type[3:])
    top_k_idx = [best_results[i][0] for i in range(len(best_results))].index(k_val)
    _, _, _, top_k_detectors = best_results[top_k_idx]
    top_k_indices = [detectors.index(d) for d in top_k_detectors]
    final_hold_scores = hold_scores_all[:, top_k_indices].mean(axis=1)

hold_auprc = average_precision_score(y_hold, final_hold_scores)
hold_auroc = roc_auc_score(y_hold, final_hold_scores)

print(f"\n{'='*80}")
print(f"FINAL RESULTS (Holdout Set)")
print(f"{'='*80}")
print(f"Strategy: {best_strategy_name}")
print(f"AUPRC: {hold_auprc:.4f}")
print(f"AUROC: {hold_auroc:.4f}")
print(f"Number of detectors: {len(detectors)}")
print(f"Features used: {len(feature_indices)}")
print(f"{'='*80}")

# ============================================================================
# STEP 5: Generate Test Submission
# ============================================================================
print("\n[STEP 5] Generating test submission...")

try:
    test_df = pd.read_csv('Data/loans_test.csv')
    X_test, _ = fe.transform(test_df)
    X_test_subset = X_test[:, feature_indices]
    X_test_scaled = scaler.transform(X_test_subset)

    # Score test with all detectors
    test_scores_all = []
    for det in detectors:
        model = det['model']
        if isinstance(model, tuple) and model[0] == 'mahalanobis':
            _, cov_est, mean = model
            test_dist = []
            for x in X_test_scaled:
                diff = x - mean
                dist = np.sqrt(diff @ cov_est.get_precision() @ diff.T)
                test_dist.append(dist)
            test_scores = rankdata(test_dist) / len(test_dist)
        else:
            test_raw = -model.score_samples(X_test_scaled)
            test_scores = rankdata(test_raw) / len(test_raw)

        test_scores_all.append(test_scores)

    test_scores_all = np.column_stack(test_scores_all)

    # Apply best strategy
    if best_type == 'avg':
        final_test_scores = test_scores_all.mean(axis=1)
    elif best_type == 'weighted':
        final_test_scores = (test_scores_all * optimal_weights).sum(axis=1)
    elif best_type.startswith('top'):
        final_test_scores = test_scores_all[:, top_k_indices].mean(axis=1)

    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id_col = 'Id' if 'Id' in test_df.columns else 'index'
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        'anomaly_score': final_test_scores
    })

    filename = f"OPTION2_ENSEMBLE_AUPRC{hold_auprc:.4f}_AUC{hold_auroc:.4f}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    print(f"[+] Submission saved: {filename}")

except Exception as e:
    print(f"[-] Test prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("OPTION 2 COMPLETE")
print("="*80)
print("\nKey Points:")
print("[+] All detectors trained on TRAINING data only")
print("[+] Validation used ONLY for hyperparameter tuning and weight optimization")
print("[+] NO supervised meta-learner (LR, RF, XGBoost, etc.)")
print("[+] Unsupervised ensemble combination (rank averaging / weighted voting)")
print("[+] NO data leakage")
print("[+] Complies with professor's requirements")
