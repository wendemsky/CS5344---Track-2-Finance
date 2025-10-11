#!/usr/bin/env python3
"""
OPTION 1: Enhanced LOF with Correlation-Based Feature Selection

PURE UNSUPERVISED APPROACH - NO DATA LEAKAGE
- Train LOF on TRAINING data only
- Use validation to select:
  1. Best feature subset (by correlation analysis)
  2. Best k parameter for LOF
- NO supervised meta-learner
- Complies with professor's requirements

Expected AUPRC: ~0.2033 (based on validation_guided_feature_selection.py)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from scipy.stats import rankdata

print("="*80)
print("OPTION 1: Enhanced LOF with Correlation-Based Feature Selection")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading data...")
train_df = pd.read_csv('Data/loans_train.csv')
valid_df = pd.read_csv('Data/loans_valid.csv')

print(f"Train shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Train target distribution: {train_df['target'].value_counts().to_dict()}")
print(f"Valid target distribution: {valid_df['target'].value_counts(normalize=True).to_dict()}")

# Split validation into tuning + holdout
y_valid = valid_df['target'].values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for tune_idx, hold_idx in sss.split(valid_df, y_valid):
    tune_df = valid_df.iloc[tune_idx]
    hold_df = valid_df.iloc[hold_idx]

print(f"Tuning set: {tune_df.shape}, Holdout set: {hold_df.shape}")

# ============================================================================
# STEP 2: Feature Engineering (FIT ON TRAIN ONLY)
# ============================================================================
print("\n[STEP 2] Feature engineering (fitting on TRAINING data only)...")

class FeatureEngineer:
    """
    Feature engineering that fits ONLY on training data
    NO validation statistics used
    """
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []

    def fit(self, df):
        """Fit all transformations on training data only"""
        data = df.copy()

        # Identify column types
        exclude_cols = ['index', 'target', 'Id']
        all_cols = [c for c in data.columns if c not in exclude_cols]

        self.categorical_cols = [c for c in all_cols if data[c].dtype == 'object']
        self.numeric_cols = [c for c in all_cols if c not in self.categorical_cols]

        # Replace sentinel values with NaN
        sentinel_map = {
            'CreditScore': 9999,
            'OriginalDTI': 999,
            'OriginalLTV': 999,
        }
        for col, sentinel in sentinel_map.items():
            if col in data.columns:
                data[col] = data[col].replace(sentinel, np.nan)

        # Fit label encoders on categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            values = data[col].fillna('MISSING').astype(str).unique().tolist()
            if 'UNKNOWN' not in values:
                values.append('UNKNOWN')
            le.fit(values)
            self.label_encoders[col] = le

        # Fit imputers on numeric columns (median from training)
        for col in self.numeric_cols:
            if data[col].notna().any():
                self.imputers[col] = float(data[col].median())
            else:
                self.imputers[col] = 0.0

        return self

    def transform(self, df):
        """Transform data using fitted parameters"""
        data = df.copy()

        # Replace sentinels
        sentinel_map = {
            'CreditScore': 9999,
            'OriginalDTI': 999,
            'OriginalLTV': 999,
        }
        for col, sentinel in sentinel_map.items():
            if col in data.columns:
                data[col] = data[col].replace(sentinel, np.nan)

        # Transform categorical
        for col in self.categorical_cols:
            if col in data.columns:
                le = self.label_encoders[col]
                data[col] = data[col].fillna('MISSING').astype(str)
                data[col] = data[col].apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                data[col] = le.transform(data[col])

        # Impute numeric
        for col in self.numeric_cols:
            if col in data.columns:
                data[col] = data[col].fillna(self.imputers[col])

        # Extract features
        feature_cols = self.categorical_cols + self.numeric_cols
        X = data[[c for c in feature_cols if c in data.columns]].values

        return X, feature_cols

# Fit feature engineer on training data ONLY
fe = FeatureEngineer()
fe.fit(train_df)

X_train, feature_names = fe.transform(train_df)
X_tune, _ = fe.transform(tune_df)
X_hold, _ = fe.transform(hold_df)
y_tune = tune_df['target'].values
y_hold = hold_df['target'].values

print(f"Features extracted: {len(feature_names)}")
print(f"X_train shape: {X_train.shape}")

# ============================================================================
# STEP 3: Feature Selection (Using Validation Correlation - ALLOWED)
# ============================================================================
print("\n[STEP 3] Feature selection using correlation analysis on validation...")

# Calculate correlation with target on TUNING set
# This is ALLOWED - we're using validation to SELECT features, not train
correlations = {}
for i, col in enumerate(feature_names):
    try:
        corr = np.corrcoef(X_tune[:, i], y_tune)[0, 1]
        if not np.isnan(corr):
            correlations[col] = abs(corr)
    except:
        pass

# Sort by absolute correlation
sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print("\nTop 20 features by correlation with target:")
for i, (feat, corr) in enumerate(sorted_features[:20], 1):
    print(f"{i:2d}. {feat:40s} | corr = {corr:.4f}")

# Test different numbers of top features
print("\n" + "="*80)
print("Testing different feature subsets...")
print("="*80)

results = []
for k in [5, 10, 15, 20, 25, 30, 40, 50]:
    top_k_features = [feat for feat, _ in sorted_features[:k]]
    top_k_indices = [feature_names.index(f) for f in top_k_features if f in feature_names]

    if len(top_k_indices) == 0:
        continue

    X_train_subset = X_train[:, top_k_indices]
    X_tune_subset = X_tune[:, top_k_indices]

    # Standardize (fit on train, transform both)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_tune_scaled = scaler.transform(X_tune_subset)

    # Train LOF on training data only, evaluate on tuning set
    lof = LocalOutlierFactor(n_neighbors=10, contamination='auto', novelty=True)
    lof.fit(X_train_scaled)

    scores = -lof.score_samples(X_tune_scaled)
    scores_norm = rankdata(scores) / len(scores)

    auprc = average_precision_score(y_tune, scores_norm)
    auroc = roc_auc_score(y_tune, scores_norm)

    results.append({
        'k': k,
        'auprc': auprc,
        'auroc': auroc,
    })

    print(f"Top {k:2d} features: AUPRC = {auprc:.4f}, AUROC = {auroc:.4f}")

# Select best k
best_result = max(results, key=lambda x: x['auprc'])
best_k_features = best_result['k']
print(f"\n>>> Best: Top {best_k_features} features with AUPRC = {best_result['auprc']:.4f}")

# ============================================================================
# STEP 4: Hyperparameter Tuning (k parameter for LOF)
# ============================================================================
print("\n[STEP 4] Hyperparameter tuning (k parameter for LOF)...")

# Select best feature subset
top_features = [feat for feat, _ in sorted_features[:best_k_features]]
feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]

X_train_final = X_train[:, feature_indices]
X_tune_final = X_tune[:, feature_indices]
X_hold_final = X_hold[:, feature_indices]

# Fit scaler on training
scaler_final = RobustScaler()
X_train_scaled = scaler_final.fit_transform(X_train_final)
X_tune_scaled = scaler_final.transform(X_tune_final)
X_hold_scaled = scaler_final.transform(X_hold_final)

# Grid search for best k
print("\nTesting different k values for LOF...")
best_auprc = -1
best_k = None
best_model = None

for k in [5, 6, 7, 8, 10, 12, 15, 20, 25, 30]:
    lof = LocalOutlierFactor(n_neighbors=k, contamination='auto', novelty=True)
    lof.fit(X_train_scaled)

    scores = -lof.score_samples(X_tune_scaled)
    scores_norm = rankdata(scores) / len(scores)

    auprc = average_precision_score(y_tune, scores_norm)
    auroc = roc_auc_score(y_tune, scores_norm)

    print(f"k = {k:2d}: AUPRC = {auprc:.4f}, AUROC = {auroc:.4f}")

    if auprc > best_auprc:
        best_auprc = auprc
        best_k = k
        best_model = lof

print(f"\n>>> Best k = {best_k} with AUPRC = {best_auprc:.4f}")

# ============================================================================
# STEP 5: Final Evaluation on Holdout
# ============================================================================
print("\n[STEP 5] Final evaluation on holdout set...")

# Retrain with best hyperparameters
final_lof = LocalOutlierFactor(n_neighbors=best_k, contamination='auto', novelty=True)
final_lof.fit(X_train_scaled)

# Evaluate on holdout
hold_scores = -final_lof.score_samples(X_hold_scaled)
hold_scores_norm = rankdata(hold_scores) / len(hold_scores)

hold_auprc = average_precision_score(y_hold, hold_scores_norm)
hold_auroc = roc_auc_score(y_hold, hold_scores_norm)

print(f"\n{'='*80}")
print(f"FINAL RESULTS (Holdout Set)")
print(f"{'='*80}")
print(f"AUPRC: {hold_auprc:.4f}")
print(f"AUROC: {hold_auroc:.4f}")
print(f"Features used: {best_k_features}")
print(f"LOF k parameter: {best_k}")
print(f"{'='*80}")

# ============================================================================
# STEP 6: Generate Test Submission
# ============================================================================
print("\n[STEP 6] Generating test submission...")

try:
    test_df = pd.read_csv('Data/loans_test.csv')
    X_test, _ = fe.transform(test_df)
    X_test_subset = X_test[:, feature_indices]
    X_test_scaled = scaler_final.transform(X_test_subset)

    # Predict on test set
    test_scores = -final_lof.score_samples(X_test_scaled)
    test_scores_norm = rankdata(test_scores) / len(test_scores)

    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id_col = 'Id' if 'Id' in test_df.columns else 'index'
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        'anomaly_score': test_scores_norm
    })

    filename = f"OPTION1_LOF_AUPRC{hold_auprc:.4f}_AUC{hold_auroc:.4f}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    print(f"[+] Submission saved: {filename}")

except Exception as e:
    print(f"[-] Test prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("OPTION 1 COMPLETE")
print("="*80)
print("\nKey Points:")
print("[+] All models trained on TRAINING data only")
print("[+] Validation used ONLY for feature selection and hyperparameter tuning")
print("[+] NO supervised meta-learner")
print("[+] NO data leakage")
print("[+] Complies with professor's requirements")
