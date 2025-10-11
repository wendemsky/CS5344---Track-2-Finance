# Pure Unsupervised Anomaly Detection Approach Plan

## Problem Analysis

### Current Issues with meta_learner_stack.py:
1. **DATA LEAKAGE**: Uses LogisticRegression/RandomForest trained on validation labels (lines 122-136)
   - This violates the unsupervised requirement
   - The meta-learner is a supervised model trained on `yvtr` (validation targets)
   - Professor explicitly states: "you should not fit your model on the validation set"

2. **Cluster-LOF Leakage**: Uses validation labels to select best cluster (lines 38-39)
   - Uses `roc_auc_score(yvtr, d)` to pick best k-means clustering
   - This is indirect supervision on validation data

3. **Feature Selection Leakage**: Uses validation labels to gate features (lines 93-98)
   - Decides whether to include cluster_lof based on AUPRC on validation

### What t6v2.py Does RIGHT:
✅ Trains LOF only on training data (line 124)
✅ Uses validation only for hyperparameter selection (k parameter, lines 114-119)
✅ No supervised meta-learner
✅ Clean separation of train/validation

---

## New Pure Unsupervised Approach

### Core Principle:
**Train ONLY on normal training data → Use validation ONLY for hyperparameter tuning → No supervised models**

### Architecture: Multi-Algorithm Unsupervised Ensemble

```
TRAINING DATA (normal only)
         ↓
    ┌────────────────────────────────────────┐
    │   Base Unsupervised Detectors          │
    │   (trained on normal data only)        │
    ├────────────────────────────────────────┤
    │ 1. LOF (k=5,8,10,12,15,20)            │
    │ 2. Isolation Forest (various params)   │
    │ 3. One-Class SVM                       │
    │ 4. Local Density Ratio                 │
    │ 5. Mahalanobis Distance                │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │   Unsupervised Combination             │
    │   (NO supervised meta-learner)         │
    ├────────────────────────────────────────┤
    │ - Rank averaging                       │
    │ - Weighted average (weights tuned      │
    │   on validation via grid search)       │
    │ - Max/Min combination                  │
    └────────────────────────────────────────┘
         ↓
    VALIDATION SET (for hyperparameter tuning)
         ↓
    TEST SET (final predictions)
```

---

## Detailed Implementation Plan

### Phase 1: Feature Engineering (NO Validation Statistics)

**ALLOWED:**
✅ Calculate statistics from TRAINING data only
✅ Domain-driven features (payment trends, credit risk indicators)
✅ Temporal aggregations (mean, std, slope over months)
✅ Apply same transformations to validation/test

**NOT ALLOWED:**
❌ Use validation target to select features
❌ Calculate scalers/imputers on validation set
❌ Use validation statistics for any transformations

**Features to Engineer:**
1. **Credit Risk Indicators** (from training):
   - CreditScore (already high correlation r=-0.25)
   - DTI ratio
   - LTV ratio
   - Combined risk score: `(700 - CreditScore)/100 + LTV/100 + DTI/50`

2. **Temporal Payment Patterns** (14 months of data):
   - Trend in CurrentUPB (increasing = trouble)
   - Trend in NonInterestBearingUPB
   - Volatility in payment amounts
   - Number of late payments (inferred from UPB changes)
   - Early delinquency indicators (first 3 months vs last 3 months)

3. **Loan Characteristics**:
   - Original loan amount
   - Interest rate
   - Loan term
   - Property type, occupancy

4. **Derived Features**:
   - Payment-to-income ratio evolution
   - Equity position changes
   - Days since origination

**Implementation:**
```python
class FeatureEngineer:
    def fit(self, train_df):
        # Fit scalers, imputers, encoders on TRAINING ONLY
        self.scaler = RobustScaler().fit(train_features)
        self.imputer = median values from training

    def transform(self, df):
        # Apply fitted transformations
        # Calculate temporal trends
        # Engineer domain features
        return features
```

### Phase 2: Base Unsupervised Detectors

Train multiple detectors on TRAINING data only:

**1. Local Outlier Factor (LOF) - Multiple Configurations**
```python
lof_configs = [
    {'n_neighbors': 5, 'contamination': 'auto'},
    {'n_neighbors': 8, 'contamination': 'auto'},
    {'n_neighbors': 10, 'contamination': 'auto'},
    {'n_neighbors': 12, 'contamination': 'auto'},
    {'n_neighbors': 15, 'contamination': 'auto'},
    {'n_neighbors': 20, 'contamination': 'auto'},
]
# Train each on Xtr, score on validation for hyperparameter selection
```

**2. Isolation Forest - Multiple Configurations**
```python
iforest_configs = [
    {'n_estimators': 100, 'max_samples': 256, 'contamination': 'auto'},
    {'n_estimators': 200, 'max_samples': 256, 'contamination': 'auto'},
    {'n_estimators': 300, 'max_samples': 512, 'contamination': 'auto'},
]
```

**3. One-Class SVM**
```python
ocsvm_configs = [
    {'kernel': 'rbf', 'nu': 0.1, 'gamma': 'scale'},
    {'kernel': 'rbf', 'nu': 0.05, 'gamma': 'auto'},
]
```

**4. Mahalanobis Distance**
```python
# Compute covariance from training data
cov = np.cov(Xtr.T)
inv_cov = np.linalg.pinv(cov)
mean = Xtr.mean(axis=0)
# Distance = sqrt((x - mean)^T * inv_cov * (x - mean))
```

**5. Local Density Ratio**
```python
# For each point, compute ratio of k-NN distance to average k-NN distance in training
```

### Phase 3: Hyperparameter Tuning on Validation (ALLOWED)

**Use validation to select:**
- Best k for LOF
- Best n_estimators for Isolation Forest
- Best nu for One-Class SVM
- Best feature subset (by evaluating different subsets)

**Grid Search Framework:**
```python
best_auprc = -1
best_config = None

for config in all_configs:
    detector = train_detector(Xtr, config)  # Train on training only
    scores = detector.score_samples(Xval)   # Evaluate on validation
    auprc = average_precision_score(yval, scores)

    if auprc > best_auprc:
        best_auprc = auprc
        best_config = config

# Retrain with best config
final_detector = train_detector(Xtr, best_config)
```

### Phase 4: Unsupervised Ensemble Combination

**NO supervised meta-learner (LogisticRegression/RandomForest)**

**Option A: Rank-Based Averaging (Simple & Effective)**
```python
# For each detector, convert scores to ranks [0,1]
def rank_normalize(scores):
    return (rankdata(scores) - 1) / (len(scores) - 1)

# Average ranks
ensemble_score = np.mean([rank_normalize(det.scores) for det in detectors], axis=0)
```

**Option B: Weighted Average (Weights Tuned on Validation)**
```python
# Find optimal weights via grid search on validation
# Constraint: weights sum to 1, all non-negative

from scipy.optimize import minimize

def objective(weights):
    combined = sum(w * rank_normalize(scores[i]) for i, w in enumerate(weights))
    return -average_precision_score(yval, combined)

# Optimize weights on validation
result = minimize(objective,
                  x0=np.ones(n_detectors)/n_detectors,
                  constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                  bounds=[(0, 1)] * n_detectors)

optimal_weights = result.x
```

**Option C: Max/Min Voting**
```python
# Max: Take maximum anomaly score across detectors
ensemble_score = np.max([rank_normalize(det.scores) for det in detectors], axis=0)

# Min: Take minimum (more conservative)
ensemble_score = np.min([rank_normalize(det.scores) for det in detectors], axis=0)
```

**Option D: Top-k Voting**
```python
# Average top-3 performing detectors (based on validation AUPRC)
top_k = sorted(detectors, key=lambda d: d.val_auprc, reverse=True)[:3]
ensemble_score = np.mean([rank_normalize(d.scores) for d in top_k], axis=0)
```

### Phase 5: Feature Subset Selection (Validation-Guided)

**Strategy: Test different feature groups, select best by validation AUPRC**

```python
feature_groups = {
    'all_features': all_feature_indices,
    'credit_only': [indices of CreditScore, DTI, LTV, etc.],
    'temporal_only': [indices of payment trends, UPB changes],
    'top_30_correlation': [top 30 features by abs correlation with target on validation],
    'high_variance': [features with variance > 0.01 on training],
    'domain_expert': [hand-picked features based on domain knowledge],
}

for group_name, indices in feature_groups.items():
    Xtr_subset = Xtr[:, indices]
    Xval_subset = Xval[:, indices]

    lof = LOF(k=10).fit(Xtr_subset)
    scores = lof.score_samples(Xval_subset)
    auprc = average_precision_score(yval, scores)

    print(f"{group_name}: AUPRC = {auprc:.4f}")
```

---

## Expected Performance

Based on correlation analysis:
- **Top 30 correlation features + LOF**: AUPRC = 0.2033 (already achieved)
- **Top 30 MI features + LOF**: AUPRC = 0.1961
- **Ensemble of 5-6 detectors**: AUPRC = 0.21-0.23 (estimated)

**Goal: AUPRC > 0.20 with pure unsupervised approach**

---

## Implementation Checklist

### ✅ What's ALLOWED:
1. Train detectors on training data only
2. Use validation AUPRC to select hyperparameters
3. Use validation AUPRC to select feature subsets
4. Use validation AUPRC to select detector weights
5. Feature engineering using training statistics only
6. Unsupervised combination methods (rank averaging, weighted voting)

### ❌ What's FORBIDDEN:
1. Training supervised models (LR, RF, XGBoost) on validation labels
2. Using validation labels for anything other than scoring/evaluation
3. Fitting scalers/imputers/encoders on validation data
4. Using validation statistics in feature engineering
5. Including validation samples in any fit() call

---

## Code Structure

```
unsupervised_anomaly_detection.py
├── FeatureEngineer (fits on train only)
├── BaseDetector (LOF, IForest, OCSVM, etc.)
├── HyperparameterTuner (uses validation for selection)
├── UnsupervisedEnsemble (rank averaging, weighted voting)
└── main()
    ├── Load data
    ├── Engineer features (fit on train)
    ├── Train base detectors on training data
    ├── Tune hyperparameters using validation
    ├── Combine predictions (unsupervised)
    ├── Generate test submission
```

---

## Key Differences from Previous Approach

| Aspect | Previous (meta_learner) | New (Pure Unsupervised) |
|--------|-------------------------|-------------------------|
| Meta-learner | LogisticRegression on validation labels | Rank averaging / weighted voting |
| Feature selection | Uses validation AUPRC to gate features | Uses validation AUPRC to select subset |
| Clustering | K-Means tuned by validation AUC | Optional: K-Means for stratified LOF (no label-based selection) |
| Training data | Training only ✅ | Training only ✅ |
| Validation use | Supervised meta-training ❌ | Hyperparameter tuning only ✅ |
| Supervised models | Yes ❌ | No ✅ |

---

## Timeline

1. **Day 1**: Implement FeatureEngineer + base detectors
2. **Day 2**: Implement hyperparameter tuning + feature selection
3. **Day 3**: Implement unsupervised ensemble + test on validation
4. **Day 4**: Optimize, generate test submission, iterate

---

## Success Criteria

✅ **No supervised models trained on validation**
✅ **All model fitting uses training data only**
✅ **Validation used only for hyperparameter selection**
✅ **AUPRC > 0.20 on validation holdout**
✅ **Code passes professor's review for fairness**

---

## References

- Your t6v2.py: AUPRC = 0.2018 (clean unsupervised baseline)
- Correlation analysis: Top 30 features → AUPRC = 0.2033
- Professor's guidelines: Pure unsupervised, validation for tuning only
