# Improved Anomaly Detection Approach

## Overview
This repository contains an improved anomaly detection ensemble that addresses critical issues in the original `final_ensemble t1.py` approach and implements proper anomaly detection methodologies based on academic literature.

## Key Issues Fixed

### 1. Validation Set Misuse (Critical Fix)
- **Problem**: Original code used validation set for training XGBoost calibrator (lines 254-279)
- **Impact**: Data leakage, validation doesn't simulate unseen data
- **Solution**: Validation set used ONLY for hyperparameter selection, not training

### 2. Limited Methodology
- **Problem**: Only used Isolation Forest (single approach)
- **Solution**: Implemented 4 complementary anomaly detection methods:
  - Statistical (Gaussian + Mahalanobis distance)
  - Proximity-based (LOF + k-NN distance)
  - Clustering-based (DBSCAN + Isolation Forest)
  - Reconstruction-based (PCA + SVD)

### 3. Semi-supervised Approach
- **Problem**: XGBoost calibrator required labels
- **Solution**: Pure unsupervised anomaly detection

## New Implementation: `improved_anomaly_ensemble.py`

### Architecture
```
Training Data → Preprocessing → Feature Engineering → Scaling
                                      ↓
    ┌─────────────────────── Ensemble Training ──────────────────────┐
    │                                                                  │
    ├─ Statistical     ├─ Proximity      ├─ Clustering    ├─ Reconstruction
    │  (Mahalanobis)   │  (LOF + k-NN)   │  (DBSCAN + IF) │  (PCA + SVD)
    │                  │                 │                │
    └─────────────────────── Weighted Combination ──────────────────┘
                                      ↓
                           Validation Set (hyperparameter tuning only)
                                      ↓
                              Final Anomaly Scores
```

### Key Classes

1. **StatisticalAnomalyDetector**: Gaussian distribution assumption with Mahalanobis distance
2. **ProximityAnomalyDetector**: LOF + k-NN distance for density-based detection
3. **ClusteringAnomalyDetector**: DBSCAN + Isolation Forest combination
4. **ReconstructionAnomalyDetector**: PCA + SVD reconstruction error
5. **ImprovedAnomalyEnsemble**: Main ensemble class orchestrating all methods

### Validation Usage
```python
def tune_hyperparameters(self, X_train, X_valid, y_valid):
    # Fit detector on TRAINING data only
    detector.fit(X_train)

    # Evaluate on validation set (labels used for evaluation only)
    scores = detector.score_samples(X_valid)
    auc = roc_auc_score(y_valid, scores)  # Evaluation only!
```

## Usage

```python
from improved_anomaly_ensemble import ImprovedAnomalyEnsemble

# Load data
train_df = pd.read_csv('Data/loans_train.csv')
valid_df = pd.read_csv('Data/loans_valid.csv')

# Train ensemble (validation used only for hyperparameter tuning)
model = ImprovedAnomalyEnsemble(random_state=42)
model.fit(train_df, valid_df)

# Evaluate
ap, auc = model.evaluate(valid_df)

# Predict on test set
test_scores = model.predict_proba(test_df)
```

## Expected Improvements

1. **Better Generalization**: No validation overfitting
2. **More Robust Detection**: Multiple complementary methods
3. **Diverse Anomaly Patterns**: Each method captures different anomaly types
4. **Proper ML Principles**: Clean train/validation separation
5. **Academic Foundation**: Based on established anomaly detection literature

## Files

- `improved_anomaly_ensemble.py`: Main implementation
- `approach_comparison.py`: Detailed comparison of old vs new approach
- `final_ensemble t1.py`: Original implementation (for reference)
- `README_IMPROVEMENTS.md`: This documentation

## Academic Foundation

Based on the four main categories of anomaly detection methods:

1. **Statistical Approaches**: Probabilistic models assuming normal distributions
2. **Proximity-Based**: Distance and density-based outlier detection
3. **Clustering-Based**: Points that don't belong to any cluster
4. **Reconstruction-Based**: High reconstruction error indicates anomalies

Each method captures different types of anomalies, making the ensemble more robust and comprehensive than single-method approaches.