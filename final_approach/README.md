# Final Approach: Ultra Unsupervised Ensemble

## Overview
Our final approach is an **ultra unsupervised ensemble** that combines multiple anomaly detection methods with sophisticated feature engineering and fusion strategies. It achieves **AUPRC=0.4524** on validation (0.43 on Kaggle), representing a **131% improvement** over the best baseline.

## Performance Summary

| Metric | Value | vs Best Baseline |
|--------|-------|------------------|
| **Validation AUPRC** | **0.4524** | +131% (0.1955 → 0.4524) |
| **Validation AUROC** | **0.7597** | +35% (0.5648 → 0.7597) |
| **Kaggle AUPRC** | **0.43** | +120% over baseline |

## Key Components

### 1. Feature Engineering (`feature_builder_advanced.py`)

#### Advanced Feature Builder
- **PCA Transformation**: Reduces 129 numeric features to 80 components (100% variance)
- **Robust Scaling**: RobustScaler handles outliers better than StandardScaler
- **Domain-Specific Features**:
  - **Amortization Features**: Payment ratios, balance changes, loan-to-value ratios
  - **Temporal Patterns**: Trends across 12-month history
  - **Statistical Aggregations**: Mean, median, std across time series

**Key Insight**: Domain features (especially amortization) provide the strongest signal for anomaly detection (AUPRC=0.4748 alone!).

### 2. Detector Ensemble (10 Selected Detectors)

#### Selected Detectors (AUPRC ≥ 0.16 threshold):
1. **Amortization Score** (AUPRC=0.4748) ⭐ **STRONGEST SIGNAL**
2. **LOF (k=6)** (AUPRC=0.3017)
3. **LOF (k=7)** (AUPRC=0.3016)
4. **LOF (k=5)** (AUPRC=0.3007)
5. **LOF (k=8)** (AUPRC=0.2994)
6. **LOF (k=10)** (AUPRC=0.2991)
7. **Cluster-wise LOF** (AUPRC=0.2972)
8. **LOF (k=12)** (AUPRC=0.2968)
9. **LOF (k=4)** (AUPRC=0.2966)
10. **Random Projection LOF** (AUPRC=0.2904)

#### Detector Details:

**LOF (Multi-k)**:
- K-values: {4, 5, 6, 7, 8, 10, 12}
- Captures anomalies at different density scales
- Trained on PCA-transformed features

**Cluster-wise LOF**:
- KMeans (n=12) clusters training data
- Per-cluster LOF models with optimized k
- Handles heterogeneous loan populations

**Random Projection LOF**:
- 40 random projections to 60D subspace
- LOF (k=7) on each projection
- MaxRank aggregation across projections

**Other Detectors** (not selected but tested):
- Isolation Forest, Elliptic Envelope, k-distance, PCA reconstruction, Mahalanobis distance

### 3. Calibration & Normalization

#### Train-CDF Calibration:
- Empirical CDF computed on **training scores only**
- Maps raw scores to [0,1] probability space
- Prevents validation leakage

#### Cohort Normalization:
- Z-score normalization within KMeans clusters
- Training cluster statistics applied to valid/test
- Handles heterogeneous subpopulations

### 4. Fusion Strategy

#### Tested Fusion Rules:
- **Rank-based**: max_rank, median_rank, wavg_rank, max_rank_top2, max_rank_top3, wavg_rank_top2
- **Probability-based**: p_avg, p_noisy_or, p_max
- **Cohort-normalized**: rankCoh::*, probCoh::*

#### Winner: `rank::wavg_rank_top2`
- Weighted average of rank-normalized scores
- Uses **top 2 detectors by AUPRC**: Amortization (0.4748) + LOF(k=6) (0.3017)
- Weights: Proportional to validation AUPRC

**Why this works**: Combines the strongest domain signal (amortization) with the best density-based detector (LOF), while avoiding overfitting to validation set.

## Methodology: Fully Unsupervised & Compliant

### Training Protocol (NO Leakage):
1. **Fit feature builder** on train set only (normal loans)
2. **Fit all detectors** on transformed train features only
3. **Compute calibration stats** (CDF, cluster means/stds) on train only
4. **Apply transformations** to validation/test (never fit)

### Validation Protocol:
- Use **validation performance** for:
  - Detector selection (AUPRC ≥ 0.16 threshold)
  - Fusion rule selection (best AUPRC)
  - Weight determination (proportional to AUPRC)
- **Never fit models on validation data**

### Competition Compliance:
✅ Train **only on normal data** (target=0)
✅ Validation used **only for hyperparameter tuning**
✅ No supervised learning or label leakage
✅ Metrics: AUPRC (primary), AUROC (tie-breaker)

## Files

### Core Scripts
- **`unsup_ultra_ensemble_fast_improvement.py`**: Main ensemble script
- **`feature_builder_advanced.py`**: Feature engineering pipeline

### Outputs
- **`SUB_ultra_unsup_fast_impr.csv`**: Kaggle submission file (test predictions)

## Algorithm Flow

```
1. Load Data
   ├── Train: 30,504 normal loans
   ├── Valid: 5,370 loans (12.61% anomalous)
   └── Test: 13,426 loans (unknown labels)

2. Feature Engineering (Train-only fit)
   ├── Drop missing features (ReliefRefinanceIndicator, PreHARP_Flag, SuperConformingFlag)
   ├── Robust scaling (RobustScaler)
   ├── PCA (80 components, 100% variance)
   └── Domain features (amortization ratios, temporal patterns)

3. Train Detectors (Train-only fit)
   ├── LOF (7 variants: k=4,5,6,7,8,10,12)
   ├── Cluster LOF (KMeans n=12, per-cluster LOF)
   ├── Random Projection LOF (40 bags, dim=60, k=7)
   ├── k-distance (k=3,5,7,9,11)
   ├── Isolation Forest (n=500, max_samples=1.0)
   ├── Elliptic Envelope (support=0.9)
   ├── PCA Reconstruction (n=80)
   └── Amortization score (domain-specific)

4. Calibration (Train-only stats)
   ├── CDF: Empirical cumulative distribution on train scores
   └── Cohort: Z-score stats per KMeans cluster

5. Validation Evaluation
   ├── Apply detectors to validation set
   ├── Compute AUPRC for each detector
   ├── Select top detectors (AUPRC ≥ 0.16)
   └── Test fusion rules → pick best

6. Final Prediction
   ├── Apply selected fusion rule (wavg_rank_top2)
   ├── Generate test predictions
   └── Save submission CSV

7. Results
   ├── Valid AUPRC: 0.4524
   ├── Valid AUROC: 0.7597
   └── Kaggle AUPRC: 0.43
```

## Why This Approach Works

### 1. **Domain Knowledge Integration**
The amortization score alone achieves AUPRC=0.4748, showing that domain-specific features are crucial for this task. Delinquent loans have distinctive payment patterns.

### 2. **Multi-Scale LOF Ensemble**
LOF with different k-values captures anomalies at various density scales. Some anomalies are isolated points (small k), others are small clusters (large k).

### 3. **Cluster-wise Modeling**
Loans are heterogeneous (different property types, loan amounts, regions). Cluster-wise LOF adapts to local subpopulations.

### 4. **Sophisticated Fusion**
Weighted rank aggregation of top-2 detectors balances the strongest signals while avoiding overfitting. Rank normalization handles different score scales.

### 5. **Train-only Calibration**
CDF and cohort normalization use **only training statistics**, preventing validation leakage while enabling probabilistic interpretation.

### 6. **Robust Preprocessing**
RobustScaler handles outliers (13.99% of PropertyValMethod values are outliers) better than StandardScaler.

## Comparison with Baselines

| Approach | AUPRC | AUROC | Key Idea |
|----------|-------|-------|----------|
| Isolation Forest | 0.1548 | 0.5273 | Tree-based isolation |
| LOF (k=50) | **0.1955** | 0.5648 | Best single baseline |
| Random Projection LOF | 0.1939 | 0.5630 | Subspace ensemble |
| One-Class SVM | 0.1848 | 0.5556 | Kernel-based |
| **Our Ensemble** | **0.4524** | **0.7597** | **Multi-detector + domain features** |

**Improvement**: +131% AUPRC, +35% AUROC over best baseline

## Running the Final Approach

### Requirements
```bash
pip install numpy pandas scikit-learn
```

### Execute
```bash
python final_approach/unsup_ultra_ensemble_fast_improvement.py
```

### Output
```
=== Ultra Unsupervised Ensemble Fast Improvement (Compliant) ===

--- Using PCA setting: 80 comps (from original winner) ---
Added FAST Mahalanobis detector (on scaled features).

[Per-detector AUPRC (Top 10)]:
  amort: 0.4748 ⭐
  lof_k6: 0.3017
  lof_k7: 0.3016
  ...

[Selected detectors]: ['amort', 'lof_k6', 'lof_k7', 'lof_k5', 'lof_k8',
                       'lof_k10', 'cluster_lof', 'lof_k12', 'lof_k4', 'rp_lof']

[Fusion best with Fast Improvement] Rule=rank::wavg_rank_top2
                                    AUPRC=0.4524  AUROC=0.7597

=== BEST VALID Fast Improvement ===
AUPRC=0.4524  AUROC=0.7597
Detectors used=['amort', 'lof_k6', ...]
Rule=rank::wavg_rank_top2

Saved SUB_ultra_unsup_fast_impr.csv
```

## Lessons Learned

### What Worked:
1. ✅ Domain-specific features (amortization) provide strongest signal
2. ✅ Multi-scale LOF ensemble captures diverse anomaly types
3. ✅ Cluster-wise modeling handles heterogeneity
4. ✅ Weighted top-2 fusion balances performance and simplicity
5. ✅ Robust preprocessing critical for outlier-heavy data

### What Didn't Work:
1. ❌ PCA dimensionality reduction (hurt baseline performance)
2. ❌ Isolation Forest (underperformed LOF)
3. ❌ Complex fusion rules (overfitting to validation)
4. ❌ Too many detectors (top-2 better than top-10 fusion)

### Future Improvements:
1. Deep learning autoencoders (LSTM, Transformer) for temporal patterns
2. Graph-based methods (loan networks, geographic clusters)
3. Semi-supervised learning (pseudo-labeling top predictions)
4. Adversarial validation (train/test distribution shift)
5. Stacking meta-learner (though risks overfitting)

## References

- **LOF**: Breunig et al. (2000) "LOF: Identifying Density-Based Local Outliers"
- **Random Subspace**: Ho (1998) "The Random Subspace Method"
- **Anomaly Ensemble**: Aggarwal (2017) "Outlier Ensembles"
- **Freddie Mac Dataset**: Single-Family Loan-Level Dataset

## Acknowledgments

This approach was developed for the CS5344 Big Data Analytics Technology course project, applying unsupervised machine learning to real-world loan delinquency prediction.

---

**Last Updated**: 2025-10-11
**Validation Performance**: AUPRC=0.4524, AUROC=0.7597
**Kaggle Performance**: AUPRC=0.43
**Improvement over Baseline**: +131% AUPRC
**Status**: ✅ Competition-Compliant, ✅ Leakage-Free, ✅ Fully Unsupervised
