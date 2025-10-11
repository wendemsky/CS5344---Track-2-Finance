# Baseline Models Evaluation

## Overview
This directory contains comprehensive baseline model evaluation for unsupervised anomaly detection on the loan delinquency dataset. We tested **10 different algorithms** across **3 preprocessing configurations** (27 total model variants).

## Contents

### Scripts
- **`run_all_baselines.py`**: Main script that trains and evaluates all baseline models
- **`compare_results.py`**: Generates comparison visualizations and statistical analysis

### Results (`results/`)
- **`baseline_results.csv`**: Complete results for all model configurations
- **`baseline_results.json`**: JSON format results
- **`summary_statistics.csv`**: Aggregated statistics by algorithm type
- **`best_per_algorithm.csv`**: Best configuration for each algorithm
- **Visualizations**: 4 comparison charts (see below)

## Baseline Models Tested

### 1. **Isolation Forest**
- **Type**: Ensemble, tree-based
- **Parameters**: n_estimators ∈ {100, 300, 500}, max_samples ∈ {256, 512, 'auto'}
- **Best AUPRC**: 0.1548 (n=500, samp=512)

### 2. **Local Outlier Factor (LOF)**
- **Type**: Density-based
- **Parameters**: k ∈ {5, 10, 20, 30, 50}
- **Best AUPRC**: **0.1955** (k=50) ⭐ **BEST OVERALL**

### 3. **One-Class SVM**
- **Type**: Kernel-based
- **Parameters**: kernel ∈ {rbf, linear}, nu ∈ {0.01, 0.05, 0.1}
- **Best AUPRC**: 0.1848 (rbf, nu=0.1)

### 4. **Elliptic Envelope**
- **Type**: Statistical, robust covariance
- **Parameters**: support_fraction ∈ {0.8, 0.9, 0.95}
- **Best AUPRC**: 0.1534 (support=0.8)
- **Note**: Best AUROC (0.6206) among all baselines

### 5. **MLP Autoencoder**
- **Type**: Neural network, reconstruction-based
- **Architecture**: 129 → 64 → 32 → 64 → 129
- **Status**: Skipped (TensorFlow not installed)

### 6. **DBSCAN**
- **Type**: Density clustering
- **Parameters**: eps ∈ {0.5, 1.0, 2.0}, min_samples ∈ {5, 10}
- **Best AUPRC**: 0.1807 (eps=1.0, min_samples=5)

### 7. **K-Nearest Neighbors Distance**
- **Type**: Distance-based
- **Parameters**: k ∈ {5, 10, 20, 30}
- **Best AUPRC**: 0.1882 (k=5)

### 8. **PCA Reconstruction Error**
- **Type**: Dimensionality reduction
- **Parameters**: n_components ∈ {20, 50, 80}
- **Best AUPRC**: 0.1809 (n=80)

### 9. **Random Projection LOF Ensemble**
- **Type**: Ensemble, subspace learning
- **Parameters**: B=20 projections, dim=50, k=10
- **Best AUPRC**: 0.1939 ⭐ **2nd BEST**

### 10. **Mahalanobis Distance**
- **Type**: Statistical distance
- **Best AUPRC**: 0.1738

## Top Results

### Top 10 Models by AUPRC

| Rank | Model | Config | AUPRC | AUROC | F1 |
|------|-------|--------|-------|-------|----|
| 1 | LOF(k=50) | robust_pcaNone | **0.1955** | 0.5648 | 0.2366 |
| 2 | RandomProjection_LOF(B=20) | robust_pcaNone | 0.1939 | 0.5630 | 0.2399 |
| 3 | KNN_Distance(k=5) | robust_pcaNone | 0.1882 | 0.5540 | 0.2357 |
| 4 | OneClassSVM(rbf, nu=0.1) | robust_pcaNone | 0.1848 | 0.5556 | 0.2360 |
| 5 | OneClassSVM(rbf, nu=0.1) | standard_pca80 | 0.1841 | 0.5375 | 0.2312 |
| 6 | LOF(k=5) | standard_pca80 | 0.1814 | 0.5395 | 0.2279 |
| 7 | RandomProjection_LOF(B=20) | standard_pca80 | 0.1813 | 0.5380 | 0.2287 |
| 8 | KNN_Distance(k=5) | standard_pca80 | 0.1809 | 0.5273 | 0.2281 |
| 9 | PCA_Reconstruction(n=80) | robust_pcaNone | 0.1809 | 0.5383 | 0.2265 |
| 10 | DBSCAN(eps=1.0, min_samples=5) | robust_pcaNone | 0.1807 | 0.5522 | 0.2337 |

### Key Insights

1. **LOF (k=50) is the best single baseline** with AUPRC=0.1955
2. **Random Projection LOF Ensemble is 2nd best** with AUPRC=0.1939
3. **Elliptic Envelope has highest AUROC** (0.6206) but lower AUPRC (0.1534)
4. **No PCA (robust_pcaNone) performs best** on average
5. **Robust scaling slightly outperforms standard scaling**

## Preprocessing Configuration Impact

| Configuration | Mean AUPRC | Std AUPRC | Max AUPRC | Mean AUROC |
|---------------|-----------|-----------|-----------|------------|
| **robust_pcaNone** | **0.1755** | 0.0212 | **0.1955** | **0.5537** |
| standard_pca80 | 0.1728 | 0.0137 | 0.1841 | 0.5398 |
| robust_pca80 | 0.1523 | 0.0097 | 0.1743 | 0.5443 |

**Recommendation**: Use **RobustScaler without PCA** for best performance.

## Algorithm Type Comparison

| Algorithm | Mean AUPRC | Std AUPRC | Best AUPRC | Avg Time (s) |
|-----------|------------|-----------|------------|--------------|
| **RandomProjection_LOF** | 0.1763 | 0.0067 | **0.1939** | 11.92 |
| **LOF** | 0.1773 | 0.0175 | **0.1955** | 4.69 |
| **PCA_Reconstruction** | 0.1783 | 0.0021 | 0.1809 | **0.07** ⚡ |
| **KNN_Distance** | 0.1721 | 0.0068 | 0.1882 | **0.71** ⚡ |
| **OneClassSVM** | 0.1713 | 0.0059 | 0.1848 | 66.94 🐌 |
| **DBSCAN** | 0.1653 | 0.0156 | 0.1807 | 0.43 |
| **Mahalanobis** | 0.1650 | 0.0049 | 0.1738 | **0.06** ⚡ |
| **EllipticEnvelope** | 0.1517 | 0.0009 | 0.1534 | 40.50 |
| **IsolationForest** | 0.1444 | 0.0044 | 0.1548 | 4.76 |

## Visualizations

### 1. Top 15 Models Comparison (`top15_comparison.png`)
Bar charts comparing the top 15 models across AUPRC, AUROC, and F1 metrics.

### 2. Algorithm Type Distributions (`algorithm_comparison_boxplots.png`)
Box plots showing performance distribution for each algorithm type.

### 3. Preprocessing Impact (`preprocessing_impact.png`)
Bar charts showing mean AUPRC and AUROC by preprocessing configuration.

### 4. AUPRC vs AUROC Scatter (`auprc_vs_auroc_scatter.png`)
Scatter plot of all models colored by preprocessing configuration, with top 3 models annotated.

## Comparison with Final Approach

| Approach | AUPRC | AUROC | Improvement |
|----------|-------|-------|-------------|
| **Best Baseline** (LOF k=50) | 0.1955 | 0.5648 | - |
| **Final Ensemble Approach** | **0.4524** | **0.7597** | **+131% AUPRC** |
| **Kaggle Leaderboard** | **0.43** | - | **+120% over baseline** |

Our final ensemble approach achieves **more than 2x the AUPRC** of the best baseline model!

## Key Takeaways

### What Worked Well:
1. **LOF-based methods** (LOF, Random Projection LOF) consistently top performers
2. **Distance-based methods** (KNN, Mahalanobis) are fast and effective
3. **Robust scaling without PCA** preserves important anomaly patterns
4. **Ensemble approaches** (Random Projection LOF) outperform single models

### What Didn't Work Well:
1. **Isolation Forest** underperformed (AUPRC=0.1548)
2. **PCA dimensionality reduction** generally hurt performance
3. **One-Class SVM** too slow for practical use (67s average)
4. **Elliptic Envelope** high AUROC but low AUPRC (imbalance sensitivity)

### Why Our Final Approach Works Better:
1. **Multiple LOF variants** with different k values
2. **Cluster-wise LOF** captures local patterns
3. **Domain-specific features** (amortization ratios)
4. **Sophisticated fusion** (weighted rank aggregation)
5. **Train-only calibration** prevents leakage
6. **Cohort normalization** handles heterogeneous loan populations

## Running the Baselines

### Requirements
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Execute
```bash
# Run all baselines (takes ~5-10 minutes)
python baseline_models/run_all_baselines.py

# Generate comparison visualizations
python baseline_models/compare_results.py
```

### Output
All results saved to `baseline_models/results/`:
- CSV and JSON result files
- 4 comparison visualizations
- Statistical summaries

## Next Steps

1. ✅ **Baseline Evaluation Complete**: Established performance benchmarks
2. ✅ **Best Baseline Identified**: LOF (k=50) with AUPRC=0.1955
3. → **Final Ensemble**: Combine best ideas (LOF, clustering, domain features)
4. → **Hyperparameter Tuning**: Optimize on validation set
5. → **Kaggle Submission**: Achieve 0.43 AUPRC (2x baseline improvement)

## References

- **LOF**: Breunig et al. (2000) "LOF: Identifying Density-Based Local Outliers"
- **Isolation Forest**: Liu et al. (2008) "Isolation Forest"
- **One-Class SVM**: Schölkopf et al. (2001) "Estimating the Support of a High-Dimensional Distribution"
- **Random Subspace**: Ho (1998) "The Random Subspace Method for Constructing Decision Forests"

---

**Last Updated**: 2025-10-11
**Total Models Evaluated**: 27 configurations across 10 algorithm types
**Best Baseline**: LOF (k=50), AUPRC=0.1955
**Final Approach**: Ensemble, AUPRC=0.4524 (+131% improvement)
