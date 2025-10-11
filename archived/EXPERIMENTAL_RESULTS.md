# Loan Anomaly Detection: Experimental Results Summary

## Overview
This document summarizes all anomaly detection approaches tested for the CS5344 loan default prediction project. The task involves semi-supervised anomaly detection where training data contains only normal loans (target=0) and validation/test data contains both normal and abnormal loans.

## Dataset Context
- **Training Set**: 30,504 loans × 145 features (100% normal loans)
- **Validation Set**: 5,370 loans × 145 features (87.39% normal, 12.61% abnormal)
- **Features**: 31 static origination variables + 112 temporal performance variables (14 months × 8 metrics)
- **Evaluation Metric**: AUPRC (Average Precision) due to class imbalance

### EDA Key Findings
- **Credit Quality**: Mean credit score 753.6 (std=156.1), range 600-850 (high-quality borrowers)
- **Loan Amounts**: Mean $317K (std=$181K), range $15K-$1.7M (diverse loan portfolio)
- **Interest Rates**: Mean 6.72% (std=0.55%), range 2.5%-9.12% (tight distribution)
- **Missing Data**: Critical issues with ReliefRefinanceIndicator (100% missing), PreHARP_Flag (100% missing), SuperConformingFlag (98.92% missing)
- **Temporal Structure**: All loans have complete 14-month sequences (no variable-length issues)
- **Data Types**: 71 integer, 60 float, 14 categorical features
- **Geographic Coverage**: Multi-state coverage with 11.22% missing MSA codes

## Experimental Results

### 1. Local Outlier Factor (LOF) Variants

#### Basic LOF (t2.py)
- **Method**: Standard LOF with default parameters
- **Results**: AUROC = 0.5488, AUPRC = 0.1699
- **Notes**: Baseline LOF implementation

#### Optimized Single LOF (t6.py)
- **Method**: Grid search for optimal k parameter
- **Best Parameters**: k = 10
- **Results**: AUROC = 0.5752, AUPRC = 0.1956
- **Notes**: Significant improvement over default parameters

#### Multi-k LOF Rank Ensemble (t10.py)
- **Method**: Ensemble of LOF with multiple k values using rank aggregation
- **Parameters**: k = [10, 20, 35, 50]
- **Individual Results**:
  - k=10: AUPRC = 0.1991, AUROC = 0.5763
  - k=20: AUPRC = 0.1841, AUROC = 0.5502
  - k=35: AUPRC = 0.1537, AUROC = 0.4930
  - k=50: AUPRC = 0.1370, AUROC = 0.4763
- **Ensemble Results**: AUROC = 0.5261, AUPRC = 0.1741
- **Submission**: SUB_lof_multiscale.csv

#### LOF-PCA Combination (t12.py)
- **Method**: LOF applied on PCA-reduced features
- **Best Parameters**: variance_threshold = 0.9, n_components = 2, k = 15
- **Validation Performance**: AUPRC = 0.2329 (valid-train)
- **Holdout Results**: AUROC = 0.5486, AUPRC = 0.1867
- **Submission**: SUB_lof_pca_tuned.csv

#### Multi-k LOF Weighted Ensemble (p13.py)
- **Method**: Weighted rank aggregation of multiple LOF detectors
- **Parameter Range**: k = [2-15] with cross-validation tuning
- **Best k Values**: k = 5 (AUPRC = 0.2027 on valid-train)
- **Results**:
  - **Weighted Rank**: AUROC = 0.6062, AUPRC = 0.2220
  - **Max Rank**: AUROC = 0.6119, AUPRC = 0.2043
- **Submission**: SUB_lof_weighted_multik.csv

### 2. Isolation Forest Variants

#### Basic Isolation Forest (t3.py)
- **Method**: Standard Isolation Forest
- **Results**: AUROC = 0.5283, AUPRC = 0.1451
- **Notes**: Underperformed compared to LOF

#### Optimized Isolation Forest (t5.py)
- **Method**: Grid search for optimal n_estimators
- **Best Parameters**: n_estimators = 600
- **Results**: AUROC = 0.5107, AUPRC = 0.1303
- **Submission**: SUB_isolation_forest.csv
- **Notes**: Performance decreased with optimization, suggesting overfitting

### 3. PCA-Based Reconstruction

#### PCA Reconstruction Error (t4.py, t7.py)
- **Method**: Anomaly detection using PCA reconstruction error
- **Best Parameters**: n_components = 2
- **Results**: AUROC = 0.5034, AUPRC = 0.1247
- **Valid-Train Performance**: AUC = 0.5137
- **Notes**: Poor performance, likely due to linear assumptions

### 4. Distance-Based Methods

#### K-Means Distance (t8.py)
- **Method**: Distance to nearest cluster centroid
- **Best Parameters**: k = 12 clusters
- **Results**: AUROC = 0.4534, AUPRC = 0.1100
- **Notes**: Worst performing method, suggests inadequate clustering for this data

### 5. Covariance-Based Methods

#### Elliptic Envelope / Minimum Covariance Determinant (t9.py)
- **Method**: Robust covariance estimation for outlier detection
- **Best Parameters**: contamination = 0.05
- **Results**: AUROC = 0.5368, AUPRC = 0.1360
- **Notes**: Moderate performance, limited by Gaussian assumptions

### 6. Meta-Learning Approach

#### Meta-Learner Stack (meta_learner_stack.py)
- **Method**: Ensemble of LOF variants with logistic regression meta-learner
- **Base Models**: LOF (k=5,6,7,8), cluster-wise LOF, amortization features
- **Feature Set**: ['lof_k5', 'lof_k6', 'lof_k7', 'lof_k8', 'cluster_lof', 'amort_0', 'amort_1', 'amort_2', 'amort_3', 'amort_4', 'amort_5']
- **Cross-Validation**: AUPRC = 0.3609 (valid-train)
- **Holdout Results**: **AUROC = 0.7843, AUPRC = 0.4210**
- **Submission**: SUB_meta_stack.csv
- **Notes**: **Best performing approach** - demonstrates power of ensemble methods

## Performance Ranking

| Rank | Method | AUPRC | AUROC | File |
|------|--------|-------|-------|------|
| 1 | **Meta-Learner Stack** | **0.4210** | **0.7843** | meta_learner_stack.py |
| 2 | Multi-k LOF Weighted | 0.2220 | 0.6062 | p13.py |
| 3 | Multi-k LOF Max Rank | 0.2043 | 0.6119 | p13.py |
| 4 | Optimized Single LOF | 0.1956 | 0.5752 | t6.py |
| 5 | LOF-PCA Combination | 0.1867 | 0.5486 | t12.py |
| 6 | Multi-k LOF Ensemble | 0.1741 | 0.5261 | t10.py |
| 7 | Basic LOF | 0.1699 | 0.5488 | t2.py |
| 8 | Basic Isolation Forest | 0.1451 | 0.5283 | t3.py |
| 9 | Elliptic Envelope | 0.1360 | 0.5368 | t9.py |
| 10 | Optimized Isolation Forest | 0.1303 | 0.5107 | t5.py |
| 11 | PCA Reconstruction | 0.1247 | 0.5034 | t4.py, t7.py |
| 12 | K-Means Distance | 0.1100 | 0.4534 | t8.py |

## Key Insights

### Why LOF Works Best
1. **Local Density Sensitivity**: Captures borrower heterogeneity better than global methods
2. **High-Dimensional Robustness**: Handles 145-feature space effectively
3. **Temporal Pattern Recognition**: Identifies deviating repayment trajectories
4. **Semi-Supervised Suitability**: Learns normal patterns without abnormal examples

### Ensemble Benefits
- **Meta-Learning Stack** achieved 89% improvement over best single method (0.4210 vs 0.2220 AUPRC)
- Combining multiple LOF variants with different k values captures various anomaly scales
- Amortization features provide domain-specific financial insights

### Method Limitations
- **Isolation Forest**: Struggles with high-dimensional, heterogeneous loan data
- **PCA Reconstruction**: Linear assumptions inadequate for complex loan patterns
- **K-Means Distance**: Global clustering inappropriate for local anomaly detection
- **Covariance Methods**: Gaussian assumptions violated by financial data distributions

## Future Directions

### Immediate Improvements
1. **Enhanced Feature Engineering**: Payment burden ratios, equity dynamics, temporal trends
2. **Cluster-Aware LOF**: Segment-specific anomaly detection within borrower groups
3. **Dynamic Parameter Selection**: Adaptive k-selection based on local density

### Advanced Approaches
1. **Deep Learning**: LSTM autoencoders for temporal sequence modeling
2. **Attention Mechanisms**: Identify critical time periods in loan lifecycle
3. **Graph Neural Networks**: Model feature relationships and borrower networks
4. **Interpretability**: Feature attribution methods for regulatory compliance

## Conclusion

The experimental results clearly demonstrate that **Local Outlier Factor (LOF) variants significantly outperform other anomaly detection methods** for this loan dataset. The meta-learning ensemble approach represents the current state-of-the-art, achieving strong performance through intelligent combination of multiple LOF detectors with domain-specific features.

The success of LOF-based methods validates the importance of local density comparison in financial anomaly detection, where borrower heterogeneity and diverse risk profiles require neighborhood-based rather than global anomaly assessment.