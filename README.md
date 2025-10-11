# Loan Anomaly Detection Project

**CS5344: Big Data Analytics Technology**
**Track 2: Finance - Loan Delinquency Prediction**

## Project Overview

This project implements an **unsupervised anomaly detection system** for identifying delinquent loans in the Freddie Mac Single-Family Loan-Level Dataset. Our ensemble approach achieves **AUPRC=0.4524** on validation (0.43 on Kaggle), representing a **131% improvement** over the best baseline model.

### Key Results

| Metric | Best Baseline | Our Approach | Improvement |
|--------|--------------|--------------|-------------|
| **Validation AUPRC** | 0.1955 (LOF) | **0.4524** | **+131%** |
| **Validation AUROC** | 0.5648 (LOF) | **0.7597** | **+35%** |
| **Kaggle AUPRC** | ~0.18 | **0.43** | **+139%** |

## Repository Structure

```
Project/
├── Data/                          # Dataset (train, valid, test)
│   ├── loans_train.csv           # 30,504 normal loans
│   ├── loans_valid.csv           # 5,370 loans (12.61% anomalous)
│   └── loans_test.csv            # 13,426 loans (unknown labels)
│
├── eda/                          # Exploratory Data Analysis
│   ├── comprehensive_eda.py      # Main EDA script
│   ├── outputs/                  # EDA results & visualizations
│   └── README.md                 # EDA documentation & findings
│
├── baseline_models/              # Baseline Model Evaluation
│   ├── run_all_baselines.py     # Tests 10 algorithms
│   ├── compare_results.py        # Comparison analysis
│   ├── results/                  # Baseline results & plots
│   └── README.md                 # Baseline documentation
│
├── final_approach/               # Final Ensemble Method
│   ├── unsup_ultra_ensemble_fast_improvement.py  # Main script
│   ├── feature_builder_advanced.py              # Feature engineering
│   └── README.md                                 # Methodology docs
│
├── archived/                     # Old experimental files
├── venv/                        # Python virtual environment
└── README.md                    # This file
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### 2. Run EDA
```bash
python eda/comprehensive_eda.py
```
**Output**: Statistical analysis + 10 visualizations in `eda/outputs/`

### 3. Run Baseline Models
```bash
python baseline_models/run_all_baselines.py
python baseline_models/compare_results.py
```
**Output**: 27 model evaluations + 4 comparison charts in `baseline_models/results/`

### 4. Run Final Approach
```bash
python final_approach/unsup_ultra_ensemble_fast_improvement.py
```
**Output**: `SUB_ultra_unsup_fast_impr.csv` (Kaggle submission)

## Project Phases

### Phase 1: Exploratory Data Analysis ✅

**Location**: [`eda/`](eda/)

**Key Findings**:
- **Dataset**: 30,504 train (100% normal), 5,370 valid (12.61% anomalous), 13,426 test
- **Features**: 143 total (129 numeric, 14 categorical)
- **Class Imbalance**: Highly imbalanced (12.61% anomaly rate in validation)
- **Missing Values**: 3 features with >99% missing (dropped)
- **Outliers**: 13.99% in PropertyValMethod → use RobustScaler
- **Top Features**: Amortization, Credit Score, DTI, LTV ratios

**Recommendations**:
- Use unsupervised methods (train has no anomalies)
- Apply robust scaling (outlier-heavy)
- Engineer domain-specific features (amortization patterns)
- Evaluate with AUPRC (handles imbalance)

### Phase 2: Baseline Model Evaluation ✅

**Location**: [`baseline_models/`](baseline_models/)

**Models Tested** (10 algorithms, 27 configurations):
1. **Isolation Forest** (AUPRC=0.1548)
2. **Local Outlier Factor** (AUPRC=**0.1955**) ⭐ **Best Baseline**
3. **One-Class SVM** (AUPRC=0.1848)
4. **Elliptic Envelope** (AUPRC=0.1534, AUROC=0.6206)
5. **MLP Autoencoder** (Skipped - TensorFlow optional)
6. **DBSCAN** (AUPRC=0.1807)
7. **KNN Distance** (AUPRC=0.1882)
8. **PCA Reconstruction** (AUPRC=0.1809)
9. **Random Projection LOF** (AUPRC=0.1939) ⭐ **2nd Best**
10. **Mahalanobis Distance** (AUPRC=0.1738)

**Preprocessing Comparison**:
- **Best**: RobustScaler without PCA (AUPRC=0.1755 avg)
- Standard scaling with PCA-80 (AUPRC=0.1728 avg)
- Robust scaling with PCA-80 (AUPRC=0.1523 avg)

**Key Insights**:
- LOF-based methods dominate (LOF, RP-LOF)
- No PCA performs best (preserves anomaly patterns)
- Robust scaling critical for outlier handling
- Ensemble approaches (RP-LOF) outperform single models

### Phase 3: Final Ensemble Approach ✅

**Location**: [`final_approach/`](final_approach/)

**Architecture**:
```
Input Features (143)
    ↓
Feature Engineering
    ├── Robust Scaling
    ├── PCA (80 components)
    └── Domain Features (amortization, temporal)
    ↓
10 Selected Detectors (AUPRC ≥ 0.16)
    ├── Amortization Score (AUPRC=0.4748) ⭐ STRONGEST
    ├── LOF (k=4,5,6,7,8,10,12) × 7 variants
    ├── Cluster-wise LOF (KMeans n=12)
    └── Random Projection LOF (40 bags)
    ↓
Calibration (Train-only)
    ├── CDF (empirical cumulative distribution)
    └── Cohort Normalization (cluster z-scores)
    ↓
Fusion (rank::wavg_rank_top2)
    └── Weighted avg of top-2 detectors
    ↓
Final Score (AUPRC=0.4524, AUROC=0.7597)
```

**Why It Works**:
1. **Domain Features**: Amortization score alone → AUPRC=0.4748
2. **Multi-Scale LOF**: Captures anomalies at different density levels
3. **Cluster-wise Modeling**: Handles heterogeneous loan populations
4. **Smart Fusion**: Top-2 weighted rank (amort + LOF k=6)
5. **Train-only Calibration**: No validation leakage

## Competition Compliance

✅ **Fully Unsupervised**: Train ONLY on normal data (target=0)
✅ **No Leakage**: Models never fit on validation/test sets
✅ **Validation for Tuning**: Hyperparameters & model selection only
✅ **Metrics**: AUPRC (primary), AUROC (secondary), F1 (optional)

**Verification**: All `.fit()` calls use ONLY training data. Validation labels used ONLY for evaluation metrics.

## Results Summary

### Validation Set Performance

| Model | AUPRC | AUROC | F1 |
|-------|-------|-------|----|
| Isolation Forest | 0.1548 | 0.5273 | 0.2304 |
| LOF (k=50) | 0.1955 | 0.5648 | 0.2366 |
| Random Projection LOF | 0.1939 | 0.5630 | 0.2399 |
| **Our Ensemble** | **0.4524** | **0.7597** | - |

### Kaggle Leaderboard

- **Our Submission**: AUPRC = **0.43**
- **Improvement**: +120% vs best baseline (~0.18 estimated)

### Per-Detector Performance (Our Approach)

| Detector | AUPRC | AUROC |
|----------|-------|-------|
| **Amortization Score** | **0.4748** | **0.7524** |
| LOF (k=6) | 0.3017 | 0.6603 |
| LOF (k=7) | 0.3016 | 0.6631 |
| LOF (k=5) | 0.3007 | 0.6593 |
| Random Projection LOF | 0.2904 | 0.6700 |

## Key Contributions

1. **Domain Feature Engineering**: Amortization-based features provide the strongest anomaly signal (AUPRC=0.4748)
2. **Multi-Scale LOF Ensemble**: 7 LOF variants capture anomalies at different density scales
3. **Cluster-wise Modeling**: Per-cluster LOF adapts to heterogeneous loan subpopulations
4. **Leakage-Free Calibration**: Train-only CDF and cohort normalization prevent overfitting
5. **Smart Fusion**: Top-2 weighted rank balances performance and generalization

## Visualizations

### EDA Outputs (`eda/outputs/`)
- Class distribution analysis
- Missing value patterns
- Top 20 numeric feature distributions
- Correlation heatmaps
- Outlier analysis
- Temporal patterns

### Baseline Comparisons (`baseline_models/results/`)
- Top 15 models bar chart (AUPRC, AUROC, F1)
- Algorithm type box plots
- Preprocessing configuration impact
- AUPRC vs AUROC scatter plot

## Lessons Learned

### What Worked:
✅ Domain-specific features (amortization ratios)
✅ LOF-based methods (multiple k values)
✅ Robust preprocessing (RobustScaler, no PCA)
✅ Ensemble fusion (top-2 weighted rank)
✅ Train-only calibration (leakage-free)

### What Didn't Work:
❌ PCA dimensionality reduction (hurt performance)
❌ Isolation Forest (underperformed LOF)
❌ One-Class SVM (too slow, 67s avg)
❌ Complex fusion rules (overfitting)
❌ Too many detectors in fusion (top-2 > top-10)

## Future Improvements

1. **Deep Learning**: LSTM/Transformer autoencoders for temporal patterns
2. **Graph Methods**: Loan networks, geographic clustering
3. **Semi-Supervised**: Pseudo-labeling high-confidence predictions
4. **Adversarial Validation**: Handle train/test distribution shift
5. **Feature Selection**: Automated feature importance ranking

## References

### Papers
- **LOF**: Breunig et al. (2000) "LOF: Identifying Density-Based Local Outliers"
- **Isolation Forest**: Liu et al. (2008) "Isolation Forest"
- **One-Class SVM**: Schölkopf et al. (2001) "Estimating the Support of a High-Dimensional Distribution"
- **Anomaly Ensembles**: Aggarwal (2017) "Outlier Ensembles: Position Paper"

### Dataset
- **Freddie Mac**: Single-Family Loan-Level Dataset
- **Anomaly Definition**: CURRENT LOAN DELINQUENCY STATUS ≠ 0
- **Competition**: Kaggle InClass Competition (CS5344)

## Team

**Course**: CS5344 - Big Data Analytics Technology
**Institution**: National University of Singapore (NUS)
**Semester**: AY2024/25 Semester 2

## License

This project is for educational purposes as part of the CS5344 course.

---

**Last Updated**: 2025-10-11
**Status**: ✅ Complete
**Performance**: AUPRC=0.4524 (validation), 0.43 (Kaggle)
**Improvement**: +131% over best baseline
