# Exploratory Data Analysis (EDA)

## Overview
This directory contains comprehensive exploratory data analysis for the Loan Anomaly Detection project. The EDA provides insights into data distributions, feature characteristics, and recommendations for modeling.

## Contents

### Scripts
- **`comprehensive_eda.py`**: Main EDA script that generates statistical analysis and visualizations
- **`eda_loan_analysis.py`**: Original EDA analysis (archived reference)
- **`proposal_visualizations.py`**: Visualizations for project proposal

### Outputs (`outputs/`)
All generated analysis outputs are stored in the `outputs/` directory:

#### Data Files
- **`basic_statistics.json`**: Dataset statistics (sizes, class distributions, feature counts)
- **`ks_test_results.csv`**: Kolmogorov-Smirnov test results for feature distributions
- **`correlation_with_target.csv`**: Feature correlations with target variable
- **`eda_summary_report.txt`**: Comprehensive text summary of findings

#### Visualizations
- **`class_distribution.png`**: Class balance across train/validation sets
- **`missing_values_analysis.png`**: Missing value patterns across datasets
- **`numeric_distributions.png`**: Top 20 numeric feature distributions (Normal vs Anomaly)
- **`correlation_with_target.png`**: Bar chart of top 25 feature correlations
- **`correlation_heatmap.png`**: Correlation matrix of top 15 features
- **`outlier_analysis.png`**: Outlier percentages by feature (IQR method)
- **`temporal_analysis.png`**: Temporal feature patterns by class

### Proposal Materials (`proposal_plots/`)
Visualizations created for the project proposal

## Key Findings

### 1. Dataset Characteristics

| Dataset | Size | Normal Loans | Anomalous Loans | Anomaly Rate |
|---------|------|--------------|-----------------|--------------|
| **Train** | 30,504 | 30,504 (100%) | 0 (0%) | 0.00% |
| **Validation** | 5,370 | 4,693 (87.4%) | 677 (12.6%) | 12.61% |
| **Test** | 13,426 | Unknown | Unknown | Unknown |

**Key Insight**: Training set contains **only normal loans**, making this a **pure unsupervised learning problem**. Models must learn patterns of normality from the training data alone.

### 2. Feature Analysis

- **Total Features**: 143 (129 numeric, 14 categorical)
- **High Variance Features**: Principal UPB (Unpaid Balance) features dominate variance
  - `OriginalUPB`, `0_CurrentActualUPB`, `1_CurrentActualUPB`, etc.
- **Missing Values**:
  - 100% missing: `ReliefRefinanceIndicator`, `PreHARP_Flag`
  - ~99% missing: `SuperConformingFlag`
  - ~11% missing: `MSA`

### 3. Feature Distributions

**Top Features by Variance** (represent UPB amounts in billions):
1. `OriginalUPB`: $3.29 \times 10^{10}$
2. `0_InterestBearingUPB`: $3.28 \times 10^{10}$
3. `0_CurrentActualUPB`: $3.28 \times 10^{10}$
4. `1_CurrentActualUPB`: $3.26 \times 10^{10}$
5. `1_InterestBearingUPB`: $3.26 \times 10^{10}$

**Note**: The correlation analysis with target shows NaN values because the training set has no anomalies (all target=0). Correlation can only be computed on the validation set where both classes exist.

### 4. Outliers

Using the IQR (Interquartile Range) method, several features show significant outlier percentages:

| Feature | Outlier % |
|---------|-----------|
| `PropertyValMethod` | 13.99% |
| `1_EstimatedLTV` | 9.79% |
| `MaturityDate` | 7.42% |
| `0_RemainingMonthsToLegalMaturity` | 7.41% |
| `OriginalLoanTerm` | 6.48% |

**Implication**: Robust scaling methods (RobustScaler) are recommended over StandardScaler.

### 5. Temporal Patterns

31 temporal features identified, including:
- `FirstPaymentDate`, `MaturityDate`
- Monthly reporting periods (0-12 months historical data)
- Remaining months to maturity
- First-time homebuyer flags

## Recommendations for Modeling

### 1. Feature Engineering

#### Recommended Transformations:
- **Scaling**: Use `RobustScaler` (handles outliers better than StandardScaler)
- **Dimensionality Reduction**: Apply PCA to reduce noise and computational cost
- **Domain Features**: Create amortization-based features (payment ratios, balance changes)

#### Features to Drop:
- 100% missing: `ReliefRefinanceIndicator`, `PreHARP_Flag`
- Near-constant: `SuperConformingFlag` (99% missing)
- Consider imputation for `MSA` (11% missing)

### 2. Model Selection

Given the **unsupervised nature** (train has no anomalies):

#### Recommended Approaches:
1. **Distance-Based**:
   - Local Outlier Factor (LOF) with multiple k values
   - K-Nearest Neighbors distance (k-distance)
   - Mahalanobis Distance

2. **Density-Based**:
   - DBSCAN
   - Isolation Forest

3. **Statistical**:
   - Elliptic Envelope (robust covariance)
   - PCA Reconstruction Error

4. **Deep Learning**:
   - LSTM Autoencoder (reconstruction error)
   - MLP Autoencoder

5. **Ensemble**:
   - Combine multiple detectors with weighted voting
   - Use validation set to determine optimal weights

### 3. Evaluation Strategy

**Metrics** (in order of priority):
1. **AUPRC** (Average Precision): Primary metric - handles class imbalance well
2. **AUROC** (ROC-AUC): Secondary metric - threshold-independent
3. **F1 Score**: Optional - requires threshold selection

**Cross-Validation**:
- **Train**: Fit models ONLY on normal data (target=0)
- **Validation**: Use for hyperparameter tuning and model selection
- **Critical**: NEVER fit models on validation data (causes leakage)

### 4. Feature Importance

Since training data has no anomalies, traditional correlation-based feature selection is not directly applicable. Instead:

1. Use validation set to evaluate feature importance
2. Apply variance thresholding to remove low-variance features
3. Consider domain knowledge (loan experts' insights)
4. Test feature subsets empirically using validation AUPRC

## Running the EDA

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Execute
```bash
python eda/comprehensive_eda.py
```

### Output
All results saved to `eda/outputs/`:
- Statistical summaries (JSON, CSV, TXT)
- Visualizations (PNG)

## Next Steps

1. ✅ **EDA Complete**: Understanding of data distributions and characteristics
2. → **Feature Engineering**: Implement robust preprocessing pipeline
3. → **Baseline Models**: Test 10+ unsupervised anomaly detection methods
4. → **Ensemble**: Combine best models with optimized weights
5. → **Evaluation**: Validate on holdout set, submit to Kaggle

## References

- **Dataset**: Freddie Mac Single-Family Loan-Level Dataset
- **Anomaly Definition**: Delinquency status ≠ 0 (any delinquency indicates anomaly)
- **Competition Metric**: Average Precision (AUPRC)

---

**Last Updated**: 2025-10-11
**Author**: CS5344 Project Team
