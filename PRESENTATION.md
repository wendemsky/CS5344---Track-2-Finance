# Unsupervised Loan Anomaly Detection
## CS5344 Big Data Analytics Technology - Track 2: Finance

**Team Members**: [Your Names]
**Date**: [Presentation Date]

---

## Slide 1: Title Slide
# Unsupervised Loan Anomaly Detection
### Predicting Loan Delinquency using Anomaly Detection

**CS5344 Big Data Analytics Technology**
**Track 2: Finance**

[Team Members]
[Date]

---

## Slide 2: Problem Statement       

### The Challenge
- **Objective**: Detect anomalous loans (delinquencies) using **unsupervised learning**
- **Domain**: Freddie Mac Single-Family Loan-Level Dataset
- **Key Constraint**: Train ONLY on normal loans (no anomalies in training data)

### What is a Loan Anomaly?
- **Normal (0)**: Current on payments (delinquency status = 0)
- **Anomaly (1)**: Any delinquency (30+ days late, 60+ days, REO acquisition, etc.)

### Why Unsupervised?
- Real-world scenario: Anomalies are rare and patterns evolve
- Must learn "normal behavior" and flag deviations

**[ADD VISUAL: Simple diagram showing Normal Loan vs Anomalous Loan payment patterns]**

---

## Slide 3: Dataset Overview

### Freddie Mac Loan Dataset

| Component | Description |
|-----------|-------------|
| **Training Set** | 30,504 loans (100% normal) |
| **Validation Set** | 5,370 loans (12.61% anomalous) |
| **Test Set** | 13,426 loans (unknown labels) |
| **Features** | 143 total (129 numeric, 14 categorical) |

### Feature Categories
1. **Static Features** (Origination)
   - Borrower: Credit Score, DTI, Number of Borrowers
   - Loan: Original UPB, Interest Rate, Loan Term, LTV, CLTV
   - Property: Type, State, Occupancy Status

2. **Temporal Features** (Monthly Performance - 13 months)
   - Current UPB, Interest Rate, Loan Age
   - Estimated LTV, Remaining Months to Maturity

**[ADD VISUAL: Screenshot of loans_train.csv showing sample rows and columns]**

---

## Slide 4: Evaluation Metrics

### Competition Rules
- **Primary Metric**: **Average Precision (AUPRC)**
  - Better for imbalanced data (12.61% anomaly rate)
  - Focus: Precision-recall trade-off

- **Secondary Metric**: **AUC-ROC**
  - Tie-breaker if AUPRC is equal

### Why AUPRC?
- **Class imbalance**: 87.39% normal, 12.61% anomalous
- AUROC can be misleading with imbalance
- AUPRC penalizes false positives more heavily

### Compliance Requirements
✅ Train ONLY on normal data (ytrain = 0)
✅ No fitting on validation set
✅ Validation used ONLY for hyperparameter tuning
✅ All preprocessing fit on training data only

---

## Slide 5: Exploratory Data Analysis - Key Findings

### 1. Class Distribution
- **Training**: 100% normal (by design)
- **Validation**: 87.39% normal, 12.61% anomalous
- **Imbalance ratio**: 1:6.93

**[ADD VISUAL: eda/outputs/class_distribution.png]**

### 2. Missing Values
- 3 features >99% missing (dropped): `ReliefRefinanceIndicator`, `PreHARP_Flag`, `SuperConformingFlag`
- MSA: ~11% missing (imputed)

**[ADD VISUAL: eda/outputs/missing_values_analysis.png - left panel only]**

---

## Slide 6: EDA - Distribution Analysis

### 3. Feature Distributions (Normal vs Anomaly)

**Key Observations**:
- Clear separation in some features (e.g., UPB trends, interest rates)
- Anomalous loans show irregular payment patterns
- High variance in balance changes

**[ADD VISUAL: eda/outputs/numeric_distributions.png - show 4-6 top features]**

### 4. Correlation with Target
- Top correlated features identified via validation set
- Used for feature importance understanding (NOT for training)

**[ADD VISUAL: eda/outputs/correlation_with_target.png]**

---

## Slide 7: EDA - Outliers & Insights

### 5. Outlier Analysis
- **13.99%** outliers in `PropertyValMethod`
- **9.79%** in `EstimatedLTV`
- **7.41%** in `RemainingMonthsToLegalMaturity`

**Implication**: Use **RobustScaler** instead of StandardScaler

**[ADD VISUAL: eda/outputs/outlier_analysis.png]**

### Key EDA Insights
✅ Domain features (amortization patterns) show strong signal
✅ Temporal trends matter (payment history over 13 months)
✅ Robust preprocessing essential due to outliers
✅ Unsupervised methods needed (train has no anomalies)

---

## Slide 8: Feature Engineering Strategy

### Advanced Feature Engineering Pipeline

1. **Sentinel Mapping**: Handle missing value codes (999, 9999, etc.)

2. **Temporal Engineering**: Multi-window aggregations
   - Mean, median, std across time windows
   - Trends, volatility, first differences

3. **Domain-Specific: Amortization Features** ⭐ **STRONGEST SIGNAL**
   ```
   - amort_short_mean:  Average payment shortfall
   - amort_short_70:    Fraction of months with >70% shortfall
   - amort_short_50:    Fraction of months with >50% shortfall
   ```
   **Key Insight**: Measures when borrowers fail to pay expected principal

4. **Preprocessing**:
   - RobustScaler (handles outliers)
   - PCA (80 components, 100% variance retained)

**[ADD VISUAL: Simple flowchart showing: Raw Data → Sentinel Mapping → Temporal Features → Amortization → Scaling → PCA]**

---

## Slide 9: Baseline Models - Extensive Experimentation

### 10 Baseline Algorithms Tested

| # | Algorithm | Best AUPRC | Best Config |
|---|-----------|------------|-------------|
| 1 | Isolation Forest | 0.1548 | n=500, samp=512 |
| 2 | **LOF** | **0.1955** ⭐ | **k=50, no PCA** |
| 3 | Random Projection LOF | 0.1939 | B=20, dim=50 |
| 4 | One-Class SVM | 0.1848 | RBF, nu=0.1 |
| 5 | KNN Distance | 0.1882 | k=5 |
| 6 | DBSCAN | 0.1807 | eps=1.0 |
| 7 | PCA Reconstruction | 0.1809 | n=80 |
| 8 | Mahalanobis Distance | 0.1738 | - |
| 9 | Elliptic Envelope | 0.1534 | support=0.8 |
| 10 | MLP Autoencoder | (skipped) | TensorFlow N/A |

**[ADD VISUAL: baseline_models/results/top15_comparison.png - AUPRC bar chart]**

---

## Slide 10: Baseline Comparison - Key Insights

### Preprocessing Impact

| Configuration | Mean AUPRC | Best Model |
|---------------|------------|------------|
| **Robust, No PCA** | **0.1755** ⭐ | LOF (k=50) |
| Standard + PCA-80 | 0.1728 | LOF (k=5) |
| Robust + PCA-80 | 0.1523 | LOF (k=50) |

**Finding**: **No PCA** preserves anomaly patterns better!

**[ADD VISUAL: baseline_models/results/preprocessing_impact.png]**

### Algorithm Comparison
- **LOF-based methods dominate** (LOF, RP-LOF)
- Distance-based > Tree-based for this task
- Ensemble approaches show promise

**[ADD VISUAL: baseline_models/results/algorithm_comparison_boxplots.png - first panel only]**

---

## Slide 11: Final Approach - Architecture

### Ultra Unsupervised Ensemble
**Validation AUPRC: 0.4524 | Kaggle AUPRC: 0.43**

```
Input: 143 Features
    ↓
Feature Engineering (feature_builder_advanced)
    ├── Temporal features (multi-window)
    ├── Amortization signals (payment shortfalls)
    ├── RobustScaler
    └── PCA (80 components)
    ↓
10 Selected Detectors (AUPRC ≥ 0.16 threshold)
    ├── Amortization Score        AUPRC: 0.4748 ⭐
    ├── LOF (k=4,5,6,7,8,10,12)  AUPRC: 0.30 avg
    ├── Cluster-wise LOF (n=12)   AUPRC: 0.2972
    └── Random Projection LOF      AUPRC: 0.2904
    ↓
Calibration (Train-only)
    ├── CDF: Empirical probability mapping
    └── Cohort: Z-score per cluster
    ↓
Fusion: Weighted Rank (Top-2)
    └── 61.2% Amort + 38.8% LOF(k=6)
    ↓
Final Score: 0.4524 AUPRC
```

**[ADD VISUAL: Hand-drawn or simple diagram of the pipeline flow]**

---

## Slide 12: Why This Approach Works

### Key Design Decisions

1. **Domain Features Trump ML Complexity**
   - Amortization alone: **0.4748 AUPRC**
   - Full ensemble: 0.4524 AUPRC
   - **Insight**: Payment shortfall = direct delinquency signal

2. **Multi-Scale LOF Ensemble**
   - Different k-values (4-12) capture anomalies at various density scales
   - Small k: Isolated outliers
   - Large k: Small anomaly clusters

3. **Cluster-wise Modeling**
   - Loans are heterogeneous (property types, regions, amounts)
   - KMeans (n=12) clusters + per-cluster LOF adapts locally

4. **Top-2 Fusion > Complex Ensemble**
   - Weighted by validation AUPRC
   - Avoids overfitting to validation set

---

## Slide 13: Leakage-Free Validation

### Ensuring Compliance

**What We DO** ✅:
- Fit ALL models on training data only
- Feature engineering fit on training only
- Use validation for:
  - Detector selection (AUPRC ≥ 0.16 threshold)
  - Fusion rule selection
  - Weight optimization

**What We DON'T DO** ❌:
- Never fit models on validation data
- Never use validation statistics for feature engineering
- Never train with anomalous samples

### Verification
```python
# All .fit() calls use ONLY Xtr (training data)
fb.fit(train)                          # Line 186 ✓
lof.fit(Xtr)                          # Line 65 ✓
km.fit(Xtr)                           # Line 206 ✓
pca.fit(Xtr_scaled)                   # Line 60 ✓

# Validation used ONLY for evaluation
auprc = average_precision_score(y_valid, scores)  # Line 296 ✓
```

---

## Slide 14: Results Summary

### Performance Comparison

| Approach | Valid AUPRC | Kaggle AUPRC | Improvement |
|----------|-------------|--------------|-------------|
| **Best Baseline** (LOF k=50) | 0.1955 | ~0.18 | - |
| **Our Ensemble** | **0.4524** | **0.43** | **+131%** |

### Per-Detector Performance (Top 5)

| Detector | AUPRC | AUROC |
|----------|-------|-------|
| **Amortization Score** | **0.4748** | **0.7524** |
| LOF (k=6) | 0.3017 | 0.6603 |
| LOF (k=7) | 0.3016 | 0.6631 |
| LOF (k=5) | 0.3007 | 0.6593 |
| Random Projection LOF | 0.2904 | 0.6700 |

**[ADD VISUAL: Bar chart showing AUPRC comparison - Baseline vs Our Approach]**

---

## Slide 15: Kaggle Leaderboard

### Competition Performance

**Public Leaderboard**:
- **Score**: 0.43 AUPRC
- **Improvement over baseline**: +139% (0.18 → 0.43)

### What This Means:
- Successfully identified **~43% of anomalous loans** in top predictions
- **2.4x better** than baseline LOF approach
- Achieved through domain knowledge + unsupervised ensemble

**[ADD VISUAL: Screenshot of Kaggle submission showing 0.43 score]**

---

## Slide 16: Key Insights & Lessons Learned

### What Worked ✅

1. **Domain Knowledge is King**
   - Amortization features (payment shortfalls) = strongest signal
   - Understanding loan mechanics > black-box ML

2. **Unsupervised Ensemble > Single Model**
   - Multi-scale LOF captures diverse anomaly types
   - Cluster-wise modeling handles heterogeneity

3. **Robust Preprocessing Matters**
   - RobustScaler essential for outlier-heavy data
   - No PCA preserves anomaly patterns better

4. **Keep It Compliant**
   - Train-only calibration prevents leakage
   - Validation for tuning, not training

---

## Slide 17: What Didn't Work

### Failed Approaches ❌

1. **Tree-Based Methods**
   - Isolation Forest: 0.1548 AUPRC (79% worse than LOF)
   - Decision Trees struggled with density-based anomalies

2. **PCA Dimensionality Reduction**
   - Hurt performance on average (-2.7% AUPRC)
   - Lost anomaly-specific patterns in compression

3. **One-Class SVM**
   - Slow (67s average) and mediocre (0.1848 AUPRC)
   - RBF kernel didn't capture loan payment patterns well

4. **Complex Fusion Rules**
   - Geometric mean, max, noisy-OR all performed worse
   - Simple weighted average won

---

## Slide 18: Technical Challenges & Solutions

### Challenge 1: Class Imbalance (12.61% anomalies)
**Solution**:
- AUPRC metric (handles imbalance)
- Unsupervised learning (no label needed in training)

### Challenge 2: Heterogeneous Loan Population
**Solution**:
- Cluster-wise LOF (KMeans n=12)
- Per-cluster adaptation

### Challenge 3: Outliers (14% in some features)
**Solution**:
- RobustScaler (uses IQR instead of mean/std)
- Elliptic Envelope for robust covariance

### Challenge 4: High Dimensionality (143 features)
**Solution**:
- Domain feature engineering (amortization)
- PCA for detector input (not for amortization)

---

## Slide 19: Future Improvements

### What Could Be Better?

1. **Deep Learning Approaches**
   - LSTM Autoencoders for temporal sequences
   - Transformer-based anomaly detection
   - Expected: +5-10% AUPRC boost

2. **Graph-Based Methods**
   - Loan networks (same borrower, same region)
   - Geographic clustering patterns
   - Capture relational anomalies

3. **Semi-Supervised Learning**
   - Pseudo-labeling high-confidence predictions
   - Active learning with expert feedback
   - Incremental model updates

4. **Feature Selection**
   - Automated feature importance ranking
   - Remove redundant temporal features
   - Faster inference

---

## Slide 20: Conclusion

### Summary

✅ **Problem**: Unsupervised loan delinquency detection
✅ **Approach**: Domain features + Multi-scale LOF ensemble
✅ **Result**: **0.4524 AUPRC (validation), 0.43 (Kaggle)**
✅ **Improvement**: **+131% over best baseline**

### Key Takeaways

1. **Domain Knowledge > Complex ML**
   - Amortization features alone = 0.4748 AUPRC
   - Understanding payment patterns crucial

2. **Unsupervised Methods Work**
   - LOF-based approaches dominate
   - No labels needed in training

3. **Compliance Matters**
   - Train-only fitting prevents leakage
   - Validation for tuning, not training

**[ADD VISUAL: Summary infographic with key numbers - 0.1955→0.4524, +131%, 10 baselines, 143 features]**

---

## Slide 21: Q&A

# Thank You!

### Questions?

**Contact**: [Your Email]
**Code**: [GitHub/Repository Link]
**Documentation**: See `README.md` in project folder

---

**Presentation Time**: ~10 minutes
**Total Slides**: 21 (adjust timing as needed)






intro
data 
eda
preprocessing
feature engineering
{ prof suggested feature correlation ?? }
base line models 
comparison analysis
final model - (7-10 slides)
results 

plan based on initail result - (5 slides)

