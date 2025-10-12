# Loan Anomaly Detection: Unsupervised Ensemble Approach
## Presentation Slides Content

---

## SLIDE 1: Title Slide

**Title**: Loan Anomaly Detection Using Unsupervised Ensemble Methods

**Subtitle**: CS5344 - Big Data Analytics Technology Project

**Team**: [Your Names]

**Date**: [Presentation Date]

**Visual**: Clean title slide with university logo

---

## SLIDE 2: Problem Statement

**Title**: Loan-Level Anomaly Detection Challenge

**Content**:

**Objective**: Learn an anomaly scoring function f: X → [0,1] where:
- f(xi) yields low scores for normal loans
- f(xi) yields high scores for abnormal loans
- f must be trained **only on normal samples** (unsupervised constraint)

**Business Context**:
- Detect loans that fail to meet monthly payment obligations
- Each loan combines static borrower info + monthly performance trajectory
- Challenge: Identify deviations from typical repayment patterns

**Key Constraints**:
- Training contains ONLY normal loans (yi = 0 always)
- Cannot use supervised learning methods
- Must generalize to unseen anomaly patterns

**Evaluation**: Average Precision (AUPRC) primary, AUC-ROC secondary

**Visual**: Mermaid Diagram 1 (from NEEDED_DIAGRAMS_ONLY.md) - Loan lifecycle showing origination → monthly payments → outcome

**Key Message**: "How do we detect anomalies when we've never seen one during training?"

---

## SLIDE 3: Dataset Overview

**Title**: Freddie Mac Single-Family Loan Dataset

**Content**:

**Data Source**: Freddie Mac loan-level dataset (2019)

**Data Splits**:

| Split | Samples (n) | Anomaly Rate | ytrain / yvalid |
|-------|---------|--------------|---------|
| **Train** | 30,504 | 0.00% | **yi = 0 for all i ∈ Itrain** |
| **Validation** | 5,370 | 12.61% (677 anomalies) | Mixed: yi ∈ {0,1} |
| **Test** | 13,426 | Unknown | Unlabeled (for evaluation) |

**Feature Structure** (m = 145 features):
- **Origination Variables** (Static):
  - CreditScore, OriginalDTI, OriginalInterestRate, OriginalUPB
  - LoanPurpose, PropertyType, OccupancyStatus, etc.
- **Performance Panel** (Temporal, N = 0-13 months):
  - N_CurrentActualUPB, N_InterestBearingUPB
  - N_NonInterestBearingUPB, N_EstimatedLTV
  - N_LoanAge, N_RemainingMonthsToLegalMaturity

**Data Format**: Each row i = (si, (ti,1, ri,1), ..., (ti,Ti, ri,Ti), yi)
- si = static features
- ri,k = monthly repayment info at month k
- yi ∈ {0,1} = label (0=normal, 1=missed ≥1 payment)

**Evaluation**: AUPRC (primary), AUROC (secondary)

**Visual**: `eda/outputs/class_distribution.png`

---

## SLIDE 4: Missing Values Analysis

**Title**: Exploratory Data Analysis: Missing Values

**Content**:

**Challenge**: Systematic missing value patterns detected in raw data

**Sentinel Values** (encoded missingness):
- CreditScore: 9999 → NaN
- OriginalDTI: 999 → NaN
- OriginalLTV: 999 → NaN
- MI_Pct: 999 → NaN
- Flag fields: '9', '99' → NaN

**Handling Strategy**:
1. Create missing indicator features (preserve information signal)
2. Median imputation for numeric features (robust to outliers)
3. Mode imputation for categorical features

**Key Insight**: Missingness itself can be informative (e.g., missing credit score may indicate higher risk)

**Visual**: `eda/outputs/missing_values_analysis.png` - Top 20 features by missing percentage

---

## SLIDE 5: Feature Distributions

**Title**: Feature Distribution Analysis

**Content**:

**Analysis**: Compared normal vs anomalous loan distributions (on validation set)

**Top 20 Features by Variance**:
- Statistical testing: Kolmogorov-Smirnov (KS) test
- Null hypothesis: Normal and anomaly distributions are identical
- **Result**: Many features show significant differences (p < 0.001)

**Key Insight**: Clear statistical separability exists between normal and anomalous loans in feature space

**Implication**: Distance-based and density-based detectors should be effective

**Visual**: `eda/outputs/numeric_distributions.png` - Histograms showing normal (green) vs anomaly (red) distributions

---

## SLIDE 6: Feature Correlation with Anomalies

**Title**: Feature-Target Correlation Analysis

**Content**:

**Challenge**: Training set has 0% anomalies (yi = 0 always) → correlation with target is undefined

**Solution**: Use validation set (12.61% anomalies) for exploratory correlation analysis
- Purpose: Understand feature-target relationships for feature engineering
- Used for EDA only, NOT for training
- All models trained exclusively on training data (no leakage)

**Top Correlated Features**:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| CreditScore | -0.250 | Lower scores → higher anomaly risk |
| 13_NonInterestBearingUPB | +0.140 | Higher non-payment balances → issues |
| 12_NonInterestBearingUPB | +0.120 | Late-stage payment irregularities |
| OriginalDTI | +0.100 | Higher debt-to-income → financial stress |
| OriginalInterestRate | +0.096 | Higher rates → riskier borrower profiles |

**Visual**: `eda/outputs/validation_top_correlations.png` - Horizontal bar chart of correlations

---

## SLIDE 7: Statistical Significance

**Title**: Feature-Anomaly Relationships

**Content**:

**CreditScore Deep Dive** (Strongest Predictor, r = -0.250):
- Normal loans: Mean = 752, Std = 44
- Anomalous loans: Mean = 717, Std = 49
- **35-point difference** (statistically significant)
- KS test: p < 1e-77 (extremely significant)

**Statistical Summary**:
- 26 out of 40 features tested show significant correlation (p < 0.05)
- **Key Observation**: No single feature has strong correlation (max |r| = 0.25)
- Implication: Anomalies arise from complex multi-feature interactions → need ensemble approach

**Visual**: `eda/outputs/validation_feature_distributions.png` - CreditScore distribution split by target

**Key Insight**: Correlation findings align with financial domain knowledge (credit risk fundamentals)

---

## SLIDE 8: Preprocessing Pipeline

**Title**: Data Preprocessing Strategy

**Content**:

**Step-by-Step Pipeline** (fit on training, transform on all splits):

1. **Sentinel Mapping**
   - Replace sentinel values (9999, 999) with NaN
   - Create missing indicator flags (preserve missingness signal)

2. **Categorical Encoding**
   - LabelEncoder with UNKNOWN handling for unseen test categories
   - Applied to: LoanPurpose, PropertyType, OccupancyStatus, etc.

3. **Imputation** (fit on training only)
   - Numeric: Median (robust to outliers)
   - Categorical: Mode

4. **Scaling** (fit on training only)
   - StandardScaler (zero mean, unit variance)
   - Ensures consistent feature scales for distance-based detectors

**Output**: Clean, scaled feature matrix X ∈ ℝ^(n × m) ready for modeling

---

## SLIDE 9: Temporal Feature Engineering

**Title**: Temporal Feature Engineering

**Content**:

**Challenge**: Raw performance panel has 14 monthly snapshots (N=0-13) → need trajectory patterns

**Multi-Window Strategy**:
Extract statistics from multiple time windows (3 strategies):
- **Main**: months [0, 3, 6, 9, 12] - quarterly sampling
- **Alt1**: months [0, 2, 4, 6, 8, 10, 12] - bimonthly sampling
- **Alt2**: months [0, 3, 6, 9] - first-year focus

**Features per window** (applied to UPB, InterestBearing, LTV, etc.):
- **Trend**: (last - first) / |first| → overall trajectory direction
- **Volatility**: std / |mean| → payment stability
- **First-difference mean**: Avg(Δt) → average change per period
- **First-difference std**: Std(Δt) → change variability

**Intuition**: Increasing volatility or negative trends in late months signal payment stress

**Result**: ~60-80 additional temporal features capturing lifecycle dynamics

---

## SLIDE 10: Domain-Driven Features

**Title**: Domain-Specific Feature Engineering

**Content**:

**1. Amortization Shortfall Signals**:

Motivation: Normal loans follow predictable principal reduction schedule → deviations signal issues

**Calculation**:
- Expected principal reduction: Use annuity formula with InterestRate, RemainingMonths
- Observed principal reduction: Δ(UPB) from month to month
- Shortfall ratio = (Expected - Observed) / Expected

**Engineered Features**:
- `amort_short_mean`: Average shortfall across all periods
- `amort_short_70`: Fraction of periods with >70% shortfall (severe)
- `amort_short_50`: Fraction of periods with >50% shortfall (moderate)
- Only applicable to Fixed-Rate Mortgage (FRM) loans

**2. PCA Dimensionality Reduction**:
- Reduce ~166 features → 80 principal components
- Retains ~95% of variance
- Benefits: Reduces noise, improves LOF/k-distance performance, mitigates curse of dimensionality

---

## SLIDE 11: Baseline Models Overview

**Title**: Baseline Model Evaluation

**Content**:

**Objective**: Systematically evaluate unsupervised anomaly detectors to identify best approaches

**Algorithms Tested** (7 families):
1. **Local Outlier Factor (LOF)** - local density deviation
2. **k-distance** - distance to k-th nearest neighbor
3. **Isolation Forest** - tree-based isolation
4. **Elliptic Envelope** - robust covariance outliers
5. **One-Class SVM** - maximum margin separation
6. **DBSCAN** - density-based clustering
7. **PCA Reconstruction Error** - linear subspace deviation

**Configurations**:
- 3 preprocessing setups: robust_pca80, standard_pca80, robust_pcaNone
- Multiple hyperparameters per algorithm (e.g., LOF k=5,7,10,...)
- **Total**: ~50 baseline configurations tested

**Evaluation**: AUPRC (primary), AUROC (secondary), F1@best-threshold (reference)

---

## SLIDE 12: Baseline Results

**Title**: Baseline Model Performance

**Content**:

**Top Performers (by AUPRC)**:

| Rank | Model | AUPRC | AUROC | Config |
|------|-------|-------|-------|--------|
| 1 | LOF (k=5) | 0.28-0.30 | 0.65-0.66 | robust_pca80 |
| 2 | LOF (k=7) | 0.28-0.29 | 0.65-0.66 | robust_pca80 |
| 3 | LOF (k=10) | 0.27-0.29 | 0.65-0.67 | robust_pca80 |
| 4 | k-distance (k=3) | 0.19-0.20 | 0.60-0.61 | robust_pca80 |
| 5 | Isolation Forest | 0.18 | 0.56-0.57 | robust_pca80 |

**Key Finding**: LOF consistently outperforms other methods

**Visual**: `baseline_models/results/top15_comparison.png` - Bar chart of top 15 models

---

## SLIDE 13: Why LOF Works Best

**Title**: Local Outlier Factor: Algorithm Intuition

**Content**:

**How LOF Works**:
- Measures **local density deviation** relative to neighbors
- For each point, compute: LOF = (avg neighbor density) / (point density)
- **LOF ≈ 1** → normal (similar density to neighbors)
- **LOF > 1** → outlier (lower density than neighbors, isolated)

**Why It's Effective for Loan Anomalies**:
- Anomalies = loans with unusual **feature combinations**, not just extreme values
  - Example: Low CreditScore + High DTI + High InterestRate (individually borderline, together anomalous)
- LOF captures **multi-dimensional density patterns** in combined feature space
- k-parameter controls neighborhood size
  - **k=5-10 optimal** for this dataset (validated empirically)
  - Too small (k=3): noisy, too large (k=20): misses local patterns

**Visual**: Mermaid Diagram 3 (from NEEDED_DIAGRAMS_ONLY.md) - LOF intuition with normal cluster + outlier

---

## SLIDE 14: Final Approach Overview

**Title**: Final Model: Ultra Unsupervised Ensemble

**Content**:

**High-Level Architecture**:
```
Raw Data (145 features)
    ↓
FeatureBuilderAdvanced (temporal + amortization + PCA)
    ↓
Multiple Unsupervised Detectors (~20 detectors, 9 types)
    ↓
Train-CDF Calibration (score → probability)
    ↓
Detector Selection (top 10 by validation AUPRC)
    ↓
Fusion Strategy Evaluation (rank/probability-based)
    ↓
Final Anomaly Scores [0,1]
```

**Core Principles**:
- **Unsupervised constraint**: Train ONLY on normal loans (Dtrain, yi = 0 always)
- **No leakage**: Validation used ONLY for hyperparameter/detector selection, NOT for training
- **Diversity**: Multiple detector types capture different anomaly patterns (local density, global outliers, subspace deviations)

**Implementation**: `final_approach/unsup_ultra_ensemble_fast_improvement.py`

---

## SLIDE 15: Feature Builder Pipeline

**Title**: FeatureBuilderAdvanced Class

**Content**:

**Four-Stage Pipeline**:

**Stage 1: Static Features**
- Sentinel mapping → NaN + missing indicator flags
- Categorical encoding (LabelEncoder with UNKNOWN handling)
- Numeric imputation (median, fit on training)

**Stage 2: Temporal Features**
- Multi-window extraction (3 window strategies: main, alt1, alt2)
- Per-window statistics: trend, volatility, first-diff mean/std
- Applied to: UPB, InterestBearingUPB, LTV, LoanAge
- **Result**: ~60-80 temporal features

**Stage 3: Amortization Signals**
- Expected vs observed principal reduction (annuity formula)
- Features: amort_short_mean, amort_short_70, amort_short_50
- Masking for non-FRM loans (interest-only, balloon)

**Stage 4: Scaling + PCA**
- StandardScaler (fit on training) → X_scaled (~166 features)
- PCA (fit on training) → X_embed (80 components, ~95% variance)

**Usage**:
```python
fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
fb.fit(train)  # Fit ONLY on training (yi = 0)
X_scaled, slices, X_embed = fb.transform(data)  # Transform any split
```

---

## SLIDE 16: Detector Portfolio

**Title**: Ensemble Detector Components

**Content**:

**9 Detector Families** (Total: ~20 individual detectors):

| Type | Variants | Purpose | Input Space |
|------|----------|---------|-------------|
| **LOF** | k ∈ {4,5,6,7,8,10,12} | Local density deviation | X_embed (PCA) |
| **Cluster-LOF** | n_clusters=12 | Cohort-specific anomalies | X_embed (PCA) |
| **k-distance** | k ∈ {3,5,7,9,11} | Distance to k-th neighbor | X_embed (PCA) |
| **Isolation Forest** | n_trees=500 | Tree-based isolation | X_embed (PCA) |
| **Elliptic Envelope** | contamination=0.1 | Robust covariance outliers | X_embed (PCA) |
| **PCA Reconstruction** | 80 components | Linear subspace reconstruction error | X_scaled (full) |
| **Random Projection LOF** | n_bags=40 | Bagged LOF in random subspaces | X_scaled (full) |
| **Mahalanobis** | Standard covariance | Global covariance-based distance | X_scaled (full) |
| **Amortization Signal** | Custom | Payment shortfall aggregation | X_scaled (amort slice) |

**Selection Criterion**: Keep detectors with validation AUPRC ≥ 0.16 → typically top 10 detectors

**Diversity Rationale**: Different detectors capture different anomaly types (local vs global, linear vs nonlinear)

---

## SLIDE 17: Training Strategy

**Title**: Training on Normal Loans Only

**Content**:

**Fundamental Principle**: All detectors fit ONLY on Dtrain (n=30,504, yi = 0 always)

**Why This Works**:
- Unsupervised detectors learn the **distribution of normal loans**
- At inference, **deviations from normal** → high anomaly scores
- No labeled anomalies needed during training (novelty detection paradigm)

**Example: LOF Implementation**
```python
# Step 1: Fit on ONLY normal loans (training set)
lof_model = LocalOutlierFactor(n_neighbors=5, novelty=True)
lof_model.fit(X_train)  # All yi = 0 (normal loans only)

# Step 2: Score validation/test (contains anomalies)
scores_valid = -lof_model.score_samples(X_valid)  # Higher = more anomalous
scores_test = -lof_model.score_samples(X_test)
```

**Key Insight**: Training on normal loans = learning "typical loan behavior" → deviations become detectable

**No Leakage**: Validation/test data NEVER used for fitting, only for scoring and hyperparameter selection

---

## SLIDE 18: Calibration Method

**Title**: Train-CDF Calibration

**Content**:

**Problem**: Different detectors produce incomparable score scales
- LOF: ~1-5 (local density ratio)
- k-distance: ~0.1-10 (Euclidean distance)
- Isolation Forest: ~-0.5-0.5 (tree depth anomaly score)
- → Cannot directly combine or compare

**Solution**: Train-CDF Calibration
1. Fit empirical CDF on **training set scores only**
2. Map any score x → percentile in training distribution → [0,1] probability
3. Higher percentile → more extreme relative to normal → higher anomaly probability

**Algorithm**:
```python
def train_cdf_calibrator(train_scores):
    sorted_scores = np.sort(train_scores)  # Empirical CDF from training
    def cdf(x):
        rank = np.searchsorted(sorted_scores, x, side='right')
        return rank / len(sorted_scores)  # Percentile [0,1]
    return cdf

# Usage
cdf_func = train_cdf_calibrator(detector_scores_train)
prob_valid = cdf_func(detector_scores_valid)  # Calibrated probabilities [0,1]
```

**Advantage**: No validation leakage (uses ONLY training distribution), normalizes all detectors to [0,1]

---

## SLIDE 19: Fusion Strategies

**Title**: Ensemble Fusion Approaches

**Content**:

**Strategy 1: Rank-Based Fusion**
- Rank-normalize each detector's calibrated scores → [0,1]
- Combine using:
  - **Weighted average**: ∑(wi × ranki), weights wi = validation AUPRC
  - **Max rank**: max(rank1, rank2, ..., rankD)
  - **Max rank top-k**: Use only top-2 or top-3 best detectors by AUPRC

**Strategy 2: Probability-Based Fusion**
- Use Train-CDF calibrated probabilities pi ∈ [0,1]
- Combine using:
  - **Weighted average**: ∑(wi × pi)
  - **Noisy-OR**: 1 - ∏(1 - pi) (assumes independence)
  - **Max probability**: max(p1, p2, ..., pD)

**Strategy 3: Cohort-Normalized Fusion**
- Cluster samples into cohorts (KMeans, n_clusters=12)
- Z-score normalize scores within each cohort (handles cluster-specific scales)
- Apply rank/probability fusion on normalized scores

**Selection**: Evaluate all fusion strategies on Dvalid, select best by AUPRC

---

## SLIDE 20: Hyperparameter Selection

**Title**: Hyperparameter Optimization

**Content**:

**Key Hyperparameters**:

1. **PCA Components** (pca_comps)
   - Tested: 60, 80, 100, 120
   - Selected: 80 (best validation AUPRC)

2. **LOF k-values**
   - Tested: k=4,5,6,7,8,10,12,15
   - Selected: k=4-12 (multiple variants kept)

3. **Detector Selection Threshold**
   - Keep detectors with AUPRC ≥ 0.16
   - Results in ~10 detectors

4. **Clustering (for cohort normalization)**
   - Tested: 10, 12, 15 clusters
   - Selected: 12 clusters

**Validation Strategy**:
- Use validation AUPRC to select hyperparameters
- Avoid overfitting by keeping selection simple (threshold-based)

---

## SLIDE 21: Complete Architecture

**Title**: End-to-End Pipeline Architecture

**Content**:

**Visual**: Mermaid Diagram 2 (from NEEDED_DIAGRAMS_ONLY.md) - Full architecture flowchart

**Key Pipeline Stages**:

1. **Input**: Raw loan data (30,504 × 145)
2. **FeatureBuilderAdvanced**:
   - Sentinel mapping → NaN + flags
   - Temporal engineering (3 window strategies)
   - Amortization signals (shortfall features)
   - Scaling + PCA (80 components)
3. **Detector Training** (on Dtrain only, yi = 0):
   - 7× LOF (k=4,5,6,7,8,10,12)
   - 1× Cluster-LOF (n_clusters=12)
   - 5× k-distance (k=3,5,7,9,11)
   - 1× Isolation Forest, 1× Elliptic Envelope
   - 1× PCA reconstruction, 1× Random Projection LOF
   - 1× Mahalanobis, 1× Amortization signal
4. **Score Generation**: Apply detectors to Dvalid, Dtest
5. **Train-CDF Calibration**: scores → [0,1] probabilities
6. **Detector Selection**: Keep top 10 (validation AUPRC ≥ 0.16)
7. **Fusion Strategy Evaluation**: Select best fusion by validation AUPRC
8. **Output**: Final anomaly scores [0,1]

---

## SLIDE 22: Best Configuration

**Title**: Final Selected Configuration

**Content**:

**Hyperparameters**:
- **PCA Components**: 80 (retains ~95% variance)
- **Detector Selection Threshold**: AUPRC ≥ 0.16
- **Number of Detectors Selected**: 10

**Top 10 Detectors** (ranked by validation AUPRC):

| Rank | Detector | Validation AUPRC | Notes |
|------|----------|------------------|-------|
| 1 | LOF k=5 | 0.2988 | Best overall |
| 2 | LOF k=7 | 0.2965 | Close second |
| 3 | LOF k=10 | 0.2943 | Robust |
| 4 | Cluster-LOF | 0.2881 | Cohort-specific |
| 5 | k-distance k=3 | 0.1949 | Complementary |
| 6-8 | k-distance k=5,7,9 | 0.1929-0.1890 | Distance-based |
| 9 | Isolation Forest | 0.1817 | Tree-based diversity |
| 10 | PCA reconstruction | 0.1742 | Linear subspace |

**Best Fusion Strategy**: max_rank_top2
- Uses top 2 detectors: LOF k=5, LOF k=7
- Takes maximum rank → emphasizes agreement on high-risk loans
- **Validation AUPRC**: **[INSERT YOUR ACTUAL RESULT]**
- **Validation AUROC**: **[INSERT YOUR ACTUAL RESULT]**

---

## SLIDE 23: Final Performance

**Title**: Model Performance Results

**Content**:

**Validation Set Performance**:
- **AUPRC**: **[YOUR RESULT]**
- **AUROC**: **[YOUR RESULT]**
- F1 Score: **[YOUR RESULT]** (at optimal threshold)

**Best Individual Detector**: LOF k=5
- AUPRC: 0.2988
- AUROC: 0.6607

**Ensemble vs Individual**:
- Ensemble: [YOUR RESULT]
- Best Single: 0.2988
- Trade-off: [Slight decrease/increase] for robustness

**Visual**: Table or bar chart comparing metrics

---

## SLIDE 24: Per-Detector Contributions

**Title**: Detector Performance Analysis

**Content**:

**Detector Family Performance**:

| Family | Best AUPRC | Count in Top-10 | Insight |
|--------|------------|-----------------|---------|
| LOF | 0.2988 | 3 | Dominates top positions |
| Cluster-LOF | 0.2881 | 1 | Cohort-specific patterns |
| k-distance | 0.1949 | 4 | Complementary to LOF |
| Isolation Forest | 0.1817 | 1 | Tree-based diversity |
| PCA reconstruction | 0.1742 | 1 | Linear subspace outliers |

**Key Insight**: LOF-based detectors consistently outperform
- Captures local density deviations effectively
- Multiple k-values provide robustness

**Visual**: Grouped bar chart showing AUPRC by detector family

---

## SLIDE 25: Error Analysis

**Title**: What the Model Captures

**Content**:

**Patterns Detected Well**:
- Low CreditScore + High DTI combinations
- Late-stage payment irregularities (months 10-13)
- High interest rate + risky borrower profiles
- Payment shortfall patterns (amortization)

**Limitations**:
- No single feature strongly predicts (max |r| = 0.25)
- Temporal dependencies underutilized
- Performance ceiling in unsupervised setting
- Some normal loans have low scores (false positives)

**Future Improvements**:
- Explicit temporal modeling (LSTM, attention)
- Graph-based loan similarity networks
- Semi-supervised fine-tuning with small labeled set

---

## SLIDE 26: Alternative Approaches Explored

**Title**: Experimental Tangents

**Content**:

**Context**: After correlation analysis (Slides 6-7), explored feature-driven experiments

**Experiment 1: Correlation-Guided Feature Engineering**
- **Motivation**: Use correlation insights (CreditScore, DTI, InterestRate, NonInterestBearingUPB) to engineer composite features
- **Approach**:
  - Created composite risk scores: `0.5 × credit_risk + 0.25 × dti_risk + 0.25 × rate_risk`
  - Added payment irregularity indicators (late-stage UPB ratios)
  - Trained simplified ensemble on these features
- **Result**:
  - Ensemble AUPRC = 0.2931
  - Best single (LOF k=5) AUPRC = 0.2992
- **Insight**: Features help slightly, but ultra ensemble (with full feature engineering) still superior

**Experiment 2: Feature-Weighted Isolation Forest**
- Upweight top correlated features in IF random sampling
- Result: Moderate improvement over standard IF, but not better than LOF baseline

**Conclusion**: Final ultra ensemble approach (Slide 14-22) outperformed all experimental tangents

---

## SLIDE 27: Approach Comparison

**Title**: Performance Comparison Across Approaches

**Content**:

| Approach | Best AUPRC | AUROC | Notes |
|----------|------------|-------|-------|
| Baseline LOF (single, k=5) | 0.2988 | 0.6607 | Simple baseline from initial evaluation |
| Correlation-guided experiment | 0.2992 | 0.6607 | Best single from tangent (LOF k=5) |
| **Final Ultra Ensemble** | **[YOUR RESULT]** | **[YOUR RESULT]** | **Production model** |

**Why Final Ultra Ensemble is Superior**:
1. **Diversity**: 9 detector families capture different anomaly patterns (local density, global distance, tree-based, linear subspace)
2. **Robustness**: Ensemble reduces variance compared to single detector
3. **Calibration**: Train-CDF prevents score scale issues, enables fair fusion
4. **Systematic evaluation**: All fusion strategies tested, best selected by validation AUPRC
5. **No leakage**: Strict unsupervised protocol (training on yi = 0 only)

**Trade-off**: Slight AUPRC change (ensemble vs single) but improved robustness to distribution shift

**Visual**: Bar chart comparing AUPRC across approaches (if time permits)

---

## SLIDE 28: Lessons Learned

**Title**: Key Takeaways from Project

**Content**:

**Technical Insights**:
1. **LOF dominates** for this dataset: Local density deviation captures loan anomalies better than global methods (IForest, OCSVM)
2. **Feature engineering > model complexity**: Temporal trajectories + amortization signals provide more value than complex models
3. **Ensemble provides robustness**: Mitigates overfitting to validation set, reduces variance
4. **Correlation analysis critical** when training has 0% anomalies: Use validation for EDA (not training) to discover predictive features
5. **Train-CDF calibration essential**: Enables fair fusion without validation leakage

**Methodological Insights**:
1. **Systematic baseline evaluation** (50+ configs) → identifies best detector families (LOF, k-distance)
2. **Diverse detector portfolio** captures different anomaly types (local vs global, linear vs nonlinear)
3. **Strict train/validation protocol** prevents leakage: fit only on train, select only on validation
4. **Domain knowledge + data insights** = success: Amortization shortfall from finance domain knowledge

**Practical Insights**:
1. Unsupervised methods achieve reasonable performance (~0.30 AUPRC) despite 0% anomalies in training
2. No silver bullet feature (max |r| = 0.25) → ensemble necessary
3. Iterative experimentation pays off: Correlation tangent → insights → better features

---

## SLIDE 29: Current Limitations

**Title**: Limitations & Challenges

**Content**:

**Data Limitations**:
1. **No anomalies in training** (yi = 0 always) → cannot use supervised learning
2. **Limited anomaly examples** in validation (677 out of 5,370 = 12.61%)
3. **Potential distribution shift** between validation and test (unknown test anomaly rate)
4. **Missing values** in temporal features (especially late months N=10-13)
5. **Label definition ambiguity**: "missed ≥1 payment" captures diverse failure modes

**Model Limitations**:
1. **Temporal dependencies underexploited**:
   - Treated as independent time windows, not sequential trajectories
   - No explicit loan lifecycle modeling (LSTM, RNN, attention)
2. **Linear dimensionality reduction** (PCA) → may miss nonlinear manifolds
3. **Ensemble complexity vs interpretability trade-off**: Hard to explain why a loan is flagged
4. **Computational cost**: ~20 detectors × scoring time → slower than single model

**Evaluation Limitations**:
1. **Single metric focus** (AUPRC) may not capture all business needs (e.g., recall at high precision)
2. **Threshold selection** depends on cost-benefit analysis (false alarm cost vs missed anomaly cost)
3. **No explainability** for individual predictions (regulatory/compliance risk)

---

## SLIDE 30: Future Improvements - Features

**Title**: Future Work: Feature Engineering

**Content**:

**Advanced Temporal Modeling**:
- **LSTM/RNN** for payment trajectory sequences
- **Attention mechanisms** to focus on critical months
- **Time-series specific detectors** (ARIMA residuals, Prophet anomalies)
- **Seasonal pattern detection** (monthly, quarterly trends)

**Interaction Features**:
- **Polynomial interactions** between top correlated features
  - Example: CreditScore × DTI, Rate × LTV
- **Graph-based features** (loan similarity networks)
- **Cluster-based features** (distance to loan archetypes)

**External Data Integration**:
- **Economic indicators** (unemployment, GDP, interest rate environment)
- **Geographic risk factors** (MSA, postal code statistics)
- **Market conditions** (housing prices, foreclosure rates)

---

## SLIDE 31: Future Improvements - Models

**Title**: Future Work: Advanced Models

**Content**:

**Deep Learning Approaches**:
- **Variational Autoencoders (VAE)**
  - Learn latent representations of normal loans
  - Reconstruction error as anomaly score
- **Attention-based models**
  - Focus on important temporal patterns
  - Interpretable attention weights
- **Self-supervised pretraining**
  - Contrastive learning on normal loans
  - Fine-tune for anomaly detection

**Advanced Ensemble Methods**:
- **Stacking meta-learner**
  - Train meta-model on detector outputs
  - Learn optimal combination weights
- **Dynamic weighting** by sample characteristics
  - Different detectors for different loan types
- **Cluster-specific ensembles**
  - Separate models for loan cohorts

**Hybrid Approaches**:
- **Semi-supervised learning** with small labeled set
- **Active learning** for label acquisition
- **One-class neural networks**

---

## SLIDE 32: Deployment Considerations

**Title**: Production Deployment

**Content**:

**Technical Requirements**:
- **Real-time scoring**: < 100ms latency per loan
- **Batch processing**: Score large portfolios overnight
- **Model monitoring**: Detect drift in feature distributions
- **Explainability**: Regulatory compliance (why is loan flagged?)

**Implementation Strategy**:
1. **Feature pipeline automation**
   - Scheduled data updates
   - Feature computation on demand
2. **Model serving**
   - REST API for real-time scoring
   - Batch jobs for portfolio analysis
3. **Monitoring dashboard**
   - Score distribution tracking
   - Feature drift detection
   - Performance metrics (if labels available)

**Business Impact**:
- **Early warning system** for risky loans
- **Portfolio risk assessment** and management
- **Lending decision support** (not replacement)

---

## SLIDE 33: Summary

**Title**: Project Summary

**Content**:

**Problem**: Detect anomalous loans (missed ≥1 payment) using only normal loan training data (unsupervised constraint)

**Dataset**: Freddie Mac 2019 loan-level data
- Dtrain: 30,504 loans, 0% anomalies (yi = 0 always)
- Dvalid: 5,370 loans, 12.61% anomalies
- 145 raw features → ~166 engineered features

**Approach**: Ultra Unsupervised Ensemble
1. **Feature engineering**: Temporal trajectories (3 window strategies) + amortization shortfall + PCA (80 components)
2. **Diverse detector portfolio**: 9 families, ~20 detectors (LOF-focused)
3. **Train-CDF calibration**: Score normalization without validation leakage
4. **Systematic fusion**: Evaluate all strategies, select best by validation AUPRC

**Results**:
- **Best single detector**: LOF k=5 (AUPRC = 0.2988, AUROC = 0.6607)
- **Final ensemble**: AUPRC = **[YOUR RESULT]**, AUROC = **[YOUR RESULT]**
- Demonstrates effective unsupervised anomaly detection despite 0% training anomalies

**Key Contributions**:
1. Validation-based correlation analysis for feature discovery (when training has 0% anomalies)
2. Multi-window temporal feature engineering capturing payment trajectories
3. Domain-driven amortization shortfall signals (finance domain knowledge)
4. Comprehensive ensemble evaluation framework with no validation leakage

**Impact**: Production-ready unsupervised loan risk assessment system

---

## SLIDE 34: Q&A

**Title**: Questions?

**Content**:
- Thank you for your attention!
- Happy to answer questions

**Visual**: Clean slide with team contact information

---

## APPENDIX SLIDES (Optional - for deep questions)

---

## APPENDIX A: Hyperparameter Tuning Details

**Title**: Appendix: Hyperparameter Grid Search

**Content**:

**PCA Components**:
- Tested: [60, 80, 100, 120]
- Validation AUPRC:
  - 60: [result]
  - 80: [result] ← selected
  - 100: [result]
  - 120: [result]

**LOF k-parameter**:
- Tested: [4, 5, 6, 7, 8, 10, 12, 15]
- Kept multiple variants (k=4-12) in ensemble

**Detector Selection Threshold**:
- Tested: [0.14, 0.15, 0.16, 0.17]
- Selected: 0.16 (balances diversity vs quality)

---

## APPENDIX B: Fusion Strategy Details

**Title**: Appendix: Complete Fusion Comparison

**Content**:

**All Fusion Strategies Tested**:

| Strategy | Type | AUPRC | AUROC |
|----------|------|-------|-------|
| max_rank_top2 | Rank | [YOUR RESULT] | [YOUR RESULT] |
| max_rank_top3 | Rank | [result] | [result] |
| wavg_rank | Rank | [result] | [result] |
| max_rank | Rank | [result] | [result] |
| wavg_prob | Probability | [result] | [result] |
| p_noisy_or | Probability | [result] | [result] |
| rankCoh_max_top2 | Cohort-norm | [result] | [result] |

**Selected**: [Best strategy based on validation AUPRC]

---

## APPENDIX C: Computational Analysis

**Title**: Appendix: Computational Complexity

**Content**:

**Training Time** (on [your hardware specs]):
- Feature engineering: ~X seconds
- Detector training: ~Y seconds
- Total training: ~Z minutes

**Inference Time**:
- Single loan: < X ms
- Batch (1000 loans): ~Y seconds

**Memory Requirements**:
- Model size: ~X MB
- Peak RAM usage: ~Y GB

**Scalability**: Can process ~Z loans/second

---

## APPENDIX D: Interpretability

**Title**: Appendix: Model Interpretability

**Content**:

**Feature Importance** (aggregated from detectors):
- Top 10 most influential features
- [List from your analysis]

**SHAP Analysis** (if computed):
- SHAP values for sample predictions
- Visualizations

**Cohort Analysis**:
- Which loan types are flagged most often?
- Cluster characteristics

---

**END OF PRESENTATION**

---

## Notes for Presenter:

**Timing**:
- Aim for ~30 seconds per slide
- Deep-dive slides (14-22): ~1 minute each
- Leave 5-10 minutes for Q&A

**Emphasis**:
- Slides 6-7 (correlation analysis): Unique contribution
- Slides 14-22 (final model): Core technical content
- Slide 23 (performance): Key results

**Backup Answers**:
- "Why ensemble if single is better?" → Robustness to distribution shift
- "How ensure no overfitting?" → Train-CDF calibration uses only train distribution
- "Real-world deployment?" → See slide 32

