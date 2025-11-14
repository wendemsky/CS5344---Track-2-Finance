# Experiments

This folder contains 7 key experiments exploring different methodologies for unsupervised loan anomaly detection.

## How to Run

Each experiment is standalone and can be run independently:

```bash
cd 4_Experiments
python exp1_amortization_irregularity_fusion.py
python exp2_autoencoder_deep_learning.py
python exp3_enhanced_domain_features.py
python exp4_robust_scaler_preprocessing.py
python exp5_isolation_forest_baseline.py
python exp6_lof_hyperparameter_tuning.py
python exp7_simple_ensemble_fusion.py
```

## Experiments

### Experiment 1: Amortization + Irregularity Fusion
**File:** `exp1_amortization_irregularity_fusion.py`

Combines amortization shortfall features with payment irregularity signals using weighted rank fusion.

**Results:**
- Amortization detector: AUPRC = 0.4749, AUROC = 0.7526
- Payment irregularity: AUPRC = 0.1261, AUROC = 0.5289
- Fused ensemble: AUPRC = 0.4821, AUROC = 0.7558
- Improvement: +1.5% over amortization alone

**Key Finding:** Amortization shortfall is the strongest single signal. Payment irregularity provides minimal complementary benefit.

---

### Experiment 2: Autoencoder Deep Learning
**File:** `exp2_autoencoder_deep_learning.py`

Tests deep learning MLP autoencoder for reconstruction-based anomaly detection.

**Architecture:**
- Input: 147 features
- Encoder: 147 → 73 → 36
- Bottleneck: 18
- Decoder: 18 → 36 → 73 → 147

**Results:**
- AUPRC: 0.1677
- AUROC: 0.5406

**Key Finding:** Autoencoder underperforms compared to classical methods. Reconstruction error alone is insufficient for this problem.

**Requires:** TensorFlow

---

### Experiment 3: Enhanced Domain Feature Engineering
**File:** `exp3_enhanced_domain_features.py`

Compares LOF performance on raw features vs engineered domain features.

**Results:**
- Raw features + LOF: AUPRC = 0.1209
- Engineered features + LOF: AUPRC = 0.2956
- Improvement: +144.4%

**Key Finding:** Domain feature engineering provides massive performance boost. Feature quality matters more than detector choice.

---

### Experiment 4: RobustScaler vs StandardScaler
**File:** `exp4_robust_scaler_preprocessing.py`

Evaluates preprocessing impact on LOF detector performance.

**Results:**
- StandardScaler + LOF: AUPRC = 0.1788, AUROC = 0.5231
- RobustScaler + LOF: AUPRC = 0.1955, AUROC = 0.5648
- Improvement: +9.3%

**Key Finding:** RobustScaler (median/IQR) handles outliers better than StandardScaler (mean/std) for this dataset.

---

### Experiment 5: Isolation Forest Baseline
**File:** `exp5_isolation_forest_baseline.py`

Tests Isolation Forest as alternative to LOF-based methods.

**Results:**
- IForest on PCA embeddings: AUPRC = 0.1589, AUROC = 0.5751
- IForest on scaled features: AUPRC = 0.1235, AUROC = 0.5173

**Key Finding:** Isolation Forest underperforms LOF. PCA embeddings work better than raw scaled features.

---

### Experiment 6: LOF Hyperparameter Tuning
**File:** `exp6_lof_hyperparameter_tuning.py`

Systematic evaluation of LOF k-neighbors parameter.

**Results:**

| k   | AUPRC  | AUROC  |
|-----|--------|--------|
| 5   | 0.3231 | 0.6931 |
| 10  | 0.3222 | 0.7031 |
| 20  | 0.3149 | 0.7126 |
| 30  | 0.3083 | 0.7124 |
| 50  | 0.2956 | 0.7048 |
| 70  | 0.2856 | 0.7014 |
| 100 | 0.2795 | 0.6945 |

**Best:** k=5 (AUPRC = 0.3231)

**Key Finding:** Smaller k values perform better, capturing local anomalous patterns more effectively than global density estimates.

---

### Experiment 7: Simple Ensemble Fusion
**File:** `exp7_simple_ensemble_fusion.py`

Compares fusion strategies for combining LOF, IForest, and Amortization detectors.

**Individual Detectors:**
- LOF: AUPRC = 0.2956
- IForest: AUPRC = 0.1589
- Amortization: AUPRC = 0.4749

**Fusion Results:**
- Average: AUPRC = 0.4132
- Maximum: AUPRC = 0.3306
- Weighted (by AUPRC): AUPRC = 0.4673
- Best detector only: AUPRC = 0.4749

**Key Finding:** The single best detector (Amortization) outperforms all fusion strategies. Fusion is only beneficial when detectors are diverse and of similar quality.

---

## Summary

**Top Performing Approach:** Amortization-based detector (AUPRC = 0.4749)

**Key Insights:**
1. Domain feature engineering provides 144% improvement over raw features
2. Amortization shortfall is the strongest anomaly signal
3. LOF with small k (5-10) works best for local patterns
4. RobustScaler handles outliers better than StandardScaler
5. Simple detectors on good features beat complex models on raw features
6. Fusion helps only when combining diverse, high-quality detectors

**Dependencies:**
- All experiments: numpy, pandas, scikit-learn
- exp2 only: tensorflow
