# CS5344 Big Data Analytics - Group 11 Submission

**Group Members:** Himanshu Maithani, Roheth Balamurugan
**Project:** Loan Anomaly Detection Using Unsupervised Ensemble Methods

## Submission Contents

```
Group11Codebase/
├── final_model.py          # Main code for generating predictions
├── feature_builder.py      # Feature engineering module
├── prediction.csv          # Final predictions on test set
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start - Reproduce Results

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure data files are available:**

   Place the following files in the same directory as `final_model.py` OR update the paths in the code:
   - `loans_train.csv` (30,504 loans)
   - `loans_valid.csv` (5,370 loans)
   - `loans_test.csv` (13,426 loans)

### Run the Code

```bash
python final_model.py
```

**Output:**
- Console output showing:
  - Individual detector performance metrics
  - Fusion strategy results
  - Validation AUPRC and AUROC
- `submission.csv` file containing test predictions

**Expected Validation Results:**
- AUPRC: ~0.4981
- AUROC: ~0.7865

## Methodology Overview

Our approach combines domain-aware feature engineering with an ensemble of unsupervised anomaly detectors:

### 1. Feature Engineering
- **Robust Scaling:** Handles outliers using median and IQR
- **PCA Embedding:** 80 components for density-based detectors
- **Amortization Features:** Domain-specific payment shortfall metrics
- **Temporal Features:** Payment pattern trends and volatility

### 2. Anomaly Detectors
- **RFOD** (Random Forest Outlier Detection)
- **Cohort LOF** (Cluster-specific Local Outlier Factor)
- **Random Projection LOF** (Multi-subspace density estimation)
- **One-Class SVM** (Support Vector outlier detection)
- **Amortization Score** (Domain-specific detector)

### 3. Ensemble Fusion
- **Calibration:** ECDF-based score normalization
- **Fusion Strategy:** Amortization-gated weighted ensemble
- **Train-Only Fitting:** Strict unsupervised learning (no validation leakage)

## Key Results

| Metric | Value |
|--------|-------|
| Validation AUPRC | 0.4981 |
| Validation AUROC | 0.7865 |
| Kaggle Test AUPRC | 0.478 |
| Improvement over Baseline | +155% |

## Code Structure

**`final_model.py`** - Main script that:
1. Loads and preprocesses data
2. Fits feature engineering pipeline (train-only)
3. Trains ensemble of detectors (train-only)
4. Calibrates and fuses detector scores
5. Generates predictions on test set
6. Outputs metrics and creates submission file

**`feature_builder.py`** - Feature engineering module:
- `FeatureBuilderAdvanced` class
- Handles sentinel values, scaling, PCA
- Creates amortization and temporal features
- Maintains train/validation separation

## Reproducibility

- **Fixed Random Seed:** `np.random.default_rng(42)`
- **Deterministic Results:** Same predictions on every run
- **No External Data:** Uses only provided Kaggle datasets
- **No Validation Leakage:** All models fitted only on training data

## Dependencies

Core requirements (see `requirements.txt` for versions):
- numpy
- pandas
- scikit-learn
- scipy

## Contact

For questions or issues, please contact:
- Himanshu Maithani: e1504053@u.nus.edu
- Roheth Balamurugan: e1415353@u.nus.edu

## Full Repository

Complete project code, experiments, and documentation:
https://github.com/wendemsky/CS5344---Track-2-Finance
