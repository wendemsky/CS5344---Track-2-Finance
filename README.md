# Loan Anomaly Detection Using Unsupervised Ensemble Methods

**CS5344: Big Data Analytics Technology**
**National University of Singapore**
**Group 11: Himanshu Maithani, Roheth Balamurugan**

## Project Overview

This project implements an unsupervised anomaly detection system for identifying delinquent loans in the Freddie Mac Single-Family Loan-Level Dataset. The final ensemble approach achieves:

- **Validation AUPRC: 0.4981** (AUROC: 0.7865)
- **Kaggle Test AUPRC: 0.478**
- **155% improvement** over the best baseline model (LOF with AUPRC=0.1955)

## Repository Structure

```
Project/
├── 1_Data/                      # Dataset files
├── 2_EDA/                       # Exploratory Data Analysis
├── 3_Baseline/                  # Baseline model evaluation
├── 4_Experiments/               # 7 key experiments
├── 5_Final_Approach/            # Final ensemble model
├── 6_Documentation/             # Papers and presentations
├── requirements.txt
└── README.md
```

## Quick Start - Reproduce Final Results

### 1. Setup Environment

**Prerequisites:**
- Python 3.8 or higher
- pip package manager

**Installation:**

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Data Files

Ensure the following data files are in the `1_Data/` folder:
- `loans_train.csv` (30,504 normal loans)
- `loans_valid.csv` (5,370 loans, 12.61% anomalous)
- `loans_test.csv` (13,426 loans)

### 3. Run Final Model (Main Submission)

```bash
# Generate final predictions - takes ~10-15 minutes
python 5_Final_Approach/final_model.py
```

**Output:**
- `5_Final_Approach/submission.csv` - Final Kaggle submission file
- Console output showing:
  - Individual detector performance (AUPRC, AUROC)
  - Fusion strategy results
  - Final validation metrics

**Expected Results:**
- Validation AUPRC: ~0.4981
- Validation AUROC: ~0.7865

### 4. (Optional) Run Other Components

```bash
# Exploratory Data Analysis
python 2_EDA/comprehensive_eda.py

# Baseline Models Evaluation (27 configurations)
python 3_Baseline/baseline_models.py

# Individual Experiments (7 experiments)
python 4_Experiments/exp1_amortization_irregularity_fusion.py
python 4_Experiments/exp2_autoencoder.py  # Requires TensorFlow
python 4_Experiments/exp3_enhanced_domain_features.py
python 4_Experiments/exp4_robust_scaler_preprocessing.py
python 4_Experiments/exp5_isolation_forest_baseline.py
python 4_Experiments/exp6_lof_hyperparameter_tuning.py
python 4_Experiments/exp7_simple_ensemble_fusion.py
```

## Results Summary

| Model | AUPRC | AUROC | Improvement |
|-------|-------|-------|-------------|
| Best Baseline (LOF) | 0.1955 | 0.5648 | - |
| Final Ensemble (Validation) | 0.4981 | 0.7865 | +155% |
| Final Ensemble (Kaggle) | 0.478 | - | +145% |

## Methodology

### Data

- Training: 30,504 normal loans (unsupervised learning)
- Validation: 5,370 loans (12.61% anomalous)
- Test: 13,426 loans
- Features: 143 total (129 numeric, 14 categorical)

### Approach

1. **Feature Engineering**
   - Robust scaling for outlier handling
   - PCA dimensionality reduction (80 components)
   - Domain-specific amortization features
   - Temporal payment patterns

2. **Anomaly Detection**
   - RFOD (Random Forest Outlier Detection)
   - LOF (Local Outlier Factor) variants
   - One-Class SVM
   - Cluster-wise anomaly detection

3. **Ensemble Fusion**
   - Inverse covariance weighting
   - EVT (Extreme Value Theory) calibration
   - Amortization-gated ensemble
   - Train-only fitting (no validation leakage)

### Key Innovations

- Amortization-based features provide strongest signal (AUPRC=0.4748)
- Multi-scale LOF captures anomalies at different density levels
- Robust preprocessing critical for outlier-heavy data
- Ensemble fusion with calibration improves generalization

## Folder Details

### 1_Data

Contains the three main CSV files (train, valid, test) with loan data.

### 2_EDA

Exploratory data analysis with statistical summaries and visualizations. Key findings include strong correlation of credit score and payment irregularity with anomalies.

### 3_Baseline

Evaluation of 10 baseline algorithms (27 configurations) including Isolation Forest, LOF, One-Class SVM, and others. LOF with k=50 achieved best baseline performance.

### 4_Experiments

7 experiments exploring different methodologies:
- Amortization + irregularity fusion (AUPRC=0.4821)
- Autoencoder (AUPRC=0.1677)
- Enhanced domain features (+144% improvement)
- Robust scaler preprocessing (+9% improvement)
- Isolation Forest baseline (AUPRC=0.1589)
- LOF hyperparameter tuning (best k=5, AUPRC=0.3231)
- Simple ensemble fusion strategies

### 5_Final_Approach

Final ensemble model combining RFOD, LOF variants, One-Class SVM, and domain-specific detectors. Achieves AUPRC=0.478 on Kaggle.

### 6_Documentation

Project papers and presentations.

## Dependencies

Core requirements:
- Python 3.8+
- numpy, pandas, scikit-learn, scipy
- matplotlib, seaborn

Optional:
- tensorflow (for autoencoder experiments)
- pyod (for advanced detector suite)

## Key Findings

### What Worked
- Domain-specific amortization features
- LOF-based density estimation
- Robust preprocessing without PCA
- Ensemble fusion with calibration
- Train-only fitting (leakage-free)

### What Did Not Work
- PCA dimensionality reduction
- Standard scaling (vs robust)
- Single detector approaches
- Complex fusion rules (overfitting)

## Competition Compliance & Reproducibility

### Data Leakage Prevention

Our implementation strictly follows unsupervised learning principles:

1. **Train-Only Fitting**: All models, scalers, PCA, and detectors are fitted ONLY on `loans_train.csv`
2. **No Validation Leakage**: Validation data (`loans_valid.csv`) is used ONLY for:
   - Hyperparameter selection
   - Fusion strategy selection
   - Performance evaluation
3. **Validation labels are NEVER used for training**

**Code Verification:**
- `FeatureBuilderAdvanced.fit()` is called only on training data (5_Final_Approach/final_model.py:832)
- All detectors use `novelty=True` or are fitted only on training set
- Calibration uses training distribution only

### Reproducibility

- **Fixed Random Seeds**: `np.random.default_rng(42)` ensures deterministic results
- **No External Data**: Uses only provided Kaggle datasets
- **Version-Locked Dependencies**: See `requirements.txt`
- **Expected Runtime**: 10-15 minutes on standard laptop (4-core CPU, 8GB RAM)

### Consistency Verification

The submitted `prediction.csv` matches our Kaggle submission (AUPRC=0.478) and is reproducible by running:
```bash
python 5_Final_Approach/final_model.py
```

## References

- Breunig et al. (2000) "LOF: Identifying Density-Based Local Outliers"
- Liu et al. (2008) "Isolation Forest"
- Freddie Mac Single-Family Loan-Level Dataset

## License

This project is for educational purposes as part of CS5344.
