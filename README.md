# Loan Anomaly Detection Using Unsupervised Ensemble Methods

CS5344: Big Data Analytics Technology
National University of Singapore

## Project Overview

This project implements an unsupervised anomaly detection system for identifying delinquent loans in the Freddie Mac Single-Family Loan-Level Dataset. The final ensemble approach achieves **AUPRC=0.4981** on validation (0.478 on Kaggle), representing a **155% improvement** over the best baseline model.

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

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Analysis Pipeline

```bash
# Exploratory Data Analysis
python 2_EDA/comprehensive_eda.py

# Baseline Models
python 3_Baseline/baseline_models.py

# Final Approach
python 5_Final_Approach/final_model.py
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
- Autoencoder deep learning (AUPRC=0.1677)
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

## Competition Compliance

- Fully unsupervised (train on normal data only)
- No validation leakage (train-only fitting)
- AUPRC as primary metric

## References

- Breunig et al. (2000) "LOF: Identifying Density-Based Local Outliers"
- Liu et al. (2008) "Isolation Forest"
- Freddie Mac Single-Family Loan-Level Dataset

## License

This project is for educational purposes as part of CS5344.
