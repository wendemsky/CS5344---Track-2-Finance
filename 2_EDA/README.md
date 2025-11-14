# Exploratory Data Analysis

## How to Run

```bash
python 2_EDA/comprehensive_eda.py
```

## Key Findings

### Dataset Statistics
- Train: 30,504 records (100% normal loans)
- Valid: 5,370 records (12.61% anomalous)
- Test: 13,426 records
- Total features: 143 (129 numeric, 14 categorical)

### Class Imbalance
- Training set has no anomalies (unsupervised learning required)
- Validation set: 87.39% normal, 12.61% anomalous
- Strong class imbalance necessitates AUPRC as primary metric

### Missing Values
- 3 features with >99% missing values (dropped in preprocessing)
- Most features have minimal missing data

### Outliers
- 13.99% of records contain outliers (PropertyValMethod)
- Recommendation: Use RobustScaler for preprocessing

### Top Correlated Features with Target
- CreditScore (r = -0.250) - Strongest predictor
- NonInterestBearingUPB months 12-13 (r = +0.12-0.14)
- OriginalDTI (r = +0.100)
- OriginalInterestRate (r = +0.096)

### Recommendations
- Use unsupervised anomaly detection methods
- Apply robust scaling to handle outliers
- Engineer domain-specific features (amortization patterns)
- Evaluate with AUPRC as primary metric

## Outputs

Visualizations and analysis saved to `2_EDA/outputs/`:
- Class distribution
- Missing values analysis
- Feature distributions
- Correlation heatmap
- Outlier analysis
- Temporal patterns
