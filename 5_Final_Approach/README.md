# Final Approach

## How to Run

```bash
python 5_Final_Approach/final_model.py
```

Runtime: Approximately 10-15 minutes

## Results

### Validation Performance
- AUPRC: 0.4981
- AUROC: 0.7865

### Per-Detector Performance
- Amortization: AUPRC = 0.4748
- Cohort LOF: AUPRC = 0.3242
- Random Projection LOF: AUPRC = 0.3208
- RFOD Temporal TopQ: AUPRC = 0.2891
- OCSVM: AUPRC = 0.1830

### Fusion Configuration
- Strategy: Weighted gate with amortization threshold
- Selected detectors: 6 (amort, cohort_lof, rp_lof, ocsvm, rfod_temporal, rfod_temporal_topq)
- Best fusion: wgate::amort_0.40::tau0.95

### Kaggle Performance
- AUPRC: 0.478
- Improvement over baseline: +145%

## Methodology

### Architecture

Ultra Hybrid RFOD-AGD-UWA v4 ensemble combining multiple advanced anomaly detectors:

1. **Feature Engineering** (feature_builder.py)
   - Robust scaling for outlier handling
   - PCA dimensionality reduction (80 components)
   - Domain-specific amortization features
   - Temporal payment patterns

2. **Core Detectors**
   - RFOD (Random Forest Outlier Detection) - Temporal variant
   - AGD (Amortization Gate Detector) - Domain-specific
   - UWA (Unified Weighted Aggregation)
   - Cluster-wise LOF (CLOF)
   - One-Class SVM on PCA embeddings
   - Random Projection LOF ensemble

3. **Calibration**
   - Train-only CDF calibration
   - EVT (Extreme Value Theory) tail modeling
   - Gamma parameter tuning

4. **Fusion**
   - Inverse covariance weighting
   - Amortization-gated ensemble
   - Multi-threshold fusion strategies

### Key Innovation

The model achieves high AUPRC (0.43) through:
- Strong amortization-based anomaly signals
- Multi-scale density estimation (LOF variants)
- Robust ensemble fusion with calibration
- Train-only fitting (no validation leakage)

## Files

- `final_model.py` - Main ensemble pipeline
- `feature_builder.py` - Feature engineering module
- `submission.csv` - Generated Kaggle submission file

## Dependencies

- numpy
- pandas
- scikit-learn
- scipy
