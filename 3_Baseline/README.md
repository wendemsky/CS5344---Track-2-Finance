# Baseline Models

## How to Run

```bash
python 3_Baseline/baseline_models.py
```

## Results

10 unsupervised anomaly detection algorithms tested across 3 preprocessing configurations (27 total variants).

### Top 5 Models by AUPRC

| Model | Config | AUPRC | AUROC | F1 |
|-------|--------|-------|-------|-----|
| LOF(k=50) | robust_pcaNone | 0.1955 | 0.5648 | 0.2366 |
| Random Projection LOF | robust_pcaNone | 0.1939 | 0.5630 | 0.2399 |
| KNN Distance(k=5) | robust_pcaNone | 0.1882 | 0.5540 | 0.2357 |
| OneClassSVM(rbf, nu=0.1) | robust_pcaNone | 0.1848 | 0.5556 | 0.2360 |
| OneClassSVM(rbf, nu=0.1) | standard_pca80 | 0.1841 | 0.5375 | 0.2312 |

### Best Preprocessing Configuration

Robust scaling without PCA performed best overall:
- Avg AUPRC: 0.1755
- Preserves anomaly patterns better than dimensionality reduction

### Algorithms Tested

1. Isolation Forest
2. Local Outlier Factor (LOF)
3. One-Class SVM
4. Elliptic Envelope
5. MLP Autoencoder (optional, requires TensorFlow)
6. DBSCAN
7. K-Nearest Neighbors Distance
8. PCA Reconstruction Error
9. Random Projection LOF Ensemble
10. Mahalanobis Distance

### Key Findings

- LOF-based methods dominate (LOF, Random Projection LOF)
- Robust scaling outperforms standard scaling
- PCA dimensionality reduction hurts performance
- Ensemble methods (Random Projection) improve results

## Outputs

Results saved to `3_Baseline/results/`:
- `baseline_results.csv` - Complete results
- `baseline_results.json` - JSON format
- Visualization charts
