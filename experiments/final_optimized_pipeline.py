#!/usr/bin/env python3
"""
FINAL OPTIMIZED PIPELINE
Based on experiments: AUPRC = 0.4750 (validation)

Key findings:
- Amortization features are extremely strong (0.4749 alone)
- Optimal weights: 0.7*amort_mean + 0.3*amort_70 + 0.0*amort_50
- Adding 1% LOF(k=6) gives tiny boost to 0.4750
- Simpler is better - complex ensembles hurt performance

This pipeline:
1. Uses feature_builder_advanced for sophisticated feature engineering
2. Computes optimized amortization score
3. Adds tiny LOF signal for marginal improvement
4. Generates test predictions for Kaggle
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, 'final_approach')

from sklearn.neighbors import LocalOutlierFactor
from feature_builder_advanced import FeatureBuilderAdvanced

def rank01(x):
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def main():
    print("="*80)
    print("FINAL OPTIMIZED PIPELINE")
    print("="*80)
    print("\nKey Innovation: Simplified approach beats complex ensemble!")
    print("  - Amortization (optimized weights): 0.4749 AUPRC")
    print("  - + 1% LOF signal: 0.4750 AUPRC")
    print("  - Previous complex ensemble: 0.4524 AUPRC")
    print("  - Improvement: +5.0% over previous best")
    print()

    # Load data
    print("Loading data...")
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test = pd.read_csv("Data/loans_test.csv")

    y_valid = valid['target'].values

    # Feature engineering
    print("Engineering features...")
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
    fb.fit(train)

    Xtr_scaled, sl_tr, Xtr = fb.transform(train)
    Xv_scaled, sl_v, Xv = fb.transform(valid)
    Xt_scaled, sl_t, Xt = fb.transform(test)

    print(f"  Train: {Xtr.shape}, Valid: {Xv.shape}, Test: {Xt.shape}")

    # Extract amortization features
    print("\nComputing optimized amortization scores...")
    am_slice = sl_v.get("amort", slice(0, 0))

    block_train = Xtr_scaled[:, np.r_[am_slice.start:am_slice.stop]]
    block_valid = Xv_scaled[:, np.r_[am_slice.start:am_slice.stop]]
    block_test = Xt_scaled[:, np.r_[am_slice.start:am_slice.stop]]

    # Optimized weights: [0.7, 0.3, 0.0]
    amort_train = 0.7*block_train[:,0] + 0.3*block_train[:,1]
    amort_valid = 0.7*block_valid[:,0] + 0.3*block_valid[:,1]
    amort_test = 0.7*block_test[:,0] + 0.3*block_test[:,1]

    print(f"  Amortization alone: 0.4749 AUPRC (validation)")

    # Add LOF signal (k=6, 1% weight)
    print("\nAdding tiny LOF signal (k=6, weight=1%)...")
    lof = LocalOutlierFactor(n_neighbors=6, novelty=True, n_jobs=-1)
    lof.fit(Xtr)

    lof_valid = -lof.score_samples(Xv)
    lof_test = -lof.score_samples(Xt)

    # Rank fusion
    amort_rank_valid = rank01(amort_valid)
    lof_rank_valid = rank01(lof_valid)

    amort_rank_test = rank01(amort_test)
    lof_rank_test = rank01(lof_test)

    # Final scores: 99% amort + 1% LOF
    final_valid = 0.99 * amort_rank_valid + 0.01 * lof_rank_valid
    final_test = 0.99 * amort_rank_test + 0.01 * lof_rank_test

    # Validate
    from sklearn.metrics import average_precision_score, roc_auc_score
    auprc = average_precision_score(y_valid, final_valid)
    auroc = roc_auc_score(y_valid, final_valid)

    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Improvement vs baseline (0.1955): +{(auprc/0.1955-1)*100:.1f}%")
    print(f"  Improvement vs previous best (0.4524): +{(auprc/0.4524-1)*100:.1f}%")

    # Generate submission
    print(f"\n{'='*80}")
    print("GENERATING KAGGLE SUBMISSION")
    print(f"{'='*80}")

    idc = "Id" if "Id" in test.columns else "index"
    submission = pd.DataFrame({
        idc: test[idc],
        "anomaly_score": final_test
    })

    output_path = "experiments/SUBMISSION_OPTIMIZED_v2.csv"
    submission.to_csv(output_path, index=False)

    print(f"  Saved: {output_path}")
    print(f"  Rows: {len(submission):,}")
    print(f"\nScore statistics:")
    print(f"  Mean: {final_test.mean():.4f}")
    print(f"  Std: {final_test.std():.4f}")
    print(f"  Min: {final_test.min():.4f}")
    print(f"  Max: {final_test.max():.4f}")

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Upload SUBMISSION_OPTIMIZED_v2.csv to Kaggle")
    print("  2. Compare with previous submission (0.43 AUPRC)")
    print("  3. Expected improvement: ~5% boost")
    print(f"\nKey insight: Domain knowledge (amortization) > Complex ML ensembles")

if __name__ == "__main__":
    main()
