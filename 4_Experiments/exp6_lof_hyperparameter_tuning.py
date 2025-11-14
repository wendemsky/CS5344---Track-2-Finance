#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_builder_shared import FeatureBuilderAdvanced

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

def main():
    print("EXPERIMENT 6: LOF Hyperparameter Tuning")
    print("="*70)
    print("Goal: Test different k values for LOF to find optimal neighbors")
    print()

    train = pd.read_csv("../1_Data/loans_train.csv")
    valid = pd.read_csv("../1_Data/loans_valid.csv")

    yv = valid["target"].values
    print(f"Data loaded: Train={train.shape}, Valid={valid.shape}")
    print(f"Validation anomaly rate: {yv.mean():.2%}")
    print()

    print("Building features...")
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
    fb.fit(train)

    _, _, X_tr_embed = fb.transform(train)
    _, _, X_v_embed = fb.transform(valid)
    print(f"Feature dimension: {X_tr_embed.shape[1]}")
    print()

    print("Testing different k values for LOF:")
    k_values = [5, 10, 20, 30, 50, 70, 100]
    results = []

    for k in k_values:
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
        lof.fit(X_tr_embed)
        scores = -lof.score_samples(X_v_embed)
        auprc = average_precision_score(yv, scores)
        auroc = roc_auc_score(yv, scores)
        results.append((k, auprc, auroc))
        print(f"  k={k:3d}: AUPRC={auprc:.6f}, AUROC={auroc:.6f}")

    print()
    best_k, best_auprc, best_auroc = max(results, key=lambda x: x[1])
    print(f"BEST CONFIGURATION:")
    print(f"  k={best_k}: AUPRC={best_auprc:.6f}, AUROC={best_auroc:.6f}")
    print()

    print("KEY FINDINGS:")
    print("  1. Smaller k: More sensitive to local structure, higher variance")
    print("  2. Larger k: More global view, more stable but may miss local patterns")
    print("  3. Optimal k depends on data characteristics and anomaly types")
    print(f"  4. For this dataset: k={best_k} performs best")

if __name__ == "__main__":
    main()
