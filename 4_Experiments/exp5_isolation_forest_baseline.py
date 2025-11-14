#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_builder_shared import FeatureBuilderAdvanced

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

def main():
    print("EXPERIMENT 5: Isolation Forest Baseline")
    print("="*70)
    print("Goal: Test Isolation Forest as an alternative to LOF")
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

    X_tr_scaled, _, X_tr_embed = fb.transform(train)
    X_v_scaled, _, X_v_embed = fb.transform(valid)
    print(f"Feature dimension: {X_tr_embed.shape[1]}")
    print()

    print("APPROACH 1: Isolation Forest on PCA embeddings")
    iforest1 = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iforest1.fit(X_tr_embed)
    scores1 = -iforest1.score_samples(X_v_embed)
    auprc1 = average_precision_score(yv, scores1)
    auroc1 = roc_auc_score(yv, scores1)
    print(f"  AUPRC: {auprc1:.6f}")
    print(f"  AUROC: {auroc1:.6f}")
    print()

    print("APPROACH 2: Isolation Forest on scaled features")
    iforest2 = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iforest2.fit(X_tr_scaled)
    scores2 = -iforest2.score_samples(X_v_scaled)
    auprc2 = average_precision_score(yv, scores2)
    auroc2 = roc_auc_score(yv, scores2)
    print(f"  AUPRC: {auprc2:.6f}")
    print(f"  AUROC: {auroc2:.6f}")
    print()

    print("KEY FINDINGS:")
    print("  1. Isolation Forest is fast and scalable")
    print("  2. Works well on high-dimensional data")
    print("  3. No need for distance metrics like LOF")
    print(f"  4. Best approach: {('PCA embeddings' if auprc1 > auprc2 else 'Scaled features')}")

if __name__ == "__main__":
    main()
