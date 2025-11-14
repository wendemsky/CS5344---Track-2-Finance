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
    print("EXPERIMENT 3: Enhanced Domain Feature Engineering")
    print("="*70)
    print("Goal: Compare LOF with raw vs engineered features")
    print()

    train = pd.read_csv("../1_Data/loans_train.csv")
    valid = pd.read_csv("../1_Data/loans_valid.csv")

    yv = valid["target"].values
    print(f"Data loaded: Train={train.shape}, Valid={valid.shape}")
    print()

    numeric_cols = train.select_dtypes(include=[np.number]).columns
    exclude_cols = ['target', 'Id', 'index']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X_tr_raw = train[feature_cols].fillna(0).values
    X_v_raw = valid[feature_cols].fillna(0).values

    print("BASELINE: Raw Features + LOF")
    lof1 = LocalOutlierFactor(n_neighbors=50, novelty=True)
    lof1.fit(X_tr_raw)
    scores1 = -lof1.score_samples(X_v_raw)
    auprc1 = average_precision_score(yv, scores1)
    print(f"  AUPRC: {auprc1:.6f}")
    print()

    print("WITH FEATURE ENGINEERING: ")
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
    fb.fit(train)
    _, _, X_tr_embed = fb.transform(train)
    _, _, X_v_embed = fb.transform(valid)

    lof2 = LocalOutlierFactor(n_neighbors=50, novelty=True)
    lof2.fit(X_tr_embed)
    scores2 = -lof2.score_samples(X_v_embed)
    auprc2 = average_precision_score(yv, scores2)
    print(f"  AUPRC: {auprc2:.6f}")
    print()

    improvement = ((auprc2 / auprc1) - 1) * 100
    print(f"IMPROVEMENT: {improvement:+.1f}%")

if __name__ == "__main__":
    main()
