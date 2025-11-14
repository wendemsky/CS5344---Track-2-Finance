#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_builder_shared import FeatureBuilderAdvanced
from experiment_utils import rank01

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

def main():
    print("EXPERIMENT 7: Simple Ensemble Fusion Strategies")
    print("="*70)
    print("Goal: Compare different fusion methods for combining detectors")
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

    X_tr_scaled, slices_tr, X_tr_embed = fb.transform(train)
    X_v_scaled, slices_v, X_v_embed = fb.transform(valid)
    print()

    print("Training detectors...")

    lof = LocalOutlierFactor(n_neighbors=50, novelty=True)
    lof.fit(X_tr_embed)
    scores_lof = -lof.score_samples(X_v_embed)
    auprc_lof = average_precision_score(yv, scores_lof)
    print(f"  LOF: AUPRC={auprc_lof:.6f}")

    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    iforest.fit(X_tr_embed)
    scores_if = -iforest.score_samples(X_v_embed)
    auprc_if = average_precision_score(yv, scores_if)
    print(f"  IForest: AUPRC={auprc_if:.6f}")

    amort_slice = slices_v['amort']
    scores_amort = X_v_scaled[:, amort_slice][:, 0]
    auprc_amort = average_precision_score(yv, scores_amort)
    print(f"  Amortization: AUPRC={auprc_amort:.6f}")
    print()

    print("Testing fusion strategies:")

    rank_lof = rank01(scores_lof)
    rank_if = rank01(scores_if)
    rank_amort = rank01(scores_amort)

    fusion_avg = (rank_lof + rank_if + rank_amort) / 3
    auprc_avg = average_precision_score(yv, fusion_avg)
    print(f"  1. Average: AUPRC={auprc_avg:.6f}")

    fusion_max = np.maximum(np.maximum(rank_lof, rank_if), rank_amort)
    auprc_max = average_precision_score(yv, fusion_max)
    print(f"  2. Maximum: AUPRC={auprc_max:.6f}")

    w1 = auprc_lof
    w2 = auprc_if
    w3 = auprc_amort
    fusion_weighted = (w1*rank_lof + w2*rank_if + w3*rank_amort) / (w1 + w2 + w3)
    auprc_weighted = average_precision_score(yv, fusion_weighted)
    print(f"  3. Weighted (by AUPRC): AUPRC={auprc_weighted:.6f}")

    fusion_best_only = rank_amort
    auprc_best = average_precision_score(yv, fusion_best_only)
    print(f"  4. Best detector only: AUPRC={auprc_best:.6f}")
    print()

    results = [
        ("Average", auprc_avg),
        ("Maximum", auprc_max),
        ("Weighted", auprc_weighted),
        ("Best only", auprc_best)
    ]
    best_method, best_score = max(results, key=lambda x: x[1])

    print(f"BEST FUSION: {best_method} with AUPRC={best_score:.6f}")
    print()

    print("KEY FINDINGS:")
    print("  1. Simple average can work well when detectors are diverse")
    print("  2. Weighted average gives more influence to better detectors")
    print("  3. Maximum captures extreme signals from any detector")
    print("  4. Sometimes the best single detector outperforms fusion")

if __name__ == "__main__":
    main()
