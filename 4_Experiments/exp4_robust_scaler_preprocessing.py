#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_builder_shared import FeatureBuilderAdvanced

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

def main():
    print("EXPERIMENT 4: RobustScaler vs StandardScaler Preprocessing")
    print("="*70)
    print("Goal: Compare impact of different scaling approaches on LOF performance")
    print()

    train = pd.read_csv("../1_Data/loans_train.csv")
    valid = pd.read_csv("../1_Data/loans_valid.csv")

    yv = valid["target"].values
    print(f"Data loaded: Train={train.shape}, Valid={valid.shape}")
    print(f"Validation anomaly rate: {yv.mean():.2%}")
    print()

    numeric_cols = train.select_dtypes(include=[np.number]).columns
    exclude_cols = ['target', 'Id', 'index']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X_tr_raw = train[feature_cols].fillna(0).values
    X_v_raw = valid[feature_cols].fillna(0).values

    print("APPROACH 1: StandardScaler (sensitive to outliers)")
    scaler1 = StandardScaler()
    X_tr_std = scaler1.fit_transform(X_tr_raw)
    X_v_std = scaler1.transform(X_v_raw)

    lof1 = LocalOutlierFactor(n_neighbors=50, novelty=True)
    lof1.fit(X_tr_std)
    scores1 = -lof1.score_samples(X_v_std)
    auprc1 = average_precision_score(yv, scores1)
    auroc1 = roc_auc_score(yv, scores1)
    print(f"  AUPRC: {auprc1:.6f}")
    print(f"  AUROC: {auroc1:.6f}")
    print()

    print("APPROACH 2: RobustScaler (robust to outliers)")
    scaler2 = RobustScaler()
    X_tr_rob = scaler2.fit_transform(X_tr_raw)
    X_v_rob = scaler2.transform(X_v_raw)

    lof2 = LocalOutlierFactor(n_neighbors=50, novelty=True)
    lof2.fit(X_tr_rob)
    scores2 = -lof2.score_samples(X_v_rob)
    auprc2 = average_precision_score(yv, scores2)
    auroc2 = roc_auc_score(yv, scores2)
    print(f"  AUPRC: {auprc2:.6f}")
    print(f"  AUROC: {auroc2:.6f}")
    print()

    improvement = ((auprc2 / auprc1) - 1) * 100
    print(f"IMPROVEMENT: {improvement:+.1f}%")
    print()

    print("KEY FINDINGS:")
    print("  1. RobustScaler uses median and IQR instead of mean and std")
    print("  2. More robust to outliers in the training data")
    print("  3. Can improve detector performance on skewed distributions")
    print(f"  4. In this case: {('RobustScaler wins' if auprc2 > auprc1 else 'StandardScaler wins')}")

if __name__ == "__main__":
    main()
