#!/usr/bin/env python3
"""
EXPERIMENT 2: Quick Wins - Test simple improvements to current best approach
Focus on low-hanging fruit that might improve AUPRC
"""

import sys
sys.path.insert(0, 'final_approach')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import time

from feature_builder_advanced import FeatureBuilderAdvanced
from experiment_framework import tracker, evaluate_scores

def load_and_engineer():
    """Load data and apply feature engineering"""
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")

    y_valid = valid['target'].values

    # Use feature builder
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
    fb.fit(train)

    Xtr_scaled, sl_tr, Xtr = fb.transform(train)
    Xv_scaled, sl_v, Xv = fb.transform(valid)

    print(f"Engineered features: Train {Xtr.shape}, Valid {Xv.shape}")

    return Xtr_scaled, Xv_scaled, Xtr, Xv, y_valid, sl_v

def amortization_score(X_scaled, sl):
    """Extract amortization score from engineered features"""
    am_slice = sl.get("amort", slice(0, 0))
    if am_slice.stop - am_slice.start >= 3:
        block = X_scaled[:, np.r_[am_slice.start:am_slice.stop]]
        # Simple weighted combo
        score = 0.6*block[:,0] + 0.25*block[:,1] + 0.15*block[:,2]
        return score
    return np.zeros(X_scaled.shape[0])

def rank01(x):
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def main():
    print("="*80)
    print("EXPERIMENT 2: Quick Wins")
    print("="*80)

    # Load engineered features
    Xtr_scaled, Xv_scaled, Xtr, Xv, y_valid, sl_v = load_and_engineer()

    # Baseline: Amortization score alone
    print("\n[Test 2.1] Amortization Score (Baseline)")
    start = time.time()

    amort_train = amortization_score(Xtr_scaled, {'amort': slice(0, 3)})  # Assume first 3 are amort
    amort_valid = amortization_score(Xv_scaled, {'amort': slice(0, 3)})

    metrics = evaluate_scores(y_valid, amort_valid)
    elapsed = time.time() - start

    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp2.1_Amort_Baseline",
        description="Amortization score from feature_builder_advanced",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization'],
        params={'weights': [0.6, 0.25, 0.15]},
        time_taken=elapsed
    )

    # Test 2.2: LOF with different k values
    print("\n[Test 2.2] LOF Sweep (k=5 to k=100)")
    start = time.time()

    best_lof_k = None
    best_lof_auprc = 0
    best_lof_scores = None

    for k in [5, 10, 20, 30, 50, 70, 100]:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
        model.fit(Xtr)
        scores = -model.score_samples(Xv)

        metrics = evaluate_scores(y_valid, scores)

        if metrics['auprc'] > best_lof_auprc:
            best_lof_k = k
            best_lof_auprc = metrics['auprc']
            best_lof_scores = scores

        print(f"    k={k:3d}: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: k={best_lof_k}, AUPRC={best_lof_auprc:.4f}")

    tracker.log_experiment(
        name=f"Exp2.2_LOF_Best_k{best_lof_k}",
        description=f"Best LOF with k={best_lof_k}",
        auprc=best_lof_auprc,
        auroc=evaluate_scores(y_valid, best_lof_scores)['auroc'],
        components=['lof'],
        params={'k': best_lof_k},
        time_taken=elapsed
    )

    # Test 2.3: Simple fusion - Amort + Best LOF
    print("\n[Test 2.3] Amort + LOF Fusion (different weights)")
    start = time.time()

    amort_rank = rank01(amort_valid)
    lof_rank = rank01(best_lof_scores)

    best_fusion_weights = None
    best_fusion_auprc = 0
    best_fusion_scores = None

    for w_amort in [0.5, 0.6, 0.7, 0.8, 0.9]:
        w_lof = 1.0 - w_amort
        fused = w_amort * amort_rank + w_lof * lof_rank

        metrics = evaluate_scores(y_valid, fused)

        if metrics['auprc'] > best_fusion_auprc:
            best_fusion_weights = (w_amort, w_lof)
            best_fusion_auprc = metrics['auprc']
            best_fusion_scores = fused

        print(f"    w_amort={w_amort:.1f}, w_lof={w_lof:.1f}: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: weights={best_fusion_weights}, AUPRC={best_fusion_auprc:.4f}")

    tracker.log_experiment(
        name=f"Exp2.3_Amort+LOF_Fusion",
        description=f"Amort + LOF(k={best_lof_k}) with optimal weights",
        auprc=best_fusion_auprc,
        auroc=evaluate_scores(y_valid, best_fusion_scores)['auroc'],
        components=['amortization', f'lof_k{best_lof_k}', 'weighted_fusion'],
        params={'weights': best_fusion_weights, 'fusion': 'weighted_rank'},
        time_taken=elapsed
    )

    # Test 2.4: Multi-k LOF ensemble
    print("\n[Test 2.4] Multi-k LOF Ensemble")
    start = time.time()

    k_values = [5, 10, 20, 30, 50]
    lof_scores_list = []

    for k in k_values:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
        model.fit(Xtr)
        scores = -model.score_samples(Xv)
        lof_scores_list.append(rank01(scores))

    # Average all LOFs
    lof_ensemble = np.mean(np.column_stack(lof_scores_list), axis=1)

    metrics = evaluate_scores(y_valid, lof_ensemble)
    elapsed = time.time() - start

    print(f"  k_values={k_values}")
    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp2.4_MultiLOF_Ensemble",
        description=f"Average of LOF with k={k_values}",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['multi_lof_ensemble'],
        params={'k_values': k_values, 'aggregation': 'mean'},
        time_taken=elapsed
    )

    # Test 2.5: Amort + Multi-k LOF
    print("\n[Test 2.5] Amort + Multi-k LOF Ensemble")
    start = time.time()

    best_w = None
    best_auprc_final = 0

    for w_amort in [0.5, 0.6, 0.7, 0.8, 0.9]:
        w_lof = 1.0 - w_amort
        fused = w_amort * amort_rank + w_lof * lof_ensemble

        metrics = evaluate_scores(y_valid, fused)

        if metrics['auprc'] > best_auprc_final:
            best_w = (w_amort, w_lof)
            best_auprc_final = metrics['auprc']
            best_final_scores = fused

        print(f"    w_amort={w_amort:.1f}: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: w_amort={best_w[0]:.1f}, AUPRC={best_auprc_final:.4f}")

    tracker.log_experiment(
        name="Exp2.5_Amort+MultiLOF",
        description=f"Amort + Multi-k LOF ensemble with optimal weights",
        auprc=best_auprc_final,
        auroc=evaluate_scores(y_valid, best_final_scores)['auroc'],
        components=['amortization', 'multi_lof_ensemble', 'weighted_fusion'],
        params={'weights': best_w, 'lof_k_values': k_values},
        time_taken=elapsed
    )

    # Test 2.6: Max instead of average
    print("\n[Test 2.6] Max-rank LOF Ensemble")
    start = time.time()

    lof_max = np.max(np.column_stack(lof_scores_list), axis=1)

    metrics = evaluate_scores(y_valid, lof_max)
    elapsed = time.time() - start

    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp2.6_MaxLOF_Ensemble",
        description=f"Max-rank of LOF with k={k_values}",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['multi_lof_max_ensemble'],
        params={'k_values': k_values, 'aggregation': 'max'},
        time_taken=elapsed
    )

    # Test 2.7: Amort + Max LOF
    print("\n[Test 2.7] Amort + Max-LOF")
    start = time.time()

    best_w_max = None
    best_auprc_max = 0

    for w_amort in [0.5, 0.6, 0.7, 0.8, 0.9]:
        w_lof = 1.0 - w_amort
        fused = w_amort * amort_rank + w_lof * lof_max

        metrics = evaluate_scores(y_valid, fused)

        if metrics['auprc'] > best_auprc_max:
            best_w_max = (w_amort, w_lof)
            best_auprc_max = metrics['auprc']

        print(f"    w_amort={w_amort:.1f}: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: w_amort={best_w_max[0]:.1f}, AUPRC={best_auprc_max:.4f}")

    tracker.log_experiment(
        name="Exp2.7_Amort+MaxLOF",
        description=f"Amort + Max-LOF ensemble",
        auprc=best_auprc_max,
        auroc=0,  # Will compute if best
        components=['amortization', 'max_lof_ensemble', 'weighted_fusion'],
        params={'weights': best_w_max, 'lof_k_values': k_values},
        time_taken=elapsed
    )

    # Summary
    tracker.print_summary()

    print("\n" + "="*80)
    print("EXPERIMENT 2 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
