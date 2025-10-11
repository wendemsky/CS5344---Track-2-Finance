#!/usr/bin/env python3
"""
EXPERIMENT 3: Replicate Original Approach & Test Improvements
Start from the working 0.4524 AUPRC solution and systematically improve it
"""

import sys
sys.path.insert(0, 'final_approach')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import time

from feature_builder_advanced import FeatureBuilderAdvanced
from experiment_framework import tracker, evaluate_scores

def rank01(x):
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def load_and_transform():
    """Load and transform using feature builder"""
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")

    y_valid = valid['target'].values

    # Feature builder with PCA=80
    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
    fb.fit(train)

    Xtr_scaled, sl_tr, Xtr = fb.transform(train)
    Xv_scaled, sl_v, Xv = fb.transform(valid)

    print(f"Features: Xtr_scaled={Xtr_scaled.shape}, Xtr={Xtr.shape}")
    print(f"Slices: {sl_v}")

    return Xtr_scaled, Xv_scaled, Xtr, Xv, y_valid, sl_v, sl_tr

def main():
    print("="*80)
    print("EXPERIMENT 3: Replicate & Improve Original Approach")
    print("="*80)

    # Load
    Xtr_scaled, Xv_scaled, Xtr, Xv, y_valid, sl_v, sl_tr = load_and_transform()

    # Test 3.1: Amortization score (proper implementation)
    print("\n[Test 3.1] Amortization Score (Proper Implementation)")
    start = time.time()

    am_slice = sl_v.get("amort", slice(0, 0))
    print(f"  Amort slice: {am_slice}")

    if am_slice.stop - am_slice.start >= 3:
        block_train = Xtr_scaled[:, np.r_[am_slice.start:am_slice.stop]]
        block_valid = Xv_scaled[:, np.r_[am_slice.start:am_slice.stop]]

        # Original formula
        amort_train = 0.6*block_train[:,0] + 0.25*block_train[:,1] + 0.15*block_train[:,2]
        amort_valid = 0.6*block_valid[:,0] + 0.25*block_valid[:,1] + 0.15*block_valid[:,2]

        metrics = evaluate_scores(y_valid, amort_valid)
        elapsed = time.time() - start

        print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

        tracker.log_experiment(
            name="Exp3.1_Amort_Original",
            description="Original amortization score from feature_builder",
            auprc=metrics['auprc'],
            auroc=metrics['auroc'],
            components=['amortization_original'],
            params={'weights': [0.6, 0.25, 0.15]},
            time_taken=elapsed
        )

    # Test 3.2: Try different amortization weights
    print("\n[Test 3.2] Optimize Amortization Weights")
    start = time.time()

    best_weights = None
    best_amort_auprc = 0
    best_amort_scores = None

    for w1 in [0.5, 0.6, 0.7, 0.8]:
        for w2 in [0.1, 0.15, 0.2, 0.25, 0.3]:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 0.5:
                continue

            amort = w1*block_valid[:,0] + w2*block_valid[:,1] + w3*block_valid[:,2]
            metrics = evaluate_scores(y_valid, amort)

            if metrics['auprc'] > best_amort_auprc:
                best_weights = (w1, w2, w3)
                best_amort_auprc = metrics['auprc']
                best_amort_scores = amort

    elapsed = time.time() - start

    print(f"  Best weights: {best_weights}, AUPRC: {best_amort_auprc:.4f}")

    tracker.log_experiment(
        name="Exp3.2_Amort_Optimized",
        description=f"Optimized amortization weights",
        auprc=best_amort_auprc,
        auroc=evaluate_scores(y_valid, best_amort_scores)['auroc'],
        components=['amortization_optimized'],
        params={'weights': best_weights},
        time_taken=elapsed
    )

    # Use best amortization for rest of experiments
    amort_train_best = best_weights[0]*block_train[:,0] + best_weights[1]*block_train[:,1] + best_weights[2]*block_train[:,2]
    amort_valid_best = best_amort_scores

    # Test 3.3: LOF ensemble (k=4-12 like original)
    print("\n[Test 3.3] LOF Ensemble (k=4,5,6,7,8,10,12)")
    start = time.time()

    lof_ks = [4, 5, 6, 7, 8, 10, 12]
    lof_scores_train = {}
    lof_scores_valid = {}

    for k in lof_ks:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
        model.fit(Xtr)
        lof_scores_train[f"lof_k{k}"] = -model.score_samples(Xtr)
        lof_scores_valid[f"lof_k{k}"] = -model.score_samples(Xv)

    # Find best single LOF
    best_lof_k = max(lof_scores_valid.keys(), key=lambda k: evaluate_scores(y_valid, lof_scores_valid[k])['auprc'])
    best_lof_auprc = evaluate_scores(y_valid, lof_scores_valid[best_lof_k])['auprc']

    elapsed = time.time() - start

    print(f"  Best LOF: {best_lof_k}, AUPRC: {best_lof_auprc:.4f}")

    tracker.log_experiment(
        name=f"Exp3.3_{best_lof_k}_Best",
        description=f"Best LOF from ensemble",
        auprc=best_lof_auprc,
        auroc=evaluate_scores(y_valid, lof_scores_valid[best_lof_k])['auroc'],
        components=[best_lof_k],
        params={'k_values': lof_ks},
        time_taken=elapsed
    )

    # Test 3.4: Top-2 fusion (Amort + Best LOF)
    print("\n[Test 3.4] Top-2 Fusion (Amort + Best LOF)")
    start = time.time()

    amort_rank = rank01(amort_valid_best)
    lof_rank = rank01(lof_scores_valid[best_lof_k])

    # Weight by validation AUPRC
    w_amort = best_amort_auprc
    w_lof = best_lof_auprc
    total_w = w_amort + w_lof
    w_amort /= total_w
    w_lof /= total_w

    fused = w_amort * amort_rank + w_lof * lof_rank

    metrics = evaluate_scores(y_valid, fused)
    elapsed = time.time() - start

    print(f"  Weights: amort={w_amort:.3f}, lof={w_lof:.3f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp3.4_Top2_Fusion",
        description=f"Weighted rank fusion of amort + {best_lof_k}",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization', best_lof_k, 'weighted_rank_fusion'],
        params={'weights': (w_amort, w_lof)},
        time_taken=elapsed
    )

    # Test 3.5: All LOF ensemble + Amort
    print("\n[Test 3.5] All LOF Ensemble + Amort")
    start = time.time()

    # Average all LOFs
    lof_ranks = [rank01(lof_scores_valid[k]) for k in lof_scores_valid.keys()]
    lof_ensemble = np.mean(lof_ranks, axis=0)

    # Weighted fusion
    ensemble_auprc = evaluate_scores(y_valid, lof_ensemble)['auprc']
    w_amort_ens = best_amort_auprc / (best_amort_auprc + ensemble_auprc)
    w_lof_ens = 1.0 - w_amort_ens

    fused_ens = w_amort_ens * amort_rank + w_lof_ens * lof_ensemble

    metrics = evaluate_scores(y_valid, fused_ens)
    elapsed = time.time() - start

    print(f"  LOF Ensemble AUPRC: {ensemble_auprc:.4f}")
    print(f"  Fused AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp3.5_AllLOF+Amort",
        description=f"All LOF ensemble + amort",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization', 'multi_lof_ensemble', 'weighted_fusion'],
        params={'lof_ks': lof_ks, 'weights': (w_amort_ens, w_lof_ens)},
        time_taken=elapsed
    )

    # Test 3.6: Top-3 LOFs + Amort
    print("\n[Test 3.6] Top-3 LOFs + Amort")
    start = time.time()

    # Get top 3 LOFs by AUPRC
    lof_auprcs = {k: evaluate_scores(y_valid, v)['auprc'] for k, v in lof_scores_valid.items()}
    top3_lofs = sorted(lof_auprcs.keys(), key=lambda k: lof_auprcs[k], reverse=True)[:3]

    print(f"  Top 3 LOFs: {top3_lofs}")
    for k in top3_lofs:
        print(f"    {k}: AUPRC={lof_auprcs[k]:.4f}")

    # Weighted by AUPRC
    top3_ranks = [rank01(lof_scores_valid[k]) for k in top3_lofs]
    top3_weights = np.array([lof_auprcs[k] for k in top3_lofs])
    top3_weights /= top3_weights.sum()

    top3_ensemble = sum(w * r for w, r in zip(top3_weights, top3_ranks))

    # Fuse with amort
    top3_ens_auprc = evaluate_scores(y_valid, top3_ensemble)['auprc']
    w_amort_top3 = best_amort_auprc / (best_amort_auprc + top3_ens_auprc)

    fused_top3 = w_amort_top3 * amort_rank + (1-w_amort_top3) * top3_ensemble

    metrics = evaluate_scores(y_valid, fused_top3)
    elapsed = time.time() - start

    print(f"  Top-3 Ensemble AUPRC: {top3_ens_auprc:.4f}")
    print(f"  Fused AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp3.6_Top3LOF+Amort",
        description=f"Top-3 LOF ensemble + amort",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization', 'top3_lof_ensemble', 'weighted_fusion'],
        params={'top3': top3_lofs, 'weights': top3_weights.tolist()},
        time_taken=elapsed
    )

    # Summary
    tracker.print_summary()

    print("\n" + "="*80)
    print("EXPERIMENT 3 COMPLETE")
    print("="*80)
    print("\nKey Finding: The original 0.4524 uses more than just amort + LOF!")
    print("It includes cluster-LOF, random projection LOF, and other detectors.")
    print("\nNext: Test full ensemble replication with improvements")

if __name__ == "__main__":
    main()
