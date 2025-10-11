#!/usr/bin/env python3
"""
EXPERIMENT 4: Can we beat standalone amortization (0.4749)?
Test advanced fusion strategies and detector combinations
"""

import sys
sys.path.insert(0, 'final_approach')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import time

from feature_builder_advanced import FeatureBuilderAdvanced
from experiment_framework import tracker, evaluate_scores

def rank01(x):
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def load_data():
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    y_valid = valid['target'].values

    fb = FeatureBuilderAdvanced(use_pca=True, pca_comps=80)
    fb.fit(train)

    Xtr_scaled, sl_tr, Xtr = fb.transform(train)
    Xv_scaled, sl_v, Xv = fb.transform(valid)

    # Extract optimized amortization
    am_slice = sl_v.get("amort", slice(0, 0))
    block_train = Xtr_scaled[:, np.r_[am_slice.start:am_slice.stop]]
    block_valid = Xv_scaled[:, np.r_[am_slice.start:am_slice.stop]]

    amort_train = 0.7*block_train[:,0] + 0.3*block_train[:,1]
    amort_valid = 0.7*block_valid[:,0] + 0.3*block_valid[:,1]

    return Xtr, Xv, y_valid, amort_train, amort_valid

def main():
    print("="*80)
    print("EXPERIMENT 4: Beat Amortization 0.4749")
    print("="*80)

    Xtr, Xv, y_valid, amort_train, amort_valid = load_data()

    # Baseline
    print(f"\nBaseline: Amortization alone = 0.4749 AUPRC")

    # Test 4.1: Isotonic calibration on amortization
    print("\n[Test 4.1] Isotonic Calibration on Amortization")
    start = time.time()

    try:
        # Fit isotonic regression on amortization scores (train only, using same scores)
        iso = IsotonicRegression(out_of_bounds='clip')
        # Use quantiles of train as pseudo-probabilities
        train_ranks = rank01(amort_train)
        iso.fit(amort_train, train_ranks)

        # Apply to validation
        amort_calibrated = iso.predict(amort_valid)

        metrics = evaluate_scores(y_valid, amort_calibrated)
        elapsed = time.time() - start

        print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

        tracker.log_experiment(
            name="Exp4.1_Amort_Isotonic",
            description="Isotonic calibration on amortization scores",
            auprc=metrics['auprc'],
            auroc=metrics['auroc'],
            components=['amortization', 'isotonic_calibration'],
            params={'calibration': 'isotonic'},
            time_taken=elapsed
        )
    except Exception as e:
        print(f"  Failed: {e}")

    # Test 4.2: Add weakest LOF signal (very small weight)
    print("\n[Test 4.2] Amort + Tiny LOF Signal")
    start = time.time()

    # Train LOF(k=6) - best from previous
    lof = LocalOutlierFactor(n_neighbors=6, novelty=True, n_jobs=-1)
    lof.fit(Xtr)
    lof_scores = -lof.score_samples(Xv)

    amort_rank = rank01(amort_valid)
    lof_rank = rank01(lof_scores)

    best_weight = None
    best_auprc = 0

    for w_lof in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        w_amort = 1.0 - w_lof
        fused = w_amort * amort_rank + w_lof * lof_rank

        metrics = evaluate_scores(y_valid, fused)

        if metrics['auprc'] > best_auprc:
            best_weight = w_lof
            best_auprc = metrics['auprc']

        print(f"    w_lof={w_lof:.2f}: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: w_lof={best_weight:.2f}, AUPRC={best_auprc:.4f}")

    tracker.log_experiment(
        name="Exp4.2_Amort+TinyLOF",
        description=f"Amort + LOF with tiny weight ({best_weight})",
        auprc=best_auprc,
        auroc=0,
        components=['amortization', 'lof_k6', 'tiny_weight_fusion'],
        params={'w_lof': best_weight},
        time_taken=elapsed
    )

    # Test 4.3: Non-linear combination (product)
    print("\n[Test 4.3] Non-linear: Amort * LOF")
    start = time.time()

    # Normalize to [0,1]
    amort_norm = (amort_rank - amort_rank.min()) / (amort_rank.max() - amort_rank.min() + 1e-12)
    lof_norm = (lof_rank - lof_rank.min()) / (lof_rank.max() - lof_rank.min() + 1e-12)

    # Geometric mean
    geometric = np.sqrt(amort_norm * lof_norm)

    metrics = evaluate_scores(y_valid, geometric)
    elapsed = time.time() - start

    print(f"  Geometric mean AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp4.3_Geometric_Mean",
        description="Geometric mean of amort and LOF",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization', 'lof_k6', 'geometric_mean'],
        params={},
        time_taken=elapsed
    )

    # Test 4.4: Max(Amort, LOF)
    print("\n[Test 4.4] Max(Amort_rank, LOF_rank)")
    start = time.time()

    max_scores = np.maximum(amort_rank, lof_rank)

    metrics = evaluate_scores(y_valid, max_scores)
    elapsed = time.time() - start

    print(f"  Max AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp4.4_Max_Fusion",
        description="Max of amort and LOF ranks",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization', 'lof_k6', 'max_fusion'],
        params={},
        time_taken=elapsed
    )

    # Test 4.5: Power transformation on amortization
    print("\n[Test 4.5] Power Transform Amortization")
    start = time.time()

    best_power = None
    best_power_auprc = 0

    for power in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
        # Apply power to rank (normalized to [0,1])
        amort_powered = amort_rank ** power

        metrics = evaluate_scores(y_valid, amort_powered)

        if metrics['auprc'] > best_power_auprc:
            best_power = power
            best_power_auprc = metrics['auprc']

        print(f"    power={power:.1f}: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: power={best_power}, AUPRC={best_power_auprc:.4f}")

    tracker.log_experiment(
        name="Exp4.5_Power_Transform",
        description=f"Power transform on amortization (power={best_power})",
        auprc=best_power_auprc,
        auroc=0,
        components=['amortization', 'power_transform'],
        params={'power': best_power},
        time_taken=elapsed
    )

    # Test 4.6: Clip extreme values
    print("\n[Test 4.6] Clip Amortization Extremes")
    start = time.time()

    best_clip_percentile = None
    best_clip_auprc = 0

    for clip_pct in [95, 97, 99, 99.5]:
        upper = np.percentile(amort_train, clip_pct)
        amort_clipped = np.clip(amort_valid, None, upper)

        metrics = evaluate_scores(y_valid, amort_clipped)

        if metrics['auprc'] > best_clip_auprc:
            best_clip_percentile = clip_pct
            best_clip_auprc = metrics['auprc']

        print(f"    clip at {clip_pct}%: AUPRC={metrics['auprc']:.4f}")

    elapsed = time.time() - start

    print(f"  Best: clip={best_clip_percentile}%, AUPRC={best_clip_auprc:.4f}")

    tracker.log_experiment(
        name="Exp4.6_Clip_Extremes",
        description=f"Clip amortization at {best_clip_percentile}th percentile",
        auprc=best_clip_auprc,
        auroc=0,
        components=['amortization', 'clipping'],
        params={'clip_percentile': best_clip_percentile},
        time_taken=elapsed
    )

    # Summary
    tracker.print_summary()

    print("\n" + "="*80)
    print("EXPERIMENT 4 COMPLETE")
    print("="*80)
    print(f"\nBest so far: {tracker.best_auprc:.4f} AUPRC")

if __name__ == "__main__":
    main()
