#!/usr/bin/env python3
"""
EXPERIMENT 1: Enhanced Domain Feature Engineering
Test different domain-specific feature sets to maximize amortization signal
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
import time

# Import framework
from experiment_framework import tracker, load_data, evaluate_scores

def extract_amortization_features_v1(df):
    """Original amortization features (baseline)"""
    features = []

    # Find amortization columns
    upb_cols = [c for c in df.columns if 'CurrentActualUPB' in c]
    rate_cols = [c for c in df.columns if 'CurrentInterestRate' in c]

    if len(upb_cols) >= 2:
        # Balance change rate
        df['balance_change_rate'] = (df[upb_cols[0]] - df[upb_cols[-1]]) / (df[upb_cols[0]] + 1)
        features.append('balance_change_rate')

        # Balance volatility
        upb_values = df[upb_cols].values
        df['balance_volatility'] = np.std(upb_values, axis=1) / (np.mean(upb_values, axis=1) + 1)
        features.append('balance_volatility')

    if 'OriginalUPB' in df.columns and len(upb_cols) > 0:
        # Current vs original ratio
        df['upb_ratio'] = df[upb_cols[0]] / (df['OriginalUPB'] + 1)
        features.append('upb_ratio')

    return features

def extract_amortization_features_v2(df):
    """Enhanced amortization with payment patterns"""
    features = extract_amortization_features_v1(df)

    upb_cols = [c for c in df.columns if 'CurrentActualUPB' in c]
    rate_cols = [c for c in df.columns if 'CurrentInterestRate' in c]

    if len(upb_cols) >= 3:
        # Payment acceleration/deceleration
        recent_change = df[upb_cols[0]] - df[upb_cols[2]]
        older_change = df[upb_cols[2]] - df[upb_cols[-1]]
        df['payment_acceleration'] = recent_change - older_change
        features.append('payment_acceleration')

        # Months to payoff (estimated)
        avg_payment = np.mean(df[upb_cols].diff(axis=1).values[:, 1:], axis=1)
        df['months_to_payoff'] = df[upb_cols[0]] / (np.abs(avg_payment) + 1)
        df['months_to_payoff'] = np.clip(df['months_to_payoff'], 0, 360)  # Cap at 30 years
        features.append('months_to_payoff')

    if len(rate_cols) >= 2:
        # Interest rate volatility
        rate_values = df[rate_cols].values
        df['rate_volatility'] = np.std(rate_values, axis=1)
        features.append('rate_volatility')

    return features

def extract_amortization_features_v3(df):
    """Ultra-enhanced with risk indicators"""
    features = extract_amortization_features_v2(df)

    upb_cols = [c for c in df.columns if 'CurrentActualUPB' in c]
    rate_cols = [c for c in df.columns if 'CurrentInterestRate' in c]

    if len(upb_cols) >= 3:
        # Irregular payment pattern (variance of payment sizes)
        payments = np.abs(df[upb_cols].diff(axis=1).values[:, 1:])
        df['payment_irregularity'] = np.std(payments, axis=1) / (np.mean(payments, axis=1) + 1)
        features.append('payment_irregularity')

        # Zero payment months (payment shock indicator)
        df['zero_payment_months'] = (payments < 100).sum(axis=1)
        features.append('zero_payment_months')

        # Increasing balance (negative amortization)
        balance_increases = (np.diff(df[upb_cols].values, axis=1) > 0).sum(axis=1)
        df['negative_amort_months'] = balance_increases
        features.append('negative_amort_months')

    # LTV and DTI risk scores
    if 'OriginalLTV' in df.columns:
        df['ltv_risk'] = np.where(df['OriginalLTV'] > 80, 1.0, 0.0)
        features.append('ltv_risk')

    if 'OriginalDTI' in df.columns:
        df['dti_risk'] = np.where(df['OriginalDTI'] > 43, 1.0, 0.0)
        features.append('dti_risk')

    if 'CreditScore' in df.columns:
        df['credit_risk'] = np.where(df['CreditScore'] < 620, 1.0, 0.0)
        features.append('credit_risk')

    # Combined risk score
    risk_cols = [c for c in features if 'risk' in c]
    if risk_cols:
        df['combined_risk'] = df[risk_cols].sum(axis=1) / len(risk_cols)
        features.append('combined_risk')

    return features

def build_amortization_score(df, features):
    """Build anomaly score from amortization features"""
    if not features:
        return np.zeros(len(df))

    # Normalize each feature to [0,1]
    scores = []
    for feat in features:
        if feat in df.columns:
            vals = df[feat].fillna(0).values
            # Rank normalize
            ranks = pd.Series(vals).rank(pct=True).values
            scores.append(ranks)

    if not scores:
        return np.zeros(len(df))

    # Weighted combination (equal weights for now)
    scores = np.column_stack(scores)
    return np.mean(scores, axis=1)

def run_experiment_simple_lof(X_train, X_valid, y_valid, feature_name, k=50):
    """Simple LOF baseline with feature set"""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
    model.fit(X_train_scaled)
    scores = -model.score_samples(X_valid_scaled)

    return scores

def main():
    print("="*80)
    print("EXPERIMENT 1: Enhanced Domain Feature Engineering")
    print("="*80)

    # Load data
    X_train, X_valid, y_valid, train_df, valid_df, numeric_cols = load_data()

    experiments = []

    # Test 1: Original amortization features (v1)
    print("\n[Test 1.1] Amortization Features V1 (Original)")
    start = time.time()

    train_copy = train_df.copy()
    valid_copy = valid_df.copy()

    features_v1 = extract_amortization_features_v1(train_copy)
    _ = extract_amortization_features_v1(valid_copy)

    amort_score_train_v1 = build_amortization_score(train_copy, features_v1)
    amort_score_valid_v1 = build_amortization_score(valid_copy, features_v1)

    metrics = evaluate_scores(y_valid, amort_score_valid_v1)
    elapsed = time.time() - start

    print(f"  Features: {features_v1}")
    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp1.1_AmortV1",
        description="Original amortization features (balance change, volatility, ratio)",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization_v1'],
        params={'features': features_v1},
        time_taken=elapsed
    )

    # Test 2: Enhanced amortization features (v2)
    print("\n[Test 1.2] Amortization Features V2 (Enhanced)")
    start = time.time()

    train_copy = train_df.copy()
    valid_copy = valid_df.copy()

    features_v2 = extract_amortization_features_v2(train_copy)
    _ = extract_amortization_features_v2(valid_copy)

    amort_score_train_v2 = build_amortization_score(train_copy, features_v2)
    amort_score_valid_v2 = build_amortization_score(valid_copy, features_v2)

    metrics = evaluate_scores(y_valid, amort_score_valid_v2)
    elapsed = time.time() - start

    print(f"  Features: {len(features_v2)} total")
    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp1.2_AmortV2",
        description="Enhanced amortization + payment patterns + interest rate volatility",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization_v2'],
        params={'features': features_v2[:5]},  # Sample
        time_taken=elapsed
    )

    # Test 3: Ultra-enhanced with risk indicators (v3)
    print("\n[Test 1.3] Amortization Features V3 (Ultra-Enhanced + Risk)")
    start = time.time()

    train_copy = train_df.copy()
    valid_copy = valid_df.copy()

    features_v3 = extract_amortization_features_v3(train_copy)
    _ = extract_amortization_features_v3(valid_copy)

    amort_score_train_v3 = build_amortization_score(train_copy, features_v3)
    amort_score_valid_v3 = build_amortization_score(valid_copy, features_v3)

    metrics = evaluate_scores(y_valid, amort_score_valid_v3)
    elapsed = time.time() - start

    print(f"  Features: {len(features_v3)} total")
    print(f"  New features: payment_irregularity, zero_payment_months, negative_amort_months, risk_scores")
    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp1.3_AmortV3_Risk",
        description="Ultra-enhanced amortization + risk indicators (LTV, DTI, credit)",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization_v3', 'risk_indicators'],
        params={'features': features_v3[:8]},  # Sample
        time_taken=elapsed
    )

    # Test 4: Combine best amortization with LOF
    print("\n[Test 1.4] Best Amortization + LOF(k=50)")
    start = time.time()

    # Use v3 (likely best) + LOF
    lof_scores = run_experiment_simple_lof(X_train, X_valid, y_valid, "numeric_features", k=50)

    # Rank normalize both
    amort_rank = pd.Series(amort_score_valid_v3).rank(pct=True).values
    lof_rank = pd.Series(lof_scores).rank(pct=True).values

    # Simple average
    combined_scores = (amort_rank + lof_rank) / 2

    metrics = evaluate_scores(y_valid, combined_scores)
    elapsed = time.time() - start

    print(f"  Fusion: Average of (AmortV3_rank + LOF_rank)")
    print(f"  AUPRC: {metrics['auprc']:.4f}, AUROC: {metrics['auroc']:.4f}")

    tracker.log_experiment(
        name="Exp1.4_AmortV3+LOF",
        description="Best amortization (v3) + LOF(k=50) rank average fusion",
        auprc=metrics['auprc'],
        auroc=metrics['auroc'],
        components=['amortization_v3', 'lof_k50', 'rank_average_fusion'],
        params={'k': 50, 'fusion': 'rank_average'},
        time_taken=elapsed
    )

    # Summary
    tracker.print_summary()

    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETE")
    print("="*80)
    print("\nNext: Run exp2_adaptive_lof.py to test adaptive LOF strategies")

if __name__ == "__main__":
    main()
