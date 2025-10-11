#!/usr/bin/env python3
"""
Comprehensive Baseline Models for Loan Anomaly Detection
Tests 10 unsupervised anomaly detection methods with various configurations
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Autoencoder baseline will be skipped.")

# Create output directory
OUTPUT_DIR = Path("baseline_models/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_preprocess(scaler_type='robust', pca_components=None):
    """Load data and apply preprocessing"""
    print(f"\nLoading data with {scaler_type} scaling, PCA={pca_components}...")

    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")

    # Drop target and ID columns
    y_valid = valid['target'].values

    feature_cols = [c for c in train.columns if c not in ['target', 'Id', 'index']]
    X_train = train[feature_cols].select_dtypes(include=[np.number]).fillna(0).values
    X_valid = valid[feature_cols].select_dtypes(include=[np.number]).fillna(0).values

    # Scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # PCA
    if pca_components and pca_components < X_train_scaled.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_valid_scaled = pca.transform(X_valid_scaled)
        print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    print(f"  Train shape: {X_train_scaled.shape}, Valid shape: {X_valid_scaled.shape}")

    return X_train_scaled, X_valid_scaled, y_valid

def evaluate_model(y_true, scores, model_name):
    """Evaluate model performance"""
    auprc = average_precision_score(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    # F1 at optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)

    print(f"  {model_name:35s} | AUPRC: {auprc:.4f} | AUROC: {auroc:.4f} | F1: {best_f1:.4f}")

    return {
        'model': model_name,
        'auprc': float(auprc),
        'auroc': float(auroc),
        'f1': float(best_f1)
    }

# ==================== BASELINE MODELS ====================

def baseline_1_isolation_forest(X_train, X_valid, y_valid):
    """Baseline 1: Isolation Forest with hyperparameter tuning"""
    print("\n[1/10] Isolation Forest")
    results = []

    for n_est in [100, 300, 500]:
        for max_samp in [256, 512, 'auto']:
            model = IsolationForest(
                n_estimators=n_est,
                max_samples=max_samp,
                contamination='auto',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train)
            scores = -model.score_samples(X_valid)

            result = evaluate_model(y_valid, scores, f"IsolationForest(n={n_est}, samp={max_samp})")
            results.append(result)

    return max(results, key=lambda x: x['auprc'])

def baseline_2_lof(X_train, X_valid, y_valid):
    """Baseline 2: Local Outlier Factor with multiple k values"""
    print("\n[2/10] Local Outlier Factor")
    results = []

    for k in [5, 10, 20, 30, 50]:
        model = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)
        model.fit(X_train)
        scores = -model.score_samples(X_valid)

        result = evaluate_model(y_valid, scores, f"LOF(k={k})")
        results.append(result)

    return max(results, key=lambda x: x['auprc'])

def baseline_3_one_class_svm(X_train, X_valid, y_valid):
    """Baseline 3: One-Class SVM with different kernels"""
    print("\n[3/10] One-Class SVM")
    results = []

    # Subsample for faster training
    if X_train.shape[0] > 10000:
        idx = np.random.choice(X_train.shape[0], 10000, replace=False)
        X_train_sub = X_train[idx]
    else:
        X_train_sub = X_train

    for kernel in ['rbf', 'linear']:
        for nu in [0.01, 0.05, 0.1]:
            try:
                model = OneClassSVM(kernel=kernel, nu=nu, gamma='auto')
                model.fit(X_train_sub)
                scores = -model.score_samples(X_valid)

                result = evaluate_model(y_valid, scores, f"OneClassSVM(kernel={kernel}, nu={nu})")
                results.append(result)
            except Exception as e:
                print(f"  OneClassSVM(kernel={kernel}, nu={nu}) failed: {e}")

    return max(results, key=lambda x: x['auprc']) if results else None

def baseline_4_elliptic_envelope(X_train, X_valid, y_valid):
    """Baseline 4: Elliptic Envelope (robust covariance)"""
    print("\n[4/10] Elliptic Envelope")
    results = []

    for support in [0.8, 0.9, 0.95]:
        try:
            model = EllipticEnvelope(
                contamination=1.0-support,
                support_fraction=support,
                random_state=42
            )
            model.fit(X_train)
            scores = -model.score_samples(X_valid)

            result = evaluate_model(y_valid, scores, f"EllipticEnvelope(support={support})")
            results.append(result)
        except Exception as e:
            print(f"  EllipticEnvelope(support={support}) failed: {e}")

    return max(results, key=lambda x: x['auprc']) if results else None

def baseline_5_autoencoder(X_train, X_valid, y_valid):
    """Baseline 5: MLP Autoencoder with reconstruction error"""
    print("\n[5/10] MLP Autoencoder")

    if not TF_AVAILABLE:
        print("  Skipping: TensorFlow not installed")
        return None

    input_dim = X_train.shape[1]
    encoding_dim = min(32, input_dim // 2)

    # Build autoencoder
    encoder = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(encoding_dim, activation='relu')
    ])

    decoder = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(encoding_dim,)),
        keras.layers.Dense(input_dim, activation='linear')
    ])

    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train
    autoencoder.fit(
        X_train, X_train,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        verbose=0
    )

    # Reconstruction error as anomaly score
    X_valid_pred = autoencoder.predict(X_valid, verbose=0)
    scores = np.mean((X_valid - X_valid_pred) ** 2, axis=1)

    return evaluate_model(y_valid, scores, "MLP_Autoencoder")

def baseline_6_dbscan(X_train, X_valid, y_valid):
    """Baseline 6: DBSCAN clustering-based"""
    print("\n[6/10] DBSCAN")
    results = []

    # Subsample for faster computation
    if X_train.shape[0] > 5000:
        idx = np.random.choice(X_train.shape[0], 5000, replace=False)
        X_train_sub = X_train[idx]
    else:
        X_train_sub = X_train

    for eps in [0.5, 1.0, 2.0]:
        for min_samples in [5, 10]:
            try:
                model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                model.fit(X_train_sub)

                # Distance to nearest core point as anomaly score
                from sklearn.metrics import pairwise_distances
                core_samples = X_train_sub[model.core_sample_indices_]
                if len(core_samples) > 0:
                    distances = pairwise_distances(X_valid, core_samples).min(axis=1)

                    result = evaluate_model(y_valid, distances, f"DBSCAN(eps={eps}, min_samples={min_samples})")
                    results.append(result)
            except Exception as e:
                print(f"  DBSCAN(eps={eps}, min_samples={min_samples}) failed: {e}")

    return max(results, key=lambda x: x['auprc']) if results else None

def baseline_7_knn_distance(X_train, X_valid, y_valid):
    """Baseline 7: K-Nearest Neighbors distance"""
    print("\n[7/10] KNN Distance")
    results = []

    for k in [5, 10, 20, 30]:
        model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        model.fit(X_train)

        distances, _ = model.kneighbors(X_valid)
        scores = distances[:, -1]  # k-th neighbor distance

        result = evaluate_model(y_valid, scores, f"KNN_Distance(k={k})")
        results.append(result)

    return max(results, key=lambda x: x['auprc'])

def baseline_8_pca_reconstruction(X_train, X_valid, y_valid):
    """Baseline 8: PCA Reconstruction Error"""
    print("\n[8/10] PCA Reconstruction Error")
    results = []

    for n_comp in [20, 50, 80]:
        if n_comp >= X_train.shape[1]:
            continue

        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(X_train)

        X_valid_transformed = pca.transform(X_valid)
        X_valid_reconstructed = pca.inverse_transform(X_valid_transformed)

        scores = np.mean((X_valid - X_valid_reconstructed) ** 2, axis=1)

        result = evaluate_model(y_valid, scores, f"PCA_Reconstruction(n={n_comp})")
        results.append(result)

    return max(results, key=lambda x: x['auprc'])

def baseline_9_random_projection_lof(X_train, X_valid, y_valid):
    """Baseline 9: Random Projection + LOF Ensemble"""
    print("\n[9/10] Random Projection LOF Ensemble")

    n_projections = 20
    proj_dim = min(50, X_train.shape[1])
    k = 10

    ensemble_scores = []

    for i in range(n_projections):
        # Random projection
        rng = np.random.default_rng(42 + i)
        projection_matrix = rng.normal(0, 1/np.sqrt(proj_dim), size=(X_train.shape[1], proj_dim))

        X_train_proj = X_train @ projection_matrix
        X_valid_proj = X_valid @ projection_matrix

        # LOF on projection
        model = LocalOutlierFactor(n_neighbors=k, novelty=True)
        model.fit(X_train_proj)
        scores = -model.score_samples(X_valid_proj)

        ensemble_scores.append(scores)

    # Max aggregation
    ensemble_scores = np.column_stack(ensemble_scores)
    final_scores = np.max(ensemble_scores, axis=1)

    return evaluate_model(y_valid, final_scores, f"RandomProjection_LOF(B={n_projections})")

def baseline_10_mahalanobis(X_train, X_valid, y_valid):
    """Baseline 10: Mahalanobis Distance"""
    print("\n[10/10] Mahalanobis Distance")

    try:
        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False)

        # Regularization if needed
        if not np.all(np.linalg.eigvals(cov) > 0):
            reg = 1e-6
            while not np.all(np.linalg.eigvals(cov) > 0):
                cov += reg * np.eye(cov.shape[0])
                reg *= 10

        inv_cov = np.linalg.inv(cov)

        # Mahalanobis distance
        diff = X_valid - mean
        scores = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

        return evaluate_model(y_valid, scores, "Mahalanobis_Distance")
    except Exception as e:
        print(f"  Mahalanobis Distance failed: {e}")
        return None

# ==================== MAIN RUNNER ====================

def main():
    print("="*80)
    print("COMPREHENSIVE BASELINE MODEL EVALUATION")
    print("="*80)

    all_results = []

    # Test different preprocessing configurations
    configs = [
        ('robust', 80),
        ('standard', 80),
        ('robust', None),
    ]

    for scaler_type, pca_comp in configs:
        print(f"\n{'='*80}")
        print(f"CONFIGURATION: Scaler={scaler_type}, PCA={pca_comp}")
        print(f"{'='*80}")

        X_train, X_valid, y_valid = load_and_preprocess(scaler_type, pca_comp)

        config_name = f"{scaler_type}_pca{pca_comp if pca_comp else 'None'}"

        # Run all baselines
        baseline_funcs = [
            baseline_1_isolation_forest,
            baseline_2_lof,
            baseline_3_one_class_svm,
            baseline_4_elliptic_envelope,
            baseline_5_autoencoder,
            baseline_6_dbscan,
            baseline_7_knn_distance,
            baseline_8_pca_reconstruction,
            baseline_9_random_projection_lof,
            baseline_10_mahalanobis
        ]

        for baseline_func in baseline_funcs:
            try:
                start_time = time.time()
                result = baseline_func(X_train, X_valid, y_valid)
                elapsed = time.time() - start_time

                if result:
                    result['config'] = config_name
                    result['scaler'] = scaler_type
                    result['pca_components'] = pca_comp
                    result['time_seconds'] = elapsed
                    all_results.append(result)
            except Exception as e:
                print(f"  ERROR in {baseline_func.__name__}: {e}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('auprc', ascending=False)

    results_df.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (Top 15 by AUPRC)")
    print("="*80)
    print(results_df[['model', 'config', 'auprc', 'auroc', 'f1', 'time_seconds']].head(15).to_string(index=False))

    # Save JSON
    with open(OUTPUT_DIR / "baseline_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"  - baseline_results.csv")
    print(f"  - baseline_results.json")

if __name__ == "__main__":
    main()
