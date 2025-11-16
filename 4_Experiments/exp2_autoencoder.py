#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_builder_shared import FeatureBuilderAdvanced

from sklearn.metrics import average_precision_score, roc_auc_score

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LeakyReLU
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("ERROR: TensorFlow not installed.")
    print("Install with: pip install tensorflow")
    sys.exit(1)

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(input_dim // 2)(input_layer)
    encoder = LeakyReLU(alpha=0.1)(encoder)
    encoder = Dense(input_dim // 4)(encoder)
    encoder = LeakyReLU(alpha=0.1)(encoder)

    bottleneck = Dense(input_dim // 8)(encoder)
    bottleneck = LeakyReLU(alpha=0.1)(bottleneck)

    decoder = Dense(input_dim // 4)(bottleneck)
    decoder = LeakyReLU(alpha=0.1)(decoder)
    decoder = Dense(input_dim // 2)(decoder)
    decoder = LeakyReLU(alpha=0.1)(decoder)

    output_layer = Dense(input_dim, activation='linear')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

    return autoencoder

def main():
    print("EXPERIMENT 2: Autoencoder for Anomaly Detection")
    print("="*70)
    print("Goal: Train a deep MLP autoencoder on normal loans and use")
    print("      reconstruction error as anomaly score.")
    print()

    train = pd.read_csv("../1_Data/loans_train.csv")
    valid = pd.read_csv("../1_Data/loans_valid.csv")

    yv = valid["target"].values
    print(f"Data loaded: Train={train.shape}, Valid={valid.shape}")
    print(f"Validation anomaly rate: {yv.mean():.2%}")
    print()

    print("Building features...")
    fb = FeatureBuilderAdvanced(use_pca=False, pca_comps=80)
    fb.fit(train)

    X_tr_scaled, _, _ = fb.transform(train)
    X_v_scaled, _, _ = fb.transform(valid)

    input_dim = X_tr_scaled.shape[1]
    print(f"Feature dimension: {input_dim}")
    print()

    print("Building autoencoder architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Encoder: {input_dim} -> {input_dim//2} -> {input_dim//4}")
    print(f"  Bottleneck: {input_dim//8}")
    print(f"  Decoder: {input_dim//8} -> {input_dim//4} -> {input_dim//2} -> {input_dim}")
    print()

    autoencoder = build_autoencoder(input_dim)

    print("Training autoencoder (20 epochs)...")
    print("  - Training only on normal loans")
    print("  - Using MSE loss")

    history = autoencoder.fit(
        X_tr_scaled, X_tr_scaled,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        verbose=0
    )

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Training complete")
    print(f"  Final train loss: {final_train_loss:.6f}")
    print(f"  Final val loss: {final_val_loss:.6f}")
    print()

    print("Computing reconstruction errors...")
    X_v_reconstructed = autoencoder.predict(X_v_scaled, verbose=0)
    reconstruction_errors = np.mean((X_v_scaled - X_v_reconstructed) ** 2, axis=1)

    ae_auprc = average_precision_score(yv, reconstruction_errors)
    ae_auroc = roc_auc_score(yv, reconstruction_errors)

    print("AUTOENCODER RESULTS:")
    print(f"  AUPRC: {ae_auprc:.6f}")
    print(f"  AUROC: {ae_auroc:.6f}")
    print()

    print("KEY FINDINGS:")
    print("  1. Autoencoder provides non-linear reconstruction-based detection")
    print("  2. Performance depends on architecture and training")
    print("  3. Captures different patterns than density-based methods")
    print("  4. Useful as complementary detector in ensemble")

if __name__ == "__main__":
    main()
