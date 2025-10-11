"""
Correlation Analysis: Investigate using highly correlated features as anomaly proxies

Since training data has no anomalies (target=0), this approach explores:
1. Finding features highly correlated with target in validation set
2. Using those features as proxy labels for training anomaly detectors
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading datasets...")
train_df = pd.read_csv('Data/loans_train.csv')
valid_df = pd.read_csv('Data/loans_valid.csv')

print(f"\nTrain shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"\nTarget distribution in train: {train_df['target'].value_counts()}")
print(f"Target distribution in valid: {valid_df['target'].value_counts(normalize=True)}")

# Get feature columns (excluding target)
feature_cols = [col for col in valid_df.columns if col not in ['target', 'loan_id']]

# Calculate correlations with target using validation set
print("\n" + "="*80)
print("CORRELATION ANALYSIS (using validation set)")
print("="*80)

# Identify numeric columns only
numeric_cols = valid_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"\nTotal features: {len(feature_cols)}")
print(f"Numeric features: {len(numeric_cols)}")

correlations = {}
for col in numeric_cols:
    # Handle missing values
    valid_mask = valid_df[[col, 'target']].notna().all(axis=1)
    if valid_mask.sum() < 10:  # Skip if too few valid samples
        continue

    # Calculate correlation
    corr = valid_df.loc[valid_mask, col].corr(valid_df.loc[valid_mask, 'target'])
    if not np.isnan(corr):
        correlations[col] = corr

# Sort by absolute correlation
correlations_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
correlations_df['abs_correlation'] = correlations_df['correlation'].abs()
correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)

print("\nTop 20 features by absolute correlation with target:")
print(correlations_df.head(20))

# Analyze top correlated features
print("\n" + "="*80)
print("DETAILED ANALYSIS OF TOP CORRELATED FEATURES")
print("="*80)

top_features = correlations_df.head(10).index.tolist()

for feature in top_features:
    print(f"\n{feature}:")
    print(f"  Correlation: {correlations_df.loc[feature, 'correlation']:.4f}")

    # Distribution in train vs valid
    print(f"  Train - Mean: {train_df[feature].mean():.4f}, Std: {train_df[feature].std():.4f}, Missing: {train_df[feature].isna().sum()}")
    print(f"  Valid - Mean: {valid_df[feature].mean():.4f}, Std: {valid_df[feature].std():.4f}, Missing: {valid_df[feature].isna().sum()}")

    # Distribution by target in validation
    valid_mask = valid_df[feature].notna()
    normal = valid_df.loc[valid_mask & (valid_df['target'] == 0), feature]
    anomaly = valid_df.loc[valid_mask & (valid_df['target'] == 1), feature]

    print(f"  Valid Normal (target=0) - Mean: {normal.mean():.4f}, Std: {normal.std():.4f}")
    print(f"  Valid Anomaly (target=1) - Mean: {anomaly.mean():.4f}, Std: {anomaly.std():.4f}")

# Feasibility Assessment
print("\n" + "="*80)
print("FEASIBILITY ASSESSMENT")
print("="*80)

print("\n1. POTENTIAL ISSUES WITH THIS APPROACH:")
print("   a) Data Leakage Risk:")
print("      - Using validation target to select features = indirect leakage")
print("      - Model might overfit to validation-specific patterns")
print("      - Test set performance may degrade significantly")

print("\n   b) Proxy Label Quality:")
print("      - Correlation doesn't imply causation")
print("      - High correlation feature may not distinguish anomalies well in train set")
print("      - Train set has NO anomalies, so feature distribution is unknown for anomalies")

print("\n   c) Distribution Mismatch:")
print("      - Training on proxy labels assumes similar distributions")
print("      - No guarantee proxy feature separates normal/anomaly in train set")

print("\n2. WHEN THIS APPROACH MIGHT WORK:")
print("   - If the highly correlated feature has clear bimodal distribution in train")
print("   - If feature represents a monotonic risk indicator (e.g., payment defaults)")
print("   - If domain knowledge supports the feature as anomaly indicator")

print("\n3. ALTERNATIVE APPROACHES:")
print("   a) Pure unsupervised methods (LOF, Isolation Forest, Autoencoders)")
print("   b) Semi-supervised with validation for hyperparameter tuning only")
print("   c) Domain-driven feature engineering (payment patterns, credit metrics)")
print("   d) Ensemble methods combining multiple unsupervised detectors")

# Check if top features have bimodal distributions in training set
print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS IN TRAINING SET")
print("="*80)

from scipy import stats

for feature in top_features[:5]:
    train_values = train_df[feature].dropna()

    # Test for bimodality using coefficient
    n = len(train_values)
    if n > 10:
        skewness = stats.skew(train_values)
        kurtosis = stats.kurtosis(train_values)
        bimodality_coef = (skewness**2 + 1) / kurtosis

        print(f"\n{feature}:")
        print(f"  Bimodality coefficient: {bimodality_coef:.4f} (>0.555 suggests bimodal)")
        print(f"  Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

        # Quartiles
        q25, q50, q75 = np.percentile(train_values, [25, 50, 75])
        print(f"  Quartiles: Q1={q25:.4f}, Q2={q50:.4f}, Q3={q75:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nUsing highly correlated features as proxy labels is RISKY because:")
print("1. It creates indirect data leakage from validation set")
print("2. Training set has NO anomalies, so we can't verify if proxy separates classes")
print("3. Correlation in validation doesn't guarantee separability in training")
print("\nRECOMMENDATION:")
print("- Use validation set ONLY for hyperparameter tuning and model selection")
print("- Stick with unsupervised/semi-supervised methods for training")
print("- Consider domain expertise to engineer features (payment trends, defaults)")
print("- Use ensemble of multiple unsupervised detectors")
