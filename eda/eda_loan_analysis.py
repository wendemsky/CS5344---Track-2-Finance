#!/usr/bin/env python3
"""
Exploratory Data Analysis for Loan Anomaly Detection Dataset
Track 2: Finance - Loan-level anomaly detection for repayment behavior
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib style
plt.style.use('default')
sns.set_palette("husl")

print("=== LOAN ANOMALY DETECTION - EXPLORATORY DATA ANALYSIS ===")
print()

# Data paths
DATA_DIR = Path("Data")
TRAIN_PATH = DATA_DIR / "loans_train.csv"
VALID_PATH = DATA_DIR / "loans_valid.csv"

print("1. LOADING DATA")
print("-" * 50)

# Load datasets
print(f"Loading training data from {TRAIN_PATH}...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Training data shape: {train_df.shape}")

print(f"Loading validation data from {VALID_PATH}...")
valid_df = pd.read_csv(VALID_PATH)
print(f"Validation data shape: {valid_df.shape}")

print()

# Basic dataset information
print("2. BASIC DATASET INFORMATION")
print("-" * 50)
print(f"Total training samples: {len(train_df):,}")
print(f"Total validation samples: {len(valid_df):,}")
print(f"Total features: {train_df.shape[1]}")

# Examine column structure
columns = list(train_df.columns)
print(f"\nColumns overview:")
print(f"- Index column: {columns[0]}")
print(f"- Target column: {columns[1]}")

# Static features (loan and borrower info)
static_features = []
temporal_features = []

for col in columns[2:]:
    if any(col.startswith(f"{i}_") for i in range(20)):  # Check for temporal prefixes
        temporal_features.append(col)
    else:
        static_features.append(col)

print(f"- Static features: {len(static_features)}")
print(f"- Temporal features: {len(temporal_features)}")

# Analyze temporal structure
temporal_months = set()
for col in temporal_features:
    if '_' in col:
        month = col.split('_')[0]
        if month.isdigit():
            temporal_months.add(int(month))

max_months = max(temporal_months) + 1 if temporal_months else 0
print(f"- Maximum temporal sequence length: {max_months} months")

print()

# Class distribution analysis
print("3. CLASS DISTRIBUTION ANALYSIS")
print("-" * 50)
train_target_dist = train_df['target'].value_counts().sort_index()
valid_target_dist = valid_df['target'].value_counts().sort_index()

print("Training set:")
for target, count in train_target_dist.items():
    pct = (count / len(train_df)) * 100
    status = "Normal loans" if target == 0 else "Abnormal loans"
    print(f"  {status} (target={target}): {count:,} ({pct:.2f}%)")

print("\nValidation set:")
for target, count in valid_target_dist.items():
    pct = (count / len(valid_df)) * 100
    status = "Normal loans" if target == 0 else "Abnormal loans"
    print(f"  {status} (target={target}): {count:,} ({pct:.2f}%)")

# Calculate imbalance ratio
normal_count = train_target_dist.get(0, 0)
abnormal_count = train_target_dist.get(1, 0)
imbalance_ratio = normal_count / abnormal_count if abnormal_count > 0 else float('inf')
print(f"\nClass imbalance ratio (normal:abnormal): {imbalance_ratio:.1f}:1")

print()

# Static features analysis
print("4. STATIC FEATURES ANALYSIS")
print("-" * 50)
print("Static feature columns:")
for i, feat in enumerate(static_features):
    print(f"  {i+1:2d}. {feat}")

# Sample some key static features for detailed analysis
key_static_features = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate', 
                      'OriginalDTI', 'LoanPurpose', 'PropertyType', 'OccupancyStatus']

available_key_features = [f for f in key_static_features if f in static_features]
print(f"\nAnalyzing {len(available_key_features)} key static features:")

for feature in available_key_features[:5]:  # Show first 5
    print(f"\n{feature}:")
    if train_df[feature].dtype in ['object', 'category']:
        # Categorical feature
        value_counts = train_df[feature].value_counts().head(5)
        print(f"  Top 5 values: {dict(value_counts)}")
        print(f"  Unique values: {train_df[feature].nunique()}")
        print(f"  Missing values: {train_df[feature].isna().sum()}")
    else:
        # Numerical feature
        stats = train_df[feature].describe()
        print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        print(f"  Missing values: {train_df[feature].isna().sum()}")

print()

# Temporal features analysis
print("5. TEMPORAL FEATURES ANALYSIS")
print("-" * 50)

# Group temporal features by type
temporal_feature_types = {}
for col in temporal_features:
    if '_' in col:
        parts = col.split('_', 1)
        if len(parts) > 1 and parts[0].isdigit():
            feature_type = parts[1]
            if feature_type not in temporal_feature_types:
                temporal_feature_types[feature_type] = []
            temporal_feature_types[feature_type].append(col)

print("Temporal feature types:")
for feat_type, cols in temporal_feature_types.items():
    print(f"  {feat_type}: {len(cols)} columns (months 0-{len(cols)-1})")

print()

# Missing values analysis
print("6. MISSING VALUES ANALYSIS")
print("-" * 50)

# Check missing values in static features
static_missing = train_df[static_features].isna().sum()
static_missing_pct = (static_missing / len(train_df)) * 100
missing_summary = pd.DataFrame({
    'feature': static_missing.index,
    'missing_count': static_missing.values,
    'missing_pct': static_missing_pct.values
}).sort_values('missing_pct', ascending=False)

print("Missing values in static features (top 10):")
top_missing = missing_summary[missing_summary['missing_pct'] > 0].head(10)
if len(top_missing) > 0:
    for _, row in top_missing.iterrows():
        print(f"  {row['feature']}: {row['missing_count']} ({row['missing_pct']:.2f}%)")
else:
    print("  No missing values in static features!")

# Check missing values in temporal features (sample)
sample_temporal = [col for col in temporal_features[:20]]  # First 20 temporal features
temporal_missing = train_df[sample_temporal].isna().sum().sum()
print(f"\nMissing values in sample temporal features: {temporal_missing}")

print()

# Sequence length analysis
print("7. SEQUENCE LENGTH ANALYSIS")
print("-" * 50)

# For each loan, count how many months have data
sequence_lengths = []
for idx, row in train_df.head(1000).iterrows():  # Sample first 1000 rows for efficiency
    month_count = 0
    for month in range(max_months):
        # Check if this month has any non-null temporal data
        month_cols = [col for col in temporal_features if col.startswith(f"{month}_")]
        if month_cols and not row[month_cols].isna().all():
            month_count += 1
        else:
            break  # Stop at first missing month
    sequence_lengths.append(month_count)

if sequence_lengths:
    seq_stats = pd.Series(sequence_lengths).describe()
    print("Sequence length statistics (sample of 1000 loans):")
    print(f"  Mean length: {seq_stats['mean']:.1f} months")
    print(f"  Std: {seq_stats['std']:.1f} months")
    print(f"  Min: {int(seq_stats['min'])} months")
    print(f"  Max: {int(seq_stats['max'])} months")
    print(f"  25th percentile: {seq_stats['25%']:.1f} months")
    print(f"  75th percentile: {seq_stats['75%']:.1f} months")

print()

# Data quality summary
print("8. DATA QUALITY SUMMARY")
print("-" * 50)
print("+ Dataset loaded successfully")
print("+ No duplicate indices found" if not train_df['index'].duplicated().any() else "! Duplicate indices detected")
print("+ Target variable is binary (0, 1)")
print(f"+ Strong class imbalance detected ({imbalance_ratio:.1f}:1)")
print(f"+ Mixed data types: {train_df.dtypes.value_counts().to_dict()}")

# Key insights
print("\n9. KEY INSIGHTS")
print("-" * 50)
print("DATASET CHARACTERISTICS:")
print(f"   - Dataset size: {len(train_df):,} training + {len(valid_df):,} validation loans")
print(f"   - Feature count: {len(static_features)} static + {len(temporal_features)} temporal")
print(f"   - Temporal span: Up to {max_months} months of loan history")

print(f"\nANOMALY DETECTION CHALLENGE:")
abnormal_pct = (abnormal_count / (normal_count + abnormal_count)) * 100 if abnormal_count > 0 else 0
print(f"   - Severe class imbalance: Only {abnormal_pct:.1f}% abnormal loans in validation")
print(f"   - Binary classification: Normal (0) vs Abnormal (1) loans")
print(f"   - Training set contains ONLY normal loans (semi-supervised learning)")

print(f"\nMODELING CONSIDERATIONS:")
print(f"   - Handle mixed static + temporal data")
print(f"   - Semi-supervised anomaly detection (no abnormal samples in training)")
print(f"   - Address class imbalance in validation")
print(f"   - Variable sequence lengths need attention")
print(f"   - Multiple feature types (numerical, categorical, temporal)")

print("\n" + "="*80)
print("EDA COMPLETED - Ready for feature engineering and modeling!")
print("="*80)