#!/usr/bin/env python3
"""
Visualization Script for Loan Anomaly Detection Proposal
Creates key charts to include in the project proposal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Creating visualizations for loan anomaly detection proposal...")

# Load data
DATA_DIR = Path("Data")
train_df = pd.read_csv(DATA_DIR / "loans_train.csv")
valid_df = pd.read_csv(DATA_DIR / "loans_valid.csv")

# Create output directory for plots
output_dir = Path("proposal_plots")
output_dir.mkdir(exist_ok=True)

print(f"Saving plots to: {output_dir}")

# =============================================================================
# 1. CLASS DISTRIBUTION COMPARISON
# =============================================================================
print("1. Creating class distribution chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Training set
train_counts = train_df['target'].value_counts().sort_index()
# Since training only has normal loans, create appropriate visualization
if len(train_counts) == 1:
    ax1.pie([100], labels=['Normal (0)'], autopct='%1.1f%%', 
            colors=['lightblue'], startangle=90)
    ax1.text(0, -1.3, '0% Abnormal (1)', ha='center', fontweight='bold', color='red')
else:
    ax1.pie(train_counts.values, labels=['Normal (0)', 'Abnormal (1)'], autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral'], startangle=90)
ax1.set_title('Training Set\n(30,504 loans)\nSemi-Supervised Setup', fontsize=14, fontweight='bold')

# Validation set  
valid_counts = valid_df['target'].value_counts().sort_index()
ax2.pie(valid_counts.values, labels=['Normal (0)', 'Abnormal (1)'], autopct='%1.1f%%',
        colors=['lightblue', 'lightcoral'], startangle=90)
ax2.set_title('Validation Set\n(5,370 loans)', fontsize=14, fontweight='bold')

plt.suptitle('Class Distribution: Semi-Supervised Learning Setup', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 2. KEY STATIC FEATURES DISTRIBUTION
# =============================================================================
print("2. Creating static features distribution plots...")

# Select key numerical features
key_features = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate', 'OriginalDTI']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    # Clean data (handle special missing value codes)
    if feature == 'CreditScore':
        data = train_df[feature][train_df[feature] < 9999]
    elif feature == 'OriginalDTI':
        data = train_df[feature][train_df[feature] < 999]
    else:
        data = train_df[feature]
    
    # Create histogram with KDE
    axes[i].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
    axes[i].set_title(f'{feature}\n(n={len(data):,})', fontweight='bold')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Remove empty subplot
axes[-1].remove()

plt.suptitle('Distribution of Key Static Features (Training Set)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '2_static_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 3. TEMPORAL PATTERNS VISUALIZATION
# =============================================================================
print("3. Creating temporal patterns visualization...")

# Extract temporal features for CurrentActualUPB
temporal_cols = [col for col in train_df.columns if 'CurrentActualUPB' in col and col.split('_')[0].isdigit()]
temporal_cols = sorted(temporal_cols, key=lambda x: int(x.split('_')[0]))

# Sample loans for visualization
sample_size = 100
sample_indices = np.random.choice(train_df.index, sample_size, replace=False)
sample_data = train_df.loc[sample_indices, temporal_cols]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot individual loan trajectories
months = range(len(temporal_cols))
for idx in sample_data.index[:20]:  # Show only first 20 for clarity
    values = sample_data.loc[idx].values
    ax1.plot(months, values, alpha=0.3, color='blue', linewidth=0.8)

ax1.set_title('Sample Loan Trajectories: Current Actual UPB Over Time\n(20 random loans)', fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Current Actual UPB ($)')
ax1.grid(True, alpha=0.3)

# Plot average trajectory with confidence interval
mean_trajectory = sample_data.mean(axis=0)
std_trajectory = sample_data.std(axis=0)

ax2.plot(months, mean_trajectory, color='red', linewidth=3, label='Mean')
ax2.fill_between(months, 
                 mean_trajectory - std_trajectory, 
                 mean_trajectory + std_trajectory, 
                 alpha=0.3, color='red', label='±1 Std Dev')

ax2.set_title('Average Loan Trajectory with Confidence Interval\n(100 random loans)', fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Current Actual UPB ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '3_temporal_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. MISSING VALUES HEATMAP
# =============================================================================
print("4. Creating missing values heatmap...")

# Calculate missing percentages for static features
static_features = [col for col in train_df.columns[2:] 
                  if not any(col.startswith(f"{i}_") for i in range(20))]

missing_data = []
for feature in static_features:
    missing_pct = (train_df[feature].isna().sum() / len(train_df)) * 100
    missing_data.append({'Feature': feature, 'Missing_Percent': missing_pct})

missing_df = pd.DataFrame(missing_data).sort_values('Missing_Percent', ascending=False)

# Only show features with some missing values
missing_df_filtered = missing_df[missing_df['Missing_Percent'] > 0].head(15)

if len(missing_df_filtered) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(missing_df_filtered['Feature'], missing_df_filtered['Missing_Percent'], 
                   color='coral', alpha=0.7)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, missing_df_filtered['Missing_Percent'])):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Missing Percentage (%)', fontweight='bold')
    ax.set_title('Missing Values in Static Features\n(Top 15 features with missing data)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 5. FEATURE TYPES BREAKDOWN
# =============================================================================
print("5. Creating feature types breakdown...")

# Count different types of features
static_count = len([col for col in train_df.columns[2:] 
                   if not any(col.startswith(f"{i}_") for i in range(20))])

temporal_types = {}
for col in train_df.columns[2:]:
    if any(col.startswith(f"{i}_") for i in range(20)):
        if '_' in col:
            parts = col.split('_', 1)
            if len(parts) > 1 and parts[0].isdigit():
                feature_type = parts[1]
                if feature_type not in temporal_types:
                    temporal_types[feature_type] = 0
                temporal_types[feature_type] += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Static vs Temporal breakdown
categories = ['Static Features', 'Temporal Features']
counts = [static_count, sum(temporal_types.values())]
colors = ['lightblue', 'lightgreen']

ax1.pie(counts, labels=categories, autopct='%1.0f%%', colors=colors, startangle=90)
ax1.set_title('Feature Type Distribution\n(145 total features)', fontsize=14, fontweight='bold')

# Temporal feature types breakdown
temporal_features = list(temporal_types.keys())
temporal_counts = list(temporal_types.values())

ax2.bar(range(len(temporal_features)), temporal_counts, color='lightgreen', alpha=0.8)
ax2.set_xlabel('Temporal Feature Type', fontweight='bold')
ax2.set_ylabel('Number of Features (14 months each)', fontweight='bold')
ax2.set_title('Temporal Feature Types\n(8 types × 14 months = 112 features)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(temporal_features)))
ax2.set_xticklabels(temporal_features, rotation=45, ha='right')
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '5_feature_types.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. CORRELATION HEATMAP (SAMPLE OF KEY FEATURES)
# =============================================================================
print("6. Creating correlation heatmap...")

# Select subset of key numerical features for correlation analysis
key_numerical = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate', 
                'OriginalDTI', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']

# Clean data
corr_data = train_df[key_numerical].copy()
corr_data['CreditScore'] = corr_data['CreditScore'].replace(9999, np.nan)
corr_data['OriginalDTI'] = corr_data['OriginalDTI'].replace(999, np.nan)

# Calculate correlation matrix
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})

ax.set_title('Correlation Matrix of Key Static Features\n(Training Set)', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '6_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\nVisualization complete! Generated 6 charts in '{output_dir}' folder:")
print("1. 1_class_distribution.png - Shows semi-supervised setup")
print("2. 2_static_features_distribution.png - Key feature distributions")
print("3. 3_temporal_patterns.png - Loan trajectory examples")
print("4. 4_missing_values.png - Data quality assessment")
print("5. 5_feature_types.png - Feature composition overview")
print("6. 6_correlation_heatmap.png - Feature relationships")

print(f"\nThese charts are ideal for your proposal to demonstrate:")
print("- Understanding of the semi-supervised learning challenge")
print("- Data complexity and feature richness")
print("- Temporal modeling requirements")
print("- Data quality considerations")
print("- Feature engineering opportunities")

print("\nAll plots are high-resolution (300 DPI) and ready for inclusion in your proposal!")