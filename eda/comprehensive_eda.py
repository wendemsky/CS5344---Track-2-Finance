#!/usr/bin/env python3
"""
Comprehensive EDA for Loan Anomaly Detection
Generates statistical analysis, visualizations, and insights
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, shapiro, chi2_contingency
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory
OUTPUT_DIR = Path("eda/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load train, valid, and test datasets"""
    print("Loading datasets...")
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")
    test = pd.read_csv("Data/loans_test.csv")

    print(f"Train shape: {train.shape}")
    print(f"Valid shape: {valid.shape}")
    print(f"Test shape: {test.shape}")

    return train, valid, test

def basic_statistics(train, valid):
    """Generate basic statistical summaries"""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)

    stats_dict = {}

    # Class distribution in train
    train_normals = (train['target'] == 0).sum()
    train_anomalies = (train['target'] == 1).sum()
    train_ratio = train_anomalies / train_normals if train_normals > 0 else 0

    print(f"\nTrain Set:")
    print(f"  Normal loans: {train_normals} ({train_normals/len(train)*100:.2f}%)")
    print(f"  Anomalous loans: {train_anomalies} ({train_anomalies/len(train)*100:.2f}%)")
    print(f"  Anomaly ratio: 1:{1/train_ratio:.2f}")

    stats_dict['train'] = {
        'total': len(train),
        'normals': int(train_normals),
        'anomalies': int(train_anomalies),
        'anomaly_rate': float(train_anomalies / len(train))
    }

    # Class distribution in valid
    valid_normals = (valid['target'] == 0).sum()
    valid_anomalies = (valid['target'] == 1).sum()
    valid_ratio = valid_anomalies / valid_normals if valid_normals > 0 else 0

    print(f"\nValidation Set:")
    print(f"  Normal loans: {valid_normals} ({valid_normals/len(valid)*100:.2f}%)")
    print(f"  Anomalous loans: {valid_anomalies} ({valid_anomalies/len(valid)*100:.2f}%)")
    print(f"  Anomaly ratio: 1:{1/valid_ratio:.2f}")

    stats_dict['valid'] = {
        'total': len(valid),
        'normals': int(valid_normals),
        'anomalies': int(valid_anomalies),
        'anomaly_rate': float(valid_anomalies / len(valid))
    }

    # Feature types
    feature_cols = [c for c in train.columns if c not in ['target', 'Id', 'index']]
    numeric_features = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"\nFeature Analysis:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")

    stats_dict['features'] = {
        'total': len(feature_cols),
        'numeric': len(numeric_features),
        'categorical': len(categorical_features)
    }

    # Save statistics
    with open(OUTPUT_DIR / "basic_statistics.json", 'w') as f:
        json.dump(stats_dict, f, indent=2)

    return stats_dict, feature_cols, numeric_features, categorical_features

def analyze_missing_values(train, valid, test):
    """Analyze missing values across datasets"""
    print("\n" + "="*60)
    print("MISSING VALUE ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (data, name, ax) in enumerate(zip([train, valid, test],
                                                 ['Train', 'Valid', 'Test'],
                                                 axes)):
        missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]

        if len(missing_pct) > 0:
            missing_pct[:20].plot(kind='barh', ax=ax, color='coral')
            ax.set_title(f'{name} Set - Top 20 Missing Features')
            ax.set_xlabel('Missing %')
            print(f"\n{name} - Features with >5% missing:")
            print(missing_pct[missing_pct > 5])
        else:
            ax.text(0.5, 0.5, 'No Missing Values',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} Set - No Missing Values')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    return missing_pct

def plot_class_distribution(train, valid):
    """Visualize class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train distribution
    train_counts = train['target'].value_counts()
    axes[0].bar(['Normal', 'Anomaly'], train_counts.values, color=['green', 'red'], alpha=0.7)
    axes[0].set_title('Train Set: Class Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v + 100, f'{v}\n({v/len(train)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

    # Valid distribution
    valid_counts = valid['target'].value_counts()
    axes[1].bar(['Normal', 'Anomaly'], valid_counts.values, color=['green', 'red'], alpha=0.7)
    axes[1].set_title('Validation Set: Class Distribution')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(valid_counts.values):
        axes[1].text(i, v + 100, f'{v}\n({v/len(valid)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_numeric_features(train, numeric_features):
    """Analyze numeric features: distributions and statistics"""
    print("\n" + "="*60)
    print("NUMERIC FEATURE ANALYSIS")
    print("="*60)

    # Select top features by variance for visualization
    variances = train[numeric_features].var().sort_values(ascending=False)
    top_features = variances.head(20).index.tolist()

    print(f"\nTop 10 features by variance:")
    print(variances.head(10))

    # Distribution comparison: Normal vs Anomaly
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.ravel()

    for idx, feat in enumerate(top_features):
        if idx >= 20:
            break

        normal_data = train[train['target']==0][feat].dropna()
        anomaly_data = train[train['target']==1][feat].dropna()

        axes[idx].hist(normal_data, bins=50, alpha=0.6, label='Normal', color='green', density=True)
        axes[idx].hist(anomaly_data, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
        axes[idx].set_title(f'{feat[:30]}', fontsize=9)
        axes[idx].legend(fontsize=8)
        axes[idx].tick_params(labelsize=8)

    plt.suptitle('Top 20 Numeric Features: Distribution by Class', fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Statistical tests for top features
    print("\nKolmogorov-Smirnov Test (Normal vs Anomaly):")
    ks_results = []
    for feat in top_features[:10]:
        normal_data = train[train['target']==0][feat].dropna()
        anomaly_data = train[train['target']==1][feat].dropna()

        if len(normal_data) > 0 and len(anomaly_data) > 0:
            ks_stat, p_value = stats.ks_2samp(normal_data, anomaly_data)
            ks_results.append({
                'feature': feat,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            print(f"  {feat[:40]:40s}: KS={ks_stat:.4f}, p={p_value:.4e} {'***' if p_value < 0.001 else ''}")

    # Save KS test results
    pd.DataFrame(ks_results).to_csv(OUTPUT_DIR / "ks_test_results.csv", index=False)

    return variances, ks_results

def correlation_analysis(train, numeric_features):
    """Analyze feature correlations"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Compute correlation matrix
    corr_with_target = train[numeric_features + ['target']].corr()['target'].drop('target')
    corr_with_target = corr_with_target.abs().sort_values(ascending=False)

    print("\nTop 20 features correlated with target:")
    print(corr_with_target.head(20))

    # Plot top correlations
    fig, ax = plt.subplots(figsize=(10, 8))
    top_corr = corr_with_target.head(25)
    top_corr.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Absolute Correlation with Target')
    ax.set_title('Top 25 Features by Correlation with Target')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_with_target.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation heatmap of top features
    top_features = corr_with_target.head(15).index.tolist()
    corr_matrix = train[top_features + ['target']].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Correlation Heatmap: Top 15 Features + Target')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save correlation results
    corr_with_target.to_csv(OUTPUT_DIR / "correlation_with_target.csv")

    return corr_with_target

def analyze_outliers(train, numeric_features):
    """Detect and analyze outliers using IQR method"""
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS")
    print("="*60)

    outlier_counts = {}

    for feat in numeric_features[:30]:  # Top 30 features
        data = train[feat].dropna()
        if len(data) > 0:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_pct = outliers / len(data) * 100

            if outlier_pct > 0:
                outlier_counts[feat] = outlier_pct

    # Plot outlier percentages
    if outlier_counts:
        outlier_df = pd.Series(outlier_counts).sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        outlier_df.plot(kind='barh', ax=ax, color='orange')
        ax.set_xlabel('Outlier %')
        ax.set_title('Top 20 Features by Outlier Percentage (IQR Method)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "outlier_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nTop 10 features with outliers:")
        print(outlier_df.head(10))

    return outlier_counts

def temporal_analysis(train):
    """Analyze temporal patterns if date features exist"""
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)

    # Look for date-related columns
    date_cols = [c for c in train.columns if any(x in c.lower() for x in ['date', 'time', 'year', 'month', 'quarter'])]

    if date_cols:
        print(f"\nFound {len(date_cols)} temporal columns:")
        print(date_cols[:10])

        # Analyze temporal patterns for first few date columns
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()

        for idx, col in enumerate(date_cols[:4]):
            if idx >= 4:
                break

            if train[col].dtype in [np.int64, np.float64]:
                # Numeric temporal feature
                normal_temporal = train[train['target']==0][col].dropna()
                anomaly_temporal = train[train['target']==1][col].dropna()

                axes[idx].hist(normal_temporal, bins=30, alpha=0.6, label='Normal', color='green')
                axes[idx].hist(anomaly_temporal, bins=30, alpha=0.6, label='Anomaly', color='red')
                axes[idx].set_title(f'{col}')
                axes[idx].legend()

        plt.suptitle('Temporal Feature Distributions by Class', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("\nNo temporal columns found.")

def generate_summary_report(stats_dict, corr_with_target, variances, ks_results):
    """Generate a summary report with key findings"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)

    report = []
    report.append("# Comprehensive EDA Report: Loan Anomaly Detection\n")
    report.append("=" * 70 + "\n\n")

    report.append("## 1. Dataset Overview\n")
    report.append(f"- **Train Set**: {stats_dict['train']['total']:,} samples "
                 f"({stats_dict['train']['anomaly_rate']*100:.2f}% anomalies)\n")
    report.append(f"- **Validation Set**: {stats_dict['valid']['total']:,} samples "
                 f"({stats_dict['valid']['anomaly_rate']*100:.2f}% anomalies)\n")
    report.append(f"- **Features**: {stats_dict['features']['total']} total "
                 f"({stats_dict['features']['numeric']} numeric, "
                 f"{stats_dict['features']['categorical']} categorical)\n\n")

    report.append("## 2. Key Findings\n\n")
    report.append("### 2.1 Class Imbalance\n")
    report.append(f"- The dataset is **highly imbalanced** with anomaly rates around "
                 f"{stats_dict['train']['anomaly_rate']*100:.2f}%\n")
    report.append("- This suggests unsupervised or semi-supervised methods are appropriate\n\n")

    report.append("### 2.2 Most Discriminative Features\n")
    report.append("**Top 10 by correlation with target:**\n")
    for i, (feat, corr) in enumerate(corr_with_target.head(10).items(), 1):
        report.append(f"{i}. `{feat}`: {corr:.4f}\n")
    report.append("\n")

    report.append("### 2.3 Statistical Significance\n")
    if ks_results:
        sig_count = sum(1 for r in ks_results if r['significant'])
        report.append(f"- {sig_count}/{len(ks_results)} top features show significant "
                     f"distribution differences (KS test, p<0.05)\n")
        report.append("- This indicates clear separability between normal and anomalous loans\n\n")

    report.append("## 3. Recommendations for Modeling\n\n")
    report.append("1. **Feature Engineering**:\n")
    report.append("   - Focus on high-correlation features for better anomaly detection\n")
    report.append("   - Consider domain-specific features (e.g., amortization ratios)\n")
    report.append("   - Apply robust scaling due to presence of outliers\n\n")

    report.append("2. **Model Selection**:\n")
    report.append("   - Use unsupervised methods: LOF, Isolation Forest, Autoencoders\n")
    report.append("   - Consider ensemble approaches to capture different anomaly types\n")
    report.append("   - Evaluate using AUPRC (primary) and AUROC (secondary)\n\n")

    report.append("3. **Validation Strategy**:\n")
    report.append("   - Train ONLY on normal data (target=0)\n")
    report.append("   - Use validation set for hyperparameter tuning and model selection\n")
    report.append("   - Never fit models on validation set to avoid leakage\n\n")

    # Save report
    with open(OUTPUT_DIR / "eda_summary_report.txt", 'w') as f:
        f.writelines(report)

    print("Summary report saved to:", OUTPUT_DIR / "eda_summary_report.txt")

def main():
    """Main EDA pipeline"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EDA FOR LOAN ANOMALY DETECTION")
    print("="*60)

    # Load data
    train, valid, test = load_data()

    # Basic statistics
    stats_dict, feature_cols, numeric_features, categorical_features = basic_statistics(train, valid)

    # Class distribution
    plot_class_distribution(train, valid)

    # Missing values
    analyze_missing_values(train, valid, test)

    # Numeric features
    variances, ks_results = analyze_numeric_features(train, numeric_features)

    # Correlation analysis
    corr_with_target = correlation_analysis(train, numeric_features)

    # Outlier analysis
    analyze_outliers(train, numeric_features)

    # Temporal analysis
    temporal_analysis(train)

    # Generate summary report
    generate_summary_report(stats_dict, corr_with_target, variances, ks_results)

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - basic_statistics.json")
    print("  - class_distribution.png")
    print("  - missing_values_analysis.png")
    print("  - numeric_distributions.png")
    print("  - ks_test_results.csv")
    print("  - correlation_with_target.png")
    print("  - correlation_with_target.csv")
    print("  - correlation_heatmap.png")
    print("  - outlier_analysis.png")
    print("  - temporal_analysis.png (if applicable)")
    print("  - eda_summary_report.txt")

if __name__ == "__main__":
    main()
