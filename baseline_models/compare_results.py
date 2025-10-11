#!/usr/bin/env python3
"""
Baseline Model Comparison Analysis
Generates visualizations and statistical comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load results
results = pd.read_csv("baseline_models/results/baseline_results.csv")

# Extract base model names
results['base_model'] = results['model'].str.extract(r'([A-Za-z_]+)')[0]

print("="*80)
print("BASELINE MODEL COMPARISON ANALYSIS")
print("="*80)

print("\n1. OVERALL BEST PERFORMING MODELS (Top 10 by AUPRC)")
print("-"*80)
top10 = results.nlargest(10, 'auprc')[['model', 'config', 'auprc', 'auroc', 'f1']]
print(top10.to_string(index=False))

print("\n2. BEST MODEL PER ALGORITHM TYPE")
print("-"*80)
best_per_algo = results.loc[results.groupby('base_model')['auprc'].idxmax()]
best_per_algo = best_per_algo.sort_values('auprc', ascending=False)
print(best_per_algo[['base_model', 'model', 'auprc', 'auroc', 'f1']].to_string(index=False))

print("\n3. PREPROCESSING IMPACT")
print("-"*80)
config_comparison = results.groupby('config').agg({
    'auprc': ['mean', 'std', 'max'],
    'auroc': ['mean', 'std', 'max']
}).round(4)
print(config_comparison)

# Visualization 1: Top 15 Models Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

top15 = results.nlargest(15, 'auprc')

# AUPRC
axes[0].barh(range(len(top15)), top15['auprc'].values, color='steelblue')
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels(top15['model'].str[:30], fontsize=9)
axes[0].set_xlabel('AUPRC', fontweight='bold')
axes[0].set_title('Top 15 Models by AUPRC', fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# AUROC
axes[1].barh(range(len(top15)), top15['auroc'].values, color='coral')
axes[1].set_yticks(range(len(top15)))
axes[1].set_yticklabels(top15['model'].str[:30], fontsize=9)
axes[1].set_xlabel('AUROC', fontweight='bold')
axes[1].set_title('Top 15 Models by AUROC', fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

# F1
axes[2].barh(range(len(top15)), top15['f1'].values, color='green')
axes[2].set_yticks(range(len(top15)))
axes[2].set_yticklabels(top15['model'].str[:30], fontsize=9)
axes[2].set_xlabel('F1 Score', fontweight='bold')
axes[2].set_title('Top 15 Models by F1 Score', fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_models/results/top15_comparison.png", dpi=300, bbox_inches='tight')
print("\nSaved: baseline_models/results/top15_comparison.png")

# Visualization 2: Algorithm Type Comparison (Box plots)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# AUPRC by algorithm
sns.boxplot(data=results, y='base_model', x='auprc', ax=axes[0], palette='Set2')
axes[0].set_xlabel('AUPRC', fontweight='bold')
axes[0].set_ylabel('Algorithm', fontweight='bold')
axes[0].set_title('AUPRC Distribution by Algorithm Type', fontweight='bold')

# AUROC by algorithm
sns.boxplot(data=results, y='base_model', x='auroc', ax=axes[1], palette='Set3')
axes[1].set_xlabel('AUROC', fontweight='bold')
axes[1].set_ylabel('Algorithm', fontweight='bold')
axes[1].set_title('AUROC Distribution by Algorithm Type', fontweight='bold')

# F1 by algorithm
sns.boxplot(data=results, y='base_model', x='f1', ax=axes[2], palette='Pastel1')
axes[2].set_xlabel('F1 Score', fontweight='bold')
axes[2].set_ylabel('Algorithm', fontweight='bold')
axes[2].set_title('F1 Score Distribution by Algorithm Type', fontweight='bold')

plt.tight_layout()
plt.savefig("baseline_models/results/algorithm_comparison_boxplots.png", dpi=300, bbox_inches='tight')
print("Saved: baseline_models/results/algorithm_comparison_boxplots.png")

# Visualization 3: Preprocessing Configuration Impact
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

config_summary = results.groupby('config').agg({
    'auprc': 'mean',
    'auroc': 'mean'
}).sort_values('auprc', ascending=False)

# AUPRC
axes[0].bar(range(len(config_summary)), config_summary['auprc'].values,
           color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(config_summary)])
axes[0].set_xticks(range(len(config_summary)))
axes[0].set_xticklabels(config_summary.index, rotation=45, ha='right')
axes[0].set_ylabel('Mean AUPRC', fontweight='bold')
axes[0].set_title('Mean AUPRC by Preprocessing Configuration', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# AUROC
axes[1].bar(range(len(config_summary)), config_summary['auroc'].values,
           color=['#d62728', '#9467bd', '#8c564b'][:len(config_summary)])
axes[1].set_xticks(range(len(config_summary)))
axes[1].set_xticklabels(config_summary.index, rotation=45, ha='right')
axes[1].set_ylabel('Mean AUROC', fontweight='bold')
axes[1].set_title('Mean AUROC by Preprocessing Configuration', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("baseline_models/results/preprocessing_impact.png", dpi=300, bbox_inches='tight')
print("Saved: baseline_models/results/preprocessing_impact.png")

# Visualization 4: AUPRC vs AUROC Scatter
fig, ax = plt.subplots(figsize=(12, 8))

colors = {'robust_pca80': 'blue', 'standard_pca80': 'orange', 'robust_pcaNone': 'green'}
for config in results['config'].unique():
    subset = results[results['config'] == config]
    ax.scatter(subset['auroc'], subset['auprc'],
              label=config, alpha=0.6, s=100, c=colors.get(config, 'gray'))

ax.set_xlabel('AUROC', fontsize=12, fontweight='bold')
ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
ax.set_title('AUPRC vs AUROC: All Baseline Models', fontsize=14, fontweight='bold')
ax.legend(title='Configuration', fontsize=10)
ax.grid(alpha=0.3)

# Annotate top 3 models
top3 = results.nlargest(3, 'auprc')
for idx, row in top3.iterrows():
    ax.annotate(row['model'][:20],
               (row['auroc'], row['auprc']),
               xytext=(10, 10), textcoords='offset points',
               fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig("baseline_models/results/auprc_vs_auroc_scatter.png", dpi=300, bbox_inches='tight')
print("Saved: baseline_models/results/auprc_vs_auroc_scatter.png")

# Summary Statistics Table
print("\n4. SUMMARY STATISTICS BY ALGORITHM")
print("-"*80)
summary_stats = results.groupby('base_model').agg({
    'auprc': ['count', 'mean', 'std', 'min', 'max'],
    'time_seconds': ['mean']
}).round(4)
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
print(summary_stats)

# Save summary tables
summary_stats.to_csv("baseline_models/results/summary_statistics.csv")
best_per_algo.to_csv("baseline_models/results/best_per_algorithm.csv", index=False)

print("\n"+"="*80)
print("COMPARISON ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - top15_comparison.png")
print("  - algorithm_comparison_boxplots.png")
print("  - preprocessing_impact.png")
print("  - auprc_vs_auroc_scatter.png")
print("  - summary_statistics.csv")
print("  - best_per_algorithm.csv")
