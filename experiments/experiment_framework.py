#!/usr/bin/env python3
"""
Experimentation Framework for Systematic AUPRC Optimization
Tests different components and tracks improvements
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# Create experiment tracking directory
EXP_DIR = Path("experiments/results")
EXP_DIR.mkdir(parents=True, exist_ok=True)

class ExperimentTracker:
    """Track all experiments and their results"""

    def __init__(self):
        self.experiments = []
        self.best_auprc = 0.4524  # Current best from final_approach
        self.best_experiment = None

    def log_experiment(self, name, description, auprc, auroc, components, params, time_taken):
        """Log an experiment result"""
        exp = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'description': description,
            'auprc': float(auprc),
            'auroc': float(auroc),
            'improvement_vs_baseline': float((auprc / 0.1955 - 1) * 100),  # vs best baseline
            'improvement_vs_current_best': float((auprc / self.best_auprc - 1) * 100),
            'components': components,
            'params': params,
            'time_seconds': time_taken
        }

        self.experiments.append(exp)

        # Check if new best
        if auprc > self.best_auprc:
            improvement = (auprc / self.best_auprc - 1) * 100
            print(f"\n*** NEW BEST! AUPRC: {auprc:.4f} (+{improvement:.2f}% improvement) ***")
            self.best_auprc = auprc
            self.best_experiment = exp
        else:
            diff = (auprc / self.best_auprc - 1) * 100
            print(f"\nResult: AUPRC: {auprc:.4f} ({diff:+.2f}% vs best)")

        # Save after each experiment
        self.save_results()

        return exp

    def save_results(self):
        """Save all experiments to JSON"""
        with open(EXP_DIR / "all_experiments.json", 'w') as f:
            json.dump(self.experiments, f, indent=2)

        # Save summary CSV
        df = pd.DataFrame(self.experiments)
        df.to_csv(EXP_DIR / "experiments_summary.csv", index=False)

    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*80)
        print("EXPERIMENTATION SUMMARY")
        print("="*80)
        print(f"\nTotal Experiments: {len(self.experiments)}")
        print(f"Best AUPRC: {self.best_auprc:.4f}")
        print(f"Improvement over baseline (0.1955): {(self.best_auprc/0.1955-1)*100:.1f}%")

        if self.best_experiment:
            print(f"\nBest Experiment: {self.best_experiment['name']}")
            print(f"  AUPRC: {self.best_experiment['auprc']:.4f}")
            print(f"  AUROC: {self.best_experiment['auroc']:.4f}")
            print(f"  Components: {', '.join(self.best_experiment['components'])}")

        # Show top 5
        if len(self.experiments) > 0:
            df = pd.DataFrame(self.experiments)
            df_sorted = df.nlargest(5, 'auprc')
            print("\nTop 5 Experiments:")
            print(df_sorted[['name', 'auprc', 'auroc', 'improvement_vs_current_best']].to_string(index=False))

def load_data():
    """Load and do basic preprocessing"""
    train = pd.read_csv("Data/loans_train.csv")
    valid = pd.read_csv("Data/loans_valid.csv")

    y_valid = valid['target'].values

    # Get numeric features
    feature_cols = [c for c in train.columns if c not in ['target', 'Id', 'index']]
    numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_train = train[numeric_cols].fillna(0).values
    X_valid = valid[numeric_cols].fillna(0).values

    print(f"Loaded: Train {X_train.shape}, Valid {X_valid.shape}")
    print(f"Valid anomaly rate: {y_valid.mean()*100:.2f}%")

    return X_train, X_valid, y_valid, train, valid, numeric_cols

def evaluate_scores(y_true, scores):
    """Comprehensive evaluation"""
    auprc = average_precision_score(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    # F1 at optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)

    return {
        'auprc': auprc,
        'auroc': auroc,
        'f1': best_f1
    }

# Global tracker
tracker = ExperimentTracker()

def main():
    """Initialize framework"""
    print("="*80)
    print("EXPERIMENTATION FRAMEWORK INITIALIZED")
    print("="*80)
    print(f"\nCurrent Best AUPRC: {tracker.best_auprc}")
    print(f"Goal: Beat {tracker.best_auprc} and maximize AUPRC!")
    print(f"\nResults will be saved to: {EXP_DIR}")
    print("\nReady to run experiments!")

if __name__ == "__main__":
    main()
