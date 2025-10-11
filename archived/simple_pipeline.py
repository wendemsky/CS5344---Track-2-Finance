#!/usr/bin/env python3
"""
Simple Anomaly Detection Pipeline
Following project guidelines: Use simple models as building blocks
- Decision Trees for feature importance
- Random Forest for robust anomaly scoring
- Simple clustering for outlier detection
- Clean, interpretable approach
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SimpleAnomalyPipeline:
    """
    Simple anomaly detection using basic ML models
    1. Random Forest for anomaly scoring
    2. K-Means clustering for outlier detection
    3. Simple feature engineering
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.rf_model = None
        self.kmeans_model = None
        self.imputation_values: Dict[str, float] = {}
        self.selected_features: List[str] = []
        np.random.seed(self.random_state)

    def clean_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Simple data cleaning"""
        data = df.copy()

        # Handle ID column
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        # Replace missing codes with NaN
        missing_codes = {'CreditScore': 9999, 'OriginalDTI': 999, 'OriginalLTV': 999}
        for col, code in missing_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)

        # Handle categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_cols = [c for c in categorical_cols if c not in ['index', 'Id', 'target']]

        for col in categorical_cols:
            if is_training:
                # Fit encoder on training data
                enc = LabelEncoder()
                values = data[col].fillna('MISSING').astype(str)
                enc.fit(values)
                self.encoders[col] = enc

            # Transform using fitted encoder
            enc = self.encoders[col]
            values = data[col].fillna('MISSING').astype(str)
            # Handle unseen categories
            values = values.apply(lambda x: x if x in enc.classes_ else 'MISSING')
            data[col] = enc.transform(values)

        # Handle numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = [c for c in numerical_cols if c not in ['index', 'Id', 'target']]

        if is_training:
            # Calculate medians on training data
            for col in numerical_cols:
                if col in data.columns:
                    median_val = data[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    self.imputation_values[col] = median_val

        # Fill missing values
        for col in numerical_cols:
            if col in data.columns:
                fill_value = self.imputation_values.get(col, 0.0)
                data[col] = data[col].fillna(fill_value)

        return data

    def create_simple_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple, interpretable features"""
        # Basic risk indicators
        if 'CreditScore' in data.columns:
            data['low_credit_score'] = (data['CreditScore'] < 620).astype(int)

        if 'OriginalLTV' in data.columns:
            data['high_ltv'] = (data['OriginalLTV'] > 90).astype(int)

        if 'OriginalDTI' in data.columns:
            data['high_dti'] = (data['OriginalDTI'] > 40).astype(int)

        # Combined risk flag
        risk_flags = []
        for flag in ['low_credit_score', 'high_ltv', 'high_dti']:
            if flag in data.columns:
                risk_flags.append(flag)

        if risk_flags:
            data['risk_count'] = data[risk_flags].sum(axis=1)

        # Loan size category
        if 'OriginalUPB' in data.columns:
            data['large_loan'] = (data['OriginalUPB'] > data['OriginalUPB'].quantile(0.9)).astype(int)

        return data

    def select_features(self, data: pd.DataFrame) -> List[str]:
        """Select most important features using simple criteria"""
        features = []

        # Core loan features
        core_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI',
                        'OriginalUPB', 'OriginalInterestRate', 'OriginalLoanTerm']

        for feat in core_features:
            if feat in data.columns and data[feat].nunique() > 1:
                features.append(feat)

        # Add created features
        created_features = ['low_credit_score', 'high_ltv', 'high_dti', 'risk_count', 'large_loan']
        for feat in created_features:
            if feat in data.columns:
                features.append(feat)

        # Add some temporal features (simple selection)
        temporal_cols = [c for c in data.columns if isinstance(c, str) and '_' in c and c.split('_', 1)[0].isdigit()]

        # Group by type and select key time points
        temporal_groups = {}
        for col in temporal_cols:
            if '_' in col:
                time_idx, feature_type = col.split('_', 1)
                if time_idx.isdigit():
                    temporal_groups.setdefault(feature_type, []).append(col)

        # Select first, middle, and last time points for key features
        key_temporal_types = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']

        for ftype in key_temporal_types:
            if ftype in temporal_groups:
                cols = sorted(temporal_groups[ftype], key=lambda x: int(x.split('_')[0]))
                if len(cols) >= 3:
                    # Select first, middle, last
                    selected_indices = [0, len(cols)//2, len(cols)-1]
                    for idx in selected_indices:
                        if idx < len(cols):
                            features.append(cols[idx])

        # Filter features by quality
        final_features = []
        for feat in features:
            if feat in data.columns:
                # Check for variation
                if data[feat].nunique() > 1:
                    # Check for sufficient non-null values
                    if data[feat].notna().mean() > 0.8:
                        final_features.append(feat)

        return final_features

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "SimpleAnomalyPipeline":
        """
        Fit simple anomaly detection pipeline
        Validation used ONLY for parameter selection, not training
        """
        print("=== Simple Anomaly Detection Pipeline ===")

        # Clean data
        train_clean = self.clean_data(train_df, is_training=True)
        train_clean = self.create_simple_features(train_clean)

        valid_clean = self.clean_data(valid_df, is_training=False)
        valid_clean = self.create_simple_features(valid_clean)

        # Select features
        self.selected_features = self.select_features(train_clean)
        print(f"Selected {len(self.selected_features)} features: {self.selected_features[:10]}...")

        # Prepare training data
        X_train = train_clean[self.selected_features].fillna(0)
        X_valid = valid_clean[self.selected_features].fillna(0)
        y_valid = valid_df['target']

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        print("Training models on TRAINING DATA ONLY...")

        # 1. Random Forest (using pseudo-labels from unsupervised methods)
        # Create pseudo-labels using statistical outliers
        outlier_threshold = 2.5  # Z-score threshold
        z_scores = np.abs((X_train_scaled - X_train_scaled.mean(axis=0)) / (X_train_scaled.std(axis=0) + 1e-8))
        max_z_scores = z_scores.max(axis=1)
        pseudo_labels = (max_z_scores > outlier_threshold).astype(int)

        # Balance pseudo-labels to reasonable proportion
        n_outliers = int(len(pseudo_labels) * 0.1)  # 10% outliers
        outlier_indices = np.argsort(max_z_scores)[-n_outliers:]
        pseudo_labels_balanced = np.zeros(len(pseudo_labels))
        pseudo_labels_balanced[outlier_indices] = 1

        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train_scaled, pseudo_labels_balanced)

        # 2. K-Means clustering for outlier detection
        print("Training K-Means clustering...")
        self.kmeans_model = KMeans(
            n_clusters=10,  # Multiple clusters to capture different loan types
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_model.fit(X_train_scaled)

        print("Models trained WITHOUT using validation data")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities using simple ensemble"""
        # Clean and prepare data
        clean_data = self.clean_data(df, is_training=False)
        clean_data = self.create_simple_features(clean_data)
        X = clean_data[self.selected_features].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Get Random Forest probabilities
        rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]

        # Get clustering-based anomaly scores
        cluster_distances = self.kmeans_model.transform(X_scaled)
        # Anomaly score = distance to nearest cluster center
        cluster_scores = cluster_distances.min(axis=1)

        # Normalize cluster scores to [0, 1]
        if len(cluster_scores) > 1:
            min_score = cluster_scores.min()
            max_score = cluster_scores.max()
            if max_score > min_score:
                cluster_scores_norm = (cluster_scores - min_score) / (max_score - min_score)
            else:
                cluster_scores_norm = np.full_like(cluster_scores, 0.5)
        else:
            cluster_scores_norm = np.array([0.5])

        # Simple ensemble: combine RF and clustering
        ensemble_scores = 0.6 * rf_probs + 0.4 * cluster_scores_norm

        print(f"Simple anomaly scores: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
        return ensemble_scores

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate simple pipeline"""
        y_true = valid_df['target']
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc


def main():
    """Simple anomaly detection main pipeline"""
    print("=== Simple Anomaly Detection (Project Guidelines) ===")
    print("Using simple models: Random Forest + K-Means Clustering")

    # Load data
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (anomalies: {valid_df['target'].mean():.1%})")

    # Train simple pipeline
    pipeline = SimpleAnomalyPipeline(random_state=42)
    pipeline.fit(train_df, valid_df)

    # Evaluate
    ap, auc = pipeline.evaluate(valid_df)
    print(f"\n=== Simple Pipeline Results ===")
    print(f"AUPRC: {ap:.4f} (vs random: {valid_df['target'].mean():.4f})")
    print(f"AUC-ROC: {auc:.4f}")

    improvement = (ap / valid_df['target'].mean() - 1) * 100
    print(f"Improvement over random: {improvement:.1f}%")

    # Generate submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        test_scores = pipeline.predict_proba(test_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })

        filename = f"SIMPLE_PIPELINE_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission: {filename}")

        return pipeline, filename, ap, auc

    except Exception as e:
        print(f"Test prediction failed: {e}")
        return pipeline, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== SIMPLE PIPELINE COMPLETE ===")
    print(f"AUPRC: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"File: {submission}")