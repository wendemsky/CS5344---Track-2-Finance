#!/usr/bin/env python3
"""
Decision Tree Based Anomaly Detection
Following project guidelines: Use decision trees as building blocks
- Decision Tree for anomaly pattern identification
- Random Forest ensemble for robustness
- Simple interpretable features
- Clean validation usage (no data leakage)
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DecisionTreeAnomalyDetector:
    """
    Decision Tree based anomaly detection
    Uses trees to learn anomaly patterns without validation data leakage
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.anomaly_tree = None
        self.forest_model = None
        self.imputation_values: Dict[str, float] = {}
        self.selected_features: List[str] = []
        np.random.seed(self.random_state)

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Clean preprocessing for decision trees"""
        data = df.copy()

        # Handle ID
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        # Remove target and ID columns from features
        feature_cols = [c for c in data.columns if c not in ['index', 'Id', 'target']]

        # Handle missing value codes
        missing_codes = {'CreditScore': 9999, 'OriginalDTI': 999, 'OriginalLTV': 999}
        for col, code in missing_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)

        # Encode categorical features
        categorical_cols = []
        for col in feature_cols:
            if col in data.columns and data[col].dtype == 'object':
                categorical_cols.append(col)

        for col in categorical_cols:
            if is_training:
                enc = LabelEncoder()
                values = data[col].fillna('MISSING').astype(str)
                enc.fit(values)
                self.encoders[col] = enc

            if col in self.encoders:
                enc = self.encoders[col]
                values = data[col].fillna('MISSING').astype(str)
                # Handle unseen values
                values = values.apply(lambda x: x if x in enc.classes_ else 'MISSING')
                data[col] = enc.transform(values)

        # Handle numerical features
        numerical_cols = []
        for col in feature_cols:
            if col in data.columns and col not in categorical_cols:
                if data[col].dtype in [np.number, 'int64', 'float64']:
                    numerical_cols.append(col)

        # Imputation for numerical columns
        if is_training:
            for col in numerical_cols:
                if col in data.columns:
                    median_val = data[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    self.imputation_values[col] = median_val

        for col in numerical_cols:
            if col in data.columns:
                fill_val = self.imputation_values.get(col, 0.0)
                data[col] = data[col].fillna(fill_val)

        return data

    def engineer_loan_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer simple features for loan anomaly detection"""

        # Risk thresholds based on industry standards
        if 'CreditScore' in data.columns:
            data['credit_risk_level'] = pd.cut(
                data['CreditScore'],
                bins=[0, 580, 620, 660, 720, 850],
                labels=[4, 3, 2, 1, 0],  # Higher number = higher risk
                include_lowest=True
            ).astype(float)

        if 'OriginalLTV' in data.columns:
            data['ltv_risk_level'] = pd.cut(
                data['OriginalLTV'],
                bins=[0, 80, 90, 95, 100, 200],
                labels=[0, 1, 2, 3, 4],  # Higher number = higher risk
                include_lowest=True
            ).astype(float)

        if 'OriginalDTI' in data.columns:
            data['dti_risk_level'] = pd.cut(
                data['OriginalDTI'],
                bins=[0, 28, 36, 43, 50, 100],
                labels=[0, 1, 2, 3, 4],  # Higher number = higher risk
                include_lowest=True
            ).astype(float)

        # Loan amount percentiles
        if 'OriginalUPB' in data.columns:
            data['loan_size_percentile'] = data['OriginalUPB'].rank(pct=True)

        # Interest rate vs credit score relationship
        if 'OriginalInterestRate' in data.columns and 'CreditScore' in data.columns:
            # Simple expected rate calculation
            expected_rate = np.maximum(3.0, 8.0 - (data['CreditScore'] - 600) / 100)
            data['rate_vs_expected'] = data['OriginalInterestRate'] - expected_rate

        return data

    def select_tree_features(self, data: pd.DataFrame) -> List[str]:
        """Select features suitable for decision trees"""
        features = []

        # Core loan characteristics
        core_features = [
            'CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB',
            'OriginalInterestRate', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers'
        ]

        for feat in core_features:
            if feat in data.columns and data[feat].nunique() > 1:
                features.append(feat)

        # Engineered risk features
        risk_features = [
            'credit_risk_level', 'ltv_risk_level', 'dti_risk_level',
            'loan_size_percentile', 'rate_vs_expected'
        ]

        for feat in risk_features:
            if feat in data.columns:
                features.append(feat)

        # Add key temporal features (simple selection)
        temporal_cols = [c for c in data.columns if isinstance(c, str) and '_' in c and c.split('_', 1)[0].isdigit()]

        # Group temporal features
        temporal_groups = {}
        for col in temporal_cols:
            parts = col.split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                feature_type = parts[1]
                temporal_groups.setdefault(feature_type, []).append(col)

        # Select key time points for most important temporal features
        important_temporal = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']

        for temp_type in important_temporal:
            if temp_type in temporal_groups:
                cols = sorted(temporal_groups[temp_type], key=lambda x: int(x.split('_')[0]))
                if len(cols) >= 6:
                    # Select beginning, middle, and end
                    key_indices = [0, len(cols)//3, 2*len(cols)//3, len(cols)-1]
                    for idx in key_indices:
                        if idx < len(cols):
                            features.append(cols[idx])

        # Quality filter
        final_features = []
        for feat in features:
            if feat in data.columns:
                if data[feat].nunique() > 1 and data[feat].notna().mean() > 0.9:
                    final_features.append(feat)

        return final_features

    def create_anomaly_labels(self, X: np.ndarray) -> np.ndarray:
        """Create pseudo-anomaly labels using statistical methods"""

        # Method 1: Statistical outliers (Z-score based)
        z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
        statistical_outliers = (z_scores > 2.5).any(axis=1)

        # Method 2: Isolation-based outliers (percentile based)
        # Calculate distance from median for each feature
        median_distances = np.abs(X - np.median(X, axis=0))
        total_distances = median_distances.sum(axis=1)
        distance_threshold = np.percentile(total_distances, 90)  # Top 10% as outliers
        distance_outliers = total_distances > distance_threshold

        # Combine methods
        combined_outliers = statistical_outliers | distance_outliers

        # Limit to reasonable proportion (around 10-15%)
        n_outliers = int(len(X) * 0.12)
        if combined_outliers.sum() > n_outliers:
            # Select top outliers by total distance
            outlier_indices = np.argsort(total_distances)[-n_outliers:]
            pseudo_labels = np.zeros(len(X))
            pseudo_labels[outlier_indices] = 1
        else:
            pseudo_labels = combined_outliers.astype(int)

        return pseudo_labels

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "DecisionTreeAnomalyDetector":
        """
        Train decision tree anomaly detector
        Validation used ONLY for evaluation, never for training
        """
        print("=== Decision Tree Anomaly Detection ===")

        # Preprocess data
        train_clean = self.preprocess_data(train_df, is_training=True)
        train_clean = self.engineer_loan_features(train_clean)

        valid_clean = self.preprocess_data(valid_df, is_training=False)
        valid_clean = self.engineer_loan_features(valid_clean)

        # Select features
        self.selected_features = self.select_tree_features(train_clean)
        print(f"Selected {len(self.selected_features)} features for decision trees")

        # Prepare training data
        X_train = train_clean[self.selected_features].fillna(0).values
        X_valid = valid_clean[self.selected_features].fillna(0).values

        # Scale data (helps with distance calculations)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # Create pseudo-anomaly labels from training data only
        print("Creating pseudo-anomaly labels from training data...")
        pseudo_labels = self.create_anomaly_labels(X_train_scaled)
        print(f"Created {pseudo_labels.sum()} anomaly examples out of {len(pseudo_labels)} ({pseudo_labels.mean():.1%})")

        # Train Decision Tree
        print("Training Decision Tree on TRAINING DATA ONLY...")
        self.anomaly_tree = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.anomaly_tree.fit(X_train_scaled, pseudo_labels)

        # Train Random Forest for ensemble
        print("Training Random Forest ensemble...")
        self.forest_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.forest_model.fit(X_train_scaled, pseudo_labels)

        print("Models trained WITHOUT using validation data")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities using decision tree ensemble"""

        # Preprocess data
        clean_data = self.preprocess_data(df, is_training=False)
        clean_data = self.engineer_loan_features(clean_data)
        X = clean_data[self.selected_features].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Get predictions from both models
        tree_probs = self.anomaly_tree.predict_proba(X_scaled)[:, 1]
        forest_probs = self.forest_model.predict_proba(X_scaled)[:, 1]

        # Ensemble combination
        ensemble_probs = 0.4 * tree_probs + 0.6 * forest_probs

        print(f"Decision tree anomaly scores: [{ensemble_probs.min():.6f}, {ensemble_probs.max():.6f}]")
        return ensemble_probs

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate decision tree detector"""
        y_true = valid_df['target'].values
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained models"""
        if self.forest_model is not None:
            importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.forest_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return pd.DataFrame()


def main():
    """Decision tree based anomaly detection"""
    print("=== Decision Tree Anomaly Detection Pipeline ===")
    print("Using Decision Trees + Random Forest (Project Guidelines)")

    # Load data
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (anomalies: {valid_df['target'].mean():.1%})")

    # Train detector
    detector = DecisionTreeAnomalyDetector(random_state=42)
    detector.fit(train_df, valid_df)

    # Show feature importance
    importance = detector.get_feature_importance()
    print(f"\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))

    # Evaluate
    ap, auc = detector.evaluate(valid_df)
    print(f"\n=== Decision Tree Results ===")
    print(f"AUPRC: {ap:.4f} (vs random: {valid_df['target'].mean():.4f})")
    print(f"AUC-ROC: {auc:.4f}")

    improvement = (ap / valid_df['target'].mean() - 1) * 100
    print(f"Improvement over random: {improvement:.1f}%")

    # Generate submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        test_scores = detector.predict_proba(test_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })

        filename = f"DECISION_TREE_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission: {filename}")

        return detector, filename, ap, auc

    except Exception as e:
        print(f"Test prediction failed: {e}")
        return detector, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== DECISION TREE PIPELINE COMPLETE ===")
    print(f"AUPRC: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"File: {submission}")