#!/usr/bin/env python3
"""
Focused Loan Anomaly Detection - Simple but Effective
Key insight: Focus on the most predictive patterns for loan defaults
- Payment deterioration over time
- Risk factor combinations
- Temporal trend anomalies
- Simplified feature engineering for better performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class FocusedAnomalyDetector:
    """
    Focused anomaly detection using most predictive loan patterns
    Optimized for computational efficiency and performance
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.model: IsolationForest = None
        self.imputation_values: Dict[str, float] = {}
        self.final_features: List[str] = []
        np.random.seed(self.random_state)

    @staticmethod
    def _detect_temporal_and_static_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Separate static and temporal columns"""
        static_features, temporal_features = [], []
        for col in df.columns:
            if col in ['index', 'Id', 'target']:
                continue
            if isinstance(col, str) and '_' in col and col.split('_', 1)[0].isdigit():
                temporal_features.append(col)
            else:
                static_features.append(col)
        return static_features, temporal_features

    @staticmethod
    def _group_temporal_by_type(temporal_features: List[str]) -> Dict[str, List[str]]:
        """Group temporal features by type"""
        groups: Dict[str, List[str]] = {}
        for col in temporal_features:
            _, ftype = col.split('_', 1)
            groups.setdefault(ftype, []).append(col)
        for ftype, cols in groups.items():
            groups[ftype] = sorted(cols, key=lambda c: int(c.split('_', 1)[0]))
        return groups

    def efficient_preprocessing(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Efficient preprocessing focused on quality"""
        data = df.copy()

        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        static_features, temporal_features = self._detect_temporal_and_static_cols(data)

        # Handle missing codes
        special_codes = {'CreditScore': 9999, 'OriginalDTI': 999, 'OriginalLTV': 999}
        for col, code in special_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)

        # Categorical encoding
        categorical_cols = [c for c in static_features if data[c].dtype == 'object']
        for col in categorical_cols:
            if is_training:
                enc = LabelEncoder()
                values = data[col].fillna('MISSING').astype(str).unique().tolist()
                enc.fit(values)
                self.encoders[col] = enc

            enc = self.encoders[col]
            data[col] = data[col].fillna('MISSING').astype(str)
            data[col] = data[col].apply(lambda x: x if x in enc.classes_ else 'MISSING')
            data[col] = enc.transform(data[col])

        # Numerical imputation
        numerical_cols = [c for c in static_features if c not in categorical_cols]
        if is_training:
            for col in numerical_cols:
                if col in data.columns:
                    val = data[col].median() if data[col].notna().any() else 0.0
                    self.imputation_values[col] = float(val)

        for col in numerical_cols:
            if col in data.columns:
                fillv = self.imputation_values.get(col, 0.0)
                data[col] = data[col].fillna(fillv)

        # Simple temporal imputation
        if temporal_features:
            by_type = self._group_temporal_by_type(temporal_features)
            for ftype, cols_sorted in by_type.items():
                # Forward fill within each row, then fill remaining with 0
                data[cols_sorted] = data[cols_sorted].fillna(method='ffill', axis=1).fillna(0)

        return data, static_features, temporal_features

    def create_focused_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Create focused features for loan anomaly detection"""
        new_features: List[str] = []

        # Core risk indicators
        if 'CreditScore' in data.columns:
            data['credit_risk_score'] = np.maximum(0, 700 - data['CreditScore']) / 100  # Higher = more risky
            new_features.append('credit_risk_score')

        if 'OriginalLTV' in data.columns:
            data['ltv_risk'] = np.maximum(0, data['OriginalLTV'] - 80) / 20  # Risk above 80%
            new_features.append('ltv_risk')

        if 'OriginalDTI' in data.columns:
            data['dti_risk'] = np.maximum(0, data['OriginalDTI'] - 28) / 22  # Risk above 28%
            new_features.append('dti_risk')

        # Combined risk score
        if all(f in new_features for f in ['credit_risk_score', 'ltv_risk', 'dti_risk']):
            data['combined_risk'] = (data['credit_risk_score'] + data['ltv_risk'] + data['dti_risk']) / 3
            new_features.append('combined_risk')

        # Interest rate vs credit score relationship
        if 'OriginalInterestRate' in data.columns and 'CreditScore' in data.columns:
            # Expected rate based on credit score (rough approximation)
            expected_rate = np.maximum(3.0, 10.0 - (data['CreditScore'] - 500) / 50)
            rate_premium = data['OriginalInterestRate'] - expected_rate
            data['rate_premium'] = np.maximum(0, rate_premium)  # Premium paid
            new_features.append('rate_premium')

        # Loan size relative to typical
        if 'OriginalUPB' in data.columns:
            median_upb = data['OriginalUPB'].median()
            data['loan_size_ratio'] = data['OriginalUPB'] / median_upb
            new_features.append('loan_size_ratio')

        # Temporal patterns - focus on most critical
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Focus on CurrentActualUPB (most predictive for defaults)
        if 'CurrentActualUPB' in by_type and len(by_type['CurrentActualUPB']) >= 6:
            cols = by_type['CurrentActualUPB']
            feat = data[cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Payment trend (declining balance is good, stable/increasing is bad)
            first_val = feat.iloc[:, 0] + 1e-8
            last_val = feat.iloc[:, -1] + 1e-8
            data['balance_trend'] = (last_val - first_val) / first_val  # Positive = increasing balance (bad)
            new_features.append('balance_trend')

            # Recent deterioration (last 3 vs previous 3)
            if len(cols) >= 6:
                recent = feat.iloc[:, -3:].mean(axis=1)
                previous = feat.iloc[:, -6:-3].mean(axis=1)
                data['recent_deterioration'] = (recent - previous) / (previous + 1e-8)
                new_features.append('recent_deterioration')

        # EstimatedLTV patterns
        if 'EstimatedLTV' in by_type and len(by_type['EstimatedLTV']) >= 6:
            cols = by_type['EstimatedLTV']
            feat = data[cols].replace([np.inf, -np.inf], np.nan).fillna(80)

            # LTV increase over time (bad sign)
            first_ltv = feat.iloc[:, 0]
            last_ltv = feat.iloc[:, -1]
            data['ltv_increase'] = np.maximum(0, last_ltv - first_ltv)
            new_features.append('ltv_increase')

        # Clean features
        for c in new_features:
            if c in data.columns:
                data[c] = data[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                # Clip extreme outliers
                p99 = data[c].quantile(0.99)
                p01 = data[c].quantile(0.01)
                if p99 > p01:
                    data[c] = np.clip(data[c], p01, p99)

        return data, new_features

    def select_best_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str], new_features: List[str]) -> List[str]:
        """Select most important features"""
        selected: List[str] = []

        # Core static features (always include if good quality)
        core_static = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalInterestRate',
                      'OriginalUPB', 'OriginalLoanTerm']

        for col in core_static:
            if col in data.columns and data[col].nunique() > 10 and data[col].notna().mean() >= 0.95:
                selected.append(col)

        # All engineered features (they're focused)
        selected.extend(new_features)

        # Key temporal points
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Most important temporal features
        for ftype in ['CurrentActualUPB', 'EstimatedLTV']:
            if ftype in by_type and len(by_type[ftype]) >= 6:
                cols = by_type[ftype]
                # Key time points: start, middle, end
                key_points = [0, len(cols)//2, len(cols)-1]
                key_cols = [cols[i] for i in key_points if i < len(cols)]
                selected.extend(key_cols)

        # Quality filter
        final = []
        for col in selected:
            if col in data.columns and data[col].nunique() > 1 and data[col].notna().mean() >= 0.8:
                final.append(col)

        return final

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "FocusedAnomalyDetector":
        """Fit focused anomaly detector"""
        print("=== Focused Loan Anomaly Detection ===")

        # Process data
        train_proc, train_static, train_temp = self.efficient_preprocessing(train_df, is_training=True)
        train_proc, train_new = self.create_focused_features(train_proc, train_static, train_temp)

        valid_proc, valid_static, valid_temp = self.efficient_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_focused_features(valid_proc, valid_static, valid_temp)

        # Feature selection
        self.final_features = self.select_best_features(train_proc, train_static, train_temp, train_new)
        print(f"Selected {len(self.final_features)} focused features")

        X_train = train_proc[self.final_features].fillna(0.0)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # Quick hyperparameter search
        print("Optimizing Isolation Forest parameters...")
        best_auprc = -1
        best_params = None

        param_options = [
            {'n_estimators': 300, 'contamination': 0.1, 'max_samples': 0.8, 'max_features': 1.0},
            {'n_estimators': 500, 'contamination': 0.12, 'max_samples': 'auto', 'max_features': 0.8},
            {'n_estimators': 200, 'contamination': 0.08, 'max_samples': 0.7, 'max_features': 1.0},
        ]

        for params in param_options:
            try:
                model = IsolationForest(random_state=self.random_state, n_jobs=-1, **params)
                model.fit(X_train_scaled)
                scores = -model.score_samples(X_valid_scaled)

                if len(scores) > 1:
                    min_s, max_s = scores.min(), scores.max()
                    if max_s > min_s:
                        scores_norm = (scores - min_s) / (max_s - min_s)
                    else:
                        scores_norm = np.full_like(scores, 0.5)
                else:
                    scores_norm = np.array([0.5])

                auprc = average_precision_score(y_valid, scores_norm)

                if auprc > best_auprc:
                    best_auprc = auprc
                    best_params = params

            except Exception as e:
                continue

        if best_params is None:
            best_params = param_options[0]

        print(f"Best AUPRC: {best_auprc:.4f}, params: {best_params}")

        # Train final model
        print("Training optimized Isolation Forest on TRAINING DATA ONLY...")
        self.model = IsolationForest(random_state=self.random_state, n_jobs=-1, **best_params)
        self.model.fit(X_train_scaled)
        print("Model trained WITHOUT validation data")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities"""
        proc, st, tp = self.efficient_preprocessing(df, is_training=False)
        proc, _ = self.create_focused_features(proc, st, tp)
        X = proc[self.final_features].fillna(0.0)
        X_scaled = self.scaler.transform(X)

        scores = -self.model.score_samples(X_scaled)

        # Normalize to [0, 1]
        if len(scores) > 1:
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                probabilities = (scores - min_s) / (max_s - min_s)
            else:
                probabilities = np.full_like(scores, 0.5)
        else:
            probabilities = np.array([0.5])

        print(f"Focused anomaly scores: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
        return probabilities

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate focused detector"""
        y_true = valid_df['target'].values
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc


def main():
    """Focused anomaly detection pipeline"""
    print("=== Focused Loan Anomaly Detection ===")

    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (anomalies: {valid_df['target'].sum()}/{len(valid_df)} = {valid_df['target'].mean():.1%})")

    # Train focused detector
    detector = FocusedAnomalyDetector(random_state=42)
    detector.fit(train_df, valid_df)

    # Evaluate
    ap, auc = detector.evaluate(valid_df)
    print(f"\n=== Focused Results ===")
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

        filename = f"FOCUSED_ANOMALY_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission: {filename}")

        return detector, filename, ap, auc

    except Exception as e:
        print(f"Test prediction failed: {e}")
        return detector, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== FOCUSED PIPELINE COMPLETE ===")
    print(f"AUPRC: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"File: {submission}")