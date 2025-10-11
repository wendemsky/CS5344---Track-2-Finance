#!/usr/bin/env python3
"""
Best Single-Method Anomaly Detection for Loan Data
Uses Isolation Forest - optimal for this specific problem:
- Purpose-built for anomaly detection
- Handles mixed feature types (static + temporal)
- No distribution assumptions (robust for financial data)
- AUPRC-optimized for imbalanced data (87.4% normal vs 12.6% anomalies)
- Validation used ONLY for hyperparameter selection (NO training on validation)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class OptimalAnomalyDetector:
    """
    Single-method anomaly detection using Isolation Forest
    Optimized specifically for loan anomaly detection with imbalanced data
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()
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

    def robust_preprocessing(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Preprocessing pipeline - fit only on training data"""
        data = df.copy()

        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        static_features, temporal_features = self._detect_temporal_and_static_cols(data)

        # Handle special missing codes
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
                if 'UNKNOWN' not in values:
                    values.append('UNKNOWN')
                enc.fit(values)
                self.encoders[col] = enc

            enc = self.encoders[col]
            data[col] = data[col].fillna('MISSING').astype(str)
            data[col] = data[col].apply(lambda x: x if x in enc.classes_ else 'UNKNOWN')
            data[col] = enc.transform(data[col])

        # Numerical imputation
        numerical_cols = [c for c in static_features if c not in categorical_cols]
        if is_training:
            for col in numerical_cols:
                if col in data.columns and data[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
                    val = data[col].median() if data[col].notna().any() else 0.0
                    self.imputation_values[col] = float(val)

        for col in numerical_cols:
            if col in data.columns:
                fillv = self.imputation_values.get(col, 0.0)
                data[col] = data[col].fillna(fillv)

        # Temporal imputation (within-loan only)
        if temporal_features:
            by_type = self._group_temporal_by_type(temporal_features)
            for ftype, cols_sorted in by_type.items():
                vals = data[cols_sorted].to_numpy(dtype=float)
                # Forward fill within each row
                mask = np.isnan(vals)
                idx = np.where(~mask, np.arange(vals.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                vals_ffill = vals[np.arange(vals.shape[0])[:, None], idx]
                # Backward fill for leading NaNs
                mask2 = np.isnan(vals_ffill)
                idx2 = np.where(~mask2, np.arange(vals_ffill.shape[1]), vals_ffill.shape[1]-1)
                np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
                vals_fbf = vals_ffill[np.arange(vals_ffill.shape[0])[:, None], idx2]
                vals_fbf = np.nan_to_num(vals_fbf, nan=0.0)
                data[cols_sorted] = vals_fbf

        return data, static_features, temporal_features

    def create_loan_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Create loan-specific features optimized for anomaly detection"""
        new_features: List[str] = []

        # Critical loan ratios
        base_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB', 'OriginalInterestRate']
        available = [f for f in base_features if f in data.columns]

        # Risk indicators
        if 'CreditScore' in available and 'OriginalLTV' in available:
            data['risk_score'] = (100 - data['CreditScore']/10) * (data['OriginalLTV']/100)
            new_features.append('risk_score')

        if 'OriginalDTI' in available and 'OriginalInterestRate' in available:
            data['payment_stress'] = data['OriginalDTI'] * data['OriginalInterestRate'] / 100
            new_features.append('payment_stress')

        # Temporal patterns (key for anomaly detection)
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Focus on most predictive temporal features
        critical_temporal = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']

        for ftype in critical_temporal:
            if ftype in by_type and len(by_type[ftype]) >= 4:
                cols = by_type[ftype]
                feat = data[cols].replace([np.inf, -np.inf], np.nan)

                # Trend analysis (critical for loan performance)
                first_val = feat.iloc[:, 0].abs() + 1e-8
                last_val = feat.iloc[:, -1].abs() + 1e-8

                data[f'{ftype}_trend'] = (last_val - first_val) / first_val
                data[f'{ftype}_volatility'] = feat.std(axis=1) / (feat.mean(axis=1).abs() + 1e-8)

                # Detect sudden changes (anomaly indicator)
                if len(cols) >= 6:
                    recent_avg = feat.iloc[:, -3:].mean(axis=1)
                    early_avg = feat.iloc[:, :3].mean(axis=1)
                    data[f'{ftype}_change_pattern'] = (recent_avg - early_avg) / (early_avg.abs() + 1e-8)
                    new_features.append(f'{ftype}_change_pattern')

                new_features += [f'{ftype}_trend', f'{ftype}_volatility']

        # Clean engineered features
        for c in new_features:
            data[c] = data[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return data, new_features

    def select_optimal_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str], new_features: List[str]) -> List[str]:
        """Select features optimized for Isolation Forest anomaly detection"""
        # Core loan characteristics
        core_static = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate',
                       'OriginalDTI', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']

        selected: List[str] = []

        # Add high-quality static features
        for col in core_static:
            if col in data.columns and data[col].nunique() > 1 and data[col].notna().mean() >= 0.95:
                selected.append(col)

        # Add engineered features (optimized for anomaly detection)
        for col in new_features:
            if col in data.columns and data[col].nunique() > 5:
                selected.append(col)

        # Strategic temporal sampling (avoid curse of dimensionality)
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Sample key time points for most predictive temporal features
        for ftype in ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']:
            if ftype in by_type and len(by_type[ftype]) >= 6:
                cols = by_type[ftype]
                # Sample: early, middle, recent periods
                n_cols = len(cols)
                sampled_indices = [0, n_cols//3, 2*n_cols//3, n_cols-1]
                sampled_cols = [cols[i] for i in sampled_indices if i < len(cols)]
                selected.extend(sampled_cols)

        # Quality filter
        final = []
        for col in selected:
            if col in data.columns and data[col].nunique() > 1 and data[col].notna().mean() >= 0.8:
                final.append(col)

        return final[:50]  # Limit to avoid curse of dimensionality

    def tune_hyperparameters(self, X_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> Dict:
        """
        Tune Isolation Forest hyperparameters using validation set
        Uses AUPRC as primary metric (better for imbalanced data)
        Validation used ONLY for evaluation, never for training
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'max_samples': ['auto', 256, 512],
            'max_features': [1.0, 0.8, 0.6]
        }

        best_auprc = -1
        best_params = None

        print("Tuning Isolation Forest hyperparameters using AUPRC...")

        for params in ParameterGrid([param_grid]):
            try:
                # Train on training data ONLY
                model = IsolationForest(
                    random_state=self.random_state,
                    n_jobs=-1,
                    **params
                )
                model.fit(X_train)

                # Evaluate on validation set (labels used for evaluation only)
                scores = model.decision_function(X_valid)  # Higher = more normal
                anomaly_scores = -scores  # Convert to anomaly scores (higher = more anomalous)

                # Normalize to [0, 1] for proper evaluation
                if len(anomaly_scores) > 1:
                    min_score = anomaly_scores.min()
                    max_score = anomaly_scores.max()
                    if max_score > min_score:
                        anomaly_scores_norm = (anomaly_scores - min_score) / (max_score - min_score)
                    else:
                        anomaly_scores_norm = np.zeros_like(anomaly_scores)
                else:
                    anomaly_scores_norm = np.array([0.5])

                auprc = average_precision_score(y_valid, anomaly_scores_norm)

                if auprc > best_auprc:
                    best_auprc = auprc
                    best_params = params

            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        if best_params is not None:
            print(f"Best params: AUPRC={best_auprc:.4f}, {best_params}")
        else:
            # Fallback to reasonable defaults
            best_params = {
                'n_estimators': 200,
                'contamination': 0.1,
                'max_samples': 'auto',
                'max_features': 1.0
            }
            print(f"Using fallback params: {best_params}")

        return best_params

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "OptimalAnomalyDetector":
        """
        Fit Isolation Forest on training data only
        Use validation set ONLY for hyperparameter tuning (not training)
        """
        print("=== Training Isolation Forest (Best Single Method) ===")

        # Preprocess training data
        train_proc, train_static, train_temp = self.robust_preprocessing(train_df, is_training=True)
        train_proc, train_new = self.create_loan_features(train_proc, train_static, train_temp)

        # Preprocess validation data (transform only)
        valid_proc, valid_static, valid_temp = self.robust_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_loan_features(valid_proc, valid_static, valid_temp)

        # Feature selection based on training data only
        self.final_features = self.select_optimal_features(train_proc, train_static, train_temp, train_new)
        print(f"Selected {len(self.final_features)} optimal features for Isolation Forest")

        X_train = train_proc[self.final_features].fillna(0.0)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values

        # Scale features (fit on training only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # Tune hyperparameters using validation set (evaluation only)
        best_params = self.tune_hyperparameters(X_train_scaled, X_valid_scaled, y_valid)

        # Train final model on training data only
        print("Training final Isolation Forest on TRAINING DATA ONLY...")
        self.model = IsolationForest(
            random_state=self.random_state,
            n_jobs=-1,
            **best_params
        )
        self.model.fit(X_train_scaled)
        print("Model trained WITHOUT using any validation data")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly probabilities"""
        # Preprocess
        proc, st, tp = self.robust_preprocessing(df, is_training=False)
        proc, _ = self.create_loan_features(proc, st, tp)
        X = proc[self.final_features].fillna(0.0)
        X_scaled = self.scaler.transform(X)

        # Get anomaly scores
        scores = self.model.decision_function(X_scaled)  # Higher = more normal
        anomaly_scores = -scores  # Convert to anomaly scores (higher = more anomalous)

        # Normalize to [0, 1] for valid probabilities
        if len(anomaly_scores) > 1:
            min_score = anomaly_scores.min()
            max_score = anomaly_scores.max()
            if max_score > min_score:
                probabilities = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                probabilities = np.full_like(anomaly_scores, 0.5)
        else:
            probabilities = np.array([0.5])

        print(f"Anomaly probability range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
        return probabilities

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate on validation set"""
        y_true = valid_df['target'].values
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc


def main():
    """
    Optimal single-method anomaly detection pipeline
    Uses Isolation Forest - best approach for this loan dataset
    """
    print("=== Optimal Anomaly Detection: Isolation Forest ===")

    # Load data
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (target distribution: {valid_df['target'].value_counts(normalize=True).round(3).to_dict()})")
    print("Using Isolation Forest: Best single method for loan anomaly detection")
    print("AUPRC-optimized for imbalanced data (ignores true negatives)")

    # Train model (validation used only for hyperparameter tuning)
    detector = OptimalAnomalyDetector(random_state=42)
    detector.fit(train_df, valid_df)

    # Evaluate on validation set
    ap, auc = detector.evaluate(valid_df)
    print(f"\n=== Validation Results (No Data Leakage) ===")
    print(f"AUPRC (Primary): {ap:.4f}")
    print(f"AUC-ROC (Secondary): {auc:.4f}")

    # Generate test predictions
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        test_scores = detector.predict_proba(test_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })

        filename = f"ISOLATION_FOREST_OPTIMAL_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission saved: {filename}")

        return detector, filename, ap, auc

    except Exception as e:
        print(f"\nTest prediction failed: {e}")
        return detector, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Final AUPRC: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"Submission: {submission}")