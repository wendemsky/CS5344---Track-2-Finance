#!/usr/bin/env python3
"""
Enhanced Loan Anomaly Detection
Focus on loan-specific anomaly patterns that indicate default risk:
- Payment deterioration patterns
- Unusual loan characteristics combinations
- Temporal stress indicators
- Risk concentration patterns

Key Improvements:
- Advanced temporal feature engineering for loan performance
- Loan domain-specific anomaly indicators
- One-Class SVM + Isolation Forest ensemble
- Better handling of temporal sequences
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class EnhancedLoanAnomalyDetector:
    """
    Enhanced anomaly detection focused on loan default patterns
    Combines domain expertise with advanced feature engineering
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.isolation_forest: IsolationForest = None
        self.one_class_svm: OneClassSVM = None
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
        """Enhanced preprocessing for loan data"""
        data = df.copy()

        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        static_features, temporal_features = self._detect_temporal_and_static_cols(data)

        # Handle special missing codes with domain knowledge
        special_codes = {'CreditScore': 9999, 'OriginalDTI': 999, 'OriginalLTV': 999}
        for col, code in special_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)

        # Enhanced categorical encoding
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

        # Smart numerical imputation using domain knowledge
        numerical_cols = [c for c in static_features if c not in categorical_cols]
        if is_training:
            for col in numerical_cols:
                if col in data.columns and data[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
                    # Use domain-appropriate defaults
                    if 'CreditScore' in col:
                        val = data[col].median() if data[col].notna().any() else 650
                    elif 'LTV' in col:
                        val = data[col].median() if data[col].notna().any() else 80
                    elif 'DTI' in col:
                        val = data[col].median() if data[col].notna().any() else 25
                    else:
                        val = data[col].median() if data[col].notna().any() else 0.0
                    self.imputation_values[col] = float(val)

        for col in numerical_cols:
            if col in data.columns:
                fillv = self.imputation_values.get(col, 0.0)
                data[col] = data[col].fillna(fillv)

        # Enhanced temporal imputation with forward-fill and trend analysis
        if temporal_features:
            by_type = self._group_temporal_by_type(temporal_features)
            for ftype, cols_sorted in by_type.items():
                vals = data[cols_sorted].to_numpy(dtype=float)

                # Forward fill within each row
                for i in range(vals.shape[0]):
                    row = vals[i, :]
                    last_valid = np.nan
                    for j in range(len(row)):
                        if not np.isnan(row[j]):
                            last_valid = row[j]
                        elif not np.isnan(last_valid):
                            row[j] = last_valid

                # Backward fill for leading NaNs
                for i in range(vals.shape[0]):
                    row = vals[i, :]
                    first_valid = np.nan
                    for j in range(len(row)-1, -1, -1):
                        if not np.isnan(row[j]):
                            first_valid = row[j]
                        elif not np.isnan(first_valid):
                            row[j] = first_valid

                # Fill any remaining NaNs with column medians
                for j in range(vals.shape[1]):
                    col_median = np.nanmedian(vals[:, j])
                    if np.isnan(col_median):
                        col_median = 0.0
                    vals[:, j] = np.where(np.isnan(vals[:, j]), col_median, vals[:, j])

                data[cols_sorted] = vals

        return data, static_features, temporal_features

    def create_advanced_loan_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Advanced feature engineering focused on loan default patterns"""
        new_features: List[str] = []

        # Critical loan risk indicators
        base_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB', 'OriginalInterestRate']
        available = [f for f in base_features if f in data.columns]

        # Advanced risk scoring
        if len(available) >= 3:
            # Combined risk score (multiple factors)
            risk_factors = []

            if 'CreditScore' in available:
                # Lower credit score = higher risk (invert)
                credit_risk = (850 - data['CreditScore']) / 850
                risk_factors.append(credit_risk)

            if 'OriginalLTV' in available:
                # Higher LTV = higher risk
                ltv_risk = data['OriginalLTV'] / 100
                risk_factors.append(ltv_risk)

            if 'OriginalDTI' in available:
                # Higher DTI = higher risk
                dti_risk = data['OriginalDTI'] / 50  # Cap at 50%
                risk_factors.append(dti_risk)

            if 'OriginalInterestRate' in available:
                # Higher interest rate = higher risk
                rate_risk = data['OriginalInterestRate'] / 10  # Normalize
                risk_factors.append(rate_risk)

            if risk_factors:
                combined_risk = np.mean(risk_factors, axis=0)
                data['combined_risk_score'] = combined_risk
                new_features.append('combined_risk_score')

        # Payment stress indicators
        if 'OriginalDTI' in available and 'OriginalInterestRate' in available:
            data['payment_burden'] = (data['OriginalDTI'] * data['OriginalInterestRate']) / 100
            new_features.append('payment_burden')

        if 'OriginalUPB' in available and 'OriginalInterestRate' in available:
            # Monthly payment estimate
            rate_monthly = data['OriginalInterestRate'] / 100 / 12
            data['monthly_payment_ratio'] = (data['OriginalUPB'] * rate_monthly) / 100000  # Normalize
            new_features.append('monthly_payment_ratio')

        # Loan characteristic anomalies
        if 'NumberOfUnits' in data.columns and 'OriginalUPB' in available:
            # UPB per unit (unusual values might indicate risk)
            data['upb_per_unit'] = data['OriginalUPB'] / (data['NumberOfUnits'] + 1)
            new_features.append('upb_per_unit')

        # Advanced temporal analysis for anomaly detection
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Key temporal features that indicate loan stress
        critical_temporal = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB', 'CurrentInterestRate']

        for ftype in critical_temporal:
            if ftype in by_type and len(by_type[ftype]) >= 6:
                cols = by_type[ftype]
                feat = data[cols].replace([np.inf, -np.inf], np.nan)

                # Trajectory analysis (early warning signs)
                first_half = feat.iloc[:, :len(cols)//2]
                second_half = feat.iloc[:, len(cols)//2:]

                first_avg = first_half.mean(axis=1)
                second_avg = second_half.mean(axis=1)

                # Deterioration pattern (key for default prediction)
                data[f'{ftype}_deterioration'] = (second_avg - first_avg) / (first_avg.abs() + 1e-8)
                new_features.append(f'{ftype}_deterioration')

                # Volatility (instability indicator)
                data[f'{ftype}_instability'] = feat.std(axis=1) / (feat.mean(axis=1).abs() + 1e-8)
                new_features.append(f'{ftype}_instability')

                # Recent trend (last 3-4 periods)
                if len(cols) >= 8:
                    recent = feat.iloc[:, -4:]
                    earlier = feat.iloc[:, -8:-4]
                    recent_avg = recent.mean(axis=1)
                    earlier_avg = earlier.mean(axis=1)
                    data[f'{ftype}_recent_trend'] = (recent_avg - earlier_avg) / (earlier_avg.abs() + 1e-8)
                    new_features.append(f'{ftype}_recent_trend')

                # Sudden spikes or drops (anomaly indicators)
                diffs = feat.diff(axis=1)
                max_change = diffs.abs().max(axis=1)
                data[f'{ftype}_max_change'] = max_change / (feat.mean(axis=1).abs() + 1e-8)
                new_features.append(f'{ftype}_max_change')

        # Clean engineered features
        for c in new_features:
            if c in data.columns:
                data[c] = data[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                # Clip extreme values to reduce noise
                q99 = data[c].quantile(0.99)
                q01 = data[c].quantile(0.01)
                data[c] = np.clip(data[c], q01, q99)

        return data, new_features

    def select_anomaly_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str], new_features: List[str]) -> List[str]:
        """Select features most relevant for loan anomaly detection"""
        selected: List[str] = []

        # Critical static features
        core_static = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate',
                       'OriginalDTI', 'OriginalLoanTerm']

        for col in core_static:
            if col in data.columns and data[col].nunique() > 10 and data[col].notna().mean() >= 0.9:
                selected.append(col)

        # All engineered features (they're designed for anomaly detection)
        for col in new_features:
            if col in data.columns and data[col].nunique() > 5:
                selected.append(col)

        # Strategic temporal sampling focusing on key patterns
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Include critical time points for most predictive features
        priority_temporal = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']

        for ftype in priority_temporal:
            if ftype in by_type and len(by_type[ftype]) >= 6:
                cols = by_type[ftype]
                n_cols = len(cols)
                # Sample key points: start, early-middle, late-middle, end
                key_indices = [0, n_cols//4, n_cols//2, 3*n_cols//4, n_cols-1]
                key_cols = [cols[i] for i in key_indices if i < len(cols)]
                selected.extend(key_cols)

        # Quality filter
        final = []
        for col in selected:
            if col in data.columns:
                # Check for sufficient variation and completeness
                if data[col].nunique() > 1 and data[col].notna().mean() >= 0.8:
                    # Avoid constant or near-constant features
                    std_val = data[col].std()
                    if std_val > 1e-6:
                        final.append(col)

        # Limit features to avoid curse of dimensionality while keeping most important
        return final[:60]

    def tune_ensemble_parameters(self, X_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> Tuple[Dict, Dict]:
        """Tune both Isolation Forest and One-Class SVM parameters"""

        print("Tuning ensemble parameters for maximum anomaly detection...")

        # Isolation Forest parameter grid
        if_params = {
            'n_estimators': [200, 300, 500],
            'contamination': [0.08, 0.12, 0.16],  # Closer to actual anomaly rate
            'max_samples': [0.8, 'auto'],
            'max_features': [0.8, 1.0]
        }

        # One-Class SVM parameter grid
        svm_params = {
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'nu': [0.08, 0.12, 0.16]  # Should match contamination roughly
        }

        best_if_auprc = -1
        best_if_params = None
        best_svm_auprc = -1
        best_svm_params = None

        # Tune Isolation Forest
        print("Tuning Isolation Forest...")
        for params in ParameterGrid([if_params]):
            try:
                model = IsolationForest(random_state=self.random_state, n_jobs=-1, **params)
                model.fit(X_train)
                scores = -model.score_samples(X_valid)

                # Normalize scores
                if len(scores) > 1:
                    min_s, max_s = scores.min(), scores.max()
                    if max_s > min_s:
                        scores_norm = (scores - min_s) / (max_s - min_s)
                    else:
                        scores_norm = np.full_like(scores, 0.5)
                else:
                    scores_norm = np.array([0.5])

                auprc = average_precision_score(y_valid, scores_norm)

                if auprc > best_if_auprc:
                    best_if_auprc = auprc
                    best_if_params = params

            except Exception as e:
                continue

        # Tune One-Class SVM (on smaller sample for speed)
        print("Tuning One-Class SVM...")
        n_sample = min(5000, len(X_train))
        train_sample_idx = np.random.choice(len(X_train), n_sample, replace=False)
        X_train_sample = X_train[train_sample_idx]

        for params in ParameterGrid([svm_params]):
            try:
                model = OneClassSVM(**params)
                model.fit(X_train_sample)
                scores = -model.score_samples(X_valid)

                # Normalize scores
                if len(scores) > 1:
                    min_s, max_s = scores.min(), scores.max()
                    if max_s > min_s:
                        scores_norm = (scores - min_s) / (max_s - min_s)
                    else:
                        scores_norm = np.full_like(scores, 0.5)
                else:
                    scores_norm = np.array([0.5])

                auprc = average_precision_score(y_valid, scores_norm)

                if auprc > best_svm_auprc:
                    best_svm_auprc = auprc
                    best_svm_params = params

            except Exception as e:
                continue

        # Fallback parameters
        if best_if_params is None:
            best_if_params = {'n_estimators': 300, 'contamination': 0.12, 'max_samples': 'auto', 'max_features': 1.0}
        if best_svm_params is None:
            best_svm_params = {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.12}

        print(f"Best IF AUPRC: {best_if_auprc:.4f}, params: {best_if_params}")
        print(f"Best SVM AUPRC: {best_svm_auprc:.4f}, params: {best_svm_params}")

        return best_if_params, best_svm_params

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "EnhancedLoanAnomalyDetector":
        """Fit enhanced anomaly detection ensemble"""
        print("=== Enhanced Loan Anomaly Detection ===")

        # Enhanced preprocessing
        train_proc, train_static, train_temp = self.robust_preprocessing(train_df, is_training=True)
        train_proc, train_new = self.create_advanced_loan_features(train_proc, train_static, train_temp)

        valid_proc, valid_static, valid_temp = self.robust_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_advanced_loan_features(valid_proc, valid_static, valid_temp)

        # Enhanced feature selection
        self.final_features = self.select_anomaly_features(train_proc, train_static, train_temp, train_new)
        print(f"Selected {len(self.final_features)} features optimized for loan anomaly detection")

        X_train = train_proc[self.final_features].fillna(0.0)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values

        # Advanced scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # Tune ensemble parameters
        if_params, svm_params = self.tune_ensemble_parameters(X_train_scaled, X_valid_scaled, y_valid)

        # Train ensemble
        print("Training enhanced anomaly detection ensemble...")

        self.isolation_forest = IsolationForest(random_state=self.random_state, n_jobs=-1, **if_params)
        self.isolation_forest.fit(X_train_scaled)

        # Train SVM on sample for computational efficiency
        n_sample = min(10000, len(X_train_scaled))
        train_sample_idx = np.random.choice(len(X_train_scaled), n_sample, replace=False)
        X_train_sample = X_train_scaled[train_sample_idx]

        self.one_class_svm = OneClassSVM(**svm_params)
        self.one_class_svm.fit(X_train_sample)

        print("Enhanced ensemble trained on TRAINING DATA ONLY")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using enhanced ensemble"""
        proc, st, tp = self.robust_preprocessing(df, is_training=False)
        proc, _ = self.create_advanced_loan_features(proc, st, tp)
        X = proc[self.final_features].fillna(0.0)
        X_scaled = self.scaler.transform(X)

        # Get scores from both models
        if_scores = -self.isolation_forest.score_samples(X_scaled)
        svm_scores = -self.one_class_svm.score_samples(X_scaled)

        # Normalize individual scores
        def normalize_scores(scores):
            if len(scores) > 1:
                min_s, max_s = scores.min(), scores.max()
                if max_s > min_s:
                    return (scores - min_s) / (max_s - min_s)
                else:
                    return np.full_like(scores, 0.5)
            return np.array([0.5])

        if_norm = normalize_scores(if_scores)
        svm_norm = normalize_scores(svm_scores)

        # Ensemble combination (weight based on expected strengths)
        ensemble_scores = 0.6 * if_norm + 0.4 * svm_norm  # IF slightly better for this data

        print(f"Enhanced anomaly scores range: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
        return ensemble_scores

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate enhanced ensemble"""
        y_true = valid_df['target'].values
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc


def main():
    """Enhanced anomaly detection pipeline"""
    print("=== Enhanced Loan Anomaly Detection Pipeline ===")

    # Load data
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (target: {valid_df['target'].value_counts(normalize=True).round(3).to_dict()})")

    # Train enhanced detector
    detector = EnhancedLoanAnomalyDetector(random_state=42)
    detector.fit(train_df, valid_df)

    # Evaluate
    ap, auc = detector.evaluate(valid_df)
    print(f"\n=== Enhanced Results (No Data Leakage) ===")
    print(f"AUPRC: {ap:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

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

        filename = f"ENHANCED_LOAN_ANOMALY_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission: {filename}")

        return detector, filename, ap, auc

    except Exception as e:
        print(f"Test prediction failed: {e}")
        return detector, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== ENHANCED PIPELINE COMPLETE ===")
    print(f"Final AUPRC: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"File: {submission}")