#!/usr/bin/env python3
"""
Domain Expert Loan Anomaly Detection
Focus on financial domain expertise to identify true loan anomalies:
- Payment shock patterns
- Unusual borrower profiles
- Loan-to-value deterioration
- Cash flow stress indicators
- Market timing anomalies

Key insight: Use One-Class SVM as primary method
- Better for complex, non-linear anomaly boundaries
- Excellent with high-dimensional feature spaces
- Captures subtle interaction patterns
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DomainExpertAnomalyDetector:
    """
    Domain expert anomaly detection using financial knowledge
    Primary method: One-Class SVM (better for complex patterns)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.model: OneClassSVM = None
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

    def domain_preprocessing(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Domain-expert preprocessing"""
        data = df.copy()

        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        static_features, temporal_features = self._detect_temporal_and_static_cols(data)

        # Domain-specific missing value handling
        special_codes = {'CreditScore': 9999, 'OriginalDTI': 999, 'OriginalLTV': 999}
        for col, code in special_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)

        # Smart categorical encoding
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

        # Financial domain imputation
        numerical_cols = [c for c in static_features if c not in categorical_cols]
        if is_training:
            for col in numerical_cols:
                if col in data.columns and data[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
                    if 'CreditScore' in col:
                        val = data[col].median() if data[col].notna().any() else 620  # Subprime threshold
                    elif 'LTV' in col:
                        val = data[col].median() if data[col].notna().any() else 80   # Conservative LTV
                    elif 'DTI' in col:
                        val = data[col].median() if data[col].notna().any() else 28   # QM rule
                    elif 'InterestRate' in col:
                        val = data[col].median() if data[col].notna().any() else 4.5  # Historical average
                    else:
                        val = data[col].median() if data[col].notna().any() else 0.0
                    self.imputation_values[col] = float(val)

        for col in numerical_cols:
            if col in data.columns:
                fillv = self.imputation_values.get(col, 0.0)
                data[col] = data[col].fillna(fillv)

        # Enhanced temporal processing
        if temporal_features:
            by_type = self._group_temporal_by_type(temporal_features)
            for ftype, cols_sorted in by_type.items():
                vals = data[cols_sorted].to_numpy(dtype=float)

                # Smooth forward fill (avoid unrealistic jumps)
                for i in range(vals.shape[0]):
                    row_vals = vals[i, :]
                    last_valid = np.nan
                    for j in range(len(row_vals)):
                        if not np.isnan(row_vals[j]):
                            last_valid = row_vals[j]
                        elif not np.isnan(last_valid):
                            row_vals[j] = last_valid

                # Backward fill and clean
                vals = np.where(np.isnan(vals), 0.0, vals)
                data[cols_sorted] = vals

        return data, static_features, temporal_features

    def create_financial_anomaly_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Create features based on financial domain expertise"""
        new_features: List[str] = []

        # Core financial ratios
        base_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB', 'OriginalInterestRate', 'OriginalLoanTerm']
        available = [f for f in base_features if f in data.columns]

        # 1. DEBT-TO-INCOME STRESS INDICATORS
        if 'OriginalDTI' in available:
            # DTI categories (regulatory thresholds)
            data['dti_high_risk'] = (data['OriginalDTI'] > 43).astype(float)  # QM rule
            data['dti_extreme_risk'] = (data['OriginalDTI'] > 50).astype(float)  # Very high risk
            new_features.extend(['dti_high_risk', 'dti_extreme_risk'])

        # 2. LOAN-TO-VALUE RISK CATEGORIES
        if 'OriginalLTV' in available:
            # LTV risk buckets
            data['ltv_high_risk'] = (data['OriginalLTV'] > 90).astype(float)  # High LTV
            data['ltv_extreme_risk'] = (data['OriginalLTV'] > 95).astype(float)  # Very high LTV
            new_features.extend(['ltv_high_risk', 'ltv_extreme_risk'])

        # 3. CREDIT SCORE RISK TIERS
        if 'CreditScore' in available:
            # Credit score categories
            data['credit_subprime'] = (data['CreditScore'] < 620).astype(float)  # Subprime
            data['credit_deep_subprime'] = (data['CreditScore'] < 580).astype(float)  # Deep subprime
            data['credit_excellent'] = (data['CreditScore'] > 740).astype(float)  # Prime
            new_features.extend(['credit_subprime', 'credit_deep_subprime', 'credit_excellent'])

        # 4. COMBINED RISK INDICATORS
        if len(available) >= 3:
            # Triple risk combination (all three high risk factors)
            if all(f in data.columns for f in ['dti_high_risk', 'ltv_high_risk', 'credit_subprime']):
                data['triple_risk'] = data['dti_high_risk'] * data['ltv_high_risk'] * data['credit_subprime']
                new_features.append('triple_risk')

            # Risk mismatch: excellent credit but high LTV/DTI (unusual pattern)
            if all(f in data.columns for f in ['credit_excellent', 'dti_high_risk', 'ltv_high_risk']):
                data['risk_mismatch'] = data['credit_excellent'] * (data['dti_high_risk'] + data['ltv_high_risk'])
                new_features.append('risk_mismatch')

        # 5. LOAN SIZE ANOMALIES
        if 'OriginalUPB' in available:
            # Loan size percentiles (unusually large or small loans)
            upb_p10 = data['OriginalUPB'].quantile(0.10)
            upb_p90 = data['OriginalUPB'].quantile(0.90)
            data['loan_size_extreme'] = ((data['OriginalUPB'] < upb_p10) | (data['OriginalUPB'] > upb_p90)).astype(float)
            new_features.append('loan_size_extreme')

        # 6. INTEREST RATE ANOMALIES
        if 'OriginalInterestRate' in available and 'CreditScore' in available:
            # Rate vs credit score mismatch
            expected_rate = 8.0 - (data['CreditScore'] - 600) * 0.01  # Rough approximation
            rate_deviation = abs(data['OriginalInterestRate'] - expected_rate)
            data['rate_anomaly'] = (rate_deviation > 2.0).astype(float)  # More than 2% deviation
            new_features.append('rate_anomaly')

        # 7. TEMPORAL PATTERN ANALYSIS (KEY FOR DEFAULTS)
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Critical temporal features for default prediction
        critical_temporal = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']

        for ftype in critical_temporal:
            if ftype in by_type and len(by_type[ftype]) >= 8:
                cols = by_type[ftype]
                feat = data[cols].replace([np.inf, -np.inf], np.nan).fillna(0)

                # Early warning: first 3 months vs last 3 months
                early_period = feat.iloc[:, :3].mean(axis=1)
                late_period = feat.iloc[:, -3:].mean(axis=1)

                # Deterioration pattern (critical for defaults)
                deterioration = (early_period - late_period) / (early_period.abs() + 1e-8)
                data[f'{ftype}_deterioration_severe'] = (deterioration > 0.2).astype(float)  # 20% deterioration
                new_features.append(f'{ftype}_deterioration_severe')

                # Volatility (payment instability)
                volatility = feat.std(axis=1) / (feat.mean(axis=1).abs() + 1e-8)
                data[f'{ftype}_high_volatility'] = (volatility > volatility.quantile(0.9)).astype(float)
                new_features.append(f'{ftype}_high_volatility')

                # Recent acceleration (last quarter getting worse fast)
                if len(cols) >= 12:
                    q3 = feat.iloc[:, -6:-3].mean(axis=1)  # Third quarter from end
                    q4 = feat.iloc[:, -3:].mean(axis=1)    # Last quarter
                    recent_acceleration = (q3 - q4) / (q3.abs() + 1e-8)
                    data[f'{ftype}_recent_acceleration'] = (recent_acceleration > 0.15).astype(float)
                    new_features.append(f'{ftype}_recent_acceleration')

        # 8. PORTFOLIO RISK CONCENTRATIONS
        if 'NumberOfUnits' in data.columns:
            # Multi-unit property risk
            data['multi_unit_risk'] = (data['NumberOfUnits'] > 1).astype(float)
            new_features.append('multi_unit_risk')

        if 'NumberOfBorrowers' in data.columns:
            # Single borrower risk
            data['single_borrower_risk'] = (data['NumberOfBorrowers'] == 1).astype(float)
            new_features.append('single_borrower_risk')

        # Clean and clip extreme values
        for c in new_features:
            if c in data.columns:
                data[c] = data[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                # Ensure binary features stay binary
                if data[c].max() <= 1.0 and data[c].min() >= 0.0:
                    data[c] = np.clip(data[c], 0.0, 1.0)

        return data, new_features

    def select_domain_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str], new_features: List[str]) -> List[str]:
        """Select features based on domain expertise"""
        selected: List[str] = []

        # Core financial features (always include if available and high quality)
        core_financial = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalInterestRate',
                         'OriginalUPB', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']

        for col in core_financial:
            if col in data.columns and data[col].nunique() > 5 and data[col].notna().mean() >= 0.95:
                selected.append(col)

        # All domain-engineered features (they're specifically designed)
        for col in new_features:
            if col in data.columns and data[col].nunique() > 1:
                selected.append(col)

        # Key temporal features (focus on most predictive)
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}

        # Include specific time points that matter for default prediction
        priority_temporal = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB', 'CurrentInterestRate']

        for ftype in priority_temporal:
            if ftype in by_type and len(by_type[ftype]) >= 6:
                cols = by_type[ftype]
                n_cols = len(cols)
                # Critical time points: start, 6 months, 12 months, 18 months, end
                critical_points = [0, min(5, n_cols-1), min(11, n_cols-1), min(17, n_cols-1), n_cols-1]
                critical_cols = [cols[i] for i in critical_points]
                selected.extend(critical_cols)

        # Final quality check
        final = []
        for col in selected:
            if col in data.columns and data[col].notna().mean() >= 0.8:
                # Check for sufficient variation
                if data[col].nunique() > 1:
                    std_val = data[col].std()
                    if std_val > 1e-8:
                        final.append(col)

        return list(set(final))  # Remove duplicates

    def tune_svm_parameters(self, X_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> Dict:
        """Tune One-Class SVM parameters focusing on domain-specific patterns"""
        print("Tuning One-Class SVM for financial anomaly detection...")

        param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'nu': [0.05, 0.08, 0.10, 0.12, 0.15],  # Lower values = more conservative
            'degree': [2, 3]  # For poly kernel
        }

        best_auprc = -1
        best_params = None

        # Use subset for speed while maintaining representativeness
        n_sample = min(8000, len(X_train))
        train_idx = np.random.choice(len(X_train), n_sample, replace=False)
        X_train_sample = X_train[train_idx]

        param_combinations = list(ParameterGrid([param_grid]))
        print(f"Testing {len(param_combinations)} parameter combinations...")

        for i, params in enumerate(param_combinations):
            try:
                # Skip degree parameter for non-poly kernels
                if params['kernel'] != 'poly':
                    params = {k: v for k, v in params.items() if k != 'degree'}

                model = OneClassSVM(**params)
                model.fit(X_train_sample)

                scores = -model.score_samples(X_valid)  # Higher = more anomalous

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

                if auprc > best_auprc:
                    best_auprc = auprc
                    best_params = params

                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{len(param_combinations)} combinations, best AUPRC: {best_auprc:.4f}")

            except Exception as e:
                continue

        if best_params is None:
            best_params = {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1}

        print(f"Best parameters: AUPRC={best_auprc:.4f}, {best_params}")
        return best_params

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "DomainExpertAnomalyDetector":
        """Fit domain expert anomaly detector"""
        print("=== Domain Expert Financial Anomaly Detection ===")

        # Domain preprocessing
        train_proc, train_static, train_temp = self.domain_preprocessing(train_df, is_training=True)
        train_proc, train_new = self.create_financial_anomaly_features(train_proc, train_static, train_temp)

        valid_proc, valid_static, valid_temp = self.domain_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_financial_anomaly_features(valid_proc, valid_static, valid_temp)

        # Domain feature selection
        self.final_features = self.select_domain_features(train_proc, train_static, train_temp, train_new)
        print(f"Selected {len(self.final_features)} domain-expert features")

        X_train = train_proc[self.final_features].fillna(0.0)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values

        # Standard scaling (important for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # Tune SVM parameters
        best_params = self.tune_svm_parameters(X_train_scaled, X_valid_scaled, y_valid)

        # Train final model
        print("Training domain expert One-Class SVM on FULL TRAINING DATA...")
        self.model = OneClassSVM(**best_params)
        self.model.fit(X_train_scaled)
        print("Model trained WITHOUT using validation data")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict financial anomaly probabilities"""
        proc, st, tp = self.domain_preprocessing(df, is_training=False)
        proc, _ = self.create_financial_anomaly_features(proc, st, tp)
        X = proc[self.final_features].fillna(0.0)
        X_scaled = self.scaler.transform(X)

        scores = -self.model.score_samples(X_scaled)  # Higher = more anomalous

        # Normalize to [0, 1] probabilities
        if len(scores) > 1:
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                probabilities = (scores - min_s) / (max_s - min_s)
            else:
                probabilities = np.full_like(scores, 0.5)
        else:
            probabilities = np.array([0.5])

        print(f"Domain expert anomaly scores: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
        return probabilities

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate domain expert detector"""
        y_true = valid_df['target'].values
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc


def main():
    """Domain expert anomaly detection"""
    print("=== Domain Expert Financial Anomaly Detection ===")

    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (anomalies: {valid_df['target'].sum()}/{len(valid_df)} = {valid_df['target'].mean():.1%})")

    # Train domain expert detector
    detector = DomainExpertAnomalyDetector(random_state=42)
    detector.fit(train_df, valid_df)

    # Evaluate
    ap, auc = detector.evaluate(valid_df)
    print(f"\n=== Domain Expert Results ===")
    print(f"AUPRC: {ap:.4f} (vs random: {valid_df['target'].mean():.4f})")
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

        filename = f"DOMAIN_EXPERT_SVM_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission: {filename}")

        return detector, filename, ap, auc

    except Exception as e:
        print(f"Test prediction failed: {e}")
        return detector, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== DOMAIN EXPERT COMPLETE ===")
    print(f"AUPRC: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"File: {submission}")