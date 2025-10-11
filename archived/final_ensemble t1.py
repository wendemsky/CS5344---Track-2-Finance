#!/usr/bin/env python3
"""
Final Kaggle Competition Ensemble (Leak-free)
- Treats the problem as anomaly detection + supervised calibrator
- NO label leakage, NO cross-loan temporal leakage
- Clean holdout evaluation; optional finalization before test scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import xgboost as xgb
from datetime import datetime
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class FinalLoanEnsemble:
    """Leak-free production-ready ensemble for competition"""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.models: Dict[str, object] = {}
        self.imputation_values: Dict[str, float] = {}
        self.final_features: List[str] = []
        np.random.seed(self.random_state)

    # -----------------------------
    # UTILITIES
    # -----------------------------
    @staticmethod
    def _detect_temporal_and_static_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        static_features, temporal_features = [], []
        for col in df.columns:
            if col in ['index', 'Id', 'target']:
                continue
            # temporal columns are like "0_FeatureName", "1_FeatureName", ...
            if isinstance(col, str) and '_' in col and col.split('_', 1)[0].isdigit():
                temporal_features.append(col)
            else:
                static_features.append(col)
        return static_features, temporal_features

    @staticmethod
    def _group_temporal_by_type(temporal_features: List[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for col in temporal_features:
            _, ftype = col.split('_', 1)
            groups.setdefault(ftype, []).append(col)
        # sort each group's columns by time index
        for ftype, cols in groups.items():
            groups[ftype] = sorted(cols, key=lambda c: int(c.split('_', 1)[0]))
        return groups

    # -----------------------------
    # PREPROCESSING (FIT ON TRAIN ONLY)
    # -----------------------------
    def robust_preprocessing(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str], List[str]]:
        data = df.copy()

        # Unify index column name
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']

        static_features, temporal_features = self._detect_temporal_and_static_cols(data)

        # Replace special missing codes with NaN (train-only to learn imputers)
        special_codes = {
            'CreditScore': 9999,
            'OriginalDTI': 999,
            'OriginalLTV': 999,
        }
        for col, code in special_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)

        # Categorical encoding (fit encoders on train only)
        categorical_cols = [c for c in static_features if data[c].dtype == 'object']
        for col in categorical_cols:
            if is_training:
                enc = LabelEncoder()
                # include placeholders
                values = data[col].fillna('MISSING').astype(str).unique().tolist()
                # ensure UNKNOWN present
                if 'UNKNOWN' not in values:
                    values.append('UNKNOWN')
                enc.fit(values)
                self.encoders[col] = enc
            # transform with safe handling of unseen labels
            enc = self.encoders[col]
            data[col] = data[col].fillna('MISSING').astype(str)
            data[col] = data[col].apply(lambda x: x if x in enc.classes_ else 'UNKNOWN')
            data[col] = enc.transform(data[col])

        # Numerical imputation (fit medians on train only)
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

        # Temporal imputation: PER-LOAN across time (no cross-loan fill!)
        if temporal_features:
            by_type = self._group_temporal_by_type(temporal_features)
            for ftype, cols_sorted in by_type.items():
                vals = data[cols_sorted].to_numpy(dtype=float)
                # Forward fill across time within each row
                mask = np.isnan(vals)
                idx = np.where(~mask, np.arange(vals.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                vals_ffill = vals[np.arange(vals.shape[0])[:, None], idx]
                # Backward fill for leading NaNs
                mask2 = np.isnan(vals_ffill)
                idx2 = np.where(~mask2, np.arange(vals_ffill.shape[1]), vals_ffill.shape[1]-1)
                np.minimum.accumulate(idx2[:, ::-1], axis=1, out=idx2[:, ::-1])
                vals_fbf = vals_ffill[np.arange(vals_ffill.shape[0])[:, None], idx2]
                # Zero-fill remaining all-NaN rows
                vals_fbf = np.nan_to_num(vals_fbf, nan=0.0)
                data[cols_sorted] = vals_fbf

        return data, static_features, temporal_features

    # -----------------------------
    # FEATURE ENGINEERING (TRAIN-ONLY DEFINES FEATURE LIST)
    # -----------------------------
    def create_competition_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        new_features: List[str] = []

        # Core ratios from static
        base_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB', 'OriginalInterestRate']
        available = [f for f in base_features if f in data.columns]

        if 'CreditScore' in available and 'OriginalLTV' in available:
            data['credit_ltv_ratio'] = data['CreditScore'] / (data['OriginalLTV'] + 1.0)
            new_features.append('credit_ltv_ratio')
        if 'OriginalDTI' in available and 'CreditScore' in available:
            data['dti_credit_ratio'] = data['OriginalDTI'] / (data['CreditScore'] / 100.0 + 1.0)
            new_features.append('dti_credit_ratio')
        if all(f in available for f in ['OriginalUPB', 'OriginalInterestRate']):
            data['monthly_rate'] = data['OriginalInterestRate'] / 100.0 / 12.0
            data['payment_burden'] = data['OriginalUPB'] * data['monthly_rate']
            new_features += ['monthly_rate', 'payment_burden']

        # Temporal aggregates on priority types
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}
        for ftype in ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']:
            if ftype in by_type and len(by_type[ftype]) >= 4:
                cols = by_type[ftype]
                feat = data[cols].replace([np.inf, -np.inf], np.nan)
                first = feat.iloc[:, 0].abs() + 1.0
                mean_abs = feat.mean(axis=1).abs() + 1.0
                # trend, volatility, recent change (use last-3 safely)
                data[f'{ftype}_trend'] = (feat.iloc[:, -1] - feat.iloc[:, 0]) / first
                data[f'{ftype}_volatility'] = feat.std(axis=1) / mean_abs
                j = max(len(cols) - 3, 1)
                data[f'{ftype}_recent_change'] = (feat.iloc[:, -1] - feat.iloc[:, j]) / (feat.iloc[:, j].abs() + 1.0)
                new_features += [f'{ftype}_trend', f'{ftype}_volatility', f'{ftype}_recent_change']

        # Clean engineered
        for c in new_features:
            data[c] = data[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return data, new_features

    def select_final_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str], new_features: List[str]) -> List[str]:
        core_static = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate',
                       'OriginalDTI', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']
        selected: List[str] = []

        for col in core_static:
            if col in data.columns and data[col].nunique() > 1:
                selected.append(col)

        for col in new_features:
            if col in data.columns and data[col].nunique() > 1:
                selected.append(col)

        # Sample temporal columns to control dimensionality
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}
        for ftype in ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']:
            if ftype in by_type:
                cols = by_type[ftype]
                sampled = cols[::3]  # every 3rd month
                selected.extend(sampled)

        # Validate: keep columns with variance and sufficient non-missing
        final = []
        for col in selected:
            if col in data.columns and data[col].nunique() > 1 and data[col].notna().mean() >= 0.8:
                final.append(col)

        return final

    # -----------------------------
    # TRAINING (NO LEAKAGE)
    # -----------------------------
    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "FinalLoanEnsemble":
        """Fit IF on train (unsupervised), fit calibrator (XGB) on a subset of valid,
        and keep a clean holdout for evaluation later.
        """
        # Preprocess train (fit encoders, imputers)
        train_proc, train_static, train_temp = self.robust_preprocessing(train_df, is_training=True)
        train_proc, train_new = self.create_competition_features(train_proc, train_static, train_temp)

        # Preprocess valid (transform only)
        valid_proc, valid_static, valid_temp = self.robust_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_competition_features(valid_proc, valid_static, valid_temp)

        # Feature selection based ONLY on train
        self.final_features = self.select_final_features(train_proc, train_static, train_temp, train_new)

        X_train = train_proc[self.final_features].fillna(0.0)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values

        # Scale on train only
        self.scalers['main'] = StandardScaler()
        X_train_s = self.scalers['main'].fit_transform(X_train)
        X_valid_s = self.scalers['main'].transform(X_valid)

        # Stage 1: Isolation Forest (unsupervised)
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=400,
            contamination='auto',  # NOT using validation prevalence
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.models['isolation_forest'].fit(X_train_s)

        # Compute IF scores for valid
        if_scores_raw = self.models['isolation_forest'].score_samples(X_valid_s)
        if_scores = -if_scores_raw  # higher = more anomalous
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        X_valid_aug = np.column_stack([X_valid_s, if_scores_norm])

        # Split valid into calibrator-train vs calibrator-holdout (clean holdout)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=self.random_state)
        for idx_train, idx_hold in sss.split(X_valid_aug, y_valid):
            X_cal_train, X_hold = X_valid_aug[idx_train], X_valid_aug[idx_hold]
            y_cal_train, y_hold = y_valid[idx_train], y_valid[idx_hold]

        # Early stopping split inside calibrator-train (no touching holdout)
        X_cal_tr, X_cal_eval, y_cal_tr, y_cal_eval = train_test_split(
            X_cal_train, y_cal_train, test_size=0.2, random_state=self.random_state, stratify=y_cal_train
        )

        # Stage 2: XGBoost calibrator
        pos = max((y_cal_tr == 1).sum(), 1)
        neg = max((y_cal_tr == 0).sum(), 1)
        spw = neg / pos
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            scale_pos_weight=spw,
            eval_metric='auc',
            early_stopping_rounds=50,
            tree_method='hist',
            verbosity=0,
        )
        self.models['xgboost'].fit(
            X_cal_tr, y_cal_tr,
            eval_set=[(X_cal_eval, y_cal_eval)],
            verbose=False,
        )

        # Save holdout for evaluation
        self._holdout = {'X': X_hold, 'y': y_hold}
        return self

    # -----------------------------
    # EVALUATION ON CLEAN HOLDOUT
    # -----------------------------
    def evaluate_holdout(self) -> Tuple[float, float]:
        assert hasattr(self, '_holdout'), "Call fit() first to prepare holdout."
        Xh, yh = self._holdout['X'], self._holdout['y']
        proba = self.models['xgboost'].predict_proba(Xh)[:, 1]
        ap = average_precision_score(yh, proba)
        auc = roc_auc_score(yh, proba)
        return ap, auc

    # -----------------------------
    # FINALIZE (OPTIONAL) & PREDICT
    # -----------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Preprocess
        proc, st, tp = self.robust_preprocessing(df, is_training=False)
        proc, _ = self.create_competition_features(proc, st, tp)
        X = proc[self.final_features].fillna(0.0)
        Xs = self.scalers['main'].transform(X)
        # IF scores
        if_raw = self.models['isolation_forest'].score_samples(Xs)
        if_scores = -if_raw
        if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        X_aug = np.column_stack([Xs, if_norm])
        # Calibrator
        proba = self.models['xgboost'].predict_proba(X_aug)[:, 1]
        # Simple blend with IF score (keeps anomaly signal even if calibrator underfits)
        final_scores = 0.3 * if_norm + 0.7 * proba
        return final_scores

    def finalize_with_all_valid_labels(self, valid_df: pd.DataFrame) -> None:
        """After evaluation, optionally refit calibrator on ALL of loans_valid (labels used),
        keeping IF fixed (trained on train only). This is for TEST scoring only (no reporting).
        """
        valid_proc, valid_static, valid_temp = self.robust_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_competition_features(valid_proc, valid_static, valid_temp)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values
        X_valid_s = self.scalers['main'].transform(X_valid)
        if_raw = self.models['isolation_forest'].score_samples(X_valid_s)
        if_scores = -if_raw
        if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        X_valid_aug = np.column_stack([X_valid_s, if_norm])

        # Early stopping with a small internal split
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X_valid_aug, y_valid, test_size=0.15, random_state=self.random_state, stratify=y_valid
        )
        pos = max((y_tr == 1).sum(), 1)
        neg = max((y_tr == 0).sum(), 1)
        spw = neg / pos

        xgb_final = xgb.XGBClassifier(
            n_estimators=1200,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            scale_pos_weight=spw,
            eval_metric='auc',
            early_stopping_rounds=50,
            tree_method='hist',
            verbosity=0,
        )
        xgb_final.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
        self.models['xgboost'] = xgb_final


# -----------------------------
# MAIN PIPELINE (LEAK-FREE)
# -----------------------------

def main(finalize_for_production: bool = True):
    print("=== Starting Leak-free Competition Pipeline ===")

    # Load data
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape} (targets unique={train_df.get('target', pd.Series()).unique() if 'target' in train_df else 'N/A'})")
    print(f"Valid: {valid_df.shape} (targets distribution={valid_df['target'].value_counts(normalize=True).round(3).to_dict()})")

    # Fit model (no leakage)
    model = FinalLoanEnsemble(random_state=42)
    model.fit(train_df, valid_df)

    # Evaluate on clean holdout only
    ap, auc = model.evaluate_holdout()
    print("\n=== Clean Holdout Evaluation (no train/tune overlap) ===")
    print(f"Average Precision: {ap:.4f}")
    print(f"AUC-ROC:          {auc:.4f}")

    # Optionally finalize calibrator on ALL of valid labels before TEST scoring
    if finalize_for_production:
        print("\nFinalizing calibrator on ALL of loans_valid (for test scoring only)...")
        model.finalize_with_all_valid_labels(valid_df)

    # Test scoring & submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        scores = model.predict_proba(test_df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        sub = pd.DataFrame({id_col: test_df[id_col], 'anomaly_score': scores})
        fname = f"FINAL_LEAKFREE_AP{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        sub.to_csv(fname, index=False)
        print(f"\nSUCCESS! Submission saved: {fname}")
        return model, fname, ap, auc
    except Exception as e:
        print(f"\nTest submission skipped or failed: {e}")
        return model, None, ap, auc


if __name__ == '__main__':
    _model, _submission, _ap, _auc = main(finalize_for_production=True)
    print("\n=== PIPELINE COMPLETE ===")
    print(f"Holdout AP={_ap:.4f}, AUC={_auc:.4f}")
    if _submission:
        print(f"Submission file: {_submission}")
