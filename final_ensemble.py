#!/usr/bin/env python3
"""
Final Kaggle Competition Ensemble
Robust and high-performance approach for loan anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinalLoanEnsemble:
    """
    Production-ready ensemble for competition
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.imputation_values = {}
        
    def robust_preprocessing(self, df, is_training=True):
        """Ultra-robust preprocessing"""
        print("Robust preprocessing...")
        
        data = df.copy()
        
        # Handle ID columns
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']
        
        # Separate features
        static_features = []
        temporal_features = []
        
        for col in data.columns:
            if col in ['index', 'Id', 'target']:
                continue
            elif any(col.startswith(f"{i}_") for i in range(20)):
                temporal_features.append(col)
            else:
                static_features.append(col)
        
        print(f"Static: {len(static_features)}, Temporal: {len(temporal_features)}")
        
        # Handle special codes first
        special_codes = {
            'CreditScore': 9999,
            'OriginalDTI': 999,
            'OriginalLTV': 999
        }
        
        for col, code in special_codes.items():
            if col in data.columns:
                data[col] = data[col].replace(code, np.nan)
        
        # Categorical encoding - only for clearly categorical columns
        categorical_cols = []
        for col in static_features:
            if data[col].dtype == 'object':
                categorical_cols.append(col)
        
        for col in categorical_cols:
            if is_training:
                self.encoders[col] = LabelEncoder()
                # Handle all unique values including NaN
                values = data[col].fillna('MISSING').astype(str).unique()
                self.encoders[col].fit(list(values) + ['UNKNOWN'])
            
            # Safe encoding
            data[col] = data[col].fillna('MISSING').astype(str)
            data[col] = data[col].apply(lambda x: x if x in self.encoders[col].classes_ else 'UNKNOWN')
            data[col] = self.encoders[col].transform(data[col])
        
        # Numerical imputation - robust approach
        numerical_cols = [col for col in static_features if col not in categorical_cols]
        
        if is_training:
            for col in numerical_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    # Use median for robustness, but handle edge cases
                    valid_data = data[col].dropna()
                    if len(valid_data) > 0:
                        self.imputation_values[col] = valid_data.median()
                    else:
                        self.imputation_values[col] = 0
        
        for col in numerical_cols:
            if col in data.columns and col in self.imputation_values:
                data[col] = data[col].fillna(self.imputation_values[col])
        
        # Temporal imputation
        for col in temporal_features:
            if col in data.columns:
                # Forward fill, then backward fill, then zero
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data, static_features, temporal_features
    
    def create_competition_features(self, data, static_features, temporal_features):
        """Create features optimized for competition performance"""
        print("Creating competition features...")
        
        new_features = []
        
        # Core financial risk features
        base_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB', 'OriginalInterestRate']
        available_base = [f for f in base_features if f in data.columns]
        
        if len(available_base) >= 3:
            # Risk ratios
            if 'CreditScore' in available_base and 'OriginalLTV' in available_base:
                data['credit_ltv_ratio'] = data['CreditScore'] / (data['OriginalLTV'] + 1)
                new_features.append('credit_ltv_ratio')
            
            if 'OriginalDTI' in available_base and 'CreditScore' in available_base:
                data['dti_credit_ratio'] = data['OriginalDTI'] / (data['CreditScore'] / 100 + 1)
                new_features.append('dti_credit_ratio')
            
            # Payment burden estimate
            if all(f in available_base for f in ['OriginalUPB', 'OriginalInterestRate']):
                data['monthly_rate'] = data['OriginalInterestRate'] / 100 / 12
                data['payment_burden'] = data['OriginalUPB'] * data['monthly_rate']
                new_features.extend(['monthly_rate', 'payment_burden'])
        
        # Temporal aggregations - focus on most predictive patterns
        temporal_groups = {}
        for col in temporal_features:
            if '_' in col and col.split('_')[0].isdigit():
                feature_type = col.split('_', 1)[1]
                if feature_type not in temporal_groups:
                    temporal_groups[feature_type] = []
                temporal_groups[feature_type].append(col)
        
        # Create high-impact temporal features
        priority_types = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']
        
        for feature_type in priority_types:
            if feature_type in temporal_groups and len(temporal_groups[feature_type]) >= 8:
                cols = sorted(temporal_groups[feature_type], key=lambda x: int(x.split('_')[0]))
                feature_data = data[cols].replace([np.inf, -np.inf], np.nan)
                
                # Key temporal indicators
                data[f'{feature_type}_trend'] = (feature_data.iloc[:, -1] - feature_data.iloc[:, 0]) / (feature_data.iloc[:, 0].abs() + 1)
                data[f'{feature_type}_volatility'] = feature_data.std(axis=1) / (feature_data.mean(axis=1).abs() + 1)
                data[f'{feature_type}_recent_change'] = (feature_data.iloc[:, -1] - feature_data.iloc[:, -3]) / (feature_data.iloc[:, -3].abs() + 1)
                
                new_features.extend([f'{feature_type}_trend', f'{feature_type}_volatility', f'{feature_type}_recent_change'])
        
        # Clean up infinite and NaN values
        for col in new_features:
            if col in data.columns:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Created {len(new_features)} competition features")
        return data, new_features
    
    def select_final_features(self, data, static_features, temporal_features, new_features):
        """Select final feature set for modeling"""
        print("Selecting final features...")
        
        # Core static features - most predictive
        core_static = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate', 
                      'OriginalDTI', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']
        
        selected_features = []
        
        # Add available core static features
        for col in core_static:
            if col in data.columns and data[col].nunique() > 1:
                selected_features.append(col)
        
        # Add engineered features
        for col in new_features:
            if col in data.columns and data[col].nunique() > 1:
                selected_features.append(col)
        
        # Add key temporal features - sample to avoid overfitting
        key_temporal_types = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']
        for temp_type in key_temporal_types:
            temp_cols = [col for col in temporal_features if temp_type in col]
            if temp_cols:
                # Take every 2nd or 3rd temporal feature to reduce dimensionality
                sampled_temporal = sorted(temp_cols, key=lambda x: int(x.split('_')[0]))[::3]
                selected_features.extend(sampled_temporal)
        
        # Final validation
        final_features = []
        for col in selected_features:
            if col in data.columns:
                # Check for variance and missing data
                if data[col].nunique() > 1 and data[col].notna().sum() > len(data) * 0.8:
                    final_features.append(col)
        
        print(f"Final feature count: {len(final_features)}")
        return final_features
    
    def fit(self, train_df, valid_df):
        """Fit the competition ensemble"""
        print("=== Final Ensemble Training ===")
        
        # Process training data
        train_processed, train_static, train_temporal = self.robust_preprocessing(train_df, is_training=True)
        train_processed, train_new_features = self.create_competition_features(
            train_processed, train_static, train_temporal
        )
        
        # Process validation data
        valid_processed, valid_static, valid_temporal = self.robust_preprocessing(valid_df, is_training=False)
        valid_processed, valid_new_features = self.create_competition_features(
            valid_processed, valid_static, valid_temporal
        )
        
        # Select features
        self.final_features = self.select_final_features(
            train_processed, train_static, train_temporal, train_new_features
        )
        
        # Prepare data matrices
        X_train = train_processed[self.final_features].fillna(0)
        X_valid = valid_processed[self.final_features].fillna(0)
        y_valid = valid_df['target'].values
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_train_scaled = self.scalers['main'].fit_transform(X_train)
        X_valid_scaled = self.scalers['main'].transform(X_valid)
        
        # Stage 1: Isolation Forest (unsupervised)
        print("Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=300,
            contamination=0.126,  # Based on validation proportion
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_train_scaled)
        
        # Get anomaly scores for validation set
        if_scores_raw = self.models['isolation_forest'].score_samples(X_valid_scaled)
        if_scores = -if_scores_raw  # Higher = more anomalous
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        
        # Stage 2: XGBoost with anomaly scores
        print("Training XGBoost...")
        
        # Create augmented features
        X_valid_aug = np.column_stack([X_valid_scaled, if_scores_norm])
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=self.random_state,
            scale_pos_weight=len(y_valid[y_valid==0]) / max(len(y_valid[y_valid==1]), 1),
            eval_metric='auc',
            early_stopping_rounds=50,
            verbosity=0
        )
        
        # Train XGBoost on validation data (since it's the only labeled data)
        eval_split = int(len(X_valid_aug) * 0.75)
        X_train_xgb, X_eval_xgb = X_valid_aug[:eval_split], X_valid_aug[eval_split:]
        y_train_xgb, y_eval_xgb = y_valid[:eval_split], y_valid[eval_split:]
        
        self.models['xgboost'].fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_eval_xgb, y_eval_xgb)],
            verbose=False
        )
        
        print("=== Training Complete ===")
        return self
    
    def predict_proba(self, test_df):
        """Generate predictions"""
        print("=== Final Prediction ===")
        
        # Preprocess test data
        test_processed, test_static, test_temporal = self.robust_preprocessing(test_df, is_training=False)
        test_processed, test_new_features = self.create_competition_features(
            test_processed, test_static, test_temporal
        )
        
        # Prepare features
        X_test = test_processed[self.final_features].fillna(0)
        X_test_scaled = self.scalers['main'].transform(X_test)
        
        # Stage 1: Isolation Forest scores
        if_scores_raw = self.models['isolation_forest'].score_samples(X_test_scaled)
        if_scores = -if_scores_raw
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        
        # Stage 2: XGBoost prediction
        X_test_aug = np.column_stack([X_test_scaled, if_scores_norm])
        xgb_scores = self.models['xgboost'].predict_proba(X_test_aug)[:, 1]
        
        # Final ensemble combination
        final_scores = 0.3 * if_scores_norm + 0.7 * xgb_scores
        
        print("=== Prediction Complete ===")
        return final_scores

def main():
    """Main competition pipeline"""
    print("=== Starting Final Competition Pipeline ===")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')
    
    print(f"Training: {train_df.shape}")
    print(f"Validation: {valid_df.shape}")
    
    # Train model
    model = FinalLoanEnsemble(random_state=42)
    model.fit(train_df, valid_df)
    
    # Evaluate on validation
    print("\n=== Validation Evaluation ===")
    val_scores = model.predict_proba(valid_df)
    y_true = valid_df['target'].values
    
    ap_score = average_precision_score(y_true, val_scores)
    auc_score = roc_auc_score(y_true, val_scores)
    
    print(f"Validation AP: {ap_score:.4f}")
    print(f"Validation AUC: {auc_score:.4f}")
    
    # Generate test submission
    try:
        print(f"\n=== Generating Kaggle submission ===")
        test_df = pd.read_csv('Data/loans_test.csv')
        print(f"Test samples: {len(test_df)}")
        
        test_scores = model.predict_proba(test_df)
        
        # Create submission file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        
        submission_df = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })
        
        filename = f"FINAL_ENSEMBLE_AP{ap_score:.4f}_AUC{auc_score:.4f}_{timestamp}.csv"
        submission_df.to_csv(filename, index=False)
        
        print(f"SUCCESS! Submission saved: {filename}")
        print(f"Performance Summary:")
        print(f"   - Average Precision: {ap_score:.4f}")
        print(f"   - AUC-ROC: {auc_score:.4f}")
        print(f"   - Ready for Kaggle!")
        
        return model, filename, ap_score, auc_score
        
    except Exception as e:
        print(f"Error creating submission: {e}")
        return model, None, ap_score, auc_score

if __name__ == "__main__":
    model, submission_file, ap, auc = main()
    
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Final Scores: AP={ap:.4f}, AUC={auc:.4f}")
    if submission_file:
        print(f"Submission: {submission_file}")
    print(f"Ready for competition submission!")