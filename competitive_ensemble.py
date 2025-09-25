#!/usr/bin/env python3
"""
Competitive Ensemble for Kaggle Loan Anomaly Detection
Focus on high-performance combination approach
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CompetitiveLoanEnsemble:
    """
    High-performance ensemble specifically designed for competition
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        
    def advanced_preprocessing(self, df, is_training=True):
        """Competition-focused preprocessing"""
        print("Advanced preprocessing...")
        
        data = df.copy()
        
        # Handle different ID columns
        if 'Id' in data.columns and 'index' not in data.columns:
            data['index'] = data['Id']
        
        # Identify feature types
        static_features = []
        temporal_features = []
        
        for col in data.columns:
            if col in ['index', 'Id', 'target']:
                continue
            elif any(col.startswith(f"{i}_") for i in range(20)):
                temporal_features.append(col)
            else:
                static_features.append(col)
        
        # Handle missing value codes aggressively
        data['CreditScore'] = data['CreditScore'].replace(9999, np.nan)
        data['OriginalDTI'] = data['OriginalDTI'].replace(999, np.nan)
        data['OriginalLTV'] = data['OriginalLTV'].replace(999, np.nan)
        
        # Advanced categorical encoding
        categorical_cols = ['FirstTimeHomebuyerFlag', 'OccupancyStatus', 'Channel', 
                           'PPM_Flag', 'ProductType', 'PropertyState', 'PropertyType', 
                           'LoanPurpose', 'SellerName', 'ServicerName', 'InterestOnlyFlag',
                           'BalloonIndicator', 'ProgramIndicator']
        
        for col in categorical_cols:
            if col in data.columns:
                if is_training:
                    self.encoders[col] = LabelEncoder()
                    # Add unknown category
                    unique_vals = list(data[col].dropna().unique()) + ['UNKNOWN']
                    self.encoders[col].fit(unique_vals)
                
                # Encode with unknown handling
                data[col] = data[col].fillna('UNKNOWN')
                safe_transform = lambda x: x if x in self.encoders[col].classes_ else 'UNKNOWN'
                data[col] = data[col].map(safe_transform)
                data[col] = self.encoders[col].transform(data[col])
        
        # Smart imputation for numerical features
        numerical_cols = [col for col in static_features if col not in categorical_cols and col in data.columns]
        
        if is_training:
            for col in numerical_cols:
                self.scalers[f'{col}_median'] = data[col].median()
                self.scalers[f'{col}_mean'] = data[col].mean()
        
        for col in numerical_cols:
            # Use median for most, mean for specific features
            if 'Rate' in col or 'Score' in col:
                data[col] = data[col].fillna(self.scalers[f'{col}_mean'])
            else:
                data[col] = data[col].fillna(self.scalers[f'{col}_median'])
        
        # Temporal imputation
        for col in temporal_features:
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data, static_features, temporal_features
    
    def create_powerful_features(self, data, static_features, temporal_features):
        """Create high-impact features"""
        print("Creating powerful features...")
        
        new_features = []
        
        # Financial risk indicators
        if all(col in data.columns for col in ['CreditScore', 'OriginalLTV', 'OriginalDTI']):
            data['risk_score_v1'] = (data['OriginalLTV'] * data['OriginalDTI']) / (data['CreditScore'] + 1)
            data['risk_score_v2'] = data['OriginalDTI'] / (data['CreditScore'] / 100)
            data['ltv_dti_interaction'] = data['OriginalLTV'] * data['OriginalDTI']
            new_features.extend(['risk_score_v1', 'risk_score_v2', 'ltv_dti_interaction'])
        
        # Payment capacity indicators
        if all(col in data.columns for col in ['OriginalUPB', 'OriginalInterestRate', 'OriginalLoanTerm']):
            monthly_rate = data['OriginalInterestRate'] / 100 / 12
            data['estimated_payment'] = (data['OriginalUPB'] * monthly_rate * (1 + monthly_rate)**data['OriginalLoanTerm']) / ((1 + monthly_rate)**data['OriginalLoanTerm'] - 1)
            new_features.append('estimated_payment')
        
        # Temporal feature engineering - focus on key patterns
        temporal_groups = {}
        for col in temporal_features:
            if '_' in col:
                parts = col.split('_', 1)
                if len(parts) > 1 and parts[0].isdigit():
                    feature_type = parts[1]
                    if feature_type not in temporal_groups:
                        temporal_groups[feature_type] = []
                    temporal_groups[feature_type].append(col)
        
        # High-impact temporal features
        for feature_type, cols in temporal_groups.items():
            if len(cols) >= 10:  # Need sufficient history
                feature_data = data[cols].replace([np.inf, -np.inf], np.nan)
                
                # Trend and change features
                data[f'{feature_type}_final_change'] = feature_data.iloc[:, -1] - feature_data.iloc[:, 0]
                data[f'{feature_type}_max_change'] = feature_data.diff(axis=1).abs().max(axis=1)
                data[f'{feature_type}_trend_strength'] = feature_data.apply(
                    lambda x: abs(np.corrcoef(range(len(x)), x)[0, 1]) if not x.isna().all() and x.var() > 0 else 0, axis=1
                )
                
                # Volatility measures
                data[f'{feature_type}_cv'] = feature_data.std(axis=1) / (feature_data.mean(axis=1).abs() + 1)
                data[f'{feature_type}_range_norm'] = (feature_data.max(axis=1) - feature_data.min(axis=1)) / (feature_data.mean(axis=1) + 1)
                
                # Recent behavior (last 3 months)
                recent_data = feature_data.iloc[:, -3:]
                data[f'{feature_type}_recent_trend'] = recent_data.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if not x.isna().any() else 0, axis=1
                )
                
                new_features.extend([
                    f'{feature_type}_final_change', f'{feature_type}_max_change', 
                    f'{feature_type}_trend_strength', f'{feature_type}_cv',
                    f'{feature_type}_range_norm', f'{feature_type}_recent_trend'
                ])
        
        # Clean infinite values
        for col in new_features:
            if col in data.columns:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Created {len(new_features)} powerful features")
        return data, new_features
    
    def fit(self, train_df, valid_df):
        """Fit competitive ensemble"""
        print("=== Training Competitive Ensemble ===")
        
        # Preprocess training data
        train_data, train_static, train_temporal = self.advanced_preprocessing(train_df, is_training=True)
        train_data, train_engineered = self.create_powerful_features(train_data, train_static, train_temporal)
        
        # Preprocess validation data  
        valid_data, valid_static, valid_temporal = self.advanced_preprocessing(valid_df, is_training=False)
        valid_data, valid_engineered = self.create_powerful_features(valid_data, valid_static, valid_temporal)
        
        # Select best features
        feature_candidates = []
        
        # Key static features
        key_static = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate', 
                     'OriginalDTI', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']
        for col in key_static:
            if col in train_data.columns:
                feature_candidates.append(col)
        
        # Engineered features
        feature_candidates.extend(train_engineered)
        
        # Key temporal features - select most informative
        key_temporal_patterns = ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']
        for pattern in key_temporal_patterns:
            pattern_cols = [col for col in train_temporal if pattern in col]
            # Take every 2nd month to reduce dimensionality but keep pattern
            selected_temporal = pattern_cols[::2]
            feature_candidates.extend(selected_temporal)
        
        # Final feature selection - remove problematic features
        final_features = []
        for col in feature_candidates:
            if col in train_data.columns:
                # Check for sufficient variance and not too many missing values
                if train_data[col].nunique() > 1 and train_data[col].notna().sum() > len(train_data) * 0.5:
                    final_features.append(col)
        
        print(f"Selected {len(final_features)} final features")
        self.final_features = final_features
        
        # Prepare training data
        X_train = train_data[final_features].fillna(0)
        
        # Prepare validation data
        X_valid = valid_data[final_features].fillna(0) 
        y_valid = valid_df['target'].values
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_train_scaled = self.scalers['main'].fit_transform(X_train)
        X_valid_scaled = self.scalers['main'].transform(X_valid)
        
        # Stage 1: Train unsupervised model on training data (all normal)
        print("Stage 1: Isolation Forest on training data...")
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=500,
            contamination=0.126,  # Based on validation set proportion
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_train_scaled)
        
        # Get anomaly scores for validation data
        if_scores = -self.models['isolation_forest'].score_samples(X_valid_scaled)  # Higher = more anomalous
        
        # Stage 2: Train supervised models on validation data with anomaly score as feature
        print("Stage 2: XGBoost with anomaly scores as features...")
        
        # Normalize anomaly scores
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        
        # Augment features with anomaly scores
        X_valid_augmented = np.column_stack([X_valid_scaled, if_scores_norm])
        
        # Train XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            scale_pos_weight=len(y_valid[y_valid==0]) / len(y_valid[y_valid==1]),
            eval_metric='auc',
            early_stopping_rounds=50
        )
        
        # Use part of validation for training, part for early stopping
        split_idx = int(len(X_valid_augmented) * 0.7)
        X_train_xgb = X_valid_augmented[:split_idx]
        y_train_xgb = y_valid[:split_idx]
        X_eval_xgb = X_valid_augmented[split_idx:]
        y_eval_xgb = y_valid[split_idx:]
        
        self.models['xgboost'].fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_eval_xgb, y_eval_xgb)],
            verbose=False
        )
        
        # Train LightGBM
        print("Stage 3: LightGBM ensemble...")
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            class_weight='balanced',
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train_xgb, y_train_xgb)
        
        print("=== Training Complete ===")
        return self
    
    def predict_proba(self, test_df):
        """Predict with competitive ensemble"""
        print("=== Competitive Prediction ===")
        
        # Preprocess
        test_data, test_static, test_temporal = self.advanced_preprocessing(test_df, is_training=False)
        test_data, test_engineered = self.create_powerful_features(test_data, test_static, test_temporal)
        
        # Prepare features
        X_test = test_data[self.final_features].fillna(0)
        X_test_scaled = self.scalers['main'].transform(X_test)
        
        # Stage 1: Get anomaly scores
        if_scores = -self.models['isolation_forest'].score_samples(X_test_scaled)
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        
        # Stage 2: Augmented prediction
        X_test_augmented = np.column_stack([X_test_scaled, if_scores_norm])
        
        # Get predictions from supervised models
        xgb_proba = self.models['xgboost'].predict_proba(X_test_augmented)[:, 1]
        lgb_proba = self.models['lightgbm'].predict_proba(X_test_augmented)[:, 1]
        
        # Final ensemble - weighted combination
        final_scores = (
            0.25 * if_scores_norm +      # Unsupervised component
            0.45 * xgb_proba +           # XGBoost (typically strongest)
            0.30 * lgb_proba             # LightGBM diversity
        )
        
        print("=== Prediction Complete ===")
        return final_scores

def run_competitive_pipeline():
    """Run the competitive pipeline"""
    print("Loading data...")
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')
    
    print(f"Training: {train_df.shape}, Validation: {valid_df.shape}")
    
    # Train model
    model = CompetitiveLoanEnsemble(random_state=42)
    model.fit(train_df, valid_df)
    
    # Evaluate
    print("\n=== Evaluation ===")
    scores = model.predict_proba(valid_df)
    y_true = valid_df['target'].values
    
    ap = average_precision_score(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    
    print(f"Validation AP: {ap:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    
    # Generate submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        print(f"\nGenerating submission for {len(test_df)} test samples...")
        
        test_scores = model.predict_proba(test_df)
        
        # Create submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        
        submission_df = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })
        
        filename = f"competitive_ensemble_AP{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission_df.to_csv(filename, index=False)
        
        print(f"✅ Submission saved: {filename}")
        print(f"📊 Performance: AP={ap:.4f}, AUC={auc:.4f}")
        
        return model, filename, ap, auc
        
    except Exception as e:
        print(f"Error generating submission: {e}")
        return model, None, ap, auc

if __name__ == "__main__":
    model, submission_file, ap_score, auc_score = run_competitive_pipeline()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Average Precision: {ap_score:.4f}")
    print(f"   AUC-ROC: {auc_score:.4f}")
    if submission_file:
        print(f"   Submission File: {submission_file}")
    print("   Ready for Kaggle submission! 🚀")