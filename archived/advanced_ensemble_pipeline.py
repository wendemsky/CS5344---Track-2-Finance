#!/usr/bin/env python3
"""
Advanced Multi-Stage Ensemble for Loan Anomaly Detection
Combines unsupervised anomaly detection with supervised boosting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedLoanEnsemble:
    """
    Multi-stage ensemble combining:
    1. Unsupervised anomaly detection (Stage 1)  
    2. Feature augmentation with anomaly scores (Stage 2)
    3. Supervised boosting with pseudo-labels (Stage 3)
    4. Final ensemble combination (Stage 4)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.models = {}
        self.stage1_models = {}
        self.stage3_models = {}
        
    def preprocess_data(self, df, is_training=True):
        """Enhanced preprocessing with better categorical handling"""
        print("Advanced preprocessing...")
        
        data = df.copy()
        
        # Separate feature types
        static_features = []
        temporal_features = []
        categorical_features = []
        
        for col in data.columns[2:]:
            if any(col.startswith(f"{i}_") for i in range(20)):
                temporal_features.append(col)
            else:
                static_features.append(col)
                if data[col].dtype == 'object':
                    categorical_features.append(col)
        
        # Handle special missing codes
        data['CreditScore'] = data['CreditScore'].replace(9999, np.nan)
        data['OriginalDTI'] = data['OriginalDTI'].replace(999, np.nan)
        data['OriginalLTV'] = data['OriginalLTV'].replace(999, np.nan)
        data['EstimatedLTV_avg'] = data[[col for col in data.columns if 'EstimatedLTV' in col]].replace(999, np.nan).mean(axis=1)
        
        # Advanced categorical encoding
        for col in categorical_features:
            if is_training:
                self.encoders[col] = LabelEncoder()
                # Handle unknown values
                unique_vals = list(data[col].unique())
                if np.nan in unique_vals:
                    unique_vals.remove(np.nan)
                unique_vals.append('UNKNOWN')
                self.encoders[col].fit(unique_vals)
            
            # Encode with unknown handling
            data[col] = data[col].fillna('UNKNOWN')
            data[col] = data[col].map(lambda x: x if x in self.encoders[col].classes_ else 'UNKNOWN')
            data[col] = self.encoders[col].transform(data[col])
        
        # Advanced missing value imputation
        for col in static_features:
            if col not in categorical_features:
                if data[col].dtype in ['int64', 'float64']:
                    if is_training:
                        self.scalers[f'{col}_median'] = data[col].median()
                    data[col] = data[col].fillna(self.scalers[f'{col}_median'])
        
        # Temporal feature imputation
        for col in temporal_features:
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data, static_features, temporal_features, categorical_features
    
    def advanced_feature_engineering(self, data, static_features, temporal_features):
        """Advanced feature engineering with domain knowledge"""
        print("Advanced feature engineering...")
        
        engineered_features = []
        
        # Financial ratios and interactions
        if 'CreditScore' in data.columns and 'OriginalLTV' in data.columns:
            data['credit_ltv_interaction'] = data['CreditScore'] * (100 - data['OriginalLTV'])
            engineered_features.append('credit_ltv_interaction')
        
        if 'OriginalUPB' in data.columns and 'OriginalInterestRate' in data.columns:
            data['monthly_payment_est'] = (data['OriginalUPB'] * data['OriginalInterestRate'] / 100) / 12
            engineered_features.append('monthly_payment_est')
        
        if 'OriginalDTI' in data.columns and 'CreditScore' in data.columns:
            data['risk_score'] = data['OriginalDTI'] / (data['CreditScore'] / 100)
            engineered_features.append('risk_score')
        
        # Temporal feature engineering
        temporal_groups = {}
        for col in temporal_features:
            if '_' in col:
                parts = col.split('_', 1)
                if len(parts) > 1 and parts[0].isdigit():
                    feature_type = parts[1]
                    if feature_type not in temporal_groups:
                        temporal_groups[feature_type] = []
                    temporal_groups[feature_type].append(col)
        
        # Advanced temporal aggregations
        for feature_type, cols in temporal_groups.items():
            if len(cols) > 3:  # Need sufficient data points
                feature_data = data[cols]
                
                # Statistical moments
                data[f'{feature_type}_mean'] = feature_data.mean(axis=1)
                data[f'{feature_type}_std'] = feature_data.std(axis=1)
                data[f'{feature_type}_skew'] = feature_data.skew(axis=1)
                data[f'{feature_type}_kurt'] = feature_data.kurtosis(axis=1)
                
                # Trend analysis
                data[f'{feature_type}_linear_trend'] = feature_data.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if not x.isna().all() else 0, axis=1
                )
                
                # Volatility measures
                data[f'{feature_type}_coeff_var'] = data[f'{feature_type}_std'] / (data[f'{feature_type}_mean'].abs() + 1e-10)
                
                # Change patterns
                data[f'{feature_type}_max_change'] = feature_data.diff(axis=1).abs().max(axis=1)
                data[f'{feature_type}_total_change'] = abs(feature_data.iloc[:, -1] - feature_data.iloc[:, 0])
                
                # Add to engineered features
                new_features = [f'{feature_type}_mean', f'{feature_type}_std', f'{feature_type}_skew', 
                               f'{feature_type}_kurt', f'{feature_type}_linear_trend', f'{feature_type}_coeff_var',
                               f'{feature_type}_max_change', f'{feature_type}_total_change']
                engineered_features.extend(new_features)
        
        # Replace infinite values
        for col in engineered_features:
            if col in data.columns:
                data[col] = data[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Created {len(engineered_features)} engineered features")
        return data, engineered_features
    
    def stage1_unsupervised_training(self, X_scaled):
        """Stage 1: Train multiple unsupervised models"""
        print("Stage 1: Training unsupervised anomaly detectors...")
        
        # Model 1: Isolation Forest
        self.stage1_models['isolation_forest'] = IsolationForest(
            n_estimators=300,
            contamination=0.12,
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.stage1_models['isolation_forest'].fit(X_scaled)
        
        # Model 2: One-Class SVM
        self.stage1_models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.12
        )
        self.stage1_models['one_class_svm'].fit(X_scaled)
        
        # Model 3: PCA reconstruction
        self.stage1_models['pca'] = PCA(n_components=0.95, random_state=self.random_state)
        X_pca = self.stage1_models['pca'].fit_transform(X_scaled)
        X_reconstructed = self.stage1_models['pca'].inverse_transform(X_pca)
        reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        self.pca_threshold_90 = np.percentile(reconstruction_errors, 90)
        self.pca_threshold_95 = np.percentile(reconstruction_errors, 95)
        
        print("Stage 1 complete!")
        
    def stage1_get_anomaly_scores(self, X_scaled):
        """Get anomaly scores from Stage 1 models"""
        scores = {}
        
        # Isolation Forest scores
        if_scores = self.stage1_models['isolation_forest'].score_samples(X_scaled)
        scores['if_anomaly_score'] = -if_scores  # Invert so higher = more anomalous
        
        # One-Class SVM scores  
        svm_scores = self.stage1_models['one_class_svm'].score_samples(X_scaled)
        scores['svm_anomaly_score'] = -svm_scores
        
        # PCA reconstruction error
        X_pca = self.stage1_models['pca'].transform(X_scaled)
        X_reconstructed = self.stage1_models['pca'].inverse_transform(X_pca)
        pca_scores = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        scores['pca_anomaly_score'] = pca_scores
        
        # Ensemble of Stage 1
        normalized_scores = {}
        for name, score_array in scores.items():
            min_s, max_s = np.min(score_array), np.max(score_array)
            if max_s > min_s:
                normalized_scores[name] = (score_array - min_s) / (max_s - min_s)
            else:
                normalized_scores[name] = np.zeros_like(score_array)
        
        # Weighted combination
        weights = {'if_anomaly_score': 0.4, 'svm_anomaly_score': 0.3, 'pca_anomaly_score': 0.3}
        ensemble_score = np.zeros(len(list(scores.values())[0]))
        for name, weight in weights.items():
            ensemble_score += weight * normalized_scores[name]
        
        scores['stage1_ensemble'] = ensemble_score
        
        return scores
        
    def stage3_supervised_training(self, X_augmented, validation_df):
        """Stage 3: Train supervised models with pseudo-labels"""
        print("Stage 3: Training supervised models with validation data...")
        
        # Use validation set for supervised training
        val_processed, val_static, val_temporal, val_categorical = self.preprocess_data(validation_df, is_training=False)
        val_processed, val_engineered = self.advanced_feature_engineering(val_processed, val_static, val_temporal)
        
        # Get same features as training
        val_features = val_processed[self.final_features].fillna(0)
        val_X_scaled = self.scalers['main'].transform(val_features)
        
        # Get Stage 1 scores for validation
        val_stage1_scores = self.stage1_get_anomaly_scores(val_X_scaled)
        
        # Create augmented validation features
        val_X_augmented = np.column_stack([
            val_X_scaled,
            val_stage1_scores['if_anomaly_score'],
            val_stage1_scores['svm_anomaly_score'], 
            val_stage1_scores['pca_anomaly_score'],
            val_stage1_scores['stage1_ensemble']
        ])
        
        y_val = validation_df['target'].values
        
        # Train XGBoost
        print("Training XGBoost...")
        self.stage3_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            scale_pos_weight=len(y_val[y_val==0]) / len(y_val[y_val==1]),  # Handle imbalance
            eval_metric='auc'
        )
        self.stage3_models['xgboost'].fit(val_X_augmented, y_val)
        
        # Train LightGBM  
        print("Training LightGBM...")
        self.stage3_models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            class_weight='balanced',
            verbose=-1
        )
        self.stage3_models['lightgbm'].fit(val_X_augmented, y_val)
        
        print("Stage 3 complete!")
        
    def fit(self, train_df, validation_df):
        """Fit the multi-stage ensemble"""
        print("=== Multi-Stage Ensemble Training ===")
        
        # Preprocess training data
        data, static_features, temporal_features, categorical_features = self.preprocess_data(train_df, is_training=True)
        
        # Engineer features
        data, engineered_features = self.advanced_feature_engineering(data, static_features, temporal_features)
        
        # Select numerical features
        numerical_features = []
        for col in static_features + engineered_features:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                if data[col].nunique() > 1:  # Remove constants
                    numerical_features.append(col)
        
        # Add some temporal features directly
        key_temporal = [col for col in temporal_features if 'CurrentActualUPB' in col or 'EstimatedLTV' in col][:28]
        all_features = numerical_features + key_temporal
        
        print(f"Using {len(all_features)} features for modeling")
        self.final_features = all_features
        
        # Prepare feature matrix
        X = data[all_features].fillna(0)
        
        # Scale features
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        # Stage 1: Train unsupervised models
        self.stage1_unsupervised_training(X_scaled)
        
        # Get Stage 1 anomaly scores for training data
        stage1_scores = self.stage1_get_anomaly_scores(X_scaled)
        
        # Stage 2: Create augmented features
        X_augmented = np.column_stack([
            X_scaled,
            stage1_scores['if_anomaly_score'],
            stage1_scores['svm_anomaly_score'],
            stage1_scores['pca_anomaly_score'], 
            stage1_scores['stage1_ensemble']
        ])
        
        # Stage 3: Train supervised models using validation set
        self.stage3_supervised_training(X_augmented, validation_df)
        
        print("=== Multi-Stage Training Complete ===")
        return self
        
    def predict_proba(self, test_df):
        """Predict anomaly probabilities using all stages"""
        print("=== Multi-Stage Prediction ===")
        
        # Preprocess
        data, static_features, temporal_features, categorical_features = self.preprocess_data(test_df, is_training=False)
        data, engineered_features = self.advanced_feature_engineering(data, static_features, temporal_features)
        
        # Prepare features
        X = data[self.final_features].fillna(0)
        X_scaled = self.scalers['main'].transform(X)
        
        # Stage 1: Get anomaly scores
        stage1_scores = self.stage1_get_anomaly_scores(X_scaled)
        
        # Stage 2: Create augmented features
        X_augmented = np.column_stack([
            X_scaled,
            stage1_scores['if_anomaly_score'],
            stage1_scores['svm_anomaly_score'],
            stage1_scores['pca_anomaly_score'],
            stage1_scores['stage1_ensemble']
        ])
        
        # Stage 3: Get supervised predictions
        xgb_proba = self.stage3_models['xgboost'].predict_proba(X_augmented)[:, 1]
        lgb_proba = self.stage3_models['lightgbm'].predict_proba(X_augmented)[:, 1]
        
        # Stage 4: Final ensemble
        final_score = (
            0.3 * stage1_scores['stage1_ensemble'] +
            0.4 * xgb_proba +
            0.3 * lgb_proba
        )
        
        print("=== Prediction Complete ===")
        return final_score

def evaluate_and_submit(model, valid_df, test_df=None):
    """Evaluate model and create submission"""
    print("\n=== Model Evaluation ===")
    
    # Predict on validation
    anomaly_scores = model.predict_proba(valid_df)
    y_true = valid_df['target'].values
    
    # Calculate metrics
    ap_score = average_precision_score(y_true, anomaly_scores)
    auc_score = roc_auc_score(y_true, anomaly_scores)
    
    print(f"Average Precision: {ap_score:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    print(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.4f}")
    
    # Create submission if test data provided
    if test_df is not None:
        print("Generating test submission...")
        test_scores = model.predict_proba(test_df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_df = pd.DataFrame({
            'index': test_df['index'],
            'anomaly_score': test_scores
        })
        
        submission_filename = f"advanced_ensemble_submission_{timestamp}.csv"
        submission_df.to_csv(submission_filename, index=False)
        print(f"Submission saved: {submission_filename}")
        print(f"AP: {ap_score:.4f}, AUC: {auc_score:.4f}")
        
        return submission_filename, ap_score, auc_score
    
    return None, ap_score, auc_score

def main():
    """Main pipeline"""
    print("Loading data...")
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')
    
    print(f"Training: {train_df.shape}, Validation: {valid_df.shape}")
    
    # Train advanced ensemble
    ensemble = AdvancedLoanEnsemble(random_state=42)
    ensemble.fit(train_df, valid_df)
    
    # Evaluate and create submission
    try:
        test_df = pd.read_csv('Data/loans_test.csv') 
        submission_file, ap, auc = evaluate_and_submit(ensemble, valid_df, test_df)
        return ensemble, submission_file, ap, auc
    except FileNotFoundError:
        print("Test file not found")
        _, ap, auc = evaluate_and_submit(ensemble, valid_df)
        return ensemble, None, ap, auc

if __name__ == "__main__":
    model, submission, ap_score, auc_score = main()
    print(f"\nFinal Performance - AP: {ap_score:.4f}, AUC: {auc_score:.4f}")