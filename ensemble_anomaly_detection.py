#!/usr/bin/env python3
"""
Ensemble Anomaly Detection for Loan Default Prediction
Multi-stage approach combining different anomaly detection methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LoanAnomalyEnsemble:
    """
    Multi-stage ensemble for loan anomaly detection
    Combines multiple unsupervised methods for robust predictions
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.feature_importances = {}
        
    def preprocess_data(self, df, is_training=True):
        """
        Comprehensive data preprocessing pipeline
        """
        print("Preprocessing data...")
        
        # Make copy to avoid modifying original
        data = df.copy()
        
        # Separate static and temporal features
        static_features = []
        temporal_features = []
        
        for col in data.columns[2:]:  # Skip index and target
            if any(col.startswith(f"{i}_") for i in range(20)):
                temporal_features.append(col)
            else:
                static_features.append(col)
        
        print(f"Static features: {len(static_features)}")
        print(f"Temporal features: {len(temporal_features)}")
        
        # Handle missing value codes
        data['CreditScore'] = data['CreditScore'].replace(9999, np.nan)
        data['OriginalDTI'] = data['OriginalDTI'].replace(999, np.nan)
        
        # Fill missing values
        for col in static_features:
            if data[col].dtype in ['object', 'category']:
                data[col] = data[col].fillna('Unknown')
            else:
                data[col] = data[col].fillna(data[col].median())
        
        # Fill temporal missing values
        for col in temporal_features:
            data[col] = data[col].fillna(method='ffill').fillna(0)
        
        return data, static_features, temporal_features
    
    def engineer_features(self, data, static_features, temporal_features):
        """
        Create engineered features from static and temporal data
        """
        print("Engineering features...")
        
        engineered_features = []
        
        # Static feature engineering
        if 'CreditScore' in data.columns and 'OriginalLTV' in data.columns:
            data['credit_ltv_ratio'] = data['CreditScore'] / (data['OriginalLTV'] + 1)
            engineered_features.append('credit_ltv_ratio')
        
        if 'OriginalUPB' in data.columns and 'OriginalInterestRate' in data.columns:
            data['upb_rate_product'] = data['OriginalUPB'] * data['OriginalInterestRate']
            engineered_features.append('upb_rate_product')
        
        # Temporal feature engineering - extract patterns from sequences
        temporal_groups = {}
        for col in temporal_features:
            if '_' in col:
                parts = col.split('_', 1)
                if len(parts) > 1 and parts[0].isdigit():
                    feature_type = parts[1]
                    if feature_type not in temporal_groups:
                        temporal_groups[feature_type] = []
                    temporal_groups[feature_type].append(col)
        
        # Create temporal aggregations
        for feature_type, cols in temporal_groups.items():
            if len(cols) > 1:
                # Trend features
                data[f'{feature_type}_trend'] = data[cols].diff(axis=1).mean(axis=1)
                engineered_features.append(f'{feature_type}_trend')
                
                # Volatility features
                data[f'{feature_type}_volatility'] = data[cols].std(axis=1)
                engineered_features.append(f'{feature_type}_volatility')
                
                # Final vs initial ratio
                if not data[cols[-1]].isna().all() and not data[cols[0]].isna().all():
                    data[f'{feature_type}_final_initial_ratio'] = (data[cols[-1]] + 1) / (data[cols[0]] + 1)
                    engineered_features.append(f'{feature_type}_final_initial_ratio')
        
        print(f"Created {len(engineered_features)} engineered features")
        return data, engineered_features
    
    def prepare_features(self, data, static_features, temporal_features, engineered_features, is_training=True):
        """
        Prepare final feature matrix for modeling
        """
        print("Preparing feature matrix...")
        
        # Select numerical features only for anomaly detection
        numerical_static = []
        for col in static_features:
            if data[col].dtype in ['int64', 'float64'] and col not in ['index', 'target']:
                numerical_static.append(col)
        
        # Combine all numerical features
        all_features = numerical_static + temporal_features + engineered_features
        
        # Remove features with too many missing values or constant values
        final_features = []
        for col in all_features:
            if col in data.columns:
                missing_pct = data[col].isna().sum() / len(data)
                if missing_pct < 0.5 and data[col].nunique() > 1:
                    final_features.append(col)
        
        print(f"Selected {len(final_features)} features for modeling")
        
        X = data[final_features].fillna(0)
        
        # Scale features
        if is_training:
            self.scalers['robust'] = RobustScaler()
            self.scalers['standard'] = StandardScaler()
            
            X_robust = self.scalers['robust'].fit_transform(X)
            X_standard = self.scalers['standard'].fit_transform(X)
        else:
            X_robust = self.scalers['robust'].transform(X)
            X_standard = self.scalers['standard'].transform(X)
        
        return X, X_robust, X_standard, final_features
    
    def train_individual_models(self, X, X_robust, X_standard):
        """
        Train individual anomaly detection models
        """
        print("Training individual models...")
        
        # Model 1: Isolation Forest on robust-scaled data
        print("1. Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_robust)
        
        # Model 2: One-Class SVM on standard-scaled data
        print("2. Training One-Class SVM...")
        self.models['one_class_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        self.models['one_class_svm'].fit(X_standard)
        
        # Model 3: PCA-based anomaly detection
        print("3. Training PCA-based detector...")
        self.models['pca'] = PCA(n_components=0.95, random_state=self.random_state)
        X_pca = self.models['pca'].fit_transform(X_standard)
        
        # Reconstruction error approach
        X_reconstructed = self.models['pca'].inverse_transform(X_pca)
        self.pca_threshold = np.percentile(
            np.mean((X_standard - X_reconstructed) ** 2, axis=1), 90
        )
        
        # Model 4: Statistical outlier detection (Mahalanobis distance)
        print("4. Training statistical outlier detector...")
        self.mean_ = np.mean(X_standard, axis=0)
        self.cov_ = np.cov(X_standard.T)
        self.cov_inv = np.linalg.pinv(self.cov_)
        
        print("Individual models trained successfully!")
    
    def predict_individual_scores(self, X, X_robust, X_standard):
        """
        Get anomaly scores from all individual models
        """
        scores = {}
        
        # Isolation Forest (invert so higher = more anomalous)
        if_scores = self.models['isolation_forest'].score_samples(X_robust)
        scores['isolation_forest'] = -if_scores
        
        # One-Class SVM (distance from boundary, invert)
        svm_scores = self.models['one_class_svm'].score_samples(X_standard)
        scores['one_class_svm'] = -svm_scores
        
        # PCA reconstruction error
        X_pca = self.models['pca'].transform(X_standard)
        X_reconstructed = self.models['pca'].inverse_transform(X_pca)
        pca_scores = np.mean((X_standard - X_reconstructed) ** 2, axis=1)
        scores['pca'] = pca_scores
        
        # Mahalanobis distance
        diff = X_standard - self.mean_
        mahal_scores = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))
        scores['mahalanobis'] = mahal_scores
        
        return scores
    
    def combine_scores(self, individual_scores, method='weighted_average'):
        """
        Combine individual anomaly scores into final ensemble score
        """
        if method == 'weighted_average':
            # Weights based on typical performance of different methods
            weights = {
                'isolation_forest': 0.3,
                'one_class_svm': 0.25,
                'pca': 0.25,
                'mahalanobis': 0.2
            }
            
            # Normalize each score to [0, 1]
            normalized_scores = {}
            for model_name, scores in individual_scores.items():
                min_score, max_score = np.min(scores), np.max(scores)
                if max_score > min_score:
                    normalized_scores[model_name] = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores[model_name] = np.zeros_like(scores)
            
            # Weighted combination
            ensemble_score = np.zeros(len(list(individual_scores.values())[0]))
            for model_name, weight in weights.items():
                if model_name in normalized_scores:
                    ensemble_score += weight * normalized_scores[model_name]
                    
        elif method == 'rank_average':
            # Rank-based combination
            ranks = {}
            for model_name, scores in individual_scores.items():
                ranks[model_name] = len(scores) - np.argsort(np.argsort(scores))
            
            ensemble_score = np.mean(list(ranks.values()), axis=0)
            
        return ensemble_score
    
    def fit(self, train_df):
        """
        Fit the ensemble model
        """
        print("=== Training Ensemble Anomaly Detection ===")
        
        # Preprocess data
        data, static_features, temporal_features = self.preprocess_data(train_df, is_training=True)
        
        # Engineer features
        data, engineered_features = self.engineer_features(data, static_features, temporal_features)
        
        # Prepare features
        X, X_robust, X_standard, final_features = self.prepare_features(
            data, static_features, temporal_features, engineered_features, is_training=True
        )
        
        self.final_features = final_features
        
        # Train individual models
        self.train_individual_models(X, X_robust, X_standard)
        
        print("=== Ensemble Training Complete ===")
        
        return self
    
    def predict_proba(self, test_df):
        """
        Predict anomaly probabilities
        """
        print("=== Predicting with Ensemble ===")
        
        # Preprocess data
        data, static_features, temporal_features = self.preprocess_data(test_df, is_training=False)
        
        # Engineer features
        data, engineered_features = self.engineer_features(data, static_features, temporal_features)
        
        # Prepare features (ensure same features as training)
        data_features = data[self.final_features].fillna(0)
        
        X_robust = self.scalers['robust'].transform(data_features)
        X_standard = self.scalers['standard'].transform(data_features)
        
        # Get individual scores
        individual_scores = self.predict_individual_scores(data_features.values, X_robust, X_standard)
        
        # Combine scores
        ensemble_score = self.combine_scores(individual_scores, method='weighted_average')
        
        print("=== Prediction Complete ===")
        
        return ensemble_score

def evaluate_model(model, valid_df):
    """
    Evaluate model on validation set
    """
    print("\n=== Model Evaluation ===")
    
    # Get predictions
    anomaly_scores = model.predict_proba(valid_df)
    y_true = valid_df['target'].values
    
    # Calculate metrics
    ap_score = average_precision_score(y_true, anomaly_scores)
    auc_score = roc_auc_score(y_true, anomaly_scores)
    
    print(f"Average Precision (AP): {ap_score:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    
    # Classification at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for thresh in thresholds:
        predictions = (anomaly_scores > np.percentile(anomaly_scores, (1-thresh)*100)).astype(int)
        precision = np.sum((predictions == 1) & (y_true == 1)) / max(np.sum(predictions == 1), 1)
        recall = np.sum((predictions == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        print(f"Threshold {thresh:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return ap_score, auc_score

def main():
    """
    Main training and evaluation pipeline
    """
    print("Loading data...")
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Validation data: {valid_df.shape}")
    
    # Initialize and train ensemble
    ensemble = LoanAnomalyEnsemble(random_state=42)
    ensemble.fit(train_df)
    
    # Evaluate on validation set
    ap_score, auc_score = evaluate_model(ensemble, valid_df)
    
    # Generate submission for test set if available
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        print(f"\nGenerating submission for test set: {test_df.shape}")
        
        test_scores = ensemble.predict_proba(test_df)
        
        # Create submission file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Handle different index column names
        id_col = 'index' if 'index' in test_df.columns else 'Id'
        submission_df = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })
        
        submission_filename = f"ensemble_submission_{timestamp}.csv"
        submission_df.to_csv(submission_filename, index=False)
        print(f"Submission saved as: {submission_filename}")
        
    except FileNotFoundError:
        print("Test file not found, skipping submission generation")
    
    return ensemble, ap_score, auc_score

if __name__ == "__main__":
    model, ap, auc = main()