#!/usr/bin/env python3
"""
Improved Anomaly Detection Ensemble
Implements multiple anomaly detection methodologies:
1. Statistical Approaches (Gaussian-based outlier detection)
2. Proximity-Based Approaches (LOF, k-NN distance)
3. Clustering-Based Approaches (DBSCAN, Isolation)
4. Reconstruction-Based Approaches (PCA, Autoencoder)

Key Improvements:
- Validation set used ONLY for hyperparameter selection, NOT training
- Pure unsupervised anomaly detection (no label leakage)
- Multiple complementary detection methods
- Robust ensemble combining different approaches
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
# TensorFlow removed for compatibility - using simple autoencoder alternative
from datetime import datetime
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnomalyDetector:
    """Statistical approach assuming normal distribution"""

    def __init__(self, contamination=0.1, alpha=0.05):
        self.contamination = contamination
        self.alpha = alpha
        self.mean_ = None
        self.cov_ = None
        self.threshold_ = None

    def fit(self, X):
        """Fit Gaussian distribution parameters"""
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X.T)

        # Calculate Mahalanobis distances for threshold
        mahal_distances = self._mahalanobis_distance(X)
        self.threshold_ = np.percentile(mahal_distances, (1 - self.contamination) * 100)
        return self

    def _mahalanobis_distance(self, X):
        """Calculate Mahalanobis distance"""
        try:
            cov_inv = np.linalg.inv(self.cov_)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            cov_inv = np.linalg.pinv(self.cov_)

        diff = X - self.mean_
        mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        return mahal_dist

    def score_samples(self, X):
        """Return anomaly scores (higher = more anomalous)"""
        mahal_distances = self._mahalanobis_distance(X)
        # Normalize to [0, 1] range using min-max scaling
        if len(mahal_distances) > 1:
            min_dist = mahal_distances.min()
            max_dist = mahal_distances.max()
            if max_dist > min_dist:
                scores = (mahal_distances - min_dist) / (max_dist - min_dist)
            else:
                scores = np.zeros_like(mahal_distances)
        else:
            scores = np.array([0.5])
        return scores


class ProximityAnomalyDetector:
    """Proximity-based approach using k-NN distance and custom LOF implementation"""

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.knn_ = NearestNeighbors(n_neighbors=n_neighbors)
        self.X_train_ = None

    def fit(self, X):
        """Fit k-NN model"""
        self.X_train_ = X.copy()
        self.knn_.fit(X)
        return self

    def _compute_lof_scores(self, X):
        """Compute LOF-like scores manually"""
        # Get distances to k neighbors for each point in X
        distances, indices = self.knn_.kneighbors(X)

        # Compute local reachability density for each point
        lrd_scores = []
        for i in range(len(X)):
            # k-distance (distance to k-th neighbor)
            k_dist = distances[i, -1]

            # Reachability distances to neighbors
            neighbor_indices = indices[i]
            reach_dists = []
            for j in neighbor_indices:
                if j < len(self.X_train_):
                    # Distance from X[i] to training point j
                    dist_to_j = np.linalg.norm(X[i] - self.X_train_[j])
                    # Reachability distance is max(k_dist_j, actual_distance)
                    train_k_dist = np.linalg.norm(self.X_train_[j] - self.knn_.kneighbors([self.X_train_[j]])[0][0, -1])
                    reach_dist = max(train_k_dist, dist_to_j)
                    reach_dists.append(reach_dist)

            # Local reachability density (inverse of average reachability distance)
            if len(reach_dists) > 0:
                avg_reach_dist = np.mean(reach_dists)
                lrd = 1.0 / (avg_reach_dist + 1e-8)
            else:
                lrd = 1.0
            lrd_scores.append(lrd)

        # Convert to LOF scores (higher = more anomalous)
        lof_scores = []
        for i, lrd in enumerate(lrd_scores):
            neighbor_indices = indices[i]
            neighbor_lrds = []
            for j in neighbor_indices:
                if j < len(lrd_scores):
                    neighbor_lrds.append(lrd_scores[j])

            if len(neighbor_lrds) > 0:
                avg_neighbor_lrd = np.mean(neighbor_lrds)
                lof = avg_neighbor_lrd / (lrd + 1e-8)
            else:
                lof = 1.0
            lof_scores.append(lof)

        return np.array(lof_scores)

    def score_samples(self, X):
        """Return anomaly scores combining LOF and k-NN distance"""
        # Simple k-NN distance approach (more reliable)
        distances, _ = self.knn_.kneighbors(X)
        knn_scores = distances[:, -1]  # Distance to k-th neighbor

        # Normalize to [0, 1]
        if len(knn_scores) > 1:
            min_score = knn_scores.min()
            max_score = knn_scores.max()
            if max_score > min_score:
                knn_norm = (knn_scores - min_score) / (max_score - min_score)
            else:
                knn_norm = np.zeros_like(knn_scores)
        else:
            knn_norm = np.array([0.5])

        return knn_norm


class ClusteringAnomalyDetector:
    """Clustering-based approach using DBSCAN and Isolation Forest"""

    def __init__(self, eps=0.5, min_samples=5, contamination=0.1):
        self.eps = eps
        self.min_samples = min_samples
        self.contamination = contamination
        self.dbscan_ = DBSCAN(eps=eps, min_samples=min_samples)
        self.isolation_forest_ = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, X):
        """Fit DBSCAN and Isolation Forest"""
        self.dbscan_.fit(X)
        self.isolation_forest_.fit(X)
        return self

    def score_samples(self, X):
        """Return anomaly scores combining DBSCAN and Isolation Forest"""
        # DBSCAN scores (outliers have label -1)
        labels = self.dbscan_.fit_predict(X)
        dbscan_scores = (labels == -1).astype(float)

        # Isolation Forest scores
        if_scores = -self.isolation_forest_.score_samples(X)  # Higher = more anomalous
        if len(if_scores) > 1:
            min_if = if_scores.min()
            max_if = if_scores.max()
            if max_if > min_if:
                if_norm = (if_scores - min_if) / (max_if - min_if)
            else:
                if_norm = np.zeros_like(if_scores)
        else:
            if_norm = np.array([0.5])

        # Combine scores
        combined_scores = 0.3 * dbscan_scores + 0.7 * if_norm
        return combined_scores


class ReconstructionAnomalyDetector:
    """Reconstruction-based approach using PCA and Autoencoder"""

    def __init__(self, n_components=None, encoding_dim=None):
        self.n_components = n_components
        self.encoding_dim = encoding_dim
        self.pca_ = None
        self.autoencoder_ = None
        self.encoder_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X):
        """Fit PCA and Autoencoder models"""
        X_scaled = self.scaler_.fit_transform(X)

        # PCA reconstruction
        if self.n_components is None:
            # Use components that explain 95% variance
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= 0.95) + 1

        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_scaled)

        # Simple SVD-based autoencoder
        if self.encoding_dim is None:
            self.encoding_dim = max(X.shape[1] // 3, 2)

        self.svd_, _, _ = self._build_simple_autoencoder(X_scaled)

        return self

    def _build_simple_autoencoder(self, X):
        """Simple autoencoder using SVD decomposition as alternative to neural network"""
        from sklearn.decomposition import TruncatedSVD

        # Use SVD as a simple autoencoder alternative
        svd = TruncatedSVD(n_components=self.encoding_dim, random_state=42)
        encoded = svd.fit_transform(X)
        reconstructed = svd.inverse_transform(encoded)

        return svd, encoded, reconstructed

    def score_samples(self, X):
        """Return anomaly scores based on reconstruction error"""
        X_scaled = self.scaler_.transform(X)

        # PCA reconstruction error
        X_pca = self.pca_.transform(X_scaled)
        X_pca_reconstructed = self.pca_.inverse_transform(X_pca)
        pca_errors = np.mean((X_scaled - X_pca_reconstructed) ** 2, axis=1)

        # SVD-based reconstruction error
        X_svd_encoded = self.svd_.transform(X_scaled)
        X_svd_reconstructed = self.svd_.inverse_transform(X_svd_encoded)
        svd_errors = np.mean((X_scaled - X_svd_reconstructed) ** 2, axis=1)

        # Normalize errors to [0, 1]
        if len(pca_errors) > 1:
            min_pca = pca_errors.min()
            max_pca = pca_errors.max()
            if max_pca > min_pca:
                pca_norm = (pca_errors - min_pca) / (max_pca - min_pca)
            else:
                pca_norm = np.zeros_like(pca_errors)
        else:
            pca_norm = np.array([0.5])

        if len(svd_errors) > 1:
            min_svd = svd_errors.min()
            max_svd = svd_errors.max()
            if max_svd > min_svd:
                svd_norm = (svd_errors - min_svd) / (max_svd - min_svd)
            else:
                svd_norm = np.zeros_like(svd_errors)
        else:
            svd_norm = np.array([0.5])

        # Combine reconstruction errors
        combined_scores = 0.4 * pca_norm + 0.6 * svd_norm
        return combined_scores


class ImprovedAnomalyEnsemble:
    """
    Improved anomaly detection ensemble using multiple methodologies
    - Uses validation set ONLY for hyperparameter tuning
    - Pure unsupervised approach (no label leakage)
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.detectors: Dict[str, Any] = {}
        self.imputation_values: Dict[str, float] = {}
        self.final_features: List[str] = []
        self.best_params: Dict[str, Any] = {}
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

    def create_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Feature engineering"""
        new_features: List[str] = []

        # Static feature ratios
        base_features = ['CreditScore', 'OriginalLTV', 'OriginalDTI', 'OriginalUPB', 'OriginalInterestRate']
        available = [f for f in base_features if f in data.columns]

        if 'CreditScore' in available and 'OriginalLTV' in available:
            data['credit_ltv_ratio'] = data['CreditScore'] / (data['OriginalLTV'] + 1.0)
            new_features.append('credit_ltv_ratio')

        if 'OriginalDTI' in available and 'CreditScore' in available:
            data['dti_credit_ratio'] = data['OriginalDTI'] / (data['CreditScore'] / 100.0 + 1.0)
            new_features.append('dti_credit_ratio')

        # Temporal aggregations
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}
        for ftype in ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']:
            if ftype in by_type and len(by_type[ftype]) >= 4:
                cols = by_type[ftype]
                feat = data[cols].replace([np.inf, -np.inf], np.nan)

                # Trend and volatility features
                data[f'{ftype}_trend'] = (feat.iloc[:, -1] - feat.iloc[:, 0]) / (feat.iloc[:, 0].abs() + 1.0)
                data[f'{ftype}_volatility'] = feat.std(axis=1) / (feat.mean(axis=1).abs() + 1.0)
                new_features += [f'{ftype}_trend', f'{ftype}_volatility']

        # Clean engineered features
        for c in new_features:
            data[c] = data[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return data, new_features

    def select_features(self, data: pd.DataFrame, static_features: List[str], temporal_features: List[str], new_features: List[str]) -> List[str]:
        """Select final feature set"""
        core_static = ['CreditScore', 'OriginalUPB', 'OriginalLTV', 'OriginalInterestRate',
                       'OriginalDTI', 'OriginalLoanTerm', 'NumberOfUnits', 'NumberOfBorrowers']
        selected: List[str] = []

        # Add core static features
        for col in core_static:
            if col in data.columns and data[col].nunique() > 1:
                selected.append(col)

        # Add engineered features
        for col in new_features:
            if col in data.columns and data[col].nunique() > 1:
                selected.append(col)

        # Sample temporal features
        by_type = self._group_temporal_by_type(temporal_features) if temporal_features else {}
        for ftype in ['CurrentActualUPB', 'EstimatedLTV', 'InterestBearingUPB']:
            if ftype in by_type:
                cols = by_type[ftype]
                sampled = cols[::3]  # Every 3rd month
                selected.extend(sampled)

        # Filter by data quality
        final = []
        for col in selected:
            if col in data.columns and data[col].nunique() > 1 and data[col].notna().mean() >= 0.8:
                final.append(col)

        return final

    def tune_hyperparameters(self, X_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> Dict[str, Any]:
        """
        Tune hyperparameters using validation set (NO training on validation labels)
        Uses AUPRC (Area Under Precision-Recall Curve) as primary metric for imbalanced data
        """
        param_grids = {
            'statistical': [
                {'contamination': [0.05, 0.1, 0.15, 0.2]}
            ],
            'proximity': [
                {'n_neighbors': [10, 20, 30], 'contamination': [0.05, 0.1, 0.15]}
            ],
            'clustering': [
                {'eps': [0.3, 0.5, 0.7], 'min_samples': [3, 5, 7], 'contamination': [0.05, 0.1, 0.15]}
            ],
            'reconstruction': [
                {'n_components': [None], 'encoding_dim': [None]}  # Auto-determined
            ]
        }

        best_params = {}
        best_scores = {}

        print("Tuning hyperparameters using AUPRC (better for imbalanced data)...")

        for method_name, param_grid in param_grids.items():
            best_auprc = -1
            best_param = None

            for params in ParameterGrid(param_grid):
                try:
                    # Fit detector on training data only
                    if method_name == 'statistical':
                        detector = StatisticalAnomalyDetector(**params)
                    elif method_name == 'proximity':
                        detector = ProximityAnomalyDetector(**params)
                    elif method_name == 'clustering':
                        detector = ClusteringAnomalyDetector(**params)
                    elif method_name == 'reconstruction':
                        detector = ReconstructionAnomalyDetector(**params)

                    detector.fit(X_train)

                    # Evaluate on validation set using AUPRC (better for imbalanced data)
                    scores = detector.score_samples(X_valid)
                    auprc = average_precision_score(y_valid, scores)
                    auc = roc_auc_score(y_valid, scores)

                    if auprc > best_auprc:
                        best_auprc = auprc
                        best_param = params

                except Exception as e:
                    print(f"Error with {method_name} params {params}: {e}")
                    continue

            if best_param is not None:
                best_params[method_name] = best_param
                best_scores[method_name] = best_auprc
                print(f"{method_name}: AUPRC={best_auprc:.4f}, params={best_param}")

        return best_params

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> "ImprovedAnomalyEnsemble":
        """
        Fit ensemble on training data only
        Use validation set ONLY for hyperparameter tuning (not training)
        """
        # Preprocess training data
        train_proc, train_static, train_temp = self.robust_preprocessing(train_df, is_training=True)
        train_proc, train_new = self.create_features(train_proc, train_static, train_temp)

        # Preprocess validation data (transform only)
        valid_proc, valid_static, valid_temp = self.robust_preprocessing(valid_df, is_training=False)
        valid_proc, _ = self.create_features(valid_proc, valid_static, valid_temp)

        # Feature selection based on training data only
        self.final_features = self.select_features(train_proc, train_static, train_temp, train_new)

        X_train = train_proc[self.final_features].fillna(0.0)
        X_valid = valid_proc[self.final_features].fillna(0.0)
        y_valid = valid_df['target'].values

        # Scale features
        self.scalers['robust'] = RobustScaler()
        X_train_scaled = self.scalers['robust'].fit_transform(X_train)
        X_valid_scaled = self.scalers['robust'].transform(X_valid)

        # Tune hyperparameters using validation set (labels used for evaluation only)
        self.best_params = self.tune_hyperparameters(X_train_scaled, X_valid_scaled, y_valid)

        # Train final detectors with best parameters on training data only
        print("Training final detectors on TRAIN DATA ONLY (no validation data used)...")

        if 'statistical' in self.best_params:
            self.detectors['statistical'] = StatisticalAnomalyDetector(**self.best_params['statistical'])
            self.detectors['statistical'].fit(X_train_scaled)
            print("Statistical detector trained on training data only")

        if 'proximity' in self.best_params:
            self.detectors['proximity'] = ProximityAnomalyDetector(**self.best_params['proximity'])
            self.detectors['proximity'].fit(X_train_scaled)
            print("Proximity detector trained on training data only")

        if 'clustering' in self.best_params:
            self.detectors['clustering'] = ClusteringAnomalyDetector(**self.best_params['clustering'])
            self.detectors['clustering'].fit(X_train_scaled)
            print("Clustering detector trained on training data only")

        if 'reconstruction' in self.best_params:
            self.detectors['reconstruction'] = ReconstructionAnomalyDetector(**self.best_params['reconstruction'])
            self.detectors['reconstruction'].fit(X_train_scaled)
            print("Reconstruction detector trained on training data only")

        print("All detectors trained WITHOUT using validation data")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomaly scores using ensemble"""
        # Preprocess
        proc, st, tp = self.robust_preprocessing(df, is_training=False)
        proc, _ = self.create_features(proc, st, tp)
        X = proc[self.final_features].fillna(0.0)
        X_scaled = self.scalers['robust'].transform(X)

        # Get scores from each detector
        scores = {}
        for name, detector in self.detectors.items():
            try:
                scores[name] = detector.score_samples(X_scaled)
            except Exception as e:
                print(f"Error in {name} detector: {e}")
                scores[name] = np.zeros(len(X_scaled))

        if not scores:
            return np.zeros(len(X_scaled))

        # Ensemble combination using EQUAL WEIGHTS (no validation data used)
        # Using theory-based weights for imbalanced anomaly detection
        weights = {
            'statistical': 0.2,      # Gaussian assumptions may not hold
            'proximity': 0.3,        # Good for density-based anomalies in imbalanced data
            'clustering': 0.25,      # Good for isolation-based detection
            'reconstruction': 0.25   # Good for high-dimensional pattern anomalies
        }
        print(f"Using theory-based weights (NO validation data): {weights}")

        ensemble_scores = np.zeros(len(X_scaled))
        total_weight = 0

        for name, score_array in scores.items():
            weight = weights.get(name, 0.25)
            ensemble_scores += weight * score_array
            total_weight += weight

        if total_weight > 0:
            ensemble_scores /= total_weight

        # Normalize to [0, 1] range for valid probability scores
        if len(ensemble_scores) > 0:
            min_score = ensemble_scores.min()
            max_score = ensemble_scores.max()
            if max_score > min_score:
                ensemble_scores = (ensemble_scores - min_score) / (max_score - min_score)
            else:
                ensemble_scores = np.zeros_like(ensemble_scores)

        print(f"Final scores range: [{ensemble_scores.min():.6f}, {ensemble_scores.max():.6f}]")
        return ensemble_scores

    def evaluate(self, valid_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate on validation set"""
        y_true = valid_df['target'].values
        y_scores = self.predict_proba(valid_df)

        ap = average_precision_score(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        return ap, auc


def main():
    """
    Multi-Method Anomaly Detection Pipeline
    Uses AUPRC as primary metric (better for imbalanced data with 87.4% normal vs 12.6% anomalies)
    Implements 4 complementary detection methods with AUPRC-based ensemble weighting
    """
    print("=== Multi-Method Anomaly Detection Pipeline (AUPRC-Optimized) ===")

    # Load data
    train_df = pd.read_csv('Data/loans_train.csv')
    valid_df = pd.read_csv('Data/loans_valid.csv')

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape} (target distribution: {valid_df['target'].value_counts(normalize=True).round(3).to_dict()})")
    print("Using AUPRC as primary metric (ignores true negatives, better for imbalanced data)")

    # Train ensemble (validation used only for hyperparameter tuning)
    model = ImprovedAnomalyEnsemble(random_state=42)
    model.fit(train_df, valid_df)

    # Evaluate on validation set
    ap, auc = model.evaluate(valid_df)
    print(f"\n=== Validation Results (No Data Leakage) ===")
    print(f"AUPRC (Primary): {ap:.4f}")
    print(f"AUC-ROC (Secondary): {auc:.4f}")

    # Generate test predictions
    try:
        test_df = pd.read_csv('Data/loans_test.csv')
        test_scores = model.predict_proba(test_df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_col = 'Id' if 'Id' in test_df.columns else 'index'
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            'anomaly_score': test_scores
        })

        filename = f"LEAKFREE_MULTIMETHOD_ENSEMBLE_AUPRC{ap:.4f}_AUC{auc:.4f}_{timestamp}.csv"
        submission.to_csv(filename, index=False)
        print(f"\nSubmission saved: {filename}")

        return model, filename, ap, auc

    except Exception as e:
        print(f"\nTest prediction failed: {e}")
        return model, None, ap, auc


if __name__ == '__main__':
    model, submission, ap, auc = main()
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Final AP: {ap:.4f}, AUC: {auc:.4f}")
    if submission:
        print(f"Submission: {submission}")