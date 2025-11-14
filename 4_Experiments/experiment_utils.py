import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def rank01(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=float)
    med = np.nanmedian(x)
    x = np.nan_to_num(x, nan=med)
    s = pd.Series(x)
    return ((s.rank(method="average") - 1) / (len(s) - 1 + 1e-12)).to_numpy()

def compute_amortization_features(df):
    features = pd.DataFrame(index=df.index)

    upb_cols = [c for c in df.columns if 'CurrentActualUPB' in c]
    rate_cols = [c for c in df.columns if 'CurrentInterestRate' in c]

    if len(upb_cols) >= 2:
        df['balance_change_rate'] = (df[upb_cols[0]] - df[upb_cols[-1]]) / (df[upb_cols[0]] + 1)
        features['balance_change_rate'] = df['balance_change_rate']

        upb_values = df[upb_cols].values
        df['balance_volatility'] = np.std(upb_values, axis=1) / (np.mean(upb_values, axis=1) + 1)
        features['balance_volatility'] = df['balance_volatility']

    if 'OriginalUPB' in df.columns and len(upb_cols) > 0:
        df['upb_ratio'] = df[upb_cols[0]] / (df['OriginalUPB'] + 1)
        features['upb_ratio'] = df['upb_ratio']

    shortfall_scores = compute_amortization_shortfall(df)
    for k, v in shortfall_scores.items():
        features[k] = v

    return features

def compute_amortization_shortfall(df):
    features = {}

    interest_bearing_cols = [c for c in df.columns if 'InterestBearingUPB' in c]
    rate_cols = [c for c in df.columns if 'CurrentInterestRate' in c]
    remaining_cols = [c for c in df.columns if 'RemainingMonthsToLegalMaturity' in c]

    if len(interest_bearing_cols) < 2 or len(rate_cols) < 1 or len(remaining_cols) < 1:
        features['amort_short_mean'] = np.zeros(len(df))
        features['amort_short_70'] = np.zeros(len(df))
        features['amort_short_50'] = np.zeros(len(df))
        return features

    shortfalls = []
    for t in range(1, min(len(interest_bearing_cols), len(rate_cols), len(remaining_cols))):
        P = df[interest_bearing_cols[t]].fillna(0).values
        P_prev = df[interest_bearing_cols[t-1]].fillna(0).values
        r = (df[rate_cols[t]].fillna(0).values / 1200)
        n = df[remaining_cols[t]].fillna(1).values

        n = np.maximum(n, 1)

        interest_payment = r * P
        denom = np.maximum((1 + r)**n - 1, 1e-12)
        total_payment = P * r * (1 + r)**n / denom
        principal_payment_expected = np.maximum(total_payment - interest_payment, 0)

        principal_payment_observed = np.maximum(P_prev - P, 0)

        shortfall = np.clip((principal_payment_expected - principal_payment_observed) /
                           (principal_payment_expected + 1e-6), 0, 1)
        shortfalls.append(shortfall)

    if len(shortfalls) > 0:
        shortfalls_array = np.column_stack(shortfalls)
        features['amort_short_mean'] = np.mean(shortfalls_array, axis=1)
        features['amort_short_70'] = np.mean(shortfalls_array > 0.7, axis=1)
        features['amort_short_50'] = np.mean(shortfalls_array > 0.5, axis=1)
    else:
        features['amort_short_mean'] = np.zeros(len(df))
        features['amort_short_70'] = np.zeros(len(df))
        features['amort_short_50'] = np.zeros(len(df))

    return features

def compute_payment_irregularity(df):
    nibupb_cols = [c for c in df.columns if 'NonInterestBearingUPB' in c]

    if len(nibupb_cols) >= 10:
        late_cols = [c for c in nibupb_cols if any(f'_{i}_' in c or f'_{i}' == c[-2:] for i in [10,11,12,13])]
        if late_cols:
            late_nibupb = df[late_cols].fillna(0).values
            return np.mean(late_nibupb, axis=1)

    return np.zeros(len(df))

def prepare_features(df, scaler=None, pca=None, fit=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['target', 'Id', 'index']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X = df[feature_cols].fillna(0).values

    if fit:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X) if scaler else X

    if pca is not None or fit:
        if fit:
            pca = PCA(n_components=min(80, X_scaled.shape[1]), random_state=42)
            X_pca = pca.fit_transform(X_scaled)
        else:
            X_pca = pca.transform(X_scaled)
        return X_pca, scaler, pca

    return X_scaled, scaler, None
