# This starter kit is built for Track 3: Cybersecurity, and feel free to transform and customize it for other tasks!

# --------------------
# Import libraries
# --------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

# --------------------
# Load data
# --------------------
train_df = pd.read_csv(r'sample_processes_train.csv')
valid_df = pd.read_csv(r'sample_processes_valid.csv')

process_ids_valid = valid_df["index"].values
X_train = train_df.drop(columns=["target"])
X_valid = valid_df.drop(columns=["target"])

y_valid = valid_df["target"].values

# --------------------
# Preprocessing & Feature engineering
# --------------------
categorical_cols = ["processName", "eventName", "hostName"]
numeric_cols = ["argsNum", "returnValue", "userId", "parentProcessId", "threadId"]

X_train_feat = X_train[categorical_cols + numeric_cols]
X_valid_feat = X_valid[categorical_cols + numeric_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
        ("num", StandardScaler(with_mean=False), numeric_cols),
    ],
    sparse_threshold=1.0, 
)

# --------------------
# Baseline Model: Isolation Forest
# --------------------
model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline([("prep", preprocess), ("clf", model)])

# Train on normal only
pipe.fit(X_train_feat)

# Score validation
# IsolationForest: higher score_samples = more normal
scores_normal = pipe["clf"].score_samples(pipe["prep"].transform(X_valid_feat))
raw_anom = -scores_normal  # invert: higher = more anomalous

# Normalize to [0,1]
min_v, max_v = np.min(raw_anom), np.max(raw_anom)
anom_score = (raw_anom - min_v) / (max_v - min_v + 1e-12)

# --------------------
# Evaluation
# --------------------
ap = average_precision_score(y_valid, anom_score)
roc_auc = roc_auc_score(y_valid, anom_score)

print(f"AP: {ap:.4f}")
print(f"AUC-ROC : {roc_auc:.4f}")

# Sample Output CSV
out_df = pd.DataFrame({"index": process_ids_valid, "anomaly_score": anom_score})
out_df.to_csv("valid_pred.csv", index=False)
print("Wrote valid_pred.csv")
