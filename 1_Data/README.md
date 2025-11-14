# Dataset

## Files

- `loans_train.csv` - Training data (30,504 records, 100% normal loans)
- `loans_valid.csv` - Validation data (5,370 records, 12.61% anomalous)
- `loans_test.csv` - Test data (13,426 records, labels withheld)

## Dataset Description

Freddie Mac Single-Family Loan-Level Dataset for anomaly detection.
- Total features: 143 (129 numeric, 14 categorical)
- Target: Binary (0 = normal, 1 = delinquent)
- Anomaly definition: Current loan delinquency status != 0

## Usage

All model training must use only `loans_train.csv` (unsupervised, no anomalies).
Validation set is used only for hyperparameter tuning and model selection.
