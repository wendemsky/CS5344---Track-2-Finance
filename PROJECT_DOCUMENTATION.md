# Loan Anomaly Detection Project Documentation
**Track 2: Finance - CS5344 Big Data Analytics Project**

---

## Project Overview

### Problem Description
We study **loan-level anomaly detection for repayment behavior**. Each loan comprises:
1. **Static context vector** describing the loan and borrower (e.g., borrower attributes)
2. **Monthly sequence of performance variables** (e.g., outstanding unpaid principal balance) observed over the life of the loan

The **loan-level binary outcome** indicates whether the loan ever fails to meet its monthly obligation or remains current throughout.

### Key Challenges
- **Borrower and product heterogeneity**
- **Mixed-type feature domains**
- **Strong class imbalance** due to the relative rarity of abnormal loans
- **Semi-supervised learning** (training contains only normal loans)
- **Temporal sequence modeling** with variable lengths

### Data Formulation
For i = 1, ..., n, the i-th loan is represented as:

```
xi = (si, (ti,1, ri,1), (ti,2, ri,2), ..., (ti,Ti, ri,Ti), yi)
```

Where:
- `si` = static loan information (borrower attributes)
- `{ti,k}Ti k=1` = months with ti,1 < ... < ti,Ti
- `Ti ∈ N` = number of months for loan i
- `ri,k` = monthly repayment information vector at month ti,k
- `yi ∈ {0, 1}` = label (0=normal, 1=abnormal)

---

## Dataset Description

### Data Files
- **`loans_train.csv`**: Training data (30,504 loans, 145 features)
- **`loans_valid.csv`**: Validation data (5,370 loans, 145 features)  
- **`loans_test.csv`**: Test data (unlabeled)
- **`sample_loans_train.csv`**: Sample subset for testing

### Data Structure
- **Total Features**: 145
- **Static Features**: 31 (origination variables)
- **Temporal Features**: 112 (8 feature types × 14 months)
- **Temporal Span**: Up to 14 months of loan history

---

## Feature Descriptions

### Target Variable
| Column | Description |
|--------|-------------|
| `index` | Unique identifier assigned to each loan |
| `target` | Binary label: 0 = normal loan (no default), 1 = abnormal loan (default/anomalous) |

### Static Features (Origination Variables)

#### Borrower Information
| Column | Description | Range/Values |
|--------|-------------|--------------|
| `CreditScore` | Borrower credit score at origination | 300–850 (9999 = missing/invalid) |
| `FirstTimeHomebuyerFlag` | First-time homebuyer status | Y/N/9 (not available) |
| `OriginalDTI` | Debt-to-Income ratio (%) | >65% or missing = 999 |
| `NumberOfBorrowers` | Number of borrowers | 1–10 |

#### Loan Terms
| Column | Description | Range/Values |
|--------|-------------|--------------|
| `OriginalUPB` | Original unpaid principal balance | Nearest $1,000 |
| `OriginalLTV` | Loan-to-Value ratio at origination | Invalid = 999 |
| `OriginalCLTV` | Combined Loan-to-Value ratio | |
| `OriginalInterestRate` | Note rate at origination | |
| `OriginalLoanTerm` | Scheduled term in months | |
| `ProductType` | FRM = Fixed Rate, ARM = Adjustable Rate | |

#### Property Information
| Column | Description | Values |
|--------|-------------|---------|
| `PropertyState` | Two-letter state/territory code | |
| `PropertyType` | SF/CO/PU/MH/CP/99 | Single-Family, Condo, PUD, Manufactured, Co-op |
| `PostalCode` | Masked ZIP code | First 3 digits + "00" |
| `NumberOfUnits` | Number of dwelling units | 1–4 |
| `OccupancyStatus` | P/I/S/9 | Primary, Investment, Second Home |
| `MSA` | Metropolitan Statistical Area code | null if unknown |

#### Loan Characteristics
| Column | Description | Values |
|--------|-------------|---------|
| `Channel` | Origination channel | R/B/C/T/9 (Retail/Broker/Correspondent/TPO/Not Available) |
| `LoanPurpose` | P/C/N/R/9 | Purchase, Cash-out Refi, No Cash-out Refi, Refi Not Specified |
| `MI_Pct` | Mortgage insurance percentage | 0=none, 1–55=coverage%, 999=not available |
| `PPM_Flag` | Prepayment penalty | Y/N |
| `InterestOnlyFlag` | Interest-only payments | Y/N |
| `BalloonIndicator` | Balloon payment | Y/N |

#### Dates
| Column | Description | Format |
|--------|-------------|---------|
| `FirstPaymentDate` | First scheduled payment due date | YYYYMM |
| `MaturityDate` | Scheduled maturity date | YYYYMM |

#### Program Indicators
| Column | Description |
|--------|-------------|
| `SuperConformingFlag` | Loan exceeded conforming limits but qualified as "super conforming" |
| `PreHARP_Flag` | HARP program indicator |
| `ProgramIndicator` | Program participation indicator |
| `ReliefRefinanceIndicator` | Relief refinance program indicator |
| `PropertyValMethod` | 1=ACE, 2=Full, 3=Other, 4=ACE+PDR |

#### Service Providers
| Column | Description |
|--------|-------------|
| `SellerName` | Entity that sold the loan ("Other Sellers" if below disclosure threshold) |
| `ServicerName` | Entity servicing the loan ("Other Servicers" if below disclosure threshold) |

### Temporal Features (Performance Panel)

For each month N = 0, 1, 2, ..., 13, the following 8 features are tracked:

| Feature Pattern | Description |
|-----------------|-------------|
| `N_CurrentActualUPB` | Current unpaid principal balance (interest-bearing + non-interest-bearing) |
| `N_CurrentInterestRate` | Mortgage interest rate in effect for that period |
| `N_CurrentNonInterestBearingUPB` | Non-interest-bearing portion of UPB (e.g., deferred modification amounts) |
| `N_EstimatedLTV` | Current estimated Loan-to-Value ratio from Freddie Mac's AVM (1–998, 999=unknown) |
| `N_InterestBearingUPB` | Portion of UPB that accrues interest |
| `N_LoanAge` | Number of months since first payment date (or modification date) |
| `N_MonthlyReportingPeriod` | Period identifier in YYYYMM format |
| `N_RemainingMonthsToLegalMaturity` | Remaining months until scheduled maturity |

---

## Exploratory Data Analysis Results

### Dataset Statistics
- **Training Data**: 30,504 loans (100% normal loans)
- **Validation Data**: 5,370 loans (87.39% normal, 12.61% abnormal)
- **Feature Count**: 145 total (31 static + 112 temporal)
- **Data Types**: 71 int64, 60 float64, 14 object/categorical

### Class Distribution
```
Training Set:
- Normal loans (target=0): 30,504 (100.00%)
- Abnormal loans (target=1): 0 (0.00%)

Validation Set:
- Normal loans (target=0): 4,693 (87.39%)
- Abnormal loans (target=1): 677 (12.61%)
```

### Key Static Feature Statistics
| Feature | Mean | Std | Min | Max | Missing |
|---------|------|-----|-----|-----|---------|
| CreditScore | 753.56 | 156.05 | 600.00 | 9999.00 | 0 |
| OriginalUPB | 317,095.43 | 181,297.52 | 15,000.00 | 1,700,000.00 | 0 |
| OriginalLTV | 75.19 | 19.43 | 6.00 | 97.00 | 0 |
| OriginalInterestRate | 6.72 | 0.55 | 2.50 | 9.12 | 0 |
| OriginalDTI | 37.83 | 10.82 | 2.00 | 999.00 | 0 |

### Missing Values Analysis
- **ReliefRefinanceIndicator**: 100% missing
- **PreHARP_Flag**: 100% missing  
- **SuperConformingFlag**: 98.92% missing
- **MSA**: 11.22% missing
- **Temporal features**: No missing values

### Temporal Characteristics
- **Sequence Length**: All loans have exactly 14 months of data
- **Temporal Feature Types**: 8 performance metrics tracked monthly
- **Time Range**: Consistent reporting periods across all loans

---

## Modeling Implications

### Problem Type
This is a **semi-supervised anomaly detection** problem because:
- Training data contains ONLY normal loans (target=0)
- Validation/test data contains both normal and abnormal loans
- Must learn to identify anomalies without seeing examples during training

### Key Challenges

1. **Semi-Supervised Learning**
   - No abnormal samples in training data
   - Must learn normal patterns and detect deviations

2. **Class Imbalance**
   - Only 12.61% abnormal loans in validation
   - Standard classification metrics may be misleading

3. **Mixed Data Types**
   - Numerical: credit scores, balances, rates
   - Categorical: states, loan purposes, flags
   - Temporal: monthly performance sequences

4. **High Dimensionality**
   - 145 features with potential for curse of dimensionality
   - Need dimensionality reduction or feature selection

5. **Temporal Dependencies**
   - Monthly sequences contain important patterns
   - Variable importance across different time periods

### Recommended Approaches

1. **Anomaly Detection Models**
   - Isolation Forest
   - One-Class SVM
   - Autoencoders (for temporal patterns)
   - LSTM-based anomaly detection

2. **Feature Engineering**
   - Temporal aggregations (trends, volatility)
   - Static feature interactions
   - Domain-specific financial ratios

3. **Evaluation Metrics**
   - Average Precision (AP)
   - Area Under ROC Curve (AUC-ROC)
   - Precision-Recall curves
   - F1-score at different thresholds

4. **Validation Strategy**
   - Use validation set for model selection
   - Time-based splits if temporal order matters
   - Cross-validation on normal loans only

---

## Data Quality Notes

### Strengths
- No duplicate indices
- Consistent temporal sequence lengths
- Complete temporal performance data
- Rich feature set covering borrower, loan, and property characteristics

### Considerations
- Several features have high missingness (program indicators)
- Some categorical encodings use special values (999, 9999) for missing data
- Credit scores above 850 coded as 9999 (need handling)
- DTI values above 65% coded as 999

### Preprocessing Requirements
- Handle missing value encodings (999, 9999)
- Categorical feature encoding
- Numerical feature scaling/normalization
- Temporal sequence preprocessing
- Feature selection for high-dimensional static features

---

## Project Timeline & Next Steps

1. **Data Preprocessing** - Handle missing values, encode features
2. **Feature Engineering** - Create temporal aggregations and domain features  
3. **Model Development** - Implement and compare anomaly detection approaches
4. **Evaluation** - Use appropriate metrics for imbalanced semi-supervised learning
5. **Model Selection** - Choose best performing approach on validation set
6. **Final Testing** - Evaluate on test set and prepare submission

---

*This documentation serves as a comprehensive reference for the loan anomaly detection project, combining domain knowledge, data understanding, and analytical insights to guide model development.*