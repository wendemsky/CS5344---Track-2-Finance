# CS5344 Track 2: Finance - Loan Anomaly Detection

**Team Project for CS5344 Big Data Analytics - Track 2: Finance**

## 🎯 Project Overview

This project focuses on **loan-level anomaly detection for repayment behavior**. We develop machine learning models to identify loans that are likely to miss scheduled payments based on static borrower information and temporal repayment patterns.

### Problem Statement
- **Objective**: Detect loans that will default (anomalous) vs. loans that remain current (normal)
- **Data**: Static loan features + 14 months of temporal performance data
- **Challenge**: Semi-supervised learning (training contains only normal loans)

### Current Performance
- **Kaggle Leaderboard Score**: 0.38 (Average Precision)
- **Validation Performance**: AP=0.6141, AUC=0.8593
- **Approach**: Multi-stage ensemble combining unsupervised + supervised learning

## 📊 Dataset

### Structure
- **Training**: 30,504 loans × 145 features (100% normal loans)
- **Validation**: 5,370 loans × 145 features (87.39% normal, 12.61% abnormal)  
- **Test**: 13,426 loans × 144 features (unlabeled for competition)

### Feature Types
- **Static Features (31)**: Borrower credit, loan terms, property info
- **Temporal Features (112)**: 8 performance metrics × 14 months

## 🚀 Our Solution: Multi-Stage Ensemble

### Stage 1: Unsupervised Learning
- **Isolation Forest** trained on normal loans only
- Learns typical repayment behavior patterns
- Provides baseline anomaly scores

### Stage 2: Feature Engineering  
- **Financial Risk Ratios**: Credit-to-LTV, DTI-to-Credit ratios
- **Temporal Patterns**: Trends, volatility, recent changes
- **Domain Features**: Payment burden estimates, risk scoring

### Stage 3: Supervised Learning
- **XGBoost** trained on validation data labels
- Uses engineered features + anomaly scores
- Handles severe class imbalance (7:1 ratio)

### Stage 4: Ensemble Combination
- **Optimal weighting**: 30% unsupervised + 70% supervised
- **Robust predictions** combining different learning paradigms

## 📁 Repository Structure

```
├── Data/                           # Dataset files (gitignored)
│   ├── loans_train.csv            
│   ├── loans_valid.csv
│   └── loans_test.csv
├── final_ensemble.py              # Main competition pipeline
├── eda_loan_analysis.py           # Exploratory data analysis
├── proposal_visualizations.py     # EDA visualization charts
├── ensemble_anomaly_detection.py  # Basic ensemble approach
├── advanced_ensemble_pipeline.py  # Advanced multi-stage approach
├── competitive_ensemble.py        # Competition-focused version
├── starter_kit.py                 # Original starter code
├── PROJECT_DOCUMENTATION.md       # Complete data documentation
├── APPROACH_DOCUMENTATION.md      # Detailed methodology
├── proposal_plots/                # EDA charts for analysis
├── FINAL_ENSEMBLE_*.csv          # Best submission file
└── README.md                      # This file
```

## 🔧 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### Run the Complete Pipeline
```bash
# Run exploratory data analysis
python eda_loan_analysis.py

# Generate visualization plots
python proposal_visualizations.py

# Train and generate submission
python final_ensemble.py
```

### Expected Output
- **Validation metrics**: AP and AUC scores on validation set
- **Submission file**: CSV ready for Kaggle submission
- **Performance summary**: Model performance breakdown

## 📈 Key Results & Insights

### Model Performance
| Model Component | Contribution | Performance Impact |
|----------------|--------------|-------------------|
| Isolation Forest | 30% weight | Baseline anomaly detection |
| XGBoost + Features | 70% weight | Supervised discrimination |
| **Combined Ensemble** | **Final** | **0.38 AP on Kaggle** |

### Critical Features
1. **Credit Score / LTV Ratio**: Risk assessment combination
2. **Temporal Volatility**: Payment consistency patterns  
3. **Recent Payment Changes**: Short-term behavior shifts
4. **Payment Burden**: Monthly payment capacity estimates

### Key Insights
- **Semi-supervised approach** perfectly matches problem structure
- **Domain knowledge** significantly improves feature engineering
- **Ensemble diversity** provides robustness against overfitting
- **Temporal patterns** are highly predictive of future defaults

## 🎯 Future Improvements

### Planned Enhancements
1. **LSTM Autoencoders** for advanced temporal pattern detection
2. **Stacked Ensembles** with meta-learning
3. **Advanced Feature Engineering** with fourier transforms
4. **Hyperparameter Optimization** using Bayesian methods

### Target Performance
- **Phase 1**: Improve to 0.42+ AP with advanced features
- **Phase 2**: Reach 0.45+ AP with deep learning models
- **Final Goal**: Achieve top-tier competition performance

## 📚 Documentation

- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**: Complete dataset and feature descriptions
- **[APPROACH_DOCUMENTATION.md](APPROACH_DOCUMENTATION.md)**: Detailed methodology and future plans
- **Code Comments**: Extensive documentation in all Python files

## 🏆 Competition Strategy

### Submission Strategy
- **Conservative baseline**: Current 0.38 AP submission
- **Iterative improvement**: Multiple enhanced models
- **Robust validation**: Cross-validation and performance monitoring
- **Final selection**: Best performing model on private leaderboard

### Risk Mitigation
- **Overfitting prevention**: Strong regularization and early stopping
- **Distribution shift**: Adversarial validation techniques
- **Feature leakage**: Careful temporal causality checking

## 🤝 Team & Contributions

This project demonstrates advanced machine learning techniques for financial risk assessment, combining domain expertise with cutting-edge anomaly detection methods.

### Key Contributions
- Novel semi-supervised ensemble architecture
- Domain-aware feature engineering for financial data
- Robust preprocessing pipeline for mixed data types
- Competition-ready submission pipeline

---

**CS5344 Big Data Analytics - Track 2: Finance**  
*Loan Anomaly Detection for Repayment Behavior Analysis*