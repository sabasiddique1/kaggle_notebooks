# Loan Approval Prediction & Credit Risk Analysis (Portfolio Project)

## Business problem (bank context)
A commercial bank receives thousands of personal loan applications per month. The underwriting team wants to:
1) Understand what factors influence approvals  
2) Predict **probability of approval** for new applications  
3) Choose an **approval/decline threshold** that reflects business risk  
4) Keep the solution **explainable, fair, and production-ready**

## Dataset
- File: `data/Loan.csv`
- Target: `LoanApproved`  
  - `1` = approved  
  - `0` = declined  
- Key feature groups:
  - Demographics (age, marital status, dependents, etc.)
  - Income / debt burden (income, monthly debt payments, total DTI, etc.)
  - Credit behavior (credit score, utilization, inquiries, payment history, defaults/bankruptcy)
  - Loan request details (amount, duration, purpose)
  - Application metadata (application date)

## Approach & methodology
### 1) Data understanding & cleaning
- Parsed `ApplicationDate` and engineered `AppYear`, `AppMonth`, `AppDayOfWeek`
- Checked for duplicates and missing values (dataset is very clean)
- Validated ranges (Age, CreditScore, DTI ratios)

### 2) EDA (decision-oriented)
Key observed approval rates:
- **Overall approval rate:** ~23.9%
- **Employment status approval rate:**
  - Self-Employed: 27.8%
  - Employed: 24.0%
  - Unemployed: 18.2%
- **Education level approval rate (top to bottom):**
  - Doctorate: 44.0%
  - Master: 35.1%
  - Bachelor: 26.6%
  - Associate: 20.4%
  - High School: 14.4%
- **Credit score bands:**
  - <580: 19.3%
  - 580–629: 26.1%
  - 630–679: 38.4%
  - 680+: 62.3%
- **Derogatory history:**
  - Previous defaults = 1: 15.5% (vs 24.8% with no defaults)
  - Bankruptcy = 1: 11.1% (vs 24.6% with no bankruptcy)

### 3) Modeling & evaluation (sklearn pipelines)
Models:
- KNN
- Decision Tree
- Random Forest (final)
- Logistic Regression (extra; used for coefficient interpretability)

Preprocessing:
- Numeric: median impute + standard scaling
- Categorical: most-frequent impute + one-hot encoding

Metrics:
- ROC-AUC
- Precision-Recall (PR-AUC)
- Confusion Matrix + classification report

### 4) Decision threshold (business risk)
We used a cost-sensitive rule:
- False approval (FP) costs **5×** false decline (FN)

Using a validation split, the chosen **Random Forest threshold** was **~0.67**, which significantly reduces risky false approvals.

### 5) Interpretability
- Random Forest feature importance (ranked)
- Partial dependence plots: `CreditScore`, `TotalDebtToIncomeRatio`, `RiskScore`
- Logistic Regression coefficients for plain-English explanations

## Final model performance (holdout test)
**Random Forest**
- Test ROC-AUC: ~0.999  
- Test PR-AUC: ~0.997  
- At threshold ≈ 0.67:
  - Precision ≈ 0.984
  - Recall ≈ 0.946

> Note: Extremely high scores can indicate **label leakage** (features that indirectly encode underwriting decisions). In real bank work, we would audit and potentially remove post-decision fields (e.g., derived risk/price variables) and re-train.

## Repository structure
```
loan-approval-ml-project/
├── data/
│   └── Loan.csv
├── notebook/
│   └── ML.ipynb
└── README.md
```

## How to run
1. Create a Python environment (Python 3.9+ recommended)
2. Install dependencies:
   - pandas, numpy, matplotlib
   - scikit-learn
   - jupyter
3. Start Jupyter and open:
   - `notebook/ML.ipynb`

## Key stakeholder takeaway
The strongest approval drivers in this dataset are consistent with underwriting intuition:
- higher **CreditScore** increases approval
- higher **TotalDebtToIncomeRatio** decreases approval
- **RiskScore** and derogatory indicators (defaults/bankruptcy) materially reduce approval odds

The project also demonstrates how to convert business risk preferences into an operational **approval threshold**.
