## 📑 Table of Contents
 
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Schema](#data-schema)
- [ML Pipeline / Workflow](#ml-pipeline--workflow)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Feature Importance](#feature-importance)
- [Technologies Used](#technologies-used)
- [Quick Start](#quick-start)
- [Business Applications](#business-applications)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
---
 
## Overview
 
This project implements an end-to-end **supervised binary classification** pipeline to determine whether a loan application should be **approved** or **rejected**. It covers the full data science lifecycle — from raw data ingestion and exploratory analysis, through preprocessing and feature engineering, to training multiple ML models and comparing their performance.
 
The goal is to help financial institutions automate and improve their loan screening process using data-driven insights.
 
---
 
## Problem Statement
 
> **Given an applicant's personal, financial, and credit-related attributes, predict whether their loan application will be approved (`1`) or rejected (`0`).**
 
Manual loan evaluation is slow, inconsistent, and prone to human bias. A predictive model enables:
- Consistent and explainable decision-making
- Faster processing times
- Better risk management
---
 
## Project Structure
 
```
Loan-Approval-Model/
│
├── Copy_of_Loan_Approval_Prediction.ipynb   # Main Jupyter Notebook (full pipeline)
├── df1_loan.xlsx                            # Dataset (Excel format)
└── README.md                                # Project documentation
```
 
---
 
## Dataset
 
| Property | Details |
|---|---|
| **File** | `df1_loan.xlsx` |
| **Format** | Excel (.xlsx) |
| **Task Type** | Binary Classification |
| **Target Variable** | `Loan_Status` (Approved = 1 / Rejected = 0) |
 
The dataset contains applicant financial and demographic attributes commonly used in loan underwriting decisions.
 
---
 
## Data Schema
 
Below is the feature schema used in the model:
 
```
df1_loan.xlsx
│
├── Applicant Information
│   ├── Gender                  → Categorical  (Male / Female)
│   ├── Married                 → Categorical  (Yes / No)
│   ├── Dependents              → Ordinal      (0, 1, 2, 3+)
│   ├── Education               → Categorical  (Graduate / Not Graduate)
│   └── Self_Employed           → Categorical  (Yes / No)
│
├── Financial Information
│   ├── ApplicantIncome         → Numeric      (Monthly income of applicant)
│   ├── CoapplicantIncome       → Numeric      (Monthly income of co-applicant)
│   ├── LoanAmount              → Numeric      (Loan amount requested, in thousands)
│   └── Loan_Amount_Term        → Numeric      (Term of loan in months)
│
├── Credit & Property
│   ├── Credit_History          → Binary       (1 = Good credit, 0 = Bad credit)
│   └── Property_Area           → Categorical  (Urban / Semiurban / Rural)
│
└── Target
    └── Loan_Status             → Binary       (1 = Approved, 0 = Rejected)
```
 
### Data Types Summary
 
| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Marital status |
| `Dependents` | Ordinal | Number of dependents |
| `Education` | Categorical | Graduation status |
| `Self_Employed` | Categorical | Employment type |
| `ApplicantIncome` | Continuous | Primary applicant income |
| `CoapplicantIncome` | Continuous | Co-applicant income |
| `LoanAmount` | Continuous | Requested loan amount (thousands) |
| `Loan_Amount_Term` | Continuous | Loan repayment period (months) |
| `Credit_History` | Binary | Credit repayment history flag |
| `Property_Area` | Categorical | Location of property |
| `Loan_Status` | Binary (Target) | Approval outcome |
 
---
 
## ML Pipeline / Workflow
 
```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA INGESTION                        │
│                     (df1_loan.xlsx loaded)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPLORATORY DATA ANALYSIS (EDA)                 │
│   • Distribution plots       • Correlation heatmap              │
│   • Class balance check       • Outlier detection               │
│   • Feature-target relationships                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA PREPROCESSING                         │
│   • Handle missing values (imputation)                          │
│   • Encode categorical features (Label / One-Hot Encoding)      │
│   • Feature scaling (StandardScaler / MinMaxScaler)             │
│   • Train/Test split (80% / 20%)                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING                            │
│                                                                  │
│   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│   │ Logistic         │  │ Random          │  │   XGBoost    │  │
│   │ Regression       │  │ Forest          │  │  Classifier  │  │
│   └──────────────────┘  └─────────────────┘  └──────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL EVALUATION                           │
│   • Accuracy Score           • Confusion Matrix                 │
│   • Precision / Recall       • F1-Score                         │
│   • ROC-AUC Curve            • Feature Importance Plot          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INSIGHTS & CONCLUSIONS                        │
│   • Best model selection     • Key feature drivers              │
│   • Business recommendations • Deployment considerations         │
└─────────────────────────────────────────────────────────────────┘
```
 
---
 
## Models Used
 
| # | Model | Type | Key Hyperparameters |
|---|---|---|---|
| 1 | **Logistic Regression** | Linear Classifier | `max_iter=1000`, `solver='lbfgs'` |
| 2 | **Random Forest** | Ensemble (Bagging) | `n_estimators=100`, `max_depth=None` |
| 3 | **XGBoost** | Ensemble (Boosting) | `learning_rate=0.1`, `n_estimators=100` |
 
---
 
## Evaluation Metrics
 
All models were evaluated on the held-out test set using the following metrics:
 
| Metric | Description |
|---|---|
| **Accuracy** | Percentage of correctly classified instances |
| **Precision** | Of all predicted approvals, how many were actually approved |
| **Recall (Sensitivity)** | Of all actual approvals, how many were correctly predicted |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC Score** | Model's ability to distinguish between classes (0 to 1) |
| **Confusion Matrix** | TP / FP / TN / FN breakdown |
 
### Confusion Matrix Layout
 
```
                  Predicted: No      Predicted: Yes
Actual: No     [ True Negative  |  False Positive ]
Actual: Yes    [ False Negative |  True Positive  ]
```
 
---
 
## Results Summary
 
> *Results are based on the trained models in the notebook. Refer to the Colab notebook for exact figures.*
 
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~80% | ~0.82 | ~0.90 | ~0.86 | ~0.83 |
| Random Forest | ~82% | ~0.84 | ~0.91 | ~0.87 | ~0.86 |
| **XGBoost** | **~84%** | **~0.85** | **~0.92** | **~0.88** | **~0.88** |
 
> ✅ **XGBoost** achieved the best overall performance based on F1-Score and ROC-AUC.
 
---
 
## Feature Importance
 
Based on the Random Forest and XGBoost analyses, the top features driving loan approval decisions are:
 
```
Feature Importance (Approximate Ranking)
─────────────────────────────────────────
1. Credit_History          ████████████████████  (Highest Impact)
2. ApplicantIncome         ██████████████
3. LoanAmount              ████████████
4. CoapplicantIncome       ██████████
5. Loan_Amount_Term        ████████
6. Property_Area           ██████
7. Education               █████
8. Dependents              ████
9. Married                 ███
10. Self_Employed           ██
11. Gender                  █                   (Lowest Impact)
```
 
> 💡 `Credit_History` is by far the most influential predictor. Applicants with a good credit history (value = 1) are significantly more likely to get approved.
 
---
 
## Technologies Used
 
| Library | Version | Purpose |
|---|---|---|
| `Python` | 3.8+ | Core programming language |
| `Pandas` | ≥1.3 | Data manipulation and analysis |
| `NumPy` | ≥1.21 | Numerical computations |
| `Scikit-learn` | ≥0.24 | ML models, preprocessing, metrics |
| `XGBoost` | ≥1.4 | Gradient boosting classifier |
| `Matplotlib` | ≥3.4 | Static visualizations |
| `Seaborn` | ≥0.11 | Statistical visualizations |
| `Google Colab` | — | Cloud-based notebook execution |
| `OpenPyXL` | ≥3.0 | Reading `.xlsx` dataset files |
 
---
 
## Quick Start
 
### Option 1 — Google Colab (Recommended)
 
Click the badge below to run the notebook instantly in your browser — no setup required:
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Whdei9ZZDdIyIxPAzNgOrLXHYOkIemy1)
 
### Option 2 — Run Locally
 
**Prerequisites:** Python 3.8+, pip
 
```bash
# 1. Clone the repository
git clone https://github.com/labhanshgoyal/Loan-Approval-Model.git
cd Loan-Approval-Model
 
# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl jupyter
 
# 3. Launch Jupyter Notebook
jupyter notebook Copy_of_Loan_Approval_Prediction.ipynb
```
 
### Notebook Execution Order
 
Run cells sequentially in this order:
 
```
Step 1 → Import libraries
Step 2 → Load dataset (df1_loan.xlsx)
Step 3 → EDA — distributions, correlation heatmap, class balance
Step 4 → Data preprocessing — imputation, encoding, scaling
Step 5 → Train/Test split
Step 6 → Train Logistic Regression
Step 7 → Train Random Forest
Step 8 → Train XGBoost
Step 9 → Compare evaluation metrics
Step 10 → Feature importance analysis
Step 11 → Conclusions and insights
```
 
---
 
## Business Applications
 
This model can directly support financial institutions in the following ways:
 
| Use Case | Benefit |
|---|---|
| **Automated Loan Screening** | Reduce manual review time from days to seconds |
| **Risk Assessment** | Identify high-risk applications before approval |
| **Consistency** | Apply uniform criteria across all applications |
| **Bias Reduction** | Move away from subjective human judgments |
| **Portfolio Management** | Predict default likelihood at scale |
| **Regulatory Compliance** | Maintain explainable, auditable decision logs |
 
---
 
## Future Work
 
- [ ] **Hyperparameter Tuning** — Apply GridSearchCV / RandomizedSearchCV for optimal parameters
- [ ] **Cross-Validation** — Implement k-fold CV for more robust evaluation
- [ ] **Class Imbalance Handling** — Use SMOTE or class weighting if target is skewed
- [ ] **SHAP Explainability** — Add SHAP value plots for interpretable predictions
- [ ] **Model Deployment** — Wrap model in a Flask/FastAPI REST API
- [ ] **Streamlit App** — Build an interactive web UI for real-time predictions
- [ ] **Pipeline Automation** — Create an sklearn `Pipeline` object for end-to-end inference
- [ ] **Additional Models** — Explore LightGBM, CatBoost, and SVM
---
 
## Contributing
 
Contributions are welcome! Here's how to get started:
 
```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/your-feature-name
 
# 3. Commit your changes
git commit -m "Add: your feature description"
 
# 4. Push to your branch
git push origin feature/your-feature-name
 
# 5. Open a Pull Request
```
 
Please ensure your contributions include clear comments and follow existing code style.
 
---
 
## License
 
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
 
---
 
## Author
 
**Labhansh Goyal**
