# 🏦 Loan Approval Predictor

> A machine learning web application that predicts whether a loan application should be approved or rejected — built with Python, scikit-learn, XGBoost, and Streamlit.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://your-app-link-here.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org)

---

## 🌐 Live Demo

**[👉 Try it here](https://your-app-link-here.streamlit.app)**

Fill in an applicant's personal, income, and loan details — get an instant approval prediction with confidence score and risk level.

---

## 📌 Project Overview

This project transforms a raw loan dataset (500 applicants) into a **production-ready ML pipeline** with a deployed web application. It solves three core problems from the original exploratory notebook:

| Problem | Original Notebook | This Project |
|---|---|---|
| Missing data | `dropna()` — deleted 110 rows (22%) | `SimpleImputer` — keeps all 500 rows |
| Models trained | Logistic Regression only | Logistic Regression + Random Forest + XGBoost |
| Deployment | Not deployable | Live Streamlit web app |

---

## 🎯 Model Results

Three models were trained and evaluated on an 80/20 train-test split:

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** ⭐ | **83.0%** | **88.6%** | **83.1%** |
| Random Forest | 81.0% | 86.9% | 79.7% |
| XGBoost | 76.0% | 82.6% | 78.8% |

> **Winner: Logistic Regression** — On small, structured datasets (~500 rows), a well-tuned linear model outperforms complex ensemble methods that tend to overfit.

---

## 🗂️ Project Structure

```
Loan-Approval-Model/
│
├── app.py                          # Streamlit web application
├── df1_loan.xlsx                   # Raw dataset (500 applicants)
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── preprocess.py               # Data cleaning & feature engineering pipeline
│   ├── train.py                    # Model training, evaluation & visualization
│   └── predict.py                  # Prediction pipeline for new applicants
│
├── models/
│   ├── best_model.pkl              # Saved best model (Logistic Regression)
│   ├── imputer.pkl                 # Fitted SimpleImputer (median values)
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_names.pkl           # Column order used during training
│   └── model_name.pkl              # Name of the best model
│
└── Copy_of_Loan_Approval_Prediction.ipynb  # Original exploratory notebook
```

---

## ⚙️ ML Pipeline

```
Raw Excel Data (500 rows)
        │
        ▼
  preprocess.py
  ├── load_data()          → Load & drop irrelevant columns
  ├── clean_data()         → Fix '3+' dependents, remove Total_Income
  ├── encode_features()    → Label encode binary cols, one-hot Property_Area
  ├── impute_missing()     → Fill NaN with column medians (saves 110 rows!)
  ├── train_test_split()   → 80% train / 20% test (stratified)
  └── scale_features()     → StandardScaler on continuous columns
        │
        ▼
  train.py
  ├── Train 3 models
  ├── Evaluate (Accuracy, Precision, Recall, F1, ROC-AUC)
  ├── Generate plots (Confusion Matrix, ROC Curves, Feature Importance)
  └── Save best model + artifacts to models/
        │
        ▼
  predict.py / app.py
  └── Load artifacts → Transform input → Predict → Return result
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/labhanshgoyal/Loan-Approval-Model.git
cd Loan-Approval-Model

# 2. Create and activate conda environment
conda create -n loan-approval python=3.10
conda activate loan-approval

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model
python src/train.py

# 5. Run the web app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| ML Library | scikit-learn, XGBoost |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | joblib |
| Data Source | Excel (openpyxl) |

---

## 📊 Features

- ✅ **3 ML models** trained and compared automatically
- ✅ **Imputation over deletion** — retains all training data
- ✅ **Full evaluation suite** — Accuracy, Precision, Recall, F1, ROC-AUC
- ✅ **Visualizations** — Confusion matrices, ROC curves, Feature importance charts
- ✅ **Risk scoring** — Low / Medium / High risk labels
- ✅ **Auto-training** — Model trains automatically if artifacts not found
- ✅ **Deployed** — Live web app accessible from any device

---

## 👤 Author

**Labhansh Goyal**
- GitHub: [@labhanshgoyal](https://github.com/labhanshgoyal)

---

*Built as part of a machine learning portfolio project — transforming an exploratory notebook into a fully deployed, production-ready application.*
