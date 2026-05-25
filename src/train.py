import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0,
os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_for_training

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay)
from xgboost import XGBClassifier

DATA_PATH = "df1_loan.xlsx"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, eval_metric='logloss', verbosity=0, random_state=42)
    }

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            }
        }
        print(f"{name}: Accuracy={results[name]['metrics']['Accuracy']:.3f}, F1={results[name]['metrics'] ['F1-Score']:.3f}, AUC={results[name]['metrics'] ['ROC-AUC']:.3f}")
    return results