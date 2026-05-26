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
        print(f"{name}: Accuracy={results[name]['metrics']['Accuracy']:.3f}, F1={results[name]['metrics']['F1-Score']:.3f}, AUC={results[name]['metrics']['ROC-AUC']:.3f}")
    return results

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices", fontsize = 16, fontweight="bold")
    for ax, (name, data) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, data["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', ax = ax, cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        ax.set_title(f"{name}\nAccuracy: {data['metrics']['Accuracy']:.2%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrices.png")

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for (name, data), color in zip(results.items(), colors):
        RocCurveDisplay.from_predictions(
            y_test, data["y_proba"],
            name=f"{name} (AUC={data['metrics']['ROC-AUC']:.3f})",
            ax=plt.gca(), color=color, lw=2
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    plt.title("ROC Curves", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: roc_curves.png")