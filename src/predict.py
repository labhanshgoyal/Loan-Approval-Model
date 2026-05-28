import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_data, encode_features

MODELS_DIR = "models"

def load_artifacts(): #loads the trained model and preprocessing objects from the models folder
    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    imputer = joblib.load(f"{MODELS_DIR}/imputer.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    model_name = joblib.load(f"{MODELS_DIR}/model_name.pkl")
    return model, imputer, scaler, feature_names, model_name

def predict_single(applicant): #makes prediction for a single applicant
    model, imputer, scaler, feature_names, model_name = load_artifacts()
    df = pd.DataFrame([applicant])
    df = clean_data(df)
    df = encode_features(df)
    if 'Loan_Status' in df.columns: #New applicants don't have a Loan_Status yet, if the column exists, drop it
        df.drop('Loan_Status', axis=1, inplace=True)
    num_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    num_cols = [c for c in num_cols if c in df.columns]
    df[num_cols] = imputer.transform(df[num_cols]) #fills missing values using the median values learned from training data
    df = df.reindex(columns=feature_names, fill_value=0) #ensures the columns are in the same order as the training data
    cols_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Dependents']
    cols_to_scale = [c for c in cols_to_scale if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale]) #scale the numerical columns
    probability = model.predict_proba(df)[0][1] #model calculates probability
    approved = bool(probability >=0.5) #decides if the loan is approved or rejected based on probability
    if probability >=0.75:
        risk_level = "Low Risk"
    elif probability >=0.5:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    return {
        "approved": approved,
        "probability": round(float(probability), 4),
        "risk_level": risk_level,
        "model_name": model_name
    }