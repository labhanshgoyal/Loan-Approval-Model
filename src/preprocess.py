import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def load_data(filepath):
    df = pd.read_excel(filepath)
    df = df.iloc[:,2:]
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    df = df.copy()
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce') #coerce - converts non-numeric values to NaN
    if 'Total_Income' in df.columns:
        df.drop('Total_Income', axis=1, inplace=True)
    return df

def encode_features(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df['Property_Urban'] = (df['Property_Area'] == 'Urban').astype(int)
    df['Property_Semiurban'] = (df['Property_Area'] == 'Semiurban').astype(int)
    df['Property_Rural'] = (df['Property_Area'] == 'Rural').astype(int)
    df.drop('Property_Area', axis=1, inplace=True)
    return df

def impute_missing_values(df):
    df = df.copy()
    num_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df, imputer

def scale_features(X_train, X_test, feature_names):
    cols_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Dependents']
    cols_to_scale = [c for c in cols_to_scale if c in feature_names]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    return X_train_scaled, X_test_scaled, scaler

def preprocess_for_training(filepath):
    from sklearn.model_selection import train_test_split
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)
    df, imputer = impute_missing_values(df)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test, scaler = scale_features(X_train, X_test, feature_names)
    return X_train, X_test, y_train, y_test, imputer, scaler, feature_names