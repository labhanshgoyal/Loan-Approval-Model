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
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
    if 'Total_Income' in df.columns:
        df.drop('Total_Income', axis=1, inplace=True)
    return df
