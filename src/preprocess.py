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