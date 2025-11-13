import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             auc)
import joblib

def load_csvs_glob(paths):
    """Load and concatenate multiple CSV files into a single DataFrame."""
    dfs = []
    for p in paths:
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    return df

def inspect_df(df, n=5):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values per column (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    display(df.head(n)) # print in vs code
    
def basic_cleaning(df, drop_cols=None):
    """Strip column names, drop specified cols, drop duplicates."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.drop_duplicates()
    # replace inf/-inf and convert objects that look numeric
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def simple_impute_numeric(df, strategy='median'):
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            if strategy == 'median':
                val = df[col].median()
            elif strategy == 'mean':
                val = df[col].mean()
            else:
                val = 0
            df[col] = df[col].fillna(val)
    return df
def encode_labels(df, label_col='Label'):
    """Map multi-class labels to integers and return mapping."""
    le = LabelEncoder()
    df = df.copy()
    df[label_col] = le.fit_transform(df[label_col].astype(str))
    return df, le