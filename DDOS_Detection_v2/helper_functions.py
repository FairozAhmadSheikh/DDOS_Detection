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
    display(df.head(n))