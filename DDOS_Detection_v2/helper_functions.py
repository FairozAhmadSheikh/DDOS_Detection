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

def scale_features(X_train, X_val=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler
    return X_train_scaled, scaler

def plot_class_balance(df, label_col='Label'):
    counts = df[label_col].value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title("Class distribution")
    plt.ylabel("Count")
    plt.xlabel("Label")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_corr_heatmap(df, numeric_only=True, top_k=40):
    numeric = df.select_dtypes(include=[np.number])
    if top_k and numeric.shape[1] > top_k:
        # pick top_k features by variance to keep plot readable
        variances = numeric.var().sort_values(ascending=False).head(top_k).index
        numeric = numeric[variances]
    corr = numeric.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='RdBu_r', center=0, square=True)
    plt.title("Feature correlation heatmap (subset)")
    plt.tight_layout()
    plt.show()

def plot_roc_pr(y_true, y_score, pos_label=1):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(rec, prec, label=f'PR AUC = {pr_auc:.4f}')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()