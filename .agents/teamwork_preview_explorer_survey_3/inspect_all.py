import pickle
import joblib
import json
import os
import numpy as np
import pandas as pd

print("=== CHECKING BEST_PARAMS.PKL ===")
try:
    with open("best_params.pkl", "rb") as f:
        bp = pickle.load(f)
    print("best_params.pkl:", bp)
except Exception as e:
    print("Error best_params:", e)

print("\n=== CHECKING BORUTA.PKL ===")
try:
    with open("boruta.pkl", "rb") as f:
        boruta = pickle.load(f)
    print("boruta type:", type(boruta))
    if hasattr(boruta, 'support_'):
        print("boruta n_features:", len(boruta.support_))
        print("boruta selected count:", sum(boruta.support_))
        print("boruta ranking min/max:", min(boruta.ranking_), max(boruta.ranking_))
    if hasattr(boruta, '__dict__'):
        print("boruta keys:", list(boruta.__dict__.keys()))
except Exception as e:
    print("Error boruta:", e)

print("\n=== CHECKING SCALER.PKL ===")
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("scaler type:", type(scaler))
    if hasattr(scaler, 'mean_'):
        print("scaler mean shape:", scaler.mean_.shape)
        print("scaler var shape:", scaler.var_.shape)
except Exception as e:
    print("Error scaler:", e)

print("\n=== CHECKING SVM_MODEL.PKL ===")
try:
    with open("svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    print("svm type:", type(svm))
    print("svm params:", svm.get_params() if hasattr(svm, 'get_params') else dir(svm))
except Exception as e:
    print("Error svm:", e)

print("\n=== CHECKING PARQUET FILES ===")
for p in ["deap_features_replication.parquet", "deap_features.parquet", "deap_features_task3.parquet"]:
    if os.path.exists(p):
        df = pd.read_parquet(p)
        print(f"\nFile: {p}")
        print(f"Shape: {df.shape}")
        print("Columns first 10:", list(df.columns[:10]))
        print("Target columns:", [c for c in df.columns if 'target' in c or 'label' in c])
        if 'subject' in df.columns:
            print("Unique subjects:", sorted(df['subject'].unique()))
            print("Number of subjects:", len(df['subject'].unique()))
        if 'target' in df.columns:
            print("Target counts:\n", df['target'].value_counts())
        if 'discrete_target' in df.columns:
            print("Discrete target counts:\n", df['discrete_target'].value_counts())
