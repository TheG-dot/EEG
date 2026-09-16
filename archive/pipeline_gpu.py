import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from boruta import BorutaPy
import joblib

FEATURES_FILE = "deap_features.parquet"

def train_model():
    print(f"Loading features from {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
        
    print(f"Total dataset shape: {df.shape} ({df.shape[0]} windows, {df.shape[1]-4} features)")
    print(f"Class balance:\n{df['target'].value_counts()}")
    
    # Drop metadata columns to form our feature matrix X
    X = df.drop(columns=['subject', 'trial', 'segment', 'target']).values
    y = df['target'].values
    
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Splitting into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("\nRunning GPU-Accelerated Boruta Feature Selection...")
    print("Using XGBoost on your RTX 5050 to dramatically speed up Boruta.")
    start = time.time()
    
    xgb_model = XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        device='cuda',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    # BorutaPy expects the estimator to have fit(X, y), predict(X), etc., which XGBoost does.
    feat_selector = BorutaPy(xgb_model, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(X_train, y_train)
    
    end = time.time()
    print(f"Boruta took {end - start:.2f} seconds on GPU.")
    
    selected_indices = np.where(feat_selector.support_)[0]
    print(f"\n=> Boruta selected {len(selected_indices)} features out of {X.shape[1]}!")
    
    X_train_filtered = feat_selector.transform(X_train)
    X_test_filtered = feat_selector.transform(X_test)
    
    # If it selected no features, default to all (failsafe)
    if len(selected_indices) == 0:
        print("Boruta selected 0 features, using all features instead for fallback.")
        X_train_filtered = X_train
        X_test_filtered = X_test
    
    print("\nTraining final models on the selected features...")
    
    # 1. XGBoost (GPU)
    final_xgb = XGBClassifier(
        n_estimators=200,
        device='cuda',        
        tree_method='hist',
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    final_xgb.fit(X_train_filtered, y_train)
    acc_xgb = final_xgb.score(X_test_filtered, y_test)
    
    # 2. SVM (CPU) - to strictly match the paper
    print("Training SVM (CPU) for paper replication...")
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    svm.fit(X_train_filtered, y_train)
    acc_svm = svm.score(X_test_filtered, y_test)
    
    print(f"\n======================================")
    print(f"XGBoost (GPU) Test Accuracy : {acc_xgb * 100:.2f}%")
    print(f"SVM (Paper Replica) Accuracy: {acc_svm * 100:.2f}%")
    print(f"======================================")
    
    print("\nSaving models to disk...")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feat_selector, "boruta.pkl")
    joblib.dump(svm, "svm_model.pkl")
    final_xgb.save_model("xgboost_model.json")
    print("Models saved successfully. Pipeline complete.")

if __name__ == '__main__':
    train_model()
