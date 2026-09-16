import os
import glob
import pickle
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from feature_extraction import extract_features_1d
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import joblib

DATA_DIR = r"C:\Users\dhyan\Downloads\archive\deap-dataset\data_preprocessed_python"
FEATURES_FILE = "deap_features.parquet"

def get_discrete_stress(valence, arousal):
    """
    Returns 1 for STRESS, 0 for CALM, -1 for OTHER (to be dropped)
    Based on Hag et al. 2023 Equation 1
    """
    if valence < 3.0 and arousal > 5.0:
        return 1
    elif 4.0 < valence < 6.0 and arousal < 4.0:
        return 0
    else:
        return -1

def build_features():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    all_rows = []
    
    print(f"Processing {len(files)} DEAP subjects...")
    for fpath in tqdm(files):
        subj_id = os.path.basename(fpath).split('.')[0]
        with open(fpath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        # Extract EEG channels only (first 32) and remove 3s baseline (384 samples)
        # Resulting shape: (40 trials, 32 channels, 7680 samples)
        raw_eeg = data['data'][:, :32, 384:] 
        labels = data['labels'] # shape: (40, 4)
        
        for trial_idx in range(40):
            val = labels[trial_idx, 0]
            aro = labels[trial_idx, 1]
            target = get_discrete_stress(val, aro)
            
            # Skip trials that do not fall into our binary stress/calm classes
            if target == -1:
                continue
                
            trial_data = raw_eeg[trial_idx]
            
            # Segment the 60s trial into 16 windows of 4 seconds (480 samples)
            segments = np.split(trial_data, 16, axis=1) 
            
            for seg_idx, seg in enumerate(segments):
                row = {
                    'subject': subj_id,
                    'trial': trial_idx,
                    'segment': seg_idx,
                    'target': target
                }
                
                # Extract features for all 32 channels
                for ch in range(32):
                    ch_features = extract_features_1d(seg[ch], fs=128.0)
                    for k, v in ch_features.items():
                        row[f"CH{ch+1}_{k}"] = v
                        
                all_rows.append(row)
                
    df = pd.DataFrame(all_rows)
    df.to_parquet(FEATURES_FILE)
    print(f"Feature extraction complete! Saved to {FEATURES_FILE}. Shape: {df.shape}")
    return df

def train_model():
    if not os.path.exists(FEATURES_FILE):
        print("Feature file not found. Starting feature extraction (this may take ~20-30 mins)...")
        df = build_features()
    else:
        print(f"Loading existing features from {FEATURES_FILE}...")
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
    
    print("\nRunning Boruta Feature Selection...")
    print("This runs a Random Forest iteratively to drop useless features.")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(X_train, y_train)
    
    selected_indices = np.where(feat_selector.support_)[0]
    print(f"\n=> Boruta selected {len(selected_indices)} features out of {X.shape[1]}!")
    
    X_train_filtered = feat_selector.transform(X_train)
    X_test_filtered = feat_selector.transform(X_test)
    
    print("\nTraining final SVM classifier on the selected features...")
    # SVM parameters mirroring the paper's default setup
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    svm.fit(X_train_filtered, y_train)
    
    acc = svm.score(X_test_filtered, y_test)
    print(f"\n======================================")
    print(f"SVM Test Accuracy: {acc * 100:.2f}%")
    print(f"======================================")
    
    print("\nSaving models to disk...")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feat_selector, "boruta.pkl")
    joblib.dump(svm, "svm_model.pkl")
    print("Models saved successfully. Pipeline complete.")

if __name__ == '__main__':
    train_model()
