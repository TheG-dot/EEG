import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from extract_features import extract_features_1d

from joblib import Parallel, delayed

DATA_DIR = r"C:\Users\dhyan\Downloads\archive\deap-dataset\data_preprocessed_python"
FEATURES_FILE = "deap_features_task3.parquet"

def process_subject(fpath):
    subj_id = os.path.basename(fpath).split('.')[0]
    with open(fpath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    raw_eeg = data['data'][:, :32, 384:] 
    labels = data['labels'] 
    
    subj_rows = []
    for trial_idx in range(40):
        val = labels[trial_idx, 0]
        aro = labels[trial_idx, 1]
        
        # Task 3 targets
        discrete_target = 1 if (val < 5.0 and aro > 5.0) else 0
        continuous_target = (9.0 - val) + aro
        
        trial_data = raw_eeg[trial_idx]
        segments = np.split(trial_data, 16, axis=1) 
        
        for seg_idx, seg in enumerate(segments):
            row = {
                'subject': subj_id,
                'trial': trial_idx,
                'segment': seg_idx,
                'discrete_target': discrete_target,
                'continuous_target': continuous_target
            }
            
            for ch in range(32):
                ch_features = extract_features_1d(seg[ch], fs=128.0)
                for k, v in ch_features.items():
                    row[f"CH{ch+1}_{k}"] = v
                    
            subj_rows.append(row)
    return subj_rows

def build_features():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    
    print(f"Processing {len(files)} DEAP subjects in parallel to max out CPU...")
    results = Parallel(n_jobs=-1)(delayed(process_subject)(f) for f in files)
    
    all_rows = []
    for r in results:
        all_rows.extend(r)
                
    df = pd.DataFrame(all_rows)
    df.to_parquet(FEATURES_FILE)
    print(f"Feature extraction complete! Saved to {FEATURES_FILE}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    if not os.path.exists(FEATURES_FILE):
        build_features()
    else:
        print(f"{FEATURES_FILE} already exists. Skipping rebuild.")
