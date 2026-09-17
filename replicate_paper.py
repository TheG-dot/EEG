import os
import glob
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from extract_features import extract_features_1d
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_DIR = r"C:\Users\dhyan\Downloads\archive\deap-dataset\data_preprocessed_python"
FEATURES_FILE = "deap_features_replication.parquet"
REPORT_FILE = "Replication_Benchmark.md"

# Subjects to DROP: 3, 6, 7, 9, 17, 23, 30
DROP_SUBJECTS = ['s03', 's06', 's07', 's09', 's17', 's23', 's30']

OPTIMAL_CHANNELS = ["CH1_", "CH2_", "CH4_", "CH5_", "CH12_", "CH17_", "CH18_", "CH21_"]

def process_subject(fpath):
    subj_id = os.path.basename(fpath).split('.')[0]
    if subj_id in DROP_SUBJECTS:
        return []
        
    with open(fpath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    raw_eeg = data['data'][:, :32, 384:] 
    labels = data['labels'] 
    
    subj_rows = []
    for trial_idx in range(40):
        val = labels[trial_idx, 0]
        aro = labels[trial_idx, 1]
        
        # Strict Replication Labels (Equation 1)
        if val < 3.0 and aro > 5.0:
            target = 1 # Stress
        elif 4.0 < val < 6.0 and aro < 4.0:
            target = 0 # Calm
        else:
            continue # Drop Ambiguous Trials
            
        trial_data = raw_eeg[trial_idx]
        segments = np.split(trial_data, 16, axis=1) 
        
        for seg_idx, seg in enumerate(segments):
            row = {
                'subject': subj_id,
                'trial': trial_idx,
                'segment': seg_idx,
                'target': target
            }
            
            for ch in range(32):
                ch_features = extract_features_1d(seg[ch], fs=128.0)
                for k, v in ch_features.items():
                    row[f"CH{ch+1}_{k}"] = v
                    
            subj_rows.append(row)
    return subj_rows

def build_features():
    if os.path.exists(FEATURES_FILE):
        logging.info("Loading existing replication dataset...")
        return pd.read_parquet(FEATURES_FILE)
        
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    logging.info("Extracting features strictly for 25 valid subjects...")
    results = Parallel(n_jobs=-1)(delayed(process_subject)(f) for f in files)
    
    all_rows = []
    for r in results:
        all_rows.extend(r)
                
    df = pd.DataFrame(all_rows)
    df.to_parquet(FEATURES_FILE)
    logging.info(f"Feature extraction complete! Shape: {df.shape}")
    return df

def run_evaluation(df):
    subjects = df['subject'].unique()
    logging.info(f"Running evaluation on {len(subjects)} subjects.")
    
    models = {
        "SVM": SVC(kernel='rbf', C=1.0, gamma=1.0e-3),
        "KNN": KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
        "RLDA": LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    }
    
    results = {
        "32_Channels": {m: {'acc': [], 'prec': [], 'rec': []} for m in models},
        "8_Channels": {m: {'acc': [], 'prec': [], 'rec': []} for m in models}
    }
    
    for subj in subjects:
        subj_df = df[df['subject'] == subj]
        y = subj_df['target'].values
        
        # Define channel sets
        all_cols = [c for c in subj_df.columns if c.startswith('CH')]
        opt_cols = [c for c in all_cols if any(c.startswith(ch) for ch in OPTIMAL_CHANNELS)]
        
        X_32 = subj_df[all_cols].values
        X_8 = subj_df[opt_cols].values
        
        n_splits = min(10, min(np.bincount(y)) if len(np.unique(y)) == 2 else 1)
        if n_splits < 2:
            logging.warning(f"Subject {subj} lacks dual-state samples for CV. Skipping.")
            continue
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        subj_metrics = {
            "32_Channels": {m: {'acc': [], 'prec': [], 'rec': []} for m in models},
            "8_Channels": {m: {'acc': [], 'prec': [], 'rec': []} for m in models}
        }
        
        for train_idx, test_idx in skf.split(X_32, y):
            y_train, y_test = y[train_idx], y[test_idx]
            
            for ch_name, X_full in [("32_Channels", X_32), ("8_Channels", X_8)]:
                X_train, X_test = X_full[train_idx], X_full[test_idx]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                for m_name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    subj_metrics[ch_name][m_name]['acc'].append(accuracy_score(y_test, y_pred))
                    subj_metrics[ch_name][m_name]['prec'].append(precision_score(y_test, y_pred, zero_division=0))
                    subj_metrics[ch_name][m_name]['rec'].append(recall_score(y_test, y_pred, zero_division=0))
                    
        for ch_name in results:
            for m_name in models:
                results[ch_name][m_name]['acc'].append(np.mean(subj_metrics[ch_name][m_name]['acc']))
                results[ch_name][m_name]['prec'].append(np.mean(subj_metrics[ch_name][m_name]['prec']))
                results[ch_name][m_name]['rec'].append(np.mean(subj_metrics[ch_name][m_name]['rec']))
                
    with open(REPORT_FILE, "w") as f:
        f.write("# Hag et al. (2023) Replication Benchmark\n\n")
        f.write("This report benchmarks the strictly replicated pipeline against the base paper's methodology.\n\n")
        
        f.write("## Configuration\n")
        f.write("- **Target Logic**: Stress (Valence < 3 & Arousal > 5) vs Calm (4 < Valence < 6 & Arousal < 4)\n")
        f.write("- **Subjects Dropped**: 3, 6, 7, 9, 17, 23, 30\n")
        f.write("- **Evaluation**: Per-Subject 10-Fold Stratified Cross-Validation (Averaged across 25 subjects)\n\n")
        
        for ch_name in ["32_Channels", "8_Channels"]:
            f.write(f"### {ch_name} Evaluation\n")
            f.write("| Model | Accuracy | Precision | Recall |\n")
            f.write("|-------|----------|-----------|--------|\n")
            
            for m_name in models:
                mean_acc = np.mean(results[ch_name][m_name]['acc']) * 100
                mean_prec = np.mean(results[ch_name][m_name]['prec']) * 100
                mean_rec = np.mean(results[ch_name][m_name]['rec']) * 100
                f.write(f"| {m_name} | {mean_acc:.2f}% | {mean_prec:.2f}% | {mean_rec:.2f}% |\n")
            f.write("\n")
            
    logging.info("Evaluation complete! Benchmark report saved.")

if __name__ == "__main__":
    df = build_features()
    run_evaluation(df)
