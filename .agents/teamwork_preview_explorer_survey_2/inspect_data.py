import os
import glob
import pickle
import numpy as np
import pandas as pd

def check_raw_deap():
    raw_dir = r"C:\Users\dhyan\Downloads\archive\deap-dataset\data_preprocessed_python"
    print("=== CHECKING RAW DEAP DIRECTORY ===")
    if os.path.exists(raw_dir):
        files = sorted(glob.glob(os.path.join(raw_dir, "*.dat")))
        print(f"Found {len(files)} files in {raw_dir}:")
        print([os.path.basename(f) for f in files])
        return files
    else:
        print(f"Raw directory NOT FOUND: {raw_dir}")
        return []

def check_data_folder():
    print("\n=== CHECKING workspace/data FOLDER ===")
    data_dir = r"C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction\data"
    files = glob.glob(os.path.join(data_dir, "*"))
    print(f"Files in {data_dir}: {files}")
    for f in files:
        sz = os.path.getsize(f)
        print(f"  {f}: {sz} bytes")
        if f.endswith(".dat"):
            try:
                with open(f, "rb") as fp:
                    d = pickle.load(fp, encoding="latin1")
                    print(f"  keys: {d.keys()}")
                    if "data" in d:
                        print(f"  data shape: {d['data'].shape}, dtype: {d['data'].dtype}")
                    if "labels" in d:
                        print(f"  labels shape: {d['labels'].shape}, dtype: {d['labels'].dtype}")
                        print(f"  first 3 labels:\n{d['labels'][:3]}")
            except Exception as e:
                print(f"  Error reading {f}: {e}")

def inspect_parquet(fpath):
    print(f"\n=======================================================")
    print(f"INSPECTING: {os.path.basename(fpath)}")
    print(f"Path: {fpath}")
    if not os.path.exists(fpath):
        print("FILE DOES NOT EXIST!")
        return
    sz = os.path.getsize(fpath)
    print(f"Size: {sz / (1024*1024):.2f} MB ({sz} bytes)")
    
    df = pd.read_parquet(fpath)
    print(f"Shape: {df.shape} (rows, cols)")
    print(f"Columns ({len(df.columns)}):")
    meta_cols = [c for c in df.columns if not c.startswith("CH")]
    feature_cols = [c for c in df.columns if c.startswith("CH")]
    print(f"  Non-CH (metadata/target) cols ({len(meta_cols)}): {meta_cols}")
    print(f"  CH feature cols ({len(feature_cols)}): total {len(feature_cols)}")
    print(f"  First 10 feature cols: {feature_cols[:10]}")
    print(f"  Last 5 feature cols: {feature_cols[-5:]}")
    
    # Metadata summary
    for c in meta_cols:
        val_counts = df[c].value_counts(dropna=False)
        nunique = df[c].nunique(dropna=False)
        if nunique <= 10:
            print(f"  Column '{c}' unique values:\n{val_counts}")
        else:
            print(f"  Column '{c}' nunique: {nunique}, sample: {df[c].unique()[:5]}")
            
    if 'subject' in df.columns:
        subjects = sorted(df['subject'].unique())
        print(f"Unique subjects ({len(subjects)}): {subjects}")
        print("Rows per subject:")
        print(df['subject'].value_counts().sort_index())
        
        if 'target' in df.columns:
            print("\nTarget counts per subject:")
            ct = pd.crosstab(df['subject'], df['target'])
            print(ct)
        elif 'discrete_target' in df.columns:
            print("\nDiscrete target counts per subject:")
            ct = pd.crosstab(df['subject'], df['discrete_target'])
            print(ct)
            
        if 'trial' in df.columns:
            print("\nTrials count per subject:")
            trials_per_subj = df.groupby('subject')['trial'].nunique()
            print(trials_per_subj)
            
        if 'segment' in df.columns:
            print("\nSegments distribution:")
            print(df['segment'].value_counts().sort_index())
            
    # Check nulls / infs in features
    print("\nChecking for NaNs and Infs in features...")
    sample_features = df[feature_cols]
    nans = sample_features.isna().sum().sum()
    print(f"Total NaNs across all features: {nans}")
    infs = np.isinf(sample_features.values).sum()
    print(f"Total Infs across all features: {infs}")
    
    # Check feature summary stats (min, max, mean, std across some features)
    print("\nFeature statistics sample (first 5 features):")
    print(sample_features[feature_cols[:5]].describe().T[['mean', 'std', 'min', 'max']])

if __name__ == "__main__":
    check_raw_deap()
    check_data_folder()
    
    workspace = r"C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction"
    parquets = [
        "deap_features_replication.parquet",
        "deap_features.parquet",
        "deap_features_task3.parquet"
    ]
    for p in parquets:
        inspect_parquet(os.path.join(workspace, p))
