import pandas as pd
import numpy as np

df = pd.read_parquet("deap_features_replication.parquet")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

subjects = sorted(df['subject'].unique())
print(f"Number of subjects: {len(subjects)}")

stats = []
for s in subjects:
    sdf = df[df['subject'] == s]
    trials = sdf['trial'].unique()
    n_stress_trials = len(sdf[sdf['target'] == 1]['trial'].unique())
    n_calm_trials = len(sdf[sdf['target'] == 0]['trial'].unique())
    n_stress_segs = (sdf['target'] == 1).sum()
    n_calm_segs = (sdf['target'] == 0).sum()
    total_segs = len(sdf)
    stats.append({
        'subject': s,
        'n_trials': len(trials),
        'stress_trials': n_stress_trials,
        'calm_trials': n_calm_trials,
        'total_segs': total_segs,
        'stress_segs': n_stress_segs,
        'calm_segs': n_calm_segs,
        'min_class_segs': min(n_stress_segs, n_calm_segs)
    })

stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

print("\nSummary of segments per subject:")
print(stats_df[['total_segs', 'stress_segs', 'calm_segs', 'min_class_segs']].describe())
