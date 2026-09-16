import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

FEATURES_FILE = "deap_features_task3.parquet"
REPORT_FILE = "Task3_Analysis.md"
OPTIMAL_CHANNELS = ["CH1_", "CH2_", "CH4_", "CH5_", "CH12_", "CH17_", "CH18_", "CH21_"]

def filter_to_8_channels(X_df):
    cols_to_keep = [c for c in X_df.columns if any(c.startswith(ch) for ch in OPTIMAL_CHANNELS)]
    return X_df[cols_to_keep]

def generate_analysis_and_regression():
    logging.info(f"Loading {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
    
    continuous_target = df['continuous_target']
    feature_cols = [c for c in df.columns if c.startswith('CH')]
    
    logging.info("Computing correlation matrix...")
    correlations = df[feature_cols].apply(lambda col: continuous_target.corr(col))
    top_correlations = correlations.abs().sort_values(ascending=False).head(20)
    
    # Write to Markdown
    with open(REPORT_FILE, "w") as f:
        f.write("# Task 3 Analysis: Continuous Stress Index\n\n")
        f.write("## Feature Correlation with Stress Index\n")
        f.write("The top 20 EEG features most strongly correlated with the continuous `(9 - Valence) + Arousal` Stress Index:\n\n")
        f.write("| Feature | Absolute Pearson Correlation |\n")
        f.write("|---------|------------------------------|\n")
        for feat, corr_val in top_correlations.items():
            f.write(f"| `{feat}` | {corr_val:.4f} |\n")
            
    logging.info("Starting XGBoost Regression Pipeline...")
    X_df = df[feature_cols]
    X_8_channels = filter_to_8_channels(X_df).values
    y = continuous_target.values
    groups = df['subject'].values
    
    gkf = GroupKFold(n_splits=5)
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    for train_idx, val_idx in gkf.split(X_8_channels, y, groups=groups):
        X_train, X_val = X_8_channels[train_idx], X_8_channels[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        regressor = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            device='cuda',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_val)
        
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        r2_scores.append(r2_score(y_val, y_pred))
        
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)
    mean_r2 = np.mean(r2_scores)
    
    with open(REPORT_FILE, "a") as f:
        f.write("\n## Regression Pipeline Performance\n")
        f.write("An `XGBRegressor` was trained on the 8 universally optimal channels to predict the continuous Stress Index using Subject-wise 5-Fold Cross Validation.\n\n")
        f.write("| Metric | Mean CV Score |\n")
        f.write("|--------|---------------|\n")
        f.write(f"| RMSE   | {mean_rmse:.4f} |\n")
        f.write(f"| MAE    | {mean_mae:.4f} |\n")
        f.write(f"| R^2    | {mean_r2:.4f} |\n")
        
    logging.info(f"Regression complete. RMSE: {mean_rmse:.4f}, R2: {mean_r2:.4f}")

if __name__ == "__main__":
    generate_analysis_and_regression()
