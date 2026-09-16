import pandas as pd
import numpy as np
import optuna
import joblib
import json
import logging
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
import traceback
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURES_FILE = "deap_features_task3.parquet"

# The 8 universally optimal channels from the paper
OPTIMAL_CHANNELS = ["CH1_", "CH2_", "CH4_", "CH5_", "CH12_", "CH17_", "CH18_", "CH21_"]

def load_data():
    logger.info("Loading dataset...")
    df = pd.read_parquet(FEATURES_FILE)
    groups = df['subject'].values
    y = df['discrete_target'].values
    X_df = df.drop(columns=['subject', 'trial', 'segment', 'discrete_target', 'continuous_target'])
    return X_df, y, groups

def filter_to_8_channels(X_df):
    cols_to_keep = [c for c in X_df.columns if any(c.startswith(ch) for ch in OPTIMAL_CHANNELS)]
    return X_df[cols_to_keep]

def objective(trial, X_df, y, groups):
    try:
        # 1. Feature Space Pruning
        use_8_channels = trial.suggest_categorical("use_8_channels", [True, False])
        if use_8_channels:
            X_curr = filter_to_8_channels(X_df).values
        else:
            X_curr = X_df.values
            
        # 2. Model selection (Restricted to GPU XGBoost to maintain 100% utilization)
        model_type = "XGBoost"
        
        # 3. Cross-Validation (GroupKFold to prevent data leakage across subjects)
        gkf = GroupKFold(n_splits=5)
        f1_scores = []
        
        for train_idx, val_idx in gkf.split(X_curr, y, groups=groups):
            X_train, X_val = X_curr[train_idx], X_curr[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            # Hyperparameters
            params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 400),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 9),
                "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "device": "cuda",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBClassifier(**params)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            # Use macro F1 to handle any potential class imbalance
            f1_scores.append(f1_score(y_val, y_pred, average='macro'))
            
        mean_f1 = float(np.mean(f1_scores))
        
        # Log to JSONL
        trial_info = {
            "trial_number": trial.number,
            "model_type": model_type,
            "use_8_channels": use_8_channels,
            "params": trial.params,
            "mean_f1": mean_f1
        }
        with open("trials_log.jsonl", "a") as f:
            f.write(json.dumps(trial_info) + "\n")
            
        return mean_f1
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with exception: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a bad score instead of crashing
        return 0.0

def main():
    X_df, y, groups = load_data()
    
    study_name = "deap_overnight_optimization"
    storage_name = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    def save_best_model(study, trial):
        if study.best_trial.number == trial.number:
            logger.info(f"New best F1 score: {trial.value:.4f}! Saving configuration...")
            joblib.dump(trial.params, "best_params.pkl")
            
    logger.info("Starting autonomous overnight optimization loop...")
    
    # Run 100 trials
    study.optimize(
        lambda t: objective(t, X_df, y, groups), 
        n_trials=100, 
        callbacks=[save_best_model],
        catch=(Exception,)
    )
    
    logger.info(f"Optimization complete. Best F1: {study.best_value:.4f}")
    
if __name__ == "__main__":
    main()
