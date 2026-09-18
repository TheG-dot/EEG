import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_parquet("deap_features_replication.parquet")
subjects = sorted(df["subject"].unique())
print(f"Loaded {len(df)} rows across {len(subjects)} subjects.")

OPTIMAL_CHANNELS = ["CH1_", "CH2_", "CH4_", "CH5_", "CH12_", "CH17_", "CH18_", "CH21_"]
all_feat_cols = [c for c in df.columns if c.startswith("CH")]
opt_feat_cols = [c for c in all_feat_cols if any(c.startswith(ch) for ch in OPTIMAL_CHANNELS)]

print(f"Total features: {len(all_feat_cols)}, 8-channel features: {len(opt_feat_cols)}")

def evaluate_pipeline(scaler_cls, feature_mode, model_factory, k_best=None):
    subj_accs, subj_precs, subj_recs = [], [], []
    
    for s in subjects:
        sdf = df[df["subject"] == s]
        y = sdf["target"].values
        
        if feature_mode == "8ch":
            X = sdf[opt_feat_cols].values
        else:
            X = sdf[all_feat_cols].values
            
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_accs, fold_precs, fold_recs = [], [], []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. Scale strictly on train
            scaler = scaler_cls()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # 2. Feature selection strictly on train
            if k_best is not None and k_best < X_train.shape[1]:
                selector = SelectKBest(score_func=f_classif, k=k_best)
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)
                
            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_precs.append(precision_score(y_test, y_pred, zero_division=0))
            fold_recs.append(recall_score(y_test, y_pred, zero_division=0))
            
        subj_accs.append(np.mean(fold_accs))
        subj_precs.append(np.mean(fold_precs))
        subj_recs.append(np.mean(fold_recs))
        
    return np.mean(subj_accs) * 100, np.mean(subj_precs) * 100, np.mean(subj_recs) * 100

configs = [
    # Baseline SVM replicate_paper.py
    ("replicate_paper SVM (gamma=1e-3, C=1.0)", StandardScaler, "32ch", None, lambda: SVC(kernel='rbf', C=1.0, gamma=1e-3)),
    ("SVM (gamma='scale', C=1.0)", StandardScaler, "32ch", None, lambda: SVC(kernel='rbf', C=1.0, gamma='scale')),
    ("SVM (gamma='scale', C=5.0)", StandardScaler, "32ch", None, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    ("SVM (gamma='scale', C=10.0)", StandardScaler, "32ch", None, lambda: SVC(kernel='rbf', C=10.0, gamma='scale')),
    ("SVM RobustScaler (gamma='scale', C=5.0)", RobustScaler, "32ch", None, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    ("SVM RobustScaler (gamma='scale', C=10.0)", RobustScaler, "32ch", None, lambda: SVC(kernel='rbf', C=10.0, gamma='scale')),
    
    # 8 Channels SVM
    ("8ch SVM (gamma='scale', C=5.0)", StandardScaler, "8ch", None, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    ("8ch SVM RobustScaler (gamma='scale', C=5.0)", RobustScaler, "8ch", None, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    
    # Feature Selection + SVM
    ("SelectKBest(k=100) + SVM C=5.0", StandardScaler, "32ch", 100, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    ("SelectKBest(k=150) + SVM C=5.0", StandardScaler, "32ch", 150, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    ("SelectKBest(k=200) + SVM C=5.0", StandardScaler, "32ch", 200, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    ("SelectKBest(k=100) RobustScaler + SVM C=5.0", RobustScaler, "32ch", 100, lambda: SVC(kernel='rbf', C=5.0, gamma='scale')),
    
    # Random Forest & Extra Trees
    ("RandomForest(n=200)", StandardScaler, "32ch", None, lambda: RandomForestClassifier(n_estimators=200, random_state=42)),
    ("ExtraTrees(n=200)", StandardScaler, "32ch", None, lambda: ExtraTreesClassifier(n_estimators=200, random_state=42)),
    ("ExtraTrees(n=300, max_depth=15)", StandardScaler, "32ch", None, lambda: ExtraTreesClassifier(n_estimators=300, max_depth=15, random_state=42)),
    ("8ch ExtraTrees(n=200)", StandardScaler, "8ch", None, lambda: ExtraTreesClassifier(n_estimators=200, random_state=42)),
    ("SelectKBest(k=150) + ExtraTrees(n=200)", StandardScaler, "32ch", 150, lambda: ExtraTreesClassifier(n_estimators=200, random_state=42)),

    # XGBoost
    ("XGBoost(n=150, max_depth=4, lr=0.05)", StandardScaler, "32ch", None, lambda: XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', n_jobs=2)),
    ("XGBoost(n=200, max_depth=5, lr=0.05, subsample=0.8)", StandardScaler, "32ch", None, lambda: XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=2)),
    ("SelectKBest(k=150) + XGBoost", StandardScaler, "32ch", 150, lambda: XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', n_jobs=2)),

    # HistGradientBoosting
    ("HistGradientBoosting()", StandardScaler, "32ch", None, lambda: HistGradientBoostingClassifier(random_state=42, max_iter=150)),

    # MLP
    ("MLP(hidden=(128, 64), alpha=0.01)", StandardScaler, "32ch", None, lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=200, random_state=42)),
    ("SelectKBest(k=150) + MLP", StandardScaler, "32ch", 150, lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=200, random_state=42)),
]

print("\n--- RUNNING BENCHMARKS ---")
for name, scaler, fmode, k_b, m_fact in configs:
    acc, prec, rec = evaluate_pipeline(scaler, fmode, m_fact, k_b)
    pass_bench = "PASS [?]" if (acc > 85.0 and prec > 80.0 and rec > 80.0) else "FAIL [X]"
    print(f"{name:50s} | Acc: {acc:6.2f}% | Prec: {prec:6.2f}% | Rec: {rec:6.2f}% | {pass_bench}")
