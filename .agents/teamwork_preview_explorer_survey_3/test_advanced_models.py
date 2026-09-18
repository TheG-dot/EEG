import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_parquet("deap_features_replication.parquet")
subjects = sorted(df["subject"].unique())
all_feat_cols = [c for c in df.columns if c.startswith("CH")]

def evaluate_pipeline(config_fn):
    subj_accs, subj_precs, subj_recs = [], [], []
    
    for s in subjects:
        sdf = df[df["subject"] == s]
        y = sdf["target"].values
        X = sdf[all_feat_cols].values
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        fold_accs, fold_precs, fold_recs = [], [], []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model, k_best = config_fn(y_train)
            
            if k_best is not None:
                selector = SelectKBest(score_func=f_classif, k=k_best)
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_precs.append(precision_score(y_test, y_pred, zero_division=0))
            fold_recs.append(recall_score(y_test, y_pred, zero_division=0))
            
        subj_accs.append(np.mean(fold_accs))
        subj_precs.append(np.mean(fold_precs))
        subj_recs.append(np.mean(fold_recs))
        
    return np.mean(subj_accs) * 100, np.mean(subj_precs) * 100, np.mean(subj_recs) * 100

def get_scale_pos_weight(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / max(1, n_pos)

experiments = [
    # 1. XGBoost balanced
    ("XGBoost balanced (depth=3, n=150)", 
     lambda y_train: (XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, scale_pos_weight=get_scale_pos_weight(y_train), random_state=42, eval_metric='logloss', n_jobs=2), None)),
    
    # 2. XGBoost balanced (depth=4, colsample=0.7)
    ("XGBoost balanced (depth=4, colsample=0.7, n=200)", 
     lambda y_train: (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, colsample_bytree=0.7, subsample=0.8, scale_pos_weight=get_scale_pos_weight(y_train), random_state=42, eval_metric='logloss', n_jobs=2), None)),
     
    # 3. XGBoost balanced + SelectKBest 150
    ("SelectKBest(150) + XGBoost balanced (depth=4)", 
     lambda y_train: (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, colsample_bytree=0.7, subsample=0.8, scale_pos_weight=get_scale_pos_weight(y_train), random_state=42, eval_metric='logloss', n_jobs=2), 150)),

    # 4. ExtraTrees balanced
    ("ExtraTrees balanced (n=300, max_depth=12)", 
     lambda y_train: (ExtraTreesClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=2), None)),

    # 5. SelectKBest(150) + ExtraTrees balanced
    ("SelectKBest(150) + ExtraTrees balanced", 
     lambda y_train: (ExtraTreesClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=2), 150)),

    # 6. SVM balanced C=5.0
    ("SVM balanced C=5.0", 
     lambda y_train: (SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', random_state=42), None)),

    # 7. SelectKBest(200) + SVM balanced C=5.0
    ("SelectKBest(200) + SVM balanced C=5.0", 
     lambda y_train: (SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', random_state=42), 200)),

    # 8. Soft Voting: XGBoost + ExtraTrees + SVM
    ("Voting (XGBoost + ExtraTrees + SVM balanced)", 
     lambda y_train: (VotingClassifier(
         estimators=[
             ('xgb', XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=get_scale_pos_weight(y_train), random_state=42, eval_metric='logloss', n_jobs=2)),
             ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42, n_jobs=2)),
             ('svm', SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', probability=True, random_state=42))
         ],
         voting='soft'
     ), None)),

    # 9. SelectKBest(150) + Soft Voting (XGBoost + ExtraTrees + SVM)
    ("SelectKBest(150) + Voting (XGB + ET + SVM)", 
     lambda y_train: (VotingClassifier(
         estimators=[
             ('xgb', XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=get_scale_pos_weight(y_train), random_state=42, eval_metric='logloss', n_jobs=2)),
             ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42, n_jobs=2)),
             ('svm', SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', probability=True, random_state=42))
         ],
         voting='soft'
     ), 150)),

    # 10. Stacking: XGBoost + ExtraTrees -> LogisticRegression
    ("Stacking (XGBoost + ExtraTrees + SVM -> LR)", 
     lambda y_train: (StackingClassifier(
         estimators=[
             ('xgb', XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=get_scale_pos_weight(y_train), random_state=42, eval_metric='logloss', n_jobs=2)),
             ('et', ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42, n_jobs=2)),
             ('svm', SVC(kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', probability=True, random_state=42))
         ],
         final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
         cv=5
     ), 150))
]

print("\n--- RUNNING ADVANCED EXPERIMENTS ---")
for name, cfg_fn in experiments:
    acc, prec, rec = evaluate_pipeline(cfg_fn)
    pass_bench = "PASS [?]" if (acc > 85.0 and prec > 80.0 and rec > 80.0) else "FAIL [X]"
    print(f"{name:55s} | Acc: {acc:6.2f}% | Prec: {prec:6.2f}% | Rec: {rec:6.2f}% | {pass_bench}")
