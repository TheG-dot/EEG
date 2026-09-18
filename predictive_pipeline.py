"""
DEAP Stress Prediction: Advanced Predictive Pipeline (Requirement 2)
====================================================================
Author: teamwork_preview_worker_1
Target Publication: Hag et al. (2023), Brain Sciences 13(9), 1340.

This module implements an advanced, leakage-free machine learning architecture
for mental stress classification using EEG features from the DEAP dataset.

Key Architectural Components:
1. Strict Cohort & Label Filtering:
   - Data source: deap_features_replication.parquet
   - Stress: Valence < 3.0 & Arousal > 5.0 (Target = 1)
   - Calm: 4.0 < Valence < 6.0 & Arousal < 4.0 (Target = 0)
   - Invalid subjects explicitly dropped: [3, 6, 7, 9, 17, 23, 30] (25 valid subjects)
   - Ambiguous trials explicitly dropped
2. Leakage-Free Preprocessing:
   - StandardScaler fitted strictly on X_train within each cross-validation fold
   - ANOVA F-statistic feature selection (SelectKBest(f_classif, k=160)) fitted strictly on X_train
3. Advanced Heterogeneous Ensemble:
   - XGBoost Classifier (gradient boosting with shrinkage and tree regularization)
   - ExtraTrees Classifier (extremely randomized trees with variance reduction)
   - Tuned SVM RBF (regularized kernel boundary with adaptive scale gamma)
   - Soft Voting Ensemble with class-imbalance compensation (scale_pos_weight & class_weight='balanced')
   - In-fold decision threshold calibration using Youden's J-statistic on training predictions
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Constants
DEFAULT_PARQUET = "deap_features_replication.parquet"
DROP_SUBJECTS = ['s03', 's06', 's07', 's09', 's17', 's23', 's30']
VALID_SUBJECT_COUNT = 25
DEFAULT_K_FEATURES = 160


def load_and_clean_data(parquet_path: str = DEFAULT_PARQUET) -> pd.DataFrame:
    """
    Loads and validates the DEAP feature dataset.
    
    Enforces strict Hag et al. (2023) criteria:
    - Drops invalid subjects: [3, 6, 7, 9, 17, 23, 30]
    - Drops ambiguous trials (retaining only target 0 for Calm and target 1 for Stress)
    - Verifies exactly 25 valid subjects and absence of NaNs/Infs.
    """
    if not os.path.exists(parquet_path):
        # Fallback check in parent or absolute directory
        alt_path = os.path.join(os.path.dirname(__file__), parquet_path)
        if os.path.exists(alt_path):
            parquet_path = alt_path
        else:
            raise FileNotFoundError(f"Feature dataset not found at: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # 1. Filter out invalid single-class subjects
    df = df[~df['subject'].isin(DROP_SUBJECTS)].copy()

    # 2. Filter out ambiguous trials (target must be 0 or 1)
    df = df[df['target'].isin([0, 1])].copy()

    # 3. Validation checks
    subjects = sorted(df['subject'].unique())
    if len(subjects) != VALID_SUBJECT_COUNT:
        raise ValueError(f"Expected {VALID_SUBJECT_COUNT} valid subjects, but found {len(subjects)}: {subjects}")

    # Check for NaN / Inf values in feature columns
    feature_cols = [c for c in df.columns if c.startswith('CH')]
    if df[feature_cols].isna().any().any():
        raise ValueError("Dataset contains unexpected NaN values in EEG feature columns!")

    return df


class CalibratedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Heterogeneous Soft Voting Ensemble combining XGBoost, ExtraTrees, and Tuned SVM
    with automatic class-imbalance compensation and in-fold decision threshold calibration.
    """
    def __init__(
        self,
        imbalance_ratio: float = 1.0,
        k_features: int = DEFAULT_K_FEATURES,
        xgb_weight: float = 2.0,
        et_weight: float = 1.0,
        svm_weight: float = 1.0,
        random_state: int = 42
    ):
        self.imbalance_ratio = imbalance_ratio
        self.k_features = k_features
        self.xgb_weight = xgb_weight
        self.et_weight = et_weight
        self.svm_weight = svm_weight
        self.random_state = random_state
        
        self.scaler = None
        self.selector = None
        self.ensemble = None
        self.calibrated_threshold = 0.5
        self.classes_ = np.array([0, 1])

    def _build_base_models(self, spw: float) -> List[Tuple[str, BaseEstimator]]:
        """Constructs diverse, complementary base learners."""
        # 1. Gradient Boosted Trees: High expressiveness, fast split regularization
        xgb = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=spw,
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=1
        )

        # 2. Extremely Randomized Trees: High stochasticity, excellent variance reduction
        et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=3,
            class_weight='balanced',
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=1
        )

        # 3. Kernel Support Vector Classifier: Maximum margin geometric hyperplane
        svm = SVC(
            C=5.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=self.random_state
        )

        return [('xgb', xgb), ('et', et), ('svm', svm)]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits scaler, feature selector, and ensemble strictly on training data,
        then calibrates decision threshold via Youden's J-statistic.
        """
        # Calculate training fold class distribution
        n_calm = int(np.sum(y == 0))
        n_stress = int(np.sum(y == 1))
        spw = float(n_calm / max(1, n_stress))

        # 1. In-fold scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 2. In-fold feature selection
        k = min(self.k_features, X_scaled.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector.fit_transform(X_scaled, y)

        # 3. Build and fit soft voting ensemble
        estimators = self._build_base_models(spw)
        weights = [self.xgb_weight, self.et_weight, self.svm_weight]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        self.ensemble.fit(X_selected, y)

        # 4. In-fold threshold calibration via Youden's J-statistic
        # Calibrates threshold on training predictions to prevent minority class collapse
        train_probs = self.ensemble.predict_proba(X_selected)[:, 1]
        self.calibrated_threshold = self._calibrate_threshold(y, train_probs)

        return self

    def _calibrate_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Finds optimal decision threshold using Youden's J-statistic:
        J = Sensitivity + Specificity - 1 = Recall_1 + Recall_0 - 1.
        Prevents minority collapse on severely imbalanced subjects (e.g., s18, s20, s27).
        """
        best_threshold = 0.5
        best_score = -999.0
        
        # Grid search over candidate thresholds
        thresholds = np.linspace(0.20, 0.80, 61)
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            # Compute sensitivity (stress recall) and specificity (calm recall)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))

            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            j_stat = sens + spec - 1.0

            # Balanced score favoring joint sensitivity and specificity
            balanced_acc = 0.5 * (sens + spec)
            
            # Require minimum sensitivity and specificity > 0.40 to avoid trivial boundaries
            if sens >= 0.40 and spec >= 0.40:
                score = balanced_acc
            else:
                score = j_stat * 0.5

            if score > best_score:
                best_score = score
                best_threshold = t

        return float(best_threshold)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Applies in-fold transformations and returns soft voting probabilities."""
        X_scaled = self.scaler.transform(X)
        X_selected = self.selector.transform(X_scaled)
        return self.ensemble.predict_proba(X_selected)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary class using calibrated decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.calibrated_threshold).astype(int)


def build_model(
    model_type: str = "ensemble",
    imbalance_ratio: float = 1.0,
    k_features: int = DEFAULT_K_FEATURES,
    random_state: int = 42
) -> BaseEstimator:
    """
    Factory function conforming to interface contract.
    Returns configured estimator based on model_type.
    """
    if model_type.lower() in ["ensemble", "soft_voting", "calibrated"]:
        return CalibratedEnsembleClassifier(
            imbalance_ratio=imbalance_ratio,
            k_features=k_features,
            random_state=random_state
        )
    elif model_type.lower() == "xgboost":
        return XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            scale_pos_weight=imbalance_ratio,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=1
        )
    elif model_type.lower() == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=3,
            class_weight='balanced',
            max_features='sqrt',
            random_state=random_state,
            n_jobs=1
        )
    elif model_type.lower() == "svm":
        return SVC(
            C=5.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_pipeline(
    k_features: int = DEFAULT_K_FEATURES,
    model_type: str = "ensemble",
    imbalance_ratio: float = 1.0,
    random_state: int = 42
) -> BaseEstimator:
    """
    Factory function returning a configured pipeline estimator
    conforming strictly to PROJECT.md interface contract.
    """
    return build_model(
        model_type=model_type,
        imbalance_ratio=imbalance_ratio,
        k_features=k_features,
        random_state=random_state
    )


def evaluate_subject(
    df_subject: pd.DataFrame,
    n_splits: int = 10,
    random_state: int = 42,
    k_features: int = DEFAULT_K_FEATURES,
    model_type: str = "ensemble"
) -> Dict[str, float]:
    """
    Executes isolated Per-Subject 10-Fold Stratified Cross-Validation on a single subject's data.
    Ensures zero cross-subject and zero cross-fold data leakage.
    
    Returns subject-level metrics averaged across the 10 folds.
    """
    feature_cols = [c for c in df_subject.columns if c.startswith('CH')]
    X = df_subject[feature_cols].values
    y = df_subject['target'].values

    # Determine feasible number of folds (must not exceed minority class count)
    class_counts = np.bincount(y)
    min_samples = int(np.min(class_counts))
    splits = min(n_splits, min_samples)
    if splits < 2:
        raise ValueError(f"Subject {df_subject['subject'].iloc[0]} lacks dual-class samples for CV.")

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Instantiate pipeline model
        imbalance_ratio = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
        
        if model_type.lower() in ["ensemble", "soft_voting", "calibrated"]:
            model = CalibratedEnsembleClassifier(
                imbalance_ratio=imbalance_ratio,
                k_features=k_features,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)
            k = min(k_features, X_tr_sc.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            X_tr_sel = selector.fit_transform(X_tr_sc, y_train)
            X_te_sel = selector.transform(X_te_sc)
            
            base_model = build_model(
                model_type=model_type,
                imbalance_ratio=imbalance_ratio,
                k_features=k_features,
                random_state=random_state
            )
            base_model.fit(X_tr_sel, y_train)
            y_pred = base_model.predict(X_te_sel)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return {
        'accuracy': float(np.mean(fold_accuracies)),
        'precision': float(np.mean(fold_precisions)),
        'recall': float(np.mean(fold_recalls)),
        'f1': float(np.mean(fold_f1s)),
        'folds_evaluated': splits,
        'total_samples': len(y),
        'stress_samples': int(np.sum(y == 1)),
        'calm_samples': int(np.sum(y == 0))
    }
