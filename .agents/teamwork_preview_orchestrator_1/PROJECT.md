# Project: DEAP Dataset Stress Prediction Pipeline Optimization

## Architecture
- **Input Data**: `deap_features_replication.parquet` (pre-extracted 640 features: 32 channels × 20 multi-domain features; 229 trials; 3,664 segments across 25 valid subjects).
- **Target Definition**:
  - Stress: Valence < 3.0 and Arousal > 5.0 (Target = 1)
  - Calm: 4.0 < Valence < 6.0 and Arousal < 4.0 (Target = 0)
  - Ambiguous trials: Dropped
  - Invalid subjects: Explicitly dropped subjects [3, 6, 7, 9, 17, 23, 30]
  - Valid cohort: Exactly 25 subjects
- **Evaluation Scheme**: Isolated Per-Subject 10-Fold Stratified Cross-Validation.
- **Preprocessing Pipeline**: In-fold `StandardScaler` + in-fold feature selection (`SelectKBest(f_classif, k=160)`). All transformations strictly fit only on `X_train` per fold to ensure zero data leakage.
- **Model Family**: Advanced Tree Ensembles (GPU/Hist XGBoost), Regularized SVM RBF (`gamma='scale'`, class-weighted), ExtraTrees, and Soft Voting Ensembles with class-imbalance compensation.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Paper Replication Audit (R1) | Forensic audit of replicate_paper.py diagnosing 8% gap vs Hag et al. (2023) | M1 | Survey |
| 2 | Codebase Leakage & Methodological Audit (R1) | Audit of archive/ scripts, boruta.pkl, scaler.pkl, and 01-03 pipeline scripts | M1 | Survey |
| 3 | Subject & Ambiguous Trial Filtering (R2/R3) | Explicitly drop subjects 3,6,7,9,17,23,30 and ambiguous trials; 25 subjects | M2, M3 | ORIGINAL_REQUEST |
| 4 | In-Fold Leakage-Free Preprocessing (R2/R3) | StandardScaler and SelectKBest strictly fit on train folds | M2, M3 | Survey |
| 5 | Class-Imbalance Compensation (R2) | scale_pos_weight in XGBoost, class_weight='balanced' in SVM/ExtraTrees | M2 | Survey |
| 6 | Advanced Model Architecture (R2) | XGBoost + ExtraTrees + SVM Soft Voting Ensemble exceeding benchmarks | M2 | ORIGINAL_REQUEST |
| 7 | Automated Benchmarking Script (R3) | evaluate_benchmark.py computing 25-subject averaged Acc, Prec, Rec via 10-fold stratified CV | M3 | ORIGINAL_REQUEST |
| 8 | Benchmark Target Achievement | Subject-averaged Accuracy > 85.0%, Precision > 80.0%, Recall > 80.0% | M2, M3 | ORIGINAL_REQUEST |
| 9 | Forensic Integrity Audit & Stress Testing | Verification by Reviewer, Challenger, and Forensic Auditor | M4 | System Spec |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 1 | Diagnostics & Baseline Audit Report (R1) | Synthesize comprehensive audit document `Diagnostics_and_Baseline_Audit.md` | Survey | IN_PROGRESS |
| 2 | Advanced Predictive Pipeline (R2) | Implement `predictive_pipeline.py` with advanced models, in-fold preprocessing, and class weighting | M1 | PLANNED |
| 3 | Automated Benchmarking Script (R3) | Implement `evaluate_benchmark.py` running isolated 10-fold stratified CV per subject, reporting metrics | M2 | PLANNED |
| 4 | Verification, Challenger Stress Testing & Forensic Audit | Reviewers, Challengers, and Forensic Auditor execution | M3 | PLANNED |

## Interface Contracts
### `predictive_pipeline.py` ↔ `evaluate_benchmark.py`
- `build_model(model_type, imbalance_ratio, random_state)` -> returns configured scikit-learn compatible estimator (e.g., XGBoost, VotingClassifier).
- `get_pipeline(k_features, model_type, imbalance_ratio, random_state)` -> returns Pipeline or step-by-step fit/predict callable.
- `evaluate_subject(df_subject, n_splits=10, random_state=42)` -> returns dict `{'accuracy': float, 'precision': float, 'recall': float}`.

## Code Layout
- `Diagnostics_and_Baseline_Audit.md`: R1 comprehensive audit report at project root.
- `predictive_pipeline.py`: R2 model definitions, pipeline builder, and training logic.
- `evaluate_benchmark.py`: R3 automated benchmarking script executing per-subject 10-fold stratified CV across 25 subjects and displaying final metrics.
- `deap_features_replication.parquet`: 25-subject clean dataset.
