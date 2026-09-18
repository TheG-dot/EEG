# Handoff Report: DEAP Dataset Artifacts, Feature Files, Labels, and Filtering Requirements (R2 & R3)

**Author**: `teamwork_preview_explorer_survey_2` (Explorer)  
**Parent Agent**: `teamwork_preview_orchestrator_1` (`0136f96f-05fa-4a06-95df-f615e9916891`)  
**Workspace Path**: `C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction`  
**Report Path**: `C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction\.agents\teamwork_preview_explorer_survey_2\report.md`  
**Date**: 2026-09-18  

---

## 1. Observation

1. **Parquet Files In Workspace**:
   - `deap_features_replication.parquet`: 22,706,802 bytes, shape `(3664, 644)`. Columns: `['subject', 'trial', 'segment', 'target', 'CH1_line_length', ... 'CH32_katz_fd']`. Contains exactly 25 subjects (`s01, s02, s04, s05, s08, s10, s11, s12, s13, s14, s15, s16, s18, s19, s20, s21, s22, s24, s25, s26, s27, s28, s29, s31, s32`).
   - `deap_features.parquet`: 27,041,683 bytes, shape `(4320, 644)`. Contains all 32 subjects (4,320 rows = 270 trials $\times$ 16 segments).
   - `deap_features_task3.parquet`: 129,952,136 bytes, shape `(20480, 645)`. Contains all 32 subjects and all 40 trials without ambiguous filtering (20,480 rows = 1,280 trials $\times$ 16 segments).
   - Feature equivalence: `np.allclose(df_rep[feature_cols].values, df_all_filtered[feature_cols].values)` returned `True`. The feature values in `deap_features_replication.parquet` are identical to the valid subject subset in `deap_features.parquet`.

2. **Data Integrity & Feature Scale**:
   - Total NaNs: 0. Total Infs: 0. Zero-variance features: 0.
   - Dynamic range: minimum value $-3.627$, maximum value $68,437,068.216$. Relative band powers `rp_*` are bounded in $[0, 1]$, fractal dimension `katz_fd` in $[1.05, 1.98]$, while line length and wavelet energies are in $[10^1, 10^7]$.
   - Dtypes: 640 `float64` feature columns, 3 `int64` metadata columns (`trial`, `segment`, `target`), 1 `str` column (`subject`).

3. **Workspace Data Directory**:
   - `data/s01.dat`: 160,540,652 bytes. Inspection revealed `mean=0.000105, std=1.000192, min=-5.335, max=5.360`, generated synthetically by `archive/process_deap_targets.py` lines 123–130. It is a mock test artifact, NOT genuine DEAP EEG data.
   - Genuine DEAP dataset resides at: `C:\Users\dhyan\Downloads\archive\deap-dataset\data_preprocessed_python\*.dat`.

4. **DEAP Full Cohort Label Breakdown (32 Subjects, 1,280 Trials)**:
   - Evaluated using Hag et al. (2023) Equation 1:
     - Stress ($V < 3.0 \land A > 5.0$): **129 trials (10.08%)**
     - Calm ($4.0 < V < 6.0 \land A < 4.0$): **141 trials (11.02%)**
     - Ambiguous (Neither): **1,010 trials (78.91%)**

5. **Seven Dropped Subjects Breakdown**:
   - `s03`: 8 Calm, 0 Stress (100% Calm)
   - `s06`: 5 Calm, 0 Stress (100% Calm)
   - `s07`: 0 Calm, 1 Stress (100% Stress)
   - `s09`: 0 Calm, 3 Stress (100% Stress)
   - `s17`: 4 Calm, 0 Stress (100% Calm)
   - `s23`: 14 Calm, 0 Stress (100% Calm)
   - `s30`: 6 Calm, 0 Stress (100% Calm)
   - Total trials dropped from invalid subjects: 41 trials (37 Calm, 4 Stress) = 656 segments.

6. **Twenty-Five Valid Subjects Distribution**:
   - Total valid trials: **229 trials** (125 Stress [54.59%], 104 Calm [45.41%]).
   - Total segments: **3,664 segments** (2,000 Stress [54.59%], 1,664 Calm [45.41%]).
   - Segment counts per subject range from 32 (s26) to 272 (s22).
   - Minority class segment counts per subject:
     - s18: 16 Stress vs 48 Calm (min = 16)
     - s26: 16 Stress vs 16 Calm (min = 16)
     - s27: 16 Stress vs 128 Calm (min = 16)
     - All other 22 subjects have $\ge 32$ minority class segments.

7. **Per-Subject Baseline Benchmark (RLDA)**:
   - Executing isolated per-subject 10-fold Stratified CV with RLDA and StandardScaler:
     - Mean Accuracy: 77.15%, Mean Precision: 75.04%, Mean Recall: 75.05%.
     - Imbalanced subjects collapse: `s27` precision is 28.33%, recall is 30.00%; `s18` precision is 48.33%.

---

## 2. Logic Chain

1. **Equivalence & Selection of Source of Truth**:
   - Observations 1 and 2 prove that `deap_features_replication.parquet` contains pre-extracted, clean, zero-NaN features that exactly match `deap_features.parquet` after filtering. Thus, `deap_features_replication.parquet` is the optimal, complete single source of truth for R2 and R3.

2. **Necessity of Dropping Subjects 3, 6, 7, 9, 17, 23, 30**:
   - Observations 4 and 5 establish that each of these 7 subjects experienced only a single emotional state (either Calm only or Stress only).
   - In isolated per-subject evaluation, binary classifiers require samples from both positive and negative classes to establish a separating hyperplane. Furthermore, stratified cross-validation requires at least two classes to partition folds.
   - Therefore, dropping these 7 subjects is an unavoidable requirement of isolated per-subject evaluation.

3. **Confirmation of 25 Valid Subjects**:
   - Observation 6 proves that all remaining 25 subjects possess at least one trial in each class (e.g., s26 has 1 Stress and 1 Calm; s18 has 1 Stress and 3 Calm; s27 has 1 Stress and 8 Calm). Hence, exactly 25 valid subjects exist.

4. **10-Fold Stratified Cross-Validation Feasibility**:
   - From Observation 6, trial-level 10-fold CV is strictly impossible because no subject has $\ge 10$ trials in both classes (s26 has only 2 trials; s18 and s27 have only 1 Stress trial).
   - However, since each trial is windowed into 16 segments of 3.75s, segment-level sample counts are $16 \times \text{trial count}$.
   - The absolute minimum minority segment count across all 25 subjects is 16 (in s18, s26, s27).
   - Because $16 \ge 10$, a 10-fold stratified split allocates at least 1 minority class sample per fold across all 25 subjects (e.g., 6 folds get 2 samples and 4 folds get 1 sample). Thus, 10-fold Stratified CV is 100% mathematically feasible at the segment level.

5. **Readiness and Required Pipeline Steps**:
   - Observation 2 demonstrates an extreme feature scale discrepancy ($\Delta > 10^7$).
   - Observation 7 demonstrates that without class imbalance handling, subjects with skewed distributions (like s27 and s18) experience severe precision/recall collapse.
   - Therefore, while the parquet data is structurally complete, downstream models must incorporate fold-isolated scaling (`StandardScaler`) and class weighting (`class_weight='balanced'` / `scale_pos_weight`).

---

## 3. Caveats

1. **Intra-Trial Temporal Correlation**: Windowing trials into 16 segments and applying random StratifiedKFold means segments from the same 60s trial may appear in both train and test folds. This faithfully reproduces the methodology of Hag et al. (2023), but represents a known segment-level evaluation constraint rather than trial-level independence.
2. **Subject s26 Sample Size**: Subject `s26` has only 2 valid trials (32 segments total; 16 Stress and 16 Calm). While a 10-fold split is mathematically valid (1–2 samples per test fold), model generalization estimates for s26 have high fold-to-fold variance.
3. **Optuna and Model Files in Workspace**: Files like `best_params.pkl`, `svm_model.pkl`, and `boruta.pkl` in the workspace were trained on Task 3 data or global splits, and should NOT be reused for R2/R3 without retraining inside the 25-subject per-subject CV scheme.

---

## 4. Conclusion

1. **Source Data**: Downstream agents (Worker, Modeler, Evaluator) must load `deap_features_replication.parquet`. No re-extraction from raw EEG `.dat` files is necessary.
2. **Subject Cohort**: Strictly evaluate on the 25 confirmed valid subjects (`s01, s02, s04, s05, s08, s10, s11, s12, s13, s14, s15, s16, s18, s19, s20, s21, s22, s24, s25, s26, s27, s28, s29, s31, s32`).
3. **Filtering**: Explicitly exclude subjects `3, 6, 7, 9, 17, 23, 30` and drop ambiguous trials (already completed in `deap_features_replication.parquet`).
4. **Evaluation Setup**: Run Per-Subject 10-Fold Stratified Cross-Validation on the 3,664 segment instances.
5. **Preprocessing Mandate**: In each CV fold, apply `StandardScaler` strictly fitted on `X_train` and transform `X_test`. Apply class weighting (`class_weight='balanced'` or `scale_pos_weight`) to safeguard minority class precision and recall on imbalanced subjects like `s27`, `s18`, `s14`, and `s24`.

---

## 5. Verification Method

To independently verify all findings in this report, execute the following commands in the workspace environment:

1. **Verify Parquet Shape, Subject Count, and Zero Missing Values**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_parquet('deap_features_replication.parquet'); assert df.shape == (3664, 644); assert df['subject'].nunique() == 25; assert not df.isnull().values.any(); print('Dataset verification PASSED.')"
   ```

2. **Verify 25 Valid Subject IDs and Dropped Subject Exclusion**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_parquet('deap_features_replication.parquet'); dropped = {'s03', 's06', 's07', 's09', 's17', 's23', 's30'}; assert len(set(df['subject']).intersection(dropped)) == 0; assert len(df['subject'].unique()) == 25; print('Subject filtering verification PASSED.')"
   ```

3. **Verify Feasibility of 10-Fold Stratified CV Across All 25 Subjects**:
   ```powershell
   python -c "import pandas as pd, numpy as np; df = pd.read_parquet('deap_features_replication.parquet'); [assert min(np.bincount(df[df['subject']==s]['target'])) >= 10 for s in df['subject'].unique()]; print('10-Fold CV Feasibility verification PASSED for all 25 subjects.')"
   ```

4. **Verify Single-Class Status of Dropped Subjects**:
   ```powershell
   python -c "import pandas as pd; df = pd.read_parquet('deap_features.parquet'); dropped = ['s03', 's06', 's07', 's09', 's17', 's23', 's30']; [assert df[df['subject']==s]['target'].nunique() == 1 for s in dropped]; print('Dropped subject single-class verification PASSED.')"
   ```

**Invalidation Conditions**:
- If any of the 25 subjects has fewer than 10 segments in either class, the segment-level 10-fold CV claim is invalidated.
- If any of the 7 dropped subjects possesses dual-class trials under Hag et al. Eq. 1, the dropped-subject justification is invalidated.
