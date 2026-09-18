# Dispatch Assignment

## 2026-09-19T01:07:35+05:30

You are the Project Orchestrator for the DEAP dataset stress prediction project.

Workspace directory: C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction
Working directory: C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction\.agents\teamwork_preview_orchestrator_1
Original request file: C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction\ORIGINAL_REQUEST.md

Mission:
Analyze the DEAP dataset replication pipeline to diagnose the accuracy gap compared to the Hag et al. (2023) paper baseline (~85%). Implement corrections and leverage advanced model architectures to exceed the paper's benchmark accuracy.

Requirements & Acceptance Criteria:
1. Diagnostics and Baseline Audit (R1): Audit existing replicate_paper.py to identify methodological flaws, data leakage, scaling issues, or feature engineering discrepancies causing the 8% accuracy gap.
2. Advanced Model Optimization (R2): Develop a new predictive pipeline utilizing advanced machine learning architectures (e.g. XGBoost, Deep Learning, Ensembles). Strictly train and evaluate on the 25-subject filtered dataset where targets are explicitly defined as "Stress" (Valence < 3 & Arousal > 5) vs "Calm" (4 < Valence < 6 & Arousal < 4). Explicitly drop ambiguous trials and invalid subjects (3, 6, 7, 9, 17, 23, 30).
3. Robust Evaluation Strategy (R3): Write an automated benchmarking script calculating subject-averaged Accuracy, Precision, and Recall using Per-Subject 10-Fold Stratified Cross-Validation to rigorously prevent data leakage.
4. Benchmarks to achieve:
   - Subject-averaged Accuracy strictly greater than 85.0%.
   - Subject-averaged Precision strictly greater than 80.0%.
   - Subject-averaged Recall strictly greater than 80.0%.
   - Automated evaluation script executes cleanly without errors.
   - Isolated Per-Subject Cross-Validation with explicit dropping of ambiguous trials and invalid subjects (3, 6, 7, 9, 17, 23, 30).

Orchestration Instructions:
- Decompose tasks, dispatch specialists, and manage team lifecycle under .agents/.
- Maintain your own BRIEFING.md and progress.md in your working directory. Keep them updated with regular progress.
- When all requirements and acceptance criteria are fully met and verified, message the Sentinel with your completion report and exact benchmark metrics.
