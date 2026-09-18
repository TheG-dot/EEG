# Original User Request

## 2026-09-18T19:36:56Z

# Teamwork Project Prompt

> Requested team: Full Team

Analyze the DEAP dataset replication pipeline to diagnose the accuracy gap compared to the Hag et al. (2023) paper baseline (~85%). Implement corrections and leverage advanced model architectures to exceed the paper's benchmark accuracy.

Working directory: `C:\Users\dhyan\.gemini\antigravity\scratch\deap_stress_prediction`
Integrity mode: demo

## Requirements

### R1. Diagnostics and Baseline Audit
Audit the existing `replicate_paper.py` script to identify any methodological flaws, data leakage, scaling issues, or feature engineering discrepancies causing the 8% accuracy gap compared to the published paper.

### R2. Advanced Model Optimization
Develop a new predictive pipeline utilizing advanced machine learning architectures (e.g., XGBoost, Deep Learning, Ensembles). You are permitted to use standard ML libraries, but you must strictly train and evaluate on the 25-subject filtered dataset where targets are explicitly defined as "Stress" (Valence < 3 & Arousal > 5) vs "Calm" (4 < Valence < 6 & Arousal < 4).

### R3. Robust Evaluation Strategy
Write an automated benchmarking script that calculates subject-averaged Accuracy, Precision, and Recall using Per-Subject 10-Fold Stratified Cross-Validation to rigorously prevent data leakage.

## Acceptance Criteria

### Performance Benchmarks
- [ ] The automated evaluation script successfully executes without errors.
- [ ] The model achieves a subject-averaged Accuracy strictly greater than **85.0%**.
- [ ] The model achieves a subject-averaged Precision and Recall strictly greater than **80.0%**.

### Methodological Integrity
- [ ] Evaluation metrics are computed using an isolated Per-Subject Cross-Validation approach.
- [ ] The data loading logic explicitly drops ambiguous trials and invalid subjects (3, 6, 7, 9, 17, 23, 30).
