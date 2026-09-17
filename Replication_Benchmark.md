# Hag et al. (2023) Replication Benchmark

This report benchmarks the strictly replicated pipeline against the base paper's methodology.

## Configuration
- **Target Logic**: Stress (Valence < 3 & Arousal > 5) vs Calm (4 < Valence < 6 & Arousal < 4)
- **Subjects Dropped**: 3, 6, 7, 9, 17, 23, 30
- **Evaluation**: Per-Subject 10-Fold Stratified Cross-Validation (Averaged across 25 subjects)

### 32_Channels Evaluation
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| SVM | 75.13% | 64.17% | 64.77% |
| KNN | 72.25% | 65.69% | 59.25% |
| RLDA | 77.15% | 75.04% | 75.05% |

### 8_Channels Evaluation
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| SVM | 71.46% | 55.43% | 59.91% |
| KNN | 71.02% | 64.19% | 58.97% |
| RLDA | 74.91% | 71.27% | 72.24% |

