# Explainable and Trustworthy Multimodal Deep Learning for Predictive Maintenance of Industrial Assets

**Student:** Devaprasad P | 2023AA05069  
**Programme:** M.Tech Artificial Intelligence and Machine Learning — BITS Pilani WILP  
**Supervisor:** Dr. Ashok Veilumuthu  

---

## Overview

This dissertation investigates Remaining Useful Life (RUL) prediction for industrial assets using the NASA C-MAPSS FD001 benchmark. It compares raw sensor sequences, engineered degradation features, and a multi-view fusion of both representations under a leakage-safe, traceable, three-stage experimental protocol. The study extends predictive accuracy with explainability (feature importance, SHAP, view masking) and trustworthiness (noise robustness, lifecycle-region error, prediction bias, MC Dropout uncertainty).

**Central finding:** Engineered degradation features provide a strong and comparatively consistent representation. The implemented GRU-based multi-view fusion was competitive but did not consistently outperform the strongest derived-feature or classical alternative.

---

## Research Questions

| # | Research Question |
|---|---|
| RQ1 | How do raw sensor sequences, engineered features, and their fusion compare for capped RUL prediction? |
| RQ2 | Which features and input views are most influential, and what do explanations reveal? |
| RQ3 | How robust is the fusion model under noise, masking, and across lifecycle regions? |
| RQ4 | How stable are the findings across engine cohorts and on the held-out test set? |

---

## Dataset

**NASA C-MAPSS FD001** — single operating condition, single fault mode  
- Training set: 100 engines, complete run-to-failure histories  
- Test set: 100 engines, truncated histories with endpoint RUL labels  
- 14 sensors retained (7 near-constant sensors excluded)  
- Capped RUL target: min(RUL, 125)

---

## Models

| Model | Input | Architecture |
|---|---|---|
| XGBoost | Feature Set C (57 features) | 300 trees, depth 4 |
| GRU | 30-cycle × 14-sensor sequence | GRU(64) → GRU(32) → Dense |
| DerivedOnlyMLP | 43 derived features | Dense(64) → Dense(32) → Dense(1) |
| MultiViewGRUFusion | Sequence + derived features | GRU branch + MLP branch, concatenated |

---

## Key Results

**Official endpoint test (100 test engines, one prediction each):**

| Rank | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| 1 | XGBoost | 12.25 | 9.02 | 0.907 |
| 2 | DerivedOnlyMLP | 12.83 | 9.56 | 0.898 |
| 3 | GRU | 13.29 | 9.92 | 0.890 |
| 4 | MultiViewGRUFusion | 13.38 | 9.94 | 0.889 |

Model rankings differed across the development split, repeated validation (seeds 21/42/84), and the official endpoint test — confirming that a single development split is insufficient for model selection.

---

## Project Structure

```
dissertation-rul-xai/
│
├── notebooks/                    # NB01–NB12 — sequential experimental pipeline
│
├── data/
│   ├── raw/                      # Original NASA C-MAPSS FD001 files
│   ├── interim/                  # Intermediate preprocessed data
│   └── processed/                # Feature sets A, B, C; train/val/test splits
│
├── models/
│   ├── classical/                # Saved XGBoost and other classical models
│   ├── deep_learning/            # GRU, CNN1D .keras files
│   ├── multiview/                # MultiViewGRUFusion, DerivedOnlyMLP .keras files
│   ├── final/                    # Full-data refitted models for official test
│   └── final_validation/         # Per-split models from repeated validation
│
├── reports/
│   ├── figures/                  # All generated plots and visualisations
│   ├── metrics/                  # Validation metrics CSVs (NB04–NB06)
│   ├── predictions/              # Per-model prediction files
│   ├── explainability/           # Feature importance, SHAP, view masking CSVs
│   ├── robustness/               # Noise, masking, MC Dropout, bias CSVs
│   ├── summary/                  # Mid-semester consolidated summary
│   ├── final_validation/         # Repeated validation outputs (NB10)
│   ├── final_test/               # Official endpoint evaluation outputs (NB11)
│   ├── final_summary/            # Final consolidation tables and evidence ledger (NB12)
│   └── progress_log.md           # Step-by-step project progress log
│
├── submission/                   # BITS submission artefacts
├── configs/
│   └── experiment_config.yaml
├── requirements.txt
└── README.md
```

---

## Notebook Index

| Notebook | Description | Authoritative for |
|---|---|---|
| `01_dataset_understanding_and_problem_formulation.ipynb` | EDA, RUL formulation | Dataset description |
| `02_eda_deepening_and_feature_selection.ipynb` | Feature variance/correlation analysis | Feature selection |
| `03_baseline_ready_preprocessing.ipynb` | Engine-level split, scaling, Feature Sets A/B/C | Preprocessing pipeline |
| `04_classical_baseline_modelling.ipynb` | XGBoost, Ridge, GradientBoosting, Dummy baselines | Classical results |
| `05_deep_learning_baseline_colab.ipynb` | CNN1D and GRU baselines on Colab T4 | Deep learning baseline results (authoritative) |
| `05_deep_learning_baseline.ipynb` | Local development version — not the authoritative result source | Reference only |
| `06_initial_multiview_deep_learning_model.ipynb` | MultiViewGRUFusion and DerivedOnlyMLP | Multi-view seed-42 results |
| `07_initial_explainability_analysis.ipynb` | SHAP, view masking, local cases | Explainability evidence |
| `08_initial_robustness_trustworthiness_checks.ipynb` | Noise, masking, MC Dropout, RUL-range analysis | Trustworthiness evidence |
| `09_consolidated_results_summary_and_midsem_assets.ipynb` | Mid-semester consolidation — do not modify | Mid-semester report artefacts |
| `10_repeated_validation.ipynb` | Repeated validation (seeds 21/42/84) + cycle_index ablation | Final-phase validation evidence |
| `11_final_test_evaluation.ipynb` | Full-training refit + official test evaluation | Official test results |
| `12_final_results_consolidation.ipynb` | Final evidence pack — no new training | Dissertation tables and evidence ledger |

**NB05 note:** `05_deep_learning_baseline_colab.ipynb` is the version executed on Colab T4 and is the authoritative source for the GRU (RMSE 13.1605) and CNN1D (RMSE 18.1509) reported results.

**NB01–09** are frozen (mid-semester evidence base). **NB09** must remain completely unchanged.

---

## Experimental Protocol

The study used a three-stage protocol to prevent post-hoc adjustment:

- **Stage 1 (NB01–NB09):** Development on seed-42 split (80/20 engine-level). Model selection, explainability, robustness analysis.
- **Stage 2 (NB10):** Repeated validation across seeds 21, 42, 84. Frozen epoch counts selected as median best epoch per model.
- **Stage 3 (NB11–NB12):** Full-data refit with frozen configuration. One-time official endpoint evaluation. SHA-256 pre-test freeze manifest recorded before any test predictions.

All 57 evidence items tracked in `reports/final_summary/evidence_ledger.md`.

---

## Getting Started

```bash
git clone https://github.com/devaprasadp-bits/dissertation-rul-xai.git
cd dissertation-rul-xai
pip install -r requirements.txt
```

Notebooks are designed to run sequentially (NB01 → NB12). NB05, NB10, and NB11 were run on Google Colab due to GPU requirements; all others run locally.

---

## Git State

- Tag `midsem-frozen-2026-07-26` marks the submitted mid-semester state
- Branch `final-phase` contains all final-phase additions (merged into `main`)

---

## Author

Devaprasad P — M.Tech AIML (WILP), BITS Pilani, 2026
