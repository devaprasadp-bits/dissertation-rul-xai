# Dissertation: Explainable and Trustworthy Multimodal Deep Learning for Predictive Maintenance of Industrial Assets

## Overview

This project investigates Remaining Useful Life (RUL) prediction for industrial systems using deep learning models, with a focus on model interpretability through Explainable AI (XAI) techniques.

## Project Structure

```
dissertation-rul-xai/
│
├── data/
│   ├── raw/               # Original, immutable datasets
│   ├── interim/           # Intermediate transformed data
│   └── processed/         # Final datasets ready for modelling
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_rul_label_preparation.ipynb
│   └── 03_baseline_model.ipynb
│
├── src/
│   ├── data_preparation/  # Scripts for loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and training
│   ├── evaluation/        # Metrics and evaluation utilities
│   └── explainability/    # SHAP, LIME, and other XAI methods
│
├── reports/
│   ├── figures/           # Generated plots and visualisations
│   ├── tables/            # Result tables
│   └── progress_log.md    # Weekly progress notes
│
├── configs/
│   └── experiment_config.yaml   # Hyperparameters and experiment settings
│
├── README.md
└── requirements.txt
```

## Dataset

- **CMAPSS (NASA Turbofan Engine Degradation Simulation)**  
  Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/devaprasadp-bits/dissertation-rul-xai.git
cd dissertation-rul-xai

# Install dependencies
pip install -r requirements.txt
```

## Notebook index

| Notebook | Description | Authoritative for |
|---|---|---|
| `01_dataset_understanding_and_problem_formulation.ipynb` | EDA, RUL formulation | Dataset description |
| `02_eda_deepening_and_feature_selection.ipynb` | Feature variance/correlation analysis | Feature selection |
| `03_baseline_ready_preprocessing.ipynb` | Engine-level split, scaling, Feature Sets A/B/C | Preprocessing pipeline |
| `04_classical_baseline_modelling.ipynb` | XGBoost, Ridge, GradientBoosting, Dummy baselines | Classical results |
| `05_deep_learning_baseline_colab.ipynb` | CNN1D and GRU baselines on Colab T4 | Deep learning baseline results (authoritative) |
| `05_deep_learning_baseline.ipynb` | Local version of step 5 — not the authoritative result source | Reference only |
| `06_initial_multiview_deep_learning_model.ipynb` | MultiViewGRUFusion and DerivedOnlyMLP | Multi-view seed-42 results |
| `07_initial_explainability_analysis.ipynb` | SHAP, view masking, local cases | Explainability evidence |
| `08_initial_robustness_trustworthiness_checks.ipynb` | Noise, masking, MC Dropout, RUL-range analysis | Trustworthiness evidence |
| `09_consolidated_results_summary_and_midsem_assets.ipynb` | Mid-semester consolidation — do not modify | Mid-semester report artefacts |
| `10_repeated_validation.ipynb` | Repeated validation (seeds 21/42/84) + cycle_index ablation | Final-phase validation evidence |
| `11_final_test_evaluation.ipynb` | Full-training + official test evaluation | Official test results |
| `12_final_results_consolidation.ipynb` | Final evidence pack — no training | Dissertation tables F1–F12 |

**Note on NB05 dual files:** `05_deep_learning_baseline_colab.ipynb` is the version executed on Colab T4 and is the authoritative source for the GRU (RMSE 13.1605) and CNN1D (RMSE 18.1509) reported results. `05_deep_learning_baseline.ipynb` is the local development version and should not be treated as the result source.

**NB01–09** are frozen (mid-semester evidence base). Only minor markdown corrections are permitted.  
**NB09** must remain completely unchanged.

## Git state

- Tag `midsem-frozen-2026-07-26` marks the submitted mid-semester state
- Branch `final-phase` contains all final-phase additions

## Author

M Tech (WILP) – BITS Pilani, 2026
