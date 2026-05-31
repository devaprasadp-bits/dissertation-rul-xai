# Dissertation: Remaining Useful Life (RUL) Prediction with Explainable AI (XAI)

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

## Author

BITS Pilani – Final Semester Dissertation, 2026
