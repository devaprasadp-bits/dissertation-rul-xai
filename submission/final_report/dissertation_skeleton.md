# Explainable and Trustworthy Multimodal Deep Learning for Predictive Maintenance of Industrial Assets

**Student:** Devaprasad P | 2023AA05069  
**Programme:** M.Tech AIML | AIMLCZG628T | BITS Pilani  
**Submission window:** 28 July – 2 August 2026

---

## PRELIMINARY PAGES

*(Preliminary pages use lower-case Roman numerals: i, ii, iii, …)*

### 1. Cover Page

*(Prepare using BITS specimen. Fields: title, student name, ID, programme, institution, year.)*

**STATUS: PENDING — use BITS specimen template**

---

### 2. Title Page / Inner Cover

*(Prepare using BITS specimen. Fields: full title, student name, BITS ID, supervisor name, institution, submission date, degree.)*

**STATUS: PENDING — use BITS specimen template**

---

### 3. Certificate from the Supervisor

*(Use BITS prescribed template. Must be signed by supervisor. Obtain early — critical-path item.)*

**Supervisor:** [Name to be confirmed]  
**Organisation:** [Organisation to be confirmed]  
**Date signed:** [Pending]

**STATUS: PENDING — template to be obtained; send to supervisor by 26 July**

---

### 4. Acknowledgements

*(Acknowledge supervisor, institution, and any other support. One page.)*

[DRAFT — to be completed]

---

### 5. BITS Abstract Sheet

*(BITS-prescribed format. Fields: organisation, location, project duration, start date, submission date, title, student ID and name, supervisor, additional examiner, faculty mentor, keywords, project area, student signature, supervisor signature. Abstract body must not exceed 200 words.)*

**Project Area:** Artificial Intelligence and Machine Learning — Predictive Maintenance and Prognostics

**Keywords:** Predictive Maintenance, Remaining Useful Life, NASA C-MAPSS, Multi-View Deep Learning, Explainable Artificial Intelligence, Model Robustness, Uncertainty Estimation

**Abstract (≤200 words):**

[DRAFT — to be written. Strict 200-word limit. Must cover: problem, approach, key results, conclusions.]

**STATUS: PENDING — abstract text, signatures required**

---

### 6. Table of Contents

*(To be generated from final document. Page numbers to be added on final assembly.)*

---

### 7. List of Figures

*(Every figure cited in text must appear here. Figure number and caption below figure.)*

---

### 8. List of Tables

*(Every table cited in text must appear here. Table number and title above table.)*

---

### 9. List of Abbreviations and Acronyms

| Abbreviation | Full Form |
|---|---|
| AI | Artificial Intelligence |
| APM | Asset Performance Management |
| C-MAPSS | Commercial Modular Aero-Propulsion System Simulation |
| CNN | Convolutional Neural Network |
| ES | Early Stopping |
| FD001 | Fault Dataset 001 (C-MAPSS subset) |
| GRU | Gated Recurrent Unit |
| LSTM | Long Short-Term Memory |
| MAE | Mean Absolute Error |
| MC Dropout | Monte Carlo Dropout |
| MLP | Multi-Layer Perceptron |
| MSE | Mean Squared Error |
| NASA | National Aeronautics and Space Administration |
| RMSE | Root Mean Squared Error |
| RUL | Remaining Useful Life |
| SHAP | SHapley Additive exPlanations |
| XAI | Explainable Artificial Intelligence |
| XGBoost | Extreme Gradient Boosting |

---

## MAIN REPORT

*(Chapter 1 begins on Arabic page 1.)*

---

## Chapter 1 — Introduction

### 1.1 Predictive Maintenance Context

[DRAFT — industrial asset failures, downtime cost, maintenance strategies, transition from reactive to predictive]

### 1.2 Problem Statement

[DRAFT — RUL prediction as capped regression; challenge of interpretability and trustworthiness in practice]

### 1.3 Motivation

[DRAFT — explainability for engineering acceptance; robustness before operational adoption; benchmark prototype rationale]

### 1.4 Scope

[DRAFT — FD001 single operating condition, single fault mode; numerical sensor time-series; multi-view formulation as benchmark-driven realisation of multimodal learning; distinction from native heterogeneous multimodality]

### 1.5 Limitations

[DRAFT — single dataset subset; single operating condition; benchmark prototype not deployed; window-only evaluation; exploratory uncertainty]

### 1.6 Research Questions

**RQ1:** How do raw sensor sequences, engineered degradation features and their fusion compare for capped RUL prediction on C-MAPSS FD001?

**RQ2:** What predictive contribution does each view provide, and which derived features most strongly influence the predictions?

**RQ3:** How does the fusion model behave under controlled noise, feature masking, missing-view conditions and different RUL regions?

**RQ4:** How stable are the main findings across engine-level validation splits and on the independent FD001 test engines?

### 1.7 Dissertation Objectives

[DRAFT — list objectives derived from RQs]

### 1.8 Summary of Methodology

[DRAFT — brief paragraph: pipeline, models, evaluation, explainability, trustworthiness]

### 1.9 Structure of the Report

[DRAFT — one sentence per chapter describing its content]

---

## Chapter 2 — Literature Review

### 2.1 Predictive Maintenance and Prognostics

[DRAFT]

### 2.2 RUL Prediction Using C-MAPSS

[DRAFT — benchmark studies, commonly used methods, reported results]

### 2.3 Classical Machine-Learning Methods for RUL

[DRAFT — regression baselines, gradient boosting, feature engineering]

### 2.4 CNN, RNN, LSTM and GRU Approaches

[DRAFT — sequence modelling for degradation; key architectures]

### 2.5 Multi-View and Multimodal Learning

[DRAFT — fusion strategies, view complementarity, distinction from this work]

### 2.6 Explainable AI in Predictive Maintenance

[DRAFT — SHAP, LIME, attribution methods, engineering acceptance literature]

### 2.7 Robustness and Uncertainty Estimation

[DRAFT — noise robustness, MC Dropout, calibration literature]

### 2.8 Literature Synthesis and Identified Research Gap

[DRAFT — comparison table and gap statement]

**Literature comparison table:**

| Study | Dataset | Input representation | Model | Explainability | Trustworthiness | Main limitation |
|---|---|---|---|---|---|---|
| [to be filled] | | | | | | |

---

## Chapter 3 — Dataset and Problem Formulation

### 3.1 C-MAPSS Overview

[DRAFT — four subsets FD001–FD004, operating conditions, fault modes, dataset origin]

### 3.2 FD001 Characteristics

[DRAFT — 100 train / 100 test engines, 21 sensors, single operating condition, single fault mode]

### 3.3 Input Columns and Selected Sensors

[DRAFT — all 21 sensors, 3 operational settings; variance-based filtering to 14 variable sensors; constant sensors dropped]

### 3.4 Capped RUL Definition

[DRAFT — piecewise linear health index; cap at 125 cycles; justification]

### 3.5 Training and Test Label Construction

[DRAFT — training: max_cycle − current_cycle; test: last observed cycle + RUL_FD001.txt offset; all-row reconstructed vs official endpoint distinction]

### 3.6 Multi-View Formulation

[DRAFT — View 1: raw sensor sequence 30×14; View 2: derived degradation features 43; distinction from native multimodality]

### 3.7 Scope Boundary

[DRAFT — numerical benchmark data; multi-view as benchmark-driven implementation; title clarification]

---

## Chapter 4 — Research Methodology

### 4.1 End-to-End Experimental Pipeline

[DRAFT — overview figure reference]

### 4.2 Leakage-Prevention Strategy

[DRAFT — engine-level splitting; scaler fit on training data only; rolling features computed within engine; normalized_cycle_age excluded; test set opened exactly once]

### 4.3 Engine-Level Splitting

[DRAFT — 80/20 engine split, SPLIT_SEED, MODEL_SEED separation]

### 4.4 Feature Engineering

[DRAFT — rolling mean (window=5), rolling std, delta from initial, cycle_index; Feature Sets A, B, C defined]

### 4.5 Scaling

[DRAFT — StandardScaler fit on training split only; applied to validation and test]

### 4.6 Window Construction

[DRAFT — 30-cycle sliding windows; stride=1; engines shorter than 30 cycles excluded]

### 4.7 Model Architectures

[DRAFT — XGBoost/C, GRU/B, DerivedOnlyMLP, MultiViewGRUFusion; exact configs from frozen plan]

### 4.8 Evaluation Protocol

[DRAFT — RMSE primary; MAE, R²; window-aligned comparison; per-engine metrics; NASA asymmetric score optional]

### 4.9 Reproducibility Controls

[DRAFT — SPLIT_SEEDS=[21,42,84], MODEL_SEED=42 fixed; keras clear_session; tf.keras.utils.set_random_seed]

---

## Chapter 5 — Experimental Setup

### 5.1 Hardware and Colab Environment

[DRAFT — Google Colab T4 GPU; local for NB12]

### 5.2 Package Versions

[DRAFT — reference environment manifest from NB10]

### 5.3 Split Seeds and Model Seed

[DRAFT — SPLIT_SEEDS=[21,42,84]; MODEL_SEED=42; rationale]

### 5.4 Hyperparameters

[DRAFT — table with all frozen hyperparameters per model]

### 5.5 Baseline Protocol

[DRAFT — Dummy Mean, Ridge, XGBoost; Feature Sets A, B, C]

### 5.6 Repeated-Validation Protocol

[DRAFT — three engine-level seeds; MODEL_SEED fixed; measures cohort sensitivity only]

### 5.7 Official Test Protocol

[DRAFT — full training on all 100 engines; epoch count from median of validation best epochs; test evaluated exactly once]

### 5.8 Evaluation Metrics

[DRAFT — RMSE, MAE, R²; mean error, under/over-prediction rate, % within 5/10/20 cycles; NASA asymmetric score]

---

## Chapter 6 — Predictive Results

### 6.1 Seed-42 Historical Results

[EVIDENCE: E01, E02 — NB06 window-aligned comparison table. Source: reports/metrics/multiview_vs_baselines_comparison_fd001.csv]

| Model | RMSE | MAE | R² |
|---|---|---|---|
| MultiViewGRUFusion | 12.0657 | 8.9406 | 0.9168 |
| XGBoost/C (window-aligned) | 12.4894 | 9.2675 | 0.9109 |
| DerivedOnlyMLP | 13.1451 | 9.4204 | 0.9013 |
| GRU/B | 13.1605 | 9.7182 | 0.9010 |
| CNN1D/B | 18.1509 | 14.0382 | 0.8118 |

[DRAFT — interpretation; one-split caveat; window-aligned basis]

### 6.2 Repeated-Validation Results

[EVIDENCE: E04 — NB10. PENDING]

[Table F3: per-split results; Table F4: mean ± std]

### 6.3 Per-Engine Analysis

[EVIDENCE: NB10 per-engine metrics. PENDING]

### 6.4 cycle_index Ablation

[EVIDENCE: NB10 ablation table. PENDING]

### 6.5 Official Test Results

[EVIDENCE: E05 — NB11. PENDING — execute exactly once after NB10 frozen]

### 6.6 Comparison with Prior Work

[DRAFT — with qualification; window-aligned basis; FD001 only; single operating condition]

---

## Chapter 7 — Explainability and Trustworthiness

### 7.1 XGBoost Feature Importance

[EVIDENCE: E06 — NB07 feature importance CSV. Source: reports/metrics/classical_feature_importance_fd001.csv]

[DRAFT]

### 7.2 SHAP Analysis

[EVIDENCE: E07 — NB07 SHAP summary. PENDING ARTEFACT PATH]

[DRAFT — top features, SHAP values, TreeExplainer on XGBoost; model behaviour not physical causality]

### 7.3 View Contribution and Single-View References

[EVIDENCE: E08 — NB06/NB07 view masking and single-view model results]

[DRAFT — GRU RMSE 13.16, DerivedOnlyMLP RMSE 13.15 as proper single-view references; masking RMSE 42.57 is inference-time sensitivity, not a trained single-view model]

### 7.4 View-Masking Sensitivity

[EVIDENCE: E09 — NB07 masking results. PENDING ARTEFACT PATH]

[DRAFT — derived-masked vs sequence-masked; sharp vs small performance drop]

### 7.5 Local Prediction Cases

[EVIDENCE: NB07. PENDING ARTEFACT PATH]

[DRAFT — units 2, 53, 22, 93; where model performs well and where it struggles]

### 7.6 Noise and Masking Robustness

[EVIDENCE: E03 — NB08 robustness metrics. Source: reports/metrics/ (to confirm path)]

[DRAFT — Gaussian noise std=0.20 in standardised space; small RMSE increase; noise definition in standardised units]

### 7.7 RUL-Range-Wise Errors

[EVIDENCE: NB08. PENDING ARTEFACT PATH]

[DRAFT — near-failure / mid / early-capped error regions; practically useful near failure]

### 7.8 Bias

[EVIDENCE: NB08 mean error. PENDING]

### 7.9 Perturbation Stability

[EVIDENCE: NB08. PENDING]

### 7.10 Exploratory MC Dropout Uncertainty

[EVIDENCE: NB08. PENDING ARTEFACT PATH]

[DRAFT — dispersion signal, not calibrated confidence; exploratory only]

---

## Chapter 8 — Discussion

### 8.1 RQ1 — View Comparison

[DRAFT — fusion vs single-view; seed-42 and repeated-validation evidence; qualified claim]

### 8.2 RQ2 — Feature Contribution

[DRAFT — derived features dominant; SHAP top features; cycle_index ablation result]

### 8.3 RQ3 — Model Behaviour Under Perturbation

[DRAFT — robustness, masking, RUL region, uncertainty]

### 8.4 RQ4 — Stability Across Splits and Test

[DRAFT — repeated validation; test result; cohort sensitivity finding]

### 8.5 Meaning of the Fusion Result

[DRAFT — complementary views; not simply additive; practical interpretation]

### 8.6 Strength of XGBoost

[DRAFT — consistently competitive; well-engineered features; honest discussion]

### 8.7 Dependence on Derived Features

[DRAFT — cycle_index and rolling statistics; shortcut risk; what this means]

### 8.8 Validation-versus-Test Differences

[DRAFT — engine cohort vs independent test; expected vs actual differences]

### 8.9 Relevance to the Work Environment

[DRAFT — industrial asset-health prediction; RUL estimates for maintenance planning; explainability for engineering acceptance; robustness and uncertainty before adoption; benchmark prototype; not deployed in SAP APM or customer systems]

---

## Chapter 9 — Conclusions and Future Work

### 9.1 Work Completed

[DRAFT]

### 9.2 Principal Findings

[DRAFT — structured around RQ1–RQ4]

### 9.3 Contribution

[DRAFT — leakage-aware, explainability- and trustworthiness-oriented multi-view pipeline; ablation and perturbation evidence]

### 9.4 Practical Implications

[DRAFT — maintenance planning; explainability for acceptance; robustness threshold before adoption]

### 9.5 Limitations

[DRAFT — single subset; single operating condition; benchmark prototype; window-only evaluation; exploratory uncertainty; one architecture]

### 9.6 Recommendations and Future Research

[DRAFT — FD002/FD004; native multimodal data; attention-based fusion; calibrated uncertainty; deployment validation]

---

## END MATTER

### References

[All cited works. Every in-text citation must map to an entry here. Citation audit required before submission.]

*(Format: consistent with BITS requirements. In-text citations must be added throughout all chapters.)*

### Glossary

| Term | Definition |
|---|---|
| Capped RUL | RUL value clipped at a maximum of 125 cycles |
| C-MAPSS | NASA turbofan engine degradation simulation dataset |
| Engine cohort | Set of engine units assigned to a training or validation split |
| Feature Set C | 14 raw sensors + 43 derived degradation features (rolling mean, std, delta, cycle_index) |
| Multi-view fusion | Combining two feature representations of the same data via a joint model |
| View masking | Setting one input view to zero at inference time to assess its contribution |
| Window-aligned comparison | Evaluating all models on the same validation rows that have a complete 30-cycle history |

### Appendices

**Appendix A — C-MAPSS Dataset Description**

[Full column schema, sensor descriptions, operating condition definitions]

**Appendix B — Feature Set Definitions**

[Feature Set A, B, C column lists with engineering descriptions]

**Appendix C — Model Architecture Diagrams**

[Keras model summaries and/or architecture figures for GRU, DerivedOnlyMLP, MultiViewGRUFusion]

**Appendix D — Training History Plots**

[Loss and MAE curves for all neural models across all seeds]

**Appendix E — Per-Engine Validation Metrics**

[Full per-engine RMSE table for seed 42 and repeated validation]

**Appendix F — Experimental Environment**

[Package versions, hardware, seeds — from NB10 environment manifest]

**Appendix G — NB01–09 Documentation Fix Log**

[Record of markdown-only corrections applied to mid-semester notebooks]

---

### BITS Final Dissertation Checklist

*(Attach completed and student-signed checklist as the final page of the report. This is a BITS mandatory requirement.)*

**STATUS: PENDING — obtain BITS checklist template; complete and sign before submission**

---

*Skeleton created: 2026-07-26 | Final report assembly target: 1 August 2026*
