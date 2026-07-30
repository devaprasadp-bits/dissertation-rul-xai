# Final Dissertation — Chapter Skeleton
## Explainable and Trustworthy Multimodal Deep Learning for Predictive Maintenance of Industrial Assets
### 2023AA05069 | Devaprasad P | BITS Pilani M.Tech AIML

---

## Front Matter

- Cover page (prescribed BITS format)
- Title page
- Supervisor certificate (Dr. Ashok Veilumuthu, SAP Labs India)
- Acknowledgements
- Prescribed abstract sheet (≤200 words — see abstract_draft.md)
- Table of contents
- List of figures
- List of tables
- List of abbreviations

*Front matter uses Roman numerals (i, ii, iii, …). Chapter 1 begins on Arabic page 1.*

---

## Chapter 1 — Introduction

1.1 Predictive Maintenance and Industrial Asset Management  
1.2 Remaining Useful Life Prediction  
1.3 Motivation for Explainability and Trustworthiness  
1.4 Problem Statement  
1.5 Objectives  
1.6 Research Questions  
1.7 Scope and Boundaries  
1.8 Dissertation Contributions  
1.9 Report Organisation  

---

## Chapter 2 — Literature Review

2.1 Predictive Maintenance Approaches  
2.2 RUL Prediction on C-MAPSS  
2.3 Classical Machine-Learning Models for Prognostics  
2.4 Deep Sequence Models  
2.5 Multi-View and Multimodal Learning  
2.6 Explainability for Prognostics  
2.7 Robustness and Uncertainty in Prognostic Models  
2.8 Research Gap  

---

## Chapter 3 — Dataset and Problem Formulation

3.1 NASA C-MAPSS Dataset  
3.2 FD001 Subset: Structure and Characteristics  
3.3 Sensor Selection and Operating Condition  
3.4 Capped-RUL Target Formulation  
3.5 Regression Task Definition  
3.6 Multi-View Input Formulation  
3.7 Evaluation Populations  
3.8 Scope Boundary  

---

## Chapter 4 — Data Preparation and Feature Engineering

4.1 Data-Quality Checks  
4.2 Sensor Selection Rationale  
4.3 Feature Set A: Raw Sensor Sequences  
4.4 Feature Set B: Rolling Statistical Features  
4.5 Feature Set C: Derived Degradation Features  
4.6 cycle_index and its Role  
4.7 Removed Feature: normalized_cycle_age  
4.8 Train-Only Scaling  
4.9 Engine-Level Splitting and Leakage Controls  
4.10 Sequence Construction for Neural Models  

---

## Chapter 5 — Experimental Methodology

5.1 Overall Experiment Design  
5.2 Classical Baselines: XGBoost, Ridge, RF, GBT  
5.3 Deep Sequence Baselines: CNN1D and GRU  
5.4 DerivedOnlyMLP  
5.5 MultiViewGRUFusion Architecture  
5.6 Evaluation Metrics  
5.7 Stage 1 — Original Seed-42 Development  
5.8 Stage 2 — Repeated Validation Protocol (Seeds 21, 42, 84)  
5.9 Ablation Protocol: cycle_index Removal  
5.10 Final Epoch Selection  
5.11 Full-Data Refitting (100 Training Engines)  
5.12 Pre-Test Freeze Manifest  
5.13 Official Test Procedure  
5.14 Reproducibility Artefacts  

---

## Chapter 6 — Predictive Performance Results

6.1 Historical Seed-42 Window-Aligned Comparison (Table F2)  
6.2 Repeated-Validation Split Results (Tables F3–F5)  
&nbsp;&nbsp;&nbsp;&nbsp;6.2.1 Per-Split RMSE and MAE  
&nbsp;&nbsp;&nbsp;&nbsp;6.2.2 Mean and Standard Deviation Summary  
&nbsp;&nbsp;&nbsp;&nbsp;6.2.3 Fusion Win Counts and Pairwise Differences  
6.3 Per-Engine Validation Analysis (Table F7)  
6.4 cycle_index Ablation Results (Table F6)  
6.5 Official Endpoint Test Results (Table F8)  
6.6 Comparison of Rankings Across Evaluation Populations  
6.7 Interpretation: Outcome C and Outcome D  

---

## Chapter 7 — Explainability and Trustworthiness

7.1 XGBoost Feature Importance (Table F9)  
7.2 SHAP Analysis: Feature Ranking and Dependence  
&nbsp;&nbsp;&nbsp;&nbsp;7.2.1 SHAP Feature Ranking  
&nbsp;&nbsp;&nbsp;&nbsp;7.2.2 SHAP Dependence: cycle_index  
&nbsp;&nbsp;&nbsp;&nbsp;7.2.3 SHAP Dependence: sensor_measurement_4_rmean  
&nbsp;&nbsp;&nbsp;&nbsp;7.2.4 SHAP Dependence: sensor_measurement_11_rmean  
7.3 View Masking Sensitivity (Table in §7.4 of source)  
7.4 Local Prediction Cases (Units 2, 53, 22, 93)  
7.5 Gaussian-Noise Robustness  
7.6 Random Masking Robustness (Table F10)  
7.7 Lifecycle-Region Error Analysis  
7.8 Prediction Bias  
7.9 Repeated Perturbation Stability  
7.10 MC Dropout Uncertainty  
7.11 Qualification of Operational Trustworthiness  

---

## Chapter 8 — Discussion

8.1 Research Question 1: Raw Sequences vs Engineered Features vs Fusion  
8.2 Research Question 2: Contribution of Each Input View  
8.3 Research Question 3: Robustness and Lifecycle Behaviour  
8.4 Research Question 4: Stability and Held-Out Generalisation  
8.5 Why Fusion Looked Strongest Initially  
8.6 Why XGBoost Performed Best on the Official Test  
8.7 Why Model Ranking Changes Across Cohorts  
8.8 The Value of Explainability Beyond RMSE  

---

## Chapter 9 — Limitations, Relevance and Future Work

9.1 Study Limitations  
9.2 Relevance to Industrial Predictive Maintenance  
9.3 Practical Implications for Predictive-Maintenance Systems  
9.4 Future Work: Multimodal Extensions  
9.5 Future Work: Temporal Alignment of Heterogeneous Data  
9.6 Future Work: Calibration and Cost-Sensitive Evaluation  
9.7 Future Work: Extension to FD002–FD004 and Industrial Data  

---

## Chapter 10 — Conclusion

10.1 What Was Built  
10.2 What Was Evaluated  
10.3 What Was Learned  
10.4 What Was Not Demonstrated  
10.5 Final Contribution  

---

## Appendices

**Appendix A** — Dataset Details and EDA Supplementary Figures  
**Appendix B** — Feature Correlation and Sensor Analysis  
**Appendix C** — Hyperparameter Tables  
&nbsp;&nbsp;&nbsp;&nbsp;C.1 XGBoost hyperparameters  
&nbsp;&nbsp;&nbsp;&nbsp;C.2 GRU architecture and training hyperparameters  
&nbsp;&nbsp;&nbsp;&nbsp;C.3 CNN1D hyperparameters  
&nbsp;&nbsp;&nbsp;&nbsp;C.4 DerivedOnlyMLP hyperparameters  
&nbsp;&nbsp;&nbsp;&nbsp;C.5 MultiViewGRUFusion hyperparameters  
**Appendix D** — Training Histories  
&nbsp;&nbsp;&nbsp;&nbsp;D.1 Seed-42 development training curves  
&nbsp;&nbsp;&nbsp;&nbsp;D.2 Repeated-validation training curves per seed  
&nbsp;&nbsp;&nbsp;&nbsp;D.3 Final full-data training histories  
**Appendix E** — Supplementary Model Results  
&nbsp;&nbsp;&nbsp;&nbsp;E.1 Full classical baseline results (all hyperparameter variants)  
&nbsp;&nbsp;&nbsp;&nbsp;E.2 Window-size sensitivity analysis  
&nbsp;&nbsp;&nbsp;&nbsp;E.3 Per-engine endpoint predictions  
**Appendix F** — Evidence Mapping  
&nbsp;&nbsp;&nbsp;&nbsp;F.1 Evidence ledger summary  
&nbsp;&nbsp;&nbsp;&nbsp;F.2 Notebook sequence and reproducibility steps  
**Appendix G** — Additional Figures  
**Signed Final Checklist (last page)**

---

*Writing order: Ch 3–5 → Ch 6–8 → Ch 1–2 → Ch 9–10 → front matter + appendices*
