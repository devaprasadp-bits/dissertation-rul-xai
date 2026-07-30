# Chapter 1 — Introduction

---

## 1.1 Predictive Maintenance and Industrial Asset Management

Industrial assets — turbines, engines, compressors, pumps, and rotating machinery — are the operational foundation of asset-intensive industries including aviation, energy, manufacturing, and transportation. The continuous availability of these assets directly affects operational continuity, safety, and economic performance. Unplanned asset failure can result in significant production losses, elevated maintenance costs, safety incidents, and regulatory exposure. The management of asset health is therefore a strategic concern in modern industrial operations.

Traditionally, maintenance has been organised around two approaches: reactive maintenance, which addresses failures after they occur, and fixed-schedule preventive maintenance, which replaces or services components at predetermined intervals regardless of their actual condition. Both approaches have well-known inefficiencies. Reactive maintenance allows failures to propagate to costly or catastrophic outcomes. Fixed-schedule maintenance results in unnecessary component replacement, wasted service capacity, and missed opportunities to extend the operational life of assets that are still in good condition.

Condition-based and predictive maintenance represent a more efficient alternative. In a condition-based regime, maintenance actions are triggered by evidence of degradation observed through sensor monitoring, rather than by calendar time or operating hours alone. Predictive maintenance extends this further: using statistical models and machine learning to estimate the current state of an asset and forecast how much useful life remains before failure. The target quantity in this forecasting task is the **Remaining Useful Life (RUL)** — the estimated number of remaining operating cycles or time units before an asset reaches its failure threshold.

Accurate RUL prediction enables maintenance teams to plan interventions at the optimal time: late enough to avoid unnecessary early replacement, but early enough to prevent unplanned failure. In enterprise asset management contexts, RUL predictions can feed scheduling systems that balance maintenance resource availability against fleet availability requirements. The prognostics and health management (PHM) research community has studied this problem extensively, with turbofan engine degradation being one of the most widely used benchmark settings [Saxena et al., 2008].

---

## 1.2 Remaining Useful Life Prediction

RUL prediction is formulated in this dissertation as a supervised regression problem. Given the recorded sensor history of an engine unit up to the current operating cycle, the task is to predict the number of cycles remaining before failure. This formulation requires labelled training data in which the full lifecycle of each engine unit is observed, allowing the true RUL at every cycle to be computed retrospectively.

The NASA C-MAPSS turbofan engine degradation dataset [Saxena et al., 2008] provides exactly this structure: 100 training engine units with full run-to-failure histories, and 100 test engine units with partial histories and corresponding endpoint RUL labels. C-MAPSS has become the standard benchmark for prognostic RUL studies, enabling consistent comparison across methods.

A characteristic modelling choice in this work is the use of a **capped RUL target**: the true RUL is capped at 125 cycles during training and evaluation. This reduces the dominance of early-lifecycle, high-RUL samples where accurate prediction is both computationally difficult and operationally less important. The cap is a widely used convention in the prognostics literature, though the specific value and its effect on results must be stated explicitly when comparing across studies.

---

## 1.3 Motivation for Explainability and Trustworthiness

Predictive accuracy is a necessary but not sufficient condition for the practical adoption of machine learning in maintenance decision support. A maintenance engineer who receives a predicted RUL of 15 cycles needs to understand why the model is making that prediction. Without supporting explanation, the prediction is a black box: it may be correct or it may reflect a spurious pattern in the training data, and the engineer has no way to distinguish the two.

Explainability — the ability to attribute model predictions to specific inputs, features, or input views — addresses this need. In prognostics specifically, explanations can help engineers validate that the model's predictions are driven by sensor channels known to track degradation, rather than by noise, preprocessing artefacts, or correlated but uninformative features. Feature importance analysis and SHAP attributions provide global explanations of the model's learned behaviour. Local case analysis provides instance-level explanations for specific engine predictions.

Trustworthiness is a related but broader concept. It encompasses not only explainability but also the model's reliability under realistic operational conditions: sensor noise, intermittent signal loss, different degradation patterns across asset cohorts, and variation in asset operating history. A model that performs well on a clean benchmark but degrades severely under moderate sensor noise would be unsafe to deploy in practice. Trustworthiness studies — covering robustness to perturbation, lifecycle-region error analysis, prediction bias characterisation, and exploratory uncertainty estimation — provide evidence about the conditions under which a model's predictions can be relied upon.

This dissertation treats explainability and trustworthiness as first-class evaluation criteria alongside predictive accuracy. The multi-dimensional assessment in Chapter 7 is as central to the dissertation's contribution as the RMSE comparison in Chapter 6.

---

## 1.4 Problem Statement

This dissertation studies a multi-view deep learning approach to RUL prediction on the NASA C-MAPSS FD001 benchmark under a leakage-safe, reproducible, and multi-stage experimental protocol. The work examines whether combining a raw sensor sequence view with an engineered degradation feature view improves RUL prediction over single-view and classical alternatives, and whether the fusion model's predictions are explainable and trustworthy.

---

## 1.5 Objectives

The principal objectives of this dissertation are:

1. To design and implement a leakage-safe, engine-level experimental pipeline for capped RUL regression on C-MAPSS FD001.
2. To compare raw sensor sequences, engineered degradation features, and their fusion in a multi-view GRU-based architecture against classical XGBoost and single-view neural baselines.
3. To test the stability of model performance and rankings across repeated independent engine cohort splits.
4. To conduct a one-time, pre-registered official endpoint evaluation under a frozen model protocol.
5. To explain the XGBoost model's predictions using feature importance and SHAP analysis, and to characterise the fusion model's input-view dependencies through view masking.
6. To evaluate the fusion model's trustworthiness through noise and masking robustness studies, lifecycle-region error analysis, prediction bias characterisation, and exploratory MC Dropout uncertainty estimation.
7. To report all results — including the negative fusion finding — in an honest and reproducible manner anchored to a 57-item evidence ledger.

---

## 1.6 Research Questions

Four research questions structure the experimental work and guide the interpretation in Chapters 6–8:

**RQ1:** How does predictive accuracy differ between raw sensor sequences, engineered degradation features, and their fusion in a multi-view model, when evaluated on NASA C-MAPSS FD001 under a leakage-safe, engine-level experimental protocol?

**RQ2:** Which features and input views are most influential in determining RUL predictions, and what do global and local explanations reveal about the predictive signal learned by the models?

**RQ3:** To what extent are the models robust to sensor noise and input masking, and how does prediction accuracy vary across different RUL lifecycle regions?

**RQ4:** How consistent is model performance and ranking across repeated engine-grouped validation splits, and does the held-out official endpoint evaluation support or contradict the validation-stage conclusions?

---

## 1.7 Scope and Boundaries

This dissertation is scoped to the NASA C-MAPSS FD001 subset: 100 training engines, single operating condition, single fault mode, 14 variable sensor measurements. The results and conclusions apply within this scope.

The work does not include evaluation on FD002–FD004, on real industrial datasets, or on other prognostics benchmarks. The multi-view formulation is implemented using two views derived from the same numerical sensor stream; it does not constitute native multimodality, which would require temporally aligned heterogeneous data sources. The models are not deployed in a production system. Calibrated uncertainty estimation and cost-sensitive evaluation are outside the experimental scope.

These boundaries are intentional: they allow the research questions to be addressed rigorously within the dissertation timeline while producing results that are clearly defined and reproducible.

---

## 1.8 Dissertation Contributions

This dissertation makes six contributions:

**C1 — Controlled experimental protocol:** A multi-stage evaluation combining engine-level repeated validation across three independent cohorts, per-model random-state reset, pre-test freeze manifesting with SHA-256 artefact integrity, and a single one-time official held-out evaluation.

**C2 — Multi-view representation study:** A systematic comparison of raw sensor sequences, derived degradation features, and their GRU-based fusion, demonstrating that the two views carry meaningful but asymmetric information.

**C3 — Honest negative finding:** Multi-view fusion is competitive but not consistently superior. This negative result is reported as a controlled experimental finding, not concealed.

**C4 — Feature-level and view-level explainability:** XGBoost feature importance, SHAP analysis, view masking, and local case analysis characterising what the models have learned.

**C5 — Trustworthiness evidence:** Noise and masking robustness, lifecycle-region error profiling, prediction bias characterisation, perturbation stability, and exploratory MC Dropout uncertainty.

**C6 — Reproducibility:** A 57-item evidence ledger with file paths and SHA-256 hashes, version-controlled artefacts, and a pre-test freeze manifest.

---

## 1.9 Report Organisation

The remainder of this dissertation is organised as follows:

**Chapter 2** reviews the relevant literature on predictive maintenance, C-MAPSS RUL prediction, classical and deep learning models for prognostics, multi-view and multimodal learning, explainable AI, and robustness and uncertainty.

**Chapter 3** describes the NASA C-MAPSS dataset and FD001 subset, defines the capped-RUL regression task and multi-view input formulation, introduces the three evaluation populations used in this study, and states the scope boundary.

**Chapter 4** describes the data preparation and feature engineering pipeline: sensor selection, the three feature sets, derived degradation features, cycle_index, leakage controls, and sequence construction.

**Chapter 5** presents the experimental methodology: all model architectures, the three-stage evaluation protocol, repeated-validation procedure, epoch selection, full-data refitting, pre-test freeze, and the official test procedure.

**Chapter 6** reports predictive performance results across all three evidence layers: historical seed-42 reference, repeated validation, and official endpoint test.

**Chapter 7** presents the explainability and trustworthiness analyses: XGBoost feature importance, SHAP, view masking, local cases, robustness, bias, perturbation stability, and MC Dropout uncertainty.

**Chapter 8** discusses the findings in relation to the four research questions and reflects on why fusion initially looked strongest, why XGBoost won the endpoint test, and why rankings vary across cohorts.

**Chapter 9** presents the study limitations, industrial relevance, and future work directions.

**Chapter 10** concludes the dissertation with a summary of what was built, evaluated, learned, and not demonstrated.
