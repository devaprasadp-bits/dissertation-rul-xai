# Research Questions and Contribution Statement
## Final Dissertation — 2023AA05069

---

## Research Questions

The four research questions are unchanged from the mid-semester scope, but their answers are now informed by the complete experimental record including repeated validation and the official held-out endpoint test.

**RQ1 — Feature representation effectiveness:**
How does predictive accuracy differ between raw sensor sequences, engineered degradation features, and their fusion in a multi-view model, when evaluated on the NASA C-MAPSS FD001 turbofan degradation dataset under a leakage-safe, engine-level experimental protocol?

*Evidence: E01, E05, E06, E10, E53 — Tables F2, F3, F4, F8*

**RQ2 — Explainability of model behaviour:**
Which features and input views are most influential in determining RUL predictions, and what do global and local explanations reveal about the predictive signal learned by the models?

*Evidence: E13–E24 — Tables F9, view masking; SHAP plots E16–E20*

**RQ3 — Robustness and lifecycle behaviour:**
To what extent are the models robust to sensor noise and input masking, and how does prediction accuracy vary across different RUL lifecycle regions?

*Evidence: E25–E34 — Tables F10, E27–E28, E32–E33*

**RQ4 — Validation stability and held-out generalisation:**
How consistent is model performance and ranking across repeated engine-grouped validation splits, and does the held-out official endpoint evaluation support or contradict the validation-stage conclusions?

*Evidence: E05–E12, E53 — Tables F3, F4, F5, F7, F8*

---

## Answers (Informing Discussion Chapter)

**RQ1:** Engineered degradation features are highly informative. DerivedOnlyMLP achieved the lowest mean repeated-validation RMSE (14.2308 ± 0.9167) and ranked second on the official endpoint test (RMSE 12.8295). XGBoost, using the combined 57-feature set including both raw and derived features, achieved the best official endpoint result (RMSE 12.2459). Multi-view fusion was competitive but did not consistently outperform the strongest single-view or classical model.

**RQ2:** cycle_index is the dominant feature by both XGBoost feature importance and SHAP analysis. Derived rolling-statistical features of sensors 4 and 11 are the next most influential. View masking confirms that the derived view carries substantially more predictive information than the raw-sequence view in isolation.

**RQ3:** The MultiViewGRUFusion model is robust to Gaussian noise and random feature masking within the tested perturbation levels. Prediction errors are higher in the early-life RUL region (RUL > 100) and lower in the degradation-dominant region (RUL < 50). Fusion exhibits a consistent under-prediction bias. MC Dropout provides exploratory uncertainty estimates correlated with prediction error (r ≈ 0.49), though calibration is not claimed.

**RQ4:** Model ranking is not stable across repeated validation splits. DerivedOnlyMLP had the lowest mean validation RMSE across three engine-grouped splits, but XGBoost ranked first on the official endpoint test. Fusion ranked second in mean validation RMSE but fourth on the official endpoint test. This supports the conclusion that rankings are sensitive to engine cohort and evaluation population.

---

## Final Contribution Statement

This dissertation contributes a leakage-safe, reproducible, and interpretable multi-view RUL prediction study on the NASA C-MAPSS FD001 turbofan degradation dataset. The specific contributions are:

**C1 — Controlled experimental protocol:**
A multi-stage evaluation protocol combining engine-level train–validation splitting, repeated validation across three independent engine cohorts, per-model random-state reset, pre-test freeze manifesting with SHA-256 artefact integrity, and a single one-time official held-out endpoint evaluation. This protocol substantially exceeds the validation rigour of the initial seed-42 development stage and is more reproducible than single-split comparisons common in the literature.

**C2 — Multi-view RUL representation study:**
A systematic comparison of raw sensor sequences (View A), derived degradation features (View B), and their fusion in a GRU-based multi-view architecture against classical XGBoost and single-view neural alternatives. The study demonstrates that the two views carry meaningful but asymmetric information, with the derived view contributing more predictive signal.

**C3 — Negative fusion finding reported honestly:**
The multi-view fusion model was competitive during validation but did not consistently outperform the strongest single-view or classical alternatives across repeated validation splits or the official endpoint test. This negative result is reported as a valid controlled finding rather than concealed or attributed to implementation error.

**C4 — Feature-level and view-level explainability:**
Global XGBoost feature importance and SHAP analysis identify cycle_index and derived sensor statistics as the dominant predictors. View masking experiments quantify the predictive contribution of each input view. Local case analysis illustrates model behaviour for representative engines.

**C5 — Trustworthiness evidence beyond accuracy:**
Gaussian-noise and random-masking robustness studies, lifecycle-region error analysis, prediction bias characterisation, perturbation stability under repeated runs, and exploratory MC Dropout uncertainty estimation provide a multi-dimensional trustworthiness assessment that extends beyond point-prediction accuracy metrics.

**C6 — Transparency and reproducibility:**
All experimental outcomes are anchored to a 57-item evidence ledger with file paths and SHA-256 hashes. The pre-test freeze manifest and single-use official evaluation enforce a disciplined test protocol. All code artefacts are committed and tagged in version control.

---

## What This Dissertation Does Not Claim

- Fusion superiority over single-view or classical models
- Calibrated predictive uncertainty
- Generalisation beyond FD001 (single operating condition, single fault mode)
- Native multimodality (both views derive from the same sensor stream)
- Production readiness or industrial deployment validation
- Direct comparison of capped endpoint RMSE with uncapped literature benchmarks
