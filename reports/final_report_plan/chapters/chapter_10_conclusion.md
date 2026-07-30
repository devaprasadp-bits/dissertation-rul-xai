# Chapter 10 — Conclusion

---

## 10.1 What Was Built

This dissertation constructed a complete, end-to-end Remaining Useful Life prediction pipeline for the NASA C-MAPSS FD001 turbofan engine degradation benchmark. The pipeline spans dataset loading and exploratory analysis, feature engineering, leakage-safe engine-level splitting, classical and deep-learning baseline modelling, a multi-view GRU-based fusion architecture, explainability analysis, robustness and trustworthiness characterisation, repeated validation across independent engine cohorts, full-data refitting, and a one-time official endpoint evaluation.

The principal models produced are:

- **XGBoost** trained on 57 derived and raw sensor features (Feature Set C)
- **GRU** trained on 30-cycle windows of 14 raw sensor measurements
- **DerivedOnlyMLP** trained on 43 engineered degradation features
- **MultiViewGRUFusion** combining a GRU sequence encoder with a dense degradation-feature branch

All models were trained from scratch on all 100 FD001 training engines for the final evaluation, with neural training epochs frozen prior to official test access. The experimental pipeline is fully reproducible, with all artefacts committed to version control, hashed in a pre-test freeze manifest, and catalogued in a 57-item evidence ledger.

---

## 10.2 What Was Evaluated

Three layers of evidence were produced:

**Historical development stage (seed-42 split):** In the initial single-split development stage, MultiViewGRUFusion achieved the lowest validation RMSE (12.0657) among all models evaluated, with XGBoost second (12.4894). This result motivated the choice of fusion as the primary model under study.

**Repeated validation (seeds 21, 42, 84):** Three independent engine-level splits were used to test the stability of this ranking. DerivedOnlyMLP achieved the lowest mean RMSE across the three splits (14.2308 ± 0.9167). MultiViewGRUFusion ranked second by mean RMSE (14.2787) but with higher variability (SD 1.6112). Fusion won against each individual comparator in only one of the three splits. XGBoost ranked third by mean RMSE (14.4297) with the highest cohort sensitivity (SD 2.0723).

**Official endpoint test (one evaluation, 100 test engines):** XGBoost achieved the strongest official result (RMSE 12.2459, MAE 9.0155, R² 0.9066). DerivedOnlyMLP ranked second (RMSE 12.8295), GRU third (RMSE 13.2860), and MultiViewGRUFusion fourth (RMSE 13.3782). MultiViewGRUFusion also showed the largest under-prediction bias (mean error −3.20 cycles).

Alongside accuracy metrics, the dissertation evaluated: XGBoost feature importance and SHAP analysis, view masking sensitivity, local prediction cases, Gaussian-noise and random-masking robustness, lifecycle-region error analysis, prediction bias, repeated perturbation stability, and exploratory MC Dropout uncertainty.

---

## 10.3 What Was Learned

The experimental evidence supports the following conclusions:

**Engineered degradation features are highly informative for capped RUL prediction on FD001.** Rolling means, delta-from-initial values, and the cycle index are the dominant predictors identified by both gain-based importance and SHAP analysis. A simple two-layer dense network (DerivedOnlyMLP) trained on these features achieves competitive or superior performance compared with the GRU sequence model and the multi-view fusion model across most evaluation populations.

**Multi-view fusion is competitive but not consistently superior.** The implemented GRU–MLP concatenation architecture demonstrated that the two input views carry meaningful information, but combining them did not reliably outperform the best single-view or classical alternative. The initial seed-42 result that placed fusion first was not reproduced under the broader repeated-validation and official-test protocol.

**Model rankings are sensitive to engine cohort and evaluation population.** The rank ordering changed substantially between the seed-42 development stage, the repeated-validation mean, and the official endpoint test. XGBoost ranked first, third, and second (or third depending on ordering) across the three layers, and Fusion ranked first, second, and fourth. No model maintained a consistent rank across all evaluations. This finding argues against drawing deployment conclusions from any single-split comparison.

**The derived degradation view carries the majority of the fusion model's predictive signal.** View masking confirms that complete loss of the derived view degrades fusion performance to below a constant predictor (R² −0.035), while loss of the sequence view reduces performance by approximately 29%. Attribution analyses reinforce that rolling means of sensors 4, 11, and 15, the cycle index, and delta features are the model's primary information sources.

**The model is robust to moderate noise but sensitive to derived-feature corruption.** Gaussian noise at standardised std 0.20 increases RMSE by 0.31% on the sequence view and 3.68% on the derived view. Complete masking of the derived view is catastrophic. Prediction is most accurate in the near-failure lifecycle region (RMSE 4.43 for RUL ≤ 30) and least accurate in the early or capped region (RMSE 13.53 for RUL 81–125). MC Dropout dispersion has a moderate correlation with absolute error (r ≈ 0.49), providing an exploratory uncertainty signal.

---

## 10.4 What Was Not Demonstrated

In the interest of scientific precision, this dissertation explicitly does not demonstrate:

- That multi-view fusion is superior to single-view or classical models for RUL prediction in general
- That calibrated predictive uncertainty is available from the models evaluated
- That results generalise beyond FD001 (single operating condition, single fault mode)
- That the multi-view formulation constitutes native multimodality — both views derive from the same numerical sensor stream
- That the models are ready for production deployment without further validation on field data
- That the capped endpoint RMSE reported here is directly comparable with uncapped benchmark results in the published literature
- That the negative fusion finding rules out all fusion architectures — the conclusion is specific to the GRU–MLP concatenation design evaluated here

---

## 10.5 Final Contribution

This dissertation contributes six things to the predictive-maintenance research and practice landscape:

**A controlled experimental protocol** that moves beyond single-split model comparison to include repeated engine-level validation across three independent cohorts, per-model random-state isolation, pre-test freeze manifesting with SHA-256 artefact integrity, and a single one-time official held-out evaluation. This protocol is substantially more rigorous than the single-split comparisons that dominate the prognostics benchmarking literature.

**A systematic multi-view representation study** on FD001 that compares raw sensor sequences, engineered degradation features, and their fusion under matched evaluation conditions. The study quantifies the incremental value of each view and the conditions under which their combination is and is not beneficial.

**An honest negative finding** reported as a controlled experimental result: the multi-view fusion model was competitive during validation but did not consistently outperform the strongest single-view or classical alternative. This finding is presented as evidence, not concealed or attributed to implementation failure.

**Feature-level and view-level explainability** grounded in XGBoost feature importance, SHAP analysis, view masking, and local case analysis. These analyses identify which representations drive predictions and provide interpretable characterisations of model behaviour at the feature, view, and instance level.

**A multi-dimensional trustworthiness assessment** extending beyond point-prediction accuracy: noise and masking robustness, lifecycle-region error profiling, prediction bias characterisation, perturbation stability, and exploratory MC Dropout uncertainty. These results allow a more complete assessment of the model's practical reliability than RMSE alone.

**A transparent and reproducible experimental record** comprising 12 sequenced notebooks, a 57-item evidence ledger, SHA-256 artefact manifests, and version-controlled model files. The reproducibility infrastructure ensures that each headline result can be traced to a specific output file and a specific notebook cell.

---

> The central finding is that in a controlled and reproducible multi-view RUL prediction study on NASA C-MAPSS FD001, engineered degradation features provide a strong and consistent predictive signal, while GRU-based multi-view fusion is competitive but not consistently superior. The dissertation's contribution is not a best model claim — it is a trustworthy experimental assessment of how different representations behave under a rigorous evaluation protocol.
