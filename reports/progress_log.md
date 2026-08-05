# Dissertation Progress Log

## Step 1 Completed: Dataset Understanding and Problem Formulation

Loaded FD001 train/test/RUL files. Defined column schema and ML task (RUL regression). Generated RUL labels for training data using max-cycle minus current-cycle. Constructed test RUL for all rows using provided final RUL values. Performed data quality checks (no missing values, no duplicates, 5 constant sensors identified). Created dataset summary, feature profile, correlation ranking, and train-test distribution comparison. Documented leakage prevention strategy and preliminary multi-view grouping. Noted FD001 limitation (one operating condition).

Key artefacts:
- 01_dataset_understanding_and_problem_formulation.ipynb
- dataset_summary_fd001.csv, data_quality_summary_fd001.csv, feature_profile_fd001.csv, sensor_rul_correlation_fd001.csv

Key decision:
- Engine-level splitting required to prevent temporal leakage across train/test boundaries.

---

## Step 2 Completed: EDA Deepening and Feature Selection

Completed deeper EDA on C-MAPSS FD001 to support modelling decisions. Features were categorized into constant, near-constant, and variable groups. A feature decision table was created with Keep / Review / Drop candidate status. Fourteen variable sensor measurements were selected as the filtered raw sensor feature set. Sensor-RUL correlation ranking, lifecycle trend plots, and sensor-vs-RUL plots were generated. Initial modelling implications were documented, including the need for classical baselines, nonlinear models, derived degradation features, and careful handling of operating settings in FD001.

Key artefacts:
- 02_eda_deepening_and_feature_selection.ipynb
- feature_decision_table_fd001.csv, selected_features_fd001.csv, eda_modelling_implications_fd001.csv

Key finding:
- 14 sensors showed meaningful variation and stronger association with RUL compared to constant and near-constant sensors.

---

## Step 3 Completed: Baseline-Ready Preprocessing

Applied RUL capping at 125 cycles. Performed engine-level train/val split (80/20, seed=42) to prevent temporal leakage. Fitted StandardScaler on training split only. Created three feature sets: A (24 raw features), B (14 EDA-filtered sensors), C (57 features = 14 sensors + rolling mean/std/delta for each + cycle_index). Removed normalized_cycle_age due to leakage (uses the final cycle of each unit, which would not be known at prediction time). Used cycle_index as the renamed cycle feature to avoid column collision with time_in_cycles metadata. Saved train/val/test CSVs for all three feature sets.

Key artefacts:
- 03_baseline_ready_preprocessing.ipynb
- train/val/test CSVs for feature sets A, B, C
- feature_sets_fd001.txt, train_val_split_fd001.csv

Key decision:
- normalized_cycle_age removed (leakage). cycle_index used as the temporal position feature instead.

---

## Step 4 Completed: Classical Baseline Modelling

Trained five classical baseline models (DummyMean, Ridge, RandomForest, GradientBoosting, XGBoost) across Feature Sets A, B, and C. Feature Set C, which includes derived degradation features, performed clearly better than raw Feature Sets A and B in the validation experiments. The best initial model was XGBoost on Feature Set C with validation RMSE 11.72, MAE 8.32, and R² 0.92. A small sensitivity check further improved RMSE slightly to 11.68.

GradientBoosting and RandomForest on Feature Set C also performed well, with RMSE values of 12.37 and 12.82 respectively. Feature Sets A and B showed similar validation performance, indicating that removing constant and near-constant sensors did not materially affect the baseline result for FD001. Ridge Regression on Feature Set C achieved RMSE 15.77, which was better than tree-based models on raw feature sets, suggesting that the derived degradation features carry useful signal even for a linear model.

Key artefacts:
- 04_classical_baseline_modelling.ipynb
- classical_baseline_results_fd001.csv, sensitivity_check_best_classical_fd001.csv, classical_feature_importance_fd001.csv
- 15 saved model files (models/classical/)
- 15 prediction files (reports/predictions/)
- Comparison plots, actual-vs-predicted, error distribution, engine trajectory, feature importance figures

Key findings:
- Derived degradation features (Set C) reduced validation RMSE by approximately 40% compared to the best raw-feature baselines.
- Even Ridge on Set C outperformed tree models on Sets A/B, suggesting that the engineered features carry strong signal.
- A small sensitivity check showed only modest improvement over the initial XGBoost configuration.

---

## Step 5 Completed: Deep Learning Baseline Modelling

Trained two sequence-based deep learning models (1D-CNN and GRU) on Feature Set B using time-windowed inputs (window size 30 cycles). Each input sample consists of 14 sensor measurements over 30 consecutive cycles, with the capped RUL at the final cycle as the target. The train/val split remained engine-level (80/20, seed=42), consistent with prior steps.

GRU outperformed CNN1D on the validation set: RMSE 13.16, MAE 9.72, R² 0.901. CNN1D achieved RMSE 18.11, MAE 14.01, R² 0.813. Both models fell short of the XGBoost/Set C classical baseline (RMSE 11.72, R² 0.920). This is understandable at this stage: the classical model benefits from explicitly engineered rolling degradation statistics, while the deep learning models received raw sensor windows and had to learn those trends implicitly.

A window size sensitivity check using CNN1D (ws=15, 30, 50) showed diminishing returns beyond window size 30. The GRU mean prediction error was -1.33 cycles, indicating a slight under-prediction bias.

A window-aligned comparison was also performed. The deep learning models are evaluated only on rows with a complete 30-cycle window history, which excludes the first 29 cycles of each validation engine (580 rows, 13.5% of the validation set). XGBoost was re-evaluated on the same subset: RMSE 12.49, MAE 9.27, R² 0.911. GRU on the same basis: RMSE 13.16, MAE 9.72, R² 0.901. The gap narrows slightly but XGBoost still leads.

Key artefacts:
- 05_deep_learning_baseline_colab.ipynb
- deep_learning_baseline_results_fd001.csv, classical_vs_deep_baseline_comparison_fd001.csv, classical_vs_deep_window_aligned_comparison_fd001.csv, window_size_sensitivity_deep_fd001.csv
- CNN1D_B_window30_fd001.keras, GRU_B_window30_fd001.keras
- Prediction files and figures (training curves, actual-vs-predicted, error distribution, engine trajectories, RMSE comparison)

Key findings:
- GRU captured temporal degradation patterns more effectively than CNN1D on this dataset.
- Deep learning baselines on raw sensor windows did not exceed the classical baseline built on engineered features, which is consistent with expectations at this stage.
- Window size 30 is a reasonable choice for CNN1D; extending to 50 gave negligible improvement.
- On a window-aligned basis, the XGBoost/GRU gap narrows from 2.0 to 0.7 RMSE points, but the ordering holds.
- These results establish the deep learning reference point before developing the multi-view model in subsequent steps.

---

## Step 6 Completed: Initial Multi-View Deep Learning Model

Implemented an initial multi-view deep learning model combining two views: a raw sensor sequence view (Feature Set B, 30-cycle window, 14 sensors) processed by a GRU encoder, and a derived degradation feature view (43 features: rolling mean, rolling std, delta from initial, cycle_index) processed by an MLP branch. The two representations are concatenated and passed through dense layers to predict capped RUL.

A derived-only MLP was also trained as an ablation model to evaluate the degradation feature view independently.

Results on window-aligned validation set:

| Model               | Views                                     | RMSE    | MAE    | R²     |
|---------------------|-------------------------------------------|---------|--------|--------|
| MultiViewGRUFusion  | sensor sequence + derived degradation     | 12.0657 | 8.9406 | 0.9168 |
| XGBoost/C           | engineered features (classical baseline)  | 12.4894 | 9.2675 | 0.9109 |
| DerivedOnlyMLP      | derived degradation only                  | 13.1451 | 9.4204 | 0.9013 |
| GRU/B               | sensor sequence only (Step 5 baseline)    | 13.1605 | 9.7182 | 0.9010 |
| CNN1D/B             | sensor sequence only (Step 5 baseline)    | 18.1509 | 14.038 | 0.8118 |

The fusion model achieved the lowest validation RMSE among all evaluated models, including the classical XGBoost baseline. This supports the multi-view formulation: combining raw sensor sequence information with derived degradation features improves over either view alone and over the best classical model.

Key artefacts:
- 06_initial_multiview_deep_learning_model.ipynb
- multiview_deep_learning_results_fd001.csv, multiview_vs_baselines_comparison_fd001.csv
- MultiViewGRUFusion_window30_fd001.keras, DerivedOnlyMLP_window30_fd001.keras
- Prediction files and figures (training curves, actual-vs-predicted, error distribution, engine trajectories, RMSE comparison)

Key findings:
- MultiViewGRUFusion (RMSE 12.07) outperformed XGBoost/C (RMSE 12.49) and GRU/B (RMSE 13.16) on the window-aligned validation set.
- DerivedOnlyMLP (RMSE 13.15) matched GRU/B, indicating that the derived feature view carries comparable signal to the raw sensor sequence view independently.
- The fusion of both views provided the best validation result, suggesting that the two views are complementary.
- This is the first result in the dissertation where a deep learning model achieved lower validation RMSE than the window-aligned classical baseline.

---

## Step 7 Completed: Initial Explainability Analysis

Applied initial explainability analysis to the best classical baseline (XGBoost/C) and the multi-view fusion model (MultiViewGRUFusion).

**XGBoost Feature Importance and SHAP (Global):**

Native feature importance identified rolling mean features as the dominant predictors: `sensor_measurement_4_rmean` (0.276), `sensor_measurement_11_rmean` (0.214), and `sensor_measurement_15_rmean` (0.179) together accounted for over 66% of total importance.

SHAP analysis on 1000 validation samples confirmed this ranking. Top 3 features by mean absolute SHAP value:

| Feature | Mean absolute SHAP |
|---------|------------|
| sensor_measurement_4_rmean | 6.65 |
| sensor_measurement_11_rmean | 5.53 |
| cycle_index | 5.35 |

SHAP dependence plots for the top 3 features were generated to inspect directional contributions to predicted RUL.

**View Masking Explainability (MultiViewGRUFusion):**

| Condition | RMSE | MAE | R² |
|-----------|------|-----|----|
| Both views available | 12.0657 | 8.9406 | 0.9168 |
| Sensor sequence masked (derived only) | 15.5505 | 12.6154 | 0.8618 |
| Derived view masked (seq only) | 42.5692 | 36.1661 | -0.0354 |

Masking the sensor sequence view increased RMSE by 3.5 points. Masking the derived degradation view collapsed model performance (RMSE 42.57, R² −0.04), indicating that the derived feature view carries the dominant predictive signal in the fusion model. The sensor sequence view provides complementary information that improves accuracy when both views are combined.

**Local Case Analysis:**

Four representative predictions were inspected: good prediction at capped boundary (unit 2, cycle 30, actual=125, pred=125.0), a mid-range good prediction away from the capped boundary (unit 53, cycle 154, actual=41, pred=41.01, error=0.007), largest under-prediction (unit 22, cycle 61, error=−50.21), and largest over-prediction (unit 93, cycle 112, error=+52.82). Per-case view masking showed that the derived view is the primary driver in all cases, with sensor masking having a smaller impact.

Key artefacts:
- 07_initial_explainability_analysis.ipynb
- xgboost_feature_importance_fd001.csv, xgboost_shap_feature_ranking_fd001.csv
- multiview_view_masking_explainability_fd001.csv, local_view_masking_cases_multiview_fd001.csv, local_prediction_cases_multiview_fd001.csv
- Figures: xgboost_feature_importance_top20, shap_summary, shap_bar, shap_dependence (×3), multiview_view_masking_rmse

Key findings:
- Rolling mean degradation features dominate XGBoost predictions; raw sensor values and rolling std features contribute far less.
- SHAP and native importance rankings are broadly consistent, with sensor_measurement_4_rmean and sensor_measurement_11_rmean as the top two features under both methods.
- The derived degradation view is the primary driver of the fusion model's predictive performance; masking it degrades R² to near zero.
- The sensor sequence view provides complementary signal that improves RMSE by ~3.5 points when combined with the derived view.
- These results are model-level explanations and do not imply physical causality.

---

## Step 8 Completed: Initial Robustness and Trustworthiness Checks

Evaluated the MultiViewGRUFusion model under controlled robustness conditions including Gaussian noise injection, random feature masking, missing-view conditions, RUL range-wise error analysis, prediction bias analysis, repeated perturbation stability, and MC Dropout uncertainty estimation.

**Baseline:** RMSE 12.0657, MAE 8.9406, R² 0.9168, mean error −2.28 cycles (slight under-prediction tendency).

**Noise robustness:** The model showed strong resilience to sensor sequence noise. At noise_std=0.20, RMSE increased by only 0.31% (RMSE 12.10). Derived feature noise caused more degradation: at noise_std=0.20, RMSE increased by 3.68% (RMSE 12.51). Combined noise at the same level produced a 4.35% increase (RMSE 12.59).

**Random masking:** Sensor sequence masking at 30% raised RMSE by 3.95% (RMSE 12.54). Derived feature masking at 30% raised RMSE by 47.16% (RMSE 17.76), again confirming higher sensitivity in the derived view. Combined masking at 30% produced the largest degradation among non-view-masking conditions (RMSE 19.05, +57.85%).

**View masking (from Step 7):** Masking the sensor sequence view raised RMSE to 15.55 (+28.88%). Masking the derived view collapsed performance to RMSE 42.57 (+252.81%, R² −0.04).

**RUL range-wise error:**

| RUL range | Samples | RMSE | MAE | Mean error |
|-----------|---------|------|-----|------------|
| near_failure (0–30) | 620 | 4.43 | 3.31 | +2.21 |
| mid_degradation (31–80) | 1000 | 12.06 | 8.96 | +4.24 |
| early_or_capped (81–125) | 2091 | 13.53 | 10.60 | −6.72 |

Near-failure predictions are the most accurate. The model tends to over-predict in the near-failure and mid-degradation ranges, and under-predict in the early/capped range.

**Prediction bias:** 56.5% of predictions were under-predictions. 40.0% fell within 5 cycles, 64.3% within 10 cycles, 90.5% within 20 cycles.

**Repeated perturbation stability (5 seeds, noise_std=0.1):** RMSE std was 0.009 for sensor sequence noise, 0.027 for derived feature noise, and 0.034 for combined noise — indicating stable behaviour across random seeds.

**MC Dropout uncertainty:** Mean prediction std was 7.47 cycles across 30 forward passes. Correlation between prediction uncertainty and absolute error was 0.49, suggesting the uncertainty signal has moderate alignment with actual errors.

Key artefacts:
- 08_initial_robustness_trustworthiness_checks.ipynb
- robustness_summary_multiview_fd001.csv, rul_range_error_analysis_multiview_fd001.csv, prediction_bias_summary_multiview_fd001.csv, baseline_predictions_multiview_fd001.csv
- repeated_noise_stability_multiview_fd001.csv, repeated_noise_stability_summary_multiview_fd001.csv
- mc_dropout_uncertainty_multiview_fd001.csv, mc_dropout_uncertainty_summary_multiview_fd001.csv
- Figures: robustness_summary_rmse_multiview_fd001.png, rul_range_error_multiview_fd001.png, uncertainty_vs_error_multiview_fd001.png

Key findings:
- The model showed resilience to sensor sequence perturbations but was more sensitive to derived feature perturbations, consistent with the explainability results from Step 7.
- Near-failure predictions (RUL 0–30) were the most accurate (RMSE 4.43), while the early/capped region showed higher error and a systematic under-prediction bias.
- 90.5% of predictions fell within 20 cycles of the actual RUL.
- Repeated perturbation stability analysis showed low RMSE variance across random seeds, indicating stable model behaviour.
- MC Dropout uncertainty showed moderate correlation (0.49) with absolute error, providing an initial uncertainty signal.
- These checks provide model-level trustworthiness evidence. They do not prove operational reliability.

---

## Step 9 Completed: Consolidated Results Summary and Mid-Sem Evidence Pack

Consolidated all experimental outputs from Steps 1–8 into a structured report-ready evidence pack. No new modelling experiments were introduced. The step loaded saved metrics, explainability outputs, and robustness results, and produced 12 summary artefacts.

**Consolidated model comparison (window-aligned validation basis):**

| Model | Input representation | RMSE | MAE | R² |
|-------|----------------------|------|-----|----|
| MultiViewGRUFusion | sensor sequence + derived degradation | 12.0657 | 8.9406 | 0.9168 |
| XGBoost | Feature Set C | 12.4894 | 9.2675 | 0.9109 |
| DerivedOnlyMLP | derived degradation only | 13.1451 | 9.4204 | 0.9013 |
| GRU | Feature Set B (window=30) | 13.1605 | 9.7182 | 0.9010 |
| CNN1D | Feature Set B (window=30) | 18.1509 | 14.0382 | 0.8118 |

All 10 selected report figures were verified on disk. The asset checklist confirmed all mid-sem inputs are ready.

Key artefacts:
- 09_consolidated_results_summary_and_midsem_assets.ipynb
- consolidated_model_comparison_fd001.csv, best_model_summary_fd001.csv
- stepwise_progress_summary_fd001.csv, dissertation_objective_coverage_fd001.csv
- explainability_summary_fd001.csv, trustworthiness_summary_fd001.csv, robustness_report_table_fd001.csv
- top_shap_features_fd001.csv, key_findings_fd001.csv, limitations_and_next_steps_fd001.csv
- selected_midsem_figures_fd001.csv, midsem_asset_checklist_fd001.csv

Key outputs:
- Best current model: MultiViewGRUFusion, RMSE 12.07, R² 0.917, on window-aligned validation.
- 8 key findings documented covering feature engineering, model comparison, explainability, robustness, and uncertainty.
- 6 limitations documented with mitigations, including FD001-only scope, benchmark-driven multi-view formulation, and uncalibrated MC Dropout.
- All dissertation title components mapped to generated evidence.
- Mid-sem report asset checklist: all 11 items ready.

---

## Step 10 Completed: Repeated Validation Across Engine Cohorts

Retrained the four principal models (XGBoost, GRU, DerivedOnlyMLP, MultiViewGRUFusion) across three engine-level validation splits (seeds 21, 42, 84) with a fixed model seed of 42. Each split used 80 training engines and 20 validation engines. Neural-model epoch counts were not fixed in advance; each run used early stopping. The median best epoch across the three splits was computed per model and saved as the frozen epoch count for Stage 3.

**Repeated validation results (window-aligned, per split):**

| Split seed | Lowest-RMSE model | RMSE |
|---|---|---:|
| 21 | DerivedOnlyMLP | 14.02 |
| 42 | XGBoost | 12.53 |
| 84 | GRU | 14.87 |

No model ranked first across all three splits.

**Mean RMSE summary across three splits:**

| Model | Mean RMSE | SD RMSE |
|---|---:|---:|
| DerivedOnlyMLP | 14.2308 | 0.92 |
| MultiViewGRUFusion | 14.2787 | 1.61 |
| XGBoost | 14.4297 | 2.07 |
| GRU | 14.4671 | 0.42 |

**Frozen epoch counts selected (median across splits):** GRU = 12, DerivedOnlyMLP = 59, MultiViewGRUFusion = 12. XGBoost has no epoch parameter.

**cycle_index ablation (seed-42 split):** Removing cycle_index increased RMSE by 16.3% for DerivedOnlyMLP, 15.0% for MultiViewGRUFusion, and 11.4% for XGBoost.

A seed-42 reproduction gate confirmed that the seed-42 split results matched the historical development run within tolerance before the repeated-validation protocol proceeded.

Key artefacts:
- 10_repeated_validation_final_protocol.ipynb
- repeated_validation_split_metrics_fd001.csv, repeated_validation_summary_fd001.csv
- repeated_validation_pairwise_comparison_fd001.csv, repeated_validation_fusion_comparison_fd001.csv
- cycle_index_ablation_fd001.csv, per_engine_validation_metrics_fd001.csv
- frozen_experiment_config.json, seed42_reproduction_manifest.json
- Training history CSVs per split (split_seed_21/, split_seed_42/, split_seed_84/)
- Figures: repeated_validation_split_rmse_fd001.png, repeated_validation_mean_rmse_fd001.png

Key findings:
- Model rankings are sensitive to engine-cohort composition; no single model ranked first in all three splits.
- DerivedOnlyMLP had the lowest mean RMSE; GRU had the lowest spread (SD 0.42).
- Fusion ranked second by mean RMSE but had the highest spread after XGBoost.
- cycle_index is a material predictor in all models that use the derived view.
- Three splits provide descriptive cohort-sensitivity evidence but are insufficient for formal statistical inference.

---

## Step 11 Completed: Official Endpoint Evaluation

Refitted all four principal models using all 100 FD001 training engines and the frozen epoch counts from Step 10 (GRU = 12, DerivedOnlyMLP = 59, MultiViewGRUFusion = 12). XGBoost was retrained with the frozen configuration (n_estimators = 300, max_depth = 4, learning_rate = 0.05). Preprocessing scalers were fitted on all 100 training engines. The pre-test freeze manifest (SHA-256 hashes of models and scalers) was recorded before any test predictions were made. The official endpoint test was run once; no re-evaluation was permitted after viewing the results.

**Official endpoint test results (100 test engines, one prediction per engine):**

| Rank | Model | RMSE | MAE | R² | Mean error |
|---|---|---:|---:|---:|---:|
| 1 | XGBoost | 12.2459 | 9.0155 | 0.9066 | +0.50 |
| 2 | DerivedOnlyMLP | 12.8295 | 9.5588 | 0.8975 | −0.55 |
| 3 | GRU | 13.2860 | 9.9167 | 0.8901 | +0.76 |
| 4 | MultiViewGRUFusion | 13.3782 | 9.9372 | 0.8885 | −3.20 |

XGBoost ranked first on the official test despite ranking third by mean RMSE in repeated validation. MultiViewGRUFusion, which ranked first in the historical seed-42 development comparison, ranked fourth. The fusion model also showed the largest under-prediction bias (mean error −3.20 cycles).

Key artefacts:
- 11_official_endpoint_evaluation.ipynb
- final_test_endpoint_metrics_fd001.csv, final_test_endpoint_predictions_fd001.csv
- final_epoch_selection_fd001.csv
- pretest_freeze_manifest.json, frozen_official_test_protocol.json, official_test_evaluation_manifest.json
- Full training history CSVs for GRU, DerivedOnlyMLP, MultiViewGRUFusion
- Figures: official_test_rmse_fd001.png, official_test_actual_vs_predicted_XGBoost_fd001.png, official_test_actual_vs_predicted_MultiViewGRUFusion_fd001.png

Key findings:
- XGBoost achieved the strongest official endpoint result (RMSE 12.25, R² 0.91).
- MultiViewGRUFusion was not consistently superior; it ranked fourth on the official test.
- The ranking across the three evaluation layers (development, repeated validation, official test) was different for all four models, confirming that a single development split is insufficient for model selection.
- The pre-test freeze manifest ensures no post-test adjustments affected the reported results.

---

## Step 12 Completed: Final Consolidation and Report Asset Assembly

Consolidated all experimental results from Steps 1–11 into report-ready summary tables and figures. No new modelling experiments were introduced. The step assembled the cross-layer ranking comparison, final model comparison, research-question evidence mapping, limitations table, claims and qualifications, explainability summary, trustworthiness summary, and the final report asset checklist. The evidence ledger was completed with all 57 items verified.

**Cross-layer ranking summary:**

| Model | Stage 1 rank | Stage 2 rank (mean) | Stage 3 rank |
|---|---:|---:|---:|
| MultiViewGRUFusion | 1 | 2 | 4 |
| XGBoost | 2 | 3 | 1 |
| DerivedOnlyMLP | 3 | 1 | 2 |
| GRU | 4 | 4 | 3 |

Key artefacts:
- 12_final_consolidation_and_report_assets.ipynb
- final_model_comparison_fd001.csv, research_question_evidence_fd001.csv
- final_limitations_fd001.csv, final_claims_and_qualifications_fd001.csv
- final_explainability_table_fd001.csv, final_trustworthiness_table_fd001.csv
- cycle_index_ablation_report_fd001.csv, fusion_comparison_report_fd001.csv
- repeated_validation_report_table_fd001.csv, repeated_validation_split_report_fd001.csv
- final_test_report_table_fd001.csv, per_engine_report_table_fd001.csv
- seed42_historical_reference_fd001.csv, experimental_protocol_summary_fd001.csv
- final_report_asset_checklist_fd001.csv, final_summary_manifest.json
- Figures: validation_test_rank_comparison_fd001.png, repeated_validation_split_rmse_fd001.png, repeated_validation_mean_rmse_fd001.png

Key findings:
- All 57 evidence items verified across Sections 1–7 of the evidence ledger.
- No model ranked consistently first across all three evaluation layers.
- The central finding is that engineered degradation features provide a strong and comparatively consistent representation; fusion was competitive but not consistently superior.
- Final report asset checklist confirmed all figures, tables, and supporting CSVs are present on disk.
