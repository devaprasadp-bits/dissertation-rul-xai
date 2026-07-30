# Evidence Ledger — Final Dissertation
## Dissertation: Explainable and Trustworthy Multimodal Deep Learning for Predictive Maintenance
## Student: 2023AA05069 | Created: 2026-07-26

**Rule:** Every headline number and figure in the final report must map to a saved output artefact via an entry in this ledger.

**Status values:** `Verified` = artefact exists and value confirmed | `Pending` = planned evidence whose source notebook or artefact has not yet been completed and verified | `Check` = artefact exists but path/value needs confirmation before writing to report

---

## Section 1 — Predictive Performance (Chapter 6)

| Evidence ID | Result / Figure | Source notebook | Source file (relative to project root) | Report section | Status |
|---|---|---|---|---|---|
| E01 | Seed-42 window-aligned comparison table (all 5 models) | NB06 | `reports/metrics/multiview_vs_baselines_comparison_fd001.csv` | §6.1, Table F2 | Verified |
| E01a | MultiViewGRUFusion RMSE 12.0657, MAE 8.9406, R² 0.9168 | NB06 | `reports/metrics/multiview_vs_baselines_comparison_fd001.csv` | §6.1 | Verified |
| E01b | XGBoost/C window-aligned RMSE 12.4894, MAE 9.2675, R² 0.9109 | NB06 | `reports/metrics/classical_vs_deep_window_aligned_comparison_fd001.csv` | §6.1 | Verified |
| E01c | GRU/B RMSE 13.1605, MAE 9.7182, R² 0.9010 | NB05 | `reports/metrics/deep_learning_baseline_results_fd001.csv` | §6.1 | Verified |
| E01d | DerivedOnlyMLP RMSE 13.1451, MAE 9.4204, R² 0.9013 | NB06 | `reports/metrics/multiview_deep_learning_results_fd001.csv` | §6.1 | Verified |
| E01e | CNN1D/B RMSE 18.1509, MAE 14.0382, R² 0.8118 | NB05 | `reports/metrics/deep_learning_baseline_results_fd001.csv` | §6.1 | Verified |
| E01f | XGBoost/C **full-validation** RMSE 11.7218, MAE 8.3182, R² 0.9203 — evaluated on all validation rows (not window-aligned; different sample population from E01b) | NB04 | `reports/metrics/classical_baseline_results_fd001.csv` | §6.1 note | Verified |
| E02 | RMSE comparison bar chart (multi-view vs baselines) | NB06 | `reports/figures/multiview_vs_baselines_rmse_fd001.png` | §6.1 | Verified |
| E03 | Actual vs predicted scatter — MultiViewGRUFusion | NB06 | `reports/figures/actual_vs_predicted_multiview_fd001.png` | §6.1 | Verified |
| E04 | Sample engine prediction trajectories — MultiViewGRUFusion | NB06 | `reports/figures/sample_engine_prediction_trajectories_multiview_fd001.png` | §6.1 | Verified |
| E05 | Repeated-validation split results (3 seeds × 4 models) | NB10 | `reports/final_validation/repeated_validation_split_metrics_fd001.csv` | §6.2, Table F3 | Verified |
| E06 | Repeated-validation mean ± std summary | NB10 | `reports/final_validation/repeated_validation_summary_fd001.csv` | §6.2, Table F4 | Verified |
| E07 | Fusion pairwise RMSE differences per split | NB10 | `reports/final_validation/repeated_validation_pairwise_comparison_fd001.csv` | §6.2, Table F5 | Verified |
| E07a | Fusion percentage comparisons and win counts | NB10 | `reports/final_validation/repeated_validation_fusion_comparison_fd001.csv` | §6.2, Table F5 | Verified |
| E08 | Per-engine validation metrics | NB10 | `reports/final_validation/per_engine_validation_metrics_fd001.csv` | §6.3, Table F7 | Verified |
| E09 | cycle_index ablation table | NB10 | `reports/final_validation/cycle_index_ablation_fd001.csv` | §6.4, Table F6 | Verified |
| E10 | Official test endpoint metrics (all models) | NB11 | `reports/final_test/final_test_endpoint_metrics_fd001.csv` | §6.5, Table F8 | Verified |
| E11 | Official test endpoint predictions | NB11 | `reports/final_test/final_test_endpoint_predictions_fd001.csv` | §6.5 | Verified |
| E12 | Final epoch selection table | NB11 | `reports/final_test/final_epoch_selection_fd001.csv` | §5.7 | Verified |

---

## Section 2 — Explainability (Chapter 7)

| Evidence ID | Result / Figure | Source notebook | Source file (relative to project root) | Report section | Status |
|---|---|---|---|---|---|
| E13 | XGBoost feature importance table (top 20) | NB04/NB07 | `reports/explainability/xgboost_feature_importance_fd001.csv` | §7.1 | Verified |
| E14 | XGBoost feature importance bar chart | NB07 | `reports/figures/xgboost_feature_importance_top20_fd001.png` | §7.1 | Verified |
| E15 | SHAP feature ranking table | NB07 | `reports/explainability/xgboost_shap_feature_ranking_fd001.csv` | §7.2 | Verified |
| E16 | SHAP summary plot (beeswarm or bar) | NB07 | `reports/figures/shap_summary_xgboost_fd001.png` | §7.2 | Verified |
| E17 | SHAP bar chart | NB07 | `reports/figures/shap_bar_xgboost_fd001.png` | §7.2 | Verified |
| E18 | SHAP dependence — cycle_index | NB07 | `reports/figures/shap_dependence_cycle_index_fd001.png` | §7.2 | Verified |
| E19 | SHAP dependence — sensor_measurement_4_rmean | NB07 | `reports/figures/shap_dependence_sensor_measurement_4_rmean_fd001.png` | §7.2 | Verified |
| E20 | SHAP dependence — sensor_measurement_11_rmean | NB07 | `reports/figures/shap_dependence_sensor_measurement_11_rmean_fd001.png` | §7.2 | Verified |
| E21 | View masking sensitivity table | NB07 | `reports/explainability/multiview_view_masking_explainability_fd001.csv` | §7.4 | Verified |
| E22 | View masking RMSE bar chart | NB07 | `reports/figures/multiview_view_masking_rmse_fd001.png` | §7.4 | Verified |
| E23 | Local prediction cases (units 2, 53, 22, 93) | NB07 | `reports/explainability/local_prediction_cases_multiview_fd001.csv` | §7.5 | Verified |
| E24 | Local view masking cases | NB07 | `reports/explainability/local_view_masking_cases_multiview_fd001.csv` | §7.5 | Verified |

---

## Section 3 — Trustworthiness and Robustness (Chapter 7)

| Evidence ID | Result / Figure | Source notebook | Source file (relative to project root) | Report section | Status |
|---|---|---|---|---|---|
| E25 | Robustness summary table (noise, masking, full) | NB08 | `reports/robustness/robustness_summary_multiview_fd001.csv` | §7.6 | Verified |
| E26 | Robustness RMSE comparison chart | NB08 | `reports/figures/robustness_summary_rmse_multiview_fd001.png` | §7.6 | Verified |
| E27 | RUL range-wise error analysis | NB08 | `reports/robustness/rul_range_error_analysis_multiview_fd001.csv` | §7.7 | Verified |
| E28 | RUL range error bar chart | NB08 | `reports/figures/rul_range_error_multiview_fd001.png` | §7.7 | Verified |
| E29 | Prediction bias summary | NB08 | `reports/robustness/prediction_bias_summary_multiview_fd001.csv` | §7.8 | Verified |
| E30 | Noise stability repeated runs | NB08 | `reports/robustness/repeated_noise_stability_multiview_fd001.csv` | §7.9 | Verified |
| E31 | Noise stability summary | NB08 | `reports/robustness/repeated_noise_stability_summary_multiview_fd001.csv` | §7.9 | Verified |
| E32 | MC Dropout uncertainty per-sample | NB08 | `reports/robustness/mc_dropout_uncertainty_multiview_fd001.csv` | §7.10 | Verified |
| E33 | MC Dropout uncertainty summary | NB08 | `reports/robustness/mc_dropout_uncertainty_summary_multiview_fd001.csv` | §7.10 | Verified |
| E34 | Uncertainty vs error scatter | NB08 | `reports/figures/uncertainty_vs_error_multiview_fd001.png` | §7.10 | Verified |

---

## Section 4 — Baseline and Classical Results (Chapter 6 / Appendices)

| Evidence ID | Result / Figure | Source notebook | Source file | Report section | Status |
|---|---|---|---|---|---|
| E35 | Full classical baseline results table | NB04 | `reports/metrics/classical_baseline_results_fd001.csv` | §6.1 / App E | Verified |
| E36 | Classical RMSE comparison chart | NB04 | `reports/figures/classical_baseline_rmse_comparison_fd001.png` | Appendix | Verified |
| E37 | Classical vs deep comparison (pre-alignment) | NB05 | `reports/metrics/classical_vs_deep_baseline_comparison_fd001.csv` | §6.1 note | Verified |
| E38 | Window-aligned classical vs deep | NB05 | `reports/metrics/classical_vs_deep_window_aligned_comparison_fd001.csv` | §6.1 | Verified |
| E39 | Sensitivity check — best classical hyperparams | NB04 | `reports/metrics/sensitivity_check_best_classical_fd001.csv` | Appendix | Verified |
| E40 | Window size sensitivity (CNN1D) | NB05 | `reports/metrics/window_size_sensitivity_deep_fd001.csv` | Appendix | Verified |

---

## Section 5 — Training Histories (Appendix D)

| Evidence ID | Result / Figure | Source notebook | Source file | Report section | Status |
|---|---|---|---|---|---|
| E41 | GRU training curve | NB05 | `reports/figures/training_curve_GRU_fd001.png` | Appendix D | Verified |
| E42 | CNN1D training curve | NB05 | `reports/figures/training_curve_CNN1D_fd001.png` | Appendix D | Verified |
| E43 | DerivedOnlyMLP training curve | NB06 | `reports/figures/training_curve_DerivedOnlyMLP_fd001.png` | Appendix D | Verified |
| E44 | MultiViewGRUFusion training curve | NB06 | `reports/figures/training_curve_MultiViewGRUFusion_fd001.png` | Appendix D | Verified |
| E45 | Training curves per seed (NB10) | NB10 | `reports/final_validation/split_seed_XX/training_history_*.csv` | Appendix D | Verified |

---

## Section 6 — Dataset and EDA (Chapter 3 / Appendix A–B)

| Evidence ID | Result / Figure | Source notebook | Source file | Report section | Status |
|---|---|---|---|---|---|
| E46 | RUL distribution plot | NB01/NB03 | `reports/figures/rul_distribution_fd001.png` | §3.4 | Verified |
| E47 | RUL capping comparison plot | NB03 | `reports/figures/rul_capping_comparison_fd001.png` | §3.4 | Verified |
| E48 | Sample RUL degradation curves | NB01 | `reports/figures/sample_rul_curves_fd001.png` | §3.2 | Verified |
| E49 | Sensor trends — selected sensors | NB02 | `reports/figures/sensor_trends_selected_fd001.png` | §3.3 | Verified |
| E50 | Sensor vs RUL correlation | NB02 | `reports/figures/sensor_vs_rul_selected_fd001.png` | §3.3 / §4.4 | Verified |
| E51 | Feature correlation heatmap | NB02 | `reports/figures/correlation_heatmap_fd001.png` | Appendix B | Verified |
| E52 | Cycle count distribution | NB01 | `reports/figures/cycle_count_distribution_fd001.png` | §3.2 | Verified |

---

## Section 7 — Final Consolidation (NB12 outputs — Chapter 6 tables)

| Evidence ID | Result / Figure | Source notebook | Source file | Report section | Status |
|---|---|---|---|---|---|
| E53 | Final model comparison table | NB12 | `reports/final_summary/final_model_comparison_fd001.csv` | §6.1, §6.5 | Verified |
| E54 | Research question evidence table | NB12 | `reports/final_summary/research_question_evidence_fd001.csv` | §8.1–8.4 | Verified |
| E55 | Final limitations table | NB12 | `reports/final_summary/final_limitations_fd001.csv` | §9.5 | Verified |
| E56 | Final claims and qualifications | NB12 | `reports/final_summary/final_claims_and_qualifications_fd001.csv` | §8 | Verified |

---

## Ledger summary

| Source | Verified | Pending | Check | Total |
|---|---|---|---|---|
| NB01–09 (mid-semester) | 43 | 0 | 0 | 43 |
| NB10 (repeated validation) | 7 | 0 | 0 | 7 |
| NB11 (test evaluation) | 3 | 0 | 0 | 3 |
| NB12 (consolidation) | 4 | 0 | 0 | 4 |
| **Total** | **57** | **0** | **0** | **57** |

**All 57 evidence items verified.**

---

## Pending artefact checklist (update as NB10/NB11/NB12 complete)

- [x] E05 — Repeated-validation split metrics (NB10)
- [x] E06 — Repeated-validation summary mean ± std (NB10)
- [x] E07 — Fusion pairwise comparison (NB10)
- [x] E07a — Fusion percentage comparisons and win counts (NB10)
- [x] E08 — Per-engine validation metrics (NB10)
- [x] E09 — cycle_index ablation (NB10)
- [x] E45 — Training histories per seed (NB10)

- [x] E10 — Official test endpoint metrics (NB11)
- [x] E11 — Official test predictions (NB11)
- [x] E12 — Final epoch selection (NB11)
- [ ] E53 — Final model comparison (NB12)
- [ ] E54 — Research question evidence (NB12)
- [ ] E55–E56 — Limitations and claims tables (NB12)

---

*Ledger created: 2026-07-26 | NB10 verified: 2026-07-28 | NB11 verified: 2026-07-30 | NB12 verified: 2026-07-30 | All 57 items complete*
