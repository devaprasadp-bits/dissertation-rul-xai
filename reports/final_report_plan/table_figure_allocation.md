# Table and Figure Allocation
## Final Dissertation — 2023AA05069

*Maps every NB12 table (F1–F12) and every evidence figure to the report chapter and section where it should appear. Use this as the definitive placement guide when writing.*

---

## Tables

### Chapter 5 — Experimental Methodology

| Table | Label in Report | Content | Source File | Section |
|-------|----------------|---------|-------------|---------|
| F1 | Table 5.1 | Experimental protocol summary | `reports/final_summary/experimental_protocol_summary_fd001.csv` | §5.1 |

### Chapter 6 — Predictive Performance Results

| Table | Label in Report | Content | Source File | Section |
|-------|----------------|---------|-------------|---------|
| F2 | Table 6.1 | Historical seed-42 window-aligned comparison (5 models) | `reports/metrics/multiview_vs_baselines_comparison_fd001.csv` | §6.1 |
| F3 | Table 6.2 | Repeated-validation split results (3 seeds × 4 models) | `reports/final_validation/repeated_validation_split_metrics_fd001.csv` | §6.2.1 |
| F4 | Table 6.3 | Repeated-validation mean ± SD summary | `reports/final_validation/repeated_validation_summary_fd001.csv` | §6.2.2 |
| F5 | Table 6.4 | Fusion win counts and pairwise RMSE differences | `reports/final_validation/repeated_validation_fusion_comparison_fd001.csv` | §6.2.3 |
| F6 | Table 6.5 | cycle_index ablation results | `reports/final_validation/cycle_index_ablation_fd001.csv` | §6.4 |
| F7 | Table 6.6 | Per-engine validation performance summary | `reports/final_validation/per_engine_validation_metrics_fd001.csv` | §6.3 |
| F8 | Table 6.7 | Official endpoint test results (4 models) | `reports/final_test/final_test_endpoint_metrics_fd001.csv` | §6.5 |
| E53 | Table 6.8 | Final validation-test model comparison (both populations) | `reports/final_summary/final_model_comparison_fd001.csv` | §6.6 |

### Chapter 7 — Explainability and Trustworthiness

| Table | Label in Report | Content | Source File | Section |
|-------|----------------|---------|-------------|---------|
| F9 | Table 7.1 | Explainability findings summary | `reports/final_summary/final_explainability_table_fd001.csv` | §7.1 |
| F10 | Table 7.2 | Trustworthiness findings summary | `reports/final_summary/final_trustworthiness_table_fd001.csv` | §7.5 |

### Chapter 8 — Discussion

| Table | Label in Report | Content | Source File | Section |
|-------|----------------|---------|-------------|---------|
| F11 | Table 8.1 | Research question evidence mapping | `reports/final_summary/research_question_evidence_fd001.csv` | §8.1 |

### Chapter 9 — Limitations

| Table | Label in Report | Content | Source File | Section |
|-------|----------------|---------|-------------|---------|
| F12 | Table 9.1 | Limitations and mitigations | `reports/final_summary/final_limitations_fd001.csv` | §9.1 |

### Appendix

| Table | Label in Report | Content | Source File | Location |
|-------|----------------|---------|-------------|---------|
| E56 | Table A.1 | Claims and qualifications | `reports/final_summary/final_claims_and_qualifications_fd001.csv` | Appendix F |
| — | Table C.1–C.5 | Hyperparameter tables (per model) | Internal notes | Appendix C |
| — | Table E.1 | Full classical baseline results | `reports/metrics/classical_baseline_results_fd001.csv` | Appendix E |
| — | Table E.2 | Window-size sensitivity (CNN1D) | `reports/metrics/window_size_sensitivity_deep_fd001.csv` | Appendix E |

---

## Figures

### Chapter 3 — Dataset and Problem Formulation

| Figure | Label | Content | Source File | Section |
|--------|-------|---------|-------------|---------|
| E46 | Figure 3.1 | RUL distribution | `reports/figures/rul_distribution_fd001.png` | §3.4 |
| E47 | Figure 3.2 | RUL capping comparison | `reports/figures/rul_capping_comparison_fd001.png` | §3.4 |
| E48 | Figure 3.3 | Sample RUL degradation curves | `reports/figures/sample_rul_curves_fd001.png` | §3.2 |
| E52 | Figure 3.4 | Engine cycle count distribution | `reports/figures/cycle_count_distribution_fd001.png` | §3.2 |

### Chapter 4 — Feature Engineering

| Figure | Label | Content | Source File | Section |
|--------|-------|---------|-------------|---------|
| E49 | Figure 4.1 | Sensor trends — selected sensors | `reports/figures/sensor_trends_selected_fd001.png` | §4.2 |
| E50 | Figure 4.2 | Sensor vs RUL correlation | `reports/figures/sensor_vs_rul_selected_fd001.png` | §4.4 |

### Chapter 6 — Predictive Performance Results

| Figure | Label | Content | Source File | Section |
|--------|-------|---------|-------------|---------|
| E02 | Figure 6.1 | RMSE comparison bar chart (seed-42 development) | `reports/figures/multiview_vs_baselines_rmse_fd001.png` | §6.1 |
| E03 | Figure 6.2 | Actual vs predicted scatter — Fusion (seed-42) | `reports/figures/actual_vs_predicted_multiview_fd001.png` | §6.1 |
| E04 | Figure 6.3 | Sample engine prediction trajectories — Fusion | `reports/figures/sample_engine_prediction_trajectories_multiview_fd001.png` | §6.1 |
| Fig-Final-1 | Figure 6.4 | Repeated-validation mean RMSE bar chart | `reports/final_summary/repeated_validation_mean_rmse_fd001.png` | §6.2.2 |
| Fig-Final-2 | Figure 6.5 | Repeated-validation per-split RMSE | `reports/final_summary/repeated_validation_split_rmse_fd001.png` | §6.2.1 |
| Fig-Final-3 | Figure 6.6 | Official endpoint test RMSE bar chart | `reports/final_summary/official_test_rmse_fd001.png` | §6.5 |
| Fig-Final-4 | Figure 6.7 | Actual vs predicted — XGBoost (official test) | `reports/final_summary/official_test_actual_vs_predicted_XGBoost_fd001.png` | §6.5 |
| Fig-Final-5 | Figure 6.8 | Actual vs predicted — Fusion (official test) | `reports/final_summary/official_test_actual_vs_predicted_MultiViewGRUFusion_fd001.png` | §6.5 |
| Fig-Final-6 | Figure 6.9 | Validation vs test rank comparison | `reports/final_summary/validation_test_rank_comparison_fd001.png` | §6.6 |

### Chapter 7 — Explainability and Trustworthiness

| Figure | Label | Content | Source File | Section |
|--------|-------|---------|-------------|---------|
| E14 | Figure 7.1 | XGBoost feature importance bar chart (top 20) | `reports/figures/xgboost_feature_importance_top20_fd001.png` | §7.1 |
| E16 | Figure 7.2 | SHAP summary (beeswarm) | `reports/figures/shap_summary_xgboost_fd001.png` | §7.2 |
| E17 | Figure 7.3 | SHAP bar chart | `reports/figures/shap_bar_xgboost_fd001.png` | §7.2 |
| E18 | Figure 7.4 | SHAP dependence — cycle_index | `reports/figures/shap_dependence_cycle_index_fd001.png` | §7.2.2 |
| E19 | Figure 7.5 | SHAP dependence — sensor_4_rmean | `reports/figures/shap_dependence_sensor_measurement_4_rmean_fd001.png` | §7.2.3 |
| E20 | Figure 7.6 | SHAP dependence — sensor_11_rmean | `reports/figures/shap_dependence_sensor_measurement_11_rmean_fd001.png` | §7.2.4 |
| E22 | Figure 7.7 | View masking RMSE bar chart | `reports/figures/multiview_view_masking_rmse_fd001.png` | §7.3 |
| E26 | Figure 7.8 | Robustness summary RMSE comparison | `reports/figures/robustness_summary_rmse_multiview_fd001.png` | §7.5 |
| E28 | Figure 7.9 | RUL range-wise error bar chart | `reports/figures/rul_range_error_multiview_fd001.png` | §7.7 |
| E34 | Figure 7.10 | MC Dropout uncertainty vs error scatter | `reports/figures/uncertainty_vs_error_multiview_fd001.png` | §7.10 |

### Appendix

| Figure | Label | Content | Source File | Location |
|--------|-------|---------|-------------|---------|
| E51 | Figure B.1 | Feature correlation heatmap | `reports/figures/correlation_heatmap_fd001.png` | Appendix B |
| E36 | Figure E.1 | Classical baseline RMSE comparison | `reports/figures/classical_baseline_rmse_comparison_fd001.png` | Appendix E |
| E41 | Figure D.1 | GRU training curve (seed-42) | `reports/figures/training_curve_GRU_fd001.png` | Appendix D |
| E42 | Figure D.2 | CNN1D training curve (seed-42) | `reports/figures/training_curve_CNN1D_fd001.png` | Appendix D |
| E43 | Figure D.3 | DerivedOnlyMLP training curve (seed-42) | `reports/figures/training_curve_DerivedOnlyMLP_fd001.png` | Appendix D |
| E44 | Figure D.4 | MultiViewGRUFusion training curve (seed-42) | `reports/figures/training_curve_MultiViewGRUFusion_fd001.png` | Appendix D |

---

## Caption Formatting Rules (BITS requirement)

- **Figures:** Caption goes **below** the figure. Format: *Figure N.M: Description.*
- **Tables:** Caption goes **above** the table. Format: *Table N.M: Description.*
- All figures and tables must be numbered and cited in the text before they appear.
- All figures and tables must appear in the List of Figures / List of Tables in front matter.

---

## Total Figure and Table Count

| Location | Tables | Figures |
|----------|--------|---------|
| Main chapters | 14 | 19 |
| Appendices | 4 | 6 |
| **Total** | **18** | **25** |
