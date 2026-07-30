# Chapter 7 — Explainability and Trustworthiness

The analyses in this chapter were conducted during the initial development stage (Stage 1, NB07–NB08) using the seed-42 split. All explainability results refer to models trained on the seed-42 development split unless stated otherwise. The robustness and trustworthiness checks were applied to the MultiViewGRUFusion model from that stage. These findings are not repeated across the three validation seeds because their purpose is to characterise model behaviour, not to measure prediction accuracy stability.

---

## 7.1 XGBoost Feature Importance

XGBoost provides native feature importance as the total gain accumulated across all tree splits that use a given feature. The top-20 features by importance are shown in Table 7.1 and Figure 7.1.

**Table 7.1: XGBoost feature importance — top 20 features (gain-normalised)**

| Rank | Feature | Importance |
|---|---|---|
| 1 | sensor_measurement_4_rmean | 0.2758 |
| 2 | sensor_measurement_11_rmean | 0.2137 |
| 3 | sensor_measurement_15_rmean | 0.1791 |
| 4 | sensor_measurement_2_rmean | 0.0475 |
| 5 | sensor_measurement_17_rmean | 0.0426 |
| 6 | sensor_measurement_11_delta | 0.0266 |
| 7 | sensor_measurement_9_delta | 0.0252 |
| 8 | sensor_measurement_3_rmean | 0.0227 |
| 9 | sensor_measurement_21_rmean | 0.0194 |
| 10 | cycle_index | 0.0177 |

*Source: E13 — `reports/explainability/xgboost_feature_importance_fd001.csv`*

Three rolling-mean features dominate the importance ranking: sensor_4_rmean (0.2758), sensor_11_rmean (0.2137), and sensor_15_rmean (0.1791). Together these three features account for approximately 67% of the total importance. Rolling-mean features capture the local average level of a sensor, which is a direct reflection of the degradation state. The raw sensor values and rolling-standard-deviation features collectively account for very little importance, confirming that smoothed degradation summaries are substantially more informative to the tree model than instantaneous readings or noise characterisation.

`cycle_index` ranks tenth by gain-based importance (0.0177). This relatively modest gain-based rank contrasts with its SHAP rank and its ablation impact (Section 6.4), illustrating that gain-based importance can understate the contribution of features that are important but also correlated with other features.

---

## 7.2 SHAP Analysis

SHAP (SHapley Additive exPlanations) provides a theoretically grounded feature attribution that accounts for feature interactions. Mean absolute SHAP values represent each feature's average marginal contribution to the model's output across the evaluation set. Figure 7.2 shows the beeswarm summary plot; Figure 7.3 shows the mean absolute SHAP bar chart.

### 7.2.1 SHAP Feature Ranking

**Table 7.2: SHAP feature ranking — top 15 features (mean |SHAP|)**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | sensor_measurement_4_rmean | 6.651 |
| 2 | sensor_measurement_11_rmean | 5.529 |
| 3 | cycle_index | 5.352 |
| 4 | sensor_measurement_15_rmean | 3.879 |
| 5 | sensor_measurement_11_delta | 3.026 |
| 6 | sensor_measurement_15_delta | 2.558 |
| 7 | sensor_measurement_9_delta | 2.503 |
| 8 | sensor_measurement_20_delta | 2.359 |
| 9 | sensor_measurement_14_delta | 2.217 |
| 10 | sensor_measurement_8_delta | 2.100 |
| 11 | sensor_measurement_2_rmean | 1.981 |
| 12 | sensor_measurement_4_delta | 1.905 |
| 13 | sensor_measurement_17_rmean | 1.686 |
| 14 | sensor_measurement_13_delta | 1.594 |
| 15 | sensor_measurement_9_rmean | 1.476 |

*Source: E15 — `reports/explainability/xgboost_shap_feature_ranking_fd001.csv`*

The SHAP ranking confirms the importance of rolling-mean features for sensors 4, 11, and 15, but also elevates `cycle_index` to third place (mean |SHAP| 5.352). In the SHAP analysis, `cycle_index` is nearly as impactful as sensor_11_rmean, despite ranking only tenth in gain-based importance. This is consistent with the ablation result (Section 6.4), which showed a 15–16% RMSE increase when `cycle_index` is removed. The SHAP analysis reveals that gain-based importance underestimates the contribution of `cycle_index`, likely because it correlates with delta features that partially absorb its signal in tree splits.

Delta features — which capture cumulative drift from the engine's initial state — appear strongly in the SHAP top-15 (ranks 5–14), whereas they were less prominent in gain-based importance. This reinforces the interpretation that XGBoost uses a combination of local level (rolling mean), temporal position (cycle_index), and cumulative change (delta) to estimate RUL.

Rolling standard-deviation features rank very low in both the gain and SHAP analyses, indicating that local sensor volatility is not a primary predictor in this model on FD001.

### 7.2.2 SHAP Dependence: cycle_index

The SHAP dependence plot for `cycle_index` (Figure 7.4) shows a broadly negative relationship: as `cycle_index` increases (engine ages), the SHAP contribution to predicted RUL decreases. This is the expected monotonic relationship — older engines are assigned lower RUL by the model. The relationship is approximately linear over most of the lifecycle range, with some scatter for high cycle values where the capped-RUL target flattens the gradient.

### 7.2.3 SHAP Dependence: sensor_measurement_4_rmean

The dependence plot for sensor_4_rmean (Figure 7.5) shows that higher values of this rolling average are associated with higher SHAP contributions (higher predicted RUL), while lower values indicate the model is increasing its RUL reduction. This reflects sensor 4's known degradation behaviour in FD001, where the rolling mean tends to decrease as the engine degrades toward failure.

### 7.2.4 SHAP Dependence: sensor_measurement_11_rmean

The dependence plot for sensor_11_rmean (Figure 7.6) shows a broadly positive relationship with SHAP contribution, indicating that the model interprets higher values of this rolling average as associated with more remaining life. This sensor is among the most important across both gain-based and SHAP rankings, and its rolling mean captures a smooth degradation trajectory that the model uses directly for RUL estimation.

---

## 7.3 View Masking Sensitivity

View masking tests the sensitivity of the trained MultiViewGRUFusion model to the loss of each input view. At inference time, one view is replaced with its mean value (zero in standardised space, approximating the training mean), while the other view retains its true values. The model weights are unchanged.

**Table 7.3: View masking results (MultiViewGRUFusion, seed-42 split)**

| Condition | RMSE | MAE | R² | RMSE increase |
|---|---|---|---|---|
| Both views available (baseline) | 12.0657 | 8.9406 | 0.9168 | — |
| Sensor sequence view masked | 15.5505 | 12.6154 | 0.8618 | +28.9% |
| Derived degradation view masked | 42.5692 | 36.1661 | −0.0354 | +252.8% |

*Source: E21 — `reports/explainability/multiview_view_masking_explainability_fd001.csv`*

Figure 7.7 shows the RMSE bar chart for the three masking conditions.

When the sensor sequence view is masked, performance degrades by 28.9% (RMSE 15.5505). The model retains meaningful predictive ability, indicating that the derived degradation view alone contains sufficient signal for useful RUL estimation.

When the derived degradation view is masked, performance collapses entirely (RMSE 42.5692, R² −0.0354 — below the performance of a constant predictor). This shows that the trained MultiViewGRUFusion is critically dependent on the derived degradation branch. The raw sequence branch, when operating without the derived view, cannot compensate.

This asymmetry is important context for interpreting the fusion result: the model has learned a representation that heavily leverages the derived view. The sensor sequence view contributes incrementally — the masked-sequence RMSE (15.55) is meaningfully worse than the full baseline (12.07), confirming that the raw sequence branch does add value — but the derived view is the load-bearing input. This is why the DerivedOnlyMLP (which uses only the derived view) achieves strong performance independently.

A qualification applies: view masking tests the trained fusion model's sensitivity, not the intrinsic separability of the information in each view. The GRU baseline (which uses only the sequence view and is trained independently) achieves RMSE 13.16 — substantially better than the masked-sequence fusion model (15.55). The masking result reflects the trained weight distribution of the fusion model, not an upper bound on what a dedicated raw-sequence model can achieve.

---

## 7.4 Local Prediction Cases

Table 7.4 documents four representative prediction cases drawn from the MultiViewGRUFusion validation set (seed-42 split), illustrating the range of model behaviour at the level of individual engine cycles.

**Table 7.4: Local prediction cases (MultiViewGRUFusion, seed-42 split)**

| Case type | Engine unit | Cycle | True RUL (capped) | Predicted RUL | Error |
|---|---|---|---|---|---|
| Good prediction — early lifecycle | 2 | 30 | 125 | 125.00 | 0.00 |
| Good prediction — mid-lifecycle | 53 | 154 | 41 | 41.01 | +0.01 |
| Largest over-prediction | 93 | 112 | 43 | 95.82 | +52.82 |
| Largest under-prediction | 22 | 61 | 125 | 74.79 | −50.21 |

*Source: E23 — `reports/explainability/local_prediction_cases_multiview_fd001.csv`*

Engine 2 at cycle 30 and engine 53 at cycle 154 demonstrate near-perfect predictions. The early-lifecycle prediction (engine 2) correctly assigns the capped target of 125, while the mid-lifecycle prediction (engine 53) matches the true RUL of 41 cycles almost exactly.

Engine 93 shows the largest over-prediction: at cycle 112, the true remaining life is 43 cycles, but the model predicts 95.82 — an error of +52.82 cycles. This suggests the model has not yet identified the engine's degradation rate as unusually fast relative to its cycle count.

Engine 22 shows the largest under-prediction: at cycle 61, the true RUL is at the cap (125), but the model predicts only 74.79 — an error of −50.21 cycles. The model is treating this early-lifecycle engine as more degraded than it actually is, possibly because its sensor patterns resemble those of engines with shorter remaining lives.

These cases confirm that the model's worst errors occur in the early-lifecycle and high-RUL region, consistent with the range-wise error analysis in Section 7.7.

---

## 7.5 Gaussian-Noise Robustness

Controlled Gaussian noise was added to the standardised inputs at varying levels to test the model's stability under sensor perturbation. Results are drawn from the robustness summary table.

**Table 7.5: Gaussian-noise robustness (MultiViewGRUFusion, baseline RMSE 12.0657)**

| Perturbation | Noise std | RMSE | RMSE increase | % increase |
|---|---|---|---|---|
| Sensor sequence noise | 0.05 | 12.070 | +0.005 | +0.04% |
| Sensor sequence noise | 0.10 | 12.086 | +0.021 | +0.17% |
| Sensor sequence noise | 0.20 | 12.103 | +0.037 | +0.31% |
| Derived feature noise | 0.05 | 12.092 | +0.026 | +0.22% |
| Derived feature noise | 0.10 | 12.170 | +0.104 | +0.87% |
| Derived feature noise | 0.20 | 12.510 | +0.444 | +3.68% |
| Combined noise | 0.05 | 12.104 | +0.038 | +0.31% |
| Combined noise | 0.10 | 12.212 | +0.147 | +1.22% |
| Combined noise | 0.20 | 12.591 | +0.525 | +4.35% |

*Source: E25 — `reports/robustness/robustness_summary_multiview_fd001.csv`*

Figure 7.8 shows the robustness RMSE comparison across conditions.

The model is highly stable to noise on the raw sequence view: even at noise standard deviation 0.20 in the standardised space, the RMSE increase is only 0.31%. Noise on the derived feature view causes proportionally larger degradation (3.68% at std 0.20), consistent with the view masking result that the model depends more heavily on the derived view. Combined noise at std 0.20 causes a 4.35% increase — the largest absolute degradation across single-noise conditions, but still well within a practical tolerance range.

These results suggest that moderate sensor noise does not substantially compromise the model's predictive capability. The greater sensitivity to derived-feature noise reflects the model's architectural dependence on that branch.

---

## 7.6 Random Masking Robustness

As a complementary perturbation test, random masking was applied by setting a random fraction of input values to zero (the standardised training mean) at inference time.

**Table 7.6: Random masking robustness (selected conditions)**

| Masking target | Mask rate | RMSE | % increase |
|---|---|---|---|
| Sensor sequence | 10% | 12.159 | +0.77% |
| Sensor sequence | 20% | 12.347 | +2.33% |
| Sensor sequence | 30% | 12.543 | +3.95% |
| Derived features | 10% | 13.309 | +10.3% |
| Derived features | 20% | 15.355 | +27.3% |
| Derived features | 30% | 17.756 | +47.2% |
| Combined | 10% | 13.728 | +13.8% |
| Combined | 20% | 16.184 | +34.1% |
| Combined | 30% | 19.046 | +57.9% |

*Source: E25 — `reports/robustness/robustness_summary_multiview_fd001.csv`*

The pattern mirrors the noise results: the model is comparatively tolerant of missing sequence features but degrades rapidly when derived features are masked. At 30% derived-feature masking the RMSE more than doubles the baseline increase. Combined 30% masking causes an RMSE of 19.05 (58% above baseline), reflecting the model's dependence on both views being substantially intact.

---

## 7.7 Lifecycle-Region Error Analysis

To understand where in the engine lifecycle prediction errors concentrate, samples were grouped into three RUL ranges and metrics computed within each range.

**Table 7.7: RUL range-wise error analysis (MultiViewGRUFusion, seed-42 split)**

| RUL range | Label | Sample count | RMSE | MAE | R² | Mean error |
|---|---|---|---|---|---|---|
| 0–30 cycles | Near-failure | 620 | **4.43** | 3.31 | 0.755 | +2.21 |
| 31–80 cycles | Mid-degradation | 1,000 | 12.06 | 8.96 | 0.301 | +4.24 |
| 81–125 cycles | Early or capped | 2,091 | 13.53 | 10.60 | 0.044 | −6.72 |

*Source: E27 — `reports/robustness/rul_range_error_analysis_multiview_fd001.csv`*

Figure 7.9 shows the range-wise error bar chart.

Prediction is most accurate in the near-failure region (RMSE 4.43), where the short remaining life constrains both the true label and the prediction to a narrow range. The mid-degradation region has a moderate RMSE (12.06) but a low R² (0.301), indicating that the model captures the general trend but with considerable scatter. The early or capped region — which contains the majority of samples (2,091) — shows the highest absolute RMSE (13.53) and the lowest R² (0.044). The near-zero R² in the capped region reflects the target compression effect: because all labels in this range are clipped to a maximum of 125, the target variance is low and even modest absolute errors can dominate the explained variance.

The mean error changes sign across regions: the model tends to slightly over-predict in the near-failure and mid-degradation regions (+2.21 and +4.24), and substantially under-predicts in the early or capped region (−6.72). This pattern is consistent with the model being calibrated to the degradation-active region of the lifecycle while being less well-tuned to early-lifecycle or capped-RUL assignments.

---

## 7.8 Prediction Bias

**Table 7.8: Prediction bias summary (MultiViewGRUFusion, seed-42 split)**

| Metric | Value |
|---|---|
| Mean error (prediction − actual) | −2.28 cycles |
| Median error | −1.71 cycles |
| Under-prediction rate | 56.5% |
| Over-prediction rate | 40.0% |
| Within 5 cycles | 40.0% |
| Within 10 cycles | 64.3% |
| Within 20 cycles | 90.5% |

*Source: E29 — `reports/robustness/prediction_bias_summary_multiview_fd001.csv`*

The seed-42 fusion model has a moderate aggregate under-prediction tendency: 56.5% of predictions are below the true RUL, with a mean error of −2.28 cycles. This is consistent with the negative mean errors observed across the repeated-validation splits (mean −3.52 cycles, Table 6.3) and the official endpoint test (mean −3.20 cycles, Table 6.7). The bias is therefore a systematic characteristic of this model, not an artefact of the specific validation cohort.

Approximately 90.5% of predictions fall within 20 cycles of the true RUL, and 64.3% fall within 10 cycles. The fraction within 5 cycles (40.0%) reflects the difficulty of precise RUL estimation in the degradation-active region.

The business implications of this bias depend on the maintenance cost function, which has not been modelled here. Under-prediction (predicting less remaining life than is actually available) would trigger maintenance earlier than necessary. Whether this is preferable to over-prediction depends on the relative cost of unnecessary maintenance versus failure.

---

## 7.9 Repeated Perturbation Stability

To confirm that the robustness measurements are themselves stable (not dominated by stochastic noise in the perturbation draws), the perturbation experiment was repeated five times at a fixed noise level.

**Table 7.9: Repeated perturbation stability (5 runs, noise std 0.10)**

| Perturbation type | Mean RMSE across runs | SD of RMSE |
|---|---|---|
| Sensor sequence noise | 12.079 | 0.009 |
| Derived feature noise | 12.160 | 0.027 |
| Combined noise | 12.224 | 0.034 |

*Source: E31 — `reports/robustness/repeated_noise_stability_summary_multiview_fd001.csv`*

The standard deviations across repeated runs are very small (0.009–0.034 RMSE units), confirming that the robustness findings reported in Section 7.5 are stable and not sensitive to the particular random draw of perturbation values.

---

## 7.10 MC Dropout Uncertainty

Exploratory predictive uncertainty was estimated using Monte Carlo (MC) Dropout [Gal and Ghahramani, 2016]: the MultiViewGRUFusion model's dropout layers were kept active at inference time, and 30 stochastic forward passes were performed per input sample. The standard deviation of the 30 predictions serves as a measure of predictive dispersion.

**Table 7.10: MC Dropout uncertainty summary (30 passes, seed-42 validation set)**

| Metric | Value |
|---|---|
| MC passes per sample | 30 |
| Mean predictive std across samples | 7.47 cycles |
| Median predictive std across samples | 7.72 cycles |
| Pearson correlation: dispersion vs \|error\| | 0.49 |

*Source: E33 — `reports/robustness/mc_dropout_uncertainty_summary_multiview_fd001.csv`*

Figure 7.10 shows the scatter of predictive uncertainty (std) versus absolute prediction error.

The moderate positive correlation (r ≈ 0.49) between predictive dispersion and absolute error suggests that the MC Dropout uncertainty has some practical value as a signal for identifying inputs where the model is less confident. Samples with higher predictive spread tend, on average, to have larger absolute errors.

However, this finding should be interpreted carefully. MC Dropout uncertainty is not a calibrated confidence interval: the dispersion of 30 stochastic passes reflects dropout variability under training regularisation, not a posterior distribution over predictions. The uncertainty estimates have not been calibrated against held-out data. They are reported here as exploratory evidence that the model's internal variability has a modest correspondence with prediction difficulty.

---

## 7.11 Qualification of Operational Trustworthiness

The trustworthiness analyses collectively show that the MultiViewGRUFusion model:

- is robust to small and moderate Gaussian perturbations of either input view;
- degrades gracefully under random feature masking at low mask rates, but substantially under high derived-feature masking rates;
- is critically dependent on the derived degradation view and not robust to its complete loss;
- is most accurate near the point of failure (RUL 0–30) and less accurate in the early or capped-RUL region;
- has a systematic under-prediction tendency across validation splits and the official test;
- shows moderate alignment between predictive dispersion and prediction difficulty under MC Dropout.

These properties support qualified operational trustworthiness in conditions where the derived degradation features can be reliably computed (i.e., where the sensor stream is intact), but would require further investigation before use in a safety-critical maintenance setting. Specifically, the uncalibrated uncertainty estimates, the lack of cost-sensitive evaluation, and the restriction to FD001 (single operating condition, single fault mode) are material limitations for any generalisation beyond this benchmark.

---

*Figures referenced:*
*Figure 7.1 (E14 — XGBoost feature importance bar chart), Figure 7.2 (E16 — SHAP beeswarm), Figure 7.3 (E17 — SHAP bar chart), Figure 7.4 (E18 — SHAP dependence cycle_index), Figure 7.5 (E19 — SHAP dependence sensor_4_rmean), Figure 7.6 (E20 — SHAP dependence sensor_11_rmean), Figure 7.7 (E22 — view masking RMSE bar), Figure 7.8 (E26 — robustness RMSE comparison), Figure 7.9 (E28 — RUL range error bar), Figure 7.10 (E34 — MC Dropout uncertainty scatter)*
