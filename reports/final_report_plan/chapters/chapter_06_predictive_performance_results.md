# Chapter 6 — Predictive Performance Results

The results in this chapter are organised across three distinct evidence layers, as established in Section 3.7. These layers must not be conflated because they use different evaluation populations and serve different purposes.

**Table 6.0: Evidence layer summary**

| Layer | Population | Purpose | Report section |
|---|---|---|---|
| Historical seed-42 reference | All eligible windows, 20 validation engines, seed-42 split | Documents original model-development outcome | §6.1 |
| Repeated validation (seeds 21, 42, 84) | All eligible windows, 20 validation engines per split | Tests stability of rankings across engine cohorts | §6.2–6.3 |
| Official endpoint test | One endpoint per 100 test engines | Final held-out evaluation, conducted once | §6.5 |

---

## 6.1 Historical Seed-42 Window-Aligned Comparison

During the initial development stage (Stage 1, NB06), all five principal models were evaluated on the seed-42 validation split using window-aligned metrics — all eligible 30-cycle windows from the 20 validation engines. The results are shown in Table 6.1 and Figure 6.1.

**Table 6.1: Historical seed-42 window-aligned comparison (all models)**

| Model | Input representation | RMSE | MAE | R² |
|---|---|---|---|---|
| **MultiViewGRUFusion** | Seq (30×14) + Derived (43) | **12.0657** | **8.9406** | **0.9168** |
| XGBoost/C | Feature Set C (57 features) | 12.4894 | 9.2675 | 0.9109 |
| DerivedOnlyMLP | Derived (43) | 13.1451 | 9.4204 | 0.9013 |
| GRU/B | Raw sequence (30×14) | 13.1605 | 9.7182 | 0.9010 |
| CNN1D/B | Raw sequence (30×14) | 18.1509 | 14.0382 | 0.8118 |

*Source: E01 — `reports/metrics/multiview_vs_baselines_comparison_fd001.csv`*

In this initial development stage, MultiViewGRUFusion achieved the lowest validation RMSE (12.0657), followed by XGBoost/C (12.4894). DerivedOnlyMLP and GRU/B performed similarly at 13.14 and 13.16 respectively. CNN1D was substantially weaker and is excluded from subsequent cross-model comparisons.

These results are retained as a historical reference to document the model-development starting point. They reflect a single engine cohort (seed-42 split) evaluated under window-aligned conditions. The full repeated-validation analysis in Section 6.2 tests whether these rankings hold across different engine cohorts.

A note on the full-validation XGBoost result (E01f): on the seed-42 split evaluated over all training-set validation rows (not window-aligned), XGBoost/C achieved RMSE 11.7218. This figure is not directly comparable with the window-aligned results in Table 6.1 because it uses a different evaluation population.

Actual vs predicted scatter and sample engine prediction trajectories for MultiViewGRUFusion are shown in Figure 6.2 and Figure 6.3.

---

## 6.2 Repeated-Validation Split Results

### 6.2.1 Per-Split RMSE and MAE

Table 6.2 reports validation RMSE, MAE, and R² for each model across all three engine-grouped splits (seeds 21, 42, and 84).

**Table 6.2: Repeated-validation split results (4 models × 3 seeds)**

| Split seed | Model | RMSE | MAE | R² | Mean error |
|---|---|---|---|---|---|
| 21 | DerivedOnlyMLP | 14.0158 | 9.9778 | 0.8865 | +0.55 |
| 21 | GRU | 14.5044 | 11.1430 | 0.8785 | −1.35 |
| 21 | MultiViewGRUFusion | 14.5132 | 10.7874 | 0.8783 | −3.56 |
| 21 | XGBoost | 14.1129 | 9.9489 | 0.8850 | +0.16 |
| 42 | DerivedOnlyMLP | 13.4408 | 9.6558 | 0.8968 | −0.17 |
| 42 | GRU | 14.0285 | 10.9565 | 0.8876 | −1.48 |
| 42 | MultiViewGRUFusion | 12.5632 | 9.1574 | 0.9098 | −1.81 |
| 42 | XGBoost | 12.5341 | 9.3171 | 0.9102 | −2.26 |
| 84 | DerivedOnlyMLP | 15.2359 | 11.0725 | 0.8677 | −1.30 |
| 84 | GRU | 14.8684 | 11.5329 | 0.8740 | −2.98 |
| 84 | MultiViewGRUFusion | 15.7599 | 12.1711 | 0.8584 | −5.20 |
| 84 | XGBoost | 16.6421 | 12.2032 | 0.8421 | −3.22 |

*Source: E05 — `reports/final_validation/repeated_validation_split_metrics_fd001.csv`*

Figure 6.4 shows the mean RMSE bar chart across models. Figure 6.5 shows per-split RMSE for each model.

Several observations are evident from this table. First, there is substantial variation in absolute RMSE across splits for all models, reflecting genuine differences in the engine cohorts assigned to each split's validation set. The seed-42 split produces the strongest results for XGBoost (12.53) and MultiViewGRUFusion (12.56) because this cohort happened to be particularly well-represented by the seed-42 training engines. The seed-84 split produces weaker results for all models, suggesting that its validation engines present a more challenging prediction task.

Second, DerivedOnlyMLP shows the lowest per-split RMSE in two of the three splits (seeds 21 and 84). XGBoost leads on seed 42. MultiViewGRUFusion leads only on seed 42 as well, where it ties closely with XGBoost.

### 6.2.2 Mean and Standard Deviation Summary

Table 6.3 summarises repeated-validation performance by model, averaged across all three splits.

**Table 6.3: Repeated-validation mean ± SD summary (3 splits)**

| Rank | Model | Mean RMSE | SD RMSE | Mean MAE | Mean R² | Mean error |
|---|---|---|---|---|---|---|
| 1 | DerivedOnlyMLP | **14.2308** | 0.9167 | 10.2354 | 0.8837 | −0.31 |
| 2 | MultiViewGRUFusion | 14.2787 | 1.6112 | 10.7053 | 0.8822 | −3.52 |
| 3 | XGBoost | 14.4297 | 2.0723 | 10.4897 | 0.8791 | −1.77 |
| 4 | GRU | 14.4671 | 0.4212 | 11.2108 | 0.8800 | −1.94 |

*Source: E06 — `reports/final_validation/repeated_validation_summary_fd001.csv`*

DerivedOnlyMLP achieves the lowest mean RMSE (14.2308) across the three splits and also the lowest spread (SD 0.9167), indicating the most consistent behaviour across engine cohorts among the four models. MultiViewGRUFusion ranks second by mean RMSE (14.2787) but shows considerably higher spread (SD 1.6112), reflecting its strong performance on seed 42 and weaker performance on seeds 21 and 84. XGBoost has the highest spread (SD 2.0723), driven by its notably weak result on seed 84 (RMSE 16.64). GRU shows the lowest spread of all (SD 0.4212) but has the highest mean RMSE.

It is important to note that three splits do not constitute sufficient evidence for formal statistical inference about model superiority. These summaries are descriptive: they indicate the direction and magnitude of differences observed across the three cohorts used in this study.

### 6.2.3 Fusion Win Counts and Pairwise Differences

Table 6.4 shows, for each split, whether MultiViewGRUFusion achieved a lower RMSE than each comparator and by what percentage.

**Table 6.4: Fusion vs comparators — pairwise RMSE comparison**

| Split seed | vs XGBoost | vs GRU | vs DerivedOnlyMLP | Fusion wins (of 3) |
|---|---|---|---|---|
| 21 | −2.84% (Fusion worse) | −0.06% (Fusion worse) | −3.55% (Fusion worse) | 0 |
| 42 | −0.23% (Fusion worse) | +10.45% (Fusion better) | +6.53% (Fusion better) | 2 |
| 84 | +5.30% (Fusion better) | −6.00% (Fusion worse) | −3.44% (Fusion worse) | 1 |
| **Total** | **1/3** | **1/3** | **1/3** | — |

*Source: E07a — `reports/final_validation/repeated_validation_fusion_comparison_fd001.csv`*

MultiViewGRUFusion wins against each individual comparator in only one of the three splits. Across all nine pairwise comparisons (3 splits × 3 comparators), Fusion wins 3 times and loses 6 times. This is the primary evidence for **Outcome C**: the fusion model does not demonstrate consistent superiority over the strongest single-view or classical alternatives.

---

## 6.3 Per-Engine Validation Analysis

Table 6.5 summarises per-engine RMSE statistics to characterise the distribution of prediction quality across individual validation engines. Full per-engine results are provided in Appendix E.

**Table 6.5: Per-engine RMSE summary across all splits (4 models)**

| Model | Min engine RMSE | Median engine RMSE | Max engine RMSE | % engines with RMSE < 15 |
|---|---|---|---|---|
| XGBoost | ~5.1 (engine 34, seed 84) | ~11.7 | ~32.2 (engine 57) | ~73% |
| DerivedOnlyMLP | ~6.3 (engine 79, seed 84) | ~11.7 | ~33.1 (engine 57) | ~72% |
| GRU | ~7.4 (engine 78, seed 84) | ~13.0 | ~24.8 (engine 57) | ~65% |
| MultiViewGRUFusion | ~6.8 (engine 43, seed 84) | ~11.8 | ~27.0 (engine 57) | ~68% |

*Source: E08 — `reports/final_validation/per_engine_validation_metrics_fd001.csv`*

A notable outlier is engine 57, which consistently produces the highest per-engine RMSE across all models and all splits. This engine appears to have an unusual degradation trajectory that all models struggle to predict accurately. Engine 69 also produces elevated RMSE due to its long cycle history (333 cycles) and the resulting challenge of predicting its early-lifecycle (high-RUL) windows.

Per-engine analysis also reveals that the mean error (prediction − actual) is negative for most engines in the fusion model across seeds 21 and 84, confirming the systematic under-prediction tendency observed in the summary statistics.

---

## 6.4 cycle_index Ablation Results

Table 6.6 reports the effect of removing `cycle_index` from the feature set. The ablation was conducted on the seed-42 split only.

**Table 6.6: cycle_index ablation results (seed-42 final-protocol split)**

| Model | RMSE with cycle_index | RMSE without cycle_index | RMSE increase | % increase |
|---|---|---|---|---|
| DerivedOnlyMLP | 13.4408 | 15.6341 | +2.1933 | +16.3% |
| MultiViewGRUFusion | 12.5632 | 14.4477 | +1.8845 | +15.0% |
| XGBoost | 12.5341 | 13.9673 | +1.4332 | +11.4% |

*Source: E09 — `reports/final_validation/cycle_index_ablation_fd001.csv`*

Removing `cycle_index` causes a substantial RMSE increase for all three models. The effect is largest for DerivedOnlyMLP (+16.3%), which relies exclusively on the derived feature view and therefore depends more heavily on the temporal-position signal. XGBoost shows the smallest increase (+11.4%), reflecting its ability to use raw sensor values as partial substitutes for the temporal position information.

These results confirm that `cycle_index` is a material predictor in all three models that use derived features. The ablation is descriptive: the magnitude of the increase reflects the combined effect of removing temporal-position information and any correlated degradation patterns captured through `cycle_index`.

---

## 6.5 Official Endpoint Test Results

The official FD001 endpoint evaluation was conducted once, under the pre-registered protocol described in Section 5.12–5.13. Each model produced one prediction per test engine (100 predictions total). Results are shown in Table 6.7 and Figure 6.6.

**Table 6.7: Official endpoint test results (capped-RUL formulation)**

| Rank | Model | RMSE | MAE | Median AE | R² | Mean error |
|---|---|---|---|---|---|---|
| **1** | **XGBoost** | **12.2459** | **9.0155** | **6.1492** | **0.9066** | **+0.50** |
| 2 | DerivedOnlyMLP | 12.8295 | 9.5588 | 7.1031 | 0.8975 | −0.55 |
| 3 | GRU | 13.2860 | 9.9167 | 7.4554 | 0.8901 | +0.76 |
| 4 | MultiViewGRUFusion | 13.3782 | 9.9372 | 6.8791 | 0.8885 | −3.20 |

*Source: E10 — `reports/final_test/final_test_endpoint_metrics_fd001.csv`*

XGBoost achieves the strongest result on the official endpoint evaluation across all primary metrics: RMSE, MAE, median absolute error, and R². The RMSE gaps relative to XGBoost are:

- DerivedOnlyMLP: 4.77% higher RMSE
- GRU: 8.49% higher RMSE
- MultiViewGRUFusion: 9.25% higher RMSE

XGBoost and DerivedOnlyMLP are both relatively unbiased (mean errors +0.50 and −0.55 respectively). GRU is also close to zero (+0.76). MultiViewGRUFusion shows a notable under-prediction bias (mean error −3.20 cycles), consistent with the negative mean error observed in repeated validation (−3.52 cycles, Table 6.3).

Figures 6.7 and 6.8 show actual vs predicted scatter for XGBoost and MultiViewGRUFusion on the official test, illustrating the difference in bias between the two models.

These results are reported as official FD001 endpoint evaluation under the capped-RUL formulation (cap 125, predictions clipped to [0, 125]). They are not directly comparable with published results using uncapped official RUL targets or the asymmetric NASA scoring function.

---

## 6.6 Comparison of Rankings Across Evaluation Populations

Table 6.8 places all three evidence layers side by side to illustrate how rankings vary with the evaluation population.

**Table 6.8: Model rankings across evaluation populations**

| Model | Seed-42 dev rank (RMSE) | Repeated-val rank (mean RMSE) | Official test rank (RMSE) |
|---|---|---|---|
| XGBoost | 2 (12.49) | 3 (14.43) | **1 (12.25)** |
| DerivedOnlyMLP | 3 (13.15) | **1 (14.23)** | 2 (12.83) |
| GRU | 4 (13.16) | 4 (14.47) | 3 (13.29) |
| MultiViewGRUFusion | **1 (12.07)** | 2 (14.28) | 4 (13.38) |

*Source: E53 — `reports/final_summary/final_model_comparison_fd001.csv`*

Figure 6.9 presents this rank comparison visually.

The ranking changes materially across populations. MultiViewGRUFusion, which ranked first in the initial seed-42 development, ranks last on the official endpoint test. XGBoost, which ranked third in repeated validation, ranks first on the official endpoint test. DerivedOnlyMLP is the most consistent performer: it ranks first or second across all three populations.

---

## 6.7 Interpretation: Outcome C and Outcome D

**Outcome C — Fusion is not consistently superior**

The primary finding of the repeated-validation and official-test stages is that MultiViewGRUFusion does not demonstrate consistent superiority over the strongest single-view or classical alternatives. Across the three repeated-validation splits, Fusion wins against each comparator in only one split out of three. On the official endpoint test, Fusion ranks last. The initial seed-42 result that placed Fusion first was not reproduced under the broader validation protocol.

This is not a failed result. It is a controlled negative experimental finding: the multi-view fusion architecture is competitive and demonstrates that both input views carry meaningful information, but the current architecture does not reliably leverage that information to outperform the best single-view or classical alternative.

**Outcome D — Model ranking depends on engine cohort and evaluation population**

A secondary finding is that model rankings are sensitive to which engines are used for evaluation. The shift in XGBoost's rank from third (repeated validation) to first (official test) illustrates this sensitivity. The difference between repeated-validation RMSE and official-test RMSE should not be interpreted as a generalisation gap, because the two populations are fundamentally different: repeated validation evaluates all eligible windows from the validation split, while the official test evaluates one endpoint prediction per test engine. The comparison is descriptive only.

---

*Figures referenced:*
*Figure 6.1 (E02 — RMSE bar chart, seed-42), Figure 6.2 (E03 — actual vs predicted, Fusion, seed-42), Figure 6.3 (E04 — prediction trajectories), Figure 6.4 (Fig-Final-1 — repeated-val mean RMSE), Figure 6.5 (Fig-Final-2 — per-split RMSE), Figure 6.6 (Fig-Final-3 — official test RMSE bar), Figure 6.7 (Fig-Final-4 — actual vs predicted, XGBoost, official test), Figure 6.8 (Fig-Final-5 — actual vs predicted, Fusion, official test), Figure 6.9 (Fig-Final-6 — rank comparison)*
