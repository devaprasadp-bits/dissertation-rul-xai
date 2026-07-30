# Chapter 8 — Discussion

This chapter interprets the experimental findings presented in Chapters 6 and 7 in relation to the four research questions. It also examines the broader implications of the results and reflects on what the experimental design contributes beyond the headline accuracy numbers.

---

## 8.1 RQ1: Raw Sequences vs Engineered Features vs Fusion

**Question:** How do raw sensor sequences, engineered degradation features, and their fusion compare for capped RUL prediction on FD001?

**Answer:** Engineered degradation features provide a strong and consistent predictive representation. Multi-view fusion is competitive on some engine cohorts but does not consistently outperform the best single-view or classical alternative.

The most informative finding is that DerivedOnlyMLP — trained exclusively on 43 engineered degradation features — achieved the lowest mean RMSE across the three repeated-validation splits (14.2308) and ranked second on the official endpoint test (12.8295). XGBoost — trained on a 57-feature set combining the same derived features with the 14 raw sensor values — ranked third in repeated validation but first on the official endpoint test (12.2459). GRU — using only the 30-cycle raw sequence — showed the most consistent but weakest validation performance (mean RMSE 14.4671).

These results establish that the engineered degradation features — rolling means, delta-from-initial, and cycle_index — encode the dominant predictive signal in this dataset and formulation. A model trained only on these features (DerivedOnlyMLP) is competitive with or superior to both the raw-sequence model and the fusion model on most evaluation populations. The addition of raw sensor values in XGBoost's Feature Set C provides a further improvement on the official test, suggesting that the raw readings carry complementary signal that the derived features do not fully capture — but only when processed by the tree model rather than the GRU encoder.

The performance of MultiViewGRUFusion is more nuanced. In the initial seed-42 development stage, Fusion ranked first (RMSE 12.0657), suggesting that the combination of the GRU sequence encoder and the derived-feature dense branch can leverage both views effectively. However, this result did not replicate consistently across repeated validation (Fusion ranked second by mean RMSE but showed higher variance than DerivedOnlyMLP) and reversed on the official endpoint test (Fusion ranked fourth). The discussion of why fusion's performance is variable appears in Section 8.5.

---

## 8.2 RQ2: Contribution of Each Input View

**Question:** What predictive contribution does each view provide, and which derived features most strongly influence predictions?

**Answer:** The derived degradation view carries the majority of the fusion model's predictive information. The raw sensor sequence view provides a complementary contribution, but the trained fusion model depends critically on the derived view. Among individual features, rolling means of key sensors and cycle_index are the most influential.

View masking provides the clearest evidence of asymmetric view contribution. When the derived view is masked (replaced with the training mean), the fusion model's RMSE rises from 12.07 to 42.57 — an increase of 253% and an R² of −0.035, below the constant predictor. When the sequence view is masked, RMSE rises from 12.07 to 15.55 — a 29% increase that retains useful predictive ability. The fusion model therefore has a strong architectural dependency on the derived degradation branch.

This finding is consistent with the SHAP analysis. The top SHAP features for XGBoost are sensor_4_rmean, sensor_11_rmean, and cycle_index — all derived features — with mean |SHAP| values of 6.65, 5.53, and 5.35 respectively. Rolling standard-deviation features contribute negligibly. The gain-based importance ranking is dominated by the same rolling-mean features, though it underestimates cycle_index relative to SHAP.

The cycle_index ablation (Section 6.4) quantifies the contribution of the temporal-position feature specifically: removing it causes RMSE increases of 16.3% (DerivedOnlyMLP), 15.0% (MultiViewGRUFusion), and 11.4% (XGBoost). The large impact of a single feature underscores how much of the RUL signal is tied to knowing where in the engine's lifecycle the current observation falls. This is not a surprising finding — cycle number is the most direct proxy for remaining life — but it confirms that the model is not relying solely on the physical degradation signatures in the sensor readings.

The observation that the raw sensor sequence view provides only incremental gain over the derived-view branch raises a question for future architectures: could a more expressive sequence encoder (e.g., transformer, multi-head attention) extract more useful signal from the raw sequence? In the current GRU-based design, the raw sequence branch's contribution is real but modest compared with the derived branch.

---

## 8.3 RQ3: Robustness and Lifecycle Behaviour

**Question:** How does the fusion model behave under controlled noise, masking, missing-view conditions, and different RUL regions?

**Answer:** The model is robustly stable to small and moderate noise on either view independently, but degrades substantially under high derived-feature masking. Prediction accuracy is best in the near-failure lifecycle region and weakest in the early or capped-RUL region.

The Gaussian-noise robustness results are encouraging: at standardised noise std 0.20 — a relatively large perturbation in the scaled feature space — RMSE increases are 0.31% (sequence only), 3.68% (derived only), and 4.35% (combined). For practical sensor noise of moderate magnitude, these increases are within an acceptable range for many predictive-maintenance applications.

The higher sensitivity to derived-feature noise compared with sequence noise is consistent with the view-masking finding. Because the model depends more on the derived view, any corruption of that branch has a proportionally larger effect. The random masking experiments, which simulate intermittent feature unavailability, show more severe degradation at high mask rates for derived features (47.2% RMSE increase at 30% masking) than for sequence features (3.95% at 30% masking), reinforcing the same asymmetry.

The lifecycle-region analysis reveals an important characteristic of the task rather than just the model: near-failure prediction (RUL 0–30) has an RMSE of 4.43, compared with 13.53 for the early or capped region (RUL 81–125). This pattern arises partly because the near-failure region has a narrow label range (0–30) that constrains prediction errors, and partly because the degradation signal is stronger and more consistent in that region. The capped region contains the bulk of the samples (2,091 of 3,711), and the target cap at 125 reduces label variance, making the near-zero R² (0.044) in that region an artefact of the capping rather than a sign of the model's complete failure to predict.

The systematic under-prediction bias (mean error −2.28 on the seed-42 validation set, consistent with −3.52 across repeated validation and −3.20 on the official test) likely reflects the model's calibration to the degradation-active region, where the capped target pulls labels downward. The bias is a systematic property of the trained model and not noise.

The MC Dropout uncertainty correlation (r ≈ 0.49) suggests that the model's internal variability under stochastic dropout does capture some signal about prediction difficulty. This is practically useful for flagging predictions where additional caution may be warranted. However, because the correlation is moderate rather than strong, and the uncertainty is uncalibrated, it should be treated as a qualitative signal rather than a quantitative reliability measure.

---

## 8.4 RQ4: Stability and Held-Out Generalisation

**Question:** How stable are the main findings across engine-level validation splits and on independent FD001 test engines?

**Answer:** Model rankings are sensitive to the engine cohort used for evaluation. The repeated-validation and official-test rankings differ, demonstrating that conclusions drawn from any single split — including the initial seed-42 development split — cannot be assumed to be stable.

The most striking manifestation of this instability is the rank reversal between the repeated-validation mean ranking (DerivedOnlyMLP 1st, Fusion 2nd, XGBoost 3rd, GRU 4th) and the official endpoint test ranking (XGBoost 1st, DerivedOnlyMLP 2nd, GRU 3rd, Fusion 4th). No model holds the same rank across both populations. The initial seed-42 development result (Fusion 1st) is consistent with neither the repeated-validation mean nor the official test.

Per-split data shows why this happens. Seed 42 produces unusually strong results for both XGBoost (12.53) and MultiViewGRUFusion (12.56), while seed 84 produces unusually weak results for both (16.64 and 15.76 respectively). This variance reflects genuine differences in which engines are assigned to each cohort: some cohorts include engines with atypical degradation trajectories, while others are more representative. Three splits are not enough to marginalise this cohort-level variance, which is why the standard deviations in Table 6.3 are substantial (up to ±2.07 for XGBoost).

The official endpoint test draws from a population of 100 test engines with potentially different characteristics to any validation cohort. Because only one endpoint prediction per engine is evaluated (rather than all windows), the test metric is also sensitive to how well each model captures the final degradation state specifically, rather than performance averaged across all lifecycle points. XGBoost, with its 57-feature tabular representation, appears particularly well-suited to this endpoint task.

These findings motivate two practical recommendations for future studies in this area. First, any single-split conclusion should be confirmed across multiple independent cohort draws before being treated as robust. Second, the choice of evaluation population — all windows versus endpoint-only — should be matched to the intended deployment context.

---

## 8.5 Why Fusion Looked Strongest Initially

The seed-42 development result placed MultiViewGRUFusion first with RMSE 12.0657, ahead of XGBoost (12.4894) and DerivedOnlyMLP (13.1451). Several factors contributed to this initial result that did not persist under the broader protocol.

First, the seed-42 split is a single engine cohort. In this specific cohort, the 20 validation engines happened to be a population on which the fusion model's combined representation was particularly effective. The seed-42 split also produces some of the most favourable results for fusion (RMSE 12.56) in the repeated-validation protocol, confirming that the initial result was a genuine property of that cohort rather than a calculation error.

Second, the initial development stage allowed unlimited tuning against the seed-42 validation set — architectures, hyperparameters, and training procedures were all adjusted with visibility into this set's performance. The fusion model may therefore be somewhat overfit to the seed-42 cohort in the sense of being the configuration most suited to that specific population, even if the configuration was determined by principled choices rather than exhaustive search.

Third, in the seed-42 development stage, DerivedOnlyMLP and GRU were evaluated without the final-protocol training configuration (no per-model random-state reset, earlier training durations). The repeated-validation protocol with fixed MODEL_SEED and consistent training settings produced improved results for DerivedOnlyMLP (from 13.1451 in seed-42 development to 13.4408 in the seed-42 final-protocol split), suggesting that the initial comparison somewhat underestimated DerivedOnlyMLP relative to the fusion model.

---

## 8.6 Why XGBoost Performed Best on the Official Endpoint Test

XGBoost's first-place official endpoint result (RMSE 12.2459) contrasts with its third-place repeated-validation mean ranking (14.4297). Several reasons may explain this.

First, the official endpoint evaluation uses exactly one prediction per engine, aligned to the engine's last observed cycle. At this final observation point, the derived features — particularly cycle_index, rolling means, and delta values — capture the cumulative degradation state most clearly. XGBoost, operating on a tabular 57-feature representation that includes all 14 raw sensor readings, can use these features in combination without the sequence architecture's overhead, and may be more effective at extracting the final-state signal from the full feature set.

Second, neural models trained with fixed epochs (GRU at 12, Fusion at 12, DerivedOnlyMLP at 59) may be more sensitive to the difference between training on 80 engines (validation stage) and training on 100 engines (full-data training in Stage 3). The increase in training data changes the distribution of epochs needed for convergence, and the frozen epoch counts — derived as medians across three validation splits — may not be equally optimal for the full-data training. XGBoost, as a gradient boosting method with 300 estimators and a fixed learning rate, is less sensitive to this change.

Third, the endpoint evaluation population is quite different from the window-aligned validation populations. Endpoint predictions require estimating RUL at the precise cycle where the test engine's history terminates, which may weight accurately capturing late-lifecycle degradation dynamics. XGBoost's engineered features, especially the rolling means and cycle_index, may be particularly well-calibrated to this regime.

---

## 8.7 Why Model Rankings Change Across Cohorts

The rank variability across splits and the official test can be attributed to two interacting factors: cohort composition and evaluation-population structure.

**Cohort composition** determines which engines appear in the validation or test set. Engine 57, for example, is an outlier that produces high RMSE for all models in every split it appears in. Splits with or without such outlier engines will show different absolute RMSE values. Because only 20 engines form each validation cohort, a single difficult engine can shift the mean RMSE by one or two points — enough to change rankings.

**Evaluation-population structure** determines whether metrics are aggregated over all windows or just the endpoint. The repeated-validation metrics are computed over all eligible 30-cycle windows from the validation engines; high-RUL windows dominate numerically because there are more of them early in the lifecycle. The official test evaluates one point per engine regardless of cycle count. A model that performs well in the capped high-RUL region may rank higher in repeated-validation metrics than in endpoint-only evaluation, and vice versa.

These two factors together mean that no single rank from any single evaluation population can be treated as the definitive model ranking. The combination of all three evidence layers — seed-42 development, repeated validation, and official endpoint test — gives a more complete picture than any one of them alone.

---

## 8.8 The Value of Explainability Beyond RMSE

The RMSE-based results in Chapter 6 rank models by overall prediction accuracy but say nothing about how or why a model makes specific predictions, whether its behaviour is consistent with known physics, or how it behaves when sensor data is degraded. The explainability and trustworthiness analyses in Chapter 7 address these questions and provide complementary evidence for model assessment.

Feature importance and SHAP results establish that the XGBoost model's predictions are driven by interpretable quantities: the rolling averages of sensors known to degrade monotonically (sensors 4, 11, 15), the cumulative drift from initial sensor readings (delta features), and the cycle count as a temporal-position proxy. This alignment between learned feature importance and physical domain knowledge provides confidence that the model is not exploiting spurious correlations.

The view masking results establish a quantitative claim about information content: the derived view carries substantially more predictive information than the raw sequence view in the trained fusion architecture. This is useful for future architecture decisions — if a simpler model without the sequence branch achieves similar performance, the engineering cost of maintaining the GRU encoder may not be justified.

Lifecycle-region analysis shows where the model is reliable and where it is not. Knowing that the model is most accurate in the near-failure region (RMSE 4.43) but substantially less reliable in the early or capped region (RMSE 13.53) allows a deployment strategy to be designed accordingly — for example, issuing low-uncertainty maintenance recommendations only when the predicted RUL falls below a threshold.

The prediction bias characterisation reveals a systematic conservative tendency (under-prediction). In a maintenance context, an operator using the model's output would receive predictions that consistently suggest slightly less remaining life than is actually available. Whether this is operationally acceptable depends on the cost-benefit structure of the specific maintenance decision, which this dissertation has not modelled.

Finally, the MC Dropout uncertainty signal — while exploratory and uncalibrated — shows that the model's own variability contains some information about prediction difficulty. An operator system could use high dispersion across MC passes as a flag to escalate to human review. This is a concrete operational use case for uncertainty estimation even without full calibration.

Taken together, the explainability and trustworthiness evidence extends the dissertation's contribution beyond a model comparison. It provides a characterisation of the model's learned behaviour, its sensitivities, its failure modes, and the qualitative conditions under which its predictions are most and least reliable. These properties are arguably as important as the RMSE for any practical deployment assessment.

---

*Tables referenced: Table 8.1 (F11 — research question evidence mapping)*
