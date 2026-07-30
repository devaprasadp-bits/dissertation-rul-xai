# Chapter 9 — Limitations, Relevance and Future Work

---

## 9.1 Study Limitations

The limitations of this dissertation are grouped by their nature and are summarised in Table 9.1. Each limitation is described below with its impact on the interpretation of results and the direction it points for future work.

**Table 9.1: Study limitations and mitigations**

| ID | Limitation | Impact | Treatment in this work |
|---|---|---|---|
| L01 | Scope limited to C-MAPSS FD001 | Findings are for one operating condition and one fault mode | All conclusions explicitly scoped to FD001 |
| L02 | Multi-view, not native multimodality | Both views from same sensor stream | Term multi-view used throughout; boundary clearly stated |
| L03 | Three repeated-validation splits | Descriptive cohort sensitivity, not formal statistical test | All variability statements labelled as descriptive |
| L04 | Validation and test evaluation units differ | Window-level vs endpoint-level metrics not directly comparable | Rankings compared descriptively only |
| L05 | 125-cycle RUL cap | Compresses early-life targets; emphasises degradation region | Cap stated in all result tables and abstract |
| L06 | One fusion architecture | Negative fusion result is architecture-specific | Conclusion limited to GRU–MLP concatenation |
| L07 | Material dependence on cycle_index | Temporal position contributes signal; may not generalise under shifted lifecycle distributions | Ablation conducted and reported |
| L08 | Neural attribution mainly at view level | Individual timestep and sensor attribution not available for fusion model | View masking and local cases used as bounded attribution |
| L09 | MC Dropout uncertainty uncalibrated | Dispersion is exploratory, not a confidence interval | Reported as exploratory evidence only |
| L10 | NASA asymmetric scoring not principal metric | Study does not weight late predictions more heavily | RMSE, MAE, R² and mean error all reported |
| L11 | Benchmark prototype, not production deployment | Latency, drift, sensor quality and cost decisions not validated | Positioned as applied benchmark research |

*Source: E55 — `reports/final_summary/final_limitations_fd001.csv`*

**L01 — FD001 scope:** The entire experimental pipeline — feature engineering choices, model architectures, hyperparameters, and training procedures — was designed and tuned for FD001's single operating condition and single fault mode. FD002–FD004 introduce multiple operating regimes and additional fault modes that would require operating-condition normalisation and potentially different feature engineering. The quantitative results in Chapters 6 and 7 should not be extrapolated to these subsets or to real industrial assets without re-evaluation.

**L02 — Multi-view vs native multimodality:** Both input views in this study derive from the same 14 numerical sensor measurements. The multi-view formulation is a useful proxy for studying representation combinations under controlled conditions, but it does not exercise the full challenges of true multimodality: temporal alignment across asynchronous data streams, different sampling rates, missing modalities during deployment, and feature space heterogeneity. The dissertation's contribution to multimodal learning is therefore methodological (a controlled evaluation framework) rather than a direct demonstration on heterogeneous data.

**L03 — Three validation splits:** The repeated-validation protocol with three seeds provides descriptive evidence of cohort sensitivity, but three configurations cannot support formal statistical tests of model superiority. Confidence intervals on mean RMSE differences would require many more independent splits. The standard deviations in Table 6.3 should be interpreted as indicative variability estimates, not as precise characterisations of the true performance distribution.

**L04 — Evaluation unit mismatch:** Repeated-validation metrics average over all eligible 30-cycle windows from each validation cohort, while the official endpoint test evaluates one prediction per engine at the final cycle. These are fundamentally different aggregations. The shift in model rankings between the two populations (Section 6.6) is partly an artefact of this difference, making any generalisation claim from one population to the other imprecise.

**L05 — RUL cap at 125:** The cap at 125 cycles reduces the dynamic range of the prediction task and compresses the target distribution for early-lifecycle engines. Results in this dissertation are specific to this cap value. Studies using uncapped targets or alternative caps will produce different absolute RMSE values and potentially different model rankings. The capped results reported here are not directly comparable with uncapped benchmark literature.

**L06 — Single fusion architecture:** The negative fusion result — that MultiViewGRUFusion does not consistently outperform the strongest single-view model — applies to the specific GRU–MLP concatenation architecture implemented here. Alternative fusion strategies (attention-based fusion, gated fusion, uncertainty-weighted fusion, cross-modal transformers) may produce different results and are not ruled out by this finding.

**L07 — cycle_index dependence:** The ablation confirms that cycle_index contributes 11–16% RMSE reduction across the three models that use it. This feature is available at prediction time in the C-MAPSS setting, where the cycle counter is part of the raw data. However, it is essentially a proxy for the engine's total operating hours to date. In deployment settings where engines enter service at different ages, have variable usage intensities, or where operating-hour records are incomplete, the value of this feature may differ substantially.

**L08 — View-level neural attribution:** The explainability analysis for MultiViewGRUFusion is limited to view masking (Section 7.3) and local case prediction differences (Section 7.4). Individual feature importance within the GRU branch — which sensor at which timestep most influenced a given prediction — is not provided. Gradient-based attribution methods (Integrated Gradients, GradCAM for time series) could provide finer-grained attribution but were outside the scope of this study.

**L09 — Uncalibrated uncertainty:** The MC Dropout uncertainty estimates are exploratory. Without calibration against a held-out set — such as comparing predicted confidence intervals with empirical coverage — the dispersion values cannot be interpreted as probability statements. A model that reports high dispersion on a given input may be genuinely uncertain or may simply have learned a representation that activates many dropout paths.

**L10 — NASA asymmetric scoring:** The official NASA prognostics evaluation also uses an asymmetric scoring function that penalises late predictions (predicting more remaining life than available) more heavily than early predictions. This scoring function is not the primary evaluation metric here. Comparisons with papers using the NASA score require matching the scoring formula and the target formulation.

**L11 — Benchmark prototype:** The system described in this dissertation was not deployed in an industrial setting. Operational considerations such as real-time inference latency, sensor data quality monitoring, concept drift under changing operating conditions, maintenance scheduling integration, and human-in-the-loop review have not been addressed.

---

## 9.2 Relevance to Industrial Predictive Maintenance

Despite the limitations above, the findings of this dissertation are relevant to the design and evaluation of predictive-maintenance systems in several concrete ways.

**Feature engineering matters more than architecture complexity.** The result that DerivedOnlyMLP — a simple two-layer dense network trained on 43 engineered features — performs comparably to or better than a GRU-based fusion model across multiple evaluation conditions has direct practical implications. In industrial settings where computational resources, interpretability requirements, and deployment simplicity are important, an engineered-feature approach may be preferable to a more complex sequence model. The finding is consistent with the broader machine-learning literature's observation that good features often matter more than sophisticated architectures on structured tabular data.

**Single-split validation is insufficient for deployment decisions.** The rank reversal between seed-42 development (Fusion first) and the broader evaluation (XGBoost first on the endpoint test, DerivedOnlyMLP first by mean repeated validation) illustrates the risk of using a single validation cohort to select and certify a model. Industrial predictive-maintenance deployments should evaluate candidate models across multiple temporal or asset-level splits before selecting a production candidate.

**Explainability supports maintenance decision-making.** The SHAP and feature importance analyses identify which sensor readings drive predictions at a given operating point. An operator interface that surfaces the top contributing features for each prediction can help maintenance engineers validate the model's reasoning against their domain knowledge and flag predictions that rely on unexpected feature patterns.

**Lifecycle-region calibration informs deployment thresholds.** The finding that prediction accuracy is highest near failure (RMSE 4.43 for RUL 0–30) and lowest in the early or capped region (RMSE 13.53 for RUL 81–125) suggests that prediction-driven maintenance decisions should carry higher confidence thresholds in the degradation-active window. A deployment system might issue high-confidence maintenance alerts only when the predicted RUL falls below 30 cycles and treat earlier predictions as indicative estimates requiring periodic confirmation.

**Honest reporting of negative results supports system safety.** Reporting that fusion was not consistently superior, rather than presenting the best single-seed result as the system's performance, provides a more reliable basis for deployment decisions. Overclaiming model performance in predictive-maintenance contexts can lead to under-maintenance (if operators trust overly optimistic estimates) or over-maintenance (if the model's actual in-deployment performance falls short of the claimed benchmark).

---

## 9.3 Practical Implications for Predictive-Maintenance Systems

Several practical design recommendations emerge from this study:

1. **Prioritise derived degradation features.** Rolling means, delta-from-initial, and a cycle counter are the most cost-effective features for this task. Any deployment pipeline should invest in computing and maintaining these features reliably before attempting sequence modelling or fusion architectures.

2. **Test model stability before deployment.** Repeated evaluation on independent asset cohorts — analogous to the three-split protocol used here — should be a standard step in the model validation workflow before production deployment.

3. **Use view masking to verify input reliability.** An operational system should monitor the availability and quality of both input views. The complete-derived-view masking result (RMSE 42.57, R² −0.035) demonstrates that silent sensor failure on the derived feature inputs would produce misleading predictions. Input quality monitoring should trigger fallback procedures or model alerts.

4. **Report bias alongside accuracy.** The systematic under-prediction tendency (mean error −2.28 to −3.52 cycles) should be characterised and communicated to maintenance operators. A calibration offset may improve decision-making by correcting for this systematic shift.

5. **Treat uncertainty estimates as flags, not guarantees.** The MC Dropout dispersion signal (r ≈ 0.49 with absolute error) can be incorporated into a human-review workflow: predictions with high dispersion are flagged for manual assessment, while predictions with low dispersion are accepted automatically. This does not require full calibration to provide operational value.

---

## 9.4 Future Work: Multimodal Extensions

The multi-view architecture implemented here uses two views derived from the same sensor stream. A natural extension is to incorporate genuinely different data modalities:

- **Inspection images:** Periodic visual inspection images of engine components could provide a complementary degradation signal. Temporal alignment between image timestamps and sensor cycle records would be required.
- **Maintenance logs:** Unstructured maintenance text records (component replacements, anomaly notes) contain expert knowledge about degradation events that sensor readings alone may not capture. NLP-based feature extraction could bridge text and sensor modalities.
- **Acoustic or vibration signals:** Accelerometer or acoustic emission sensors, where available, can detect degradation signatures (bearing wear, imbalance) that thermal and pressure sensors may not resolve.

These extensions would require the temporal alignment infrastructure that the current scope explicitly excludes. The controlled multi-view framework established in this dissertation provides a baseline against which the incremental value of true heterogeneous modalities could be measured.

---

## 9.5 Future Work: Temporal Alignment of Heterogeneous Data

Native multimodal fusion introduces the technical challenge of aligning data streams with different sampling rates, latencies, and availability patterns. A sensor recording every second, an inspection image taken every two weeks, and a maintenance log entry at irregular intervals occupy fundamentally different temporal scales. Future work should develop alignment strategies (interpolation, event-based synchronisation, learned temporal embeddings) that preserve the integrity of each modality while enabling joint modelling.

---

## 9.6 Future Work: Calibration and Cost-Sensitive Evaluation

Two evaluation dimensions that this dissertation leaves unaddressed are probabilistic calibration and cost-sensitive scoring.

For calibration: the MC Dropout uncertainty estimates should be evaluated using calibration curves and expected calibration error metrics, and alternative uncertainty estimation approaches (conformal prediction, Bayesian neural networks, ensemble methods) should be compared.

For cost-sensitive evaluation: the NASA asymmetric scoring function should be evaluated alongside RMSE, and sensitivity to the scoring function's asymmetry parameter should be studied. Industrial maintenance cost functions — which weight the cost of premature maintenance, deferred maintenance, and unplanned failure differently — could be incorporated into model selection criteria.

---

## 9.7 Future Work: Extension to FD002–FD004 and Industrial Data

The most direct validation of the pipeline's generalisability is evaluation on the remaining C-MAPSS subsets (FD002–FD004), which introduce six operating conditions, two fault modes, and their combinations. Operating-condition normalisation and regime-aware feature engineering would need to be added. Beyond C-MAPSS, evaluation on industrial datasets with confirmed operating histories, failure labels, and multiple asset classes would provide the most practically relevant evidence of the approach's utility.

---

*Table in this chapter: Table 9.1 (F12 — limitations and mitigations)*
