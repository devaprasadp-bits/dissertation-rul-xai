# Chapter 2 — Literature Review

---

## 2.1 Predictive Maintenance Approaches

Maintenance strategies for industrial assets range from purely reactive (fix on failure) to purely preventive (replace on schedule) to condition-based and predictive approaches that use real-time or historical operational data to guide maintenance decisions [Achouch et al., 2022]. Condition-based maintenance (CBM) monitors measurable degradation indicators — vibration levels, temperature trends, pressure differentials — and triggers maintenance when a threshold is crossed. Predictive maintenance extends CBM by forecasting the remaining useful life before the threshold is reached, enabling proactive scheduling rather than reactive response.

The prognostics and health management (PHM) framework formalises this forecasting task. PHM systems typically comprise four components: sensing and data acquisition, feature extraction, health state estimation, and remaining life prediction [Saxena et al., 2008]. Machine learning has become the dominant approach at the feature extraction and prediction stages, replacing or supplementing traditional physics-based degradation models in data-rich industrial settings.

The turbofan engine domain has been particularly well-studied because run-to-failure engine data is difficult to obtain in practice, and the C-MAPSS simulation provides a controlled and reproducible benchmark for algorithm development and comparison.

---

## 2.2 RUL Prediction on C-MAPSS

The C-MAPSS dataset [Saxena et al., 2008] has been the primary benchmark for prognostic RUL studies for over fifteen years. The four subsets — FD001 through FD004 — vary in operating conditions (1 or 6) and fault modes (1 or 2), providing a graduated difficulty structure.

Early approaches on C-MAPSS used classical machine learning models: support vector regression, random forest, and gradient boosting with hand-crafted features derived from sensor statistics [Heimes, 2008; Ramasso and Saxena, 2014]. These methods established that engineered features, particularly rolling statistics and delta-from-initial values, are effective predictors on this dataset.

Deep learning methods — beginning with recurrent architectures — substantially expanded the performance frontier. Li et al. [2018] demonstrated that deep convolutional networks (CNN1D) applied to sliding-window sensor inputs could improve over classical baselines on FD001 and FD003. The window-based formulation, in which a fixed-length segment of sensor history is used as the model input, became the standard deep learning input structure for C-MAPSS. GRU-based and LSTM-based architectures further improved performance by modelling sequential dependencies across the input window [Cho et al., 2014].

A persistent finding across C-MAPSS studies is that feature engineering — particularly rolling means, rolling standard deviations, and delta features — provides a strong baseline that is difficult to outperform with raw-input deep learning alone. The relative competitiveness of classical and deep learning methods has been noted across multiple benchmark comparisons [Cummins et al., 2024].

This dissertation contributes to the C-MAPSS literature by evaluating a multi-view fusion architecture under a repeated-validation and pre-registered official-test protocol, reporting the negative finding that fusion does not consistently outperform the best single-view or classical alternative.

---

## 2.3 Classical Machine-Learning Models for Prognostics

Classical machine-learning models — linear regression, random forest, gradient boosting trees, and XGBoost — remain competitive for RUL prediction on structured sensor data when effective features are engineered. Their key advantages are computational efficiency, interpretability via feature importance, and the absence of sequence-construction overhead.

XGBoost [Chen and Guestrin, 2016] is a regularised gradient boosting framework that handles tabular data with missing values, supports efficient tree construction with second-order gradient approximations, and provides native feature importance estimates. On C-MAPSS FD001, XGBoost with derived degradation features is among the strongest classical baselines when evaluated under matched conditions.

The interpretability advantage of tree models — particularly through SHAP attributions [Lundberg and Lee, 2017] — is a material benefit for maintenance applications, where domain engineers need to validate that model predictions are driven by physically meaningful sensor patterns. This is a key motivation for including XGBoost in the comparison and conducting a full SHAP analysis in this dissertation.

---

## 2.4 Deep Sequence Models

Sequence models are a natural fit for RUL prediction because engine degradation manifests as temporal trends in sensor readings. The two primary architectures evaluated in this dissertation are CNN1D and GRU.

**CNN1D** applies one-dimensional convolution filters to a windowed sensor input, extracting local temporal patterns within the window. Li et al. [2018] demonstrated that stacked CNN layers with global pooling can match or exceed LSTM performance on C-MAPSS. The architecture is computationally efficient and parallelisable, but it treats each position in the window independently and cannot model long-range sequential dependencies.

**GRU** (Gated Recurrent Unit) [Cho et al., 2014] is a recurrent architecture that explicitly models sequential dependencies through a gating mechanism controlling information flow across timesteps. The GRU's update and reset gates allow it to selectively retain or discard information from the past, making it effective for capturing gradual degradation trends over the input window. GRU is generally more computationally efficient than LSTM while achieving comparable performance on sequence modelling tasks.

In this dissertation, GRU substantially outperforms CNN1D on FD001 under matched conditions (RMSE 13.16 vs 18.15 in the seed-42 development stage), and GRU is used as the sequence encoder branch in the multi-view fusion model.

---

## 2.5 Multi-View and Multimodal Learning

Multi-view learning refers to methods that leverage multiple representations or views of the same underlying object to improve learning performance [Baltrušaitis et al., 2019]. In prognostics, different feature representations of the same sensor stream — raw time series, statistical summaries, frequency-domain features — constitute natural views.

Multimodal learning is a broader category in which physically distinct data modalities are combined: sensor readings and inspection images, or time-series data and maintenance text records. True multimodal fusion introduces additional challenges of temporal alignment, missing modalities, and heterogeneous feature spaces [Baltrušaitis et al., 2019].

The architecture implemented in this dissertation is a controlled multi-view fusion: the raw sensor sequence view (30 × 14) and the derived degradation feature view (43 features) are both derived from the same 14 sensor measurements, processed by separate branches (GRU encoder and dense network respectively), and fused by concatenation followed by a shared prediction head. This design allows systematic examination of whether the temporal-pattern information in the raw sequence and the degradation-summary information in the derived features can be combined beneficially.

The fusion architecture is a common pattern in multi-view learning: the late-fusion design processes each view independently and combines the learned representations before the prediction layer. Alternative fusion strategies — attention-weighted fusion, gated fusion, cross-view attention — are outside the current experimental scope but represent natural extensions.

The negative fusion finding in this dissertation — that the GRU–MLP concatenation architecture does not consistently outperform the best single-view model — is consistent with the general observation in multi-view learning that naive concatenation does not always improve over the stronger single-view baseline, particularly when the views carry asymmetric amounts of information and one view is significantly more informative than the other.

---

## 2.6 Explainability for Prognostics

The explainable AI (XAI) literature has produced a range of methods for attributing model predictions to input features [Vilone and Longo, 2021]. In the context of prognostics and predictive maintenance, three approaches are most relevant.

**Feature importance** from tree models provides a global ranking of features by their contribution to the ensemble's predictive accuracy. Gain-based importance, as computed by XGBoost, measures the total improvement in the objective function attributable to splits using each feature. While intuitive, gain-based importance can be biased by feature correlation structures.

**SHAP (SHapley Additive exPlanations)** [Lundberg and Lee, 2017] provides theoretically grounded attributions based on the Shapley value from cooperative game theory. SHAP values satisfy desirable axioms (efficiency, symmetry, dummy, linearity) and can be used for both global feature ranking (mean absolute SHAP across the dataset) and local instance-level attribution (how each feature contributed to a specific prediction). SHAP has become the preferred attribution method in settings where feature correlation is a concern, as it more accurately distributes credit among correlated predictors.

**View masking** is a model-agnostic approach applicable to multi-view architectures: one input view is replaced with a neutral value (typically the training mean) and the change in prediction accuracy measures the model's dependence on that view. This approach provides a coarser but more interpretable attribution at the view level, complementing the feature-level SHAP analysis.

Cummins et al. [2024] survey explainability methods in the predictive maintenance context and note that while SHAP-based methods have been applied to classical prognostics models, their systematic integration with multi-view deep learning architectures and view-level interpretation remains less developed. This dissertation contributes to this gap by combining SHAP for XGBoost with view masking for the fusion model.

---

## 2.7 Robustness and Uncertainty in Prognostic Models

The deployment reliability of a prognostics model depends not only on its nominal accuracy but on its behaviour under realistic operational perturbations: sensor noise, intermittent data loss, missing features, and variation in asset populations.

Noise robustness studies in the prognostics literature typically inject controlled Gaussian noise into the test inputs and measure the resulting change in prediction accuracy. The magnitude of acceptable noise degradation depends on the application, but even small RMSE increases under realistic noise levels can have material consequences for maintenance planning.

Missing-view or missing-modality robustness is a related concern for multi-view architectures. If one input branch fails — due to sensor damage, data pipeline failure, or network latency — the system must either degrade gracefully or fall back to a single-view configuration. Testing the model under complete view masking establishes whether graceful degradation is possible or whether the architecture is critically dependent on both views.

Predictive uncertainty estimation provides a complementary measure of model reliability. MC Dropout [Gal and Ghahramani, 2016] is a practical approximation to Bayesian inference: by keeping dropout active at test time and performing multiple stochastic forward passes, a distribution of predictions is obtained, whose dispersion serves as an exploratory uncertainty estimate. While MC Dropout does not provide calibrated confidence intervals without further calibration, the correlation between predictive dispersion and prediction error has been used in the literature as an operational uncertainty signal.

Calibrated uncertainty — where stated confidence levels correspond to observed coverage rates — remains an open problem for deep learning prognostics models. Conformal prediction, Bayesian deep learning, and ensemble-based uncertainty estimation are active research directions. This dissertation uses MC Dropout as an exploratory starting point and explicitly does not claim calibration.

---

## 2.8 Research Gap

The literature review identifies several gaps that this dissertation addresses:

1. **Single-split evaluation dominance.** The majority of C-MAPSS RUL prediction studies report results from a single train–validation split. This makes it difficult to distinguish model superiority from split-specific luck. This dissertation addresses this gap through repeated validation across three independent engine cohorts.

2. **Limited multi-view evaluation under matched conditions.** Multi-view and multimodal fusion architectures for prognostics have been proposed, but their comparison against single-view alternatives under matched training, validation, and test protocols is rare. This dissertation provides such a comparison, including a negative finding about the fusion model's consistency.

3. **Explainability limited to classical models.** SHAP-based analyses in the prognostics literature are predominantly applied to classical models. Systematic view-level attribution for multi-view deep learning architectures — combining view masking with per-feature SHAP on the classical comparator — is less developed. This dissertation contributes to this gap.

4. **Trustworthiness as a secondary concern.** Robustness studies, lifecycle-region error analysis, prediction bias, and uncertainty estimation are often treated as supplementary analyses rather than primary evaluation criteria. This dissertation elevates trustworthiness to a first-class dimension of model assessment, alongside predictive accuracy.

5. **Reproducibility and test-protocol rigour.** Published prognostic results often lack detailed information about train–test protocol, feature leakage prevention, and the number of times the test set was consulted. The pre-test freeze manifest and one-time official evaluation in this dissertation provide an unusually transparent test protocol for a postgraduate prognostics study.

---

## References

Achouch, M., Dimitrova, M., Ziane, R., et al. (2022). On Predictive Maintenance in Industry 4.0: Overview, Models, and Challenges. *Applied Sciences*, 12(16), 8081.

Baltrušaitis, T., Ahuja, C., and Morency, L.-P. (2019). Multimodal Machine Learning: A Survey and Taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423–443.

Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *Proceedings of EMNLP*, 1724–1734.

Cummins, L., Sommers, A., Ramezani, S. B., et al. (2024). Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities. *arXiv preprint arXiv:2401.07871*.

Gal, Y., and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1050–1059.

Heimes, F. O. (2008). Recurrent Neural Networks for Remaining Useful Life Estimation. *Proceedings of the International Conference on Prognostics and Health Management*, 1–6.

Li, X., Ding, Q., and Sun, J.-Q. (2018). Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks. *Reliability Engineering & System Safety*, 172, 1–11.

Lundberg, S. M., and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS)*, 4765–4774.

Ramasso, E., and Saxena, A. (2014). Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets. *International Journal of Prognostics and Health Management*, 5(2).

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). Damage Propagation Modelling for Aircraft Engine Run-to-Failure Simulation. *Proceedings of the International Conference on Prognostics and Health Management*.

Vilone, G., and Longo, L. (2021). Notions of Explainability and Evaluatability Approaches for Explainable Artificial Intelligence. *Information Fusion*, 76, 89–106.
