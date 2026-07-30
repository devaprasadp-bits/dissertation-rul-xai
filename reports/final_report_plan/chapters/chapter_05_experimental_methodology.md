# Chapter 5 — Experimental Methodology

---

## 5.1 Overall Experiment Design

The experimental methodology follows a three-stage structure designed to establish initial model development evidence, test stability across independent engine cohorts, and produce a final held-out evaluation under a controlled protocol.

**Stage 1 — Seed-42 development (NB01–NB09):**
All modelling decisions — feature sets, architectures, hyperparameters, and training procedures — were made during an initial development phase using a single engine-level split with random seed 42. This stage produced the historical reference results reported in Section 6.1. All explainability and trustworthiness studies (Chapters 7) were also conducted during this stage.

**Stage 2 — Repeated validation (NB10):**
After the initial development stage, the four principal models were retrained under a consistent protocol across three independent engine-grouped splits (seeds 21, 42, and 84). The purpose was to test whether the seed-42 rankings were stable across different engine cohorts. No modelling decisions were changed based on these results; the repeated-validation protocol was predetermined.

**Stage 3 — Official endpoint evaluation (NB11–NB12):**
Each model was retrained on all 100 available training engines using a fixed epoch count derived from the repeated-validation stage. The official NASA endpoint labels were accessed only after a pre-test artefact manifest was created and verified. The official test was conducted exactly once, and no model parameters or configurations were modified afterwards.

This three-stage design imposes a clear separation between model development, validation, and final evaluation, reducing the risk of implicit overfitting to the validation population.

---

## 5.2 Classical Baselines: XGBoost, Ridge, RF, and GBT

Four classical machine-learning models were trained and evaluated as baselines: Ridge Regression, Random Forest, Gradient Boosting Trees (GBT), and XGBoost. A Dummy Mean regressor (always predicting the training-set mean RUL) was also included as a lower-bound reference.

All classical models were evaluated across the three feature sets (A, B, and C). The best-performing configuration for each model family is reported in the results. XGBoost with Feature Set C is the representative classical result used in all cross-model comparisons because it consistently achieved the strongest performance among classical models.

The XGBoost configuration used throughout this dissertation is:

**Table 5.1: XGBoost hyperparameters**

| Parameter | Value |
|---|---|
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 4 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | reg:squarederror |
| random_state | 42 |
| Input features | Feature Set C (57 features) |

Unlike the neural models, XGBoost does not require sequence windowing. It is trained on all available rows from the training engines (20,631 rows for the full-data training in Stage 3; approximately 16,340 rows in the seed-42 development split).

---

## 5.3 Deep Sequence Baselines: CNN1D and GRU

Two deep learning baseline models were trained on the raw sensor sequence view (Feature Set B, 30-cycle windows, shape (30, 14)).

**CNN1D architecture:**

```
Input: (30, 14)
Conv1D(64, kernel_size=5, activation='relu', padding='same')
Conv1D(64, kernel_size=5, activation='relu', padding='same')
GlobalAveragePooling1D()
Dense(50, activation='relu')
Dropout(0.2)
Dense(1)
```

**GRU architecture:**

```
Input: (30, 14)
GRU(64, return_sequences=True)
GRU(32)
Dense(50, activation='relu')
Dropout(0.2)
Dense(1)
```

Both models use the Adam optimiser with learning rate 0.001, MSE loss, and MAE as a monitored metric. They were trained with batch size 256, a maximum of 30 epochs, early stopping with patience 10 (monitoring validation loss), and ReduceLROnPlateau with factor 0.5 and patience 5. The GRU model consistently outperformed CNN1D and is the representative deep-learning-baseline result in all cross-model comparisons.

---

## 5.4 DerivedOnlyMLP

The DerivedOnlyMLP is a single-view neural network trained exclusively on the derived degradation feature view (Feature Set C minus the raw sensors, i.e., the 43 derived features). It was included as an ablation model to quantify the predictive signal carried by the derived view independently of the raw-sequence view.

**DerivedOnlyMLP architecture:**

```
Input: (43,)
Dense(64, activation='relu')
Dropout(0.2)
Dense(32, activation='relu')
Dense(1)
```

Training uses Adam with learning rate 0.001, MSE loss, batch size 128, a maximum of 60 epochs, early stopping with patience 8, and ReduceLROnPlateau with factor 0.5 and patience 4.

---

## 5.5 MultiViewGRUFusion Architecture

MultiViewGRUFusion is the primary multi-view model. It takes both input views simultaneously and combines them through a learned fusion layer.

**Sensor-sequence branch:**

```
Input: sensor_sequence_view — shape (30, 14)
GRU(64, return_sequences=False)
Dropout(0.2)
→ 64-dimensional representation
```

**Derived-degradation branch:**

```
Input: degradation_feature_view — shape (43,)
Dense(64, activation='relu')
Dropout(0.2)
Dense(32, activation='relu')
→ 32-dimensional representation
```

**Fusion head:**

```
Concatenate([seq_x, der_x])  → 96-dimensional fused vector
Dense(64, activation='relu')
Dropout(0.2)
Dense(32, activation='relu')
Dense(1)
```

The model is compiled with Adam (learning rate 0.001), MSE loss, and MAE metric. Training uses batch size 256, maximum 60 epochs, early stopping with patience 8, and ReduceLROnPlateau with factor 0.5 and patience 4. The architecture is summarised in Table 5.2.

**Table 5.2: MultiViewGRUFusion architecture summary**

| Component | Configuration |
|---|---|
| Sensor branch input | (30, 14) — 30-cycle window, 14 sensors |
| Sensor branch encoder | GRU(64, return_sequences=False) + Dropout(0.2) |
| Derived branch input | (43,) — 43 derived degradation features |
| Derived branch encoder | Dense(64, relu) + Dropout(0.2) + Dense(32, relu) |
| Fusion | Concatenate → Dense(64, relu) + Dropout(0.2) + Dense(32, relu) + Dense(1) |
| Optimiser | Adam, lr=0.001 |
| Loss | MSE |

---

## 5.6 Evaluation Metrics

The primary evaluation metrics are:

- **RMSE** — Root Mean Squared Error. Penalises larger errors more heavily.
- **MAE** — Mean Absolute Error. Average absolute error in cycles; directly interpretable.
- **R²** — Coefficient of determination. Proportion of target variance explained by the model.
- **Mean error** — Mean of (prediction − actual). Positive values indicate over-prediction; negative values indicate under-prediction (conservative bias).

All metrics are computed on capped RUL targets (cap 125) with predictions clipped to [0, 125].

---

## 5.7 Stage 1 — Original Seed-42 Development

The initial development stage used a single engine-level split with seed 42: 80 training engines and 20 validation engines. All modelling decisions — architectures, feature sets, hyperparameters, and training procedures — were finalised during this stage without access to the official test labels.

The validation metrics from this stage are referred to as the **historical seed-42 reference** throughout this dissertation. They reflect the window-aligned evaluation (all eligible 30-cycle windows from the 20 validation engines). The MultiViewGRUFusion model achieved the lowest validation RMSE in this initial development stage (RMSE 12.0657), which motivated the choice of fusion as the primary model under study.

---

## 5.8 Stage 2 — Repeated Validation Protocol

After the initial development stage, a repeated-validation protocol was designed to test whether the seed-42 rankings were stable across different engine cohorts.

**Protocol details:**

- Three independent engine-level splits were generated with seeds 21, 42, and 84.
- In each split, 80 engines are assigned to training and 20 to validation by random draw without replacement.
- The **model random state** is held fixed at 42 across all three splits. Only the data cohort changes; weight initialisation and dropout are identical across splits.
- All four principal models (XGBoost, GRU, DerivedOnlyMLP, MultiViewGRUFusion) were trained and evaluated on each split under identical hyperparameters.
- Callbacks (early stopping, ReduceLROnPlateau) were instantiated fresh for each model–split combination; callback state was not carried over between runs.
- TensorFlow session was cleared and the random seed was reset before each neural model training.
- XGBoost `random_state` was fixed at 42 for all splits.

The purpose of fixing MODEL_SEED while varying SPLIT_SEED is to isolate the effect of the engine cohort from the effect of random weight initialisation. Any observed variation in RMSE across splits can therefore be attributed to differences in which engines appear in the validation set, not to stochastic training variation.

---

## 5.9 Ablation Protocol: cycle_index Removal

A targeted ablation was conducted within the seed-42 final-protocol split to quantify the contribution of `cycle_index`. The ablation removes `cycle_index` from Feature Set C, reducing the derived feature count from 43 to 42. XGBoost and DerivedOnlyMLP were retrained on this reduced feature set; the RMSE increase is reported as the cycle_index contribution.

This ablation was conducted using the seed-42 split only and is reported in Section 6.4. It was not conducted across all three validation seeds; the result is therefore indicative rather than a repeated-validation finding.

---

## 5.10 Final Epoch Selection

Because Stage 3 trains neural models on all 100 engines without a validation set, early stopping cannot be used. The training duration must be predetermined. The epoch counts were derived from the repeated-validation stage as follows:

For each neural model, the best epoch (epoch at which validation loss was minimised) was recorded across all three repeated-validation splits. The **median** of the three best epochs was selected as the final training epoch count.

**Table 5.3: Final epoch selection**

| Model | Best epoch (seed 21) | Best epoch (seed 42) | Best epoch (seed 84) | Median (frozen) |
|---|---|---|---|---|
| GRU | 12 | 12 | 26 | **12** |
| DerivedOnlyMLP | 55 | 60 | 59 | **59** |
| MultiViewGRUFusion | 12 | 9 | 12 | **12** |

The median was chosen because it is robust to outlier epochs from any single split. XGBoost does not require epoch selection; it uses the same configuration as in the development stage.

---

## 5.11 Full-Data Refitting (100 Training Engines)

In Stage 3, each model was retrained from scratch on all 100 FD001 training engines (20,631 rows for tabular models; 17,731 window samples for neural models using the 30-cycle window). No validation set and no early stopping were used during this final refit. Neural models trained for exactly the number of epochs determined in Section 5.10.

The full training data uses the same preprocessing pipeline as the validation-stage training, with scalers refitted on the full 100-engine training set.

---

## 5.12 Pre-Test Freeze Manifest

Before reading any official test labels, a SHA-256 artefact integrity manifest was created recording the cryptographic hashes of all model files, training history files, scalers, the feature manifest, the epoch selection file, and the frozen protocol file. This manifest was saved to `reports/final_test/pretest_freeze_manifest.json`.

The purpose of the pre-test manifest is to provide verifiable evidence that no model or preprocessing artefact was modified after the official test labels were accessed. The evaluation manifest (saved after official test evaluation) records the same set of hashes; the two manifests can be compared to confirm artefact integrity.

---

## 5.13 Official Test Procedure

The official evaluation was conducted under the following constraints:

- The execution control flag `AUTHORISE_OFFICIAL_TEST` was set to `True` only after the pre-test freeze manifest was reviewed.
- Official test data (`test_FD001.txt` and `RUL_FD001.txt`) was read only within the authorised evaluation cell.
- Each model produced exactly **one prediction per test engine** (100 predictions per model), aligned to the final observed cycle of each test engine.
- Predictions were clipped to [0, 125] before metric computation.
- Official test endpoint RUL values were capped at 125 for consistency with the training target formulation.
- The evaluation was conducted **once**; the flag `OVERWRITE_OFFICIAL_TEST` was not set to `True` at any point.
- No model parameters, hyperparameters, or feature sets were modified after viewing the official test outcomes.

---

## 5.14 Reproducibility Artefacts

All experimental outputs are anchored to a 57-item evidence ledger. Each evidence item records the source notebook, the output file path relative to the project root, and the role of the output in the final dissertation. The NB12 consolidation notebook independently recomputes the repeated-validation summaries from split-level CSVs and independently recomputes the official-test metrics from the saved prediction CSV, providing an internal consistency check.

The full notebook sequence, SHA-256 hashes of key artefacts, and the signed evidence ledger are provided in Appendix F.

**Table 5.4: Notebook sequence (complete)**

| Notebook | Stage | Purpose |
|---|---|---|
| NB01 | Development | Dataset loading, RUL generation, initial checks |
| NB02 | Development | EDA, feature profiling, sensor selection |
| NB03 | Development | RUL capping, feature sets, scaling, engine-level split |
| NB04 | Development | Classical baseline modelling (Ridge, RF, GBT, XGBoost) |
| NB05 | Development | CNN1D and GRU deep learning baselines |
| NB06 | Development | DerivedOnlyMLP and MultiViewGRUFusion; seed-42 reference |
| NB07 | Development | Explainability: feature importance, SHAP, view masking |
| NB08 | Development | Robustness, bias, perturbation stability, MC Dropout |
| NB09 | Development | Mid-semester consolidated results |
| NB10 | Repeated validation | Three-seed repeated protocol; epoch selection; ablation |
| NB11 | Official test | Full-data refit; pre-test freeze; one-time evaluation |
| NB12 | Consolidation | Evidence recomputation; Tables F1–F12; claims and limitations |

---

*Tables in this chapter: Table 5.1, Table 5.2, Table 5.3, Table 5.4 (F1 from NB12 maps to Table 5.1 in this chapter)*
