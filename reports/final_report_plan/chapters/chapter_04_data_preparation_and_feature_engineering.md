# Chapter 4 — Data Preparation and Feature Engineering

---

## 4.1 Data-Quality Checks

Before any modelling, the raw FD001 files were inspected for data-quality issues. The training file and test file were loaded and the following checks were performed:

- **Missing values:** No missing values were found in any column of the training or test data.
- **Duplicate rows:** No duplicate rows were identified.
- **Column consistency:** Both files contain 26 columns in the expected order (unit identifier, cycle counter, 3 operating settings, 21 sensor measurements).
- **Cycle integrity:** Each engine unit's cycle counter starts at 1 and increments without gaps.
- **Operating settings:** The three operating-setting columns are near-constant across all rows in FD001, consistent with the single operating condition of this subset.

These checks confirmed that the raw data is clean and suitable for direct use in feature engineering and preprocessing.

---

## 4.2 Sensor Selection Rationale

The raw FD001 data contains 21 sensor measurement columns (s1–s21). Exploratory data analysis revealed that a subset of these sensors are constant or near-constant across the entire dataset and therefore carry no informative degradation signal. Seven sensors were excluded: sensors 1, 5, 6, 10, 16, 18, and 19. These sensors show near-zero variance and no meaningful correlation with RUL in FD001.

The 14 retained sensors, which form the basis of all model inputs in this dissertation, are:

> sensor_measurement_2, sensor_measurement_3, sensor_measurement_4, sensor_measurement_7, sensor_measurement_8, sensor_measurement_9, sensor_measurement_11, sensor_measurement_12, sensor_measurement_13, sensor_measurement_14, sensor_measurement_15, sensor_measurement_17, sensor_measurement_20, sensor_measurement_21

Sensor vs RUL correlation analysis is shown in Figure 4.2. Sensor trends over the degradation lifecycle are shown in Figure 4.1. These sensors correspond to operational characteristics including compressor discharge temperature, turbine inlet temperature, fan speed, bypass ratio, and bleed enthalpy, among others.

The operating-setting columns are not used as model inputs because they are near-constant in FD001 and therefore do not contribute to the degradation signal.

---

## 4.3 Feature Set A: Full Raw Inputs

Feature Set A comprises all available raw operational inputs: the 3 operating-setting columns and the 21 original sensor measurement columns. This produces 24 features. Feature Set A was used in the classical baseline experiments as the broadest raw-feature baseline, providing a starting point to identify the value of sensor selection.

The 7 near-constant sensor columns are included in Feature Set A but are excluded from Feature Sets B and C. Their presence in Feature Set A has negligible effect on classical model performance due to their zero variance, but they add noise for linear models.

---

## 4.4 Feature Set B: Filtered Raw Sensors

Feature Set B contains the 14 selected variable sensor measurements described in Section 4.2. These 14 features are the filtered raw sensor input used for all sequence-based neural models (CNN1D and GRU), as well as the sensor sequence branch of MultiViewGRUFusion. Each input sample for a sequence model is a window of 30 consecutive cycles of these 14 sensors, producing an input tensor of shape (30, 14).

Feature Set B was also used as an intermediate classical baseline to measure the effect of removing non-informative sensors from Feature Set A.

---

## 4.5 Feature Set C: Derived Degradation Features

Feature Set C is the primary feature set for classical machine-learning models and the derived-view branch of the multi-view architecture. It extends Feature Set B by adding 43 engineered features derived from the same 14 sensors, bringing the total to 57 features.

The 43 derived features are generated per engine unit using a rolling window of 5 cycles:

- **Rolling mean** (`_rmean`): For each of the 14 sensors, the rolling mean over the preceding 5 cycles. This smooths short-term fluctuations and captures the local average sensor level. (14 features)
- **Rolling standard deviation** (`_rstd`): For each of the 14 sensors, the rolling standard deviation over the preceding 5 cycles. This captures local sensor volatility, which may increase as degradation progresses. Missing values at the start of each engine's history (fewer than 5 preceding cycles) are filled with zero. (14 features)
- **Delta from initial value** (`_delta`): For each of the 14 sensors, the difference between the current cycle reading and the first cycle reading of that engine unit. This captures cumulative drift from the engine's initial operating state. (14 features)
- **cycle_index**: A direct copy of the cycle counter (`time_in_cycles`) for each engine unit, representing the engine's temporal position in its lifecycle. (1 feature)

The composition of Feature Set C is summarised in Table 4.1.

**Table 4.1: Feature Set C composition**

| Component | Count | Description |
|---|---|---|
| Raw sensor measurements | 14 | Selected variable sensors (Feature Set B) |
| Rolling mean per sensor | 14 | 5-cycle rolling mean, computed within engine |
| Rolling std per sensor | 14 | 5-cycle rolling standard deviation, computed within engine |
| Delta from initial per sensor | 14 | Current value minus first-cycle value of that engine |
| cycle_index | 1 | Current cycle number for the engine unit |
| **Total** | **57** | Feature Set C (used by XGBoost and DerivedOnlyMLP) |

All rolling statistics are computed strictly within each engine unit's own history. No information from other engines or from future cycles is used. This preserves the leakage-safe structure of the preprocessing pipeline.

---

## 4.6 cycle_index and Its Role

The `cycle_index` feature is a direct copy of the `time_in_cycles` column — the cycle counter for each engine unit. It is introduced as a separate named feature to distinguish it from the metadata column and to ensure that the cycle counter enters the model only through the explicit feature set (not through metadata leakage).

`cycle_index` is the most informative single feature in Feature Set C, as confirmed by both XGBoost feature importance and SHAP analysis (Chapter 7). As engines age, the cycle count increases monotonically and is directly correlated with proximity to failure. The `cycle_index` ablation experiment (Section 6.4) quantifies the degradation in prediction accuracy when this feature is removed.

---

## 4.7 Removal of normalized_cycle_age

An earlier version of the feature engineering code included a feature named `normalized_cycle_age`, defined as the ratio of the current cycle to the engine's total lifecycle length:

> **normalized_cycle_age = time_in_cycles / max_cycle_of_engine**

This feature was identified as a **data leakage source** and was removed. The denominator — the maximum cycle of the engine — is the cycle at which the engine fails. During real deployment, the failure cycle is unknown in advance. Using this feature would therefore give the model access to information that cannot be known at prediction time. All results in this dissertation are produced without `normalized_cycle_age`.

The `cycle_index` feature is the leakage-safe alternative: it uses only the current cycle number, which is directly observable at any prediction point.

---

## 4.8 Train-Only Scaling

StandardScaler normalisation is applied to all feature sets before modelling. The scaler is fitted on the **training split only** and then applied identically to the training, validation, and test data. This prevents information about the scale or distribution of the validation or test data from influencing the training process.

Three separate scalers are maintained: one for Feature Set A, one for Feature Set B, and one for Feature Set C. The final full-training scalers (used in NB11) are fitted on all 100 training engines.

---

## 4.9 Engine-Level Splitting and Leakage Controls

For development-stage experiments, the 100 training engines are split into training and validation subsets at the **engine level**. Complete engine trajectories are assigned to either the training or the validation set; no engine appears in both. This is essential for leakage prevention: because cycles from the same engine are temporally correlated, a row-level random split would result in future cycles of a given engine appearing in training while earlier cycles appear in validation, or vice versa, giving the model an inflated and unrealistic performance estimate.

The seed-42 development split assigns 80 engines to training (80%) and 20 engines to validation (20%), as shown in Table 4.2.

**Table 4.2: Seed-42 engine-level split**

| Subset | Engines | Rows |
|---|---|---|
| Training | 80 | 16,340 |
| Validation | 20 | 4,291 |
| Official test (separate) | 100 | 13,096 (rows); 100 endpoints |

For the repeated-validation protocol (Chapter 5, Section 5.8), the same 80/20 engine-level split procedure is repeated with three different random seeds (21, 42, and 84). The model random state is kept fixed at 42 across all splits to isolate the effect of the data cohort.

The full leakage-prevention decisions are summarised in Table 4.3.

**Table 4.3: Leakage prevention decisions**

| Area | Decision |
|---|---|
| Train-validation split | Engine-level; no row-level randomisation |
| Scaler fitting | Fitted on training split only; applied to all splits |
| Rolling statistics | Computed within each engine unit using past cycles only |
| Delta features | Computed from the engine's own first cycle |
| Sequence windows | Each window spans cycles within one engine only |
| cycle_index | Uses observable cycle number; leakage-prone `normalized_cycle_age` removed |
| Metadata exclusion | unit_number, time_in_cycles, RUL, RUL_capped excluded from model inputs |
| Official test | Labels accessed only after pre-test freeze manifest; evaluated exactly once |

---

## 4.10 Sequence Construction for Neural Models

The CNN1D and GRU baseline models, as well as the raw-sequence branch of MultiViewGRUFusion, require a fixed-length time series as input. Training samples for these models are constructed by sliding a window of 30 consecutive cycles over each engine's training history:

- For each engine unit, all possible 30-cycle windows are extracted. If an engine has fewer than 30 cycles, it is excluded from the windowed dataset.
- Each window is labelled with the RUL (capped at 125) of its final cycle.
- The window covers cycles *t − 29* through *t*, where *t* is the prediction point.

This construction is performed separately for training and validation data. For the **official test**, each test engine provides exactly one window aligned to its final observed cycle (the endpoint), giving one prediction per engine.

The window size of 30 cycles was selected based on a sensitivity analysis reported in Appendix E (Table E.2), which evaluated CNN1D performance under window sizes of 20, 30, and 40 cycles. The 30-cycle window provided a good balance between capturing sufficient temporal history and maintaining training efficiency.

---

*Figures referenced: Figure 4.1 (E49 — sensor trends), Figure 4.2 (E50 — sensor vs RUL correlation)*  
*Tables in this chapter: Table 4.1, Table 4.2, Table 4.3*
