# Chapter 3 — Dataset and Problem Formulation

---

## 3.1 NASA C-MAPSS Dataset

The dataset used in this dissertation is the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset [Saxena et al., 2008]. C-MAPSS is a widely used prognostics benchmark that provides simulated run-to-failure degradation trajectories for aircraft turbofan engines under controlled operating conditions.

Each engine unit in the dataset is observed over a series of operating cycles. For every cycle, the record contains the engine unit identifier, the cycle counter, three operating condition variables, and twenty-one sensor measurements representing different aspects of engine behaviour during its lifecycle. As the engine accumulates operating cycles, degradation progresses monotonically until the unit reaches its failure point. This structure makes the dataset suitable for studying how sensor readings change over time and how remaining useful life can be estimated from a partial operating history.

The dataset consists of four subsets — FD001 through FD004 — each varying in the number of operating conditions and fault modes. The full dataset was created using a thermodynamic model of turbofan degradation and is freely available from the NASA Prognostics Center of Excellence.

---

## 3.2 FD001 Subset: Structure and Characteristics

This dissertation uses the FD001 subset exclusively. FD001 operates under a single operating condition and a single fault mode, making it the most controlled subset in the collection. This controlled setting is appropriate for the initial benchmark study described here, as it eliminates the additional confounding factor of regime-dependent sensor variation and allows the modelling pipeline, feature engineering, and multi-view formulation to be developed and evaluated without operating-condition normalisation.

The FD001 training set contains 100 engine units. The test set also contains 100 engine units, each observed up to a point prior to failure with an associated endpoint RUL provided in a separate file. The training and test engines are independent populations.

The column structure of the raw FD001 data is shown in Table 3.1.

**Table 3.1: FD001 raw column structure**

| Column group | Count | Description |
|---|---|---|
| Unit identifier | 1 | Engine unit number (1–100 in train; 1–100 in test) |
| Cycle counter | 1 | Time in operating cycles, starting at 1 for each unit |
| Operating settings | 3 | Operating condition variables (near-constant in FD001) |
| Sensor measurements | 21 | Engine sensor measurement columns (s1–s21) |
| **Total** | **26** | Input data columns before target construction |

The three operating settings are near-constant in FD001 due to the single operating condition; they are not used as model inputs. Of the 21 sensor measurements, several are constant or nearly constant across the entire FD001 dataset and carry no degradation information. Exploratory data analysis identified 14 variable sensor measurements as informative. The remaining 7 sensors are excluded from all feature sets used in this work.

The training engines have cycle lengths ranging from 128 to 362 cycles, with a median of approximately 206 cycles. The total number of training rows is 20,631. The distribution of engine cycle lengths is shown in Figure 3.4.

---

## 3.3 Sensor Selection and Operating Condition

The 14 sensor measurements retained after exploratory filtering are the core inputs to all models in this study. These sensors correspond to measurements such as fan speed, low-pressure compressor discharge temperature, high-pressure turbine inlet temperature, bypass ratio, and bleed enthalpy, among others. Full sensor names and their correlation with RUL are shown in Figure 4.2.

The single operating condition in FD001 means that operating-setting normalisation, which is required for FD002–FD004, is not needed here. This simplifies the preprocessing pipeline and removes one potential source of variance in the results.

---

## 3.4 Capped-RUL Target Formulation

For the training data, the true failure cycle of each engine unit is known. The RUL for any cycle within the training history is computed as:

> **RUL = final cycle of unit − current cycle**

For the official test data, the endpoint RUL for each test engine is provided in a separate file by NASA. This file gives the number of remaining cycles at the last observed cycle of each test unit.

In practical predictive maintenance, very high RUL values during the early operating life of an asset are often less meaningful than behaviour in the degradation-active region. A model trained to minimise error uniformly across all RUL ranges tends to be dominated by early-life, high-RUL samples where exact prediction is both difficult and operationally unimportant. To address this, a **capped-RUL target** is used throughout this dissertation:

> **Capped RUL = min(RUL, 125)**

All training targets are capped at 125 cycles. The official test endpoint RUL values are also capped at 125 for consistency. Of the 100 FD001 test endpoint labels, 11 exceed 125 cycles and are thus capped. Predictions from all models are clipped to the range [0, 125] before metric computation.

The effect of this capping is illustrated in Figure 3.1 and Figure 3.2. Figure 3.1 shows the raw RUL distribution in the training set. Figure 3.2 shows the comparison between uncapped and capped training targets.

An important reporting boundary follows from this formulation: the RMSE and other metrics reported throughout this dissertation reflect performance on the **capped-RUL regression task**. They are not directly comparable with results in papers using uncapped RUL targets or the asymmetric NASA scoring function, unless evaluation definitions are explicitly matched.

---

## 3.5 Regression Task Definition

The prediction task is a supervised regression problem: given the sensor history of an engine up to the current cycle, predict the capped RUL at that point. Formally, for a given engine unit *u* at cycle *t*:

> **ŷ(u, t) = f( x(u, t) )**

where *x(u, t)* is the input representation (described in Section 3.6) and *f* is the trained predictor. The target is *y(u, t) = min(RUL(u, t), 125)*.

The primary evaluation metrics are Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score. RMSE penalises larger errors more heavily, which is appropriate for predictive maintenance where large prediction errors can have material consequences for maintenance planning. MAE gives the average absolute error in cycles and is directly interpretable. R² indicates the proportion of target variance explained by the model. Mean prediction error (prediction − actual) is also reported to characterise prediction bias.

---

## 3.6 Multi-View Input Formulation

A central design choice in this dissertation is the formulation of the input as two complementary views of the same underlying sensor stream. This multi-view formulation is distinct from native multimodality (which involves physically different data sources such as images, audio, or maintenance logs) because both views derive from the same 14 numerical sensor measurements.

**View A — Raw sensor sequence view**

The raw sensor sequence view presents the 14 selected sensor measurements as a time-ordered window of 30 consecutive cycles. Each input sample is a matrix of shape (30, 14): 30 time steps, 14 features. This view gives the model direct access to recent temporal dynamics in the sensor signals.

**View B — Derived degradation feature view**

The derived degradation feature view is a flat vector of 43 engineered features, constructed from the same 14 sensors using rolling statistics computed within each engine's history: rolling mean (14 features), rolling standard deviation (14 features), delta from the engine's initial-cycle value (14 features), and one cycle_index feature (1 feature). The rolling window for these statistics is 5 cycles. This view summarises the degradation trajectory rather than exposing raw temporal dynamics.

The two views are summarised in Table 3.2.

**Table 3.2: Multi-view input formulation**

| View | Input representation | Shape | Processed by |
|---|---|---|---|
| Raw sensor sequence | 30-cycle window of 14 sensor measurements | (30, 14) | GRU encoder branch |
| Derived degradation features | Rolling mean, std, delta, cycle_index | (43,) | Dense branch |

This formulation allows the study to examine whether a model combining temporal sensor history with derived degradation summaries outperforms models using either view alone, and to quantify the contribution of each view through masking experiments.

---

## 3.7 Evaluation Populations

A recurring theme in the results chapters is that different experiments use different **evaluation populations**. These populations must not be conflated because their RMSE values are not directly comparable.

Three evaluation populations are used in this dissertation:

**Table 3.3: Evaluation populations**

| Population | Description | Used in |
|---|---|---|
| Seed-42 window-aligned validation | All eligible 30-cycle windows from validation engines under the original seed-42 split | Chapter 6, §6.1 (historical reference) |
| Repeated-validation (3 splits) | All eligible windows from validation engines across splits seeded 21, 42, and 84 | Chapter 6, §6.2–6.3 (main stability evidence) |
| Official test endpoint | One endpoint prediction per FD001 test engine (100 predictions total) | Chapter 6, §6.5 (held-out final evaluation) |

Because the repeated-validation metrics are computed over all eligible windows per split, while the official test metrics are computed over exactly one endpoint prediction per engine, the absolute RMSE values of the two populations are not directly comparable. The ranking of models across populations may be compared descriptively, but differences in absolute RMSE between populations do not represent a generalisation gap.

---

## 3.8 Scope Boundary

This dissertation is scoped to the C-MAPSS FD001 benchmark under the constraints summarised in Table 3.4. This scope is intentional: it keeps the experimental pipeline reproducible and feasible within the dissertation timeline while still permitting meaningful exploration of the multi-view, explainability, and robustness questions.

**Table 3.4: Scope of this dissertation**

| Aspect | Scope |
|---|---|
| Dataset | NASA C-MAPSS FD001 only |
| Asset type | Simulated turbofan engine degradation |
| Operating condition | Single (FD001); no regime-normalisation required |
| Fault mode | Single (FD001) |
| Prediction target | Capped Remaining Useful Life (cap at 125 cycles) |
| Problem type | Regression |
| Input views | Raw sensor sequence (30 × 14) and derived degradation features (43) |
| Data modality | Numerical time-series only |
| Multi-view vs multimodal | Multi-view (both views from same sensor stream); not native multimodality |
| Explainability methods | Feature importance, SHAP, view masking, local case analysis |
| Trustworthiness methods | Noise and masking robustness, lifecycle-region error, bias, MC Dropout |
| Production deployment | Not claimed |
| Generalisation to FD002–FD004 | Out of scope |

The scope is defined here to prevent overreading of the results reported in subsequent chapters. Chapter 9 discusses the limitations that arise directly from these scope decisions.

---

*Figures referenced: Figure 3.1 (E46 — RUL distribution), Figure 3.2 (E47 — capping comparison), Figure 3.3 (E48 — sample degradation curves), Figure 3.4 (E52 — cycle count distribution)*  
*Tables in this chapter: Table 3.1, Table 3.2, Table 3.3, Table 3.4*
