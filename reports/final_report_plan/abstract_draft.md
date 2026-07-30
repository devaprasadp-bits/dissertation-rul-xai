# Final Abstract
## 2023AA05069 | Devaprasad P

*Prescribed abstract — ≤200 words. Use this text verbatim for the BITS abstract sheet.*

---

This dissertation presents a leakage-safe and reproducible multi-view study of Remaining Useful Life (RUL) prediction for turbofan engines using the NASA C-MAPSS FD001 dataset. The work compares raw sensor sequences, engineered degradation features, and a GRU-based multi-view fusion architecture against classical XGBoost across a controlled three-stage evaluation: original seed-42 development, repeated validation on three independent engine-grouped splits, and a single one-time official held-out endpoint evaluation.

On repeated validation, DerivedOnlyMLP achieved the lowest mean RMSE of 14.23, while MultiViewGRUFusion was competitive at 14.28. On the official held-out endpoint test, XGBoost ranked first with RMSE 12.25, followed by DerivedOnlyMLP at 12.83, GRU at 13.29, and MultiViewGRUFusion at 13.38. Multi-view fusion did not consistently outperform the strongest single-view or classical alternatives across cohorts or the final evaluation.

Explainability analysis using SHAP and feature importance confirms that engineered degradation features, particularly cycle index and derived sensor statistics, carry the dominant predictive signal. Trustworthiness studies covering noise robustness, range-wise error analysis, prediction bias, and exploratory MC Dropout uncertainty extend the evaluation beyond accuracy metrics. The dissertation contributes a controlled experimental assessment of how different representations behave in predictive maintenance rather than a single best-model claim.

---

*Word count: 192*

---

## Notes for Abstract Sheet

- Keep exactly as written above; do not reword for brevity
- RMSE values are rounded to 2 decimal places for readability in the abstract
- Full precision values: XGBoost 12.2459, DerivedOnlyMLP 12.8295, GRU 13.2860, Fusion 13.3782
- The abstract sheet requires the student number, dissertation title, supervisor name and date — fill those from the prescribed template
