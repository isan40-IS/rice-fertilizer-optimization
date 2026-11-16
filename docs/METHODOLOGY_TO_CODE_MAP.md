# Methodology-to-Code Map

| Thesis component | Code |
|---|---|
| EDA and data preparation | `data/` and notebooks 00–01 |
| Structural NPK missingness | `data/missing_values.py` |
| NPK ratios and total | `data/feature_engineering.py` |
| LR, RF, XGBoost, CatBoost | `modeling/model_registry.py` |
| CV and randomized search | `validation/splitters.py`, `modeling/tuner.py` |
| MAE, RMSE, R² | `modeling/metrics.py` |
| Interpretability | `modeling/interpretation.py` |
| Surrogate model | `surrogate/predictor.py` |
| Decision/state separation | `surrogate/scenario.py` |
| Objective and penalties | `optimization/objective.py`, `penalties.py` |
| Differential Evolution | `optimization/differential_evolution.py` |
| Productivity | `evaluation/productivity.py` |
| Fertilizer efficiency | `evaluation/fertilizer_efficiency.py` |
| Sensitivity | `evaluation/sensitivity.py` |
| Input stability | `evaluation/input_stability.py` |
| Algorithm stability | `evaluation/algorithm_stability.py` |
