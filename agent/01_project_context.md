# 01 — Project Context

## Identity and Research Aim
Working title: **Pengembangan Sistem Optimasi Pemupukan Tanaman Padi Menggunakan Pendekatan Machine Learning dan Metaheuristic Optimization**.

This undergraduate Information Systems thesis investigates a **surrogate-based fertilizer optimization** workflow for rice. It uses historical agronomic observations to learn predicted crop yield from fertilizer doses and land conditions, then searches for candidate N/P/K doses with **Differential Evolution (DE)**. The system is a model-based decision-support research prototype, not a field-proven fertilizer prescription.

## Research Questions
1. How can machine learning predict rice yield from N/P/K fertilizer doses, soil characteristics, and environmental factors?
2. How can a trained surrogate model support optimization of N/P/K dosage using metaheuristic optimization?
3. How can optimization quality be assessed through predicted productivity, fertilizer efficiency, and solution stability?
4. How do soil and environmental conditions relate to changes in optimized doses, assessed by sensitivity analysis?

## Conceptual Model
Inputs:
- **Decision variables** `x = [N, P, K]`: fertilizer application rates, in kg/ha.
- **State variables** `s`: observed/estimated soil and environmental conditions; held constant within one optimization scenario.

Prediction: `y_hat = f_hat(x; s)`, where `y_hat` is predicted wet-basis rice yield (kg/ha).

Optimization: choose `x*` to improve the surrogate's predicted yield relative to a reference dose `x_base`, while respecting approved constraints and penalty terms.

`delta_y_hat = f_hat(x; s) - f_hat(x_base; s)`.

This predicted improvement is **not** evidence of actual field improvement or a causal treatment effect.

## Dataset
Research dataset: **Towards Site-Specific Nutrient Management Strategies: An Open Database in Senegal** (Figshare DOI: `10.6084/m9.figshare.27022285.v1`). The original multi-crop compilation spans 1989–2022, approximately **2,992 records and 125 attributes** before research-specific filtering. The thesis focuses on rice. Actual row count after filtering must be read from the current data artifacts.

Primary target in the provided config: `yield.wb.kg_ha`.

Identifiers/context available in earlier project work: `record_id`, `id` (article), `exp.id`, `year`, `location`, and composite experimental groups. These are not automatically valid prediction features; distinguish grouping/identification from permissible model inputs.

Feature families:
- Fertilizer: N/P/K rates and, when enabled, derived total/ratios.
- Soil: e.g. `initial.ph.soil`, `soc.percentage`, `soil_type`; `clay.perc1` was removed in the latest reported minimalist experiment because of extensive missingness.
- Environment: e.g. `avprec.mm`.

Exact active feature list must be verified in the latest repository/configuration. Do not infer its complete membership from this overview.

## Predictive Modeling
Original comparative design: Linear Regression, Random Forest, XGBoost, CatBoost; MAE, RMSE, R², and cross-validation/holdout evaluation. The earlier code snapshot uses a group-based strategy with `group_article_experiment` and randomized hyperparameter search.

**Latest reported experimental direction (November 2026):** focus on native missing-value handling with XGBoost/CatBoost, a `minimalist_feature` profile, and a selected balanced dataset. This is a new experimental baseline, not proof that every older config file has been updated.

## Optimization and Analysis
- Optimizer: Differential Evolution.
- Fixed within a scenario: soil and environmental state `s` and the fitted surrogate.
- Varied by optimizer: N, P, K, and any derived fertilizer input generated consistently from them.
- Objective: maximize predicted improvement over the same-model baseline, net of approved penalties.
- Hard constraints: agronomically and empirically justified dose/search bounds, with optional approved total-dose constraint.
- Soft constraints: proposal discusses penalties on excessive total fertilizer and extreme NPK composition.
- Evaluation: predicted yield improvement, fertilizer-use-efficiency indicators, sensitivity to state changes, stability under perturbations/optimizer seeds, and agronomic plausibility.

## Scope and Limits
- Uses secondary Senegal experimental data, not new field trials.
- Studies rice rather than all crops.
- Does not claim universal applicability to Indonesian rice fields.
- Does not directly optimize monetary profit, supply chains, logistics, or measured environmental damage.
- Expert review of agronomic plausibility is a planned validation stage, not automatically completed.

## Fundamental Research Risks
1. High and nonrandom missingness in soil/environment variables.
2. Potential selection bias from retaining more complete rows.
3. Correlation within source studies and experiments; overly optimistic evaluation from leakage.
4. Predictive accuracy may be driven by context rather than dose-response learning.
5. Optimizer may exploit surrogate errors or poorly supported dose combinations.
6. Predictions, model explanations, and response curves must not be labeled causal effects without suitable evidence.

## Researcher Collaboration Preferences
Use Indonesian, straightforward but analytically rigorous. Work as a thinking partner: explain assumptions, mechanisms, alternatives, risks, and downstream effects. Challenge weak reasoning constructively. Avoid long explanations for trivial edits; avoid large code changes without justification.

## Primary References
- Thesis proposal PDF in the research repository (locate actual path).
- Current `configs/`, `src/`, notebook outputs, and experiment artifacts.
- Decisions, findings, and status files under `agent/`.
