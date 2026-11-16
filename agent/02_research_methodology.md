# 02 — Research Methodology and Scientific Guardrails

## Implementation Update: Config-Controlled Missing Handling (2026-11-17)
The researcher explicitly authorized implementation of no-imputation training, retaining the current feature cohort; the broad-feature importance experiment is deferred. Current `state_imputation.mode` in `configs/decisions.yaml` is `none`, with `default_method: none` and empty overrides. `plan` remains available as an opt-in per-feature imputation mode. Numerical NaNs reach the estimator; categoricals remain one-hot encoded without imputation. The advanced MICE example is `configs/examples/decision.example.yaml` and is not loaded automatically. Validation design, dataset, target and optimization coefficients are unchanged. Historical audit statements below concern the pre-fix code or old artifacts, not a new native-model result.


## End-to-End Logic
1. Define objectives and agronomic questions.
2. Inspect/filter the Senegal dataset for rice and valid yield observations.
3. Audit variable semantics, missingness, structural zeroes, target units, duplicates, and experimental dependence.
4. Select a defensible modeling feature set and document dataset construction.
5. Separate holdout data and build cross-validation according to the approved strategy.
6. Fit preprocessing **within training folds**; fit/tune/evaluate candidate predictors.
7. Diagnose residuals, validation stability, feature dependence, and NPK response behavior.
8. Select/freeze a surrogate and its preprocessing/feature schema.
9. Run Differential Evolution over allowed N/P/K combinations for fixed state scenarios.
10. Assess predicted improvement, fertilizer efficiency, sensitivity, optimization stability, extrapolation risks, and agronomic plausibility.

The stages can be iterative, but the latest supervisor direction is to **proceed to optimization without deleting more rows from the current active dataset**.

## Data and Missingness
- Original data includes partially observed soil/environment fields; missingness varies by variable and may overlap unevenly across experiments.
- Reconstruct fertilizer structural zeroes only when supported by treatment indicators and explicit rules. Unknown fertilizer dose is not necessarily a true zero.
- Never impute the observed yield target for supervised training.
- Earlier approach: feature-specific simple imputers and `IterativeImputer` (MICE-style) with Bayesian Ridge, inside `ColumnTransformer`/`Pipeline`.
- Latest reported experiment: **native XGBoost/CatBoost missing handling**, **remove `clay.perc1`**, prioritize select soil/environment state variables over some source-identification attributes, and apply MILM-based row selection to a designated dataset.
- Do not assert native missing handling solves missing-not-at-random bias, sample-selection bias, or missingness-related extrapolation.
- When investigating performance gains, note that model family, feature set, missing strategy, and sample composition changed together; causal attribution requires controlled ablations.

### MILP Dataset Selection (Previously Reported as MILM)
The earlier researcher report called this process **MILM**. The 2026-11-17 audit located **Mixed Integer Linear Programming (MILP)** in `notebooks/exploration_notebooks/02_balancing_data.ipynb`, using `scipy.optimize.milp`.

Verified formulation: binary row-retention variables, a hard retained-row count, complete target and N/P/K as eligibility requirements, and nonnegative slack for each column's missingness limit. The objective minimizes summed excess missing fractions. It does not balance article/group representation, yield values, or dose distributions. Target values are not part of the selection objective; target completeness is an eligibility condition.

The stored notebook run retains 800 of 1,682 rows and reports solver optimality, while all three soft missingness targets remain unmet. This is consistent with the formulation; optimality does not mean all targets are satisfied. Exact generation parameters for the configured 1,000-row cohort remain incompletely traced. See FIND-005 in `06_research_findings.md`.

This documents an inspected historical construction method, not approval for additional row selection or a methodological change.

Treat the selected dataset as a **specific experimental cohort**, not a random representative sample without evidence. Log per-column missingness, retained groups/years, target distribution, and before/after characteristics when artifacts are available.

## Modeling
**Audit clarification — 2026-11-17:** Native missing handling remains the intended experiment, not the verified behavior of the existing minimalist bundles. Both inspected XGBoost/CatBoost `*_native` bundles use MICE. The notebook omits the `imputation_mode` argument and the resolver routes method `none` into the MICE list. Do not use these outputs as evidence of native-versus-MICE superiority. See FIND-007 and DEC-006. No methodology change or corrected experiment was executed during the audit.

Target: `yield.wb.kg_ha` (kg/ha). Check if all preprocessing preserves its intended scale and units.

Potential families from proposal: Linear Regression, Random Forest, XGBoost, CatBoost. The new baseline reportedly uses XGBoost and CatBoost natively handling missing values. Do not assume the code's training pathway enables native handling simply because those estimators support it; inspect the preprocessor for any imputation that still runs.

Metrics: MAE, RMSE (kg/ha), and R², across appropriate folds and holdout. Compare with meaningful simple baselines. Report train-vs-validation gaps and variations across groups if available. Holdout must not guide iterative tuning choices without an explicit evaluation plan.

## Splitting and Leakage
The uploaded historical configuration uses `group_kfold`, `group_article_experiment`, five CV splits, a 20% holdout, and seed 42. The training implementation separates train/test and uses `GroupShuffleSplit` for group-aware holdout and `GroupKFold` within training.

Validate:
- No record/group overlap across holdout boundary.
- No target-derived features, leaked group statistics, or target information in predictors.
- Imputers, encoders, scalers, and learned transformations fitted per fold rather than once on the whole dataset.
- Any dataset balancing, feature selection, or row filtering performed with target/outcome information is audited for evaluation consequences.
- CV used in tuning matches the intended generalization scenario.

If a new method improves random-CV performance but worsens group generalization, do not declare it better solely based on random-CV metrics.

## Surrogate Suitability Beyond Predictive Error
A surrogate suitable for NPK optimization should be evaluated not only for aggregate yield accuracy but also for its response when `x=[N,P,K]` changes while `s` stays fixed. Questions:
- Do predictions vary meaningfully and plausibly across supported fertilizer dose ranges?
- Is the model exploiting source/study proxies rather than fertilizer response?
- Does it return similar predictions across many distinct fertilizer combinations?
- Are recommended candidates near model/domain boundaries or sparse regions?
- Are engineered fertilizer features recalculated whenever doses change?
- Are model response curves being interpreted as *modeled associations*, not causal treatment effects?

Horizontal prediction bands in actual-versus-predicted plots are a finding to diagnose, not automatic proof of a specific failure mode.

## Differential Evolution Formulation
For a fixed state scenario `s`, let `x=[N,P,K]`.

`y_base_hat = f_hat(x_base; s)`

`delta_y_hat(x;s) = f_hat(x;s) - y_base_hat`

`J(x;s) = delta_y_hat(x;s) - lambda_T * Phi_T(x) - lambda_E * Phi_E(x)`

Find the highest-scoring feasible `x` within approved agronomic/data-supported bounds using DE. DE is a heuristic search algorithm; the best found solution is not a mathematical guarantee of a global optimum.

The proposal discusses:
- Per-nutrient lower/upper bounds derived from data support intersected with agronomic acceptability.
- A total-dose soft threshold and optional hard maximum.
- Composition proportion bounds and squared violation penalties.

Do not silently invent `lambda_T`, `lambda_E`, proportions, feasibility thresholds, or reference scenario policies. Existing older values are provisional until checked against the latest approved configuration and agronomic justification.

Note: maximizing `delta_y_hat` for a fixed state and fixed baseline is mathematically equivalent to maximizing `f_hat` **if penalties do not depend on the baseline**; the delta formulation clarifies comparison, not causal improvement.

### Optimization Correctness Checks
- The optimizer changes only N/P/K.
- Soil/weather state remains fixed for one run.
- Baseline and candidate yield both come from the **same** fitted model under the **same** state.
- Any derived `total_npk` and ratios are regenerated using the same training rules.
- Input features, order, units, types, categories, and missing indicators align with the trained pipeline.
- Minimize-vs-maximize sign convention is correct in the DE API.
- Hard constraints are respected and penalties are correctly scaled.
- Fixed seeds and repeated runs support reproducibility and stability analysis.
- Compare selected candidate with the baseline and simple search strategies when feasible.

## Evaluation of Optimization
- **Productivity:** model-predicted yield and delta versus model-predicted baseline.
- **Efficiency:** calculate fertilizer-use-efficiency measures according to the proposal, carefully handling zero/negative denominator and interpretation.
- **Sensitivity:** perturb one state variable at a time, rerun optimization, and compare N/P/K changes; normalize cautiously when scales differ.
- **Stability:** perturb state inputs and repeat optimizer seeds; distinguish input stability from algorithmic stability.
- **Agronomic review:** check boundary concentration, unsupported combinations, implausible dose-response patterns, and expert feedback when available.

## Scientific Reporting Rules
For every meaningful experiment report:
1. Dataset version and exact feature set.
2. Split strategy and seed.
3. Model, parameters, preprocessing, and tuning budget.
4. Quantitative evaluation with uncertainty/variation where available.
5. Constraints, limitations, and what conclusions the result does *not* justify.
6. Paths to actual evidence artifacts.

Never present predicted or simulated fertilizer recommendations as empirically verified field prescriptions.
