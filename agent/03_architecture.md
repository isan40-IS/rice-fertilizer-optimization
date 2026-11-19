# 03 — Architecture and Code Navigation

## Experiment-D diagnostics (2026-11-17, latest)
`evaluation/surrogate_diagnostics.py` loads/validates the frozen C candidate, dataset/schema/split/CV hashes and identities; supplies one shared scenario preparation/prediction path, numeric parsing validation, NPK feature recomputation using the bundle pseudocount, training-only continuous distance references, conservative empirical pairing, and explicit manual-parameter evidence. No fitting or optimization occurs. `notebooks/2_model_experiment/experiment_D.ipynb` covers provenance, data dictionary/distributions, holdout diagnostics, editable playground, dose/state/interaction responses, matching/domain diagnostics, consistency checks and parameter worksheet. Outputs live only in a unique `artifacts/experiment_D/<run_id>/`; C source artifacts/configs remain unchanged. Fertilizer control semantics are checked against metadata: positive NPK with unchanged `No fertilizer` is rejected; automatic positive-dose sweeps skip control cases. Domain distances have no invented cutoff; composition bounds remain soft. Notebook is locally ignored by Git, as are other experiment notebooks.


## Experiment-C implementation (2026-11-17, latest)
`tuner.py` dispatches finite-list random/grid/Optuna search with the legacy API, score maximization, consistent training folds, grid cap, parameter validation (including CatBoost public defaults), optional lazy Optuna import and fail-fast failure exports. `trainer.py` records histories/summary/provenance/fold manifests and supports holdout reporting disabled. `pipeline_factory.NonEmptyColumns` optionally filters all-missing columns on each fit's training data only. Importance name/value mismatches now raise instead of silently truncating.
`pipelines/run_experiment_c.py` and `scripts/run_experiment_c.py` run fresh method/search-seed studies, snapshot effective overrides, check fingerprints, and freeze a CV-only candidate. Holdout is a separate explicit action. `configs/experiment_C/` isolates C settings (minimalist, no imputation, two models, fixed model/split seeds); the B dataset is read-only and C preprocessing refuses overwrite. Notebook C separates guarded execution and reading/analysis. Optional dependency extra `tuning` / `requirements-tuning.txt` is available. Optuna 5.1.0 tested. Persistent resume, continuous domains and pruning are explicitly unsupported in phase one. See `docs/EXPERIMENT_C_USAGE.md`.


## Experiment-B Configuration Readiness (2026-11-17, Latest)
Active settings now use the broad `feature_importance_candidates` profile, separate processed output `data/processed/experiment_B_candidates.csv` and artifacts root `artifacts/experiment_B`. The cohort reference is the existing 1,000-row minimalist CSV. `data/cohort.py` aligns the same IDs/order; input/reference hashes, exact count and preserved reference values are checked before writing. Preprocessing unions declared profile numerics with configured numeric columns, parses decimal commas, validates coordinate ranges, reconstructs existing structural zeroes, regenerates NPK features and refuses reference/raw overwrite. Original preprocessing without cohort mode remains available as `all_eligible`; no cohort change was made here. Output audit records processed/cohort hashes. Details/commands: `docs/EXPERIMENT_B_PLAN.md`. Earlier implementation/audit sections below are historical snapshots.


## Implementation Update (2026-11-17, After Audit)
- `resolve_imputation_plan` reads global `state_imputation.mode`; `none` suppresses all imputers, while `plan` applies configured methods. Per-column `none` is now distinct from MICE. Categorical methods are respected instead of always applying mode imputation.
- `build_preprocessor` can infer mode from the resolved plan; `trainer.py` explicitly forwards it. Broader profiles listing N/P/K numerically do not cause duplicate dose inputs.
- Training stores `imputation_mode` in metric summaries and bundles plus the effective `state_imputation` configuration in bundles. The fitted pipeline continues to govern inference.
- Active config keeps `minimalist_features`, the same 1,000-row dataset and grouped validation. XGBoost/CatBoost are enabled; the other model definitions are retained but disabled. MICE settings are in the opt-in `configs/examples/decision.example.yaml`.
- Experiment A's training cells now call shared `run_training`, use config settings without local overrides, retain selection weights and create a fresh run ID. Changed training-cell outputs/execution counts were cleared; later analysis is explicitly historical. No notebook training was run.
- Corrected focused pipeline/config tests and added regression checks for native NaNs, MICE/simple/none combinations, categorical missingness, no target/dose imputation, serialization and trainer mode wiring. Entire suite: **39 passed**. Tests use synthetic data; historical scientific bundles remain unchanged.

The audit section below is retained as pre-fix evidence; implementation problems described there are corrected in source, while the inspected old artifacts still have their audited behavior.


## Important Version Warning
The historical sections below describe the earlier uploaded partial snapshot and are retained for traceability. A direct repository/artifact audit on **2026-11-17** verified the current components listed next. Read the current audit before relying on a historical path or setting.

## Verified Current Architecture — 2026-11-17
- Library root: `src/rice_fertilizer/`; thin entry scripts in `scripts/`; orchestration in `pipelines/`.
- `configs/decisions.yaml` selects `minimalist_features`; `configs/paths.yaml` points to `data/processed/model_dataset_balanced_1000_minimalist_504622.csv`.
- Actual model input is profile numeric + categorical + global decision features: pH, precipitation, SOC, total NPK, three NPK ratios, fertilizer type, and direct N/P/K doses (11 columns). The minimalist profile excludes clay, soil type, and nutrient-source categories.
- `modeling/pipeline_factory.py` supports `imputation_mode="none"` for numerical passthrough; categoricals still use one-hot encoding rather than native CatBoost categorical inputs. `modeling/trainer.py` does not pass this mode and uses the default `plan`.
- `modeling/imputation.py` accepts method `none` but incorrectly appends it to `mice_columns`. In `notebooks/2_model_experiment/experiment_A.ipynb`, the loop computes `imputation_mode` but omits it from `build_model_pipeline()`. Inspected native-labeled XGBoost/CatBoost bundles for both minimalist runs contain MICE; see FIND-007.
- The notebook overrides the entire `configs["models"]` mapping, losing selection settings. Its ranking receives an empty configuration and produces all-zero selection scores; see FIND-008.
- Dataset selection is MILP in `notebooks/exploration_notebooks/02_balancing_data.ipynb`. The stored run is 800 rows; precise generation parameters for the 1,000-row artifact are not fully recovered.
- Experiment artifacts: `artifacts/exp_A_comp_miss_val_mech_minimalist/` (1,000 rows) and `artifacts/exp_A_comp_miss_val_mech_minimalist_383327/` (800 rows). The notebook training section uses the latter while later analysis references the former; do not merge their results.
- The model registry and main model configuration also include LightGBM and Extra Trees, in addition to the original four families. Experiment A compares five native-labeled/imputed variants, including LightGBM.
- Optimization is implemented in `optimization/` (`differential_evolution.py`, `optimizer.py`, `objective.py`, constraints and penalties); orchestration is `pipelines/run_optimization.py`, exposed by `scripts/optimize_scenarios.py`.
- `surrogate/candidate_builder.py` reconstructs direct N/P/K and derived features with `add_npk_features`; `SurrogatePredictor` uses the fitted pipeline and bundle pseudocount. Scenario state remains fixed. Runtime response/semantic checks are still pending.
- `evaluation/baseline.py` computes group-mean doses from the context dataframe supplied by the optimization pipeline (currently the full configured processed dataset). This does not use yield values for group means, but deployment/unseen-group policy remains open.
- Evaluation/reporting implementations exist for productivity, fertilizer efficiency, sensitivity, input/algorithm stability, domain support, tables and plots. Existence is not proof of a completed scientific evaluation.
- Active objective construction requires non-null `yield_scale`, `lambda_total`, and `lambda_composition`; these are still unset. No DE execution was performed during the audit.
- `tests/integration/test_model_pipeline.py` still calls an old pipeline interface, and `tests/e2e/test_smoke_configuration.py` still expects unset feature/bound settings. Historical test-success documentation is not a verified current test result; pytest was not run in this audit.

## Conceptual Pipeline
`Raw Senegal data → data validation and cleaning → rice selection → structural-zero handling → dataset/feature construction → train/holdout split → CV and model tuning → prediction evaluation → selected surrogate bundle → DE optimization → sensitivity/stability/agronomic evaluation`

This is the intended architecture. Historically the uploaded snapshot chiefly covered training and validation; the current checkout also contains the optimizer and evaluation implementations described above.

## Located Components in Uploaded Snapshot
Paths below are **logical roles**; determine exact current repository locations by searching filenames/imports.

| File | Role |
| --- | --- |
| `trainer.py`  | `run_training`: loads processed data/config, selects feature profile, splits data, checks leakage, loops over models, tunes, evaluates, exports, selects a model |
| `splitters.py` | `make_holdout_and_cv`: random/group/temporal strategies; `ExpandingWindowSplit` |
| `leakage_checks.py` | Group and record overlap checks; forbidden feature detection helper |
| `imputation.py` | Resolves per-column MICE/simple-imputation plan; constructs imputers |
| `pipeline_factory.py`  | Builds `ColumnTransformer`/`Pipeline`; old snapshot imputes configured state features |
| `model_registry.py` | Constructs Linear Regression, Random Forest, XGBoost, CatBoost regressors |
| `tuner.py` | `RandomizedSearchCV` with optional no-search direct fit |
| `evaluator.py` | CV fold evaluation, summary, holdout evaluation |
| `metrics.py` | MAE, RMSE, R² calculation |
| `interpretation.py` | Built-in and permutation feature importance |
| `model_selector.py` | Weighted standardized metric ranking and eligibility check |
| `serializer.py` | Saves/loads joblib model bundles |

## Historical Training Control Flow
1. `run_training(configs, run_dir)` reads `processed_dataset`.
2. Gets `feature_profile`, target, split strategy, and feature lists from config.
3. Resolves state-feature imputation plan.
4. Calls `make_holdout_and_cv` to produce holdout and training-only CV.
5. Writes split assignments and leakage-check summaries.
6. Loops over enabled model definitions from model config:
   - create estimator;
   - assemble preprocessing+model pipeline;
   - random-search hyperparameters (or fit directly if no search space);
   - evaluate cross-validation and holdout;
   - save predictions, metrics, importance, model bundle, best params.
7. Ranks candidates and writes selected model bundle.

**Caution:** In the old snapshot, `pipeline_factory.py` explicitly builds MICE/simple imputation branches; merely selecting XGBoost or CatBoost from the registry would not mean that those missing values reach the estimators natively. Audit and document the new native-missing pathway.

## Historical Config Snapshot
- `decisions.yaml`: `feature_profile: modeling_features`; `validation_strategy: group_kfold`; group `group_article_experiment`; target `yield.wb.kg_ha`; `baseline_policy: group_mean`; MICE configuration for selected soil/environment fields.
- `validation.yaml`: holdout 0.20, 5 folds, seed 42, group/random/temporal strategies.
- `models.yaml`: four enabled models, randomized search with 30 iterations, negative RMSE as tuning score, weighted selection metrics.
- `evaluation.yaml`: marginal-step rules, sensitivity and input/algorithm stability seed settings; `fraction_of_iqr` remains null in old snapshot.

These are **old uploaded configuration facts**, not proof of settings in the newest experiment.

## Historical New-Experiment Navigation (Researcher Report; Corrected Above)
- Feature profile: `minimalist_feature` (exact config filename/location to locate).
- Dataset identifier: `model_dataset_balanced_1000_minimalist_504622` (find actual file/extension/path).
- Notebook: `notebooks/2_model_experiment/experiment_A.ipynb`.
- Dataset-selection procedure: `balancing_dataset` / MILM (locate actual implementation).
- Focus models: native missing XGBoost and CatBoost.

### Efficient Notebook Inspection
Avoid opening an entire large `.ipynb` as raw text. Use `nbformat` or JSON parsing to summarize:
- code cell sources and execution order;
- output types and traceback/error summaries;
- metrics/tables relevant to the task;
- filenames of exported plots/artifacts.

Prefer exported small CSV/JSON metrics when available. If a chart is relevant, inspect the plot image or the specific notebook cell rather than making claims from text alone. Distinguish stored outputs from a reproducibly rerun notebook.

## Model Bundle Contract (Old Snapshot)
The saved bundle contains the fitted pipeline plus metadata including model name, feature profile, numeric/categorical/feature columns, target, ratio pseudocount, validation strategy, CV summary, holdout metrics, best params, training statistics, training reference numeric data, and observed categorical levels.

Before optimization, inspect the **actual new bundle** and verify prediction schema, native missing support, feature engineering, and serialization loading.

## Repository Discovery Checklist
1. Locate `configs/features.yaml`, any updated `decisions.yaml`, and new native-missing switches.
2. Trace `balancing_dataset` generation and dataset lineage.
3. Find entry scripts for preprocess/train/optimization, not only library functions.
4. Find and inspect optimizer implementation and decision/evaluation configuration.
5. Identify tests, experiment outputs, and model artifact directories.
6. Update this architecture map when actual current code diverges from the uploaded snapshot.

## Change Impact Rules
- Changing preprocessing affects train/inference/DE-input consistency.
- Changing selected features affects model bundles and optimizer feature schemas.
- Changing group assignment affects validation interpretation.
- Changing search bounds/penalties affects scientific claims about optimality.
- Changing balancing/row retention can change target distribution and experiment coverage.

When fixing a component, inspect upstream assumptions and downstream consumers before editing.
