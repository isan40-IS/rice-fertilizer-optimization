# 05 - Current Working State

## no_engineered results reviewed (2026-11-19, latest)
Current experiment_B code cells point to `artifacts/no_engineered/ta_20261119T040155Z`; legacy notebook Markdown is not evidence for this run. Verified actual dataset/hash/reference values, 31 inputs (88 transformed), identical 808/192 B split IDs, no group/record overlap, 7 overlapping articles, no imputer and 1,954 preserved numerical holdout NaNs. Researcher training/export are complete; the pending-training statements in preparation history below are superseded as current status.
Random search: 30 completed unique candidates per model, search seed 11; estimator/split seed 42. CatBoost selected: CV RMSE 1898.157 ?523.927, R2 0.612106; holdout RMSE 1432.413, R2 0.739598. XGBoost: CV RMSE 1902.103 ?652.454, R2 0.603020; holdout RMSE 1520.929, R2 0.706421. Two-model CV RMSE difference is only 3.946; holdout is previously inspected. N/P lead permutation on both; K is negative for XGBoost (-6.431) but positive for CatBoost (48.662 kg/ha). XGBoost soil_type_nan alone has 34.376% built-in importance. No causal masking or accuracy improvement from derivative removal is established.
Documentation renamed/expanded to `docs/NO_ENGINEERED.md` with findings and implications. Cell 12 still asserts 97 inputs and stores AssertionError; current artifacts and downstream 31-input numerical outputs were independently checked read-only. Notebook was not fixed or rerun. Optional family permutation remains unexecuted. Training has only 33 unique NPK combinations, and generic optimization holds product rates/concentrations/sources fixed while changing doses; scenario semantics must be reviewed. Suggested next action: frozen CatBoost response/domain diagnostics, with special attention to K and product/source consistency, then researcher-approved objective decisions before DE. C/D candidate/history remain unchanged; no final replacement or optimizer values approved. No fit, training, new permutation/response curves or DE performed during this documentation task. See FIND-017 and updated backlog.

## Historical preparation state (before the researcher run)

## no_engineered configuration ready (2026-11-19, latest root configuration)
Researcher requested a reduced profile based on experiment-B XGBoost >=0.5% importance, without derivatives or latitude/longitude, retaining research-required features. Root config now selects `no_engineered` (31 inputs: 17 numeric / 14 categorical), output `data/processed/no_engineered.csv`, artifacts `artifacts/no_engineered/`. Historical B/C/D artifacts and isolated C configs remain unchanged. Earlier root-active-profile statements below are historical; immediately before this change the actual root decisions selected `minimalist_features` despite stale B comments.
Selection evidence is the broad B run `ta_20261116T203731Z`: 46 encoded columns map to 38 raw features; excluding seven derivatives/two selected coordinates leaves 29, plus initial pH/fertilizer type = 31. At preparation time the notebook had an inconsistent 11-input aggregation; this is historical, not the current no_engineered aggregation. See DEC-020 and `docs/NO_ENGINEERED.md` for selected names, lineage and normal commands.
Minimal preprocessing fix retains reference columns as output-only lineage even when the model profile excludes them. Ten focused tests and the full 86-test suite passed; temporary real-data preprocessing preserved 1,000 rows, reference values/hash and identical B holdout IDs (808/192). No final-path export, model training, notebook execution or DE performed. Next: researcher-run comparison; no improved scores, masking mechanism or replacement optimization surrogate is claimed. Native missing handling, group validation, model settings and unresolved optimization coefficients are unchanged.


## Experiment-D implemented and verified (2026-11-17, latest)
Researcher approved implementation of `docs/EXPERIMENT_D_PLAN.md` after discussion. Experiment C has now been run by the researcher: Optuna only, 30 trials x seeds [11,29,47] x two models x 5 folds; its frozen candidate is XGBoost/search seed 11 from `exp_C_20261117T055252Z_b321014c`. Earlier statements below that C research runs remain pending are historical.
Experiment D reuses that fitted model and original 808/192 split without fitting, changing configs/cohort, or executing DE. Notebook and shared diagnostic helper are implemented. Complete outputs: `artifacts/experiment_D/exp_D_20261117T123759Z_33092529/`. Holdout diagnostic MAE 1042.957, RMSE 1466.330 kg/ha, R2 0.727121; this is previously inspected holdout, not a pristine test. Seven compatible positive-dose scenarios yield 21 nutrient-response curves; one sampled control scenario is skipped with a reason. Training contains only 33 distinct NPK combinations; repeated treatments make row-level distance quantiles zero. Empirical pairing remains insufficient evidence for controlled comparison because metadata/design are unconfirmed. Read FIND-016 and D output tables before claiming surrogate suitability.
Verification: 85-test suite passed, with 15 focused D tests also rerun after final helper updates; the notebook is executed with model/preprocessor fit disabled. Source hashes are checked before/after. Next: researcher review of response/domain evidence, treatment comparability, bounds/baseline policy and unapproved optimization coefficients. A 60-trial C run is still only a suggestion, not a prerequisite or approved configuration change. No agronomic recommendation or optimization result is claimed.


## Experiment-C implementation complete (2026-11-17, latest)
User authorized implementation of the tuning plan. Random/grid/Optuna finite-space backends, legacy config fallback, uniform exports, independent search seeds, isolated C configs/runner and guarded notebook are implemented. C keeps the researcher-selected current `minimalist_features` (11 inputs), the existing 1,000-row B CSV read-only, native missing handling and unchanged 808/192 group split. No cohort change or new profile selection occurred. All-missing filtering is fit-fold-only and does not delete rows.
Default C comparison: random/Optuna x search seeds [11,29,47] x XGBoost/CatBoost; 30 candidates per model/study. Model/split seeds remain 42; threads/trial/CV sequential for initial controlled studies. Holdout reports are disabled during comparisons and require an explicit frozen-candidate evaluation. Optuna 5.1.0 installed in project environment; **70 tests passed**, including synthetic repeated-model studies, no holdout calls during search, explicit later evaluation, reproducibility and failure cases. Readiness/guard checks used the real dataset without fitting a research model. Full C research studies have NOT run. No C metrics/winner claimed.
Use `python scripts/run_experiment_c.py` or explicitly enable the notebook training flag. Read `docs/EXPERIMENT_C_USAGE.md`; resume/continuous distributions remain unsupported and rejected, rather than silently ignored. Default/root `models.yaml` still selects random; C overrides are isolated. Earlier planning-only status below is historical.


## Experiment-C tuning design (2026-11-17, planning only)
Researcher requested a Markdown design and Optuna usage/repetition guidance. Created `docs/EXPERIMENT_C_TUNING_PLAN.md`: compatible random/grid/Optuna dispatcher, shared finite search space, separate search/model/split seeds, lazy dependency, uniform scoring/refit/artifacts, proposed three-seed fair-budget experiment and experiment-C analysis workflow. Clarifies one study is valid; repeat studies assess search variability, while resume adds budget and is not an independent repeat. No tuning implementation/config/dependency/notebook change or experiment execution occurred. Study count/budget and final feature profile remain proposed.


## Experiment-B Notebook Diagnostics (2026-11-17, Latest)
Researcher created/trained run `artifacts/experiment_B/ta_20261116T203731Z` and requested additions to `notebooks/2_model_experiment/experiment_B.ipynb`. Notebook now includes snapshot/hash/cohort audits, feature coverage, native-NaN/encoder mapping checks, original-feature built-in aggregation, permutation interpretation, train-validation gaps, training-mean baseline, group residuals, duplicate-input and spatial coverage checks. Added code was verified separately against saved bundles without refitting; original result outputs retained except corrected setup. Optional joint NPK/coordinate permutation is disabled by default and was not run.
Verified: 808/192 split; 39 all-missing inputs, 3 constant observed; 7 overlapping articles (no group overlap); holdout numeric NaNs preserved (6,452); all 192 reported-coordinate pairs seen in training. Mean-training baseline holdout RMSE 2812.128; XGBoost/CatBoost holdout RMSE 1440.205/1440.017, CV mean R2 0.626900/0.609425. Holdout has 67 unique input combinations, 55/67 unique predictions. No feature selection, row deletion or DE decision was made. Notebook contains interpretation and limits, not causal claims. Earlier statements of unexecuted user training below are historical.


## Active Configuration Ready for User Run (2026-11-17, Latest)
- User requested complete experiment-B configurations. Active profile: `feature_importance_candidates` (97 inputs), XGBoost/CatBoost only, no imputation, same grouped validation/seed/selection settings. Minimum CV R2 remains 0.60; no threshold was relaxed.
- Raw source: `data/input/data.csv`; read-only cohort reference: `data/processed/model_dataset_balanced_1000_minimalist_504622.csv`; new output: `data/processed/experiment_B_candidates.csv`; artifacts: `artifacts/experiment_B/<run_id>/`.
- `data.yaml` configures exact 1,000-ID reference mode, source/reference hashes, all 44 numeric source/target fields and coordinate bounds. Preprocessing supports cohort alignment, numeric-profile parsing and preservation checks; it fails rather than silently changing rows/labels or overwriting the reference.
- Verification: **47 tests passed**. Temporary real-data preprocessing produced 1,000 rows / 106 columns / 97 model inputs / 196 encoded columns, preserved all baseline values and 32,901 numeric missing entries, and reproduced the same 808/192 holdout IDs. Twenty precipitation strings coerce to missing already present in the reference; no new missingness relative to baseline was introduced there.
- Final-path preprocessing and research-model training were not executed. Run `python scripts/preprocess.py` then `python scripts/train_models.py`, or `python scripts/run_full_experiment.py`, from the repository root in the project environment. The enriched final CSV is created by those preprocessing commands.
- Unit/timing interpretation and importance caveats remain as documented. Per-fold/coherent-family importance and experiment-B analysis notebook are still deferred. No new scientific scores, selected surrogate or DE results exist.


**Updated:** 2026-11-17, after direct repository/artifact audit and explicit researcher request for documentation updates.

## Latest Task: Feature-Importance Preparation (2026-11-17, After No-Imputation Fix)
- Researcher requested a broad metadata-based profile and importance audit/plan; experiment B is explicitly deferred until after researcher-run modeling.
- Added optional `feature_importance_candidates`: 50 numeric + 47 categorical = 97 unique inputs, with all seven NPK-derived features. Missingness did not determine exclusion; final pH/EC, outcomes/IDs and unreviewed mixed-unit layer-1 P/K are excluded for validity. Generic soil measurement timing still needs review.
- Only XGBoost/CatBoost are enabled. Active `minimalist_features`, no-imputation settings and dataset path were not changed by this task; existing selection settings were also left intact.
- `docs/EXPERIMENT_B_PLAN.md` records source mappings, exclusions, readiness gates, access paths and importance limitations. Current minimalist data lacks 86 candidate inputs; same-ID enrichment, numeric parsing and unit/timing checks are pending, not executed. Do not train the broad profile using the current 20-column CSV directly.
- Audit: built-in importance is on transformed columns, permutation importance on raw inputs through the pipeline. Silent length truncation, missing-score fallback, NPK/derived-feature consistency and holdout-based feature selection require attention before experiment B. Per-fold/family-level importance and original-feature aggregation are proposals, not implemented features.
- Validation for this task: schema/name/count/duplicate/derived-feature checks, static inspection and installed-library/primary-documentation verification. No training, preprocessing dataset export, importance calculation, new scores or experiment-B notebook execution.

## Latest Implementation State (After Audit, 2026-11-17)
The researcher authorized points 1-5: centralized optional imputation and active no-imputation, with advanced MICE retained as an example. Implemented and checked:
- `state_imputation.mode: none` in `configs/decisions.yaml`; `default_method: none`, empty column overrides and no active MICE block. Same minimalist profile, dataset, target, grouped split and optimization settings.
- Resolver, preprocessor and trainer honor the switch; no-imputation applies to numeric and categorical imputer branches. Numerical NaNs are preserved; categoricals still use one-hot encoding. Mode/settings are exported in future bundles/metrics.
- Active models: XGBoost/CatBoost. Other definitions are retained disabled.
- Optional advanced MICE example: `configs/examples/decision.example.yaml`. Versioned examples present in the workspace are retained. Broader profiles require matching dataset columns; the example is not automatically active.
- Experiment A now calls the shared trainer without model/imputation overrides and creates a fresh run. Changed training outputs are cleared; historical analysis remains explicitly separate.
- **Verification: 39 tests passed.** Synthetic checks cover native fit/predict/reload and trainer mode export; diagnostic preprocessing preserved all 1,192 numeric NaNs in the active dataset. No research-data predictive model or feature-importance experiment was trained/run, and no old bundles/metrics were overwritten.

The source-level native/selection wiring and stale-test blockers from the audit below are resolved. Historical metrics and bundle findings remain valid for their old runs; no corrected scientific metrics or final surrogate selection exist yet. Next: a separately requested controlled baseline training; broad-feature importance remains deferred. DE coefficients/baseline justification remain open.

## Historical Audit Snapshot (Before Implementation)
The remaining audit snapshot is retained for provenance. Its statements about code/config/test blockers describe pre-fix state; use the latest implementation state above for current behavior.

## Current Phase
Transition from predictive modeling to Differential Evolution optimization. Reported supervisor direction remains: proceed toward optimization without further row removal. The audit found implementation/selection problems that must be separated from the intended native-missing experiment.

## Verified Configured Baseline
- Profile: `minimalist_features` (earlier conversation used singular `minimalist_feature`).
- Dataset: `data/processed/model_dataset_balanced_1000_minimalist_504622.csv`.
- SHA-256: `94ab8b338849d0c46cabbff786e232d233eceaf9f69c74b47297f50ca56e27b6`.
- Shape: 1,000 rows, 20 columns; 16 articles, 266 article-experiment groups; target and N/P/K have no missing values.
- Actual 11 inputs: direct N/P/K; `initial.ph.soil`, `avprec.mm`, `soc.percentage`; `total_npk`, `ratio_n_p`, `ratio_n_k`, `ratio_p_k`; `fertilizer_type`.
- Missingness: pH 46.60%, SOC 50.60%, precipitation 22.00%, fertilizer type 0%.
- Reference run: `artifacts/exp_A_comp_miss_val_mech_minimalist/`; split 808 training / 192 holdout, matching record IDs and zero article-experiment group overlap.
- Validation: GroupShuffleSplit holdout fraction 0.20, seed 42; five-fold GroupKFold inside training. Group fraction need not equal row fraction. This does not establish article-disjoint evaluation.

This records the configured cohort, not a newly approved final surrogate or a completed immutable experiment manifest.

## Notebook and Run Separation
`notebooks/2_model_experiment/experiment_A.ipynb` contains mixed execution counts and multiple runs. Its training section names `exp_A_comp_miss_val_mech_minimalist_383327`, whose artifacts use 800 rows (666 training / 134 holdout). Later analysis references the 1,000-row minimalist run and imputation run `ta_20261111T010140Z`. Do not treat all outputs as one experiment.

## Verified Stored Metrics
Source: each run's `metrics/model_comparison.csv`; no rerun. MAE/RMSE are kg/ha. Names preserve artifact labels; **native-labeled bundles use MICE** (FIND-007).

| Rows | Model label | CV R2 mean | CV RMSE mean | Holdout R2 | Holdout MAE | Holdout RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | xgboost_native | 0.647536 | 1806.583 | 0.724958 | 1064.934 | 1472.128 |
| 1,000 | catboost_native | 0.632722 | 1844.487 | 0.726492 | 1060.367 | 1468.018 |
| 800 | xgboost_native | 0.622853 | 1858.499 | 0.796846 | 885.416 | 1316.777 |
| 800 | catboost_native | 0.612961 | 1903.386 | 0.810142 | 844.064 | 1272.958 |

Earlier reported R2 around 0.72 matches the 1,000-row holdout, not CV mean. Earlier approximate MAE/RMSE of 1,000-1,200 does not match its RMSE. Cohort/holdout composition differs between 800 and 1,000 rows; comparisons do not isolate native-missing benefits.

## Verified Blockers and Risks
1. **Native-label mismatch:** notebook calculates but does not pass `imputation_mode`; resolver puts method `none` into the MICE list. Inspected XGBoost/CatBoost native-labeled bundles for both cohorts use MICE and categorical mode imputation. The 1,000-row XGBoost holdout has 267 missing input entries, zero after transformation.
2. **Selection configuration lost:** notebook overrides `configs["models"]`, dropping selection settings; all experiment-A selection scores are zero. Saved selection names `xgboost_native`; weighted-metric superiority is not established.
3. **MILP provenance/coverage:** implementation in `notebooks/exploration_notebooks/02_balancing_data.ipynb` stores an 800-row run. Exact 1,000-row generation settings remain incompletely traced. Compared with current pre-balancing data (1,682 rows), articles fall from 28 to 16, groups from 350 to 266, mean yield from 5136.99 to 4420.09 kg/ha.
4. **Repeated inputs:** 192 holdout rows have 46 unique raw/transformed feature combinations and 42 unique XGBoost predictions. This supports feature duplication as an explanation for repeated predictions; the full visual-band mechanism is unresolved.
5. **DE objective incomplete:** implementation exists but enabled normalization/penalties require `yield_scale`, `lambda_total`, `lambda_composition`, which are null. Bounds and group-mean baseline remain provisional. Perturbations and several validation thresholds are unset.
6. **Optimizer semantics:** derived NPK reconstruction exists; runtime dose-response/support checks and fertilizer-type consistency when doses change remain pending.
7. **Tests:** some tests use old interfaces/assumptions. Historical test-success text is not current verification. Pytest was not run.

## Audit Work and Limits (Before Implementation)
- Read AGENTS.md and all seven agent documents; inspected relevant code, configs, notebook sources/stored outputs, metrics, splits and fitted bundles.
- Computed dataset hashes/statistics, verified split/target alignment, and inspected fitted transformers/transformed holdout input without fitting models.
- Updated agent documents 02-07 under the explicit memory-update protocol. No research code, config, dataset or experiment artifact was changed; no training, notebook rerun, DE run or pytest execution occurred.
- Existing `.gitignore` modification predates this work and was left untouched. `git check-ignore` confirms the updated agent documents are ignored by repository rules; ordinary `git diff` does not display these edits. Their contents were checked directly.

## Historical Recommended Next Actions (At Audit Time)
1. Finish a reproducible baseline manifest and isolate run references without changing the cohort.
2. Correct/test native-missing and selection-config wiring; retain cohort/group design for necessary comparisons and preserve historical mislabeled evidence.
3. Explicitly select/freeze a surrogate using corrected evidence; check schema, fixed-state dose response, category semantics and domain support.
4. Resolve/justify bounds, baseline and objective coefficients with researcher authority, then run initial DE on limited scenarios/seeds.
5. Continue feasibility/boundary diagnostics, sensitivity, stability and agronomic review. Keep Optuna and broad feature search in backlog unless a demonstrated blocker requires them.

## Restrictions
- No further row removal without explicit approval.
- No silent split, target, coefficient or dataset changes.
- Native handling remains experimental intent, not established by current labels/scores.
- A selected-model file does not establish scientifically justified model selection.
- Model predictions/improvements are not causal or field-validated effects.

## Pending Researcher Verification
Final surrogate choice; exact 1,000-row MILP generation settings; agronomic provenance/approval of bounds; baseline policy for unseen groups; penalty weights/yield scale; perturbations and validation thresholds. No new scientific decision was finalized by this update.
