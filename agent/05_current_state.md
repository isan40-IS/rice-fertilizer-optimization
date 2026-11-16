# 05 - Current Working State

**Updated:** 2026-11-17, after direct repository/artifact audit and explicit researcher request for documentation updates.

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

## Completed Work and Limits
- Read AGENTS.md and all seven agent documents; inspected relevant code, configs, notebook sources/stored outputs, metrics, splits and fitted bundles.
- Computed dataset hashes/statistics, verified split/target alignment, and inspected fitted transformers/transformed holdout input without fitting models.
- Updated agent documents 02-07 under the explicit memory-update protocol. No research code, config, dataset or experiment artifact was changed; no training, notebook rerun, DE run or pytest execution occurred.
- Existing `.gitignore` modification predates this work and was left untouched. `git check-ignore` confirms the updated agent documents are ignored by repository rules; ordinary `git diff` does not display these edits. Their contents were checked directly.

## Recommended Next Actions (Not Executed)
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
