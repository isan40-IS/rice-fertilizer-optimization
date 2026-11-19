# 06 — Research Findings and Diagnostic Evidence

## FIND-018 - Selected legacy XGBoost and input recovery preflight
**Date/status:** 2026-11-19; verified saved bundle/CSV/JSON/configs and temporary preprocessing/inference checks; no fitting.
**Source:** artifacts/ta_20261027T232115Z; docs/SURROGATE_SELECTION_ta_20261027T232115Z.json.
**Observations:** Selected XGBoost has 16 inputs, random_kfold and MICE/median/categorical preprocessing. CV RMSE 1591.015224 (SD 133.776503), R2 0.680139 (SD 0.050166). Holdout RMSE 1649.202029, R2 0.659784, n=337. Saved leakage checks show zero record overlap and 174 shared article-experiment groups. Current configured 1,682-row CSV lacks five required features.
**Recovery check:** Restored preprocessing in a temporary directory yields 1,682 rows/25 columns. All saved split record IDs/order and holdout targets match. All eight training-numeric sorted observed-value distributions match saved training references. All 337 saved XGBoost holdout predictions reproduce (max absolute error 2.274e-13). Temporary files removed; dataset not exported or overwritten.
**Interpretation:** The saved inference path and a recoverable compatible input dataset are supported. Original CSV byte identity is unknown because the legacy bundle lacks dataset hashes. Small random-CV variation does not establish cross-experiment generalization or optimization stability. Researcher's stability rationale is distinct from these observations. No NPK response, joint-domain suitability or agronomic validation is established by this preflight.

## Evidence Standard
- **Verified:** directly inspected in current data/code/output.
- **Reported:** researcher described the observation, original output not yet inspected here.
- **Hypothesis:** plausible explanation requiring tests.
- **Interpretation:** reasoned implication, not an observed causal mechanism.

Do not upgrade a finding's evidence status without checking the cited artifact.

## FIND-001 — Extensive Missingness in Soil/Environment Features
**Audit addition (verified, 2026-11-17):** Current `data/processed/model_dataset.csv` has 1,682 rows: missing pH 68.25%, SOC 70.63%, precipitation 46.08%. Configured 1,000-row cohort: 46.60%, 50.60%, 22.00%. Original percentages below remain historical, reported values.

**Historical report status (audit additions above take precedence):** Reported, supported by earlier research exploration.
**Observation:** Several soil/environment variables have high missingness. Earlier analyses cited roughly 63.7% for `initial.ph.soil`, 67.3% for `soc.percentage`, and 79.7% for `clay.perc1` in an earlier data state. These percentages must not be assumed to describe the new selected dataset.
**Implication:** Imputation, native missing handling, feature inclusion, and row selection can materially change the training sample and inferred patterns.
**Caution:** Missingness mechanism (MCAR/MAR/MNAR) is not established simply by these rates.
**Next evidence:** Current missingness tables, per-study breakdown, and post-selection comparison.

## FIND-002 — Overlap Structure Changes Feature-Availability Trade-offs
**Status:** Reported.
**Observation:** A missingness matrix showed that some `..._Source` attributes contain many observed values, while records missing such source values can contain useful observations in `avprec.mm`, `soc.percentage`, or `initial.ph.soil`.
**Researcher interpretation:** Prefer keeping scientifically useful state-variable measurements rather than maximizing marginal completeness of source-related attributes.
**Caution:** Precise list of source fields, joint intersection counts, and effect on sample retention are not yet independently verified.
**Next evidence:** Pairwise/joint completeness counts and record-ID coverage by experimental group.

## FIND-003 — Native Missing and Minimalist Dataset Associated With Better Scores
**Audit correction (2026-11-17):** Original report below is retained as history, not a verified native-handling result. In `artifacts/exp_A_comp_miss_val_mech_minimalist/metrics/model_comparison.csv`, XGBoost has CV R2 0.647536 / holdout R2 0.724958, MAE 1064.934 and RMSE 1472.128 kg/ha; CatBoost has CV R2 0.632722 / holdout R2 0.726492, MAE 1060.367 and RMSE 1468.018 kg/ha. FIND-007 confirms MICE in both native-labeled bundles. Native-handling benefits are not established. Separate 800-row results appear in `05_current_state.md`; comparisons across cohorts do not isolate an intervention.

**Historical report status (audit additions above take precedence):** Reported.
**Observation:** The experiment using the new minimalist dataset and native XGBoost/CatBoost missing handling reportedly achieved R² around 0.72 and MAE/RMSE around 1,000–1,200 kg/ha.
**Source:** `notebooks/2_model_experiment/experiment_A.ipynb`, according to researcher.
**Interpretation:** The new *configuration as a whole* appears more promising than the earlier reported high-error pipeline.
**Not established:** Which individual intervention caused the improvement. Model, features, missing handling, and row selection changed together.
**Next evidence:** Actual notebook output, model-specific folds/holdout metrics, and comparable old experimental results.

## FIND-004 — Horizontal Bands in Predicted Yield
**Audit addition (verified numerical diagnostic, 2026-11-17):** The 1,000-row run has 192 holdout rows, 46 unique raw and transformed feature combinations, and 42 unique XGBoost predictions (CatBoost: 46). Source: configured dataset joined by `row_index` to the run's `metrics/*_holdout_predictions.csv`, plus fitted XGBoost preprocessing. Saved targets match dataset targets. This supports repeated inputs as one explanation; it does not explain every visual band. No new plot inspection/response sweep or row deletion occurred.

**Historical report status (audit additions above take precedence):** Reported; root cause unknown.
**Observation:** Predicted values cluster into horizontal stripes near several yield levels in an actual-versus-predicted scatter plot (researcher mentions around 5,000, 7,000, 8,000 kg/ha).
**Plausible hypotheses (not findings):**
1. Tree-ensemble piecewise-constant predictions and repeated leaf/ensemble outputs.
2. Multiple rows sharing identical or near-identical effective feature values.
3. Missingness patterns or feature compression limiting model discrimination.
4. Study/experiment structure or target clusters.
5. Plotting/rounding/aggregation artifact.

**Discriminating diagnostics:**
- Inspect distinct predicted values, frequency table, and numeric precision; check if apparent bands are exact repeats or visual clustering.
- Plot residuals and predicted-value histogram by model and by experimental group.
- Compare train, CV/holdout, and subgroups; inspect effective features for repeatedly predicted samples.
- Check whether groups or missingness indicators explain clustering.
- Perturb N/P/K at fixed state to inspect local surrogate response across dose ranges.

**Do not conclude** that the model is incorrect solely because of banding; assess predictive utility and response behavior.

## FIND-005 — Row Selection Can Bias the Analyzed Population
**Audit addition (verified cohort shift, 2026-11-17):** Current pre-balancing dataset has 1,682 rows / 28 articles / 350 groups / mean yield 5136.99 kg/ha. Configured 1,000-row cohort has 16 articles / 266 groups / mean yield 4420.09 kg/ha. Composition shift is observed; its bias magnitude and causal contribution to scores are not established.
**Implementation evidence:** `notebooks/exploration_notebooks/02_balancing_data.ipynb` uses MILP with exact row count, complete target/NPK eligibility, soft missing limits and an excess-missingness objective. It has no article/group/dose/yield balancing objective. Stored 800-row output reports optimality but exceeds all three limits: SOC 38.25% vs 20%, pH 33.25% vs 20%, precipitation 27.50% vs 10%. Exact 1,000-row generation parameters remain open. MILM wording below is historical.

**Historical report status (audit additions above take precedence):** Methodological risk, not demonstrated failure.
**Basis:** The researcher used MILM to retain a targeted subset with reduced missingness.
**Risk:** Differences in location, year, experiment, target distribution, or fertilizer treatment coverage may limit representativeness.
**Suggested audit without deleting more rows:** Compare retained vs excluded rows, target distribution, dose distributions, year/group coverage, missingness, and covariate support.

## FIND-006 — Yield Prediction Accuracy Is Not Equivalent to Optimization Validity
**Status:** Scientific principle from the thesis design and surrogate-model limitations.
**Reasoning:** Predicting yield across heterogeneous land conditions differs from estimating how predicted yield changes when only fertilizer dose changes. An optimizer may exploit model artifacts in sparsely observed regions.
**Diagnostics:** Fixed-state response curves, constraint checks, nearest-training-support or density checks, consistency across seeds/models/scenarios, and expert review.

## FIND-007 — Native Missing Handling Must Be Verified in the Pipeline
**Status:** Verified implementation/label mismatch, 2026-11-17; supersedes earlier unknown status.
**Code evidence:** `modeling/imputation.py` accepts `none` but appends it to `mice_columns`. Experiment A computes `imputation_mode` but omits it from `build_model_pipeline()`, whose default is `plan`; main `trainer.py` also uses that default.
**Artifact evidence:** Inspected native-labeled XGBoost/CatBoost bundles under both minimalist runs contain BayesianRidge IterativeImputer on all seven profile numeric features, plus categorical most-frequent imputation and one-hot encoding. On the 1,000-row XGBoost holdout, 267 missing input entries become zero after transformation.
**Interpretation:** Existing labels/metrics do not demonstrate native missing handling. Nominal native/imputation branches also provide different numeric predictor sets to MICE; this is not the intended ablation. One-hot categoricals are not native CatBoost categorical handling.
**Limits:** Bundles were loaded/preprocessing transformed, with no refitting or corrected experiment.
**Next step:** Focused missing-preservation and fit/predict checks after separately authorized implementation correction; preserve cohort/group design and historical artifacts.

## FIND-008 - Experiment-A Selection Settings Lost
**Status:** Verified, 2026-11-17.
**Evidence:** Notebook replaces `configs["models"]` with specifications and calls `rank_models(..., model_cfg.get("selection", {}))`. Weights are absent; `model_selector.py` adds no weighted terms to initial zero scores. Both minimalist metric CSVs show all-zero scores and select `xgboost_native`.
**Interpretation:** Stored selection does not demonstrate superiority under intended ranking; this does not automatically invalidate every prediction/metric.
**Next step:** Preserve explicit selection configuration and review selection against comparable, corrected evidence. No replacement winner was approved.

## FIND-009 - Notebook Provenance and Optimization Readiness
**Status:** Verified code/output observations, 2026-11-17.
**Evidence:** Experiment-A training names the 800-row `_383327` run; later comparisons reference the 1,000-row run and `ta_20261111T010140Z`. Snapshots are created before notebook model overrides. Later group analysis searches `configs/`, whereas normal snapshots use `config_snapshot/`, and can fall back to project settings. Snapshots alone do not capture all effective notebook settings.
**Optimization evidence:** DE, candidate NPK rebuilding and evaluation exist. `yield_scale`, `lambda_total`, `lambda_composition` remain null while corresponding options are enabled; objective construction requires these values. Perturbations/validation thresholds remain unset. Group-mean baseline uses context doses, not yield values; unseen-group policy remains open.
**Limit:** Implementation existence does not prove completed DE, agronomic validation or runtime response correctness. No DE was run during the audit.

## FIND-010 - Historical Test Claims Need Revalidation
**Status:** Verified static interface/assumption mismatch, 2026-11-17.
**Evidence:** `tests/integration/test_model_pipeline.py` calls an older pipeline interface; `tests/e2e/test_smoke_configuration.py` expects null profile/bounds despite current settings. `docs/VALIDATION_STATUS.md` records historical checks.
**Limit:** Pytest was not executed; no current pass/fail count is asserted. Repair/revalidation is proposed work.


## FIND-011 - Corrected Config-Driven No-Imputation Path
**Date/status:** 2026-11-17, verified implementation and regression checks after explicit researcher authorization.
**Evidence:** Resolver separates `none` from MICE; global mode is forwarded by trainer and inferred from plans when needed. Notebook training uses the shared trainer, retaining selection settings. 39 tests passed, including synthetic XGBoost/CatBoost fit/predict/reload, mode-driven trainer exports, mixed optional imputation and target/dose protection.
**Data diagnostic:** Fitting only the new preprocessor for inspection on the active 1,000-row dataset preserved all 1,192 numeric NaNs. No predictive model was fitted on research data; the diagnostic preprocessor was not saved or used for score evaluation.
**Interpretation:** No-imputation is now implemented/verified. This is not evidence of improved yield prediction or a newly selected scientific surrogate. FIND-007/008 remain valid descriptions of historical code/artifacts; their source-level causes have been corrected.
**Limits/next step:** Obtain a new controlled training run when requested. Broader feature importance, final surrogate selection and DE remain pending.

## FIND-012 - Metadata and Feature-Importance Audit for Experiment B
**Date/status:** 2026-11-17, verified metadata/schema/code observations; no importance results.
**Evidence:** Read `data/metadata/metadata.xlsx` sheet Metadata and actual 125-column raw CSV. Metadata explicitly identifies final pH/EC as after treatment. Actual density headers are `bd.1/.2/.3`; layer-1 soil P/K have multiple reported units. New optional profile contains 50 numeric/47 categorical inputs, including seven engineered features. Existing 20-column minimalist CSV lacks 86 of these inputs. Many broader numeric source values use decimal commas, while preprocessing currently coerces only the 11 configured numeric fields.
**Importance evidence:** `interpretation.py` correctly uses fitted transformed feature names for built-in scores and scores the fitted pipeline for raw-column permutations. XGBoost default tree importance is normalized gain; CatBoost non-ranking regression uses PredictionValuesChange. Built-in values are not directly comparable across models. The helper silently truncates name/value mismatches with `min`; trainer forwards `None` if scoring is absent, overriding the helper default. Active scoring is explicit negative RMSE.
**Interpretation risks:** Independent dose/derived permutations create inconsistent NPK representations; correlations can mask individual reliance. Current permutations use holdout, not fold validation. Selecting features with them makes the same holdout adaptive evaluation. Missingness, method/depth and spatial context may act as proxies. Proposed CV/family-level methods are not yet implemented or approved as a changed evaluation design.
**Next:** Follow `docs/EXPERIMENT_B_PLAN.md` after researcher-run modeling; establish same-cohort enrichment, numeric parsing, unit/timing checks, then implement agreed importance diagnostics. Importance is not causal agronomic evidence.

## FIND-013 - Reference-Cohort Preprocessing Verified for Broad Inputs
**Date/status:** 2026-11-17, verified implementation and temporary real-data preflight.
**Evidence:** Configured preprocessing yields 1,000 rows, 106 dataset columns, 97 input columns and 196 encoded columns. It reproduces all reference values/IDs/order and the original 808/192 holdout IDs; the baseline file hash is unchanged. No-imputation preprocessing preserves 32,901 numeric missing entries. Parsing records 20 nonnumeric precipitation strings becoming missing, matching existing reference missingness. All 47 tests passed, including altered-label, missing-ID, duplicate-ID, changed-hash, row-loss and overwrite failure checks.
**Interpretation:** Configuration/data readiness is demonstrated for the preserved cohort, not improved predictive performance, causal validity, final agronomic comparability or a newly trained surrogate. Generic soil measurement timing and previously documented importance limitations still apply.
**Limit:** No final dataset was exported; preflight used a temporary directory. No research-model training, importance experiment or DE run occurred.

## FIND-014 - Trained Experiment-B Run and Diagnostic Coverage
**Date/status:** 2026-11-17; verified from user-created notebook, saved CSVs/bundles and separately executed diagnostic code. No model refit by the agent.
**Source:** `artifacts/experiment_B/ta_20261116T203731Z`; diagnostic details/interpretation in `notebooks/2_model_experiment/experiment_B.ipynb`.
**Results:** XGBoost CV mean R2 0.626900 and RMSE 1853.726; CatBoost 0.609425 and 1899.908. Holdout RMSE 1440.205/1440.017 (R2 0.736758/0.736826); training-mean baseline RMSE 2812.128. Seven articles overlap across partitions despite zero article-experiment group overlap. 39/97 inputs are entirely missing; 3 have a single observed value. All 192 complete reported-coordinate holdout pairs also occur in training. Both pipelines preserve all 6,452 numeric NaNs on holdout without an imputer.
**Interpretation:** Predictive utility on this cohort/holdout is supported; causal dose response and new-location generalization are not established. Holdout RMSE difference (~0.188 kg/ha) is descriptive, not demonstrated superiority. Repeated raw inputs (67 distinct among 192) partly explain repeated predictions. No features or rows were discarded.
**Pending:** Optional joint-family permutation, fold-based discovery, spatial ablation and surrogate response diagnostics; no such results are claimed.

## FIND-015 - Tuning Backend and Experiment-C Software Verification
**Date/status:** 2026-11-17, verified implementation/testing, not scientific model results.
**Evidence:** 70 tests pass with Optuna 5.1.0, including random/grid/Optuna refit/contracts, negative-score sign, same-seed reproducibility on deterministic synthetic data, invalid spaces/grid caps, group protection, missing optional dependency, failure exports, all-missing train-only filtering, a multi-model/multi-seed synthetic experiment and frozen holdout evaluation. Real-data readiness confirms 1,000 rows, 11 minimalist inputs and the same 808/192 split without research fitting.
**Implementation finding:** CatBoost get_params omits valid defaults such as depth/iterations; validation now also uses its public constructor signature. This was found by a failing synthetic integration test and corrected; no field observations were altered.
**Limits:** No C research study or improved predictive metric is claimed. Exact reproducibility across hardware/versions is not established. Whole-cohort/native-model suitability and DE decisions remain separate research questions.

## Finding Template
`FIND-XXX | Date | Evidence status | Observation | Source/artifact | Interpretation | Alternative explanations | Limitations | Next verification`.

## FIND-016 - Frozen Experiment-D Diagnostic Evidence
**Date/status:** Verified, 2026-11-17; model inference only, no training/DE.
**Source:** `artifacts/experiment_D/exp_D_20261117T123759Z_33092529/` and frozen C candidate `exp_C_20261117T055252Z_b321014c/optuna_seed_11/models/xgboost.joblib`.
**Predictive observations:** Original 192-row holdout: MAE 1042.95694, RMSE 1466.33000 kg/ha, R2 0.72712062. It is previously inspected holdout, not independent confirmatory evidence. No comparison against old experiments is treated as controlled model/feature ablation.
**Responsiveness:** Eight deterministically selected training scenarios; one `No fertilizer` control is skipped for positive-dose sweeps to preserve treatment semantics. The remaining seven yield 21 response curves. Median amplitudes across these scenarios: N 4042.80, P 3041.14, K 998.84 kg/ha. These are model-function ranges on exploratory grids, include potentially weak/out-of-range combinations, and are not causal fertilizer effects. Detailed no-verified-violation summaries are exported separately; absence of a range violation does not prove support. State/missingness changes also alter predictions; masking effects are not agronomic effects.
**Coverage:** Training 808 rows contains only 33 distinct NPK combinations. Leave-one-out nearest distance is zero for 800 rows, so Q50/Q90/Q95 of row-level reference are zero; distinct-combination distance is reported separately. Do not automatically turn those quantiles into a reliability cutoff.
**Pairing:** 63 one-nutrient and 143 multi-nutrient candidate pairs within holdout groups are descriptive. Zero controlled pairs are approved under default unconfirmed metadata/design policy; repeated doses and incomplete context are not silently aggregated or equated. Thus controlled response ranking remains insufficient evidence, not model failure or a zero performance score.
**Limits/next:** Current C CSV has 20 columns, limiting management/replicate matching; full metadata enrichment would require a separately reviewed read-only lineage. Review curves/domain evidence and literature/worksheet decisions before optimization. A good aggregate prediction score is not agronomic validation.


## FIND-017 - no_engineered run: prediction, missingness and optimization risks
**Date/status:** 2026-11-19, verified from current experiment_B code/output and researcher-created artifacts; agent performed read-only integrity and saved-model inference checks, no refit.
**Source:** `artifacts/no_engineered/ta_20261119T040155Z`; full interpretation/evidence links in `docs/NO_ENGINEERED.md`. Notebook legacy Markdown is excluded as a source of current results. Dataset SHA-256 `2162ca28d54b8620397b300d44dc47b3fb6bbb83821c5da0cfd60d407892b743`; reference values/hash and B split IDs match. 1,000 rows / 44 CSV columns / 31 model inputs / 88 transformed inputs; 808/192 rows, 212/54 groups, no record/group overlap but 7 overlapping articles.
**Results:** Random search 30 unique completed candidates/model, search seed 11, estimator/split seeds 42. CatBoost selected by configured CV-weighted ranking: CV RMSE 1898.156631 (SD 523.927085), R2 0.612106; holdout MAE/RMSE/R2 1059.520391/1432.413045/0.739598. XGBoost CV RMSE 1902.102799 (SD 652.453904), R2 0.603020; holdout 1099.414776/1520.929172/0.706421. Baseline training-mean holdout RMSE 2812.128220. CV difference 3.946 kg/ha is small relative to fold variation; no significance or nested validation claimed.
**Reliance:** Permutation N/P/K = 596.853/439.309/-6.431 kg/ha for XGBoost; 437.247/429.060/48.662 for CatBoost. Rank correlation 0.531436. Built-in XGBoost soil_type = 39.429%, including soil_type_nan alone 34.376%; CatBoost N/P/rainfall = 27.772/13.098/11.779%. Missingness/source reliance remains; importance is not causal response, and negative K permutation is not a negative agronomic effect.
**Coverage:** No all-missing inputs; ph.method has only one observed value (pH in water) and 65.1% missing. bd.2/bd.3 have zero measured holdout values; their zero permutation does not validate observed-value response. Both bundles preserve all 1,954 numeric holdout NaNs without imputation. 67 unique holdout inputs; 55/67 unique predictions XGBoost/CatBoost. Training contains 33 unique NPK combinations; training Spearman P-K 0.797172. Domain/response evidence from the older C/D model cannot be transferred to this candidate.
**Reproducibility issue:** Current code cell 12 still asserts 97 unique inputs and stores AssertionError. Downstream outputs match 31-input artifacts, verified independently; a clean Run All remains blocked until assertion context is corrected. No notebook change or execution performed here.
**Interpretation/next:** CatBoost is a candidate for response/domain diagnostics, not an approved final replacement. Product rate/concentration/source inputs are fixed by generic optimization while doses change, so joint treatment semantics need review. Derivative-removal masking remains a hypothesis: features, coordinates and search seed also changed versus B. No new row deletion, feature removal, bounds/weights, model replacement or DE approval follows from these observations. See backlog for proposed next checks.
