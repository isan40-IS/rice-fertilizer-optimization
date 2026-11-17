# 04 — Scientific Decision Log

## Status Vocabulary
- **Confirmed:** explicitly adopted research decision, subject to future documented revision.
- **Active experimental:** currently used in the reported experiment, not necessarily final for thesis.
- **Superseded:** formerly used; replaced in the latest reported workflow, retained for traceability.
- **Provisional:** chosen/configured but justification or approval remains incomplete.
- **Open:** not decided; must not be invented by the agent.
- **Suggested:** outside suggestion, not an approved implementation decision.

When a decision changes, preserve the prior entry and link the replacement. Do not rewrite historical evidence.

## DEC-001 — Research Focus and Architecture
**Status:** Confirmed.
**Decision:** Rice yield prediction with ML surrogate, followed by N/P/K optimization using Differential Evolution, then sensitivity, stability, efficiency, and agronomic interpretation.
**Rationale:** Research questions and proposal methodology.
**Consequence:** Avoid optimizing unrelated objectives or modifying land conditions as if they were decision variables.

## DEC-002 — Prediction Target
**Status:** Confirmed in uploaded training configuration.
**Decision:** `yield.wb.kg_ha`.
**Rationale:** Selected yield measure in previous research decisions.
**Audit 2026-11-17:** Config, notebook and bundles use this target key. Error units are reported as kg/ha; raw-data unit harmonization was not independently audited.

## DEC-003 — Group-Aware Validation
**Status:** Confirmed design; reverified in active config, notebook and saved splits on 2026-11-17.
**Decision:** `group_kfold` and `group_article_experiment` for grouped validation; earlier code uses group-aware holdout.
**Rationale:** Prevent closely related observations from the same experiment appearing across train/validation boundaries.
**Trade-off:** More realistic cross-experiment evaluation may yield worse metrics than random splits.
**Restriction:** Do not switch to random splitting solely to improve scores.
**Audit:** The 1,000-row run has 808 training / 192 holdout rows with matching record IDs and zero group overlap. Groups are article-experiment pairs; article-disjoint evaluation is not established.

## DEC-004 — Original Model Comparison
**Status:** Superseded as the immediate active modeling focus; still part of original proposal.
**Decision:** Compare Linear Regression, Random Forest, XGBoost, CatBoost.
**Rationale:** Linear baseline, bagging baseline, boosting candidates.
**History:** Older `models.yaml` enabled all four; later experiment focuses on native missing XGBoost/CatBoost.
**Note:** Do not erase prior comparisons from the research narrative if they were actually conducted.

## DEC-005 — Original MICE-Based State Imputation
**Status:** Superseded as the intended missingness approach by DEC-006; **still present in inspected minimalist bundles**. The earlier status described intent rather than verified execution.
**Earlier approach:** `IterativeImputer`/Bayesian Ridge for selected soil and environmental fields; categorical mode imputation and default median elsewhere, according to old config.
**Reason for reconsideration:** Extensive missingness and practical modeling difficulties reported by researcher.
**Risk:** MICE with very high missingness may rely heavily on model assumptions; native missing handling also has limitations.
**Replacement:** DEC-006.

## DEC-006 — Native Missing-Value Models
**Status:** Active experimental intention, implemented by DEC-015; historical bundles retain the audited MICE behavior.
**Decision:** Evaluate XGBoost and CatBoost with native missing-value handling instead of MICE for the latest experiment.
**Rationale:** Avoid forced imputation of highly incomplete state variables; reported performance improved in the new experimental configuration.
**Caution:** Several changes occurred simultaneously, so improvement cannot be attributed to native missing handling alone.
**Audit 2026-11-17:** Intent is unchanged, but FIND-007 verifies MICE in both inspected native-labeled XGBoost/CatBoost bundles for both minimalist cohorts. Existing metrics do not validate native handling or its superiority. Correcting/rerunning the experiment remains pending; no replacement model was approved.

## DEC-007 — Minimalist Feature Profile
**Status:** Active experimental.
**Decision:** Use the historically reported `minimalist_feature` (verified key: `minimalist_features`); remove `clay.perc1` because of very high missingness and deprioritize some source-related attributes to retain key state variables such as `avprec.mm`, `initial.ph.soil`, `soc.percentage`.
**Rationale:** Missingness-overlap analysis showed feature-availability trade-offs. The researcher prioritized the scientific need for soil/environment state variables.
**Risks:** Feature removal may reduce useful predictive signal; retention choices may alter sample coverage.
**Audit 2026-11-17:** Config/bundle input has 11 columns: direct N/P/K; pH, precipitation, SOC; total NPK and three ratios; fertilizer type. Clay, soil type and N/P/K source categories are excluded.

## DEC-008 — MILP-Based Row Selection (Historically Reported as MILM)
**Status:** Active experimental historical dataset-construction choice; further deletions restricted.
**Decision:** Use the reported MILM process to retain a designated number of observations while seeking lower missingness across selected columns, accepting that missingness targets may not all be met.
**Reported dataset identifier:** `model_dataset_balanced_1000_minimalist_504622`.
**Rationale:** Prioritize retained sample count while reducing missingness burden.
**Risks:** Possible sample-selection bias and changes in study/year/target distribution.
**Audit 2026-11-17:** MILP located in `notebooks/exploration_notebooks/02_balancing_data.ipynb`: exact retained count plus soft missingness limits. Stored run is 800 rows; configured active cohort remains 1,000 rows. Exact 1,000-row generation settings remain unresolved. DEC-009 still restricts further deletion.

## DEC-009 — Supervisor Direction After Latest Guidance
**Status:** Confirmed as reported by researcher.
**Decision:** Continue to optimization and **do not remove further data rows** from the active dataset at this stage.
**Implication:** Freeze the chosen experimental dataset for the immediate optimization baseline; address open model concerns through analysis or separate experiments, not silent replacement.

## DEC-010 — Hyperparameter Tuning
**Status:** Existing old method confirmed; replacement only suggested.
**Existing code:** `RandomizedSearchCV`, historically 30 iterations, scoring by negative RMSE.
**Suggestion:** Nesha proposed Optuna.
**Decision:** Do not migrate automatically. Consider after optimization milestone or when evidence shows tuning is the bottleneck. Compare using identical folds, budgets, and evaluation methodology.

## DEC-011 — Feature Importance Selection
**Status:** Suggested, not approved as sole selection rule.
**Suggestion:** Train one model with broader feature coverage, inspect feature importance, and select features based on findings.
**Caution:** Importance can be biased by missingness, correlations, proxy variables, and study effects; it is not proof of causal relevance or optimality for dose-response modeling.
**Restriction:** Do not discard state/decision features solely because one estimator ranks them low.

## DEC-012 — Optimization Constraints and Penalties
**Status:** Provisional/open in the old uploaded configuration.
**Old configured bounds:** N 60–180, P 6.5–17.5, K 25–83 kg/ha; total soft 200, total hard 300; composition bounds were specified.
**Unresolved old values:** `lambda_total`, `lambda_composition`, `yield_scale`; several sensitivity/validation thresholds remain null.
**Audit 2026-11-17:** Bounds and null coefficients remain in current config. Enabled normalization/penalties require the null values, so objective construction is not ready. No values were filled.
**Policy:** These numbers are historical configured values, **not confirmed agronomic recommendations**. Verify their provenance, feasibility, units, and approval before claiming scientific validity or using them in final experiments.

## DEC-013 — Baseline Yield Reference
**Status:** Provisional/older configured choice.
**Old setting:** `baseline_policy: group_mean`.
**Methodological intent:** Compare model-predicted baseline and optimized doses under the same fixed state using the same surrogate.
**Verify:** How group-mean doses are computed, whether group availability introduces leakage, and how an unseen scenario receives its baseline.

## DEC-014 — Immediate Work Prioritization
**Status:** Active planning priority.
**Decision:** Establish a traceable surrogate baseline and proceed to DE optimization. Keep additional feature-search and Optuna work in backlog unless it blocks validity or execution.
**Rationale:** Supervisor direction and risk of endless reworking of prediction experiments.

## Audit Record: 2026-11-17 (No New Scientific Decision)
The researcher authorized documentation updates following the read-only audit. No research scope, evaluation design, dataset, coefficients, selected model or methodology was changed. DEC-005/006 now distinguish the intended transition from actual MICE behavior. DEC-007/008 correct identifiers and implementation terminology, preserving historical reports. Existing selection artifacts name `xgboost_native`, but all-zero scores do not establish a weighted-metric winner (FIND-008).

Corrective work in `07_research_backlog.md` is proposed, not executed or automatically approved by this update. DEC-009 and the open decisions in DEC-012/013 remain in force.


## DEC-015 - Config-Controlled No-Imputation Baseline
**Date:** 2026-11-17, after the repository audit.
**Status:** Confirmed by explicit researcher instruction; implementation verified, scientific training results pending.
**Decision:** Centralize imputation selection in `configs/decisions.yaml`, activate `state_imputation.mode: none` with the existing minimalist profile/cohort, preserve optional MICE/advanced-feature settings in an example, and defer broad-feature importance experiments.
**Implementation:** Resolver/pipeline/trainer now honor no-imputation; notebook uses the shared trainer. Active models are XGBoost/CatBoost; other definitions are retained disabled. Modes/settings are recorded in new bundles/metric summaries. Example: `configs/examples/decision.example.yaml`.
**Relationship to history:** Implements DEC-006's intention and replaces the active MICE settings described in DEC-005. DEC-006 remains historical experimental intent, now implemented by DEC-015; old native-labeled artifacts still use MICE. No historical result is relabeled or overwritten.
**Verification:** 39 synthetic-data tests passed; diagnostic transform of the active dataset preserved all 1,192 numeric missing entries. No research-model training, score comparison, DE or broad-feature importance experiment was run.
**Risks/open items:** Native handling does not resolve selection/missingness bias. Final surrogate choice and optimization decisions in DEC-012/013 remain open. Broader feature examples require a dataset with the corresponding columns and a reviewed lineage.

## DEC-016 - Broad Candidate Profile for Feature-Importance Planning
**Date:** 2026-11-17, after DEC-015.
**Status:** Researcher-requested exploratory preparation completed; profile is not active and experiment is deferred.
**Request:** Use XGBoost/CatBoost only; derive a broad numeric/categorical profile from `data/metadata/metadata.xlsx`, including implemented engineered features, without missingness-based feature exclusions. Audit importance and plan experiment B after researcher-run modeling.
**Result:** Added `feature_importance_candidates` (50 numeric / 47 categorical, 97 unique inputs) to `configs/features.yaml`. Source and schema/exclusion rationale are documented in `docs/EXPERIMENT_B_PLAN.md`. Outcomes, explicit post-treatment pH/EC, identifiers and unreviewed mixed-unit layer-1 P/K measures are excluded for semantics/validity, not missingness. All-missing candidate columns are retained.
**Limits:** Active profile/dataset/model settings unchanged. No broader cohort, row deletion, unit conversion, new validation strategy or final feature selection is approved. Timing of generic soil measures and measurement comparability remain execution checks. Current CSV lacks 86 inputs; enrichment/parsing is pending.
**Next:** Review data readiness and experiment-B implementation only when requested; no model or importance experiment was executed in this task.

## DEC-017 - Runnable Experiment-B Configuration on the Existing Cohort
**Date/status:** 2026-11-17; explicitly authorized by the researcher asking to complete all experiment configs.
**Decision/implementation:** Activate the broad profile with native numerical missing handling and XGBoost/CatBoost, use raw data to enrich the same 1,000 reference IDs into a separate dataset, and isolate run artifacts under `artifacts/experiment_B`. Parse all selected numeric source features; preserve original labels/doses/state values, row order and grouped split. Pin source/reference hashes and fail on mismatches rather than silently selecting/replacing data.
**Supersedes:** DEC-016's inactive-profile/data-preparation status only; its exploratory scope and feature rationale remain. No additional rows are removed or added, no unit conversion is assumed, and no importance-based selection is finalized. Existing model eligibility threshold and optimization coefficients are unchanged.
**Verification:** 47 tests passed; temporary preprocessing of real data preserved 1,000 reference rows and the same 808/192 split. No final dataset export or research-model training was performed. User execution commands are in `docs/EXPERIMENT_B_PLAN.md`.

## DEC-018 - Configurable Tuning and Controlled Experiment C
**Date/status:** 2026-11-17; implementation explicitly requested by researcher, implemented/tested. Research results pending.
**Decision:** Preserve random search and add grid/Optuna selection from configuration. Isolate C configs/artifacts, hold the selected minimalist schema, cohort and folds fixed, and separate search seeds from estimator/split seeds. Remove all-missing columns only inside training-fit preprocessing. Compare all configured studies, freeze a candidate by minimum eligible mean CV RMSE, and evaluate holdout only by a separate explicit action. Existing eligibility threshold remains 0.60; no unapproved optimizer coefficient is filled.
**Evidence:** `docs/EXPERIMENT_C_USAGE.md`, runner/notebook C and 70 passing tests; no research training execution. The earlier DEC-010 description of random-only implementation is historical; random remains available and no scientific superiority of Optuna is established.
**Limits:** Phase one accepts finite value lists, TPE without pruning and fresh sequential studies. Persistent resume/continuous domains are not implemented. Three search seeds assess search variability, not independent dataset replication. No final surrogate or agronomic recommendation is adopted.

## Decision Addition Format
`DEC-XXX — Title | Date | Status | Decision | Evidence | Rationale | Alternatives | Risks | Affected artifacts | Reviewer/approval | Supersedes`.
