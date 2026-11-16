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
**Status:** Active experimental.
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

## Decision Addition Format
`DEC-XXX — Title | Date | Status | Decision | Evidence | Rationale | Alternatives | Risks | Affected artifacts | Reviewer/approval | Supersedes`.
