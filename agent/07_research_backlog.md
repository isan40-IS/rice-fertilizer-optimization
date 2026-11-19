# 07 — Research Backlog and Experiments

## Experiment-D progress and remaining decisions (2026-11-17, latest)
D diagnostics/playground/parameter evidence are implemented under explicit researcher approval. Notebook uses the frozen C model; model/data/config hashes are preserved, and no DE is run. See current_state and FIND-016 for executed evidence. This supersedes the pending response/domain-diagnostic software tasks only; it does not approve optimizer coefficients or establish causal/field validity.
Remaining priorities: inspect joint dose support and ratio behavior near zero; confirm treatment context (including fertilizer category), management metadata and replicate design before controlled ranking claims; review the worksheet against literature and choose nutrient basis/bounds, total/composition trade-offs, yield scale, baseline policy and warning policy. Final distance cutoff remains undecided. DE stability requires actual optimization/perturbation later. Optional 60-trial tuning or numeric int/float search domains remain future experiments, not necessary automatic next steps.


## Experiment-C implementation progress (2026-11-17)
Multi-backend tuning, seed-study runner, effective snapshots, CV-only candidate freezing, explicit holdout step and notebook C are implemented under the researcher request. 70 tests passed; research studies remain unexecuted. Next: user-run bounded C studies, inspect all seed histories/runtime/eligibility, then response/domain diagnostics for DE. Persistent resume, continuous spaces, pruning, nested CV and further feature exploration remain future work; no new results or agronomic decisions follow from passing software tests.


## Experiment-C design artifact (2026-11-17)
`docs/EXPERIMENT_C_TUNING_PLAN.md` records the requested multi-backend development plan, Optuna usage, reproducibility/resume distinctions and proposed repeated-seed analysis. Planning deliverable complete; implementation, dependency installation, final C profile/budget, notebook creation and study execution remain pending. This does not supersede the existing random-search implementation or establish a new scientific result.


## Experiment-B analysis progress (2026-11-17)
User-run training artifacts now exist for `ta_20261116T203731Z`. Notebook diagnostics/interpretation were added and verified read-only (FIND-014); this supersedes earlier pending-training status for this run. Coverage, provenance, aggregation and error/spatial diagnostics are available. Optional joint permutations remain disabled/unexecuted; training-fold importance, spatial ablation and final feature selection remain pending proposals. No new row removal or methodology change is authorized by this progress note.


## Prioritization Rules
- **P0:** necessary to honor current supervisor direction or prevent scientifically invalid optimization.
- **P1:** important to evaluate and interpret optimization once a baseline is running.
- **P2:** useful model enhancement or exploratory diagnostics that should not block immediate optimization by default.

Backlog items are proposals, not approval to modify the current dataset/methodology. Prefer smaller tests that isolate a single hypothesis.

## Implementation Progress After Audit (2026-11-17)
Researcher authorized no-imputation implementation (DEC-015). TASK-002 source correction and targeted verification are complete; future research-data training remains pending. TASK-014 notebook/config wiring correction is complete; historical selections are preserved and a new scientific model selection is pending. TASK-015 existing-test repair/regression checks are complete: **39 tests passed**. No broad-feature importance experiment was run. Keep TASK-001 provenance/freeze, TASK-003 response readiness, open objective decisions and TASK-004 DE work pending.

Progress/proposal text below records audit-time status. Implementation is now authorized and completed only to the extent stated here; historical result comparison, new training, broader features and final methodology choices are not automatically complete.

## P0 — Baseline Verification and Initial Optimization

### TASK-001 — Verify Latest Experiment Artifacts
**Goal:** Establish a reproducible current baseline.
**Actions:** Find dataset file/hash, feature list, actual output metrics, split/group assignments, best estimator/bundle, preprocessing behavior, config snapshot, seeds.
**Acceptance:** A concise machine-readable run summary with paths and actual holdout/CV metrics; no guessed values.
**Progress 2026-11-17:** Dataset/hash, 11 inputs, both cohort metrics, split IDs/no group overlap, bundles and preprocessing verified. A machine-readable immutable manifest, effective notebook overrides and full 1,000-row generation lineage remain pending; acceptance is not fully met.

### TASK-002 — Audit Native Missing Pipeline
**Goal:** Ensure XGBoost/CatBoost actually use the intended missing-value pathway.
**Actions:** Inspect active pipeline and sample transformed model input; inspect fit/predict compatibility; run focused tests.
**Acceptance:** The full inference path for missing state features is traceable and matches experimental claims.
**Progress 2026-11-17:** Audit completed and mismatch verified (FIND-007); corrected native execution is pending. Proposed correction: explicitly handle resolver `none` and pass `imputation_mode` through callers; verify missing values reach intended estimators and inference matches training. Preserve cohort/group design and historical artifacts. This entry alone does not authorize implementation or retraining.

### TASK-003 — Check Optimization Readiness
**Goal:** Ensure trained surrogate can score candidate N/P/K consistently.
**Actions:** Inspect feature order/units, derived NPK recomputation, state-feature handling, missing values, baseline-dose construction, model bundle loading.
**Acceptance:** Repeated calls with unchanged inputs are reproducible; changing dose changes the intended fields only; no unexpected NaNs/errors.
**Progress 2026-11-17:** Bundle schema and candidate NPK recomputation inspected; runtime dose-response checks remain pending. Include fertilizer-type consistency when doses change, large ratios near zero denominators, baseline context policy and training support.

### TASK-004 — Execute First Differential Evolution Baseline
**Goal:** Produce first feasible model-based N/P/K recommendations on fixed state scenarios.
**Actions:** Validate bounds/penalties; run with controlled seeds; compare with baseline; save objective, N/P/K, predicted yields, predicted delta, constraint/penalty diagnostics.
**Acceptance:** Reproducible experiment with no fabricated parameters and explicit limitations.
**Progress 2026-11-17:** Implementation located; no DE run performed. Current enabled objective requires unset `yield_scale`, `lambda_total`, `lambda_composition`. Resolve researcher decisions/readiness first; do not invent values or disable terms silently.

## P1 — Optimization Trustworthiness

### TASK-005 — Fertilizer Response Diagnostics
**Question:** Does the surrogate respond meaningfully to N/P/K variation at a fixed state?
**Method:** One-at-a-time and limited joint dose sweeps inside supported ranges; show curves and data-support diagnostics.
**Risk:** Model response is associational, not a measured causal response.

### TASK-006 — Constraint and Boundary Diagnostics
**Question:** Do optimized doses crowd against boundaries or unusual composition patterns?
**Method:** Report boundary distances, total dose, NPK proportions, feasible rate, and sensitivity to penalty choices.

### TASK-007 — Stability and Sensitivity
**Question:** Are recommendations stable across DE seeds and small realistic state perturbations?
**Method:** Separate algorithm stability from input stability; honor configured seed schemes where valid; preserve scenario eligibility.

### TASK-008 — Agronomic Plausibility Review
**Question:** Are dose ranges and predicted responses reasonable for the dataset/domain?
**Method:** Literature-supported checks and expert feedback; document decisions and unresolved concerns.

## P2 — Prediction Research Backlog

### TASK-009 — Diagnose Horizontal Prediction Bands
**Question:** Are repeated predictions model structure, feature compression, missingness effects, or visualization artifacts?
**Method:** Compare predicted-value precision/counts, distribution plots, experiment-wise residuals, feature duplication, and fixed-state dose sweeps.
**Decision gate:** Only modify model if there is evidence that banding harms generalization or optimization suitability.
**Progress 2026-11-17:** 192 holdout rows have 46 unique raw/transformed feature combinations and 42 unique XGBoost predictions (FIND-004). Supports repeated-input explanation without explaining every visual band; no row deletion.

### TASK-010 — Controlled Missingness/Feature Ablations
**Question:** Which change was responsible for better performance?
**Method:** Under matched train/test groups, compare one factor at a time: feature profile, native missing handling, imputation, row cohort. Do not add row deletions to the active baseline.
**Caution:** Keep dataset/population differences explicit and avoid unfair comparisons.

### TASK-011 — Broader Feature Importance Study (Nesha's Suggestion)
**Question:** Which measured or proxy variables contribute to predictive utility, and are dose/state features meaningful?
**Method:** Train one interpretable baseline on a scientifically valid broader feature set if feasible; use permutation/group-aware importance and stability; inspect correlated features and leakage risks.
**Caution:** Importance does not establish causal fertilizer response and should not alone decide which essential state variables to drop.

### TASK-012 — Evaluate Optuna vs RandomizedSearchCV (Nesha's Suggestion)
**Question:** Does adaptive search improve validated performance per unit compute?
**Method:** Same dataset, folds, target, search space, scoring, and comparable time/trial budget. Report validation vs holdout separately.
**Decision gate:** Adopt only if gains and implementation complexity justify it. Do not let tuning delay the optimization milestone without a concrete reason.

### TASK-013 — MILP Data-Selection Audit (Earlier Report: MILM)
**Question:** How did row selection change study/year/dose/target coverage?
**Method:** Compare included/excluded groups and distributions, inspect algorithm and criteria, preserve original IDs and processing manifest.
**Restriction:** This is an audit, not permission to delete more rows from the active dataset.
**Progress 2026-11-17:** MILP algorithm/stored 800-row run located; 1,000-row cohort statistics/hash verified (FIND-005). Exact generation settings, solver evidence and full retained/excluded coverage comparison remain pending.

## Additional Corrective Proposals

### TASK-014 - Experiment Selection and Effective-Config Traceability (P0)
**Problem:** Model override loses selection weights (FIND-008); notebook mixes cohort/run references and snapshots omit overrides (FIND-009).
**Method:** Preserve explicit weights/effective settings; bind analyses to dataset hash and saved splits. Keep historical artifacts unchanged; review winner after comparable evidence is available.
**Acceptance:** Intended weights recorded, ranking traceable, each analysis linked to one cohort/run. This does not preselect an estimator.

### TASK-015 - Revalidate Current Tests (P1)
**Problem:** Existing tests use old interface/config assumptions (FIND-010).
**Method:** Update focused tests when implementation work is authorized; verify meaningful native-input preservation/selection regression checks and report actual execution results.
**Limit:** No tests were modified or executed by the documentation update.

### TASK-016 - Experiment B: Broad Feature-Importance Study (Explicitly Deferred)
**Request/status:** Researcher requested preparation, not execution. Metadata-based profile and audit/plan completed on 2026-11-17; see DEC-016/FIND-012 and `docs/EXPERIMENT_B_PLAN.md`. This concretizes TASK-011 without changing the active baseline.
**Pending readiness:** Enrich the same approved record IDs into a separate dataset, parse all selected numeric fields, review units/measurement timing and validate schema. No row selection or dataset replacement is authorized by this entry.
**Proposed implementation:** Assert feature-name/value alignment, export method/scoring/provenance and mappings, retain raw permutation repetitions, plan training-group-CV importance and coherent NPK-family permutations. Compare both models with coverage/cardinality/fold stability; do not automatically select/remove features from holdout ranks.
**Restriction:** Do not execute experiment B, train a broad model, or activate the new profile until requested. No importance result or reduced feature set exists yet.

### TASK-017 - Experiment-B Runnable Configuration and Cohort Preparation
**Status:** Completed implementation/configuration and temporary verification, 2026-11-17 (DEC-017/FIND-013).
**Done:** Separate output/artifact paths; exact reference cohort with provenance/value checks; all numeric source parsing; coordinate checks; 47 passing tests and real-data preflight preserving 1,000 IDs and the holdout. This completes TASK-016's technical enrichment/parsing readiness, not its feature-importance analysis.
**Pending:** User-run final preprocessing/training, review metric eligibility and data/importance diagnostics, then explicitly requested experiment-B analyses. Preserve unit/timing cautions, holdout-use restrictions and DE decision gates. No scientific result is marked complete merely because configs are runnable.

## Next-Task Selection
**Historical recommendation after audit (implementation progress above supersedes completed items):** finish TASK-001 provenance -> TASK-002 and TASK-014 correction -> necessary matched-cohort comparison and explicit surrogate selection -> TASK-003 with minimal TASK-005 response checks -> resolve open coefficients/bounds/baseline -> TASK-004. Keep corrective work bounded to support the optimization milestone. Broad feature search/Optuna remain P2. Researcher approval is required for scientific configuration changes.

After each experiment, record: hypothesis, dataset and split, exact modification, verification artifact, outcome, uncertainty, decision, and whether it changes the active baseline.
