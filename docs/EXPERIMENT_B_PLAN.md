# Experiment B: metadata-based feature importance

Status: configurations and preprocessing support prepared, 2026-11-17.
Preprocessing was verified on real data in a temporary directory: 1,000 rows,
106 dataset columns, 97 model inputs, identical reference values and holdout IDs
(808 training / 192 holdout). All 47 tests passed. No research-model training,
new scientific importance analysis, feature selection, or experiment-B notebook
execution was performed. The final dataset is created when the user runs preprocessing.

## Ready-to-run configuration

| File | Active setting |
| --- | --- |
| `configs/paths.yaml` | Raw input `data/input/data.csv`; read-only cohort reference `data/processed/model_dataset_balanced_1000_minimalist_504622.csv`; new output `data/processed/experiment_B_candidates.csv`; artifacts under `artifacts/experiment_B/` |
| `configs/data.yaml` | `cohort.mode: reference`, 1,000 rows, pinned source/reference SHA-256; all 44 numeric source/target fields; latitude/longitude range checks |
| `configs/decisions.yaml` | `feature_importance_candidates`, no imputation, same target and group validation |
| `configs/models.yaml` | Only XGBoost/CatBoost; existing search/scoring settings and minimum CV R2 0.60 retained |
| `configs/validation.yaml` | Existing seed 42, five-fold GroupKFold and 20% group holdout retained |

Run from the repository root with the project Python environment active:

```powershell
python scripts/preprocess.py
python scripts/train_models.py
```

Or keep data preparation/training audits together in one run directory:

```powershell
python scripts/run_full_experiment.py
```

These commands are instructions for the user; they were not run on the final
experiment paths during configuration preparation. Each run uses a fresh directory
under `artifacts/experiment_B/`. Training alone requires the enriched CSV to exist.
The baseline CSV is never the output path. Missing source/reference IDs, hash
changes, a changed reference value/order/count or invalid coordinates cause a
failure rather than silent cohort changes. `data/cohort.py` and the preprocessing
orchestrator enforce these checks. The selected profile's numeric source columns
are always added to the coercion list, including future profile changes.

Preflight preserved all 32,901 missing numeric entries through the no-imputation
preprocessor (196 encoded columns). Coercion recorded 20 nonnumeric precipitation
values; their missingness already matches the reference cohort. No text extraction
or unit conversion was invented. Existing optimizer/evaluation coefficients remain
unset because they are not required for prediction training.

Model eligibility still requires mean CV R2 >= 0.60 as previously configured by
the researcher. If neither model passes, training exports metrics/importance but
does not produce a selected-model bundle; do not silently lower the threshold.
Importance investigation can still inspect both models' exported diagnostic tables.

## Scope and current configuration

- Compare XGBoost and CatBoost only; these are already the only enabled models
  in `configs/models.yaml`. Keep native numerical missing handling (`mode: none`).
- New optional profile: `profiles.feature_importance_candidates` in
  `configs/features.yaml`: **50 numeric + 47 categorical = 97 unique inputs**,
  including all seven implemented NPK-derived features and direct N/P/K.
- The active profile is now `feature_importance_candidates` following the user's
  configuration request. Output paths are separate; selection rules, seeds,
  grouped splitting and optimizer settings are preserved.
- This is a diagnostic candidate pool, not an approved final feature set or a
  schema ready for NPK optimization. No missingness threshold selected features.

## Metadata and inclusion rationale

Source: `data/metadata/metadata.xlsx`, sheet `Metadata`, checked against all 125
actual headers and example values in `data/input/data.csv`.

| Family | Treatment in profile | Metadata evidence |
| --- | --- | --- |
| Spatial context | Reported/approximate latitude and longitude numeric; keep separate | A9:A12 |
| Experiment duration | `exp_year` numeric, not publication year or article ID | A18 |
| Fertilization | N/P/K/S nutrient doses, product rates, concentrations, additional nutrient/fertilizer rates numeric; source/type/name categorical | A32:A53 |
| Soil nutrients | Layers kept separate; eligible nutrient measures numeric and accompanying units/methods/depths categorical | A81:A115 |
| Physical/chemical soil | Density, SOC/SOM, total N, clay, initial pH/EC numeric; texture/type/method/depth/soil-water ratio categorical | A117:A150 |
| Environment | Temperature, precipitation, slope and elevation numeric | A151:A154 |
| Management | Rotation, previous crop, tillage categorical | A25:A27 |
| Engineered NPK | Total, three ratios, three proportions numeric | `data/feature_engineering.py`, not metadata |

Metadata names are not always literal CSV headers: `bd1/bd2/bd3` map to
`bd.1/bd.2/bd.3`; `crop rotation` maps to `crop_rotation`. The metadata repeats an
additional-nutrient rate label for fertilizer and nutrient quantities; retain
the two distinct actual CSV headers and document their meaning separately.
`soc.method` is present in the actual data and is included as measurement context.

Depth intervals such as `0-20` and ratios such as `1:2.5` are categorical; no
unapproved midpoint/depth averaging or conversion is introduced. Several soil
measurements are not explicitly labeled initial by the metadata. Their timing
must be checked before claiming they are deployable pre-treatment predictors.
Method/depth/source columns and coordinates can act as study proxies; their
importance is predictive evidence, not proof of agronomic influence.

### Explicit exclusions (not based on missingness)

- All `yield.*` (target itself and related outcome/measurement information),
  all `biomass.*`, and `final.ph.soil` / `final.ec.soil.dS_m`. Metadata A55:A66
  describes outcomes; A144/A149 explicitly identify after-treatment measures.
- Article/record/group/experiment IDs, authors, DOI/URL, publication year,
  treatment name and comments: identification/free-text proxies or leakage risk.
- `year` and `location`: leave as evaluation/context metadata for now. Year has
  mixed representations (including ranges), and location is an explicit site
  proxy. Coordinates are included as an exploratory spatial proxy; this is not
  a demonstration of generalization to unseen locations.
- Experimental design, replicate count and study category: methodological
  descriptors, not primary agronomic inputs. `Additional_descriptor` mixes
  free-text context and needs a separate semantic extraction design.
- `crop` and `country`: scope labels, constant for the rice/Senegal task.
- `Nutrients applied (NPKS)`: retain for structural-zero reconstruction; omit
  from model inputs as a redundant treatment indicator.
- `P.soil.layer1` / its unit and `K.soil.layer1` / its unit are deferred for a
  **unit audit**, not for missingness. Rice observations report P in `ppm` and
  `mg_kg`, and K in `meq_100g`, `cmol_kg` and `ppm`. Establish/document equivalent
  units or conversions before pooling. Adding a unit category alone does not
  establish harmonization. Other retained layers require the same audit before
  execution; no unit conversion was performed in this task.

Entirely missing or constant candidate columns are retained as requested, but
must be flagged as uninformative in this cohort. A zero importance for such a
column is not evidence that the corresponding agronomic quantity is irrelevant.

## Data preparation design and implemented readiness checks

The current 1,000-row minimalist CSV has only 20 columns and lacks **86 of the
97 candidate inputs** (including three additional derived proportions).
Changing only `feature_profile` and running training will therefore fail.
An already fitted minimalist model can only provide importance for its own
training inputs. Assessing the 97-input profile requires a newly trained broad
model after data preparation; extra columns cannot be attached retrospectively
to an existing fitted model to obtain meaningful importance for them.

The following design guided the implementation. Steps 1-3 and schema checks were
verified by temporary preprocessing; unit/timing interpretation remains a research
limitation, and no final dataset export/model training was performed:

1. Preserve the current 1,000 record IDs and saved group/split assignments.
   Reattach raw columns using the existing `add_record_id` convention, with
   one-to-one ID, row-count and original feature/target consistency checks.
   Do not rerun balancing, remove rows or overwrite the active dataset.
2. Preserve approved N/P/K and target values; regenerate all engineered features
   using the existing pseudocount and zero-dose convention. Write a separate
   enriched experiment-B dataset and record its hash and lineage.
3. Convert **all declared numeric source features** with the existing
   decimal-comma parsing rules; audit newly coerced missing values and ambiguous
   strings. Current preprocessing coerces only `configs/data.yaml.numeric_columns`
   originally had 11 fields, which was insufficient. The active list now covers
   all 44 source/target fields; preprocessing also includes selected numeric source
   columns automatically. Native missing
   support does not accept unparsed numeric strings as numerical inputs.
4. Review units/layers/measurement timing and ambiguous metadata. Keep unit/depth
   annotations; do not silently assume all soil measures are comparable or initial.
5. Confirm predictor schema, numeric dtypes, categorical missing representation,
   finite nonmissing numeric values and no target/outcome/ID predictors. Preserve
   numerical NaNs; one-hot encode categories without imputation. Log complete,
   partial, constant and all-missing columns without deleting rows.
6. Enable this profile with a separate run/config snapshot only when preparation
   is reviewed. A separate broader cohort would be a new population decision,
   not an automatic replacement of the active baseline.

## Audit of current importance implementation

Sources: `modeling/interpretation.py`, `modeling/trainer.py`, installed model
property implementations, and official documentation linked below.

### What currently works

- `built_in_feature_importance` gets transformed feature names from the fitted
  preprocessor and values from the fitted estimator. Order comes from the same
  fitted pipeline, which is the correct basis for matching names to model inputs.
- XGBoost tree default importance is normalized gain; CatBoost regression uses
  PredictionValuesChange. Their scales and meanings differ; do not average or
  directly compare their raw importance values as if they were the same metric.
- Categorical built-in scores belong to one-hot output columns, not whole raw
  categorical variables. Aggregate via fitted transformer/encoder mappings;
  do not split names heuristically on underscores (raw names contain them).
- `permutation_feature_importance` permutes the raw input DataFrame and scores the
  **entire fitted pipeline**. One raw categorical column is shuffled before
  encoding, so its one-hot indicators stay internally consistent as a block.
- Current configured scoring is `neg_root_mean_squared_error`, with 10 repeats,
  seed 42 and one permutation worker. Positive importance equals the increase
  in RMSE after permutation, in kg/ha. Negative means shuffling improved the
  score on these data; zero/negative is not an automatic feature-deletion rule.

### Changes/checks to plan before feature selection

1. Replace silent `min(len(names), len(values))` truncation with an explicit
   dimension assertion/error. A mismatch currently hides a schema problem.
2. Export the importance definition, scoring, evaluation partition, model,
   dataset/hash, fold, seed, original-to-transformed mapping and feature coverage.
   If scoring is omitted, trainer currently passes `None`, causing estimator
   default scoring rather than the helper's negative-RMSE default; use an explicit
   fallback in future work. Active scoring is already explicit, so current config
   is unaffected.
3. Keep signed per-repeat permutation values. `importance_std` is repeat-to-repeat
   dispersion, not a confidence interval and not fold/group generalization
   uncertainty. Report fold variation separately; do not sum standard deviations.
4. Standard column permutation breaks deterministic NPK relationships: shuffling
   raw N while leaving total/ratios/proportions unchanged creates inconsistent
   input. Per-column results measure reliance on individual representations,
   not a valid fertilizer response. Add a joint NPK-family permutation using the
   same row permutation across doses and all deterministic derivatives. This
   preserves internal algebra, but can still break relationships
   with soil/management context; it is a model-reliance diagnostic, not a causal
   or necessarily in-domain agronomic intervention. A separate
   coherent one-nutrient perturbation must recompute derivatives, but can still
   produce unsupported combinations; treat it as a sensitivity diagnostic.
5. SOC/SOM, coordinates, doses/product rates/concentrations and NPK derivatives
   can be strongly dependent. Individual permutation importance can be low because
   another feature substitutes for it. Plan justified family-level permutations;
   their effects need not sum to the individual effects.
6. Native missingness and laboratory/study context can supply predictive signal.
   Include nonmissing counts/groups and all-missing flags in reporting. A learned
   split/importance score does not isolate physical feature effects from missingness
   or study proxies. One-hot treatment is not CatBoost native categorical training.
7. Current trainer calculates permutation importance on `X_test/y_test` only,
   for the final full-training estimator, not per CV fold. Using those same
   holdout importances to select features and then reporting that holdout as an
   untouched final test would be adaptive evaluation. For feature discovery,
   propose training-partition group-CV importance on validation folds, with
   preprocessing/refitting inside each fold and no holdout-guided choice. If
   hyperparameter selection uses the same folds, label it development analysis;
   nested evaluation would be needed for a stronger independent performance claim.
   Existing holdout has already been inspected in earlier research, so do not
   retroactively claim it is pristine. No new split policy is adopted here.
8. Check prediction quality and group residuals before interpreting rankings.
   A weak model's low importance is weak evidence about feature usefulness.

## Proposed experiment-B sequence (not executed)

1. Record the run/dataset/profile/hash and establish schema/coverage/unit readiness.
2. Inspect both models' CV/holdout predictive performance; retain both for comparison.
3. Present built-in transformed scores, a traceable original-feature aggregation,
   and categorical cardinality/coverage beside the rankings.
4. For discovery, compute raw-column and scientifically justified family-level
   permutation importance on training CV validation folds. Preserve group-aware
   splitting and control random seeds/repeat counts. Existing holdout CSVs may be
   read descriptively, with their selection-use limitation explicitly noted.
5. Compare rankings/stability across models, folds and feature families; show
   signed RMSE changes with uncertainty appropriately labeled. Keep importance
   scales separate. SHAP is optional future work, not a requirement or implemented
   capability of the current pipeline.
6. Propose a reduced profile with scientific rationale and controlled validation;
   do not automatically discard N/P/K or required state variables based on ranks.
   No selected profile becomes the DE surrogate automatically.

## Access after a training run

The current trainer automatically exports both methods for each enabled model:

```text
artifacts/experiment_B/<run_id>/interpretation/xgboost_built_in_importance.csv
artifacts/experiment_B/<run_id>/interpretation/catboost_built_in_importance.csv
artifacts/experiment_B/<run_id>/interpretation/xgboost_permutation_importance.csv
artifacts/experiment_B/<run_id>/interpretation/catboost_permutation_importance.csv
```

Built-in columns: `feature, importance`. Permutation columns:
`feature, importance_mean, importance_std`. Fold-level importance, original-feature
built-in aggregation, family-level permutations and raw repeat values are **not
currently exported**. Proposed experiment B would add these separately.

Reading existing files needs no retraining:

```python
from pathlib import Path
import pandas as pd

run = Path("artifacts/experiment_B/<actual_run_id>")
for model in ("xgboost", "catboost"):
    built_in = pd.read_csv(run / "interpretation" / f"{model}_built_in_importance.csv")
    permutation = pd.read_csv(run / "interpretation" / f"{model}_permutation_importance.csv")
    display(built_in.head(20))
    display(permutation.head(20))
```

All named scores describe model-specific predictive reliance/associations, not
causal agronomic effects or field-validated fertilizer response.

## Primary documentation

- [scikit-learn: permutation importance, scoring and correlated features](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [XGBoost Python API: importance_type and feature_importances_](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [CatBoost: PredictionValuesChange and feature importance](https://catboost.ai/docs/en/concepts/fstr)
