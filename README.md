# Rice Fertilizer Optimization

Reproducible research code for a rice NPK fertilizer optimization thesis using:

1. tabular-data preprocessing,
2. yield prediction with Linear Regression, Random Forest, XGBoost, and CatBoost,
3. machine-learning surrogate modeling,
4. Differential Evolution optimization,
5. productivity, fertilizer-efficiency, sensitivity, and stability evaluation.

The repository deliberately contains **no dataset**. Runtime directories under `data/` and `artifacts/` are created automatically when a pipeline is executed.

## Scientific scope

The predictive model approximates:

```text
predicted_yield = f(N, P, K, soil_state, environment_state)
```

During optimization, soil and environmental state variables remain fixed while N, P, and K are changed by Differential Evolution. The optimization score is designed as:

```text
normalized yield improvement
- lambda_total × total-dose penalty
- lambda_composition × extreme-composition penalty
```

Several scientific values are intentionally left unset because they require researcher or agronomic decisions. They include the feature profile, validation design, NPK bounds, baseline, penalty settings, perturbations, agronomic thresholds, and the pseudocount used by NPK-ratio features. They are listed in `configs/decisions.yaml` and explained in `README_HIZKIA.md` and `docs/DECISION_CHECKLIST.md`.

## Installation

```bash
python -m venv .venv
source .venv\Scripts\activate
# .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -e .
```

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Prepare the data

1. Create only the input directory by running `python scripts/prepare_data_directory.py`.
2. Put the CSV file at `data/input/data.csv`.
3. Do not create `data/interim/` or `data/processed/` manually; pipelines create them.
4. Confirm that the CSV column names match `configs/data.yaml` and the selected profile in `configs/features.yaml`.
5. Fill all required decisions in `configs/decisions.yaml`.

## Main commands

```bash
python scripts/preprocess.py
python scripts/train_models.py
python scripts/optimize_scenarios.py --model-bundle artifacts/<training_run>/selected_model/model_bundle.joblib
python scripts/evaluate_optimization.py \
  --model-bundle artifacts/<training_run>/selected_model/model_bundle.joblib \
  --optimization-results artifacts/<optimization_run>/optimization/optimization_results.csv
```

Run preprocessing and training in one run directory:

```bash
python scripts/run_full_experiment.py
```

Optimization is deliberately a separate step because the NPK bounds, baseline, penalties, and agronomic rules must be decided first.

## Outputs

Generated runtime files are stored under:

```text
data/interim/
data/processed/
artifacts/<run_id>/
```

Typical artifacts include data audits, fold metrics, model bundles, split assignments, optimization results, convergence histories, evaluation tables, and figures.

## Important limitations

- Optimization results are model-based simulations, not field-validated fertilizer prescriptions.
- Missing values must not automatically be interpreted as zero.
- Soil measurements with different units, depths, or laboratory methods must not be merged without valid harmonization rules.
- Feature importance is not causal evidence.
- A high predictive score does not automatically guarantee a safe surrogate model for optimization.

See `README_HIZKIA.md` for the Indonesian operational guide.
