# Project Tree

Tree berikut menggambarkan struktur proyek saat ini. Isi `artifacts/`,
`data/`, dan `notebooks/exploration_results/` adalah hasil runtime atau
eksplorasi sehingga tidak dirinci satu per satu.

```text
rice-fertilizer-optimization/
├── .gitignore
├── Makefile
├── README.md
├── README_HIZKIA.md
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── artifacts/
│   └── ta_<timestamp>/
│       ├── config_snapshot/
│       ├── data_audit/
│       └── run_metadata.json
├── configs/
│   ├── examples/
│   │   └── decisions.example.yaml
│   ├── data.yaml
│   ├── decisions.yaml
│   ├── evaluation.yaml
│   ├── features.yaml
│   ├── models.yaml
│   ├── optimization.yaml
│   ├── paths.yaml
│   └── validation.yaml
├── data/
│   ├── input/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY_EXPECTED.md
│   ├── DECISION_CHECKLIST.md
│   ├── METHODOLOGY_TO_CODE_MAP.md
│   ├── PROJECT_TREE.md
│   └── VALIDATION_STATUS.md
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_feature_feasibility.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_surrogate_diagnostics.ipynb
│   ├── 04_optimization_results.ipynb
│   ├── 05_sensitivity_stability.ipynb
│   ├── exploration_notebooks/
│   └── exploration_results/
├── scripts/
│   ├── evaluate_optimization.py
│   ├── optimize_scenarios.py
│   ├── prepare_data_directory.py
│   ├── preprocess.py
│   ├── run_full_experiment.py
│   ├── setup_env.py
│   └── train_models.py
├── src/
│   └── rice_fertilizer/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── exceptions.py
│       ├── schemas.py
│       ├── utils.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── cleaner.py
│       │   ├── crop_filter.py
│       │   ├── feature_engineering.py
│       │   ├── loader.py
│       │   ├── missing_values.py
│       │   ├── preprocessing_pipeline.py
│       │   └── validator.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── agronomic_validation.py
│       │   ├── algorithm_stability.py
│       │   ├── baseline.py
│       │   ├── fertilizer_efficiency.py
│       │   ├── input_stability.py
│       │   ├── perturbation.py
│       │   ├── productivity.py
│       │   └── sensitivity.py
│       ├── modeling/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   ├── interpretation.py
│       │   ├── metrics.py
│       │   ├── model_registry.py
│       │   ├── model_selector.py
│       │   ├── pipeline_factory.py
│       │   ├── serializer.py
│       │   ├── trainer.py
│       │   └── tuner.py
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── bounds.py
│       │   ├── config_factory.py
│       │   ├── constraints.py
│       │   ├── convergence.py
│       │   ├── differential_evolution.py
│       │   ├── objective.py
│       │   ├── optimizer.py
│       │   └── penalties.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── common.py
│       │   ├── run_evaluation.py
│       │   ├── run_full_experiment.py
│       │   ├── run_optimization.py
│       │   ├── run_preprocessing.py
│       │   └── run_training.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── export.py
│       │   ├── plots.py
│       │   └── tables.py
│       ├── surrogate/
│       │   ├── __init__.py
│       │   ├── candidate_builder.py
│       │   ├── domain_guard.py
│       │   ├── predictor.py
│       │   └── scenario.py
│       └── validation/
│           ├── __init__.py
│           ├── domain_checks.py
│           ├── leakage_checks.py
│           └── splitters.py
└── tests/
	├── conftest.py
	├── e2e/
	│   └── test_smoke_configuration.py
	├── integration/
	│   ├── test_model_pipeline.py
	│   └── test_optimizer_integration.py
	└── unit/
		├── test_baseline.py
		├── test_efficiency.py
		├── test_feature_engineering.py
		├── test_missing_values.py
		├── test_objective.py
		├── test_penalties.py
		├── test_productivity.py
		└── test_sensitivity.py
```

Folder `.rfo/` adalah virtual environment lokal dan tidak ditampilkan.
Folder `artifacts/` dan subfolder `data/` diisi oleh script pipeline.
