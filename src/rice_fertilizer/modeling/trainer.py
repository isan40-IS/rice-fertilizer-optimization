from __future__ import annotations
from pathlib import Path
from typing import Any
from copy import deepcopy
import logging
import pandas as pd
from ..config import require_value, resolve_path
from ..exceptions import ConfigurationError
from ..utils import save_json
from ..validation.splitters import make_holdout_and_cv
from ..validation.leakage_checks import check_group_overlap, check_record_overlap
from .imputation import resolve_imputation_plan
from .model_registry import create_estimator
from .pipeline_factory import build_model_pipeline
from .tuner import tune_pipeline
from .evaluator import evaluate_cv, evaluate_holdout, summarize_cv
from .interpretation import built_in_feature_importance, permutation_feature_importance
from .model_selector import rank_models
from .serializer import save_model_bundle

LOGGER = logging.getLogger(__name__)

def run_training(configs: dict[str, Any], run_dir: Path) -> dict[str, Path]:
    processed_path = resolve_path(configs["paths"]["processed_dataset"])
    if not processed_path.exists():
        raise FileNotFoundError("Processed data not found. Run scripts/preprocess.py first.")
    df = pd.read_csv(processed_path)

    profile_name = require_value(configs["decisions"], "feature_profile")
    validation_strategy = require_value(configs["decisions"], "validation_strategy")
    target = require_value(configs["decisions"], "target_column")
    decision_features = configs["features"]["decision_features"]
    group_column = configs["decisions"].get("group_column")
    profile = configs["features"]["profiles"][profile_name]
    numeric_features = profile["numeric"]
    categorical_features = profile["categorical"]
    feature_columns = list(dict.fromkeys(
        numeric_features + categorical_features + decision_features
    ))
    missing = [column for column in feature_columns + [target] if column not in df.columns]
    if missing:
        raise ConfigurationError(f"Processed data is missing selected columns: {missing}")

    imputation_plan = resolve_imputation_plan(
        numeric_features, categorical_features,
        configs["decisions"], configs["data"]["nutrient_rates"], target,
    )
    
    split = make_holdout_and_cv(
        df, validation_strategy, group_column, configs["validation"]
    )
    train_df = df.iloc[split.train_indices].copy()
    test_df = df.iloc[split.test_indices].copy()
    X_train = train_df[feature_columns].reset_index(drop=True)
    y_train = train_df[target].reset_index(drop=True)
    X_test = test_df[feature_columns]
    y_test = test_df[target]

    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_assignments = pd.DataFrame({
        "row_index": list(split.train_indices) + list(split.test_indices),
        "split": ["train"] * len(split.train_indices) + ["test"] * len(split.test_indices),
    }).sort_values("row_index")
    if "record_id" in df.columns:
        split_assignments["record_id"] = df.loc[
            split_assignments["row_index"], "record_id"
        ].to_numpy()
    split_assignments.to_csv(split_dir / "split_assignments.csv", index=False)

    leakage = {"record_overlap": check_record_overlap(train_df, test_df, "record_id")}
    if group_column:
        leakage["group_overlap"] = check_group_overlap(train_df, test_df, group_column)
    save_json(leakage, split_dir / "leakage_checks.json")

    models_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    interpretation_dir = run_dir / "interpretation"
    for directory in [models_dir, metrics_dir, interpretation_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    summaries = []
    bundles = {}
    model_cfg = configs["models"]
    for model_name, spec in model_cfg["models"].items():
        if not spec.get("enabled", True):
            continue

        # Flag : Update run model apa
        print("Training model: %s", model_name)
        LOGGER.info("Training model: %s", model_name)

        estimator = create_estimator(spec["class"], spec.get("params", {}))
        pipeline = build_model_pipeline(
            estimator=estimator,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            decision_features=decision_features,
            model_class=spec["class"],
            imputation_plan=imputation_plan,
            imputation_mode=imputation_plan.mode,
        )
        tuned, search_info = tune_pipeline(
            pipeline,
            spec.get("search_space", {}),
            X_train,
            y_train,
            split.cv,
            groups=split.groups,
            n_iter=int(model_cfg.get("search_iterations", 30)),
            scoring=model_cfg.get("scoring", "neg_root_mean_squared_error"),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        )

        fold_metrics = evaluate_cv(tuned, X_train, y_train, split.cv, split.groups)
        cv_summary = summarize_cv(fold_metrics)
        holdout_metrics, predictions = evaluate_holdout(tuned, X_test, y_test)

        fold_metrics.insert(0, "model", model_name)
        fold_metrics.to_csv(metrics_dir / f"{model_name}_fold_metrics.csv", index=False)
        predictions.to_csv(
            metrics_dir / f"{model_name}_holdout_predictions.csv",
            index_label="row_index",
        )
        built_in_feature_importance(tuned).to_csv(
            interpretation_dir / f"{model_name}_built_in_importance.csv", index=False
        )
        permutation_feature_importance(
            tuned,
            X_test,
            y_test,
            scoring=model_cfg.get("scoring"),
            n_repeats=int(model_cfg.get("permutation_importance_repeats", 10)),
            random_state=int(model_cfg.get("random_state", 42)),
            n_jobs=int(model_cfg.get("permutation_importance_n_jobs", 1)),
        ).to_csv(
            interpretation_dir / f"{model_name}_permutation_importance.csv", index=False
        )

        summary = {
            "model": model_name,
            "imputation_mode": imputation_plan.mode,
            **cv_summary,
            **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
            "best_search_score": search_info["best_score"],
        }
        summaries.append(summary)
        bundle = {
            "pipeline": tuned,
            "model_name": model_name,
            "feature_profile": profile_name,
            "imputation_mode": imputation_plan.mode,
            "state_imputation": deepcopy(configs["decisions"]["state_imputation"]),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "target_column": target,
            "ratio_pseudocount": float(
                require_value(
                    configs["decisions"],
                    "feature_engineering.ratio_pseudocount",
                )
            ),
            "validation_strategy": validation_strategy,
            "group_column": group_column,
            "cv_summary": cv_summary,
            "holdout_metrics": holdout_metrics,
            "best_params": search_info["best_params"],
            "training_numeric_statistics": train_df[numeric_features].describe().to_dict(),
            "training_target_statistics": train_df[target].describe().to_dict(),
            "training_reference_numeric": train_df[numeric_features].copy(),
            "training_categorical_levels": {
                column: sorted(train_df[column].dropna().astype(str).unique().tolist())
                for column in categorical_features
            },
        }
        bundles[model_name] = bundle
        save_model_bundle(bundle, models_dir / f"{model_name}.joblib")
        save_json(search_info["best_params"], models_dir / f"{model_name}_best_params.json")

    ranked = rank_models(pd.DataFrame(summaries), model_cfg.get("selection", {}))
    ranked.to_csv(metrics_dir / "model_comparison.csv", index=False)
    selected = ranked.loc[ranked["selected"]]
    if selected.empty:
        raise ConfigurationError("No model satisfied the selection rules")
    selected_name = str(selected.iloc[0]["model"])
    selected_dir = run_dir / "selected_model"
    selected_path = save_model_bundle(
        bundles[selected_name], selected_dir / "model_bundle.joblib"
    )
    save_json({"selected_model": selected_name}, selected_dir / "selection.json")
    return {
        "selected_model": selected_path,
        "model_comparison": metrics_dir / "model_comparison.csv",
        "split_assignments": split_dir / "split_assignments.csv",
    }
