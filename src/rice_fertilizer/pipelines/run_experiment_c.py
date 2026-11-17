"""Controlled repeated-seed tuning. Training and final holdout are separate actions."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import argparse
import json
import platform

import pandas as pd

from ..config import load_all_configs, project_root, resolve_path
from ..exceptions import ConfigurationError
from ..modeling.trainer import run_training
from ..modeling.serializer import load_model_bundle
from ..modeling.evaluator import evaluate_holdout
from ..modeling.interpretation import permutation_feature_importance
from ..utils import make_run_id, save_json, hash_file, configure_logging
from .common import initialize_run


def run_experiment(config_dir=None):
    config_dir = Path(config_dir) if config_dir else project_root()/"configs/experiment_C"
    base = load_all_configs(config_dir)
    settings = base["experiment"]
    methods, seeds = settings["methods"], settings["search_seeds"]
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"random", "grid", "optuna"}:
        raise ConfigurationError("experiment.methods must be unique supported methods")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ConfigurationError("experiment.search_seeds must be unique nonnegative integers")
    if base["models"].get("scoring") != "neg_root_mean_squared_error":
        raise ConfigurationError("Experiment C selection requires negative RMSE scoring")
    # Check dependency and input before creating any research run.
    if "optuna" in methods:
        try:
            import optuna
        except ImportError as exc:
            raise ConfigurationError('Install Optuna: pip install -e ".[tuning]"') from exc
    dataset = resolve_path(base["paths"]["processed_dataset"])
    group = resolve_path(base["paths"]["artifacts_root"])/(make_run_id("exp_C")+"_"+uuid4().hex[:8])
    group.mkdir(parents=True, exist_ok=False)
    manifest = {"group_dir": str(group), "dataset_sha256": hash_file(dataset),
                "methods": methods, "search_seeds": seeds, "python": platform.python_version(),
                "status": "running", "runs": []}
    save_json(manifest, group/"experiment_manifest.json")
    tables = []
    common_fingerprint = None
    try:
        for method in methods:
            # Grid has no sampling randomness; run it once, not three identical times.
            for seed in (seeds[:1] if method == "grid" else seeds):
                if hash_file(dataset) != manifest["dataset_sha256"]:
                    raise ConfigurationError("Dataset changed during experiment; refusing next study")
                run_id = f"{method}_seed_{seed}"
                overrides = {"paths": {"artifacts_root": str(group)}, "models": {
                    "tuning": {"method": method, "search_seed": seed},
                    "report_holdout": False, "require_selected": False}}
                configs, run_dir = initialize_run(config_dir, run_id, overrides)
                outputs = run_training(configs, run_dir)
                provenance = json.loads((run_dir/"training_manifest.json").read_text())
                fingerprint = (provenance["dataset_sha256"], provenance["feature_columns"], provenance["cv_manifest_sha256"])
                if common_fingerprint is not None and fingerprint != common_fingerprint:
                    raise ConfigurationError("Dataset/features/folds differ across experiment studies")
                common_fingerprint = fingerprint
                comparison = pd.read_csv(outputs["model_comparison"])
                comparison["run_dir"] = str(run_dir)
                tables.append(comparison)
                manifest["runs"].append({"run_dir": str(run_dir), "method": method, "search_seed": seed})
                save_json(manifest, group/"experiment_manifest.json")
                pd.concat(tables, ignore_index=True).to_csv(group/"study_comparison.csv", index=False)
        result = pd.concat(tables, ignore_index=True)
        eligible = result.loc[result.eligible].sort_values(
            ["rmse_mean", "model", "tuning_method", "search_seed"])
        # Freeze one candidate using CV only. No holdout metrics enter this decision.
        winner = None if eligible.empty else eligible.iloc[0].to_dict()
        save_json({"rule": "minimum mean CV RMSE among eligible model/studies; deterministic tie order", "candidate": winner}, group/"frozen_candidate.json")
        manifest["status"] = "complete"
        save_json(manifest, group/"experiment_manifest.json")
        return group
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(manifest, group/"experiment_manifest.json")
        raise


def evaluate_frozen_candidate(group_dir):
    group = Path(group_dir).resolve()
    frozen = json.loads((group/"frozen_candidate.json").read_text())
    candidate = frozen["candidate"]
    if candidate is None:
        raise ConfigurationError("No candidate met the configured CV eligibility threshold")
    run = Path(candidate["run_dir"])
    configs = load_all_configs(run/"config_snapshot")
    bundle = load_model_bundle(run/"models"/f"{candidate['model']}.joblib")
    data_path = resolve_path(configs["paths"]["processed_dataset"])
    if hash_file(data_path) != bundle["provenance"]["dataset_sha256"]:
        raise ConfigurationError("Dataset changed since tuning")
    df = pd.read_csv(data_path)
    split = pd.read_csv(run/"splits/split_assignments.csv")
    test = df.iloc[split.loc[split['split'].eq('test'), 'row_index']]
    metrics, predictions = evaluate_holdout(bundle["pipeline"], test[bundle["feature_columns"]], test[bundle["target_column"]])
    out = group/"selected_holdout"
    out.mkdir(exist_ok=True)
    predictions.to_csv(out/"predictions.csv", index_label="row_index")
    save_json({"candidate": candidate, "holdout_metrics": metrics,
               "note": "Previously inspected holdout; descriptive, not pristine confirmatory evaluation"}, out/"evaluation.json")
    permutation_feature_importance(bundle["pipeline"], test[bundle["feature_columns"]], test[bundle["target_column"]],
        scoring=configs["models"]["scoring"], n_repeats=configs["models"].get("permutation_importance_repeats",10),
        random_state=configs["models"].get("random_state",42), n_jobs=1).to_csv(out/"permutation_importance.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(project_root()/"configs/experiment_C"))
    parser.add_argument("--evaluate-selected", metavar="GROUP_DIR", help="Evaluate an already frozen candidate; no tuning")
    args = parser.parse_args()
    configure_logging()
    if args.evaluate_selected:
        print(evaluate_frozen_candidate(args.evaluate_selected))
    else:
        print("Experiment group:", run_experiment(args.config_dir))


if __name__ == "__main__":
    main()
