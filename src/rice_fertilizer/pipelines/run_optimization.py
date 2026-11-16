from __future__ import annotations
import pandas as pd
from .common import initialize_run
from ..config import require_value, resolve_path
from ..evaluation.baseline import resolve_baseline
from ..modeling.serializer import load_model_bundle
from ..optimization.bounds import bounds_from_decisions
from ..optimization.config_factory import objective_config_from_configs
from ..optimization.optimizer import optimize_scenario
from ..surrogate.predictor import SurrogatePredictor
from ..surrogate.scenario import build_observation_scenarios
from ..utils import configure_logging, save_dataframe

def main(run_id: str | None = None, model_bundle_path: str | None = None):
    configure_logging()
    configs, run_dir = initialize_run(run_id=run_id)
    if not model_bundle_path:
        raise FileNotFoundError("Provide --model-bundle from a training run")
    processed_path = resolve_path(configs["paths"]["processed_dataset"])
    if not processed_path.exists():
        raise FileNotFoundError("Run preprocessing first")
    df = pd.read_csv(processed_path)
    bundle = load_model_bundle(model_bundle_path)
    predictor = SurrogatePredictor(bundle, configs["data"]["nutrient_rates"])

    decision_columns = set(configs["data"]["nutrient_rates"].values())
    engineered = set(configs["features"]["engineered_features"])
    state_features = [
        column for column in bundle["feature_columns"]
        if column not in decision_columns and column not in engineered
    ]
    baseline_policy = require_value(configs["decisions"], "baseline_policy")
    group_column = configs["decisions"].get("group_column")
    target = require_value(configs["decisions"], "target_column")
    baseline_resolver = lambda row: resolve_baseline(
        baseline_policy, row, df, configs["data"]["nutrient_rates"],
        group_column=group_column, target_column=target,
    )
    scenarios = build_observation_scenarios(
        df,
        state_features,
        configs["data"]["nutrient_rates"],
        configs["optimization"]["scenario"]["scenario_id_column"],
        metadata_columns=["id", "exp.id", "year", "location"],
        max_scenarios=configs["optimization"]["scenario"].get("max_scenarios"),
        baseline_resolver=baseline_resolver,
    )

    bounds = bounds_from_decisions(configs["decisions"])
    objective_config = objective_config_from_configs(configs)
    de_config = configs["optimization"]["differential_evolution"]
    seeds = configs["optimization"].get("seeds", [42])
    warning_fraction = configs["decisions"]["agronomic_validation"].get(
        "boundary_warning_fraction"
    )

    result_rows, histories = [], []
    for scenario in scenarios:
        for seed in seeds:
            result, history = optimize_scenario(
                scenario, predictor, bounds, objective_config, de_config, int(seed),
                warning_fraction,
            )
            result_rows.append({**result.to_dict(), **scenario.metadata})
            if not history.empty:
                histories.append(history)
    results = pd.DataFrame(result_rows)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    output_dir = run_dir / "optimization"
    save_dataframe(results, output_dir / "optimization_results")
    if not history.empty:
        save_dataframe(history, output_dir / "convergence_history")
    print(f"Optimization results: {output_dir}")
    return run_dir, {
        "results": output_dir / "optimization_results.csv",
        "history": output_dir / "convergence_history.csv",
    }

if __name__ == "__main__":
    main()
