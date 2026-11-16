from __future__ import annotations
import numpy as np
import pandas as pd
from .common import initialize_run
from ..config import require_value, resolve_path
from ..evaluation.agronomic_validation import validate_solution
from ..evaluation.algorithm_stability import summarize_algorithm_stability
from ..evaluation.baseline import resolve_baseline
from ..evaluation.fertilizer_efficiency import (
    average_efficiency,
    nutrient_marginal_efficiency,
    resolve_marginal_step,
    total_marginal_efficiency,
)
from ..evaluation.input_stability import (
    normalized_recommendation_shift,
    summarize_input_stability,
)
from ..evaluation.perturbation import (
    generate_random_perturbations,
    perturbation_size,
)
from ..evaluation.sensitivity import evaluate_local_sensitivity
from ..modeling.serializer import load_model_bundle
from ..optimization.bounds import bounds_from_decisions
from ..optimization.config_factory import objective_config_from_configs
from ..optimization.optimizer import optimize_scenario
from ..optimization.penalties import nutrient_proportions
from ..reporting.export import export_excel
from ..reporting.plots import (
    save_baseline_vs_optimal_doses,
    save_yield_improvement_histogram,
)
from ..reporting.tables import optimization_summary, sensitivity_summary
from ..schemas import FertilizerVector
from ..surrogate.predictor import SurrogatePredictor
from ..surrogate.scenario import build_observation_scenarios
from ..utils import configure_logging, save_dataframe, save_json
from ..validation.domain_checks import NumericDomainSupport

def main(
    run_id: str | None = None,
    model_bundle_path: str | None = None,
    optimization_results_path: str | None = None,
):
    configure_logging()
    configs, run_dir = initialize_run(run_id=run_id)
    if not model_bundle_path or not optimization_results_path:
        raise FileNotFoundError(
            "Provide both --model-bundle and --optimization-results"
        )
    df = pd.read_csv(resolve_path(configs["paths"]["processed_dataset"]))
    bundle = load_model_bundle(model_bundle_path)
    results = pd.read_csv(optimization_results_path)
    predictor = SurrogatePredictor(bundle, configs["data"]["nutrient_rates"])
    bounds = bounds_from_decisions(configs["decisions"])
    objective_config = objective_config_from_configs(configs)
    de_config = configs["optimization"]["differential_evolution"]
    warning_fraction = configs["decisions"]["agronomic_validation"].get(
        "boundary_warning_fraction"
    )

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
        baseline_resolver=baseline_resolver,
    )
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}

    domain_support = None
    if bundle.get("training_reference_numeric") is not None:
        domain_support = NumericDomainSupport(bundle["numeric_features"]).fit(
            bundle["training_reference_numeric"]
        )

    marginal_cfg = configs["evaluation"]["marginal_step"]
    proportion_bounds = configs["decisions"]["optimization"]["proportion_bounds"]
    min_props = np.array([
        proportion_bounds[n]["min"] for n in ("N", "P", "K")
    ], dtype=float) if all(
        proportion_bounds[n]["min"] is not None for n in ("N", "P", "K")
    ) else None
    max_props = np.array([
        proportion_bounds[n]["max"] for n in ("N", "P", "K")
    ], dtype=float) if all(
        proportion_bounds[n]["max"] is not None for n in ("N", "P", "K")
    ) else None
    max_training_distance = configs["decisions"]["agronomic_validation"].get(
        "max_training_distance"
    )

    evaluated_rows = []
    for _, row in results.iterrows():
        scenario = scenario_map.get(str(row["scenario_id"]))
        if scenario is None:
            continue
        optimum = FertilizerVector(
            float(row["optimal_n"]), float(row["optimal_p"]), float(row["optimal_k"])
        )
        baseline = FertilizerVector(
            float(row["baseline_n"]), float(row["baseline_p"]), float(row["baseline_k"])
        )
        total_eff = total_marginal_efficiency(
            float(row["optimal_yield"]),
            float(row["baseline_yield"]),
            optimum,
            baseline,
        )
        nutrient_eff = {}
        for index, nutrient in enumerate(("N", "P", "K")):
            step = resolve_marginal_step(
                index, bounds,
                marginal_cfg["method"],
                marginal_cfg["value"],
                marginal_cfg["minimum_step"],
            )
            nutrient_eff[nutrient] = nutrient_marginal_efficiency(
                predictor, scenario, optimum, index, step, bounds
            )
        candidate = predictor.candidate_frame(optimum.as_array(), scenario)
        training_distance = (
            domain_support.distance(candidate) if domain_support is not None else None
        )
        proportions = nutrient_proportions(optimum.as_array())
        validation = validate_solution(
            optimum.as_array(), bounds, proportions,
            min_props, max_props, warning_fraction,
            training_distance, max_training_distance,
        )
        evaluated_rows.append({
            **row.to_dict(),
            "average_efficiency": average_efficiency(
                float(row["optimal_yield"]), optimum
            ),
            "delta_total_npk": total_eff["delta_total_npk"],
            "total_marginal_efficiency": total_eff["marginal_efficiency"],
            "total_marginal_status": total_eff["status"],
            "marginal_efficiency_n": nutrient_eff["N"]["value"],
            "marginal_efficiency_p": nutrient_eff["P"]["value"],
            "marginal_efficiency_k": nutrient_eff["K"]["value"],
            "marginal_direction_n": nutrient_eff["N"]["direction"],
            "marginal_direction_p": nutrient_eff["P"]["direction"],
            "marginal_direction_k": nutrient_eff["K"]["direction"],
            "prop_n": proportions[0],
            "prop_p": proportions[1],
            "prop_k": proportions[2],
            "training_domain_distance": training_distance,
            "agronomic_status": validation["status"],
            "agronomic_warnings": "|".join(validation["warnings"]),
        })
    evaluated = pd.DataFrame(evaluated_rows)

    evaluation_dir = run_dir / "evaluation"
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    save_dataframe(evaluated, evaluation_dir / "evaluated_optimization_results")
    summary = optimization_summary(evaluated)
    save_dataframe(summary, tables_dir / "optimization_summary", save_parquet=False)

    algorithm_rows = []
    if configs["evaluation"]["algorithm_stability"].get("enabled", True):
        for scenario_id, group in evaluated.groupby("scenario_id"):
            if len(group) < 2:
                continue
            algorithm_rows.append({
                "scenario_id": scenario_id,
                **summarize_algorithm_stability(group),
            })
    algorithm_summary = pd.DataFrame(algorithm_rows)
    if not algorithm_summary.empty:
        save_dataframe(
            algorithm_summary,
            evaluation_dir / "algorithm_stability_summary",
            save_parquet=False,
        )

    notes = {}
    sensitivity_results = pd.DataFrame()
    sensitivity_cfg = configs["evaluation"]["sensitivity"]
    active_perturbations = {
        feature: spec
        for feature, spec in configs["decisions"].get("perturbations", {}).items()
        if spec.get("value") is not None and feature in state_features
    }
    if sensitivity_cfg.get("enabled", True) and active_perturbations:
        sensitivity_frames = []
        limited = scenarios[: int(sensitivity_cfg.get("scenario_limit", 10))]
        optimizer_function = lambda scenario, seed: optimize_scenario(
            scenario, predictor, bounds, objective_config, de_config, int(seed),
            warning_fraction,
        )
        for scenario in limited:
            for feature, spec in active_perturbations.items():
                if pd.isna(scenario.state_values.get(feature)):
                    continue
                delta = perturbation_size(feature, spec, df)
                state_series = pd.to_numeric(df[feature], errors="coerce")
                state_scale = float(
                    state_series.quantile(0.75) - state_series.quantile(0.25)
                )
                sensitivity_frames.append(evaluate_local_sensitivity(
                    scenario, feature, delta, state_scale,
                    optimizer_function, bounds,
                    sensitivity_cfg.get("common_seeds", [11, 29, 47]),
                ))
        if sensitivity_frames:
            sensitivity_results = pd.concat(sensitivity_frames, ignore_index=True)
            save_dataframe(
                sensitivity_results,
                evaluation_dir / "sensitivity_results",
            )
            save_dataframe(
                sensitivity_summary(sensitivity_results),
                tables_dir / "sensitivity_summary",
                save_parquet=False,
            )
    else:
        notes["sensitivity"] = (
            "Skipped because no non-null perturbation values were configured."
        )

    input_stability_summary = pd.DataFrame()
    input_cfg = configs["evaluation"]["input_stability"]
    fraction = input_cfg.get("fraction_of_iqr")
    if input_cfg.get("enabled", True) and fraction is not None:
        stability_rows = []
        numeric_state_features = [
            feature for feature in state_features
            if feature in bundle["numeric_features"]
        ]
        limited = scenarios[: int(input_cfg.get("scenario_limit", 10))]
        for scenario in limited:
            base_group = evaluated.loc[
                evaluated["scenario_id"].astype(str) == scenario.scenario_id
            ]
            if base_group.empty:
                continue
            base_row = base_group.sort_values(
                "optimization_score", ascending=False
            ).iloc[0]
            original = np.array([
                base_row["optimal_n"], base_row["optimal_p"], base_row["optimal_k"]
            ])
            shifts = []
            perturbed_scenarios = generate_random_perturbations(
                scenario, numeric_state_features, df,
                int(input_cfg.get("random_perturbations", 20)),
                float(fraction),
                int(input_cfg.get("random_seed", 42)),
            )
            seed = int(configs["evaluation"]["sensitivity"].get(
                "common_seeds", [11]
            )[0])
            for perturbed in perturbed_scenarios:
                perturbed_result, _ = optimize_scenario(
                    perturbed, predictor, bounds, objective_config,
                    de_config, seed, warning_fraction,
                )
                perturbed_vector = np.array([
                    perturbed_result.optimal_n,
                    perturbed_result.optimal_p,
                    perturbed_result.optimal_k,
                ])
                shift = normalized_recommendation_shift(
                    original, perturbed_vector, bounds
                )
                shifts.append(shift)
                stability_rows.append({
                    "scenario_id": scenario.scenario_id,
                    "perturbation_id": perturbed.scenario_id,
                    "normalized_shift": shift,
                    "optimal_n": perturbed_result.optimal_n,
                    "optimal_p": perturbed_result.optimal_p,
                    "optimal_k": perturbed_result.optimal_k,
                })
            if shifts:
                stability_summary = summarize_input_stability(shifts)
                threshold = configs["decisions"]["agronomic_validation"].get(
                    "max_input_stability_index"
                )
                if threshold is None:
                    stability_status = "threshold_not_configured"
                else:
                    stability_status = (
                        "stable"
                        if stability_summary["input_stability_mean"] <= float(threshold)
                        else "warning_unstable"
                    )
                stability_rows.append({
                    "scenario_id": scenario.scenario_id,
                    "perturbation_id": "__SUMMARY__",
                    **stability_summary,
                    "input_stability_status": stability_status,
                })
        input_stability_summary = pd.DataFrame(stability_rows)
        if not input_stability_summary.empty:
            save_dataframe(
                input_stability_summary,
                evaluation_dir / "input_stability",
            )
    else:
        notes["input_stability"] = (
            "Skipped because input_stability.fraction_of_iqr is null."
        )

    if not evaluated.empty:
        save_yield_improvement_histogram(
            evaluated, figures_dir / "yield_improvement_histogram.png"
        )
        save_baseline_vs_optimal_doses(
            evaluated, figures_dir / "baseline_vs_optimal_doses.png"
        )

    sheets = {
        "evaluated_results": evaluated,
        "optimization_summary": summary,
        "algorithm_stability": algorithm_summary,
        "sensitivity": sensitivity_results,
        "input_stability": input_stability_summary,
    }
    export_excel(
        {name: frame for name, frame in sheets.items() if not frame.empty},
        run_dir / "reports" / "evaluation_outputs.xlsx",
    )
    save_json(notes, evaluation_dir / "evaluation_notes.json")
    print(f"Evaluation outputs: {evaluation_dir}")
    return run_dir, {"evaluated_results": evaluation_dir / "evaluated_optimization_results.csv"}

if __name__ == "__main__":
    main()
