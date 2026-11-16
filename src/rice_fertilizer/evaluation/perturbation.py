from __future__ import annotations
import pandas as pd
from ..schemas import OptimizationScenario

def perturbation_size(feature: str, specification: dict, reference_data: pd.DataFrame) -> float:
    value = specification.get("value")
    if value is None:
        raise ValueError(f"Perturbation value for '{feature}' is not decided")
    method = specification.get("method", "absolute")
    if method == "absolute":
        return float(value)
    series = pd.to_numeric(reference_data[feature], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Cannot derive perturbation for empty feature '{feature}'")
    scale = {
        "iqr_fraction": series.quantile(0.75) - series.quantile(0.25),
        "std_fraction": series.std(ddof=1),
        "range_fraction": series.max() - series.min(),
    }.get(method)
    if scale is None:
        raise ValueError(f"Unknown perturbation method: {method}")
    return float(value) * float(scale)

def perturb_scenario(
    scenario: OptimizationScenario, feature: str, delta: float, direction: int
) -> OptimizationScenario:
    if feature not in scenario.state_values:
        raise KeyError(f"Scenario does not contain '{feature}'")
    value = scenario.state_values[feature]
    if pd.isna(value):
        raise ValueError(f"Cannot perturb missing state feature '{feature}'")
    state = dict(scenario.state_values)
    state[feature] = float(value) + int(direction) * float(delta)
    return OptimizationScenario(
        f"{scenario.scenario_id}::{feature}::{direction:+d}",
        state,
        scenario.baseline_dose,
        dict(scenario.metadata),
    )

def generate_random_perturbations(
    scenario: OptimizationScenario,
    numeric_state_features: list[str],
    reference_data: pd.DataFrame,
    count: int,
    fraction_of_iqr: float,
    random_seed: int,
) -> list[OptimizationScenario]:
    import numpy as np
    rng = np.random.default_rng(random_seed)
    iqr = {
        feature: (
            pd.to_numeric(reference_data[feature], errors="coerce").quantile(0.75)
            - pd.to_numeric(reference_data[feature], errors="coerce").quantile(0.25)
        )
        for feature in numeric_state_features
    }
    scenarios = []
    for index in range(count):
        state = dict(scenario.state_values)
        for feature in numeric_state_features:
            if pd.isna(state.get(feature)) or pd.isna(iqr[feature]):
                continue
            state[feature] = float(state[feature]) + rng.uniform(
                -fraction_of_iqr * iqr[feature],
                fraction_of_iqr * iqr[feature],
            )
        scenarios.append(OptimizationScenario(
            f"{scenario.scenario_id}::random::{index}",
            state,
            scenario.baseline_dose,
            dict(scenario.metadata),
        ))
    return scenarios
