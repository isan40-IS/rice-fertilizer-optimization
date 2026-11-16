from __future__ import annotations
import numpy as np
import pandas as pd
from ..schemas import NutrientBounds, OptimizationScenario
from .perturbation import perturb_scenario

def normalized_sensitivity(
    x_minus: np.ndarray,
    x_plus: np.ndarray,
    delta: float,
    state_scale: float,
    bounds: NutrientBounds,
):
    raw = (np.asarray(x_plus) - np.asarray(x_minus)) / (2.0 * delta)
    normalized = state_scale * raw / (bounds.upper_array() - bounds.lower_array())
    return raw, normalized, float(np.mean(normalized ** 2))

def evaluate_local_sensitivity(
    scenario: OptimizationScenario,
    feature: str,
    delta: float,
    state_scale: float,
    optimizer_function,
    bounds: NutrientBounds,
    seeds: list[int],
) -> pd.DataFrame:
    minus = perturb_scenario(scenario, feature, delta, -1)
    plus = perturb_scenario(scenario, feature, delta, +1)
    rows = []
    for seed in seeds:
        minus_result, _ = optimizer_function(minus, seed)
        plus_result, _ = optimizer_function(plus, seed)
        x_minus = np.array([
            minus_result.optimal_n, minus_result.optimal_p, minus_result.optimal_k
        ])
        x_plus = np.array([
            plus_result.optimal_n, plus_result.optimal_p, plus_result.optimal_k
        ])
        raw, normalized, index = normalized_sensitivity(
            x_minus, x_plus, delta, state_scale, bounds
        )
        for nutrient, raw_value, normalized_value in zip(
            ("N", "P", "K"), raw, normalized
        ):
            rows.append({
                "scenario_id": scenario.scenario_id,
                "feature": feature,
                "seed": seed,
                "delta": delta,
                "nutrient": nutrient,
                "raw_sensitivity": float(raw_value),
                "normalized_sensitivity": float(normalized_value),
                "sensitivity_index": index,
            })
    return pd.DataFrame(rows)
