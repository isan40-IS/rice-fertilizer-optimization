from __future__ import annotations
import numpy as np
from ..schemas import FertilizerVector, NutrientBounds, OptimizationScenario

def average_efficiency(
    predicted_yield: float, dose: FertilizerVector, epsilon: float = 1e-8
) -> float:
    return float(predicted_yield / (dose.total() + epsilon))

def total_marginal_efficiency(
    optimized_yield: float,
    baseline_yield: float,
    optimized_dose: FertilizerVector,
    baseline_dose: FertilizerVector,
    epsilon: float = 1e-8,
) -> dict:
    delta_yield = float(optimized_yield - baseline_yield)
    delta_dose = float(optimized_dose.total() - baseline_dose.total())
    if abs(delta_dose) <= epsilon:
        value, status = np.nan, "no_dose_change"
    else:
        value = float(delta_yield / delta_dose)
        if delta_yield >= 0 and delta_dose > 0:
            status = "increased_yield_increased_dose"
        elif delta_yield >= 0 and delta_dose < 0:
            status = "increased_yield_reduced_dose"
        elif delta_yield < 0 and delta_dose > 0:
            status = "decreased_yield_increased_dose"
        else:
            status = "decreased_yield_reduced_dose"
    return {
        "delta_yield": delta_yield,
        "delta_total_npk": delta_dose,
        "marginal_efficiency": value,
        "status": status,
    }

def nutrient_marginal_efficiency(
    predictor,
    scenario: OptimizationScenario,
    optimum: FertilizerVector,
    nutrient_index: int,
    step_size: float,
    bounds: NutrientBounds,
) -> dict:
    base = optimum.as_array()
    lower, upper = bounds.lower_array(), bounds.upper_array()
    base_yield = predictor.predict_yield(base, scenario)
    if base[nutrient_index] + step_size <= upper[nutrient_index]:
        shifted = base.copy()
        shifted[nutrient_index] += step_size
        value = (predictor.predict_yield(shifted, scenario) - base_yield) / step_size
        direction = "forward"
    elif base[nutrient_index] - step_size >= lower[nutrient_index]:
        shifted = base.copy()
        shifted[nutrient_index] -= step_size
        value = (base_yield - predictor.predict_yield(shifted, scenario)) / step_size
        direction = "backward"
    else:
        return {"value": np.nan, "direction": "unavailable", "step": step_size}
    return {"value": float(value), "direction": direction, "step": float(step_size)}

def resolve_marginal_step(
    nutrient_index: int,
    bounds: NutrientBounds,
    method: str,
    value: float,
    minimum_step: float,
) -> float:
    if method == "absolute":
        return max(float(value), float(minimum_step))
    if method == "range_fraction":
        ranges = bounds.upper_array() - bounds.lower_array()
        return max(float(value) * float(ranges[nutrient_index]), float(minimum_step))
    raise ValueError(f"Unknown marginal step method: {method}")
