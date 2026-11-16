import numpy as np
from ..schemas import NutrientBounds

def validate_solution(
    x: np.ndarray,
    bounds: NutrientBounds,
    proportions: np.ndarray,
    min_proportions: np.ndarray | None,
    max_proportions: np.ndarray | None,
    boundary_warning_fraction: float | None,
    training_distance: float | None = None,
    max_training_distance: float | None = None,
) -> dict:
    x = np.asarray(x, dtype=float)
    warnings = []
    if boundary_warning_fraction is not None:
        lower, upper = bounds.lower_array(), bounds.upper_array()
        margin = (upper - lower) * float(boundary_warning_fraction)
        if np.any(x <= lower + margin) or np.any(x >= upper - margin):
            warnings.append("near_decision_boundary")
    if min_proportions is not None and max_proportions is not None:
        if np.any(proportions < min_proportions) or np.any(proportions > max_proportions):
            warnings.append("extreme_nutrient_composition")
    if (
        training_distance is not None
        and max_training_distance is not None
        and training_distance > max_training_distance
    ):
        warnings.append("low_training_domain_support")
    return {"status": "acceptable" if not warnings else "warning", "warnings": warnings}
