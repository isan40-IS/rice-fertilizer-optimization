import numpy as np
from ..schemas import NutrientBounds

def check_bounds(x: np.ndarray, bounds: NutrientBounds) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(np.all(x >= bounds.lower_array()) and np.all(x <= bounds.upper_array()))

def boundary_status(
    x: np.ndarray,
    bounds: NutrientBounds,
    warning_fraction: float | None,
) -> str:
    if warning_fraction is None:
        return "not_configured"
    x = np.asarray(x, dtype=float)
    lower, upper = bounds.lower_array(), bounds.upper_array()
    margin = (upper - lower) * float(warning_fraction)
    if np.any(x < lower + margin) or np.any(x > upper - margin):
        return "near_boundary"
    return "inside_bounds"
