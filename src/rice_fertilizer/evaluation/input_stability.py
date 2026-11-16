import numpy as np
from ..schemas import NutrientBounds

def normalized_recommendation_shift(
    original: np.ndarray, perturbed: np.ndarray, bounds: NutrientBounds
) -> float:
    ranges = bounds.upper_array() - bounds.lower_array()
    return float(np.mean(((np.asarray(perturbed) - np.asarray(original)) / ranges) ** 2))

def summarize_input_stability(shifts: list[float]) -> dict:
    values = np.asarray(shifts, dtype=float)
    return {
        "input_stability_mean": float(np.mean(values)),
        "input_stability_median": float(np.median(values)),
        "input_stability_max": float(np.max(values)),
        "input_stability_iqr": float(
            np.quantile(values, 0.75) - np.quantile(values, 0.25)
        ),
    }
