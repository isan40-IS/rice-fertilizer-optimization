import numpy as np

def total_dose_penalty(
    x: np.ndarray, total_soft: float, total_max: float, epsilon: float = 1e-8
) -> float:
    if total_max < total_soft:
        raise ValueError("total_max must be >= total_soft")
    total = float(np.asarray(x, dtype=float).sum())
    violation = max(0.0, (total - total_soft) / (total_max - total_soft + epsilon))
    return float(violation ** 2)

def nutrient_proportions(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = float(x.sum())
    return np.zeros(3) if abs(total) <= epsilon else x / (total + epsilon)

def extreme_composition_penalty(
    x: np.ndarray,
    min_proportions: np.ndarray,
    max_proportions: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    proportions = nutrient_proportions(x, epsilon)
    minimum = np.asarray(min_proportions, dtype=float)
    maximum = np.asarray(max_proportions, dtype=float)
    if np.any(minimum > maximum):
        raise ValueError("Minimum proportion cannot exceed maximum")
    lower = np.maximum(0.0, minimum - proportions)
    upper = np.maximum(0.0, proportions - maximum)
    return float(np.sum(lower**2 + upper**2))
