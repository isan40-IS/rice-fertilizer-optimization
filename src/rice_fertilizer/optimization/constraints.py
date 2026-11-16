import numpy as np

def total_dose(x: np.ndarray) -> float:
    return float(np.asarray(x, dtype=float).sum())

def total_dose_feasible(x: np.ndarray, total_max: float | None) -> bool:
    return True if total_max is None else total_dose(x) <= float(total_max)
