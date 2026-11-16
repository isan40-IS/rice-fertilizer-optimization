import numpy as np
from rice_fertilizer.evaluation.sensitivity import normalized_sensitivity
from rice_fertilizer.schemas import NutrientBounds

def test_normalized_sensitivity():
    bounds = NutrientBounds(0, 200, 0, 100, 0, 100)
    raw, normalized, index = normalized_sensitivity(
        np.array([50, 20, 20]),
        np.array([60, 25, 15]),
        1.0, 2.0, bounds,
    )
    assert raw.shape == (3,)
    assert normalized.shape == (3,)
    assert index >= 0
