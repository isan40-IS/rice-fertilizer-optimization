import numpy as np
from rice_fertilizer.optimization.penalties import (
    extreme_composition_penalty,
    nutrient_proportions,
    total_dose_penalty,
)

def test_total_penalty_zero_below_soft_threshold():
    assert total_dose_penalty(np.array([50, 20, 20]), 100, 200) == 0.0

def test_total_penalty_reaches_one_at_maximum():
    assert abs(total_dose_penalty(np.array([100, 50, 50]), 100, 200) - 1.0) < 1e-6

def test_composition_penalty_zero_inside_range():
    value = extreme_composition_penalty(
        np.array([60.0, 20.0, 20.0]),
        np.array([0.4, 0.1, 0.1]),
        np.array([0.8, 0.4, 0.4]),
    )
    assert value < 1e-12

def test_zero_vector_proportions():
    assert np.allclose(nutrient_proportions(np.zeros(3)), np.zeros(3))
