import numpy as np

from rice_fertilizer.optimization.objective import (
    ObjectiveConfiguration, SurrogateObjective,
)
from rice_fertilizer.schemas import FertilizerVector, OptimizationScenario


class DummyPredictor:
    def predict_yield(self, decision_vector, scenario):
        return 100.0 + float(np.sum(decision_vector))


def make_config(**overrides):
    values = dict(
        yield_scale=10.0,
        lambda_total=0.0,
        lambda_composition=0.0,
        enable_total_penalty=False,
        enable_composition_penalty=False,
        enable_total_hard_constraint=False,
        total_soft=None,
        total_max=None,
        min_proportions=None,
        max_proportions=None,
    )
    values.update(overrides)
    return ObjectiveConfiguration(**values)


def test_objective_uses_scaled_yield_improvement():
    scenario = OptimizationScenario("s", {}, FertilizerVector(1, 1, 1))
    objective = SurrogateObjective(DummyPredictor(), scenario, make_config())
    components = objective.components(np.array([2.0, 2.0, 2.0]))
    assert components.yield_improvement == 3.0
    assert abs(components.score - 0.3) < 1e-8


def test_objective_can_disable_normalization_for_diagnostics():
    scenario = OptimizationScenario("s", {}, FertilizerVector(1, 1, 1))
    objective = SurrogateObjective(
        DummyPredictor(), scenario,
        make_config(normalize_objective=False),
    )
    components = objective.components(np.array([2.0, 2.0, 2.0]))
    assert components.score == 3.0
