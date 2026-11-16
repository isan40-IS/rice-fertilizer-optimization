from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..schemas import ObjectiveComponents, OptimizationScenario
from .constraints import total_dose_feasible
from .penalties import total_dose_penalty, extreme_composition_penalty

@dataclass
class ObjectiveConfiguration:
    yield_scale: float
    lambda_total: float
    lambda_composition: float
    enable_total_penalty: bool
    enable_composition_penalty: bool
    enable_total_hard_constraint: bool
    total_soft: float | None
    total_max: float | None
    min_proportions: np.ndarray | None
    max_proportions: np.ndarray | None
    use_yield_improvement: bool = True
    normalize_objective: bool = True
    infeasible_cost: float = 1e12
    epsilon: float = 1e-8

class SurrogateObjective:
    def __init__(self, predictor, scenario: OptimizationScenario, config: ObjectiveConfiguration):
        self.predictor = predictor
        self.scenario = scenario
        self.config = config
        self.baseline_yield = predictor.predict_yield(
            scenario.baseline_dose.as_array(), scenario
        )

    def components(self, x: np.ndarray) -> ObjectiveComponents:
        x = np.asarray(x, dtype=float)
        feasible = total_dose_feasible(x, self.config.total_max)
        if self.config.enable_total_hard_constraint and not feasible:
            return ObjectiveComponents(
                float("nan"), self.baseline_yield, float("nan"), float("nan"),
                float("nan"), float("nan"), 0.0, 0.0,
                -self.config.infeasible_cost, False,
            )
        predicted = self.predictor.predict_yield(x, self.scenario)
        improvement = predicted - self.baseline_yield
        normalized_improvement = (
            improvement / (self.config.yield_scale + self.config.epsilon)
        )
        objective_value = (
            improvement if self.config.use_yield_improvement else predicted
        )
        scaled_objective = (
            objective_value / (self.config.yield_scale + self.config.epsilon)
            if self.config.normalize_objective else objective_value
        )

        total_pen = 0.0
        if self.config.enable_total_penalty:
            if self.config.total_soft is None or self.config.total_max is None:
                raise ValueError("Total penalty enabled but thresholds are missing")
            total_pen = total_dose_penalty(
                x, self.config.total_soft, self.config.total_max, self.config.epsilon
            )

        composition_pen = 0.0
        if self.config.enable_composition_penalty:
            if self.config.min_proportions is None or self.config.max_proportions is None:
                raise ValueError("Composition bounds are missing")
            composition_pen = extreme_composition_penalty(
                x, self.config.min_proportions, self.config.max_proportions,
                self.config.epsilon,
            )

        score = (
            scaled_objective
            - self.config.lambda_total * total_pen
            - self.config.lambda_composition * composition_pen
        )
        return ObjectiveComponents(
            predicted, self.baseline_yield, improvement, normalized_improvement,
            objective_value, scaled_objective, total_pen, composition_pen,
            float(score), feasible,
        )

    def scipy_cost(self, x: np.ndarray) -> float:
        return float(-self.components(x).score)
