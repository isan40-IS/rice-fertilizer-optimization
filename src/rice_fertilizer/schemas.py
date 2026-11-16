from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np

@dataclass(frozen=True)
class FertilizerVector:
    nitrogen: float
    phosphorus: float
    potassium: float
    def as_array(self) -> np.ndarray:
        return np.array([self.nitrogen, self.phosphorus, self.potassium], dtype=float)
    def total(self) -> float:
        return float(self.nitrogen + self.phosphorus + self.potassium)

@dataclass(frozen=True)
class NutrientBounds:
    n_min: float
    n_max: float
    p_min: float
    p_max: float
    k_min: float
    k_max: float
    def as_scipy_bounds(self) -> list[tuple[float, float]]:
        return [(self.n_min, self.n_max), (self.p_min, self.p_max), (self.k_min, self.k_max)]
    def lower_array(self) -> np.ndarray:
        return np.array([self.n_min, self.p_min, self.k_min], dtype=float)
    def upper_array(self) -> np.ndarray:
        return np.array([self.n_max, self.p_max, self.k_max], dtype=float)

@dataclass(frozen=True)
class OptimizationScenario:
    scenario_id: str
    state_values: dict[str, Any]
    baseline_dose: FertilizerVector
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ObjectiveComponents:
    predicted_yield: float
    baseline_yield: float
    yield_improvement: float
    normalized_yield_improvement: float
    objective_value: float
    scaled_objective_value: float
    total_penalty: float
    composition_penalty: float
    score: float
    feasible: bool

@dataclass
class OptimizationResult:
    scenario_id: str
    seed: int
    baseline_n: float
    baseline_p: float
    baseline_k: float
    baseline_yield: float
    optimal_n: float
    optimal_p: float
    optimal_k: float
    optimal_yield: float
    yield_improvement: float
    total_penalty: float
    composition_penalty: float
    optimization_score: float
    success: bool
    convergence_message: str
    iterations: int
    evaluations: int
    runtime_seconds: float
    domain_status: str = "not_checked"
    agronomic_status: str = "not_checked"
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
