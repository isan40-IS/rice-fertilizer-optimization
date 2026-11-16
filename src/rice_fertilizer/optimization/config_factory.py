from __future__ import annotations
import numpy as np
from ..config import require_value
from .objective import ObjectiveConfiguration

def objective_config_from_configs(configs: dict) -> ObjectiveConfiguration:
    decisions = configs["decisions"]
    opt = decisions["optimization"]
    objective_settings = configs["optimization"].get("objective", {})
    normalize_objective = bool(
        objective_settings.get("normalize_yield_improvement", True)
    )
    yield_scale = (
        require_value(decisions, "optimization.yield_scale")
        if normalize_objective else 1.0
    )
    lambda_total = (
        require_value(decisions, "optimization.lambda_total")
        if opt.get("enable_total_penalty") else 0.0
    )
    lambda_composition = (
        require_value(decisions, "optimization.lambda_composition")
        if opt.get("enable_composition_penalty") else 0.0
    )
    total_soft = (
        require_value(decisions, "optimization.total_soft")
        if opt.get("enable_total_penalty") else None
    )
    total_max = (
        require_value(decisions, "optimization.total_max")
        if opt.get("enable_total_penalty") or opt.get("enable_total_hard_constraint")
        else None
    )
    minimum = maximum = None
    if opt.get("enable_composition_penalty"):
        minimum = np.array([
            require_value(decisions, f"optimization.proportion_bounds.{nutrient}.min")
            for nutrient in ("N", "P", "K")
        ], dtype=float)
        maximum = np.array([
            require_value(decisions, f"optimization.proportion_bounds.{nutrient}.max")
            for nutrient in ("N", "P", "K")
        ], dtype=float)
    return ObjectiveConfiguration(
        yield_scale=float(yield_scale),
        lambda_total=float(lambda_total),
        lambda_composition=float(lambda_composition),
        enable_total_penalty=bool(opt.get("enable_total_penalty")),
        enable_composition_penalty=bool(opt.get("enable_composition_penalty")),
        enable_total_hard_constraint=bool(opt.get("enable_total_hard_constraint")),
        total_soft=None if total_soft is None else float(total_soft),
        total_max=None if total_max is None else float(total_max),
        min_proportions=minimum,
        max_proportions=maximum,
        use_yield_improvement=bool(
            objective_settings.get("use_yield_improvement", True)
        ),
        normalize_objective=normalize_objective,
        infeasible_cost=float(objective_settings["infeasible_cost"]),
        epsilon=float(objective_settings.get("epsilon", 1e-8)),
    )
