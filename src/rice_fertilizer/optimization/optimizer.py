from time import perf_counter
import pandas as pd
from ..schemas import NutrientBounds, OptimizationResult, OptimizationScenario
from ..surrogate.domain_guard import boundary_status
from .convergence import ConvergenceTracker
from .differential_evolution import run_de
from .objective import ObjectiveConfiguration, SurrogateObjective

def optimize_scenario(
    scenario: OptimizationScenario,
    predictor,
    bounds: NutrientBounds,
    objective_config: ObjectiveConfiguration,
    de_config: dict,
    seed: int,
    boundary_warning_fraction: float | None = None,
):
    objective = SurrogateObjective(predictor, scenario, objective_config)
    tracker = ConvergenceTracker(objective)
    started = perf_counter()
    result = run_de(
        objective.scipy_cost, bounds.as_scipy_bounds(), de_config, seed, tracker.callback
    )
    runtime = perf_counter() - started
    components = objective.components(result.x)
    record = OptimizationResult(
        scenario.scenario_id, int(seed),
        scenario.baseline_dose.nitrogen,
        scenario.baseline_dose.phosphorus,
        scenario.baseline_dose.potassium,
        components.baseline_yield,
        float(result.x[0]), float(result.x[1]), float(result.x[2]),
        components.predicted_yield,
        components.yield_improvement,
        components.total_penalty,
        components.composition_penalty,
        components.score,
        bool(result.success),
        str(result.message),
        int(result.nit),
        int(result.nfev),
        float(runtime),
        boundary_status(result.x, bounds, boundary_warning_fraction),
    )
    history = pd.DataFrame(tracker.history)
    if not history.empty:
        history.insert(0, "scenario_id", scenario.scenario_id)
        history.insert(1, "seed", seed)
    return record, history
