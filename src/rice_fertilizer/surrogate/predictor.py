from __future__ import annotations
import numpy as np
from ..schemas import OptimizationScenario
from .candidate_builder import build_candidate_dataframe

class SurrogatePredictor:
    def __init__(self, model_bundle: dict, nutrient_columns: dict[str, str]):
        self.bundle = model_bundle
        self.pipeline = model_bundle["pipeline"]
        self.feature_columns = list(model_bundle["feature_columns"])
        self.nutrient_columns = nutrient_columns
        self.ratio_pseudocount = float(model_bundle.get("ratio_pseudocount", 1e-8))
        self._configure_single_candidate_prediction()

    def _configure_single_candidate_prediction(self) -> None:
        """Avoid parallel-startup overhead during thousands of one-row predictions."""
        model = getattr(self.pipeline, "named_steps", {}).get("model")
        if model is None or not hasattr(model, "get_params"):
            return
        params = model.get_params(deep=False)
        updates = {}
        if "n_jobs" in params:
            updates["n_jobs"] = 1
        if "thread_count" in params:
            updates["thread_count"] = 1
        if updates:
            model.set_params(**updates)

    def candidate_frame(self, decision_vector, scenario: OptimizationScenario):
        return build_candidate_dataframe(
            decision_vector, scenario, self.feature_columns, self.nutrient_columns,
            self.ratio_pseudocount,
        )

    def predict_yield(self, decision_vector, scenario: OptimizationScenario) -> float:
        prediction = self.pipeline.predict(
            self.candidate_frame(decision_vector, scenario)
        )
        return float(np.asarray(prediction).ravel()[0])
