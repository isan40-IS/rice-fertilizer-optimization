from __future__ import annotations
import numpy as np
import pandas as pd
from ..data.feature_engineering import add_npk_features
from ..schemas import OptimizationScenario

def build_candidate_dataframe(
    decision_vector: np.ndarray,
    scenario: OptimizationScenario,
    feature_columns: list[str],
    nutrient_columns: dict[str, str],
    ratio_pseudocount: float = 1e-8,
) -> pd.DataFrame:
    decision_vector = np.asarray(decision_vector, dtype=float)
    if decision_vector.shape != (3,):
        raise ValueError("Decision vector must be [N, P, K]")
    row = dict(scenario.state_values)
    row[nutrient_columns["N"]] = float(decision_vector[0])
    row[nutrient_columns["P"]] = float(decision_vector[1])
    row[nutrient_columns["K"]] = float(decision_vector[2])
    frame = add_npk_features(
        pd.DataFrame([row]),
        nutrient_columns["N"],
        nutrient_columns["P"],
        nutrient_columns["K"],
        ratio_pseudocount=ratio_pseudocount,
    )
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Candidate cannot reproduce trained feature schema: {missing}")
    return frame[feature_columns]
