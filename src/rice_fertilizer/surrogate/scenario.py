from __future__ import annotations
from collections.abc import Callable
import pandas as pd
from ..schemas import FertilizerVector, OptimizationScenario
from ..exceptions import ConfigurationError

def build_observation_scenarios(
    df: pd.DataFrame,
    state_features: list[str],
    nutrient_columns: dict[str, str],
    scenario_id_column: str = "record_id",
    metadata_columns: list[str] | None = None,
    max_scenarios: int | None = None,
    baseline_resolver: Callable[[pd.Series], FertilizerVector] | None = None,
) -> list[OptimizationScenario]:
    metadata_columns = metadata_columns or []
    scenarios = []
    source = df.head(max_scenarios) if max_scenarios else df
    for row_index, row in source.iterrows():
        if baseline_resolver is None:
            values = [row[nutrient_columns[key]] for key in ("N", "P", "K")]
            if any(pd.isna(value) for value in values):
                continue
            baseline = FertilizerVector(*map(float, values))
        else:
            try:
                baseline = baseline_resolver(row)
            except (ValueError, TypeError):
                continue
            if any(pd.isna(value) for value in baseline.as_array()):
                continue
        scenario_id = str(row.get(scenario_id_column, row_index))
        scenarios.append(OptimizationScenario(
            scenario_id=scenario_id,
            state_values={feature: row.get(feature) for feature in state_features},
            baseline_dose=baseline,
            metadata={
                column: row.get(column)
                for column in metadata_columns if column in row.index
            },
        ))
    if not scenarios:
        raise ConfigurationError(
            "No valid scenarios could be built. Check baseline policy and NPK values."
        )
    return scenarios
