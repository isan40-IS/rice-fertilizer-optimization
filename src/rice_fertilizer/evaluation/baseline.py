from __future__ import annotations
import pandas as pd
from ..schemas import FertilizerVector
from ..exceptions import ConfigurationError

def observed_baseline(row: pd.Series, nutrient_columns: dict[str, str]) -> FertilizerVector:
    return FertilizerVector(
        float(row[nutrient_columns["N"]]),
        float(row[nutrient_columns["P"]]),
        float(row[nutrient_columns["K"]]),
    )

def zero_baseline(row: pd.Series, nutrient_columns: dict[str, str]) -> FertilizerVector:
    return FertilizerVector(0.0, 0.0, 0.0)

def group_mean_baseline(
    row: pd.Series,
    context: pd.DataFrame,
    nutrient_columns: dict[str, str],
    group_column: str,
) -> FertilizerVector:
    group = context.loc[context[group_column] == row[group_column]]
    return FertilizerVector(*[
        float(pd.to_numeric(group[nutrient_columns[n]], errors="coerce").mean())
        for n in ("N", "P", "K")
    ])

def historical_best_baseline(
    row: pd.Series,
    context: pd.DataFrame,
    nutrient_columns: dict[str, str],
    group_column: str,
    target_column: str,
) -> FertilizerVector:
    group = context.loc[context[group_column] == row[group_column]].copy()
    if group.empty:
        raise ConfigurationError("Historical-best group is empty")
    best = group.loc[pd.to_numeric(group[target_column], errors="coerce").idxmax()]
    return observed_baseline(best, nutrient_columns)

def resolve_baseline(
    policy: str,
    row: pd.Series,
    context: pd.DataFrame,
    nutrient_columns: dict[str, str],
    group_column: str | None = None,
    target_column: str | None = None,
) -> FertilizerVector:
    if policy == "observed":
        return observed_baseline(row, nutrient_columns)
    if policy == "zero":
        return zero_baseline(row, nutrient_columns)
    if policy == "group_mean":
        if not group_column:
            raise ConfigurationError("group_mean baseline requires group_column")
        return group_mean_baseline(row, context, nutrient_columns, group_column)
    if policy == "historical_best":
        if not group_column or not target_column:
            raise ConfigurationError(
                "historical_best requires group_column and target_column"
            )
        return historical_best_baseline(
            row, context, nutrient_columns, group_column, target_column
        )
    raise ConfigurationError(f"Unknown baseline policy: {policy}")
