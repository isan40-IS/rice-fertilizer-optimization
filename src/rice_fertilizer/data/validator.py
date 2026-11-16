from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from ..exceptions import DataValidationError

def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required source columns: {missing}")

def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in result.columns:
            rows.append({"column": column, "status": "absent", "new_missing": None})
            continue
        before = int(result[column].isna().sum())
        result[column] = (
            result[column]
            .astype("string")
            .str.replace(",", ".", regex=False)
        )
        result[column] = pd.to_numeric(result[column], errors="coerce")
        rows.append({
            "column": column,
            "status": "coerced",
            "new_missing": int(result[column].isna().sum()) - before,
        })
    return result, pd.DataFrame(rows)

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "non_missing": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "unique": int(series.nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"])

def find_numeric_anomalies(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        valid = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append({
            "column": column,
            "count": int(valid.size),
            "negative_count": int((valid < 0).sum()),
            "zero_count": int((valid == 0).sum()),
            "min": float(valid.min()) if not valid.empty else np.nan,
            "q01": float(valid.quantile(0.01)) if not valid.empty else np.nan,
            "q25": float(valid.quantile(0.25)) if not valid.empty else np.nan,
            "median": float(valid.median()) if not valid.empty else np.nan,
            "q75": float(valid.quantile(0.75)) if not valid.empty else np.nan,
            "q99": float(valid.quantile(0.99)) if not valid.empty else np.nan,
            "max": float(valid.max()) if not valid.empty else np.nan,
        })
    return pd.DataFrame(rows)

def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict[str, int]:
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "subset_duplicate_rows": int(df.duplicated(subset=subset).sum()) if subset else 0,
    }

def validate_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise DataValidationError(f"Target column not found: {target}")
    valid = pd.to_numeric(df[target], errors="coerce").dropna()
    if valid.empty:
        raise DataValidationError(f"Target column '{target}' has no usable numeric values.")
    if (valid < 0).any():
        raise DataValidationError(f"Target column '{target}' contains negative values.")
