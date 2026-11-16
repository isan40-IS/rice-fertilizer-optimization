from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..exceptions import DataValidationError

def load_csv(path: str | Path, read_config: dict) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}. Put your data at data/input/data.csv or update configs/paths.yaml."
        )
    df = pd.read_csv(
        path,
        sep=read_config.get("sep", ","),
        encoding=read_config.get("encoding", "utf-8"),
        low_memory=read_config.get("low_memory", False),
        na_values=read_config.get("missing_tokens", []),
        keep_default_na=True,
    )
    if df.empty:
        raise DataValidationError("The input CSV contains no rows.")
    return df

def add_record_id(df: pd.DataFrame, column: str = "record_id") -> pd.DataFrame:
    result = df.copy()
    if column not in result.columns:
        result.insert(0, column, range(len(result)))
    if result[column].duplicated().any():
        raise DataValidationError(f"Record identifier '{column}' is not unique.")
    return result

def build_composite_column(
    df: pd.DataFrame, output_column: str, source_columns: list[str], separator: str = "::"
) -> pd.DataFrame:
    missing = [col for col in source_columns if col not in df.columns]
    if missing:
        raise DataValidationError(f"Cannot build {output_column}; missing columns: {missing}")
    result = df.copy()
    result[output_column] = (
        result[source_columns].astype("string").fillna("<MISSING>").agg(separator.join, axis=1)
    )
    return result
