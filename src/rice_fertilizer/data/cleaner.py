from __future__ import annotations
import pandas as pd

def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    count = int(df.duplicated().sum())
    return df.drop_duplicates().copy(), count

def standardize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip().replace({"": pd.NA})
    return result

def filter_valid_target_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    mask = pd.to_numeric(df[target], errors="coerce").notna()
    return df.loc[mask].copy(), int((~mask).sum())

def add_iqr_outlier_flags(df: pd.DataFrame, columns: list[str], multiplier: float = 1.5) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        series = pd.to_numeric(result[column], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        result[f"{column}__iqr_outlier"] = (
            (series < q1 - multiplier * iqr) | (series > q3 + multiplier * iqr)
        )
    return result

def filter_valid_decision_rows(
    df: pd.DataFrame,
    decision_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Menghapus baris yang masih memiliki missing value pada
    decision variables setelah structural-zero reconstruction.

    Decision variables tidak diimputasi karena nilainya
    merepresentasikan dosis pupuk aktual.
    """
    valid_mask = df[decision_columns].notna().all(axis=1)

    removed = int((~valid_mask).sum())

    return df.loc[valid_mask].copy(), removed