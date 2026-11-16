from __future__ import annotations
import pandas as pd
from ..exceptions import DataValidationError

def filter_crop(
    df: pd.DataFrame,
    column: str,
    accepted_labels: list[str],
    case_sensitive: bool = False,
) -> pd.DataFrame:
    if column not in df.columns:
        raise DataValidationError(f"Crop column not found: {column}")
    values = df[column].astype("string").str.strip()
    if case_sensitive:
        mask = values.isin(accepted_labels)
    else:
        accepted = {str(value).strip().casefold() for value in accepted_labels}
        mask = values.str.casefold().isin(accepted)
    result = df.loc[mask].copy()
    if result.empty:
        raise DataValidationError(
            f"No crop rows matched labels {accepted_labels} in column '{column}'."
        )
    return result
