"""Enrich an existing cohort without selecting new observations or changing labels."""
from __future__ import annotations

import pandas as pd

from ..exceptions import DataValidationError


def align_reference_cohort(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    record_id: str,
    expected_rows: int,
) -> pd.DataFrame:
    for name, frame in (("source", source), ("reference", reference)):
        if record_id not in frame:
            raise DataValidationError(f"{name} cohort is missing {record_id}")
        if frame[record_id].isna().any() or frame[record_id].duplicated().any():
            raise DataValidationError(f"{name} cohort requires unique, nonmissing IDs")
    if len(reference) != expected_rows:
        raise DataValidationError(
            f"Reference cohort has {len(reference)} rows, expected {expected_rows}"
        )
    missing = reference.loc[~reference[record_id].isin(source[record_id]), record_id]
    if not missing.empty:
        raise DataValidationError(f"Reference IDs absent from source: {missing.head().tolist()}")
    return source.set_index(record_id).loc[reference[record_id]].reset_index()


def validate_reference_preserved(result: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Require all existing values, IDs and row ordering to remain unchanged."""
    missing = reference.columns.difference(result.columns)
    if not missing.empty:
        raise DataValidationError(f"Enriched data lost reference columns: {missing.tolist()}")
    try:
        pd.testing.assert_frame_equal(
            # Normalize pandas nullable NA versus CSV-loaded numpy NaN only;
            # observed values, missingness masks and row order must still match.
            result[reference.columns].reset_index(drop=True).convert_dtypes(),
            reference.reset_index(drop=True).convert_dtypes(),
            check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9,
        )
    except AssertionError as exc:
        raise DataValidationError(
            "Enrichment changed reference rows/order/values; no output was written. "
            + str(exc)[:700]
        ) from exc
