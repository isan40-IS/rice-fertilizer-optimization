from __future__ import annotations
import re
from typing import Any
import pandas as pd
from ..constants import DECISION_ORDER
from ..exceptions import DataValidationError

def _normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())

def parse_applied_nutrients(
    value: Any,
    control_labels: list[str],
    explicit_label_map: dict[str, list[str]] | None = None,
) -> tuple[set[str] | None, str]:
    """Conservative parsing of nutrient-application labels."""
    normalized = _normalize_label(value)
    if not normalized:
        return None, "indicator_missing"
    if normalized in {_normalize_label(item) for item in control_labels}:
        return set(), "control_label"
    explicit = {
        _normalize_label(key): {nutrient.upper() for nutrient in nutrients}
        for key, nutrients in (explicit_label_map or {}).items()
    }
    if normalized in explicit:
        return explicit[normalized], "explicit_map"
    compact = re.sub(r"[^NPKS]", "", normalized)
    leftover = re.sub(r"[NPKS\s,+/&-]", "", normalized)
    if compact and not leftover:
        return set(compact), "automatic_npk_pattern"
    return None, "ambiguous_label"

def reconstruct_structural_zeroes(
    df: pd.DataFrame,
    indicator_column: str,
    nutrient_rate_columns: dict[str, str],
    control_labels: list[str],
    explicit_label_map: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if indicator_column not in df.columns:
        raise DataValidationError(f"Structural-zero indicator column missing: {indicator_column}")
    result = df.copy()
    parsed = result[indicator_column].apply(
        lambda value: parse_applied_nutrients(value, control_labels, explicit_label_map)
    )
    audit_rows = []
    for nutrient in DECISION_ORDER:
        column = nutrient_rate_columns[nutrient]
        if column not in result.columns:
            raise DataValidationError(f"Nutrient rate column missing: {column}")
        original_missing = result[column].isna()
        reasons, zero_flags = [], []
        for idx, (applied_set, parse_status) in parsed.items():
            if not bool(original_missing.loc[idx]):
                reasons.append("observed_value"); zero_flags.append(False)
            elif applied_set is not None and nutrient not in applied_set:
                result.at[idx, column] = 0.0
                reasons.append("not_applied_by_indicator"); zero_flags.append(True)
            elif applied_set is not None and nutrient in applied_set:
                reasons.append("applied_but_rate_missing"); zero_flags.append(False)
            else:
                reasons.append(parse_status); zero_flags.append(False)
        result[f"{nutrient}_missing_reason"] = reasons
        result[f"{nutrient}_was_structural_zero"] = zero_flags
        audit_rows.append({
            "nutrient": nutrient,
            "rate_column": column,
            "missing_before": int(original_missing.sum()),
            "structural_zero_filled": int(sum(zero_flags)),
            "missing_after": int(result[column].isna().sum()),
        })
    label_audit = pd.DataFrame({
        "indicator_value": result[indicator_column],
        "parse_status": [status for _, status in parsed],
        "parsed_nutrients": [
            "" if values is None else "".join(sorted(values)) for values, _ in parsed
        ],
    })
    return result, pd.DataFrame(audit_rows), label_audit
