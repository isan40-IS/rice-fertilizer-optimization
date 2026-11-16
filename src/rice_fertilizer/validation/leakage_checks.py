from __future__ import annotations
import pandas as pd

def check_group_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, group_column: str) -> dict:
    overlap = sorted(
        set(train_df[group_column].dropna().astype(str))
        & set(test_df[group_column].dropna().astype(str))
    )
    return {
        "group_column": group_column,
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:20],
        "passed": len(overlap) == 0,
    }

def check_record_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, record_id_column: str) -> dict:
    overlap = set(train_df[record_id_column]) & set(test_df[record_id_column])
    return {"overlap_count": len(overlap), "passed": len(overlap) == 0}

def detect_forbidden_features(features: list[str], forbidden: list[str]) -> list[str]:
    forbidden_set = {item.casefold() for item in forbidden}
    return [feature for feature in features if feature.casefold() in forbidden_set]
