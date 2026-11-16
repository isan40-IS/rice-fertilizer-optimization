from __future__ import annotations
import numpy as np
import pandas as pd

def add_npk_features(
    df: pd.DataFrame,
    n_column: str = "N.rate.kg_ha",
    p_column: str = "P.rate.kg_ha",
    k_column: str = "K.rate.kg_ha",
    ratio_pseudocount: float = 1e-8,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    result = df.copy()
    n = pd.to_numeric(result[n_column], errors="coerce")
    p = pd.to_numeric(result[p_column], errors="coerce")
    k = pd.to_numeric(result[k_column], errors="coerce")
    total = n + p + k
    result["total_npk"] = total
    if ratio_pseudocount < 0:
        raise ValueError("ratio_pseudocount must be non-negative")
    result["ratio_n_p"] = n / (p + ratio_pseudocount)
    result["ratio_n_k"] = n / (k + ratio_pseudocount)
    result["ratio_p_k"] = p / (k + ratio_pseudocount)
    result["prop_n"] = n / (total + epsilon)
    result["prop_p"] = p / (total + epsilon)
    result["prop_k"] = k / (total + epsilon)
    result.loc[total.eq(0), ["prop_n", "prop_p", "prop_k"]] = 0.0
    return result

def select_feature_frame(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
    decision_features: list[str],
    passthrough_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys(
        (passthrough_columns or []) + numeric_features + categorical_features + decision_features + [target]
    ))
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Selected feature columns are missing: {missing}")
    return df[columns].copy()
