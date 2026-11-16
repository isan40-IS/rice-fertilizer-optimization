import pandas as pd

def rank_models(model_summary: pd.DataFrame, selection_config: dict) -> pd.DataFrame:
    result = model_summary.copy()
    score = pd.Series(0.0, index=result.index)
    for metric, weight in selection_config.get("metric_weights", {}).items():
        if metric not in result:
            continue
        values = pd.to_numeric(result[metric], errors="coerce")
        std = values.std(ddof=0)
        score += float(weight) * ((values - values.mean()) / (std if std else 1.0)).fillna(0)
    result["selection_score"] = score
    minimum_r2 = selection_config.get("minimum_r2")
    result["eligible"] = True if minimum_r2 is None else result["r2_mean"] >= float(minimum_r2)
    result = result.sort_values(["eligible", "selection_score"], ascending=[False, True])
    result["selected"] = False
    if result["eligible"].any():
        result.loc[result[result["eligible"]].index[0], "selected"] = True
    return result.reset_index(drop=True)
