import pandas as pd

def optimization_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "baseline_yield", "optimal_yield", "yield_improvement",
        "baseline_n", "baseline_p", "baseline_k",
        "optimal_n", "optimal_p", "optimal_k",
        "optimization_score", "runtime_seconds",
    ]
    rows = []
    for metric in metrics:
        if metric not in results.columns:
            continue
        series = pd.to_numeric(results[metric], errors="coerce")
        rows.append({
            "metric": metric,
            "mean": series.mean(),
            "std": series.std(ddof=1),
            "median": series.median(),
            "q25": series.quantile(0.25),
            "q75": series.quantile(0.75),
            "min": series.min(),
            "max": series.max(),
        })
    return pd.DataFrame(rows)

def sensitivity_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby(["feature", "nutrient"], dropna=False)
        .agg(
            raw_sensitivity_median=("raw_sensitivity", "median"),
            normalized_sensitivity_median=("normalized_sensitivity", "median"),
            normalized_sensitivity_iqr=(
                "normalized_sensitivity",
                lambda x: x.quantile(0.75) - x.quantile(0.25),
            ),
            sensitivity_index_median=("sensitivity_index", "median"),
        )
        .reset_index()
    )
