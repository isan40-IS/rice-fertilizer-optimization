import pandas as pd

def evaluate_algorithm_stability(optimizer_function, scenario, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        result, _ = optimizer_function(scenario, seed)
        rows.append(result.to_dict())
    return pd.DataFrame(rows)

def summarize_algorithm_stability(results: pd.DataFrame) -> dict:
    summary = {}
    for column in [
        "optimal_n", "optimal_p", "optimal_k",
        "optimal_yield", "optimization_score",
    ]:
        summary[f"{column}_mean"] = float(results[column].mean())
        summary[f"{column}_std"] = float(results[column].std(ddof=1))
        summary[f"{column}_iqr"] = float(
            results[column].quantile(0.75) - results[column].quantile(0.25)
        )
    summary["success_rate"] = float(results["success"].mean())
    return summary
