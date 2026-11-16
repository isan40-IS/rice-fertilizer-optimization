from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_actual_vs_predicted(predictions: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(predictions["actual"], predictions["predicted"], alpha=0.6)
    low = min(predictions["actual"].min(), predictions["predicted"].min())
    high = max(predictions["actual"].max(), predictions["predicted"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set(xlabel="Actual yield", ylabel="Predicted yield", title="Actual vs predicted yield")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def save_convergence(history: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for (_, seed), group in history.groupby(["scenario_id", "seed"]):
        ax.plot(group["generation"], group["score"], alpha=0.5)
    ax.set(xlabel="Generation", ylabel="Optimization score", title="DE convergence")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def save_yield_improvement_histogram(results: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(results["yield_improvement"].dropna(), bins=20)
    ax.set(
        xlabel="Predicted yield improvement",
        ylabel="Scenario count",
        title="Distribution of predicted yield improvement",
    )
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def save_baseline_vs_optimal_doses(results: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "Baseline": [
            results["baseline_n"].median(),
            results["baseline_p"].median(),
            results["baseline_k"].median(),
        ],
        "Optimal": [
            results["optimal_n"].median(),
            results["optimal_p"].median(),
            results["optimal_k"].median(),
        ],
    }, index=["N", "P", "K"])
    ax = summary.plot(kind="bar")
    ax.set(ylabel="Dose", title="Median baseline and optimized dose")
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
