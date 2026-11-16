import numpy as np
import pandas as pd
from sklearn.base import clone
from .metrics import regression_metrics

def evaluate_cv(estimator, X, y, cv, groups=None) -> pd.DataFrame:
    rows = []
    iterator = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold, (train_idx, validation_idx) in enumerate(iterator, start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_metrics = regression_metrics(y.iloc[train_idx], model.predict(X.iloc[train_idx]))
        validation_metrics = regression_metrics(
            y.iloc[validation_idx], model.predict(X.iloc[validation_idx])
        )
        rows.append({
            "fold": fold,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })
    return pd.DataFrame(rows)

def summarize_cv(fold_metrics: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for metric in ["mae", "rmse", "r2"]:
        values = fold_metrics[f"validation_{metric}"]
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
    return summary

def evaluate_holdout(estimator, X_test, y_test):
    predictions = estimator.predict(X_test)
    frame = pd.DataFrame({
        "actual": np.asarray(y_test, dtype=float),
        "predicted": np.asarray(predictions, dtype=float),
        "residual": np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float),
    }, index=X_test.index)
    return regression_metrics(y_test, predictions), frame
