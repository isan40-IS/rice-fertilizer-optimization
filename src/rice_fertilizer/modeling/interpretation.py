import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def transformed_feature_names(fitted_pipeline) -> list[str]:
    return list(fitted_pipeline.named_steps["preprocessor"].get_feature_names_out())

def built_in_feature_importance(fitted_pipeline) -> pd.DataFrame:
    model = fitted_pipeline.named_steps["model"]
    names = transformed_feature_names(fitted_pipeline)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    if len(names) != len(values):
        raise ValueError(f"Feature importance schema mismatch: {len(names)} names / {len(values)} values")
    return pd.DataFrame({
        "feature": names,
        "importance": values,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

def permutation_feature_importance(
    fitted_pipeline, X, y, scoring="neg_root_mean_squared_error",
    n_repeats=10, random_state=42, n_jobs=1,
) -> pd.DataFrame:
    result = permutation_importance(
        fitted_pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=n_jobs,
    )
    return pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
