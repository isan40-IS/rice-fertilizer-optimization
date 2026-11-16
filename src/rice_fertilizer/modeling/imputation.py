from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
# IterativeImputer is experimental in scikit-learn; must be explicitly
# enabled before it becomes importable from sklearn.impute.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from ..exceptions import ConfigurationError

_MICE_ESTIMATORS = {
    "bayesian_ridge": BayesianRidge,
    "extra_trees": ExtraTreesRegressor,
    "decision_tree": DecisionTreeRegressor,
}

_VALID_METHODS = {"median", "most_frequent", "mice", "none",}


@dataclass(frozen=True)
class ResolvedImputationPlan:
    """
    Fully resolved imputation plan for every state feature the caller
    asked about. Columns are partitioned into three disjoint groups by
    method — this partition is what pipeline_factory consumes to
    build the ColumnTransformer branches.
    """
    median_columns: tuple[str, ...]
    most_frequent_columns: tuple[str, ...]
    mice_columns: tuple[str, ...]
    mice_params: dict


def _build_mice_estimator(estimator_name: str | None):
    if estimator_name is None:
        raise ConfigurationError(
            "state_imputation.mice_shared.estimator is not set. "
            "Choose one of: " + ", ".join(sorted(_MICE_ESTIMATORS))
        )
    if estimator_name not in _MICE_ESTIMATORS:
        raise ConfigurationError(
            f"Unknown MICE estimator '{estimator_name}'. "
            f"Choose one of: {sorted(_MICE_ESTIMATORS)}"
        )
    return _MICE_ESTIMATORS[estimator_name]()


def resolve_imputation_plan(
    numeric_features: list[str],
    categorical_features: list[str],
    decisions: dict,
    nutrient_rate_columns: dict[str, str],
    target_column: str,
) -> ResolvedImputationPlan:
    """
    Resolve the imputation plan for every state feature present in the
    given numeric + categorical feature lists.

    Raises ConfigurationError if:
      - state_imputation section is missing from decisions.yaml,
      - a decision variable (N/P/K rate) or the target column is
        explicitly configured under state_imputation.columns,
      - a state feature has no resolvable method (neither a column
        override nor default_method is set),
      - method is unknown,
      - a categorical column is configured with method "mice"
        (IterativeImputer cannot natively model string categories),
      - method "mice" is used anywhere but mice_shared.estimator is
        missing or unknown.
    """
    decision_columns = set(nutrient_rate_columns.values())
    state_imputation = decisions.get("state_imputation")
    if state_imputation is None:
        raise ConfigurationError(
            "configs/decisions.yaml is missing the 'state_imputation' section."
        )

    columns_cfg = state_imputation.get("columns", {})
    default_method = state_imputation.get("default_method")

    # Fail loudly if someone configured imputation for a column this
    # module must never touch — catches the mistake at config-read
    # time, not silently downstream in a fitted pipeline.
    forbidden_configured = (set(columns_cfg) & decision_columns) | (
        {target_column} if target_column in columns_cfg else set()
    )
    if forbidden_configured:
        raise ConfigurationError(
            f"state_imputation.columns configures forbidden column(s) "
            f"{sorted(forbidden_configured)} — decision variables (N/P/K "
            "rate) and the target column must never be imputed here. "
            "They are handled by structural-zero reconstruction."
        )

    all_features = [*numeric_features, *categorical_features]
    categorical_set = set(categorical_features)

    median_columns: list[str] = []
    most_frequent_columns: list[str] = []
    mice_columns: list[str] = []

    for column in all_features:
        if column in decision_columns or column == target_column:
            continue  # out of scope for this module by design

        method = columns_cfg.get(column, {}).get("method") or default_method
        if method is None:
            raise ConfigurationError(
                f"No imputation method resolved for '{column}'. Set "
                f"state_imputation.columns.{column}.method or "
                "state_imputation.default_method."
            )
        if method not in _VALID_METHODS:
            raise ConfigurationError(
                f"Unknown imputation method '{method}' for '{column}'. "
                f"Choose one of: {sorted(_VALID_METHODS)}"
            )
        if method == "mice" and column in categorical_set:
            raise ConfigurationError(
                f"'{column}' is categorical; IterativeImputer ('mice') cannot "
                "model string categories directly. Use 'most_frequent' instead."
            )

        if method == "median":
            median_columns.append(column)
        elif method == "most_frequent":
            most_frequent_columns.append(column)
        else:
            mice_columns.append(column)

    mice_params: dict = {}
    if mice_columns:
        mice_shared = state_imputation.get("mice_shared", {})
        mice_params = {
            "estimator": mice_shared.get("estimator"),
            "max_iter": int(mice_shared.get("max_iter", 10)),
            "tol": float(mice_shared.get("tol", 1e-3)),
            "initial_strategy": mice_shared.get("initial_strategy", "median"),
            "random_state": mice_shared.get("random_state", 42),
        }
        # Validate the estimator now, at config-resolution time, so a
        # missing/unknown estimator fails before any fold is fit.
        _build_mice_estimator(mice_params["estimator"])

    return ResolvedImputationPlan(
        median_columns=tuple(median_columns),
        most_frequent_columns=tuple(most_frequent_columns),
        mice_columns=tuple(mice_columns),
        mice_params=mice_params,
    )


def build_mice_imputer(mice_params: dict) -> IterativeImputer:
    """
    Build the single shared IterativeImputer for every column resolved
    to method "mice". All columns passed into this transformer act as
    each other's predictors (see module docstring, limitation 1).
    """
    estimator = _build_mice_estimator(mice_params.get("estimator"))
    return IterativeImputer(
        estimator=estimator,
        max_iter=int(mice_params.get("max_iter", 10)),
        tol=float(mice_params.get("tol", 1e-3)),
        initial_strategy=mice_params.get("initial_strategy", "median"),
        random_state=mice_params.get("random_state", 42),
        add_indicator=True,
    )


def build_simple_imputer(strategy: str) -> SimpleImputer:
    """strategy is "median" or "most_frequent"; add_indicator=True in
    both cases so the model can see which rows were originally missing."""
    return SimpleImputer(strategy=strategy, add_indicator=True)
