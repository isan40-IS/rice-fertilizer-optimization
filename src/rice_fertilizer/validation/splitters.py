from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from ..exceptions import ConfigurationError

@dataclass
class SplitResult:
    train_indices: np.ndarray
    test_indices: np.ndarray
    cv: object
    groups: pd.Series | None

def make_holdout_and_cv(
    df: pd.DataFrame,
    strategy: str,
    group_column: str | None,
    validation_config: dict,
) -> SplitResult:
    indices = np.arange(len(df))
    random_state = int(validation_config.get("random_state", 42))
    test_size = float(validation_config.get("holdout_fraction", 0.2))
    n_splits = int(validation_config.get("n_splits", 5))

    if strategy == "random_kfold":
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, shuffle=True
        )
        return SplitResult(
            train_idx, test_idx,
            KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            None,
        )

    if strategy in {"group_kfold", "group_shuffle"}:
        if not group_column or group_column not in df.columns:
            raise ConfigurationError(
                f"Group strategy requires an existing group_column, got: {group_column}"
            )
        groups_all = df[group_column].astype("string")
        holdout = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(holdout.split(indices, groups=groups_all))
        train_groups = groups_all.iloc[train_idx].reset_index(drop=True)
        cv = (
            GroupKFold(n_splits=n_splits)
            if strategy == "group_kfold"
            else GroupShuffleSplit(
                n_splits=n_splits, test_size=test_size, random_state=random_state
            )
        )
        return SplitResult(train_idx, test_idx, cv, train_groups)

    year_column = validation_config["temporal"]["year_column"]
    if year_column not in df.columns:
        raise ConfigurationError(f"Temporal column not found: {year_column}")
    years = pd.to_numeric(df[year_column], errors="coerce")

    if strategy == "temporal_holdout":
        cutoff = validation_config["temporal"].get("cutoff_year")
        if cutoff is None:
            raise ConfigurationError("Set validation.temporal.cutoff_year")
        train_idx = indices[years.to_numpy() < cutoff]
        test_idx = indices[years.to_numpy() >= cutoff]
        if not len(train_idx) or not len(test_idx):
            raise ConfigurationError("Temporal cutoff produced an empty train or test set")
        return SplitResult(
            train_idx, test_idx,
            ExpandingWindowSplit(
                years.iloc[train_idx],
                n_splits=n_splits,
                minimum_train_years=int(
                    validation_config["temporal"].get("minimum_train_years", 3)
                ),
            ),
            None,
        )

    if strategy == "forward_chaining":
        if years.dropna().empty:
            raise ConfigurationError("No valid years for forward chaining")
        ordered = np.argsort(years.fillna(years.min()).to_numpy())
        cut = int(len(ordered) * (1.0 - test_size))
        return SplitResult(
            ordered[:cut], ordered[cut:],
            ExpandingWindowSplit(
                years.iloc[ordered[:cut]],
                n_splits=n_splits,
                minimum_train_years=int(
                    validation_config["temporal"].get("minimum_train_years", 3)
                ),
            ),
            None,
        )

    raise ConfigurationError(f"Unsupported validation strategy: {strategy}")

class ExpandingWindowSplit:
    def __init__(
        self, years: pd.Series, n_splits: int = 5, minimum_train_years: int = 3
    ):
        self.years = pd.to_numeric(years, errors="coerce").reset_index(drop=True)
        self.n_splits = n_splits
        self.minimum_train_years = max(1, int(minimum_train_years))

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        unique_years = np.array(sorted(self.years.dropna().unique()))
        if len(unique_years) <= self.minimum_train_years:
            raise ValueError(
                "Not enough distinct years for the configured minimum training window"
            )
        eligible_validation_years = unique_years[self.minimum_train_years:]
        validation_years = eligible_validation_years[
            -min(self.n_splits, len(eligible_validation_years)):
        ]
        values = self.years.to_numpy()
        for year in validation_years:
            train_idx = np.where(values < year)[0]
            validation_idx = np.where(values == year)[0]
            if len(train_idx) and len(validation_idx):
                yield train_idx, validation_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        available = max(0, self.years.nunique() - self.minimum_train_years)
        return min(self.n_splits, available)
