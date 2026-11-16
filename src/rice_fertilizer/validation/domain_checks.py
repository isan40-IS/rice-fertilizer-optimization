from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class NumericDomainSupport:
    def __init__(self, numeric_features: list[str]):
        self.numeric_features = numeric_features
        self.scaler = StandardScaler()
        self.neighbors = NearestNeighbors(n_neighbors=1)
        self.medians: pd.Series | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "NumericDomainSupport":
        numeric = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        self.medians = numeric.median()
        scaled = self.scaler.fit_transform(numeric.fillna(self.medians).to_numpy())
        self.neighbors.fit(scaled)
        self._fitted = True
        return self

    def distance(self, candidate: pd.DataFrame) -> float:
        if not self._fitted or self.medians is None:
            raise RuntimeError("NumericDomainSupport must be fitted first")
        numeric = candidate[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        scaled = self.scaler.transform(numeric.fillna(self.medians).to_numpy())
        distances, _ = self.neighbors.kneighbors(scaled)
        return float(distances[0, 0])

def numeric_range_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append({
            "column": column,
            "min": series.min() if not series.empty else np.nan,
            "max": series.max() if not series.empty else np.nan,
            "q01": series.quantile(0.01) if not series.empty else np.nan,
            "q99": series.quantile(0.99) if not series.empty else np.nan,
        })
    return pd.DataFrame(rows)
