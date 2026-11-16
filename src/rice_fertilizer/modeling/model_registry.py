from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from ..exceptions import ConfigurationError

def create_estimator(model_class: str, params: dict):
    if model_class == "linear_regression":
        return LinearRegression(**params)
    if model_class == "random_forest":
        return RandomForestRegressor(**params)
    if model_class == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ConfigurationError("xgboost is not installed") from exc
        return XGBRegressor(**params)
    if model_class == "extra_trees":
        return ExtraTreesRegressor(**params)
    if model_class == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ConfigurationError("catboost is not installed") from exc
        return CatBoostRegressor(**params)
    if model_class == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ConfigurationError("lightgbm is not installed") from exc
        return LGBMRegressor(**params)
    raise ConfigurationError(f"Unknown model class: {model_class}")
