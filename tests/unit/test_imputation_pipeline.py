from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rice_fertilizer.config import load_yaml
from rice_fertilizer.exceptions import ConfigurationError
from rice_fertilizer.modeling.imputation import resolve_imputation_plan
from rice_fertilizer.modeling.pipeline_factory import build_preprocessor


RATES = {"N": "N", "P": "P", "K": "K"}


def resolve(settings, numeric=None, categorical=None):
    return resolve_imputation_plan(
        numeric or ["soil"], categorical or [],
        {"state_imputation": settings}, RATES, "yield",
    )


def test_none_method_does_not_create_or_validate_mice():
    plan = resolve({"default_method": "none"})
    assert plan.none_columns == ("soil",)
    assert plan.mice_columns == ()
    assert plan.mice_params == {}


def test_global_none_overrides_legacy_column_imputation():
    plan = resolve({
        "mode": "none", "default_method": "mice",
        "columns": {"soil": {"method": "mice"}},
    })
    X = pd.DataFrame({"soil": [1.0, np.nan, 3.0]})
    pre = build_preprocessor(["soil"], [], [], False, plan)
    transformed = pre.fit_transform(X)
    np.testing.assert_array_equal(np.isnan(transformed[:, 0]), X.soil.isna())
    np.testing.assert_allclose(transformed[[0, 2], 0], [1.0, 3.0])


def test_no_imputation_includes_categorical_missing_as_category():
    X = pd.DataFrame({
        "soil": [1.0, np.nan, 3.0],
        "type": ["A", np.nan, "A"],
    })
    plan = resolve({"mode": "none"}, categorical=["type"])
    pre = build_preprocessor(["soil"], ["type"], [], False, plan)
    transformed = pre.fit_transform(X)
    assert np.isnan(transformed[1, 0])
    assert not np.array_equal(transformed[0, 1:], transformed[1, 1:])
    assert "imputer" not in pre.named_transformers_["categorical_complete"].named_steps
    # An unseen category at inference must not change the input schema.
    assert pre.transform(pd.DataFrame({"soil": [np.nan], "type": ["B"]})).shape[1] == transformed.shape[1]


def test_plan_supports_mixed_mice_simple_and_none_inside_folds():
    X = pd.DataFrame({
        "soil": [1.0, 2.0, np.nan, 4.0],
        "rain": [10.0, np.nan, 30.0, 40.0],
        "keep": [np.nan, 2.0, 3.0, 4.0],
        "type": ["A", np.nan, "A", "B"],
    })
    plan = resolve({
        "mode": "plan", "default_method": "median",
        "mice_shared": {"estimator": "bayesian_ridge"},
        "columns": {"soil": {"method": "mice"}, "keep": {"method": "none"}},
    }, numeric=["soil", "rain", "keep"], categorical=["type"])
    pre = build_preprocessor(["soil", "rain", "keep"], ["type"], [], False, plan)
    # Only the training slice may determine statistics, not validation values.
    pre.fit(X.iloc[:3])
    median = pre.named_transformers_["numeric_median"].named_steps["imputer"]
    assert median.statistics_[0] == 20.0
    transformed = pre.transform(X)
    names = list(pre.get_feature_names_out())
    assert np.isnan(transformed[0, names.index("numeric_complete__keep")])
    assert np.isfinite(transformed[:, names.index("numeric_mice__soil")]).all()
    assert np.isfinite(transformed[:, names.index("numeric_median__rain")]).all()


def test_plan_can_preserve_categorical_missing_without_mode_imputation():
    plan = resolve({
        "mode": "plan", "default_method": "median",
        "columns": {"type": {"method": "none"}},
    }, categorical=["type"])
    pre = build_preprocessor(["soil"], ["type"], [], False, plan)
    pre.fit(pd.DataFrame({"soil": [1.0, np.nan], "type": ["A", np.nan]}))
    assert "imputer" not in pre.named_transformers_["categorical_complete"].named_steps


@pytest.mark.parametrize("column", ["N", "P", "K", "yield"])
@pytest.mark.parametrize("mode", ["plan", "none"])
def test_target_and_doses_cannot_be_configured_for_imputation(column, mode):
    with pytest.raises(ConfigurationError, match="forbidden"):
        resolve({"mode": mode, "default_method": "median", "columns": {column: {"method": "mice"}}})


def test_invalid_global_mode_is_not_silently_accepted():
    with pytest.raises(ConfigurationError, match="state_imputation.mode"):
        resolve({"mode": "native"})


def test_broader_profiles_do_not_duplicate_decision_features():
    plan = resolve({"mode": "none"}, numeric=["soil", "N"])
    pre = build_preprocessor(["soil", "N"], [], ["N"], False, plan)
    result = pre.fit_transform(pd.DataFrame({"soil": [np.nan, 2.0], "N": [10.0, 20.0]}))
    assert result.shape == (2, 2)
    assert list(pre.get_feature_names_out()) == ["numeric_complete__soil", "decision_features__N"]


def test_advanced_example_resolves_mice_without_becoming_active():
    root = Path(__file__).resolve().parents[2]
    example = load_yaml(root / "configs/examples/decision.example.yaml")
    features = load_yaml(root / "configs/features.yaml")
    profile = features["profiles"][example["feature_profile"]]
    plan = resolve_imputation_plan(
        profile["numeric"], profile["categorical"], example,
        {"N": "N.rate.kg_ha", "P": "P.rate.kg_ha", "K": "K.rate.kg_ha"},
        "yield.wb.kg_ha",
    )
    assert plan.mode == "plan"
    assert "clay.perc1" in plan.mice_columns
    active = load_yaml(root / "configs/decisions.yaml")
    assert active["state_imputation"]["mode"] == "none"
