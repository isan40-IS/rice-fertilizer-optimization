import builtins
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from rice_fertilizer.exceptions import ConfigurationError
from rice_fertilizer.modeling.tuner import tune_pipeline
from rice_fertilizer.modeling.pipeline_factory import build_preprocessor
from rice_fertilizer.modeling.imputation import resolve_imputation_plan


@pytest.fixture
def search_data():
    x = pd.DataFrame({"x": np.arange(24, dtype=float)})
    y = pd.Series(np.sin(x.x)*3+x.x)
    groups = np.repeat(np.arange(6), 4)
    pipeline = Pipeline([("model", DecisionTreeRegressor(random_state=42))])
    return pipeline, x, y, GroupKFold(3), groups


def run(data, method, **extra):
    pipe, x, y, cv, groups = data
    cfg = {"method": method, "search_seed": 11, "cv_n_jobs": 1,
           "random": {"n_iter": 3}, "grid": {"max_candidates": 10},
           "optuna": {"n_trials": 3}}
    cfg.update(extra)
    return tune_pipeline(pipe, {"model__max_depth": [1, 2, 3]}, x, y, cv,
                         groups, n_jobs=1, tuning_config=cfg)


@pytest.mark.parametrize("method", ["random", "grid", "optuna"])
def test_backends_fit_score_and_record_folds(search_data, method):
    if method == "optuna":
        pytest.importorskip("optuna")
    fitted, info = run(search_data, method)
    assert len(fitted.predict(search_data[1])) == 24
    assert info["method"] == method and info["best_score"] < 0
    assert info["n_complete"] == 3
    assert info["best_score"] == max(r["mean_score"] for r in info["results"])
    assert all("fold_2_score" in r for r in info["results"])
    assert info["n_unique"] <= info["n_complete"]
    json.dumps(info)  # exportable metadata, not a required live Study object


@pytest.mark.parametrize("method", ["random", "optuna"])
def test_fixed_search_seed_reproduces_deterministic_study(search_data, method):
    if method == "optuna":
        pytest.importorskip("optuna")
    _, a = run(search_data, method)
    _, b = run(search_data, method)
    assert [(r["params"], r["mean_score"]) for r in a["results"]] == [(r["params"], r["mean_score"]) for r in b["results"]]
    assert a["fold_sha256"] == b["fold_sha256"]


def test_legacy_config_still_uses_random(search_data):
    p, x, y, cv, g = search_data
    _, info = tune_pipeline(p, {"model__max_depth": [1, 2]}, x, y, cv, g, n_iter=2, n_jobs=1)
    assert info["method"] == "random" and info["n_complete"] == 2


def test_empty_space_direct_fit_and_invalid_backend(search_data):
    p, x, y, cv, g = search_data
    fit, info = tune_pipeline(p, {}, x, y, cv, g)
    assert info["method"] == "direct" and info["best_score"] is None
    assert len(fit.predict(x)) == len(y)
    with pytest.raises(ConfigurationError, match="Unknown tuning"):
        tune_pipeline(p, {}, x, y, cv, g, tuning_config={"method": "typo"})


@pytest.mark.parametrize("cfg", [
    {"method": "grid", "grid": {"max_candidates": 2}},
    {"method": "random", "random": {"n_iter": 0}},
    {"method": "optuna", "optuna": {"n_trials": 0}},
    {"method": "optuna", "optuna": {"resume": True}},
    {"method": "optuna", "optuna": {"n_jobs": 2}},
    {"method": "optuna", "optuna": {"timeout_seconds": -1}},
])
def test_invalid_settings_rejected_before_fit(search_data, cfg):
    with pytest.raises(ConfigurationError):
        run(search_data, **cfg)


@pytest.mark.parametrize("space", [{"model__typo": [1]}, {"model__max_depth": []}, {"model__max_depth": {"low": 1, "high": 3}}])
def test_invalid_space(search_data, space):
    p, x, y, cv, g = search_data
    with pytest.raises(ConfigurationError):
        tune_pipeline(p, space, x, y, cv, g)


def test_group_overlap_rejected(search_data):
    p, x, y, _, g = search_data
    with pytest.raises(ConfigurationError, match="Group overlap"):
        tune_pipeline(p, {"model__max_depth": [1]}, x, y,
                      [(np.array([0, 1, 4]), np.array([2, 3, 5]))], g)


def test_nonfinite_or_failed_trial_never_becomes_winner(search_data, tmp_path):
    pytest.importorskip("optuna")
    p, x, y, cv, g = search_data
    with pytest.raises(ValueError):
        tune_pipeline(p, {"model__max_depth": [-1]}, x, y, cv, g,
                      failure_path=tmp_path/"failure.json",
                      tuning_config={"method": "optuna", "optuna": {"n_trials": 1}})
    record = json.loads((tmp_path/"failure.json").read_text())
    assert record["state"] == "FAILED" and record["partial_results"][-1]["state"] == "FAIL"


def test_random_does_not_require_optuna(search_data, monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("test missing optional dependency")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    _, info = run(search_data, "random")
    assert info["method"] == "random"
    with pytest.raises(ConfigurationError, match="Optuna unavailable"):
        run(search_data, "optuna")


def test_drop_all_missing_is_fit_only_not_holdout_selected():
    settings = {"state_imputation": {"mode": "none"}}
    plan = resolve_imputation_plan(["empty", "observed"], ["category"], settings, {}, "target")
    pre = build_preprocessor(["empty", "observed"], ["category"], [], False,
                             plan, drop_all_missing=True)
    train = pd.DataFrame({"empty": [np.nan, np.nan], "observed": [1., 2.], "category": [np.nan, np.nan]})
    pre.fit(train)
    holdout = pd.DataFrame({"empty": [99.], "observed": [3.], "category": ["new"]})
    assert pre.transform(holdout).shape == (1, 1)
    assert pre.get_feature_names_out().tolist() == ["numeric_complete__observed"]
