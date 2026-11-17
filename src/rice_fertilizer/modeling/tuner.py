"""Configuration-driven, training-only hyperparameter search.

Finite lists are shared by all backends. Continuous spaces and persistent/resumed
Optuna studies are deliberately rejected until their contracts are implemented.
"""
from __future__ import annotations

import hashlib
import json
from inspect import signature, Parameter
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, cross_validate

from ..exceptions import ConfigurationError
from ..utils import save_json


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def tune_pipeline(
    pipeline, search_space: dict, X, y, cv, groups=None, n_iter: int = 30,
    scoring: str = "neg_root_mean_squared_error", n_jobs: int = -1,
    random_state: int = 42, tuning_config: dict | None = None, failure_path=None,
):
    settings = dict(tuning_config or {})
    method = settings.get("method", "random")
    if method not in {"random", "grid", "optuna"}:
        raise ConfigurationError(f"Unknown tuning method: {method}")
    if not isinstance(scoring, str) or not scoring:
        raise ConfigurationError("Tuning requires an explicit sklearn scoring name")
    seed = settings.get("search_seed", random_state)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("search_seed must be a nonnegative integer")
    parallel = settings.get("cv_n_jobs", n_jobs)
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel == 0:
        raise ConfigurationError("cv_n_jobs must be a nonzero integer")
    valid_parameters = set(pipeline.get_params())
    # CatBoost get_params lists explicitly supplied values, omitting valid defaults
    # such as depth/iterations. Its public constructor declares those parameters.
    for step_name, step in getattr(pipeline, "named_steps", {}).items():
        valid_parameters.update(
            f"{step_name}__{name}" for name, param in signature(type(step).__init__).parameters.items()
            if name != "self" and param.kind not in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
        )
    for name, choices in search_space.items():
        if name not in valid_parameters:
            raise ConfigurationError(f"Unknown pipeline parameter: {name}")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise ConfigurationError(f"{name}: phase-one search space requires a nonempty list")
        if any(not (v is None or isinstance(v, (str, int, float, bool))) for v in choices):
            raise ConfigurationError(f"{name}: choices must be scalar or null")
        if any(isinstance(v, float) and not np.isfinite(v) for v in choices):
            raise ConfigurationError(f"{name}: nonfinite choices are not supported")
    budget = None
    if method == "random":
        budget = _positive(settings.get("random", {}).get("n_iter", n_iter), "random.n_iter")
    elif method == "grid":
        cap = _positive(settings.get("grid", {}).get("max_candidates", 1000), "grid.max_candidates")
        budget = len(ParameterGrid(search_space)) if search_space else 0
        if search_space and len(ParameterGrid(search_space)) > cap:
            raise ConfigurationError(f"Grid has {len(ParameterGrid(search_space))} candidates, exceeds {cap}")
    else:
        opt = settings.get("optuna", {})
        budget = _positive(opt.get("n_trials", n_iter), "optuna.n_trials")
        if opt.get("sampler", "tpe") != "tpe" or opt.get("pruner", "none") != "none":
            raise ConfigurationError("Initial Optuna adapter supports TPE and no pruning only")
        if opt.get("storage") is not None or opt.get("resume", False):
            raise ConfigurationError("Persistent/resumed studies are not supported; start a fresh study")
        if opt.get("n_jobs", 1) != 1:
            raise ConfigurationError("Optuna trials must be sequential for this experiment")
        timeout = opt.get("timeout_seconds")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not np.isfinite(timeout) or timeout <= 0):
            raise ConfigurationError("timeout_seconds must be positive or null")
    started = perf_counter()
    info = {"method": method, "requested_method": method, "scoring": scoring,
            "direction": "maximize", "search_seed": seed, "requested_candidates": budget,
            "effective_config": settings, "search_space": search_space,
            "search_space_sha256": _digest(search_space)}
    if not search_space:
        fitted = clone(pipeline).fit(X, y)
        return fitted, {**info, "method": "direct", "best_params": {}, "best_score": None,
                        "results": [], "n_complete": 0, "n_unique": 0,
                        "runtime_seconds": perf_counter()-started}
    if hasattr(cv, "split"):
        folds = list(cv.split(X, y, groups))
    else:
        folds = list(cv)
    if not folds:
        raise ConfigurationError("No CV folds available")
    group_values = np.asarray(groups) if groups is not None else None
    for train, valid in folds:
        if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
            raise ConfigurationError("Invalid or overlapping CV indices")
        if group_values is not None and set(group_values[train]) & set(group_values[valid]):
            raise ConfigurationError("Group overlap inside tuning fold")
    fold_ids = [[np.asarray(a).tolist(), np.asarray(b).tolist()] for a, b in folds]
    info["fold_sha256"] = _digest(fold_ids)
    rows = []
    def record_failure(exc):
        if failure_path is not None:
            save_json({**info, "state": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "partial_results": rows}, failure_path)
    if method in {"random", "grid"}:
        kwargs = dict(estimator=pipeline, cv=folds, scoring=scoring, n_jobs=parallel,
                      refit=True, return_train_score=True, error_score="raise")
        if method == "random":
            search = RandomizedSearchCV(param_distributions=search_space,
                                        n_iter=budget, random_state=seed, **kwargs)
        else:
            search = GridSearchCV(param_grid=search_space, **kwargs)
        try:
            search.fit(X, y)
        except Exception as exc:
            record_failure(exc)
            raise
        results = search.cv_results_
        for i, params in enumerate(results["params"]):
            scores = [float(results[f"split{k}_test_score"][i]) for k in range(len(folds))]
            if not np.isfinite(scores).all():
                raise ConfigurationError("Nonfinite CV score; search result rejected")
            rows.append({"number": i, "params": params, "state": "COMPLETE",
                         "mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)),
                         "mean_fit_seconds": float(results["mean_fit_time"][i]),
                         "mean_score_seconds": float(results["mean_score_time"][i]),
                         **{f"fold_{k}_score": score for k, score in enumerate(scores)}})
        fitted, best_params, best_score = search.best_estimator_, search.best_params_, float(search.best_score_)
    else:
        try:
            import optuna
        except ImportError as exc:
            raise ConfigurationError('Optuna unavailable. Install optional dependency: pip install -e ".[tuning]"') from exc
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.NopPruner(), study_name=opt.get("study_name"),
        )
        info["study_name"] = study.study_name
        def objective(trial):
            params = {name: trial.suggest_categorical(name, list(choices)) for name, choices in search_space.items()}
            trial_start = perf_counter()
            try:
                evaluated = cross_validate(clone(pipeline).set_params(**params), X, y,
                    cv=folds, scoring=scoring, n_jobs=parallel, error_score="raise")
                scores = evaluated["test_score"]
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite CV score")
                trial.set_user_attr("fold_scores", scores.tolist())
                rows.append({"number": trial.number, "params": params, "state": "COMPLETE",
                             "mean_score": float(scores.mean()), "std_score": float(scores.std()),
                             "mean_fit_seconds": float(evaluated["fit_time"].mean()),
                             "mean_score_seconds": float(evaluated["score_time"].mean()),
                             "duration_seconds": perf_counter()-trial_start,
                             **{f"fold_{k}_score": float(score) for k, score in enumerate(scores)}})
                return float(scores.mean())
            except Exception as exc:
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
                rows.append({"number": trial.number, "params": params, "state": "FAIL",
                             "error": f"{type(exc).__name__}: {exc}"})
                raise  # fail fast; never turn an error into a winning score
        try:
            study.optimize(objective, n_trials=budget, timeout=opt.get("timeout_seconds"), n_jobs=1)
        except Exception as exc:
            record_failure(exc)
            raise
        if not rows:
            raise ConfigurationError("No valid Optuna trials completed")
        best_params, best_score = study.best_params, float(study.best_value)
        fitted = clone(pipeline).set_params(**best_params).fit(X, y)
    return fitted, {**info, "best_params": best_params, "best_score": best_score,
                    "results": rows, "n_complete": len(rows), "n_failed": 0,
                    "n_unique": len({_digest(r["params"]) for r in rows}),
                    "runtime_seconds": perf_counter()-started}
