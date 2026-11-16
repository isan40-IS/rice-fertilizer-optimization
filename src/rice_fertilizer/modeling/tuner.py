from sklearn.model_selection import RandomizedSearchCV

def tune_pipeline(
    pipeline, search_space: dict, X, y, cv, groups=None, n_iter: int = 30,
    scoring: str = "neg_root_mean_squared_error", n_jobs: int = -1,
    random_state: int = 42,
):
    if not search_space:
        pipeline.fit(X, y)
        return pipeline, {"best_params": {}, "best_score": None}
    search = RandomizedSearchCV(
        pipeline, search_space, n_iter=n_iter, scoring=scoring, cv=cv,
        n_jobs=n_jobs, random_state=random_state, refit=True,
        return_train_score=True, error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }
