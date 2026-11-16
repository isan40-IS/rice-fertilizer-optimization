from rice_fertilizer.data.feature_engineering import add_npk_features
from rice_fertilizer.modeling.model_registry import create_estimator
from rice_fertilizer.modeling.pipeline_factory import build_model_pipeline
from rice_fertilizer.modeling.metrics import regression_metrics
import joblib
import numpy as np
import pandas as pd
import pytest

from rice_fertilizer.config import load_all_configs
from rice_fertilizer.modeling.imputation import resolve_imputation_plan

def test_random_forest_pipeline_runs(synthetic_data):
    df = add_npk_features(synthetic_data)
    numeric = [
        "N.rate.kg_ha", "P.rate.kg_ha", "K.rate.kg_ha",
        "avprec.mm", "total_npk",
    ]
    categorical = ["soil_type", "fertilizer_type"]
    X, y = df[numeric + categorical], df["yield.wb.kg_ha"]
    estimator = create_estimator(
        "random_forest", {"n_estimators": 20, "random_state": 1}
    )
    pipeline = build_model_pipeline(
        estimator, numeric, categorical, decision_features=[],
        model_class="random_forest", imputation_mode="none",
    )
    pipeline.fit(X.iloc[:100], y.iloc[:100])
    prediction = pipeline.predict(X.iloc[100:])
    assert len(prediction) == 20
    assert regression_metrics(y.iloc[100:], prediction)["rmse"] >= 0


@pytest.mark.parametrize("model_class,params", [
    ("xgboost", {"n_estimators": 3, "max_depth": 2, "n_jobs": 1}),
    ("catboost", {"iterations": 3, "depth": 2, "thread_count": 1,
                  "verbose": False, "allow_writing_files": False}),
])
def test_native_pipeline_fit_predict_and_reload(model_class, params, synthetic_data, tmp_path):
    df = synthetic_data.copy()
    df.loc[::3, "avprec.mm"] = np.nan
    plan = resolve_imputation_plan(
        ["avprec.mm"], ["fertilizer_type"],
        {"state_imputation": {"mode": "none"}},
        {"N": "N.rate.kg_ha", "P": "P.rate.kg_ha", "K": "K.rate.kg_ha"},
        "yield.wb.kg_ha",
    )
    columns = ["avprec.mm", "fertilizer_type", "N.rate.kg_ha"]
    pipeline = build_model_pipeline(
        create_estimator(model_class, params), ["avprec.mm"], ["fertilizer_type"],
        ["N.rate.kg_ha"], model_class, plan,
    )
    pipeline.fit(df.loc[:79, columns], df.loc[:79, "yield.wb.kg_ha"])
    pre = pipeline.named_steps["preprocessor"]
    transformed = pre.transform(df[columns])
    assert np.isnan(transformed[:, 0]).sum() == df["avprec.mm"].isna().sum()
    prediction = pipeline.predict(df.loc[80:, columns])
    assert np.isfinite(prediction).all()
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(joblib.load(path).predict(df.loc[80:, columns]), prediction)


@pytest.mark.parametrize("mode", ["none", "plan"])
def test_training_honors_decision_mode_and_saves_it(mode, synthetic_data, tmp_path, monkeypatch):
    from rice_fertilizer.modeling import trainer

    configs = load_all_configs()
    # This synthetic fixture exercises mode wiring, independently of the active study.
    configs["decisions"]["feature_profile"] = "minimalist_features"
    configs["models"]["selection"]["minimum_r2"] = None
    df = add_npk_features(synthetic_data)
    df["initial.ph.soil"] = 6.5
    df["soc.percentage"] = 1.2
    df.loc[::3, ["initial.ph.soil", "soc.percentage", "avprec.mm"]] = np.nan
    df["group_article_experiment"] = df["id"].astype(str) + "::" + df["exp.id"].astype(str)
    data_path = tmp_path / "synthetic.csv"
    df.to_csv(data_path, index=False)
    configs["paths"]["processed_dataset"] = str(data_path)
    configs["validation"]["n_splits"] = 2
    configs["decisions"]["state_imputation"] = {
        "mode": mode, "default_method": "median", "columns": {},
    }
    configs["models"]["models"] = {
        "xgboost": {"class": "xgboost", "params": {"n_estimators": 3, "n_jobs": 1}},
    }
    # This is a pipeline regression test, not a feature-importance experiment.
    empty_importance = lambda *args, **kwargs: pd.DataFrame(columns=["feature", "importance"])
    monkeypatch.setattr(trainer, "built_in_feature_importance", empty_importance)
    monkeypatch.setattr(trainer, "permutation_feature_importance", empty_importance)
    outputs = trainer.run_training(configs, tmp_path / "run")
    bundle = joblib.load(outputs["selected_model"])
    assert bundle["imputation_mode"] == mode
    assert bundle["state_imputation"] == configs["decisions"]["state_imputation"]
    comparison = pd.read_csv(outputs["model_comparison"])
    assert comparison["imputation_mode"].tolist() == [mode]
    transformed = bundle["pipeline"].named_steps["preprocessor"].transform(df[bundle["feature_columns"]])
    if mode == "none":
        assert np.isnan(transformed).sum() == df[bundle["feature_columns"]].isna().sum().sum()
    else:
        assert not np.isnan(transformed).any()
