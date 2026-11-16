from rice_fertilizer.data.feature_engineering import add_npk_features
from rice_fertilizer.modeling.model_registry import create_estimator
from rice_fertilizer.modeling.pipeline_factory import build_model_pipeline
from rice_fertilizer.modeling.metrics import regression_metrics

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
    pipeline = build_model_pipeline(estimator, numeric, categorical, "random_forest")
    pipeline.fit(X.iloc[:100], y.iloc[:100])
    prediction = pipeline.predict(X.iloc[100:])
    assert len(prediction) == 20
    assert regression_metrics(y.iloc[100:], prediction)["rmse"] >= 0
