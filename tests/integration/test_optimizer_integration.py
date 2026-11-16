from rice_fertilizer.data.feature_engineering import add_npk_features
from rice_fertilizer.modeling.model_registry import create_estimator
from rice_fertilizer.modeling.pipeline_factory import build_model_pipeline
from rice_fertilizer.optimization.objective import ObjectiveConfiguration
from rice_fertilizer.optimization.optimizer import optimize_scenario
from rice_fertilizer.schemas import (
    FertilizerVector, NutrientBounds, OptimizationScenario,
)
from rice_fertilizer.surrogate.predictor import SurrogatePredictor

def test_optimizer_connects_to_surrogate(synthetic_data):
    df = add_npk_features(synthetic_data)
    numeric = [
        "N.rate.kg_ha", "P.rate.kg_ha", "K.rate.kg_ha",
        "avprec.mm", "total_npk",
    ]
    categorical = ["soil_type", "fertilizer_type"]
    features = numeric + categorical
    estimator = create_estimator(
        "random_forest", {"n_estimators": 30, "random_state": 1}
    )
    pipeline = build_model_pipeline(estimator, numeric, categorical, "random_forest")
    pipeline.fit(df[features], df["yield.wb.kg_ha"])
    predictor = SurrogatePredictor({
        "pipeline": pipeline,
        "feature_columns": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_profile": "test",
    }, {"N": "N.rate.kg_ha", "P": "P.rate.kg_ha", "K": "K.rate.kg_ha"})
    row = df.iloc[0]
    scenario = OptimizationScenario(
        "s1",
        {
            "avprec.mm": row["avprec.mm"],
            "soil_type": row["soil_type"],
            "fertilizer_type": row["fertilizer_type"],
        },
        FertilizerVector(
            row["N.rate.kg_ha"], row["P.rate.kg_ha"], row["K.rate.kg_ha"]
        ),
    )
    bounds = NutrientBounds(0, 180, 0, 80, 0, 120)
    config = ObjectiveConfiguration(
        yield_scale=1000,
        lambda_total=0,
        lambda_composition=0,
        enable_total_penalty=False,
        enable_composition_penalty=False,
        enable_total_hard_constraint=False,
        total_soft=None,
        total_max=None,
        min_proportions=None,
        max_proportions=None,
    )
    result, _ = optimize_scenario(
        scenario, predictor, bounds, config,
        {"maxiter": 5, "popsize": 5, "polish": False, "workers": 1},
        seed=1,
    )
    assert 0 <= result.optimal_n <= 180
    assert 0 <= result.optimal_p <= 80
    assert 0 <= result.optimal_k <= 120
