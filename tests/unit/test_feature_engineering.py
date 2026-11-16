import pandas as pd
from rice_fertilizer.data.feature_engineering import add_npk_features

def test_npk_features_are_consistent():
    df = pd.DataFrame({
        "N.rate.kg_ha": [60.0],
        "P.rate.kg_ha": [30.0],
        "K.rate.kg_ha": [10.0],
    })
    result = add_npk_features(df)
    assert result.loc[0, "total_npk"] == 100.0
    assert abs(result.loc[0, "ratio_n_p"] - 2.0) < 1e-8
    assert abs(result.loc[0, "prop_n"] - 0.6) < 1e-8

def test_zero_total_has_zero_proportions():
    df = pd.DataFrame({
        "N.rate.kg_ha": [0.0],
        "P.rate.kg_ha": [0.0],
        "K.rate.kg_ha": [0.0],
    })
    result = add_npk_features(df)
    assert result.loc[0, ["prop_n", "prop_p", "prop_k"]].sum() == 0
